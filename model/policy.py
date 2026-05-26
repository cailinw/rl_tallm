from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from model.context_builder import parse_model_output


@dataclass
class PolicyStep:
    action: Dict
    action_text: str
    logprob: Optional[torch.Tensor]


class PolicyModel:
    """
    Policy wrapper with two modes:
    - Heuristic mode (default fallback): random/latency-biased tool picker.
    - HF mode: autoregressive text generation + logprob extraction for GRPO updates.
    """

    def __init__(self, model_name: str, seed: int = 42, training_cfg: Optional[Dict] = None):
        self.model_name = model_name
        self.rng = random.Random(seed)
        self.training_cfg = training_cfg or {}

        self.use_hf_policy = bool(self.training_cfg.get("use_hf_policy", True))
        self.enable_policy_updates = bool(self.training_cfg.get("enable_policy_updates", True))
        self.policy_device = str(
            self.training_cfg.get("policy_device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_new_tokens = int(self.training_cfg.get("max_new_tokens", 128))
        self.temperature = float(self.training_cfg.get("temperature", 0.8))
        self.top_p = float(self.training_cfg.get("top_p", 0.95))
        self.learning_rate = float(self.training_cfg.get("learning_rate", 5e-6))
        self.max_grad_norm = float(self.training_cfg.get("max_grad_norm", 1.0))

        self.hf_model = None
        self.hf_tokenizer = None
        self.optimizer = None
        self.trainable = False

        if self.use_hf_policy:
            self._init_hf_model()

    def _init_hf_model(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception:
            self.use_hf_policy = False
            return

        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        self.hf_model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.policy_device)
        self.hf_model.train(self.enable_policy_updates)
        if self.enable_policy_updates:
            self.optimizer = torch.optim.AdamW(self.hf_model.parameters(), lr=self.learning_rate)
            self.trainable = True

    def generate_action(
        self,
        prompt: str,
        available_tools: List[Dict],
        force_final_prob: float = 0.2,
    ) -> Dict:
        step = self.generate_action_with_trace(
            prompt=prompt,
            available_tools=available_tools,
            force_final_prob=force_final_prob,
        )
        return step.action

    def generate_action_with_trace(
        self,
        prompt: str,
        available_tools: List[Dict],
        force_final_prob: float = 0.2,
    ) -> PolicyStep:
        if self.use_hf_policy and self.hf_model is not None and self.hf_tokenizer is not None:
            return self._generate_hf(prompt, available_tools)

        if not available_tools or self.rng.random() < force_final_prob:
            text = json.dumps({"type": "final_answer", "answer": "best effort final answer"})
            return PolicyStep(action=parse_model_output(text), action_text=text, logprob=None)

        # Favor faster tool prior as a sensible default policy.
        sorted_tools = sorted(available_tools, key=lambda t: float(t.get("default_latency_ms", 500.0)))
        pick = sorted_tools[0] if self.rng.random() < 0.8 else self.rng.choice(sorted_tools)
        action_json = json.dumps({"type": "tool_call", "tool_name": pick["name"], "tool_args": {"query": "default"}})
        return PolicyStep(action=parse_model_output(action_json), action_text=action_json, logprob=None)

    def _generate_hf(self, prompt: str, available_tools: List[Dict]) -> PolicyStep:
        del available_tools  # tools are represented in the prompt text
        assert self.hf_model is not None
        assert self.hf_tokenizer is not None

        encoded = self.hf_tokenizer(prompt, return_tensors="pt").to(self.policy_device)
        prompt_len = encoded["input_ids"].shape[1]
        gen = self.hf_model.generate(
            **encoded,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.hf_tokenizer.pad_token_id,
            eos_token_id=self.hf_tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        seq = gen.sequences
        gen_ids = seq[:, prompt_len:]
        action_text = self.hf_tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        action = parse_model_output(action_text)

        logprob_sum: Optional[torch.Tensor] = None
        if self.trainable and gen_ids.shape[1] > 0:
            # Recompute token logprobs with gradients enabled.
            full_input = seq[:, :-1]
            full_target = seq[:, 1:]
            attn = torch.ones_like(full_input, dtype=torch.long, device=self.policy_device)
            outputs = self.hf_model(input_ids=full_input, attention_mask=attn)
            token_logp = torch.log_softmax(outputs.logits, dim=-1)
            start = prompt_len - 1
            gen_len = gen_ids.shape[1]
            pred_slice = token_logp[:, start : start + gen_len, :]
            target_slice = full_target[:, start : start + gen_len]
            picked = pred_slice.gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)
            logprob_sum = picked.sum()
        return PolicyStep(action=action, action_text=action_text, logprob=logprob_sum)

    def apply_grpo_loss(self, loss: torch.Tensor) -> float:
        if not self.trainable or self.optimizer is None:
            return 0.0
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hf_model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return float(loss.detach().item())
