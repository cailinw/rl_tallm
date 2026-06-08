"""
Evaluate zero-shot Qwen and online-GRPO Qwen on OpenCATP/CATP tasks.

The full OpenCATP path executes generated plans with OpenCATP's Plan runner and
records task_score, cost_price, exec_time, qop, and validity. Use --dry_run to
exercise the parser/training loop without the full OpenCATP dataset installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


TOOLS = [
    "image_classification",
    "image_colorization",
    "object_detection",
    "image_deblurring",
    "image_denoising",
    "image_super_resolution",
    "image_captioning",
    "text_to_image",
    "visual_question_answering",
    "sentiment_analysis",
    "question_answering",
    "text_summarization",
    "text_generation",
    "machine_translation",
    "mask_filling",
]

DEFAULT_TRAIN_SEQ_TASKS = [
    1,
    2,
    3,
    4,
    5,
    7,
    9,
    10,
    11,
    14,
    15,
    16,
    17,
    18,
    19,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    34,
    35,
    37,
    38,
    39,
    41,
    42,
    43,
    44,
    45,
    47,
    48,
    49,
    50,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    63,
    64,
    65,
    66,
    67,
    68,
    70,
    71,
    72,
    73,
    75,
    76,
    77,
    79,
    80,
    82,
    83,
    84,
    85,
    86,
]

DEFAULT_TEST_SEQ_TASKS = [0, 6, 8, 12, 13, 20, 21, 31, 36, 40, 46, 51, 61, 62, 69, 74, 78, 81]


@dataclass
class CatpExample:
    task_id: int
    sample_id: int
    task_info: str
    sample_info: Dict[str, Any]


@dataclass
class EvalResult:
    policy: str
    split: str
    task_id: int
    sample_id: int
    valid: bool
    task_score: float
    cost_price: float
    exec_time: float
    qop: float
    reward: float
    plan: str
    error: str = ""
    raw_response: str = ""


@dataclass
class TrainMetric:
    step: int
    mean_reward: float
    policy_loss: float


def normalize_tool_name(value: str) -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "input": "input_of_query",
        "input_query": "input_of_query",
        "input_of_query": "input_of_query",
        "colorization": "image_colorization",
        "text_to_image_generation": "text_to_image",
        "fill_mask": "mask_filling",
    }
    return aliases.get(text, text)


def plan_to_string(plan: Sequence[Any]) -> str:
    return json.dumps(plan, ensure_ascii=True, sort_keys=True)


def sanitize_plan(raw_plan: Any, max_tools: int = 5) -> List[Any]:
    """Convert model JSON into OpenCATP's [tool, [deps], ...] format."""
    if isinstance(raw_plan, dict):
        raw_plan = raw_plan.get("plan") or raw_plan.get("steps") or raw_plan.get("tools")

    plan: List[Any] = []
    seen_tools = set()

    if isinstance(raw_plan, list) and raw_plan and all(isinstance(x, str) for x in raw_plan):
        for idx, tool in enumerate(raw_plan[:max_tools]):
            tool_name = normalize_tool_name(tool)
            if tool_name not in TOOLS or tool_name in seen_tools:
                continue
            dep = "input_of_query" if idx == 0 else plan[-2]
            plan.extend([tool_name, [dep]])
            seen_tools.add(tool_name)
        return plan

    if isinstance(raw_plan, list):
        previous_tool = "input_of_query"
        for item in raw_plan[:max_tools]:
            if isinstance(item, dict):
                tool_name = normalize_tool_name(item.get("tool") or item.get("name") or item.get("tool_name") or "")
                deps = item.get("dependencies") or item.get("deps") or item.get("inputs") or [previous_tool]
            elif isinstance(item, (list, tuple)) and item:
                tool_name = normalize_tool_name(item[0])
                deps = item[1] if len(item) > 1 else [previous_tool]
            else:
                continue

            if tool_name not in TOOLS or tool_name in seen_tools:
                continue
            if not isinstance(deps, list):
                deps = [deps]
            norm_deps = [normalize_tool_name(str(dep)) for dep in deps if str(dep).strip()]
            norm_deps = [dep for dep in norm_deps if dep == "input_of_query" or dep in seen_tools]
            if not norm_deps:
                norm_deps = [previous_tool]
            plan.extend([tool_name, norm_deps])
            previous_tool = tool_name
            seen_tools.add(tool_name)

    return plan


def parse_plan_from_text(text: str, max_tools: int = 5) -> List[Any]:
    text = text.strip()
    candidates = [text]
    if "```" in text:
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        candidates = fenced + candidates
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    if "[" in text and "]" in text:
        candidates.append(text[text.find("[") : text.rfind("]") + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        plan = sanitize_plan(payload, max_tools=max_tools)
        if plan:
            return plan
    mentioned_tools = []
    normalized_text = text.lower().replace("-", "_").replace(" ", "_")
    for tool in TOOLS:
        if tool in normalized_text:
            mentioned_tools.append(tool)
    if mentioned_tools:
        return sanitize_plan(mentioned_tools, max_tools=max_tools)
    return []


def reward_from_metrics(
    valid: bool,
    task_score: float,
    cost_price: float,
    exec_time: float,
    alpha: float,
    latency_weight: float,
    invalid_penalty: float,
) -> float:
    if not valid:
        return invalid_penalty
    if not all(math.isfinite(float(value)) for value in (task_score, cost_price, exec_time)):
        return invalid_penalty
    # QOP already combines task score and price; add an explicit runtime term so
    # the online policy sees live critical-path time directly.
    return alpha * float(task_score) + (1.0 - alpha) * float(-cost_price) - latency_weight * float(exec_time)


def finite_or(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def normalize_metric_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        normalized = {}
        for key, value in payload.items():
            if "text" in str(key) and not isinstance(value, str):
                if isinstance(value, (list, tuple)):
                    normalized[key] = " ".join(str(item) for item in value)
                else:
                    normalized[key] = str(value)
            else:
                normalized[key] = value
        return normalized
    return payload


class DryRunCatpEvaluator:
    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self.tool_quality = {tool: self.rng.uniform(0.45, 0.95) for tool in TOOLS}
        self.tool_latency = {tool: self.rng.uniform(0.05, 0.9) for tool in TOOLS}
        self.tool_cost = {tool: 0.01 + 0.08 * self.tool_latency[tool] for tool in TOOLS}

    def make_examples(
        self,
        split: str,
        n: int,
        task_ids: Optional[Sequence[int]] = None,
        sample_ids_by_task: Optional[Dict[str, Sequence[int]]] = None,
    ) -> List[CatpExample]:
        task_ids = list(task_ids) if task_ids is not None else (
            DEFAULT_TRAIN_SEQ_TASKS if split == "train" else DEFAULT_TEST_SEQ_TASKS
        )
        examples = []
        for i, task_id in enumerate(task_ids[:n]):
            examples.append(
                CatpExample(
                    task_id=task_id,
                    sample_id=0,
                    task_info=f"Dry-run OpenCATP task {task_id}: choose tools to transform input into target output.",
                    sample_info={"has_image": task_id < 105, "has_text": 107 <= task_id <= 125, "dry_run": True},
                )
            )
        return examples

    def evaluate_plan(self, example: CatpExample, plan: List[Any], policy: str, split: str) -> EvalResult:
        tools = [plan[i] for i in range(0, len(plan), 2) if i < len(plan)]
        valid = bool(tools) and all(tool in TOOLS for tool in tools)
        if valid:
            quality = sum(self.tool_quality[t] for t in tools) / max(1, len(tools))
            coverage = min(1.0, 0.35 + 0.18 * len(set(tools)))
            task_score = min(1.0, 0.5 * quality + 0.5 * coverage)
            exec_time = sum(self.tool_latency[t] for t in tools)
            cost_price = sum(self.tool_cost[t] for t in tools)
            qop = 0.5 * task_score - 0.5 * cost_price
            error = ""
        else:
            task_score = -2.0
            exec_time = 0.0
            cost_price = 2.0
            qop = -2.0
            error = "invalid_plan"
        reward = reward_from_metrics(valid, task_score, cost_price, exec_time, 0.5, 0.1, -1.0)
        return EvalResult(
            policy=policy,
            split=split,
            task_id=example.task_id,
            sample_id=example.sample_id,
            valid=valid,
            task_score=task_score,
            cost_price=cost_price,
            exec_time=exec_time,
            qop=qop,
            reward=reward,
            plan=plan_to_string(plan),
            error=error,
        )


class OpenCatpEvaluator:
    def __init__(self, opencatp_root: Path, alpha: float, latency_weight: float, invalid_penalty: float):
        dataset_path = opencatp_root / "dataset" / "opencatp"
        samples_path = dataset_path / "test_task_samples.json"
        if not samples_path.exists():
            raise FileNotFoundError(
                f"Missing OpenCATP dataset at {dataset_path}. Expected {samples_path}. "
                "Download the OpenCATP dataset or symlink it there."
            )

        sys.path.insert(0, str(opencatp_root))
        from src.config import GlobalTaskConfig, GlobalPathConfig, GlobalMetricsConfig  # type: ignore
        from src.data_loader import TaskDataset  # type: ignore
        from src.metrics.evaluator import calculate_qop, calculate_task_score  # type: ignore
        from src.plan import Plan  # type: ignore
        from src.catpllm.utils.utils import get_task_and_sample_info  # type: ignore

        self.global_task_config = GlobalTaskConfig
        self.global_path_config = GlobalPathConfig
        self.global_metrics_config = GlobalMetricsConfig
        self.task_dataset_cls = TaskDataset
        self.calculate_qop = calculate_qop
        self.calculate_task_score = calculate_task_score
        self.plan_cls = Plan
        self.get_task_and_sample_info = get_task_and_sample_info
        self.alpha = alpha
        self.latency_weight = latency_weight
        self.invalid_penalty = invalid_penalty

    def make_examples(
        self,
        split: str,
        n: int,
        task_ids: Optional[Sequence[int]] = None,
        sample_ids_by_task: Optional[Dict[str, Sequence[int]]] = None,
    ) -> List[CatpExample]:
        task_ids = list(task_ids) if task_ids is not None else (
            DEFAULT_TRAIN_SEQ_TASKS if split == "train" else DEFAULT_TEST_SEQ_TASKS
        )
        if sample_ids_by_task is not None:
            samples_by_task = {str(task_id): list(sample_ids) for task_id, sample_ids in sample_ids_by_task.items()}
        elif split == "test":
            samples_by_task = self.global_task_config.default_test_task_samples
        else:
            samples_by_task = {str(task_id): [0] for task_id in task_ids}
        examples: List[CatpExample] = []
        for task_id in task_ids:
            sample_ids = samples_by_task.get(str(task_id), [0])
            for sample_id in sample_ids:
                task_info, sample_info = self.get_task_and_sample_info(
                    task_id, int(sample_id), self.global_path_config.data_path
                )
                examples.append(
                    CatpExample(
                        task_id=task_id,
                        sample_id=int(sample_id),
                        task_info=task_info,
                        sample_info=sample_info,
                    )
                )
                if len(examples) >= n:
                    return examples
        return examples

    def evaluate_plan(self, example: CatpExample, plan: List[Any], policy: str, split: str) -> EvalResult:
        if not plan:
            return EvalResult(policy, split, example.task_id, example.sample_id, False, -2.0, 2.0, 0.0, -2.0, -1.0, "[]", "empty_plan")

        task_dataset = self.task_dataset_cls(self.global_path_config.data_path, task_id=example.task_id)
        batch = task_dataset[task_dataset.sample_ids.index(example.sample_id)]
        try:
            opencatp_plan = self.plan_cls(plan)
            result = opencatp_plan.execute(batch["input"])
            if result is None:
                raise ValueError("plan_execution_returned_none")
            result = normalize_metric_payload(result)
            output = normalize_metric_payload(batch["output"])
            task_score = finite_or(self.calculate_task_score(result, output, sequential=example.task_id < 200), 0.0)
            cost_price = finite_or(opencatp_plan.price, float(self.global_metrics_config.cost_penalty))
            exec_time = finite_or(opencatp_plan.exec_time, 0.0)
            qop = finite_or(self.calculate_qop(task_score, cost_price), -2.0)
            valid = True
            error = ""
        except Exception as exc:
            task_score = float(self.global_metrics_config.score_penalty)
            cost_price = float(self.global_metrics_config.cost_penalty)
            exec_time = 0.0
            qop = -2.0
            valid = False
            error = str(exc) or repr(exc)

        reward = reward_from_metrics(
            valid,
            task_score,
            cost_price,
            exec_time / 1000.0,
            self.alpha,
            self.latency_weight,
            self.invalid_penalty,
        )
        return EvalResult(
            policy=policy,
            split=split,
            task_id=example.task_id,
            sample_id=example.sample_id,
            valid=valid,
            task_score=task_score,
            cost_price=cost_price,
            exec_time=exec_time,
            qop=qop,
            reward=reward,
            plan=plan_to_string(plan),
            error=error,
        )


def build_prompt(example: CatpExample, max_tools: int) -> str:
    return (
        "You are an OpenCATP tool planner. Choose a short executable plan.\n"
        "Available tools:\n"
        + "\n".join(f"- {tool}" for tool in TOOLS)
        + "\n\n"
        f"Task: {example.task_info}\n"
        f"Sample metadata: {json.dumps(example.sample_info, ensure_ascii=True)}\n\n"
        "Return only JSON in this form:\n"
        '{"plan":[{"tool":"tool_name","dependencies":["input_of_query"]}]}\n'
        f"Use at most {max_tools} tools. Dependencies must be input_of_query or prior tools."
    )


class HeuristicPlanPolicy:
    def __init__(self, seed: int, max_tools: int):
        self.rng = random.Random(seed)
        self.max_tools = max_tools

    def sample_plan(self, example: CatpExample, trainable: bool = False) -> Tuple[List[Any], str, Optional[torch.Tensor]]:
        del trainable
        task_id = example.task_id
        if 0 <= task_id <= 14:
            pool = ["image_denoising", "image_deblurring", "image_super_resolution", "image_colorization"]
        elif 15 <= task_id <= 104:
            pool = ["image_captioning", "image_classification", "object_detection"]
        elif 200 <= task_id <= 229:
            pool = ["image_captioning", "image_classification", "object_detection", "machine_translation"]
        elif example.sample_info.get("has_image"):
            pool = ["image_denoising", "image_deblurring", "image_super_resolution", "image_captioning", "object_detection"]
        elif example.sample_info.get("has_text"):
            pool = ["text_summarization", "machine_translation", "sentiment_analysis", "text_generation", "mask_filling"]
        else:
            pool = list(TOOLS)
        n = min(self.max_tools, self.rng.randint(1, min(3, len(pool))))
        chosen = self.rng.sample(pool, k=n)
        plan = sanitize_plan(chosen, max_tools=self.max_tools)
        return plan, json.dumps({"plan": chosen}), None

    def apply_loss(self, loss: torch.Tensor) -> float:
        del loss
        return 0.0

    def save(self, output_dir: Path) -> None:
        del output_dir


class QwenPlanPolicy:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: str,
        learning_rate: float,
        max_new_tokens: int,
        max_tools: int,
        train: bool,
        use_lora: bool,
        load_in_4bit: bool,
        load_adapter: Optional[str] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.max_new_tokens = max_new_tokens
        self.max_tools = max_tools
        self.trainable = train
        self.use_lora = (use_lora and train) or bool(load_adapter)
        self.load_adapter = load_adapter

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

        kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": device}

        tokenizer_source = load_adapter or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not load_in_4bit:
            self.model = self.model.to(device)

        if load_adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, load_adapter, is_trainable=train)
        elif self.use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if load_in_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                ),
            )

        self.model.train(train)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate) if train and params else None

    def sample_plan(self, example: CatpExample, trainable: bool = False) -> Tuple[List[Any], str, Optional[torch.Tensor]]:
        prompt_text = build_prompt(example, self.max_tools)
        messages = [
            {"role": "system", "content": "You are a cost-aware OpenCATP tool planner. Return only valid JSON."},
            {"role": "user", "content": prompt_text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = prompt_text
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = encoded["input_ids"].shape[1]
        generate_kwargs: Dict[str, Any] = {
            "do_sample": trainable,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
        }
        if trainable:
            generate_kwargs.update({"temperature": 0.8, "top_p": 0.95})
        gen = self.model.generate(**encoded, **generate_kwargs)
        seq = gen.sequences
        gen_ids = seq[:, prompt_len:]
        text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        plan = parse_plan_from_text(text, max_tools=self.max_tools)

        logprob_sum: Optional[torch.Tensor] = None
        if trainable and self.optimizer is not None and gen_ids.shape[1] > 0:
            full_input = seq[:, :-1]
            full_target = seq[:, 1:]
            attn = torch.ones_like(full_input, dtype=torch.long, device=device)
            outputs = self.model(input_ids=full_input, attention_mask=attn)
            token_logp = torch.log_softmax(outputs.logits, dim=-1)
            start = prompt_len - 1
            gen_len = gen_ids.shape[1]
            pred_slice = token_logp[:, start : start + gen_len, :]
            target_slice = full_target[:, start : start + gen_len]
            logprob_sum = pred_slice.gather(-1, target_slice.unsqueeze(-1)).squeeze(-1).sum()
        return plan, text, logprob_sum

    def apply_loss(self, loss: torch.Tensor) -> float:
        if self.optimizer is None:
            return 0.0
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.optimizer.step()
        return float(loss.detach().item())

    def save(self, output_dir: Path) -> None:
        if self.use_lora and self.trainable:
            adapter_dir = output_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(adapter_dir)
            self.tokenizer.save_pretrained(adapter_dir)


class OnlineGRPO:
    def __init__(self, policy: Any, evaluator: Any, group_size: int):
        self.policy = policy
        self.evaluator = evaluator
        self.group_size = group_size

    def train(self, examples: Sequence[CatpExample], steps: int) -> List[TrainMetric]:
        metrics: List[TrainMetric] = []
        if not examples:
            return metrics
        for step in range(1, steps + 1):
            example = examples[(step - 1) % len(examples)]
            rewards: List[float] = []
            logprobs: List[torch.Tensor] = []
            for _ in range(self.group_size):
                plan, _, logprob = self.policy.sample_plan(example, trainable=True)
                result = self.evaluator.evaluate_plan(example, plan, policy="qwen_online_grpo", split="train")
                rewards.append(result.reward)
                if logprob is not None:
                    logprobs.append(logprob)
            reward_t = torch.tensor(rewards, dtype=torch.float32)
            if reward_t.numel() > 1:
                adv = (reward_t - reward_t.mean()) / reward_t.std(unbiased=False).clamp_min(1e-6)
            else:
                adv = torch.zeros_like(reward_t)
            loss_value = 0.0
            if logprobs:
                loss_terms = []
                for logprob, advantage in zip(logprobs, adv):
                    loss_terms.append(-(advantage.to(device=logprob.device, dtype=logprob.dtype) * logprob))
                loss_value = self.policy.apply_loss(torch.stack(loss_terms).mean())
            metrics.append(TrainMetric(step=step, mean_reward=float(reward_t.mean().item()), policy_loss=loss_value))
        return metrics


def evaluate_policy(policy: Any, evaluator: Any, examples: Sequence[CatpExample], name: str, split: str) -> List[EvalResult]:
    rows = []
    for example in examples:
        plan, raw_response, _ = policy.sample_plan(example, trainable=False)
        result = evaluator.evaluate_plan(example, plan, policy=name, split=split)
        result.raw_response = raw_response
        rows.append(result)
    return rows


def summarize(rows: Sequence[EvalResult]) -> Dict[str, Any]:
    valid = [r for r in rows if r.valid]
    return {
        "n": len(rows),
        "valid_rate": len(valid) / max(1, len(rows)),
        "mean_task_score": sum(r.task_score for r in rows) / max(1, len(rows)),
        "mean_cost_price": sum(r.cost_price for r in rows) / max(1, len(rows)),
        "mean_exec_time": sum(r.exec_time for r in rows) / max(1, len(rows)),
        "mean_qop": sum(r.qop for r in rows) / max(1, len(rows)),
        "mean_reward": sum(r.reward for r in rows) / max(1, len(rows)),
    }


def write_outputs(output_dir: Path, rows: Sequence[EvalResult], train_metrics: Sequence[TrainMetric], config: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    (output_dir / "summary.json").write_text(json.dumps(summarize(rows), indent=2, sort_keys=True))
    with (output_dir / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else list(EvalResult.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    with (output_dir / "train_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrainMetric.__dataclass_fields__))
        writer.writeheader()
        for row in train_metrics:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opencatp_root", default="external/OpenCATP-LLM")
    parser.add_argument("--output_dir", default="experiments/results/catp_online_grpo")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--policy_backend", choices=["heuristic", "qwen"], default="qwen")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--train_steps", type=int, default=8)
    parser.add_argument("--n_train", type=int, default=4)
    parser.add_argument("--n_eval", type=int, default=4)
    parser.add_argument("--train_task_ids", default=None, help="Comma-separated OpenCATP task ids for training.")
    parser.add_argument("--eval_task_ids", default=None, help="Comma-separated OpenCATP task ids for evaluation.")
    parser.add_argument(
        "--train_samples",
        default=None,
        help="Explicit train samples as 'task:sample,sample;task:sample,sample'.",
    )
    parser.add_argument(
        "--eval_samples",
        default=None,
        help="Explicit eval samples as 'task:sample,sample;task:sample,sample'.",
    )
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_tools", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--latency_weight", type=float, default=0.1)
    parser.add_argument("--invalid_penalty", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--load_adapter",
        default=None,
        help="Evaluate a saved PEFT/LoRA adapter as qwen_online_grpo without retraining.",
    )
    parser.add_argument("--no_force_exit", action="store_true", help="Do not force-exit after real OpenCATP runs.")
    return parser.parse_args()


def parse_task_ids(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_sample_ids(value: Optional[str]) -> Optional[Dict[str, List[int]]]:
    if not value:
        return None
    samples_by_task: Dict[str, List[int]] = {}
    for task_spec in value.split(";"):
        task_spec = task_spec.strip()
        if not task_spec:
            continue
        task_id, raw_sample_ids = task_spec.split(":", maxsplit=1)
        samples_by_task[task_id.strip()] = [
            int(part.strip()) for part in raw_sample_ids.split(",") if part.strip()
        ]
    return samples_by_task


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    opencatp_root = Path(args.opencatp_root).expanduser().resolve()
    if args.dry_run:
        evaluator = DryRunCatpEvaluator(seed=args.seed)
    else:
        evaluator = OpenCatpEvaluator(
            opencatp_root=opencatp_root,
            alpha=args.alpha,
            latency_weight=args.latency_weight,
            invalid_penalty=args.invalid_penalty,
        )

    train_examples = evaluator.make_examples(
        "train",
        args.n_train,
        task_ids=parse_task_ids(args.train_task_ids),
        sample_ids_by_task=parse_sample_ids(args.train_samples),
    )
    eval_examples = evaluator.make_examples(
        "test",
        args.n_eval,
        task_ids=parse_task_ids(args.eval_task_ids),
        sample_ids_by_task=parse_sample_ids(args.eval_samples),
    )

    if args.policy_backend == "heuristic":
        trained_policy = HeuristicPlanPolicy(seed=args.seed, max_tools=args.max_tools)
        baseline_policy = HeuristicPlanPolicy(seed=args.seed, max_tools=args.max_tools)
        baseline_rows = evaluate_policy(baseline_policy, evaluator, eval_examples, "qwen_baseline", "test")
    else:
        baseline_policy = QwenPlanPolicy(
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            learning_rate=args.learning_rate,
            max_new_tokens=args.max_new_tokens,
            max_tools=args.max_tools,
            train=not bool(args.load_adapter),
            use_lora=args.use_lora and not bool(args.load_adapter),
            load_in_4bit=args.load_in_4bit,
            load_adapter=None,
        )
        # Single-GPU friendly baseline: evaluate the base model before either
        # training in-place or loading the saved online-GRPO adapter.
        baseline_rows = evaluate_policy(baseline_policy, evaluator, eval_examples, "qwen_baseline", "test")
        trained_policy = baseline_policy
        if args.load_adapter:
            del baseline_policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trained_policy = QwenPlanPolicy(
                model_name=args.model_name,
                device=args.device,
                dtype=args.dtype,
                learning_rate=args.learning_rate,
                max_new_tokens=args.max_new_tokens,
                max_tools=args.max_tools,
                train=False,
                use_lora=False,
                load_in_4bit=args.load_in_4bit,
                load_adapter=args.load_adapter,
            )

    trainer = OnlineGRPO(trained_policy, evaluator, group_size=args.group_size)
    train_metrics = [] if args.load_adapter else trainer.train(train_examples, steps=args.train_steps)
    grpo_rows = evaluate_policy(trained_policy, evaluator, eval_examples, "qwen_online_grpo", "test")

    output_dir = Path(args.output_dir)
    all_rows = baseline_rows + grpo_rows
    write_outputs(output_dir, all_rows, train_metrics, vars(args))
    trained_policy.save(output_dir)

    print(json.dumps({"baseline": summarize(baseline_rows), "online_grpo": summarize(grpo_rows)}, indent=2, sort_keys=True))
    print(f"Saved results to {output_dir}")
    sys.stdout.flush()
    sys.stderr.flush()
    if not args.dry_run and not args.no_force_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
