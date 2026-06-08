"""
Controlled ablation: does RL *training* help the SAME LLM on the arena?

This is the fair counterpart to the cross-paradigm plot in rl_tool_arena.py.
Here the model class, the prompt, and the inputs are held FIXED; the ONLY
difference between the two LLM policies is the GRPO policy-gradient update:

    qwen_zeroshot : Qwen2.5-Instruct, LoRA adapter DISABLED  (no training)
    qwen_grpo     : the SAME checkpoint, after LoRA GRPO fine-tuning on reward

Both read the identical natural-language tool descriptions + objective produced
by rl_tool_arena.build_arena_qwen_messages, and both are scored with the same
expected_reward / oracle-normalization as every other policy. So any gap
between them is attributable to RL training alone.

The GRPO update mirrors training/grpo_trainer.py: sample a group of G
completions per context, compute group-relative advantages
(r - mean) / (std + eps), and ascend sum_g adv_g * logprob(completion_g).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl_tool_arena import (  # noqa: E402
    Context,
    GRPOConfig,
    RewardConfig,
    Tool,
    TabularGRPOPolicy,
    build_arena_qwen_messages,
    build_world,
    evaluate_action_fn,
    evaluate_policy_expected,
    latency_greedy_action,
    normalized_score,
    oracle_action,
    quality_greedy_action,
    reliability_greedy_action,
    sample_reward,
    wtb_category_distribution,
    _action_expectations,
)


# --------------------------------------------------------------------------- #
# Shared parsing (identical rule to ArenaQwenPolicy in rl_tool_arena.py)
# --------------------------------------------------------------------------- #
def parse_tool_choice(raw: str, tool_names: Sequence[str]) -> Optional[int]:
    text = raw
    if "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    chosen = None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            chosen = payload.get("tool") or payload.get("tool_name") or payload.get("name")
    except Exception:
        chosen = None
    if isinstance(chosen, str) and chosen in tool_names:
        return list(tool_names).index(chosen)
    for i, name in enumerate(tool_names):
        if name in raw:
            return i
    return None  # unparseable


# --------------------------------------------------------------------------- #
# Qwen GRPO trainer (LoRA)
# --------------------------------------------------------------------------- #
@dataclass
class QwenGRPOConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    lr: float = 1e-5
    group_size: int = 8
    total_steps: int = 300
    eval_interval: int = 30
    max_new_tokens: int = 24
    temperature: float = 1.0
    top_p: float = 0.95
    micro_batch: int = 2
    grad_clip: float = 1.0
    parse_fail_penalty: float = -1.0  # reward assigned to unparseable completions
    seed: int = 0


class QwenGRPOArena:
    def __init__(self, cfg: QwenGRPOConfig, dtype: str = "bfloat16"):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.cfg = cfg

        torch_dtype = torch.bfloat16
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch_dtype, device_map="cuda", trust_remote_code=True
        )
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(self.model, lora)
        self.model.print_trainable_parameters()
        self.device = next(self.model.parameters()).device
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=cfg.lr
        )
        self.parse_failures = 0
        self.train_steps = 0
        self.diverse_steps = 0  # steps where the group sampled >1 distinct action
        self.uniq_sum = 0.0

    # ---- prompt helper ---------------------------------------------------- #
    def _prompt_ids(self, ctx: Context, reward_cfg: RewardConfig):
        messages = build_arena_qwen_messages(ctx, reward_cfg, reveal_numbers=False)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return enc

    # ---- greedy action for evaluation ------------------------------------ #
    def select_action(self, ctx: Context, reward_cfg: RewardConfig) -> int:
        torch = self.torch
        enc = self._prompt_ids(ctx, reward_cfg)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(gen[0], skip_special_tokens=True).strip()
        names = [t.name for t in ctx.tools]
        idx = parse_tool_choice(raw, names)
        return 0 if idx is None else idx

    def zeroshot_action(self, ctx: Context, reward_cfg: RewardConfig) -> int:
        # Same model, adapter disabled => the untrained baseline.
        with self.model.disable_adapter():
            return self.select_action(ctx, reward_cfg)

    # ---- token log-probs of completions (with grad) ---------------------- #
    def _completion_logprobs(self, full_ids, prompt_len: int):
        torch = self.torch
        attn = (full_ids != self.tokenizer.pad_token_id).long()
        out = self.model(input_ids=full_ids, attention_mask=attn)
        # Only the completion region needs grads; slice BEFORE the (huge) softmax
        # to avoid materializing log-probs over the full sequence x vocab.
        logits = out.logits[:, prompt_len - 1 : -1, :]      # predicts tokens [prompt_len : end]
        targets = full_ids[:, prompt_len:]                  # [G, comp_len]
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_logp = torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)
        comp_mask = (targets != self.tokenizer.pad_token_id).float()
        seq_logp = (tok_logp * comp_mask).sum(dim=1) / comp_mask.sum(dim=1).clamp_min(1.0)
        return seq_logp

    # ---- one GRPO step on a single context ------------------------------- #
    def _grpo_step(self, ctx: Context, reward_cfg: RewardConfig, rng: np.random.Generator) -> Dict[str, float]:
        torch = self.torch
        enc = self._prompt_ids(ctx, reward_cfg)
        prompt_ids = enc["input_ids"]
        prompt_len = prompt_ids.shape[1]
        names = [t.name for t in ctx.tools]

        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                num_return_sequences=self.cfg.group_size,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # gen: [G, prompt_len + comp_len]
        rewards = np.zeros(self.cfg.group_size, dtype=np.float64)
        chosen_actions: List[int] = []
        for g in range(self.cfg.group_size):
            comp = gen[g, prompt_len:]
            raw = self.tokenizer.decode(comp, skip_special_tokens=True).strip()
            idx = parse_tool_choice(raw, names)
            if idx is None:
                self.parse_failures += 1
                rewards[g] = self.cfg.parse_fail_penalty
                chosen_actions.append(-1)
            else:
                rewards[g] = sample_reward(ctx.tools[idx], reward_cfg, rng)[0]
                chosen_actions.append(idx)

        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)

        # Micro-batch the grad forward/backward so memory is bounded by
        # micro_batch sequences, not the whole group.
        self.optimizer.zero_grad()
        total_loss = 0.0
        G = self.cfg.group_size
        mb = max(1, self.cfg.micro_batch)
        for start in range(0, G, mb):
            sl = slice(start, min(start + mb, G))
            seq_logp = self._completion_logprobs(gen[sl], prompt_len)
            loss_mb = -(adv_t[sl] * seq_logp).sum() / G
            loss_mb.backward()
            total_loss += float(loss_mb.item())
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], self.cfg.grad_clip
        )
        self.optimizer.step()
        n_uniq = len(set(chosen_actions))
        self.train_steps += 1
        self.uniq_sum += n_uniq
        if n_uniq > 1:
            self.diverse_steps += 1
        return {
            "loss": total_loss,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "uniq_actions": float(n_uniq),
        }

    # ---- full training loop ---------------------------------------------- #
    def train(self, contexts: Sequence[Context], reward_cfg: RewardConfig,
              random_v: float, oracle_v: float) -> List[Dict[str, float]]:
        rng = np.random.default_rng(self.cfg.seed)
        ctx_probs = np.array([c.prob for c in contexts])
        history: List[Dict[str, float]] = []

        def _eval_norm() -> Dict[str, float]:
            stats = evaluate_action_fn(contexts, lambda c: self.select_action(c, reward_cfg), reward_cfg)
            return {
                "expected_reward": stats["expected_reward"],
                "normalized_score": normalized_score(stats["expected_reward"], random_v, oracle_v),
                "success_prob": stats["success_prob"],
                "latency_s": stats["latency_s"],
                "timeout_prob": stats["timeout_prob"],
            }

        history.append({"step": 0.0, **_eval_norm()})
        print(f"[step 0] qwen_grpo norm={history[-1]['normalized_score']:.3f}")

        for step in range(1, self.cfg.total_steps + 1):
            ctx_idx = int(rng.choice(len(contexts), p=ctx_probs))
            info = self._grpo_step(contexts[ctx_idx], reward_cfg, rng)
            self._last_info = info
            if step % self.cfg.eval_interval == 0:
                ev = _eval_norm()
                history.append({"step": float(step), **ev})
                print(
                    f"[step {step}] loss={info['loss']:.3f} train_r={info['reward_mean']:.3f} "
                    f"r_std={info['reward_std']:.3f} uniq={info['uniq_actions']:.1f} "
                    f"qwen_grpo norm={ev['normalized_score']:.3f}"
                )
        return history


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _random_metrics(contexts: Sequence[Context], reward_cfg: RewardConfig) -> Dict[str, float]:
    agg = {"expected_reward": 0.0, "success_prob": 0.0, "latency_s": 0.0, "timeout_prob": 0.0}
    for ctx in contexts:
        stats = [_action_expectations(ctx, a, reward_cfg) for a in range(len(ctx.tools))]
        for k in agg:
            agg[k] += ctx.prob * float(np.mean([s[k] for s in stats]))
    return agg


@dataclass
class SummaryRow:
    policy: str
    normalized_score: float
    expected_reward: float
    success_prob: float
    latency_s: float
    timeout_prob: float


def _write_dicts(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", default=None)
    p.add_argument("--n_contexts", type=int, default=12)
    p.add_argument("--tools_per_context", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--r_correct", type=float, default=1.0)
    p.add_argument("--w_latency", type=float, default=0.3)
    p.add_argument("--r_timeout", type=float, default=1.0)

    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--total_steps", type=int, default=300)
    p.add_argument("--eval_interval", type=int, default=30)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--micro_batch", type=int, default=2)

    p.add_argument("--tabular_steps", type=int, default=4000)
    p.add_argument("--output_dir", default="experiments/results/qwen_grpo_arena")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    context_names = None
    context_probs = None
    if args.dataset_path:
        try:
            context_names, context_probs = wtb_category_distribution(args.dataset_path)
            args.n_contexts = len(context_names)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"WARN: could not load WildToolBench categories ({exc}); using synthetic contexts.")

    contexts = build_world(
        n_contexts=args.n_contexts,
        tools_per_context=args.tools_per_context,
        seed=args.seed,
        context_names=context_names,
        context_probs=context_probs,
    )
    reward_cfg = RewardConfig(r_correct=args.r_correct, w_latency=args.w_latency, r_timeout_penalty=args.r_timeout)

    # Analytic baselines / endpoints.
    rnd = _random_metrics(contexts, reward_cfg)
    ora = evaluate_action_fn(contexts, lambda c: oracle_action(c, reward_cfg), reward_cfg)
    random_v, oracle_v = rnd["expected_reward"], ora["expected_reward"]

    def _row(name: str, stats: Dict[str, float]) -> Dict[str, float]:
        return asdict(SummaryRow(
            policy=name,
            normalized_score=normalized_score(stats["expected_reward"], random_v, oracle_v),
            expected_reward=stats["expected_reward"],
            success_prob=stats["success_prob"],
            latency_s=stats["latency_s"],
            timeout_prob=stats["timeout_prob"],
        ))

    summary: List[Dict[str, float]] = []
    summary.append(_row("random", rnd))
    summary.append(_row("latency_greedy", evaluate_action_fn(contexts, latency_greedy_action, reward_cfg)))
    summary.append(_row("quality_greedy", evaluate_action_fn(contexts, quality_greedy_action, reward_cfg)))
    summary.append(_row("reliability_greedy", evaluate_action_fn(contexts, reliability_greedy_action, reward_cfg)))

    # Tabular GRPO reference (different model class; for context only).
    tab = TabularGRPOPolicy(contexts, GRPOConfig(total_steps=args.tabular_steps, seed=args.seed))
    tab.train(reward_cfg)
    summary.append(_row("tabular_grpo_rl", evaluate_policy_expected(tab, reward_cfg)))

    # The controlled pair: same Qwen, adapter off vs GRPO-trained.
    trainer = QwenGRPOArena(
        QwenGRPOConfig(
            model_name=args.model,
            lr=args.lr,
            group_size=args.group_size,
            total_steps=args.total_steps,
            eval_interval=args.eval_interval,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            micro_batch=args.micro_batch,
            seed=args.seed,
        ),
        dtype=args.dtype,
    )

    zs = evaluate_action_fn(contexts, lambda c: trainer.zeroshot_action(c, reward_cfg), reward_cfg)
    summary.append(_row("qwen_zeroshot", zs))
    print(f"qwen_zeroshot norm={normalized_score(zs['expected_reward'], random_v, oracle_v):.3f}")

    curve = trainer.train(contexts, reward_cfg, random_v, oracle_v)

    grpo_final = evaluate_action_fn(contexts, lambda c: trainer.select_action(c, reward_cfg), reward_cfg)
    summary.append(_row("qwen_grpo", grpo_final))
    summary.append(_row("oracle", ora))

    _write_dicts(out_dir / "summary.csv", summary)
    _write_dicts(out_dir / "qwen_grpo_curve.csv", [{"seed": args.seed, **c} for c in curve])

    print("\npolicy                norm[0..1]  exp_reward  success  latency_s  timeout")
    order = ["random", "latency_greedy", "quality_greedy", "reliability_greedy",
             "qwen_zeroshot", "qwen_grpo", "tabular_grpo_rl", "oracle"]
    by_name = {r["policy"]: r for r in summary}
    for name in order:
        if name in by_name:
            r = by_name[name]
            print(f"{name:20s} {r['normalized_score']:10.3f} {r['expected_reward']:11.3f} "
                  f"{r['success_prob']:8.3f} {r['latency_s']:10.3f} {r['timeout_prob']:8.3f}")
    frac_div = trainer.diverse_steps / max(trainer.train_steps, 1)
    mean_uniq = trainer.uniq_sum / max(trainer.train_steps, 1)
    print(f"\nparse_failures during training: {trainer.parse_failures}")
    print(f"exploration: mean_uniq_actions/group={mean_uniq:.2f}, "
          f"steps_with_action_diversity={trainer.diverse_steps}/{trainer.train_steps} ({frac_div:.0%})")

    _plot(out_dir, curve, by_name)
    print(f"\nWrote results to {out_dir}")


def _plot(out_dir: Path, curve: Sequence[Dict[str, float]], by_name: Dict[str, Dict[str, float]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    steps = [c["step"] for c in curve]
    norm = [c["normalized_score"] for c in curve]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, norm, label="qwen_grpo (LoRA, learned)", color="tab:blue", linewidth=2, marker="o")
    if "qwen_zeroshot" in by_name:
        ax.axhline(by_name["qwen_zeroshot"]["normalized_score"], color="tab:orange",
                   linewidth=2, label="qwen_zeroshot (same model, no training)")
    if "tabular_grpo_rl" in by_name:
        ax.axhline(by_name["tabular_grpo_rl"]["normalized_score"], color="tab:green",
                   linestyle="-.", label="tabular GRPO (reference)")
    for name, style in [("quality_greedy", ":"), ("latency_greedy", "--")]:
        if name in by_name:
            ax.axhline(by_name[name]["normalized_score"], color="gray", linestyle=style, alpha=0.7, label=name)
    ax.axhline(1.0, color="black", alpha=0.6, label="oracle")
    ax.axhline(0.0, color="gray", alpha=0.4, label="random")
    ax.set_xlabel("GRPO training steps")
    ax.set_ylabel("normalized score  (0 = random, 1 = oracle)")
    ax.set_title("Same Qwen: RL training (GRPO) vs zero-shot")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "qwen_grpo_curve.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
