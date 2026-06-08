"""
RL Tool Arena: a sound, self-contained study of cost-aware tool selection.

Research question
-----------------
When candidate tools present a genuine quality / latency / reliability
tradeoff, can an RL agent LEARN a cost-aware tool-selection policy from
reward alone (without being told tool costs) that beats latency-agnostic and
naive latency-greedy heuristics and approaches an oracle?

Why this is a sound RL setup (addresses common failure modes)
-------------------------------------------------------------
1. There is actual learning: a softmax policy is trained with a
   group-relative policy gradient -- the SAME advantage estimator
   (r - mean) / std used by training/grpo_trainer.py -- so it is a
   tractable analog of the LLM GRPO trainer in this repo.
2. The agent does NOT observe latent tool costs/quality. It must estimate
   tool value purely from sampled reward => a real RL problem, not sorting a
   number it was handed.
3. The environment is constructed so that NO fixed single-axis heuristic is
   optimal across contexts/weightings. We demonstrate this directly with a
   sweep over the latency weight w (see --mode sweep): latency-greedy only
   wins for large w, quality-greedy only for small w, and RL tracks the
   oracle across the whole range.

Honesty note
------------
The tool latency / quality / reliability values are SIMULATED. Task-context
frequencies are grounded in the real WildToolBench category distribution when
available, but the claim here is about the learning algorithm, not about
real-world tool latencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
@dataclass
class Tool:
    name: str
    quality: float        # P(useful output | not timed out)
    latency_s: float      # expected wall-clock latency in seconds
    timeout_prob: float   # P(timeout / hard failure)


@dataclass
class RewardConfig:
    r_correct: float = 1.0
    w_latency: float = 0.3
    r_timeout_penalty: float = 1.0


@dataclass
class Context:
    name: str
    prob: float
    tools: List[Tool]


def build_world(
    n_contexts: int,
    tools_per_context: int,
    seed: int,
    context_names: Optional[Sequence[str]] = None,
    context_probs: Optional[Sequence[float]] = None,
) -> List[Context]:
    """Construct tool sets with a genuine quality/latency/reliability tradeoff.

    Design choices that create real tension (no trivial optimum):
    - Higher-quality tools tend to be SLOWER (quality positively tied to latency).
    - FASTER tools tend to be FLAKIER (higher timeout probability).
    So "always fastest" buys speed but loses quality and reliability, while
    "always highest quality" pays a latency cost. The best choice depends on
    the context and on the latency weight w.
    """
    rng = np.random.default_rng(seed)
    contexts: List[Context] = []
    for c in range(n_contexts):
        cname = context_names[c] if context_names is not None and c < len(context_names) else f"ctx_{c}"
        cprob = float(context_probs[c]) if context_probs is not None and c < len(context_probs) else 1.0
        tools: List[Tool] = []
        for k in range(tools_per_context):
            quality = float(rng.uniform(0.45, 0.97))
            # latency grows with quality (slower = better) plus noise, clipped.
            latency = float(np.clip(0.15 + 1.7 * quality + rng.normal(0.0, 0.25), 0.1, 2.2))
            # faster tools are flakier; add per-tool noise so it isn't perfectly monotone.
            timeout = float(np.clip(0.28 * (1.0 - latency / 2.2) + rng.normal(0.0, 0.04), 0.0, 0.6))
            tools.append(Tool(name=f"{cname}_tool{k}", quality=quality, latency_s=latency, timeout_prob=timeout))
        contexts.append(Context(name=cname, prob=cprob, tools=tools))

    total = sum(c.prob for c in contexts)
    for c in contexts:
        c.prob = c.prob / total if total > 0 else 1.0 / len(contexts)
    return contexts


def expected_reward(tool: Tool, reward_cfg: RewardConfig) -> float:
    """Analytic expected reward for choosing `tool` (used for eval/oracle only)."""
    success_prob = (1.0 - tool.timeout_prob) * tool.quality
    return (
        reward_cfg.r_correct * success_prob
        - reward_cfg.w_latency * tool.latency_s
        - reward_cfg.r_timeout_penalty * tool.timeout_prob
    )


def sample_reward(tool: Tool, reward_cfg: RewardConfig, rng: np.random.Generator) -> Tuple[float, Dict[str, float]]:
    """Stochastic reward the LEARNER actually observes (no latent values exposed)."""
    timed_out = rng.random() < tool.timeout_prob
    success = (not timed_out) and (rng.random() < tool.quality)
    # Latency jitter; latency is incurred whether or not the call succeeds.
    latency = max(0.02, float(rng.normal(tool.latency_s, 0.1 * tool.latency_s)))
    reward = (
        reward_cfg.r_correct * float(success)
        - reward_cfg.w_latency * latency
        - reward_cfg.r_timeout_penalty * float(timed_out)
    )
    return reward, {"success": float(success), "latency_s": latency, "timed_out": float(timed_out)}


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
def oracle_action(ctx: Context, reward_cfg: RewardConfig) -> int:
    return int(np.argmax([expected_reward(t, reward_cfg) for t in ctx.tools]))


def latency_greedy_action(ctx: Context) -> int:
    return int(np.argmin([t.latency_s for t in ctx.tools]))


def quality_greedy_action(ctx: Context) -> int:
    return int(np.argmax([t.quality for t in ctx.tools]))


def reliability_greedy_action(ctx: Context) -> int:
    return int(np.argmin([t.timeout_prob for t in ctx.tools]))


# --------------------------------------------------------------------------- #
# Qwen-as-policy (a meaningful zero-shot LLM baseline inside the arena)
# --------------------------------------------------------------------------- #
def _latency_tier(latency_s: float) -> str:
    if latency_s < 0.6:
        return "very low latency (fast)"
    if latency_s < 1.2:
        return "moderate latency"
    return "high latency (slow but thorough)"


def _accuracy_tier(quality: float) -> str:
    if quality < 0.6:
        return "basic/approximate accuracy"
    if quality < 0.8:
        return "good accuracy"
    return "high accuracy"


def _reliability_tier(timeout_prob: float) -> str:
    if timeout_prob < 0.1:
        return "very reliable (rarely fails)"
    if timeout_prob < 0.25:
        return "occasionally unstable"
    return "frequently times out / unstable"


def tool_description(tool: Tool, reveal_numbers: bool) -> Dict[str, object]:
    """Realistic API-style description for an LLM.

    We expose estimated latency (typically advertised/measurable) plus
    qualitative accuracy and reliability tiers (as a real tool catalog might).
    With reveal_numbers=True we additionally expose exact latent values, which
    turns the task into 'can the LLM compute the argmax' (a diagnostic upper
    bound), rather than 'can it reason from a realistic description'.
    """
    desc: Dict[str, object] = {
        "name": tool.name,
        "estimated_latency_ms": round(tool.latency_s * 1000.0, 0),
        "accuracy": _accuracy_tier(tool.quality),
        "reliability": _reliability_tier(tool.timeout_prob),
    }
    if reveal_numbers:
        desc["success_quality_prob"] = round(tool.quality, 3)
        desc["timeout_prob"] = round(tool.timeout_prob, 3)
    return desc


def build_arena_qwen_messages(
    ctx: Context, reward_cfg: RewardConfig, reveal_numbers: bool
) -> List[Dict[str, str]]:
    tools = [tool_description(t, reveal_numbers) for t in ctx.tools]
    objective = (
        "Scoring for your choice (you will be judged on expected net value):\n"
        f"  +{reward_cfg.r_correct} for a correct/useful result\n"
        f"  -{reward_cfg.w_latency} per SECOND of latency\n"
        f"  -{reward_cfg.r_timeout_penalty} if the tool times out / fails\n"
        "A tool that times out yields no correct result AND incurs the timeout penalty.\n"
        "Pick the single tool that maximizes expected net value for this objective."
    )
    user = (
        f"Task category: {ctx.name}\n\n"
        f"Candidate tools (choose exactly one):\n{json.dumps(tools, ensure_ascii=False, indent=2)}\n\n"
        f"{objective}\n\n"
        'Return only JSON: {"tool": "<tool_name>"}'
    )
    return [
        {"role": "system", "content": "You are a cost-aware tool-selection policy. Return only JSON. Do not explain."},
        {"role": "user", "content": user},
    ]


class ArenaQwenPolicy:
    def __init__(self, model_name: str, device: str = "auto", dtype: str = "auto", max_new_tokens: int = 32):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "Qwen baseline requires torch and transformers in the active environment."
            ) from exc

        torch_dtype = "auto"
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
        )
        self.max_new_tokens = max_new_tokens
        self.parse_failures = 0
        self._cache: Dict[str, int] = {}

    def _cache_key(self, ctx: Context, reward_cfg: RewardConfig, reveal_numbers: bool) -> str:
        tool_sig = "|".join(f"{t.name}:{t.quality:.3f}:{t.latency_s:.3f}:{t.timeout_prob:.3f}" for t in ctx.tools)
        return f"{reward_cfg.w_latency:.4f}|{int(reveal_numbers)}|{ctx.name}|{tool_sig}"

    def select_action(self, ctx: Context, reward_cfg: RewardConfig, reveal_numbers: bool) -> int:
        key = self._cache_key(ctx, reward_cfg, reveal_numbers)
        if key in self._cache:
            return self._cache[key]

        messages = build_arena_qwen_messages(ctx, reward_cfg, reveal_numbers)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_device = next(self.model.parameters()).device
        encoded = self.tokenizer(prompt, return_tensors="pt").to(model_device)
        output_ids = self.model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen = output_ids[:, encoded["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(gen[0], skip_special_tokens=True).strip()

        action = self._parse_action(raw, ctx)
        self._cache[key] = action
        return action

    def _parse_action(self, raw: str, ctx: Context) -> int:
        names = [t.name for t in ctx.tools]
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
        if isinstance(chosen, str) and chosen in names:
            return names.index(chosen)
        # Fallback: substring match against the raw output.
        for i, name in enumerate(names):
            if name in raw:
                return i
        self.parse_failures += 1
        return 0  # deterministic fallback to first tool


@dataclass
class GRPOConfig:
    lr: float = 0.2
    group_size: int = 8
    total_steps: int = 4000
    eval_interval: int = 200
    adv_eps: float = 1e-6
    seed: int = 0


class TabularGRPOPolicy:
    """Softmax policy over tools per context, trained with group-relative
    advantage identical in spirit to training/grpo_trainer.py:
        advantages = (rewards - rewards.mean()) / rewards.std().clamp_min(eps)
    """

    def __init__(self, contexts: Sequence[Context], cfg: GRPOConfig):
        self.contexts = list(contexts)
        self.cfg = cfg
        self.n_ctx = len(self.contexts)
        self.max_actions = max(len(c.tools) for c in self.contexts)
        self.logits = np.zeros((self.n_ctx, self.max_actions), dtype=np.float64)
        # Mask invalid actions for ragged tool counts.
        self.valid = np.zeros((self.n_ctx, self.max_actions), dtype=bool)
        for i, c in enumerate(self.contexts):
            self.valid[i, : len(c.tools)] = True
        self.logits[~self.valid] = -1e9

    def _policy(self, ctx_idx: int) -> np.ndarray:
        z = self.logits[ctx_idx].copy()
        z[~self.valid[ctx_idx]] = -1e9
        z -= z.max()
        e = np.exp(z)
        e[~self.valid[ctx_idx]] = 0.0
        return e / e.sum()

    def greedy_action(self, ctx_idx: int) -> int:
        probs = self._policy(ctx_idx)
        return int(np.argmax(probs))

    def train(self, reward_cfg: RewardConfig) -> List[Dict[str, float]]:
        rng = np.random.default_rng(self.cfg.seed)
        ctx_probs = np.array([c.prob for c in self.contexts])
        history: List[Dict[str, float]] = []

        for step in range(1, self.cfg.total_steps + 1):
            ctx_idx = int(rng.choice(self.n_ctx, p=ctx_probs))
            ctx = self.contexts[ctx_idx]
            probs = self._policy(ctx_idx)

            # Sample a group of actions for this context (GRPO group).
            actions = rng.choice(self.max_actions, size=self.cfg.group_size, p=probs)
            rewards = np.array([sample_reward(ctx.tools[a], reward_cfg, rng)[0] for a in actions])

            # Group-relative advantage (mirrors grpo_trainer.py).
            adv = (rewards - rewards.mean()) / (rewards.std() + self.cfg.adv_eps)

            # REINFORCE gradient of log pi for a softmax over actions:
            #   d/dlogit_a logpi(a_g) = 1{a == a_g} - pi(a)
            grad = np.zeros(self.max_actions)
            for a_g, adv_g in zip(actions, adv):
                onehot = np.zeros(self.max_actions)
                onehot[a_g] = 1.0
                grad += adv_g * (onehot - probs)
            grad /= self.cfg.group_size

            self.logits[ctx_idx] += self.cfg.lr * grad
            self.logits[ctx_idx][~self.valid[ctx_idx]] = -1e9

            if step % self.cfg.eval_interval == 0 or step == 1:
                history.append({"step": float(step), **evaluate_policy_expected(self, reward_cfg)})

        return history


# --------------------------------------------------------------------------- #
# Evaluation (analytic expectations of the *greedy* policy => low variance)
# --------------------------------------------------------------------------- #
def _action_expectations(ctx: Context, action: int, reward_cfg: RewardConfig) -> Dict[str, float]:
    tool = ctx.tools[action]
    return {
        "expected_reward": expected_reward(tool, reward_cfg),
        "success_prob": (1.0 - tool.timeout_prob) * tool.quality,
        "latency_s": tool.latency_s,
        "timeout_prob": tool.timeout_prob,
    }


def evaluate_action_fn(contexts: Sequence[Context], action_fn, reward_cfg: RewardConfig) -> Dict[str, float]:
    agg = {"expected_reward": 0.0, "success_prob": 0.0, "latency_s": 0.0, "timeout_prob": 0.0}
    for ctx in contexts:
        stats = _action_expectations(ctx, action_fn(ctx), reward_cfg)
        for k in agg:
            agg[k] += ctx.prob * stats[k]
    return agg


def evaluate_policy_expected(policy: TabularGRPOPolicy, reward_cfg: RewardConfig) -> Dict[str, float]:
    agg = {"expected_reward": 0.0, "success_prob": 0.0, "latency_s": 0.0, "timeout_prob": 0.0}
    for i, ctx in enumerate(policy.contexts):
        stats = _action_expectations(ctx, policy.greedy_action(i), reward_cfg)
        for k in agg:
            agg[k] += ctx.prob * stats[k]
    return agg


def normalized_score(value: float, random_v: float, oracle_v: float) -> float:
    denom = oracle_v - random_v
    if abs(denom) < 1e-9:
        return 0.0
    return (value - random_v) / denom


# --------------------------------------------------------------------------- #
# WildToolBench grounding (optional)
# --------------------------------------------------------------------------- #
def wtb_category_distribution(dataset_path: str) -> Tuple[List[str], List[float]]:
    from collections import Counter
    from data.toolbench_loader import load_tasks_from_benchmark

    cfg = {
        "data": {
            "benchmark": "wild_tool_bench",
            "wild_tool_bench_path": dataset_path,
            "strict_benchmark_loading": True,
            "allow_synthetic_fallback": False,
            "min_tools_per_task": 2,
        }
    }
    tasks = load_tasks_from_benchmark(cfg, split="train")
    counts = Counter(str(t.get("category", "Unknown")) for t in tasks)
    names = sorted(counts)
    total = sum(counts.values())
    probs = [counts[n] / total for n in names]
    return names, probs


# --------------------------------------------------------------------------- #
# Experiment drivers
# --------------------------------------------------------------------------- #
@dataclass
class FinalRow:
    policy: str
    w_latency: float
    seed: int
    expected_reward: float
    normalized_score: float
    success_prob: float
    latency_s: float
    timeout_prob: float


def baseline_action_fns():
    return {
        "random": None,  # handled specially (mean over actions)
        "latency_greedy": latency_greedy_action,
        "quality_greedy": quality_greedy_action,
        "reliability_greedy": reliability_greedy_action,
        "oracle": None,  # handled specially
    }


def evaluate_all_policies(
    contexts: List[Context],
    reward_cfg: RewardConfig,
    grpo_cfg: GRPOConfig,
    qwen_policy: "Optional[ArenaQwenPolicy]" = None,
    qwen_reveal_numbers: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    # Random = uniform over actions -> mean expected reward per context.
    def random_eval() -> Dict[str, float]:
        agg = {"expected_reward": 0.0, "success_prob": 0.0, "latency_s": 0.0, "timeout_prob": 0.0}
        for ctx in contexts:
            stats = [_action_expectations(ctx, a, reward_cfg) for a in range(len(ctx.tools))]
            for k in agg:
                agg[k] += ctx.prob * float(np.mean([s[k] for s in stats]))
        return agg

    results: Dict[str, Dict[str, float]] = {}
    results["random"] = random_eval()
    results["latency_greedy"] = evaluate_action_fn(contexts, latency_greedy_action, reward_cfg)
    results["quality_greedy"] = evaluate_action_fn(contexts, quality_greedy_action, reward_cfg)
    results["reliability_greedy"] = evaluate_action_fn(contexts, reliability_greedy_action, reward_cfg)
    results["oracle"] = evaluate_action_fn(contexts, lambda c: oracle_action(c, reward_cfg), reward_cfg)

    if qwen_policy is not None:
        label = "qwen_full_info" if qwen_reveal_numbers else "qwen_zeroshot"
        results[label] = evaluate_action_fn(
            contexts,
            lambda c: qwen_policy.select_action(c, reward_cfg, qwen_reveal_numbers),
            reward_cfg,
        )

    policy = TabularGRPOPolicy(contexts, grpo_cfg)
    curve = policy.train(reward_cfg)
    results["grpo_rl"] = evaluate_policy_expected(policy, reward_cfg)
    return results, curve


def _maybe_build_qwen(args) -> "Optional[ArenaQwenPolicy]":
    if not getattr(args, "include_qwen", False):
        return None
    print(f"Loading Qwen baseline: {args.qwen_model} (reveal_numbers={bool(args.qwen_reveal_numbers)})")
    return ArenaQwenPolicy(
        model_name=args.qwen_model,
        device=args.qwen_device,
        dtype=args.qwen_dtype,
    )


def run_train_mode(args) -> None:
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

    qwen_policy = _maybe_build_qwen(args)
    final_rows: List[FinalRow] = []
    curves: List[Dict[str, float]] = []
    for seed in args.seeds:
        contexts = build_world(
            n_contexts=args.n_contexts,
            tools_per_context=args.tools_per_context,
            seed=seed,
            context_names=context_names,
            context_probs=context_probs,
        )
        reward_cfg = RewardConfig(
            r_correct=args.r_correct, w_latency=args.w_latency, r_timeout_penalty=args.r_timeout
        )
        grpo_cfg = GRPOConfig(
            lr=args.lr,
            group_size=args.group_size,
            total_steps=args.total_steps,
            eval_interval=args.eval_interval,
            seed=seed,
        )
        results, curve = evaluate_all_policies(
            contexts, reward_cfg, grpo_cfg, qwen_policy=qwen_policy, qwen_reveal_numbers=args.qwen_reveal_numbers
        )

        random_v = results["random"]["expected_reward"]
        oracle_v = results["oracle"]["expected_reward"]
        for policy_name, stats in results.items():
            final_rows.append(
                FinalRow(
                    policy=policy_name,
                    w_latency=args.w_latency,
                    seed=seed,
                    expected_reward=stats["expected_reward"],
                    normalized_score=normalized_score(stats["expected_reward"], random_v, oracle_v),
                    success_prob=stats["success_prob"],
                    latency_s=stats["latency_s"],
                    timeout_prob=stats["timeout_prob"],
                )
            )
        for point in curve:
            curves.append(
                {
                    "seed": seed,
                    "step": point["step"],
                    "rl_expected_reward": point["expected_reward"],
                    "rl_normalized_score": normalized_score(point["expected_reward"], random_v, oracle_v),
                }
            )

    _write_csv(out_dir / "final_metrics.csv", final_rows)
    _write_dicts_csv(out_dir / "learning_curve.csv", curves)
    _print_final_summary(final_rows)
    _maybe_plot_training(out_dir, final_rows, curves)
    print(f"\nWrote results to {out_dir}")


def run_sweep_mode(args) -> None:
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

    qwen_policy = _maybe_build_qwen(args)
    weights = [round(w, 4) for w in np.linspace(args.w_min, args.w_max, args.w_steps)]
    rows: List[FinalRow] = []
    for w in weights:
        for seed in args.seeds:
            contexts = build_world(
                n_contexts=args.n_contexts,
                tools_per_context=args.tools_per_context,
                seed=seed,
                context_names=context_names,
                context_probs=context_probs,
            )
            reward_cfg = RewardConfig(r_correct=args.r_correct, w_latency=w, r_timeout_penalty=args.r_timeout)
            grpo_cfg = GRPOConfig(
                lr=args.lr,
                group_size=args.group_size,
                total_steps=args.total_steps,
                eval_interval=max(args.total_steps, 1),
                seed=seed,
            )
            results, _ = evaluate_all_policies(
                contexts, reward_cfg, grpo_cfg, qwen_policy=qwen_policy, qwen_reveal_numbers=args.qwen_reveal_numbers
            )
            random_v = results["random"]["expected_reward"]
            oracle_v = results["oracle"]["expected_reward"]
            for policy_name, stats in results.items():
                rows.append(
                    FinalRow(
                        policy=policy_name,
                        w_latency=w,
                        seed=seed,
                        expected_reward=stats["expected_reward"],
                        normalized_score=normalized_score(stats["expected_reward"], random_v, oracle_v),
                        success_prob=stats["success_prob"],
                        latency_s=stats["latency_s"],
                        timeout_prob=stats["timeout_prob"],
                    )
                )

    _write_csv(out_dir / "sweep_metrics.csv", rows)
    _print_sweep_summary(rows, weights)
    _maybe_plot_sweep(out_dir, rows, weights)
    print(f"\nWrote sweep results to {out_dir}")


# --------------------------------------------------------------------------- #
# IO + reporting helpers
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, rows: Sequence[FinalRow]) -> None:
    if not rows:
        return
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_dicts_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_by_policy(rows: Sequence[FinalRow], key: str) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for row in rows:
        acc.setdefault(row.policy, []).append(getattr(row, key))
    return {p: float(np.mean(v)) for p, v in acc.items()}


def _print_final_summary(rows: Sequence[FinalRow]) -> None:
    reward = _mean_by_policy(rows, "expected_reward")
    norm = _mean_by_policy(rows, "normalized_score")
    success = _mean_by_policy(rows, "success_prob")
    latency = _mean_by_policy(rows, "latency_s")
    timeout = _mean_by_policy(rows, "timeout_prob")
    order = [
        "random",
        "latency_greedy",
        "quality_greedy",
        "reliability_greedy",
        "qwen_zeroshot",
        "qwen_full_info",
        "grpo_rl",
        "oracle",
    ]
    print(f"\n{'policy':20s} {'exp_reward':>11s} {'norm[0..1]':>11s} {'success':>9s} {'latency_s':>10s} {'timeout':>9s}")
    for p in order:
        if p not in reward:
            continue
        print(
            f"{p:20s} {reward[p]:11.3f} {norm[p]:11.3f} {success[p]:9.3f} {latency[p]:10.3f} {timeout[p]:9.3f}"
        )


def _print_sweep_summary(rows: Sequence[FinalRow], weights: Sequence[float]) -> None:
    print("\nBest policy by latency weight w (normalized score, mean over seeds):")
    has_qwen = any(r.policy in ("qwen_zeroshot", "qwen_full_info") for r in rows)
    qwen_label = "qwen_full_info" if any(r.policy == "qwen_full_info" for r in rows) else "qwen_zeroshot"
    header = f"{'w':>7s}  {'best_fixed_heuristic':24s} {'rl_norm':>8s} {'best_fixed_norm':>16s}"
    if has_qwen:
        header += f" {'qwen_norm':>10s}"
    print(header)
    for w in weights:
        wr = [r for r in rows if abs(r.w_latency - w) < 1e-9]
        by_policy: Dict[str, List[float]] = {}
        for r in wr:
            by_policy.setdefault(r.policy, []).append(r.normalized_score)
        means = {p: float(np.mean(v)) for p, v in by_policy.items()}
        fixed = {p: means[p] for p in ("latency_greedy", "quality_greedy", "reliability_greedy") if p in means}
        best_fixed = max(fixed, key=fixed.get)
        line = f"{w:7.3f}  {best_fixed:24s} {means.get('grpo_rl', float('nan')):8.3f} {fixed[best_fixed]:16.3f}"
        if has_qwen:
            line += f" {means.get(qwen_label, float('nan')):10.3f}"
        print(line)


def _maybe_plot_training(out_dir: Path, rows: Sequence[FinalRow], curves: Sequence[Dict[str, float]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    # Learning curve (mean over seeds).
    steps = sorted({c["step"] for c in curves})
    rl_norm = [float(np.mean([c["rl_normalized_score"] for c in curves if c["step"] == s])) for s in steps]

    norm = _mean_by_policy(rows, "normalized_score")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, rl_norm, label="GRPO RL (learned)", color="tab:blue", linewidth=2)
    for p, style in [
        ("latency_greedy", "--"),
        ("quality_greedy", "-."),
        ("reliability_greedy", ":"),
        ("random", "--"),
    ]:
        if p in norm:
            ax.axhline(norm[p], linestyle=style, color="gray", alpha=0.8, label=f"{p}")
    for qlabel, color in [("qwen_zeroshot", "tab:orange"), ("qwen_full_info", "tab:brown")]:
        if qlabel in norm:
            ax.axhline(norm[qlabel], linestyle="-", color=color, alpha=0.9, linewidth=2, label=qlabel)
    ax.axhline(1.0, linestyle="-", color="black", alpha=0.6, label="oracle")
    ax.set_xlabel("training steps")
    ax.set_ylabel("normalized score  (0 = random, 1 = oracle)")
    ax.set_title("Learning cost-aware tool selection from reward")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=140)
    plt.close(fig)


def _maybe_plot_sweep(out_dir: Path, rows: Sequence[FinalRow], weights: Sequence[float]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    def series(policy: str) -> List[float]:
        out = []
        for w in weights:
            vals = [r.normalized_score for r in rows if r.policy == policy and abs(r.w_latency - w) < 1e-9]
            out.append(float(np.mean(vals)) if vals else float("nan"))
        return out

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(weights, series("grpo_rl"), label="GRPO RL (learned)", color="tab:blue", linewidth=2, marker="o")
    ax.plot(weights, series("latency_greedy"), label="latency-greedy", color="tab:red", linestyle="--", marker="s")
    ax.plot(weights, series("quality_greedy"), label="quality-greedy", color="tab:green", linestyle="-.", marker="^")
    ax.plot(weights, series("reliability_greedy"), label="reliability-greedy", color="tab:purple", linestyle=":", marker="v")
    for qlabel, color in [("qwen_zeroshot", "tab:orange"), ("qwen_full_info", "tab:brown")]:
        if any(r.policy == qlabel for r in rows):
            ax.plot(weights, series(qlabel), label=qlabel.replace("_", "-"), color=color, linewidth=2, marker="D")
    ax.axhline(1.0, color="black", alpha=0.6, label="oracle")
    ax.axhline(0.0, color="gray", alpha=0.5, label="random")
    ax.set_xlabel("latency weight  w")
    ax.set_ylabel("normalized score  (0 = random, 1 = oracle)")
    ax.set_title("No fixed heuristic is optimal across w; RL adapts")
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_curve.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["train", "sweep"], default="train")
    parser.add_argument("--dataset_path", default=None, help="Optional WildToolBench jsonl to ground context frequencies.")
    parser.add_argument("--n_contexts", type=int, default=12)
    parser.add_argument("--tools_per_context", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])

    parser.add_argument("--r_correct", type=float, default=1.0)
    parser.add_argument("--w_latency", type=float, default=0.3)
    parser.add_argument("--r_timeout", type=float, default=1.0)

    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--total_steps", type=int, default=4000)
    parser.add_argument("--eval_interval", type=int, default=200)

    parser.add_argument("--w_min", type=float, default=0.0)
    parser.add_argument("--w_max", type=float, default=1.0)
    parser.add_argument("--w_steps", type=int, default=11)

    parser.add_argument("--include_qwen", action="store_true", help="Evaluate a zero-shot Qwen policy inside the arena.")
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--qwen_device", default="auto")
    parser.add_argument("--qwen_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument(
        "--qwen_reveal_numbers",
        action="store_true",
        help="Expose exact latent quality/timeout to Qwen (diagnostic upper bound instead of realistic descriptions).",
    )

    parser.add_argument("--output_dir", default="experiments/results/rl_tool_arena")
    args = parser.parse_args()

    if args.mode == "train":
        run_train_mode(args)
    else:
        run_sweep_mode(args)


if __name__ == "__main__":
    main()
