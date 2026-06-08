from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.toolbench_loader import load_tasks_from_benchmark


@dataclass(frozen=True)
class Condition:
    name: str
    latency_multiplier: float
    timeout_rate: float
    timeout_threshold_ms: float


@dataclass
class EpisodeMetrics:
    policy: str
    condition: str
    seed: int
    task_idx: int
    category: str
    n_available_tools: int
    n_required_tools: int
    n_tool_calls: int
    required_tool_recall: float
    required_tool_hit: float
    all_required_tools_hit: float
    mean_latency_ms: float
    total_latency_ms: float
    timeout_rate: float
    timeout_events: int
    proxy_utility: float


@dataclass
class SummaryMetrics:
    policy: str
    condition: str
    seed: int
    n_tasks: int
    mean_required_tool_recall: float
    required_tool_hit_rate: float
    all_required_tools_hit_rate: float
    mean_tool_calls: float
    mean_latency_ms: float
    mean_total_latency_ms: float
    timeout_rate: float
    mean_proxy_utility: float


@dataclass
class QwenAction:
    policy: str
    condition: str
    task_idx: int
    selected_tools: List[str]
    raw_output: str


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(base_path: str, override_path: str | None) -> Dict[str, Any]:
    base_cfg = yaml.safe_load(Path(base_path).read_text()) or {}
    if not override_path:
        return base_cfg
    override_cfg = yaml.safe_load(Path(override_path).read_text()) or {}
    return _deep_merge(base_cfg, override_cfg)


def _stable_bucket(name: str, n_buckets: int) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_buckets


def assign_tool_latencies(tasks: Sequence[Dict[str, Any]], profile: str = "hash") -> Dict[str, float]:
    """Assign deterministic synthetic latency because WildToolBench does not annotate cost."""
    tool_names = sorted({tool for task in tasks for tool in task.get("available_tools", [])})
    if profile == "flat":
        return {name: 500.0 for name in tool_names}

    buckets = [120.0, 250.0, 500.0, 900.0, 1500.0]
    return {name: buckets[_stable_bucket(name, len(buckets))] for name in tool_names}


def choose_tools(
    policy: str,
    task: Dict[str, Any],
    latencies_ms: Dict[str, float],
    rng: random.Random,
    max_calls: int,
) -> List[str]:
    available = list(task.get("available_tools", []))
    if not available or max_calls <= 0:
        return []

    n_calls = min(max_calls, len(available))
    if policy == "random":
        return rng.sample(available, k=n_calls)
    if policy == "latency_aware":
        return sorted(available, key=lambda name: (latencies_ms.get(name, 500.0), name))[:n_calls]
    if policy == "slow":
        return sorted(available, key=lambda name: (latencies_ms.get(name, 500.0), name), reverse=True)[:n_calls]
    if policy == "gold_oracle":
        required = [name for name in task.get("required_tools", []) if name in available]
        required = sorted(required, key=lambda name: (latencies_ms.get(name, 500.0), name))
        extras = [
            name
            for name in sorted(available, key=lambda name: (latencies_ms.get(name, 500.0), name))
            if name not in set(required)
        ]
        return (required + extras)[:n_calls]
    raise ValueError(f"Unknown policy: {policy}")


def _tool_details_for_prompt(
    task: Dict[str, Any],
    latencies_ms: Dict[str, float],
    condition: Condition,
    include_latency: bool,
) -> List[Dict[str, Any]]:
    details_by_name = {
        item.get("name"): item
        for item in task.get("tool_details", [])
        if isinstance(item, dict) and item.get("name")
    }
    prompt_tools: List[Dict[str, Any]] = []
    for name in task.get("available_tools", []):
        detail = details_by_name.get(name, {})
        tool = {
            "name": name,
            "description": detail.get("description", ""),
        }
        if include_latency:
            tool["estimated_latency_ms"] = round(latencies_ms.get(name, 500.0) * condition.latency_multiplier, 1)
            tool["timeout_risk"] = condition.timeout_rate
        prompt_tools.append(tool)
    return prompt_tools


def build_qwen_prompt(
    task: Dict[str, Any],
    latencies_ms: Dict[str, float],
    condition: Condition,
    max_calls: int,
    include_latency: bool,
) -> List[Dict[str, str]]:
    prompt_tools = _tool_details_for_prompt(task, latencies_ms, condition, include_latency)
    latency_instruction = (
        "Prefer tools that can answer the task with lower estimated latency and lower timeout risk. "
        "Do not choose a fast tool if it is irrelevant."
        if include_latency
        else "Choose the tools that are most relevant to solving the task."
    )
    user_content = (
        f"Task:\n{task.get('query', '')}\n\n"
        f"Available tools:\n{json.dumps(prompt_tools, ensure_ascii=False, indent=2)}\n\n"
        f"Select up to {max_calls} tool names from the available tools. {latency_instruction}\n"
        'Return only valid JSON in this exact shape: {"tools": ["tool_name"]}'
    )
    return [
        {
            "role": "system",
            "content": "You are a careful tool-selection policy. Return only JSON. Do not explain.",
        },
        {"role": "user", "content": user_content},
    ]


def parse_selected_tools(raw_output: str, available_tools: Sequence[str], max_calls: int) -> List[str]:
    available = set(available_tools)
    candidates: List[str] = []
    text = raw_output.strip()
    json_text = text
    if "{" in text and "}" in text:
        json_text = text[text.find("{") : text.rfind("}") + 1]
    elif "[" in text and "]" in text:
        json_text = text[text.find("[") : text.rfind("]") + 1]

    try:
        payload = json.loads(json_text)
        if isinstance(payload, dict):
            raw_tools = (
                payload.get("tools")
                or payload.get("tool_names")
                or payload.get("selected_tools")
                or payload.get("tool")
                or []
            )
        else:
            raw_tools = payload
        if isinstance(raw_tools, str):
            raw_tools = [raw_tools]
        if isinstance(raw_tools, list):
            candidates = [str(item).strip() for item in raw_tools if str(item).strip()]
    except Exception:
        candidates = []

    selected: List[str] = []
    for name in candidates:
        if name not in available or name in selected:
            continue
        selected.append(name)
        if len(selected) >= max_calls:
            break
    return selected


class QwenToolSelector:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 96,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Qwen policies require torch and transformers. Activate the venv and install dependencies."
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
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens

    def select_tools(
        self,
        task: Dict[str, Any],
        latencies_ms: Dict[str, float],
        condition: Condition,
        max_calls: int,
        include_latency: bool,
    ) -> tuple[List[str], str]:
        messages = build_qwen_prompt(
            task=task,
            latencies_ms=latencies_ms,
            condition=condition,
            max_calls=max_calls,
            include_latency=include_latency,
        )
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_device = next(self.model.parameters()).device
        encoded = self.tokenizer(prompt, return_tensors="pt").to(model_device)
        output_ids = self.model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[:, encoded["input_ids"].shape[1] :]
        raw_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        selected = parse_selected_tools(
            raw_output,
            available_tools=task.get("available_tools", []),
            max_calls=max_calls,
        )
        return selected, raw_output


def build_qwen_action_cache(
    *,
    policies: Sequence[str],
    tasks: Sequence[Dict[str, Any]],
    conditions: Sequence[Condition],
    latencies_ms: Dict[str, float],
    max_calls: int,
    model_name: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    output_dir: Path,
) -> Dict[tuple[str, str, int], List[str]]:
    qwen_policies = [policy for policy in policies if policy.startswith("qwen_")]
    if not qwen_policies:
        return {}

    selector = QwenToolSelector(
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
    )
    actions: List[QwenAction] = []
    cache: Dict[tuple[str, str, int], List[str]] = {}
    for condition in conditions:
        for task_idx, task in enumerate(tasks):
            for policy in qwen_policies:
                include_latency = policy == "qwen_latency"
                selected, raw_output = selector.select_tools(
                    task=task,
                    latencies_ms=latencies_ms,
                    condition=condition,
                    max_calls=max_calls,
                    include_latency=include_latency,
                )
                cache[(policy, condition.name, task_idx)] = selected
                actions.append(
                    QwenAction(
                        policy=policy,
                        condition=condition.name,
                        task_idx=task_idx,
                        selected_tools=selected,
                        raw_output=raw_output,
                    )
                )
                print(
                    f"qwen_action condition={condition.name} policy={policy} "
                    f"task={task_idx + 1}/{len(tasks)} selected={selected}"
                )

    (output_dir / "qwen_actions.json").write_text(json.dumps([asdict(action) for action in actions], indent=2))
    return cache


def simulate_episode(
    *,
    policy: str,
    condition: Condition,
    seed: int,
    task_idx: int,
    task: Dict[str, Any],
    latencies_ms: Dict[str, float],
    max_calls: int,
    latency_penalty_per_second: float,
    timeout_penalty: float,
    selected_tools: Optional[List[str]] = None,
) -> EpisodeMetrics:
    rng = random.Random(f"{seed}:{condition.name}:{policy}:{task_idx}")
    chosen_tools = selected_tools
    if chosen_tools is None:
        chosen_tools = choose_tools(policy, task, latencies_ms, rng, max_calls=max_calls)

    observed_latencies: List[float] = []
    timeout_events = 0
    for tool_name in chosen_tools:
        base_latency = latencies_ms.get(tool_name, 500.0)
        jitter = max(0.05, rng.gauss(1.0, 0.15))
        latency = base_latency * condition.latency_multiplier * jitter
        timed_out = latency > condition.timeout_threshold_ms or rng.random() < condition.timeout_rate
        observed_latencies.append(latency)
        timeout_events += int(timed_out)

    required = set(task.get("required_tools", []))
    used = set(chosen_tools)
    if required:
        required_recall = len(required & used) / len(required)
        required_hit = float(bool(required & used))
        all_required_hit = float(required.issubset(used))
    else:
        required_recall = 1.0
        required_hit = 1.0
        all_required_hit = 1.0

    total_latency = sum(observed_latencies)
    timeout_rate = timeout_events / max(1, len(chosen_tools))
    proxy_utility = (
        required_recall
        - latency_penalty_per_second * (total_latency / 1000.0)
        - timeout_penalty * timeout_events
    )

    return EpisodeMetrics(
        policy=policy,
        condition=condition.name,
        seed=seed,
        task_idx=task_idx,
        category=str(task.get("category", "Unknown")),
        n_available_tools=len(task.get("available_tools", [])),
        n_required_tools=len(required),
        n_tool_calls=len(chosen_tools),
        required_tool_recall=required_recall,
        required_tool_hit=required_hit,
        all_required_tools_hit=all_required_hit,
        mean_latency_ms=(total_latency / max(1, len(observed_latencies))) if observed_latencies else 0.0,
        total_latency_ms=total_latency,
        timeout_rate=timeout_rate,
        timeout_events=timeout_events,
        proxy_utility=proxy_utility,
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def summarize(episodes: Sequence[EpisodeMetrics]) -> SummaryMetrics:
    if not episodes:
        raise ValueError("Cannot summarize zero episodes")
    first = episodes[0]
    total_calls = sum(ep.n_tool_calls for ep in episodes)
    total_timeouts = sum(ep.timeout_events for ep in episodes)
    return SummaryMetrics(
        policy=first.policy,
        condition=first.condition,
        seed=first.seed,
        n_tasks=len(episodes),
        mean_required_tool_recall=_mean(ep.required_tool_recall for ep in episodes),
        required_tool_hit_rate=_mean(ep.required_tool_hit for ep in episodes),
        all_required_tools_hit_rate=_mean(ep.all_required_tools_hit for ep in episodes),
        mean_tool_calls=_mean(ep.n_tool_calls for ep in episodes),
        mean_latency_ms=_mean(ep.mean_latency_ms for ep in episodes),
        mean_total_latency_ms=_mean(ep.total_latency_ms for ep in episodes),
        timeout_rate=total_timeouts / max(1, total_calls),
        mean_proxy_utility=_mean(ep.proxy_utility for ep in episodes),
    )


def parse_condition(raw: str) -> Condition:
    parts = raw.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid condition '{raw}'. Expected name:latency_multiplier:timeout_rate:timeout_threshold_ms"
        )
    name, multiplier, timeout_rate, threshold = parts
    return Condition(
        name=name,
        latency_multiplier=float(multiplier),
        timeout_rate=float(timeout_rate),
        timeout_threshold_ms=float(threshold),
    )


def write_csv(path: Path, rows: Sequence[Any]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run small latency-aware policy ablations on normalized tool-use tasks."
    )
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument("--override_config", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--benchmark", default="wild_tool_bench", choices=["wild_tool_bench", "catp_llm"])
    parser.add_argument("--n_tasks", type=int, default=256)
    parser.add_argument("--max_calls", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["random", "latency_aware", "slow", "gold_oracle"],
        choices=["random", "latency_aware", "slow", "gold_oracle", "qwen_no_latency", "qwen_latency"],
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "normal:1.0:0.0:5000",
            "degraded_2x:2.0:0.0:5000",
            "degraded_5x_timeout:5.0:0.15:5000",
        ],
        help="Condition specs: name:latency_multiplier:timeout_rate:timeout_threshold_ms",
    )
    parser.add_argument("--latency_profile", choices=["hash", "flat"], default="hash")
    parser.add_argument("--latency_penalty_per_second", type=float, default=0.03)
    parser.add_argument("--timeout_penalty", type=float, default=0.5)
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--qwen_device", default="auto")
    parser.add_argument("--qwen_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--qwen_max_new_tokens", type=int, default=96)
    parser.add_argument("--output_dir", default="experiments/results/wtb_latency_ablation")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override_config)
    cfg.setdefault("data", {})
    cfg["data"]["benchmark"] = args.benchmark
    cfg["data"]["strict_benchmark_loading"] = True
    cfg["data"]["allow_synthetic_fallback"] = False
    if args.dataset_path:
        if args.benchmark == "wild_tool_bench":
            cfg["data"]["wild_tool_bench_path"] = args.dataset_path
        else:
            cfg["data"]["catp_llm_path"] = args.dataset_path

    tasks, diagnostics = load_tasks_from_benchmark(cfg, split="train", return_diagnostics=True)
    tasks = tasks[: args.n_tasks]
    if not tasks:
        raise RuntimeError("No tasks loaded for experiment.")

    conditions = [parse_condition(raw) for raw in args.conditions]
    latencies_ms = assign_tool_latencies(tasks, profile=args.latency_profile)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qwen_action_cache = build_qwen_action_cache(
        policies=args.policies,
        tasks=tasks,
        conditions=conditions,
        latencies_ms=latencies_ms,
        max_calls=args.max_calls,
        model_name=args.qwen_model,
        device=args.qwen_device,
        dtype=args.qwen_dtype,
        max_new_tokens=args.qwen_max_new_tokens,
        output_dir=output_dir,
    )

    all_episodes: List[EpisodeMetrics] = []
    summaries: List[SummaryMetrics] = []
    for seed in args.seeds:
        for condition in conditions:
            for policy in args.policies:
                episodes = [
                    simulate_episode(
                        policy=policy,
                        condition=condition,
                        seed=seed,
                        task_idx=task_idx,
                        task=task,
                        latencies_ms=latencies_ms,
                        max_calls=args.max_calls,
                        latency_penalty_per_second=args.latency_penalty_per_second,
                        timeout_penalty=args.timeout_penalty,
                        selected_tools=qwen_action_cache.get((policy, condition.name, task_idx)),
                    )
                    for task_idx, task in enumerate(tasks)
                ]
                all_episodes.extend(episodes)
                summaries.append(summarize(episodes))

    write_csv(output_dir / "episodes.csv", all_episodes)
    write_csv(output_dir / "summary.csv", summaries)
    (output_dir / "summary.json").write_text(json.dumps([asdict(row) for row in summaries], indent=2))
    (output_dir / "diagnostics.json").write_text(
        json.dumps(
            {
                "data": diagnostics,
                "n_tasks_used": len(tasks),
                "n_unique_tools": len(latencies_ms),
                "latency_profile": args.latency_profile,
                "policies": args.policies,
                "qwen_model": args.qwen_model if any(policy.startswith("qwen_") for policy in args.policies) else None,
                "conditions": [asdict(condition) for condition in conditions],
                "max_calls": args.max_calls,
                "seeds": args.seeds,
            },
            indent=2,
        )
    )

    print(f"Wrote experiment results to {output_dir}")
    for row in summaries:
        print(
            f"{row.condition:20s} {row.policy:14s} seed={row.seed} "
            f"recall={row.mean_required_tool_recall:.3f} "
            f"hit={row.required_tool_hit_rate:.3f} "
            f"latency={row.mean_total_latency_ms:.1f}ms "
            f"timeout={row.timeout_rate:.3f} "
            f"utility={row.mean_proxy_utility:.3f}"
        )


if __name__ == "__main__":
    main()
