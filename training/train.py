from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.api_registry import APIRegistry, ToolSpec
from data.toolbench_loader import load_tasks_from_benchmark, split_by_category
from environment.api_executor import APIExecutor
from environment.latency_tracker import LatencyTracker
from environment.tool_env import ToolEnv
from model.policy import PolicyModel
from training.grpo_trainer import OnlineGRPOTrainer


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(base_path: str, override_path: str | None) -> Dict[str, Any]:
    base_cfg = yaml.safe_load(Path(base_path).read_text()) or {}
    if not override_path:
        return base_cfg
    override_cfg = yaml.safe_load(Path(override_path).read_text()) or {}
    return _deep_merge(base_cfg, override_cfg)


def build_registry(tasks: list[dict]) -> APIRegistry:
    registry = APIRegistry()
    seen = set()
    for task in tasks:
        cat = task.get("category", "Unknown")
        for tool_name in task.get("available_tools", []):
            if tool_name in seen:
                continue
            seen.add(tool_name)
            default_latency = 120.0 if "fast" in tool_name.lower() else 900.0 if "slow" in tool_name.lower() else 450.0
            registry.register(
                ToolSpec(
                    name=tool_name,
                    description=f"{tool_name} ({cat})",
                    schema={"type": "object", "properties": {"query": {"type": "string"}}},
                    category=cat,
                    default_latency_ms=default_latency,
                )
            )
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument("--override_config", default=None)
    parser.add_argument("--output_dir", default="checkpoints/latency_aware_rl")
    parser.add_argument("--debug_steps", type=int, default=0)
    parser.add_argument("--print_data_diagnostics", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.override_config)
    tasks, diagnostics = load_tasks_from_benchmark(config, split="train", return_diagnostics=True)
    splits = split_by_category(tasks, config)
    train_tasks = splits["train"] if splits["train"] else tasks
    if args.print_data_diagnostics:
        print(json.dumps(diagnostics, indent=2, sort_keys=True))
        print(
            json.dumps(
                {
                    "n_train_split_tasks": len(train_tasks),
                    "n_test_split_tasks": len(splits["test"]),
                    "n_other_split_tasks": len(splits["other"]),
                },
                indent=2,
                sort_keys=True,
            )
        )

    registry = build_registry(train_tasks)
    tracker = LatencyTracker()
    env = ToolEnv(
        api_registry=registry,
        executor=APIExecutor(
            registry,
            mode=config["environment"].get("executor_mode", "replay"),
            trace_dir=config["environment"].get("trace_dir", "data/latency_traces"),
            call_log_dir=config["environment"].get("call_log_dir", "data/call_logs"),
            timeout_threshold_ms=int(config["environment"].get("timeout_threshold_ms", 5000)),
        ),
        latency_tracker=tracker,
        reward_config=config,
        max_steps=int(config["training"].get("max_steps_per_episode", 8)),
    )
    policy = PolicyModel(
        config["model"]["name"],
        seed=int(config["training"].get("random_seed", 42)),
        training_cfg=config.get("training", {}),
    )
    trainer = OnlineGRPOTrainer(config=config, policy=policy, env=env, registry=registry)

    total_steps = args.debug_steps or int(config["training"].get("total_train_steps", 200))
    metrics = trainer.train(train_tasks, total_steps=total_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output_dir / "train_metrics.json").write_text(json.dumps([m.__dict__ for m in metrics], ensure_ascii=True, indent=2))
    if metrics:
        last = metrics[-1]
        print(
            f"Finished {len(metrics)} steps | reward={last.mean_reward:.3f} "
            f"| turns={last.mean_turns:.2f} | timeout_rate={last.timeout_rate:.3f}"
        )
    else:
        print("No metrics produced. Check task loading.")


if __name__ == "__main__":
    main()
