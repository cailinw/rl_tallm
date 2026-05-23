from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _first_non_empty(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, "", [], {}):
            return row[key]
    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        # If it is a chat-style object, prefer "content".
        content = value.get("content")
        if isinstance(content, str):
            return content.strip()
    if isinstance(value, list):
        # For chat/message lists, concatenate user-visible strings.
        parts = []
        for item in value:
            if isinstance(item, dict):
                c = item.get("content")
                if isinstance(c, str):
                    parts.append(c.strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        return " ".join([p for p in parts if p]).strip()
    return ""


def _extract_tool_name(item: Any) -> Optional[str]:
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, dict):
        candidate = _first_non_empty(
            item,
            keys=[
                "name",
                "tool_name",
                "api_name",
                "function_name",
                "id",
            ],
        )
        text = _normalize_text(candidate)
        return text or None
    return None


def _extract_tools(row: Dict[str, Any]) -> List[str]:
    raw = _first_non_empty(
        row,
        keys=[
            "available_tools",
            "tools",
            "tool_names",
            "apis",
            "api_list",
            "candidate_tools",
        ],
    )
    if raw is None:
        return []
    if isinstance(raw, list):
        names: List[str] = []
        for item in raw:
            name = _extract_tool_name(item)
            if name:
                names.append(name)
        # Keep deterministic order while deduping.
        seen = set()
        deduped = []
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        return deduped
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


def _extract_required_tools(row: Dict[str, Any], available_tools: List[str]) -> List[str]:
    raw = _first_non_empty(
        row,
        keys=[
            "required_tools",
            "gold_tools",
            "solution_tools",
            "reference_tools",
        ],
    )
    if isinstance(raw, list):
        names = []
        for item in raw:
            name = _extract_tool_name(item)
            if name:
                names.append(name)
        return names[:]
    if isinstance(raw, str) and raw.strip():
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [available_tools[0]] if available_tools else []


def _extract_query(row: Dict[str, Any]) -> str:
    # Ordered for benchmark-agnostic robustness.
    direct = _first_non_empty(
        row,
        keys=[
            "query",
            "instruction",
            "question",
            "prompt",
            "user_query",
            "input",
        ],
    )
    query = _normalize_text(direct)
    if query:
        return query

    # Chat style structures.
    messages = _first_non_empty(row, keys=["messages", "conversation", "dialogue"])
    query = _normalize_text(messages)
    return query


def _extract_ground_truth(row: Dict[str, Any]) -> str:
    value = _first_non_empty(
        row,
        keys=[
            "ground_truth",
            "answer",
            "final_answer",
            "reference_answer",
            "gold_answer",
            "target",
            "output",
        ],
    )
    return _normalize_text(value)


def _extract_category(row: Dict[str, Any], available_tools: List[str]) -> str:
    value = _first_non_empty(
        row,
        keys=[
            "category",
            "tool_category",
            "task_category",
            "domain",
            "topic",
        ],
    )
    category = _normalize_text(value)
    if category:
        return category
    if available_tools:
        # Heuristic fallback from tool prefixes.
        first = available_tools[0].split("_")[0].strip()
        return first.capitalize() if first else "Unknown"
    return "Unknown"


def _normalize_common_task(row: Dict[str, Any], benchmark: str) -> Dict[str, Any]:
    tools = _extract_tools(row)
    return {
        "query": _extract_query(row),
        "available_tools": tools,
        "ground_truth": _extract_ground_truth(row),
        "required_tools": _extract_required_tools(row, tools),
        "category": _extract_category(row, tools),
        "source_benchmark": benchmark,
    }


def _normalize_toolbench_task(row: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_common_task(row, benchmark="toolbench")


def _normalize_live_api_bench_task(row: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_common_task(row, benchmark="live_api_bench")


def _pick_repo_and_normalizer(data_cfg: Dict[str, Any], benchmark: str):
    if benchmark == "toolbench":
        return data_cfg.get("toolbench_repo", "ToolBench/ToolBench"), _normalize_toolbench_task
    if benchmark == "live_api_bench":
        return data_cfg.get("live_api_bench_repo", "LiveAPIBench/LiveAPIBench"), _normalize_live_api_bench_task
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _load_hf_dataset(repo: str, split: str, data_cfg: Dict[str, Any]):
    from datasets import load_dataset

    config_name = data_cfg.get("dataset_config_name")
    if config_name:
        return load_dataset(repo, config_name, split=split)
    return load_dataset(repo, split=split)


def _build_diagnostics(
    normalized_tasks: List[Dict[str, Any]],
    dropped_tasks: int,
    benchmark: str,
    split: str,
) -> Dict[str, Any]:
    categories = Counter(t.get("category", "Unknown") for t in normalized_tasks)
    missing_query = sum(1 for t in normalized_tasks if not t.get("query"))
    missing_gt = sum(1 for t in normalized_tasks if not t.get("ground_truth"))
    tool_count_hist = Counter(len(t.get("available_tools", [])) for t in normalized_tasks)

    return {
        "benchmark": benchmark,
        "split": split,
        "n_tasks": len(normalized_tasks),
        "n_dropped": dropped_tasks,
        "missing_query": missing_query,
        "missing_ground_truth": missing_gt,
        "category_counts": dict(categories),
        "tool_count_histogram": dict(sorted(tool_count_hist.items(), key=lambda x: x[0])),
    }


def _synthetic_tasks(benchmark: str, n: int = 200) -> List[Dict[str, Any]]:
    rng = random.Random(7)
    categories = ["Weather", "Finance", "News", "Travel", "Sports"]
    all_tools = {
        "Weather": ["weather_fast", "weather_slow"],
        "Finance": ["stock_quote_fast", "stock_quote_slow"],
        "News": ["news_search_fast", "news_search_slow"],
        "Travel": ["flight_lookup_fast", "flight_lookup_slow"],
        "Sports": ["sports_score_fast", "sports_score_slow"],
    }
    tasks: List[Dict[str, Any]] = []
    for i in range(n):
        category = rng.choice(categories)
        tools = all_tools[category]
        city = rng.choice(["San Francisco", "New York", "Tokyo", "Paris"])
        tasks.append(
            {
                "query": f"{category} query {i} for {city}",
                "available_tools": tools,
                "ground_truth": f"{category} answer {i}",
                "required_tools": [tools[0]],
                "category": category,
                "source_benchmark": benchmark,
            }
        )
    return tasks


def load_tasks_from_benchmark(
    config: Dict[str, Any],
    split: str = "train",
    return_diagnostics: bool = False,
):
    data_cfg = config.get("data", {})
    benchmark = data_cfg.get("benchmark", "toolbench").strip().lower()
    strict_loading = bool(data_cfg.get("strict_benchmark_loading", False))
    allow_synthetic_fallback = bool(data_cfg.get("allow_synthetic_fallback", True))

    repo, normalizer = _pick_repo_and_normalizer(data_cfg, benchmark)

    try:
        dataset = _load_hf_dataset(repo=repo, split=split, data_cfg=data_cfg)
        tasks = [normalizer(dict(row)) for row in dataset]
    except Exception as exc:
        if strict_loading or not allow_synthetic_fallback:
            raise RuntimeError(
                f"Failed to load benchmark dataset for benchmark={benchmark}, repo={repo}, split={split}"
            ) from exc
        # Fallback for local development if dataset access is unavailable.
        tasks = _synthetic_tasks(benchmark)

    min_tools = data_cfg.get("min_tools_per_task", 2)
    filtered = [t for t in tasks if len(t.get("available_tools", [])) >= min_tools]
    diagnostics = _build_diagnostics(
        normalized_tasks=filtered,
        dropped_tasks=max(0, len(tasks) - len(filtered)),
        benchmark=benchmark,
        split=split,
    )
    if return_diagnostics:
        return filtered, diagnostics
    return filtered


def split_by_category(tasks: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    data_cfg = config.get("data", {})
    train_categories = set(data_cfg.get("train_categories", []))
    test_categories = set(data_cfg.get("test_categories", []))

    train, test, other = [], [], []
    for task in tasks:
        category = task.get("category", "")
        if category in train_categories:
            train.append(task)
        elif category in test_categories:
            test.append(task)
        else:
            other.append(task)
    return {"train": train, "test": test, "other": other}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["toolbench", "live_api_bench"], default="toolbench")
    parser.add_argument("--split", default="train")
    parser.add_argument("--repo", default=None, help="Optional dataset repo override.")
    parser.add_argument("--dataset_config_name", default=None, help="Optional HF dataset config name.")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Disable synthetic fallback and fail hard on load errors.")
    parser.add_argument("--inspect", action="store_true", help="Print normalization diagnostics JSON.")
    args = parser.parse_args()

    config = {
        "data": {
            "benchmark": args.benchmark,
            "toolbench_repo": args.repo or "ToolBench/ToolBench",
            "live_api_bench_repo": args.repo or "LiveAPIBench/LiveAPIBench",
            "dataset_config_name": args.dataset_config_name,
            "strict_benchmark_loading": args.strict,
            "allow_synthetic_fallback": not args.strict,
            "min_tools_per_task": 2,
        }
    }
    tasks, diagnostics = load_tasks_from_benchmark(config, split=args.split, return_diagnostics=True)
    if args.test:
        print(f"Loaded {len(tasks)} tasks from {args.benchmark} ({args.split})")
        if tasks:
            print(tasks[0])
    if args.inspect:
        print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
