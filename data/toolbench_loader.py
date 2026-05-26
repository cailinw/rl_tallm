from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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
        text = raw.strip()
        if not text:
            return []
        parsed = None
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    parsed = None
        if isinstance(parsed, list):
            names = []
            for item in parsed:
                name = _extract_tool_name(item)
                if name:
                    names.append(name)
            if names:
                return names
        return [x.strip() for x in text.split(",") if x.strip()]
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


def _normalize_catp_llm_task(row: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_common_task(row, benchmark="catp_llm")


def _normalize_wild_tool_bench_task(row: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_common_task(row, benchmark="wild_tool_bench")


def _pick_path_and_normalizer(data_cfg: Dict[str, Any], benchmark: str):
    if benchmark == "catp_llm":
        return data_cfg.get("catp_llm_path"), _normalize_catp_llm_task
    if benchmark == "wild_tool_bench":
        return data_cfg.get("wild_tool_bench_path"), _normalize_wild_tool_bench_task
    raise ValueError(
        f"Unsupported benchmark: {benchmark}. Supported benchmarks: catp_llm, wild_tool_bench."
    )


def _load_json_file(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return [x for x in payload["data"] if isinstance(x, dict)]
    return []


def _load_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _load_records_from_path(dataset_path: str, split: str) -> List[Dict[str, Any]]:
    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    if path.is_file():
        if path.suffix == ".jsonl":
            return _load_jsonl_file(path)
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            if isinstance(payload, dict) and split in payload and isinstance(payload[split], list):
                return [x for x in payload[split] if isinstance(x, dict)]
            if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
                return [x for x in payload["data"] if isinstance(x, dict)]
            if isinstance(payload, list):
                return [x for x in payload if isinstance(x, dict)]
            raise ValueError(f"Unsupported JSON structure in dataset file: {path}")
        raise ValueError(f"Unsupported dataset file extension for {path}. Use .json or .jsonl.")

    split_candidates = [
        path / f"{split}.jsonl",
        path / f"{split}.json",
        path / split / "data.jsonl",
        path / split / "data.json",
        path / "data" / f"{split}.jsonl",
        path / "data" / f"{split}.json",
        path / "data" / "Wild-Tool-Bench.jsonl",
        path / "wild-tool-bench" / "data" / "Wild-Tool-Bench.jsonl",
    ]
    for candidate in split_candidates:
        if candidate.exists():
            if candidate.suffix == ".jsonl":
                return _load_jsonl_file(candidate)
            if candidate.suffix == ".json":
                return _load_json_file(candidate)
    raise FileNotFoundError(
        f"No supported dataset file found under {path} for split={split}. "
        "Expected .json/.jsonl in root, split/, data/, or WildToolBench default paths."
    )


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
    benchmark = data_cfg.get("benchmark", "catp_llm").strip().lower()
    strict_loading = bool(data_cfg.get("strict_benchmark_loading", False))
    allow_synthetic_fallback = bool(data_cfg.get("allow_synthetic_fallback", True))

    dataset_path, normalizer = _pick_path_and_normalizer(data_cfg, benchmark)
    if not dataset_path and strict_loading:
        raise RuntimeError(
            f"Missing dataset path for benchmark={benchmark}. "
            f"Set data.{benchmark}_path in config."
        )

    try:
        if not dataset_path:
            raise FileNotFoundError(f"Missing dataset path for benchmark={benchmark}")
        dataset = _load_records_from_path(dataset_path=dataset_path, split=split)
        tasks = [normalizer(dict(row)) for row in dataset if isinstance(row, dict)]
    except Exception as exc:
        if strict_loading or not allow_synthetic_fallback:
            raise RuntimeError(
                f"Failed to load benchmark dataset for benchmark={benchmark}, "
                f"path={dataset_path}, split={split}"
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
    parser.add_argument("--benchmark", choices=["catp_llm", "wild_tool_bench"], default="catp_llm")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_path", default=None, help="Local path to JSON/JSONL dataset file or root directory.")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Disable synthetic fallback and fail hard on load errors.")
    parser.add_argument("--inspect", action="store_true", help="Print normalization diagnostics JSON.")
    args = parser.parse_args()

    config = {
        "data": {
            "benchmark": args.benchmark,
            "catp_llm_path": args.dataset_path if args.benchmark == "catp_llm" else None,
            "wild_tool_bench_path": args.dataset_path if args.benchmark == "wild_tool_bench" else None,
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
