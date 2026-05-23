from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.toolbench_loader import load_tasks_from_benchmark, split_by_category


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument("--n_eval_tasks", type=int, default=200)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    tasks = load_tasks_from_benchmark(config, split="test")
    splits = split_by_category(tasks, config)
    test_tasks = splits["test"][: args.n_eval_tasks]
    categories = sorted({t["category"] for t in test_tasks})
    print(json.dumps({"n_test_tasks": len(test_tasks), "categories": categories}, indent=2))


if __name__ == "__main__":
    main()
