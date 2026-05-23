from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


DEGRADATION_CONFIGS = [
    {"target": "all", "multiplier": 2.0, "timeout_rate": 0.0},
    {"target": "all", "multiplier": 5.0, "timeout_rate": 0.0},
    {"target": "top50pct", "multiplier": 1.0, "timeout_rate": 0.15},
    {"target": "top50pct", "multiplier": 3.0, "timeout_rate": 0.10},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base_config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    multipliers = config.get("eval", {}).get("degradation_multipliers", [2.0, 5.0])
    timeout_rates = config.get("eval", {}).get("degradation_timeout_rates", [0.0, 0.15])
    combos = [{"target": "all", "multiplier": m, "timeout_rate": t} for m in multipliers for t in timeout_rates]
    print(json.dumps({"default_configs": DEGRADATION_CONFIGS, "resolved_configs": combos}, indent=2))


if __name__ == "__main__":
    main()
