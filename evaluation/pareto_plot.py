from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="checkpoints/latency_aware_rl/train_metrics.json")
    parser.add_argument("--output", default="figures/pareto.png")
    args = parser.parse_args()

    results = json.loads(Path(args.results_file).read_text())
    rewards = [float(r.get("mean_reward", 0.0)) for r in results]
    turns = [float(r.get("mean_turns", 0.0)) for r in results]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.scatter(turns, rewards, s=14)
    plt.xlabel("Avg Tool Calls / Task (proxy)")
    plt.ylabel("Mean Reward (proxy)")
    plt.title("Accuracy-Latency Proxy Pareto")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
