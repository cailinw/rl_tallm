"""
Aggregate and plot the α-ablation results for online GRPO.

Reads episodes.csv from three output dirs (α=0.2, 0.5, 0.8), computes
aligned metrics on the shared (task_id, sample_id) pairs, and produces:
  - alpha_ablation_summary.csv   — per-α aggregate metrics
  - alpha_ablation_per_task.csv  — per-α per-task metrics
  - alpha_ablation_overall.png   — bar chart: score / time / cost / reward
  - alpha_ablation_per_task_score.png  — grouped bars per task
  - alpha_ablation_per_task_time.png   — grouped bars per task
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALPHA_DIRS = {
    "0.2": "experiments/results/catp_grpo_alpha_0.2",
    "0.5": "experiments/results/catp_qwen_online_loaded_8tasks_327eval",
    "0.8": "experiments/results/catp_grpo_alpha_0.8",
}

METRICS = ["task_score", "exec_time_ms", "cost_price", "reward"]
METRIC_LABELS = {
    "task_score": "Mean Task Score ↑",
    "exec_time_ms": "Mean Exec Time (ms) ↓",
    "cost_price": "Mean Cost ↓",
    "reward": "Mean Reward ↑",
}
COLORS = {"0.2": "#e07b54", "0.5": "#4c8fbd", "0.8": "#5aad6a"}


def finite(v, fallback=-1.0):
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def load_episodes(result_dir: str):
    """Return only the GRPO policy rows from episodes.csv."""
    path = Path(result_dir) / "episodes.csv"
    if not path.exists():
        raise FileNotFoundError(f"episodes.csv not found in {result_dir}")
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            # keep only online-GRPO rows; skip baseline rows
            if "grpo" in row.get("policy", "").lower() or "online" in row.get("policy", "").lower():
                rows.append(row)
    return rows


def key(row):
    return (int(row["task_id"]), int(row["sample_id"]))


def compute_aligned(all_rows: dict[str, list]) -> dict[str, list]:
    """Keep only (task_id, sample_id) pairs present in all alpha variants."""
    sets = [set(key(r) for r in rows) for rows in all_rows.values()]
    common = sets[0].intersection(*sets[1:])
    return {alpha: [r for r in rows if key(r) in common]
            for alpha, rows in all_rows.items()}


def summarise(rows: list, alpha_val: float) -> dict:
    if not rows:
        return {}
    scores, times, costs, rewards = [], [], [], []
    for r in rows:
        s = finite(r.get("task_score"), -1.0)
        t = finite(r.get("exec_time_ms") or r.get("exec_time"), 9999.0)
        c = finite(r.get("cost_price"), 9.0)
        valid = r.get("valid", "true").lower() in ("true", "1", "yes")
        if not valid:
            rew = -1.0
        else:
            rew = alpha_val * s - (1 - alpha_val) * c - 0.1 * (t / 1000)
        scores.append(s)
        times.append(t)
        costs.append(c)
        rewards.append(rew)
    # only count GRPO rows (skip baseline rows)
    return {
        "n": len(rows),
        "mean_task_score": float(np.mean(scores)),
        "mean_exec_time_ms": float(np.mean(times)),
        "mean_cost_price": float(np.mean(costs)),
        "mean_reward": float(np.mean(rewards)),
    }


def per_task_summarise(rows: list, alpha_val: float) -> dict:
    by_task = defaultdict(list)
    for r in rows:
        by_task[int(r["task_id"])].append(r)
    result = {}
    for tid, task_rows in sorted(by_task.items()):
        result[tid] = summarise(task_rows, alpha_val)
        result[tid]["task_id"] = tid
    return result


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_overall(summary: dict, out_dir: Path):
    alphas = sorted(summary.keys(), key=float)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    metric_keys = ["mean_task_score", "mean_exec_time_ms", "mean_cost_price", "mean_reward"]
    labels = ["Task Score ↑", "Exec Time (ms) ↓", "Cost ↓", "Reward ↑"]
    x = np.arange(len(alphas))
    width = 0.5
    for ax, mk, label in zip(axes, metric_keys, labels):
        vals = [summary[a].get(mk, 0.0) for a in alphas]
        colors = [COLORS[a] for a in alphas]
        bars = ax.bar(x, vals, width, color=colors, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"α={a}" for a in alphas])
        ax.set_title(label, fontsize=10)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        ax.set_xlabel("α value")
    fig.suptitle("Online GRPO: α ablation (325 aligned samples, 8 tasks)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_ablation_overall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'alpha_ablation_overall.png'}")


def plot_per_task(per_task: dict, metric_key: str, ylabel: str, out_path: Path):
    alphas = sorted(per_task.keys(), key=float)
    task_ids = sorted({tid for pt in per_task.values() for tid in pt})
    n_tasks = len(task_ids)
    x = np.arange(n_tasks)
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 1.5), 4))
    for i, alpha in enumerate(alphas):
        vals = [per_task[alpha].get(tid, {}).get(metric_key, 0.0) for tid in task_ids]
        offset = (i - len(alphas) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=f"α={alpha}",
                      color=COLORS[alpha], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Task {t}" for t in task_ids], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Online GRPO α ablation — {ylabel}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments/results/alpha_ablation")
    parser.add_argument("--alpha_02_dir", default=ALPHA_DIRS["0.2"])
    parser.add_argument("--alpha_05_dir", default=ALPHA_DIRS["0.5"])
    parser.add_argument("--alpha_08_dir", default=ALPHA_DIRS["0.8"])
    args = parser.parse_args()

    dirs = {
        "0.2": args.alpha_02_dir,
        "0.5": args.alpha_05_dir,
        "0.8": args.alpha_08_dir,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading episodes...")
    all_rows = {}
    for alpha, d in dirs.items():
        try:
            all_rows[alpha] = load_episodes(d)
            print(f"  α={alpha}: {len(all_rows[alpha])} episodes from {d}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    if len(all_rows) < 2:
        print("Need at least 2 α variants to compare. Exiting.")
        return

    print("Computing aligned subset...")
    aligned = compute_aligned(all_rows)
    n_overlap = len(next(iter(aligned.values())))
    print(f"  Aligned samples: {n_overlap}")

    # Overall summary
    summary = {}
    summary_rows = []
    for alpha, rows in sorted(aligned.items(), key=lambda x: float(x[0])):
        s = summarise(rows, float(alpha))
        summary[alpha] = s
        summary_rows.append({"alpha": alpha, **s})
        print(f"  α={alpha}: score={s['mean_task_score']:.4f}  "
              f"time={s['mean_exec_time_ms']:.1f}ms  "
              f"cost={s['mean_cost_price']:.4f}  reward={s['mean_reward']:.4f}")

    write_csv(out_dir / "alpha_ablation_summary.csv", summary_rows,
              ["alpha", "n", "mean_task_score", "mean_exec_time_ms", "mean_cost_price", "mean_reward"])

    # Per-task summary
    per_task = {}
    pt_rows = []
    for alpha, rows in aligned.items():
        per_task[alpha] = per_task_summarise(rows, float(alpha))
        for tid, m in per_task[alpha].items():
            pt_rows.append({"alpha": alpha, **m})
    write_csv(out_dir / "alpha_ablation_per_task.csv", pt_rows,
              ["alpha", "task_id", "n", "mean_task_score", "mean_exec_time_ms",
               "mean_cost_price", "mean_reward"])

    # Metadata
    meta = {"n_overlap": n_overlap, "alpha_dirs": dirs}
    with open(out_dir / "alpha_ablation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Plots
    print("Generating plots...")
    plot_overall(summary, out_dir)
    plot_per_task(per_task, "mean_task_score", "Mean Task Score ↑",
                  out_dir / "alpha_ablation_per_task_score.png")
    plot_per_task(per_task, "mean_exec_time_ms", "Mean Exec Time (ms) ↓",
                  out_dir / "alpha_ablation_per_task_time.png")
    print("Done.")


if __name__ == "__main__":
    main()
