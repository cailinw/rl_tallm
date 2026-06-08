from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt


PENALTY_SCORE = -2.0
PENALTY_COST = 2.0
PENALTY_QOP = -2.0
DEFAULT_ALPHA = 0.5
DEFAULT_LATENCY_WEIGHT = 0.1
DEFAULT_INVALID_PENALTY = -1.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def read_online_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def summarize_online(rows: List[Dict[str, Any]], policy: str) -> Dict[str, Any]:
    selected = [r for r in rows if r["policy"] == policy]
    return {
        "method": policy,
        "n": len(selected),
        "valid_rate": mean(1.0 if r["valid"] == "True" else 0.0 for r in selected),
        "mean_task_score": mean(float(r["task_score"]) for r in selected),
        "mean_cost_price": mean(float(r["cost_price"]) for r in selected),
        "mean_exec_time_ms": mean(float(r["exec_time"]) for r in selected),
        "mean_qop": mean(float(r["qop"]) for r in selected),
        "mean_reward": mean(float(r["reward"]) for r in selected),
    }


def reward_from_metrics(
    valid: bool,
    task_score: float,
    cost_price: float,
    exec_time_ms: float,
    alpha: float = DEFAULT_ALPHA,
    latency_weight: float = DEFAULT_LATENCY_WEIGHT,
    invalid_penalty: float = DEFAULT_INVALID_PENALTY,
) -> float:
    if not valid:
        return invalid_penalty
    return alpha * task_score + (1.0 - alpha) * (-cost_price) - latency_weight * (exec_time_ms / 1000.0)


def iter_catp_entries(payload: Dict[str, Any]):
    for task_id, samples in payload.items():
        if not isinstance(samples, dict):
            continue
        for sample_id, item in samples.items():
            if isinstance(item, dict):
                yield str(task_id), str(sample_id), item


def summarize_catp(results_dir: Path) -> Dict[str, Any]:
    rows = read_catp_rows(results_dir)
    return summarize_catp_rows(rows, method="catp_llm_offline")


def read_catp_rows(results_dir: Path) -> List[Dict[str, Any]]:
    valid_path = results_dir / "valid_plans.json"
    invalid_path = results_dir / "invalid_plans.json"
    valid_payload = json.loads(valid_path.read_text()) if valid_path.exists() else {}
    invalid_payload = json.loads(invalid_path.read_text()) if invalid_path.exists() else {}

    rows: List[Dict[str, Any]] = []
    for task_id, sample_id, item in iter_catp_entries(valid_payload):
        task_score = float(item.get("task_score", PENALTY_SCORE))
        cost_price = float(item.get("cost_price", PENALTY_COST))
        exec_time = float(item.get("exec_time", 0.0) or 0.0)
        rows.append(
            {
                "task_id": task_id,
                "sample_id": sample_id,
                "valid": 1.0,
                "task_score": task_score,
                "cost_price": cost_price,
                "exec_time": exec_time,
                "qop": float(item.get("qop", PENALTY_QOP) if item.get("qop") is not None else PENALTY_QOP),
                "reward": reward_from_metrics(True, task_score, cost_price, exec_time),
            }
        )
    for task_id, sample_id, item in iter_catp_entries(invalid_payload):
        task_score = float(item.get("task_score", PENALTY_SCORE))
        cost_price = float(item.get("cost_price", PENALTY_COST))
        exec_time = float(item.get("exec_time", 0.0) or 0.0)
        rows.append(
            {
                "task_id": task_id,
                "sample_id": sample_id,
                "valid": 0.0,
                "task_score": task_score,
                "cost_price": cost_price,
                "exec_time": exec_time,
                "qop": float(item.get("qop", PENALTY_QOP) if item.get("qop") is not None else PENALTY_QOP),
                "reward": reward_from_metrics(False, task_score, cost_price, exec_time),
            }
        )
    return rows


def summarize_catp_rows(rows: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    return {
        "method": method,
        "n": len(rows),
        "valid_rate": mean(r["valid"] for r in rows),
        "mean_task_score": mean(r["task_score"] for r in rows),
        "mean_cost_price": mean(r["cost_price"] for r in rows),
        "mean_exec_time_ms": mean(r["exec_time"] for r in rows),
        "mean_qop": mean(r["qop"] for r in rows),
        "mean_reward": mean(r["reward"] for r in rows),
    }


def online_keys(rows: List[Dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(row["task_id"]), str(row["sample_id"])) for row in rows}


def filter_online_by_keys(rows: List[Dict[str, Any]], keys: set[tuple[str, str]]) -> List[Dict[str, Any]]:
    return [row for row in rows if (str(row["task_id"]), str(row["sample_id"])) in keys]


def filter_catp_by_keys(rows: List[Dict[str, Any]], keys: set[tuple[str, str]]) -> List[Dict[str, Any]]:
    return [row for row in rows if (str(row["task_id"]), str(row["sample_id"])) in keys]


def read_online_training(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "step": float(row["step"]),
                    "mean_reward": float(row["mean_reward"]),
                    "policy_loss": float(row["policy_loss"]),
                }
            )
    return rows


def read_catp_losses(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    for idx, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if line:
            rows.append({"step": float(idx), "value": float(line)})
    return rows


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "method",
        "n",
        "valid_rate",
        "mean_task_score",
        "mean_cost_price",
        "mean_exec_time_ms",
        "mean_qop",
        "mean_reward",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_training(online_rows: List[Dict[str, float]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    if online_rows:
        steps = [r["step"] for r in online_rows]
        axes[0].plot(steps, [r["mean_reward"] for r in online_rows], marker="o", color="tab:blue")
        axes[1].plot(steps, [r["policy_loss"] for r in online_rows], marker="o", color="tab:orange")
    axes[0].set_title("Online GRPO Mean Reward")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Mean reward")
    axes[1].set_title("Online GRPO Policy Loss")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Policy loss")
    fig.suptitle("Online GRPO Training Progress")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close()


def plot_catp_training(catp_rows: List[Dict[str, float]], output_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    if catp_rows:
        plt.plot([r["step"] for r in catp_rows], [r["value"] for r in catp_rows], marker="o", label="CATP-LLM train loss")
    plt.title("CATP-LLM Offline Training Loss")
    plt.xlabel("Loss update")
    plt.ylabel("Train loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_final_metrics(summary_rows: List[Dict[str, Any]], output_path: Path) -> None:
    metrics = ["valid_rate", "mean_task_score", "mean_qop", "mean_reward"]
    methods = [r["method"] for r in summary_rows]
    x = range(len(methods))
    width = 0.2
    plt.figure(figsize=(10, 4.8))
    for idx, metric in enumerate(metrics):
        offsets = [pos + (idx - 1.5) * width for pos in x]
        plt.bar(offsets, [float(r[metric]) for r in summary_rows], width=width, label=metric)
    plt.title("Final OpenCATP Quality Metrics by Method")
    plt.xlabel("Method")
    plt.ylabel("Metric value")
    plt.xticks(list(x), methods, rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_latency_cost(summary_rows: List[Dict[str, Any]], output_path: Path) -> None:
    methods = [r["method"] for r in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(methods, [float(r["mean_exec_time_ms"]) for r in summary_rows], color="tab:purple")
    axes[0].set_title("Mean Execution Time")
    axes[0].set_ylabel("Milliseconds")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(methods, [float(r["mean_cost_price"]) for r in summary_rows], color="tab:green")
    axes[1].set_title("Mean Cost Price")
    axes[1].set_ylabel("Cost")
    axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle("Final OpenCATP Latency and Cost Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online_results_dir", required=True)
    parser.add_argument("--catp_results_dir", required=True)
    parser.add_argument("--catp_train_losses", required=True)
    parser.add_argument("--output_dir", default="experiments/results/catp_three_way_compare")
    args = parser.parse_args()

    online_dir = Path(args.online_results_dir)
    catp_dir = Path(args.catp_results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    online_rows = read_online_rows(online_dir / "episodes.csv")
    catp_rows = read_catp_rows(catp_dir)
    summary_rows = [
        summarize_online(online_rows, "qwen_baseline"),
        summarize_online(online_rows, "qwen_online_grpo"),
        summarize_catp_rows(catp_rows, method="catp_llm_offline"),
    ]
    write_summary_csv(out_dir / "final_metrics.csv", summary_rows)
    (out_dir / "final_metrics.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True))

    baseline_rows = [row for row in online_rows if row["policy"] == "qwen_baseline"]
    grpo_rows = [row for row in online_rows if row["policy"] == "qwen_online_grpo"]
    overlap_keys = online_keys(baseline_rows) & online_keys(grpo_rows) & online_keys(catp_rows)
    aligned_summary_rows = [
        summarize_online(filter_online_by_keys(online_rows, overlap_keys), "qwen_baseline"),
        summarize_online(filter_online_by_keys(online_rows, overlap_keys), "qwen_online_grpo"),
        summarize_catp_rows(filter_catp_by_keys(catp_rows, overlap_keys), method="catp_llm_offline"),
    ]
    aligned_metadata = {
        "n_overlap": len(overlap_keys),
        "overlap_examples": [
            {"task_id": task_id, "sample_id": sample_id}
            for task_id, sample_id in sorted(overlap_keys, key=lambda item: (int(item[0]), int(item[1])))
        ],
    }
    write_summary_csv(out_dir / "aligned_final_metrics.csv", aligned_summary_rows)
    (out_dir / "aligned_final_metrics.json").write_text(
        json.dumps({"metadata": aligned_metadata, "summary": aligned_summary_rows}, indent=2, sort_keys=True)
    )

    online_training = read_online_training(online_dir / "train_metrics.csv")
    catp_losses = read_catp_losses(Path(args.catp_train_losses))
    with (out_dir / "training_curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "step", "value"])
        writer.writeheader()
        for row in online_training:
            writer.writerow(
                {
                    "method": "qwen_online_grpo_mean_reward",
                    "step": row["step"],
                    "value": row["mean_reward"],
                }
            )
            writer.writerow(
                {
                    "method": "qwen_online_grpo_policy_loss",
                    "step": row["step"],
                    "value": row["policy_loss"],
                }
            )
        for row in catp_losses:
            writer.writerow({"method": "catp_llm_train_loss", **row})

    plot_training(online_training, out_dir / "training_progress.png")
    plot_catp_training(catp_losses, out_dir / "catp_offline_training_loss.png")
    plot_final_metrics(summary_rows, out_dir / "final_metrics.png")
    plot_latency_cost(summary_rows, out_dir / "latency_cost_metrics.png")
    plot_final_metrics(aligned_summary_rows, out_dir / "aligned_final_metrics.png")
    plot_latency_cost(aligned_summary_rows, out_dir / "aligned_latency_cost_metrics.png")
    print(
        json.dumps(
            {
                "aligned": {"metadata": aligned_metadata, "summary": aligned_summary_rows},
                "output_dir": str(out_dir),
                "summary": summary_rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
