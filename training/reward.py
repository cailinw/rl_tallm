from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def fuzzy_answer_match(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    pred_counts = Counter(pred_toks)
    gold_counts = Counter(gold_toks)
    overlap = sum((pred_counts & gold_counts).values())
    precision = overlap / max(1, len(pred_toks))
    recall = overlap / max(1, len(gold_toks))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_reward(
    trajectory: List[dict],
    final_answer: str,
    ground_truth: str,
    required_tools: List[str],
    config: Dict,
) -> Dict:
    reward_cfg = config.get("reward", config)
    training_cfg = config.get("training", config)
    w_latency = float(training_cfg.get("w_latency", 0.3))

    sim = fuzzy_answer_match(final_answer, ground_truth)
    if sim >= 0.99:
        r_task = float(reward_cfg.get("r_task_correct", 1.0))
    elif sim >= 0.7:
        r_task = float(reward_cfg.get("r_task_partial", 0.3))
    else:
        r_task = 0.0

    timeout_events = sum(int(turn.get("timed_out", False)) for turn in trajectory)
    r_timeout = float(reward_cfg.get("r_timeout_penalty", -2.0)) * timeout_events

    required_tools = set(required_tools or [])
    used_tools = {turn.get("tool_name") for turn in trajectory if turn.get("tool_name")}
    missing_required = len(required_tools - used_tools)
    r_required = -0.2 * missing_required if required_tools else 0.0

    deltas = [max(0.0, float(turn.get("latency_delta", 0.0))) for turn in trajectory if turn.get("tool_name")]
    apply_latency = (r_task > 0.0) if reward_cfg.get("latency_penalty_only_on_success", True) else True
    r_latency = (-w_latency * (sum(deltas) / max(1, len(deltas)))) if (apply_latency and deltas) else 0.0

    total = r_task + r_latency + r_timeout + r_required
    return {
        "total": total,
        "r_task": r_task,
        "r_latency": r_latency,
        "r_timeout": r_timeout,
        "r_required": r_required,
        "similarity": sim,
        "breakdown": {
            "n_turns": len(trajectory),
            "timeout_events": timeout_events,
            "missing_required_tools": missing_required,
            "mean_positive_latency_delta": (sum(deltas) / max(1, len(deltas))) if deltas else 0.0,
        },
    }
