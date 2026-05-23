from __future__ import annotations

from collections import defaultdict, deque
from statistics import mean, pstdev
from typing import Deque, Dict


class LatencyTracker:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._latencies: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self._timeouts: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, tool_name: str, latency_ms: float, timed_out: bool = False) -> None:
        self._latencies[tool_name].append(float(latency_ms))
        self._timeouts[tool_name].append(1 if timed_out else 0)

    def _stats(self, tool_name: str) -> tuple[float, float]:
        vals = list(self._latencies[tool_name])
        if not vals:
            return 500.0, 100.0
        if len(vals) == 1:
            return vals[0], max(vals[0] * 0.1, 1.0)
        return mean(vals), max(pstdev(vals), 1.0)

    def get_delta(self, tool_name: str, latency_ms: float) -> float:
        mu, sigma = self._stats(tool_name)
        return (float(latency_ms) - mu) / sigma

    def get_tier(self, tool_name: str, latency_ms: float) -> str:
        z = self.get_delta(tool_name, latency_ms)
        if z < -0.5:
            return "FAST"
        if z > 0.5:
            return "SLOW"
        return "MEDIUM"

    def get_stats(self, tool_name: str) -> dict:
        mu, sigma = self._stats(tool_name)
        timeouts = list(self._timeouts[tool_name])
        timeout_rate = (sum(timeouts) / len(timeouts)) if timeouts else 0.0
        return {"mean": mu, "std": sigma, "timeout_rate": timeout_rate}
