from __future__ import annotations

import random
from typing import Any, Dict, List


class TaskSampler:
    def __init__(self, tasks: List[Dict[str, Any]], seed: int = 42):
        self.tasks = tasks
        self.rng = random.Random(seed)
        self._idx = 0
        self.rng.shuffle(self.tasks)

    def sample(self, n: int = 1) -> List[Dict[str, Any]]:
        if not self.tasks:
            return []
        out: List[Dict[str, Any]] = []
        for _ in range(n):
            out.append(self.tasks[self._idx % len(self.tasks)])
            self._idx += 1
        return out

    def sample_one(self) -> Dict[str, Any]:
        samples = self.sample(1)
        if not samples:
            raise ValueError("No tasks available.")
        return samples[0]
