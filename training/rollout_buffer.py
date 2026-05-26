from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Episode:
    task: Dict[str, Any]
    trajectory: List[Dict[str, Any]]
    final_answer: str
    reward: float
    logprob_sum: Optional[Any] = None
    info: Dict[str, Any] = field(default_factory=dict)


class RolloutBuffer:
    def __init__(self):
        self.episodes: List[Episode] = []

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)

    def clear(self) -> None:
        self.episodes.clear()

    def __len__(self) -> int:
        return len(self.episodes)
