from __future__ import annotations

import json
import random
from typing import Dict, List

from model.context_builder import parse_model_output


class PolicyModel:
    """
    Lightweight policy wrapper.
    - Uses heuristic/random behavior by default for local bring-up.
    - Interface is compatible with replacing internals by HF/TRL generation.
    """

    def __init__(self, model_name: str, seed: int = 42):
        self.model_name = model_name
        self.rng = random.Random(seed)

    def generate_action(
        self,
        prompt: str,
        available_tools: List[Dict],
        force_final_prob: float = 0.2,
    ) -> Dict:
        _ = prompt
        if not available_tools or self.rng.random() < force_final_prob:
            return {"type": "final_answer", "answer": "best effort final answer"}

        # Favor faster tool prior as a sensible default policy.
        sorted_tools = sorted(available_tools, key=lambda t: float(t.get("default_latency_ms", 500.0)))
        pick = sorted_tools[0] if self.rng.random() < 0.8 else self.rng.choice(sorted_tools)
        action_json = json.dumps({"type": "tool_call", "tool_name": pick["name"], "tool_args": {"query": "default"}})
        return parse_model_output(action_json)
