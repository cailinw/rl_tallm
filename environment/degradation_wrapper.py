from __future__ import annotations

import random
from typing import Any, Dict

from environment.api_executor import APIExecutor


class DegradationWrapper:
    def __init__(self, executor: APIExecutor, config: Dict[str, Any]):
        self.executor = executor
        self.config = config

    async def call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        output = await self.executor.call(tool_name, args)
        targets = set(self.config.get("target_tools", []))
        if targets and tool_name not in targets and "all" not in targets:
            return output

        mode = self.config.get("mode", "uniform")
        multiplier = float(self.config.get("latency_multiplier", 1.0))
        timeout_rate = float(self.config.get("timeout_rate", 0.0))

        if mode == "spike" and random.random() < 0.2:
            multiplier = max(multiplier, 3.0)

        output["latency_ms"] = float(output["latency_ms"]) * multiplier
        if random.random() < timeout_rate:
            output["success"] = False
            output["timed_out"] = True
            output["error"] = "timeout"
            output["result"] = ""
        return output
