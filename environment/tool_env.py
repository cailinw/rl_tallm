from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.api_registry import APIRegistry
from environment.api_executor import APIExecutor
from environment.latency_tracker import LatencyTracker
from training.reward import compute_reward


class ToolEnv:
    def __init__(
        self,
        api_registry: APIRegistry,
        executor: APIExecutor,
        latency_tracker: LatencyTracker,
        reward_config: Dict[str, Any],
        max_steps: int = 8,
    ):
        self.api_registry = api_registry
        self.executor = executor
        self.latency_tracker = latency_tracker
        self.reward_config = reward_config
        self.max_steps = max_steps
        self.current_task: Dict[str, Any] = {}
        self.trajectory: List[Dict[str, Any]] = []
        self.done = False
        self.final_answer = ""

    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.current_task = task
        self.trajectory = []
        self.done = False
        self.final_answer = ""
        return {
            "query": task.get("query", ""),
            "available_tools": task.get("available_tools", []),
            "step_count": 0,
            "history": [],
        }

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode already finished. Call reset().")

        step_info: Dict[str, Any] = {"action": action}
        reward = 0.0

        if action.get("type") == "final_answer":
            self.final_answer = action.get("answer", "")
            self.done = True
            step_info["event"] = "final_answer"
        elif action.get("type") == "tool_call":
            tool_name = action.get("tool_name", "")
            tool_args = action.get("tool_args", {})
            if not self.api_registry.has_tool(tool_name):
                obs = {
                    "tool_result": "",
                    "latency_ms": 0.0,
                    "latency_tier": "SLOW",
                    "latency_delta": 3.0,
                    "step_count": len(self.trajectory) + 1,
                }
                step_info.update({"error": "unknown_tool", "tool_name": tool_name})
                self.trajectory.append({**obs, **step_info})
                if len(self.trajectory) >= self.max_steps:
                    self.done = True
                return obs, reward, self.done, step_info

            out = asyncio.run(self.executor.call(tool_name, tool_args))
            latency_ms = float(out.get("latency_ms", 0.0))
            self.latency_tracker.update(tool_name, latency_ms, timed_out=bool(out.get("timed_out", False)))
            latency_delta = self.latency_tracker.get_delta(tool_name, latency_ms)
            latency_tier = self.latency_tracker.get_tier(tool_name, latency_ms)
            obs = {
                "tool_result": out.get("result", ""),
                "latency_ms": latency_ms,
                "latency_tier": latency_tier,
                "latency_delta": latency_delta,
                "step_count": len(self.trajectory) + 1,
            }
            step_info.update(
                {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "success": out.get("success", False),
                    "error": out.get("error"),
                    "timed_out": out.get("timed_out", False),
                }
            )
            self.trajectory.append({**obs, **step_info})
        else:
            raise ValueError(f"Unknown action type: {action.get('type')}")

        if len(self.trajectory) >= self.max_steps:
            self.done = True

        if self.done:
            reward_parts = compute_reward(
                trajectory=self.trajectory,
                final_answer=self.final_answer,
                ground_truth=self.current_task.get("ground_truth", ""),
                required_tools=self.current_task.get("required_tools", []),
                config=self.reward_config,
            )
            reward = float(reward_parts["total"])
            step_info["reward_breakdown"] = reward_parts

        observation = {
            "query": self.current_task.get("query", ""),
            "available_tools": self.current_task.get("available_tools", []),
            "step_count": len(self.trajectory),
            "history": self.trajectory,
        }
        return observation, reward, self.done, step_info


def _build_smoke_registry() -> APIRegistry:
    from data.api_registry import ToolSpec

    registry = APIRegistry()
    registry.register(
        ToolSpec(
            name="weather_fast",
            description="Fast weather lookup",
            schema={"type": "object", "properties": {"city": {"type": "string"}}},
            category="Weather",
            default_latency_ms=80.0,
        )
    )
    registry.register(
        ToolSpec(
            name="weather_slow",
            description="Slow weather lookup",
            schema={"type": "object", "properties": {"city": {"type": "string"}}},
            category="Weather",
            default_latency_ms=900.0,
        )
    )
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--n_episodes", type=int, default=5)
    args = parser.parse_args()
    if not args.smoke_test:
        return

    registry = _build_smoke_registry()
    executor = APIExecutor(registry, mode="replay")
    tracker = LatencyTracker()
    env = ToolEnv(
        api_registry=registry,
        executor=executor,
        latency_tracker=tracker,
        reward_config={"w_latency": 0.3, "r_task_correct": 1.0, "r_task_partial": 0.3, "r_timeout_penalty": -2.0},
    )

    for i in range(args.n_episodes):
        obs = env.reset(
            {
                "query": f"weather question {i}",
                "available_tools": ["weather_fast", "weather_slow"],
                "ground_truth": f"Weather answer {i}",
                "required_tools": ["weather_fast"],
                "category": "Weather",
            }
        )
        _ = obs
        env.step({"type": "tool_call", "tool_name": "weather_fast", "tool_args": {"city": "SF"}})
        _, reward, done, info = env.step({"type": "final_answer", "answer": f"Weather answer {i}"})
        print(f"episode={i}, done={done}, reward={reward:.3f}, breakdown={info.get('reward_breakdown', {})}")


if __name__ == "__main__":
    main()
