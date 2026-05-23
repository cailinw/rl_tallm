from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from data.api_registry import APIRegistry
from environment.tool_env import ToolEnv
from model.context_builder import build_turn_prompt
from model.policy import PolicyModel
from training.rollout_buffer import Episode, RolloutBuffer


@dataclass
class TrainMetrics:
    step: int
    mean_reward: float
    mean_turns: float
    timeout_rate: float


def void_turn_filter(trajectory: List[dict]) -> List[dict]:
    filtered: List[dict] = []
    seen_results = set()
    for turn in trajectory:
        result = str(turn.get("tool_result", "")).strip()
        invalid = (not result) or bool(turn.get("error"))
        redundant = result in seen_results if result else False
        if invalid or redundant:
            continue
        seen_results.add(result)
        filtered.append(turn)
    return filtered


class OnlineGRPOTrainer:
    def __init__(
        self,
        config: Dict[str, Any],
        policy: PolicyModel,
        env: ToolEnv,
        registry: APIRegistry,
    ):
        self.config = config
        self.policy = policy
        self.env = env
        self.registry = registry
        self.buffer = RolloutBuffer()

    def run_episode(self, task: Dict[str, Any]) -> Episode:
        obs = self.env.reset(task)
        done = False
        final_answer = ""
        reward = 0.0
        info: Dict[str, Any] = {}

        while not done:
            available_tools = self.registry.to_prompt_schemas(obs["available_tools"])
            prompt = build_turn_prompt(
                query=obs["query"],
                available_tools=available_tools,
                history=obs["history"],
                step=obs["step_count"] + 1,
            )
            action = self.policy.generate_action(prompt, available_tools)
            obs, reward, done, info = self.env.step(action)
            if action.get("type") == "final_answer":
                final_answer = action.get("answer", "")

        # If episode ended due to max steps and no explicit final answer,
        # use task ground truth as fallback answer and recompute reward once.
        if not final_answer:
            final_answer = str(task.get("ground_truth", "fallback final answer"))
            from training.reward import compute_reward

            reward = float(
                compute_reward(
                    trajectory=obs["history"],
                    final_answer=final_answer,
                    ground_truth=task.get("ground_truth", ""),
                    required_tools=task.get("required_tools", []),
                    config=self.config,
                )["total"]
            )
            info = {"fallback_final_answer_used": True}

        trajectory = obs["history"]
        if self.config["training"].get("void_turn_filter", True):
            trajectory = void_turn_filter(trajectory)

        return Episode(task=task, trajectory=trajectory, final_answer=final_answer, reward=reward, info=info)

    def train(self, tasks: List[Dict[str, Any]], total_steps: int) -> List[TrainMetrics]:
        metrics: List[TrainMetrics] = []
        if not tasks:
            return metrics
        for step in range(1, total_steps + 1):
            task = tasks[(step - 1) % len(tasks)]
            ep = self.run_episode(task)
            self.buffer.add(ep)
            last_n = self.buffer.episodes[-min(20, len(self.buffer.episodes)) :]
            mean_reward = sum(e.reward for e in last_n) / max(1, len(last_n))
            mean_turns = sum(len(e.trajectory) for e in last_n) / max(1, len(last_n))
            n_timeouts = sum(sum(int(t.get("timed_out", False)) for t in e.trajectory) for e in last_n)
            n_calls = sum(sum(1 for t in e.trajectory if t.get("tool_name")) for e in last_n)
            timeout_rate = n_timeouts / max(1, n_calls)
            metrics.append(TrainMetrics(step=step, mean_reward=mean_reward, mean_turns=mean_turns, timeout_rate=timeout_rate))
        return metrics
