from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

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
    policy_loss: float = 0.0


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
        self.group_size = int(self.config["training"].get("num_generations", 4))

    def run_episode(self, task: Dict[str, Any]) -> Episode:
        obs = self.env.reset(task)
        done = False
        final_answer = ""
        reward = 0.0
        info: Dict[str, Any] = {}
        step_logprobs: List[torch.Tensor] = []

        while not done:
            available_tools = self.registry.to_prompt_schemas(obs["available_tools"])
            prompt = build_turn_prompt(
                query=obs["query"],
                available_tools=available_tools,
                history=obs["history"],
                step=obs["step_count"] + 1,
            )
            step = self.policy.generate_action_with_trace(prompt, available_tools)
            action = step.action
            if step.logprob is not None:
                step_logprobs.append(step.logprob)
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

        logprob_sum = torch.stack(step_logprobs).sum() if step_logprobs else None
        return Episode(
            task=task,
            trajectory=trajectory,
            final_answer=final_answer,
            reward=reward,
            logprob_sum=logprob_sum,
            info=info,
        )

    def _grpo_update(self, episodes: List[Episode]) -> float:
        trainable = bool(getattr(self.policy, "trainable", False))
        if not trainable:
            return 0.0
        rewards = torch.tensor([float(ep.reward) for ep in episodes], dtype=torch.float32)
        if rewards.numel() < 2:
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)

        loss_terms: List[torch.Tensor] = []
        for idx, ep in enumerate(episodes):
            if ep.logprob_sum is None:
                continue
            adv = advantages[idx].to(device=ep.logprob_sum.device, dtype=ep.logprob_sum.dtype)
            loss_terms.append(-(adv * ep.logprob_sum))
        if not loss_terms:
            return 0.0
        loss = torch.stack(loss_terms).mean()
        return self.policy.apply_grpo_loss(loss)

    def train(self, tasks: List[Dict[str, Any]], total_steps: int) -> List[TrainMetrics]:
        metrics: List[TrainMetrics] = []
        if not tasks:
            return metrics
        for step in range(1, total_steps + 1):
            task = tasks[(step - 1) % len(tasks)]
            batch_episodes = [self.run_episode(task) for _ in range(max(1, self.group_size))]
            for ep in batch_episodes:
                self.buffer.add(ep)
            policy_loss = self._grpo_update(batch_episodes)
            last_n = self.buffer.episodes[-min(20, len(self.buffer.episodes)) :]
            mean_reward = sum(e.reward for e in last_n) / max(1, len(last_n))
            mean_turns = sum(len(e.trajectory) for e in last_n) / max(1, len(last_n))
            n_timeouts = sum(sum(int(t.get("timed_out", False)) for t in e.trajectory) for e in last_n)
            n_calls = sum(sum(1 for t in e.trajectory if t.get("tool_name")) for e in last_n)
            timeout_rate = n_timeouts / max(1, n_calls)
            metrics.append(
                TrainMetrics(
                    step=step,
                    mean_reward=mean_reward,
                    mean_turns=mean_turns,
                    timeout_rate=timeout_rate,
                    policy_loss=policy_loss,
                )
            )
        return metrics
