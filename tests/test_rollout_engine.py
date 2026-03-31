from types import SimpleNamespace

import torch

from nanorllm.agents.math_agent import MathAgent
from nanorllm.envs.math_env import MathEnv
from nanorllm.rewards.math_reward import math_reward
from nanorllm.rollout.engine import RolloutEngine


class DummyLLM:
    def generate(self, messages, args):
        return {
            "text": "wrong",
            "prompt_ids": torch.tensor([1, 2]),
            "response_ids": torch.tensor([3]),
            "response_logprobs": torch.tensor([-0.1]),
        }


def test_run_episode_does_not_append_empty_step_when_hitting_max_steps():
    engine = RolloutEngine()
    agent = MathAgent("sys")
    env = MathEnv(reward_fn=math_reward, max_turn=5)
    args = SimpleNamespace(max_steps=1, max_turn=5, max_new_tokens=1, temperature=1.0)

    rollout = engine.run_episode(
        agent,
        env,
        DummyLLM(),
        {"task_id": "t1", "question": "1+1", "answer": "2"},
        args,
    )

    assert len(rollout.trajectory.steps) == 1
    assert len(rollout.step_views) == 1
    assert rollout.trajectory.terminated is True
    assert rollout.trajectory.termination_reason == "max_steps"

    final_step = rollout.trajectory.steps[-1]
    assert final_step.prompt_messages is not None
    assert final_step.model_response == "wrong"
    assert final_step.done is False
    assert final_step.info["termination_reason"] == "max_steps"
