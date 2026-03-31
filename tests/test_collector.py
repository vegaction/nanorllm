from nanorllm.core.trajectory import Rollout, Step, StepRolloutView, Trajectory
from nanorllm.rollout.collector import execute_tasks, stats_rollout

import torch


def test_stats_rollout_uses_token_counts_for_prompt_and_response_lengths():
    rollout = Rollout(
        trajectory=Trajectory(
            task_id="t1",
            steps=[
                Step(prompt_messages=[{"role": "user", "content": "q1"}]),
                Step(prompt_messages=[{"role": "user", "content": "q2"}]),
            ],
        ),
        step_views=[
            StepRolloutView(
                prompt_ids=torch.tensor([1, 2, 3]),
                response_ids=torch.tensor([4, 5]),
                response_logprobs=torch.tensor([-0.1, -0.2]),
            ),
            StepRolloutView(
                prompt_ids=torch.tensor([1, 2, 3, 4]),
                response_ids=torch.tensor([5]),
                response_logprobs=torch.tensor([-0.3]),
            ),
        ],
    )

    stats = stats_rollout(rollout)

    assert stats == {
        "num_steps": 2,
        "prompt_length": 7,
        "response_length": 3,
    }


def test_execute_tasks_attaches_run_metadata():
    def rollout_fn(task):
        return Rollout(
            trajectory=Trajectory(task_id=task["task_id"], steps=[Step()]),
            step_views=[
                StepRolloutView(
                    prompt_ids=torch.tensor([1, 2]),
                    response_ids=torch.tensor([3]),
                    response_logprobs=torch.tensor([-0.1]),
                ),
            ],
        )

    tasks = [{"task_id": "gsm8k-001", "question": "1+1", "answer": "2"}]

    rollouts = execute_tasks(tasks, num_samples_per_task=2, rollout_fn=rollout_fn)

    assert len(rollouts) == 2
    assert rollouts[0].run_id == "gsm8k-001_sample1"
    assert rollouts[1].run_id == "gsm8k-001_sample2"
    assert rollouts[0].stats == {
        "num_steps": 1,
        "prompt_length": 2,
        "response_length": 1,
    }
    assert "rollout_time" in rollouts[0].timing
