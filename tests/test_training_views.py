import torch

from nanorllm.core.trajectory import EpisodeRollout, Step, StepRolloutView, Trajectory
from nanorllm.trainer.collate import (
    transform_episode_samples,
    transform_step_samples,
)


def test_transform_step_samples_records_step_metadata():
    rollout = EpisodeRollout(
        trajectory=Trajectory(
            task_id="t2",
            steps=[Step(prompt_messages=[{"role": "user", "content": "q"}])],
        ),
        step_views=[
            StepRolloutView(
                prompt_ids=torch.tensor([1, 2]),
                response_ids=torch.tensor([3, 4]),
                response_logprobs=torch.tensor([-0.1, -0.2]),
            ),
        ],
        advantage=1.25,
    )

    samples = transform_step_samples(rollout)

    assert len(samples) == 1
    assert samples[0].view == "step"
    assert samples[0].task_id == "t2"
    assert samples[0].step_index == 0
    assert torch.equal(samples[0].old_logprobs, torch.tensor([0.0, 0.0, -0.1, -0.2]))


def test_transform_episode_samples_returns_prefix_compatible_episode_view():
    rollout = EpisodeRollout(
        trajectory=Trajectory(
            task_id="t3",
            steps=[
                Step(prompt_messages=[{"role": "user", "content": "q1"}]),
                Step(
                    prompt_messages=[
                        {"role": "user", "content": "q1"},
                        {"role": "assistant", "content": "a1"},
                        {"role": "user", "content": "q2"},
                    ]
                ),
            ],
        ),
        step_views=[
            StepRolloutView(
                prompt_ids=torch.tensor([1, 2]),
                response_ids=torch.tensor([9]),
                response_logprobs=torch.tensor([-0.1]),
            ),
            StepRolloutView(
                prompt_ids=torch.tensor([1, 2, 9, 5]),
                response_ids=torch.tensor([6]),
                response_logprobs=torch.tensor([-0.3]),
            ),
        ],
        advantage=0.75,
    )

    prompt_map = {
        ("q1",): torch.tensor([1, 2]),
        ("q1", "a1", "q2"): torch.tensor([1, 2, 9, 5]),
    }

    def tokenize_messages(messages, add_generation_prompt=True):
        key = tuple(message["content"] for message in messages)
        return prompt_map[key]

    sample = transform_episode_samples(rollout, tokenize_messages)

    assert sample.view == "prefix-compatible-episode-as-sequence"
    assert sample.task_id == "t3"
    assert sample.step_index is None
    assert sample.metadata == {"num_steps": 2}
    assert torch.equal(sample.input_ids, torch.tensor([1, 2, 9, 5, 6]))
    assert torch.equal(sample.loss_mask, torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0]))
