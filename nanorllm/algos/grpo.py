from collections import defaultdict

import math

from nanorllm.core.trajectory import EpisodeRollout


RELATIVE_SIGNAL_EPS = 1e-8


def has_relative_signal(
    rollout_group: list[EpisodeRollout],
    eps: float = RELATIVE_SIGNAL_EPS,
) -> bool:
    if not rollout_group:
        return False

    rewards = [rollout.trajectory.final_reward for rollout in rollout_group]
    return max(rewards) - min(rewards) > eps


def compute_group_relative_reward(rollout_group: list[EpisodeRollout]) -> dict[int, float]:
    group_scores = [rollout.trajectory.final_reward for rollout in rollout_group]
    avg_reward = sum(group_scores) / len(group_scores)
    variance = sum((x-avg_reward)**2 for x in group_scores) / len(group_scores)
    std_reward = math.sqrt(variance)
    return {
        idx: (rollout.trajectory.final_reward - avg_reward) / (std_reward + RELATIVE_SIGNAL_EPS)
        for idx, rollout in enumerate(rollout_group)
    }



def group_by_task_id(episode_outputs: list[EpisodeRollout]) -> dict[str | None, list[EpisodeRollout]]:
    grouped_episode_outputs: dict[str | None, list[EpisodeRollout]] = defaultdict(list)
    for rollout in episode_outputs:
        grouped_episode_outputs[rollout.trajectory.task_id].append(rollout)
    return grouped_episode_outputs


def compute_advantage(
    grouped_episode_outputs: dict[str | None, list[EpisodeRollout]],
) -> dict[str | None, list[EpisodeRollout]]:
    rollouts = []
    for task_id, rollout_group in grouped_episode_outputs.items():
        if not has_relative_signal(rollout_group):
            continue

        episode_advantages = compute_group_relative_reward(rollout_group)
        for idx, rollout in enumerate(rollout_group):
            rollout.advantage = episode_advantages[idx]
            rollouts.append(rollout)
    return rollouts



        

        



