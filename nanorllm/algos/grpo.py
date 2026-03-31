from collections import defaultdict
import math

from nanorllm.core.trajectory import Rollout


RELATIVE_SIGNAL_EPS = 1e-8


def has_relative_signal(
    rollout_group: list[Rollout],
    eps: float = RELATIVE_SIGNAL_EPS,
) -> bool:
    if not rollout_group:
        return False

    rewards = [rollout.trajectory.final_reward for rollout in rollout_group]
    return max(rewards) - min(rewards) > eps



def compute_advantage(
    grouped_rollouts: dict[str | None, list[Rollout]],
) -> list[Rollout]:
    rollouts = []
    for task_id, rollout_group in grouped_rollouts.items():
        # GRPO only learns from relative differences within the same task group.
        # If every rollout got (almost) the same reward, normalization would
        # collapse to ~0 and this group would not provide a useful preference signal.
        if not has_relative_signal(rollout_group):
            continue

        group_scores = [rollout.trajectory.final_reward for rollout in rollout_group]
        avg_reward = sum(group_scores) / len(group_scores)
        variance = sum((x-avg_reward)**2 for x in group_scores) / len(group_scores)
        std_reward = math.sqrt(variance)

        for rollout in rollout_group:
            rollout.advantage = (rollout.trajectory.final_reward - avg_reward) / (std_reward + RELATIVE_SIGNAL_EPS)
            rollouts.append(rollout)

    return rollouts



def group_by_task_id(rollouts: list[Rollout]) -> dict[str | None, list[Rollout]]:
    grouped_rollouts: dict[str | None, list[Rollout]] = defaultdict(list)
    for rollout in rollouts:
        grouped_rollouts[rollout.trajectory.task_id].append(rollout)
    return grouped_rollouts

        
