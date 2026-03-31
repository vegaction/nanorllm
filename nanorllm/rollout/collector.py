import logging
import time
from typing import Any

from nanorllm.core.trajectory import EpisodeRollout

logger = logging.getLogger(__name__)


def stats_rollout(episode_rollout: EpisodeRollout) -> dict[str, Any]:
    # Keep prompt/response stats in the same unit: rollout token counts.
    num_steps = len(episode_rollout.trajectory.steps)
    prompt_length = sum(int(step_view.prompt_ids.numel()) for step_view in episode_rollout.step_views)
    response_length = sum(int(step_view.response_ids.numel()) for step_view in episode_rollout.step_views)
    return {
        "num_steps": num_steps,
        "prompt_length": prompt_length,
        "response_length": response_length,
    }

def execute_tasks(tasks, num_samples_per_task, rollout_fn) -> list[EpisodeRollout]:
    episode_outputs = []
    total_rollouts = len(tasks) * num_samples_per_task
    rollout_idx = 0
    for  task in tasks:

        for sample_idx in range(1, num_samples_per_task + 1):
            rollout_idx += 1

            run_id = f"{task.get('task_id', 'task')}_sample{sample_idx}"
            logger.info(f"Starting rollout {rollout_idx}/{total_rollouts} for task_id={task.get('task_id', 'task')} sample_idx={sample_idx}")
            start_time = time.perf_counter()
            rollout_result = rollout_fn(task)
            rollout_time = time.perf_counter() - start_time
            logger.info(f"Finished rollout {rollout_idx}/{total_rollouts} for task_id={task.get('task_id', 'task')} sample_idx={sample_idx} in {rollout_time:.2f} seconds")
            
            rollout_result.run_id = run_id
            rollout_result.stats = stats_rollout(rollout_result)
            rollout_result.timing['rollout_time'] = rollout_time
            episode_outputs.append(rollout_result)
    return episode_outputs
