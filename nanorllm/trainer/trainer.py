import logging
import time

from nanorllm.algos.grpo import  compute_advantage, group_by_task_id
from nanorllm.core.trajectory import TrainSample, EpisodeRollout
from nanorllm.trainer.collate import collate_train_batch, transform_episode_samples, transform_step_samples
from nanorllm.trainer.loss import compute_policy_loss

import torch

logger = logging.getLogger(__name__)


def collect_rollouts(tasks, num_samples_per_task, rollout_fn) -> list[EpisodeRollout]:
    """
    Collect multi-sample rollout results for each task.

    Each episode output keeps the trajectory-level semantics and the step-level
    training samples together as `(trajectory, samples)`.
    """
    episode_outputs = []
    total_rollouts = len(tasks) * num_samples_per_task
    rollout_idx = 0
    for task_idx, task in enumerate(tasks, start=1):
        logger.info(
            "Starting task %s/%s (%s): %s",
            task_idx,
            len(tasks),
            task.get("task_id", "unknown-task"),
            task["question"],
        )
        for sample_idx in range(1, num_samples_per_task + 1):
            rollout_idx += 1
            rollout_start = time.perf_counter()
            logger.info(
                "Collecting rollout %s/%s for task %s (sample %s/%s)",
                rollout_idx,
                total_rollouts,
                task.get("task_id", "unknown-task"),
                sample_idx,
                num_samples_per_task,
            )
            rollout_result = rollout_fn(task)
            episode_outputs.append(rollout_result)
            rollout_seconds = time.perf_counter() - rollout_start
            logger.info(
                "Finished rollout %s/%s in %.2fs: final_reward=%s termination=%s",
                rollout_idx,
                total_rollouts,
                rollout_seconds,
                rollout_result.trajectory.final_reward,
                rollout_result.trajectory.termination_reason,
            )
        logger.info("Completed task %s/%s", task_idx, len(tasks))
    return episode_outputs




def build_samples_from_episode_outputs(
    episode_outputs: list[EpisodeRollout],
    policy,
    args):
  
    grouped_episode_outputs = group_by_task_id(episode_outputs)
    rollouts = compute_advantage(
        grouped_episode_outputs
    )
    samples = []
    if args.mode in {'step', 'step_as_sequence'}:
        for rollout in rollouts:
            samples.extend(transform_step_samples(rollout)) 
    elif args.mode in {
        'prefix-compatible-episode-as-sequence',
        'prefix_episode',
        'token',
        'episode',
    }:
        for rollout in rollouts:
            samples.append(transform_episode_samples(rollout, policy.tokenize_messages)) 
    else:
        raise ValueError(f"Unsupported training view mode: {args.mode}")
    return samples




def aggregate_train_metrics(minibatch_metrics):
    if not minibatch_metrics:
        return {"loss": 0.0, "avg_advantage": 0.0}

    weighted_loss_sum = 0.0
    weighted_advantage_sum = 0.0
    total_samples = 0

    for minibatch_metric in minibatch_metrics:
        num_samples = minibatch_metric["num_samples"]
        weighted_loss_sum += float(minibatch_metric["loss"].item()) * num_samples
        weighted_advantage_sum += float(minibatch_metric["advantage"].item()) * num_samples
        total_samples += num_samples

    return {
        "loss": weighted_loss_sum / total_samples,
        "avg_advantage": weighted_advantage_sum / total_samples,
    }


def iter_minibatches(samples, train_batch_size):
    for i in range(0, len(samples), train_batch_size):
        yield samples[i: i+train_batch_size]
    




def run_train_epoch(
    tasks,
    rollout_fn,
    policy,
    tokenizer,
    optimizer,
    args
):
    
    logger.info(
        "Starting train epoch: tasks=%s samples_per_task=%s max_steps=%s max_new_tokens=%s",
        len(tasks),
        args.num_samples_per_task,
        args.max_steps,
        args.max_new_tokens,
    )
    episode_outputs = collect_rollouts(tasks, args.num_samples_per_task, rollout_fn)
    logger.info("Collected %s episode rollouts", len(episode_outputs))
    samples = build_samples_from_episode_outputs(episode_outputs, policy, args)
    logger.info("Built %s train samples for mode=%s", len(samples), args.mode)
    train_start = time.perf_counter()

    minibatch_metrics = []
    batch = None

    if not samples:
        metrics = aggregate_train_metrics(minibatch_metrics)
        logger.info(
            "Finished train epoch in %.2fs with metrics=%s",
            time.perf_counter() - train_start,
            metrics,
        )
        trajectories = [rollout.trajectory for rollout in episode_outputs]
        return {
            "episode_outputs": episode_outputs,
            "trajectories": trajectories,
            "samples": samples,
            "batch": batch,
            "metrics": metrics,
        }

    policy.model.train()
    for batch_samples in iter_minibatches(samples, args.train_batch_size):
        if batch_samples:
            batch = collate_train_batch(batch_samples, tokenizer, args, device=policy.device)
            optimizer.zero_grad()

            logger.info("Running train step on batch with shape=%s", tuple(batch["input_ids"].shape))
            logits = policy.forward(batch['input_ids'], batch['attention_mask'])
            loss = compute_policy_loss(logits, batch, args)

            loss.backward()
            optimizer.step()
            metrics = {
                        "loss": loss.detach(),
                        "advantage": batch['advantages'].detach().mean(),
                        "num_samples": int(batch['advantages'].shape[0]),
                    }
            minibatch_metrics.append(metrics)
    metrics = aggregate_train_metrics(minibatch_metrics)

    logger.info(
        "Finished train epoch in %.2fs with metrics=%s",
        time.perf_counter() - train_start,
        metrics,
    )
    trajectories= [rollout.trajectory for rollout in episode_outputs]
    return {
        "episode_outputs": episode_outputs,
        "trajectories": trajectories,
        "samples": samples,
        "batch": batch,
        "metrics": metrics,
    }
