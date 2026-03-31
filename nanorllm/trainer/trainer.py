import logging
import time

from nanorllm.algos.grpo import  compute_advantage, group_by_task_id
from nanorllm.core.trajectory import Rollout
from nanorllm.trainer.collate import collate_train_batch, transform_episode_samples, transform_step_samples
from nanorllm.trainer.loss import compute_policy_loss
from nanorllm.rollout.collector import execute_tasks
import torch

logger = logging.getLogger(__name__)



def build_samples_from_rollouts(
    rollouts: list[Rollout],
    policy,
    args):
  
    grouped_rollouts = group_by_task_id(rollouts)
    training_rollouts = compute_advantage(
        grouped_rollouts
    )
    samples = []
    if args.mode == 'step':
        for rollout in training_rollouts:
            samples.extend(transform_step_samples(rollout)) 
    elif args.mode == 'prefix-compatible-episode-as-sequence':
        for rollout in training_rollouts:
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
    rollouts = execute_tasks(tasks, args.num_samples_per_task, rollout_fn)
    logger.info("Collected %s rollouts", len(rollouts))
    samples = build_samples_from_rollouts(rollouts, policy, args)
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
        trajectories = [rollout.trajectory for rollout in rollouts]
        return {
            "rollouts": rollouts,
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
    trajectories= [rollout.trajectory for rollout in rollouts]
    return {
        "rollouts": rollouts,
        "trajectories": trajectories,
        "samples": samples,
        "batch": batch,
        "metrics": metrics,
    }
