from typing import Any
import torch
from nanorllm.core.trajectory import Rollout, TrainSample


def _resolve_prompt_ids(rollout: Rollout, step_index: int, tokenize_messages) -> torch.Tensor:
    # We treat the last visible prompt as the truth source for the episode view.
    # For append-only chat agents this is effectively equivalent to an rLLM-like
    # cumulative replay view, but it is stricter because the final prompt already
    # contains any env/user turns inserted between model responses.
    if step_index >= len(rollout.trajectory.steps):
        raise ValueError("prefix-compatible-episode-as-sequence view requires prompt_messages on trajectory steps")
    prompt_messages = rollout.trajectory.steps[step_index].prompt_messages
    if prompt_messages is None:
        raise ValueError("prefix-compatible-episode-as-sequence view requires prompt_messages on trajectory steps")
    prompt_ids = tokenize_messages(prompt_messages, add_generation_prompt=True).view(-1).detach().cpu()
    if not torch.equal(prompt_ids, rollout.step_views[step_index].prompt_ids):
        raise ValueError("step prompt_ids do not match retokenized prompt_messages")
    return prompt_ids


def transform_episode_samples(rollout: Rollout, tokenize_messages) -> TrainSample:
    if not rollout.step_views:
        raise ValueError("prefix-compatible-episode-as-sequence view requires at least one step_view")

    last_step_index = len(rollout.step_views) - 1
    prompt_ids = _resolve_prompt_ids(rollout, last_step_index, tokenize_messages)
    response_ids = rollout.step_views[last_step_index].response_ids
    input_ids = torch.concat([prompt_ids, response_ids], dim=0)

    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    old_logprobs = torch.zeros_like(input_ids, dtype=torch.float32)
    for step_index, step_view in enumerate(rollout.step_views):
        step_prompt_ids = _resolve_prompt_ids(rollout, step_index, tokenize_messages)
        current_len = step_prompt_ids.shape[-1]
        response_len = step_view.response_ids.shape[-1]

        # 如果 agent 不是追加历史，而是重写了上下文，或者有 summary/compression 逻辑，把前面的长历史压缩成了后面的短 prompt，这种情况下就不满足 prefix-compatible 的要求了
        if current_len > prompt_ids.shape[-1] or not torch.equal(prompt_ids[:current_len], step_prompt_ids):
            raise ValueError("prefix-compatible-episode-as-sequence view requires each step prompt_ids to be a prefix of the final prompt_ids")
        loss_mask[current_len : current_len + response_len] = 1.0
        old_logprobs[current_len : current_len + response_len] = step_view.response_logprobs

    return TrainSample(input_ids=input_ids, loss_mask=loss_mask, old_logprobs=old_logprobs, advantage=rollout.advantage, view_kind="prefix-compatible-episode-as-sequence", task_id=rollout.trajectory.task_id, metadata={"num_steps": len(rollout.step_views)})


def transform_step_samples(rollout: Rollout) -> list[TrainSample]:
    train_samples = []
    for step_index, step_view in enumerate(rollout.step_views):
        input_ids = torch.concat([step_view.prompt_ids, step_view.response_ids], dim=0)
        loss_mask = torch.concat([torch.zeros_like(step_view.prompt_ids, dtype=torch.float32), torch.ones_like(step_view.response_ids, dtype=torch.float32)], dim=0)
        old_logprobs = torch.concat([torch.zeros_like(step_view.prompt_ids, dtype=torch.float32), step_view.response_logprobs.to(dtype=torch.float32)], dim=0)
        train_samples.append(TrainSample(input_ids=input_ids, loss_mask=loss_mask, old_logprobs=old_logprobs, advantage=rollout.advantage, view_kind="step", task_id=rollout.trajectory.task_id, step_index=step_index))
    return train_samples


def collate_train_batch(samples: list[TrainSample], tokenizer, args, device: str | None = None) -> dict[str, torch.Tensor]:
    '''
    把step-level的 train sample 整理成模型需要的batch。找到这批样本的最长序列，用来后续的padding补齐

    input_ids: [B, T]
    attention_mask: [B, T]
    labels: [B, T]
    response_mask: [B, T]
    advantages: [B]

    response_mask 和 attention_mask 的区别：
    attention_mask：参与模型前向的token（self.model(input_ids=input_ids, attention_mask=attention_mask)）
    response_mask：参与 policy loss的token（seq_probs = masked_sequence_logprobs(token_probs, batch['response_mask'])）
    '''
    pad_token_id = tokenizer.pad_token_id

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_response_mask = []
    batch_advantages = []
    batch_old_logprobs = []

    max_len = 0

    for sample in samples:
        input_ids = sample.input_ids
        old_logprobs = sample.old_logprobs
        loss_mask = sample.loss_mask

        if input_ids.shape[0] > args.max_length:
            start = input_ids.shape[0] - args.max_length
            input_ids = input_ids[start:]
            old_logprobs = old_logprobs[start:]
            loss_mask = loss_mask[start:]

        max_len = max(max_len, input_ids.shape[0])
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)
        batch_advantages.append(torch.tensor(sample.advantage, dtype=torch.float32))
        batch_old_logprobs.append(old_logprobs)
        batch_response_mask.append(loss_mask)

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    padded_response_mask = []
    padded_old_logprobs = []

    for input_ids, attention_mask, labels, response_mask, old_logprobs in zip(batch_input_ids, batch_attention_mask, batch_labels, batch_response_mask, batch_old_logprobs, strict=True):
        pad_len = max_len - input_ids.shape[0]
        padded_input_ids.append(torch.nn.functional.pad(input_ids, (0, pad_len), value=pad_token_id))
        padded_attention_mask.append(torch.nn.functional.pad(attention_mask, (0, pad_len), value=0))
        padded_labels.append(torch.nn.functional.pad(labels, (0, pad_len), value=pad_token_id))
        padded_response_mask.append(torch.nn.functional.pad(response_mask, (0, pad_len), value=0))
        padded_old_logprobs.append(torch.nn.functional.pad(old_logprobs, (0, pad_len), value=0.0))

    batch = {
        "input_ids": torch.stack(padded_input_ids, dim=0),
        "attention_mask": torch.stack(padded_attention_mask, dim=0),
        "labels": torch.stack(padded_labels, dim=0),
        "loss_mask": torch.stack(padded_response_mask, dim=0),
        "response_mask": torch.stack(padded_response_mask, dim=0),
        "advantages": torch.stack(batch_advantages, dim=0),
        "old_logprobs": torch.stack(padded_old_logprobs, dim=0),
    }
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}
    return batch
