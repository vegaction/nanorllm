import torch
import torch.nn.functional as F


def compute_token_logprobs(logits: torch.Tensor, labels: torch.Tensor, args) -> torch.Tensor:
    '''
    输入logits 返回labels对应的token log_probs
    1. 因为logits的物理含义是下一个token的logits分布，所以logits直接对应真正关注的labels，但是eos 位置不需要保留了（不关心）
    2. labels 需要向后shift 一位，因为labels和input_ids 一样，labels shift一位后是logits对应要解码的token
    3. 取log_softmax, 一般还需要先除以temperature
    4. 通过gather 获取shifted_labels 对应的结果，最终返回[B, T]
    '''
    shifted_logits = torch.concat(
        [logits.new_zeros(logits.shape[0], 1, logits.shape[-1]), logits],
        dim=1,
    )
    log_probs = F.log_softmax(shifted_logits / args.temperature, dim=-1) # 注意这里是log_softmax，这里计算量比较大，容易炸显存？
    token_logprobs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return  token_logprobs


def loss_agg(policy_loss, response_mask, mode):
    if mode ==  'seq-mean-token-mean':
        return policy_loss.sum(dim=1) / (response_mask.sum(dim=1) + 1e-8)
    else:
        raise NotImplementedError


def compute_policy_loss(logits: torch.Tensor, batch: list, args) -> torch.Tensor:
    token_probs = compute_token_logprobs(logits, batch['labels'], args)
    old_logprobs = batch['old_logprobs']
    advantages = batch['advantages'].unsqueeze(1)

    ratio = torch.exp(token_probs-old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)

    policy_loss = -torch.min(ratio* advantages, clipped_ratio* advantages)  * batch['loss_mask']
    loss = loss_agg(policy_loss, batch['loss_mask'], args.loss_agg_mode)
    return loss.mean()


def summarize_batch_metrics(
    advantages: torch.Tensor,
    loss: torch.Tensor,
) -> dict[str, float]:
    return {
        "loss": float(loss.detach().item()),
        "avg_advantage": float(advantages.detach().mean().item()),
    }
