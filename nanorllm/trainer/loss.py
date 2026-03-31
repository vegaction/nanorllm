import torch
import torch.nn.functional as F


def compute_token_logprobs(logits: torch.Tensor, labels: torch.Tensor, args) -> torch.Tensor:
    '''
    输入logits 返回labels对应的token log_probs
    1. 因为logits的物理含义是下一个token的logits分布，所以logits直接对应的labels是input_ids向后shift一位的结果（也就是模型要预测的token），所以需要在logits前面拼接一行全0的logits来对齐labels
    2. 取log_softmax, 一般还需要先除以temperature
    3. 通过gather 获取shifted_labels 对应的结果，最终返回[B, T]
    '''
    shifted_logits = torch.concat(
        [logits.new_zeros(logits.shape[0], 1, logits.shape[-1]), logits],
        dim=1,
    )
    log_probs = F.log_softmax(shifted_logits / args.temperature, dim=-1) # 注意这里是log_softmax，这里计算量比较大，为什么通常要异步 rollout和 trainer，因为这里容易炸显存
    token_logprobs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return  token_logprobs


def loss_agg(policy_loss, response_mask, mode):
    if mode ==  'seq-mean-token-mean':
        return policy_loss.sum(dim=1) / (response_mask.sum(dim=1) + 1e-8) # token mean
    else:
        raise NotImplementedError


def compute_policy_loss(logits: torch.Tensor, batch: list, args) -> torch.Tensor:
    token_probs = compute_token_logprobs(logits, batch['labels'], args)
    old_logprobs = batch['old_logprobs'] # 在transform_step_samples里已经把prompt部分的old_logprobs置0了，所以这里直接用就行
    advantages = batch['advantages'].unsqueeze(1)

    ratio = torch.exp(token_probs-old_logprobs) # ratio = πθ(a|s) / πθ_old(a|s) = exp(log πθ(a|s) - log πθ_old(a|s))
    clipped_ratio = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
    # advantage 可正可负，所以需要在里面乘，如果 advantage 是正的，ratio越大越好，clipped_ratio限制了ratio的上界；如果advantage是负的，ratio越小越好，clipped_ratio限制了ratio的下界
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
