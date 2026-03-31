# nanorllm

最小化复现 `agentic RL` 的核心闭环，当前只保留一条真实主路径：

题目 -> multi-turn rollout -> terminal reward -> grouped advantage -> train batch -> policy update

这个仓库不是完整复刻 `rLLM`，而是把其中最值得先吃透的那条链路压缩成一个可以读懂、可以跑通、可以改动的小版本。

## 当前定位

当前唯一主入口是 [examples/train_math_grpo.py](/Users/sl/caitian/nanorllm/examples/train_math_grpo.py)。

当前主 demo 固定为 `multi-turn math self-refine`。
这条主路径属于典型的 cumulative chat agent 设定，因此 `prefix-compatible-episode-as-sequence` 在这里是合理默认，而不是冷门特例。

和 `rLLM` 对齐的地方：

- 保留 `agent / env / rollout / trainer` 四层结构
- 保留 `Trajectory / Step` 作为交互语义对象
- 让 rollout 额外产出 `Rollout`，并从中派生训练样本供 trainer 消费
- 用同题多采样后的 grouped advantage 做最小 GRPO-lite

刻意不做的地方：

- Ray / verl / async worker
- critic
- 通用 workflow 系统
- tool use / search
- 大规模工程化配置

## 当前主线

现在仓库里只把这一条路径当作主路径：

- `HFCausalPolicy` 负责本地 `transformers` causal LM 的前向和采样
- `MathAgent` 负责维护 messages 和写入 `Trajectory`
- `MathEnv` 负责判断答案、给 retry feedback、控制 episode 结束
- `RolloutEngine` 负责驱动 agent-env-policy 循环
- `execute_tasks` 负责批量 rollout 收集，并补 run metadata
- `trainer` 负责 `rollouts -> grouped advantage -> collate -> loss -> optimizer.step()`

当前训练范式是一个最小的 `on-policy` PPO/GRPO-style actor-only loop：

- rollout 由当前 policy 现采
- `old_logprobs` 来自这次 rollout 时的旧策略
- trainer 直接在这批新鲜样本上做 clipped ratio 更新
- 当前没有 replay buffer，也没有长期混用历史策略数据

## 目录

```text
nanorllm/
  nanorllm/
    agents/
      base.py
      math_agent.py
    algos/
      grpo.py
    core/
      trajectory.py
      types.py
    datasets/
      simple_math.py
    envs/
      base.py
      math_env.py
    policy/
      base.py
      hf_causal.py
    rollout/
      collector.py
      engine.py
    trainer/
      collate.py
      loss.py
      trainer.py
    utils/
      util.py
  examples/
    train_math_grpo.py
```

## 核心对象

`Step`

- 一轮 `observation -> model_response/action -> env feedback` 的记录单元
- 只保留交互语义，不直接存训练用 token/logprob 字段

`Trajectory`

- 一个完整 episode 的容器
- 最小字段包括 `task_id`、`steps`、`final_reward`、`terminated`、`termination_reason`

`StepRolloutView`

- rollout 阶段记录的一步训练视图
- 保存 `prompt_ids / response_ids / response_logprobs`

`Rollout`

- rollout/collector 对 trainer 的统一输出对象
- 绑定 `trajectory`、`step_views`、原始 `task`
- 允许挂载 `run_id`、`stats`、`timing` 这类收集侧 metadata

`TrainSample`

- trainer 消费的序列训练样本
- 当前支持 `step`、`prefix-compatible-episode-as-sequence` 两种视图名
- `advantage` 写在训练样本上，而不是写回 `Step`

`MathAgent`

- 维护对话历史
- 把 env observation 和 model response 写入当前 `Trajectory`
- 不做 reward 判断

`MathEnv`

- 提供题目
- 检查 `\boxed{...}` 中的答案
- 回答错误时返回 retry feedback

`RolloutEngine`

- 驱动一条完整 episode
- 输出 `Rollout`

`execute_tasks`

- 批量执行 `tasks x num_samples_per_task`
- 调用 `rollout_fn(task)` 收集 `Rollout`
- 为每条 rollout 补 `run_id / stats / timing`

`GRPO-lite`

- `group_by_task_id`
- `compute_advantage`
- `Rollout -> TrainSample`

`Trainer`

- 消费 collector 产出的 `Rollout`
- 再把 grouped advantage 写回对应的训练视图
- 组 batch，计算 loss，执行 `optimizer.step()`

## 任务格式

当前 task schema 固定为：

```python
{
    "task_id": "gsm8k-001",
    "question": "...",
    "answer": "42",
}
```

环境规则固定为：

- `reset(task)` 返回 `{"question": ...}`
- 回答正确：`done=True, reward=1.0`
- 回答错误且未超轮数：`done=False, reward=0.0`，返回 retry feedback
- 最后一轮仍错误：`done=True, reward=0.0`

## 运行

先确保：

- 已创建虚拟环境 `.venv`
- 已安装依赖

运行主入口：

```bash
.venv/bin/python examples/train_math_grpo.py
```

## 数据流

当前最小训练流程：

```python
rollouts = execute_tasks(tasks, num_samples_per_task, rollout_fn)
grouped = group_by_task_id(rollouts)
rollouts = compute_advantage(grouped)
samples = [
    transform_episode_samples(rollout, policy.tokenize_messages)
    for rollout in rollouts
]
batch = collate_train_batch(samples, tokenizer, args)
logits = policy.forward(batch["input_ids"], batch["attention_mask"])
loss = compute_policy_loss(logits, batch, args)
```

当前 rollout 的最小 episode 输出形态是：

```python
Rollout(
    trajectory=...,
    step_views=[
        StepRolloutView(...),
        StepRolloutView(...),
    ],
    run_id="gsm8k-001_sample1",
    stats={"num_steps": 2, ...},
    timing={"rollout_time": 1.23},
)
```

当前 trainer 消费的主样本形态是：

```python
TrainSample(
    input_ids=...,
    loss_mask=...,
    old_logprobs=...,
    advantage=0.5,
    view_kind="prefix-compatible-episode-as-sequence",
)
```

也就是说，当前 trainer 主线依赖 rollout 阶段预先缓存好的 token ids 和 old logprobs，但这些训练字段不再挂在 `Step` 上，而是走单独的训练样本视图。

从训练范式上看，这里更接近“单轮采样后立刻更新”的最小 on-policy loop，而不是 off-policy / replay-buffer 风格的数据复用。

## 设计边界

当前边界是：

- `agent` 只关心对话状态和 trajectory 写入
- `env` 只关心环境转移和 reward
- `rollout engine` 负责 episode 循环，并额外产出该 episode 对应的 `StepRolloutView`
- `collector` 负责批量执行 task、重复采样、补 run metadata
- `trainer` 只关心 `rollouts -> grouped advantage -> batch -> loss -> step`

当前保留的一个现实折中是：

- rollout 的训练事实和交互语义仍然绑定在同一条 episode 输出上，但已经拆成 `Trajectory`、`StepRolloutView` 和 `TrainSample` 三条视图

这样做的目的是先把最小训练闭环跑稳，同时保留 `trajectory -> its step_views / train samples` 的归属关系；如果后面要进一步向 `rLLM` 的分层靠拢，再把这层 episode 输出收成更明确的 `RolloutResult`/`TokenTrajectory`。

## 目前已经有的训练能力

- grouped advantage
- response-only loss mask
- old logprobs
- clipped policy objective
- 可选 KL 项

所以现在更准确的说法不是“纯最小 REINFORCE”，而是“一个很小的 PPO/GRPO-like 训练闭环”。

## 当前不做的事

- async rollout
- actor / learner 解耦
- distributed training
- replay / buffer 系统
- checkpoint / resume
- 通用 config 系统
- 多环境 / 多 agent 抽象

## 建议的下一步

按这个顺序继续最顺：

### Phase 1: 收紧 collector 和 rollout 契约

- 固定 `execute_tasks(...) -> list[Rollout]` 这层接口
- 明确 `run_id / stats / timing / task` 哪些字段属于 collector 输出契约
- 固定 `Rollout -> TrainSample` 的最小契约
- 明确当前默认训练视图是 `prefix-compatible-episode-as-sequence`
- 继续把 `step` / `prefix-compatible-episode-as-sequence` 的适用边界写清楚

### Phase 2: 把配置从脚本里拿出来

- 增加一个最小 `TrainConfig`
- 把 model name、lr、max_steps、max_new_tokens、num_samples_per_task 收进 config
- 让 [examples/train_math_grpo.py](/Users/sl/caitian/nanorllm/examples/train_math_grpo.py) 只负责组装依赖和启动

### Phase 3: 补可解释性和调试能力

- 打印每条 trajectory 的 question / response / reward / advantage
- 在 train step 输出 `avg_reward`、`success_rate`
- 增加一个保存最近 rollout 的 debug dump

### Phase 4: 再决定是否继续向 `rLLM` 靠

如果你想更贴近 `rLLM`：

- 抽 workflow 层
- 让 agent/env 交互接口更稳定
- 把 trainer 和 rollout 的耦合再降一层

如果你想更偏教学 demo：

- 保持现在的四层结构
- 继续把每一层写得更短更直白
- 用 math 这个 demo 先把 PPO/GRPO 的关键概念讲透
