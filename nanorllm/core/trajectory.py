from typing import Any, Literal
from dataclasses import dataclass, field
import torch

# 一轮agent-env 交互的记录单元
@dataclass
class Step:
    observation: Any = None # 这一步开始时，env给了agent什么输入——env.state。对于math self-refine 来说就是原题/retry feedback
    prompt_messages: list[dict] | None = None #送进模型的信息
    model_response: str | None = None # 模型训练对象，优化这段输出对应的token概率
    action: Any = None # response 的结构化版本。代表给env的接口，而不是给trainer的文本值
    # 以下是环境返回后需要更新的，即环境反馈
    reward: float = 0.0 # 可能会有per-step reward
    done: bool = False # 真正结束的是执行完这一轮action之后，所以除了Trajectory的terminated外还需要有step-level 的结束标志
    info: dict[str, Any] = field(default_factory=dict) # 能兼容更多env metadata info，方便调试



# rollout/trainer 的交换对象
@dataclass
class Trajectory:
    task_id: str | None = None
    steps: list[Step] = field(default_factory=list)
    final_reward: float = 0.0
    terminated: bool = False
    termination_reason: str | None = None

@dataclass
class StepRolloutView:
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    response_logprobs: torch.Tensor


@dataclass
class EpisodeRollout:
    trajectory: Trajectory
    step_views: list[StepRolloutView] = field(default_factory=list)
    task: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    advantage: float | None = None


    run_id: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainSample:
    input_ids: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None
    old_logprobs: torch.Tensor | None = None
    advantage: float | None = None
    view_kind: Literal["step", "prefix-compatible-episode-as-sequence"] | None = None
    task_id: str | None = None
    step_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def view(self) -> Literal["step", "prefix-compatible-episode-as-sequence"] | None:
        return self.view_kind
