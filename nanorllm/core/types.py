from dataclasses import dataclass, field
from typing import Any

@dataclass
class Action:
   value: Any

@dataclass
class RewardOutput: 
   # RewardOutput 是 reward function 的内部 richer result，适合在 env 内部使用，trainer 直接用里面的reward
   reward: float = 0.0
   metadata: dict[str, Any] = field(default_factory=dict)
   is_correct: bool | None = None