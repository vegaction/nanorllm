from dataclasses import dataclass, field
from typing import Any, Protocol
from nanorllm.core.types import RewardOutput 

class RewardFunction(Protocol):
    def __call__(self, task: dict[str, Any], action: str) -> RewardOutput: ...
