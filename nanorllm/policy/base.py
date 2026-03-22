from abc import ABC, abstractmethod

import torch


class BasePolicy(ABC):
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device

    @property
    @abstractmethod
    def model(self) -> torch.nn.Module:
        raise NotImplementedError

    @property
    @abstractmethod
    def tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        raise NotImplementedError

    def parameters(self):
        return self.model.parameters()


def build_policy(model_name: str, device: str) -> BasePolicy:
    from nanorllm.policy.hf_causal import HFCausalPolicy

    return HFCausalPolicy(model_name=model_name, device=device)
