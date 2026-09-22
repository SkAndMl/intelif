import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class LoraConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, cfg: LoraConfig):
        super().__init__()

        self.base = base
        self.base.requires_grad_(False)

        self.lora_a = nn.Parameter(torch.empty(cfg.rank, base.in_features, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, cfg.rank, dtype=base.weight.dtype))
        self.dropout = nn.Dropout(cfg.dropout)
        self.scaling = cfg.alpha / cfg.rank

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        update = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)

        return self.base(x) + update * self.scaling


def apply_lora(root: nn.Module, cfg: LoraConfig) -> list[nn.Parameter]:
    for module in list(root.modules()):
        for name, child in list(module.named_children()):
            if name in cfg.targets and isinstance(child, nn.Linear):
                setattr(module, name, LoraLinear(child, cfg))

    return [p for name, p in root.named_parameters() if "lora_" in name]


def lora_state_dict(root: nn.Module) -> dict[str, Tensor]:
    return {name: p.detach() for name, p in root.named_parameters() if "lora_" in name}
