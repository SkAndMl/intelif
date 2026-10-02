from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


class LoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, config: LoraConfig):
        super().__init__()

        self.base = base
        self.base.requires_grad_(False)

        self.lora_A = nn.Linear(base.in_features, config.r, bias=False)
        self.lora_B = nn.Linear(config.r, base.out_features, bias=False)

        self.dropout = nn.Dropout(config.lora_dropout)
        self.scaling = config.lora_alpha / config.r

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        base = self.base(x)

        lora = self.lora_B(self.lora_A(self.dropout(x)))

        return base + self.scaling * lora


def inject_lora(
    module: nn.Module,
    config: LoraConfig,
):

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in config.target_modules:
            setattr(
                module,
                name,
                LoraLinear(
                    base=child,
                    config=config,
                ),
            )
        else:
            inject_lora(child, config)


@torch.no_grad()
def merge_lora(module: nn.Module) -> None:
    for name, child in module.named_children():
        if not isinstance(child, LoraLinear):
            merge_lora(child)
            continue

        weight = child.base.weight
        delta = child.scaling * (
            child.lora_B.weight.float() @ child.lora_A.weight.float()
        )

        weight.copy_((weight.float() + delta.to(weight.device)).to(weight.dtype))
        setattr(module, name, child.base)
