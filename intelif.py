from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llama import LlamaModel, ModelConfig
from lora import LoraConfig, inject_lora

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


def create_orthogonal_tensors(choices: int, dim: int) -> torch.Tensor:
    assert choices <= dim

    rand = torch.randn(dim, choices)
    q, _ = torch.linalg.qr(rand)

    return q.T


@dataclass
class IntelIfConfig:
    model_id: str
    choice_dim: int = 256
    temperature: float = 0.07
    lora_config: LoraConfig | None = None
    freeze_base: bool = True


class IntelIf(nn.Module):
    def __init__(self, cfg: IntelIfConfig, base_model_cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self.base_model_cfg = base_model_cfg

        self.base_model = LlamaModel.from_pretrained(
            model_id=cfg.model_id, cfg=base_model_cfg
        )

        if self.cfg.freeze_base:
            self.base_model.requires_grad_(False)

        if self.cfg.lora_config is not None:
            inject_lora(
                self.base_model,
                self.cfg.lora_config,
            )

        self.model_to_choice = nn.Linear(
            in_features=self.base_model_cfg.model_dim,
            out_features=self.cfg.choice_dim,
            bias=False,
        )
        self.choice_to_model = nn.Linear(
            in_features=self.cfg.choice_dim,
            out_features=self.base_model_cfg.model_dim,
            bias=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        lengths: Tensor,
        choice_slots: Tensor,
        num_choices: int,
    ) -> Tensor:

        assert choice_slots.shape == input_ids.shape

        b = lengths.shape[0]

        text_embeds: Tensor = self.base_model.embed_tokens(input_ids)  # b, t, model_dim

        choice_embeddings = create_orthogonal_tensors(
            num_choices, self.cfg.choice_dim
        ).to(input_ids.device)  # num_choices, choice_dim

        choice_model_proj: Tensor = self.choice_to_model(
            choice_embeddings.to(text_embeds.dtype)
        )  # num_choices, model_dim

        choice_proj_per_position = choice_model_proj[
            choice_slots.clamp(min=0)
        ]  # b, t, model_dim

        input_embeds = torch.where(
            choice_slots[..., None] >= 0, choice_proj_per_position, text_embeds
        )

        last_hidden_state = self.base_model.forward_embeddings(input_embeds, lengths)
        decision_state = last_hidden_state[
            torch.arange(b, device=lengths.device), lengths - 1
        ]

        choice_state: Tensor = self.model_to_choice(decision_state)  # b, choice_dim
        choice_state = F.normalize(choice_state, dim=-1)

        choice_logits = choice_state.float() @ choice_embeddings.transpose(
            0, 1
        )  # b, num_choices

        return choice_logits / self.cfg.temperature
