from dataclasses import dataclass

import torch
from torch import Tensor, nn

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

        self.scorer = nn.Linear(
            in_features=self.base_model_cfg.model_dim,
            out_features=1,
            bias=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        lengths: Tensor,
        choice_slots: Tensor,
        choice_mask: Tensor,
    ) -> Tensor:

        assert choice_slots.shape == input_ids.shape
        assert choice_mask.shape[0] == input_ids.shape[0]

        input_embeds: Tensor = self.base_model.embed_tokens(input_ids)

        last_hidden_state = self.base_model.forward_embeddings(
            input_embeds, lengths
        )  # b, t, model_dim

        mask = choice_slots >= 0
        assert (mask.sum(dim=1) == choice_mask.sum(dim=1)).all()

        batch_indices, token_indices = mask.nonzero(as_tuple=True)
        slot_indices = choice_slots[batch_indices, token_indices]
        assert choice_mask[batch_indices, slot_indices].all()

        score_states = last_hidden_state[batch_indices, token_indices]
        scores: Tensor = self.scorer(score_states).squeeze(-1)

        logits = scores.new_full(choice_mask.shape, float("-inf"))
        logits[batch_indices, slot_indices] = scores

        return logits
