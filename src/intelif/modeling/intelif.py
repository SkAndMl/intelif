from torch import Tensor, nn

from intelif.modeling.qwen import Qwen3Model

MODEL_ID = "Qwen/Qwen3-4B"


class IntelIfModel(nn.Module):
    def __init__(self, base_model: Qwen3Model):
        super().__init__()

        self.base_model = base_model
        self.scorer = nn.Linear(
            in_features=base_model.cfg.model_dim,
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
        score_states = score_states.to(self.scorer.weight.dtype)
        scores: Tensor = self.scorer(score_states).squeeze(-1)

        logits = scores.new_full(choice_mask.shape, float("-inf"))
        logits[batch_indices, slot_indices] = scores

        return logits
