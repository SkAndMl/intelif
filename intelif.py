import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from llama import LlamaModel, ModelConfig, fetch_hf_state
from lora import LoraConfig, apply_lora

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


@dataclass
class IntelIfConfig:
    d_action: int = 256
    normalize_decision: bool = True
    temperature: float = 0.07
    learn_temperature: bool = True
    max_logit_scale: float = 100.0
    freeze_backbone: bool = True


def sample_action_anchors(
    num_choices: int,
    d_action: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if num_choices > d_action:
        raise ValueError(f"cannot build {num_choices} orthogonal anchors in {d_action} dimensions")

    noise = torch.randn(d_action, num_choices, generator=generator, device=device, dtype=torch.float32)
    basis, upper = torch.linalg.qr(noise, mode="reduced")
    signs = torch.where(torch.diagonal(upper) < 0, -1.0, 1.0)

    return (basis * signs).T.to(dtype)


class LlamaIntelIf(nn.Module):
    def __init__(self, cfg: ModelConfig, decision: IntelIfConfig | None = None):
        super().__init__()

        self.cfg = cfg
        self.decision = decision or IntelIfConfig()

        self.model = LlamaModel(cfg)
        self.anchor_proj = nn.Linear(self.decision.d_action, cfg.model_dim, bias=False)
        self.decision_proj = nn.Linear(cfg.model_dim, self.decision.d_action, bias=False)

        log_scale = torch.tensor(math.log(1.0 / self.decision.temperature))

        if self.decision.learn_temperature:
            self.log_scale = nn.Parameter(log_scale)
        else:
            self.register_buffer("log_scale", log_scale, persistent=False)

        self.anchor_proj.weight.data.normal_(0.0, cfg.initialize_range)
        self.decision_proj.weight.data.normal_(0.0, cfg.initialize_range)

        if self.decision.freeze_backbone:
            self.model.requires_grad_(False)

    def apply_lora(self, cfg: LoraConfig) -> list[nn.Parameter]:
        return apply_lora(self.model, cfg)

    def match_anchor_scale_to_embeddings(self) -> None:
        embeddings = self.model.embed_tokens.weight.data

        self.anchor_proj.weight.data.normal_(0.0, embeddings.pow(2).mean().sqrt().item())

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def build_inputs_embeds(self, input_ids: Tensor, anchor_slots: Tensor, anchors: Tensor) -> Tensor:
        embeds = self.model.embed_tokens(input_ids)
        anchor_embeds = self.anchor_proj(anchors.to(embeds.dtype))
        filled = anchor_embeds[anchor_slots.clamp(min=0)]

        return torch.where(anchor_slots[..., None] >= 0, filled, embeds)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        anchor_slots: Tensor,
        decision_position: Tensor,
        anchors: Tensor,
        choice_mask: Tensor,
    ) -> Tensor:
        embeds = self.build_inputs_embeds(input_ids, anchor_slots, anchors)
        hidden = self.model(inputs_embeds=embeds, attention_mask=attention_mask)

        h = hidden[torch.arange(hidden.shape[0], device=hidden.device), decision_position]
        z = self.decision_proj(h)

        if self.decision.normalize_decision:
            z = F.normalize(z, dim=-1)

        scale = self.log_scale.exp().clamp(max=self.decision.max_logit_scale)
        logits = (z @ anchors.to(z.dtype).T) * scale

        return logits.masked_fill(~choice_mask, torch.finfo(logits.dtype).min)


def from_pretrained(
    model_id: str = DEFAULT_MODEL_ID,
    cfg: ModelConfig | None = None,
    decision: IntelIfConfig | None = None,
    lora: LoraConfig | None = None,
    dtype: torch.dtype = torch.float32,
) -> LlamaIntelIf:
    model = LlamaIntelIf(cfg or ModelConfig(), decision)

    state = fetch_hf_state(model_id)
    state = {k.removeprefix("model."): v for k, v in state.items() if k != "lm_head.weight"}
    model.model.load_state_dict(state, strict=True)

    model.match_anchor_scale_to_embeddings()

    if lora is not None:
        model.apply_lora(lora)

    return model.to(dtype)


@dataclass
class DecisionExample:
    input_ids: Tensor
    anchor_slots: Tensor
    decision_position: int
    num_choices: int
    labels_by_row: list[str]
    gold_row: int | None = None


@dataclass
class DecisionBatch:
    input_ids: Tensor
    attention_mask: Tensor
    anchor_slots: Tensor
    decision_position: Tensor
    choice_mask: Tensor
    gold_row: Tensor | None
    labels_by_row: list[list[str]] = field(default_factory=list)

    def model_inputs(self, anchors: Tensor) -> dict[str, Tensor]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "anchor_slots": self.anchor_slots,
            "decision_position": self.decision_position,
            "anchors": anchors,
            "choice_mask": self.choice_mask,
        }


class ChoicePromptBuilder:
    def __init__(self, tokenizer, anchor_before_description: bool = True):
        self.tokenizer = tokenizer
        self.anchor_before_description = anchor_before_description
        self.placeholder_id = tokenizer.pad_token_id

    def _extend(self, ids: list[int], slots: list[int], text: str) -> None:
        encoded = self.tokenizer(text, add_special_tokens=False)["input_ids"]

        ids.extend(encoded)
        slots.extend([-1] * len(encoded))

    def _extend_anchor(self, ids: list[int], slots: list[int], row: int) -> None:
        ids.append(self.placeholder_id)
        slots.append(row)

    def build(
        self,
        state: str,
        question: str,
        choices: dict[str, str],
        gold: str | None = None,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> DecisionExample:
        labels = list(choices)
        k = len(labels)

        if k < 2:
            raise ValueError("a choice needs at least two candidates")

        if gold is not None and gold not in choices:
            raise ValueError(f"gold label {gold!r} is not among the candidates")

        order = torch.randperm(k, generator=generator).tolist() if shuffle else list(range(k))
        rows = torch.randperm(k, generator=generator).tolist() if shuffle else list(range(k))

        ids: list[int] = []
        slots: list[int] = []
        labels_by_row: list[str | None] = [None] * k

        self._extend(ids, slots, f"STATE\n{state}\n\nQUESTION\n{question}\n\nCHOICES\n")

        for position, source in enumerate(order):
            label = labels[source]
            row = rows[position]
            labels_by_row[row] = label

            if self.anchor_before_description:
                self._extend_anchor(ids, slots, row)
                self._extend(ids, slots, f" {choices[label]}\n")
            else:
                self._extend(ids, slots, f"{choices[label]} ")
                self._extend_anchor(ids, slots, row)
                self._extend(ids, slots, "\n")

        self._extend(ids, slots, "\nDECISION:")

        gold_row = labels_by_row.index(gold) if gold is not None else None

        return DecisionExample(
            input_ids=torch.tensor(ids, dtype=torch.long),
            anchor_slots=torch.tensor(slots, dtype=torch.long),
            decision_position=len(ids) - 1,
            num_choices=k,
            labels_by_row=labels_by_row,
            gold_row=gold_row,
        )

    def collate(self, examples: list[DecisionExample]) -> DecisionBatch:
        width = max(len(e.input_ids) for e in examples)
        k_max = max(e.num_choices for e in examples)

        input_ids = torch.full((len(examples), width), self.placeholder_id, dtype=torch.long)
        attention_mask = torch.zeros(len(examples), width, dtype=torch.long)
        anchor_slots = torch.full((len(examples), width), -1, dtype=torch.long)
        choice_mask = torch.zeros(len(examples), k_max, dtype=torch.bool)

        for index, example in enumerate(examples):
            length = len(example.input_ids)
            input_ids[index, :length] = example.input_ids
            attention_mask[index, :length] = 1
            anchor_slots[index, :length] = example.anchor_slots
            choice_mask[index, : example.num_choices] = True

        gold = [e.gold_row for e in examples]

        return DecisionBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            anchor_slots=anchor_slots,
            decision_position=torch.tensor([e.decision_position for e in examples], dtype=torch.long),
            choice_mask=choice_mask,
            gold_row=None if any(g is None for g in gold) else torch.tensor(gold, dtype=torch.long),
            labels_by_row=[e.labels_by_row for e in examples],
        )


def decision_loss(logits: Tensor, gold_row: Tensor) -> Tensor:
    return F.cross_entropy(logits.float(), gold_row)


def probabilities_by_label(logits: Tensor, labels_by_row: list[list[str]]) -> list[dict[str, float]]:
    probs = logits.float().softmax(dim=-1)

    return [
        {label: probs[index, row].item() for row, label in enumerate(labels)}
        for index, labels in enumerate(labels_by_row)
    ]
