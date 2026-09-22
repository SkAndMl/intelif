import pytest
import torch
from transformers import AutoTokenizer

from intelif import (
    DEFAULT_MODEL_ID,
    ChoicePromptBuilder,
    IntelIfConfig,
    LlamaIntelIf,
    decision_loss,
    from_pretrained,
    probabilities_by_label,
    sample_action_anchors,
)
from llama import ModelConfig
from lora import LoraConfig, LoraLinear

STATE = "Why did I get charged twice for the same card transaction?"
QUESTION = "What best describes the user's intent?"
CHOICES = {
    "duplicate_card_payment": "A card transaction was charged more than once.",
    "cash_withdrawal_fee": "The user was charged a fee for withdrawing cash.",
    "cash_withdrawal_unrecognized": "The user does not recognize an ATM withdrawal.",
    "card_payment_reversed": "A card payment was reversed.",
}

TINY = ModelConfig(
    model_dim=64,
    head_dim=16,
    num_heads=4,
    num_kv_heads=2,
    num_layers=2,
    intermediate_size=128,
)
TINY_DECISION = IntelIfConfig(d_action=32)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)


@pytest.fixture(scope="module")
def builder(tokenizer):
    return ChoicePromptBuilder(tokenizer)


@pytest.fixture
def tiny():
    torch.manual_seed(0)

    return LlamaIntelIf(TINY, TINY_DECISION).eval()


@pytest.fixture(scope="module")
def pretrained():
    return from_pretrained(dtype=torch.float32).eval()


@pytest.mark.parametrize("k", [2, 5, 16, 32])
def test_anchors_are_unit_norm_and_mutually_orthogonal(k):
    anchors = sample_action_anchors(k, 256)
    gram = anchors @ anchors.T

    assert anchors.shape == (k, 256)
    assert torch.allclose(anchors.norm(dim=-1), torch.ones(k), atol=1e-5)
    assert torch.allclose(gram, torch.eye(k), atol=1e-5)


def test_anchors_reject_more_choices_than_dimensions():
    with pytest.raises(ValueError):
        sample_action_anchors(17, 16)


def test_anchors_differ_across_calls():
    assert not torch.allclose(sample_action_anchors(8, 64), sample_action_anchors(8, 64))


def test_anchors_are_reproducible_from_a_generator():
    first = sample_action_anchors(8, 64, generator=torch.Generator().manual_seed(7))
    second = sample_action_anchors(8, 64, generator=torch.Generator().manual_seed(7))

    assert torch.equal(first, second)


def test_builder_assigns_each_choice_one_anchor_row(builder):
    example = builder.build(STATE, QUESTION, CHOICES, gold="duplicate_card_payment")
    rows = example.anchor_slots[example.anchor_slots >= 0].tolist()

    assert sorted(rows) == list(range(len(CHOICES)))
    assert sorted(example.labels_by_row) == sorted(CHOICES)
    assert example.num_choices == len(CHOICES)


def test_builder_gold_row_points_at_the_gold_label(builder):
    generator = torch.Generator().manual_seed(3)

    for _ in range(25):
        example = builder.build(
            STATE, QUESTION, CHOICES, gold="cash_withdrawal_fee", generator=generator
        )

        assert example.labels_by_row[example.gold_row] == "cash_withdrawal_fee"


def test_builder_shuffles_both_order_and_anchor_binding(builder):
    generator = torch.Generator().manual_seed(11)
    seen_bindings = set()
    seen_orders = set()

    for _ in range(40):
        example = builder.build(STATE, QUESTION, CHOICES, generator=generator)
        seen_bindings.add(tuple(example.labels_by_row))
        seen_orders.add(tuple(example.anchor_slots[example.anchor_slots >= 0].tolist()))

    assert len(seen_bindings) > 1
    assert len(seen_orders) > 1


def test_builder_decision_position_is_the_final_token(builder):
    example = builder.build(STATE, QUESTION, CHOICES, shuffle=False)

    assert example.decision_position == len(example.input_ids) - 1
    assert example.anchor_slots[example.decision_position] == -1


def test_builder_rejects_unknown_gold(builder):
    with pytest.raises(ValueError):
        builder.build(STATE, QUESTION, CHOICES, gold="not_a_label")


def test_builder_rejects_single_candidate(builder):
    with pytest.raises(ValueError):
        builder.build(STATE, QUESTION, {"only": "the only option"})


def test_builder_can_place_anchor_after_description(tokenizer):
    before = ChoicePromptBuilder(tokenizer, anchor_before_description=True)
    after = ChoicePromptBuilder(tokenizer, anchor_before_description=False)

    first_slot = lambda b: (b.build(STATE, QUESTION, CHOICES, shuffle=False).anchor_slots >= 0).nonzero()[0].item()

    assert first_slot(after) > first_slot(before)


@torch.no_grad()
def test_injection_writes_anchor_vectors_into_their_slots(tiny, builder):
    example = builder.build(STATE, QUESTION, CHOICES, shuffle=False)
    batch = builder.collate([example])
    anchors = sample_action_anchors(example.num_choices, TINY_DECISION.d_action)

    embeds = tiny.build_inputs_embeds(batch.input_ids, batch.anchor_slots, anchors)
    expected = tiny.anchor_proj(anchors)
    slots = batch.anchor_slots[0]

    for position in (slots >= 0).nonzero().flatten().tolist():
        assert torch.equal(embeds[0, position], expected[slots[position]])


@torch.no_grad()
def test_injection_leaves_other_positions_untouched(tiny, builder):
    example = builder.build(STATE, QUESTION, CHOICES, shuffle=False)
    batch = builder.collate([example])
    anchors = sample_action_anchors(example.num_choices, TINY_DECISION.d_action)

    embeds = tiny.build_inputs_embeds(batch.input_ids, batch.anchor_slots, anchors)
    plain = tiny.model.embed_tokens(batch.input_ids)
    keep = batch.anchor_slots < 0

    assert torch.equal(embeds[keep], plain[keep])


@torch.no_grad()
def test_logits_are_equivariant_to_anchor_row_permutation(tiny, builder):
    example = builder.build(STATE, QUESTION, CHOICES, shuffle=False)
    batch = builder.collate([example])
    k = example.num_choices
    anchors = sample_action_anchors(k, TINY_DECISION.d_action)

    permutation = torch.randperm(k)
    permuted_anchors = torch.empty_like(anchors)
    permuted_anchors[permutation] = anchors

    permuted_slots = batch.anchor_slots.clone()
    slots = permuted_slots >= 0
    permuted_slots[slots] = permutation[permuted_slots[slots]]

    original = tiny(**batch.model_inputs(anchors))
    permuted = tiny(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        anchor_slots=permuted_slots,
        decision_position=batch.decision_position,
        anchors=permuted_anchors,
        choice_mask=batch.choice_mask,
    )

    assert torch.allclose(permuted[0, permutation], original[0], atol=1e-6)


@torch.no_grad()
def test_logits_depend_on_the_anchor_basis(tiny, builder):
    example = builder.build(STATE, QUESTION, CHOICES, shuffle=False)
    batch = builder.collate([example])

    first = tiny(**batch.model_inputs(sample_action_anchors(example.num_choices, 32)))
    second = tiny(**batch.model_inputs(sample_action_anchors(example.num_choices, 32)))

    assert not torch.allclose(first, second, atol=1e-4)


@torch.no_grad()
def test_unavailable_choices_are_masked_out(tiny, builder):
    small = builder.build(STATE, QUESTION, dict(list(CHOICES.items())[:2]), shuffle=False)
    large = builder.build(STATE, QUESTION, CHOICES, shuffle=False)
    batch = builder.collate([small, large])
    anchors = sample_action_anchors(large.num_choices, TINY_DECISION.d_action)

    logits = tiny(**batch.model_inputs(anchors))
    probs = logits.softmax(dim=-1)

    assert batch.choice_mask[0].tolist() == [True, True, False, False]
    assert probs[0, 2:].sum().item() == 0.0
    assert probs[0, :2].sum().item() == pytest.approx(1.0, abs=1e-6)
    assert probs[1].sum().item() == pytest.approx(1.0, abs=1e-6)


@torch.no_grad()
def test_padding_does_not_change_logits(tiny, builder):
    short = builder.build("Fees?", QUESTION, CHOICES, shuffle=False)
    long = builder.build(STATE * 3, QUESTION, CHOICES, shuffle=False)
    anchors = sample_action_anchors(len(CHOICES), TINY_DECISION.d_action)

    alone = tiny(**builder.collate([short]).model_inputs(anchors))
    padded = tiny(**builder.collate([short, long]).model_inputs(anchors))

    assert torch.allclose(alone[0], padded[0], atol=1e-4)


def test_gradients_reach_projections_and_lora_only(builder):
    torch.manual_seed(0)
    model = LlamaIntelIf(TINY, TINY_DECISION)
    model.apply_lora(LoraConfig(rank=4))

    example = builder.build(STATE, QUESTION, CHOICES, gold="duplicate_card_payment")
    batch = builder.collate([example])
    anchors = sample_action_anchors(example.num_choices, TINY_DECISION.d_action)

    decision_loss(model(**batch.model_inputs(anchors)), batch.gold_row).backward()

    assert model.anchor_proj.weight.grad is not None
    assert model.decision_proj.weight.grad is not None
    assert model.log_scale.grad is not None

    lora_grads = [p.grad for n, p in model.named_parameters() if "lora_b" in n]
    assert lora_grads and all(g is not None and g.abs().sum() > 0 for g in lora_grads)

    assert model.model.embed_tokens.weight.grad is None
    assert model.model.layers[0].self_attn.q_proj.base.weight.grad is None
    assert model.model.layers[0].mlp.gate_proj.weight.grad is None


def test_trainable_parameters_exclude_the_backbone():
    torch.manual_seed(0)
    model = LlamaIntelIf(TINY, TINY_DECISION)
    model.apply_lora(LoraConfig(rank=4))

    trainable = sum(p.numel() for p in model.trainable_parameters())
    expected = (
        model.anchor_proj.weight.numel()
        + model.decision_proj.weight.numel()
        + model.log_scale.numel()
        + sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    )

    assert trainable == expected
    assert trainable < sum(p.numel() for p in model.parameters()) / 10


def test_model_has_no_lm_head(tiny):
    assert not hasattr(tiny, "lm_head")
    assert not any("lm_head" in name for name, _ in tiny.named_parameters())


@torch.no_grad()
def test_lora_is_identity_at_initialization(builder):
    torch.manual_seed(0)
    model = LlamaIntelIf(TINY, TINY_DECISION).eval()

    example = builder.build(STATE, QUESTION, CHOICES, shuffle=False)
    batch = builder.collate([example])
    anchors = sample_action_anchors(example.num_choices, TINY_DECISION.d_action)

    before = model(**batch.model_inputs(anchors))
    model.apply_lora(LoraConfig(rank=4))
    after = model(**batch.model_inputs(anchors))

    assert isinstance(model.model.layers[0].self_attn.q_proj, LoraLinear)
    assert torch.allclose(before, after, atol=1e-6)


def test_anchor_scale_matches_embedding_scale(pretrained):
    anchors = sample_action_anchors(16, pretrained.decision.d_action)

    with torch.no_grad():
        injected = pretrained.anchor_proj(anchors).norm(dim=-1).mean()

    tokens = pretrained.model.embed_tokens.weight.norm(dim=-1).mean()

    assert injected.item() == pytest.approx(tokens.item(), rel=0.15)


@torch.no_grad()
def test_end_to_end_probabilities_are_a_distribution_over_labels(pretrained, builder):
    example = builder.build(STATE, QUESTION, CHOICES, gold="duplicate_card_payment")
    batch = builder.collate([example])
    anchors = sample_action_anchors(example.num_choices, pretrained.decision.d_action)

    logits = pretrained(**batch.model_inputs(anchors))
    probabilities = probabilities_by_label(logits, batch.labels_by_row)[0]

    assert logits.shape == (1, example.num_choices)
    assert set(probabilities) == set(CHOICES)
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)
