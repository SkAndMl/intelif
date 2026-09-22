import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama import ModelConfig, from_pretrained

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
PROMPTS = [
    "Why did I get charged twice for the same card transaction?",
    "The capital of France is",
    "def fibonacci(n):",
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def reference():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    )

    return model.eval()


@pytest.fixture(scope="module")
def ours():
    return from_pretrained(MODEL_ID, dtype=torch.float32)


@pytest.fixture(scope="module")
def batch(tokenizer):
    return tokenizer(PROMPTS, return_tensors="pt", padding=True)


def test_config_matches_checkpoint():
    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).config
    cfg = ModelConfig()

    assert cfg.model_dim == hf.hidden_size
    assert cfg.head_dim == hf.head_dim
    assert cfg.num_heads == hf.num_attention_heads
    assert cfg.num_kv_heads == hf.num_key_value_heads
    assert cfg.num_layers == hf.num_hidden_layers
    assert cfg.vocab_size == hf.vocab_size
    assert cfg.intermediate_size == hf.intermediate_size
    assert cfg.rms_norm_eps == hf.rms_norm_eps
    assert cfg.tie_word_embeddings == hf.tie_word_embeddings
    assert cfg.context_length == hf.max_position_embeddings


def test_parameter_count_matches(reference, ours):
    assert sum(p.numel() for p in ours.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )


def test_weights_are_tied(ours):
    assert ours.lm_head.weight is ours.model.embed_tokens.weight


@torch.no_grad()
def test_hidden_states_match_layer_by_layer(reference, ours, batch):
    captured = []
    handles = [
        layer.register_forward_hook(lambda _m, _i, out: captured.append(out))
        for layer in ours.model.layers
    ]

    try:
        final = ours.model(**batch)
    finally:
        for handle in handles:
            handle.remove()

    reference_states = reference.model(**batch, output_hidden_states=True).hidden_states
    expected = reference_states[1:]
    captured = captured[:-1] + [final]
    keep = batch["attention_mask"].bool()

    assert len(captured) == len(expected) == ours.cfg.num_layers

    for index, (mine, theirs) in enumerate(zip(captured, expected)):
        diff = (mine[keep] - theirs[keep]).abs().max().item()
        scale = theirs[keep].abs().max().item()

        assert diff / scale < 1e-4, f"layer {index}: max abs diff {diff:.3e}, scale {scale:.3e}"


@torch.no_grad()
def test_logits_match(reference, ours, batch):
    mine = ours(**batch)
    theirs = reference(**batch).logits
    keep = batch["attention_mask"].bool()

    mine, theirs = mine[keep], theirs[keep]
    diff = (mine - theirs).abs().max().item()

    assert diff < 2e-3, f"max abs logit diff {diff:.3e}"
    assert torch.equal(mine.argmax(-1), theirs.argmax(-1))
    assert torch.allclose(mine.log_softmax(-1), theirs.log_softmax(-1), atol=2e-3)


@torch.no_grad()
def test_logits_match_single_unpadded_sequence(reference, ours, tokenizer):
    ids = tokenizer(PROMPTS[0], return_tensors="pt")["input_ids"]
    diff = (ours(input_ids=ids) - reference(input_ids=ids).logits).abs().max().item()

    assert diff < 2e-3, f"max abs logit diff {diff:.3e}"


@torch.no_grad()
def test_greedy_continuation_matches(reference, ours, tokenizer):
    ids = tokenizer(PROMPTS[1], return_tensors="pt")["input_ids"]

    for _ in range(16):
        ids = torch.cat([ids, ours(input_ids=ids)[:, -1:].argmax(-1)], dim=-1)

    expected = reference.generate(ids[:, : -16], max_new_tokens=16, do_sample=False)

    assert torch.equal(ids, expected)


@torch.no_grad()
def test_inputs_embeds_path_matches_input_ids(ours, batch):
    embeds = ours.model.embed_tokens(batch["input_ids"])

    from_ids = ours(**batch)
    from_embeds = ours(inputs_embeds=embeds, attention_mask=batch["attention_mask"])

    assert torch.equal(from_ids, from_embeds)


@torch.no_grad()
def test_padding_does_not_change_unpadded_logits(ours, tokenizer):
    ids = tokenizer(PROMPTS[0], return_tensors="pt")["input_ids"]
    padded = torch.cat([ids, torch.full((1, 7), tokenizer.pad_token_id)], dim=-1)
    mask = torch.cat([torch.ones_like(ids), torch.zeros(1, 7, dtype=torch.long)], dim=-1)

    unpadded_logits = ours(input_ids=ids)
    padded_logits = ours(input_ids=padded, attention_mask=mask)[:, : ids.shape[1]]

    assert torch.allclose(unpadded_logits, padded_logits, atol=1e-4)


def test_rejects_both_input_ids_and_embeds(ours, batch):
    with pytest.raises(ValueError):
        ours(input_ids=batch["input_ids"], inputs_embeds=ours.model.embed_tokens(batch["input_ids"]))


def test_rejects_neither_input(ours):
    with pytest.raises(ValueError):
        ours()
