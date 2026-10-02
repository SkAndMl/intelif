import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from intelif.model import Intelif
from intelif.modeling.intelif import IntelIfModel
from intelif.modeling.lora import LoraConfig, inject_lora, merge_lora
from intelif.modeling.qwen import ModelConfig, Qwen3Model

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())
TINY = {
    "model_dim": 64,
    "head_dim": 32,
    "num_heads": 4,
    "num_kv_heads": 2,
    "num_layers": 2,
    "rope_base": 1e6,
    "vocab_size": 151936,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-6,
    "context_length": 40960,
}


def tiny_network(seed: int = 0) -> IntelIfModel:
    torch.manual_seed(seed)
    base = Qwen3Model(
        ModelConfig(**TINY, dtype=torch.float32, gradient_checkpointing=False)
    )
    for parameter in base.parameters():
        parameter.data.normal_(0, 0.05)

    network = IntelIfModel(base)
    network.requires_grad_(False)

    lora = CONFIG["lora"]
    inject_lora(
        network.base_model,
        LoraConfig(**{**lora, "target_modules": tuple(lora["target_modules"])}),
    )
    for name, parameter in network.named_parameters():
        if "lora_" in name or name == "scorer.weight":
            parameter.data.normal_(0, 0.05)

    return network.eval()


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        CONFIG["base_model"], revision=CONFIG["base_revision"]
    )


@pytest.fixture(scope="session")
def model(tokenizer) -> Intelif:
    network = tiny_network()
    merge_lora(network.base_model)

    return Intelif(network, tokenizer, CONFIG, device="cpu")
