import torch
from conftest import CONFIG, tiny_network

from intelif.modeling.lora import LoraLinear, merge_lora


def logits(network, input_ids, choice_slots, choice_mask):
    with torch.no_grad():
        return network(
            input_ids=input_ids,
            lengths=torch.full((input_ids.shape[0],), input_ids.shape[1]),
            choice_slots=choice_slots,
            choice_mask=choice_mask,
        )


def test_lora_targets_match_config():
    network = tiny_network()
    targets = {
        name.rsplit(".", 1)[-1]
        for name, module in network.named_modules()
        if isinstance(module, LoraLinear)
    }

    assert targets == set(CONFIG["lora"]["target_modules"])


def test_merge_matches_unmerged():
    input_ids = torch.randint(0, 1000, (2, 16))
    choice_slots = torch.full_like(input_ids, -1)
    choice_slots[:, [4, 9, 14]] = torch.tensor([0, 1, 2])
    choice_mask = torch.ones(2, 3, dtype=torch.bool)

    network = tiny_network()
    before = logits(network, input_ids, choice_slots, choice_mask)

    merge_lora(network.base_model)
    after = logits(network, input_ids, choice_slots, choice_mask)

    assert not any(isinstance(m, LoraLinear) for m in network.modules())
    assert torch.allclose(before, after, atol=1e-5)
