from collections.abc import Iterator
from random import Random

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch import Tensor
from transformers import PreTrainedTokenizerBase

DATASET_REVISIONS = {
    "mteb/banking77": "18072d2685ea682290f7b8924d94c62acc19c0b2",
    "SetFit/amazon_massive_intent_en-US": "f7672a018e8ceb37fc0184dcfbb7e665155ffea6",
    "DeepPavlov/clinc150": "d835118ecd5ffe5488d22e9e58d1c23d18c33229",
}

# Chosen once with random.Random(42).sample(sorted(BANKING77 labels), 20).
# Keep this list fixed across experiments.
HELD_OUT_BANKING_INTENTS = frozenset(
    {
        "activate_my_card",
        "apple_pay_or_google_pay",
        "atm_support",
        "card_acceptance",
        "card_arrival",
        "card_delivery_estimate",
        "card_linking",
        "card_payment_not_recognised",
        "declined_cash_withdrawal",
        "declined_transfer",
        "edit_personal_details",
        "exchange_charge",
        "extra_charge_on_statement",
        "get_disposable_virtual_card",
        "supported_cards_and_currencies",
        "unable_to_verify_identity",
        "verify_my_identity",
        "why_verify_identity",
        "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    }
)

QUESTION = "What is the intent?"


def readable_intent(name: str) -> str:
    return name.replace("_", " ")


def banking_test_rows(banking, held_out: bool) -> list[dict[str, str]]:
    return [
        {
            "state": row["text"],
            "source": "banking77",
            "gold_intent": row["label_text"],
            "choice_pool": "banking77_final",
        }
        for row in banking["test"]
        if (row["label_text"] in HELD_OUT_BANKING_INTENTS) == held_out
    ]


def massive_rows(massive, split: str) -> list[dict[str, str]]:
    return [
        {
            "state": row["text"],
            "source": "massive",
            "gold_intent": row["label_text"],
        }
        for row in massive[split]
    ]


def load_intent_data(seed: int = 42) -> dict:
    banking = load_dataset(
        "mteb/banking77", revision=DATASET_REVISIONS["mteb/banking77"]
    )
    massive = load_dataset(
        "SetFit/amazon_massive_intent_en-US",
        revision=DATASET_REVISIONS["SetFit/amazon_massive_intent_en-US"],
    )
    clinc = load_dataset(
        "DeepPavlov/clinc150", revision=DATASET_REVISIONS["DeepPavlov/clinc150"]
    )
    clinc_intents = load_dataset(
        "DeepPavlov/clinc150",
        "intents",
        revision=DATASET_REVISIONS["DeepPavlov/clinc150"],
    )["intents"]

    banking_names = set(banking["train"].unique("label_text"))
    if len(banking_names) != 77 or not HELD_OUT_BANKING_INTENTS < banking_names:
        raise ValueError("BANKING77 labels changed; review the held-out intent list")

    choices = {
        "banking77": sorted(banking_names - HELD_OUT_BANKING_INTENTS),
        "banking77_final": sorted(banking_names),
        "massive": sorted(massive["train"].unique("label_text")),
        "clinc150": sorted(row["name"] for row in clinc_intents),
    }
    if len(choices["massive"]) != 60 or len(choices["clinc150"]) != 150:
        raise ValueError("An intent catalog changed; review the dataset mapping")

    banking_train = [
        {
            "state": row["text"],
            "source": "banking77",
            "gold_intent": row["label_text"],
        }
        for row in banking["train"]
        if row["label_text"] not in HELD_OUT_BANKING_INTENTS
    ]

    banking_train, banking_val = train_test_split(
        banking_train,
        test_size=0.1,
        random_state=seed,
        stratify=[row["gold_intent"] for row in banking_train],
    )

    clinc_names = {row["id"]: row["name"] for row in clinc_intents}
    clinc_test = [
        {
            "state": row["utterance"],
            "source": "clinc150",
            "gold_intent": clinc_names[row["label"]],
        }
        for row in clinc["test"]
        if row["label"] is not None
    ]

    return {
        "train": banking_train + massive_rows(massive, "train"),
        "validation": {
            "banking77": banking_val,
            "massive": massive_rows(massive, "validation"),
        },
        "final": {
            "banking77_seen": banking_test_rows(banking, False),
            "banking77_held_out": banking_test_rows(banking, True),
            "massive": massive_rows(massive, "test"),
            "clinc150": clinc_test,
        },
        "choices": choices,
    }


def construct_input_text(state: str, choices: list[str], anchor_token: str) -> str:
    text = f"STATE:\n{state}\nQUESTION:\n{QUESTION}\nCHOICES:\n"
    for choice in choices:
        text += f"{readable_intent(choice)} {anchor_token}\n"
    return text + "DECISION:"


def make_batches(
    examples: list[dict[str, str]],
    choices: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    device: str,
    rng: Random,
    shuffle: bool = False,
) -> Iterator[dict[str, Tensor]]:

    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer needs a pad token for choice anchors")

    rows = examples.copy()
    if shuffle:
        rng.shuffle(rows)

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        input_ids, choice_slots = [], []
        lengths, targets, choice_counts = [], [], []

        for row in batch:
            catalog = choices[row.get("choice_pool", row["source"])]

            if row["gold_intent"] not in catalog:
                raise ValueError(f"Unknown intent: {row['source']}/{row['gold_intent']}")

            candidates = catalog.copy()
            rng.shuffle(candidates)

            targets.append(candidates.index(row["gold_intent"]))
            choice_counts.append(len(candidates))

            text = construct_input_text(row["state"], candidates, tokenizer.pad_token)
            ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

            anchor_mask = ids == tokenizer.pad_token_id
            if anchor_mask.sum().item() != len(candidates):
                raise ValueError("Expected one choice anchor per candidate")

            slots = torch.full_like(ids, -1)
            slots[anchor_mask] = torch.arange(len(candidates))

            input_ids.append(ids)
            choice_slots.append(slots)
            lengths.append(len(ids))

        max_len = max(lengths)
        max_choices = max(choice_counts)

        padded_ids = torch.full(
            (len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long
        )
        padded_slots = torch.full((len(batch), max_len), -1, dtype=torch.long)
        choice_mask = torch.zeros((len(batch), max_choices), dtype=torch.bool)

        for i, (ids, slots) in enumerate(zip(input_ids, choice_slots)):
            padded_ids[i, : lengths[i]] = ids
            padded_slots[i, : lengths[i]] = slots
            choice_mask[i, : choice_counts[i]] = True

        yield {
            "input_ids": padded_ids.to(device),
            "choice_slots": padded_slots.to(device),
            "choice_mask": choice_mask.to(device),
            "lengths": torch.tensor(lengths, device=device),
            "targets": torch.tensor(targets, device=device),
        }
