import json
import re
from collections.abc import Iterator
from random import Random

import torch
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from intelif.render import encode_question
from intelif.types import Choice

load_dotenv(find_dotenv())

DATASET_REVISIONS = {
    "mteb/banking77": "18072d2685ea682290f7b8924d94c62acc19c0b2",
    "SetFit/amazon_massive_intent_en-US": "f7672a018e8ceb37fc0184dcfbb7e665155ffea6",
    "DeepPavlov/clinc150": "d835118ecd5ffe5488d22e9e58d1c23d18c33229",
    "Salesforce/xlam-function-calling-60k": "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
    "osunlp/early-experience": "7f1dfbe4f7a100ee1337843b1abbd2c25b8cf7ce",
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
TOOL_QUESTION = "Which tool should be called?"
ACTION_QUESTION = "Which action should the agent take next?"

ACTIONS_MARKER = "Your admissible actions of the current situation are:"
ACTION_PATTERN = re.compile(r"^\s*\[?(['\"])(.*)\1\]?[,.]*\s*$", re.MULTILINE)
MAX_STATE_CHARS = 4000
MAX_ACTION_CHOICES = 64


def readable_intent(name: str) -> str:
    return name.replace("_", " ")


def xlam_rows(xlam) -> list[dict]:
    rows = []

    for row in xlam:
        tools = json.loads(row["tools"])
        names = [tool["name"] for tool in tools]
        called = {call["name"] for call in json.loads(row["answers"])}

        if len(called) != 1 or not called < set(names) or len(set(names)) != len(names):
            continue

        choices = [f"{tool['name']}: {tool.get('description', '')}" for tool in tools]
        gold = called.pop()

        rows.append(
            {
                "state": row["query"],
                "source": "xlam",
                "question": TOOL_QUESTION,
                "choices": choices,
                "gold_intent": choices[names.index(gold)],
                "group": gold,
            }
        )

    return rows


def agent_rows(dataset, source: str) -> list[dict]:
    rows, seen = [], set()

    for row in dataset:
        prompt: str = row["messages"][-2]["content"]
        state, listing = prompt.split(ACTIONS_MARKER)
        listing = listing.split("Now it's your turn")[0]

        actions = [action for _, action in ACTION_PATTERN.findall(listing)]
        answer = row["messages"][-1]["content"]
        gold = re.search(r"<action>(.*?)</action>", answer, re.DOTALL).group(1).strip()

        if (
            prompt in seen
            or gold not in actions
            or len(set(actions)) != len(actions)
            or len(actions) > MAX_ACTION_CHOICES
            or len(state) > MAX_STATE_CHARS
        ):
            continue

        seen.add(prompt)
        rows.append(
            {
                "state": state.strip(),
                "source": source,
                "question": ACTION_QUESTION,
                "choices": actions,
                "gold_intent": gold,
                "group": re.search(r"Your task is to: (.*)", state).group(1),
            }
        )

    return rows


def split_by_group(rows: list[dict], seed: int) -> tuple[list, list, list]:
    groups = sorted({row["group"] for row in rows})
    train_groups, held_out = train_test_split(groups, test_size=0.1, random_state=seed)

    train_groups = set(train_groups)
    val_groups = set(held_out[: len(held_out) // 2])
    test_groups = set(held_out[len(held_out) // 2 :])

    return (
        [row for row in rows if row["group"] in train_groups],
        [row for row in rows if row["group"] in val_groups],
        [row for row in rows if row["group"] in test_groups],
    )


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

    xlam = load_dataset(
        "json",
        data_files="hf://datasets/Salesforce/xlam-function-calling-60k"
        f"@{DATASET_REVISIONS['Salesforce/xlam-function-calling-60k']}"
        "/xlam_function_calling_60k.json",
        split="train",
    )
    alfworld = load_dataset(
        "osunlp/early-experience",
        "alfworld",
        split="expert",
        revision=DATASET_REVISIONS["osunlp/early-experience"],
    )
    webshop = load_dataset(
        "json",
        data_files="hf://datasets/osunlp/early-experience"
        f"@{DATASET_REVISIONS['osunlp/early-experience']}"
        "/webshop/expert_sft.jsonl",
        split="train",
    )

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

    xlam_train, xlam_val, xlam_test = split_by_group(xlam_rows(xlam), seed)
    alfworld_train, alfworld_val, alfworld_test = split_by_group(
        agent_rows(alfworld, "alfworld"), seed
    )
    webshop_train, webshop_val, webshop_test = split_by_group(
        agent_rows(webshop, "webshop"), seed
    )

    return {
        "train": banking_train
        + massive_rows(massive, "train")
        + xlam_train
        + alfworld_train
        + webshop_train,
        "validation": {
            "banking77": banking_val,
            "massive": massive_rows(massive, "validation"),
            "xlam": xlam_val,
            "alfworld": alfworld_val,
            "webshop": webshop_val,
        },
        "final": {
            "banking77_seen": banking_test_rows(banking, False),
            "banking77_held_out": banking_test_rows(banking, True),
            "massive": massive_rows(massive, "test"),
            "clinc150": clinc_test,
            "xlam": xlam_test,
            "alfworld": alfworld_test,
            "webshop": webshop_test,
        },
        "choices": choices,
    }


def encode_row(
    row: dict,
    choices: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    rng: Random,
) -> dict:
    catalog = row.get("choices") or choices[row.get("choice_pool", row["source"])]

    if row["gold_intent"] not in catalog:
        raise ValueError(f"Unknown intent: {row['source']}/{row['gold_intent']}")

    candidates = catalog.copy()
    rng.shuffle(candidates)

    question = Choice(
        criteria={readable_intent(candidate): None for candidate in candidates},
        instructions=row.get("question", QUESTION),
    )

    if len(question.criteria) != len(candidates):
        raise ValueError("Two candidates render to the same label")

    encoded = encode_question(tokenizer, row["state"], question, tokenizer.pad_token_id)

    return {
        "input_ids": encoded["input_ids"],
        "choice_slots": encoded["choice_slots"],
        "target": candidates.index(row["gold_intent"]),
        "choice_count": len(candidates),
    }


def collate_batch(
    batch: list[dict], pad_token_id: int, device: str
) -> dict[str, Tensor]:
    lengths = [len(example["input_ids"]) for example in batch]
    max_len = max(lengths)
    max_choices = max(example["choice_count"] for example in batch)

    padded_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_slots = torch.full((len(batch), max_len), -1, dtype=torch.long)
    choice_mask = torch.zeros((len(batch), max_choices), dtype=torch.bool)

    for i, example in enumerate(batch):
        padded_ids[i, : lengths[i]] = example["input_ids"]
        padded_slots[i, : lengths[i]] = example["choice_slots"]
        choice_mask[i, : example["choice_count"]] = True

    return {
        "input_ids": padded_ids.to(device),
        "choice_slots": padded_slots.to(device),
        "choice_mask": choice_mask.to(device),
        "lengths": torch.tensor(lengths, device=device),
        "targets": torch.tensor(
            [example["target"] for example in batch], device=device
        ),
    }


def make_batches(
    examples: list[dict[str, str]],
    choices: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_tokens: int,
    device: str,
    rng: Random,
    shuffle: bool = False,
    bucket_size: int = 2048,
) -> Iterator[dict[str, Tensor]]:

    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer needs a pad token for choice anchors")

    rows = examples.copy()
    if shuffle:
        rng.shuffle(rows)

    for start in range(0, len(rows), bucket_size):
        encoded = [
            encode_row(row, choices, tokenizer, rng)
            for row in rows[start : start + bucket_size]
        ]
        encoded.sort(key=lambda example: len(example["input_ids"]))

        batches, batch, longest = [], [], 0

        for example in encoded:
            length = len(example["input_ids"])

            if batch and (
                len(batch) == batch_size
                or max(longest, length) * (len(batch) + 1) > max_tokens
            ):
                batches.append(batch)
                batch, longest = [], 0

            batch.append(example)
            longest = max(longest, length)

        batches.append(batch)

        if shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield collate_batch(batch, tokenizer.pad_token_id, device)
