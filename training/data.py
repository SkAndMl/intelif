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
from intelif.types import Choice, Noul

load_dotenv(find_dotenv())

SEED = 42
VALIDATION_SIZE = 500
FINAL_SIZE = 1000
TRAIN_CAPS = {"mnli": 40000, "snli": 20000, "qqp": 20000, "paws": 20000}

DATASET_REVISIONS = {
    "mteb/banking77": "18072d2685ea682290f7b8924d94c62acc19c0b2",
    "SetFit/amazon_massive_intent_en-US": "f7672a018e8ceb37fc0184dcfbb7e665155ffea6",
    "DeepPavlov/clinc150": "d835118ecd5ffe5488d22e9e58d1c23d18c33229",
    "Salesforce/xlam-function-calling-60k": "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
    "osunlp/early-experience": "7f1dfbe4f7a100ee1337843b1abbd2c25b8cf7ce",
    "nyu-mll/glue": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
    "stanfordnlp/snli": "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
    "google/boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
    "google-research-datasets/paws": "161ece9501cf0a11f3e48bd356eaa82de46d6a09",
    "tau/commonsense_qa": "94630fe30dad47192a8546eb75f094926d47e155",
    "allenai/openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
    "allenai/sciq": "2c94ad3e1aafab77146f384e23536f97a4849815",
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

INTENT_QUESTION = "What is the intent?"
TOOL_QUESTION = "Which tool should be called?"
ACTION_QUESTION = "Which action should the agent take next?"
NLI_QUESTION = "How does the hypothesis relate to the premise?"
NLI_NOUL = "Does the premise entail the hypothesis?"
MCQ_QUESTION = "Which answer is correct?"

NLI_LABELS = {
    "entailment": "The hypothesis must be true if the premise is true.",
    "neutral": "The hypothesis might or might not be true given the premise.",
    "contradiction": "The hypothesis cannot be true if the premise is true.",
}

ACTIONS_MARKER = "Your admissible actions of the current situation are:"
ACTION_PATTERN = re.compile(r"^\s*\[?(['\"])(.*)\1\]?[,.]*\s*$", re.MULTILINE)
MAX_STATE_CHARS = 4000
MAX_ACTION_CHOICES = 64
LETTERS = "ABCDEFGHIJ"


def readable(label: str) -> str:
    return label.replace("_", " ")


def catalog_row(state, instructions: str, criteria: dict, gold: str, source: str):
    return {
        "kind": "catalog",
        "state": state,
        "instructions": instructions,
        "criteria": criteria,
        "gold": gold,
        "source": source,
    }


def letters_row(state, instructions: str, options: list[str], gold: int, source: str):
    return {
        "kind": "letters",
        "state": state,
        "instructions": instructions,
        "options": options,
        "gold": gold,
        "source": source,
    }


def noul_row(state, instructions: str, answer: bool, source: str):
    return {
        "kind": "noul",
        "state": state,
        "instructions": instructions,
        "gold": "true" if answer else "false",
        "source": source,
    }


def build_question(row: dict, rng: Random) -> tuple[Choice | Noul, str]:
    if row["kind"] == "noul":
        return Noul(instructions=row["instructions"]), row["gold"]

    if row["kind"] == "letters":
        order = list(range(len(row["options"])))
        rng.shuffle(order)
        criteria = {LETTERS[i]: row["options"][j] for i, j in enumerate(order)}

        return (
            Choice(criteria=criteria, instructions=row["instructions"]),
            LETTERS[order.index(row["gold"])],
        )

    keys = list(row["criteria"])
    rng.shuffle(keys)
    criteria = {key: row["criteria"][key] for key in keys}

    return Choice(criteria=criteria, instructions=row["instructions"]), row["gold"]


def sample(rows: list, size: int, seed: int = SEED) -> list:
    return Random(seed).sample(rows, min(size, len(rows)))


def split_by_group(rows: list[dict], groups: list) -> tuple[list, list, list]:
    unique = sorted(set(groups))
    train_groups, held_out = train_test_split(unique, test_size=0.1, random_state=SEED)
    train_groups = set(train_groups)
    val_groups = set(held_out[: len(held_out) // 2])

    train, val, test = [], [], []
    for row, group in zip(rows, groups):
        if group in train_groups:
            train.append(row)
        elif group in val_groups:
            val.append(row)
        else:
            test.append(row)

    return train, val, test


def intent_data() -> tuple[list, dict, dict]:
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

    banking_names = sorted(banking["train"].unique("label_text"))
    if len(banking_names) != 77 or not HELD_OUT_BANKING_INTENTS < set(banking_names):
        raise ValueError("BANKING77 labels changed; review the held-out intent list")

    seen_banking = {
        readable(n): None for n in banking_names if n not in HELD_OUT_BANKING_INTENTS
    }
    all_banking = {readable(n): None for n in banking_names}
    massive_catalog = {
        readable(n): None for n in sorted(massive["train"].unique("label_text"))
    }
    clinc_names = {row["id"]: row["name"] for row in clinc_intents}
    clinc_catalog = {readable(n): None for n in sorted(clinc_names.values())}

    def banking_rows(split: str, catalog: dict, keep) -> list:
        return [
            catalog_row(
                row["text"],
                INTENT_QUESTION,
                catalog,
                readable(row["label_text"]),
                "banking77",
            )
            for row in banking[split]
            if keep(row["label_text"])
        ]

    def massive_rows(split: str) -> list:
        return [
            catalog_row(
                row["text"],
                INTENT_QUESTION,
                massive_catalog,
                readable(row["label_text"]),
                "massive",
            )
            for row in massive[split]
        ]

    banking_train = banking_rows(
        "train", seen_banking, lambda n: n not in HELD_OUT_BANKING_INTENTS
    )
    banking_train, banking_val = train_test_split(
        banking_train,
        test_size=0.1,
        random_state=SEED,
        stratify=[row["gold"] for row in banking_train],
    )

    clinc_test = [
        catalog_row(
            row["utterance"],
            INTENT_QUESTION,
            clinc_catalog,
            readable(clinc_names[row["label"]]),
            "clinc150",
        )
        for row in clinc["test"]
        if row["label"] is not None
    ]

    train = banking_train + massive_rows("train")
    validation = {"banking77": banking_val, "massive": massive_rows("validation")}
    final = {
        "banking77_seen": banking_rows(
            "test", all_banking, lambda n: n not in HELD_OUT_BANKING_INTENTS
        ),
        "banking77_held_out": banking_rows(
            "test", all_banking, lambda n: n in HELD_OUT_BANKING_INTENTS
        ),
        "massive": massive_rows("test"),
        "clinc150": clinc_test,
    }

    return train, validation, final


def xlam_data() -> tuple[list, list, list]:
    xlam = load_dataset(
        "json",
        data_files="hf://datasets/Salesforce/xlam-function-calling-60k"
        f"@{DATASET_REVISIONS['Salesforce/xlam-function-calling-60k']}"
        "/xlam_function_calling_60k.json",
        split="train",
    )

    rows, groups = [], []
    for row in xlam:
        tools = json.loads(row["tools"])
        names = [tool["name"] for tool in tools]
        called = {call["name"] for call in json.loads(row["answers"])}

        if len(called) != 1 or not called < set(names) or len(set(names)) != len(names):
            continue

        gold = called.pop()
        criteria = {tool["name"]: tool.get("description") or None for tool in tools}
        rows.append(catalog_row(row["query"], TOOL_QUESTION, criteria, gold, "xlam"))
        groups.append(gold)

    return split_by_group(rows, groups)


def agent_data(source: str) -> tuple[list, list, list]:
    revision = DATASET_REVISIONS["osunlp/early-experience"]
    if source == "alfworld":
        dataset = load_dataset(
            "osunlp/early-experience", "alfworld", split="expert", revision=revision
        )
    else:
        dataset = load_dataset(
            "json",
            data_files=f"hf://datasets/osunlp/early-experience@{revision}"
            "/webshop/expert_sft.jsonl",
            split="train",
        )

    rows, groups, seen = [], [], set()
    for row in dataset:
        prompt = row["messages"][-2]["content"]
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
            catalog_row(
                state.strip(),
                ACTION_QUESTION,
                {action: None for action in actions},
                gold,
                source,
            )
        )
        groups.append(re.search(r"Your task is to: (.*)", state).group(1))

    return split_by_group(rows, groups)


def nli_row(premise: str, hypothesis: str, label: int, source: str, rng: Random):
    name = list(NLI_LABELS)[label]
    state = f"Premise: {premise}\nHypothesis: {hypothesis}"
    style = rng.random()

    if style < 0.3:
        return noul_row(state, NLI_NOUL, name == "entailment", source)

    if style < 0.55:
        return letters_row(state, NLI_QUESTION, list(NLI_LABELS), label, source)

    return catalog_row(state, NLI_QUESTION, dict(NLI_LABELS), name, source)


def pair_data() -> tuple[list, dict, dict]:
    glue = DATASET_REVISIONS["nyu-mll/glue"]
    mnli = load_dataset("nyu-mll/glue", "mnli", revision=glue)
    snli = load_dataset(
        "stanfordnlp/snli", revision=DATASET_REVISIONS["stanfordnlp/snli"]
    )
    qqp = load_dataset("nyu-mll/glue", "qqp", revision=glue)
    paws = load_dataset(
        "google-research-datasets/paws",
        "labeled_final",
        revision=DATASET_REVISIONS["google-research-datasets/paws"],
    )
    boolq = load_dataset("google/boolq", revision=DATASET_REVISIONS["google/boolq"])

    rng = Random(SEED)

    def nli(split, source: str, size: int) -> list:
        rows = [row for row in split if row["label"] in (0, 1, 2)]
        return [
            nli_row(row["premise"], row["hypothesis"], row["label"], source, rng)
            for row in sample(rows, size)
        ]

    def qqp_rows(split, size: int) -> list:
        return [
            noul_row(
                {"question1": row["question1"], "question2": row["question2"]},
                "Do the two questions ask the same thing?",
                row["label"] == 1,
                "qqp",
            )
            for row in sample(list(split), size)
        ]

    def paws_rows(split, size: int) -> list:
        return [
            noul_row(
                f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}",
                "Are the two sentences paraphrases of each other?",
                row["label"] == 1,
                "paws",
            )
            for row in sample(list(split), size)
        ]

    def boolq_rows(rows: list) -> list:
        return [
            noul_row(
                row["passage"],
                row["question"].capitalize() + "?",
                row["answer"],
                "boolq",
            )
            for row in rows
        ]

    qqp_held = sample(list(qqp["validation"]), VALIDATION_SIZE + FINAL_SIZE, SEED + 1)
    boolq_held = sample(
        list(boolq["validation"]), VALIDATION_SIZE + FINAL_SIZE, SEED + 1
    )

    train = (
        nli(mnli["train"], "mnli", TRAIN_CAPS["mnli"])
        + nli(snli["train"], "snli", TRAIN_CAPS["snli"])
        + qqp_rows(qqp["train"], TRAIN_CAPS["qqp"])
        + paws_rows(paws["train"], TRAIN_CAPS["paws"])
        + boolq_rows(list(boolq["train"]))
    )
    validation = {
        "mnli": nli(mnli["validation_matched"], "mnli", VALIDATION_SIZE),
        "snli": nli(snli["validation"], "snli", VALIDATION_SIZE),
        "qqp": qqp_rows(qqp_held[:VALIDATION_SIZE], VALIDATION_SIZE),
        "paws": paws_rows(paws["validation"], VALIDATION_SIZE),
        "boolq": boolq_rows(boolq_held[:VALIDATION_SIZE]),
    }
    final = {
        "mnli_mismatched": nli(mnli["validation_mismatched"], "mnli", FINAL_SIZE),
        "snli": nli(snli["test"], "snli", FINAL_SIZE),
        "qqp": qqp_rows(qqp_held[VALIDATION_SIZE:], FINAL_SIZE),
        "paws": paws_rows(paws["test"], FINAL_SIZE),
        "boolq": boolq_rows(boolq_held[VALIDATION_SIZE:]),
    }

    return train, validation, final


def mcq_data() -> tuple[list, dict, dict]:
    csqa = load_dataset(
        "tau/commonsense_qa", revision=DATASET_REVISIONS["tau/commonsense_qa"]
    )
    obqa = load_dataset(
        "allenai/openbookqa", "main", revision=DATASET_REVISIONS["allenai/openbookqa"]
    )
    sciq = load_dataset("allenai/sciq", revision=DATASET_REVISIONS["allenai/sciq"])

    def labelled(rows, source: str, stem: str) -> list:
        out = []
        for row in rows:
            labels = list(row["choices"]["label"])
            if row["answerKey"] not in labels:
                continue
            out.append(
                letters_row(
                    row[stem],
                    MCQ_QUESTION,
                    list(row["choices"]["text"]),
                    labels.index(row["answerKey"]),
                    source,
                )
            )
        return out

    def sciq_rows(rows) -> list:
        out = []
        for row in rows:
            options = [
                row["correct_answer"],
                row["distractor1"],
                row["distractor2"],
                row["distractor3"],
            ]
            state = (
                {"context": row["support"], "question": row["question"]}
                if row["support"]
                else row["question"]
            )
            out.append(letters_row(state, MCQ_QUESTION, options, 0, "sciq"))
        return out

    csqa_validation = labelled(csqa["validation"], "csqa", "question")
    csqa_validation = sample(csqa_validation, len(csqa_validation))

    train = (
        labelled(csqa["train"], "csqa", "question")
        + labelled(obqa["train"], "obqa", "question_stem")
        + sciq_rows(sciq["train"])
    )
    validation = {
        "csqa": csqa_validation[:VALIDATION_SIZE],
        "obqa": labelled(obqa["validation"], "obqa", "question_stem"),
        "sciq": sample(sciq_rows(sciq["validation"]), VALIDATION_SIZE),
    }
    final = {
        "csqa": csqa_validation[VALIDATION_SIZE:],
        "obqa": labelled(obqa["test"], "obqa", "question_stem"),
        "sciq": sample(sciq_rows(sciq["test"]), FINAL_SIZE),
    }

    return train, validation, final


def load_data(smoke: bool = False) -> dict:
    train, validation, final = intent_data()

    for source, (src_train, src_val, src_final) in {
        "xlam": xlam_data(),
        "alfworld": agent_data("alfworld"),
        "webshop": agent_data("webshop"),
    }.items():
        train += src_train
        validation[source] = src_val
        final[source] = src_final

    for src_train, src_val, src_final in (pair_data(), mcq_data()):
        train += src_train
        validation.update(src_val)
        final.update(src_final)

    if smoke:
        train = sample(train, 3000)
        validation = {k: sample(v, 32) for k, v in validation.items()}
        final = {k: sample(v, 32) for k, v in final.items()}

    return {"train": train, "validation": validation, "final": final}


def encode_row(
    row: dict, tokenizer: PreTrainedTokenizerBase, anchor_id: int, rng: Random
) -> dict:
    question, gold = build_question(row, rng)
    encoded = encode_question(tokenizer, row["state"], question, anchor_id)

    return {
        "input_ids": encoded["input_ids"],
        "choice_slots": encoded["choice_slots"],
        "target": encoded["keys"].index(gold),
        "choice_count": len(encoded["keys"]),
    }


def collate_batch(batch: list[dict], pad_id: int, device: str) -> dict[str, Tensor]:
    lengths = [len(example["input_ids"]) for example in batch]
    width = max(lengths)
    choices = max(example["choice_count"] for example in batch)

    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    choice_slots = torch.full((len(batch), width), -1, dtype=torch.long)
    choice_mask = torch.zeros((len(batch), choices), dtype=torch.bool)

    for i, example in enumerate(batch):
        input_ids[i, : lengths[i]] = example["input_ids"]
        choice_slots[i, : lengths[i]] = example["choice_slots"]
        choice_mask[i, : example["choice_count"]] = True

    return {
        "input_ids": input_ids.to(device),
        "lengths": torch.tensor(lengths, device=device),
        "choice_slots": choice_slots.to(device),
        "choice_mask": choice_mask.to(device),
        "targets": torch.tensor([e["target"] for e in batch], device=device),
    }


def make_batches(
    rows: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    anchor_id: int,
    batch_size: int,
    max_tokens: int,
    device: str,
    rng: Random,
    shuffle: bool = False,
    bucket_size: int = 2048,
) -> Iterator[dict[str, Tensor]]:
    rows = rows.copy()
    if shuffle:
        rng.shuffle(rows)

    for start in range(0, len(rows), bucket_size):
        encoded = [
            encode_row(row, tokenizer, anchor_id, rng)
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
            yield collate_batch(batch, anchor_id, device)
