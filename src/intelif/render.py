import json

import torch
from transformers import PreTrainedTokenizerBase

from intelif.types import Choice, JSONContent, Noul, NoulCriteria, Score

PROMPT_FORMAT = "intelif-v1"
EMPTY_STATE = "(empty)"

DEFAULT_INSTRUCTIONS = {
    "choice": "Which option applies?",
    "noul": "Is the statement true?",
    "score": "Which level applies?",
}
NOUL_DESCRIPTIONS = {"true": "Yes", "false": "No"}


def to_text(value: JSONContent) -> str:
    if isinstance(value, str):
        return value

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def state_text(state: JSONContent | None) -> str:
    if state in ("", None, {}, []):
        return EMPTY_STATE

    return to_text(state)


def option_line(key: str, description: JSONContent | None) -> str:
    if description is None or to_text(description) == key:
        return key

    return f"{key}: {to_text(description)}"


def question_options(question: Choice | Noul | Score) -> dict[str, JSONContent | None]:
    if isinstance(question, Choice):
        return dict(question.criteria)

    if isinstance(question, Noul):
        criteria = question.criteria or NoulCriteria()
        options = {"true": criteria.true, "false": criteria.false}

        return {
            outcome: NOUL_DESCRIPTIONS[outcome] if description is None else description
            for outcome, description in options.items()
        }

    return {
        str(level): description for level, description in enumerate(question.criteria)
    }


def render_chunks(
    state: JSONContent | None, question: Choice | Noul | Score
) -> list[str]:
    instructions = question.instructions
    if instructions is None:
        instructions = DEFAULT_INSTRUCTIONS[question.type]

    chunks = [
        f"STATE:\n{state_text(state)}\nQUESTION:\n{to_text(instructions)}\nCHOICES:\n"
    ]

    for key, description in question_options(question).items():
        chunks[-1] += f"{option_line(key, description)} "
        chunks.append("\n")

    chunks[-1] += "DECISION:"

    return chunks


def encode_question(
    tokenizer: PreTrainedTokenizerBase,
    state: JSONContent | None,
    question: Choice | Noul | Score,
    anchor_id: int,
) -> dict:
    chunks = render_chunks(state, question)
    input_ids, choice_slots = [], []

    for index, chunk in enumerate(chunks):
        chunk_ids = tokenizer(
            chunk, add_special_tokens=False, split_special_tokens=True
        )["input_ids"]
        input_ids += chunk_ids
        choice_slots += [-1] * len(chunk_ids)

        if index < len(chunks) - 1:
            input_ids.append(anchor_id)
            choice_slots.append(index)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "choice_slots": torch.tensor(choice_slots, dtype=torch.long),
        "keys": list(question_options(question)),
    }
