import sys
from random import Random

import pytest
from conftest import ROOT

pytest.importorskip("datasets")
sys.path.insert(0, str(ROOT / "training"))

from data import (
    build_question,
    catalog_row,
    encode_row,
    letters_row,
    noul_row,
)

from intelif.types import Choice, Noul


def test_letters_gold_follows_the_shuffle():
    options = ["red", "green", "blue", "yellow"]
    row = letters_row("state", "Which color?", options, 2, "test")

    for seed in range(20):
        question, gold = build_question(row, Random(seed))

        assert isinstance(question, Choice)
        assert question.criteria[gold] == "blue"


def test_catalog_and_noul_questions():
    catalog = catalog_row("state", "Intent?", {"a": None, "b": "B"}, "b", "test")
    question, gold = build_question(catalog, Random(0))
    assert set(question.criteria) == {"a", "b"} and gold == "b"

    noul = noul_row("state", "True?", False, "test")
    question, gold = build_question(noul, Random(0))
    assert isinstance(question, Noul) and gold == "false"


def test_encoded_target_points_at_gold(tokenizer):
    row = letters_row("state", "Which color?", ["red", "green", "blue"], 1, "test")
    anchor = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    example = encode_row(row, tokenizer, anchor, Random(3))

    assert example["choice_count"] == 3
    assert 0 <= example["target"] < 3
    assert int((example["choice_slots"] >= 0).sum()) == 3
