import pytest

pytest.importorskip("decision_index")

from decision_index.engines import Unsupported, validate

from intelif.integrations.decision_index import IntelifEngine
from intelif.model import Intelif
from intelif.render import render_chunks
from intelif.types import normalize_questions

RAGTRUTH_CRITERIA = {
    "false": "All content of the response is supported by the context in the prompt.",
    "true": "The response contains content that is not supported by the context in the prompt.",
}

REQUESTS = {
    "string criteria": (
        "Where is my card?",
        {
            "q": {
                "type": "choice",
                "instructions": "Intent?",
                "criteria": {"a": "card arrival", "b": "refund"},
            }
        },
    ),
    "none, dict and list descriptions": (
        {"song": [1, 2, 3]},
        {
            "q": {
                "type": "choice",
                "instructions": {"task": "pick"},
                "criteria": {
                    "phishing": None,
                    "chord_0": {"names": "CM", "pitch_classes": [0, 4, 7]},
                    "A": ["#bfb978", "#ffeaba"],
                },
            }
        },
    ),
    "noul with criteria": (
        {"prompt": "p", "response": "r"},
        {
            "h": {
                "type": "noul",
                "instructions": "Hallucinated?",
                "criteria": RAGTRUTH_CRITERIA,
            }
        },
    ),
    "noul with empty criteria": (
        "email body",
        {"s": {"type": "noul", "instructions": "Urgent tone?", "criteria": {}}},
    ),
    "255 options": (
        "pick one",
        {
            "q": {
                "type": "choice",
                "instructions": "Pick",
                "criteria": {f"o{i}": f"option {i}" for i in range(255)},
            }
        },
    ),
    "several questions": (
        "shared",
        {
            f"k{i}": {
                "type": "choice",
                "instructions": f"q{i}",
                "criteria": {"x": "X", "y": "Y"},
            }
            for i in range(4)
        },
    ),
    "empty state": (
        "",
        {
            "q": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"yes": "yes", "no": "no"},
            }
        },
    ),
}


@pytest.fixture(scope="module")
def engine(model):
    return IntelifEngine(client=model)


@pytest.mark.parametrize("name", REQUESTS)
def test_responses_pass_kit_validation(engine, name):
    state, questions = REQUESTS[name]
    response, _ = engine(state, questions)

    validate(questions, response)


def test_noul_criteria_are_rendered():
    question = normalize_questions(
        {
            "h": {
                "type": "noul",
                "instructions": "Hallucinated?",
                "criteria": RAGTRUTH_CRITERIA,
            }
        }
    )["h"]
    prompt = "".join(render_chunks("state", question))

    assert f"true: {RAGTRUTH_CRITERIA['true']}" in prompt
    assert f"false: {RAGTRUTH_CRITERIA['false']}" in prompt


def test_context_limit_is_unsupported(model):
    small = Intelif(model.network, model.tokenizer, model.config, max_tokens=32)
    engine = IntelifEngine(client=small)

    with pytest.raises(Unsupported):
        engine("word " * 100, REQUESTS["string criteria"][1])


def test_unknown_question_type_is_unsupported(engine):
    with pytest.raises(Unsupported):
        engine("state", {"q": {"type": "score", "criteria": ["low", "high"]}})
