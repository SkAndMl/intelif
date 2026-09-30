from collections.abc import Mapping
from functools import cached_property
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_serializer,
)

from intelif.errors import IntelifValidationError

type JSONContent = str | dict[str, Any] | list[Any]


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_serializer(mode="wrap")
    def _omit_unset(self, handler) -> dict[str, Any]:
        return {key: value for key, value in handler(self).items() if value is not None}


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    true: JSONContent | None = None
    false: JSONContent | None = None


class Choice(_Question):
    type: Literal["choice"] = "choice"
    criteria: dict[str, JSONContent | None]
    instructions: JSONContent | None = None

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, criteria: dict) -> dict:
        if not criteria:
            raise ValueError("at least one criterion is required")

        if any(not key for key in criteria):
            raise ValueError("criteria keys must be non-empty strings")

        return criteria


class Noul(_Question):
    type: Literal["noul"] = "noul"
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class Score(_Question):
    type: Literal["score"] = "score"
    criteria: list[JSONContent] = Field(min_length=1)
    instructions: JSONContent | None = None


type Question = Annotated[Choice | Noul | Score, Field(discriminator="type")]

_question_adapter = TypeAdapter(Question)


def normalize_questions(
    questions: Mapping[str, Any],
) -> dict[str, Choice | Noul | Score]:
    if not questions:
        raise IntelifValidationError("at least one question is required")

    normalized = {}

    for name, question in questions.items():
        if isinstance(question, (Choice, Noul, Score)):
            normalized[name] = question

        try:
            normalized[name] = _question_adapter.validate_python(question)
        except ValidationError as error:
            raise IntelifValidationError(f"question `{name}`: {error}") from error

    return normalized


class _Answer(BaseModel):
    model_config = ConfigDict(frozen=True)


class ChoiceAnswer(_Answer):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class NoulAnswer(_Answer):
    type: Literal["noul"] = "noul"
    noul: float


class ScoreAnswer(_Answer):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[int, JSONContent]
    probabilities: dict[int, float]


type Answer = Annotated[
    ChoiceAnswer | NoulAnswer | ScoreAnswer, Field(discriminator="type")
]


class Usage(_Answer):
    input_tokens: int
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    usage: Usage
    answers: dict[str, Answer]

    @cached_property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ChoiceAnswer)
        }

    @cached_property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, NoulAnswer)
        }

    @cached_property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ScoreAnswer)
        }
