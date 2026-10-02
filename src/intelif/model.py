import threading
from collections.abc import Mapping
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from intelif.errors import IntelifModelError, IntelifUnsupportedError
from intelif.hub import (
    ADAPTER_FILE,
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    load_config,
    resolve_file,
)
from intelif.modeling.intelif import IntelIfModel
from intelif.modeling.lora import LoraConfig, inject_lora, merge_lora
from intelif.modeling.qwen import ModelConfig, Qwen3Model
from intelif.render import encode_question
from intelif.types import (
    Answer,
    Choice,
    ChoiceAnswer,
    JSONContent,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
    normalize_questions,
)

DEFAULT_DTYPES = {
    "cuda": torch.bfloat16,
    "mps": torch.float16,
    "cpu": torch.float32,
}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def resolve_dtype(dtype: str | torch.dtype | None, device: str) -> torch.dtype:
    if dtype is None:
        return DEFAULT_DTYPES[torch.device(device).type]

    if isinstance(dtype, str):
        return getattr(torch, dtype)

    return dtype


def load_adapter(network: IntelIfModel, adapter_path: str) -> None:
    lora_names = {name for name, _ in network.named_parameters() if "lora_" in name}
    expected = lora_names | {"scorer.weight"}
    state = load_file(adapter_path)

    if set(state) != expected:
        missing, unexpected = expected - set(state), set(state) - expected
        raise IntelifModelError(
            f"adapter does not match the model: {len(missing)} missing, "
            f"{len(unexpected)} unexpected tensors"
        )

    network.load_state_dict(state, strict=False)


def build_network(config: dict, dtype: torch.dtype, adapter_path: str) -> IntelIfModel:
    base_cfg = ModelConfig(
        **config["base_config"], dtype=dtype, gradient_checkpointing=False
    )
    base_model = Qwen3Model.from_pretrained(
        config["base_model"], base_cfg, revision=config["base_revision"]
    )

    network = IntelIfModel(base_model)
    network.requires_grad_(False)

    lora = config["lora"]
    inject_lora(
        network.base_model,
        LoraConfig(**{**lora, "target_modules": tuple(lora["target_modules"])}),
    )
    load_adapter(network, adapter_path)
    merge_lora(network.base_model)

    return network


def collate(examples: list[dict], pad_id: int, device: str) -> dict[str, torch.Tensor]:
    lengths = [len(example["input_ids"]) for example in examples]
    width, most = max(lengths), max(len(example["keys"]) for example in examples)

    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    choice_slots = torch.full((len(examples), width), -1, dtype=torch.long)
    choice_mask = torch.zeros((len(examples), most), dtype=torch.bool)

    for i, example in enumerate(examples):
        input_ids[i, : lengths[i]] = example["input_ids"]
        choice_slots[i, : lengths[i]] = example["choice_slots"]
        choice_mask[i, : len(example["keys"])] = True

    return {
        "input_ids": input_ids.to(device),
        "lengths": torch.tensor(lengths, device=device),
        "choice_slots": choice_slots.to(device),
        "choice_mask": choice_mask.to(device),
    }


def build_answer(
    question: Choice | Noul | Score, keys: list[str], probabilities: list[float]
) -> Answer:
    by_key = dict(zip(keys, probabilities))

    if isinstance(question, Choice):
        choice = max(keys, key=by_key.__getitem__)

        return ChoiceAnswer(
            choice=choice,
            confidence=by_key[choice],
            probabilities=by_key,
        )

    if isinstance(question, Noul):
        return NoulAnswer(noul=by_key["true"])

    levels = {int(key): probability for key, probability in by_key.items()}
    return ScoreAnswer(
        score=sum(level * probability for level, probability in levels.items()),
        confidence=max(levels.values()),
        legend=dict(enumerate(question.criteria)),
        probabilities=levels,
    )


class Intelif:
    def __init__(
        self,
        network: IntelIfModel,
        tokenizer: PreTrainedTokenizerBase,
        config: dict,
        *,
        device: str = "cpu",
        max_tokens: int | None = None,
        max_batch_tokens: int = 32768,
    ):
        self.network = network.to(device).eval()
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.max_tokens = max_tokens or config["max_tokens"]
        self.max_batch_tokens = max_batch_tokens

        self.anchor_id = tokenizer.convert_tokens_to_ids(config["anchor_token"])
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.anchor_id

        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls,
        model: str = DEFAULT_MODEL,
        *,
        revision: str | None = DEFAULT_REVISION,
        device: str | None = None,
        dtype: str | torch.dtype | None = None,
        adapter_path: str | None = None,
        max_tokens: int | None = None,
        max_batch_tokens: int = 32768,
        token: str | None = None,
    ) -> "Intelif":
        config = load_config(model, revision, token)
        device = device or default_device()

        tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"],
            revision=config["base_revision"],
            token=token,
        )

        network = build_network(
            config,
            resolve_dtype(dtype, device),
            adapter_path or resolve_file(model, ADAPTER_FILE, revision, token),
        )

        return cls(
            network,
            tokenizer,
            config,
            device=device,
            max_tokens=max_tokens,
            max_batch_tokens=max_batch_tokens,
        )

    @property
    def name(self) -> str:
        return self.config["name"]

    def system_one(
        self, state: JSONContent | None, questions: Mapping[str, Any]
    ) -> SystemOneResponse:
        normalized = normalize_questions(questions)
        encoded = {
            name: self._encode(state, question) for name, question in normalized.items()
        }

        with self._lock:
            probabilities = self._score(list(encoded.values()))

        answers = {
            name: build_answer(question, encoded[name]["keys"], probs)
            for (name, question), probs in zip(normalized.items(), probabilities)
        }
        input_tokens = sum(len(example["input_ids"]) for example in encoded.values())

        return SystemOneResponse(
            model=self.name,
            usage=Usage(input_tokens=input_tokens),
            answers=answers,
        )

    def choice(
        self,
        state: JSONContent | None,
        criteria: Mapping[str, JSONContent | None],
        instructions: JSONContent | None = None,
    ) -> ChoiceAnswer:
        question = Choice(criteria=dict(criteria), instructions=instructions)
        return self.system_one(state, {"choice": question}).choices["choice"]

    def noul(
        self,
        state: JSONContent | None,
        instructions: JSONContent | None = None,
        criteria: Mapping[str, JSONContent | None] | None = None,
    ) -> NoulAnswer:
        question = Noul(instructions=instructions, criteria=criteria)
        return self.system_one(state, {"noul": question}).nouls["noul"]

    def score(
        self,
        state: JSONContent | None,
        criteria: list[JSONContent],
        instructions: JSONContent | None = None,
    ) -> ScoreAnswer:
        question = Score(criteria=criteria, instructions=instructions)
        return self.system_one(state, {"score": question}).scores["score"]

    def _encode(
        self, state: JSONContent | None, question: Choice | Noul | Score
    ) -> dict:
        example = encode_question(self.tokenizer, state, question, self.anchor_id)

        length = len(example["input_ids"])

        if length > self.max_tokens:
            raise IntelifUnsupportedError(
                f"prompt of {length} tokens exceeds the {self.max_tokens}-token context window"
            )

        return example

    def _groups(self, examples: list[dict]) -> list[list[dict]]:
        groups, current, longest = [], [], 0

        for example in examples:
            length = len(example["input_ids"])
            padded = max(longest, length) * (len(current) + 1)

            if current and padded > self.max_batch_tokens:
                groups.append(current)
                current, longest = [], 0

            current.append(example)
            longest = max(longest, length)

        return groups + [current]

    def _score(self, examples: list[dict]) -> list[list[float]]:
        return [
            probabilities
            for group in self._groups(examples)
            for probabilities in self._score_safely(group)
        ]

    def _score_safely(self, examples: list[dict]) -> list[list[float]]:
        try:
            return self._forward(examples)
        except torch.OutOfMemoryError:
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

            if len(examples) == 1:
                raise IntelifUnsupportedError(
                    f"out of memory on a {len(examples[0]['input_ids'])}-token prompt"
                ) from None

            return [
                probabilities
                for example in examples
                for probabilities in self._score_safely([example])
            ]

    @torch.inference_mode()
    def _forward(self, examples: list[dict]) -> list[list[float]]:
        batch = collate(examples, self.pad_id, self.device)
        probabilities = self.network(**batch).float().softmax(-1)

        return [
            probabilities[i, : len(example["keys"])].tolist()
            for i, example in enumerate(examples)
        ]
