from typing import Any

import torch
from decision_index.engines import Engine, Unsupported

from intelif.errors import IntelifUnsupportedError
from intelif.hub import DEFAULT_MODEL, DEFAULT_REVISION
from intelif.model import Intelif

SUPPORTED_TYPES = ("choice", "noul")


def resolved_revision(model: str, revision: str | None) -> str | None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError, HFValidationError

    try:
        return HfApi().model_info(model, revision=revision).sha
    except (HfHubHTTPError, HFValidationError, OSError, ValueError):
        return revision


class IntelifEngine(Engine):
    name = "intelif"
    latency = (
        "Device-synchronized in-process request wall time of Intelif.system_one, "
        "including prompt rendering and tokenization; excludes model loading."
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        revision: str | None = DEFAULT_REVISION,
        device: str | None = None,
        dtype: str | None = None,
        max_tokens: int | None = None,
        max_batch_tokens: int = 32768,
        client: Intelif | None = None,
        **options: Any,
    ):
        super().__init__(**options)

        self.client = client or Intelif.from_pretrained(
            model,
            revision=revision,
            device=device,
            dtype=dtype,
            max_tokens=max_tokens,
            max_batch_tokens=int(max_batch_tokens),
        )
        self.device = self.client.device

        self.provenance = {
            "kind": "intelif",
            "package": "intelif",
            "model": model,
            "revision": revision,
            "resolved_revision": resolved_revision(model, revision),
            "base_model": self.client.config["base_model"],
            "base_revision": self.client.config["base_revision"],
            "lora": self.client.config["lora"],
            "prompt_format": self.client.config["prompt_format"],
            "device": self.device,
            "context_limit_tokens": self.client.max_tokens,
            "policy": (
                "Intelif.system_one as published: one fixed intelif-v1 rendering for "
                "every benchmark, an anchor token after each option, a linear scorer "
                "on each anchor's final hidden state, and a softmax over options. "
                "noul questions are scored as the options true/false, described by "
                "their criteria when given and by Yes/No otherwise. Prompts over the "
                "context window are refused as unsupported, never truncated. No "
                "option is filtered."
            ),
        }

    def runtime(self) -> dict:
        info = {"torch": torch.__version__, "device": self.device}
        if self.device.startswith("cuda"):
            info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())

        return info

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def __call__(self, state, questions: dict):
        for question in questions.values():
            if question["type"] not in SUPPORTED_TYPES:
                raise Unsupported(f"unsupported question type {question['type']}")

        try:
            response = self.client.system_one(state, questions)
        except IntelifUnsupportedError as error:
            raise Unsupported(str(error)) from None

        return response.model_dump(mode="json"), None
