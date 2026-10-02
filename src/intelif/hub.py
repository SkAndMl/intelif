import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from intelif.errors import IntelifModelError
from intelif.render import PROMPT_FORMAT

DEFAULT_MODEL = "UserMoonlight/intelif-qwen3-4b"
DEFAULT_REVISION = "v0.1"

CONFIG_FILE = "config.json"
ADAPTER_FILE = "adapter.safetensors"
CONFIG_VERSION = 1


def resolve_file(
    model: str, filename: str, revision: str | None, token: str | None
) -> str:
    local = Path(model) / filename
    if local.is_file():
        return str(local)

    return hf_hub_download(model, filename, revision=revision, token=token)


def load_config(model: str, revision: str | None, token: str | None) -> dict:
    path = resolve_file(model, CONFIG_FILE, revision, token)
    config = json.loads(Path(path).read_text())

    if config.get("intelif_config_version") != CONFIG_VERSION:
        raise IntelifModelError(
            f"unsupported intelif_config_version {config.get('intelif_config_version')!r}"
        )

    if config.get("prompt_format") != PROMPT_FORMAT:
        raise IntelifModelError(
            f"unsupported prompt_format {config.get('prompt_format')!r}"
        )

    if config.get("architecture") != "qwen3":
        raise IntelifModelError(
            f"unsupported architecture {config.get('architecture')!r}"
        )

    return config
