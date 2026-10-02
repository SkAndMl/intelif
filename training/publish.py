import json
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

load_dotenv(find_dotenv())

REPO_NAME = "intelif-qwen3-4b"
ADAPTER_PATH = "adapter.safetensors"
LOG_PATH = "train.log"
RESULTS_PATH = "results.json"


def metrics_table(metrics: dict[str, dict[str, float]]) -> str:
    rows = [
        f"| {name} | {values['accuracy']:.4f} | {values['loss']:.4f} |"
        for name, values in metrics.items()
    ]

    return "| split | accuracy | loss |\n|---|---|---|\n" + "\n".join(rows)


def key_value_table(values: dict) -> str:
    rows = [f"| {key} | `{value}` |" for key, value in values.items()]

    return "| name | value |\n|---|---|\n" + "\n".join(rows)


def build_readme(results: dict) -> str:
    hyperparams = results["hyperparams"]
    best = results["best"]

    return f"""---
base_model: {hyperparams["model_id"]}
tags:
- intelif
- lora
---

# {REPO_NAME}

LoRA adapters and choice scorer for Intelif on top of `{hyperparams["model_id"]}`.

- `{ADAPTER_PATH}`: trainable weights (LoRA A/B and scorer) from the best checkpoint
- `{LOG_PATH}`: full training log
- `{RESULTS_PATH}`: everything in this card as JSON

## Held-out results

Best checkpoint at step {best["step"]} (summed validation loss {best["val_loss"]:.4f}).

{metrics_table(results["final"])}

## Validation at best checkpoint

{metrics_table(best["validation"])}

## Training data

{key_value_table(results["train_rows"])}

## Hyperparameters

{key_value_table(hyperparams)}
"""


def get_repo_id(api: HfApi) -> str:
    return f"{api.whoami()['name']}/{REPO_NAME}"


def download_adapter(revision: str | None = None) -> str:
    return hf_hub_download(get_repo_id(HfApi()), ADAPTER_PATH, revision=revision)


def upload_run(private: bool = True) -> str:
    results = json.loads(Path(RESULTS_PATH).read_text())

    api = HfApi()
    repo_id = get_repo_id(api)
    api.create_repo(repo_id, private=private, exist_ok=True)

    api.create_commit(
        repo_id=repo_id,
        operations=[
            CommitOperationAdd(ADAPTER_PATH, ADAPTER_PATH),
            CommitOperationAdd(LOG_PATH, LOG_PATH),
            CommitOperationAdd(RESULTS_PATH, RESULTS_PATH),
            CommitOperationAdd("README.md", build_readme(results).encode()),
        ],
        commit_message=f"Upload run (best step {results['best']['step']})",
    )

    return f"https://huggingface.co/{repo_id}"


if __name__ == "__main__":
    print(upload_run())
