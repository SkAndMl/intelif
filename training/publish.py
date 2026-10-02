import argparse
import json
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi

load_dotenv(find_dotenv())

REPO_NAME = "intelif-qwen3-4b"
ADAPTER_PATH = "adapter.safetensors"
CONFIG_PATH = "config.json"
LOG_PATH = "train.log"
RESULTS_PATH = "results.json"


def metrics_table(metrics: dict[str, dict[str, float]]) -> str:
    rows = [
        f"| {name} | {values['accuracy']:.4f} | {values['loss']:.4f} | {values['ece']:.4f} |"
        for name, values in metrics.items()
    ]

    return "| split | accuracy | loss | ece |\n|---|---|---|---|\n" + "\n".join(rows)


def key_value_table(values: dict) -> str:
    rows = [f"| {key} | `{value}` |" for key, value in values.items()]

    return "| name | value |\n|---|---|\n" + "\n".join(rows)


def build_readme(results: dict) -> str:
    hyperparams = results["hyperparams"]
    best = results["best"]
    lora = hyperparams["lora"]

    return f"""---
base_model: {hyperparams["base_model"]}
tags:
- intelif
- lora
---

# {REPO_NAME}

LoRA adapters (r={lora["r"]} on {", ".join(lora["target_modules"])}) and a linear
choice scorer for Intelif on top of `{hyperparams["base_model"]}`.

- `{ADAPTER_PATH}`: trainable weights (LoRA A/B and scorer) from the best checkpoint
- `{CONFIG_PATH}`: the model spec read by `intelif.Intelif.from_pretrained`
- `{LOG_PATH}`: full training log
- `{RESULTS_PATH}`: everything in this card as JSON

## Held-out results

Best checkpoint at step {best["step"]} of {hyperparams["total_steps"]} (summed validation loss {best["val_loss"]:.4f}).

{metrics_table(results["final"])}

## Training data

{key_value_table(results["train_rows"])}

## Hyperparameters

{key_value_table(hyperparams)}
"""


def get_repo_id(api: HfApi) -> str:
    return f"{api.whoami()['name']}/{REPO_NAME}"


def upload_run(run_dir: Path, private: bool = True) -> str:
    run_dir = Path(run_dir)
    results = json.loads((run_dir / RESULTS_PATH).read_text())

    api = HfApi()
    repo_id = get_repo_id(api)
    api.create_repo(repo_id, private=private, exist_ok=True)

    commit = api.create_commit(
        repo_id=repo_id,
        operations=[
            CommitOperationAdd(name, str(run_dir / name))
            for name in (ADAPTER_PATH, CONFIG_PATH, LOG_PATH, RESULTS_PATH)
        ]
        + [CommitOperationAdd("README.md", build_readme(results).encode())],
        commit_message=f"Upload run (best step {results['best']['step']})",
    )

    return f"https://huggingface.co/{repo_id} at revision {commit.oid}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a training run.")
    parser.add_argument("run_dir", nargs="?", default="runs/train")
    print(upload_run(Path(parser.parse_args().run_dir)))
