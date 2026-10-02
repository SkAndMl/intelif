import argparse
import json
import math
import time
from pathlib import Path
from random import Random

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from data import QUESTION, load_intent_data, make_batches, readable_intent
from hf import ADAPTER_PATH, REPO_NAME
from intelif import MODEL_ID, IntelIf, IntelIfConfig
from lora import LoraConfig
from qwen import ModelConfig

ECE_BINS = 15
RESULTS_PATH = "baseline_results.json"
LM_METHODS = {
    "zero_shot_lm": "raw",
    "zero_shot_chat": "chat",
    "zero_shot_chat_no_think": "chat_no_think",
}


def row_choices(row: dict, catalogs: dict[str, list[str]]) -> list[str]:
    return row.get("choices") or catalogs[row.get("choice_pool", row["source"])]


def summarize(confidences: list[float], correct: list[bool], nlls: list[float]) -> dict:
    bins = [[] for _ in range(ECE_BINS)]
    for confidence, is_correct in zip(confidences, correct):
        bins[min(int(confidence * ECE_BINS), ECE_BINS - 1)].append(
            (confidence, is_correct)
        )

    ece = sum(
        abs(sum(c for c, _ in b) - sum(ok for _, ok in b)) for b in bins if b
    ) / len(correct)

    return {
        "accuracy": sum(correct) / len(correct),
        "nll": sum(nlls) / len(nlls),
        "ece": ece,
    }


def chance_metrics(rows: list[dict], catalogs: dict[str, list[str]]) -> dict:
    sizes = [len(row_choices(row, catalogs)) for row in rows]

    return {
        "accuracy": sum(1 / size for size in sizes) / len(sizes),
        "nll": sum(math.log(size) for size in sizes) / len(sizes),
        "ece": 0.0,
    }


def lm_prompt(state: str, question: str, choices: list[str]) -> str:
    text = f"STATE:\n{state}\nQUESTION:\n{question}\nCHOICES:\n"
    text += "".join(f"{readable_intent(choice)}\n" for choice in choices)

    return text + "DECISION:"


def chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    state: str,
    question: str,
    choices: list[str],
    thinking: bool,
) -> str:
    body = f"STATE:\n{state}\nQUESTION:\n{question}\nCHOICES:\n"
    body += "".join(f"{readable_intent(choice)}\n" for choice in choices)
    body += "\nReply with exactly one of the choices, copied verbatim."

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def lm_inputs(
    tokenizer: PreTrainedTokenizerBase,
    row: dict,
    candidates: list[str],
    style: str,
) -> tuple[str, list[str]]:
    question = row.get("question", QUESTION)

    if style == "raw":
        return (
            lm_prompt(row["state"], question, candidates),
            [f" {readable_intent(choice)}\n" for choice in candidates],
        )

    return (
        chat_prompt(
            tokenizer, row["state"], question, candidates, thinking=style == "chat"
        ),
        [f"{readable_intent(choice)}<|im_end|>" for choice in candidates],
    )


@torch.no_grad()
def choice_logprobs(
    lm: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    continuations: list[str],
    device: str,
) -> Tensor:
    prefix = tokenizer.encode(prompt)
    choices = [tokenizer.encode(text) for text in continuations]

    length = len(prefix) + sum(len(ids) for ids in choices)
    mask = torch.zeros(length, length, dtype=torch.bool)
    mask[: len(prefix), : len(prefix)] = torch.ones(
        len(prefix), len(prefix), dtype=torch.bool
    ).tril()

    positions = list(range(len(prefix)))
    predict_at, targets, owners = [], [], []
    start = len(prefix)

    for index, ids in enumerate(choices):
        end = start + len(ids)
        mask[start:end, : len(prefix)] = True
        mask[start:end, start:end] = torch.ones(
            len(ids), len(ids), dtype=torch.bool
        ).tril()

        positions += range(len(prefix), len(prefix) + len(ids))
        predict_at += [len(prefix) - 1] + list(range(start, end - 1))
        targets += ids
        owners += [index] * len(ids)
        start = end

    input_ids = torch.tensor(
        [prefix + [t for ids in choices for t in ids]], device=device
    )
    bias = torch.zeros(length, length, dtype=lm.dtype).masked_fill(
        ~mask, torch.finfo(lm.dtype).min
    )
    hidden = lm.model(
        input_ids=input_ids,
        attention_mask=bias[None, None].to(device),
        position_ids=torch.tensor([positions], device=device),
    ).last_hidden_state[0]

    logprobs = lm.lm_head(hidden[predict_at]).float().log_softmax(-1)
    token_scores = logprobs.gather(-1, torch.tensor(targets, device=device)[:, None])

    return torch.zeros(len(choices), device=device).index_add_(
        0, torch.tensor(owners, device=device), token_scores.squeeze(-1)
    )


def lm_metrics(
    lm: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    rows: list[dict],
    catalogs: dict[str, list[str]],
    device: str,
    style: str,
) -> dict:
    rng = Random(42)
    confidences, correct, nlls = [], [], []

    for row in rows:
        candidates = row_choices(row, catalogs).copy()
        rng.shuffle(candidates)

        prompt, continuations = lm_inputs(tokenizer, row, candidates, style)
        scores = choice_logprobs(lm, tokenizer, prompt, continuations, device)

        probs = scores.log_softmax(-1)
        target = candidates.index(row["gold_intent"])

        confidences.append(probs.max().exp().item())
        correct.append(probs.argmax().item() == target)
        nlls.append(-probs[target].item())

    return summarize(confidences, correct, nlls)


@torch.no_grad()
def intelif_metrics(
    model: IntelIf,
    tokenizer: PreTrainedTokenizerBase,
    rows: list[dict],
    catalogs: dict[str, list[str]],
    device: str,
) -> dict:
    confidences, correct, nlls = [], [], []

    for batch in make_batches(rows, catalogs, tokenizer, 32, 4096, device, Random(42)):
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
        ):
            logits = model(
                input_ids=batch["input_ids"],
                lengths=batch["lengths"],
                choice_slots=batch["choice_slots"],
                choice_mask=batch["choice_mask"],
            )

        probs = logits.float().log_softmax(-1)
        targets = batch["targets"]

        confidences += probs.max(-1).values.exp().tolist()
        correct += (probs.argmax(-1) == targets).tolist()
        nlls += (-probs.gather(-1, targets[:, None]).squeeze(-1)).tolist()

    return summarize(confidences, correct, nlls)


def resolve_adapter(path: str) -> str:
    if Path(path).exists():
        return path

    repo_id = f"{HfApi().whoami()['name']}/{REPO_NAME}"
    return hf_hub_download(repo_id, ADAPTER_PATH)


def load_intelif(adapter_path: str, device: str) -> IntelIf:
    model = IntelIf(
        cfg=IntelIfConfig(model_id=MODEL_ID, lora_config=LoraConfig()),
        base_model_cfg=ModelConfig(gradient_checkpointing=False),
    ).to(device)

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    incompatible = model.load_state_dict(
        load_file(adapter_path, device=device), strict=False
    )
    if incompatible.unexpected_keys or set(incompatible.missing_keys) & trainable:
        raise RuntimeError("The adapter does not match the model")

    return model.eval()


def print_table(results: dict) -> None:
    methods = ["chance", *LM_METHODS, "intelif"]
    header = "| split | " + " | ".join(
        f"{method} acc | {method} nll | {method} ece" for method in methods
    )
    print(header + " |")
    print("|---" * (1 + 3 * len(methods)) + "|")

    for split, by_method in results.items():
        cells = []
        for method in methods:
            metrics = by_method.get(method)
            cells += (
                [f"{metrics[key]:.4f}" for key in ("accuracy", "nll", "ece")]
                if metrics
                else ["-"] * 3
            )
        print(f"| {split} | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--adapter", default=ADAPTER_PATH)
    parser.add_argument("--skip-lm", action="store_true")
    parser.add_argument("--skip-intelif", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_intent_data()
    catalogs = data["choices"]
    splits = {
        name: Random(0).sample(rows, min(args.limit, len(rows))) if args.limit else rows
        for name, rows in data["final"].items()
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    results = {
        name: {"chance": chance_metrics(rows, catalogs)}
        for name, rows in splits.items()
    }

    if not args.skip_intelif:
        adapter = resolve_adapter(args.adapter)
        print(f"loading adapter from {adapter}")

        model = load_intelif(adapter, device)

        for name, rows in splits.items():
            start = time.perf_counter()
            results[name]["intelif"] = intelif_metrics(
                model, tokenizer, rows, catalogs, device
            )
            print(
                f"intelif/{name}: {results[name]['intelif']} ({time.perf_counter() - start:.0f}s)"
            )

        del model
        torch.cuda.empty_cache()

    if not args.skip_lm:
        lm = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        lm = lm.to(device).eval()

        for method, style in LM_METHODS.items():
            for name, rows in splits.items():
                start = time.perf_counter()
                results[name][method] = lm_metrics(
                    lm, tokenizer, rows, catalogs, device, style
                )
                print(
                    f"{method}/{name}: {results[name][method]} ({time.perf_counter() - start:.0f}s)"
                )

    with open(RESULTS_PATH, "w") as file:
        json.dump(results, file, indent=2)

    print_table(results)


if __name__ == "__main__":
    main()
