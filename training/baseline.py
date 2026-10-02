import argparse
import json
import math
import time
from pathlib import Path
from random import Random

import torch
from data import SEED, build_question, load_data
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from intelif import Intelif
from intelif.hub import DEFAULT_MODEL, DEFAULT_REVISION
from intelif.render import option_line, question_options, state_text, to_text

ECE_BINS = 15
LM_METHODS = {
    "zero_shot_lm": "raw",
    "zero_shot_chat": "chat",
    "zero_shot_chat_no_think": "chat_no_think",
}


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


def questions(rows: list[dict]) -> list[tuple]:
    rng = Random(SEED)
    return [(row, *build_question(row, rng)) for row in rows]


def chance_metrics(items: list[tuple]) -> dict:
    sizes = [len(question_options(question)) for _, question, _ in items]

    return {
        "accuracy": sum(1 / size for size in sizes) / len(sizes),
        "nll": sum(math.log(size) for size in sizes) / len(sizes),
        "ece": 0.0,
    }


def lm_body(row: dict, question, options: dict) -> str:
    text = f"STATE:\n{state_text(row['state'])}\n"
    text += f"QUESTION:\n{to_text(question.instructions)}\nCHOICES:\n"
    text += "".join(f"{option_line(key, value)}\n" for key, value in options.items())

    return text


def lm_inputs(
    tokenizer: PreTrainedTokenizerBase, row: dict, question, style: str
) -> tuple[str, list[str], list[str]]:
    options = question_options(question)
    keys = list(options)
    body = lm_body(row, question, options)

    if style == "raw":
        return body + "DECISION:", [f" {key}\n" for key in keys], keys

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": body
                + "\nReply with exactly one option key, copied verbatim.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=style == "chat",
    )

    return prompt, [f"{key}<|im_end|>" for key in keys], keys


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
    items: list[tuple],
    device: str,
    style: str,
) -> dict:
    confidences, correct, nlls = [], [], []

    for row, question, gold in items:
        prompt, continuations, keys = lm_inputs(tokenizer, row, question, style)
        probs = choice_logprobs(lm, tokenizer, prompt, continuations, device)
        probs = probs.log_softmax(-1)
        target = keys.index(gold)

        confidences.append(probs.max().exp().item())
        correct.append(probs.argmax().item() == target)
        nlls.append(-probs[target].item())

    return summarize(confidences, correct, nlls)


def intelif_metrics(model: Intelif, items: list[tuple]) -> dict:
    confidences, correct, nlls = [], [], []

    for row, question, gold in items:
        answer = model.system_one(row["state"], {"q": question}).answers["q"]
        probabilities = (
            answer.probabilities
            if answer.type == "choice"
            else {"true": answer.noul, "false": 1 - answer.noul}
        )
        predicted = max(probabilities, key=probabilities.get)

        confidences.append(probabilities[predicted])
        correct.append(predicted == gold)
        nlls.append(-math.log(max(probabilities[gold], 1e-12)))

    return summarize(confidences, correct, nlls)


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
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--out", default="runs/baseline_results.json")
    parser.add_argument("--skip-lm", action="store_true")
    parser.add_argument("--skip-intelif", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_data()
    splits = {
        name: questions(
            Random(0).sample(rows, min(args.limit, len(rows))) if args.limit else rows
        )
        for name, rows in data["final"].items()
    }
    results = {
        name: {"chance": chance_metrics(items)} for name, items in splits.items()
    }

    if not args.skip_intelif:
        model = Intelif.from_pretrained(args.model, revision=args.revision)

        for name, items in splits.items():
            start = time.perf_counter()
            results[name]["intelif"] = intelif_metrics(model, items)
            print(
                f"intelif/{name}: {results[name]['intelif']} ({time.perf_counter() - start:.0f}s)"
            )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_lm:
        config = json.loads(Path(args.config).read_text())
        tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"], revision=config["base_revision"]
        )
        lm = AutoModelForCausalLM.from_pretrained(
            config["base_model"], revision=config["base_revision"], dtype=torch.bfloat16
        )
        lm = lm.to(device).eval()

        for method, style in LM_METHODS.items():
            for name, items in splits.items():
                start = time.perf_counter()
                results[name][method] = lm_metrics(lm, tokenizer, items, device, style)
                print(
                    f"{method}/{name}: {results[name][method]} ({time.perf_counter() - start:.0f}s)"
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print_table(results)


if __name__ == "__main__":
    main()
