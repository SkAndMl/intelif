import argparse
import json
import math
import time
from collections import Counter, deque
from pathlib import Path
from random import Random

import torch
from data import DATASET_REVISIONS, SEED, TRAIN_CAPS, load_data, make_batches
from publish import ADAPTER_PATH, CONFIG_PATH, LOG_PATH, RESULTS_PATH, upload_run
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoTokenizer

from intelif.modeling.intelif import IntelIfModel
from intelif.modeling.lora import LoraConfig, inject_lora
from intelif.modeling.qwen import ModelConfig, Qwen3Model

BATCH_SIZE = 32
MAX_TOKENS = 4096
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
EPOCHS = 1
LOG_EVERY = 200
SMOKE_STEPS = 50
ECE_BINS = 15


def expected_calibration_error(confidences: list, correct: list) -> float:
    bins = [[0.0, 0.0, 0] for _ in range(ECE_BINS)]
    for confidence, is_correct in zip(confidences, correct):
        bucket = bins[min(int(confidence * ECE_BINS), ECE_BINS - 1)]
        bucket[0] += confidence
        bucket[1] += is_correct
        bucket[2] += 1

    return sum(abs(c - k) for c, k, n in bins if n) / max(1, len(correct))


def build_model(config: dict, device: str, checkpointing: bool) -> IntelIfModel:
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base_cfg = ModelConfig(
        **config["base_config"], dtype=dtype, gradient_checkpointing=checkpointing
    )
    base_model = Qwen3Model.from_pretrained(
        config["base_model"], base_cfg, revision=config["base_revision"]
    )

    model = IntelIfModel(base_model)
    model.requires_grad_(False)

    lora = config["lora"]
    inject_lora(
        model.base_model,
        LoraConfig(**{**lora, "target_modules": tuple(lora["target_modules"])}),
    )
    model.scorer.requires_grad_(True)

    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Intelif adapter.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--out", default="runs/train")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / LOG_PATH).write_text("")

    def log(message: str) -> None:
        print(message, flush=True)
        with open(out / LOG_PATH, "a") as file:
            file.write(message + "\n")

    log(f"running on {device}")

    data = load_data(args.smoke)
    log(f"train rows: {dict(Counter(row['source'] for row in data['train']))}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"], revision=config["base_revision"]
    )
    anchor_id = tokenizer.convert_tokens_to_ids(config["anchor_token"])
    model = build_model(config, device, args.checkpointing)

    def batches(rows: list, rng: Random, shuffle: bool = False):
        return make_batches(
            rows, tokenizer, anchor_id, BATCH_SIZE, MAX_TOKENS, device, rng, shuffle
        )

    def autocast():
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
        )

    def forward(batch: dict) -> torch.Tensor:
        return model(
            input_ids=batch["input_ids"],
            lengths=batch["lengths"],
            choice_slots=batch["choice_slots"],
            choice_mask=batch["choice_mask"],
        )

    @torch.no_grad()
    def evaluate(rows: list) -> dict:
        model.eval()
        total_loss, confidences, correct = 0.0, [], []

        for batch in batches(rows, Random(SEED)):
            with autocast():
                logits = forward(batch)

            probs = logits.float().log_softmax(-1)
            total_loss -= probs.gather(-1, batch["targets"][:, None]).sum().item()
            confidences += probs.max(-1).values.exp().tolist()
            correct += (probs.argmax(-1) == batch["targets"]).tolist()

        model.train()

        return {
            "loss": total_loss / len(correct),
            "accuracy": sum(correct) / len(correct),
            "ece": expected_calibration_error(confidences, correct),
        }

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    log(f"trainable parameters: {sum(p.numel() for p in trainable):,}")

    def save_adapter() -> None:
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in model.state_dict().items()
                if name in trainable_names
            },
            out / ADAPTER_PATH,
            metadata={
                "lora": json.dumps(config["lora"]),
                "prompt_format": config["prompt_format"],
            },
        )

    batches_per_epoch = sum(1 for _ in batches(data["train"], Random(0), True))
    total_steps = max(1, EPOCHS * batches_per_epoch // GRAD_ACCUM_STEPS)
    if args.smoke:
        total_steps = min(total_steps, SMOKE_STEPS)
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    hyperparams = {
        "base_model": config["base_model"],
        "base_revision": config["base_revision"],
        "lora": config["lora"],
        "trainable_parameters": sum(p.numel() for p in trainable),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_tokens": MAX_TOKENS,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": WARMUP_RATIO,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "max_grad_norm": MAX_GRAD_NORM,
        "gradient_checkpointing": args.checkpointing,
        "train_caps": TRAIN_CAPS,
        "seed": SEED,
        "dataset_revisions": DATASET_REVISIONS,
        "smoke": args.smoke,
    }
    log(f"hyperparams: {json.dumps(hyperparams)}")

    recent_losses, recent_accuracies = deque(maxlen=30), deque(maxlen=30)
    step, micro_step, window_rows, window_tokens = 0, 0, 0, 0
    window_start = time.perf_counter()
    best_val_loss, best = float("inf"), {}
    train_rng = Random(SEED)

    for epoch in range(EPOCHS):
        for batch in batches(data["train"], train_rng, shuffle=True):
            micro_step += 1

            with autocast():
                logits = forward(batch)
                loss_sum = nn.functional.cross_entropy(
                    logits.float(), batch["targets"], reduction="sum"
                )

            loss_sum.backward()

            rows = batch["targets"].numel()
            window_rows += rows
            window_tokens += batch["input_ids"].numel()
            recent_losses.append(loss_sum.item() / rows)
            recent_accuracies.append(
                (logits.argmax(-1) == batch["targets"]).float().mean().item()
            )

            if micro_step % GRAD_ACCUM_STEPS != 0:
                continue

            step += 1
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.div_(window_rows)
            window_rows = 0

            grad_norm = nn.utils.clip_grad_norm_(
                trainable, MAX_GRAD_NORM, error_if_nonfinite=True
            ).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step == 1 or step % LOG_EVERY == 0 or step == total_steps:
                elapsed = time.perf_counter() - window_start
                peak = (
                    torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0
                )
                log(
                    f"[epoch={epoch + 1}/{EPOCHS}, step={step}/{total_steps}] "
                    f"lr: {scheduler.get_last_lr()[0]:.2e}, "
                    f"tokens/s: {window_tokens / elapsed:.0f}, "
                    f"peak_memory: {peak:.1f}GiB, "
                    f"rolling_loss: {sum(recent_losses) / len(recent_losses):.4f}, "
                    f"rolling_accuracy: {sum(recent_accuracies) / len(recent_accuracies):.4f}, "
                    f"grad_norm: {grad_norm:.4f}"
                )

                validation = {}
                for name, split_rows in data["validation"].items():
                    validation[name] = evaluate(split_rows)
                    metrics = validation[name]
                    log(
                        f"  validation/{name}: loss={metrics['loss']:.4f}, "
                        f"accuracy={metrics['accuracy']:.4f}, ece={metrics['ece']:.4f}"
                    )

                total_val_loss = sum(m["loss"] for m in validation.values())
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    best = {
                        "step": step,
                        "val_loss": total_val_loss,
                        "validation": validation,
                    }
                    save_adapter()

                window_tokens = 0
                window_start = time.perf_counter()

            if step >= total_steps:
                break

        if step >= total_steps:
            break

    incompatible = model.load_state_dict(
        load_file(out / ADAPTER_PATH, device=device), strict=False
    )
    if incompatible.unexpected_keys or set(incompatible.missing_keys) & trainable_names:
        raise RuntimeError("The saved adapter does not match the model")

    final = {}
    for name, split_rows in data["final"].items():
        final[name] = evaluate(split_rows)
        metrics = final[name]
        log(
            f"final/{name}: loss={metrics['loss']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, ece={metrics['ece']:.4f}"
        )

    (out / CONFIG_PATH).write_text(json.dumps(config, indent=2) + "\n")
    (out / RESULTS_PATH).write_text(
        json.dumps(
            {
                "hyperparams": hyperparams,
                "best": best,
                "final": final,
                "train_rows": dict(Counter(row["source"] for row in data["train"])),
            },
            indent=2,
        )
    )

    if args.smoke or args.no_upload:
        log("skipping upload")
        return

    log(f"uploaded to {upload_run(out)}")


if __name__ == "__main__":
    main()
