import json
import math
import time
from collections import Counter, deque
from random import Random

import torch
from data import DATASET_REVISIONS, load_intent_data, make_batches
from publish import ADAPTER_PATH, LOG_PATH, RESULTS_PATH, upload_run
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from intelif.modeling.intelif import MODEL_ID, IntelIfModel
from intelif.modeling.lora import LoraConfig, inject_lora
from intelif.modeling.qwen import ModelConfig, Qwen3Model

batch_size = 32
max_tokens = 16384
device = "cuda" if torch.cuda.is_available() else "cpu"

open(LOG_PATH, "w").close()


def log(message: str) -> None:
    print(message)
    with open(LOG_PATH, "a") as file:
        file.write(message + "\n")


log(f"running on {device}")

data = load_intent_data()
train_rng = Random(42)

lora_config = LoraConfig()
base_model_cfg = ModelConfig()

BASE_MODEL = "Qwen/Qwen3-4B"
BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

lora_config = LoraConfig()
base_model_cfg = ModelConfig(gradient_checkpointing=False)

model = IntelIfModel(
    Qwen3Model.from_pretrained(
        BASE_MODEL,
        base_model_cfg,
        revision=BASE_REVISION,
    )
).to(device)

model.base_model.requires_grad_(False)

inject_lora(model.base_model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def get_batches(
    examples: list[dict[str, str]],
    rng: Random,
    shuffle: bool = False,
):
    return make_batches(
        examples=examples,
        choices=data["choices"],
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_tokens=max_tokens,
        device=device,
        rng=rng,
        shuffle=shuffle,
    )


def eval_score(examples: list[dict[str, str]]) -> tuple[torch.Tensor, float]:
    model.eval()
    total_loss, total_correct, total_examples = 0, 0, 0
    for batch in get_batches(examples, Random(42)):
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
            ),
        ):
            logits: torch.Tensor = model(
                input_ids=batch["input_ids"],
                lengths=batch["lengths"],
                choice_slots=batch["choice_slots"],
                choice_mask=batch["choice_mask"],
            )
            total_loss += torch.nn.functional.cross_entropy(
                logits, target=batch["targets"], reduction="sum"
            )

        total_correct += (
            (logits.argmax(dim=-1) == batch["targets"]).float().sum().item()
        )
        total_examples += batch["targets"].numel()

    model.train()
    return total_loss / total_examples, total_correct / total_examples


epochs = 2
log_every = 200
rolling_window = 30
max_grad_norm = 1.0
grad_accum_steps = 2
warmup_ratio = 0.05
learning_rate = 3e-4

trainable_params = [
    parameter for parameter in model.parameters() if parameter.requires_grad
]
lora_a_params = [
    parameter
    for name, parameter in model.named_parameters()
    if name.endswith("lora_A.weight") and parameter.requires_grad
]
lora_b_params = [
    parameter
    for name, parameter in model.named_parameters()
    if name.endswith("lora_B.weight") and parameter.requires_grad
]

if not lora_a_params or len(lora_a_params) != len(lora_b_params):
    raise RuntimeError("LoRA adapters were not injected as expected")


def gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    if any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("A trainable parameter did not receive a gradient")
    return torch.linalg.vector_norm(
        torch.stack(
            [parameter.grad.detach().float().norm() for parameter in parameters]
        )
    ).item()


trainable_names = {
    name for name, parameter in model.named_parameters() if parameter.requires_grad
}


def save_adapter() -> None:
    save_file(
        {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
            if name in trainable_names
        },
        ADAPTER_PATH,
    )


optimizer = torch.optim.AdamW(
    params=trainable_params,
    lr=learning_rate,
)

batches_per_epoch = sum(1 for _ in get_batches(data["train"], Random(0), shuffle=True))
total_steps = epochs * batches_per_epoch // grad_accum_steps
warmup_steps = max(1, int(warmup_ratio * total_steps))


def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps

    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

recent_losses = deque(maxlen=rolling_window)
recent_accuracies = deque(maxlen=rolling_window)

hyperparams = {
    "model_id": MODEL_ID,
    "lora_r": lora_config.r,
    "lora_alpha": lora_config.lora_alpha,
    "lora_dropout": lora_config.lora_dropout,
    "lora_target_modules": list(lora_config.target_modules),
    "base_dtype": str(base_model_cfg.dtype),
    "gradient_checkpointing": base_model_cfg.gradient_checkpointing,
    "epochs": epochs,
    "batch_size": batch_size,
    "max_tokens": max_tokens,
    "grad_accum_steps": grad_accum_steps,
    "learning_rate": learning_rate,
    "warmup_ratio": warmup_ratio,
    "warmup_steps": warmup_steps,
    "total_steps": total_steps,
    "max_grad_norm": max_grad_norm,
    "seed": 42,
    "dataset_revisions": DATASET_REVISIONS,
}

log(f"hyperparams: {json.dumps(hyperparams)}")
log(f"Training {len(lora_a_params)} LoRA adapters with full choice catalogs")

step = 0
micro_step = 0
window_rows = 0
window_tokens = 0
window_start = time.perf_counter()
best_val_loss = float("inf")
best = {}

for epoch in range(epochs):
    for batch in get_batches(data["train"], train_rng, shuffle=True):
        micro_step += 1

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
        ):
            logits = model(
                input_ids=batch["input_ids"],
                lengths=batch["lengths"],
                choice_slots=batch["choice_slots"],
                choice_mask=batch["choice_mask"],
            )

            loss_sum = torch.nn.functional.cross_entropy(
                logits, target=batch["targets"], reduction="sum"
            )

        loss_sum.backward()

        rows = batch["targets"].numel()
        window_rows += rows
        window_tokens += batch["input_ids"].numel()

        loss = loss_sum.item() / rows
        accuracy = (logits.argmax(dim=-1) == batch["targets"]).float().mean().item()
        recent_losses.append(loss)
        recent_accuracies.append(accuracy)

        if micro_step % grad_accum_steps != 0:
            continue

        step += 1

        for parameter in trainable_params:
            if parameter.grad is not None:
                parameter.grad.div_(window_rows)

        window_rows = 0

        scorer_grad_norm = model.scorer.weight.grad.norm().item()
        lora_a_grad_norm = gradient_norm(lora_a_params)
        lora_b_grad_norm = gradient_norm(lora_b_params)

        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=max_grad_norm,
            error_if_nonfinite=True,
        ).item()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step == 1 or step % log_every == 0 or step == total_steps:
            elapsed = time.perf_counter() - window_start
            peak_memory = (
                torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
            )

            log(
                f"[epoch={epoch + 1}/{epochs}, step={step}/{total_steps}] "
                f"lr: {scheduler.get_last_lr()[0]:.2e}, "
                f"tokens/s: {window_tokens / elapsed:.0f}, "
                f"peak_memory: {peak_memory:.1f}GiB, "
                f"train loss: {loss:.4f}, train accuracy: {accuracy:.4f}, "
                f"rolling_loss: {sum(recent_losses) / len(recent_losses):.4f}, "
                f"rolling_accuracy: {sum(recent_accuracies) / len(recent_accuracies):.4f}, "
                f"grad_norms (before clipping): "
                f"scorer={scorer_grad_norm:.4f}, "
                f"lora_A={lora_a_grad_norm:.4f}, "
                f"lora_B={lora_b_grad_norm:.4f}, "
                f"total={total_grad_norm:.4f}"
            )

            total_val_loss = 0
            validation = {}

            for name, examples in data["validation"].items():
                val_loss, val_accuracy = eval_score(examples)
                log(
                    f"  validation/{name}: "
                    f"loss={val_loss.item():.4f}, accuracy={val_accuracy:.4f}"
                )

                total_val_loss += val_loss
                validation[name] = {"loss": val_loss.item(), "accuracy": val_accuracy}

            if total_val_loss.item() < best_val_loss:
                best_val_loss = total_val_loss.item()
                best = {
                    "step": step,
                    "val_loss": best_val_loss,
                    "validation": validation,
                }
                save_adapter()

            window_tokens = 0
            window_start = time.perf_counter()


incompatible = model.load_state_dict(
    load_file(ADAPTER_PATH, device=device), strict=False
)
if incompatible.unexpected_keys or set(incompatible.missing_keys) & trainable_names:
    raise RuntimeError("The saved adapter does not match the model")

final = {}

for name, examples in data["final"].items():
    test_loss, test_accuracy = eval_score(examples)
    log(f"final/{name}: loss={test_loss.item():.4f}, accuracy={test_accuracy:.4f}")

    final[name] = {"loss": test_loss.item(), "accuracy": test_accuracy}

with open(RESULTS_PATH, "w") as file:
    json.dump(
        {
            "hyperparams": hyperparams,
            "best": best,
            "final": final,
            "train_rows": dict(Counter(row["source"] for row in data["train"])),
        },
        file,
        indent=2,
    )

log(f"uploaded to {upload_run()}")
