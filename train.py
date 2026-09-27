import math
from collections import deque
from random import Random

import torch
from transformers import AutoTokenizer

from data import load_intent_data, make_batches
from intelif import MODEL_ID, IntelIf, IntelIfConfig
from lora import LoraConfig
from qwen import ModelConfig

batch_size = 4
max_tokens = 8192
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"running on {device}")

data = load_intent_data()
train_rng = Random(42)

model = IntelIf(
    cfg=IntelIfConfig(
        model_id=MODEL_ID,
        lora_config=LoraConfig(),
    ),
    base_model_cfg=ModelConfig(),
).to(device)

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


epochs = 5
log_every = 300
rolling_window = 30
max_grad_norm = 1.0
grad_accum_steps = 8
warmup_ratio = 0.05

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


optimizer = torch.optim.AdamW(
    params=trainable_params,
    lr=3e-4,
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

print(f"Training {len(lora_a_params)} LoRA adapters with full choice catalogs")

step = 0
micro_step = 0
best_val_loss = float("inf")

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

            loss = torch.nn.functional.cross_entropy(logits, target=batch["targets"])

        (loss / grad_accum_steps).backward()

        accuracy = (logits.argmax(dim=-1) == batch["targets"]).float().mean().item()
        recent_losses.append(loss.item())
        recent_accuracies.append(accuracy)

        if micro_step % grad_accum_steps != 0:
            continue

        step += 1

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

        if step == 1 or step % log_every == 0:
            print(
                f"[epoch={epoch + 1}/{epochs}, step={step}] "
                f"lr: {scheduler.get_last_lr()[0]:.2e}, "
                f"train loss: {loss.item():.4f}, train accuracy: {accuracy:.4f}, "
                f"rolling_loss: {sum(recent_losses) / len(recent_losses):.4f}, "
                f"rolling_accuracy: {sum(recent_accuracies) / len(recent_accuracies):.4f}, "
                f"grad_norms (before clipping): "
                f"scorer={scorer_grad_norm:.4f}, "
                f"lora_A={lora_a_grad_norm:.4f}, "
                f"lora_B={lora_b_grad_norm:.4f}, "
                f"total={total_grad_norm:.4f}"
            )

            total_val_loss = 0

            for name, examples in data["validation"].items():
                val_loss, val_accuracy = eval_score(examples)
                print(
                    f"  validation/{name}: "
                    f"loss={val_loss.item():.4f}, accuracy={val_accuracy:.4f}"
                )

                total_val_loss += val_loss

            if total_val_loss.item() < best_val_loss:
                best_val_loss = total_val_loss.item()
                torch.save(model.state_dict(), "best_model.pt")


model.load_state_dict(torch.load("best_model.pt"))

for name, examples in data["final"].items():
    test_loss, test_accuracy = eval_score(examples)
    print(f"final/{name}: loss={test_loss.item():.4f}, accuracy={test_accuracy:.4f}")
