from collections import deque
from random import Random

import torch
from transformers import AutoTokenizer

from data import load_intent_data, make_batches
from intelif import IntelIf, IntelIfConfig
from llama import ModelConfig
from lora import LoraConfig

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
batch_size = 8
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

recent_losses = deque(maxlen=rolling_window)
recent_accuracies = deque(maxlen=rolling_window)

print(f"Training {len(lora_a_params)} LoRA adapters with full choice catalogs")

step = 0
best_val_loss = float("inf")

for epoch in range(epochs):
    for batch in get_batches(data["train"], train_rng, shuffle=True):
        step += 1

        optimizer.zero_grad()
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

        loss.backward()

        scorer_grad_norm = model.scorer.weight.grad.norm().item()
        lora_a_grad_norm = gradient_norm(lora_a_params)
        lora_b_grad_norm = gradient_norm(lora_b_params)

        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=max_grad_norm,
            error_if_nonfinite=True,
        ).item()

        optimizer.step()

        accuracy = (logits.argmax(dim=-1) == batch["targets"]).float().mean().item()
        recent_losses.append(loss.item())
        recent_accuracies.append(accuracy)

        if step == 1 or step % log_every == 0:
            print(
                f"[epoch={epoch + 1}/{epochs}, step={step}] "
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
