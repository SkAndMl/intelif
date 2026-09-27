from collections import deque
from collections.abc import Iterator
from typing import Literal

import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from intelif import IntelIf, IntelIfConfig
from llama import ModelConfig
from lora import LoraConfig

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
batch_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"running on {device}")

ds = load_dataset("mteb/banking77")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()

CHOICES = set(train_df["label_text"].unique()).union(
    set(test_df["label_text"].unique())
)
CHOICES = sorted(CHOICES)


train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["label"],
)


model = IntelIf(
    cfg=IntelIfConfig(
        model_id=MODEL_ID,
        lora_config=LoraConfig(),
    ),
    base_model_cfg=ModelConfig(),
).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def construct_input_text(
    state: str,
    question: str,
    choices: list[str],
    anchor_token: str,
    choice_first: bool = False,
):
    text = f"STATE:\n{state}\n"
    text += f"QUESTION:\n{question}\n"
    text += "CHOICES:\n"
    for choice in choices:
        if choice_first:
            text += f"{choice} {anchor_token}\n"
        else:
            text += f"{anchor_token} {choice}\n"

    text += "DECISION:"

    return text


def construct_input_ids(
    text: str, anchor_token_id: int, choice_order: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:

    input_ids = torch.tensor(tokenizer.encode(text))
    anchor_mask = input_ids == anchor_token_id

    if anchor_mask.sum().item() != len(choice_order):
        raise ValueError()

    choice_slots = torch.full_like(input_ids, -1)
    choice_slots[anchor_mask] = torch.tensor(choice_order, dtype=torch.long)

    return input_ids, choice_slots


def get_data(
    split: Literal["train", "val", "test"] = "train",
) -> Iterator[dict[str, torch.Tensor]]:
    df: pd.DataFrame | None = None
    match split:
        case "train":
            df = train_df
            df = df.sample(frac=1)
        case "val":
            df = val_df
        case "test":
            df = test_df
        case _:
            raise ValueError()

    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start : start + batch_size]
        bs = len(batch_df)

        input_ids, choice_slots = [], []
        lengths, targets = [], []

        max_len = 0

        for _, row in batch_df.iterrows():
            gold_label_index = CHOICES.index(row["label_text"])
            prompt_order = torch.randperm(n=len(CHOICES)).tolist()
            target_index = prompt_order.index(gold_label_index)

            text = construct_input_text(
                state=row["text"],
                question="What is the intent?",
                choices=[CHOICES[_] for _ in prompt_order],
                anchor_token=tokenizer.pad_token,
                choice_first=True,
            )

            _input_ids, _choice_slots = construct_input_ids(
                text=text,
                anchor_token_id=tokenizer.pad_token_id,
                choice_order=prompt_order,
            )

            input_ids.append(_input_ids)
            choice_slots.append(_choice_slots)
            lengths.append(len(_input_ids))
            targets.append(target_index)

            max_len = max(max_len, lengths[-1])

        input_ids_tensor = torch.full(
            size=(bs, max_len),
            fill_value=tokenizer.pad_token_id,
            dtype=torch.long,
        )
        choice_slots_tensor = torch.full(
            size=(bs, max_len), fill_value=-1, dtype=torch.long
        )

        for i in range(bs):
            input_ids_tensor[i, : lengths[i]] = input_ids[i]
            choice_slots_tensor[i, : lengths[i]] = choice_slots[i]

        yield {
            "input_ids": input_ids_tensor.to(device),
            "choice_slots": choice_slots_tensor.to(device),
            "lengths": torch.tensor(lengths, device=device),
            "targets": torch.tensor(targets, device=device),
        }


def eval_score(split: Literal["val", "test"]) -> tuple[torch.Tensor, float]:
    model.eval()
    total_loss, total_correct, total_examples = 0, 0, 0
    for batch in get_data(split):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits: torch.Tensor = model(
                input_ids=batch["input_ids"],
                lengths=batch["lengths"],
                choice_slots=batch["choice_slots"],
                num_choices=len(CHOICES),
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


epochs = 10
steps_per_epoch = len(train_df) // batch_size  # 7994 steps
log_every = 100
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

print(
    f"Training {len(lora_a_params)} LoRA adapters on a fixed batch with fresh anchors"
)


step = 0

for epoch in range(epochs):
    for batch in get_data("train"):
        step += 1

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=batch["input_ids"],
                lengths=batch["lengths"],
                choice_slots=batch["choice_slots"],
                num_choices=len(CHOICES),
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
            val_loss, val_accuracy = eval_score("val")

            print(
                f"[epoch={epoch + 1}/{epochs}, step={step}] "
                f"train loss: {loss.item():.4f}, train accuracy: {accuracy:.4f}, "
                f"val loss: {val_loss.item():.4f}, val accuracy: {val_accuracy:.4f}, "
                f"rolling_loss: {sum(recent_losses) / len(recent_losses):.4f}, "
                f"rolling_accuracy: {sum(recent_accuracies) / len(recent_accuracies):.4f}, "
                f"grad_norms (before clipping): "
                f"scorer={scorer_grad_norm:.4f}, "
                f"lora_A={lora_a_grad_norm:.4f}, "
                f"lora_B={lora_b_grad_norm:.4f}, "
                f"total={total_grad_norm:.4f}"
            )

    test_loss, test_accuracy = eval_score("test")
    print(
        f"[epoch={epoch + 1}/{epochs}] "
        f"test loss: {test_loss.item():.4f}, test accuracy: {test_accuracy:.4f}"
    )
