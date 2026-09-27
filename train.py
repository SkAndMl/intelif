from collections import deque

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from intelif import IntelIf, IntelIfConfig
from llama import ModelConfig
from lora import LoraConfig

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

ds = load_dataset("mteb/banking77")
df = ds["train"].to_pandas()

top_4 = df["label_text"].value_counts().head(4).index.to_list()

df_4 = df[df["label_text"].isin(top_4)]


model = IntelIf(
    cfg=IntelIfConfig(
        model_id=MODEL_ID,
        lora_config=LoraConfig(),
    ),
    base_model_cfg=ModelConfig(),
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def construct_input_text(
    state: str, question: str, choices: list[str], anchor_token: str
):
    text = f"STATE:\n{state}\n"
    text += f"QUESTION:\n{question}\n"
    text += "CHOICES:\n"
    for choice in choices:
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


batch_df = df_4.sample(n=16)

CHOICES = sorted(top_4)


def construct_batch() -> dict[str, torch.Tensor]:
    input_ids = []
    choice_slots = []
    lengths = []
    targets = []

    max_len = 0

    for _, row in batch_df.iterrows():
        gold_label_index = CHOICES.index(row["label_text"])
        prompt_order = torch.randperm(n=len(CHOICES)).tolist()
        choice_order = torch.randperm(n=len(CHOICES)).tolist()

        prompt_order_target_index = prompt_order.index(gold_label_index)
        target_index = choice_order[prompt_order_target_index]

        text = construct_input_text(
            state=row["text"],
            question="What is the intent?",
            choices=[CHOICES[_] for _ in prompt_order],
            anchor_token=tokenizer.pad_token,
        )

        _input_ids, _choice_slots = construct_input_ids(
            text=text,
            anchor_token_id=tokenizer.pad_token_id,
            choice_order=choice_order,
        )

        input_ids.append(_input_ids)
        choice_slots.append(_choice_slots)
        lengths.append(len(_input_ids))
        targets.append(target_index)

        max_len = max(max_len, lengths[-1])

    bs = len(input_ids)

    input_ids_tensor = torch.full(size=(bs, max_len), fill_value=tokenizer.pad_token_id)
    choice_slots_tensor = torch.full(size=(bs, max_len), fill_value=-1)

    for i in range(bs):
        input_ids_tensor[i, : lengths[i]] = input_ids[i]
        choice_slots_tensor[i, : lengths[i]] = choice_slots[i]

    return {
        "input_ids": input_ids_tensor,
        "choice_slots": choice_slots_tensor,
        "lengths": torch.tensor(lengths),
        "targets": torch.tensor(targets),
    }


steps = 300
log_every = 10
rolling_window = 20
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

batch = construct_batch()
recent_losses = deque(maxlen=rolling_window)
recent_accuracies = deque(maxlen=rolling_window)

print(
    f"Training {len(lora_a_params)} LoRA adapters on a fixed batch with fresh anchors"
)

for step in range(1, steps + 1):
    optimizer.zero_grad()

    logits = model(
        input_ids=batch["input_ids"],
        lengths=batch["lengths"],
        choice_slots=batch["choice_slots"],
        num_choices=len(CHOICES),
    )

    loss = torch.nn.functional.cross_entropy(logits, target=batch["targets"])

    loss.backward()

    choice_to_model_grad_norm = model.choice_to_model.weight.grad.norm().item()
    model_to_choice_grad_norm = model.model_to_choice.weight.grad.norm().item()
    lora_a_grad_norm = gradient_norm(lora_a_params)
    lora_b_grad_norm = gradient_norm(lora_b_params)

    total_grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable_params, max_norm=max_grad_norm, error_if_nonfinite=True
    ).item()

    optimizer.step()

    accuracy = (logits.argmax(dim=-1) == batch["targets"]).float().mean().item()
    recent_losses.append(loss.item())
    recent_accuracies.append(accuracy)

    if step == 1 or step % log_every == 0:
        print(
            f"[step={step}/{steps}] "
            f"loss: {loss.item():.4f}, accuracy: {accuracy:.4f}, "
            f"rolling_loss: {sum(recent_losses) / len(recent_losses):.4f}, "
            f"rolling_accuracy: {sum(recent_accuracies) / len(recent_accuracies):.4f}, "
            f"grad_norms (before clipping): "
            f"choice_to_model={choice_to_model_grad_norm:.4f}, "
            f"model_to_choice={model_to_choice_grad_norm:.4f}, "
            f"lora_A={lora_a_grad_norm:.4f}, "
            f"lora_B={lora_b_grad_norm:.4f}, "
            f"total={total_grad_norm:.4f}"
        )
