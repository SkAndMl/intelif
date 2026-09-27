# Intelif — Project Brief

> Working name: **Intelif** ("intelligent if")  
> Goal: build a small open decision model that returns probability distributions over arbitrary choices supplied at inference time.

## 1. Project objective

The project is inspired by TypeSafe's **Jev / System One** API design, but it is not intended to reproduce Jev's private model architecture or training recipe. The public Jev interface is useful because it treats AI as a **decision primitive for software**, not as a text generator.

Our initial research question is narrower:

> Can a small pretrained language model be post-trained to make accurate decisions over **dynamic, previously unseen choice sets**, using a Headless-AD-style output representation instead of a fixed classifier or ordinary LM-token logits?

The first implementation should optimize for fast experimentation and falsifiability rather than completeness. Start with **SmolLM2-360M-Instruct** and a controlled intent-classification setup. Do not build the game demo, model router, or full API until the core variable-choice experiment works.

---

## 2. Jev: product/API inspiration

### 2.1 Mental model

Jev exposes a compact contract:

```text
shared state + typed questions -> typed probabilistic answers
```

The caller sends some JSON-compatible **state** and one or more questions about that same state. The questions can be evaluated together, which is useful when several independent decisions depend on the same input.

The public API centers on three primitives:

1. **Choice** — choose one option from a caller-provided set.
2. **Noul** — answer a yes/no proposition as a probability of true.
3. **Score** — place the state on an ordered scale and return a distribution / expected score.

The important design principle is that the caller supplies the decision space. Application code owns the policy around the result: thresholds, routing, fallbacks, side effects, and human review.

### 2.2 Example public API shape

Conceptually:

```python
response = client.system_one(
    state={
        "ticket": "I was charged twice and need the duplicate refunded today.",
        "account_tier": "business",
    },
    questions={
        "intent": Choice(
            instructions="What is the customer's main request?",
            criteria={
                "refund": "The customer wants money returned.",
                "technical_help": "The customer needs a bug or integration fixed.",
                "information": "The customer is asking for information only.",
                "other": "None of the other options clearly fits.",
            },
        ),
        "urgent": Noul(
            instructions="Does the ticket explicitly communicate time pressure?"
        ),
        "frustration": Score(
            instructions="How frustrated does the customer appear?",
            criteria=[
                "Calm and neutral",
                "Concerned but civil",
                "Very angry or using strong language",
            ],
        ),
    },
)
```

A `Choice` result includes the chosen label and a probability distribution over the supplied options. A `Noul` result is a probability in `[0, 1]`. A `Score` result is based on an ordered set of levels and can return a fractional expected score.

### 2.3 API principles we want to copy

We do **not** need to clone Jev's endpoint names or SDK implementation exactly. We do want to preserve these product ideas:

- **State is separate from the question.**
- **Choice sets are supplied at inference time.**
- **Return distributions, not only argmax labels.**
- **Multiple questions can share one state.**
- **The model makes judgments; application code makes policy decisions.**
- The API should feel like a semantic/probabilistic `if` statement rather than a chat completion.

### 2.4 Initial Intelif API scope

For v0, implement only **Choice**. `Noul` and `Score` can come later.

Target high-level interface:

```python
result = intelif.choice(
    state="Why did I get charged twice for the same card transaction?",
    question="What best describes the user's intent?",
    choices={
        "duplicate_card_payment": "A card transaction was charged more than once.",
        "cash_withdrawal_fee": "The user was charged a fee for withdrawing cash.",
        "cash_withdrawal_unrecognized": "The user does not recognize an ATM withdrawal.",
    },
)

result.choice
result.probabilities
```

The implementation underneath this API is the research focus described below.

---

## 3. Headless-AD: method we are adapting

Source paper: **In-Context Reinforcement Learning for Variable Action Spaces** (Sinii et al., ICML 2024), which proposes **Headless-AD**.

### 3.1 Why ordinary classifiers are a problem

A standard classifier / Algorithm Distillation model produces:

```text
hidden state -> fixed linear head -> logits over fixed action IDs
```

If the model is trained with `N` output classes, the final layer has `N` outputs. This ties the architecture to a specific action-set size and associates each output dimension with a fixed action meaning.

Headless-AD removes that dependency. The paper reports that simply permuting action semantics hurts standard AD, and increasing the action-set size cannot be handled without changing the output layer and retraining.

### 3.2 Headless-AD components

The paper introduces four relevant ideas:

1. **Remove the fixed output classifier.**  
   The model predicts a vector in action-embedding space instead of logits tied to action IDs.

2. **Random action embeddings.**  
   At each training step, each available action is assigned a fresh random embedding. The mapping is consistent within the context but changes across training steps. This prevents the network from attaching permanent semantics to one action vector.

3. **Action-set prompt.**  
   The embeddings for all currently available actions are included in the model input so the model knows what choices are valid.

4. **Contrastive / InfoNCE objective.**  
   The predicted action vector should be similar to the correct action embedding and dissimilar to the other available action embeddings.

The paper further constrains action embeddings to be **unit length and mutually orthogonal**. This reduces interference between options. Exact orthogonality requires action-embedding dimension `d_action >= number_of_choices`.

### 3.3 Core objective

Let:

- `h` = final transformer hidden state at the decision position;
- `W_out` = learned projection from model hidden size to action-embedding size;
- `z = W_out h` = predicted action vector;
- `E = [e_1, ..., e_K]` = current action embeddings;
- `tau` = temperature.

Then:

```text
logit_i = dot(z, e_i) / tau
```

and the loss for gold choice `y` is:

```text
L = -log exp(logit_y) / sum_i exp(logit_i)
```

This is ordinary cross-entropy over dynamically constructed logits and is equivalent to the InfoNCE objective in this setup.

### 3.4 Important distinction from the paper

We are **not** reproducing the full Algorithm Distillation / in-context RL setting.

Original Headless-AD learns action semantics from interaction histories containing observations, actions, and rewards. Our first experiment is a supervised semantic-decision task. Each candidate action already has a natural-language description.

We are borrowing the **output parameterization and invariance mechanism**:

```text
random action anchors
+ action-set prompt
+ direct prediction in action space
+ contrastive loss
```

and adapting it to:

```text
state + question + candidate descriptions -> decision over arbitrary candidates
```

Do not describe v0 as a reproduction of Headless-AD. It is a Headless-AD-inspired adaptation for language decision tasks.

---

## 4. v0 model: SmolLM2-360M-Instruct

Use:

```text
HuggingFaceTB/SmolLM2-360M-Instruct
```

Relevant architecture facts from the official config:

- architecture: `LlamaForCausalLM`
- hidden size: `960`
- layers: `32`
- attention heads: `15`
- KV heads: `5`
- max position embeddings: `8192`
- Apache-2.0 license

### 4.1 Do not use the LM head for the decision

The pretrained LM head can remain physically present in the checkpoint, but the decision path should bypass it.

Conceptually:

```text
SmolLM2 transformer
      |
      v
final hidden state at DECISION position: h in R^960
      |
      v
learned decision projection W_out
      |
      v
predicted action vector z in R^d_action
      |
      v
similarity against current action embeddings
      |
      v
probabilities over current choices
```

Initial default:

```text
d_action = 256
```

This supports up to 256 exactly orthogonal candidate anchors, which is enough for the planned v0 tests.

### 4.2 Binding natural-language choices to random action anchors

This is the main adaptation we need to implement carefully.

For each batch step, generate an orthonormal matrix:

```text
E in R^(K_max x d_action)
```

where each row `e_i` is a random unit action anchor. Use the same positional anchors across the batch for that step, as in Headless-AD, but **randomly permute which semantic choice occupies each position for every training example**. Regenerate `E` on the next optimizer step.

Each candidate should be represented by both:

1. its natural-language description; and
2. its current random action anchor.

Recommended v0 representation: project the action anchor into model hidden space and inject it as a continuous pseudo-token next to its choice description.

```text
W_in: R^256 -> R^960
```

Conceptual sequence:

```text
STATE
<state text>

QUESTION
<question text>

CHOICES
[action_anchor_0] <description of choice 0>
[action_anchor_1] <description of choice 1>
...
[action_anchor_K-1] <description of choice K-1>

DECISION
```

`[action_anchor_i]` is **not** a vocabulary token. Construct it by projecting `e_i` through `W_in` and inserting that vector through `inputs_embeds`.

At the `DECISION` position, obtain `h`, compute `z = W_out h`, and score all candidate anchors using dot products.

### 4.3 Training parameters

Train at least:

- `W_in`
- `W_out`
- LoRA adapters on SmolLM2
- optionally a learned decision marker embedding if implementation requires one

Keep the original LM head unused.

Start with LoRA rather than full fine-tuning. The goal of v0 is to validate the method cheaply, not maximize the final model.

Do not prematurely optimize LoRA target modules, quantization, kernels, or serving. First make the loss decrease and the controlled generalization tests work.

---

## 5. Initial verification dataset

The dataset should be designed to answer one specific question:

> Does the model learn to select among arbitrary semantic choices, including choices whose classes were never seen during training, rather than memorizing a fixed classifier?

### 5.1 Training sources

#### BANKING77

Use as the main fine-grained training source.

- 13,083 English customer-service queries
- 77 banking intents
- fine-grained single-domain labels
- CC-BY-4.0

Examples include closely related classes such as card-payment issues, cash-withdrawal issues, pending transactions, fees, refunds, and identity verification. These are useful because they naturally produce hard negatives.

#### MASSIVE — English only

Use to broaden the semantic domain beyond banking.

- 60 intents
- 18 domains
- 19,521 utterances per language in MASSIVE 1.1
- multilingual source dataset; v0 should use **English only**
- CC-BY-4.0

### 5.2 Held-out evaluation source

#### CLINC150

Do **not** train on CLINC150 in the first experiment.

Use it as an unseen-taxonomy evaluation set. The key purpose is to test whether the trained model can receive a completely new set of intent labels/descriptions at inference time and select among them without parameter updates.

### 5.3 Additional within-domain held-out-label split

CLINC150 changes both the taxonomy and the input distribution. We therefore also need a controlled unseen-label test within BANKING77.

Before training:

1. deterministically choose **20 of the 77 BANKING77 intents** using a fixed seed;
2. persist the held-out intent list in the repository;
3. remove every example whose gold label belongs to those 20 intents from training;
4. evaluate those intents separately.

This gives three distinct evaluation regimes:

```text
A. seen BANKING77 / MASSIVE intents
B. unseen BANKING77 intents, same domain
C. unseen CLINC150 taxonomy, new domain/taxonomy
```

Do not silently change the held-out split once experiments begin.

---

## 6. Dynamic training-example construction

Do not materialize ordinary fixed-class classification examples such as:

```text
text -> global class ID 42
```

The model must never rely on a permanent output index.

### 6.1 Canonical stored record

Store source data in a semantic form, e.g.:

```json
{
  "state": "Why did I get charged twice for the same card transaction?",
  "question": "What best describes the user's intent?",
  "gold_label": "transaction charged twice",
  "source": "banking77"
}
```

Candidate sets should be created dynamically by the data collator / sampler.

### 6.2 Candidate-set sampling

For every presentation of an example:

1. include the gold label;
2. sample `K - 1` negatives;
3. randomize the order;
4. use the resulting **local index only for that example**;
5. regenerate / reshuffle on later presentations.

Initial training distribution:

```text
K sampled from 2..16
```

A useful starting mixture is approximately:

```text
70% hard negatives
30% random negatives
```

Do not treat this ratio as sacred; it is a starting point.

### 6.3 Hard negatives

Random negatives alone will make the task too easy.

Hard negatives should be semantically similar labels. Example for:

```text
transaction charged twice
```

Good negatives include:

```text
card payment not recognized
card payment fee charged
card payment reversed
pending card payment
cash withdrawal not recognized
```

For v0, build a per-label nearest-neighbor table using embeddings of the **label text / description only**. A lightweight sentence-embedding model is fine for mining negatives. This model is not a teacher for the gold answer; it is only used to choose difficult distractors.

### 6.4 Label descriptions

For the very first pass, normalize dataset intent names into readable text:

```text
transaction_charged_twice -> "transaction charged twice"
card_payment_not_recognised -> "card payment not recognised"
```

Do not generate elaborate LLM-written descriptions until the simple setup is working. Later, we can add richer descriptions and paraphrases to reduce dependence on dataset label naming style.

---

## 7. Batch-level action-anchor generation

For a batch whose largest candidate set is `K_max`:

1. sample a random matrix `R in R^(d_action x K_max)`;
2. obtain an orthonormal basis with QR decomposition;
3. use the first `K_max` orthonormal vectors as action anchors;
4. share these positional anchors across the batch for this optimizer step;
5. for each sample, randomly permute semantic candidates onto anchor positions;
6. regenerate the anchors at the next training step.

Example pseudocode:

```python
R = torch.randn(d_action, k_max, device=device)
Q, _ = torch.linalg.qr(R, mode="reduced")
E = Q.T  # [k_max, d_action], unit and mutually orthogonal
```

For sample `b` with `K_b < K_max`, only the first `K_b` anchors are valid. Mask the rest before softmax.

The semantic-to-anchor assignment must be random. Otherwise the model can learn that anchor position `i` corresponds to a persistent intent.

---

## 8. Forward pass and loss

Pseudo-flow:

```python
# E: [K, d_action]
# W_in: d_action -> hidden_size
# W_out: hidden_size -> d_action

anchor_tokens = W_in(E)  # [K, hidden_size]

inputs_embeds = build_sequence(
    state_text,
    question_text,
    [(anchor_tokens[i], choice_text[i]) for i in range(K)],
    decision_marker,
)

hidden = model(inputs_embeds=inputs_embeds).last_hidden_state
h = hidden[:, decision_position, :]           # [B, 960]
z = W_out(h)                                  # [B, d_action]

logits = einsum("bd,bkd->bk", z, E_batch) / temperature
logits = mask_invalid_choices(logits)

loss = cross_entropy(logits, gold_local_index)
```

This cross-entropy is the InfoNCE loss because the gold action anchor is the positive and all other currently available anchors are negatives.

Initial implementation should prefer correctness and inspectability over clever batching.

---

## 9. Evaluation plan

A normal test-set accuracy number is insufficient. The entire point is variable-choice generalization.

### 9.1 Core metrics

Report at least:

- top-1 accuracy
- negative log likelihood
- Brier score
- expected calibration error (ECE)

Calibration metrics matter because the intended API returns probabilities, although we should not claim good calibration merely because the model is trained with softmax/InfoNCE.

### 9.2 Seen-class performance

Evaluate held-out examples whose intent was present during training.

This establishes whether the model can solve the basic semantic task at all.

### 9.3 Unseen BANKING77 labels

Evaluate examples from the 20 BANKING77 intents completely excluded from training.

The candidate descriptions are supplied at inference time. No gradient update is allowed.

This is the cleanest first test of new action semantics with minimal domain shift.

### 9.4 CLINC150 unseen taxonomy

Evaluate on CLINC150 without training on its intent classes.

This tests transfer to a new label taxonomy and a broader distribution shift.

### 9.5 Choice-count extrapolation

Train with:

```text
K <= 16
```

Then evaluate at:

```text
K = 2, 4, 8, 16, 32, 64, 150
```

Plot accuracy and NLL versus `K`.

The architecture should mechanically support these larger sets because `d_action = 256`; whether model quality survives is an empirical question.

### 9.6 Permutation invariance

For the same `(state, question, candidate set)`:

1. evaluate many random candidate permutations;
2. map predictions back to semantic labels;
3. measure how often the selected semantic label changes;
4. compare probability vectors after undoing the permutation.

The desired behavior is semantic invariance to candidate ordering.

### 9.7 Fresh-anchor invariance

Repeat the same example with several independently generated orthonormal action-anchor sets.

Predictions should remain stable after mapping back to semantic choices. This verifies that the model is not relying on properties of one specific random basis.

---

## 10. Required baselines

At minimum, compare against:

1. **Random choice** — sanity floor.
2. **SmolLM2-360M-Instruct generation** — prompt it as ordinary multiple choice and parse its answer.
3. **SmolLM2-360M token-logit readout** — SemIf-style baseline: present candidates as A/B/C/... and softmax only the LM-head logits for those option tokens.
4. **Our Headless-AD-inspired model** — random action anchors + direct action-vector prediction + InfoNCE.

A useful later ablation is the same LoRA-tuned backbone with token-logit readout. That isolates whether gains come from decision post-training generally or specifically from the headless action representation.

---

## 11. v0 success criteria

Do not define success as "training loss decreases."

The experiment is interesting if we see all of the following:

- good accuracy on seen intents;
- materially above-random performance on completely held-out BANKING77 intents;
- useful zero-shot performance on CLINC150's unseen taxonomy;
- low sensitivity to candidate permutation;
- low sensitivity to fresh random action-anchor bases;
- graceful degradation as the number of candidates grows beyond the training range;
- a measurable advantage over the frozen SmolLM2 token-logit baseline on at least some of the variable-choice generalization tests.

If the model only performs well on seen labels or collapses when anchors are regenerated, the method has not demonstrated the desired property.

---

## 12. Suggested repository milestones

### Milestone 0 — Baselines

- Load SmolLM2-360M-Instruct.
- Build normal multiple-choice generation baseline.
- Build token-logit / SemIf-style baseline.
- Build deterministic evaluation harness.

### Milestone 1 — Dataset pipeline

- Load BANKING77 and MASSIVE English.
- Persist the 20-intent BANKING77 holdout manifest.
- Normalize labels into readable descriptions.
- Implement dynamic candidate sampling.
- Implement hard-negative pools.
- Unit-test permutation and gold-label correctness.

### Milestone 2 — Headless model

- Implement orthonormal anchor generation.
- Implement `W_in` pseudo-token injection.
- Bypass LM head.
- Implement `W_out` decision projection.
- Implement masked dynamic logits + InfoNCE/CE.
- Verify gradients reach LoRA + `W_in` + `W_out`.

### Milestone 3 — Small overfit test

Before any real training:

- take a few hundred examples;
- confirm the model can overfit them;
- regenerate anchors while training;
- verify accuracy after changing the anchor basis and choice order.

Do not launch a long run before this works.

### Milestone 4 — Main v0 run

Train on:

```text
BANKING77 train intents (excluding 20 held-out intents)
+ MASSIVE English
```

with dynamic `K in [2, 16]`.

Evaluate all axes in Section 9.

### Milestone 5 — API shell

Only after the model is validated, expose:

```python
intelif.choice(...)
```

Return:

```python
ChoiceResult(
    choice: str,
    probabilities: dict[str, float],
)
```

Later extend the API with `noul` and `score`.

### Milestone 6 — demos

After the model/API are credible:

- dynamic game-action demo;
- LLM/model-router demo;
- public benchmark + ablations;
- release model weights, training code, dataset transforms, and evaluation harness.

---

## 13. Non-goals for v0

Do **not** spend initial time on:

- reproducing Jev's proprietary training method;
- recreating full RL / Algorithm Distillation trajectories;
- full fine-tuning a larger model;
- multilingual training;
- game environments;
- model routing datasets;
- optimizing serving latency;
- building `score` / `noul` before `choice` works;
- sophisticated calibration layers;
- a production SDK.

The first result should answer one technical question cleanly.

---

## 14. Implementation invariants

Coding agents should preserve these invariants unless an experiment explicitly changes them:

1. **No persistent global output class IDs.**
2. **Candidate order is randomized.**
3. **Action anchors are regenerated during training.**
4. **Gold labels are defined semantically, then converted to local indices only after candidate sampling/permutation.**
5. **LM head is not used by the headless model.**
6. **Only currently available choices appear in the contrastive denominator.**
7. **Invalid/padded choices are masked before softmax.**
8. **Train/test intent splits are persisted and reproducible.**
9. **CLINC150 is evaluation-only for the first experiment.**
10. **No result should be described as calibrated without measuring calibration.**

---

## 15. Open questions / expected ablations

Do not block v0 on these; record them for later.

- `d_action`: 64 vs 128 vs 256 vs larger.
- Whether to L2-normalize the predicted vector `z` before scoring.
- Temperature: fixed vs learned.
- LoRA target modules and rank.
- Anchor before vs after candidate-description text.
- One anchor pseudo-token vs multiple projected anchor tokens.
- Hard-negative ratio.
- Training `K` distribution.
- Rich natural-language criteria vs normalized label names.
- Frozen backbone + projections only vs LoRA.
- Headless representation vs post-trained token-logit representation.
- Calibration via held-out temperature scaling.

---

## 16. References

### Jev / TypeSafe

- TypeSafe official Python SDK: https://github.com/typesafe-ai/typesafe-sdk-python
- TypeSafe official JavaScript SDK: https://github.com/typesafe-ai/typesafe-sdk-js
- TypeSafe organization / System One projects: https://github.com/typesafe-ai

### Headless-AD

- Sinii et al., *In-Context Reinforcement Learning for Variable Action Spaces*, ICML 2024 / arXiv:2312.13327
- Reference implementation: https://github.com/corl-team/headless-ad

### Backbone

- SmolLM2-360M-Instruct: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct

### Datasets

- BANKING77: https://huggingface.co/datasets/PolyAI/banking77
- MASSIVE: https://huggingface.co/datasets/AmazonScience/massive
- CLINC150: https://huggingface.co/datasets/DeepPavlov/clinc150

### Baseline

- SemIf: https://github.com/TheoLeeCJ/SemIf

---

## 17. One-sentence summary for coding agents

**Build a SmolLM2-360M decision model that receives state + question + dynamically supplied semantic choices, binds each choice to a fresh random orthonormal action anchor, predicts the correct anchor from the final transformer hidden state, trains with InfoNCE, and is evaluated primarily on unseen labels, unseen taxonomies, candidate permutations, fresh anchor bases, and action-set sizes larger than those seen during training.**
