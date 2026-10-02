---
id: llme-distillation
title: "Knowledge Distillation"
sidebar_label: "5 · Knowledge distillation"
sidebar_position: 5
slug: /llm-engineering/knowledge-distillation
description: "Train a small student to copy a large teacher: soft targets and temperature, the T-squared correction, sequence-level and on-policy distillation for language models, what a student cannot inherit, and the terms-of-use traps."
tags: [distillation, knowledge-distillation, temperature, kl-divergence, on-policy, student-teacher, llm]
---

import Infographic from '@site/src/components/Infographic';
import DistillationTemperatureLab from '@site/src/components/viz/DistillationTemperatureLab';

**In one line.** Distillation trains a small student on the full probability distribution of a large teacher, not just its final answer, so the student inherits how the teacher weighs the wrong options as well as the right one.

:::note Not from a lecture
This chapter is written for this site from the sources under Further reading, mainly Hinton, Vinyals and Dean (2015), the sequence-level and on-policy papers, and the Gemma 2 and Qwen3 technical reports. Every number below is printed by the code in this chapter, except the figures that name their source.
:::

## The idea in plain words

A model trained on hard labels sees only "this image is a car". The teacher that already knows cars says something richer: 96% car, 3% truck, 1% bus, almost nothing carrot. The tiny probabilities carry the lesson. A car is far more like a truck than a carrot, and a student that copies those proportions learns that similarity structure for free. The small probabilities of the wrong answers are where that knowledge hides.

The catch is that a confident teacher squeezes those small numbers towards zero, so they barely move the loss. **Temperature** fixes this. Divide the logits by $T$ before the softmax and the distribution flattens: the wrong answers grow large enough to matter. Train the student against the flattened teacher, at the same temperature.

<Infographic src="/img/llme/knowledge-distillation-temperature.svg" alt="Four bar charts of a five-class teacher at temperatures 1, 2, 4 and 8, a table of student accuracy on digits with hard and soft targets, and two cards on the T-squared correction and the KL values." caption="Temperature and what soft targets buy. The bar values, gradient norms and KL figures are printed by block 1, the table by block 2." />

For language models the same idea works one token at a time: at every position the teacher gives a distribution over the whole vocabulary, which is a far richer signal than the single next token in the text. Three recipes follow, and they differ in whose text the student learns on.

<Infographic src="/img/llme/knowledge-distillation-recipes.svg" alt="Three lanes: word-level distillation on a fixed corpus, sequence-level distillation on teacher-generated text, and on-policy distillation on student samples graded by the teacher, with a table of KL after 60 steps." caption="Whose text does the student learn on? The KL table is printed by block 4, the layer-copy numbers by block 3." />

## How it works

### Soft targets and the loss

Let $z$ be the logits. The tempered softmax is $p_i = \exp(z_i / T) / \sum_j \exp(z_j / T)$. At $T=1$ it is the ordinary softmax. The distillation loss of Hinton et al. is a weighted average of two terms, a soft one against the teacher at temperature $T$ and a hard one against the true label at temperature 1:

$$\mathcal{L} = \alpha \, \mathrm{CE}\big(y, q_1\big) + (1 - \alpha)\, T^2 \, \mathrm{KL}\big(p_T \,\|\, q_T\big)$$

where $p_T$ is the teacher's tempered distribution and $q_T$ the student's. The factor $T^2$ is not decoration. The paper observes that the gradients from soft targets scale as $1/T^2$, so multiplying by $T^2$ keeps the soft and hard terms in the same proportion when you change the temperature. Block 1 measures it: the gradient norm falls by a factor of four between $T=2$ and $T=4$, and the norm times $T^2$ stays close to 1 (0.9646, 0.9683, 0.9168, 0.8799 for $T$ of 2, 4, 8, 20).

The paper also shows what a high temperature approaches. If the logits are zero-meaned, the gradient becomes proportional to $(z_i - v_i)/(N T^2)$, so distillation turns into plain **logit matching**, a squared error between student and teacher logits. Block 1 prints the gap between the true scaled gradient and that approximation: 0.4149 at $T=1$, 0.0657 at $T=20$ and 0.0138 at $T=100$. At lower temperatures the loss pays less attention to very negative logits, which are barely constrained when the teacher was trained and may simply be noise. That is why the best temperature in the paper's smallest student was a middling 2.5 to 4, not the highest.

### Language models: tokens, text and the student's own samples

For an autoregressive language model, word-level distillation means a KL at every token position between the teacher's next-token distribution and the student's. Gemma 2's report states the objective as minimising $-\sum_x P_T(x \mid x_c) \log P_S(x \mid x_c)$ over the vocabulary, where $x_c$ is the context. It needs the two models to share a tokenizer, since the distributions must line up index by index. When they do not, you fall back to training the student on text.

Three recipes differ in whose text the student sees.

| Recipe | Student trains on | Loss | Weakness |
| --- | --- | --- | --- |
| Word-level (supervised) | a fixed corpus, human or teacher written | KL to the teacher's token distribution | the student never sees its own mistakes |
| Sequence-level (Kim and Rush, 2016) | the teacher's decoded outputs (beam search in the paper) | ordinary cross-entropy on those outputs | throws away the teacher's probabilities, keeps one path |
| On-policy (GKD, Agarwal et al.) | sequences the student itself samples | KL, reverse KL or a generalised JSD, scored by the teacher | needs the teacher during training, and sampling costs time |

The on-policy argument is about a mismatch. A student trained on fixed text sees prefixes the teacher or a human wrote. At inference it must continue from prefixes it wrote itself, a distribution it was never trained on. The GKD paper's algorithm mixes the two with a student data fraction $\lambda$, and it allows other divergences than the forward KL, because a student too small to match the whole distribution may do better by concentrating on the teacher's main mode. TRL exposes these as `lmbda`, `beta` (0 for forward KL, 1 for reverse KL, in between a generalised JSD) and `seq_kd`.

## A real system that works this way

**Gemma 2** is the clearest large-scale case. Its report says the 2B and 9B models were trained with knowledge distillation instead of next-token prediction, using a large teacher on more than 50 times the compute-optimal number of tokens, so distillation stood in for data the team did not have. A controlled comparison in the report trains a 2B model on 500B tokens either from scratch or distilled from a 7B teacher, and finds distillation better. The 27B model in that release was trained from scratch.

**Qwen3** uses distillation for its smaller models. The technical report describes a "strong-to-weak" pipeline: an off-policy phase on teacher outputs in both thinking modes, then an on-policy phase in which the student generates and is fine-tuned to match the logits of Qwen3-32B or Qwen3-235B-A22B by minimising KL. On Qwen3-8B the report compares reinforcement learning with on-policy distillation from the same starting checkpoint and finds distillation better on AIME'24 (74.4 against 67.6) at about one tenth of the GPU hours. That is the report's own comparison on math and code queries, not a general law.

**DistilBERT** is the classic case: 40% fewer parameters than BERT, 60% faster, and 97% of its language-understanding performance according to its abstract, trained with a loss combining masked language modelling, distillation and a cosine loss on hidden states.

## What a student can and cannot inherit

| Inherits well | Does not inherit |
| --- | --- |
| The teacher's behaviour on the distribution you distilled on | Whatever exceeds the student's capacity |
| Style, format and the ranking of wrong answers | Behaviour on inputs unlike the transfer set |
| Some regularisation: the paper describes matching soft targets as a regulariser for the small net | The teacher's mistakes are inherited too, with the same confidence |

The honest ceiling is the teacher, and in practice the student lands below it. Hinton's paper reports that the 800-unit MNIST net trained plainly made 146 test errors, the large regularised teacher 67, and the distilled small net 74. Distillation recovered most of the gap and not all of it. A student also cannot repair a teacher that is wrong; it learns the wrong answers faster.

## Code you can run

All four blocks run on CPU in `.lecture-import/venv-llm` (Python 3.14, torch 2.14.1, transformers 5.18.0, datasets 5.0.1, scikit-learn 1.9.1), seeded. The language-model blocks download `SmolLM2-135M-Instruct` and the Dolly dataset from the Hugging Face Hub the first time they run. The Dolly responses (`databricks/databricks-dolly-15k`, CC BY-SA 3.0 per its dataset card) are only used as ordinary text to run the student on.

### 1. Temperature, the gradient and the T-squared correction

The teacher is five fixed logits for car, truck, bus, cat and carrot. The student is a deliberately confused set of logits that puts truck above car.

```python
import numpy as np
import torch
import torch.nn.functional as F

classes = ["car", "truck", "bus", "cat", "carrot"]
teacher_logits = torch.tensor([9.0, 5.5, 4.6, 1.0, -2.0])

print("teacher distribution at each temperature")
print("T      " + "  ".join(f"{c:>7s}" for c in classes) + "   entropy (nats)")
for T in (1, 2, 4, 8, 20):
    p = F.softmax(teacher_logits / T, dim=0)
    entropy = -(p * p.log()).sum().item()
    print(f"{T:<5d}  " + "  ".join(f"{v:7.4f}" for v in p.tolist()) + f"   {entropy:.3f}")

student_logits = torch.tensor([6.0, 6.2, 3.0, 2.5, 0.0])
print("\nsoft-target loss and the T squared correction")
print("T      KL(teacher || student)   gradient norm   gradient norm x T^2")
for T in (1, 2, 4, 8, 20):
    z = student_logits.clone().requires_grad_(True)
    p = F.softmax(teacher_logits / T, dim=0)
    log_q = F.log_softmax(z / T, dim=0)
    kl = F.kl_div(log_q, p, reduction="sum")
    kl.backward()
    g = z.grad.norm().item()
    print(f"{T:<5d}  {kl.item():22.5f}   {g:13.5f}   {g * T * T:19.4f}")

print("\nhigh temperature is close to matching logits")
v = teacher_logits - teacher_logits.mean()
z = student_logits - student_logits.mean()
N = len(classes)
for T in (1, 4, 20, 100):
    zz = student_logits.clone().requires_grad_(True)
    p = F.softmax(teacher_logits / T, dim=0)
    F.kl_div(F.log_softmax(zz / T, dim=0), p, reduction="sum").backward()
    exact = zz.grad * T * T
    approx = (z - v) / N
    print(f"T={T:<4d} max |T^2 x gradient - (z - v)/N| = {(exact - approx).abs().max().item():.4f}")
```

At $T=1$ the teacher says 0.9589 car and 0.0290 truck, so the student's confusion is punished by a KL of 0.66750 but the useful information about *how* it is wrong is buried in tiny numbers. At $T=4$ the teacher says 0.5131 car, 0.2139 truck, 0.1708 bus, and the same student scores 0.10767. The lab below is this exact example. Its defaults (T of 4) reproduce the KL 0.10767 and the scaled gradient norm 0.9683 printed above. Drag the truck logit and the mixed-loss slider to see how the hard label and the soft targets pull on the student.

<DistillationTemperatureLab />

### 2. A teacher and a student on digits

Hinton's own small experiment used MNIST. This block uses the 8x8 digits that ship with scikit-learn, so it runs in seconds. The teacher is a two-hidden-layer network with 256 units and 85,002 parameters, trained on only 300 labelled rows. The student has 8 units per layer and 682 parameters. Five seeds are averaged.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(4)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=600, random_state=0, stratify=y)
X_lab, X_unl, y_lab, _ = train_test_split(X_rest, y_rest, train_size=300, random_state=0, stratify=y_rest)
X_lab, y_lab = torch.tensor(X_lab), torch.tensor(y_lab)
X_unl = torch.tensor(X_unl)
X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)
print(f"labelled {len(X_lab)}, unlabelled transfer rows {len(X_unl)}, test {len(X_test)}")

def mlp(hidden, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 10))

def accuracy(model):
    with torch.no_grad():
        return (model(X_test).argmax(1) == y_test).float().mean().item()

def train(model, loss_fn, epochs=400, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model).backward()
        opt.step()
    return model

def hard_loss(model):
    return F.cross_entropy(model(X_lab), y_lab)

def soft_loss(teacher_logits_lab, T, alpha, X_transfer=None, teacher_logits_transfer=None):
    p_lab = F.softmax(teacher_logits_lab / T, dim=1)
    def fn(model):
        out = model(X_lab)
        loss = alpha * F.cross_entropy(out, y_lab)
        soft = F.kl_div(F.log_softmax(out / T, dim=1), p_lab, reduction="batchmean") * T * T
        loss = loss + (1 - alpha) * soft
        if X_transfer is not None:
            p_t = F.softmax(teacher_logits_transfer / T, dim=1)
            out_t = model(X_transfer)
            loss = loss + (1 - alpha) * F.kl_div(F.log_softmax(out_t / T, dim=1), p_t, reduction="batchmean") * T * T
        return loss
    return fn

seeds = range(5)
results = {"teacher (256 units)": [], "student, hard labels": [], "student, soft T=1": [],
           "student, soft T=4": [], "student, 0.3 hard + 0.7 soft T=4": [],
           "student, soft T=4 + 897 unlabelled rows": []}
for seed in seeds:
    teacher = train(mlp(256, seed), hard_loss, epochs=300)
    with torch.no_grad():
        t_lab = teacher(X_lab)
        t_unl = teacher(X_unl)
    results["teacher (256 units)"].append(accuracy(teacher))
    results["student, hard labels"].append(accuracy(train(mlp(8, seed + 100), hard_loss)))
    results["student, soft T=1"].append(accuracy(train(mlp(8, seed + 100), soft_loss(t_lab, 1, 0.0))))
    results["student, soft T=4"].append(accuracy(train(mlp(8, seed + 100), soft_loss(t_lab, 4, 0.0))))
    results["student, 0.3 hard + 0.7 soft T=4"].append(accuracy(train(mlp(8, seed + 100), soft_loss(t_lab, 4, 0.3))))
    results["student, soft T=4 + 897 unlabelled rows"].append(
        accuracy(train(mlp(8, seed + 100), soft_loss(t_lab, 4, 0.0, X_unl, t_unl))))

n_params = lambda m: sum(p.numel() for p in m.parameters())
print(f"teacher parameters {n_params(mlp(256, 0)):,}, student parameters {n_params(mlp(8, 0)):,}")
print("\nmodel                                        test accuracy, mean over 5 seeds (min to max)")
for name, vals in results.items():
    print(f"{name:44s} {np.mean(vals):.4f}  ({min(vals):.4f} to {max(vals):.4f})")
```

Three results matter. First, soft targets at $T=1$ change nothing (0.8787, identical to hard labels), because a confident teacher at temperature 1 is nearly one-hot and says nothing the label did not. Second, at $T=4$ the same student reaches 0.9213 from the same 300 rows: the teacher's wrong-answer structure is information the labels lack, and the student got it for free. Third, because soft targets need no labels, you can also feed the student rows the teacher can label for itself. Adding 897 unlabelled rows lifts it to 0.9347, against a teacher at 0.9557. The student never catches the teacher here, and it should not be expected to: it has 0.8% of the parameters.

The seed spread is large (hard labels range from 0.8517 to 0.9167 across seeds), so read the ordering, not the third decimal place.

### 3. Building a student by copying layers

DistilBERT initialised its student by taking one layer out of two from the teacher. The block below does the same idea on `SmolLM2-135M-Instruct`: the student keeps layers 0, 6, 12, 18, 24 and 29 of the teacher's 30, and trains with token-level KL at $T=2$ on 512 real Dolly responses, batch 16. A second student with the same shape but random weights shows what the copied layers are worth. Held-out KL is measured at $T=1$ on 64 other responses.

```python
import copy
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(name)
tok.padding_side = "right"
teacher = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

keep = [0, 6, 12, 18, 24, 29]

def make_student(from_teacher):
    if from_teacher:
        model = copy.deepcopy(teacher)
    else:
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), dtype=torch.float32)
    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in keep])
    model.config.num_hidden_layers = len(keep)
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = i
    return model

count = lambda m: sum(p.numel() for p in m.parameters())
print(f"teacher {count(teacher):,} parameters, {teacher.config.num_hidden_layers} layers")
student = make_student(True)
print(f"student {count(student):,} parameters, {student.config.num_hidden_layers} layers, copied from teacher layers {keep}")

data = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=0)
texts = [r["response"] for r in data if 60 < len(r["response"]) < 400][:600]
train_text, held_out = texts[:512], texts[512:576]

def encode(batch_text):
    enc = tok(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=32)
    return enc["input_ids"], enc["attention_mask"]

def distill_loss(model, ids, mask, T):
    with torch.no_grad():
        t = teacher(input_ids=ids, attention_mask=mask).logits
    s = model(input_ids=ids, attention_mask=mask).logits
    p = F.softmax(t / T, dim=-1)
    per_token = (p * (p.clamp_min(1e-12).log() - F.log_softmax(s / T, dim=-1))).sum(-1)
    m = mask.float()
    agree = ((s.argmax(-1) == t.argmax(-1)).float() * m).sum() / m.sum()
    return (per_token * m).sum() / m.sum() * T * T, agree

ids_ho, mask_ho = encode(held_out)

def evaluate(model):
    model.eval()
    with torch.no_grad():
        kl, agree = distill_loss(model, ids_ho, mask_ho, 1.0)
    model.train()
    return kl.item(), agree.item()

def run(model, steps=60, batch=16, lr=5e-5):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    for step in range(1, steps + 1):
        pick = torch.randint(0, len(train_text), (batch,), generator=g).tolist()
        ids, mask = encode([train_text[i] for i in pick])
        loss, _ = distill_loss(model, ids, mask, 2.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in (1, 20, 40, 60):
            print(f"  step {step:3d}: training loss (T=2, scaled by T^2) {loss.item():.3f}")
    return evaluate(model)

kl0, a0 = evaluate(student)
print(f"\nlayer-copied student before training: held-out KL {kl0:.3f}, top-1 agreement {a0:.3f}")
kl1, a1 = run(student)
print(f"layer-copied student after 60 steps: held-out KL {kl1:.3f}, top-1 agreement {a1:.3f}")

random_student = make_student(False)
rk0, ra0 = evaluate(random_student)
print(f"\nrandomly initialised student of the same shape before training: held-out KL {rk0:.3f}, top-1 agreement {ra0:.3f}")
rk1, ra1 = run(random_student)
print(f"random-init student after 60 steps: held-out KL {rk1:.3f}, top-1 agreement {ra1:.3f}")
```

The copied layers start at a held-out KL of 10.385 and end at 4.721 after 60 steps, with 12.0% top-1 agreement on the teacher's next token. The random student starts lower, at 8.267, only because an untrained network predicts something close to uniform, and ends higher, at 5.976 with 9.6% agreement, so the head start from the teacher's weights is real but the student still agrees with the teacher on about one token in eight. Sixty steps on 135M-parameter teachers is a demonstration of the mechanism and nothing like a trained student.

### 4. Three recipes, same student, same budget

Each regime starts from the same layer-copied student and takes 60 steps of batch 8. The student is then judged twice: by its KL against the teacher on the teacher's own continuations of 64 held-out prompts, and by its KL on continuations it samples itself.

```python
import copy
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(name)
tok.padding_side = "left"
teacher = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
keep = [0, 6, 12, 18, 24, 29]

def make_student():
    model = copy.deepcopy(teacher)
    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in keep])
    model.config.num_hidden_layers = len(keep)
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = i
    return model

data = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=0)
responses = [r["response"] for r in data if 60 < len(r["response"]) < 400][:600]
PREFIX, NEW = 6, 16

def prefixes(texts):
    ids = [tok(t, add_special_tokens=False)["input_ids"][:PREFIX] for t in texts]
    ids = [x for x in ids if len(x) == PREFIX]
    return torch.tensor(ids)

train_prefix = prefixes(responses[:400])
held_prefix = prefixes(responses[400:464])

def rollout(model, prompt_ids, sample, seed=0):
    torch.manual_seed(seed)
    options = {"do_sample": True, "temperature": 1.0, "top_k": 0} if sample else {"do_sample": False}
    with torch.no_grad():
        return model.generate(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), max_new_tokens=NEW,
                              min_new_tokens=NEW, pad_token_id=tok.eos_token_id, **options)

def continuation_kl(student, full_ids):
    with torch.no_grad():
        t = teacher(input_ids=full_ids).logits[:, PREFIX - 1:-1]
    s = student(input_ids=full_ids).logits[:, PREFIX - 1:-1]
    p = F.softmax(t, dim=-1)
    return (p * (p.clamp_min(1e-12).log() - F.log_softmax(s, dim=-1))).sum(-1).mean()

def evaluate(student):
    student.eval()
    teacher_roll = rollout(teacher, held_prefix, sample=False)
    student_roll = rollout(student, held_prefix, sample=True, seed=1)
    student_greedy = rollout(student, held_prefix, sample=False)
    with torch.no_grad():
        kl_t = continuation_kl(student, teacher_roll).item()
        kl_s = continuation_kl(student, student_roll).item()
    match = (student_greedy[:, PREFIX:] == teacher_roll[:, PREFIX:]).float().mean().item()
    student.train()
    return kl_t, kl_s, match

STEPS, BATCH, LR = 60, 8, 5e-5
teacher_texts = rollout(teacher, train_prefix, sample=False)
print(f"teacher generated {len(teacher_texts)} greedy continuations of {NEW} tokens")

def train(mode):
    torch.manual_seed(0)
    student = make_student()
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    for step in range(STEPS):
        pick = torch.randint(0, len(train_prefix), (BATCH,), generator=g)
        if mode == "off-policy (teacher logits on fixed text)":
            ids = tok([responses[int(i)] for i in pick], return_tensors="pt", padding=True, truncation=True,
                      max_length=PREFIX + NEW, padding_side="right")["input_ids"]
            ids = ids[:, :PREFIX + NEW]
            if ids.shape[1] < PREFIX + NEW:
                continue
            loss = continuation_kl(student, ids)
        elif mode == "sequence-level (teacher text, cross-entropy)":
            ids = teacher_texts[pick]
            logits = student(input_ids=ids).logits[:, PREFIX - 1:-1]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, PREFIX:].reshape(-1))
        else:
            student.eval()
            ids = rollout(student, train_prefix[pick], sample=True, seed=step)
            student.train()
            loss = continuation_kl(student, ids)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
    return student

base = make_student()
b = evaluate(base)
print(f"{'before training':45s} KL on teacher text {b[0]:.3f}  KL on own samples {b[1]:.3f}  greedy token match {b[2]:.3f}")
for mode in ("off-policy (teacher logits on fixed text)", "sequence-level (teacher text, cross-entropy)", "on-policy (student samples, teacher logits)"):
    r = evaluate(train(mode))
    print(f"{mode:45s} KL on teacher text {r[0]:.3f}  KL on own samples {r[1]:.3f}  greedy token match {r[2]:.3f}")
```

Each recipe wins on the ground it trained on. Word-level distillation on fixed text is best on the teacher's text (4.662). Sequence-level training, which learns from one decoded path and ignores the teacher's probabilities, is behind on both measures (5.013 and 3.131). On-policy distillation is best on the student's own samples (2.168 against 2.579), the quantity that matters when the student generates for users. This is one run, a few prompts and 60 steps, so it demonstrates why the GKD authors proposed on-policy training and does not reproduce their benchmark gains.

## Production snippets (not run here)

:::warning Not run in this environment
The snippet below was not executed. It follows the TRL `GKDTrainer` documentation, whose class lives under `trl.experimental` in the installed TRL 1.14.1 and prints an instability warning on import. The model names are the documentation's own example, not a recommendation.
:::

```python
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

train = Dataset.from_dict({"messages": [[
    {"role": "user", "content": "Hi, how are you?"},
    {"role": "assistant", "content": "I'm great thanks"},
]] * 100})

args = GKDConfig(output_dir="gkd-model", per_device_train_batch_size=1, lmbda=1.0, beta=0.5, seq_kd=False)
GKDTrainer(model=student, teacher_model=teacher, args=args, processing_class=tokenizer, train_dataset=train).train()
```

## Designing with it

1. **Pick the student first.** Distil into the smallest model that meets the latency budget, then ask whether any teacher can lift it far enough. A teacher cannot lift a student past its own capacity.
2. **Prefer shared tokenizers.** With the same vocabulary you can match full distributions. With different ones you are limited to sequence-level distillation on generated text.
3. **Shape the transfer set.** It is the distribution the student learns. Make it look like production prompts, and add unlabelled prompts: soft targets need no labels.
4. **Start with a mixed loss and a moderate temperature,** $T$ between 2 and 4, then tune.
5. **Evaluate on the student's own samples,** and compare against fine-tuning the student on the same data with hard labels.
6. **Check the terms of use before you generate data.** It is a legal question, not a technical one.

:::warning Terms of use can forbid distilling a model
Providers differ, and the wording changes. As I read them in this session: Anthropic's Commercial Terms (effective 17 June 2025) say the customer may not access the services to build a competing product or service, including to train competing AI models, except as expressly approved. A copy of OpenAI's Terms of Use dated 11 December 2024 forbids using Output to develop models that compete with OpenAI (openai.com itself refused an automated fetch, so check the live page). The Llama 3.1 Community Licence (version date 23 July 2024), in the copy I read, allows using outputs to improve another AI model but requires that a model you distribute carries "Llama" at the start of its name and "Built with Llama" be displayed. Open-weight licences differ again. This is not legal advice: read the current terms of the model you would use as a teacher, and keep a record of which teacher produced which data.
:::

**Failure modes to name**

- *Temperature too low:* soft targets are nearly one-hot and add nothing (block 2, $T=1$).
- *Distilling the teacher's mistakes:* the student copies errors with the same confidence.
- *Transfer set mismatch:* a student distilled on one kind of text degrades on another.
- *Judging only on teacher text:* the student's own samples are what users see.

## Where this stands in 2026

:::info Industry view

- **Distillation is used for small models at scale.** Gemma 2 (2024) trained its 2B and 9B models with it, and Qwen3 (2025) used a strong-to-weak pipeline for its lightweight models, both from their own technical reports.
- **On-policy distillation is the current direction for post-training small models.** Qwen3's report found it better than reinforcement learning from the same checkpoint at about a tenth of the GPU hours on its test set, and TRL ships a GKD trainer, still under its experimental namespace in version 1.14.1.
- **The legal side is as important as the technical side.** Provider terms differ on whether outputs may be used to train other models; read them for the exact model and date.
- **KL against the teacher is a training signal, not a result.** Judge the student on task performance, with the methods in [LLM evaluation methods](/docs/llm-evals/llm-eval-methods) course.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why is the soft-target loss multiplied by $T^2$?</summary>

The gradient of the tempered KL with respect to the student logits is proportional to $1/T^2$ at high temperature, so without the factor the soft term would fade as the temperature rises while the hard term stayed the same size. Multiplying by $T^2$ keeps the two terms in a steady ratio, which block 1 shows: the gradient norm times $T^2$ stays between 0.88 and 0.97 for $T$ from 2 to 20.<br /><em>Authored · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> In block 2, soft targets at T=1 gave exactly the hard-label accuracy but T=4 did better. Explain.</summary>

At $T=1$ the confident teacher is almost one-hot, so its distribution carries the same information as the label. At $T=4$ the probabilities of the wrong classes grow to a size the loss can see, and they encode which digits look alike. That similarity structure is not in the labels, and it is why the student reached 0.9213 against 0.8787.<br /><em>Authored · interpretation</em>

</details>

<details>
<summary><strong>Q3.</strong> Your student and teacher use different tokenizers. Which recipes are still open?</summary>

Token-level KL needs the two distributions to line up over the same vocabulary, so word-level and on-policy KL are not directly available. Sequence-level distillation still works because it needs only the teacher's decoded text, which the student tokenizes in its own way, though it discards the teacher's probabilities.<br /><em>Authored · applied</em>

</details>

<details>
<summary><strong>Q4.</strong> In block 4 the on-policy student has the lowest KL on its own samples but not on the teacher's text. Is that a contradiction?</summary>

No. Each recipe fits the distribution it trained on. The on-policy student trained on its own samples, so it is best there (2.168), while the word-level student trained on fixed text that resembles the teacher's, so it is best on the teacher's text (4.662). What you should care about depends on where the student will run: at inference it conditions on its own previous tokens.<br /><em>Authored · interpretation</em>

</details>

<details>
<summary><strong>Q5.</strong> A vendor offers a strong hosted model. You want to distil its answers into a 1B model for your product. What do you do before generating any data?</summary>

Read the vendor's current terms for the exact model, since several providers restrict using outputs to train competing models and some licences instead require attribution or naming. Record the teacher, version and date for each batch of generated data. If the terms forbid it, use a teacher whose licence permits it.<br /><em>Authored · applied</em>

</details>

## Further reading

- [Hinton, Vinyals and Dean, "Distilling the Knowledge in a Neural Network" (2015)](https://arxiv.org/abs/1503.02531): soft targets, temperature, the $T^2$ factor and the logit-matching limit.
- [Sanh et al., "DistilBERT" (2019)](https://arxiv.org/abs/1910.01108): distillation during pre-training, the triple loss and layer-copy initialisation.
- [Kim and Rush, "Sequence-Level Knowledge Distillation" (EMNLP 2016)](https://arxiv.org/abs/1606.07947): training the student on the teacher's decoded sequences.
- [Agarwal et al., "On-Policy Distillation of Language Models" (2023)](https://arxiv.org/abs/2306.13649): GKD, the student data fraction and the divergence choices.
- [Gemma Team, "Gemma 2: Improving Open Language Models at a Practical Size" (2024)](https://arxiv.org/abs/2408.00118): distillation for 2B and 9B pre-training.
- [Qwen Team, "Qwen3 Technical Report" (2025)](https://arxiv.org/abs/2505.09388): the strong-to-weak distillation pipeline and the on-policy against reinforcement learning comparison.
- [TRL GKD trainer documentation](https://huggingface.co/docs/trl/main/en/gkd_trainer): `lmbda`, `beta` and `seq_kd`.
- Terms read for the caution above: [Anthropic Commercial Terms](https://www.anthropic.com/legal/commercial-terms), [OpenAI Terms of Use](https://openai.com/policies/terms-of-use/), [Llama 3.1 Community Licence](https://www.llama.com/llama3_1/license/).
- Related chapters on this site: [parameter-efficient fine-tuning with LoRA](/docs/theory/dnn/parameter-efficient-fine-tuning-lora), [RLHF and instruction tuning](/docs/theory/dnn/rlhf-and-instruction-tuning), [synthetic data generation](/docs/llm-engineering/synthetic-data-generation).

## Check yourself

- I can explain why a confident teacher's one-hot-like output hides the information a student needs, and how temperature exposes it.
- I can write the mixed distillation loss with the $T^2$ factor and say why the factor is there.
- I can say what high-temperature distillation reduces to, and why a very high temperature is not always best.
- I can distinguish word-level, sequence-level and on-policy distillation by the text the student trains on.
- I can explain the train and inference mismatch that on-policy distillation addresses.
- I can state what a student cannot inherit, and why the teacher's mistakes transfer as well.
- I can name the terms-of-use questions to answer before generating teacher data.
