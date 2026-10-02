---
id: llme-preference-tuning
title: "Preference Tuning: DPO, ORPO and Friends"
sidebar_label: "Preference tuning (DPO, ORPO)"
sidebar_position: 4
slug: /llm-engineering/preference-tuning-dpo-orpo
description: "Teach a model which of two answers is better without a reward model or reinforcement learning: the DPO loss derived and implemented from scratch, the effect of beta, a real DPOTrainer run, and how ORPO, SimPO and KTO differ."
tags: [dpo, preference-tuning, orpo, simpo, kto, alignment, trl, rlhf]
---

import Infographic from '@site/src/components/Infographic';
import DpoLossLab from '@site/src/components/viz/DpoLossLab';

**In one line.** Preference tuning shows the model pairs of answers, one better and one worse, and nudges it to make the better one relatively more likely than before, measured against a frozen copy of itself.

:::note Not from a lecture
This chapter is written for this site from the DPO, ORPO, SimPO and KTO papers and the TRL documentation listed under Further reading. The reinforcement-learning route to the same goal is covered in [RLHF and instruction tuning](/docs/theory/dnn/rlhf-and-instruction-tuning) and [PPO, GRPO and LLM post-training](/docs/theory/drl/ppo-grpo-and-llm-post-training); this chapter does not repeat it. Versions used: TRL 1.14.1, PEFT 0.21.2, Transformers 5.18.0, PyTorch 2.14.1 (CPU).
:::

## The idea in plain words

Supervised fine-tuning shows a model one good answer per prompt and says "say this". That works when you can write the good answer. Often you cannot write it, but you can recognise it: given two replies, you know which one you prefer. Polite against curt, grounded against invented, strict JSON against chatty prose. Preference tuning learns from exactly that signal.

The older route, reinforcement learning from human feedback, first trains a separate **reward model** on the preference pairs and then runs a reinforcement-learning loop that samples answers from the model and pushes it towards high reward without drifting too far from where it started. It works and it is heavy: two extra models, sampling during training, and a loop that is easy to destabilise.

**Direct Preference Optimization** (DPO), from Rafailov and colleagues, removes the middle. The paper's observation is that the reward model and the policy are two views of the same thing: there is a closed-form relation between the optimal policy and the reward, so the preference data can be used to train the policy directly with a simple classification loss. No reward model, no sampling during training.

Here is the intuition without the algebra. Keep a frozen copy of the starting model, the **reference**. For a prompt with a chosen answer and a rejected answer, ask how much more likely each answer has become under the model being trained than under the reference. Call those two numbers the **log-ratios**. DPO rewards the model when the chosen answer's log-ratio is larger than the rejected one's. The reference is what stops the model from simply making everything it likes arbitrarily probable.

<Infographic src="/img/llme/preference-tuning-dpo-orpo-dpo-loss.svg" alt="The DPO loss worked on one pair: policy and reference log-probabilities, the two log-ratios, the implicit rewards, the loss and the effect of beta" caption="One pair through the loss, with the numbers printed by block 1. Policy equal to reference gives loss ln 2 = 0.6931." />

<Infographic src="/img/llme/preference-tuning-dpo-orpo-methods-compare.svg" alt="A table comparing DPO, SimPO, ORPO and KTO on whether they need a reference model, what data they need and their loss on the same toy pair" caption="Four objectives on one toy pair, printed by block 4. The losses are on different scales and are not a ranking of the methods." />

## How it works

### Preference data

A preference example is a prompt, a `chosen` answer and a `rejected` answer. TRL's documentation says the explicit-prompt form is recommended, and the trainer accepts both plain strings and conversational messages, applying the chat template for you. Where the pairs come from decides everything downstream: human comparisons, answers from two models ranked by a judge, or programmatic corruptions of a good answer. A pair is only informative if the two answers differ in the way you care about and in no other way. If the chosen answers are also systematically longer, the model learns to be long.

A cheaper input exists for KTO, covered below: one answer per example with a single desirable or undesirable label (TRL calls it unpaired preference data).

### The loss, from the objective

The RLHF objective asks for a policy that earns high reward while staying close to the reference, with beta controlling how close. The DPO paper's key step is that the optimal policy for that objective has a closed form, so the reward can be written in terms of the policy: it is beta times the log-ratio of the policy to the reference, plus a term that depends only on the prompt. Put that into the Bradley-Terry model of preferences, in which the probability that the chosen answer wins is the sigmoid of the reward difference, and the prompt-only term cancels. What is left is the loss TRL's documentation writes:

$$\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\!\left(\beta\left[\log\frac{\pi_\theta(y^{+}\mid x)}{\pi_{\mathrm{ref}}(y^{+}\mid x)} - \log\frac{\pi_\theta(y^{-}\mid x)}{\pi_{\mathrm{ref}}(y^{-}\mid x)}\right]\right)$$

Each piece has a name worth knowing, because the trainer logs them.

- **Implicit reward** of an answer: beta times its log-ratio. TRL logs the averages as `rewards/chosen` and `rewards/rejected`.
- **Margin**: chosen reward minus rejected reward, logged as `rewards/margins`.
- **Reward accuracy**: the share of pairs with a positive margin, logged as `rewards/accuracies`.

Block 1 computes all of it by hand for one pair and checks the gradient with autograd. The gradient has a clear reading: the update pushes the chosen log-probability up and the rejected one down by the same amount, scaled by beta times the weight sigmoid(-margin). A pair the model already ranks correctly with a big margin has a small weight and barely moves the model; a pair it ranks wrongly has a weight near one. At the start of training the policy equals the reference, the margin is zero and the loss is ln 2, which is 0.6931. If your first logged loss is not that number, something is mis-wired.

### What beta does

Beta scales the log-ratios before the sigmoid, so it sets how far the policy must move before the loss is satisfied. TRL's configuration describes it as the deviation from the reference: a higher beta means less deviation. Block 1 shows the same pair under five betas. To push the loss down to 0.1 the log-ratio gap must reach 225.2 at beta 0.01, 22.5 at 0.1 and 4.5 at 0.5. A small beta lets the policy wander far from the reference before the loss notices; a large one saturates early and stays near it. TRL's default is 0.1.

### Implementing it on a real model

Block 2 writes the loop from scratch on `SmolLM2-135M-Instruct` with a LoRA adapter as the policy. The reference needs no second copy of the model: with the adapter switched off, the model is exactly the reference, so `disable_adapter()` supplies the reference log-probabilities. The first step's loss is 0.6931 as predicted. Over 16 steps the loss falls to 0.2185 and the margin grows to 3.492.

Look at the last two columns, which are the log-ratios of the chosen and rejected answers. By step 16 the chosen answer is 13.653 nats more likely than under the reference and the rejected one is 21.268 nats less likely. Both moved, the rejected more. TRL's documentation says the same of real runs: in practice DPO is typically achieved by suppressing the likelihood of dispreferred completions rather than by raising the likelihood of preferred ones.

Then the unflattering part. After those 16 steps the three sample generations in block 2 are still not JSON. The tuned model has learned that the JSON answer is relatively better than the alternatives, but greedy decoding still starts with something else. This is why preference tuning follows supervised fine-tuning in the standard recipe: SFT installs the behaviour, preference tuning refines which of several acceptable behaviours is better. The SmolLM2 model card describes the same order, supervised fine-tuning then DPO.

### A real `DPOTrainer` run

Block 3 does it the standard way, on top of an SFT stage. It runs 40 steps of supervised LoRA tuning on the ticket router, merges the adapter into the model, and then trains with TRL's `DPOTrainer` and a new LoRA adapter. Chosen is the correct queue as JSON and rejected is a wrong queue as JSON, so the pairs differ only in the label. With a `peft_config` and no `ref_model`, the trainer uses the starting model as the reference, as its documentation states.

What it shows, in order: the logged training metrics, a held-out evaluation of the implicit rewards on 23 pairs the model never saw, and the held-out tickets scored directly. Read the three results separately.

1. **The training metrics move as the theory says.** Loss falls from 0.6519 to 0.3931, the margin rises, and reward accuracy reaches 0.9844. Note `rewards/chosen` stays near zero and `rewards/rejected` falls to -0.7720: the same suppression pattern as block 2.
2. **The held-out preference accuracy rises from 0 to 0.738.** The starting 0.000 is not a failure: the margin is exactly zero when policy equals reference, and reward accuracy counts only strictly positive margins. The held-out margin is 0.4576, so the preference generalised to unseen tickets.
3. **The held-out right-queue score by answer scoring goes from 17 to 19 of 23.** The scoring picks the queue whose JSON answer the model finds most likely. Two more tickets is a small, one-seed change on 23 cases. The earlier dry run of this same experiment with 32 steps and different pair sampling gave 17 and 17. Treat the result as showing that the mechanism works and the effect on task accuracy is small and noisy here.

That last point is the honest summary of preference tuning on a toy: the training signal is easy to fit and the loss and margin look great, while the quality you care about moves by little. Always evaluate on the task, not on the DPO metrics.

### Neighbours of DPO

DPO is the reference point; the others change what the loss needs. All four losses below are computed on the same toy pair in block 4, and the numbers are on different scales, so they show the mechanics and not which method is better.

| Method | Reference model | Data | Idea |
| --- | --- | --- | --- |
| DPO | yes | chosen and rejected | beta times the log-ratio difference through a sigmoid |
| SimPO | no | chosen and rejected | reward is the average log-probability per token, scaled by beta, with a target margin gamma |
| ORPO | no | chosen and rejected | the ordinary SFT loss plus lambda times a log odds-ratio penalty on the disfavoured answer |
| KTO | yes | one answer with a desirable or undesirable label | utility of each answer from prospect theory, around an estimated KL reference point |

The formulas, as the papers and TRL write them:

- SimPO: the reward is `beta / |y|` times the log-probability of the answer, and the loss is `-log sigmoid(beta / |y_w| * log p(y_w) - beta / |y_l| * log p(y_l) - gamma)`. The authors emphasise that it needs no reference model, and that the target margin gamma asks the winner to beat the loser by at least that much. They report gains of up to 6.4 points on AlpacaEval 2 and 7.5 on Arena-Hard over DPO; these are the authors' benchmark claims.
- ORPO: odds are `P / (1 - P)` with the length-normalised likelihood; the odds-ratio loss is `-log sigmoid(log(odds(y_w) / odds(y_l)))`; the objective adds it to the SFT loss with weight lambda. The authors report results from 125M to 7B parameters. In TRL, lambda is the `beta` argument, default 0.1.
- KTO: the loss is `w(y) * (1 - v(x, y))` with `v` a sigmoid of beta times the log-ratio minus an estimated KL term for desirable answers, and with the sign reversed for undesirable ones. TRL estimates the KL term from mismatched pairs inside the batch, which is why it requires a per-step batch larger than one, and recommends a per-step batch of at least 4.

Two practical notes from the documentation in this TRL release. DPO is a stable trainer, and TRL's DPO configuration lists a `sigmoid_norm` loss which it describes as the SimPO authors' length-normalised variant of the sigmoid loss. ORPO has moved to `trl.experimental.orpo`: it is available but marked experimental, which matters if you pin versions. KTO is a stable trainer and accepts unpaired data, or paired data which it splits into labelled answers.

### Verifiable rewards instead of preferences

When the answer can be checked by a program (the JSON parses, the code passes its tests, the sum is right) you do not need preference pairs at all: you can use the check as a reward in an online method such as GRPO. That route is covered in [PPO, GRPO and LLM post-training](/docs/theory/drl/ppo-grpo-and-llm-post-training). One result relevant to cost: the LoRA study cited in the previous chapter reports that reinforcement learning needs very little adapter capacity, around one bit per episode by its estimate, so a small-rank LoRA can be enough there even where it is not for supervised learning.

## A real system that works this way

**SmolLM2-135M-Instruct,** the model throughout this series, is built with exactly this pipeline. Its model card says that post-training was supervised fine-tuning followed by DPO on UltraFeedback, and its supervised data set is public. Every step in this chapter and the previous one is a miniature of the model's own recipe, run on a task small enough for a laptop.

**TRL itself** is the other real system: the DPO trainer in this chapter is the reference implementation that the documentation describes, with a dozen loss variants selectable by `loss_type` (sigmoid by default, then hinge, IPO, `sigmoid_norm` and others), logging the same reward metrics as the code in block 2 computes by hand.

## Code you can run

Four blocks. Block 1 takes a second. Block 2 takes about a minute. Block 3 trains twice and takes about a minute and a half, which is over the target for a single block; it is the real `DPOTrainer` run and cannot be shortened without losing the SFT stage. Block 4 takes a second.

### 1. The DPO loss by hand, with the gradient checked

Toy log-probabilities, written down so you can change them.

```python
import math

import torch
import torch.nn.functional as F

policy_chosen, policy_rejected = -12.0, -20.5
reference_chosen, reference_rejected = -14.0, -19.0
beta = 0.1

ratio_chosen = policy_chosen - reference_chosen
ratio_rejected = policy_rejected - reference_rejected
reward_chosen, reward_rejected = beta * ratio_chosen, beta * ratio_rejected
margin = reward_chosen - reward_rejected
loss = -math.log(1 / (1 + math.exp(-margin)))
weight = 1 / (1 + math.exp(margin))
print(f"log-ratio chosen {ratio_chosen:+.1f}, rejected {ratio_rejected:+.1f}")
print(f"implicit rewards: chosen {reward_chosen:+.3f}, rejected {reward_rejected:+.3f}, margin {margin:.3f}")
print(f"loss {loss:.4f}   gradient weight sigmoid(-margin) {weight:.4f}")

pc = torch.tensor(policy_chosen, requires_grad=True)
pr = torch.tensor(policy_rejected, requires_grad=True)
auto = -F.logsigmoid(beta * ((pc - reference_chosen) - (pr - reference_rejected)))
auto.backward()
print(f"autograd loss {auto.item():.4f}; d loss / d chosen log-prob {pc.grad.item():+.4f}, rejected {pr.grad.item():+.4f}")
print(f"closed form -beta * weight = {-beta * weight:+.4f}, +beta * weight = {beta * weight:+.4f}")

at_start = -math.log(1 / (1 + math.exp(-0.0)))
print(f"\npolicy equal to reference: margin 0, loss {at_start:.4f} (ln 2 = {math.log(2):.4f})")

print("\nsame log-ratios (+2.0 and -1.5), different beta:")
print(f"{'beta':>5} {'margin':>7} {'loss':>7} {'grad weight':>12} {'gap needed for loss 0.1':>25}")
target = -math.log(math.exp(0.1) - 1)
for b in (0.01, 0.05, 0.1, 0.5, 1.0):
    m = b * (ratio_chosen - ratio_rejected)
    print(f"{b:>5} {m:>7.3f} {-math.log(1 / (1 + math.exp(-m))):>7.4f} {1 / (1 + math.exp(m)):>12.4f} {target / b:>25.1f}")
```

What it prints:

```text
log-ratio chosen +2.0, rejected -1.5
implicit rewards: chosen +0.200, rejected -0.150, margin 0.350
loss 0.5334   gradient weight sigmoid(-margin) 0.4134
autograd loss 0.5334; d loss / d chosen log-prob -0.0413, rejected +0.0413
closed form -beta * weight = -0.0413, +beta * weight = +0.0413

policy equal to reference: margin 0, loss 0.6931 (ln 2 = 0.6931)

same log-ratios (+2.0 and -1.5), different beta:
 beta  margin    loss  grad weight   gap needed for loss 0.1
 0.01   0.035  0.6758       0.4913                     225.2
 0.05   0.175  0.6095       0.4564                      45.0
  0.1   0.350  0.5334       0.4134                      22.5
  0.5   1.750  0.1602       0.1480                       4.5
  1.0   3.500  0.0298       0.0293                       2.3
```

The implicit rewards of +0.200 and -0.150 give a margin of 0.350 and a loss of 0.5334. Autograd agrees with the closed form: the chosen log-probability's gradient is -0.0413 and the rejected one's is +0.0413, which is beta times the weight 0.4134. At the start of training the loss is 0.6931. The last table is the beta story in numbers.

### Try it: the DPO loss and beta

The lab's defaults are block 1's numbers: log-ratios +2.0 and -1.5 at beta 0.1, so rewards +0.200 and -0.150, margin 0.350, loss 0.5334 and gradient weight 0.4134. Set both log-ratios to zero and the loss is 0.6931. Move beta to 0.5 and the same log-ratios give 0.1602. The curves show the loss and the gradient weight against the gap between the two log-ratios.

<DpoLossLab />

### 2. DPO from scratch on a real model

```python
import json
import os
import random

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)

QUEUES = ["billing", "technical", "account"]
TICKETS = [
    ("Hi, I was charged twice for my subscription this month.", "billing"),
    ("Hello, the app crashes every time I open the dashboard.", "technical"),
    ("Hey, I forgot my password and the reset email never arrives.", "account"),
    ("Please send me a receipt for last month's payment.", "billing"),
    ("I get a 500 error when I upload a file.", "technical"),
    ("Please delete my account and all my data.", "account"),
    ("The card on file was declined but my bank says it is fine.", "billing"),
    ("Pages take more than a minute to load.", "technical"),
]
CHATTY = [
    "Thanks for reaching out! This sounds like a {q} question, and I would be happy to help with that.",
    "Sure, I can help. I think this ticket belongs in the {q} queue.",
]
rng = random.Random(0)
pairs = []
for i in range(64):
    ticket, queue = TICKETS[i % len(TICKETS)]
    chosen = json.dumps({"queue": queue})
    if i % 2 == 0:
        rejected = rng.choice(CHATTY).format(q=queue)
    else:
        rejected = json.dumps({"queue": rng.choice([q for q in QUEUES if q != queue])})
    pairs.append((ticket, chosen, rejected))

model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
policy = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM"))


def sequence_logp(m, ticket, completion):
    prompt = [{"role": "user", "content": ticket}]
    p = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    f = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": completion}], tokenize=True, return_dict=True)["input_ids"]
    x = torch.tensor([f])
    logp = torch.log_softmax(m(input_ids=x).logits[0, :-1], -1).gather(-1, x[0, 1:, None])[:, 0]
    return logp[len(p) - 1:].sum()


beta, batch_size = 0.1, 8
optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=2e-4)
policy.train()
print(f"{'step':>4} {'loss':>7} {'reward acc':>11} {'margin':>7} {'chosen ratio':>13} {'rejected ratio':>15}")
for step in range(1, 17):
    batch = [pairs[(step * batch_size + k) % len(pairs)] for k in range(batch_size)]
    optimizer.zero_grad()
    stats = []
    for ticket, chosen, rejected in batch:
        pc, pr = sequence_logp(policy, ticket, chosen), sequence_logp(policy, ticket, rejected)
        with torch.no_grad(), policy.disable_adapter():
            rc, rr = sequence_logp(policy, ticket, chosen), sequence_logp(policy, ticket, rejected)
        margin = beta * ((pc - rc) - (pr - rr))
        (-F.logsigmoid(margin) / batch_size).backward()
        stats.append((-F.logsigmoid(margin).item(), float(margin.item() > 0), margin.item(), (pc - rc).item(), (pr - rr).item()))
    optimizer.step()
    if step in (1, 2, 4, 8, 12, 16):
        mean = [sum(s[i] for s in stats) / batch_size for i in range(5)]
        print(f"{step:>4} {mean[0]:>7.4f} {mean[1]:>11.3f} {mean[2]:>7.3f} {mean[3]:>13.3f} {mean[4]:>15.3f}")

policy.eval()
print("\ngenerations after 16 steps:")
for ticket, queue in TICKETS[:3]:
    ids = tokenizer.apply_chat_template([{"role": "user", "content": ticket}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = policy.generate(**ids, max_new_tokens=14, do_sample=False)
    print(f"  {ticket[:52]:<52} -> {tokenizer.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()!r}")
```

What it prints:

```text
step    loss  reward acc  margin  chosen ratio  rejected ratio
   1  0.6931       0.000   0.000         0.000           0.000
   2  0.6101       1.000   0.181         0.998          -0.812
   4  0.4922       0.875   0.531         3.065          -2.248
   8  0.3316       0.875   1.577         7.842          -7.927
  12  0.2839       0.875   2.351        11.834         -11.681
  16  0.2185       0.875   3.492        13.653         -21.268

generations after 16 steps:
  Hi, I was charged twice for my subscription this mon -> '"Hey there, welcome to Hugging Face. I\'m here to'
  Hello, the app crashes every time I open the dashboa -> '"Hang on, it\'s likely due to a busy day or'
  Hey, I forgot my password and the reset email never  -> '"Hey there, welcome to Hugging Face! I\'m here to'
```

### 3. SFT, then a real `DPOTrainer` run

```python
import json
import os
import random

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import PrinterCallback
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

torch.manual_seed(0)
name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)

ISSUES = {
    "billing": ["I was charged twice for my subscription this month", "my invoice shows an amount I do not recognise", "I need a refund for the annual plan I bought by mistake", "the card on file was declined but my bank says it is fine", "please send me a receipt for last month's payment", "I was billed after I cancelled", "the price on my invoice is higher than the one I was quoted", "I want to change the credit card used for payments"],
    "technical": ["the app crashes every time I open the dashboard", "the export button does nothing when I click it", "I get a 500 error when I upload a file", "pages take more than a minute to load", "the mobile app freezes on the login screen", "the API returns an empty list for my project", "notifications stopped arriving on my phone", "the report is missing data from last week"],
    "account": ["I forgot my password and the reset email never arrives", "I need to change the email address on my account", "please delete my account and all my data", "I cannot enable two-factor authentication", "my teammate needs access to our workspace", "my account was locked after too many login attempts", "I want to change the name shown on my profile", "how do I transfer ownership of the workspace to someone else"],
}
OPENERS = ["Hi,", "Hello,", "Hi team,", "Good morning,", "", "Hey,"]
CLOSERS = ["Thanks.", "Please help.", "Thank you in advance.", "", "This is urgent.", "Regards, Sam"]
QUEUES = list(ISSUES)

rng = random.Random(0)
train, test, seen = [], [], set()
for queue, issues in ISSUES.items():
    for k, issue in enumerate(issues):
        for _ in range(8 if k < 6 else 4):
            text = " ".join(p for p in [rng.choice(OPENERS), issue[0].upper() + issue[1:] + ".", rng.choice(CLOSERS)] if p)
            if k < 6:
                train.append({"ticket": text, "queue": queue})
            elif text not in seen:
                seen.add(text)
                test.append({"ticket": text, "queue": queue})
rng.shuffle(train)


def prompt(t):
    return [{"role": "user", "content": t}]


def answer(q):
    return [{"role": "assistant", "content": json.dumps({"queue": q})}]


def pick_queue_by_score(model, rows):
    model.eval()
    right = 0
    for r in rows:
        p = tokenizer.apply_chat_template(prompt(r["ticket"]), add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
        scores = {}
        for q in QUEUES:
            f = tokenizer.apply_chat_template(prompt(r["ticket"]) + answer(q), tokenize=True, return_dict=True)["input_ids"]
            x = torch.tensor([f])
            with torch.no_grad():
                logp = torch.log_softmax(model(input_ids=x).logits[0, :-1], -1).gather(-1, x[0, 1:, None])[:, 0]
            scores[q] = logp[len(p) - 1:].sum().item()
        right += max(scores, key=scores.get) == r["queue"]
    return right


sft_data = Dataset.from_list([{"prompt": prompt(r["ticket"]), "completion": answer(r["queue"])} for r in train])
sft = SFTTrainer(
    model=AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32),
    args=SFTConfig(output_dir="sft-stage", per_device_train_batch_size=8, max_steps=40, learning_rate=5e-4, warmup_steps=3, report_to="none",
                   use_cpu=True, max_length=128, seed=0, save_strategy="no", gradient_checkpointing=False, bf16=False, disable_tqdm=True, logging_steps=40),
    train_dataset=sft_data,
    processing_class=tokenizer,
    peft_config=LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM"),
)
sft.remove_callback(PrinterCallback)
sft.train()
sft_model = sft.model.merge_and_unload().eval()
print(f"after SFT: right queue by answer score {pick_queue_by_score(sft_model, test)} of {len(test)} held-out tickets")


def pairs_for(rows):
    out = []
    for r in rows:
        wrong = rng.choice([q for q in QUEUES if q != r["queue"]])
        out.append({"prompt": prompt(r["ticket"]), "chosen": answer(r["queue"]), "rejected": answer(wrong)})
    return Dataset.from_list(out)


trainer = DPOTrainer(
    model=sft_model,
    args=DPOConfig(output_dir="dpo-stage", per_device_train_batch_size=8, max_steps=24, learning_rate=2e-4, beta=0.1, warmup_steps=2, report_to="none",
                   use_cpu=True, max_length=128, seed=0, save_strategy="no", gradient_checkpointing=False, bf16=False, disable_tqdm=True, logging_steps=8),
    train_dataset=pairs_for(train[:96]),
    eval_dataset=pairs_for(test),
    processing_class=tokenizer,
    peft_config=LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.remove_callback(PrinterCallback)
before = trainer.evaluate()
trainer.train()
after = trainer.evaluate()

print(f"\n{'step':>4} {'loss':>7} {'rewards/chosen':>15} {'rewards/rejected':>17} {'rewards/margins':>16} {'rewards/accuracies':>19}")
for h in trainer.state.log_history:
    if "loss" in h and "rewards/margins" in h:
        print(f"{int(h['step']):>4} {h['loss']:>7.4f} {h['rewards/chosen']:>15.4f} {h['rewards/rejected']:>17.4f} {h['rewards/margins']:>16.4f} {h['rewards/accuracies']:>19.4f}")
print(f"\nheld-out pairs, implicit reward accuracy: before {before['eval_rewards/accuracies']:.3f}, after {after['eval_rewards/accuracies']:.3f}")
print(f"held-out pairs, implicit reward margin:   before {before['eval_rewards/margins']:.4f}, after {after['eval_rewards/margins']:.4f}")
print(f"after DPO: right queue by answer score {pick_queue_by_score(trainer.model, test)} of {len(test)} held-out tickets")
```

What it prints:

```text
after SFT: right queue by answer score 17 of 23 held-out tickets

step    loss  rewards/chosen  rewards/rejected  rewards/margins  rewards/accuracies
   8  0.6519          0.0009           -0.0872           0.0882              0.6562
  16  0.5135          0.0012           -0.4195           0.4207              0.9844
  24  0.3931         -0.0048           -0.7720           0.7671              0.9844

held-out pairs, implicit reward accuracy: before 0.000, after 0.738
held-out pairs, implicit reward margin:   before 0.0000, after 0.4576
after DPO: right queue by answer score 19 of 23 held-out tickets
```

### 4. DPO, SimPO, ORPO and KTO on the same toy pair

```python
import math

beta, gamma, lam = 0.1, 0.5, 0.1
chosen_logp, rejected_logp = -12.0, -20.5
chosen_ref, rejected_ref = -14.0, -19.0
chosen_len, rejected_len = 8, 10


def log_sigmoid(x):
    return -math.log(1 + math.exp(-x))


dpo = -log_sigmoid(beta * ((chosen_logp - chosen_ref) - (rejected_logp - rejected_ref)))

simpo_beta = 2.0
simpo = -log_sigmoid(simpo_beta * chosen_logp / chosen_len - simpo_beta * rejected_logp / rejected_len - gamma)

avg_c, avg_r = chosen_logp / chosen_len, rejected_logp / rejected_len
odds = lambda avg: math.exp(avg) / (1 - math.exp(avg))
l_or = -log_sigmoid(math.log(odds(avg_c) / odds(avg_r)))
l_sft = -avg_c
orpo = l_sft + lam * l_or


def kto_loss(logratio, desirable, kl, beta=0.1):
    inner = beta * (logratio - kl) if desirable else beta * (kl - logratio)
    return 1 - 1 / (1 + math.exp(-inner))


kto_good = kto_loss(chosen_logp - chosen_ref, True, 0.0)
kto_bad = kto_loss(rejected_logp - rejected_ref, False, 0.0)

print(f"{'method':<8}{'reference model':<17}{'data':<26}{'loss on the toy pair':>22}")
print(f"{'DPO':<8}{'yes':<17}{'chosen and rejected':<26}{dpo:>22.4f}")
print(f"{'SimPO':<8}{'no':<17}{'chosen and rejected':<26}{simpo:>22.4f}")
print(f"{'ORPO':<8}{'no':<17}{'chosen and rejected':<26}{orpo:>22.4f}")
print(f"{'KTO':<8}{'yes':<17}{'one label per answer':<26}{kto_good:>11.4f} and {kto_bad:.4f}")
print(f"\nORPO parts: SFT term {l_sft:.4f}, odds-ratio term {l_or:.4f} times lambda {lam}")
print(f"average log-probabilities: chosen {avg_c:.3f}, rejected {avg_r:.3f}")
print("toy setting: chosen 8 tokens, rejected 10 tokens; beta, gamma and lambda are illustrative")
```

What it prints:

```text
method  reference model  data                        loss on the toy pair
DPO     yes              chosen and rejected                       0.5334
SimPO   no               chosen and rejected                       0.4375
ORPO    no               chosen and rejected                       1.5415
KTO     yes              one label per answer           0.4502 and 0.4626

ORPO parts: SFT term 1.5000, odds-ratio term 0.4150 times lambda 0.1
average log-probabilities: chosen -1.500, rejected -2.050
toy setting: chosen 8 tokens, rejected 10 tokens; beta, gamma and lambda are illustrative
```

SimPO and ORPO take no reference model, so the toy pair needs only the policy's own log-probabilities and the token counts. ORPO's loss of 1.5415 is mostly its SFT term (1.5000): it is a different quantity from the others and should not be compared with them.

## Production snippets (not run here)

At realistic scale a DPO run uses a preference dataset such as the one in TRL's quick start and a GPU. The dataset name and the adapter-learning-rate guidance come from the TRL DPO documentation. Not run in this environment.

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

trainer = DPOTrainer(
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
    args=DPOConfig(
        output_dir="dpo-router",
        beta=0.1,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        bf16=True,
        gradient_checkpointing=True,
        max_length=1024,
    ),
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.train()
```

Not run in this environment.

## Designing with it

1. **Do SFT first.** Block 2 shows a model whose chosen answer became far more likely and whose greedy output still did not change.
2. **Make pairs differ in one way.** If the chosen answers are also longer, politer or more confident, the model learns that too.
3. **Log the first loss.** It should be 0.6931. If not, the reference and policy are not the same at step zero.
4. **Watch both rewards, not only the margin.** In both runs here `rewards/chosen` stayed near zero while `rewards/rejected` fell. A margin can grow because the rejected answers become unlikely while the chosen ones do not improve.
5. **Hold out pairs and a task metric.** Reward accuracy on training pairs reaches 0.98 quickly; the task score moved by two tickets of 23.
6. **Keep the learning rate low and beta at its default first.** TRL's DPO default is 1e-6 and its guide suggests about 1e-5 for adapters; this chapter used 2e-4 on a tiny dataset for a few steps and that is not a recipe.
7. **Pick the method by the data you have.** Pairs and a good SFT model: DPO. Only thumbs up or down: KTO. No spare memory for a reference model: SimPO or ORPO.

## Where this stands in 2026

:::info Industry view
- **DPO is the documented default for offline preference tuning.** TRL lists it as the trainer for preference data, supports adapters directly, and logs the reward metrics used here; a dozen loss variants hang off the same trainer.
- **The family keeps growing.** TRL's DPO page lists loss types from IPO and hinge to SimPO's normalised variant and several more. The KTO paper was last revised in September 2026, which is a sign that the area is still moving.
- **Status differs by method in the library.** In this TRL release ORPO lives under `trl.experimental` while DPO and KTO are stable trainers. Check the release you pin.
- **Benchmark claims are the authors' own.** SimPO's gains over DPO and ORPO's AlpacaEval figures are from the respective abstracts and were measured on particular models and benchmarks; I did not reproduce them.
- **Verifiable rewards have taken over where they apply.** For tasks with a checkable answer, online methods with programmatic rewards are the current alternative to preference pairs; see the DRL chapter.
- **I could not verify** any claim about which labs use which method for their production models; none is made here.
:::

## Practice questions

<details>
<summary><strong>Q1.</strong> The first logged DPO loss in your run is 0.52. What is likely wrong and how would you check?</summary>

At step zero the policy equals the reference, so the margin is zero and the loss must be ln 2, 0.6931. A lower value means the policy already differs from the reference, for example a reference loaded from a different checkpoint, an adapter that was already trained, or dropout active on one side. Check that both log-probabilities come from the same weights with the adapter off and dropout disabled.<br /><em>Authored · applied</em>

</details>

<details>
<summary><strong>Q2.</strong> In block 1 the pair has log-ratios +2.0 and -1.5. Compute the margin and the loss at beta 0.5.</summary>

The gap is 3.5, the margin is 0.5 x 3.5 = 1.75, and the loss is -log sigmoid(1.75) = 0.1602. The gradient weight is sigmoid(-1.75) = 0.1480. A higher beta satisfies the same pair with a much smaller loss and gradient, so the policy stays closer to the reference.<br /><em>Authored · applied</em>

</details>

<details>
<summary><strong>Q3.</strong> Block 2's margin reaches 3.492 and reward accuracy 0.875, yet generations are still not JSON. Why, and what should come first?</summary>

DPO optimises the relative likelihood of the chosen answer against the rejected one, measured against the reference. The chosen answer became 13.653 nats more likely, but it started so unlikely that other openings still win under greedy decoding. Supervised fine-tuning should come first to install the behaviour; preference tuning then refines it.<br /><em>Authored · interpretation</em>

</details>

<details>
<summary><strong>Q4.</strong> Block 3 reports held-out reward accuracy rising from 0.000 to 0.738. Why is 0.000 the correct starting value?</summary>

Reward accuracy is the share of pairs with a strictly positive margin. With policy equal to reference every margin is exactly zero, so none counts. The starting value says nothing about quality, only that the model has not yet moved.<br /><em>Authored · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Training margin and reward accuracy look excellent but the task score barely moved. List two possible reasons.</summary>

The pairs may differ in a way that does not match the task: here a wrong label as the only difference, on templated sentences, which a small model can separate without learning to classify better. And the optimiser can raise the margin mainly by lowering the rejected answers' likelihood, which does not require improving the chosen ones. Evaluate on the task and look at both rewards.<br /><em>Authored · interpretation</em>

</details>

<details>
<summary><strong>Q6.</strong> Your team has user ratings, thumbs up or thumbs down on single answers, and no paired comparisons. Which method fits and what must you watch?</summary>

KTO takes one answer per example with a desirable or undesirable label. Watch the balance between the two labels (TRL provides desirable and undesirable weights for it), use a per-step batch of at least 4 because the KL term is estimated from other examples in the batch, and keep the learning rate in the range TRL recommends.<br /><em>Authored · applied</em>

</details>

## Further reading

- [Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"](https://arxiv.org/abs/2305.18290): the method and its derivation.
- [Hugging Face TRL: DPO trainer](https://huggingface.co/docs/trl/dpo_trainer): loss types, `beta`, logged metrics, adapter guidance.
- [Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model"](https://arxiv.org/abs/2403.07691) and its [TRL page](https://huggingface.co/docs/trl/orpo_trainer).
- [Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward"](https://arxiv.org/abs/2405.14734).
- [Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization"](https://arxiv.org/abs/2402.01306) and its [TRL page](https://huggingface.co/docs/trl/kto_trainer).
- [Hugging Face TRL: dataset formats and types](https://huggingface.co/docs/trl/dataset_formats): preference and unpaired preference data.
- [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct): the SFT then DPO recipe of the model used here.
- Theory: [RLHF and instruction tuning](/docs/theory/dnn/rlhf-and-instruction-tuning), [PPO, GRPO and LLM post-training](/docs/theory/drl/ppo-grpo-and-llm-post-training). Previous: [supervised fine-tuning with LoRA](/docs/llm-engineering/supervised-fine-tuning-with-lora).

## Check yourself

- I can explain why DPO needs no reward model and what the frozen reference is for.
- I can compute the implicit rewards, margin, loss and gradient weight for a preference pair, and say what the loss is at step zero.
- I can say what beta controls and how the needed log-ratio gap changes with it.
- I can implement the DPO step on a real model with a LoRA adapter, getting reference log-probabilities by switching the adapter off.
- I can run `DPOTrainer`, read `rewards/chosen`, `rewards/rejected`, `rewards/margins` and `rewards/accuracies`, and explain why reward accuracy starts at zero.
- I can say what ORPO, SimPO and KTO change, and name the data and reference model each needs.
- I can say why preference tuning follows SFT and why I must evaluate on the task and not on the DPO metrics.
