---
id: llme-sft-lora
title: "Supervised Fine-Tuning with LoRA"
sidebar_label: "SFT with LoRA"
sidebar_position: 3
slug: /llm-engineering/supervised-fine-tuning-with-lora
description: "Fine-tune a small chat model with TRL's SFTTrainer and a LoRA adapter on a laptop CPU, watch the loss fall, compare before and after, count the trainable parameters, merge the adapter and choose the main hyperparameters."
tags: [fine-tuning, sft, lora, peft, trl, qlora, adapters, hyperparameters]
---

import Infographic from '@site/src/components/Infographic';
import LoraRankLab from '@site/src/components/viz/LoraRankLab';

**In one line.** Supervised fine-tuning with LoRA freezes the model, trains two thin matrices beside each chosen weight, and so teaches a new behaviour by updating about two percent of the parameters, in a file small enough to email.

:::note Not from a lecture
This chapter is written for this site from the TRL, PEFT and Transformers documentation, the LoRA, QLoRA and follow-up papers, and the SmolLM2 model card listed under Further reading. The theory of why low-rank updates work is in the [LoRA and PEFT chapter](/docs/theory/dnn/parameter-efficient-fine-tuning-lora) and is not repeated. This chapter is the workflow, run for real. Versions used: TRL 1.14.1, PEFT 0.21.2, Transformers 5.18.0, PyTorch 2.14.1 (CPU).
:::

## The idea in plain words

Fine-tuning a model normally means changing all its weights. For a model with billions of parameters that needs the weights, a gradient for each, and two running statistics per weight for the Adam optimiser, which is why full fine-tuning needs many times the memory of the model itself. LoRA avoids almost all of that. The original weights stay frozen. Next to a chosen weight matrix you add two small matrices, A and B, whose product is a low-rank correction, and you train only those. The **rank** is how thin they are.

You pay for the saving in two ways that matter in practice. First, the correction can express only so much, so a low rank has less capacity to learn than a full update. Second, you now have an extra file, the **adapter**, and a choice at serving time: keep it separate or merge it into the base weights.

This chapter takes the support-ticket router of the previous two chapters and trains it. The base model, `SmolLM2-135M-Instruct`, given only a ticket, chats politely and never produces the JSON we need: 0 valid answers out of 23 held-out tickets. After 60 training steps on a CPU, with an adapter of 2.4 million trainable parameters, all 23 answers are valid JSON and 19 name the right queue. Those are tickets about issues the model never saw during training, so it is a test of generalising and not of remembering.

<Infographic src="/img/llme/supervised-fine-tuning-with-lora-lora-params.svg" alt="A frozen weight matrix with a trainable low-rank pair beside it, the SmolLM2 module shapes, and the trainable parameter counts for rank 8 on three target sets" caption="What gets trained: counts printed by block 2. Rank 8 on all linear layers is 2,442,240 parameters, 1.82 percent of the base." />

<Infographic src="/img/llme/supervised-fine-tuning-with-lora-before-after.svg" alt="Before and after tuning on 23 held-out tickets: valid JSON, right queue, prompt tokens, and the training loss by step" caption="The measured result of block 1: loss from 3.5810 to 0.0087, and 0 to 23 valid answers, with a prompt 4.8 times shorter than six worked examples." />

## How it works

### What the trainer does

`SFTTrainer` from TRL takes the model, an `SFTConfig`, a dataset and, for LoRA, a `peft_config`. Pass a `LoraConfig` and the trainer wraps the model for you. The dataset type decides the loss (see the previous chapter): with prompt-completion data the loss is computed on the completion only. The loss itself is the ordinary next-token cross-entropy, with the labels shifted by one position and masked positions ignored.

Block 1 shows the whole workflow in one run: build data, measure the base model, train, measure again. Three details of the configuration are worth knowing.

- **Float32 on CPU.** TRL loads the model in float32 unless you pass a dtype in `model_init_kwargs`; its documentation notes this differs from plain `from_pretrained`, which follows the checkpoint. On a CPU, float32 is the safe choice.
- **A higher learning rate than full fine-tuning.** TRL's guide says adapters typically use a learning rate near 1e-4 because only new parameters are learned. Block 1 uses 3e-4 for a very short run on a tiny dataset. A published study of LoRA found the best learning rate to be consistently about ten times the full fine-tuning rate, and about fifteen times for runs shorter than around 100 steps; this run is of that kind.
- **Gradient checkpointing off.** TRL turns it on by default to save memory on GPUs. At this size it only slows the CPU run.

### What the hyperparameters mean

| Parameter | What it controls | Default in PEFT | Practical guidance |
| --- | --- | --- | --- |
| `r` | rank of the update, hence capacity | 8 | start at 8 to 16; raise it for harder or larger data |
| `lora_alpha` | scale of the update: it is multiplied by `lora_alpha / r` | 8 | commonly set to r or 2r; a fixed ratio makes learning rates comparable across ranks |
| `target_modules` | which weights get an adapter | None, so set it explicitly | `"all-linear"` covers attention and MLP layers |
| `lora_dropout` | dropout on the adapter input | 0.0 | small values (0.05) can help on small data |
| `use_rslora` | scale by `lora_alpha / sqrt(r)` instead | off | helps at high ranks, where the plain ratio shrinks the update |

Two published findings shape the defaults. The "LoRA Without Regret" study (29 September 2025) reports that LoRA matches full fine-tuning when it is applied to all layers, especially the MLP layers, and when it is not capacity-constrained, meaning the trainable parameters exceed the information to be learned; attention-only LoRA underperformed even at matched parameter counts. The earlier "LoRA Learns Less and Forgets Less" study found that, at typical settings, LoRA substantially underperforms full fine-tuning on code and mathematics, while preserving the base model's behaviour outside the target domain better, and that full fine-tuning learns weight changes of a rank 10 to 100 times higher than typical LoRA settings. Both agree that LoRA is a capacity trade: you give up some ability to learn in return for forgetting less and costing much less.

For a behaviour this narrow (a fixed output format and three labels), capacity is not the constraint. For teaching a model a large body of new capability it can be.

### Counting what you train

Block 2 counts the trainable parameters for seven ranks and three target sets, using PEFT on the model skeleton, and checks every count against the formula `layers x rank x (in + out)` summed over the modules. The base has 134,515,008 parameters. Rank 8 on query and value projections only trains 460,800 of them, 0.34 percent. Rank 8 on all linear layers trains 2,442,240, 1.82 percent, which is the adapter used in block 1. Rank 64 on all linear layers trains 19,537,920, 14.52 percent: no longer a small adapter.

The memory arithmetic that follows is an estimate, not a measurement: it counts weights, gradients and Adam's two moments (4, 4 and 8 bytes per float32 parameter) and leaves out activations and framework overhead. On those terms full fine-tuning holds 2,152 MB of training state for this tiny model and LoRA holds the frozen base plus 39.1 MB, or 27 percent. With the frozen base in bfloat16, the base shrinks from 538 MB to 269 MB. The lab below does the same arithmetic interactively. The original paper reports the effect at a very different scale: for GPT-3 175B, a 10,000-fold reduction in trainable parameters against full fine-tuning with Adam and a 3-fold reduction in GPU memory, with no additional inference latency.

### The adapter, merging and serving

After training you have an adapter: here 9.8 MB against a 538 MB base. There are two ways to serve it. **Keep it separate**: load the base once and attach different adapters per task or customer, at the cost of a small extra computation on every layer. **Merge it**: `merge_and_unload()` folds the update into the base weights and returns a plain model with no adapter layers, with no extra inference cost, but you now store a full-size model per variant. PEFT also provides `merge_adapter()` and `unmerge_adapter()` to switch, and `add_weighted_adapter()` to combine several adapters. Note that `merge_and_unload()` is not in place: assign its return value.

Block 3 trains briefly, saves the adapter and the merged model, and checks that merging changed nothing it should not have: the merged model's logits differ from the adapter-on logits by about 2e-4, and both give the same answer.

### Evaluating before and after

The comparison to keep is the one in block 1: the same held-out set, the same decoding, before and after, plus the strongest prompt-only baseline you could have used instead. Here that is the six-example prompt from the first chapter. The tuned model reaches 19 of 23 correct queues with a 47-token prompt; the base model with six worked examples reaches 13 of 23 with a 225-token prompt. The tuned model is both better on this test and cheaper per request, which is the situation in which tuning pays.

Do not over-read it: it is one seed, a synthetic task, 23 tickets, and templated language. The training loss reaches 0.0087, a sign of memorising these sentence patterns; the held-out score is the evidence that something more than memorising happened.

### Hyperparameters, measured

Block 4 changes one thing at a time on the same data with 30 training steps, scoring the average loss on the answer tokens of the held-out tickets. Alpha is set to twice the rank so the scaling factor stays at 2.

| Rank | Modules | Trainable | Held-out answer loss |
| --- | --- | ---: | ---: |
| 8 | q, v | 460,800 | 2.8097 |
| 1 | all linear | 305,280 | 3.4644 |
| 8 | all linear | 2,442,240 | 0.5242 |
| 64 | all linear | 19,537,920 | 0.1132 |

Before any training the loss is 4.8644. Two readings, both for this toy only. Adding the MLP layers helped far more than going from rank 1 to rank 8 on attention alone: rank 8 on all linear layers has about five times the parameters of the query and value adapter and a loss of 0.5242 against 2.8097, which agrees in direction with the finding that all-layer LoRA is better. And a higher rank learned faster in 30 steps. With one seed and 30 steps this says nothing about where the curves end up after a long run.

### QLoRA

QLoRA applies the same adapter to a base model stored in 4 bits. Its paper introduced a 4-bit data type called NormalFloat (NF4), double quantisation of the quantisation constants, and paged optimisers to absorb memory spikes, and reports fine-tuning a 65-billion-parameter model on a single 48 GB GPU. The kernels need a GPU, so the snippet below is not run here.

## A real system that works this way

**Guanaco, from the QLoRA paper,** is the clearest published example of the recipe at scale. The authors report that their best model family reached 99.3 percent of ChatGPT's performance level on the benchmark they used, with 24 hours of fine-tuning on a single GPU, and that they applied the method to over a thousand models of several architectures and sizes. That is the authors' own claim about their own benchmark, and the models are from 2023; the useful part is the recipe (4-bit base, LoRA adapters, paged optimiser), not the leaderboard position.

**SmolLM2-135M-Instruct,** the model in this chapter, was itself produced by supervised fine-tuning followed by preference optimisation, according to its model card. Tuning it again for one narrow task, as here, is the ordinary use of the technique.

## Code you can run

Four runnable blocks and one snippet that is not run. Block 1 is the one that matters; it takes about a minute. Blocks 2, 3 and 4 take seconds, seconds and about a minute. First run downloads the model. PEFT prints a harmless warning about a missing config file when saving the adapter.

### 1. Train the router with `SFTTrainer` and LoRA, before and after

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
from trl import SFTConfig, SFTTrainer

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
print(f"{len(train)} training tickets from 18 issues, {len(test)} held-out tickets from 6 unseen issues")

SHOTS = [
    ("My invoice shows an amount I do not recognise.", "billing"),
    ("The mobile app freezes on the login screen.", "technical"),
    ("I cannot enable two-factor authentication.", "account"),
    ("I want to change the credit card used for payments.", "billing"),
    ("The API returns an empty list for my project.", "technical"),
    ("I want to change the name shown on my profile.", "account"),
]
SYSTEM = 'Classify the support ticket. Reply with JSON only, for example {"queue": "billing"}. Queues: billing, technical, account.'


def evaluate(model, shots):
    model.eval()
    prefix = []
    if shots:
        prefix = [{"role": "system", "content": SYSTEM}]
        for t, q in SHOTS[:shots]:
            prefix += [{"role": "user", "content": t}, {"role": "assistant", "content": json.dumps({"queue": q})}]
    valid = right = 0
    outputs = []
    for row in test:
        ids = tokenizer.apply_chat_template(prefix + [{"role": "user", "content": row["ticket"]}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=12, do_sample=False)
        text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        outputs.append(text)
        try:
            valid += 1
            right += json.loads(text)["queue"] == row["queue"]
        except (ValueError, KeyError, TypeError):
            valid -= 1
    return valid, right, outputs, ids["input_ids"].shape[1]


model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
for label, shots in (("base, ticket only", 0), ("base, 6 worked examples", 6)):
    valid, right, outputs, n_tokens = evaluate(model, shots)
    print(f"{label:<26} prompt {n_tokens:>3} tokens   valid JSON {valid:>2}/{len(test)}   right queue {right:>2}/{len(test)}")
before = evaluate(model, 0)[2][:3]

dataset = Dataset.from_list([
    {"prompt": [{"role": "user", "content": r["ticket"]}], "completion": [{"role": "assistant", "content": json.dumps({"queue": r["queue"]})}]}
    for r in train
])
config = SFTConfig(
    output_dir="sft-router", per_device_train_batch_size=8, max_steps=60, learning_rate=3e-4, lr_scheduler_type="linear",
    warmup_steps=4, logging_steps=10, report_to="none", use_cpu=True, max_length=128, seed=0, save_strategy="no",
    gradient_checkpointing=False, bf16=False, disable_tqdm=True,
)
lora = LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", lora_dropout=0.0, task_type="CAUSAL_LM")
trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, processing_class=tokenizer, peft_config=lora)
trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in trainer.model.parameters())
print(f"\ntrainable parameters {trainable:,} of {total:,} ({trainable / total:.2%})")
trainer.remove_callback(PrinterCallback)
trainer.train()
curve = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
print("training loss by step: " + ", ".join(f"{s} {l:.4f}" for s, l in curve))

valid, right, outputs, n_tokens = evaluate(trainer.model, 0)
print(f"\ntuned, ticket only         prompt {n_tokens:>3} tokens   valid JSON {valid:>2}/{len(test)}   right queue {right:>2}/{len(test)}")
for row, b, a in zip(test[:3], before, outputs[:3]):
    print(f"\nticket:  {row['ticket']}\nbefore:  {b!r}\nafter:   {a!r}   (true queue {row['queue']})")
```

What it prints:

```text
144 training tickets from 18 issues, 23 held-out tickets from 6 unseen issues
base, ticket only          prompt  47 tokens   valid JSON  0/23   right queue  0/23
base, 6 worked examples    prompt 225 tokens   valid JSON 23/23   right queue 13/23

trainable parameters 2,442,240 of 136,957,248 (1.78%)
training loss by step: 10 3.5810, 20 0.9011, 30 0.2312, 40 0.0399, 50 0.0203, 60 0.0087

tuned, ticket only         prompt  47 tokens   valid JSON 23/23   right queue 19/23

ticket:  Hello, The price on my invoice is higher than the one I was quoted. Please help.
before:  "Hello! I'm sorry to hear that your invoice is higher"
after:   '{"queue": "billing"}'   (true queue billing)

ticket:  Hi, The price on my invoice is higher than the one I was quoted. This is urgent.
before:  "I'm sorry to hear that your invoice is higher than the"
after:   '{"queue": "billing"}'   (true queue billing)

ticket:  Hey, The price on my invoice is higher than the one I was quoted. Thank you in advance.
before:  "Hey, I hope you're doing well! I'm here"
after:   '{"queue": "billing"}'   (true queue billing)
```

The base model never produces valid JSON from the bare ticket: 0 of 23. With six worked examples it produces valid JSON for all 23 but picks the right queue for 13. After 60 steps the loss has fallen from 3.5810 at step 10 to 0.0087 and the tuned model answers 23 of 23 in valid JSON and 19 of 23 correctly, from a 47-token prompt. The three before-and-after generations show the behavioural change: from a polite paragraph to the exact JSON requested.

### 2. Counting parameters and memory

```python
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

name = "HuggingFaceTB/SmolLM2-135M-Instruct"
config = AutoConfig.from_pretrained(name)
hidden, inter, layers = config.hidden_size, config.intermediate_size, config.num_hidden_layers
kv = config.num_key_value_heads * config.head_dim
shapes = {
    "q_proj": (hidden, hidden), "k_proj": (hidden, kv), "v_proj": (hidden, kv), "o_proj": (hidden, hidden),
    "gate_proj": (hidden, inter), "up_proj": (hidden, inter), "down_proj": (inter, hidden),
}
print(f"hidden {hidden}, intermediate {inter}, layers {layers}, key/value width {kv}")

TARGETS = {
    "q,v": ["q_proj", "v_proj"],
    "q,k,v,o": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all linear": list(shapes),
}


def formula(rank, modules):
    return layers * sum(rank * (shapes[m][0] + shapes[m][1]) for m in modules)


with torch.device("meta"):
    base = AutoModelForCausalLM.from_config(config)
total = sum(p.numel() for p in base.parameters())
print(f"base parameters {total:,}\n")

print(f"{'rank':>4}  " + "  ".join(f"{t:>22}" for t in TARGETS))
for rank in (1, 2, 4, 8, 16, 32, 64):
    cells = []
    for modules in TARGETS.values():
        peft_model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=modules))
        counted = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        assert counted == formula(rank, modules)
        cells.append(f"{counted:>12,} ({counted / total:5.2%})")
        base = peft_model.unload()
    print(f"{rank:>4}  " + "  ".join(f"{c:>22}" for c in cells))

rank, modules = 8, TARGETS["all linear"]
trainable = formula(rank, modules)
mb = 1_000_000
print(f"\nrank 8, all linear: {trainable:,} trainable parameters")
print(f"adapter weights {trainable * 4 / mb:.1f} MB, gradients {trainable * 4 / mb:.1f} MB, Adam moments {trainable * 8 / mb:.1f} MB")
print(f"frozen base: {total * 4 / mb:.0f} MB in float32, {total * 2 / mb:.0f} MB in bfloat16")
print(f"full fine-tuning with Adam in float32 would hold weights, gradients and moments: {total * 16 / mb:.0f} MB")
print(f"LoRA holds the frozen base plus {trainable * 16 / mb:.1f} MB: {(total * 4 + trainable * 16) / (total * 16):.0%} of that")
print("estimates only: activations, the data and framework overhead are not counted")
```

What it prints:

```text
hidden 576, intermediate 1536, layers 30, key/value width 192
base parameters 134,515,008

rank                     q,v                 q,k,v,o              all linear
   1          57,600 (0.04%)         115,200 (0.09%)         305,280 (0.23%)
   2         115,200 (0.09%)         230,400 (0.17%)         610,560 (0.45%)
   4         230,400 (0.17%)         460,800 (0.34%)       1,221,120 (0.91%)
   8         460,800 (0.34%)         921,600 (0.69%)       2,442,240 (1.82%)
  16         921,600 (0.69%)       1,843,200 (1.37%)       4,884,480 (3.63%)
  32       1,843,200 (1.37%)       3,686,400 (2.74%)       9,768,960 (7.26%)
  64       3,686,400 (2.74%)       7,372,800 (5.48%)     19,537,920 (14.52%)

rank 8, all linear: 2,442,240 trainable parameters
adapter weights 9.8 MB, gradients 9.8 MB, Adam moments 19.5 MB
frozen base: 538 MB in float32, 269 MB in bfloat16
full fine-tuning with Adam in float32 would hold weights, gradients and moments: 2152 MB
LoRA holds the frozen base plus 39.1 MB: 27% of that
estimates only: activations, the data and framework overhead are not counted
```

### Try it: rank, target modules and memory

The defaults (rank 8, all linear layers, `lora_alpha` 16, float32 base) reproduce block 2: 2,442,240 trainable parameters, 1.82 percent of the base, 9.8 MB of adapter weights, and LoRA training state at 27 percent of full fine-tuning. Change the rank to 64 to see 19,537,920 (14.52 percent), pick "q, v" to see how few parameters attention-only adapters train, and switch the base to bfloat16 to see the frozen weights halve.

<LoraRankLab />

### 3. Merging the adapter

```python
import json
import os
import tempfile

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import PrinterCallback
from trl import SFTConfig, SFTTrainer

torch.manual_seed(0)
name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)

tickets = [
    ("Hi, I was charged twice for my subscription this month. Thanks.", "billing"),
    ("Hello, the app crashes every time I open the dashboard.", "technical"),
    ("Hey, I forgot my password and the reset email never arrives.", "account"),
    ("Hi team, please send me a receipt for last month's payment.", "billing"),
    ("Good morning, I get a 500 error when I upload a file.", "technical"),
    ("Hello, please delete my account and all my data.", "account"),
] * 8
dataset = Dataset.from_list([
    {"prompt": [{"role": "user", "content": t}], "completion": [{"role": "assistant", "content": json.dumps({"queue": q})}]}
    for t, q in tickets
])
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="sft-merge", per_device_train_batch_size=8, max_steps=30, learning_rate=3e-4, report_to="none", use_cpu=True,
                   max_length=128, seed=0, save_strategy="no", gradient_checkpointing=False, bf16=False, logging_steps=30, disable_tqdm=True),
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.remove_callback(PrinterCallback)
trainer.train()
peft_model = trainer.model.eval()

batch = tokenizer(
    [tokenizer.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False) for t, _ in tickets[:6]],
    return_tensors="pt", padding=True, add_special_tokens=False,
)
with torch.no_grad():
    adapter_logits = peft_model(**batch).logits
    with peft_model.disable_adapter():
        base_logits = peft_model(**batch).logits
print(f"adapter changes the logits by up to {(adapter_logits - base_logits).abs().max():.3f} (adapter on versus off)")

ids = tokenizer.apply_chat_template([{"role": "user", "content": tickets[0][0]}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
with torch.no_grad():
    out = peft_model.generate(**ids, max_new_tokens=12, do_sample=False)
adapter_answer = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

with tempfile.TemporaryDirectory() as adapter_dir, tempfile.TemporaryDirectory() as merged_dir:
    peft_model.save_pretrained(adapter_dir)
    merged = peft_model.merge_and_unload().eval()
    merged.save_pretrained(merged_dir)
    size = lambda d: sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".safetensors")) / 1e6
    print(f"saved adapter {size(adapter_dir):.1f} MB, saved merged model {size(merged_dir):.1f} MB")

with torch.no_grad():
    merged_logits = merged(**batch).logits
print(f"merged model versus adapter-on logits: largest difference {(merged_logits - adapter_logits).abs().max():.2e}")
print(f"merged model versus the original base logits: largest difference {(merged_logits - base_logits).abs().max():.3f}")

def answer(model):
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=12, do_sample=False)
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


print("adapter-on answer:   ", repr(adapter_answer))
print("merged model answer:", repr(answer(merged)))
print("has adapter layers left:", any("lora" in n for n, _ in merged.named_parameters()))
```

What it prints:

```text
adapter changes the logits by up to 20.397 (adapter on versus off)
saved adapter 9.8 MB, saved merged model 538.1 MB
merged model versus adapter-on logits: largest difference 2.30e-04
merged model versus the original base logits: largest difference 20.397
adapter-on answer:    '{"queue": "billing"}'
merged model answer: '{"queue": "billing"}'
has adapter layers left: False
```

### 4. Changing rank and target modules, one at a time

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
from trl import SFTConfig, SFTTrainer

name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)

ISSUES = {
    "billing": ["I was charged twice for my subscription this month", "my invoice shows an amount I do not recognise", "I need a refund for the annual plan I bought by mistake", "the card on file was declined but my bank says it is fine", "please send me a receipt for last month's payment", "I was billed after I cancelled", "the price on my invoice is higher than the one I was quoted", "I want to change the credit card used for payments"],
    "technical": ["the app crashes every time I open the dashboard", "the export button does nothing when I click it", "I get a 500 error when I upload a file", "pages take more than a minute to load", "the mobile app freezes on the login screen", "the API returns an empty list for my project", "notifications stopped arriving on my phone", "the report is missing data from last week"],
    "account": ["I forgot my password and the reset email never arrives", "I need to change the email address on my account", "please delete my account and all my data", "I cannot enable two-factor authentication", "my teammate needs access to our workspace", "my account was locked after too many login attempts", "I want to change the name shown on my profile", "how do I transfer ownership of the workspace to someone else"],
}
OPENERS = ["Hi,", "Hello,", "Hi team,", "Good morning,", "", "Hey,"]
CLOSERS = ["Thanks.", "Please help.", "Thank you in advance.", "", "This is urgent.", "Regards, Sam"]

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
dataset = Dataset.from_list([
    {"prompt": [{"role": "user", "content": r["ticket"]}], "completion": [{"role": "assistant", "content": json.dumps({"queue": r["queue"]})}]}
    for r in train
])


def answer_loss(model, rows):
    model.eval()
    total = count = 0
    for r in rows:
        prompt = [{"role": "user", "content": r["ticket"]}]
        p = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
        f = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": json.dumps({"queue": r["queue"]})}], tokenize=True, return_dict=True)["input_ids"]
        labels = torch.tensor([[-100] * len(p) + f[len(p):]])
        with torch.no_grad():
            loss = model(input_ids=torch.tensor([f]), labels=labels).loss.item()
        n = len(f) - len(p)
        total += loss * n
        count += n
    return total / count


def run(rank, modules, lr=3e-4, steps=30):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(output_dir="sft-sweep", per_device_train_batch_size=8, max_steps=steps, learning_rate=lr, warmup_steps=3, report_to="none",
                       use_cpu=True, max_length=128, seed=0, save_strategy="no", gradient_checkpointing=False, bf16=False, logging_steps=steps, disable_tqdm=True),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=modules, task_type="CAUSAL_LM"),
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    trainer.remove_callback(PrinterCallback)
    trainer.train()
    return trainable, answer_loss(trainer.model, test)


base_loss = answer_loss(AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32), test)
print(f"held-out answer loss before any training: {base_loss:.4f}")
print(f"{'rank':>4} {'alpha':>5}  {'modules':<12} {'trainable':>10}  {'held-out answer loss after 30 steps':>36}")
for rank, label, modules in [(8, "q,v", ["q_proj", "v_proj"]), (1, "all linear", "all-linear"), (8, "all linear", "all-linear"), (64, "all linear", "all-linear")]:
    trainable, loss = run(rank, modules)
    print(f"{rank:>4} {2 * rank:>5}  {label:<12} {trainable:>10,}  {loss:>36.4f}")
```

What it prints:

```text
held-out answer loss before any training: 4.8644
rank alpha  modules       trainable   held-out answer loss after 30 steps
   8    16  q,v             460,800                                2.8097
   1     2  all linear      305,280                                3.4644
   8    16  all linear    2,442,240                                0.5242
  64   128  all linear   19,537,920                                0.1132
```

## Production snippets (not run here)

QLoRA with TRL needs a GPU and the bitsandbytes kernels. The `quantization_config` argument of `SFTTrainer` is documented as the way to combine a quantised load with a `peft_config`; it applies when the model is given as a string. Not run in this environment.

```python
import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
trainer = SFTTrainer(
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
    args=SFTConfig(
        output_dir="qlora-router",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        num_train_epochs=2,
        bf16=True,
        gradient_checkpointing=True,
        packing=True,
        max_length=512,
    ),
    train_dataset=dataset,
    quantization_config=quantization,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.train()
```

Not run in this environment. A real run would use a model large enough to need the quantisation, and `dataset` as built in the previous chapter.

## Designing with it

1. **Start from the strongest prompt baseline and beat it.** A tuned model that does not beat six good examples on a held-out set is a maintenance cost with no benefit.
2. **Hold out by issue, not by row.** Block 1 measures on unseen issues; a random split would flatter the model.
3. **Target all linear layers first, then reduce.** The published findings and block 4 both favour including the MLP layers.
4. **Keep alpha tied to rank.** `lora_alpha = 2 * r` or `r` keeps the scaling steady, so a learning rate that worked at one rank is a sensible start at another.
5. **Watch held-out loss, not training loss.** Block 1's training loss goes to 0.0087; that is memorisation as much as learning.
6. **Decide merge or keep separate by serving shape.** Many tasks on one base favour separate adapters; one product favours a merged model.
7. **Record versions.** TRL and PEFT change defaults between releases: the numbers in this chapter come from TRL 1.14.1 and PEFT 0.21.2.

## Where this stands in 2026

:::info Industry view
- **LoRA with PEFT and TRL is the standard low-cost route** and is a first-class path in both libraries' documentation, with `peft_config` accepted directly by the SFT, DPO and KTO trainers.
- **The capacity trade is now well documented.** Between the 2024 and 2025 studies cited above, the guidance converged: apply LoRA to all layers, expect lower capacity than full fine-tuning on large new-skill datasets, and tune the learning rate separately from full fine-tuning.
- **Defaults move.** TRL's SFT defaults here include a chunked cross-entropy loss and, at this version, gradient checkpointing on and a learning rate of 2e-5; a recipe copied from an older tutorial may not do what it says.
- **Prefer to verify, not to trust, headline numbers.** The 99.3 percent figure is from the QLoRA authors' abstract; I did not reproduce it, and it measures a 2023 benchmark.
- **I could not verify** anything about GPU throughput or QLoRA memory on real hardware, since none was available.
:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Block 2 shows rank 8 on q and v trains 460,800 parameters and on all linear layers 2,442,240. Derive the first number.</summary>

A LoRA pair on a weight of shape in by out adds rank x (in + out) parameters. Query and value projections here are 576 to 576 and 576 to 192. At rank 8 that is 8 x (576 + 576) = 9,216 plus 8 x (576 + 192) = 6,144, so 15,360 per layer, and 30 layers give 460,800.<br /><em>Authored · applied</em>

</details>

<details>
<summary><strong>Q2.</strong> Why can the adapter be tiny while the model it changes is not, and what does that cost in capacity?</summary>

The update to a weight matrix is constrained to a low-rank product, so it has far fewer degrees of freedom than the full matrix. A narrow task (a format and three labels) needs few. A task that must add a lot of new capability may need more than a low-rank update can hold, which is what the "learns less" finding describes.<br /><em>Authored · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> After training, the loss is 0.0087 but 4 of 23 held-out tickets are wrong. What does that tell you, and what would you do next?</summary>

The model fits the training sentences almost perfectly but does not generalise to every unseen issue, which is overfitting to a small templated dataset. Next steps: more varied training phrasing, held-out loss for early stopping, and a larger and more realistic validation set. Adding rank does not fix an overfit.<br /><em>Authored · interpretation</em>

</details>

<details>
<summary><strong>Q4.</strong> When would you keep the adapter separate rather than merging it?</summary>

When one base model serves many tasks or customers: you load the base once and attach the right 10 MB adapter per request group. When you serve one product, merging removes the adapter's extra computation and simplifies deployment, at the price of storing a full model per variant.<br /><em>Authored · applied</em>

</details>

<details>
<summary><strong>Q5.</strong> Block 4 gave 0.5242 for rank 8 on all linear and 2.8097 for rank 8 on q and v. Name two reasons not to conclude "always use all linear layers".</summary>

It is one seed, 30 steps and one synthetic task, and the models' losses were measured before training had converged. The all-linear adapter also has about five times the parameters, so the comparison mixes target modules with capacity. The published findings point the same way, but a project should run its own sweep on its own data.<br /><em>Authored · interpretation</em>

</details>

<details>
<summary><strong>Q6.</strong> The merged model's logits differ from the adapter-on logits by 2.3e-4. Is something wrong?</summary>

No. Merging adds the low-rank product into the weights, which changes the order of floating-point operations. A difference of that size in float32 is expected rounding. A large difference would indicate a bug, for example merging into a lower-precision copy or using the wrong scaling.<br /><em>Authored · interpretation</em>

</details>

## Further reading

- [Hugging Face TRL: SFT trainer](https://huggingface.co/docs/trl/sft_trainer): `SFTConfig`, adapters with `peft_config`, QLoRA via `quantization_config`.
- [Hugging Face PEFT: LoraConfig reference](https://huggingface.co/docs/peft/package_reference/lora) and [LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora): parameters, `all-linear`, rsLoRA, merging.
- [Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685): the original method.
- [Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314): NF4, double quantisation, paged optimisers.
- [Biderman et al., "LoRA Learns Less and Forgets Less"](https://arxiv.org/abs/2405.09673): where LoRA falls short of full fine-tuning and what it preserves.
- [Thinking Machines, "LoRA Without Regret"](https://thinkingmachines.ai/blog/lora/): when LoRA matches full fine-tuning.
- [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).
- Theory: [LoRA, adapters and prefix tuning](/docs/theory/dnn/parameter-efficient-fine-tuning-lora). Previous: [preparing data for fine-tuning](/docs/llm-engineering/preparing-data-for-fine-tuning). Next: [preference tuning with DPO and ORPO](/docs/llm-engineering/preference-tuning-dpo-orpo).

## Check yourself

- I can fine-tune a small chat model with `SFTTrainer` and a `LoraConfig` and show a before and after on held-out data.
- I can derive the number of trainable parameters from rank and module shapes, and check it against PEFT.
- I can say what `r`, `lora_alpha`, `target_modules` and `use_rslora` change and what I would start with.
- I can estimate the training memory of LoRA against full fine-tuning and say what the estimate leaves out.
- I can merge an adapter, verify the merge numerically, and choose between merged and separate serving.
- I can say what published studies found about LoRA's capacity and why one toy sweep proves little.
