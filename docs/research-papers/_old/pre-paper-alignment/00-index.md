---
id: research-papers-index
title: "Research Papers: From the Idea to the Implementation"
sidebar_label: "Start here"
sidebar_position: 0
slug: /research-papers
description:
  "Fourteen original papers explained through their motivation, methods,
  equations, experiments and complete runnable teaching implementations."
tags: [research-papers]
---

Read each paper as a sequence of decisions: what problem the authors faced, what
they changed, how the method works, and what the experiments actually support.

These chapters follow your requested priority order. ResNet and DDPM come last.
Every chapter now opens with a one-minute explanation before moving from
intuition to equations, architecture, code, experiments and limitations. It then
connects the original work to later upgrades and ends with realistic interview
scenarios and worked answer expectations.

## What every chapter gives you

| Layer | What you should get from it |
| --- | --- |
| Paper in one minute | The problem, key idea and historical importance without jargon |
| Architecture flow | A Mermaid diagram that turns the paper's training or inference path into one visual story |
| Concept walkthrough | Intuition first, then notation, shapes and the original architecture |
| Real-world example | A concrete workflow plus the boundary between documented use and illustration |
| Interactive lab | A local, accessible control panel for changing the paper's important variables and inspecting exact values |
| Progressive implementation | The runnable script split into setup, core method, training/inference and verification stages |
| Evidence and limitations | What the experiments support, what the metrics mean and what can still fail |
| Further evolution | Primary-source links to later papers or official model releases, with the change each introduced |
| Interview practice | Scenario questions covering design, maths, implementation, debugging, evaluation and production trade-offs |

The embedded original paper remains available at the end of every chapter, so
the explanation can always be checked against the authors' figures, tables and
appendices.

## The reading order

Years below refer to the original preprint or report. BERT's conference
publication was in 2019, ResNet's in 2016, and ReAct's in 2023.

| No. | Year | Paper                                                          | Area                        | Why it matters                                                 | Concepts to learn                                                        |
| --- | ---- | -------------------------------------------------------------- | --------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 1   | 2017 | [Attention Is All You Need](/docs/research-papers/transformer) | Sequence modelling          | Builds an encoder–decoder around attention                     | Q/K/V, heads, masks, positions, residuals, training and decoding         |
| 2   | 2018 | [BERT](/docs/research-papers/bert)                             | Language understanding      | Learns representations using both sides of a token             | MLM, NSP, WordPiece, segment embeddings, downstream heads                |
| 3   | 2018 | [GPT-1](/docs/research-papers/gpt-1)                           | Generative pre-training     | Transfers a causal language model to supervised tasks          | Next-token loss, task transformations, fine-tuning, auxiliary objectives |
| 4   | 2019 | [GPT-2](/docs/research-papers/gpt-2)                           | Zero-shot transfer          | Studies tasks expressed as text continuations                  | WebText, byte-level BPE, scale, likelihood, decoding                     |
| 5   | 2020 | [GPT-3](/docs/research-papers/gpt-3)                           | In-context learning         | Tests scale and examples supplied in the prompt                | Zero/one/few-shot learning, fixed weights, evaluation, contamination     |
| 6   | 2020 | [RAG](/docs/research-papers/rag)                               | Retrieval and generation    | Combines external document memory with a generator             | Dense retrieval, latent documents, sequence/token marginalisation        |
| 7   | 2021 | [LoRA](/docs/research-papers/lora)                             | Efficient adaptation        | Learns small updates to frozen weight matrices                 | Rank, parameter counts, initialisation, merging, adapter checkpoints     |
| 8   | 2022 | [InstructGPT](/docs/research-papers/instructgpt)               | Instruction following       | Trains with demonstrations and human preferences               | SFT, reward models, PPO, advantages, reference KL, trade-offs            |
| 9   | 2022 | [ReAct](/docs/research-papers/react)                           | Tool-using agents           | Interleaves model decisions with environment feedback          | Actions, observations, state, demonstrations, recovery, evaluation       |
| 10  | 2021 | [CLIP](/docs/research-papers/clip)                             | Vision and language         | Makes text descriptions usable as visual class targets         | Dual encoders, contrastive loss, temperature, zero-shot classification   |
| 11  | 2023 | [LLaMA](/docs/research-papers/llama)                           | Efficient foundation models | Examines strong models with practical inference costs          | Training tokens, RMSNorm, SwiGLU, RoPE, data mixtures                    |
| 12  | 2025 | [DeepSeek-R1](/docs/research-papers/deepseek-r1)               | Reasoning post-training     | Studies outcome-reward learning and distilled reasoning models | R1-Zero, GRPO, cold start, rejection sampling, distillation              |
| 13  | 2015 | [ResNet](/docs/research-papers/resnet)                         | Deep vision networks        | Makes identity-preserving deep networks easier to optimise     | Degradation, residuals, gradients, projections, bottlenecks              |
| 14  | 2020 | [DDPM](/docs/research-papers/ddpm)                             | Image generation            | Learns to reverse a gradual Gaussian corruption process        | Schedules, variational objective, noise prediction, U-Net, sampling      |

## How to work through a chapter

Start with the motivating example. Before running the program, follow the tensor
shapes or probability calculation on paper. Then run the complete script,
inspect its outputs and connect each function to the method it implements.

Use the interactive lab before opening the code. Change one control at a time,
state what you expect to happen, and then check the plotted value or table. The
labs run locally in the page and expose a data-table alternative where a chart
is used. After that, use the staged code tabs; open the collapsed complete script
only when you are ready to run or modify the whole program.

The experiment sections explain how to interpret the paper's evidence. A
benchmark result always belongs to a particular dataset, scoring rule, model and
evaluation setting. The embedded PDF lets you inspect the full tables, appendix
and original wording without leaving the chapter.

Use the checklist to test understanding rather than just recognising
terminology. Being able to explain why an alternative would fail is a stronger
check than remembering an equation's name.

For interview practice, answer each scenario aloud before reading the worked
response. A strong answer should make assumptions explicit, use a concrete
example, discuss at least one failure mode and end with the metric or experiment
that would validate the decision.

## Run the complete examples

[Download all 14 Python programs and the run guide](/examples/research-papers/research-paper-examples.zip).

Each script is self-contained. It includes its imports, data generation or local
environment, model or runner, full execution loop and output checks. The
model-training examples use small datasets so they can run on a CPU without
downloading checkpoints or using API credentials. ReAct also supports a
separately configured model endpoint; its default mode tests the runner with an
explicit fixture.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python attention.py
```

The examples were checked with Python 3.12 and PyTorch 2.14.0. Run them in a
working directory where they can write demo checkpoints; DDPM also writes
generated images and ReAct writes a trace.

| Chapter     | Script                                                     | Complete runnable experiment                                      | What it does not reproduce                                        |
| ----------- | ---------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| Transformer | [attention.py](/examples/research-papers/attention.py)     | Encoder–decoder training and autoregressive sequence reversal     | Translation corpus, original scale and beam search                |
| BERT        | [bert.py](/examples/research-papers/bert.py)               | MLM + NSP pre-training, then classification fine-tuning           | WordPiece, natural-language corpus and original checkpoints       |
| GPT-1       | [gpt1.py](/examples/research-papers/gpt1.py)               | Causal pre-training followed by supervised fine-tuning            | Published task suite and original training scale                  |
| GPT-2       | [gpt2.py](/examples/research-papers/gpt2.py)               | Character language modelling and top-k generation                 | WebText, byte-level BPE and zero-shot benchmark claims            |
| GPT-3       | [gpt3.py](/examples/research-papers/gpt3.py)               | Learn to answer using a randomly changing prompt dictionary       | Natural-language few-shot transfer and sparse attention           |
| RAG         | [rag.py](/examples/research-papers/rag.py)                 | Joint retriever/generator training with both marginalisations     | DPR/BART scale, large index and original beam decoding            |
| LoRA        | [lora.py](/examples/research-papers/lora.py)               | Fit, save, reload and merge a low-rank adapter                    | Full-LLM task adaptation                                          |
| InstructGPT | [instructgpt.py](/examples/research-papers/instructgpt.py) | SFT, preference reward training and clipped PPO                   | Human annotation and multi-token language-model rollouts          |
| ReAct       | [react.py](/examples/research-papers/react.py)             | Stateful tools, parsing, feedback, stopping and model integration | Original benchmark environments; default fixture is not a model   |
| CLIP        | [clip.py](/examples/research-papers/clip.py)               | Contrastive encoder training and text-vector classification       | Natural-language encoders and unseen-concept transfer             |
| LLaMA       | [llama.py](/examples/research-papers/llama.py)             | Full narrow decoder with RoPE, RMSNorm and SwiGLU                 | Original tokenizer, data, KV cache and distributed training       |
| DeepSeek-R1 | [deepseek_r1.py](/examples/research-papers/deepseek_r1.py) | Cold start, GRPO, verified filtering and student training         | Original architecture and complete large-scale multi-stage recipe |
| ResNet      | [resnet.py](/examples/research-papers/resnet.py)           | Convolutional residual training and held-out classification       | ImageNet models and benchmark results                             |
| DDPM        | [ddpm.py](/examples/research-papers/ddpm.py)               | Time-conditioned U-Net training and complete reverse sampling     | Original image datasets, architecture and sampling scale          |

These programs are educational implementations written for the chapters.
Original author repositories, where available, are linked within the relevant
explanation. A complete runnable teaching experiment is different from
reproducing a published result; the tables above make that boundary concrete.

## Keep the contributions separate

| If you want to change…                 | Start with…              | The central operation                          |
| -------------------------------------- | ------------------------ | ---------------------------------------------- |
| How tokens exchange information        | Transformer, BERT, LLaMA | Change architecture or attention access        |
| How a model transfers to a task        | GPT-1, GPT-2, GPT-3      | Fine-tune weights or change the prompt         |
| Where answers get information          | RAG, ReAct               | Retrieve documents or interact with tools      |
| How many weights adaptation trains     | LoRA                     | Constrain the weight update                    |
| Which responses the model prefers      | InstructGPT, DeepSeek-R1 | Optimise supervised/preference/outcome signals |
| How images and text share meaning      | CLIP                     | Contrastive alignment of embeddings            |
| How deep representations are optimised | ResNet                   | Learn residual changes                         |
| How new images are generated           | DDPM                     | Learn and sample reverse denoising transitions |

Several of these ideas can coexist in one system. A decoder can use residual
connections, be adapted with LoRA, trained on preference feedback and connected
to retrieval tools. Understanding the separate operations makes those
combinations easier to reason about.

## Interview coverage across the collection

The scenarios at the end of the chapters collectively cover the following
interview dimensions:

| Interview dimension | Representative chapters |
| --- | --- |
| Tensor shapes, equations and objectives | Transformer, BERT, LoRA, CLIP, DDPM |
| Architecture selection | Transformer, BERT, GPT family, ResNet, LLaMA |
| Training and adaptation | GPT-1, LoRA, InstructGPT, DeepSeek-R1 |
| Debugging and failure isolation | RAG, ReAct, LLaMA, DDPM |
| Production latency, memory and cost | Transformer, LoRA, LLaMA, DDPM |
| Evaluation, leakage and experiment design | BERT, GPT-2, GPT-3, CLIP |
| Safety, authorization and trustworthy evidence | RAG, InstructGPT, ReAct, CLIP |
| Real-time system design | RAG policy assistant, incident agent, vision search and high-volume model serving |
