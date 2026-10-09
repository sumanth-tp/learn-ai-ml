---
id: afr-dspy
title: "Automatic Prompt Optimisation and DSPy"
sidebar_label: "5 · Prompt optimisation"
sidebar_position: 5
slug: /agentic-frontier/automatic-prompt-optimisation-and-dspy
description: "Treat a prompt as something a program can search for: DSPy signatures, modules, metrics and optimisers from the current 3.4 release, run on a stand-in model and on a real small model, with the failures that matter: weak teachers, tiny dev sets and a model too small to follow the format."
tags: [dspy, prompt-optimisation, few-shot, bootstrapping, gepa, mipro, prompts, evaluation]
---

import Infographic from '@site/src/components/Infographic';
import PromptSearchLab from '@site/src/components/viz/PromptSearchLab';

**In one line.** Instead of rewriting a prompt by hand until it feels right, you write the program, a few dozen examples and a scoring function, and an optimiser searches for the demonstrations and instructions that score best.

:::note Not from a lecture
Written for this site from the DSPy documentation (the `current` pages and the 3.4 migration guide, read on 8 October 2026) and the three papers under Go deeper. Everything runs against DSPy 3.4.0 on Python 3.14.6. Two models are used: a deterministic stand-in that copies the label of the most similar demonstration (stated plainly wherever it appears), and Qwen2.5-0.5B-Instruct from the Hugging Face hub on a CPU. I did not run the instruction-writing optimisers (MIPROv2, GEPA), which need a capable second model; they are described from the documentation and papers.
:::

:::tip Before you start
You should already know:

- what a prompt with few-shot examples is, and why examples change a model's behaviour ([prompt, retrieve or fine-tune](/docs/llm-engineering/prompt-retrieve-or-fine-tune));
- the idea of an evaluation set and a metric ([LLM evaluation workflow](/docs/llm-evals/evaluation-workflow));
- what a held-out test set is for ([model evaluation](/docs/theory/ml/model-evaluation)).

Reading time: about 45 minutes with the code. Block 5 runs a small model on the CPU and takes a few minutes.

After this chapter you can:

- read a DSPy program as signature, module, examples and metric, and say what an optimiser changes;
- run a labelled-demo, a bootstrapping and a random-search optimiser and read their scores honestly;
- name the failure modes: a weak teacher, a small dev set, and a metric that rewards the wrong thing.
:::

## In 30 seconds

A prompt is a set of knobs: the instruction, the examples shown, the order, the format. You normally turn them by hand, run a few cases and trust your eye. That does not scale, and it breaks when you change the model, because a prompt that suits one model can confuse another.

Think of tuning a guitar by ear against tuning it with a tuner. The tuner needs a reference note (your examples and metric) and then turns the pegs for you. DSPy is the tuner for prompts: you describe what goes in and what comes out, give it examples and a way to score an answer, and it tries variations and keeps the best. It does not make a weak model clever. It finds out what a given model needs to be shown.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Signature | A declaration of a task's inputs and outputs | `message -> queue` |
| Module | A way of calling the model for a signature | `dspy.Predict`, `dspy.ChainOfThought` |
| Demonstration (demo) | An example shown inside the prompt | "message: I was charged twice. queue: billing" |
| Metric | A function that scores one prediction | 1 if the queue is right, else 0 |
| Optimiser | A search that improves a program against a metric (DSPy also calls it a teleprompter) | `BootstrapFewShot` |
| Compile | Run an optimiser and get back an improved program | `optimiser.compile(program, trainset=...)` |
| Bootstrapping | Keeping the program's own correct runs as demonstrations | a trace that passed the metric |
| Teacher | The program that produces traces to bootstrap from | the zero-shot program, or a better one |
| Dev set | Examples used to choose between candidates | 24 messages |
| Test set | Examples never used for any choice, kept for the final score | 48 messages |

## The idea in plain words

Suppose you run a help desk and want a model to send each message to one of four queues. You write a prompt: "Route this message to billing, shipping, technical or account." You try five messages, three come out right, you reword the prompt, and now four do. Is it better, or lucky? A different model next month, and you start again.

The smallest example of an alternative has three parts. First, a few labelled examples: "charged twice soon" is billing, "parcel late again" is shipping, "app crashes today" is technical. Second, a score: right or wrong. Third, a rule for what goes in the prompt. If the model answers by finding the example most like the new message, then adding an example that looks like "tracking stuck please" teaches it that kind of message. In the worked example below, three demonstrations get 3 of 4 test messages right and a fourth demonstration gets 4 of 4.

In words: for many tasks, what a model needs is not cleverer wording but the right examples, and which examples are right is an empirical question. DSPy turns that question into a search with a score, so it can be re-run when the model, the data or the task changes. The precise statement is longer: DSPy separates the **program** (what steps, with what inputs and outputs) from its **parameters** (the instructions and demonstrations in each prompt, and optionally the model weights), and an optimiser tunes the parameters against a metric.

<Infographic src="/img/afr/dspy-from-prompts-to-programs.svg" alt="Four boxes you write (signature, module, examples, metric), a loop the optimiser runs (run, score, propose, keep the best) and a table of four optimisers with what each changes and whether it needs a second model." caption="The top row is everything you write. The loop below is what compile does. The table says what each optimiser is allowed to change." />

## Worked example, step by step

The stand-in model of the first code block works like this. Given a message, it looks at the demonstrations in the prompt, finds the one that shares the most words with the message, and if they share at least two words it copies that demonstration's label. Otherwise it answers `billing`. That is a crude imitation of in-context learning, and it is deterministic, which makes the arithmetic checkable.

Take three demonstrations: "charged twice soon" (billing), "parcel late again" (shipping), "app crashes today" (technical). Four test messages:

1. "charged twice please" shares "charged twice" with the first demonstration: 2 words, so the answer is billing. Right.
2. "parcel late today" shares "parcel late" with the second: 2 words, shipping. Right.
3. "tracking stuck soon" shares only "soon" with the first demonstration: 1 word, below 2, so the default answer is billing. The truth is shipping. Wrong.
4. "app crashes again" shares "app crashes" with the third: 2 words, technical. Right.

Three of four right is 0.75. Now add a fourth demonstration, "tracking stuck please" (shipping). Message 3 now shares "tracking stuck" with it: 2 words, shipping. Right. Four of four.

An optimiser is a way of finding that fourth demonstration without being told which message was missed. It tries candidate sets of demonstrations, scores each on examples it has labels for, and keeps the set that scores best. The first code block reproduces the 3 of 4 and 4 of 4.

<Infographic src="/img/afr/dspy-worked-example.svg" alt="Three demonstrations, a table of four messages with the demonstration each matches, shared words, answer and truth, and the effect of adding a fourth demonstration." caption="Read the table row by row: the third message is the one with no demonstration to match. The panel below it shows the fix." />

## How it works

### Signatures: say what, not how

A signature names inputs and outputs. The short form is a string, `"message -> queue"`. A class gives each field a description and the task an instruction, as in the code below. DSPy's documentation makes a point worth remembering: the model reads the field names, so naming is "the cheapest optimization". A field called `research_request` will do better than one called `request`, and `"a, b -> c"` leaves the model guessing. Types matter too: `contains_pun: bool` or `haikus: list[str]` tell DSPy how to parse the answer and to warn when it cannot.

### Modules: how to call the model

A module wraps a signature in a strategy. `dspy.Predict` asks once. `dspy.ChainOfThought` adds a reasoning field first. `dspy.ReAct` interleaves tool calls. The documentation lists others, among them `BestOfN`, `Refine`, `MultiChainComparison`, `ProgramOfThought` and `CodeAct`. Modules can contain modules, so a program is ordinary Python with several model calls. The optimiser sees each call as a separate set of parameters.

### What goes into the prompt

An **adapter** turns a signature and its demonstrations into a request. The default is `ChatAdapter`. The second code block prints what it builds: a system message that lists input and output fields and the instruction, a user and an assistant message for each demonstration, then the real input, ending with the instruction to answer in the fields' format and finish with a marker. DSPy then parses the reply back into typed fields. Format failures surface as parse errors, which matters for small models (block 5).

### The metric and the data

A **metric** takes the gold example and the prediction and returns a score. Exact match is the simplest. It can also be a function that checks the output's shape, runs code, or calls a judge model. A metric is where you say what "good" means, so it is also where an optimiser will find loopholes. Split your examples three ways: a **train** set the optimiser learns from, a **dev** set it uses to choose, and a **test** set you look at once. The GEPA page of the documentation says the same: keep a final test set that influenced neither step.

### The optimisers, by what they change

The DSPy API reference (checked 8 October 2026) lists these optimisers: `LabeledFewShot`, `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch` (also `BootstrapRS`), `BootstrapFinetune`, `COPRO`, `MIPROv2`, `GEPA`, `SIMBA`, `BetterTogether`, `Ensemble`, `KNN`, `KNNFewShot` and `InferRules`.

| Optimiser | What it does | Needs |
| --- | --- | --- |
| `LabeledFewShot(k)` | Puts `k` labelled training examples into the prompt | labels only |
| `BootstrapFewShot` | Runs a teacher program on training inputs and keeps traces that pass the metric as demonstrations, up to `max_bootstrapped_demos` (default 4), padded with labelled ones up to `max_labeled_demos` (default 16) | a metric |
| `BootstrapFewShotWithRandomSearch` | Tries seeds -3 (zero-shot), -2 (labelled), -1 (bootstrapped) and then shuffled bootstraps, scores each on a dev set, keeps the best | a metric, a dev set |
| `MIPROv2` | Proposes instructions and bootstraps demonstrations, then searches the combinations with Bayesian optimisation. `auto` can be `light`, `medium` or `heavy` | a metric, a model that writes instructions |
| `GEPA` | Evolves instruction text by reflecting on failures; the metric can return written feedback as well as a score. The docs' example budget `auto="light"` evaluates about six candidate prompts | a metric, a reflection model |
| `BootstrapFinetune` | Uses traces to fine-tune weights instead of changing the prompt | a model you can tune |

Three papers sit behind these. The DSPy paper (Khattab and colleagues, arXiv 2310.03714, October 2023) reports that within minutes of compiling, a few lines let GPT-3.5 and llama2-13b-chat bootstrap pipelines that beat standard few-shot prompting by roughly 25% and 65% on its tasks. The MIPRO paper (Opsahl-Ong and colleagues, arXiv 2406.11695, June 2024) reports that it beat baseline optimisers on five of seven multi-stage programs with Llama-3-8B, by up to 13% accuracy. The GEPA paper (Agrawal and colleagues, arXiv 2507.19457, July 2025, an ICLR 2026 oral) reports a 6% average gain over a reinforcement-learning method (GRPO) on six tasks with up to 35 times fewer rollouts, and over 10% over MIPROv2. Those are the authors' tasks and models. The DSPy documentation also quotes two company results (a Shopify task made about 75 times cheaper and twice as reliable, and a Dropbox accuracy doubling); I could not check them independently.

### What a compiled program is

The optimiser returns a program with the same code and different parameters: demonstrations attached to each predictor and possibly rewritten instructions. You save it to a file and load it later. Nothing about the model changed, so there is no training run and no new weights, unless you chose `BootstrapFinetune`.

### Where it breaks

- **A weak teacher teaches one thing.** Bootstrapping keeps traces that pass the metric. If the teacher is right only on the easy class, every demonstration is that class. Block 3 shows it.
- **A tiny dev set picks noise.** With 24 dev examples, one answer moves the score by 0.042. Candidates that differ by one answer are tied.
- **The metric is the target.** An optimiser raises the metric, not the thing you meant. Check outputs by eye after every search.
- **The model cannot do the task.** Better demonstrations do not give a 0.5-billion-parameter model the ability to tell four queues apart. Block 5 shows the ceiling.
- **A prompt tuned for one model is tuned for that model.** Re-run the search when you change the model; that is a feature of the approach, not a bug.

### A note on the 3.4 API

DSPy 3.4 is a transition release. Custom language models written as `BaseLM` subclasses with a `forward` method still work but warn, and are scheduled for removal in 3.5. The replacement is a small **engine** object with a `complete(request)` method that returns a response, passed as `dspy.LM("name", engine=engine)`. The code below uses that interface for its stand-in and for the local model, so it is the form to learn.

## Code you can run

Five blocks, each self-contained. They use `dspy` 3.4.0, `tiktoken` 0.14.0, `torch` 2.14.1 and `transformers` 5.18.0. Blocks 1 to 4 use deterministic stand-in models and run in seconds. Block 5 loads Qwen2.5-0.5B-Instruct (about 1 GB, downloaded once) and runs about a hundred generations on the CPU. The stand-in is **not a language model**: it copies the label of the most similar demonstration. It exists so the optimisers' mechanics are visible and exactly repeatable.

### 1. The worked example, reproduced

First the four messages and three demonstrations from the worked example, run through a real `dspy.Predict` with the stand-in engine, then with the fourth demonstration added.

```python
import re
import warnings

warnings.filterwarnings("ignore")
import dspy
from dspy.lm15 import Message, Response, Usage

class NearestDemoEngine:
    def complete(self, request):
        texts = ["".join(getattr(part, "text", "") for part in m.parts) for m in request.messages]
        grab = lambda text: re.search(r"message ## \]\]\n(.*?)(?:\n|$)", text).group(1)
        demos = [(grab(a), re.search(r"queue ## \]\]\n(\w+)", b).group(1)) for a, b in zip(texts[0:-1:2], texts[1:-1:2])]
        query = set(grab(texts[-1]).split())
        shared = lambda demo: len(query & set(demo[0].split()))
        best = max(demos, key=shared, default=None)
        label = best[1] if best and shared(best) >= 2 else "billing"
        return Response(id=None, model=request.model, message=Message.assistant(f"[[ ## queue ## ]]\n{label}\n\n[[ ## completed ## ]]"),
                        finish_reason="stop", usage=Usage())

dspy.configure(lm=dspy.LM("stand-in/nearest-demo", engine=NearestDemoEngine(), cache=False))

class Route(dspy.Signature):
    """Route a customer message to a queue."""
    message: str = dspy.InputField()
    queue: str = dspy.OutputField()

demo = lambda message, queue: dspy.Example(message=message, queue=queue)
three = [demo("charged twice soon", "billing"), demo("parcel late again", "shipping"), demo("app crashes today", "technical")]
queries = [("charged twice please", "billing"), ("parcel late today", "shipping"), ("tracking stuck soon", "shipping"), ("app crashes again", "technical")]

for label, demos in (("three demos", three), ("four demos", three + [demo("tracking stuck please", "shipping")])):
    program = dspy.Predict(Route)
    program.demos = demos
    answers = [program(message=message).queue for message, _ in queries]
    right = sum(answer == truth for answer, (_, truth) in zip(answers, queries))
    print(f"{label}: answers {answers}, {right} of {len(queries)} right")
```

**Reading the output.** With three demonstrations the answers are billing, shipping, billing, technical: 3 of 4 right, because the third message has no matching demonstration. With the fourth the answers become billing, shipping, shipping, technical: 4 of 4.

**Line by line.**

- `NearestDemoEngine.complete` is the stand-in. It reads the demonstration pairs out of the request messages, picks the one sharing the most words with the query and copies its label if at least two words are shared.
- `dspy.LM("stand-in/nearest-demo", engine=..., cache=False)` is the 3.4 way to plug in a custom backend. `cache=False` makes sure every call reaches the engine.
- `program.demos = demos` attaches demonstrations by hand. An optimiser does the same thing after a search.

### 2. The prompt DSPy builds

What does DSPy actually send? An engine that only records the request tells us, once without a demonstration and once with one.

```python
import warnings

warnings.filterwarnings("ignore")
import dspy
from dspy.lm15 import Message, Response, Usage

seen = []

class RecordingEngine:
    def complete(self, request):
        seen.append(request)
        return Response(id=None, model=request.model, message=Message.assistant("[[ ## queue ## ]]\nshipping\n\n[[ ## completed ## ]]"),
                        finish_reason="stop", usage=Usage())

dspy.configure(lm=dspy.LM("recording/none", engine=RecordingEngine(), cache=False))

class Route(dspy.Signature):
    """Route a customer message to a queue."""
    message: str = dspy.InputField()
    queue: str = dspy.OutputField(desc="one of billing, shipping, technical, account")

program = dspy.Predict(Route)
program(message="My parcel has not arrived.")
program.demos = [dspy.Example(message="I was charged twice.", queue="billing")]
result = program(message="My parcel has not arrived.")

for number, request in enumerate(seen, 1):
    print(f"--- request {number}: {len(request.messages)} messages after the system prompt")
    if number == 1:
        print(request.system)
    for message in request.messages:
        print(f"[{message.role}]", "".join(part.text for part in message.parts).strip().replace("\n", " | "))
print("parsed answer:", result.queue, "| type:", type(result).__name__)
```

**Reading the output.** Request 1 is a system message and one user message. The system message lists the input field `message` and the output field `queue` (with its description), shows the format with `{message}` and `{queue}` placeholders, and ends with the signature's docstring as the objective. The user message carries the real input and the instruction to reply with the `queue` field and the completion marker. Request 2 has the same system message and three messages: the demonstration as a user message, the demonstration's answer as an assistant message, then the real input. The parsed answer is `shipping`, returned as a `Prediction` object with a `queue` attribute.

**Line by line.**

- `RecordingEngine` stores each request and returns a fixed reply in the format the adapter expects. The adapter parses it back into fields.
- Setting `program.demos` after the first call changes what the next prompt contains. That is all an optimiser changes about demonstrations.

### 3. Four optimisers, one stand-in

Now the optimisers themselves, on the stand-in. The task has four queues and 24 openers (six per queue), each with four ending words. 96 messages are split 24 train, 24 dev and 48 test. We run labelled demonstrations at six sizes, bootstrapping at two sizes, and random search.

```python
import contextlib
import io
import random
import re
import warnings

warnings.filterwarnings("ignore")
import dspy
import tiktoken
from dspy.lm15 import Message, Response, Usage
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, LabeledFewShot

QUEUES = ["billing", "shipping", "technical", "account"]
OPENERS = {
    "billing": ["charged twice", "invoice wrong", "refund payment", "card billed", "discount missing", "unexpected fee"],
    "shipping": ["parcel late", "tracking stuck", "courier left", "delivery delayed", "address change", "box damaged"],
    "technical": ["app crashes", "cannot log in", "error page", "export fails", "sync stopped", "nothing loads"],
    "account": ["change email", "close profile", "add teammate", "reset two-factor", "update company", "delete data"],
}
pool = [(f"{phrase} {tag}", queue) for queue in QUEUES for phrase in OPENERS[queue] for tag in ("soon", "again", "today", "please")]
random.Random(1).shuffle(pool)
pool = [dspy.Example(message=m, queue=q).with_inputs("message") for m, q in pool]
train, dev, test = pool[:24], pool[24:48], pool[48:]
counter = tiktoken.get_encoding("o200k_base")

class NearestDemoEngine:
    def __init__(self):
        self.calls, self.prompt_tokens = 0, []

    def complete(self, request):
        self.calls += 1
        texts = ["".join(getattr(part, "text", "") for part in m.parts) for m in request.messages]
        self.prompt_tokens.append(len(counter.encode((request.system or "") + "".join(texts))))
        grab = lambda text: re.search(r"message ## \]\]\n(.*?)(?:\n|$)", text).group(1)
        demos = [(grab(a), re.search(r"queue ## \]\]\n(\w+)", b).group(1)) for a, b in zip(texts[0:-1:2], texts[1:-1:2]) if "queue ##" in b]
        query = set(grab(texts[-1]).split())
        best = max(demos, key=lambda d: len(query & set(d[0].split())) / len(query | set(d[0].split())), default=None)
        label = best[1] if best and len(query & set(best[0].split())) >= 2 else "billing"
        return Response(id=None, model=request.model, message=Message.assistant(f"[[ ## queue ## ]]\n{label}\n\n[[ ## completed ## ]]"),
                        finish_reason="stop", usage=Usage())

engine = NearestDemoEngine()
dspy.configure(lm=dspy.LM("stand-in/nearest-demo", engine=engine, cache=False))

class Route(dspy.Signature):
    """Route a customer message to a queue."""
    message: str = dspy.InputField()
    queue: str = dspy.OutputField(desc="one of billing, shipping, technical, account")

metric = lambda gold, pred, trace=None: pred.queue == gold.queue
base = dspy.Predict(Route)

def score(program, data):
    engine.prompt_tokens.clear()
    accuracy = sum(metric(e, program(message=e.message)) for e in data) / len(data)
    return accuracy, sum(engine.prompt_tokens) / len(engine.prompt_tokens)

print("labelled demos   dev    test   prompt tokens")
for k in (0, 4, 8, 12, 16, 24):
    program = LabeledFewShot(k=k).compile(base, trainset=train) if k else base
    print(f"{k:14d} {score(program, dev)[0]:6.3f} {score(program, test)[0]:6.3f} {score(program, test)[1]:8.0f}")

lines, quiet = [], io.StringIO()
with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
    for demos in (4, 12):
        program = BootstrapFewShot(metric=metric, max_bootstrapped_demos=demos, max_labeled_demos=demos).compile(base, trainset=train)
        queues = sorted({d["queue"] for d in program.predictors()[0].demos})
        lines.append(f"bootstrapped, up to {demos:2d}: dev {score(program, dev)[0]:.3f}  test {score(program, test)[0]:.3f}  queues in its demos: {queues}")
    search = BootstrapFewShotWithRandomSearch(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=4,
                                              num_candidate_programs=8, num_threads=1).compile(base, trainset=train, valset=dev)
print("\n".join(lines))
print("random search, 8 shuffled candidates plus 3 fixed ones: seed, dev accuracy, test accuracy")
for entry in sorted(search.candidate_programs, key=lambda e: e["seed"]):
    print(f"  seed {entry['seed']:2d}  {entry['score'] / 100:.3f}  {score(entry['program'], test)[0]:.3f}")
winner = search.candidate_programs[0]
print(f"winner on dev: seed {winner['seed']}, dev {winner['score'] / 100:.3f}, test {score(winner['program'], test)[0]:.3f}")
```

**Reading the output.** The first table is `LabeledFewShot` with `k` demonstrations. Accuracy climbs with `k`: dev 0.250, 0.375, 0.458, 0.500, 0.583, 0.625 and test 0.250, 0.375, 0.479, 0.583, 0.604, 0.771. The average prompt grows from 135 to 618 tokens, so the last row costs about 4.6 times the first per call. The stand-in only copies, so more demonstrations means more of the 24 openers are covered.

**What surprised me.** Bootstrapping did worse than just using labelled examples. `BootstrapFewShot` with up to 4 demonstrations scored 0.250 on both sets, and all its demonstrations were `billing`. The reason is the teacher. With no demonstrations the stand-in answers `billing` to everything, so the only training traces that "pass" are billing messages, and every bootstrapped demonstration teaches billing. With up to 12 demonstrations it picked up all four queues but scored 0.333 dev and 0.312 test, still below 12 labelled demonstrations (0.500 and 0.583).

Random search made 11 candidates (seeds -3 to 7): dev scores between 0.250 and 0.375, test scores 0.250 to 0.375. The winner on dev was seed -2, the plain labelled-demonstrations candidate, at dev 0.375 and test 0.375. So 11 candidate programs did not beat 12 labelled demonstrations by hand. Candidates with dev scores within one answer of each other (0.042) are tied, and the test scores of the dev-tied candidates differ (seed 6: dev 0.375, test 0.333).

**What this does not show.** A real language model generalises to phrasing it has not seen, which this stand-in cannot. The lesson is about the search: more and better-covering demonstrations help, a biased teacher poisons bootstrapping, and a small dev set cannot rank near-equal candidates.

**Line by line.**

- `pool` splits 96 messages 24, 24 and 48 with a fixed seed, so every run is the same.
- `score` returns accuracy and the average prompt size, counted with the same tokenizer as chapter 1. The engine records the size of each request it receives.
- `contextlib.redirect_stdout` hides DSPy's progress bars; the lines we want are collected and printed afterwards.
- `num_candidate_programs=8` plus the three fixed seeds gives 11 candidates. `valset=dev` tells the search which examples to score candidates on.
- `search.candidate_programs` holds every candidate, so we can score each one on the test set. DSPy itself only returns the winner.

### 4. An intermediate step nobody labelled

The reason DSPy bootstraps rather than only labelling is multi-step programs. Here a program first names the problem in two words, then routes. The training data only has the final queue. Nobody wrote the two-word problem for any example, so labelled demonstrations cannot supply them.

```python
import contextlib
import io
import random
import re
import warnings

warnings.filterwarnings("ignore")
import dspy
from dspy.lm15 import Message, Response, Usage
from dspy.teleprompt import BootstrapFewShot, LabeledFewShot

CAUSES = {"billing": ["charged twice", "refund payment", "invoice wrong"], "shipping": ["parcel late", "tracking stuck", "courier left"],
          "technical": ["app crashes", "cannot log in", "export fails"], "account": ["change email", "close profile", "add teammate"]}
rows = [(f"{cause} {tag}", queue) for queue, causes in CAUSES.items() for cause in causes for tag in ("soon", "again", "today")]
random.Random(4).shuffle(rows)
data = [dspy.Example(message=m, queue=q).with_inputs("message") for m, q in rows]
train, test = data[:12], data[12:]

def fields(text):
    return dict(re.findall(r"\[\[ ## (\w+) ## \]\]\n(.*?)(?=\n\n\[\[ ##|\n*$)", text, re.S))

class TwoStepEngine:
    def complete(self, request):
        texts = ["".join(getattr(part, "text", "") for part in m.parts) for m in request.messages]
        query = fields(texts[-1])
        if "`problem`" in request.system.split("Your output fields are:")[1].split("All interactions")[0]:
            reply = "[[ ## problem ## ]]\n" + " ".join(query["message"].split()[:2]) + "\n\n[[ ## completed ## ]]"
        else:
            demos = [(fields(a), fields(b)) for a, b in zip(texts[0:-1:2], texts[1:-1:2])]
            words = set((query["problem"] + " " + query["message"]).split())
            overlap = lambda d: len(words & set((d[0].get("problem", "") + " " + d[0]["message"]).split()))
            best = max(demos, key=overlap, default=None)
            reply = f"[[ ## queue ## ]]\n{best[1]['queue'] if best and overlap(best) >= 3 else 'billing'}\n\n[[ ## completed ## ]]"
        return Response(id=None, model=request.model, message=Message.assistant(reply), finish_reason="stop", usage=Usage())

dspy.configure(lm=dspy.LM("stand-in/two-step", engine=TwoStepEngine(), cache=False))

class Gist(dspy.Signature):
    """Name the main problem in two words."""
    message: str = dspy.InputField()
    problem: str = dspy.OutputField()

class Decide(dspy.Signature):
    """Route a customer message to a queue: billing, shipping, technical or account."""
    message: str = dspy.InputField()
    problem: str = dspy.InputField()
    queue: str = dspy.OutputField()

class Pipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gist, self.decide = dspy.Predict(Gist), dspy.Predict(Decide)

    def forward(self, message):
        return self.decide(message=message, problem=self.gist(message=message).problem)

metric = lambda gold, pred, trace=None: pred.queue == gold.queue
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    labelled = LabeledFewShot(k=6).compile(Pipeline(), trainset=train)
    bootstrapped = BootstrapFewShot(metric=metric, max_bootstrapped_demos=6, max_labeled_demos=6).compile(Pipeline(), trainset=train)
for name, program in (("labelled demos", labelled), ("bootstrapped demos", bootstrapped)):
    for module, predictor in program.named_predictors():
        invented = [d["problem"] for d in predictor.demos if d.get("problem")]
        print(f"{name:19s} {module:7s} {len(predictor.demos)} demos, {len(invented)} carry a problem nobody labelled: {invented}")
print("accuracy on", len(test), "held-out messages:",
      f"labelled {sum(metric(e, labelled(message=e.message)) for e in test) / len(test):.3f},",
      f"bootstrapped {sum(metric(e, bootstrapped(message=e.message)) for e in test) / len(test):.3f}")
print("fields in the training data:", sorted(train[0].keys()))
```

**Reading the output.** With `LabeledFewShot` both modules receive 6 demonstrations and none carries a `problem` field, because the data has none. With `BootstrapFewShot` both modules also receive 6, but one of them carries a problem, `cannot log`, which the program itself produced on a training example and which then passed the metric. Held-out accuracy is 0.417 for both. The training fields printed last are only `message` and `queue`.

**What surprised me.** Only one of the 12 training examples produced a passing trace. The teacher starts weak, and its default answer is `billing`, so passing traces are rare. That is the **cold start**: bootstrapping needs a teacher that is right sometimes. In a real program you would start with a stronger model as the teacher, or with a small set of hand-written traces.

**Line by line.**

- `Pipeline.forward` calls `gist`, then passes its output to `decide`. DSPy sees two predictors.
- `named_predictors()` lists each predictor and its demonstrations, so we can see what each module will be shown.
- `fields` parses the `[[ ## name ## ]]` sections of a message. The stand-in uses both the problem and the message when matching, so labelled demonstrations (which have no problem) still work for the first step of the teacher's run.

### 5. A real small model

Finally the same DSPy code against a real language model: Qwen2.5-0.5B-Instruct, greedy decoding, on the CPU. The test messages use openers the training messages never contain, so the model has to generalise. We compare zero-shot, 8 labelled demonstrations and bootstrapping.

```python
import collections
import contextlib
import io
import warnings

warnings.filterwarnings("ignore")
import dspy
import torch
from dspy.lm15 import Message, Response, Usage
from dspy.teleprompt import BootstrapFewShot, LabeledFewShot
from transformers import AutoModelForCausalLM, AutoTokenizer

OPENERS = {
    "billing": ["I was charged twice for", "The invoice shows the wrong amount for", "Please refund the payment for", "My card was billed for", "There is an unexpected fee on", "Cancel the recurring charge on"],
    "shipping": ["My parcel has not arrived for", "The tracking page is stuck on", "The courier left", "Delivery is two weeks late for", "The box was damaged on arrival for", "Where is"],
    "technical": ["The app crashes when I open", "I cannot log in to", "The page shows an error on", "Export to PDF fails in", "Nothing loads on", "The upload hangs in"],
    "account": ["Please change the email on", "I want to close", "How do I add a teammate to", "Reset the two-factor setup for", "Delete all personal data from", "Transfer ownership of"],
}
OBJECTS = ["my order", "the premium plan", "the mobile app", "my workspace"]

def rows(openers):
    return [dspy.Example(message=f"{OPENERS[q][i]} {obj}.", queue=q).with_inputs("message") for i in openers for obj in OBJECTS for q in OPENERS]

train = rows(range(4))[::3]
test = rows([4, 5])

name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

class LocalEngine:
    def complete(self, request):
        chat = [{"role": "system", "content": request.system}] if request.system else []
        chat += [{"role": m.role, "content": "".join(getattr(p, "text", "") for p in m.parts)} for m in request.messages]
        ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=24, do_sample=False, stop_strings=["[[ ## completed ## ]]"], tokenizer=tokenizer)
        text = tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return Response(id=None, model=request.model, message=Message.assistant(text), finish_reason="stop", usage=Usage())

dspy.configure(lm=dspy.LM("local/qwen2.5-0.5b", engine=LocalEngine(), cache=False))

class Route(dspy.Signature):
    """Route a customer message to a queue."""
    message: str = dspy.InputField()
    queue: str = dspy.OutputField(desc="one of billing, shipping, technical, account")

def evaluate(program, label):
    answers, right = collections.Counter(), 0
    for example in test:
        try:
            answer = program(message=example.message).queue.strip().lower()
        except Exception:
            answer = "unparsed"
        answers[answer] += 1
        right += answer == example.queue
    print(f"{label:28s} {right:2d} / {len(test)} correct  answers: {dict(answers)}")

metric = lambda gold, pred, trace=None: pred.queue.strip().lower() == gold.queue
base = dspy.Predict(Route)
print("training queues in order:", [e.queue[:4] for e in train[:8]], "...")
evaluate(base, "zero-shot")
evaluate(LabeledFewShot(k=8).compile(base, trainset=train), "8 labelled demos")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    bootstrapped = BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=4).compile(base, trainset=train)
evaluate(bootstrapped, "bootstrapped 4 + 4")
```

**Reading the output.** The training queues alternate (`bill`, `acco`, `tech`, `ship`, and so on), so labelled and bootstrapped demonstrations cover all four queues. Zero-shot gets 3 of 32 right. Its answers are mostly `billing` and, more telling, 24 of the 32 carry stray characters (`> billing`, `>billing`, `>billing]`) so even correct labels fail the exact-match metric. With 8 labelled demonstrations the model gets 11 of 32 right and the format is clean, but it answers billing 8 times, account 16 times, technical 8 times and never shipping. Bootstrapped demonstrations get 15 of 32, answering billing 25 times and account 7.

**What surprised me.** The biggest gain is the format: from 3 to 11 correct is mostly stray characters disappearing once the prompt contains examples. After that a 0.5-billion-parameter model still leans on one or two labels, and the best of the three programs scores under 50%. A 0.5-billion model cannot tell four support queues apart for unseen phrasing, whatever demonstrations it sees. Bootstrapping scored highest here but its answers are the most lopsided, so 15 of 32 is not evidence that it generalises better; with 32 test messages one answer is 3 percentage points.

**What I did not run.** Random search, MIPROv2 and GEPA against this model. An earlier trial of random search with seven candidates took 53 minutes on a heavily loaded machine, and the instruction-writing optimisers need a model that can write good instructions, which this one cannot. Their results are not claimed.

**Line by line.**

- `rows` builds messages from six openers per queue; openers 0 to 3 make the training set and 4 and 5 the test set, so test phrasing is unseen. `rows(range(4))[::3]` takes every third message, which keeps the queues balanced.
- `LocalEngine.complete` turns the request into a chat template, generates greedily and stops at DSPy's completion marker. That marker stop is why each call takes about two seconds.
- `evaluate` catches parse failures as `unparsed` instead of crashing, and counts what the model answered, which is how the missing `shipping` shows up.
- `metric` strips and lowercases the answer before comparing, which forgives case and spaces but not a stray `>`.

## The lab

<PromptSearchLab />

The lab replays the printed results of block 3 (the stand-in). With 8 labelled demonstrations selected, dev accuracy is 0.458, test 0.479 and the prompt about 297 tokens; with all 11 candidates tried, the best on dev is 0.375 and its test score 0.375.

**What each control does.**

- **labelled demonstrations** picks one of the sweep sizes from block 3 and shows its dev and test accuracy and average prompt size in the sentence under the chart.
- **candidates tried** reveals the random-search candidates in order: blue is dev accuracy, orange is test accuracy, and the red frame marks the candidate with the best dev score so far.
- **show data** lists every program with its dev and test accuracy.

**Try it yourself.**

1. Step **labelled demonstrations** from 0 to 24. Dev goes from 0.250 to 0.625 and test from 0.250 to 0.771, while the prompt grows from 135 to 618 tokens. Past 12 demonstrations each extra one buys less accuracy and costs 20 tokens on every call.
2. Drag **candidates tried** from 1 to 2. The red frame jumps to seed -2 (dev 0.375, test 0.375). Drag on to 11: it never moves, although seeds 1, 3, 4 and 5 score 0.333 on dev and seed 6 ties at 0.375. The search found nothing better than the starting point because every candidate is a different handful of examples.
3. Compare **labelled demonstrations** = 4 (0.375 dev, 0.375 test, 215 tokens) with the 11-candidate winner (0.375, 0.375). The search spent hundreds of model calls (264 dev evaluations alone) and matched four labelled examples. Then set 12 demonstrations (0.500 dev, 0.583 test): a plain increase in labelled data beat the search.

<Infographic src="/img/afr/dspy-results.svg" alt="A table of dev and test accuracy and prompt tokens for six labelled demonstration counts, a table showing bootstrapping from a weak teacher, and a table of a real small model's results with zero-shot, labelled and bootstrapped demonstrations." caption="Left: the stand-in sweep. Right: the weak-teacher failure. Bottom: the real model, with what it actually answered." />

## Designing with it

| Situation | Start with | Why |
| --- | --- | --- |
| A new task and a few labelled examples | `LabeledFewShot` at several `k`, scored on dev | The cheapest baseline; measure before searching |
| A multi-step program with no labels for the middle | `BootstrapFewShot` with a strong teacher | It is the only one that fills the unlabelled steps |
| 50 or more examples and a metric you trust | `BootstrapFewShotWithRandomSearch`, then `MIPROv2` | Joint search over instructions and demonstrations |
| The failure reasons can be written down | `GEPA` with a metric that returns feedback text | It learns from why an answer failed, not only whether |
| A cheap model needs to match an expensive one | Optimise the cheap model's prompt, then compare on test | The docs' stated reason; check it on your data |
| The model changed | Re-run the search and compare on the same test set | Prompts are tied to a model |

Rules of thumb. Measure the zero-shot baseline first, because it often shows a format problem and not a reasoning problem. Keep the test set out of every decision. Look at a handful of compiled prompts and outputs by eye; a metric can be gamed. Treat score differences smaller than one example as ties. And budget: every candidate costs a pass over the dev set, so a search of 11 candidates on 24 examples is 264 model calls before the test run.

For metrics and evaluation sets see the [evaluation workflow](/docs/llm-evals/evaluation-workflow) and [custom model evals](/docs/llm-evals/custom-model-evals). For hand-written agents that this can sit on top of, see [create_agent](/docs/genai/langchain-advanced/create-agent), and for fine-tuning as the alternative, [supervised fine-tuning with LoRA](/docs/llm-engineering/supervised-fine-tuning-with-lora).

## Where this stands in 2026

:::info Industry view
DSPy's documentation now leads with GEPA for prompt optimisation, and describes the workflow as: give a training set and a metric, let the optimiser generate instruction variations with a model, run the examples, keep the best. Its own pages claim that an optimised small model can match a hand-prompted large one, quoting a Shopify and a Dropbox case; I treat those as the project's claims. The library itself is moving: 3.4 changes how custom models plug in and schedules removals for 3.5, so pin the version and read the migration guide before upgrading.

Not settled: how much of the gain from instruction optimisation survives a model upgrade, how to stop an optimiser overfitting a small dev set, and how to write metrics that do not get gamed. The papers report gains on their tasks; the only gains this chapter measured are the stand-in's and a 0.5-billion model's, which say little about frontier models.
:::

## Common mistakes

- **Tuning on the test set.** It feels like checking your work. Once the test score informs a choice it is a dev score, and it will flatter you. Keep it sealed until the end.
- **Trusting a one-answer difference.** On 24 dev examples, 0.375 against 0.333 is one message. Treat those as ties, enlarge the dev set, or look at the test scores.
- **Bootstrapping from a teacher that is wrong most of the time.** It feels automatic. In block 3 every bootstrapped demonstration was `billing`. Check which classes your demonstrations cover, and use a stronger teacher or labelled examples.
- **Optimising before measuring zero-shot.** It feels like the point of the tool. Zero-shot showed 24 of 32 answers with a stray character, which no instruction search would have explained. Look at raw outputs first.
- **Expecting demonstrations to fix a model that is too small.** It feels like prompt engineering can do anything. The 0.5-billion model never answered `shipping` with any demonstrations. Change the model, not only the prompt.

## Practice questions

<details>
<summary><strong>Easy.</strong> In the worked example, why is "tracking stuck soon" answered `billing` with three demonstrations?</summary>

It shares only the word "soon" with the closest demonstration, "charged twice soon". The stand-in needs two shared words to copy a label, so it falls back to its default, `billing`. The fourth demonstration, "tracking stuck please", shares "tracking stuck" and fixes it.

</details>

<details>
<summary><strong>Easy.</strong> What are the three things you must supply to run an optimiser, and what is the output?</summary>

A program (signature and module), examples with the right answers (split into train, dev and test), and a metric. The output is the same program with better parameters attached: demonstrations and possibly rewritten instructions. It can be saved to a file and loaded later.

</details>

<details>
<summary><strong>Medium.</strong> Block 3 shows `BootstrapFewShot` with up to 4 demonstrations scoring 0.250 while 4 labelled demonstrations score 0.375. Why, and what would you change?</summary>

The zero-shot stand-in answers `billing` to every message, so the only traces that pass the metric are billing messages, and all four bootstrapped demonstrations are billing. The program is shown one class and still answers `billing` elsewhere. Remedies: use a teacher that is right on more classes (a program with labelled demonstrations, or a stronger model), raise `max_bootstrapped_demos` so other classes get in (12 covered all four queues, though it scored 0.333 on dev), or mix labelled demonstrations that you know cover every class.

</details>

<details>
<summary><strong>Medium.</strong> A search tries 11 candidates on a 24-example dev set and the best scores 0.375 where the next scores 0.333. Is the best really better?</summary>

Not demonstrably. With 24 examples, 0.375 is 9 correct and 0.333 is 8 correct: one answer. The same candidates differ on the test set in the other direction (seed 6 scored 0.375 on dev and 0.333 on test). To decide, use a larger dev set, average over several splits, or compare on the test set once and report the uncertainty. A one-example gap is a tie.

</details>

<details>
<summary><strong>Stretch.</strong> In block 5 the 0.5-billion model scores 3 of 32 zero-shot and 11 of 32 with 8 labelled demonstrations, yet never answers `shipping`. Separate the part of the gain that is about format from the part that is about the task.</summary>

In the zero-shot answers, 24 of 32 contain stray characters before or around the label, so many correct labels are lost to the exact-match metric. The prompt with demonstrations removes them, so part of the 8-answer gain is parsing. The rest is the model picking labels from the demonstrations, but it still gives only three of the four labels and no `shipping`, so the task is not learned. To separate the two, score with a lenient metric (label contained in the output) as well as the strict one. If the lenient zero-shot score is much higher, the first gain was format.

</details>

<details>
<summary><strong>Stretch.</strong> You have 40 labelled support tickets, a small model that handles them poorly, and a strong model you can afford for a few hundred calls. Outline a plan using what this chapter covers.</summary>

Split the tickets (for example 20 train, 10 dev, 10 test) and measure zero-shot and labelled few-shot on dev first, reading raw outputs for format problems. Use the strong model as the teacher in `BootstrapFewShot` so that passing traces cover all classes, and check the class mix of the demonstrations. If the metric can explain failures in words, try `GEPA` with the strong model as reflection model and the small model as the student. Compare finalists once on test, treating differences of one ticket as ties. Pin the library and model versions, save the compiled program, and keep the test set for the re-run when either changes.

</details>

## Go deeper

All opened on 8 October 2026.

- DSPy documentation, `current` pages: getting started (expanding signatures, GEPA optimisation), the API reference for signatures, modules and optimisers (`BootstrapFewShot`, `MIPROv2`, `GEPA`), and the DSPy 3.4 LM migration guide (custom engines, deprecations scheduled for 3.5). The `dspy` 3.4.0 package, installed from PyPI.
- Omar Khattab and colleagues, "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines", arXiv 2310.03714, submitted 5 October 2023.
- Krista Opsahl-Ong and colleagues, "Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs" (MIPRO), arXiv 2406.11695, submitted 17 June 2024, revised 6 October 2024.
- Lakshya A. Agrawal and colleagues, "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning", arXiv 2507.19457, submitted 25 July 2025, revised 14 February 2026, ICLR 2026 oral.
- Qwen2.5-0.5B-Instruct from the Hugging Face hub, used as the real model in block 5.
- On this site: [Context engineering](/docs/agentic-frontier/context-engineering) (what a prompt costs), [evaluation workflow](/docs/llm-evals/evaluation-workflow), [LLM-as-a-judge methods](/docs/llm-evals/llm-eval-methods), [prompt, retrieve or fine-tune](/docs/llm-engineering/prompt-retrieve-or-fine-tune), [tuning embedding models and rerankers](/docs/llm-engineering/tuning-embedding-models-and-rerankers).

**Not verified here.** MIPROv2, GEPA, COPRO, SIMBA and `BootstrapFinetune` were not run; descriptions come from the documentation and papers. No frontier model was used, so no claim is made about gains on one. The Shopify and Dropbox figures are the documentation's claims. Block 5's numbers are for one small model with greedy decoding; a different model, seed or prompt would move them. I did not look up when DSPy 3.4.0 was released.

## Check yourself

- I can describe a DSPy program as signature, module, examples and metric, and say what an optimiser changes.
- I can run and compare labelled, bootstrapped and random-search programs on a dev set and a held-out test set.
- I can explain why bootstrapping from a weak teacher gives biased demonstrations and what to do about it.
- I can read raw outputs to separate a format failure from a task failure.
- I can tell when a difference between two candidates is smaller than the noise of the dev set.

## Where to go next

This is the last chapter in this group. Back to the start: [Context engineering](/docs/agentic-frontier/context-engineering), where the prompt an optimiser builds has to fit in a budget. Related: [LLM evaluation workflow](/docs/llm-evals/evaluation-workflow), because an optimiser is only as good as its metric.
