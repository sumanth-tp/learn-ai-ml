---
id: agentic-course-llm-evaluation
title: "08. Chatbot and RAG Evaluation with LangSmith and LLM as a Judge (Complete Agentic AI Course in 10 Hours)"
sidebar_label: "8 - LLM evaluation"
sidebar_position: 8
slug: /projects/agentic-ai-complete-course/llm-evaluation
description:
  "Measure a chatbot and a RAG app instead of trusting your eyes: build LangSmith datasets, write LLM-as-a-judge evaluators for correctness, relevance, groundedness and retrieval relevance, run experiments, compare models and read the scores."
tags:
  [
    agentic-ai,
    evaluation,
    llm-as-a-judge,
    langsmith,
    rag,
    datasets,
    experiments,
    groundedness,
  ]
---

import Infographic from '@site/src/components/Infographic';

> **Part 8 of 9** ·
> [Watch on YouTube](https://www.youtube.com/watch?v=rV3HJ4LEZ7k&t=33775s) ·
> 9:22:55 to 10:30:25 · Notebook: `RAG-Tutorials/1-rag_evaluation.ipynb` (the
> instructor's RAG tutorials folder). Notes follow the video in order.

This chapter shows how to put numbers on an LLM application: you build a small
test dataset in LangSmith, let a second LLM grade your app's answers, and read
the scores for a plain chatbot first and then for a RAG pipeline.

## The end of the guardrails demo (9:22:55)

The first half-minute of this section is the tail of the previous chapter. The
instructor finishes the healthcare chatbot, the project that combines every
guardrail middleware, says he will explain it properly in a separate video
later, and signs off. Nothing here belongs to evaluation, so the real topic
starts at 9:23:20.

## What this module is about (9:23:20)

He introduces the module as a response to what viewers asked for most: how do
you apply evaluation metrics to whatever you built, whether that is a chatbot, a
RAG application or an agentic workflow? His point is blunt: for any LLM
application, some evaluation technique is not optional.

The plan for the module, in his words turned into a list:

- It runs for roughly an hour of video and is itself a series of short videos.
- It uses **LangChain** for the application code and **LangSmith** as the place
  where datasets, runs and scores are stored and displayed. He reminds you that
  earlier parts of the course already touched LangGraph and LangSmith.
- This first series is the starting point. Later series are meant to cover
  things like building datasets in a more systematic way, regression testing
  and human annotation. They are announced, not shown here.
- The code and written material are shared in the video description, and he
  asks you to implement everything yourself.

### The LangSmith pages he opens (9:24:30)

Before any code he shows the LangSmith evaluation pages on the LangChain
website. The headline reads "Harden your application with LangSmith
evaluation", with the sub-line that you should not ship on vibes alone and
should measure your LLM application by testing it across its development
lifecycle. Scrolling down he reaches a "Dataset construction" section: a strong
testing framework starts with a reference dataset, which is often a tedious job,
and LangSmith lets you save debugging and production traces into datasets. The
example on screen is a "Chat extraction examples" table of input and output
pairs, for instance a question about the largest planet in the solar system next
to the expected answer. That table is exactly the shape of the dataset he
builds later: an input column and a reference output column.

## Why evaluate a chatbot at all (9:25:30)

He switches to an Excalidraw page and draws the simplest possible system: an
input goes into a chatbot and an output comes out. Then he lists the questions
that appear the moment you have such a chatbot.

1. **Which LLM should I use?** You could pick an OpenAI model, a Google Gemini
   model or an open-source model served by Groq. Cost is one factor, but he
   stresses that accuracy for your specific use case matters more than price.
2. **How do I decide an LLM is good enough for this use case?** You need a
   ground truth. For a given input there must be an expected output, so the
   response the model generates can be compared against it. That is why you
   need data shaped as "for this input, this output".
3. **Which evaluation metrics?** And a follow-on question: who actually does the
   comparing? His answer for this video is **LLM as a judge**: a second LLM,
   driven by a prompt, compares the generated output with the expected one and
   produces a score. For RAG he will show the same idea with a different set of
   metrics.

<Infographic
  src="/img/agentic-course/08-chatbot-eval-plan.svg"
  alt="The chatbot evaluation plan: input to chatbot to output, data holding input and ground-truth output, three questions about which LLM, ground truth and metrics, LLM as a judge with a prompt tracked in LangSmith, and four numbered steps."
  caption="Redrawn from the instructor's Excalidraw page at 9:30:30."
/>

He then writes the steps he will follow, which become the skeleton of the first
half of the chapter:

| Step | What happens | Where it appears below |
| --- | --- | --- |
| 1 | Gather data points: pairs of an input and the output you expect | Datasets in LangSmith |
| 2 | Build an LLM as a judge, a prompt that grades each answer | Define metrics |
| 3 | Apply the evaluation metrics to your chatbot's outputs | Run evaluations |
| 4 | Compare several LLM models and keep the one with the best result | Compare models |

He uses LangSmith because the tracking can be done entirely in the LangChain
cloud, so every run is visible there. (He says "LangGraph cloud" at one point;
the product he is using is LangSmith.)

## Setup (9:30:40)

He opens the notebook, whose first markdown cell is the tutorial introduction:
a short definition of RAG, the three things you will learn (how to create test
datasets, how to run the application on them and how to measure it with
different metrics), and the three-step overview of an evaluation workflow
(create a dataset of questions and expected answers, run your app on those
questions, use evaluators to score factors such as answer relevance, answer
accuracy and retrieval quality). The tutorial's running example is a bot that
answers questions about three blog posts by Lilian Weng, which you will meet in
the RAG half.

Then he adds a heading, "Chatbot Evaluation", and installs the two libraries he
needs.

### Installing with uv (9:31:00)

```bash
uv add langsmith openai
```

He first types the package name wrongly (`langmith`), and `uv` refuses with a
"No solution found when resolving dependencies" error: the misspelt package is
not in the registry, so the project's requirements cannot be satisfied. The help
text even offers a `--frozen` flag, but the right fix is just to spell it
`langsmith`. The corrected command resolves 253 packages and audits 232. Check
the name before you chase an environment problem.

### LangSmith account and API key (9:32:00)

- Search for LangSmith and sign up. The tagline he reads out: a unified
  observability and evaluation platform where teams can debug, test and monitor
  AI application performance, whether or not they build with LangChain.
- The workspace home page has "Get started" shortcuts (set up tracing, run an
  evaluation, try out the playground) and a table of tracing projects. The left
  sidebar groups the product into **Observability** (tracing projects,
  monitoring), **Evaluation** (datasets and experiments, annotation queues),
  **Prompt engineering** (prompts, playground) and **LangGraph Platform**
  (deployments).
- To get a key: Settings, then API Keys, then create one. He names it
  `evaluation`. The dialog asks for a description, a key type (personal access
  token or service key) and an expiry (never, 30 days, 90 days, one year or
  custom).
- He copies the key into his `.env` file as `LANGSMITH_API_KEY`, next to the
  `OPENAI_API_KEY` he already had.

:::danger Never show your .env
His `.env` file, with live keys, is on screen for several seconds. Do not copy
that habit. Keep real values out of recordings, screenshots and commits, and
rotate any key that has been displayed. In your own file the lines are simply
`OPENAI_API_KEY=...` and `LANGSMITH_API_KEY=...`.
:::

### The first notebook cell (9:33:20)

```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
```

What each line does:

- `load_dotenv()` reads the `.env` file, and `os.getenv` pulls the two keys out
  so they can be copied into the process environment, where the OpenAI and
  LangSmith libraries look for them.
- `LANGSMITH_TRACING="true"` switches on tracing, so every wrapped call and
  every decorated function from here on is recorded in LangSmith without any
  extra code.

:::note Environment variable names
The video uses `LANGSMITH_TRACING` and `LANGSMITH_API_KEY`, which are the
current names. Older tutorials use `LANGCHAIN_TRACING_V2` and
`LANGCHAIN_API_KEY`; they still work in some versions but prefer the new ones.
:::

## Step 1: create the data points (9:34:40)

Before writing the code he shows where the data will live. In LangSmith,
**Evaluation** contains **Datasets and Experiments**. A dataset is a table of
examples; an experiment is one run of your application over that dataset,
scored by evaluators. The existing list in his account shows datasets from
earlier work (for instance "Lilian Weng Blogs Q&A" and "QA Example Dataset"),
each with an experiment count, an example count and a created date.

### Creating a dataset from code (9:36:40)

The code needs a `Client`, which is the object that talks to LangSmith and
uploads data. He builds the cell live:

1. `client = Client()`.
2. Pick a dataset name and call `client.create_dataset(dataset_name)`. This
   creates an empty dataset.
3. Add rows with `client.create_examples(...)`. While typing he browses the
   other methods on the client (`create_example`, `create_chat_example`,
   `create_annotation_queue` and so on) and reads the docstring: an example is a
   row in a dataset holding an input and an expected output. He starts to type
   the singular `create_example`, then settles on the plural
   `create_examples`, which takes the dataset id (`dataset.id`) and a list of
   examples.
4. Each example is a dictionary with an `inputs` key and an `outputs` key, each
   holding key-value pairs. He generated the five question and answer pairs with
   ChatGPT while reading the documentation.

```python
from langsmith import Client

client = Client()

# Define dataset: these are your test cases
dataset_name = "Chatbots Evaluation"
dataset = client.create_dataset(dataset_name)
client.create_examples(
    dataset_id=dataset.id,
    examples=[
        {
            "inputs": {"question": "What is LangChain?"},
            "outputs": {"answer": "A framework for building LLM applications"},
        },
        {
            "inputs": {"question": "What is LangSmith?"},
            "outputs": {"answer": "A platform for observing and evaluating LLM applications"},
        },
        {
            "inputs": {"question": "What is OpenAI?"},
            "outputs": {"answer": "A company that creates Large Language Models"},
        },
        {
            "inputs": {"question": "What is Google?"},
            "outputs": {"answer": "A technology company known for search"},
        },
        {
            "inputs": {"question": "What is Mistral?"},
            "outputs": {"answer": "A company that creates Large Language Models"},
        }
    ]
)
```

Output (the second and later lines in the original are LangSmith rate-limit
warnings, explained below):

```text
{'example_ids': ['596db3b7-bbbc-46fd-9f94-69bdc0791732',
  '575b5134-ad1f-41de-8269-dbe299794e30',
  '2864cc97-108b-4df9-929c-dc99dd7b82d7',
  'f642340f-a193-4c1b-8289-0962d5e1fadd',
  '01a10cb2-d736-4c95-9dd6-e629b758ef2f'],
 'count': 5}
```

:::note The dataset name conflict
His first run failed with a "conflict for dataset" error, because a dataset
called "Simple Chatbot Evaluation" already existed from an earlier attempt, and
LangSmith will not create two datasets with one name. He changed the name and
reran. The notebook in the repository carries the final name,
`"Chatbots Evaluation"`, which is what the code above uses. The abandoned
"Simple Chatbot Evaluation" dataset stays in his account with zero examples.
Re-running this cell with the same name will fail the same way, so either
change the name or delete the old dataset in the UI.
:::

:::note Rate-limit warnings in the output
The notebook's saved output also contains `LangSmithRateLimitError` messages
(HTTP 429, "tenant exceeded usage limits ... monthly_traces of 100"). That is
the free tier's monthly trace cap, which his account had used up from earlier
videos. Your code still runs, but traces stop being stored. He comments on it
again at 10:04.
:::

### What the dataset looks like in LangSmith (9:40:00)

Refreshing the **Chatbots Evaluation** dataset in the browser shows all five
rows with a created time, a split called `base`, an **Inputs** column and a
**Reference Outputs** column. The reference output is the ground truth he
talked about on the board.

| Input (question) | Reference output (ground truth) |
| --- | --- |
| What is LangChain? | A framework for building LLM applications |
| What is LangSmith? | A platform for observing and evaluating LLM applications |
| What is OpenAI? | A company that creates Large Language Models |
| What is Google? | A technology company known for search |
| What is Mistral? | A company that creates Large Language Models |

He adds an aside worth keeping: this step can be automated. If a team is
annotating inputs and outputs in a spreadsheet, you can read that CSV or Excel
file and push the rows into LangSmith in the same way, instead of typing
examples into a Python list.

He closes this part by recapping what the code did: create a client, choose a
dataset name, create an empty dataset, then add any number of examples to it.

## Step 2: define the metrics, LLM as a judge (9:41:20)

A recap of the board, then the next piece. The model under test produces an
output; the judge is a second LLM call that looks at the question, the expected
answer and the produced answer, and says whether the produced answer is
correct. He builds two evaluators: one graded by an LLM, one by plain Python.

### The wrapped OpenAI client (9:43:20)

He imports `openai` and `wrappers` from `langsmith`, then wraps the client with
`wrappers.wrap_openai(openai.OpenAI())`. Hovering over `wrap_openai` shows its
docstring: it patches the OpenAI client to make it traceable and supports the
chat and responses APIs for both sync and async clients. In practice this means
every call made through `openai_client` shows up as a trace in LangSmith with no
other changes.

He also defines `eval_instructions`, the system prompt for the judge. It casts
the judge as an expert professor whose job is grading students' answers. This
is the "teacher grading a quiz" framing that he reuses in every judge prompt
later.

### The correctness evaluator (9:44:40)

An evaluator in LangSmith is simply a function. This one takes three
dictionaries, which LangSmith fills in by argument name:

- `inputs`: the example's input (here `{"question": ...}`),
- `outputs`: what your application returned (here `{"response": ...}`),
- `reference_outputs`: the ground truth stored in the dataset (here
  `{"answer": ...}`).

It returns a boolean. Inside, the function builds a prompt that names the
question, the real answer and the predicted answer, asks for `CORRECT` or
`INCORRECT`, and sends it to `gpt-4o-mini` at temperature 0 with the system
prompt above. The last line compares the reply with the string `"CORRECT"` to
produce `True` or `False`.

```python
import openai
from langsmith import wrappers
 
openai_client=wrappers.wrap_openai(openai.OpenAI())

eval_instructions = "You are an expert professor specialized in grading students' answers to questions."

def correctness(inputs:dict,outputs:dict, reference_outputs:dict)->bool:
      user_content = f"""You are grading the following question:
    {inputs['question']}
    Here is the real answer:
    {reference_outputs['answer']}
    You are grading the following predicted answer:
    {outputs['response']}
    Respond with CORRECT or INCORRECT:
    Grade:
    """
      response=openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                  {"role":"system","content":eval_instructions},
                  {"role":"user","content":user_content}
            ]
      ).choices[0].message.content

      return response == "CORRECT"
```

Walking through the important lines:

- The `user_content` f-string is the judge's whole view of the world: the
  question, the ground truth and the answer under test. Nothing else.
- `temperature=0` keeps the judge as repeatable as a language model can be.
- `.choices[0].message.content` pulls out the text of the first reply.
- `return response == "CORRECT"` turns the text into a boolean. He explains
  that this is why the prompt insists on a single word.

<Infographic
  src="/img/agentic-course/08-judge-call.svg"
  alt="Three inputs, the question, the real answer and the chatbot's response, are put into one user prompt, sent with a system prompt to the judge LLM gpt-4o-mini, which answers CORRECT or INCORRECT, and a comparison turns that into True or False."
  caption="Explanatory board (not shown in the video): what the first correctness evaluator sends to the judge and how its reply becomes a score."
/>

:::warning This judge is fragile
Because the function tests `response == "CORRECT"`, a reply of `Correct.` or
`CORRECT.` or a short explanation scores `False`, even when the judge meant
"yes". It works here because the prompt is strict and temperature is 0. In the
RAG half he replaces this style with structured output, which removes the
problem. Prefer that version for anything you will rely on.
:::

### The concision evaluator (9:48:20)

The second metric is deliberately simple. It makes no LLM call; it only
compares lengths:

```python
## Concisions- checks whether the actual output is less than 2x the length of the expected result.

def concision(outputs: dict, reference_outputs: dict) -> bool:
    return int(len(outputs["response"]) < 2 * len(reference_outputs["answer"]))
```

If the response is shorter than twice the length of the reference answer, the
function returns 1, otherwise 0. Passing means "not too wordy". He says plainly
that this is one example of a cheap, custom metric you can invent yourself, and
that a response must satisfy the rule to pass.

| Metric | How it is scored | Inputs it reads | Output |
| --- | --- | --- | --- |
| `correctness` (first version) | LLM judge answers CORRECT or INCORRECT | `inputs`, `outputs`, `reference_outputs` | `True` or `False` |
| `concision` | Python: `len(response) < 2 * len(reference answer)` | `outputs`, `reference_outputs` | `1` or `0` |

## Step 3: run the evaluation (9:50:00)

Now the experiment itself. Three pieces are needed: the application, a thin
function that connects each dataset row to the application, and the call to
`client.evaluate`.

### The application under test (9:50:40)

First a default instruction, then a function that stands in for the chatbot.
`my_app` takes a question, a model name and an instruction string, and returns
the model's reply text. The model argument matters later, because it is how he
swaps models.

```python
default_instructions = "Respond to the users question in a short, concise manner (one short sentence)."
def my_app(question: str, model: str = "gpt-4o-mini", instructions: str = default_instructions) -> str:
    return openai_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
    ).choices[0].message.content
```

The default instruction asks for a short, concise answer of one short sentence.
That instruction is also why `concision` is a fair test for this bot.

### The target function (9:51:20)

`client.evaluate` calls your system once per dataset row and passes it that
row's `inputs` dictionary. `ls_target` is the adapter: it picks the question out
of the dictionary, calls `my_app`, and wraps the reply in a dictionary whose key,
`response`, is the key the evaluators read.

```python
### Call my_app for every datapoints
def ls_target(inputs: str) -> dict:
    return {"response": my_app(inputs["question"])}
```

(The type hint says `str` but `inputs` is a dictionary, as the body shows. It is
harmless, but if you copy the pattern, hint it as `dict`.)

### Running it (9:52:20)

While typing the call, he opens the signature tooltip and reads out the
parameters: the target, the data, the evaluators, a summary-evaluators option,
metadata, an experiment prefix, a description, repetitions, concurrency and an
upload flag. Four are needed here:

| Argument | Meaning | Value used |
| --- | --- | --- |
| first argument | your AI system, called once per row | `ls_target` |
| `data` | the dataset to run over | `dataset_name` |
| `evaluators` | the list of metric functions | `[correctness, concision]` |
| `experiment_prefix` | the name stem; LangSmith adds a random suffix | `"openai-4o-mini-chatbot"` |

```python
## Run our evaluation
experiment_results=client.evaluate(
    ls_target, ## Your AI system
    data=dataset_name,
    evaluators=[correctness,concision],
    experiment_prefix="openai-4o-mini-chatbot"
)
```

Output:

```text
View the evaluation results for experiment: 'openai-4o-mini-chatbot-ac3151d5' at:
https://smith.langchain.com/o/.../datasets/.../compare?selectedSessions=...

5it [00:10,  2.04s/it]
```

The `5it` is the progress bar: five rows processed in about ten seconds. (A
`TqdmWarning` about `IProgress` also appears; it only asks you to install
`ipywidgets` and is harmless.) The printed link opens the experiment, and the
experiment name carries the suffix `ac3151d5`, which he tells you to remember.

### Reading the result in LangSmith (9:54:00)

In **Datasets and Experiments** the dataset **Chatbots Evaluation** now shows
one experiment. Opening it lists the experiments with an average score for each
evaluator, a bar chart per metric, and a latency figure. For this first run the
averages are:

- **Correctness: 0.60.** Three of the five answers were judged correct.
- **Concision: 0.40.** Two of the five answers were short enough.

Opening the experiment shows one row per example with the question, the
reference output, the model's output and a coloured score cell for each
evaluator: green for pass, red for fail. He points out that this view lets you
compare what the model said against the ground truth next to the score.

<Infographic
  src="/img/agentic-course/08-dataset-to-experiment.svg"
  alt="A dataset feeds each question to the target function, which calls my_app and returns a response; the response and the dataset's reference outputs go to the correctness and concision evaluators, whose scores form one experiment in LangSmith."
  caption="Explanatory board (not shown in the video): how client.evaluate wires the dataset, the target and the evaluators into one experiment."
/>

The per-row result, redrawn from the LangSmith screen (the green and red chips
are the pass and fail cells):

| Question | Concision | Correctness |
| --- | --- | --- |
| What is LangChain? | pass (1.00) | pass (1.00) |
| What is Google? | fail (0.00) | pass (1.00) |
| What is OpenAI? | fail (0.00) | pass (1.00) |
| What is LangSmith? | pass (1.00) | fail (0.00) |
| What is Mistral? | fail (0.00) | fail (0.00) |

The two averages (0.40 and 0.60) fall straight out of this table. Note how the
two metrics disagree about the same answer: a model can be right but wordy
(Google and OpenAI) or short but wrong (LangSmith). That is the main reason to
use more than one metric.

## Step 4: compare another model (9:55:00)

He now answers his original question, "which model?". He keeps the same
dataset, same evaluators and same `my_app`, and only changes which model the
target passes in. `my_app` already has a `model` parameter, so the change is one
extra argument in `ls_target`:

```python
### Call my_app for every datapoints
def ls_target(inputs: str) -> dict:
    return {"response": my_app(inputs["question"],model="gpt-4-turbo")}
```

Then he runs the evaluation again under a new prefix, so the second experiment
is stored beside the first:

```python
## Run our evaluation
experiment_results=client.evaluate(
    ls_target, ## Your AI system
    data=dataset_name,
    evaluators=[correctness,concision],
    experiment_prefix="openai-4-turbo-chatbot"
)
```

Output:

```text
View the evaluation results for experiment: 'openai-4-turbo-chatbot-ae4ea2ed' at:
https://smith.langchain.com/o/.../datasets/.../compare?selectedSessions=...

5it [00:09,  1.90s/it]
```

When the page refreshes it has two experiments. The comparison, read from the
LangSmith screens:

| Experiment | Model | Concision | Correctness |
| --- | --- | --- | --- |
| `openai-4o-mini-chatbot-ac3151d5` | `gpt-4o-mini` | 0.40 | 0.60 |
| `openai-4-turbo-chatbot-ae4ea2ed` | `gpt-4-turbo` | 0.00 | 1.00 |

<Infographic
  src="/img/agentic-course/08-chatbot-results.svg"
  alt="The LangSmith experiments table with gpt-4o-mini at concision 0.40 and correctness 0.60 and gpt-4-turbo at concision 0.00 and correctness 1.00, with the per-example pass and fail chips for the first experiment."
  caption="Redrawn from the LangSmith result screens at 9:54:45 to 9:57:00."
/>

His commentary stumbles on the numbers: he first expects the new model's
correctness to be 1, then reads 0.6 off the chart (which is the first
experiment's figure) and concludes that if he must choose between the two he
would pick `gpt-4o-mini`. He ends by suggesting you try other models and
GPT versions yourself and watch the results in LangSmith.

:::note Reading the comparison correctly
The screens show something different from the sentence he lands on. On
**correctness**, `gpt-4-turbo` won clearly (1.00 against 0.60). `gpt-4o-mini`
only wins on **concision** (0.40 against 0.00), and, though he does not say so
here, on price. So "choose `gpt-4o-mini`" is a defensible decision only if you
value brevity and cost over accuracy, and it should be stated as that trade-off,
not as "the better model". Two cautions on top: with five examples one flipped
row moves correctness by 0.20, and the judge is itself an LLM, so treat these
numbers as a demonstration of the workflow rather than a benchmark. Also,
`gpt-4-turbo` is an older model that OpenAI has been retiring; substitute any
current model name you have access to.
:::

He sums up the chatbot half: you create data points, make an LLM a judge, define
several evaluation metrics and compare several models, then pick the one you
prefer. The next videos move to RAG, where the data is your own or a company's
documents, so the metrics change.

## RAG evaluation: the plan (9:57:30)

Back in Excalidraw he starts a second page titled RAG evaluation. The three
things to cover are written down:

1. How to create test datasets.
2. How to run the RAG app over those test datasets.
3. How to measure RAG performance, using different evaluation metrics.

He ticks off LangSmith as the tool that will track all three.

### The LangSmith diagram of the four metrics (9:58:40)

To explain which metrics apply to a RAG application, he pastes in a diagram from
the LangSmith documentation. It shows a question going to a search step (a
database icon, fed by a stack of documents), which returns relevant documents
that fill an LLM's context window, which produces an answer. A reference answer
sits to the right of the answer. Four coloured brackets mark the checks:

- **Retrieval relevance:** are the retrieved documents relevant to the question?
  This judges the search step itself.
- **Groundedness:** is the answer grounded in the documents? Did the model stay
  inside the retrieved text, or did it make something up?
- **Correctness:** does the answer match the ground-truth reference answer?
- **Answer relevance:** does the answer address the question that was asked?

<Infographic
  src="/img/agentic-course/08-four-metrics.svg"
  alt="A RAG pipeline from question through search over documents, relevant documents, an LLM and an answer, with four coloured checks: answer relevance from question to answer, retrieval relevance from question to relevant documents, groundedness from documents to answer, and correctness between answer and reference answer."
  caption="Redrawn from the LangSmith documentation diagram the instructor shows at 9:59:00."
/>

He adds that accuracy is the key thing, but that these four give you a more
complete picture of where a RAG application is good or bad.

### Retrieval metrics versus generation metrics

He does not use these two words, but the diagram splits cleanly that way, and
the split is the most useful way to remember it. A RAG app has two stages, so
you measure each stage separately.

| Metric | Stage it judges | Compares | Needs a reference answer? | Question it answers |
| --- | --- | --- | --- | --- |
| Retrieval relevance | Retrieval | retrieved documents vs the question | No | Did the search bring back material about this question? |
| Groundedness | Generation | answer vs the retrieved documents | No | Is every claim supported by what was retrieved? |
| Answer relevance | Generation | answer vs the question | No | Did the model actually answer what was asked? |
| Correctness | Generation | answer vs the reference answer | Yes | Does the answer agree with the ground truth? |

:::note Not from the video
The video's retrieval check is one yes or no judgement per question. Search
teams also use ranked, label-based retrieval metrics such as precision at k,
recall at k and mean reciprocal rank, which need a human-labelled list of the
right documents for every question. They are more exact but cost more to build.
The LLM-judged version used here is a quick approximation.
:::

### The three steps for RAG (10:00:40)

He lays out the work in the order he will do it, and later ticks each one:

| Step | What | Detail |
| --- | --- | --- |
| 1 | Create the RAG | data ingestion, a retriever, and generation |
| 2 | Create test data | a question and its answer; the answer is the ground truth |
| 3 | Create evaluation metrics | the four above, each using an LLM as a judge |

His reasoning for the judge: LLMs are strong and improving all the time, so why
not use one to implement the evaluation metrics?

<Infographic
  src="/img/agentic-course/08-rag-eval-whiteboard.svg"
  alt="The RAG evaluation page: three questions on creating test datasets, running the RAG app on them and measuring performance with different metrics, and three experiment steps, RAG, test data and evaluation metrics as an LLM judge."
  caption="Redrawn from the instructor's Excalidraw page at 10:02:30."
/>

## Step 1: build the RAG (10:02:40)

This part is quick, because he has built RAG several times already in the
course. Three blog posts by Lilian Weng are loaded (on agents, prompt
engineering and adversarial attacks on LLMs), split into small chunks, embedded,
and stored in an in-memory vector store. The URLs come from the LangSmith
documentation, and he says you can use any sources you like.

```python
## RAG
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# List of URLs to load documents from
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)

# Add the document chunks to the "vector store" using OpenAIEmbeddings
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(),
)

# With langchain we can easily turn any vector store into a retrieval component:
retriever = vectorstore.as_retriever(k=6)
```

Line by line, in his terms:

- `WebBaseLoader(url).load()` fetches each page as documents. The nested list
  comprehension flattens the list of lists into one list, `docs_list`.
- `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250,
  chunk_overlap=0)` cuts the text into pieces of about 250 tokens (counted with
  the tiktoken tokenizer) with no overlap.
- `InMemoryVectorStore.from_documents(...)` embeds each chunk with
  `OpenAIEmbeddings()` and keeps the vectors in memory. There is no external
  database, which is why this is quick to set up and gone when the kernel ends.
- `vectorstore.as_retriever(...)` turns the store into a retriever object that
  takes a question and returns documents.

The cell takes a while to run because the pages are long. The output is only a
`USER_AGENT environment variable not set` warning, which asks you to identify
your requests when scraping. Setting `USER_AGENT` to any descriptive string
silences it.

:::note The retriever returns 4 documents, not 6
`as_retriever(k=6)` looks as though it asks for six chunks, but `k` is not a
direct argument of `as_retriever`. It belongs inside `search_kwargs`, and passed
this way it is silently ignored. The notebook's own output proves it: the call
below returns four documents, which is the default. The correct form is
`vectorstore.as_retriever(search_kwargs={"k": 6})`. Nothing downstream breaks,
but your evaluation results then describe a four-chunk retriever, not a
six-chunk one.
:::

### Trying the retriever (10:04:00)

```python
retriever.invoke("what is agents")
```

Output (trimmed to the shape; each item is a `Document` with metadata such as
the source URL and title, and the chunk text):

```text
[Document(id='618dc4b1-...', metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'title': "LLM Powered Autonomous Agents | Lil'Log", ...}, page_content='...'),
 Document(id='1fd0573b-...', ...),
 Document(id='837abf53-...', ...),
 Document(id='97716229-...', ...)]
```

He also points at a red LangSmith rate-limit message under the output and says
free accounts allow a limited number of requests; the evaluation still works for
the demo.

### The generation part (10:04:40)

He wants a chat model, goes to use `llm`, and gets `NameError: name 'llm' is not
defined`, because the notebook kernel had lost it. He shrugs, defines it again
with `init_chat_model`, and shows the object. This is a small on-camera mistake
worth keeping: the later cell depends on `llm`, so define it before `rag_bot`.

```python
from langchain.chat_models import init_chat_model
llm=init_chat_model("openai:gpt-4o-mini")
llm
```

Output:

```text
ChatOpenAI(client=..., model_name='gpt-4o-mini', model_kwargs={}, openai_api_key=SecretStr('**********'))
```

Next the bot itself. `@traceable` from LangSmith, placed on a function, records
every call to it as a trace, in the same spirit as the wrapped OpenAI client.
Inside, `rag_bot` does three things in order: retrieve documents for the
question, join their text into one string and build a system prompt around it,
then call the LLM and return both the answer and the documents.

```python
from langsmith import traceable

## Add decorator
@traceable()
def rag_bot(question:str)->dict:
    ## Relevant context
    docs=retriever.invoke(question)
    docs_string = " ".join(doc.page_content for doc in docs)

    instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions.       Use the following source documents to answer the user's questions.       If you don't know the answer, just say that you don't know.       Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}"""
    
    ## llm invoke

    ai_msg=llm.invoke([
         {"role": "system", "content": instructions},
        {"role": "user", "content": question},

    ])
    return {"answer":ai_msg.content,"documents":docs}
```

Notes on the code:

- The system prompt tells the model to use the supplied documents, to say it does
  not know if the documents do not contain the answer, and to answer in three
  sentences at most.
- The return value is a dictionary with two keys, `answer` and `documents`. The
  documents are returned on purpose: two of the four evaluators need to see them.

<Infographic
  src="/img/agentic-course/08-rag-pipeline.svg"
  alt="Ingestion and retrieval built once from three blog posts through a loader, a token splitter, an in-memory vector store and a retriever, then the rag_bot function that retrieves, joins documents into a prompt, calls the LLM and returns answer and documents."
  caption="Explanatory board (not shown in the video): the RAG app that is being evaluated."
/>

Trying it with a vague question:

```python
rag_bot("What is agents")
```

Output (trimmed):

```text
{'answer': 'Agents refer to autonomous entities, particularly in the context of artificial intelligence, that can perceive their environment, make decisions, and take actions to achieve specific goals. In the documents, agents can be powered by large language models (LLMs) and possess features like memory, planning, and reflection to enhance their behavior and interaction. They are often designed to simulate human-like behavior or solve complex problems in various applications.',
 'documents': [Document(id='618dc4b1-...', metadata={...}, page_content='...'), ...]}
```

He reads the answer and notes that the whole context, the documents, comes back
with it. That completes the first of the three steps: data ingestion, retriever
and generation.

## Step 2: create the test data (10:10:00)

Now the dataset for RAG. As before, a `Client`, a list of examples, a dataset
name and `create_examples`. Each question is about something the three blog
posts answer, and each output is the ground truth he wants the RAG app to
reproduce. The three questions are about how the ReAct agent uses
self-reflection, the types of bias that arise with few-shot prompting, and five
types of adversarial attacks.

```python
from langsmith import Client

client=Client()

# Define the examples for the dataset
examples = [
    {
        "inputs": {"question": "How does the ReAct agent use self-reflection? "},
        "outputs": {"answer": "ReAct integrates reasoning and acting, performing actions - such tools like Wikipedia search API - and then observing / reasoning about the tool outputs."},
    },
    {
        "inputs": {"question": "What are the types of biases that can arise with few-shot prompting?"},
        "outputs": {"answer": "The biases that can arise with few-shot prompting include (1) Majority label bias, (2) Recency bias, and (3) Common token bias."},
    },
    {
        "inputs": {"question": "What are five types of adversarial attacks?"},
        "outputs": {"answer": "Five types of adversarial attacks are (1) Token manipulation, (2) Gradient based attack, (3) Jailbreak prompting, (4) Human red-teaming, (5) Model red-teaming."},
    }
]

### create the daatset and example in LAngsmith
dataset_name="RAG Test Evaluation"
dataset = client.create_dataset(dataset_name=dataset_name)
client.create_examples(
    dataset_id=dataset.id,
    examples=examples
)
```

Output:

```text
{'example_ids': ['22098524-fe71-46c2-b86f-d0c8668808e6',
  '93bc3994-6e14-4bd8-b446-9a6c7eb712be',
  'd8baca57-c7a2-41a0-ae72-0d52b4cda043'],
 'count': 3}
```

Back in LangSmith he opens the new dataset, **RAG Test Evaluation**, and shows
the three rows, with the question under Inputs and the answer under Reference
Outputs. (A banner on that screen says the account has exceeded its monthly
retention-limit usage; that is the free-tier cap again.)

| Input (question) | Reference output (ground truth) |
| --- | --- |
| How does the ReAct agent use self-reflection? | ReAct integrates reasoning and acting, performing actions such as tools like a Wikipedia search API, and then observing and reasoning about the tool outputs. |
| What are the types of biases that can arise with few-shot prompting? | Majority label bias, recency bias and common token bias. |
| What are five types of adversarial attacks? | Token manipulation, gradient based attack, jailbreak prompting, human red-teaming and model red-teaming. |

(The last two reference answers are condensed here from the full sentences in the
code above.) He then ticks step two and says the next video will cover the
evaluators.

## Step 3: the four evaluators (10:12:40)

He adds a markdown heading, "Evaluators or Metrics", and pastes the same
four-metric diagram next to the code, so he can implement the metrics in the
order of the board. Every evaluator follows one recipe, so he builds the first
carefully and then the rest quickly.

<Infographic
  src="/img/agentic-course/08-four-evaluators.svg"
  alt="A table of the four evaluators, correctness, relevance, groundedness and retrieval relevance, showing which part of the RAG app each judges, what it compares, which fields it reads, which judge model it uses and which key it returns."
  caption="Explanatory board (not shown in the video): the four evaluators of this chapter compared."
/>

### Metric 1: correctness, response against reference answer (10:13:40)

The notebook cell opens with a short markdown note, which he reads out:

- **Goal:** measure how similar or correct the RAG chain's answer is relative to
  a ground-truth answer.
- **Mode:** requires a ground truth, the reference answer, supplied through the
  dataset.
- **Evaluator:** use an LLM as a judge to assess correctness.

Because the judge must reply in a fixed shape, he first defines the shape, a
`TypedDict`. A `TypedDict` is a dictionary type with named fields, and
`Annotated` lets each field carry a description that LangChain passes to the
model as part of the schema. He writes two fields: an explanation and a
boolean. The comment in the code explains the order: models write their fields
in the order they are listed, so putting the explanation first makes the model
reason before it commits to the verdict.

Then the grading prompt, again in the teacher grading a quiz style. It gives
the judge three criteria: grade only on factual accuracy relative to the ground
truth, reject conflicting statements, and accept extra information as long as it
does not contradict the ground truth. True means all criteria are met. The
prompt asks for step-by-step reasoning and tells the judge not to state the
answer first.

Then the judge model. He switches from `init_chat_model` to `ChatOpenAI`, because
only the class gives him `with_structured_output`, which makes the model reply
in the shape of `CorrectnessGrade`. The options he picks are `method="json_schema"`
and `strict=True`, which tell the API to follow the schema exactly.

Finally the evaluator function. Its signature matches the chatbot one: `inputs`,
`outputs` and `reference_outputs`, returning a boolean. It packs the question,
the ground truth and the student answer into one message, calls the judge, and
returns the `correct` field.

#### The error on camera, and the fix (10:26:00)

When he first wrote the schema, he typed the description as the second item
inside `Annotated` (spelling tidied here; this is the broken form, not the notebook's):

```python
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, "Explain your reasoning for the score"]
    correct: Annotated[bool, "True if the answer is correct, False otherwise."]
```

Everything seems fine until the evaluation runs. Much later, at 10:26, running
the RAG experiment fails with an "Error running evaluator" message for the `correctness` evaluator, and the cause he reads is an invalid schema in `CorrectnessGrade`.
His diagnosis: the description needs a different slot. In the three-item form
`Annotated[type, default, description]`, the middle item is the default value,
and `...` (Python's `Ellipsis`) means "no default, required". With only two items
the text lands in the default slot, which an OpenAI strict JSON schema does not
accept. The fix is to insert `...` as the middle item, giving the version in the
notebook below. (The video's error text is only partly legible, so the
explanation of why the two-item form fails is the standard behaviour rather
than a line-by-line reading of the traceback.)

```python
from typing_extensions import Annotated,TypedDict

## Correctness Output Schema

# Grade output schema
class CorrectnessGrade(TypedDict):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

## correctness prompt

correctness_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

from langchain_openai import ChatOpenAI

grader_llm=ChatOpenAI(model="gpt-4o-mini",temperature=0).with_structured_output(CorrectnessGrade,
                                                                         method="json_schema",strict=True)
## evaluator
def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""

    # Run evaluator
    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions}, 
        {"role": "user", "content": answers}
    ])
    return grade["correct"]
```

:::warning This redefines correctness
Notebook cell 22 defines a new `correctness` function with the same name as the
chatbot one. The RAG version reads `outputs["answer"]`, while the chatbot
version reads `outputs["response"]`. If you re-run the chatbot experiments after
this cell, they will fail with a `KeyError`. Either re-run the earlier cell
first, or give one of the two functions a different name.
:::

### Metric 2: relevance, response against input (10:20:40)

The notebook note: the flow is the same as above, but it looks only at the inputs
and outputs, with no reference outputs. Without a reference answer you cannot
grade accuracy, but you can still grade relevance, that is, whether the model
addressed the user's question. This is the metric to reach for when you have a
question but no ground truth.

He builds it by copying the same structure: a `RelevanceGrade` class, a prompt
(the answer must be concise and relevant, and must help to answer the question),
a judge model and a function. Two differences from correctness: the judge here is
`gpt-4o`, and the function takes only `inputs` and `outputs`, since there is no
reference.

```python
# Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]

# Grade prompt
relevance_instructions="""You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM
relevance_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(RelevanceGrade, method="json_schema", strict=True)

# Evaluator
def relevance(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer helpfulness."""
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions}, 
        {"role": "user", "content": answer}
    ])
    return grade["relevant"]
```

### Metric 3: groundedness, response against retrieved documents (10:22:45)

The note: another useful way to evaluate answers without reference answers is to
check whether the response is justified by, or grounded in, the retrieved
documents. This is the hallucination check. The prompt gives the judge the
retrieved text as FACTS and tells it the answer must not contain hallucinated
information outside the scope of the facts.

The evaluator receives `outputs` containing both `answer` and `documents`, so
that is why `rag_bot` returned both. It joins the documents' text into the FACTS
string.

```python
# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "Provide the score on if the answer hallucinates from the documents"]

# Grade prompt
grounded_instructions = """You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM 
grounded_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(GroundedGrade, method="json_schema", strict=True)

# Evaluator
def groundedness(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer groundedness."""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke([{"role": "system", "content": grounded_instructions}, {"role": "user", "content": answer}])
    return grade["grounded"]
```

### Metric 4: retrieval relevance, retrieved documents against input (10:24:20)

The last one judges the retriever, not the model. It gives the judge the question
and the retrieved text and asks only whether the facts are related to the
question. The prompt is lenient on purpose: the aim is to catch facts that are
completely unrelated, so any keyword or semantic link counts as relevant, and
some unrelated material among relevant material is fine.

```python
# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the retrieved documents are relevant to the question, False otherwise"]

# Grade prompt
retrieval_relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a set of FACTS provided by the student. 

Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM
retrieval_relevance_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """An evaluator for document relevance"""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"

    # Run evaluator
    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions}, 
        {"role": "user", "content": answer}
    ])
    return grade["relevant"]
```

His summary of the pattern: it is all about prompting an LLM. If you can do
correctness, you can do the others.

| Evaluator | Reads from `inputs` | Reads from `outputs` | Reads from `reference_outputs` | Judge model | Returned key |
| --- | --- | --- | --- | --- | --- |
| `correctness` | `question` | `answer` | `answer` | `gpt-4o-mini` | `correct` |
| `relevance` | `question` | `answer` | none | `gpt-4o` | `relevant` |
| `groundedness` | none | `documents`, `answer` | none | `gpt-4o` | `grounded` |
| `retrieval_relevance` | `question` | `documents` | none | `gpt-4o` | `relevant` |

## Run the RAG evaluation (10:25:00)

The target wraps `rag_bot`, and `client.evaluate` is called with the new dataset
and all four evaluators. Two arguments are new: `metadata`, which tags the
experiment with free-form notes you can filter on later, and the pandas call at
the end to look at the results locally.

```python
def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])

experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[correctness, groundedness, relevance, retrieval_relevance],
    experiment_prefix="rag-doc-relevance",
    metadata={"version": "LCEL context, gpt-4-0125-preview"},
)
# Explore results locally as a dataframe if you have pandas installed
experiment_results.to_pandas()
```

First attempt, as above: the run fails because of the schema error, which he
fixes (see the correctness section). Second attempt: `client.evaluate` runs the
three questions, and each one costs a RAG call and four judge calls, which is why
it takes noticeably longer than the chatbot run (the progress bar on screen
reads `15.42s/it`, the saved notebook `13.64s/it`).

:::note The metadata label is a leftover
The `metadata={"version": "LCEL context, gpt-4-0125-preview"}` value is copied
from the LangSmith documentation example. This bot does not use LCEL and its
model is `gpt-4o-mini`, so the label is wrong for this run. Metadata does not
change behaviour, but a wrong label misleads you later when you filter
experiments, so write what is true, for example the model and the retriever
settings.
:::

Output:

```text
View the evaluation results for experiment: 'rag-doc-relevance-aea19ef6' at:
https://smith.langchain.com/o/.../datasets/.../compare?selectedSessions=...

3it [00:40, 13.64s/it]
```

The `to_pandas()` call returns a DataFrame with the columns `inputs.question`,
`outputs.answer`, `outputs.documents`, `error`, `reference.answer`, one
`feedback.<name>` column per evaluator, `execution_time`, `example_id` and `id`.
The feedback columns from the saved notebook:

| Question | `feedback.correctness` | `feedback.groundedness` | `feedback.relevance` | `feedback.retrieval_relevance` |
| --- | --- | --- | --- | --- |
| How does the ReAct agent use self-reflection? | True | False | True | False |
| What are the types of biases that can arise with few-shot prompting? | True | True | True | True |
| What are five types of adversarial attacks? | True | True | True | True |

He wonders aloud whether pandas is installed before running the cell. If
`to_pandas()` complains, install it with `uv add pandas`.

### Reading the result in LangSmith (10:27:20)

The experiment appears under **RAG Test Evaluation** with one bar chart per
evaluator and a table row for the experiment. He reads the aggregates out:
relevance 1.0, correctness 1.0 and groundedness around 0.5. Opening the
experiment gives, per row, the input, the reference output, the generated output
and one coloured cell per evaluator, plus the latency, the token count and the
cost of each run, which he calls really useful.

<Infographic
  src="/img/agentic-course/08-rag-results.svg"
  alt="A results table for the rag-doc-relevance experiment with three questions and four evaluators; the first question fails groundedness and retrieval relevance, the other two pass everything; averages are 1.00, 0.67, 1.00 and 0.67."
  caption="Redrawn from the LangSmith result screens at 10:28:00 to 10:28:30."
/>

:::note Why 0.5 on the chart and 0.67 in the table
The groundedness bar he reads out at 10:28 shows 0.50, and the retrieval
relevance cell next to it shows 0.00, because that screen was captured while the
last judge calls were still finishing. The finished experiment, shown a few
seconds later and matching the notebook's DataFrame, has groundedness and
retrieval relevance at 0.67 each (two of three rows pass). When you read live
results, wait for every row to complete before drawing a conclusion.
:::

## How to read the scores

He stops at "this is how you decide", so here is the reading that the table
supports. Treat the four columns together.

- **Correctness 1.00, relevance 1.00.** On these three questions the answers were
  right and on topic.
- **Groundedness 0.67 and retrieval relevance 0.67.** They fail on the same row,
  the ReAct self-reflection question. The bot's answer matched the reference, yet
  the judge found it not supported by the retrieved text, and judged the
  retrieved text off-topic. Two readings are plausible: the retriever fetched
  weak chunks for that question and the model answered from its own knowledge, or
  the judge was strict. The way to find out is to open that row's trace and read
  the documents.
- The lesson: a correct answer is not the same as a **trustworthy RAG answer**.
  If groundedness is low while correctness is high, the model may be succeeding
  without using your documents, and that will fail on questions that depend on
  private data.

<Infographic
  src="/img/agentic-course/08-reading-scores.svg"
  alt="A diagnosis table: retrieval relevance False means the search fetched the wrong chunks; groundedness False with retrieval True means the model added facts; correct but not grounded means the answer came from model memory; correctness False with grounded True means missing documents or a wrong reference; relevance False means the answer drifts."
  caption="Explanatory board (not shown in the video): reading the four scores together to find what to fix first."
/>

:::note Not from the video
Four general cautions for LLM-as-a-judge. Judges are themselves LLMs, so they can
be wrong or inconsistent even at temperature 0. A judge from the same model
family as the app can be too kind to it. Three examples is far too small for
conclusions; real datasets need dozens to hundreds of questions. And spot-check
the judge by reading some rows yourself, which is what the human annotation
series he announces is for.
:::

## Wrapping up RAG evaluation (10:29:00)

He recaps what the module did. Using LangChain and LangSmith, he built a RAG
pipeline, a test dataset and four custom LLM-as-a-judge evaluators, one per
metric on the LangSmith diagram: correctness, groundedness, retrieval relevance
and answer relevance. You can add more metrics for any extra technique you put
into your RAG pipeline. His stated aim was to show how to evaluate a chatbot and
a RAG application by writing your own metrics, and he promises further videos in
the series.

## What comes next (10:30:00)

The next section, covered in the next chapter, switches topic to **LLM
gateways**: what they are, why they belong in any application that calls LLMs,
and a practical implementation. He draws a first picture of a chatbot, a RAG app
and an app all feeding into one provider box with OpenAI, Gemini and Claude
inside.

## What you can now do

- I can explain why a chatbot or RAG app needs an evaluation step and why
  accuracy, cost and conciseness are separate questions.
- I can create a LangSmith dataset from code with `create_dataset` and
  `create_examples`, using `inputs` and `outputs` dictionaries, and know why a
  dataset name must be unique.
- I can write an evaluator as a function of `inputs`, `outputs` and
  `reference_outputs`, and explain how its return value becomes a score.
- I can build an LLM-as-a-judge with a grading prompt and a structured-output
  schema, and explain why the explanation field comes before the verdict.
- I can wrap an app in a target function and run `client.evaluate`, then read
  the averages and per-row results in LangSmith.
- I can compare two models on the same dataset and say which trade-off the
  numbers show rather than just which score is higher.
- I can separate retrieval metrics from generation metrics and name the four
  RAG metrics, what each compares and whether it needs a reference answer.
- I can read a results table by combining correctness, groundedness, relevance
  and retrieval relevance to decide whether to fix retrieval or generation.
