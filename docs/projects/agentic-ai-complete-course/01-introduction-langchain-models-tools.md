---
id: agentic-course-langchain-models-and-tools
title: "01. Introduction, LangChain setup, models and tools (Complete Agentic AI Course in 10 Hours)"
sidebar_label: "1 - LangChain: setup, models, tools"
sidebar_position: 1
slug: /projects/agentic-ai-complete-course/langchain-models-and-tools
description:
  "The course plan, a uv-based LangChain v1 project, your first create_agent
  agent, calling OpenAI, Gemini and Groq models, streaming and batching, and
  building tools by hand with @tool and bind_tools."
tags:
  [
    agentic-ai,
    langchain,
    langchain-v1,
    uv,
    create-agent,
    init-chat-model,
    streaming,
    batch,
    tools,
    bind-tools,
  ]
---

import Infographic from '@site/src/components/Infographic';

> **Part 1 of 9** ·
> [Watch on YouTube](https://www.youtube.com/watch?v=rV3HJ4LEZ7k&t=0s) · 0:00:00
> to 1:06:40 (introduction, then the first hour of the LangChain section) ·
> Notebooks: `updatedlangchain/1-langchainintro.ipynb`,
> `updatedlangchain/2-modelintegration.ipynb`,
> `updatedlangchain/3-tools.ipynb` in the Langchain-V1-Crash-Course repository.
> Notes follow the video in order.
>
> From Krish Naik's *Complete Agentic AI Course In 10 Hours*. Boards captioned
> *Redrawn from…* recreate something the instructor shows on screen. Boards
> captioned *Explanatory board (not shown in the video)* are additions of my
> own.

This chapter takes you from an empty folder to a working LangChain v1 project:
you set it up with `uv`, build a first agent, call three different model
providers, stream and batch their answers, and wire a Python function to a
model as a tool.

## About the whole course (0:00 to 2:31)

The video opens with the instructor, Krish, explaining what the next ten and a
half hours contain and why he made them. The short version: in the previous
four to five months, generative AI and agentic AI have changed a lot, and this
single recording is his attempt to cover the important parts of that change
"in one shot", in the order you would need them.

He lays out the plan aloud. Nine sections follow one another in a single video:

| #   | Section                                                   | Starts                                                                              | What he says it covers                                                                                    |
| --- | --------------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 1   | Introduction                                              | [0:00:00](https://www.youtube.com/watch?v=rV3HJ4LEZ7k&t=0s)                         | This plan, and who the course is for                                                                      |
| 2   | LangChain                                                 | [0:02:31](https://www.youtube.com/watch?v=rV3HJ4LEZ7k&t=151s)                       | Generative AI and agentic AI with the new LangChain version 1: agents, models, tools, messages, memory, middleware |
| 3   | LangGraph                                                 | [2:35:12](https://www.youtube.com/watch?v=rV3HJ4LEZ7k&t=9312s)                      | A complete LangGraph crash course, focused on building agentic AI applications                           |
| 4   | RAG                                                       | [5:02:29](https://www.youtube.com/watch?v=rV3HJ4LEZ7k&t=18149s)                     | How to implement RAG, covering traditional RAG and also agentic RAG                                       |
| 5   | Vectorless RAG                                            | [7:10:43](https://www.youtube.com/watch?v=rV3HJ4LEZ7k&t=25843s)                     | RAG without a vector search step, and how it differs from traditional vector RAG                          |
| 6   | Deep Agents                                               | [8:02:11](https://www.youtube.com/watch?v=rV3HJ4LEZ7k&t=28931s)                     | Deep research agents, with a practical implementation                                                     |
| 7   | Guardrails                                                | [8:45:43](https://www.youtube.com/watch?v=rV3HJ4LEZ7k&t=31543s)                     | The AI security side: keeping an LLM application inside rules                                             |
| 8   | LLM Evaluation                                            | [9:22:55](https://www.youtube.com/watch?v=rV3HJ4LEZ7k&t=33775s)                     | Techniques for evaluating LLM applications, using open-source libraries                                   |
| 9   | LLM Gateways                                              | [10:30:25](https://www.youtube.com/watch?v=rV3HJ4LEZ7k&t=37825s)                    | About 30 to 40 minutes on what an LLM gateway is, plus its implementation                                 |

<Infographic
  src="/img/agentic-course/01-course-map.svg"
  alt="The nine sections of the course grouped into building agents, giving agents knowledge, and making them safe and shippable, each with its start time."
  caption="Explanatory board (not shown in the video): the nine sections from the instructor's plan, grouped by purpose, with the start times he gives."
/>

A few points he makes along the way, in his order:

- **The RAG pair is a comparison, not two separate topics.** After building
  traditional RAG (and its agentic variant), he builds a vectorless version and
  then asks what actually differs between the two approaches.
- **Open-source libraries are preferred** in the evaluation part, so you are not
  tied to one vendor's tooling.
- **The video is a lot, and he knows it.** He says plainly that nobody finishes
  this in a day: expect about a month of study. The payoff he promises is that
  after learning all of it you will be able to answer interview questions on
  these topics.
- **Timestamps are provided** in the video description, so you can jump to a
  section. These notes mirror that: this chapter is part 1, and each later
  chapter takes one slice of the same video.
- **He asks for a like target** (5,000 for this video, and a thousand again at
  the start of the LangChain section). It is not technical, but it tells you the
  audience he has in mind: people following his channel's playlists.

### Who it is for, and what comes with it

The section at 2:31 makes the audience clearer. If you have been following his
LangChain and LangGraph playlists, where he has uploaded many videos and built
end-to-end projects, this course is the refreshed, one-shot version. The reason
for refreshing is LangChain's new major version, **v1**. It changes how agents
are created, changes how memory is applied, and introduces a new idea called
**middleware**. He says he is recording the LangGraph updates in parallel, and
that a newer topic, **Deep Agents**, is also coming, so everything is being
recorded as he goes.

Three code repositories accompany the course: one for the LangChain v1 notebooks
(this chapter uses its `updatedlangchain` folder), one for the LangGraph crash
course, and one for the RAG tutorials. The LangChain repository also holds the
guardrails notebook and the LLM gateway notebook that later parts use.

:::note My addition: what to know first
The video never states prerequisites, but every step is Python code, so basic
Python (functions, f-strings, `for` loops, dictionaries) is the one thing you
need. Everything about LangChain, agents and tools is taught from zero below.
:::

## LangChain version 1 and a tour of the docs (2:31 to 5:00)

LangChain recently released **version 1**, and its documentation changed a lot
with it. The first thing he does is open the LangChain site, click through to
**Docs**, and show that the home page now offers three Python frameworks side by
side: **LangChain**, **LangGraph** and **Deep Agents** (there is a TypeScript
tab as well, which this course does not use). Beneath them sits **LangSmith**,
the platform for observing, evaluating, prompting and deploying what you build.

<Infographic
  src="/img/agentic-course/01-docs-map.svg"
  alt="The LangChain docs home page with LangChain, LangGraph and Deep Agents, and the LangChain sidebar of core components with the ones covered in chapter 1 highlighted."
  caption="Redrawn from the LangChain docs pages shown at 3:45 to 5:00. The highlighted entries are the ones this chapter covers."
/>

His plan is to cover all three modules, in an updated way, so that you always
stay current with the documentation. He then scrolls the LangChain overview and
reads out what changed in the v1 docs, which doubles as the syllabus for the
LangChain half of the course:

| Topic from the docs sidebar | What it is, in plain words                                                  | Where in these notes             |
| --------------------------- | --------------------------------------------------------------------------- | -------------------------------- |
| Agents                      | The new `create_agent` way of building an agent                              | "Your first agent" below         |
| Models                      | How to plug in different model providers                                    | "Model integration" below        |
| Tools                       | How a model calls a function to fetch information or act                     | "Tools" below                    |
| Streaming                   | Showing the answer while it is still being produced (and batching)           | "Streaming and batch" below      |
| Messages                    | The types of message: human, AI, tool (and system)                           | The next chapter                 |
| Short-term memory           | Keeping a conversation's state between turns                                 | Later chapters                   |
| Structured output           | Getting typed, predictable answers out of a model                            | Later chapters                   |
| Middleware                  | New in v1: built-in and custom hooks around the agent loop                   | Later chapters                   |
| Guardrails                  | Checks on what goes into and comes out of the agent                          | Later chapters (and part 7)      |

:::note What "v1" means for old tutorials
In v1, the supported way to build an agent is `langchain.agents.create_agent`.
It replaces the older prebuilt `create_react_agent` that lived in LangGraph's
prebuilt module, and a lot of tutorials online still show the older style. If
an example imports from `langgraph.prebuilt` to make a simple agent, it is the
previous generation. Also, because functionality is moved between packages
across releases, his advice, repeated below, is to work with recent versions.
:::

## Tools for the course: uv and an IDE (5:00 to 7:20)

Two choices before any code.

**An IDE.** He uses **Google Antigravity**, an "agentic" IDE in the same family as
VS Code and Cursor: it has an agent panel that can write code with you, and it
is free for a limited number of requests (not unlimited). He downloads it by
searching for "Google Antigravity" and running the installer (an `.exe` on
Windows; other platforms have their own downloads). Two habits he mentions about it: it offers
a lot of autocomplete suggestions as you type, which is convenient for him, and
he tells you not to rely on them while learning. Type each line yourself so you
understand it. Any editor works. The course needs only a terminal and the
ability to run a Jupyter notebook.

**uv.** He uses the **uv package manager**, an extremely fast Python package and
project manager written in Rust. If you have used `pip` and `venv`, uv does both
jobs and also keeps a record of which versions you installed. The next sections
walk through it step by step.

## Install uv (7:20 to 9:20)

He googles "uv package manager", opens the Astral installation page, and picks the
command for his system. The page offers a standalone installer for macOS and
Linux and one for Windows. On Windows he opens a PowerShell terminal inside the
IDE (the terminal menu lets you choose between PowerShell, Command Prompt and
others) and pastes the command; on macOS or Linux you paste the `curl` command
into any terminal. He had already installed uv, so he only shows the page.

```bash
# macOS or Linux (from the uv installation page)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows: run the PowerShell command from the same page, for example
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Check it worked with `uv --version`. You only install uv once per machine. For
the rest of the video he then switches that same terminal to Command Prompt,
which is why the screen shows `cmd` paths such as `E:\agenticAI\langchainupdated>`;
the commands are identical in any shell.

## Create the project and the virtual environment (9:20 to 12:40)

First he creates an empty folder called `langchainupdated` and opens it in the
IDE. A virtual environment is "always a good practice for any project": it keeps
this project's libraries separate from every other project on the machine.

**1. Initialise the folder as a working repository.**

```bash
uv init
```

This turns the folder into a uv project. It writes `pyproject.toml` (the
project's name, Python version and, later, its dependencies), a
`.python-version` file, a placeholder `main.py`, a `README.md` and a
`.gitignore`. On his machine it picks **Python 3.13**. His point: when Python
releases a newer version, running `uv init` again will pick the newest one, and
it is a good habit to work with recent versions.

**2. Create the virtual environment.**

On camera he first types `uv venv/` with a trailing slash by mistake. uv answers
that `venv/` is an unrecognised subcommand. He leaves the error in the video and
retypes the command correctly:

```bash
uv venv
```

```text
Using CPython 3.13.2
Creating virtual environment at: .venv
Activate with: .venv\Scripts\activate
```

Output: a `.venv` folder appears next to the project files. This is the
environment that will hold the packages.

**3. Activate it.** uv prints the activation command, so you copy it. On Windows
it runs a script inside `.venv\Scripts`:

```bash
# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate
```

Once it works, the terminal prompt gets the project name in front of it, here
`(langchainupdated)`. That prefix is how you know the environment is active.

<Infographic
  src="/img/agentic-course/01-uv-setup-flow.svg"
  alt="Eight steps for setting up a uv project: install uv, uv init, uv venv, activate, requirements.txt, uv add -r, the .env file, and uv add ipykernel."
  caption="Explanatory board (not shown in the video): the terminal walkthrough from 5:00 to 20:00 as one flow, with the commands he typed."
/>

:::tip A shortcut he does not show
`uv add` creates the `.venv` for you if it does not exist, and `uv run python
file.py` runs code inside it without activating anything. The manual
`uv venv` plus activate steps are still worth knowing because the notebook
kernel in the IDE needs to find that `.venv`.
:::

## Install the libraries (12:40 to 16:00)

Installing used to mean `pip install` and then separately remembering which
versions you got. With uv the dependency list is stored in `pyproject.toml`
automatically.

He creates a file named `requirements.txt` and stresses that it must sit
**outside** the `.venv` folder, next to `pyproject.toml`. He lists the libraries
he will need, with **no version numbers**, so that he gets the newest release of
each:

```text
langchain
langchain_community
langchain-openai
langchain-groq
python-dotenv
langchain-google-genai
```

What each one is for, in the order he explains them:

| Package                  | What it gives you                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `langchain`              | The framework itself: `create_agent`, `init_chat_model`, `@tool`, messages                                  |
| `langchain_community`    | A grab-bag of community integrations, including ready-made tools (he says he will need it later)            |
| `langchain-openai`       | The OpenAI integration (`ChatOpenAI`)                                                                       |
| `langchain-groq`         | The Groq integration (`ChatGroq`), because he also wants to use Groq-hosted models                          |
| `python-dotenv`          | `load_dotenv()`, which reads your API keys from a `.env` file                                               |
| `langchain-google-genai` | The Google Gemini integration (`ChatGoogleGenerativeAI`)                                                    |

He wants every example in the course to be shown with several providers, which
is why all three integrations are installed now. Then the install command:

```bash
uv add -r requirements.txt
```

The way he frames it: with plain pip you wrote `pip install -r requirements.txt`
(and uv also supports `uv pip install -r requirements.txt` if you want exactly
that), but `uv add -r requirements.txt` is better because it installs the
packages **and records each one in `pyproject.toml`**. The install prints a few
warnings; he says they can be ignored.

| You want to…                          | Classic pip and venv              | uv                                         |
| ------------------------------------- | --------------------------------- | ------------------------------------------ |
| Create the environment                | `python -m venv .venv`            | `uv venv`                                  |
| Install from a requirements file      | `pip install -r requirements.txt` | `uv add -r requirements.txt` (also records the packages) |
| Add one library                       | `pip install langchain`           | `uv add langchain`                         |
| Keep a record of installed versions   | `pip freeze > requirements.txt`   | Automatic: `pyproject.toml` (and `uv.lock`) |

He then opens `pyproject.toml` to read off what was installed. The versions that
matter, because the rest of the course was recorded against them:

```toml
requires-python = ">=3.13"
dependencies = [
    "ipykernel>=7.1.0",
    "langchain>=1.1.0",
    "langchain-community>=0.4.1",
    "langchain-google-genai>=3.2.0",
    "langchain-groq>=1.1.0",
    "langchain-openai>=1.1.0",
    "python-dotenv>=1.2.1",
]
```

(`ipykernel` is added a few minutes later, below.) **LangChain 1.1.0** is the
version in use. He will hand over this `pyproject.toml` with the course, so you
can see the base versions. Tomorrow's releases will not break your copy, because
you know the baseline. His advice is still to use recent versions: many
functions get deprecated or moved to other libraries as the ecosystem evolves.

Because the `pyproject.toml` also has a `description` line, he mentions you may
fill that in if you like. It is not required.

## API keys and the .env file (16:00 to 18:00)

He needs three keys, one per provider. He creates each in the provider's own
console, shown briefly:

1. **Google API key** from Google AI Studio. In the API Keys page he chooses
   "Create API key", picks a project, and names the key (he calls it
   `langchain updated`).
2. **Groq API key** from the Groq console, again "Create API key". Groq is a
   service that runs open-source models (such as Qwen, which appears below) on
   its own fast hardware.
3. **OpenAI API key** from the OpenAI platform.

He then creates a file named `.env` in the project root and pastes the three
keys in, one `NAME=value` per line. The names are the ones the libraries look
for automatically:

```text
OPENAI_API_KEY=...
GROQ_API_KEY=...
GOOGLE_API_KEY=...
```

:::danger Keys on screen
In the video the `.env` file is briefly visible on screen. Treat any key that has
appeared in a video, a screenshot or a repository as leaked: revoke it in the
console and create a new one. In your own project, add `.env` to `.gitignore`
before the first commit, and never paste a real key into a notebook cell. The
chapter code below always reads keys from the environment.
:::

## Jupyter kernel (17:20 to 19:20)

Notebooks need a **kernel**, the process that actually runs your Python. For the
notebook to see this project's packages, the project needs `ipykernel`:

```bash
uv add ipykernel
```

He recaps the two ways to add libraries, because they are the heart of the uv
workflow:

- `uv add <library name>`: add one library (for example `uv add langchain`;
  he runs this again and uv answers "Resolved 109 packages" and "Audited 103
  packages", because it was already installed).
- `uv add -r requirements.txt`: add everything listed in the file, the same
  idea as a requirements install.

The section summary he gives: a virtual environment, a `requirements.txt` with
the recent libraries, one command to install them all, and `pyproject.toml`
showing exactly what you got. Next he starts on the LangChain docs and agents.

## The first notebook: 1-langchainintro (19:20 to 22:00)

Inside the project he creates a folder `updatedlangchain` (he deletes an earlier
empty one first) and a new notebook in it, `1-langchainintro.ipynb`. Two things
to do before the first cell:

1. **Select the kernel.** In the notebook's kernel picker choose the Python
   environment `.venv` (it shows as `.venv (Python 3.13.2)`). This is why
   `uv add ipykernel` mattered.
2. **Add a markdown heading.** A markdown cell titled "Langchain Version V1",
   plus an "Agents" subheading, run with the play button.

To prove the kernel works he runs a trivial code cell (`1 + 1`). The `.env` file
is already in the folder tree, so everything is ready. The first real cell loads
the keys:

```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
```

Line by line:

- `load_dotenv()` finds the `.env` file and copies each `NAME=value` line into
  the process's environment variables.
- `os.getenv("OPENAI_API_KEY")` reads one of those variables, and the assignment
  to `os.environ[...]` sets it explicitly. Strictly, `load_dotenv()` has already
  done the work and libraries such as `langchain-openai` read the variable
  themselves, so the last line is belt and braces. It does no harm and makes it
  obvious which key the cell is about.
- The cell prints nothing. Silence means success.

:::warning A trap in that last line
If `OPENAI_API_KEY` is missing from `.env`, `os.getenv` returns `None`, and
assigning `None` to `os.environ[...]` raises a `TypeError` (environment values
must be strings). If you hit that error, the key name in `.env` does not match,
or the notebook is not running from a folder where `load_dotenv()` can find the
file.
:::

## Your first agent (22:00 to 37:20)

### What an agent is, drawn on the whiteboard (22:00 to 27:20)

Before writing the agent he switches to an Excalidraw page headed "Agents" and
builds a picture in steps. This is the foundation for everything that follows,
so it is worth reading slowly.

**Step 1: a plain LLM.** Take an LLM, any LLM. It could be an OpenAI model, a
Gemini model, a Groq-hosted model or any open-source model. The input goes in
(for example "write me a 200-word paragraph on artificial intelligence"), and
the output comes out (a 200-word paragraph). This is a simple **generative AI
application**: input, LLM, output. In the early days of generative AI that was
exciting enough. Nowadays everybody talks about **agents**, and he calls them a
very handy and important topic.

**Step 2: where the plain LLM breaks.** Ask the same LLM "give me today's AI
news". An LLM has a **training cut-off date**: it learned from data up to a
point and knows nothing about today. So this is the problem with a plain LLM:
for any question that needs current information, it cannot answer by itself.
He stresses this ("really important for you all to understand").

**Step 3: add a tool.** The LLM needs to depend on some outside **tool**: a
third-party API, a Google search, or anything else that is connected to current
data (here, AI news). With the tool attached, a different thing happens when the
question arrives: the LLM makes a **decision**. It recognises "I cannot answer
this on my own, I must use the tool that can". The tool is called, and what the
tool returns is called the **context**. The LLM then reads the context and writes
the output. The output now answers the question, with fresh information.

That whole arrangement, an LLM that decides when to route a question to a tool,
gets the context back, and then answers, is a **basic agent**. In his words, the
agent can autonomously make simple decisions: when to route, which kind of query
needs what, and how to solve the task.

<Infographic
  src="/img/agentic-course/01-agent-whiteboard.svg"
  alt="Two-step whiteboard: a plain LLM turning input into output, then the same LLM with a tool that returns context, forming a basic agent."
  caption="Redrawn from the instructor's Excalidraw page, drawn between 22:15 and 29:30 (final state at 29:30, shown here in two steps)."
/>

He also writes **ReAct** in the corner of the board. Before LangChain v1, building
an agent was more work: you created an LLM, created a tool separately, linked the
tool to the LLM yourself, and used an architecture called the ReAct architecture
to run the loop. His claim is that creating the same thing has become much
simpler in LangChain v1.

:::note ReAct did not go away
`create_agent` still runs the same reason-then-act loop that ReAct describes: the
model reasons, calls a tool, reads the result, and repeats until it can answer.
What v1 removed is the work of wiring it up by hand. Under the hood the agent is
a compiled **LangGraph** graph, which the notebook draws for you a few minutes
later. So "ReAct" names the pattern, and `create_agent` is how you get it ready-made.
:::

### Creating the agent with `create_agent` (27:20 to 29:20)

He opens a new cell and imports the factory function:

```python
from langchain.agents import create_agent
```

Then he builds an agent. Its inputs are the model, a list of tools and a system
prompt. He notes the model can be supplied in different ways. The simplest is just a name
string (a ready-made model object also works, as the model section shows). For
OpenAI he passes `gpt-5`, the recent flagship OpenAI model at the time. His first attempt
also includes a `verbose=True` argument, which he mentions he will explain
(it gives more information about each invocation):

```python
agent=create_agent(
    model="gpt-5",
    tools=[],
    system_prompt="You are a helpful assistant.",
    verbose=True
)
```

```text
TypeError: create_agent() got an unexpected keyword argument 'verbose'
```

He keeps the error in the video and removes the argument, because
`create_agent` has no `verbose` flag. The lesson for you: if you copy code from
older LangChain tutorials, expect a few arguments that no longer exist. To
watch what an agent did, look at the messages it returns (you will do exactly
that in a few minutes).

:::note A real flag for debugging
`create_agent` does accept `debug=True`, which makes the underlying graph print
each step as it runs. That is the closest thing to what `verbose` meant before.
It is not shown in the video.
:::

The working version, with `tools` empty because no tool exists yet:

```python
from langchain.agents import create_agent

agent=create_agent(
    model="gpt-5",
    tools=[],
    system_prompt="You are a helpful assistant."
)
agent
```

```text
<langgraph.graph.state.CompiledStateGraph object at 0x...>
```

What each argument does:

- `model="gpt-5"`: a string is enough. LangChain looks at the name, works out
  that it is an OpenAI model, and creates the chat model for you. (You meet this
  mechanism properly in the model section below.) It uses `OPENAI_API_KEY`.
- `tools=[]`: the list of functions the agent may call. Empty for now.
- `system_prompt`: standing instructions that go to the model on every call.

The output tells you something important: the thing you get back is a
`CompiledStateGraph` from **LangGraph**. In a notebook, evaluating the variable
also draws it. The picture matches the whiteboard: a `__start__` node, a `model`
node and an `__end__` node. With no tools, the agent is just input, LLM and
output.

### Adding a tool, and the graph changes (29:20 to 32:00)

He defines a plain Python function, `get_weather`. It takes a city as a string
and returns a string:

```python
from langchain.agents import create_agent

def get_weather(city:str)-> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

agent=create_agent(
    model="gpt-5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)
agent
```

Line by line:

- `def get_weather(city:str)-> str:` declares the function with **type hints**:
  the argument is a string and so is the return value.
- The text in triple quotes straight after the `def` line is a **docstring**. He
  calls it important, and it is: the model never sees your code, it sees the
  function's name, its argument names and types, and this docstring.
- The function body simply returns a sentence saying the weather is sunny, for
  any city. He is upfront that this is fake: in a real tool you would call a
  weather API (or a database) here and return what comes back.
- `tools=[get_weather]` hands the function to the agent. LangChain wraps a plain
  function into a tool automatically.

The graph drawn under the cell now changes. Beside `__start__`, `model` and
`__end__` there is a new **`tools`** node, connected to `model` in both
directions.

<Infographic
  src="/img/agentic-course/01-agent-graph.svg"
  alt="Two agent graphs: with no tools the flow is start, model, end; with get_weather there is also a tools node the model can loop through before ending."
  caption="Redrawn from the graph the notebook renders at 29:15 (no tools) and 31:45 (with get_weather)."
/>

The meaning, as he explains it: when a question about the weather arrives, the
model decides to go to the tools node, the tool runs, the result comes back to
the model, and only then does the agent finish. The edges to `tools` and to
`__end__` are the model's choice at run time, which is why the diagram shows two
exits from `model`.

### Running the agent, and the error on the way (32:00 to 36:40)

To run it you call `agent.invoke`. He deliberately tries the most obvious thing
first and leaves the mistake in the video. Here is the sequence, because each
failure is instructive.

**Mistake 1: pass a plain string.**

```python
### run the agent
agent.invoke("What is the weather like in New York?")
```

```text
InvalidUpdateError: Expected dict, got What is the weather like in New York?
```

Why: an agent made by `create_agent` is a graph with a **state**, and that state
is a dictionary with a key called `messages`. `invoke` expects you to provide
that dictionary, not bare text. (He admits he first thought it would work, and
decides to show the error rather than cut it.)

**The fix: a dictionary with a `messages` list.** Each message is a dictionary
with a `role` and `content`. Here the role is `user`, which means "a human
message":

```python
### run the agent
response=agent.invoke({"messages":[{"role":"user","content":"What is the weather like in New York?"}]})
```

This one runs, but it takes a few seconds, because the agent makes more than one
call to the model, as you will see. The cell prints nothing because the result
is assigned to `response`. He says there will be more about message types as
they go on. For now note that this user message is a *human message*, and that
he mentions you may also pass a `HumanMessage` object instead of a dictionary.

**Reading the result.** Printing `response["messages"]` shows the whole
conversation the agent had, in order:

```python
response["messages"]
```

```text
[HumanMessage(content='What is the weather like in New York?', ...),
 AIMessage(content='', ..., tool_calls=[{'name': 'get_weather', 'args': {'city': 'New York'}, 'id': 'call_...', 'type': 'tool_call'}], ...),
 ToolMessage(content='The weather in New York is sunny.', name='get_weather', ...),
 AIMessage(content="It's sunny in New York right now.", ...)]
```

| Step | Message                          | Who produced it           | What it says                                                                 |
| ---- | -------------------------------- | ------------------------- | ---------------------------------------------------------------------------- |
| 1    | `HumanMessage`                   | You                       | The question                                                                 |
| 2    | `AIMessage` with `tool_calls`    | The model (call 1)        | No text yet: instead a request to run `get_weather` with `city='New York'`   |
| 3    | `ToolMessage`                    | The tool, run by the agent | What the function returned: "The weather in New York is sunny."              |
| 4    | `AIMessage`                      | The model (call 2)        | The final answer in natural language                                         |

This is the agent loop from the whiteboard, made visible. The model first decides
to use the tool (step 2). The tool runs and its result becomes context (step 3).
The model then writes the output (step 4). The model was therefore called
**twice**: once to choose the tool, once to turn the result into a sentence.

**How does the model know to call `get_weather`?** He asks it himself. The answer
is the docstring. When the tool is attached, the model is given the function's
name, its arguments and its docstring ("Get the weather for a city"). From that
description it works out that a weather question should be sent to this tool.
That is why a vague or missing docstring leads to a tool that is never chosen.

**Getting just the answer.** To see only the last message's text, take the last
item of the list:

```python
response["messages"][-1].content
```

`[-1]` means "the last message", which is the final `AIMessage`; `.content` is its
text. Remove `[-1]` and you get the entire conversation again, as above. He types
the `[-1].content` version and then goes back to the whole list.

**A second way to pass the input, and a typo that does not matter.**

```python
agent.invoke({"messages":"What is the weather in New Yourk"})
```

Here `messages` is just a string. LangChain treats a string as a single human
message, so this works too: you do not even need to say that it is a human
message. He types "New Yourk" by mistake and leaves it, and the result is nice:
the model fixes the spelling when it calls the tool, so the tool receives
`New York`, and the final answer is along the lines of "It's sunny in New York.
Did you mean New York City? If you want details such as temperature or a
forecast, let me know". It shows a model reading intent, not matching text.

:::note Why a plain string failed but this one worked
`agent.invoke("...")` passes a string where the **state dictionary** should be,
which is what `InvalidUpdateError` complains about. In `{"messages": "..."}` the
string sits inside the `messages` key, where LangChain knows how to turn text
into a message. The top level must always be a dictionary.
:::

The word he uses for this is an **autonomous agent**: the input arrives, the model
decides which tool to call, gets the context, and gives the output, with nothing
in your code saying "if the question is about weather". He points out that this
is one tool, and you can create as many as you like (the next sections show how
tools are really defined).

### Which version are we on? (36:40 to 37:20)

Last, he checks the library version in a cell at the top of the notebook:

```python
import langchain
langchain.__version__
```

```text
'1.1.0'
```

This is the "LangChain 1.1.0" he mentioned from `pyproject.toml`. Agents work
autonomously on the task assigned to them. That closes notebook 1. The next video
section, he says, will cover how to integrate different models and messages.

## Model integration (37:20 to 50:00)

Notebook 2, `2-modelintegration.ipynb`, is headed "Models Integration With
OpenAI, Google Gemini and GROQ". The question it answers: how do you call a
model, in the updated LangChain, and what are the different ways of doing it?
He uses three providers: **OpenAI** (the GPT models), **Google Gemini**, and
**Groq** (which hosts many open-source models).

First cell: load all three keys, one for each provider.

```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
```

All three names must exist in `.env`. The cell prints nothing.

<Infographic
  src="/img/agentic-course/01-model-integration.svg"
  alt="A table of OpenAI, Gemini and Groq with their key name, package, init_chat_model string and provider class, and a flow showing both routes lead to the same model interface."
  caption="Explanatory board (not shown in the video): the six model cells of this notebook as one table, and why they behave the same."
/>

### OpenAI, using `init_chat_model` (40:00 to 43:20)

`init_chat_model` is LangChain's general-purpose way to create a chat model from
a name. Import it from `langchain.chat_models`. He first mistypes the model name,
which is a useful thing to see:

```python
from langchain.chat_models import init_chat_model
model=init_chat_model("gtp-4.1")
model
```

```text
ValueError: Unable to infer model provider for model='gtp-4.1'. Please specify 'model_provider' directly.
```

The model name was spelled `gtp` instead of `gpt`. LangChain guesses the provider
from names it recognises (`gpt-...` is OpenAI), and a name it cannot recognise
produces this error, which tells you to name the provider yourself (for example
`init_chat_model("gtp-4.1", model_provider="openai")`; the real fix here is the
spelling).
He fixes the spelling:

```python
from langchain.chat_models import init_chat_model
model=init_chat_model("gpt-4.1")
model
```

```text
ChatOpenAI(profile={'max_input_tokens': 1047576, 'max_output_tokens': 32768, 'image_inputs': True, ..., 'tool_calling': True, 'structured_output': True, ...}, client=..., model_name='gpt-4.1', ...)
```

Reading the output: the object is a `ChatOpenAI`, so `init_chat_model` chose the
OpenAI class for you. The `profile` is a summary of what this model can do (its
largest input and output sizes, whether it handles images, tool calling and
structured output). The API key is shown masked. He mentions that OpenAI offers
many models (4.1, 4.5 and others) and you only change the name string to switch
between them.

To call the model, use `invoke`. The input can be a plain string, which is
treated as one human message:

```python
## invoke the model
response=model.invoke("Hello How are you?")
response
```

```text
AIMessage(content="Hello! I'm just a program, but I'm here and ready to help you. How can I assist you today?", ..., response_metadata={'token_usage': {...}, 'model_name': 'gpt-4.1-2025-04-14', ...}, usage_metadata={'input_tokens': 12, 'output_tokens': 23, 'total_tokens': 35, ...})
```

The reply is an **`AIMessage`**: whenever a model produces output in LangChain it
arrives as an AI message, and what you type in is a human message. The message
carries both the text (`content`) and metadata such as token counts and the exact
model version that answered. To get only the text:

```python
response.content
```

```text
"Hello! I'm just a program, but I'm here and ready to help you. How can I assist you today?"
```

### Gemini, using `init_chat_model` (42:40 to 45:00)

For Google, he reuses the same function, with a **provider prefix**. The string
has the form `provider:model`:

```python
import os
from langchain.chat_models import init_chat_model

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = init_chat_model("google_genai:gemini-2.5-flash-lite")
response = model.invoke("Why do parrots talk?")
response.content
```

```text
'Parrots don\'t "talk" in the same way humans do, with understanding and intent behind every word. Instead, they are remarkable ...'
```

Notes on this cell:

- `google_genai:` before the model name tells LangChain which integration to use.
  The first call took a while to answer (about 25 seconds on screen), which he
  puts down to it being the first request.
- This is a different model from the one the repository notebook shows
  (`gemini-2.5-flash`): he recorded with the lite version and changes it a few
  minutes later (see below).
- The question, "Why do parrots talk?", is the one he uses for Gemini and Groq.
  The answers differ in style because they are different models.

:::note Pick the prefix on purpose
A bare name such as `gemini-...` may be treated by `init_chat_model` as a Google
Vertex AI model, which needs a different login. With an AI Studio key from
`GOOGLE_API_KEY`, the `google_genai:` prefix is the safe way. Model names also
change often (older ones are retired), so when a name errors, check the
provider's current model list. The pattern, not the exact name, is the lesson.
:::

### The other way: provider classes (44:40 to 47:20)

`init_chat_model` is a convenience. Each provider package also exports its own
class, and you can construct it directly.

```python
### ChatOpenAI

from langchain_openai import ChatOpenAI
model=ChatOpenAI(model="gpt-4.1")
response=model.invoke("Hello How are you?")
response
```

His thought, in order: first check `requirements.txt` to make sure
`langchain-openai` is installed (it is), then import `ChatOpenAI` from
`langchain_openai`, give it the model name, and call it. The answer is similar to
before, just worded a little differently, as models do. He also points out that
`init_chat_model("gpt-4.1")` showed `ChatOpenAI` in its output: `init_chat_model`
is simply choosing this class for you. (While typing, the IDE's autocomplete
offered something like `ChatOpenAI(model_name="gpt-4o", temperature=0)`; he does
not use it.)

For Gemini, the equivalent class lives in `langchain-google-genai`:

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI



model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
response = model.invoke("Why do parrots talk?")
response
```

```text
AIMessage(content='Parrots talk for a fascinating mix of reasons, stemming from their **evolutionary adaptations, social needs, and cognitive abilities.** ...', ...)
```

Same question, same Gemini family, a different object type but the same kind of
answer. He sums up the two ways: use `init_chat_model` with the provider prefix,
or use the provider's own class (`ChatOpenAI` for OpenAI, `ChatGoogleGenerativeAI`
for Gemini).

### Groq (47:20 to 49:20)

The third provider, Groq, follows the same two routes. The first uses
`init_chat_model` with a `groq:` prefix, here with Alibaba's open-source
**Qwen 3** model (32 billion parameters) which Groq hosts:

```python
import os
from langchain.chat_models import init_chat_model

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = init_chat_model("groq:qwen/qwen3-32b")
response = model.invoke("Why do parrots talk?")
response
```

```text
AIMessage(content="<think>\nOkay, so why do parrots talk? Let me think about this. I know parrots are known for mimicking human speech, but why? ...", ...)
```

The second route uses the class from `langchain-groq`:

```python
import os
from langchain_groq import ChatGroq



model = ChatGroq(model="qwen/qwen3-32b")
response = model.invoke("Why do parrots talk?")
response
```

```text
AIMessage(content='<think>\nOkay, so I need to figure out why parrots talk. Let\'s start by recalling what I know about parrots. ...', ...)
```

What the output tells you: the text begins with a `<think>` block. Qwen 3 is a
**reasoning model**, and it writes its thinking out before the answer; the real
answer comes after the closing think tag. Note the model name has no provider
prefix when you use `ChatGroq` directly (`qwen/qwen3-32b`), because the class
already knows the provider. With `init_chat_model` the prefix is what carries
that information (`groq:qwen/qwen3-32b`).

| Provider      | `init_chat_model` string               | Provider class               | Package                  |
| ------------- | -------------------------------------- | ---------------------------- | ------------------------ |
| OpenAI        | `"gpt-4.1"`                            | `ChatOpenAI`                 | `langchain-openai`       |
| Google Gemini | `"google_genai:gemini-2.5-flash-lite"` | `ChatGoogleGenerativeAI`     | `langchain-google-genai` |
| Groq          | `"groq:qwen/qwen3-32b"`                | `ChatGroq`                   | `langchain-groq`         |

He closes the section by saying the integration really is this easy: two ways for
each provider, and you can change the model by changing a string. As a final
demonstration he goes back to the Gemini `init_chat_model` cell, changes
`gemini-2.5-flash-lite` to `gemini-2.5-flash` ("let's say I want the flash
model") and runs it; the answer comes back the same way. That edited version is
what the repository notebook holds:

```python
import os
from langchain.chat_models import init_chat_model

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = init_chat_model("google_genai:gemini-2.5-flash")
response = model.invoke("Why do parrots talk?")
response.content
```

## Streaming and batch (50:00 to 57:20)

Next heading in the notebook: "Streaming And Batch". It contains two ideas that
you will use constantly once you build a chatbot.

### The problem with `invoke` for long answers

With `invoke`, nothing appears until the model has written the entire answer.
He demonstrates with a longer request:

```python
model.invoke("Write me a 200 words paragraph on Artificial Intelligence")
```

You sit and wait for several seconds (about seven on screen), then the whole
`AIMessage` appears at once. For a short reply nobody minds. For a long one,
the user stares at a blank screen.

:::note Which model is `model` here?
The `model` variable is whatever you assigned last. By the time of this section
he has re-run several cells, and the output style here looks like the Gemini
flash-lite model. Any of the three models behaves the same for `invoke`,
`stream` and `batch`.
:::

### `stream`

Most models can **stream**: send the answer in small pieces while it is still
being generated. Showing text progressively makes the application feel much
faster, especially for long answers. Calling `stream` returns an iterator (a
generator) that yields **chunks** as they arrive.

He first calls it without a loop, which only shows that it is a generator:

```python
model.stream("Write me a 200 words paragraph on Artificial Intelligence")
```

```text
<generator object BaseChatModel.stream at 0x...>
```

Nothing has been printed because the generator is lazy: you have to loop over it
to receive the pieces. So he adds a `for` loop and prints each chunk's `.text`:

```python
for chunk in model.stream("Write me a 200 words paragraph on Artificial Intelligence"):
    print(chunk.text)
```

The text now arrives while the model is still writing, but each chunk lands on its
own line, which is hard to read. To see where the chunk boundaries fall, he
changes the print: `end="|"` replaces the newline with a vertical bar after each
chunk, and `flush=True` forces Python to display it immediately rather than
buffering:

```python
for chunk in model.stream("Write me a 200 words paragraph on Artificial Intelligence"):
    print(chunk.text, end="|", flush=True)
```

```text
Artificial Intelligence (AI) represents the simulation of human intelligence processes by machines, especially computer systems. At its core, AI empowers systems to perceive their environment, learn from data, reason, solve problems, and even make decisions with minimal human intervention. This| sophisticated technology encompasses various subfields, ...
```

Each `|` marks the end of a chunk, so you can see a chunk is a few words to a
sentence, not a single character. He then repeats the same idea with a second
question, `"Why do parrots have colorful feathers?"`, first with the streaming loop
and then with `invoke`, so that you feel the difference side by side:

```python
model.invoke("Why do parrots have colorful feathers?")
```

```python
for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)
```

In the video he gets a `SyntaxError` while editing these cells (a leftover
fragment of the previous cell), and simply retypes the line. With `invoke` he
waits and then everything appears; with `stream`, text appears as it is
generated. His point: when you build chatbots for a company, you will use
streaming most of the time, but there are cases where you want batch.

### `batch`

**Batch** means sending a collection of independent requests to a model together,
so that they are processed in parallel. Here he sends three unrelated questions in
one call:

```python
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response)
```

```text
content='Parrots are renowned for their dazzling array of colors, and there isn\'t just one reason for this vibrant plumage. ...' additional_kwargs={} ...
content='Airplanes fly by expertly manipulating four fundamental forces: **Lift, Weight, Thrust, and Drag.** ...' ...
content='**Quantum computing** is a new type of computing that harnesses the principles of quantum mechanics ...' ...
```

How it works: `model.batch` takes a **list** of inputs and returns a **list** of
`AIMessage`s in the same order. All three prompts go out in parallel, and all three
answers arrive together. The loop afterwards just prints each reply.

You can also limit how many calls run at once, with the `config` argument:

```python
model.batch(
    ["Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"],
    config={
        'max_concurrency': 5,  # Limit to 5 parallel calls
    }
)
```

If you send ten questions with `max_concurrency` set to five, only five are in
flight at the same time; as soon as one finishes, the next is sent. This
protects you from provider rate limits.

<Infographic
  src="/img/agentic-course/01-invoke-stream-batch.svg"
  alt="Three lanes comparing invoke, which waits for the whole answer, stream, which prints chunks as they arrive, and batch, which runs several prompts in parallel."
  caption="Explanatory board (not shown in the video): invoke, stream and batch side by side."
/>

| Method             | Input                      | What you get back                         | Use it when                                                                  |
| ------------------ | -------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------- |
| `model.invoke(x)`  | One prompt                 | One `AIMessage`, after the full answer    | Scripts and short answers                                                    |
| `model.stream(x)`  | One prompt                 | An iterator of chunks, arriving as written | Chatbots: the user sees text immediately                                     |
| `model.batch([...])` | A list of independent prompts | A list of `AIMessage`s, in input order   | Many separate questions at once, such as scoring a list of documents          |

:::note Batch is about speed, not necessarily cost
The notebook (and the video) says batching "can significantly improve
performance and reduce costs". `batch` runs your calls in parallel on your side,
which certainly cuts waiting time. Each prompt is still billed for its own
tokens, so cost only falls if you use a provider's separate discounted batch
service. Also, `max_concurrency` is a cap on simultaneous calls, not a rule that
requests are grouped in fives.
:::

## Tools (57:20 to 1:06:00)

He says: "we move one step ahead and talk about tools". Recall the whiteboard: an
agent is an LLM connected to a tool, and a tool is just a piece of functionality.
It can be an API request, a built-in tool, a news-reporting tool, a Google search
tool: any independent function. Now he shows how tools are actually created.

The notebook is `3-tools.ipynb`. Its first markdown cell quotes the LangChain
definition: models can request to call tools that perform tasks such as fetching
data from a database, searching the web or running code. A tool is a **pairing**
of two things:

1. **A schema**: the tool's name, a description and its argument definitions
   (often written as a JSON schema).
2. **A function or coroutine** to execute.

He starts by creating a model and checking it works, exactly as before. This time
he uses the Groq-hosted Qwen model:

```python
import os
from langchain.chat_models import init_chat_model

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = init_chat_model("groq:qwen/qwen3-32b")
response = model.invoke("Why do parrots talk?")
response
```

```text
AIMessage(content='<think>\nOkay, so I need to figure out why parrots talk. ...', ...)
```

The response confirms the model works, and its text shows what a plain LLM
returns. Now he needs to attach a tool to it.

### Defining a tool with `@tool`

```python
from langchain.tools import tool

@tool
def get_weather(location:str)->str:
    """Get the weather at a location"""
    return f"It's sunny in {location}"


model_with_tools=model.bind_tools([get_weather])
```

Line by line:

- `from langchain.tools import tool` imports `tool`. It is used as a
  **decorator**: written as `@tool` on the line above a function, it turns that
  function into a LangChain tool.
- `def get_weather(location:str)->str:` is the function. The argument is called
  `location` and it is a string. This is the **argument definition** from the
  schema.
- The docstring `"""Get the weather at a location"""` is the **description**. He
  says it plays a very important role: when the tool is bound to the model, the
  model learns what the function does from this sentence. That is why he wrote it.
  The function's name becomes the tool's name.
- The body returns "It's sunny in" the location. He hard-codes the answer for
  simplicity, and reminds you that here you could make an API request or query a
  database and return real data.
- `model.bind_tools([get_weather])` returns a **new** model object that knows
  about the tool, here saved as `model_with_tools`. The original `model` is
  untouched. Nothing is printed.

He notes this is one of two ways to give a model a tool. The other, which you saw
in notebook 1, is `create_agent(model=..., tools=[...])`, which does the whole
loop for you. Binding tools is the older, more manual route that people used
before, and it exposes what the agent is doing inside.

:::note `@tool` or a plain function?
In notebook 1 the agent accepted a plain function with no decorator, and that
worked because `create_agent` wraps functions for you. `@tool` is what you reach
for when you call `bind_tools` yourself, and when you want to control the tool's
name, description or argument schema explicitly.
:::

### Asking a question: the model requests a tool call (1:03:00 to 1:04:00)

```python
response = model_with_tools.invoke("What's the weather like in Boston?")
print(response)
for tool_call in response.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
```

```text
content='' additional_kwargs={'reasoning_content': 'Okay, the user is asking about the weather in Boston. I need to use the get_weather function. ...', 'tool_calls': [{'id': '...', 'function': {'arguments': '{"location":"Boston"}', 'name': 'get_weather'}, 'type': 'function'}]} response_metadata={...} ... tool_calls=[{'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': '...', 'type': 'tool_call'}] ...
Tool: get_weather
Args: {'location': 'Boston'}
```

What happened, and what is missing. The model did **not** answer the question and
did **not** run your function. Its `content` is an empty string. Instead it
returned a **request**: "please call `get_weather` with `location='Boston'`". The
reasoning text in the output shows the model deciding that it needs the
`get_weather` function and that it requires a `location`, which is Boston here.
The loop at the end walks through `response.tool_calls`, a list of these requests,
printing each tool's name (`get_weather`) and its arguments (`{'location':
'Boston'}`).

His summary: this is the simplest way of working with a tool. Use the decorator,
give a schema or docstring, and bind it to the model. Either do that, or, if you
are directly creating an agent, define the function and pass it to `create_agent`
with a model name and the tool list. Either way the model decides *which* function
to call and *with what arguments*.

### The tool execution loop (1:04:00 to 1:06:00)

Binding only produces the request. Someone has to run the function and show the
model its result. He calls the full procedure the **tool execution loop**, and he
pastes it in from the LangChain docs:

```python
# Step 1: Model generates tool calls
messages = [{"role": "user", "content": "What's the weather in Boston?"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# Step 2: Execute tools and collect results
for tool_call in ai_msg.tool_calls:
    # Execute the tool with the generated arguments
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# Step 3: Pass results back to model for final response
final_response = model_with_tools.invoke(messages)
print(final_response.text)
# "The current weather in Boston is 72°F and sunny."
```

```text
The weather in Boston is sunny.
```

He runs it step by step, so here is what each step does:

1. **The model generates tool calls.** `messages` starts as a list holding one user
   message. `model_with_tools.invoke(messages)` returns an `AIMessage` that holds
   the tool call (as above). That message is appended to `messages`, so the
   conversation so far is: user question, then the model's request.
2. **Execute the tools and collect the results.** For each request in
   `ai_msg.tool_calls`, `get_weather.invoke(tool_call)` runs the real function with
   the arguments the model chose. Passing the whole tool-call object (not just its
   arguments) makes the tool return a `ToolMessage`, which carries the result *and* the
   id of the call it answers. That result is appended too.
3. **Pass the results back for the final response.** `invoke` is called again, this
   time with the whole list. The model now sees the question, its own request, and
   the tool's answer, which is the **context**, and it writes the final sentence.
   `.text` gives the text of that reply.

He prints `messages` to show the conversation the loop built:

```python
messages
```

```text
[{'role': 'user', 'content': "What's the weather in Boston?"},
 AIMessage(content='', additional_kwargs={'reasoning_content': '...', 'tool_calls': [...]}, ..., tool_calls=[{'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': '...', 'type': 'tool_call'}], ...),
 ToolMessage(content="It's sunny in Boston", name='get_weather', tool_call_id='...')]
```

Three entries: the user's role dictionary, the AI message with the request, and the
tool message when the tool was executed ("It's sunny in Boston"). The final answer
is not appended in this example: it was only printed.

<Infographic
  src="/img/agentic-course/01-tool-loop.svg"
  alt="A three-lane flow showing your code asking the model, the model requesting get_weather, your code running the tool, and the model writing the final answer, with the messages list growing underneath."
  caption="Explanatory board (not shown in the video): the three steps of the tool execution loop and how the messages list grows."
/>

:::note The comment in the pasted code
The last line, `# "The current weather in Boston is 72°F and sunny."`, comes from the
LangChain documentation's example output. It is only a comment. The real output
here is "The weather in Boston is sunny.", because the tool in this notebook
always answers "sunny" and never reports a temperature.
:::

This loop is exactly what the `create_agent` agent from notebook 1 did for you:
call the model, run any tool it asked for, give the result back, and repeat until
the model answers without asking for a tool. Seeing it written out by hand is
the point of this notebook.

His closing remarks on tools:

- That was a quick revision of how to work with a tool, "a very basic way of
  creating this".
- Inside the function you can write any logic you want, and you can also use the
  **built-in tools** that LangChain provides (the reason `langchain_community` was
  installed), by calling them directly in code.
- The most important thing is **what response the tool generates**: a model can
  only reason over what the tool hands back, so a tool that returns clear, useful
  text produces a better answer.

## What comes next (1:06:00)

At 1:06:40 he moves to the next topic, **messages**: the system message, the AI
message and the human message, and how their `role`, `content` and metadata work.
You have already met three of them in this chapter without naming them all:
`HumanMessage` (your question), `AIMessage` (the model's reply or tool request)
and `ToolMessage` (a tool's result). The next chapter in this series begins there.

## What you can now do

- [ ] I can say what the nine sections of the course are, where each starts in the
      video, and why the course was recorded around LangChain v1.
- [ ] I can create a uv project (`uv init`, `uv venv`, activate), install packages
      with `uv add -r requirements.txt`, and read the versions back from
      `pyproject.toml`.
- [ ] I can store API keys in a `.env` file, load them with `load_dotenv()`, and
      explain why `.env` must never be committed.
- [ ] I can explain why a plain LLM cannot answer "today's AI news" and how adding a
      tool and getting back context turns it into a basic agent.
- [ ] I can build an agent with `create_agent`, invoke it with a `messages`
      dictionary, and explain why a bare string fails with `InvalidUpdateError`.
- [ ] I can read the message list an agent returns (human, AI with tool calls, tool,
      final AI) and explain how the model knew to call `get_weather`.
- [ ] I can create a chat model for OpenAI, Gemini or Groq in two ways
      (`init_chat_model("provider:model")` or the provider class) and read the
      `AIMessage` it returns.
- [ ] I can choose between `invoke`, `stream` and `batch`, loop over a stream with
      `print(chunk.text, end="|", flush=True)`, and cap parallelism with
      `max_concurrency`.
- [ ] I can define a tool with `@tool`, attach it with `bind_tools`, read the
      `tool_calls` the model requests, and run the three-step tool execution loop
      by hand.
