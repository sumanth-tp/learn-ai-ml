---
id: agentic-course-deep-agents
title: "06. Deep Agents: Planning, Sub-Agents and a File System (Complete Agentic AI Course in 10 Hours)"
sidebar_label: "6 - Deep Agents"
sidebar_position: 6
slug: /projects/agentic-ai-complete-course/deep-agents
description: "Why shallow and ReAct agents break on big tasks, the four parts of a deep agent (planning tool, sub-agents, system prompt, file system), and a first working deep agent built with the deepagents library, Groq and Tavily."
tags: [agentic-ai, deep-agents, deepagents, langgraph, langchain, middleware, tavily, groq]
---

import Infographic from '@site/src/components/Infographic';

> **Part 6 of 9** · [Watch on YouTube](https://www.youtube.com/watch?v=rV3HJ4LEZ7k) ·
> Notebook: `deepagentscourse/deeoagentsdemo/1-basicsdeepagent.ipynb`. The instructor shared this notebook only as a
> Google Drive file, so it is not in a repository. Every line of code below was read from the video frames (enlarged
> to read them) and cross-checked against what he says. Notes follow the video in order.

This chapter explains what makes an agent "deep" (it plans, delegates, keeps files and follows a rich system prompt),
and then builds the smallest possible deep agent with the `deepagents` library so you can see the difference from an
ordinary agent in the graph it compiles and in the result it returns.

:::note What this chapter covers and what it leaves for later
The instructor splits Deep Agents into two videos. This part is the theory plus a first "hello, deep agent" notebook.
Customising the model, system prompt and tools, then the backend, sub-agents and interrupts, is promised for part two
and is not in this video. Where this chapter goes beyond the video it says so in a clearly marked **addition**.
:::

## Where this fits in the course

By now the course has covered building generative AI applications, building independent agents that can carry out a
task, the different types of agent, and getting several agents to collaborate in multi-agent applications. Deep agents
are the next step up. The plan for this video is to understand how deep agents differ from what you have built so far,
see a little code that creates one, and leave the full practical treatment (customisation, sub-agents, backends,
interrupts) for the next video.

## First, what a plain agent looks like

The instructor opens a whiteboard with two headings: "Agents, which we will call shallow agents" and "Deep agents".
He starts with the simplest agent there is, because the word "shallow" only makes sense after you have seen it.

Take an LLM and give it an input. The LLM behaves like the brain of the system: it reads the input and decides
between two options. It can generate the answer itself, or it can reach out to a tool. A tool can be anything outside
the model, usually a third-party service.

His example is a question the model cannot answer from memory: "What is the current temperature in Bangalore, or
Paris?" A language model has no live data, so it is connected to a tool, such as SerpAPI, Tavily or any weather API.
The model asks the tool, the tool replies that the temperature in Paris is so-and-so, and that reply becomes the
output. This is a perfectly real agent, and it is the pattern most people have built in their generative AI projects.
The instructor calls it a **shallow agent**.

<Infographic
  src="/img/agentic-course/06-shallow.svg"
  alt="A shallow agent: input goes to an LLM, which either calls a tool such as SERP API and returns the result, or answers directly. Below, three limits: no explicit planning, complex queries cannot be handled, limited context retention."
  caption="Redrawn from the whiteboard."
/>

### Why he calls it shallow

He gives three reasons, and the whole chapter is really about fixing them.

1. **There is no explicit planning.** A query arrives, the LLM decides on one action (call a tool, or answer), and an
   output leaves. There is a single piece of logic applied once. Nothing ever breaks the request into steps.
2. **Complex queries cannot be handled.** Suppose you ask for today's recent AI news, how it relates to economics, and
   what the best recent developments in physics are. That is really three or four questions folded together. To answer
   it properly the request has to be decomposed into smaller queries that are solved separately, and a one-pass agent
   has no mechanism to do that.
3. **Context retention is limited.** An agent needs a good amount of context to behave sensibly. Here a single flow
   produces a single output, so very little context is built up or kept.

:::note A small simplification on the board
On the whiteboard the tool's reply goes straight to the output, and he says the model is not consulted again. In
practice, frameworks usually pass the tool result back to the model once so it can phrase the final answer. The
argument does not change: there is still one decision, no plan and no memory worth the name.
:::

## The ReAct agent

"Fine," he anticipates, "but surely we have seen better agents than that." The best known is the **ReAct agent**.

Again an LLM, but now it is connected to many tools: a Wikipedia tool, a search API tool, a Tavily tool, as many as
you like. The LLM is given a system prompt. When an input arrives, the model chooses which tool to call. After the tool
returns, its output goes back to the LLM as new context, and the model decides whether it needs another tool. The
important word is **loop**. The agent may act any number of times, each time based on what it just observed.

His worked example is "What is 2 + 2, and then multiply by 5?". The first part is solved and its result becomes
context, the second part is solved with that context, and the combined context produces the final answer.

<Infographic
  src="/img/agentic-course/06-react.svg"
  alt="A ReAct agent: an LLM with a system prompt loops with many tools, acting and observing, and then produces an output. A bracket labels it still a shallow agent, with five missing things."
  caption="Redrawn from the whiteboard."
/>

This is a real improvement, because the model can keep going until the query is solved. But the instructor still files
it under shallow agents, and the board says why. Underneath it is just an LLM plus tools, going round a loop. There is
no planning, no structured plan, no deep reasoning, no state management and no persistent memory. A tool is called, a
result returns, and the loop continues. That is all.

:::note What ReAct stands for
Aloud, he expands the name as "act" plus "read". The usual expansion is **Reason + Act**: the model reasons about what
to do, acts by calling a tool, observes the result, and repeats. His point about the loop is unaffected.
:::

| | Shallow agent | ReAct agent | Deep agent |
| --- | --- | --- | --- |
| Shape | LLM + tool, one pass | LLM + tools in a loop | Plan, delegate, remember |
| Number of model decisions | One | Many, as the loop repeats | Many, across a to-do list |
| Planning | None | None | A to-do list written first |
| Splits big tasks | No | Only implicitly, one step at a time | Yes, into sub-agents |
| Memory beyond the chat | None | None | A file system, shared with sub-agents |
| Behaviour steered by | A short prompt | A short prompt | A rich system prompt |

## Deep agents

A deep agent "works completely differently", so he refuses to call it shallow. His examples are the deep research modes
in ChatGPT and Claude, and Manus. He also says that his own team is building a product on a deep agent, which he names
on the board (I read it as Xenodocs, from a low-resolution frame), and will announce soon. A lot of product work is
happening in this direction.

The architecture is different, and he captures it as **four properties**. Draw a deep agent in the middle of the board
and hang these four things on it.

<Infographic
  src="/img/agentic-course/06-four-parts.svg"
  alt="A deep agent in the centre with four components around it: a planning tool, sub agents, a system prompt and a file system acting as persistent memory. Claude Code is noted as the example."
  caption="Redrawn from the whiteboard."
/>

1. **A planning tool.** When a query arrives it does not go straight to a tool or straight to an answer. A planning
   step happens first.
2. **Sub-agents.** Workers that carry out the items the plan produced.
3. **A system prompt.** The standing instructions that say how the agent should behave.
4. **A file system.** A place to keep things, which acts as persistent memory.

### The example he keeps coming back to: Claude Code

To make this concrete he points at Claude Code, which he calls an amazing deep agent. It is used for far more than
writing code, because it has planning and decomposition built in.

People assume Claude Code is only for coding, and that is how he first thought of it, but it plans and decomposes work too. He opens a public copy of Claude Code's system prompt in a browser. It begins by saying that the assistant is
Anthropic's official CLI for Claude, an interactive tool that helps users with software engineering tasks, and it goes
on with rules such as assisting only with defensive security tasks and refusing to create or improve code that could be
used maliciously. The lesson is that a deep agent has a long, deliberate system prompt, and that this one is readable
by anyone.

:::note About that prompt
The page he shows is a copy kept on a third-party public repository, not a file published by Anthropic, and Claude
Code's real prompt changes between releases. Treat it as an illustration of the idea (a long prompt covering tone,
style, safety rules and tool use), not as the current text.
:::

### Planning tool, with the Paris trip

Back on the board, he takes the first component. Whenever a query arrives, the first important module is the planning
tool. In Claude Code the plan is simply a **to-do list**. He gives "the thousand-foot view" with a request:

> Plan a holiday to Paris, budget 100k rupees, 3 nights and 4 days.

As soon as this reaches the deep agent, planning happens first, and the plan is a to-do list. Day one is travelling to
Paris and staying at a particular hotel at a particular price. Day two is breakfast, then the Eiffel Tower. Day three
is another place. Day four is the flight back to India. Each day carries its cost, and the list says what to book and
what not to book.

### Sub-agents

A list is not work. Someone has to execute it, so the next component is the sub-agents. Sub-agent one makes sure the
first item is carried out, sub-agent two takes the next, then three and four. The to-do list decides how many workers
the job needs, and each worker is responsible for its own slice.

### System prompt and file system

To carry out the work properly the agents need a system prompt that says how they should behave: tone, coding style,
anything you want to pin down. Claude Code's prompt, shown earlier, is the model for this.

Then comes the file system, which he stresses is very important. Think of it as **persistent memory that every
sub-agent can reach**. A sub-agent can do its piece, save the outcome into the file system as a specific file or a
shared note, and the other sub-agents can pick it up. This is how the workers communicate with each other without
passing huge messages around.

Put together, a deep agent plans, creates sub-agents to solve the planned tasks, follows a system prompt, and uses a
file system as persistent memory between those sub-agents.

<Infographic
  src="/img/agentic-course/06-planning-to-subagents.svg"
  alt="A request to plan a Paris holiday becomes a to-do list from the planning tool, which is handed out to four sub agents. A system prompt guides them and a shared file system acts as persistent memory."
  caption="Redrawn from the whiteboard, combining his two sketches of the to-do list and the sub-agents."
/>

### A second example: researching and writing a blog

For a more everyday case, imagine you give the deep research agent a blog topic and ask it to research the topic and
produce the blog. First it makes a to-do list. The tasks he writes down are:

1. Research the topic.
2. Do more research, for example from research papers or other source material.
3. Write the blog.
4. Run a copyright check.

Each item gets a sub-agent with the right powers. The first sub-agent has internet access. The second can reach arXiv,
the research-paper site. The third is a specialist at writing blogs. The fourth checks for copied material, which he
says also happens through the internet. His board shows the first three sub-agents explicitly and the last one in
speech, so the copyright worker's tool is the least certain part of the sketch. He adds that these tasks can then be
done in parallel.

<Infographic
  src="/img/agentic-course/06-blog-example.svg"
  alt="A blog topic becomes a to-do list of four items: research, more research, write the blog and copyright check. Each is handled by a sub agent with internet access, arXiv access, writing skill or internet access respectively."
  caption="Redrawn from the whiteboard."
/>

:::tip Why splitting the work helps
Each sub-agent gets only the tools and instructions it needs for its piece. That keeps every worker's context small and
focused, and it means a long job does not have to fit inside one overloaded conversation. The notebook later shows the
same idea from the other side, with a tool result that is too big for the conversation being moved into a file.
:::

With that, the theory is done. The instructor says it is a basic understanding, and moves to build a basic deep agent
with some tools.

## Project setup

He shows the whole setup from an empty folder, because the environment is part of the lesson. The folder is called
`deepagentscourse` and he opens it in Google Antigravity, an editor in the VS Code family. Any IDE is fine. The commands
below were run in the editor's terminal (a Windows command prompt, hence the backslashes).

```bash
uv init
uv venv
.venv\Scripts\activate        # on macOS or Linux: source .venv/bin/activate
```

- `uv init` creates a project in the current folder. The terminal answers `Initialized project 'deepagentscourse'`, and
  you get `pyproject.toml`, `main.py`, `README.md` and `.python-version`.
- `uv venv` makes the virtual environment. It reported `Using CPython 3.13.2` and `Creating virtual environment at:
  .venv`, and printed the activation command that he then ran.
- Activation puts the environment's name in the prompt, `(deepagentscourse)`.

Next he writes a `requirements.txt`. He starts with the library the whole video is about, adds the LangChain pieces,
and extends the file twice as he needs more.

```text
deepagents
langchain
langchain-openai
langchain-groq
ipykernel
tavily-python
python-dotenv
```

| Package | Why it is there |
| --- | --- |
| `deepagents` | A standalone library for agents that tackle complex, multi-step tasks. It is built on LangGraph and inspired by Claude Code, deep research and Manus. |
| `langchain` | Provides `create_agent` and `init_chat_model`, which the notebook uses. |
| `langchain-openai` | In case an OpenAI model is wanted. It is installed but not used in this video. |
| `langchain-groq` | The chat model integration for Groq, which is the model provider used in the run. |
| `ipykernel` | Lets the Jupyter notebook attach to this virtual environment's Python. |
| `tavily-python` | The Tavily client, for real-time internet search. Added. |
| `python-dotenv` | Loads the `.env` file. Added. |

He installs with:

```bash
uv add -r requirements.txt
```

The terminal reported `Resolved 89 packages` on the first run. Installing `deepagents` also brings in LangGraph, which
he points out, because the library is built on it. He then re-runs the same command each time he adds a line to the
file.

He stresses why LangGraph matters here. It is what lets you build complex, multi-agent workflows, and it is stateful:
it has a **state** data structure that remembers information about the workflow and can share it. Deep agents inherit
all of that, which is why things like persistence across a conversation come for free.

The install printed these versions, which are worth pinning if you want your graph to look like the one in the video
(the version of `deepagents` itself scrolled off screen):

| Package | Version printed |
| --- | --- |
| `langchain` | 1.2.3 |
| `langchain-core` | 1.2.6 |
| `langchain-groq` | 1.1.1 |
| `langchain-openai` | 1.1.7 |
| `langchain-anthropic` | 1.3.1 |
| `langchain-google-genai` | 4.1.3 |
| `langgraph` | 1.0.5 |
| `langgraph-checkpoint` | 3.0.1 |
| `langgraph-prebuilt` | 1.0.5 |
| `langgraph-sdk` | 0.3.1 |
| `langsmith` | 0.6.2 |
| `openai` | 2.14.0 |

:::warning Versions move quickly
`deepagents` was young and fast-moving when this was recorded. With the pins above and `deepagents` 0.3.1 I could
rebuild exactly the graph he shows. In `deepagents` 0.7.21, the newest release at the time of writing, the default
setup is different: a generic model gets no `write_todos` tool and no summarisation node by default. If your graph
picture or tool list differs from this chapter, check your installed version before assuming you made a mistake.
:::

### Keys in the `.env` file

He creates a `.env` file next to the notebook with four variables: `OPENAI_API_KEY`, `GROQ_API_KEY`,
`GOOGLE_API_KEY` and `TAVILY_API_KEY`. The Tavily key is for internet search. The OpenAI, Groq and Google keys are for
the model providers, and in this video only Groq and Tavily end up being used.

```text
OPENAI_API_KEY="..."
GROQ_API_KEY="..."
GOOGLE_API_KEY="..."
TAVILY_API_KEY="..."
```

:::danger Keys shown on screen
In the recording the `.env` file is open and the values are readable. Never reuse that practice. If you ever show a
real key on screen, in a commit or in a chat, treat it as leaked and rotate it. Add `.env` to `.gitignore`.
:::

To get a Tavily key you sign in at the Tavily website (he continues with Google), and the dashboard shows an API key
you can copy. Tavily is a real-time internet search service built for LLM tools.

## The notebook

He creates a folder `deeoagentsdemo` (the spelling is his) and a notebook `1-basicsdeepagent.ipynb` inside it. First he
selects the kernel, the `.venv` environment with Python 3.13.2. Then he adds two Markdown cells for reference, so you
can reread the definition later. They read, in substance:

> **Deep Agents overview.** Build agents that can plan, use subagents, and leverage file systems for complex tasks.
> `deepagents` is a standalone library for building agents that can tackle complex, multi-step tasks. Built on
> LangGraph and inspired by applications like Claude Code, Deep Research, and Manus, deep agents come with planning
> capabilities, file systems for context management, and the ability to spawn subagents.
>
> **When to use deep agents.** Use deep agents when you need agents that can:
> - Handle complex, multi-step tasks that require planning and decomposition
> - Manage large amounts of context through file system tools
> - Delegate work to specialized subagents for context isolation
> - Persist memory across conversations and threads

He admits he should have put these in earlier, but the point is that the four bullets restate what was just explained
on the board. Notice the phrases "context management" and "context isolation". They are the long-horizon story: a file
system holds the bulky material so the conversation does not have to, and a sub-agent works in its own context so its
mess does not pollute the main one. Persistence across conversations comes from LangGraph, which the library sits on.

## Cell 1: environment variables

He starts a new code cell headed `### Basic deep agent`.

```python
### Basic deep agent

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
```

What it does, line by line:

- `import os` gives access to environment variables.
- `from dotenv import load_dotenv` and `load_dotenv()` read the `.env` file and place its entries into the process
  environment. He had to add `python-dotenv` to `requirements.txt` first, which is why that package appears late.
- Each `os.environ[...] = os.getenv(...)` line copies a variable back into the environment. He runs the cell and
  nothing is printed, which is what success looks like. He also says you could add Groq or Tavily lines "wherever you
  want"; he sets OpenAI, Groq and Tavily here and skips Google.

:::note These three assignments are redundant, and one is risky
`load_dotenv()` has already put every variable from `.env` into `os.environ`, so the three lines change nothing. They
are also fragile: if a key is missing from `.env`, `os.getenv` returns `None`, and assigning `None` into `os.environ`
raises a `TypeError`. The simplest safe version is just `load_dotenv()`. The lines are kept above because they are what
the video runs.
:::

## Cell 2: the web search tool

Before building the agent he wants a tool. A deep agent without tools still has its planning and file tools, but a
researcher needs the internet, so he builds a web search tool around the Tavily client.

He has shown Tavily as a tool for agents in the LangChain section, so this is a recap with a new twist. First he tries the import and creates the client. He types the client first, with the key read from the environment
even though it is already set:

```python
from tavily import TavilyClient
tavily_client=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
```

Then he turns it into a function the agent can call. The function is where most of the teaching happens, because the
function signature **is** the tool's interface.

### How the function was built, and the order bug

He writes the parameters one at a time:

- `query: str`, the search text.
- `max_results: int = 5`, a cap of five results.
- `topic`, typed as a `Literal` so it can only take a fixed set of values. He imports `Literal` from `typing` in its
  own cell first (`from typing import Literal`) and checks that the import works. The values he lists are `"sports"`,
  `"news"` and `"finance"`, with a default of `"general"`.
- `include_raw_content: bool = False`, so results do not carry the full page text by default.

He explains that these are the parameters the Tavily client needs, which is why he hard-codes them with allowed
options. If you press into the `search` call in the editor, a tooltip shows its full signature, in this order:
`query`, `search_depth`, `topic`, `time_range`, `start_date`, `end_date`, `days`, `max_results`, `include_domains`,
`exclude_domains`, `include_answer`, `include_raw_content`, and more.

His first draft of the call passed the values positionally, as `max_results, include_raw_content, topic`. He then
points out that the order has to match the client's own signature, and the tooltip shows it does not: after `query`
the next positional slots are `search_depth`, `topic` and `time_range`, so his three values would have landed in the
wrong parameters. His fix is to pass each one **by keyword**, which removes the problem. He also notes (and the tooltip confirms) how he found out what to pass: he read the Tavily
documentation page, and the `Literal` is simply the set of news categories he wants the search to support.

His default `"general"` was not in his list the first time. He corrects it by adding `"general"` as the first
entry of the `Literal`. Here is the final merged cell as it stood when he ran it, with a comment heading:

```python
### Tools- Internet search
from tavily import TavilyClient
from typing import Literal

tavily_client=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query:str,max_results:int=5,
topic: Literal["general","sports","news","finance"]="general",
include_raw_content:bool=False):
    """Run a web search"""
    return tavily_client.search(query,
    max_results=max_results,include_raw_content=include_raw_content,topic=topic)
```

Reading it:

- The `Literal[...]` annotation does real work. The agent library turns the signature into a schema that the model
  sees, so the model learns that `topic` may only be one of those four strings.
- The docstring `"""Run a web search"""` is what the model reads as the tool's description. His is minimal, and a
  more informative docstring (when to use it, what comes back) makes the model use the tool better.
- The function returns whatever `tavily_client.search` returns, a dictionary with the query, the results, and fields
  such as follow-up questions and images. That dictionary is what the agent will receive as the tool result.
- He runs the cell, and nothing is printed because it only defines things.

:::note "sports" is not a valid Tavily topic
Tavily's `topic` accepts `"general"`, `"news"` and `"finance"`, which you can see in the signature tooltip on screen.
The `"sports"` entry he adds is not in that list, and calling the tool with it would be rejected or ignored. The
argument that the `Literal` should mirror what the client accepts is right; take `"sports"` out in your own copy.
:::

He stops to say that the same `Literal` idea applies to anything: you decide how many categories you want to expose.
The tool is now ready to be handed to the agent.

## Creating the deep agent

Now the part the video is named for. To create a deep agent you need a prompt, a model, and the agent itself.

### The import, compared with a normal agent

In LangChain, an ordinary agent is created with `create_agent` from `langchain.agents`. It takes an LLM and the tools
you want to integrate. A deep agent is created with `create_deep_agent`, imported from the `deepagents` package. His
first cell has these headings and the first attempt:

```python
### Create a deep agent
## Prompt

## agent

from deepagents import create_deep_agent

create_deep_agent(
    models=,
    tools=[web_search],
    system_prompt="Act as a researcher"
)
```

That cell was a work in progress (note `models=` with no value). He explains the three things `create_deep_agent`
needs:

- **`tools`**: the list of functions the agent can call. Here, `[web_search]`.
- **`system_prompt`**: the standing instruction. His is deliberately tiny, `"Act as a researcher"`. The prompt can be
  simple or elaborate, and it sits on top of the library's own built-in instructions.
- **`model`**: which LLM to use. This is a string or a chat model object.

### Choosing a model: `init_chat_model` and Groq

The model is the next cell. He imports `init_chat_model` and builds a Groq model. The first typing shows a typo in the
import path (`lanchain.chat`) that he corrects on the next attempt.

```python
from langchain.chat_models import init_chat_model

model=init_chat_model("groq:qwen/qwen3-32b")
model
```

- `init_chat_model` creates a chat model from a provider-prefixed string. `"groq:qwen/qwen3-32b"` means "the Groq
  provider, model `qwen/qwen3-32b`". The Groq key is already in the environment from cell 1, which is why nothing else
  is needed.
- His first spelling of the model name was wrong (he types it as "quen"), which he notices and fixes before running.
- The last line, `model`, displays the object. The notebook shows `ChatGroq(profile={'max_input_tokens': 131072,
  'max_output_tokens': 16384, ...` and so on. Those numbers are useful: the model accepts about 131 thousand tokens in
  and 16 thousand out.

He mentions the other ways to build the model, all of which he has covered in the LangChain section: `init_chat_model`
with another provider prefix (OpenAI, Google Gemini), or the concrete classes `ChatGroq`, `ChatOpenAI` and
`ChatGoogleGenerativeAI`.

:::note The library's own default
On screen at the end of the video, the Deep Agents docs say that `deepagents` uses `claude-sonnet-4-5-20250929` by
default and that you can pass any supported model identifier string or LangChain model object. He supplies Groq's
Qwen3 32B instead. Groq's catalogue of model names changes, so if `qwen/qwen3-32b` is not available to you, pick a
current tool-calling model from Groq's list.
:::

### Running it, and the first error

With the model defined, he puts it into the agent cell and runs it:

```python
## agent

from deepagents import create_deep_agent

deepagent=create_deep_agent(
    model=model,
    tools=[web_search],
    system_prompt="Act as a researcher"
)
deepagent
```

The first run failed. The error said that there was an unexpected keyword argument `models` and asked "did you mean
`model`?". The parameter is **`model`**, singular. It is a good illustration of reading Python's error message: it
named the fix. After correcting it the cell ran in about a second. Ending the cell with the bare name `deepagent`
makes the notebook draw the compiled graph, which is the picture he comes back to in a minute.

## A normal agent next to a deep agent

To make the difference visible, he builds an ordinary agent with the same model and tool.

```python
## Basic Agent
from langchain.agents import create_agent

simple_agent=create_agent(
    model=model,
    tool=[web_search]
)
simple_agent
```

This failed too, with `TypeError: create_agent() got an unexpected keyword argument 'tool'. Did you mean 'tools'?`. He
changes `tool` to `tools`, and it works:

```python
simple_agent=create_agent(
    model=model,
    tools=[web_search]
)
simple_agent
```

Both mistakes are the same kind: a singular where a plural belongs (`models`, `tool`). The correct parameter names are
`model` and `tools`.

The notebook prints each agent's graph, and this is where the difference shows. "Almost everything is the same," he
says: both have a model and tools. What a deep agent adds is a set of middleware, and middleware shows up in the graph.

<Infographic
  src="/img/agentic-course/06-simple-vs-deep-graph.svg"
  alt="Two LangGraph pictures side by side. The simple agent has start, model, tools and end. The deep agent adds PatchToolCallsMiddleware.before_agent, SummarizationMiddleware.before_model and TodoListMiddleware.after_model around the model."
  caption="Redrawn from the two graph pictures printed in the notebook: the simple agent, the deep agent."
/>

### Reading the deep agent's graph

The simple agent's graph is small: `__start__`, then `model`, which either goes to `__end__` or to `tools`, and tools
go back to `model`. It is the loop from the ReAct board.

The deep agent's graph has extra boxes. Reading it from the top:

| Node in the graph | What it is for |
| --- | --- |
| `PatchToolCallsMiddleware.before_agent` | Runs once when the agent starts. If an earlier tool call in the history never received a result (an interrupted or cancelled call), it patches in a stand-in result so the history stays valid. The instructor calls it the "patch tool calls hook". |
| `SummarizationMiddleware.before_model` | Runs before each model call. When the conversation grows too long, it compresses older messages into a summary so the model keeps room to think. |
| `model` | The LLM decides the next step. |
| `TodoListMiddleware.after_model` | Runs after each model call. This is the to-do list machinery from the whiteboard: the model's plan lives in a to-do list that the agent uses to keep track of how execution is going. |
| `tools` | Where the tool calls are executed. |

His summary is that when a task is given to a deep agent it is divided into sub-tasks, and each one needs tracking, and
the to-do list is how that is tracked. As the conversation grows, summarisation happens automatically. He ties this back
to the LangChain module, where he explained middleware as hooks you can place at specific points in a workflow, such as
before the agent, before the model or after the model. A deep agent comes with those hooks pre-installed.

:::note Where the other deep-agent parts are
Only middleware that adds a node to the loop appears in the picture. The file-system tools (`ls`, `read_file`,
`write_file`, `edit_file`, `glob`, `grep`), the `task` tool that launches sub-agents, and the `write_todos` tool are
all supplied by middleware too, but they add tools and prompt text rather than a hook node. When I built the same agent
with `deepagents` 0.3.1 and the versions listed earlier, the `tools` node held those tools plus `web_search`, and the
graph matched the video.
:::

:::note The to-do hook is narrower than it sounds
The instructor says the to-do list is created "automatically" by this hook. More precisely, `TodoListMiddleware`
supplies the `write_todos` tool and guidance on when to use it, and its `after_model` hook is a safety check that
stops the model from updating the list several times in one turn. Whether a to-do list is created at all is the
model's decision. You will see in a moment that for a simple question it does not make one.
:::

## Calling the agent

A deep agent is invoked exactly like any LangGraph agent: you pass a dictionary with a `messages` list, each message
having a role and content.

### Two slips on the way

His first attempt at the cell was incomplete: it contained `{"role":"user",""}`, a dictionary with a stray empty string
and no value, which Python rejected as `SyntaxError: expression expected after dictionary key and ':'`. He then
typed the full message. The question first typed was "what is langgraph", and he decided to ask "what is deep agent"
instead, so the question he actually ran has the spelling "What is deepagent?".

```python
result = deepagent.invoke({"messages": [{"role": "user", "content": "What is deepagent?"}]})
result
```

- `invoke` runs the whole graph to completion and returns the final state.
- The input has a single user message. `role` is `"user"` and `content` is the question.
- Ending the cell with `result` prints the state.

### What happens inside while you wait

He explains the run before showing the result. The question enters the graph. The model decides whether it needs a tool
call. For a question like this it will do an internet search. Whatever hook is relevant fires on the way: the model can
make a to-do list for how to resolve the question and split it into sub-tasks, it calls the internet search tool, the
tool result comes back to the model's context and summarisation applies. It takes a while because a deep agent is doing
research, not answering from memory. He notes that streaming would help you watch progress and says he will show it
later, but he does not do so in this video. While waiting, the notebook's cell timer ticked through 17 s, 32 s and
47 s, and finished at **52.9 s**.

<Infographic
  src="/img/agentic-course/06-invoke-flow.svg"
  alt="One invoke call: the question passes through middleware hooks to the model, which plans, calls web_search or answers. A large tool result is saved to the virtual file system, and the returned result holds messages, files and optionally todos."
  caption="Explanatory board (not shown in the video), based on the notebook run."
/>

### The output

The printed state is a dictionary with a `messages` list and a `files` dictionary. Trimmed to what is legible on screen:

```text
{'messages': [HumanMessage(content='What is deepagent?', additional_kwargs={}, response_metadata={}, id='164961ce-...'),
              AIMessage(content='', additional_kwargs={'reasoning_content': 'Okay, the user is asking, "What is deepagent?" Let me start by un...'}, ...),
              ToolMessage(content='Tool result too large, the result of this tool call 3fetmqatt was saved in the filesystem at this path: /la...', ...),
              AIMessage(content="DeepAgent is an end-to-end deep reasoning agent introduced in a 2025 research paper by Xiaoxi Li and colleagu...", ...)],
 'files': {'/large_tool_results/3fetmqatt': {'content': ['{"query": "deepagent", "follow_up_questions": null, "answer": null, "images": [], ...'],
                                            'created_at': '2026-01-09T06:16:45.591676+00:00',
                                            'modified_at': '2026-01-09T06:16:45.591676+00:00'}}}
```

Read it from the top. The first message is his question. The second is the model's turn: its `content` is empty,
because this Qwen model on Groq puts its thinking in `additional_kwargs['reasoning_content']` and then asks for a tool
call. The third is the tool's reply, and it is not the search results. It is a short notice saying that the tool result
was too large and has been saved into the file system. The fourth message is the model's final answer.

Next, `files`. The agent has saved the large search result as a file named `/large_tool_results/3fetmqatt`, whose
content is the Tavily response, with created and modified timestamps. The dictionary key tells you the path.

He shows the last answer next:

```python
result["messages"][-1].content
```

This prints the final message's text, an explanation of "DeepAgent" with numbered capabilities, such as
"General Reasoning: capable of handling both structured (API/tool use) and unstructured ..." and "demonstrated superior
results on eight benchmarks, including ToolHop (tool discovery) and GAIA (complex reasoning)". The output is a long
line in the notebook, so only the end is visible. Then he looks at the files:

```python
result['files']
```

which prints the dictionary above, starting with `{'/large_tool_results/3fetmqatt': {'content': ['{"query": "deepagent", "follow_up_questions": null, "answer": null, "images": [], ...`. His explanation is that these
are some of the files the agent made in order to preserve context. The deep agent summarises as it goes, and when a
tool result is very large it moves the content into a file, so that the conversation stays small.

:::warning The answer is about a different "DeepAgent"
Read the answer carefully: it describes "DeepAgent", an end-to-end reasoning agent from a 2025 research paper. That is
not the LangChain `deepagents` library the video is about. The question was ambiguous, the model searched the web for
the word and answered about the first thing it found. The agent worked correctly, the prompt was the problem. For a
cleaner demo, ask "What is the LangChain deepagents library?" or give a topic with no namesake.
:::

:::note Those files are not on your disk, and no to-do list appeared
He says the content may be saved "in some hard disk file". By default it is not. `deepagents` uses a **state backend**,
a virtual file system that lives inside the agent's LangGraph state. The files persist within one conversation thread
and are visible in the result, but they are not written to your drive and are not shared across threads. To write to a
real folder or keep files between threads you choose a different backend, which is part-two material (see the addition
at the end of this chapter). Also note that `result` here has only `messages` and `files`. There is no `todos` key,
because the model did not call the planning tool for this simple lookup. The planning tool is something the model
chooses to use when a task needs it.
:::

## Simple agent versus deep agent, in his words

He closes the notebook part with the comparison. With `create_deep_agent` a number of middleware hooks are applied for
you, and you can create any number of tools on top. With a plain `create_agent` agent you just get nodes, a graph and
edges communicating with one another. You can of course add middleware to a simple agent yourself and give it the same
hooks. The point of the demonstration is the clear idea of what a deep agent is, not that simple agents are incapable.

| | `create_agent` (simple agent) | `create_deep_agent` |
| --- | --- | --- |
| Import | `from langchain.agents import create_agent` | `from deepagents import create_deep_agent` |
| Arguments used here | `model=`, `tools=` | `model=`, `tools=`, `system_prompt=` |
| Graph | `model` and `tools` loop | Adds patch, summarisation and to-do middleware around the model |
| Extra tools you did not write | None | Planning, file and sub-agent tools |
| Large tool results | Stay in the conversation | Moved into a file, the conversation keeps a pointer |
| Where to customise | Add middleware yourself | Model, system prompt, tools, backend, sub-agents, interrupts |

### Errors you may hit, in one place

| Mistake | Fix |
| --- | --- |
| Search call passed `max_results, include_raw_content, topic` by position, so the order mattered | Pass them by keyword |
| Default topic `"general"` was not one of the allowed values | Add `"general"` to the `Literal` |
| `create_deep_agent(models=...)` | `model=` (singular) |
| `create_agent(tool=[...])` | `tools=` (plural) |
| Half-typed message dictionary, `SyntaxError` | Complete the `role` and `content` pair |
| Question ran as "What is deepagent?", ambiguous | See the warning above |

## What comes in part two

He ends by saying this was the initial part of building deep agents, and that many topics remain and he does not want to
make the video too long. Part two customises the agent: the **model**, **system prompt** and **tools**, and then the
extra features, **backend**, **sub-agents** and **interrupts**. At the very end he opens the LangChain docs page on
customising Deep Agents, whose diagram is exactly this map: `create_deep_agent` branches into core config (model,
system prompt, tools) and features (backend, subagents, interrupts), and everything feeds into a customised agent.

<Infographic
  src="/img/agentic-course/06-customisation-map.svg"
  alt="create_deep_agent branches into core config with model, system prompt and tools, and features with backend, subagents and interrupts, all feeding a customised agent."
  caption="Redrawn from the LangChain docs diagram he shows."
/>

The video then moves straight on to the next series, on guardrails, which is the next chapter.

## Addition: the sub-agent, backend and long-horizon pieces the video only names

:::note Not from the video
This section is an **addition**. The instructor defers sub-agents and backends to part two, but the brief for this
chapter includes how a deep agent handles long jobs, so here is the minimum you need. I ran the scripted-model snippet end to
end against `deepagents` 0.3.1 with the versions in the table above, and confirmed that the sub-agent dictionary is
accepted (the `task` tool lists it) and that the backend classes and arguments exist. The sub-agent and backend
snippets continue the notebook's state (`web_search`, `model`) and were not run against a live model. Always check your installed version's documentation, as the API
is still moving.
:::

### The tools a deep agent gets without asking

In the compiled graph of `deepagents` 0.3.1 the `tools` node holds these built-in tools next to your own:

| Tool | Belongs to | What the model uses it for |
| --- | --- | --- |
| `write_todos` | Planning | Writes or updates the to-do list (items with a `pending`, `in_progress` or `completed` status). |
| `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` | File system | List, read, create, change and search files in the agent's workspace. |
| `execute` | File system backend | Runs shell commands, but only if the backend supports it. Otherwise it returns an error. |
| `task` | Sub-agents | Hands a piece of work to a sub-agent and returns its short answer. |

Two behaviours explain the long-horizon story. First, when a tool result is bigger than about 20,000 tokens, the file
system middleware writes it to `/large_tool_results/...` and leaves the model a short pointer, exactly what the run
above showed. The model can then `read_file` or `grep` the file in small pieces. Second, when the conversation itself
gets long, summarisation compresses it. In `deepagents` 0.3.1 the trigger is 85 percent of the model's maximum input
size when the model reports it, keeping about the last 10 percent, and otherwise a fixed token limit. Together they let
a job run for far longer than one context window.

### Planning and files with no API key

You can see the planning tool and the file system working without any provider key by scripting a stand-in model. This
is only to show the mechanics, since a real model decides for itself whether to plan.

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from deepagents import create_deep_agent


class ScriptedModel(GenericFakeChatModel):
    """A stand-in model that replays a fixed script, so no API key is needed."""

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def profile(self):
        return {"max_input_tokens": 131072}


script = iter([
    AIMessage(content="", tool_calls=[{
        "name": "write_todos", "id": "call_1",
        "args": {"todos": [
            {"content": "Research the topic", "status": "in_progress"},
            {"content": "Write notes.md", "status": "pending"},
        ]},
    }]),
    AIMessage(content="", tool_calls=[{
        "name": "write_file", "id": "call_2",
        "args": {"file_path": "/notes.md", "content": "# Notes\nhello"},
    }]),
    AIMessage(content="Done."),
])

agent = create_deep_agent(model=ScriptedModel(messages=script), tools=[],
                          system_prompt="Act as a researcher")
result = agent.invoke({"messages": [{"role": "user", "content": "Plan and write notes"}]})

print(list(result.keys()))   # ['messages', 'todos', 'files']
print(result["todos"])
print(result["files"]["/notes.md"]["content"])
```

Output:

```text
['messages', 'todos', 'files']
[{'content': 'Research the topic', 'status': 'in_progress'}, {'content': 'Write notes.md', 'status': 'pending'}]
['# Notes', 'hello']
```

When the model does use the planning tool, a `todos` key appears next to `messages` and `files`. A file's `content` is
stored as a list of lines.

### Defining a sub-agent

A sub-agent is a dictionary with a name, a description the main agent reads to decide when to delegate, its own system
prompt, and its own tools. The main agent hands work over with the `task` tool and gets back only the sub-agent's final
answer, so the sub-agent's many intermediate searches never enter the main conversation. That is the "context
isolation" in the notebook's markdown cell. A general-purpose sub-agent is always available as well.

```python
research_subagent = {
    "name": "research-agent",
    "description": "Digs into one question on the web and returns a short, sourced summary.",
    "system_prompt": "You are a careful researcher. Search, then reply with a summary under 200 words.",
    "tools": [web_search],
}

deepagent = create_deep_agent(
    model=model,
    tools=[web_search],
    system_prompt="Coordinate the work and delegate research to the research agent.",
    subagents=[research_subagent],
)
```

### Putting files on a real disk

To have the file tools work on a real folder, pass a backend. Use `virtual_mode=True` so paths are confined to
`root_dir`.

```python
from deepagents.backends import FilesystemBackend

deepagent = create_deep_agent(
    model=model,
    tools=[web_search],
    system_prompt="Act as a researcher",
    backend=FilesystemBackend(root_dir="./workspace", virtual_mode=True),
)
```

:::danger A file backend lets the model change real files
With a disk backend the agent can create and overwrite files under the folder you give it. Point it at a scratch
directory, keep `virtual_mode=True`, and never at a folder holding work you cannot recreate. `StoreBackend` (persist
across threads through a LangGraph store) and `CompositeBackend` (route different paths to different backends) are the
other options in the same package.
:::

## What you can now do

- I can explain, with the Bangalore and Paris example, why a single-pass LLM-plus-tool agent is called shallow, and
  name its three limits: no explicit planning, no handling of complex queries, limited context retention.
- I can explain what a ReAct loop adds (repeated acting and observing) and why it is still shallow.
- I can name the four parts of a deep agent (planning tool, sub-agents, system prompt, file system) and say what each
  contributes, using the Paris trip and the blog example.
- I can explain why a file system acts as persistent memory shared by sub-agents, and why moving bulky results into
  files helps a long task.
- I can set up a project with `uv`, a virtual environment, a `requirements.txt` and a `.env`, and keep keys out of
  version control.
- I can write a typed `web_search` tool around the Tavily client, call `search` with keyword arguments, and explain why
  the `Literal` and the docstring matter to the model.
- I can create a deep agent with `create_deep_agent(model=..., tools=[...], system_prompt=...)`, build the model with
  `init_chat_model("groq:qwen/qwen3-32b")`, and recognise the `model` and `tools` spelling errors.
- I can compare a `create_agent` graph with a `create_deep_agent` graph and say what the patch, summarisation and
  to-do middleware do.
- I can invoke a deep agent, read `messages` and `files` in the result, and explain the `/large_tool_results/` entry.
- I can spot that an ambiguous question produces an answer about the wrong thing, and that `files` is a virtual file
  system and not your hard disk.
