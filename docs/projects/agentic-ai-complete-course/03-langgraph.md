---
id: agentic-course-langgraph
title: "3. LangGraph Crash Course: Chatbot, Tools, ReAct, Memory, Streaming, Human in the Loop and MCP (Complete Agentic AI Course in 10 Hours)"
sidebar_label: "3 - LangGraph"
sidebar_position: 3
slug: /projects/agentic-ai-complete-course/langgraph
description:
  "Learn LangGraph's state, nodes and edges by building a chatbot, then add tools, a ReAct loop, memory, streaming, human approval and an MCP server and client."
tags:
  [
    agentic-ai,
    langgraph,
    langchain,
    state-graph,
    tools,
    react-agent,
    memory,
    streaming,
    human-in-the-loop,
    mcp,
  ]
---

import Infographic from '@site/src/components/Infographic';

> **Part 3 of 9** ·
> [Watch on YouTube](https://www.youtube.com/watch?v=rV3HJ4LEZ7k) ·
> Notebooks: `Agentic-LanggraphCrash-course/1-BasicChatbot/chatbot.ipynb` and
> `2-HumanAssistance/humanintheloop.ipynb`. The MCP part is
> typed live in a separate project that has no notebook, so its code is read
> from the screen. Notes follow the video in order.

By the end of this chapter you can build a LangGraph chatbot from first principles: a typed state, nodes that
change it, edges that route it, and then tools, a ReAct loop, memory, streaming, a human approval step and a
small MCP server and client.

:::note What this video contains, and what it does not
The instructor opens by describing a three-part LangGraph course. The recording you are reading about is
**Part 1 only** (fundamentals), plus a self-contained video on building MCP servers from scratch. His announced
Part 2 (workflows, multi-agent systems, the functional API, debugging in LangGraph Studio) and Part 3
(end-to-end projects, deployment, evaluation) are **not in this video**. The author's repository does contain
notebooks for debugging, multi-agent systems and multimodal RAG; they are covered at the end of this chapter
under a clearly marked section for additions, so that nothing in the repository is left out.
:::

## What the LangGraph course covers

The video starts on an Excalidraw page that lays out the whole plan. He splits the LangGraph crash course into
three parts and tells you what to expect from each, so that you know what you are signing up for before the
first line of code.

<Infographic
  src="/img/agentic-course/03-roadmap.svg"
  alt="Roadmap board with three parts: Part 1 LangGraph fundamentals, Part 2 advanced LangGraph, Part 3 LangGraph agents and end to end projects"
  caption="Redrawn from the roadmap page."
/>

| Part | What it teaches | His size estimate |
| --- | --- | --- |
| 1. LangGraph fundamentals | Build a basic chatbot, integrate tools and multiple tools, add memory, add human in the loop (human feedback while the graph is running), streaming techniques, MCP, including how to build an MCP server from scratch. Along the way: states, graphs, nodes and edges, all through the **Graph API**. | About 2 h 50 m (he stresses it is only an estimate) |
| 2. Advanced LangGraph | Different kinds of workflows and agents, applications where agents communicate with other agents to solve a complex workflow, how state is managed across several agents, the **Functional API** (as an alternative to the Graph API), and debugging and monitoring in **LangGraph Studio** together with LangSmith. He calls this the step towards production-grade applications. | About 2 h |
| 3. LangGraph agents, end-to-end projects | Complete projects, an LLMOps pipeline, deployment techniques, and LLM evaluation: the metrics specific to LLMs, using LangGraph together with open-source tools such as MLflow, using AWS to track metrics, Grafana to display the reports, and Hugging Face Spaces for deployment. | Not given |

He asks viewers to download the material from the description, practise along, and share their learning on
LinkedIn and Twitter and tag him, because these are long recordings that depend on viewer support.

## Project setup with uv

He starts from an empty folder, which will be the project workspace, opens a command prompt there and launches
VS Code from it. Every project, he reminds you, begins with an environment. In his earlier videos he created
environments with conda; here he switches to **uv**, and spends a few minutes on why.

### Why uv

uv is a Python package and project manager written in Rust. Because it is compiled Rust rather than Python it
is very fast. On the benchmark chart he shows from uv's own README (installing the Trio library's dependencies
with a warm cache) the gap is large:

| Tool | Time on the README chart |
| --- | --- |
| uv | 0.06 s |
| poetry | 0.99 s |
| pdm | 1.90 s |
| pip-sync | 4.63 s |

The highlights he reads out from that README:

- It is 10 to 100 times faster than `pip`.
- It is a single tool that replaces `pip`, `pip-tools`, `pipx`, `poetry`, `pyenv`, `twine` and `virtualenv`.
- It does full project management with a universal lock file.
- It installs and manages Python versions as well, inside the same project.

To install it, copy the command for your platform from the README:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# any platform, if you already have pip
pip install uv
```

You can still use conda if you prefer it; nothing later in the chapter depends on uv.

### Create the project

He already had uv installed, so he goes straight to initialising the workspace from the integrated terminal:

```bash
uv init
```

Output: `Initialized project 'agenticlanggraph'`. The command creates five files in the folder, and he walks
through them:

| File | What it is for |
| --- | --- |
| `.gitignore` | The usual Git ignore list. |
| `.python-version` | The Python version uv will use. Here it contains `3.13`. |
| `main.py` | A starter program with a `main()` function and the usual `if __name__ == "__main__"` guard, so you can run the project from here. |
| `pyproject.toml` | The project's brief information: name, version, description, `requires-python` (here `>=3.13`) and a `dependencies` list. The list is empty because nothing is installed yet. |
| `README.md` | An empty readme. |

Next he writes a `requirements.txt` containing the three core libraries (the file grows later in the video):

```text
langgraph
langchain
langsmith
```

What each one is for, in his words: LangGraph and LangChain provide the functionality for building generative AI
applications, chatbots and agentic applications. LangSmith is used for tracking and evaluating applications,
including from the LangGraph cloud.

Then he creates and activates a virtual environment, and installs the requirements into it:

```bash
uv venv
```

Output: `Using CPython 3.13.2`, `Creating virtual environment at: .venv` and `Activate with:
.venv\Scripts\activate`. He copies that activation line into the terminal (on macOS and Linux it is
`source .venv/bin/activate`). The prompt gains the project name in brackets, which is how you know it worked.

```bash
uv add -r requirements.txt
```

This is the uv equivalent of `pip install -r requirements.txt`. He says it quickly in the video, but the
important part is that `uv add` also records the packages in `pyproject.toml` (you now see `langchain>=0.3.25`,
`langgraph>=0.4.8` and `langsmith>=0.3.45`, the versions current at the time of recording) and writes a
`uv.lock` file. Opening the lock file, you can see every installed library.

### The notebook and its kernel

He creates a folder called `1-BasicChatbot` and inside it a notebook, `1-basicchatbot.ipynb`. VS Code asks for a
kernel; he picks the `.venv` he just made (Python 3.13.2). Running notebooks in VS Code needs one more package:

```bash
uv add ipykernel
```

To check the kernel is alive he runs a deliberately broken cell first (a stray `!` after `1+`), which proves
the kernel is connected by returning an "invalid syntax" error, and then `1 + 1`, which returns `2`. He
promises to use Jupyter notebooks for the first stages and to move to ordinary Python files later (the MCP
section at the end does exactly that).

:::tip Keep your `.env` out of Git
Almost every cell later in the video needs an API key. Create a `.env` file next to the notebook, list it in
`.gitignore`, and load it with `python-dotenv`. If you ever see a key on screen, as happens briefly in this
video's tool section, treat it as exposed and generate a new one.
:::

## The three building blocks: nodes, edges and state

Before any code, he sets the scene for the whole course: from here on LangGraph's **Graph API** is used.
LangGraph also has a **Functional API**, which he says you will meet later, but in his experience the Graph API is
the easiest and best way to learn LangGraph. If you already know the Graph API well you can move to the
Functional API afterwards, and he promises to explain the difference when you get there.

He then says that LangGraph has three important components: **edges**, **nodes** and **state**. To explain
them he picks a real problem of his own.

### The use case: turn a YouTube video into a blog

He uploads a lot of videos and would like a blog post for each one. If you think about how a human would do
it, three steps appear:

1. From the YouTube video, take out the **transcript**.
2. From the transcript, write the **title** of the blog.
3. From the title and the transcript together, write the **content** of the blog.

Doing this by hand for every video is slow, but LLMs are very good at content generation, so the question
becomes: can LLMs solve this workflow using LangGraph? To answer it he draws the workflow as a graph.

<Infographic
  src="/img/agentic-course/03-blog-workflow.svg"
  alt="Whiteboard of a YouTube to blog workflow: YT URL, START, transcript node, title generator node, content generator node, END, with edges between them"
  caption="Redrawn from the whiteboard."
/>

### Nodes

A **node** is a unit of work. In the drawing:

- The graph begins at `START`. The input is the YouTube URL, which goes into the first node.
- The first node produces the transcript. Its implementation uses a third-party loader that LangChain provides
  (he remembers it as something like `YT loader`): you give it the video URL and it returns the transcript. The
  output of the node is the transcript.
- The second node is the **title generator**. Its input is the transcript. Inside, it is an LLM plus a prompt:
  the prompt tells the LLM to write a blog title for this transcript.
- The third node is the **content generator**. Its input is the title and the transcript. Inside it is again an
  LLM plus a prompt, and its output is the blog content.
- The graph finishes at `END`, where you get the output.

He makes an important point about nodes: whenever you create a node you must also give it a **node
implementation**, the function that says what the node actually does. A node is a name plus a piece of code.

### Edges

An **edge** is the arrow between two nodes. Its whole purpose is to say where the flow of information goes next,
from node to node. In the drawing there is an edge from `START` to the transcript node, one from transcript to
title generator, one from title generator to content generator, and one from content generator to `END`.

### State

Nodes and edges explain the flow, but they raise a question: how does the third node get hold of the transcript
that the first node produced? That is the job of **state**.

State is a set of variables that you declare once for the whole graph. Any node can read them and any node can
write them. For this use case he defines three variables, `transcript`, `title` and `content`. When the first node
finishes, its output is saved into `transcript`; the second node reads `transcript` and saves `title`; the third
node reads both and saves `content`. The benefit of saving values inside the state is that every node in the
graph can see them.

<Infographic
  src="/img/agentic-course/03-state-shared.svg"
  alt="Three nodes inside a StateGraph, each writing to or reading from one shared state with transcript, title and content"
  caption="Redrawn from the whiteboard."
/>

This is why a whole graph of this kind is called a **StateGraph**: it keeps the state available at every node. He
adds a warning against a common confusion. State is **not** the same thing as external memory. Memory (keeping a
conversation across separate runs) is a different feature that he adds later in this chapter. For now, state is
simply what the nodes share while one run of the graph is in progress.

| Component | Plain meaning | In the blog example |
| --- | --- | --- |
| Node | A function that does one job and may update the state | transcript, title generator, content generator |
| Edge | A connection that says which node runs next | arrows from `START` to `END` |
| State | Shared variables every node can read and write | `transcript`, `title`, `content` |

## Build a basic chatbot

With those three ideas in place he moves to the first real build. The graph is deliberately tiny: `START`, one
node called the chatbot, and `END`. The chatbot node will contain an LLM (and a prompt), take the user's input
and return the answer. Later he will add external tools, then explain the ReAct agent, and he says the
ReAct part is more impressive than the basic bot, but first you must understand how this one is built.

<Infographic
  src="/img/agentic-course/03-chatbot-state.svg"
  alt="Whiteboard with a State holding a messages list that is appended to, and a StateGraph going START to chatbot to END, with Reducers noted"
  caption="Redrawn from the whiteboard."
/>

### Imports

He adds a markdown heading, "Build A Basic Chatbot With Langgraph (GRAPH API)", and starts with the imports:

```python
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
```

What each import is for:

- `Annotated` (from `typing`) adds context-specific metadata to a type. He hovers over it in VS Code and reads the
  documentation example: `Annotated[int, runtime_check.Unsigned]` tells a hypothetical runtime-check module that this int
  is unsigned, while every other consumer can ignore the metadata and treat the type as a plain int. In LangGraph
  the metadata will be a function that says how to update a state key.
- `TypedDict` (from `typing_extensions`) lets you declare a dictionary with named, typed keys. The tooltip describes it as a
  simple typed namespace that is equivalent to a plain `dict` at runtime. His own example is `Point2D` with `x` and
  `y` as `int` and `label` as `str`; a value is just `{'x': 1, 'y': 2, 'label': 'good'}`. It is a description of
  the shape of a dictionary for type checkers, nothing is enforced while the code runs.
- `StateGraph`, `START` and `END` come from `langgraph.graph`. `StateGraph` is the class that represents the
  entire graph you drew; `START` and `END` are the special first and last nodes.
- `add_messages` is a **reducer**, the function you attach to a state key to say how new values are merged in.

### Reducers and why the state class uses one

He explains reducers with the chatbot itself. The state of this chatbot needs a variable, say `messages`, that
holds the conversation. The type of that variable should be a **list**, because every time the user speaks and
the bot replies, the new messages must be added to it. The graph can be run many times in one session, and you
want each turn to **append** rather than **replace**. If the first turn is "hi, how are you?" and the bot says
"I am good", and the user's next turn is "what is your name?", the earlier turns must still be there, otherwise
the chatbot loses the thread. A reducer is what makes this happen: instead of overwriting the variable it
combines the old value with the new one. `add_messages` is one of several kinds of reducer, and its job is only to
add messages to the list.

<Infographic
  src="/img/agentic-course/03-reducer-explained.svg"
  alt="Two lanes comparing a state key with no reducer, which is overwritten, against add_messages, which grows the list"
  caption="Explanatory board (not shown in the video)."
/>

:::note Precisely what add_messages does
He describes `add_messages` as always appending. The accurate rule, which VS Code shows in the tooltip he opens,
is that it merges two lists of messages and **updates existing messages by ID**; it is append-only unless a new
message carries the same ID as an existing one. For normal chat use this is indistinguishable from appending.
It also converts plain strings such as `"Hi"` into `HumanMessage` objects, which you will see in the outputs
below.
:::

### The State class

Now the state itself:

```python
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages:Annotated[list,add_messages]
```

Line by line:

- `class State(TypedDict)` declares the state as a dictionary-shaped type. He explains why it inherits from
  `TypedDict`: the state is going to be passed around as a dictionary, so the class describes the keys that the
  dictionary has.
- `messages:Annotated[list,add_messages]` declares one key, `messages`. Its type is a list, and the second argument to
  `Annotated` is the reducer. Read it as "a list, updated by `add_messages`". The comment he types (copied from the
  LangGraph docs) says the same thing.

Then the graph builder:

```python
graph_builder=StateGraph(State)
graph_builder
```

Output: `<langgraph.graph.state.StateGraph at 0x...>`. This cell is only on screen in the video (the notebook builds
the graph in one later cell). It shows that `graph_builder` is just a `StateGraph` object; passing `State`
tells it the shape of the state that every node will receive. Nothing can run yet.

### API key, environment and the LLM

To call a model he needs an API key. He uses Groq in this course because it is quick, though he says you can use
OpenAI or any other provider. First he adds two libraries to `requirements.txt`, `python-dotenv` and
`langchain-groq`, and installs them with `uv add -r requirements.txt` again. Then he loads the environment:

```python
import os
from dotenv import load_dotenv
load_dotenv()
```

Output: `True`, which tells you a `.env` file was found. To get a key he opens `console.groq.com`, goes to **API
Keys**, creates a key with a name, and pastes it into a new `.env` file in the project as `GROQ_API_KEY=...`.

:::warning Re-run after you edit the .env
He edits `.env` after the notebook's kernel has already started, then later hits an "invalid API key" error. The
cause is that `load_dotenv()` ran before the key was in the file. The fix is to restart the kernel (or at least
re-run `load_dotenv()`) so the environment picks up the new value. The same thing happens again when he adds the Tavily key.
:::

Now the model. There are two equivalent ways to create it:

```python
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model

llm=ChatGroq(model="llama3-8b-8192")
```

```python
llm=init_chat_model("groq:llama3-8b-8192")
llm
```

Output (both): `ChatGroq(client=..., model_name='llama3-8b-8192', ...)`. The first is the provider's own class from
`langchain-groq`. The second, `init_chat_model`, is the generic route: you pass `"<provider>:<model name>"` and it
returns the right class. He says that with OpenAI you would install `langchain-openai` and use something like
`"openai:<model name>"`. The model name comes from the Groq playground (he picks Llama 3 8B with an 8192 context).
On screen he first types a shortened name, `ChatGroq(model="llama")`, then replaces it with the full model name;
a model name must be the exact string the provider lists.

:::note Model names change
Providers retire models. `llama3-8b-8192` is the name used in the video and in the repository, but Groq has
been replacing its first-generation model names, and the same repository's multi-agent notebook already uses
`llama-3.1-8b-instant`. If a call fails with a "model not found" or deprecation error, copy a current model id
from the provider's model list and keep the rest of the code unchanged. The same applies to the Qwen model used
in the MCP section.
:::

### The chatbot node

A node needs a function. He types it in stages (first returning an empty list as a stub, then filling it in):

```python
## Node Functionality
def chatbot(state:State):
    return {"messages":[llm.invoke(state["messages"])]}
```

How to read it:

- The function takes the current `state`, typed as `State`. LangGraph calls every node with the current state.
- `state["messages"]` is the conversation so far. It is passed straight to `llm.invoke`, so the LLM sees the
  whole list.
- The function **returns a dictionary of updates**, not the whole state: `{"messages": [response]}`. Because the
  `messages` key has the `add_messages` reducer, this reply is appended to the existing list, which is exactly what
  we wanted.

### Building the graph: nodes, edges, compile

```python
graph_builder=StateGraph(State)

## Adding node
graph_builder.add_node("llmchatbot",chatbot)
## Adding Edges
graph_builder.add_edge(START,"llmchatbot")
graph_builder.add_edge("llmchatbot",END)

## compile the graph
graph=graph_builder.compile()
```

- `StateGraph(State)` starts the builder with the state shape.
- `add_node("llmchatbot", chatbot)` registers a node. The **first argument is the name** you choose (he deliberately
  uses `llmchatbot` to show it can be anything); the **second is the function**, the node implementation.
- `add_edge(START, "llmchatbot")` and `add_edge("llmchatbot", END)` draw the two arrows. Note that edges use
  the **node names**, not the functions.
- `compile()` turns the blueprint into something you can run. He stresses that until you compile, you cannot
  execute the graph.

### Visualising the graph

```python
## Visualize the graph
from IPython.display import Image,display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass
```

`graph.get_graph()` returns the graph's structure and `.draw_mermaid_png()` renders it as a Mermaid picture, which
`display(Image(...))` shows in the notebook. The `try/except` is there because drawing needs optional extras (and
by default a network call to render the picture); if it fails, the rest still works. The picture shows `__start__`,
`llmchatbot` and `__end__`, connected in a line.

<Infographic
  src="/img/agentic-course/03-rendered-graphs.svg"
  alt="Three small rendered graphs: basic chatbot, streaming demo SuperBot, and human in the loop with chatbot and tools"
  caption="Redrawn from the graph images the notebooks render."
/>

### Running it, and two errors

To run a compiled graph you call `invoke`. His first attempt passes a plain string:

```python
graph.invoke("Hi")
```

This fails with `InvalidUpdateError`. The reason is that `invoke` expects a dictionary of **state keys**, because the input
is itself an update to the state. A bare string means nothing to the graph. The fix is to supply the key:

```python
graph.invoke({"messages":"Hi"})
```

This time the graph starts but fails inside the `llmchatbot` node with a 401 `invalid_api_key` error. That is the
environment issue described in the warning above: he restarts the kernel, re-runs the cells (the graph-builder cell
he had pasted earlier is not needed again) and tries again. Now it works:

```python
response=graph.invoke({"messages":"Hi"})
```

```python
response["messages"]
```

Output: a list with a `HumanMessage(content='Hi', ...)` followed by `AIMessage(content="Hi! It's nice to meet you.
Is there something I can help you with or would you like to chat?", ...)`.

He points at the output to make the key point of this section. He sent the plain string `Hi`; the graph turned it
into a `HumanMessage` and stored it in the `messages` list, and the node's reply, an `AIMessage`, was **appended**
after it. That is the reducer at work. Everything about the conversation is in that list.

To read just the answer, take the last message:

```python
response["messages"][-1].content
```

Output: `"Hi! It's nice to meet you. Is there something I can help you with or would you like to chat?"`. Index `-1` is
the last message and `.content` is its text. He says that if you understand this, you will be able to execute any
graph or workflow you can imagine.

### Streaming the response

`invoke` returns the final state. The other way to run a graph is `stream`, which yields results as the graph goes
through its nodes. He first prints each event as it arrives:

```python
for event in graph.stream({"messages":"Hi How are you?"}):
    print(event)
```

Each event is a dictionary **keyed by the node name** (here `llmchatbot`) whose value is what that node returned,
so you see the `AIMessage` inside `{'llmchatbot': {'messages': [...]}}`. To get only the text, he loops over the
event's values:

```python
for event in graph.stream({"messages":"Hi How are you?"}):
    for value in event.values():
        print(value["messages"][-1].content)
```

Output: `Hi! I'm just a language model, so I don't have emotions or feelings like humans do, but I'm functioning
properly and ready to help you with any questions or tasks you may have! ...`. Only the AI message shows because
the default streaming mode reports the update each node produced, and the human message was not produced by a
node. He promises a full explanation of the streaming modes later in the video; it is in the section "Streaming
in LangGraph" below.

## Chatbot with tools

The next step is to connect the chatbot to the outside world. He motivates it with a question: suppose the
chatbot is just an LLM with a prompt, wired as `START` to chatbot to `END`, and the user asks, "Provide me the
recent AI news." Will the LLM be able to answer? No. The LLM has no live information, and it may not have been
trained on recent data, so it needs an **external tool**.

<Infographic
  src="/img/agentic-course/03-tool-need.svg"
  alt="Whiteboard: user asks for the recent AI news, the chatbot LLM cannot answer, so it makes a tool call to a ToolNode holding Tavily, add, subtract and custom tools, then END"
  caption="Redrawn from the whiteboard."
/>

What should happen instead is that the chatbot recognises it cannot answer and makes a **tool call**. The tool
could be any third-party API, for example a search engine; the one he chooses is **Tavily**, a web-search API
built for LLMs. So the graph gains a second node, a **tool node**, and the request goes `START` to chatbot to tool
node to `END`. The tool node can hold many tools: Tavily search, and custom tools such as add or subtract, or any
function you write.

### How does the LLM know a tool exists?

This is the question he spends the most time on. The answer has two parts:

1. **Binding.** The LLM is bound to the tools. If you write a custom function, say `add`, you also write its
   **doc string**. The doc string tells the LLM what the tool does and what inputs (arguments) it needs. When a
   user's request matches that description, the LLM decides to call the tool.
2. **A tool node that executes the call.** Binding only tells the LLM that tools exist. When the LLM does make a
   tool call, something has to run it. That is the `ToolNode`.

He adds a third piece, a **tool condition**, which routes the flow to the tool node when the LLM made a tool call
and to `END` otherwise. All three are implemented in the code that follows.

### The target graph

Before writing code he shows the graph he is going to build: a tool-calling LLM node, a tools node, and `END`.

<Infographic
  src="/img/agentic-course/03-tool-calling-graph.svg"
  alt="Graph with start, tool_calling_llm, tools and end, annotated: node is LLM plus tools, tool node holds Tavily search and custom tools, the LLM reads each tool's doc string"
  caption="Redrawn from the graph image and annotations."
/>

- `tool_calling_llm` is the first node. Its implementation is **an LLM with the tools bound to it**.
- From that node there are **two paths**. If the LLM decides it must use a tool, the flow goes to the `tools` node,
  which holds Tavily search and the custom tools and produces a tool message. If it can answer directly, the flow
  goes straight to `END`.
- The LLM decides by reading each tool's doc string. His analogy for binding: imagine the LLM holds weapons to
  solve your input; binding tells it which weapons it has, and the tool node is the act of actually using one.

### Add Tavily

First `langchain-tavily` goes into `requirements.txt` and is installed again with `uv add -r requirements.txt`.
Then he opens `tavily.com` (an internet search service built for LLMs and RAG), logs in, copies the key it
shows (the free tier is enough for this course) and adds it to the `.env` as `TAVILY_API_KEY`. Since the `.env`
changed, he restarts the kernel and re-runs the earlier cells (imports, `load_dotenv`, the LLM). Then:

```python
from langchain_tavily import TavilySearch

tool=TavilySearch(max_results=2)
tool.invoke("What is langgraph")
```

`TavilySearch(max_results=2)` creates the search tool, limiting it to two results. Calling `invoke` with a string
runs one search. Output: a dictionary with the keys `query`, `follow_up_questions`, `answer`, `images`, a
`results` list (each result has `title`, `url`, `content` and a `score`), and `response_time`. The first result
is a DataCamp tutorial titled "LangGraph Tutorial: What Is LangGraph and How to Use It?" and the second is a
GeeksforGeeks article. So the tool works on its own before the LLM is involved.

### Add a custom tool and write its doc string

To show that any Python function can be a tool, and how the LLM learns about it, he writes one:

```python
## Custom function
def multiply(a:int,b:int)->int:
    """Multiply a and b

    Args:
        a (int): first int
        b (int): second int

    Returns:
        int: output int
    """
    return a*b
```

The type hints (`a:int`, `b:int`, `->int`) and the doc string (a summary line, an `Args` section and a `Returns`
section) are what the LLM will see. He generates the doc string skeleton with VS Code's doc-string helper and fills
in the descriptions.

:::warning The bug he hits, and the fix
The first time he types this function, he leaves out the `return a*b` line, so the function body is only a doc
string and returns `None`. Everything appears to work (the LLM makes the right call, `multiply(a=2, b=3)`) but
the tool message comes back as `null`. You will see this live in the "Running it" section below. Adding the
`return` line fixes it.
:::

Now collect the tools and bind them:

```python
tools=[tool,multiply]
```

```python
llm_with_tool=llm.bind_tools(tools)
```

```python
llm_with_tool
```

`bind_tools` returns a new runnable that carries the tools' names, descriptions and argument schemas with every
request. Output: `RunnableBinding(bound=ChatGroq(...), kwargs={'tools': [{'type': 'function', 'function': {'name':
'tavily_search', 'description': 'A search engine optimized for comprehensive, accurate, and trusted results.
...'}}, ...]})`. He points out that you can read from the output exactly which functions the model is connected to: `tavily_search`
and `multiply`.

### Build the tool-calling graph

```python
## Stategraph
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

## Node definition
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tool.invoke(state["messages"])]}

## Grpah
builder=StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))

## Add Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition
)
builder.add_edge("tools",END)

## compile the graph
graph=builder.compile()

from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

The new pieces, in the order he explains them:

- **`ToolNode`** (from `langgraph.prebuilt`) turns your list of tools into a ready-made node. Each tool has its own
  implementation, and `ToolNode(tools)` wraps all of them so that, given an AI message that contains tool calls,
  it runs them and returns the results as tool messages. In `add_node("tools", ToolNode(tools))` the string is the
  node name and the `ToolNode` object is the node implementation.
- **`tool_calling_llm`**, the node function, is the same shape as `chatbot` except that it calls
  `llm_with_tool.invoke(...)`. It does not call the plain `llm`, because only the bound version knows about the tools.
- **`add_conditional_edges`** is used because the first node has two outgoing paths. A normal edge always goes
  to the same place. A conditional edge asks a function where to go. Whenever a node has more than one outgoing
  edge, you add them as conditional edges.
- **`tools_condition`** (also from `langgraph.prebuilt`) is that function, already written for you. It applies two
  rules: if the latest message from the assistant is a tool call, route to the node named `tools`; if it is not,
  route to `END`. He emphasises that the tool node must be named exactly `tools` for this to work, which is
  why he gives it that name.
- **`builder.add_edge("tools", END)`** says that after the tools have run, the graph finishes. (This is the line he
  will change in the ReAct section.)

| `tools_condition` input | Where it routes |
| --- | --- |
| Last AI message contains a tool call | the node named `tools` |
| Last AI message is plain text | `END` |

Output: the rendered graph shows `__start__`, `tool_calling_llm`, `tools` and `__end__`, with a dotted line from
`tool_calling_llm` to `__end__` (the conditional path with no tool call), a solid line to `tools`, and a solid line
from `tools` to `__end__`.

:::note The errors he hits while typing this
Because he typed the cell live after restarting the kernel, three small errors appear. They are worth recognising:

- `NameError: name 'State' is not defined`. The kernel had been restarted, so the `State` class cell must be re-run.
- `RuntimeError: Node 'tool_calling_llm' already present` after `ValueError: Node name must be provided if action is
  not a function`. He first added the node without its function, which failed, and re-running the cell on the same
  builder object tried to add the name a second time. Define the function before the cell that uses it, and create a
  fresh `StateGraph(State)` when you re-run.
- `NameError: name 'Image' is not defined`. After the restart the `IPython.display` import was missing.

All are one-line fixes: re-run the missing cell.
:::

### Running it

First the Tavily path:

```python
response=graph.invoke({"messages":"What is the recent ai news"})
```

```python
response['messages'][-1].content
```

He looks at the whole list of messages and explains what is in it. There is the `HumanMessage` (the question), then an
`AIMessage` with **empty content**, because the LLM did not answer; instead it holds a `tool_calls` entry with an
ID, the function name (`tavily_search`) and arguments (`query`: `recent ai news`, among others). Then there is a
`ToolMessage`, the output of the tool, whose content is the search results (here, news items). Reading the last
message gives that tool message's content, not an answer written by the LLM.

To print it readably he loops and uses `pretty_print`:

```python
for m in response['messages']:
    m.pretty_print()
```

Output (trimmed):

```text
================================ Human Message =================================

What is the recent ai news
================================== Ai Message ==================================
Tool Calls:
  tavily_search (9hkqr0mx3)
 Call ID: 9hkqr0mx3
  Args:
    query: recent ai news
    search_depth: advanced
    time_range: day
    topic: news
================================= Tool Message =================================
Name: tavily_search

{"query": "recent ai news", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": ...
```

Now the custom tool:

```python
response=graph.invoke({"messages":"What is 5 multiplied by 2"})
for m in response['messages']:
    m.pretty_print()
```

Output:

```text
================================ Human Message =================================

What is 5 multiplied by 2
================================== Ai Message ==================================
Tool Calls:
  multiply (g6rrcbzyx)
 Call ID: g6rrcbzyx
  Args:
    a: 5
    b: 2
================================= Tool Message =================================
Name: multiply

10
```

This is the fixed version. When he first asked "what is 2 multiplied by 3", the LLM made the right call
(`a: 2`, `b: 3`) but the tool message printed `null`, then `5 * 2` also gave `null`. He realised the function
had no `return`, added `return a*b`, and then had to **re-run the cells downstream of the fix in order**: the
`multiply` definition, `tools=[tool,multiply]`, `llm_with_tool=llm.bind_tools(tools)`, and the cell that builds and
compiles the graph. The compiled graph keeps references to the old objects, so changing the function alone does
nothing until you rebuild. After that the answer is 10. He tries a compound request too ("what is 5 multiplied
by 2 and then multiply 10"), and the model issues two `multiply` calls, getting 10 and then 20 (the second
argument pair it picks is `a: 10, b: 2`).

### A request with two parts, and why it fails

Finally the case that exposes the design's limit:

```python
response=graph.invoke({"messages":"Give me the recent ai news and then multiply 5 by 10"})
for m in response['messages']:
    m.pretty_print()
```

Output (trimmed): a human message, an AI message with a `tavily_search` tool call, and the tool message of news
results, and then the graph ends. The multiplication never happens.

He stops to think about why. The sentence contains two requests. The LLM correctly made a tool call for the first
one and the Tavily tool answered, but the graph then went from `tools` straight to `END`. Nothing ever gave the
LLM a second turn to look at what was left. That sets up the next section.

## The ReAct agent architecture

His fix is to feed the tool's output **back to the LLM** instead of ending. Then the LLM becomes the main
decision-maker: after the search result arrives it still holds the second request (multiply 5 by 10), so it can make
a second tool call, get that result, and only then combine everything into a final answer.

<Infographic
  src="/img/agentic-course/03-react-brain.svg"
  alt="Whiteboard with natural input flowing into an LLM labelled brain with binding tools, which calls a ToolNode and receives the output back until it ends"
  caption="Redrawn from the whiteboard."
/>

He draws it on a fresh page: a natural input arrives ("AI news" and "multiply 5 by 5") at an LLM labelled **the
brain**, which has tools bound to it. The LLM breaks the input into its parts. For the first it knows it must call
the tool node; the tool node returns its output **to the LLM, not to the end**. The LLM still holds the second
part, so it calls the tool node again for the multiplication, gets that result back, then checks whether
anything is left. Nothing is left, so it summarises and ends. This communication pattern between the LLM and its
tools is what he calls the **ReAct agent architecture**.

He names three terms in ReAct:

- **Act**: when an input arrives, the LLM can act by making a tool call.
- **Observe**: when the tool's output comes back, the LLM observes it and decides whether to make another tool call or finish.
- **Reason**: after getting an output the LLM decides what to do next, which makes it the decision-maker.

He says this is where agent behaviour comes from, and that it is the reason agentic AI has become so popular.

:::note The usual order of the three words
In the research paper that named ReAct, the name is short for **Reason** plus **Act**, and a single turn is usually described as
reason, act, observe, repeating. The video lists the words in a different order but describes the same loop.
One more precision: a model does not literally split the sentence in two. It may return one tool call, or several in
a single message, and the loop in the graph is what lets it keep going until it stops asking for tools.
:::

### Changing one edge

How should the graph change? He copies the previous cell into a new section (markdown heading "ReAct Agent
Architecture") and asks you to say where the change is. The answer is a single line: instead of `tools` going to
`END`, it goes back to `tool_calling_llm`.

```python
## Stategraph
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

## Node definition
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tool.invoke(state["messages"])]}

## Grpah
builder=StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))

## Add Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition
)
builder.add_edge("tools","tool_calling_llm")

## compile the graph
graph=builder.compile()

from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

The only difference from the previous graph is `builder.add_edge("tools","tool_calling_llm")` in place of
`builder.add_edge("tools",END)`. The rendered picture now has an arrow from `tools` back up to `tool_calling_llm`:
a loop. It repeats until the LLM answers without a tool call, at which point `tools_condition` sends the flow to `END`.

<Infographic
  src="/img/agentic-course/03-react-graph.svg"
  alt="Two graphs side by side: before, tools goes to END; after, tools goes back to tool_calling_llm, forming a ReAct loop"
  caption="Redrawn from the two rendered graphs the notebook shows."
/>

Run the same compound request again:

```python
response=graph.invoke({"messages":"Give me the recent ai news and then multiply 5 by 10"})
for m in response['messages']:
    m.pretty_print()
```

Output (trimmed): after the `tavily_search` tool message there is now a further AI message that lists the news as
bullet points (for example a startup launching an AI-powered law firm and a chip maker laying off staff) and then
says it will multiply 5 by 10, ending with `50`. The tool output went back to the LLM, the LLM carried on with the
second request, and the loop ended on its own. You can attach any number of tools; the LLM is the one deciding
which to call.

## Adding memory to the agentic graph

Memory solves what he calls persistent checkpointing. To show why it is needed, he starts with the graph he already
has (the ReAct one, with no memory) and holds a conversation.

```python
response=graph.invoke({"messages":"Hello my name is KRish"})
for m in response['messages']:
    m.pretty_print()
```

Output: the human message, then `Nice to meet you, KRish! How are you today?`. No tool is needed, so the flow goes
straight to the end. Now the follow-up:

```python
response=graph.invoke({"messages":"What is my name"})
for m in response['messages']:
    m.pretty_print()
```

Output (trimmed):

```text
================================ Human Message =================================

What is my name
================================== Ai Message ==================================
Tool Calls:
  multiply (xgm8j8jmw)
 Call ID: xgm8j8jmw
  Args:
    a: 1
    b: 0
================================= Tool Message =================================
Name: multiply

0
================================== Ai Message ==================================

I apologize for the mistake earlier. Since the tool call id "xgm8j8jmw" yielded 0, I will assume you are asking about your name again.

Unfortunately, I don't have any information about your name, as it's not provided in the conversation. ...
```

The bot has no idea who he is, even though he introduced himself one call earlier. Each `invoke` starts from an
empty state, so the earlier turn is simply gone. There is also a small comedy in the output: a model with tools
bound sometimes reaches for a tool even when none is relevant, here a meaningless `multiply(1, 0)`. His main
point, though, is that nothing persisted between the two calls.

### The checkpointer

LangGraph's remedy is a **checkpointer**. He copies the graph cell and adds memory:

```python
## Stategraph
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

## Node definition
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tool.invoke(state["messages"])]}

## Grpah
builder=StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))

## Add Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition
)
builder.add_edge("tools","tool_calling_llm")

## compile the graph
graph=builder.compile(checkpointer=memory)

from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

Two changes from the ReAct graph: the import and creation of `MemorySaver()`, and the argument
`checkpointer=memory` in `compile`. He reads the tooltip for `MemorySaver`: it is an in-memory checkpoint saver that
stores checkpoints in memory using a `defaultdict`. After each node runs, LangGraph saves a checkpoint of the state so it
can be recalled later. The compile step is where you attach it, and he points out this is the place to remember.

:::note In-memory only, and the current name
The tooltip itself says to use this saver only for debugging or testing and recommends a database-backed saver (for
example the Postgres one) for production; if you deploy on the LangGraph Platform, no checkpointer is needed at all
because the platform supplies one. In recent LangGraph releases `MemorySaver` is an alias for `InMemorySaver`; both
work.
:::

### Threads

A checkpointer needs to know which conversation to restore. For that you give each session a unique **thread ID**
through the run's `config`:

```python
config={"configurable":{"thread_id":"1"}}

response=graph.invoke({"messages":"Hi my name is Krish"},config=config)

response
```

The `config` dictionary has a `configurable` key containing a `thread_id`. He says that in a real application you would
generate a unique value for each user session (his example: a user joins, so you make a thread for that user). You pass it as the
second argument to `invoke` as `config=config`, in addition to the messages. Output: the state, with the
human message and an AI message "Nice to meet you, Krish! ...". (On screen the first run appeared twice in the
list because the cell was executed twice on the same thread; both runs are recorded in the thread.)

```python
response['messages'][-1].content
```

Output: `'Nice to meet you, Krish! Is there something I can help you with or would you like to chat?'`.

<Infographic
  src="/img/agentic-course/03-memory-threads.svg"
  alt="Code on the left with compile checkpointer, a config with thread_id and an invoke call; on the right a MemorySaver storing the messages of thread 1 and an empty thread 2"
  caption="Explanatory board (not shown in the video)."
/>

Now the test that failed before, on the same thread:

```python
response=graph.invoke({"messages":"Hey what is my name"},config=config)

print(response['messages'][-1].content)
```

Output: `Your name is Krish.` He has just shown memory: the previous interaction was restored because the call used the
same `thread_id`. He asks one more (his typo included):

```python
response=graph.invoke({"messages":"Hey do you remember mmy name"},config=config)

print(response['messages'][-1].content)
```

Output: `Your name is Krish, right?`. In a finished application, he says, you keep the thread ID for the length of the user's
session, and the saved memory looks after the rest.

| Situation | What the second question gets |
| --- | --- |
| No checkpointer | A blank state: the bot does not know the name |
| Checkpointer, same `thread_id` | The earlier turns are restored: `Your name is Krish.` |
| Checkpointer, a different `thread_id` | A fresh conversation with no history |

## Streaming in LangGraph

Until now he has mostly used `graph.invoke`, with one early look at `stream`. This section is about the different
ways to get the response back while a graph runs. He builds a very small graph just for this, with a memory
checkpointer again.

```python
from langgraph.checkpoint.memory import MemorySaver
memory=MemorySaver()
```

```python
def superbot(state:State):
    return {"messages":[llm.invoke(state['messages'])]}
```

```python
graph=StateGraph(State)

## node
graph.add_node("SuperBot",superbot)
## Edges

graph.add_edge(START,"SuperBot")
graph.add_edge("SuperBot",END)


graph_builder=graph.compile(checkpointer=memory)


## Display
from IPython.display import Image, display
display(Image(graph_builder.get_graph().draw_mermaid_png()))
```

The node is called `SuperBot` and uses the plain `llm`; the graph is `START`, `SuperBot`, `END`. Note the naming
in this cell: the builder is called `graph`, and the **compiled** graph is stored in `graph_builder` (the names are
the other way round from the earlier cells, and every later streaming call uses `graph_builder`).

```python
## Invocation

config = {"configurable": {"thread_id": "1"}}

graph_builder.invoke({'messages':"Hi,My name is Krish And I like cricket"},config)
```

Output: a state with the human message and the AI's reply ("Hi Krish! Nice to meet you! Cricket is a great
sport, isn't it? Who's your favorite cricketer or team? ..."). He starts here because it shows the result as a
whole before comparing the streaming options.

### The two methods and the two modes

There are two methods, `.stream()` (synchronous) and `.astream()` (asynchronous), both for streaming results back. He
says that if you know Python you know what sync versus async means, and the part he cares about is the **stream
mode**, an extra parameter. He writes the definitions in a markdown cell:

- `values`: streams the full state of the graph after each node is called.
- `updates`: streams only the updates to the state of the graph after each node is called.

To make it concrete he draws a graph with Node 1, Node 2 and Node 3.

<Infographic
  src="/img/agentic-course/03-streaming-modes.svg"
  alt="Graph with Node 1, Node 2 and Node 3, the message each node writes, and what stream mode updates versus values emits at each step"
  caption="Redrawn from the whiteboard."
/>

Suppose Node 1 runs and the messages variable becomes `Hi`; Node 2 then makes it `My name is`; Node 3 makes it
`Krish`.

- With `updates`, each step reports only what that node just wrote: first `Hi`, then `My name is`, then `Krish`.
- With `values`, each step reports the whole accumulated list: first `[Hi]`, then `[Hi, My name is]`, then `[Hi, My name is, Krish]`.

He adds the multi-turn case: on a second input, `updates` again reports only the new messages, whereas `values`
reports the entire conversation, old turns first. In a streaming loop `values` therefore gives you the human message
again, with the previous conversation attached, while `updates` gives only the newest AI message.

### Seeing it in code

```python
# Create a thread
config = {"configurable": {"thread_id": "3"}}

for chunk in graph_builder.stream({'messages':"Hi,My name is Krish And I like cricket"},config,stream_mode="updates"):
    print(chunk)
```

Output (trimmed): `{'SuperBot': {'messages': [AIMessage(content="Nice to meet you, Krish! It's great to hear that you like
cricket! What's your favorite team or player in cricket?", ...)]}}`. With `updates` the chunk is keyed by the node
name and contains only the AI message. There is no human message here, as he points out.

Same call with `values`:

```python
for chunk in graph_builder.stream({'messages':"Hi,My name is Krish And I like cricket"},config,stream_mode="values"):
    print(chunk)
```

Output (trimmed): two chunks. The first is `{'messages': [HumanMessage(content='Hi,My name is Krish And I like cricket'
...)]}` and the second contains the human message and, after it, the AI message. Everything in the conversation is
printed, not just the newest line. (In the notebook this thread already held earlier turns, so the human message
shows up more than once, and he notes that the first chunk of a `values` stream is the input state before any node has
run.)

He then runs a fresh thread to separate the effects cleanly:

```python
# Create a thread
config = {"configurable": {"thread_id": "4"}}

for chunk in graph_builder.stream({'messages':"Hi,My name is Krish And I like cricket"},config,stream_mode="updates"):
    print(chunk)
```

Output: `{'SuperBot': {'messages': [AIMessage(content="Hello Krish! Nice to meet you! It's great to know that you
like cricket! Which team do you support?", ...)]}}`. Then a follow-up on the same thread in `values` mode:

```python
for chunk in graph_builder.stream({'messages':"I also like football"},config,stream_mode="values"):
    print(chunk)
```

Output (trimmed): a first chunk carrying the **previous conversation** (the first human message, the AI
reply) with the new human message "I also like football" appended, then a second chunk with the AI message
answering it. He says `values` keeps appending all the conversation, so you can stream through the whole thing, which
is useful when you want detailed information. `updates` is leaner.

| `stream_mode` | Each chunk contains | Good for |
| --- | --- | --- |
| `"updates"` | Only what the node that just ran changed, keyed by node name | Showing just the newest reply |
| `"values"` | The full state after each step | Seeing the whole conversation or debugging state |

### astream_events

A third technique gives much more detail:

```python
config = {"configurable": {"thread_id": "5"}}

async for event in graph_builder.astream_events({"messages":["Hi My name is Krish and I like to play cricket"]},config,version="v2"):
    print(event)
```

This is the asynchronous event stream. Printing each event shows many different kinds: `on_chain_start` (first for
the whole graph, `LangGraph`, then for the `SuperBot` node), then chat-model events, and a stream of
`on_chat_model_stream` events where each carries one **chunk** of the reply (`AIMessageChunk(content='Nice')`, then
`' to'`, and so on). He suggests this when you need much more detail for debugging each step, or token-by-token
output. In a notebook you can use `async for` directly; in a normal script it must run inside an `async` function.

## Human in the loop

The last LangGraph topic is **human in the loop**, which he also calls human feedback. The idea is that while a graph is
running you can **interrupt** it, ask a person, and then let it continue with the person's answer. He opens a new
whiteboard page titled "Human Feedback In the Loop".

<Infographic
  src="/img/agentic-course/03-human-loop.svg"
  alt="Whiteboard: START to Chatbot with LLM plus tools, a ToolNode holding Tavily and human assistance with feedback returning to the chatbot, then END, and a complex workflow with an interrupt"
  caption="Redrawn from the whiteboard."
/>

The graph is the one you already know: `START`, a chatbot (an LLM with tools bound), a tool node, and `END`. The
new idea is in the tool node. As before it holds Tavily, but there is now a second, custom tool for **human
assistance**. When the user's input leads the chatbot to call the human-assistance tool rather than Tavily, the
graph stops, a human supplies feedback, and the chatbot carries on using that feedback.

Why would you want this? His example is a complex workflow with two steps in which the second must not run unless
a person agrees. Between them you place an **interrupt**; if the human gives good feedback ("yes, continue"), the
workflow proceeds.

### The code

He works in a new notebook, `2-HumanAssistance/humanintheloop.ipynb`, and the first cell loads the model as before:

```python
import os
from langchain.chat_models import init_chat_model
llm=init_chat_model("groq:llama3-8b-8192")
llm
```

Then everything else is one large cell:

```python
from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import Command, interrupt

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

tool = TavilySearch(max_results=2)
tools = [tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
```

He first reads through the imports. Most are old friends: Tavily, `TypedDict`, `MemorySaver`, `StateGraph`,
`add_messages` (the reducer), `ToolNode` and `tools_condition`. Two are new: `Command` and `interrupt` from
`langgraph.types`. `interrupt` forcibly pauses the workflow so a human can answer; `Command` is how you send
the answer back and resume. There is also `tool` from `langchain_core.tools`, a **decorator** that turns a
function into a tool, so it can be bound to the LLM like Tavily was.

The heart is the `human_assistance` tool:

- `@tool` converts the function into a LangChain tool; its doc string ("Request assistance from a human.") is the
  description the LLM will read, exactly like the `multiply` doc string earlier.
- `interrupt({"query": query})` **pauses the graph** right here and surfaces the query to whoever is running it. Whatever
  you resume with is returned by the `interrupt` call.
- `return human_response["data"]` returns the human's answer, taken from the `data` key, as the tool's output.

The rest is the familiar pattern: the tools list now holds `[tool, human_assistance]`, it is bound with `bind_tools`,
and the `chatbot` node, the tool node and the same `tools_condition` routing are added. The edge `tools` to
`chatbot` makes it a ReAct-style loop, and `START` to `chatbot` starts it. Then the memory is attached and the graph
compiled and drawn:

```python
memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)
```

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

The picture shows `__start__`, `chatbot`, `tools` and `__end__`, with the loop between `chatbot` and `tools`. He observes
that the interrupt will happen inside the `tools` node, because that is where `human_assistance` lives.

:::warning A checkpointer is required
`interrupt` can only pause a run if there is somewhere to save its state. That is why the graph is compiled with
`checkpointer=memory`, and why every call below passes a `thread_id`. Without them there is nothing to resume.
:::

:::note Two details in this cell that are easy to misread
First, the comment in the `chatbot` node says parallel tool calling is disabled, but the code does not do that:
`llm.bind_tools(tools)` is called with default settings. Disabling it (for providers that support the option)
matters because, on resume, LangGraph re-runs the node that was interrupted from its start; if the model had asked
for Tavily and human help in the same message, the Tavily call would be repeated. Second, the line
`tool = TavilySearch(max_results=2)` reuses the name `tool`, replacing the decorator you imported. It only works
because the decorator was already used on the line above; do not use `@tool` again after that line in the same
notebook.
:::

### First run: the LLM picks Tavily instead

```python
user_input = "I need some expert guidance and assistance for building an AI agent. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": user_input},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

This is the streaming pattern from the last section: `stream_mode="values"` yields the state after each step, and
`pretty_print()` prints the newest message in each. His first attempt used a slightly different sentence ("I need
some expert guidance for building AI agents, could you request assistance for me?"). The LLM made a call to
**Tavily search** instead, got a blog post and a YouTube result, and answered normally. The wording of the
request decides which tool the LLM thinks fits, so he reworded it to mention "assistance" explicitly, which
matches the doc string of `human_assistance`.

Output of the reworded request:

```text
================================ Human Message =================================

I need some expert guidance and assistance for building an AI agent. Could you request assistance for me?
================================== Ai Message ==================================
Tool Calls:
  human_assistance (8jwn397aj)
 Call ID: 8jwn397aj
  Args:
    query: expert guidance and assistance for building an AI agent
```

The graph now stops after that AI message. The tool call was made and the interrupt fired, so it is waiting for a human.
There is no tool message yet.

### Resuming with the human's answer

The human writes a reply, and then the script sends it back with a `Command`:

```python
human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"data": human_response})

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

`Command(resume={"data": human_response})` says "resume the paused run, and give the interrupt this value". The
dictionary key `data` must match what the tool reads (`human_response["data"]`). Passing the command, with the
**same `config`** so it finds the same thread, to `graph.stream` continues the run.

Output (trimmed):

```text
================================== Ai Message ==================================
Tool Calls:
  human_assistance (8jwn397aj)
 Call ID: 8jwn397aj
  Args:
    query: expert guidance and assistance for building an AI agent
================================= Tool Message =================================
Name: human_assistance

We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents.
================================== Ai Message ==================================

Thank you for the recommendation! LangGraph seems like a great tool for building AI agents. I'll make sure to keep that in mind.

To further assist you, I'd like to ask a few follow-up questions:

1. What specific aspects of building an AI agent are you struggling with or would like guidance on?
2. What is your background in AI and machine learning, and what is your goal for building this agent?
3. Have you considered any specific architectures or frameworks for building your agent, or would you like some recommendations?

Please let me know, and I'll do my best to provide more tailored guidance and assistance.
```

(The first message printed again is the stored state being replayed, because `values` mode starts with the current
state.) The human's text appears as a `ToolMessage` from `human_assistance`, and the LLM then reads it, thanks the human
and asks follow-up questions. He notes that you can interrupt again, whenever you like, and supply another answer,
so a finished chatbot can take human feedback any number of times.

<Infographic
  src="/img/agentic-course/03-interrupt-resume.svg"
  alt="Three lanes for your script, the graph and the human showing stream, tool call, interrupt and pause, human answer, Command resume, and the LLM reply"
  caption="Explanatory board (not shown in the video)."
/>

:::tip If your re-runs behave oddly
On screen, the same thread was reused while he re-ran the first cell several times, so tool-call IDs changed between
runs. If a re-run produces something confusing, use a new `thread_id` for a clean experiment.
:::

## Building MCP servers and an MCP client from scratch

The video then cuts to a separate recording whose goal is to **build your own MCP servers** and plug them into an
app. It is introduced with a slide with three components.

<Infographic
  src="/img/agentic-course/03-mcp-architecture.svg"
  alt="Slide redrawn: MCP servers on the left offering tools, MCP clients in the middle keeping one to one connections, apps on the right including Claude desktop and a LangGraph agent, joined by load_mcp_tools"
  caption="Redrawn from the slide."
/>

- **MCP servers** provide context, tools and prompts to clients. Think of third-party companies building services
  (simple calculations, or integrations with third-party APIs) and exposing them. A server can offer many tools,
  such as `add()` and `divide()` on a "Math" server, or a "Data" server.
- **MCP clients** keep a one-to-one connection with a server, inside the host app.
- The **app** is the thing the user sees. It can be Claude Desktop, or any application you build yourself. In the
  slide's lower row a Python client is joined to a LangGraph agent by `load_mcp_tools`.

He says he has covered MCP in depth in an earlier module and here focuses on building it from scratch.

### The application we will build

On the whiteboard he draws the app. It is a chatbot application, built with LangChain or LangGraph, containing an LLM.
A user gives an input and the LLM must decide whether an MCP server call is needed. The MCP server has
tools: addition and multiplication, and a **weather call API**. If the user asks for the weather of New York or
Bangalore, the LLM cannot answer by itself because it has no live data, so it makes a tool call, this time through the
**MCP protocol**. Inside the app there is an **MCP client** that does the talking.

<Infographic
  src="/img/agentic-course/03-mcp-app-board.svg"
  alt="Whiteboard: an application with chatbot, LLM and MCP client sends input over the MCP protocol to an MCP server offering add, multiplication and a weather call API, with stdio and http transports"
  caption="Redrawn from the whiteboard."
/>

He describes the conversation in order. The input arrives; the MCP server provides the list of its tools and information
about each tool; the LLM decides; and the LLM passes the relevant input to the MCP server, which runs the tool and
returns the result. Then he lists what he will build:

1. An **MCP server**, from scratch, with the `FastMCP` library.
2. An **MCP client**, using LangChain's MCP adapters library.
3. Two different **transports**, because the way a client talks to a server depends on the transport. He will run one
   tool with `stdio` (standard input and output) and another with HTTP, and explain how they differ.

### Set up the project

He opens a new empty folder (`mcpdemolangchain`) in the Cursor IDE and initialises it with uv, exactly as in the first
section:

```bash
uv init
uv venv
```

Then he activates the environment by copying uv's activation command. The generated `.python-version` shows
`3.13` again. He creates `requirements.txt`:

```text
langchain-groq
langchain-mcp-adapters
mcp
```

`langchain-mcp-adapters` lets LangChain and LangGraph use MCP tools as ordinary tools. `mcp` is the Python MCP package,
and it contains `FastMCP`, which he describes as the quick, "Pythonic" way to build MCP servers and clients, with
a README example of a server in a few lines. Install them:

```bash
uv add -r requirements.txt
```

After this `pyproject.toml` lists `langchain-groq>=0.3.2`, `langchain-mcp-adapters>=0.1.7` and `mcp>=1.9.4`. Later he also
installs `langgraph` (he adds it to `requirements.txt` and runs the same command again) because the client needs
`create_react_agent`.

:::note Two FastMCP libraries
The README he shows on screen belongs to the stand-alone `fastmcp` package, but the code he types imports
`FastMCP` from `mcp.server.fastmcp`, which is the version that ships inside the official `mcp` package. That is
why `mcp` is in the requirements and `fastmcp` is not. Do not mix the two import paths.
:::

:::warning A fresh install today pulls `mcp` 2.x, which breaks this import
`mcp>=1.9.4` has no upper bound, so a new environment resolves to `mcp` 2.x. In that release the
`mcp.server.fastmcp` module the video imports no longer exists (the class was renamed `MCPServer`), so the server
files below fail with an `ImportError`. To follow the video exactly, pin the version in `requirements.txt` with
`mcp<2` before running `uv add -r requirements.txt`. The improvements chapter lists this under dependency drift.
:::

### The first server: math over stdio

He creates `mathserver.py` and builds the server:

```python
from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Math")

@mcp.tool()
def add(a:int,b:int)->int:
    """_summary_
    Add to numbers
    """
    return a+b

@mcp.tool()
def multiple(a:int,b:int)-> int:
    """Multiply two numbers"""
    return a*b

#The transport="stdio" argument tells the server to:

#Use standard input/output (stdin and stdout) to receive and respond to tool function calls

if __name__=="__main__":
    mcp.run(transport="stdio")
```

Reading it:

- `FastMCP("Math")` creates the server. The string is just the **server's name**, not a tool name (he corrects
  himself on this on camera).
- `@mcp.tool()` registers each function as a tool the server offers. The type hints and the doc string are what a
  client's LLM will read to decide which tool to call, just like the doc string in the LangChain tool earlier.
- `add` and a second function return the sum and the product. He starts with simple tools to teach the mechanics: you can write
  any tool you like, but it matters to understand the basics first.
- `mcp.run(transport="stdio")` starts the server over the **stdio** transport.

:::note Small slips on screen in this file
The doc string of `add` still contains the leftover `_summary_` text from VS Code's snippet and says "Add to
numbers"; the second function is named `multiple`, not `multiply`. The model reads the function name too, so the
tool is exposed to the LLM as `multiple`. It still works, because the doc string says "Multiply two numbers", but
a clearer pair is `add` and `multiply` with a doc string such as "Add two numbers".
:::

### What `stdio` really means

He stops to explain the transport, because he says many people write this code without being able to explain it.
`stdio` means the server uses **standard input and output** to receive and respond to tool calls. Picture the server running in a
command prompt: a client that wants to talk to it writes a request into the prompt's input and reads the answer from
its output. It does not listen on any port and has no URL. This is very helpful when you are testing locally: the server runs on your
own machine and the client talks to it directly through the command line.

### The second server: weather over HTTP

For the second server he creates `weather.py`. In real life this would call a third-party weather API; for the demo it
returns a constant:

```python
from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Weather")

@mcp.tool()
async def get_weather(location:str)->str:
    """Get the weather location."""
    return "It's always raining in California"

if __name__=="__main__":
    mcp.run(transport="streamable-http")
```

The shape is identical to the first server, with three differences. The server is named `Weather`. The tool is
`async def`, because a real implementation would await a network call. And the transport is
`"streamable-http"`. He says this is only a placeholder (it may not be true weather, he jokes) and that the point is the
structure: you can put any API-calling code inside the tool.

### Seeing the difference when you run them

He runs the weather file first:

```bash
python weather.py
```

Output:

```text
INFO:     Started server process [37360]
INFO:     Waiting for application startup.
StreamableHTTP session manager started
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

With the streamable HTTP transport the server runs as an **API service on a URL**: by default `localhost` on port 8000.
He says you can also set the URL and port yourself after the transport option, and he will mention production
deployment briefly at the end without showing a cloud deployment.

Now the math server:

```bash
python mathserver.py
```

Nothing is printed. That is correct, not a bug. A `stdio` server is waiting on its standard input; it is not an
HTTP service and prints no URL. It is meant to be started by a client, not by you.

<Infographic
  src="/img/agentic-course/03-mcp-transports.svg"
  alt="Two transports side by side: stdio, where the client spawns a child process and talks over stdin and stdout, and streamable HTTP, where a web server on localhost 8000 answers HTTP requests"
  caption="Explanatory board (not shown in the video)."
/>

| | `stdio` (math server) | Streamable HTTP (weather server) |
| --- | --- | --- |
| How the client reaches it | Starts it as a child process and talks over standard input and output | Sends HTTP requests to a URL |
| Prints anything when run by hand | No | Yes, a Uvicorn start-up log with `http://127.0.0.1:8000` |
| Needs a port or URL | No | Yes (default port 8000) |
| Who starts it | The client, from a command and arguments | You, before the client connects |
| Best for | Local testing on your machine | A server that runs as a service, possibly elsewhere |

### The client: one client, two servers

Now the piece that ties it together. He creates `client.py` and imports what he needs. The first import builds the
client:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import asyncio
```

`MultiServerMCPClient` is, per the documentation he reads on screen, a client that supports **several servers at
once**: you give it a dictionary that maps server names to their connection settings. `create_react_agent` (from
`langgraph.prebuilt`) builds a ready-made ReAct agent, the same loop you built by hand earlier, from a model and a list
of tools, and it is the part that integrates the LLM with all the MCP tools. `ChatGroq` is the model. `asyncio`
is needed because the MCP client calls are asynchronous.

The `.env` for this project holds the Groq key as `GROQ_API_KEY`. Then the body:

```python
async def main():
    client=MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":["mathserver.py"], ## Ensure correct absolute path
                "transport":"stdio",

            },
            "weather": {
                "url": "http://localhost:8000/mcp",  # Ensure server is running here
                "transport": "streamable_http",
            }

        }
    )

    import os
    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

    tools=await client.get_tools()
    model=ChatGroq(model="qwen-qwq-32b")
    agent=create_react_agent(
        model,tools
    )

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )

    print("Math response:", math_response['messages'][-1].content)

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in California?"}]}
    )
    print("Weather response:", weather_response['messages'][-1].content)

asyncio.run(main())
```

How the client config reads:

- The `"math"` entry says: to reach this server, run the command `python` with the argument `mathserver.py`, and use the
  `stdio` transport. The client itself launches the server process. He reminds you to give the **absolute
  path** if the file is in another folder; here it sits in the working directory.
- The `"weather"` entry gives a **URL** and the `streamable_http` transport. Nothing launches it; it must already be
  running, so he keeps the earlier terminal with `python weather.py` open. The `/mcp` at the end of the URL is the path on
  which the server exposes its MCP endpoint.
- `client.get_tools()` connects to both servers and returns their tools, here the math tools and the weather tool, as LangChain tools. It is awaited
  because it is async.
- `ChatGroq(model="qwen-qwq-32b")` is a Qwen 32-billion-parameter reasoning model on Groq.
- `create_react_agent(model, tools)` needs only those two arguments and gives back the agent.
- `agent.ainvoke(...)` runs it asynchronously. The message is in the role and content format; the answer is the content of the last message.
- Because `main` is `async`, it cannot be called directly. `asyncio.run(main())` runs it, as he explains.

:::warning Two spellings of the same transport
The server says `transport="streamable-http"` (with a hyphen) while the client config says `"streamable_http"`
(with an underscore). They are different APIs with different spelling conventions, and each is correct where it
appears. Copy both exactly as written.
:::

### Running the client

With the weather server still running in another terminal, he runs the client:

```bash
python client.py
```

First attempt, math only: Output:

```text
Math response: The result of (3 + 5) multiplied by 12 is **96**.

Here's the step-by-step breakdown:
1. Addition: 3 + 5 = 8
2. Multiplication: 8 × 12 = 96
```

The agent chose the `add` tool and the multiply tool, and the calls went through the **stdio** transport to a math server
the client launched itself. 8 times 12 is 96, which is right. (He reads the question aloud as "3 plus 5 times 12";
the text on screen has the brackets, and the answer 96 matches the bracketed version.)

For the weather he adds the second call. The first version of that message said "weather in NYC"; he changes it
to California because the tool is hard-coded to always answer for California. Running again:

```text
Math response: The result of (3 + 5) multiplied by 12 is **96**.
...
Weather response: The tool indicated that it's always raining in California, but in reality, California has a diverse climate ranging from Mediterranean to arid. Would you like me to provide more accurate weather information for a specific city or region in California?
```

The weather answer comes from the HTTP server. The model also adds its own knowledge that California's climate is
varied, which he finds a nice touch; the tool only returned the constant string, and what the user finally sees depends
on what your real API implementation returns.

:::note A run that failed, and what it means
One of his runs crashed with `groq.BadRequestError: Error code: 400 ... 'Failed to call a function. Please
adjust your prompt. See 'failed_generation' for more details.' ... 'code': 'tool_use_failed'`, and the failed call
shown in the message used the tool name `multiple`. This is the model producing a malformed tool call, not a bug in the
MCP code. He simply runs it again and it passes. If you see this, retry; if it keeps happening, switch to a
different model or rename the tool and sharpen its doc string.
:::

### What you have just built

He closes with a recap. One client talks to two independent MCP servers, each running on its own: the math server
over **stdio**, so the traffic goes through the command line, and the weather server over **HTTP**, where it runs at a
URL, `localhost:8000` with `/mcp`. The adapters from LangChain are what make the MCP tools usable by the agent.
The tools in the math server can be anything you want (addition, subtraction, anything), and the weather one stands for any third-party API. When you
are finished you can close the servers. He thanks you for watching, and that is the end of the Part 1 recording.

---

## Additions from the course repository, not in this video

:::note Not from the video
Everything below is an **addition**. It comes from notebooks in the author's repository
(`Agentic-LanggraphCrash-course`) that belong to his announced Part 2 but are **not taught in the recording**. They are
included so that the repository is fully covered, and the commentary is written for these notes, not taken from
the video. Check the code against your installed versions before relying on it.
:::

### Debugging with LangGraph Studio: `agent.py` and `langgraph.json`

The folder `3-Debugging` is a small project designed to be opened in **LangGraph Studio**, a visual tool for running and
inspecting a graph. It needs two files and an environment file.

<Infographic
  src="/img/agentic-course/03-studio-debug.svg"
  alt="The 3-Debugging folder with agent.py, langgraph.json and .env feeding the langgraph dev command, which starts a local server, LangGraph Studio and LangSmith traces"
  caption="Explanatory board (not shown in the video)."
/>

`agent.py` builds the graph and exposes the compiled result as a module variable:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"]="TestProject"

from langchain.chat_models import init_chat_model
llm=init_chat_model("groq:llama3-8b-8192")

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


def make_tool_graph():
    ## Graph With tool Call
    from langchain_core.tools import tool

    @tool
    def add(a:float,b:float):
        """Add two number"""
        return a+b
    tools=[add]
    tool_node=ToolNode([add])

    llm_with_tool=llm.bind_tools([add])

    def call_llm_model(state:State):
        return {"messages":[llm_with_tool.invoke(state['messages'])]}
    

        ## Grpah
    builder=StateGraph(State)
    builder.add_node("tool_calling_llm",call_llm_model)
    builder.add_node("tools",ToolNode(tools))

    ## Add Edges
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition
    )
    builder.add_edge("tools","tool_calling_llm")

    ## compile the graph
    graph=builder.compile()
    return graph

tool_agent=make_tool_graph()
```

What is new compared with the chapter so far:

- The four `os.environ[...]` lines switch on **LangSmith tracing**: `LANGSMITH_TRACING="true"` turns it on,
  `LANGSMITH_API_KEY` identifies you, and `LANGSMITH_PROJECT="TestProject"` names the project in which the runs will be recorded.
  (The `.env` uses the older variable name `LANGCHAIN_API_KEY`, which the file copies into `LANGSMITH_API_KEY`.)
- The graph is built **inside a function**, `make_tool_graph()`, and the compiled graph is returned. The last line,
  `tool_agent=make_tool_graph()`, stores it in a module-level variable. That variable is what Studio will load.
- The tool is a different one, `add`, with `float` arguments and a short doc string.

`langgraph.json` tells LangGraph where everything is:

```json
{
    "dependencies":["."],
    "graphs":{
        "tool_agent":"./agent.py:tool_agent"
    },
    "env":"../.env"
}
```

| Key | Meaning |
| --- | --- |
| `dependencies` | Where to find the project's code and packages. `["."]` means this folder. |
| `graphs` | A map from a graph name to `path/to/file.py:variable`. Here the name `tool_agent` points at the `tool_agent` variable in `agent.py`. |
| `env` | The path to the `.env` file with your keys. Here it is one folder up (`../.env`). |

To launch it you install the LangGraph command-line tool and run, from the folder that contains `langgraph.json`:

```bash
uv add "langgraph-cli[inmem]"
langgraph dev
```

This is not in the notebook (the notebook `debugging.ipynb` only rebuilds the same graph and invokes it with "What
is machine learning" and "what is 5 plus 20"); the command is the standard way to start the local development server
that Studio connects to. In Studio you see the graph drawn, can type an input, run it and inspect the state after each
node, while LangSmith records a trace of every run under the `TestProject` project.

### A simple multi-agent system

`Agents/multiaiagent.ipynb` explores two ways of making several agents cooperate over one shared state.

<Infographic
  src="/img/agentic-course/03-multi-agent.svg"
  alt="Left: a simple pipeline START to researcher to writer to END. Right: a supervisor that routes to researcher, analyst and writer and finishes with END"
  caption="Explanatory board (not shown in the video)."
/>

**Version 1: a two-agent pipeline.** The first cells set up the imports, load the Groq key, and define a state that adds
a field for routing:

```python
import os
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
```

```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
```

```python
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
```

```python
## Define the state
class AgentState(MessagesState):
    next_agent:str #ehich agent should go next 
```

`MessagesState` is a ready-made state class that already has a `messages` key with the `add_messages` reducer, so you
only add what you need on top, here `next_agent`. Two tools and the model follow:

```python
# Create simple tools
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Using Tavily for web search
    search = TavilySearchResults(max_results=3)
    results = search.invoke(query)
    return str(results)

@tool
def write_summary(content: str) -> str:
    """Write a summary of the provided content."""
    # Simple summary generation
    summary = f"Summary of findings:\n\n{content[:500]}..."
    return summary
```

```python
from langchain.chat_models import init_chat_model

llm=init_chat_model("groq:llama-3.1-8b-instant")
llm
```

Then two agent functions. Each is simply a node: it prepends a system message that defines its role, calls the model,
and returns its message plus who goes next.

```python
# Define agent functions (simpler approach)
def researcher_agent(state: AgentState):
    """Researcher agent that searches for information"""
    
    messages = state["messages"]
    
    # Add system message for context
    system_msg = SystemMessage(content="You are a research assistant. Use the search_web tool to find information about the user's request.")
    
    # Call LLM with tools
    researcher_llm = llm.bind_tools([search_web])
    response = researcher_llm.invoke([system_msg] + messages)
    
    # Return the response and route to writer
    return {
        "messages": [response],
        "next_agent": "writer"
    }
```

```python
def writer_agent(state: AgentState):
    """Writer agent that creates summaries"""
    
    messages = state["messages"]
    
    # Add system message
    system_msg = SystemMessage(content="You are a technical writer. Review the conversation and create a clear, concise summary of the findings.")
    
    # Simple completion without tools
    response = llm.invoke([system_msg] + messages)
    
    return {
        "messages": [response],
        "next_agent": "end"
    }
```

```python
# Tool executor node
def execute_tools(state: AgentState):
    """Execute any pending tool calls"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if there are tool calls to execute
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Create tool node and execute
        tool_node = ToolNode([search_web, write_summary])
        response = tool_node.invoke(state)
        return response
    
    # No tools to execute
    return state
```

```python
# Build graph
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)

# Define flow
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)
final_workflow=workflow.compile()

final_workflow
```

```python
response=final_workflow.invoke({"messages":"Reasearch about the usecase of agentic ai in business"})
```

```python
response["messages"][-1].content
```

The graph is a straight line: `researcher` then `writer` then `END`, with `set_entry_point("researcher")` as an older way of
writing `add_edge(START, "researcher")`. The output is a structured summary of agentic-AI use cases in business, written
by the writer agent. Be aware of what this version does not do: `execute_tools` is defined but never added to the graph,
so if the researcher's model asks for `search_web` the call is **not executed**, and the writer works from whatever
the conversation holds. The graph is also built on `MessagesState` rather than `AgentState`, so the `next_agent` field is
returned by nodes but is not declared in the state this graph uses.

**Version 2: a supervisor.** The second half of the notebook (headed "supervise Multi Ai Agent Architecture") replaces
the fixed line with a decision-maker. The state holds each agent's product:

```python
from typing import TypedDict, Annotated, List, Literal, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
import random
from datetime import datetime
```

```python
# ===================================
# State Definition
# ===================================

# ===================================
# State Definition
# ===================================

class SupervisorState(MessagesState):
    """State for the multi-agent system"""
    next_agent: str = ""
    research_data: str = ""
    analysis: str = ""
    final_report: str = ""
    task_complete: bool = False
    current_task: str = ""
```

The supervisor is an LLM chain whose prompt describes the team and the current progress, and asks for one word:

```python
# ===================================
# Supervisor with Groq LLM
# ===================================
from langchain_core.prompts import ChatPromptTemplate
def create_supervisor_chain():
    """Creates the supervisor decision chain"""
    
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor managing a team of agents:
        
1. Researcher - Gathers information and data
2. Analyst - Analyzes data and provides insights  
3. Writer - Creates reports and summaries

Based on the current state and conversation, decide which agent should work next.
If the task is complete, respond with 'DONE'.

Current state:
- Has research data: {has_research}
- Has analysis: {has_analysis}
- Has report: {has_report}

Respond with ONLY the agent name (researcher/analyst/writer) or 'DONE'.
"""),
        ("human", "{task}")
    ])
    
    return supervisor_prompt | llm
```

```python
def supervisor_agent(state: SupervisorState) -> Dict:
    """Supervisor decides next agent using Groq LLM"""
    
    messages = state["messages"]
    task = messages[-1].content if messages else "No task"
    
    # Check what's been completed
    has_research = bool(state.get("research_data", ""))
    has_analysis = bool(state.get("analysis", ""))
    has_report = bool(state.get("final_report", ""))
    
    # Get LLM decision
    chain = create_supervisor_chain()
    decision = chain.invoke({
        "task": task,
        "has_research": has_research,
        "has_analysis": has_analysis,
        "has_report": has_report
    })
    
    # Parse decision
    decision_text = decision.content.strip().lower()
    print(decision_text)
    
    # Determine next agent
    if "done" in decision_text or has_report:
        next_agent = "end"
        supervisor_msg = "✅ Supervisor: All tasks complete! Great work team."
    elif "researcher" in decision_text or not has_research:
        next_agent = "researcher"
        supervisor_msg = "📋 Supervisor: Let's start with research. Assigning to Researcher..."
    elif "analyst" in decision_text or (has_research and not has_analysis):
        next_agent = "analyst"
        supervisor_msg = "📋 Supervisor: Research done. Time for analysis. Assigning to Analyst..."
    elif "writer" in decision_text or (has_analysis and not has_report):
        next_agent = "writer"
        supervisor_msg = "📋 Supervisor: Analysis complete. Let's create the report. Assigning to Writer..."
    else:
        next_agent = "end"
        supervisor_msg = "✅ Supervisor: Task seems complete."
    
    return {
        "messages": [AIMessage(content=supervisor_msg)],
        "next_agent": next_agent,
        "current_task": task
    }
```

The three workers share one pattern: read what they need from the state, call the LLM with a role-specific prompt, write their
product into their own state key, and hand control back to the supervisor.

```python
# ===================================
# Agent 1: Researcher (using Groq)
# ===================================

def researcher_agent(state: SupervisorState) -> Dict:
    """Researcher uses Groq to gather information"""
    
    task = state.get("current_task", "research topic")
    
    # Create research prompt
    research_prompt = f"""As a research specialist, provide comprehensive information about: {task}

    Include:
    1. Key facts and background
    2. Current trends or developments
    3. Important statistics or data points
    4. Notable examples or case studies
    
    Be concise but thorough."""
    
    # Get research from LLM
    research_response = llm.invoke([HumanMessage(content=research_prompt)])
    research_data = research_response.content
    
    # Create agent message
    agent_message = f"🔍 Researcher: I've completed the research on '{task}'.\n\nKey findings:\n{research_data[:500]}..."
    
    return {
        "messages": [AIMessage(content=agent_message)],
        "research_data": research_data,
        "next_agent": "supervisor"
    }
```

```python
# ===================================
# Agent 2: Analyst (using Groq)
# ===================================

def analyst_agent(state: SupervisorState) -> Dict:
    """Analyst uses Groq to analyze the research"""
    
    research_data = state.get("research_data", "")
    task = state.get("current_task", "")
    
    # Create analysis prompt
    analysis_prompt = f"""As a data analyst, analyze this research data and provide insights:

Research Data:
{research_data}

Provide:
1. Key insights and patterns
2. Strategic implications
3. Risks and opportunities
4. Recommendations

Focus on actionable insights related to: {task}"""
    
    # Get analysis from LLM
    analysis_response = llm.invoke([HumanMessage(content=analysis_prompt)])
    analysis = analysis_response.content
    
    # Create agent message
    agent_message = f"📊 Analyst: I've completed the analysis.\n\nTop insights:\n{analysis[:400]}..."
    
    return {
        "messages": [AIMessage(content=agent_message)],
        "analysis": analysis,
        "next_agent": "supervisor"
    }
```

```python
# ===================================
# Agent 3: Writer (using Groq)
# ===================================

def writer_agent(state: SupervisorState) -> Dict:
    """Writer uses Groq to create final report"""
    
    research_data = state.get("research_data", "")
    analysis = state.get("analysis", "")
    task = state.get("current_task", "")
    
    # Create writing prompt
    writing_prompt = f"""As a professional writer, create an executive report based on:

Task: {task}

Research Findings:
{research_data[:1000]}

Analysis:
{analysis[:1000]}

Create a well-structured report with:
1. Executive Summary
2. Key Findings  
3. Analysis & Insights
4. Recommendations
5. Conclusion

Keep it professional and concise."""
    
    # Get report from LLM
    report_response = llm.invoke([HumanMessage(content=writing_prompt)])
    report = report_response.content
    
    # Create final formatted report
    final_report = f"""
📄 FINAL REPORT
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Topic: {task}
{'='*50}

{report}

{'='*50}
Report compiled by Multi-Agent AI System powered by Groq
"""
    
    return {
        "messages": [AIMessage(content=f"✍️ Writer: Report complete! See below for the full document.")],
        "final_report": final_report,
        "next_agent": "supervisor",
        "task_complete": True
    }
```

A **router** function turns `next_agent` into the name of the next node. It is attached to every node with conditional edges:

```python
# ===================================
# Router Function
# ===================================

def router(state: SupervisorState) -> Literal["supervisor", "researcher", "analyst", "writer", "__end__"]:
    """Routes to next agent based on state"""
    
    next_agent = state.get("next_agent", "supervisor")
    
    if next_agent == "end" or state.get("task_complete", False):
        return END
        
    if next_agent in ["supervisor", "researcher", "analyst", "writer"]:
        return next_agent
        
    return "supervisor"
```

```python
# Create workflow
workflow = StateGraph(SupervisorState)

# Add nodes
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("writer", writer_agent)

# Set entry point
workflow.set_entry_point("supervisor")

# Add routing
for node in ["supervisor", "researcher", "analyst", "writer"]:
    workflow.add_conditional_edges(
        node,
        router,
        {
            "supervisor": "supervisor",
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            END: END
        }
    )

graph=workflow.compile()
```

```python
response=graph.invoke(HumanMessage(content="What are the benefits and risks of AI in healthcare?"))
```

```python
response['final_report']
```

The run prints the supervisor's decisions as they happen (`researcher`, `analyst`, `writer`) and the final report is
the formatted document the writer built.

:::warning A bug in this notebook
`supervisor_agent` takes the task from `messages[-1]`, the **latest** message, and stores it as `current_task` on every
call. After the researcher has replied, the latest message is the researcher's own progress note, so the topic
passed to later agents is no longer the user's question. The saved output shows exactly that: the final report's
"Topic" is the analyst's progress message and it talks about a "No Task" template. A fix is to set `current_task` once, from the
first human message, and keep it. The notebook also passes a bare `HumanMessage` to `invoke`; pass a state dictionary,
`{"messages": [...]}`, as everywhere else in this chapter.
:::

The notebook ends with a markdown cell sketching a **hierarchical** system (a CEO above a research team leader with data and
market researchers, and a writing team leader with technical and summary writers). It is a design outline in a docstring,
not working code.

### Multimodal RAG over a PDF with images

`4-Multimodal/1-multimodalopenai.ipynb` builds a question-answering pipeline over a PDF that contains both text and a chart,
using **one embedding model for both**: CLIP. Text chunks and images land in the same vector space, so a text question can
retrieve either. The sample file is `multimodal_sample.pdf` (an "Annual Revenue Overview" page with a chart).

<Infographic
  src="/img/agentic-course/03-multimodal-rag.svg"
  alt="Pipeline: PDF split into text chunks and images, both embedded by CLIP into FAISS, a question retrieves top hits, a multimodal message is built and a vision LLM answers"
  caption="Explanatory board (not shown in the video)."
/>

Imports and the CLIP model:

```python
import fitz  # PyMuPDF
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
import os
import base64
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
```

```python
###Clip Model
import os
from dotenv import load_dotenv
load_dotenv()

## set up the environment
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

### initialize the Clip Model for unified embeddings
clip_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
```

Two embedding functions turn an image or a piece of text into a normalised vector (unit length, so similarity is a plain dot product):

```python
### Embedding functions
def embed_image(image_data):
    """Embed image using CLIP"""
    if isinstance(image_data, str):  # If path
        image = Image.open(image_data).convert("RGB")
    else:  # If PIL Image
        image = image_data
    
    inputs=clip_processor(images=image,return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        # Normalize embeddings to unit vector
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    
def embed_text(text):
    """Embed text using CLIP."""
    inputs = clip_processor(
        text=text, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=77  # CLIP's max token length
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        # Normalize embeddings
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
```

CLIP can only read 77 tokens of text, which is why the text is split into small chunks first:

```python
## Process PDF
pdf_path="multimodal_sample.pdf"
doc=fitz.open(pdf_path)
# Storage for all documents and embeddings
all_docs = []
all_embeddings = []
image_data_store = {}  # Store actual image data for LLM

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
```

```python
for i,page in enumerate(doc):
    ## process text
    text=page.get_text()
    if text.strip():
        ##create temporary document for splitting
        temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
        text_chunks = splitter.split_documents([temp_doc])

        #Embed each chunk using CLIP
        for chunk in text_chunks:
            embedding = embed_text(chunk.page_content)
            all_embeddings.append(embedding)
            all_docs.append(chunk)



    ## process images
    ##Three Important Actions:

    ##Convert PDF image to PIL format
    ##Store as base64 for GPT-4V (which needs base64 images)
    ##Create CLIP embedding for retrieval

    for img_index, img in enumerate(page.get_images(full=True)):
        try:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Create unique identifier
            image_id = f"page_{i}_img_{img_index}"
            
            # Store image as base64 for later use with GPT-4V
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            image_data_store[image_id] = img_base64
            
            # Embed image using CLIP
            embedding = embed_image(pil_image)
            all_embeddings.append(embedding)
            
            # Create document for image
            image_doc = Document(
                page_content=f"[Image: {image_id}]",
                metadata={"page": i, "type": "image", "image_id": image_id}
            )
            all_docs.append(image_doc)
            
        except Exception as e:
            print(f"Error processing image {img_index} on page {i}: {e}")
            continue

doc.close()
```

For each page the loop does two jobs. Text is split into 500-character chunks with 100 overlap and each chunk is embedded. Each image is
extracted, saved as base64 (the vision model will need the raw picture later), embedded, and represented in the index by a
small placeholder document `[Image: page_0_img_0]` whose metadata says `type: image`. For the sample file the store ends up
with two entries (one text chunk and one image), and the embeddings array has shape `(2, 512)`. Then the index:

```python
# Create unified FAISS vector store with CLIP embeddings
embeddings_array = np.array(all_embeddings)
embeddings_array
```

```python
# Create custom FAISS index since we have precomputed embeddings
vector_store = FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
    embedding=None,  # We're using precomputed embeddings
    metadatas=[doc.metadata for doc in all_docs]
)
vector_store
```

```python
# Initialize GPT-4 Vision model
llm = init_chat_model("openai:gpt-4.1")
llm
```

Retrieval, message building and the pipeline:

```python
def retrieve_multimodal(query, k=5):
    """Unified retrieval using CLIP embeddings for both text and images."""
    # Embed query using CLIP
    query_embedding = embed_text(query)
    
    # Search in unified vector store
    results = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=k
    )
    
    return results
```

```python
def create_multimodal_message(query, retrieved_docs):
    """Create a message with both text and images for GPT-4V."""
    content = []
    
    # Add the query
    content.append({
        "type": "text",
        "text": f"Question: {query}\n\nContext:\n"
    })
    
    # Separate text and image documents
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    
    # Add text context
    if text_docs:
        text_context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])
        content.append({
            "type": "text",
            "text": f"Text excerpts:\n{text_context}\n"
        })
    
    # Add images
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "text",
                "text": f"\n[Image from page {doc.metadata['page']}]:\n"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data_store[image_id]}"
                }
            })
    
    # Add instruction
    content.append({
        "type": "text",
        "text": "\n\nPlease answer the question based on the provided text and images."
    })
    
    return HumanMessage(content=content)
```

```python
def multimodal_pdf_rag_pipeline(query):
    """Main pipeline for multimodal RAG."""
    # Retrieve relevant documents
    context_docs = retrieve_multimodal(query, k=5)
    
    # Create multimodal message
    message = create_multimodal_message(query, context_docs)
    
    # Get response from GPT-4V
    response = llm.invoke([message])
    
    # Print retrieved context info
    print(f"\nRetrieved {len(context_docs)} documents:")
    for doc in context_docs:
        doc_type = doc.metadata.get("type", "unknown")
        page = doc.metadata.get("page", "?")
        if doc_type == "text":
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  - Text from page {page}: {preview}")
        else:
            print(f"  - Image from page {page}")
    print("\n")
    
    return response.content
```

```python
if __name__ == "__main__":
    # Example queries
    queries = [
        "What does the chart on page 1 show about revenue trends?",
        "Summarize the main findings from the document",
        "What visual elements are present in the document?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        answer = multimodal_pdf_rag_pipeline(query)
        print(f"Answer: {answer}")
        print("=" * 70)
```

For the first query the saved output retrieves two documents (the text from page 0 and the image from page 0) and the vision
model answers that revenue rose steadily over three quarters, with Q1 lowest, Q2 higher and Q3 highest, and the largest jump between
Q2 and Q3. The key idea is the message: it is a single `HumanMessage` whose content is a **list of parts**, text parts and
`image_url` parts that carry the picture as a base64 data URL, so the model reads the text excerpts and looks at the chart together.

(Note that the page numbers are zero-based in the metadata, so "page 1" in the question refers to `page: 0`.)

---

## Errors you will meet in this chapter

| Error or symptom | Cause | Fix |
| --- | --- | --- |
| `InvalidUpdateError` on `graph.invoke("Hi")` | The input must be a dict of state keys | `graph.invoke({"messages": "Hi"})` |
| 401 `invalid_api_key` inside a node | The key was added to `.env` after the kernel started | Restart the kernel or re-run `load_dotenv()` |
| `NameError: name 'State' is not defined` (or `Image`) | Kernel restarted and earlier cells not re-run | Re-run the imports and the `State` cell |
| `Node ... already present` | The same builder object had `add_node` called twice | Create a fresh `StateGraph(State)` and re-run |
| Tool message is `null` | The tool function has no `return` | Add `return a*b` and rebuild bindings and the graph |
| Second request in one sentence is ignored | `tools` goes to `END` | Add `builder.add_edge("tools","tool_calling_llm")` |
| The bot forgets your name | No checkpointer, or a different `thread_id` | Compile with `checkpointer=memory` and reuse the `thread_id` |
| Graph pauses and never continues | `interrupt` fired and nothing resumed it | Call `graph.stream(Command(resume=...), config)` with the same `config` |
| Running `python mathserver.py` prints nothing | It is a `stdio` server waiting for a client | Expected; let `client.py` start it |
| `groq.BadRequestError ... tool_use_failed` | The model produced a malformed tool call | Retry, or use a different model |

## What you can now do

- I can explain nodes, edges and state, and why a graph that keeps state is called a `StateGraph`.
- I can set up a project with `uv init`, `uv venv` and `uv add -r requirements.txt` and run a notebook in its kernel.
- I can write a `TypedDict` state with an `add_messages` reducer and explain why a reducer appends instead of replacing.
- I can build, compile, draw and run a one-node chatbot graph with `invoke` and `stream`, and read the last message.
- I can create a Groq model with `ChatGroq` or `init_chat_model("groq:...")`, and I know to restart the kernel after editing `.env`.
- I can give an LLM tools: bind a Tavily search and a custom function with a good doc string, add a `ToolNode`, and route with `tools_condition`.
- I can turn a one-shot tool graph into a ReAct agent by pointing the tools node back at the LLM, and explain act, observe and reason.
- I can add memory with `MemorySaver`, `compile(checkpointer=...)` and a `thread_id`, and explain why it is for testing only.
- I can choose between `stream_mode="updates"`, `stream_mode="values"` and `astream_events`.
- I can pause a graph with `interrupt` inside a tool, and resume it with `Command(resume=...)` on the same thread.
- I can build an MCP server with `FastMCP`, run it over `stdio` or streamable HTTP, and explain the difference.
- I can connect a `MultiServerMCPClient` to two servers and use their tools in a `create_react_agent` agent.

:::note A newer way to build the agent
The video builds agents with `create_react_agent` from `langgraph.prebuilt`. LangGraph and LangChain version 1
point to `create_agent` from `langchain.agents` as the replacement, and the older function is on its way out. The graph ideas in this chapter
(state, nodes, edges, conditional edges, checkpointers, interrupts) are unchanged; the next course part uses the newer
entry point.
:::
