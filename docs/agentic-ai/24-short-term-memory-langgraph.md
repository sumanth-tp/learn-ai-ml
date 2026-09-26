---
id: agentic-ai-short-term-memory-langgraph
title: "How To Implement Short Term Memory Using LangGraph"
sidebar_label: "24 · Short-term memory in LangGraph"
sidebar_position: 24
slug: /agentic-ai/short-term-memory-langgraph
description: "Short-term memory in LangGraph with a checkpointer and thread IDs, persisting it in PostgreSQL via Docker, and fighting the context overflow problem with trimming, deletion and summarisation."
tags: [langgraph, short-term-memory, checkpointer, postgres, trimming, summarisation]
---

> **Video 24 of 28** · [Watch on YouTube](https://www.youtube.com/watch?v=FSBkTI1QuvY) · Translated from the
> Hindi transcript. Notes follow the video section by section, in its order.

The last video explained memory at a conceptual level; this one implements short-term memory in LangGraph, makes it persistent, and then handles the context window problem with trimming, deletion and summarisation.

## The plan and the prerequisite

The previous video discussed memory around LLMs in detail: short-term memory, long-term memory, and the challenges around them. This video covers, step by step:

1. How to implement short-term memory in LangGraph.
2. The concept of **persistence**.
3. The **context window** problem from the last video, with the techniques of **trimming**, **deletion** and **summarisation**.

Watch the previous video first; without it this one will not make as much sense.

## Recap: short-term memory is a conversation buffer

LLMs have no intrinsic memory, so they cannot remember your messages. Every interaction is **stateless**: each call to `llm.invoke` is treated as a fresh conversation, which makes memory tricky. The technique devised to keep a conversation going is a **conversation buffer**: whenever you send a message to the LLM, send not only the user's current message but the entire conversation before it, concatenated. Then at any moment the `llm.invoke` call carries the context of the whole conversation. That is what was called **short-term memory**, and today it is done in the LangGraph context. The code will look familiar, because short-term memory has already been implemented earlier in this playlist.

## Checkpointer and thread ID

What inside LangGraph helps you implement short-term memory? The **checkpointer**. Its basic idea is that you store your graph's **state** somewhere **at every superstep**.

Two things are used:

- **Checkpointer**: stores the state.
- **Thread** (thread ID): every conversation, or thread, gets a thread ID, and that thread's conversation is stored against it.

At first the state is stored in **RAM**, which is obviously wiped out when the program closes. Solving that with a database is the next section, persistence.

## Demo: the graph without short-term memory

The code imports the libraries and calls `load_dotenv` (the keys are in a `.env` file). The model is `ChatOpenAI` with whatever its default model is. The state is `MessagesState`, which comes built in, so no state has to be coded. There is one node, `call_model`, and the graph is simply START → chat → END.

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

load_dotenv()

model = ChatOpenAI()


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])  # (implied, not shown in narration)
    return {"messages": [response]}  # (implied, not shown in narration)


builder = StateGraph(MessagesState)
builder.add_node("chat", call_model)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()
graph
```

```mermaid
flowchart LR
    S(["START"]) --> C["chat"] --> E(["END"])
```

No checkpointer or thread ID yet, so no short-term memory. Send "Hi, my name is Nitish" and you get the human message and an AI reply. Then send "What is my name?" to the same graph:

```python
graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Nitish"}]})
graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]})
```

```text
I'm sorry, I cannot know your name as I am a computer program.
```

That is what happens without a conversation buffer.

## Demo: adding `InMemorySaver` and a thread ID

Build the same graph again. Everything is the same except the main difference: use `InMemorySaver` from `langgraph.checkpoint.memory`, which saves your checkpoints in RAM. Create a checkpointer and pass it when compiling. Earlier, `builder.compile()` had no checkpointer; now it has one.

Then create a `config` variable carrying the thread ID. It could be generated dynamically; here it is `"thread-1"`. Every time you invoke the graph you send not only the message but also the thread ID, telling it which thread the message belongs to.

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "thread-1"}}

graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Nitish"}]}, config=config)
graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config=config)
```

"Hi, my name is Nitish" gets "Nice to meet you", and now "What is my name?" gets "Your name is Nitish". You can also expose the state to see what is stored in it:

```python
graph.get_state(config)
```

It holds four messages: "Hi, my name is Nitish", "Hi, nice to meet you", "What is my name?", "Your name is Nitish".

Now change the thread. Copy the config, make it `thread-2` as `config2`, and ask "What is my name?" with `config2`:

```python
config2 = {"configurable": {"thread_id": "thread-2"}}

graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config=config2)
```

```text
I'm sorry, I do not have the ability to know your name.
```

Switching to `config2` started a new conversation with the LLM, and in it you asked directly for your name without the earlier message, so the LLM does not know. The state for `config2` shows only those messages. This checkpointer-and-thread code has been used throughout the playlist; it is a very basic implementation of short-term memory in LangGraph.

## The problem: `InMemorySaver` lives in RAM

The biggest problem with this flow is `InMemorySaver`: it stores your state in **RAM**, so as soon as your program leaves RAM, the storage goes with it. Right now `get_state` shows the messages in thread-2. But restart the code, reload the second graph and both configs, do **not** run any of the invoke calls, and go straight to the state: nothing prints. Thread-2 has nothing stored, even though those messages were exchanged in the previous session. Config 1 shows nothing either. Everything was lost on restart because **RAM is volatile**.

That is why, for short-term memory in production systems you deploy for users, you save the state in a **production-grade database** instead of `InMemorySaver`. The LangGraph documentation recommends **Postgres**.

## Persistence with PostgreSQL through Docker

This section saves the state permanently with PostgreSQL, which solves the persistence problem. There are two or three possible setups:

1. Install Postgres on your machine and have LangGraph talk to it.
2. Install Postgres through **Docker** and have LangGraph talk to that database.

The second is used here, because the first tends to bring a lot of installation issues. Perform these steps as they are:

1. **Install Docker** from its download URL. (Docker Desktop is already installed and running on this machine.)
2. **Start Docker Desktop** after installing.
3. **Check the installation**: open a new terminal where your code is and run the version command. If a version prints, Docker is installed and running.

   ```bash
   docker --version
   ```

4. **Create a `docker-compose.yml` file.** It loads a Postgres configuration: the name of the image Docker will download, environment variables for the database's username, password and database name, and a port mapping. Keep it as it is; the file is provided to copy-paste.

   ```yaml
   services:
     postgres:
       image: postgres:16  # (implied, image name not read out)
       environment:
         POSTGRES_USER: postgres  # (implied, value not read out)
         POSTGRES_PASSWORD: postgres  # (implied, value not read out)
         POSTGRES_DB: postgres  # (implied, value not read out)
       ports:
         - "5432:5432"  # (implied, value not read out)
   ```

5. **Run the file**:

   ```bash
   docker compose up -d
   ```

6. **Check the Postgres server is running**; you should see the image running:

   ```bash
   docker ps
   ```

Now install some Python dependencies, provided in the code as one command. You probably already have LangGraph and `langchain-openai`; you need `langgraph-checkpoint-postgres` and one more package. If installing from the Jupyter notebook gives trouble, install directly from the terminal.

```bash
pip install langgraph langgraph-checkpoint-postgres langchain-openai
pip install "psycopg[binary]"  # (implied, not shown in narration)
```

### The code

Load the dependencies, call `load_dotenv`, create the LLM, and build the same graph structure with its single node; nothing is different up to here. Then provide the database URL, exactly as written in the setup.

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres"  # (implied, value not read out)
```

From here the main thing changes: all the code runs **inside a context manager**. `PostgresSaver` is another type of saver, just like `InMemorySaver`. Call `checkpointer.setup()`, then compile the graph with `checkpointer=checkpointer`. Define the thread in a `config` variable, invoke "Hi, my name is Nitish", ask "What is my name?" in the same thread, and print the response.

```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "thread-1"}}

    graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Nitish"}]}, config=config)
    result = graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config=config)

    print(result["messages"][-1].content)
```

Running it shows the LLM's most recent message. The same code is then used for a separate thread, `thread-2`, asking only "What is my name?":

```text
I can't determine your name unless you tell me.
```

Different threads run different conversation buffers, and messages are being stored per thread.

### Proving the persistence

Restart the shell. With exactly the same setup (import `PostgresSaver`, the DB URL, and say thread-2), open the context manager again, but this time **invoke nothing**: simply fetch the graph's state.

```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "thread-2"}}
    print(graph.get_state(config))
```

The first attempt fails with "builder is not defined", so run the earlier cells again (the imports, LLM, nodes and graph structure, but not the cells that send messages) and then run this. The last message appears. Switch it to thread-1 and you see thread-1's last message too: "Your name is Nitish. What else would you like to know?"

The code was restarted, removed from RAM and brought back, and the past conversation was still stored and retrievable. That is how you implement production-grade code. There is nothing special in it: use this saver instead of `InMemorySaver`, inside a context manager. Play around with the code to get comfortable.

## The context overflow problem

Every LLM has a context window: the maximum number of tokens it can process at one time while responding. If the input tokens exceed it, the response gets messed up; the LLM might hallucinate and give improper answers, so you cannot trust its output. That is why your input tokens should always be **way less** than the context window.

Short-term memory carries a huge risk of exceeding it, because of how it works: every new human message is sent with the entire past conversation history attached. Imagine a very long conversation of 500 or 1000 messages, with long outputs from the LLM, all concatenated into `llm.invoke`. You may cross the context window and get a wrong response. This is the **context overflow problem**, and the following techniques fight it.

## Technique 1: trimming

The idea is very simple. Set a **max token limit**, say 500 tokens. Every time, before sending the messages into `llm.invoke`, check whether their total token count is under 500. If it is, send them all. If the conversation has grown and the total exceeds 500, **trim**: keep the last n messages and remove the ones before. Which n? Precisely as many messages, counted from the end, as fit within 500; as soon as the total would go above 500, ignore everything older and do not send it to the LLM.

The assumption behind it: in a very long conversation, the context currently being discussed is in the last 50 to 100 messages anyway, and older context is not that useful. So you trim the old messages away.

### Trimming in code

LangChain provides a function named `trim_messages` for this. Load the environment, create the model, and define the important line, `MAX_TOKENS = 150`: whenever the LLM is invoked, the input messages must be 150 tokens or fewer; if more, trimming happens.

The main node is the same as before: take the messages from the state, invoke the model, return the response. The one big difference is that the messages go to `trim_messages` first, with the conversation history so far, a maximum of 150 tokens, and the instruction to keep the last n messages that fit and trim the rest. Tokens are counted with LangChain's `count_tokens_approximately`.

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatOpenAI()

MAX_TOKENS = 150


def call_model(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        strategy="last",  # (implied, not shown in narration)
        token_counter=count_tokens_approximately,
        max_tokens=MAX_TOKENS,
    )

    print("Current Token Count ->", count_tokens_approximately(messages=messages))  # (implied, not shown in narration)
    for message in messages:  # (implied, not shown in narration)
        print(message.content)  # (implied, not shown in narration)

    response = model.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

One important point: you are **not deleting** the messages. They stay in the state, and since `InMemorySaver` is used, in memory too. When the function is called you keep the last messages and leave out the earlier ones. You are just **not showing them to the LLM**. The graph is exactly the same flow; the only difference is the trimming inside `call_model` before the LLM is called.

### Running the trimming demo

Define a thread, `chat-1`, and send the first message, "Hi, my name is Nitish":

```python
config = {"configurable": {"thread_id": "chat-1"}}

result = graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Nitish"}]}, config=config)
```

The current token count is shown as **10**. It may not look like 10, but `count_tokens_approximately` is LangChain's own heuristic and counts other things too; the `role: user` you sent is also counted as tokens. So do not go by the exact numbers. At 10 there is no trimming, the whole message goes to the LLM, and the LLM replies.

The turns that follow:

| Message sent | Messages in state | Current token count | Trimming? |
| --- | --- | --- | --- |
| "Hi, my name is Nitish" | 1 | 10 | No |
| "I am learning LangGraph" | 3 | 40 | No |
| "Can you explain short-term memory?" | 5 | 108 | No |
| "What is my name?" | 7 | 8 | Yes |

At 108 the count is still below 150, so the whole history goes to the LLM, which replies with an explanation of short-term memory. That long reply means the next turn will cross 150. When "What is my name?" is sent, all the messages together exceed 150, so only the last n that fit within 150 are kept. The count shows **8**, and only "What is my name?" remains after trimming: the message just before it, the long explanation, probably crosses 150 on its own, so it too had to go. With only that message sent, the LLM no longer knows the name:

```text
I'm sorry, I don't know your name.
```

The example was set up to show trimming really happening. All the heavy lifting is done by `trim_messages`; your job is to set the number for your application, which you will work out with some experimentation.

## Trimming's flaw, and summarisation

Trimming has an inherent flaw. It sends the most recent n messages and **completely ignores** all older ones; the LLM does not even know they exist. That rests on the assumption that only the latest n messages are useful to keep the conversation going, and in most real-world scenarios the assumption fails. Old messages can be useful too, and ignoring them creates a kind of breaking point, or the LLM cannot carry the conversation properly.

To handle this flaw, a second and more popular technique is used: **summarisation**. It is largely like trimming, with one very important difference. You still send only the most recent n messages at any moment, but you do not ignore the older ones. You send them to **another LLM** to generate a **summary**, then send the summary plus the most recent n messages to the LLM. The LLM always has the recent messages and a summary of the past, so the context is not fully lost.

A simple example: 500 messages have been exchanged between the human and the AI, and you have decided to always send the last 100 to the LLM. A summary of the previous 400 is generated, and you send that summary plus the most recent 100. As you keep chatting, say 100 more messages arrive. You keep the most recent 100; you already have the summary of the first 400; you generate a summary of the 100 after those and **merge the two summaries**. Now you have a summary of 500 messages plus the most recent 100, and that is what you send.

This is why summarisation performs better than trimming: the context is preserved. However long the conversation, you always carry at least its summary and send it to the LLM.

One more interesting detail: after generating the summary you **delete** the old messages from the state; you do not just ignore them. That is necessary, because if you kept them in the state you would be holding both the old messages and their summary, which can cause confusion. So you keep only the summary. **Deletion plus summarisation operate together**, and deletion is a kind of prerequisite step, so it comes first.

## Technique 2: deletion

The most important import is the last one: `RemoveMessage` from LangChain's messages module, used for **permanent deletion** from the state. Then `load_dotenv` and the LLM.

The workflow has two nodes: the `chat` node for normal chatting and a `delete_old_messages` node. That node takes all the messages currently in the state and runs a simple check: if the list is longer than 10, remove the **first six**, keeping only the last four. The actual deletion is done by `RemoveMessage`, given the IDs of the messages to remove.

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import RemoveMessage

load_dotenv()

model = ChatOpenAI()


def chat_node(state: MessagesState):
    response = model.invoke(state["messages"])  # (implied, not shown in narration)
    return {"messages": [response]}  # (implied, not shown in narration)


def delete_old_messages(state: MessagesState):
    msgs = state["messages"]

    if len(msgs) > 10:
        to_remove = msgs[:6]
        return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}

    return {}  # (implied, not shown in narration)


builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_node("cleanup", delete_old_messages)

builder.add_edge(START, "chat")
builder.add_edge("chat", "cleanup")
builder.add_edge("cleanup", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

```mermaid
flowchart LR
    S(["START"]) --> C["chat"] --> D["cleanup<br/>(deletes the first 6<br/>if more than 10)"] --> E(["END"])
```

The cleanup node only does its work when the conversation has more than 10 messages.

To test it, define a thread and invoke the graph **seven times** with different messages. You would expect 14 messages in the state afterwards, seven from the user and seven from the AI. It takes a while, since the LLM is called seven times.

```python
config = {"configurable": {"thread_id": "t1"}}  # (implied, thread name not read out)

graph.invoke({"messages": [{"role": "user", "content": "Hi, I am Nitish"}]}, config=config)
graph.invoke({"messages": [{"role": "user", "content": "Tell me about LangGraph"}]}, config=config)
graph.invoke({"messages": [{"role": "user", "content": "Now explain checkpointers"}]}, config=config)
# ... four more invokes with other messages (not read out)

len(graph.get_state(config).values["messages"])
```

Will the state hold 14? No. Since 14 is more than 10, the **oldest** six messages are deleted (not the last six), and the current state has **eight**. Gone are "Hi, I am Nitish", "Tell me about LangGraph" and "Now explain checkpointers" with the LLM's answers to each; what remains are the later user messages and their answers. Deleting messages from the state with `RemoveMessage` is that simple.

:::note

The cleanup node runs after every turn, not once at the end. The deletion actually fires after the sixth turn, when the state reaches 12 messages (6 are removed, leaving 6); the seventh turn then brings it to 8. The end result, eight messages with the first three exchanges gone, is the same as described.

:::

## Technique 3: summarisation

The workflow on screen is simple. It works normally while the state holds **six or fewer** messages. As soon as it holds more than six (say seven or eight), summarisation runs: keep only the **two most recent** messages, generate a summary of all the messages before them (say the four remaining), and attach it ahead of those two. From START you go to chat; if the state has fewer than six messages, go to END; if more than six, summarise and then go to END, so the next chat sees fewer than six messages.

```mermaid
flowchart LR
    S(["START"]) --> C["chat"]
    C -->|"6 or fewer messages"| E(["END"])
    C -->|"more than 6 messages"| SU["summarize<br/>(summary + delete all<br/>but the last 2)"]
    SU --> E
```

### The state

The imports include `RemoveMessage`, because deletion is done with it. Then `load_dotenv` and the LLM. This time you **define your own state**. The last three or four examples used the built-in `MessagesState`, but that is not enough here: you additionally need a **`summary`** key to store the summary.

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import RemoveMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatOpenAI()


class ChatState(MessagesState):
    summary: str
```

### The chat node

The workflow needs two nodes, a chat node and a summary node. The chat node has two cases:

1. **No summary exists yet.** The conversation has just started and the state has six or fewer messages. Send the model only the messages.
2. **A summary has been generated** once or more (the state had exceeded six messages). Send the model the messages **plus the summary**, so it has the context of both.

In code: start an empty `messages` list; if a summary exists in the state, append it as a **system message** ("Conversation summary"); then append all the state's messages; send the final list to the model; put the response back into the state's `messages` key.

```python
def chat_node(state: ChatState):
    messages = []

    if state["summary"]:
        messages.append(SystemMessage(content=f"Conversation summary:\n{state['summary']}"))

    messages.extend(state["messages"])

    response = model.invoke(messages)
    return {"messages": [response]}
```

### The summarise node

Two things happen here: the **summary** and the **deletion**.

The summary operation has two possibilities:

1. **First-time summary.** A new conversation has crossed six messages for the first time, so for the first time a value is stored in the state's `summary` key: a kind of creation.
2. **Extending an existing summary.** A summary already exists from one or more earlier rounds, and you are editing it, in the sense of merging the summary of the new messages into the past summary.

In code, `existing_summary` takes the state's summary value, which covers both cases. If it has a value, a summary was performed before, so the prompt gives the existing summary and asks to extend it using the new conversation above (the new messages to summarise are placed above the prompt). If it is empty, the prompt simply says to summarise the conversation above; there is no old summary to merge into. `messages_for_summary` holds the messages followed by whichever prompt applies, and it goes to a model, which returns the summary.

Then comes deletion. `messages_to_delete` is everything except the **two most recent** messages. The node returns two things: the new `summary`, and `messages` with every message other than the last two removed via `RemoveMessage`.

```python
def summarize_conversation(state: ChatState):
    existing_summary = state["summary"]

    if existing_summary:
        prompt = (
            f"Existing summary:\n{existing_summary}\n\n"
            "Extend the summary using the new conversation above."
        )
    else:
        prompt = "Summarize the conversation above."

    messages_for_summary = state["messages"] + [HumanMessage(content=prompt)]

    response = model.invoke(messages_for_summary)

    messages_to_delete = state["messages"][:-2]

    return {
        "summary": response.content,
        "messages": [RemoveMessage(id=m.id) for m in messages_to_delete],
    }
```

### The condition and the graph

The summarise node should not always trigger, only when the state holds more than six messages. That is the conditional function `should_summarize`.

Then build the graph: add both nodes, an edge from START to chat, a conditional edge from chat that goes to `summarize` on True and to END on False, and finally connect `summarize` to END. Add the checkpointer and compile; the graph is exactly the one shown.

```python
def should_summarize(state: ChatState):
    return len(state["messages"]) > 6


builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_node("summarize", summarize_conversation)

builder.add_edge(START, "chat")
builder.add_conditional_edges("chat", should_summarize, {True: "summarize", False: END})
builder.add_edge("summarize", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### Testing it

Create the `config` variable and a utility function that shows whatever is currently in the state. It displays only 80 characters of each message so things are easy to see, which is why the AI messages look so short.

```python
config = {"configurable": {"thread_id": "t1"}}  # (implied, thread name not read out)


def show_state(config):  # (implied, name not read out)
    state = graph.get_state(config).values
    print("Summary:", state.get("summary"))
    print("Number of messages:", len(state["messages"]))
    for m in state["messages"]:
        print(m.type, ":", m.content[:80])
```

Round by round:

1. Invoke the graph for the first time with "Quantum physics" and, obviously, an empty summary. The output shows the message and the AI's reply. The summary is empty because there are fewer than six messages; the state holds **two**.

   ```python
   graph.invoke({"messages": [HumanMessage(content="Quantum physics")], "summary": ""}, config=config)
   show_state(config)
   ```

2. "How is Albert Einstein related to quantum physics?" The summary is still empty with **four** messages: the quantum physics question and answer, and the Einstein question and answer.
3. "What are some of Einstein's famous works?" Still empty, with **six** messages; the summary triggers only above six.
4. "Explain special theory of relativity". Now the real thing happens. The state holds only **two** messages: the total had reached eight, which is greater than six, so the two most recent ("Explain special theory of relativity" and the AI's answer) were kept and the rest were summarised into the `summary` key:

   ```text
   The conversation above discusses Albert Einstein's contributions to physics...
   ```

That is how summarisation works in principle. It is a very basic example, but the chatbots you use rely on this same principle in one way or another.

## Wrapping up

This video covered short-term memory in LangGraph: implementing it, persisting it with a Postgres database, and solving the context overflow problem with trimming and summarisation. Watch it fully and try the code yourself, and you should become very comfortable handling short-term memory in LangGraph.
