---
id: agentic-ai-long-term-memory-langgraph
title: "Long Term Memory in LangGraph"
sidebar_label: "25 · Long-term memory in LangGraph"
sidebar_position: 25
slug: /agentic-ai/long-term-memory-langgraph
description: "Long-term memory in LangGraph from scratch: memory stores and namespaces, put/get/search and semantic search, a chatbot that reads memories, a node that writes them with deduplication, the merged chatbot, and a persistent PostgresStore."
tags: [langgraph, long-term-memory, memory-store, semantic-search, postgres]
---

> **Video 25 of 28** · [Watch on YouTube](https://www.youtube.com/watch?v=KrXBcokM3Tc) · Translated from the
> Hindi transcript. Notes follow the video section by section, in its order.

After the foundations of memory and then short-term memory, this video implements long-term memory in LangGraph, starting from scratch, adding complexity step by step, and ending with how production-grade LLM systems do it.

## Where this fits

For the last two or three videos the playlist has been about memory. The video before last was a deep dive into the foundations: what memory is at a theoretical level, how you build it around an LLM, the two types (short-term and long-term), and the problems that come with each. The last video implemented short-term memory in LangGraph inside a chatbot.

This one is dedicated to **long-term memory**: how to implement it inside LangGraph and make your chatbot more powerful with it. It is especially important if you plan to become an AI engineer who builds production-grade chatbot systems, so watch it end to end.

## Recap: what long-term memory is

Take a chatbot like ChatGPT. You hold conversations in **multiple threads**, a separate thread for each conversation. In one window you talk about a technical topic such as LangChain, in another you discuss travel plans, and in a third you get into some very deep things.

If you are the company that built the chatbot, you want to improve the product over time so the user feels more and more comfortable with it. For that you need to understand the user better, so you can **personalise** the LLM's responses. The problem is that the user does not tell you everything about himself in one chat. He reveals a little in each:

- In the technical conversation, that he is a programmer and prefers Python.
- In the travel conversation, that he is planning to move or travel somewhere.
- In the deep-thought conversation, that he has a philosophical inclination or believes a little in spirituality.

These important pieces of information do not live in one thread; they are spread across many. So builders of LLM-based systems create a **memory store**, which for now you can think of as a database. Every time the user asks a question in a chat, you answer it, and you also ask your LLM whether the question hides any information about the user that could be useful later. If it does, you extract that piece and add it to the memory store.

The key property of this store is that it is **persistent**: close the chat window and the information stays there forever. Say in another chat you mention "I have to go to Mumbai in 2 months"; that is important, if temporary, information, so it too goes into the database. Pulling important pieces of information out of every conversation into persistent storage is what builds **long-term memory**.

```mermaid
flowchart LR
    T1["Thread 1<br/>LangChain (technical)<br/>'I'm a programmer, I prefer Python'"] --> M[("Memory store<br/>(persistent)")]
    T2["Thread 2<br/>Travel plans<br/>'Going to Mumbai in 2 months'"] --> M
    T3["Thread 3<br/>Deep thoughts<br/>'Philosophical inclination'"] --> M
    M -->|"checked before replying"| L["LLM reply,<br/>personalised"]
```

The store makes personalised responses possible. If the user once said his name is Nitish, that is stored. When framing a reply, the LLM always first checks whether the store holds something that would help personalise the answer. Ask "Write a code for Fibonacci series" after having once told the chatbot "I prefer programming in Python", and the LLM looks up your language preference in the store, finds Python, and writes the program in Python. That is how personalisation happens in LLM-based systems, and the component that makes it possible is long-term memory.

## A technical note: `BaseStore` and its implementations

In LangGraph the memory store concept is implemented with a class, in fact an **abstract class** called **`BaseStore`**. You will see it in several places in the code, which is why it is worth understanding first.

`BaseStore` lays out what a memory store can do:

- create new memories,
- search existing memories,
- edit existing memories,
- delete existing memories.

Other classes inherit from it:

- **`InMemoryStore`** stores memories in **RAM**. Because of that it is not very useful in production, and you would not use it in production-grade systems. It exists so you can quickly prototype and check that your whole functionality works. The code starts with it.
- **`PostgresStore`** stores memories in a **Postgres** database and is what you use in production-grade systems. It is shown at the end of the video.
- **`RedisStore`** is another production-grade implementation that stores memories in **Redis**.

```mermaid
flowchart TB
    B["<b>BaseStore</b><br/>(abstract class)<br/>create · search · edit · delete"]
    B --> I["InMemoryStore<br/>RAM, for prototyping"]
    B --> P["PostgresStore<br/>Postgres, production"]
    B --> R["RedisStore<br/>Redis, production"]
```

That is the conceptual mind map: long-term memory in LangGraph is implemented as a store, its contract is in the `BaseStore` abstract class, and several implementations inherit from it. The two that matter most here are `InMemoryStore` (RAM) and `PostgresStore` (Postgres database).

The coding comes in two parts:

1. **Memory stores on their own.** Forget for a while that this is LangGraph and focus only on stores: how to make one, create new memories in it, and search existing ones.
2. **Integrating the store with LangGraph** to build a real chatbot that both creates and reads memories.

## Part 1: working with a memory store

### `InMemoryStore` and the `BaseStore` methods

Import `InMemoryStore` from `langgraph.store.memory`. It is one implementation of `BaseStore`; it keeps things in RAM, so it is not persistent, but it is very useful for learning quickly.

Click into its source and the structure becomes clear. `InMemoryStore` inherits the `BaseStore` class, and `BaseStore` is abstract because it inherits `ABC`. Inside it you see the abstract methods, for example:

- `get`: fetch one particular memory,
- `search`: search more than one memory,
- `put`: create new memories.

### Making the store and understanding namespaces

First make an object of the class, called `store`. It now has every capability of the class: creating memories, searching them, fetching them.

To create a memory you first need a **namespace**. If you think of the memory store as Google Drive, a namespace is nothing but a **folder** inside it. Just as you organise data in folders, you organise memories in a store with namespaces. Examples:

- `users`, `u1`: a top-level folder `users` with a folder `u1` inside it, holding all of user one's memories.
- `users`, `u2`: a second folder for user two, holding all of u2's memories.
- `users`, `u1`, `profile`: one level deeper, a `profile` folder for u1's profile information, such as name, profession, age and gender.
- `users`, `u2`, `profile`: the same for u2.
- `users`, `u1`, `preferences`: memories such as "the user likes dark mode" or "the user prefers the Python programming language".

```mermaid
flowchart TB
    U["users"] --> U1["u1"]
    U --> U2["u2"]
    U1 --> P1["profile<br/>name, profession, age, gender"]
    U1 --> PR1["preferences<br/>dark mode, Python"]
    U2 --> P2["profile"]
```

Namespaces are a way to organise memory inside a memory store, and every memory you create lives in some namespace. You create a namespace as a **tuple**: string, comma, string, comma, string, as long as you like. Here it is `("users", "u1")`.

### Creating memories with `put`

`put` inserts a new memory into a particular namespace. It needs three inputs:

1. the **namespace**, the folder where the memory goes,
2. a **key** for the memory, which must be **unique**,
3. a **value**, what the memory actually is.

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

namespace = ("users", "u1")

store.put(namespace, "1", {"data": "User likes pizza"})
store.put(namespace, "2", {"data": "User prefers dark mode"})
```

The first memory is a dictionary whose `data` is "User likes pizza"; the second, with key `"2"`, is that the user prefers dark mode. Run it and the memories are added. Then a second namespace, a second folder for user two:

```python
namespace_2 = ("users", "u2")

store.put(namespace_2, "1", {"data": "User likes pasta"})
store.put(namespace_2, "2", {"data": "User prefers grid-style navigation"})
```

### Fetching one memory with `get`

`get` retrieves data and needs only two things, the namespace and the key:

```python
store.get(namespace, "1")
```

This fetches u1's first memory. Switch to `namespace_2` and you get the second user's first memory, "User likes pasta"; change the key to `"2"` and you get "User prefers grid-style navigation". So `get` fetches **one particular memory** by namespace and key.

### Fetching all memories with `search`

To fetch every memory in a namespace at once, use `search` with just the namespace. The result is a list, so loop over it and print each item:

```python
items = store.search(namespace)

for item in items:
    print(item)
```

That prints all of user one's memories; pass `namespace_2` to see all of user two's.

LangGraph has made this very simple: one `InMemoryStore` object, with `put`, `get` and `search` to create memories, retrieve one item, and retrieve them all. The concept worth learning is the namespace: you are creating folders inside the memory store.

### Where `get` and `search` fall short

So far you have seen how to **create** memory (`put`) and how to **retrieve** it (`get` and `search`). Both retrieval methods are useful, but there is a situation neither handles:

- `get` works when you have the exact key, meaning you know exactly which memory you want.
- `search` (as used so far) brings all the memories in one go: "I don't care, bring me everything".

What if you want neither one known memory nor all of them, but the **specific** memories relevant right now? Say you are talking to your chatbot about a travel plan you also discussed in the past. By now the store might hold **100 memories** about you: your name, your profession, your programming language, that you like dark mode, and much more, including that you plan to visit Mumbai in 2 months. For the current conversation only the memories around the Mumbai plan help, and perhaps only two or three of the 100 are about it.

If you use `search` to fetch all 100 and put them in your context, the LLM will only get confused and will not give good responses. The best case is to fetch exactly the two or three memories that talk about Mumbai; once they are part of the current conversation, the LLM automatically gives better personalised responses.

What you need is **semantic search**: match the meaning of the current conversation against the stored memories, and fetch only those whose meaning matches.

### Semantic search over memories

Semantic search also uses the `search` function. There are two changes.

**Change 1: give the store an embedding model.** Fetch an OpenAI embedding model, `text-embedding-3-small`, into a variable `embedding_model`. When creating the `InMemoryStore`, pass a dictionary under the key `index` saying this store can do semantic search because it has an embedding model, and how many dimensions of embeddings to generate. The store is now a slightly specialised one with embedding and semantic search capabilities.

```python
from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

store = InMemoryStore(
    index={
        "embed": embedding_model,
        "dims": 1536,  # (implied, not shown in narration)
    }
)
```

Create the namespace `("users", "u1")` again and add 10 different memories with `put`, exactly as before. The ones read out during the demo include:

```python
namespace = ("users", "u1")

store.put(namespace, "5", {"data": "User is learning machine learning"})  # memory number five
store.put(namespace, "…", {"data": "User prefers step-by-step reasoning"})
store.put(namespace, "…", {"data": "User likes examples in Python"})
store.put(namespace, "…", {"data": "User prefers dark mode in applications"})
store.put(namespace, "…", {"data": "User prefers bullet points over paragraphs"})
store.put(namespace, "…", {"data": "User prefers concise answers over long explanations"})
# ... 10 memories in total; the remaining ones are on screen but not read out
```

**Change 2: pass a query and a limit to `search`.** The query's embedding is generated, the memories' embeddings already exist, the vectors are compared, and the closest memories are returned. `limit` is how many to return.

```python
store.search(namespace, query="What is the user currently learning?", limit=1)
```

```text
User is learning machine learning
```

With `limit=3` you would get the top three matches, such as "User prefers step-by-step reasoning" and "User likes examples in Python", which it found semantically closest; but 1 is the good choice here. Another example: the memories include three or four preferences, so set the limit to 3:

```python
store.search(namespace, query="What are the user's preferences?", limit=3)
```

```text
User prefers dark mode in applications
User prefers bullet points over paragraphs
User prefers concise answers over long explanations
```

So the difference is in two places: you pass an embedding model when creating the store, and `store.search` now takes a query and a limit on top of the namespace. This is how you conduct semantic search in long-term memories. It is a very important concept, and in any reasonably proper chatbot with memory you will definitely use it, so make sure you understand it well.

## Part 2: connecting the store to a LangGraph chatbot

### The plan: a chatbot that reads existing memories

The workflow is the familiar one from this playlist, START → chat → END. What is different is that it is connected to a memory store that is **already prefilled** with some memories. When the user asks a question, the LLM first goes to the store, checks which memories are available, and uses them to personalise its response.

At this point the chatbot can **only use existing memories; it cannot create new ones**. Creating memories comes a little later.

```mermaid
flowchart LR
    S(["START"]) --> C["chat"] --> E(["END"])
    M[("Memory store<br/>(prefilled)")] -.->|"read memories"| C
```

### The store, the user and the prefilled memories

After the imports, make an `InMemoryStore`. No embedding model this time, because semantic search is not used here. Define a user ID; it is normally dynamic and comes from the front end, but assume the user is `u1`. Build the namespace from it, with a further sub-folder called `details`, so the hierarchy is `user` → `u1` → `details`, and all memories live in `details`.

The next four or five lines manually create memories for u1:

```python
import uuid  # (implied, not shown in narration)

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

user_id = "u1"
namespace = ("user", user_id, "details")

# memory wording as described in the narration; keys (implied, not shown in narration)
store.put(namespace, str(uuid.uuid4()), {"data": "User's name is Nitish"})
store.put(namespace, str(uuid.uuid4()), {"data": "User teaches AI on YouTube"})
store.put(namespace, str(uuid.uuid4()), {"data": "User prefers concise answers"})
store.put(namespace, str(uuid.uuid4()), {"data": "User likes examples in Python"})
store.put(namespace, str(uuid.uuid4()), {"data": "User is building MCP servers, Python-based projects"})
```

### The system prompt template

Now the LangGraph work begins, with a system prompt template. Reading it makes the flow clear:

- "You are a helpful assistant with memory capabilities. If user-specific memory is available, use it to personalise your responses based on what you know about the user."
- "Your goal is to provide relevant, friendly and tailored assistance that reflects the user's preferences, context and past interactions." In a nutshell: if the memories say something about the user, use it to frame a personalised answer.
- "If the user's name or relevant personal context is available, always personalise your response by":
  1. **addressing the user by name**, whenever you know it;
  2. **referencing known projects, tools or preferences**, so if you are explaining what GenAI is and you know what project the user is working on, connect the two;
  3. **adjusting the tone** to feel friendly, natural and directly aimed at the user.
- "Avoid generic phrasing when personalisation is possible. For example, instead of 'In TypeScript apps…' say 'Since your project is built with TypeScript…'."
- "In the end, suggest three relevant further questions based on the current response and the user profile."

The prompt pushes hard for memory-based personalisation. It contains a fill-in-the-blank, `user_details_content`, where the memories will be filled in.

```python
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalise your responses based on what you know about the user.

Your goal is to provide relevant, friendly and tailored assistance that reflects the user's preferences, context and past interactions.

If the user's name or relevant personal context is available, always personalise your response by:
- Addressing the user by name
- Referencing known projects, tools or preferences
- Adjusting the tone to feel friendly, natural and directly aimed at the user

Avoid generic phrasing when personalisation is possible.
For example, instead of "In TypeScript apps..." say "Since your project is built with TypeScript..."

In the end, suggest three relevant further questions based on the current response and the user profile.

{user_details_content}
"""

llm = ChatOpenAI()
```

### The chat node: `config` and `store` arrive as inputs

This node does not simply take a question and answer it; it does the memory work too. It is called `chat_node`. Like every node so far it receives the **state**, but it also receives two more things:

- **`config`**, of type `RunnableConfig`. It comes from the config you pass when you **invoke** the graph, the same mechanism used throughout the playlist to pass a thread ID for short-term memory. Here it carries which user you are chatting with. You need the user ID because it gives you the namespace, and you need the namespace because that is where the memories are.
- **`store`**, of type `BaseStore`. This is your memory store, and it arrives because you pass the store when **compiling** the graph. From it you extract the memories.

The steps inside the node:

1. Extract the user ID from `config`.
2. Build the namespace from it, exactly the namespace defined above.
3. `search` the store and bring back **all** the memories into `items`. No semantic search here, to avoid adding complexity the first time you add memory to a chatbot.
4. If `items` has anything, merge all the memories into one text block separated by hyphens (first memory, hyphen, second memory, and so on) and store it in `user_details_content`.
5. Call `format` on the system prompt template to fill `user_details_content` in, so the old memories are now inside the prompt.
6. Convert the result into a `SystemMessage`.
7. Call `llm.invoke`, sending not just the conversation so far but the system message **prepended** to it.
8. Add the response to `messages`.

```python
def chat_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id, "details")

    items = store.search(namespace)

    if items:
        user_details_content = "\n".join(f"- {it.value['data']}" for it in items)
    else:
        user_details_content = ""  # (implied, not shown in narration)

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(user_details_content=user_details_content)
    )

    response = llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}
```

This is how you pick up long-term memory and inject it into short-term memory: it goes into the LLM's context window, and the response is based on it.

The graph is one node, START → chat → END, and the store is passed to `builder.compile`:

```python
builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(store=store)
graph
```

### Demo: a personalised answer

The cells above had not all been run, so run everything again from the top. Define a config with the user ID `u1` and ask only "Explain GenAI in simple terms":

```python
config = {"configurable": {"user_id": "u1"}}

result = graph.invoke(
    {"messages": [{"role": "user", "content": "Explain GenAI in simple terms"}]},
    config,
)
```

The reply starts with "Sure, Nitish". It knows the name because the name went into the system prompt, and it got there through memory. After the answer it lists "Here are a few questions to consider", because the prompt asked for questions at the end of every response:

```text
Would you like to see specific examples of GenAI in Python?
Are you interested in exploring tools or libraries for building GenAI projects?
How would you like to incorporate GenAI concepts into your teaching material?
```

Python comes from the stored preference, and "teaching material" comes from the memory that the user teaches on YouTube. The response was personalised to the user through long-term memory. When you do this yourself on your machine and in your own project, you will have a big smile on your face.

## Creating new memories while chatting

### The plan: a `remember` node

So far only existing memories are used; ideally new memories should be created too. The workflow for that is START → **remember** → END, again connected to a memory store, but this time the store starts **empty** and new memories are added to it.

To keep things simple, this workflow does **only one thing**: create new memories. It does not use existing memories. Essentially **it is not a chatbot**; it is not actually chatting. It takes the user's message, decides whether there is a memory inside it, and if so picks it up and puts it in the store. That's it.

```mermaid
flowchart LR
    S(["START"]) --> R["remember"] --> E(["END"])
    R -.->|"write memories"| M[("Memory store<br/>(starts empty)")]
```

### The extractor LLM and the `MemoryDecision` model

After the imports, create a new store. You need an LLM here too: the user's message goes to an LLM, which decides whether there is something worth remembering in it. If there is, it extracts it and gets it stored; if not, it is skipped. This one is called the **extractor** LLM, because its job is to extract memories.

Then something new: a **Pydantic model** that controls the LLM so it gives controlled responses. When you send it a message, you want two things back:

1. **`should_write`**, True or False: is there anything in this message worth remembering? If True, do the further work; if False, nothing is written to the store.
2. **`memories`**, a list of strings, because one message may contain more than one thing worth remembering.

The model is called `MemoryDecision`. Run `with_structured_output` on the extractor LLM with it, and that is the final LLM you use. This is exactly the structured-output concept from the LangChain playlist.

```python
import uuid
from typing import List

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

store = InMemoryStore()

memory_llm = ChatOpenAI()


class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[str] = Field(default_factory=list)


memory_extractor = memory_llm.with_structured_output(MemoryDecision)
```

### The `remember` node

The node again receives three things: `state`, `config` (for the user ID, which gives the namespace where the memory is added) and `store`; the same setup as the chat node.

1. Extract the user ID and form the namespace.
2. Take the **last message** from the state, because the memory is extracted from the user's latest message.
3. Invoke the memory extractor with two things: a system message, and the user's last message.

The system message reads: "Extract long-term memories from the user's message. Only store stable user-specific info: identity, preferences, ongoing projects. Do not store transient info. Return `should_write=False` if nothing is worth storing. Each memory should be a short atomic sentence."

4. What comes back is stored in `decision`, which is a `MemoryDecision` object, so the LLM is forced to answer in that structured format.
5. `if decision.should_write`: go through `decision.memories` one by one and `store.put` each into the namespace with a generated unique key.
6. Finally return a **hard-coded** message saying the memory has been noted.

```python
MEMORY_PROMPT = """Extract long-term memories from the user's message.
Only store stable user-specific info: identity, preferences, ongoing projects.
Do not store transient info.
Return should_write=False if nothing is worth storing.
Each memory should be a short atomic sentence."""


def remember_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id, "details")

    last_msg = state["messages"][-1].content

    decision: MemoryDecision = memory_extractor.invoke(
        [SystemMessage(content=MEMORY_PROMPT), {"role": "user", "content": last_msg}]
    )

    if decision.should_write:
        for mem in decision.memories:
            store.put(namespace, str(uuid.uuid4()), {"data": mem})

    return {"messages": [{"role": "assistant", "content": "Noted."}]}


builder = StateGraph(MessagesState)
builder.add_node("remember", remember_node)
builder.add_edge(START, "remember")
builder.add_edge("remember", END)

graph = builder.compile(store=store)
```

### Demo: three memories written

Run it from the top. The config (user ID `u1`) goes to the node. Three messages, each with something worth remembering:

```python
config = {"configurable": {"user_id": "u1"}}

graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Nitish"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "I teach AI on YouTube"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "My favourite programming language is Python"}]}, config)
```

Each time the assistant says "Noted", not because the LLM decided to say so but because the reply is hard-coded. Then check the store:

```python
for item in store.search(("user", "u1", "details")):
    print(item.value["data"])
```

All three things were remembered: memories are being extracted and stored.

### The flaw: duplicate memories

Everything here is correct except one major flaw. Run the same three messages again ("Hi, my name is Nitish", "I teach AI on YouTube", "My favourite programming language is Python") and look at the store: **duplicate memories** have been created. There is no **deduplication** code, so repeatedly sending the same kind of message keeps creating redundant memories. This very big flaw has to be solved with some deduplication strategy.

### One deduplication strategy

There can be many strategies; here is one. The extractor LLM receives not only the user's latest message but also the **existing memories**. You ask it, nicely, to extract memories from the current message, and for each one to tell you whether it already exists in the store. Every memory then comes with a True or False, and you loop over the list and add only the ones that are new. That is the only change. Other strategies are also possible.

The code is the same as before (imports, store, LLM), except the Pydantic model is more expressive. `MemoryDecision` still has `should_write` and `memories`, but `memories` is now a list of **`MemoryItem`** instead of a list of strings. `MemoryItem` is itself a Pydantic model with two components: the **text** of the memory, and a **boolean** saying whether it is new or old.

```python
class MemoryItem(BaseModel):
    text: str
    is_new: bool


class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)


memory_extractor = memory_llm.with_structured_output(MemoryDecision)
```

The memory system prompt evolves too. It reads: "You are responsible for updating and maintaining accurate user memory." It is given the existing memory under CURRENT USER DETAILS, and its task is:

- "Review the user's latest message."
- "Extract user-specific info worth storing long-term."
- "For each extracted item, set `is_new=True` only if it adds new information compared to CURRENT USER DETAILS."
- "If it is basically the same meaning as something already present, set `is_new=False`."
- "Keep each memory as a short atomic sentence."
- "No speculation; only facts stated by the user."
- "If there is nothing memory-worthy, return an empty list."

The earlier explanation had it the other way round (True as existing, False as new). As the prompt is written, **True means a new memory and False means an existing one**.

```python
MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS:
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term.
- For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
- If it is basically the same meaning as something already present, set is_new=false.
- Keep each memory as a short atomic sentence.
- No speculation; only facts stated by the user.
- If there is nothing memory-worthy, return an empty list.
"""
```

The rest of the node is mostly the same. The only difference is that when adding memories it checks `is_new`, and adds a memory only when `is_new` is True:

```python
def remember_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id, "details")

    items = store.search(namespace)  # (implied, not shown in narration)
    user_details_content = "\n".join(f"- {it.value['data']}" for it in items)  # (implied, not shown in narration)

    last_msg = state["messages"][-1].content

    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=user_details_content)),
            {"role": "user", "content": last_msg},
        ]
    )

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new:
                store.put(namespace, str(uuid.uuid4()), {"data": mem.text})

    return {"messages": [{"role": "assistant", "content": "Noted."}]}
```

The graph code after it is exactly the same.

### Demo: no more duplicates

Send "My name is Nitish" and then "I like Python programming language"; the store holds those two memories. Now send "My name is Nitish" again, and "I like Python for programming" again. Check the store: still the same two, with **no duplication**. The duplication problem is solved.

## The merged chatbot: remember, then chat

You have now seen both halves: using existing memories and creating new ones. Merge them into a proper chatbot, a single flow that does both:

1. START goes to the **remember** node, which extracts memories from the user's most recent message.
2. Once memories are extracted, the flow goes to the **chat** node, which has two jobs: reply to the user's question, and make that reply contextual and personalised by first checking what the store knows about the user.
3. Then the workflow ends.

The whole workflow interacts with the memory store: the remember node **writes** memories and the chat node **reads** them. Nothing special; it just joins the two pieces studied separately.

```mermaid
flowchart LR
    S(["START"]) --> R["remember<br/>extract memories from<br/>the latest message"]
    R --> C["chat<br/>personalised reply"]
    C --> E(["END"])
    R -.->|"write"| M[("Memory store")]
    M -.->|"read"| C
```

The code is the pieces already shown, put together:

- the imports and the memory store,
- the chat system prompt, copy-pasted from before,
- **two LLMs**: one to extract memories (in the remember node) and one to chat (in the chat node),
- the memory LLM's two Pydantic models from the deduplication section, its structured output, and the memory system prompt,
- the `remember_node`, exactly as in the deduplication version,
- the chat LLM and the `chat_node`, exactly as in the first chatbot.

```python
memory_llm = ChatOpenAI()
memory_extractor = memory_llm.with_structured_output(MemoryDecision)

chat_llm = ChatOpenAI()  # used inside chat_node in place of llm

builder = StateGraph(MessagesState)
builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)
builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")
builder.add_edge("chat", END)

graph = builder.compile(store=store)

config = {"configurable": {"user_id": "u1"}}
```

### Demo: memories created and used in real time

1. "Hi, my name is Nitish" gets "Hi Nitish, it's great to meet you. How can I assist you today?" The store now holds "User name is Nitish".
2. "I teach AI on YouTube" gets "That's great, Nitish. Teaching AI on YouTube sounds like an exciting venture." Printed in full, it also asks questions back, as instructed: "What topics do you cover in your videos? Are there any specific projects or concepts your audience is particularly interested in?" The store now holds two memories.
3. "Explain GenAI simply" gets "Sure, Nitish", an explanation in bullet points (because the prompt asked for that), and questions at the end asking whether you want answers to them. The store **still holds only two memories**, which is expected: nothing in "Explain GenAI simply" is worth remembering, so the LLM sent nothing to add.

```python
graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Nitish"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "I teach AI on YouTube"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "Explain GenAI simply"}]}, config)
```

The chatbot is now creating memories in real time and using them to personalise the LLM's response.

### The flaw: `InMemoryStore` is volatile

Everything works, with one major flaw: `InMemoryStore` keeps all the memories in **RAM**. Close your code or shut down your machine and come back, and all your memories are wiped out. Restart the kernel, run all the code again, and check which memories are available: **none**, because the restart pushed everything out of RAM. Being volatile, `InMemoryStore` cannot be used for production-grade chatbot systems. The fix is a persistent memory store built on a Postgres database.

## Persistent memory with `PostgresStore`

### Running Postgres through Docker

To use `PostgresStore` you first need a Postgres database on your machine. The setup is exactly the one from the last video on short-term memory: run Postgres through Docker. The exact steps are written out in the notebook:

1. **Install Docker.** Search for Docker Desktop, download it for your machine and install it. (It is already installed here.)
2. **Check it works.** Open a new terminal and run the version command. Output like a version string means Docker is installed correctly.

   ```bash
   docker --version
   ```

3. **Run the Postgres command** from the notebook. It installs a Postgres image inside Docker and runs it. Do this step carefully.

Running it here first failed with "Docker cannot connect to the Docker daemon", because Docker was installed but not started. After starting Docker and running the command again, it complained that a container with that name was already running, left over from earlier testing. The fix was to remove that container under Containers and the image under Images, then run the command again. This time it reported that the image was not on the machine, so it downloaded the Docker image and then ran it as a container. The freshly downloaded image appears under Images, and the container, **langgraph-postgres**, is running. To check from the terminal:

```bash
docker ps
```

It shows the Postgres 16 version running on the machine. You are now ready to run the code.

### The code with `PostgresStore`

First install the libraries with the exact install command given in the notebook (already installed here).

```bash
pip install -U "psycopg[binary,pool]" langgraph-checkpoint-postgres  # (implied, not shown in narration)
```

Everything after that is exactly the same as the merged chatbot: the same system prompt, memory LLM, Pydantic models, memory extractor, memory-extraction system prompt, remember node, chat LLM, chat node and graph. The graph cannot be displayed yet because it has not been compiled: the compile now happens **inside a context manager**. You give the database's URL (keep exactly the URL given), open the context manager as `store` (last time, for short-term memory, a checkpointer sat in this position), set the store up, and compile the graph telling it the store is a `PostgresStore`. The rest is the same: the config, then the messages.

```python
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres"  # (implied, not shown in narration)

with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()

    graph = builder.compile(store=store)

    config = {"configurable": {"user_id": "u1"}}

    graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Nitish"}]}, config)
    graph.invoke({"messages": [{"role": "user", "content": "I teach AI on YouTube"}]}, config)
    result = graph.invoke({"messages": [{"role": "user", "content": "Explain GenAI simply"}]}, config)

    print(result["messages"][-1].content)

    for item in store.search(("user", "u1", "details")):
        print(item.value["data"])
```

The response starts "Sure, Nitish", explains point by point, and ends with some questions. The stored memories are two:

```text
Nitish teaches AI on YouTube
User's name is Nitish
```

### Proving persistence

That does not yet prove the memories are persistent, so restart the kernel. After the restart, run only the last cell, which shows what is currently stored in memory. Both memories are **still there, even after the restart**. Because you used the Postgres store, your memories are now persistent and will stay on your machine for many days, even if you turn the machine off.

This is the setup used in a production-grade chatbot: either Postgres or Redis.

## Wrapping up

Take these notebooks and run them on your machine, and if possible write your own code and implement this in your own project; that is how the learning sticks. This video built up, from scratch, how to implement long-term memory inside LangGraph.
