---
id: ai-security-memory
title: "Module 3: Agentic Memory Techniques"
sidebar_label: "3 · Agentic memory"
sidebar_position: 3
slug: /projects/ai-security/memory
description:
  "Thirteen memory techniques in the order they were invented: buffer,
  sliding window, summary, summary buffer and token buffer for the short
  term; vector store, entity, episodic, semantic, procedural and
  self-reflection memory for the long term; then routing between stores and
  forgetting with half-life decay. Each with its diagram, trade-offs and a
  runnable Python implementation."
tags:
  [projects, memory, agents, langmem, chromadb, summarisation, episodic-memory, ebbinghaus]
---

import Infographic from '@site/src/components/Infographic';

> **Module 3 of 4** ·
> [Watch from 2:47:50](https://www.youtube.com/watch?v=rQE3w8Qjx98&t=10070s) ·
> about three hours of the 7h48m course
>
> From Krish Naik's *The Complete AI Security Course In 8 Hours*. This module
> is taught by Chirantan Lonkar over two live sessions; the second opens at
> 4:05 with a recap, folded into the sections below. Notes follow the module
> in order. Diagrams redraw his architecture notes and notebook figures;
> blocks marked *Not from the session* are additions.

"Give the agent memory" is not one decision. It is a choice between
thirteen techniques, each invented to fix the one before it, and this module
walks that lineage from a plain list of messages to memories that fade on a
half-life.

:::warning The notebooks are not linked
Chirantan runs thirteen notebooks from a local repository called
`Agent_Memory_Techniques` and says they are on his GitHub, but the course
description doesn't link them and no URL appears on screen. The code in this
chapter was written for these notes: small, runnable implementations of what
each notebook demonstrates, using the same FinCoach example. Where the
session showed specific numbers or outputs, those are reported as the
session's, not reproduced from this code.
:::

## The project that needs all of this: MOSAIC

Chirantan opens with the production system he is building next (2:48 to
2:53), because it uses almost every idea in the module. **MOSAIC**, a
multi-agent clinical-trial intelligence engine, reads two US government
databases: ClinicalTrials.gov, with around half a million studies, and
PubMed's research papers. The business problem is scale. A researcher can
cross-check five, ten or fifteen trials by hand; nobody can do it for two
lakh (200,000) of them.

The pipeline is the familiar RAG shape. Three ingestion scripts pull and
parse the data, Google Cloud Storage keeps the raw and processed files, and
Cloud SQL (PostgreSQL with pgvector) doubles as both the relational store
and the vector database. Chunking and embedding are standard, so he skips
them.

<Infographic
  src="/img/ai-security/m3-mosaic.svg"
  alt="MOSAIC architecture: data sources, ingestion, GCP storage, processing, a LangMem memory layer with episodic, procedural and semantic memory, a supervisor with six specialist agents, a human-in-the-loop gate with a learning loop, FastAPI on Cloud Run, and observability."
  caption="Redrawn from the mentor's architecture board, 2:48 to 2:52."
/>

What makes it interesting is the memory layer, built with **LangMem**, which
uses all three long-term memory types for different jobs. **Episodic**
memory keeps past research sessions, so a question about type 2 diabetes
trials or vitamin D3 absorption can find what was found before. **Procedural**
memory holds how the agents should reason, and changes when a human rejects
a finding. **Semantic** memory keeps durable knowledge about each trial
sponsor: a credibility score, a count of broken promises, an average delay.
Six specialist agents run in parallel under a seventh, a supervisor.

Human review is not blanket. Each specialist has a confidence threshold, and
only signals below it wait in a review queue. A rejection with a reason
becomes a new procedural rule that every future session loads, which he
calls the **learning loop**. LangSmith, Cloud Logging and Secret Manager form
the observability layer.

The routing between memory stores in this system comes back at the end of
the module.

## Memory is more than memorising facts

Everyone talks about agent memory, Chirantan says, but few can explain what
it means (2:53 to 3:00). Memory is not "make my agent memorise facts". Many
people jump straight to LangMem and episodic or semantic memory without
knowing why they'd pick semantic over a simple summary. So the module traces
the **lineage**: start with the simplest technique, see what breaks, and
watch each next technique fix it.

| # | Technique | Term | The idea in one line |
| --- | --- | --- | --- |
| 1 | Conversation buffer | Short | Keep every message, re-send them all |
| 2 | Sliding window | Short | Keep only the last *k* turns |
| 3 | Summary | Short | Compress old turns into a running summary |
| 4 | Summary buffer | Short | Recent turns verbatim, older turns summarised |
| 5 | Token buffer | Short | Drop the oldest messages to stay under a token budget |
| 6 | Vector store | Long | Embed every turn; retrieve by meaning across sessions |
| 7 | Entity | Long | One structured record per person or thing, updated in place |
| 8 | Episodic | Long | Each session packaged as a timestamped episode |
| 9 | Semantic | Long | Durable facts and behaviour patterns distilled from episodes |
| 10 | Procedural | Long | Learned rules that change how the agent acts |
| 11 | Self-reflection | Long | The agent's own post-mortems, reused next time |
| 12 | Memory routing | Both | Send each read or write to the right store |
| 13 | Forgetting and decay | Both | Let unused memories fade and prune them |

<Infographic
  src="/img/ai-security/m3-lineage.svg"
  alt="The thirteen memory techniques: five short-term techniques that live in RAM, six long-term ones that live in a database, plus memory routing and forgetting."
  caption="The lineage of the module's notebooks, 2:55."
/>

He has built these thirteen notebooks and plans 26 to 28 in total; the rest,
including hierarchical and graph memory, are for a later session.

**The survey.** Everything, he says, starts with a survey paper from the
National University of Singapore,
[*Memory in the Age of AI Agents*](https://arxiv.org/abs/2512.13564). It
organises the field by three questions: what *carries* memory (its forms),
*why* agents need it (its functions), and *how* it forms, evolves and is
retrieved (its dynamics).

<Infographic
  src="/img/ai-security/m3-survey.svg"
  alt="The survey organises agent memory by forms, functions and dynamics."
  caption="Redrawn from the survey's structure as shown in the session, 2:57 to 3:00."
/>

**Does the agent even need memory?** Before any AI, he asks the class
whether classic shop-floor automation was stateless or stateful. The class
splits, which is his point: think about what statefulness buys and costs
before adding it. A **state** is simply memory shared across the schema of
your workflow.

He also points to [MAGMA](https://arxiv.org/abs/2601.03236), a multi-graph
memory architecture, since most enterprises he meets are moving to graph
memory; to Graphiti, which he'll cover later; and to the
[Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
repository, "a goldmine". Read the papers with NotebookLM or Claude if they
feel heavy.

:::note
Graphiti is described as published by the Neo4j team. It is built by
**Zep**, as open source, and uses a graph database such as Neo4j as its
store.
:::

## The running example and the shared code

Every notebook uses one character: **FinCoach**, a personal-finance
advisor for users in India, talking to Chiru, who earns ₹1,20,000 a month,
spends ₹60,000, has a ₹50,000 fixed deposit (FD) maturing soon, and is risk
averse. The same conversation is replayed through each technique so you
can see what it remembers and what it loses.

The implementations below share one helper module.

### Complete file: `memory_common.py`

*Not from the session.* One model, one tokenizer, one vector store. Put
`OPENAI_API_KEY=...` in a `.env` file next to it, and install the packages
with `pip install openai tiktoken chromadb python-dotenv`.

```python
"""Shared helpers for the memory examples: one model, one tokenizer, one vector store."""
import json
import os

import chromadb
import tiktoken
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads OPENAI_API_KEY from .env

client = OpenAI()
MODEL = "gpt-4o-mini"
ENCODING = tiktoken.get_encoding("o200k_base")  # tokenizer of the GPT-4o family

SYSTEM_PROMPT = (
    "You are FinCoach, a friendly personal-finance advisor for users in India. "
    "Answer in plain language, use rupees, and keep replies under 120 words."
)

def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

def message_tokens(messages: list[dict]) -> int:
    # Roughly 4 extra tokens per message for the role and separators.
    return sum(count_tokens(m["content"]) + 4 for m in messages)

def chat(messages: list[dict], temperature: float = 0.7, max_tokens: int = 1024) -> str:
    response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return response.choices[0].message.content

def chat_json(system: str, user: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return json.loads(response.choices[0].message.content)

def transcript(messages: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

_chroma = chromadb.PersistentClient(path="./memory_db")
_embed = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small"
)

def get_collection(name: str):
    return _chroma.get_or_create_collection(name, embedding_function=_embed)
```

`chat` uses temperature 0.7 and a 1,024-token reply cap, the settings the
session's helper uses. `chat_json` forces a JSON reply for the extraction
steps later on.

## 1. Conversation buffer memory

The first technique starts from one fact (3:00 to 3:12, recapped at 4:07).
**LLMs are stateless.** An API call has no idea what the model did a second
ago, let alone five minutes ago. That is not a bug but a feature: every
request is isolated.

The simplest fix is a **buffer**, and a buffer is just a list. Each turn, you
append the new messages and re-send the whole list. Messages have strict
roles: `system`, `user` and `assistant`, plus `tool` messages once agents
call tools. With OpenAI's API the system message sits inside the list as its
first item.

<Infographic
  src="/img/ai-security/m3-buffer.svg"
  alt="Conversation buffer: every turn re-sends the whole history, so prompt tokens grow from 149 to 976 over ten turns, with the trade-offs listed below."
  caption="Redrawn from the notebook's turn table and token analysis, 3:02 to 3:12."
/>

The pain point is visible straight away: the list only grows. Turn one
sends about 50 tokens, turn two 130, turn three 450. Even GPT-4o's
128K-token window fills eventually, and long before that the bill hurts.
With 10,000 users on turn 50, one call can reach 8,000 tokens, about 160
times the first. Early LangChain projects were built exactly like this.

In the live demo, five FinCoach turns ("Hi, I'm Chiru, my monthly take-home
is 120…", expenses, no investments, what to do with the FD, a plan based on
everything) push the prompt from 279 to 430, 602, 755 and 907 tokens. Five
turns have used **18.3%** of the token budget, before any PDF or scraped page
is attached. The notebook's ten-turn analysis grows the prompt from 149 to
976 tokens, a factor of 6.6.

| Strength | Weakness |
| --- | --- |
| Perfect recall within a session | Token cost grows every turn |
| Zero implementation complexity | Hard context-window ceiling |
| Deterministic, easy to debug | No persistence across sessions |
| No information loss or distortion | No prioritisation of important facts |

"Deterministic" matters: LLMs aren't, and enterprises want as much
determinism around them as they can get. **Recall** here means the agent's
ability to bring back something said earlier when it's needed.

**Production verdict.** Not enough on its own. No production system uses a
raw buffer as its only memory, but every one uses *something like* a buffer
for the active session, bounded by a hard token budget (technique 5 grows
out of this), persisted to a database, and backed by a long-term layer.

:::note
Two claims need tightening. "Agentic AI never uses conversation buffer
memory": inside a single run, every agent loop carries the full message
list, tool messages included; LangGraph's `messages` state is a buffer. What
production avoids is using it as the *only* memory across long sessions,
which is the mentor's own verdict. And the growth is **linear** per call
(each call re-sends one more turn) and **quadratic** in total spend, not
"exponential" as said in the recap.
:::

### Complete file: `t01_buffer.py`

*Not from the session.*

```python
from memory_common import SYSTEM_PROMPT, chat, message_tokens

class ConversationBufferMemory:
    """Keep every message and re-send the whole list on every call."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system = {"role": "system", "content": system_prompt}
        self.buffer: list[dict] = []  # the buffer is just a list

    def messages_for_api(self) -> list[dict]:
        return [self.system, *self.buffer]

    def ask(self, user_text: str) -> str:
        self.buffer.append({"role": "user", "content": user_text})
        reply = chat(self.messages_for_api())
        self.buffer.append({"role": "assistant", "content": reply})
        return reply

if __name__ == "__main__":
    memory = ConversationBufferMemory()
    turns = [
        "Hi, I'm Chiru. My monthly take-home is ₹1,20,000.",
        "My monthly expenses are ₹60,000 for rent, food and transport.",
        "I have an FD of ₹50,000 maturing in three months. I'm risk-averse.",
        "What should I do with the FD maturity amount?",
        "Based on everything I've told you, what's my plan for this year?",
    ]
    for n, text in enumerate(turns, start=1):
        memory.ask(text)
        print(f"turn {n}: prompt ≈ {message_tokens(memory.messages_for_api())} tokens")
```

Run it and the printed prompt size climbs every turn; every other technique
in this module exists to bend that line.

## 2. Sliding window memory

The next idea is obvious from the name (3:12 to 3:30, recapped at 4:10).
Instead of the whole history, send only the last *k*. Chirantan's diagram
uses Alice, since "in computer science every user is called Alice", and a
window of four messages.

<Infographic
  src="/img/ai-security/m3-sliding-window.svg"
  alt="Sliding window with k = 4: the window fills, the first message is evicted, and a fact slides out; the FinCoach example forgets the salary by turn 5."
  caption="Redrawn from the mentor's architecture notes, 3:13 to 3:15 (again at 4:11), and the notebook's FinCoach example."
/>

When the window is full, the oldest message is **evicted**, like a
contestant leaving Bigg Boss, to make room. The cost stops growing: with a
window of ten turns, turn 100 sends exactly ten turns, the same as turn 10.
Cost is constant, not compounding. But "I like Python" has slid out, so the
agent no longer knows it. The class names the drawback in chorus: context
loss.

Two precise points from the notebook:

- **The window counts turns, not tokens.** A turn is one user message plus
  one assistant reply, so a window of *k* = 5 keeps ten messages. That is
  easier to reason about than raw token counts, but the token budget is only
  approximate.
- **Evicted is not deleted.** The pair leaves the active window together,
  but it still exists somewhere. Where it goes is his homework question, and
  the answer drives the next techniques.

**Recency versus completeness.** The FinCoach walk-through uses a window of
three turns.

| Turn | User says | Window afterwards |
| --- | --- | --- |
| 1 | My salary is ₹1,20,000 | turn 1 |
| 2 | My expenses are ₹60,000 | turns 1–2 |
| 3 | I have an FD of ₹50,000 | turns 1–3 |
| 4 | What about SIPs? | turn 1 **evicted** before sending; turns 2–4 |
| 5 | How much should I invest based on my salary? | "Could you remind me of your monthly salary?" |

The user told it in turn 1 and is now frustrated. In the live demo, turn six
("What is my exact monthly take-home?") gets "I'm sorry, I don't have access
to the previous conversations." He compares it with recurrent neural
networks, which lose the early part of a long sequence, and which
transformers replaced. He also recalls a chatbot launched in India in late
2023 that began forgetting by the fifth or sixth turn, and guesses a
cost-saving window was behind it; that is his speculation.

| Strength | Weakness |
| --- | --- |
| Fixed, predictable token cost per turn | Loses early context as the window moves |
| Context window never overflows | May forget important facts from early turns |
| Simple to implement and reason about | Window size has no universal right answer |
| Good coherence over recent turns | No long-term memory across sessions |

**Production verdict.** Almost always present in production, almost never
alone. It handles recency and cost; a long-term retrieval layer, a vector
store or a knowledge graph, catches the facts that slide out. When papers
say "hybrid memory", check which short-term memory is inside it.

### Doubts · Can evicted messages go into a vector store? · 3:19

**Chirantan:** Since evicted messages aren't deleted, could you vectorise
them and retrieve them later?

**Discussion:** Yes, but it is an engineering challenge. The window is a
short-term technique and the vector store a long-term one, and joining them
adds embedding and storage cost. It is possible, and he wants the class to
think about combinations like this instead of memorising techniques.

### Doubts · When should I use a sliding window instead of a token buffer? · 3:29

**Chirantan:** Use a sliding window for the simplest bounded solution when
messages are roughly the same length. Use a token buffer when lengths vary
a lot: if some messages are 10 tokens and others 500, counting messages
wastes the budget.

### Complete file: `t02_sliding_window.py`

*Not from the session.* The window includes the turn being asked, so with
*k* = 3 the first turn is evicted before turn 4 is sent, as in the
walk-through.

```python
from memory_common import SYSTEM_PROMPT, chat
from t01_buffer import ConversationBufferMemory

class SlidingWindowMemory(ConversationBufferMemory):
    """Send only the last k turns. A turn is one user message plus one reply."""

    def __init__(self, k_turns: int = 3, system_prompt: str = SYSTEM_PROMPT):
        super().__init__(system_prompt)
        self.k_turns = k_turns
        self.evicted: list[dict] = []  # evicted from the window, not deleted

    def ask(self, user_text: str) -> str:
        # Make room for the new turn: the oldest user+assistant pair leaves together.
        while len(self.buffer) >= 2 * self.k_turns:
            self.evicted.extend(self.buffer[:2])
            del self.buffer[:2]
        self.buffer.append({"role": "user", "content": user_text})
        reply = chat(self.messages_for_api())
        self.buffer.append({"role": "assistant", "content": reply})
        return reply

if __name__ == "__main__":
    memory = SlidingWindowMemory(k_turns=3)
    for text in [
        "My salary is ₹1,20,000 a month.",
        "My expenses are ₹60,000.",
        "I have an FD of ₹50,000.",
        "What about SIPs?",  # turn 1 is evicted before this is sent
        "How much should I invest based on my salary?",
    ]:
        print(memory.ask(text)[:80])
    print("evicted:", [m["content"] for m in memory.evicted if m["role"] == "user"])
```

:::note
The diagram's *k* counts **messages** (four), while the notebook's window
counts **turns** (three). Both are valid; know which one your implementation
uses, because a four-message window holds only two turns.
:::

*The session then breaks for course promotion (3:30 to 3:37); it is left
out here.*

## 3. Summary memory

Ask the class what summary memory is and they answer at once: summarise the
previous conversation (3:37 to 3:56, recapped at 4:13). Instead of
discarding history, it **compresses** it: fewer tokens, same context.

His analogy is a proverb. You tell him a long story about your aching back
and everything going wrong, and he answers "health is wealth". Your many
words became three, and you still understood him. The notebook's version: at
the end of each chapter of a long book, write a one-page summary; you can't
quote the book any more, but you know what happened. Or a newspaper that
rewrites yesterday's news as one paragraph each morning and builds today's
edition on it.

<Infographic
  src="/img/ai-security/m3-summary.svg"
  alt="Summary memory: the buffer fills, a summariser LLM merges it into a running summary, and new turns start from the summary; progressive summarisation levels 0 to 3; lossy compression and its mitigation."
  caption="Redrawn from the mentor's architecture notes, 3:15 (again at 4:15), and the notebook, 3:48 to 3:50."
/>

Where the sliding window saved tokens and lost context, summary memory saves
tokens and **keeps** the context, which Chirantan calls a win-win. After the
first turns, the summary reads something like "Chiru earns ₹1,20,000 a
month, spends ₹60,000, has an FD of ₹50,000, is risk averse, asked about
SIPs", and every new turn starts from it.

How it works, point by point:

1. **An LLM writes the summary**, not code. You give a model such as
   GPT-4o mini the old turns and a summarisation prompt, and its paragraph
   replaces the raw turns. Generating new text that captures the essence is
   **abstractive summarisation**.
2. **A threshold triggers it**, not every turn. Summarising after every
   message would be expensive. Trigger it by turn count (the buffer exceeds
   *N* turns) or by tokens (the buffer exceeds a budget); the oldest turns
   are then compressed and replaced.
3. **The summary goes in as its own context block**, usually in the system
   message.
4. **Lossy compression is the fundamental risk.** Every summary loses
   something, and the model decides what matters, with no human in the loop.
   A rare but critical instruction, "the user is allergic to equity
   instruments", may survive the first compression and vanish by the third,
   and the finance agent then recommends equity. These **low-frequency,
   high-importance** details are exactly what summarisers drop; he likens
   it to TF-IDF, where a rare term carries the most weight. In medicine it's
   a diabetic being recommended a sugary diet.
5. **Progressive summarisation** chains summaries of summaries. When the
   summary itself grows, it is re-summarised, level by level.

Progressive summarisation is also called **hierarchical**. The demo payoff:
turn six asks "Based on my salary, am I saving enough?". With a sliding
window that failed; now the model has the first three turns as a summary
and turns four to six verbatim, and answers.

He adds one more idea, **context echo**. When several consecutive turns
stay on one topic, say investing a crore in mutual funds, each turn restates
some of the earlier context, so an agent can still answer correctly after
the original message has been evicted. It's a topic for his later context
engineering session.

| Strength | Weakness |
| --- | --- |
| Preserves key facts across long sessions | Summarisation is lossy; details get lost |
| Token cost always bounded | Each summarisation call costs tokens and latency |
| Better than a window for domain-critical facts | Quality depends on the summarisation prompt |
| Natural fit for multi-session continuity | Hard to audit what exactly was summarised |

**Production verdict.** A strong production pattern, used by long-running
coding agents, but only if engineered carefully and only as one layer of a
hybrid. The risk is the third or fourth compression, and in a real system
you're on compression cycle 129, not 3. That, he says, is why context
engineering exists: keeping context intact over long periods despite lossy
compression. The mitigation is a **domain-specific summarisation prompt**
that names the categories of fact that must survive: salary, risk profile,
goals, decisions.

### Doubts · Can we choose which facts must survive? · 3:41

**Krishna:** Can we mark facts such as a person's IDs as important so they
survive summarisation?

**Chirantan:** Yes, and he returns to it with the lossy-compression point:
name the must-keep categories in the summarisation prompt.

### Doubts · How do we know the summary kept everything? · 3:43

**Chirantan:** How would you check that every raw turn was summarised
correctly?

**Chat:** A reflective prompt, an efficient summary prompt, domain
knowledge. Aramis suggests keeping the last few turns verbatim and adding
the summary, which is exactly the next technique.

### Complete file: `t03_summary.py`

*Not from the session.* The summarisation prompt names the fact categories
that must survive, as the notebook recommends.

```python
from memory_common import SYSTEM_PROMPT, chat, transcript

SUMMARY_PROMPT = """You keep the running memory of a personal-finance conversation.
Merge the existing summary and the new messages into one short paragraph.
Always keep the user's name, income, expenses, savings and investments,
risk profile, goals, constraints (things they never want) and decisions made."""

class SummaryMemory:
    """Compress old turns into a running summary written by a second LLM call."""

    def __init__(self, trigger_messages: int = 6, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.trigger_messages = trigger_messages  # turn-based trigger
        self.summary = ""
        self.buffer: list[dict] = []

    def messages_for_api(self) -> list[dict]:
        system = self.system_prompt
        if self.summary:
            system += "\n\nSummary of the conversation so far:\n" + self.summary
        return [{"role": "system", "content": system}, *self.buffer]

    def merge_into_summary(self, messages: list[dict]) -> None:
        self.summary = chat(
            [
                {"role": "system", "content": SUMMARY_PROMPT},
                {
                    "role": "user",
                    "content": f"Existing summary:\n{self.summary or '(none)'}\n\n"
                    f"New messages:\n{transcript(messages)}",
                },
            ],
            temperature=0,
        )

    def ask(self, user_text: str) -> str:
        self.buffer.append({"role": "user", "content": user_text})
        reply = chat(self.messages_for_api())
        self.buffer.append({"role": "assistant", "content": reply})
        if len(self.buffer) >= self.trigger_messages:
            self.merge_into_summary(self.buffer)  # one summariser call...
            self.buffer = []  # ...then the buffer starts empty again
        return reply
```

:::tip Test the summariser, not just the chatbot
*Not from the session.* Plant one must-keep fact early, such as "never
suggest equity", run twenty turns, and assert it's still in `summary`
after every compression. That catches exactly the third-cycle loss
described above.
:::

## 4. Summary buffer memory

Summary buffer memory combines techniques 1 and 3 (3:56 to 4:04, recapped
at 4:15). The most recent messages stay **word for word**; older history
is summarised. His analogy is a long phone call with a friend: you can
repeat the last few sentences exactly, but of what they said twenty minutes
ago you only remember a few facts.

<Infographic
  src="/img/ai-security/m3-summary-buffer.svg"
  alt="Summary buffer: messages stay verbatim in the buffer until it passes a threshold, then the oldest are merged into a running summary; the prompt is summary plus buffer plus the new message."
  caption="Redrawn from the mentor's architecture notes, 3:57 (again at 4:17)."
/>

This answers the sliding-window homework: evicted messages are not wasted,
they are merged into the summary. Ruchi asks whether it isn't just the
sliding window; the difference is exactly that the evicted messages are
summarised and carried forward instead of dropped.

The engineering challenge is the **transition threshold**, the point where
messages move from the buffer into the summary, and how the token budget is
split between the two regions. In the Monday recap the threshold is a token
limit *T*: once the buffer passes it, the oldest messages leave.

| # | Technique | Mechanism | Gap it leaves |
| --- | --- | --- | --- |
| 1 | Buffer | Keep everything verbatim | Unbounded cost |
| 2 | Sliding window | Keep only the last N turns | Loses early context |
| 3 | Summary | Compress old turns into a summary | Recent turns still expensive |
| 4 | **Summary buffer** | **Summary of old + token-capped buffer of recent** | **The production hybrid** |

It suits customer-support agents, coaching agents and long chat
assistants: anything that needs precise recent context plus historical
awareness. In the ten-turn demo, turn ten asks "Can you give me a complete
action plan based on what we've discussed?" and gets one that uses facts
from the whole conversation; the comparison run with a plain buffer had
already used 81% of its budget.

His verdict is more reserved than the notebook's. He calls it a "boring"
technique, fine for small and medium enterprise solutions and research use,
but not how a bank with millions of customers should engineer memory. And
the next problem remains: *how long* can the context be kept? He points to
work on shrinking the KV cache as one research direction.

### Complete file: `t04_summary_buffer.py`

*Not from the session.*

```python
from memory_common import SYSTEM_PROMPT, chat, message_tokens
from t03_summary import SummaryMemory

class SummaryBufferMemory(SummaryMemory):
    """Recent turns stay verbatim; turns pushed out of the buffer join the summary."""

    def __init__(self, max_buffer_tokens: int = 400, system_prompt: str = SYSTEM_PROMPT):
        super().__init__(system_prompt=system_prompt)
        self.max_buffer_tokens = max_buffer_tokens  # the transition threshold

    def ask(self, user_text: str) -> str:
        self.buffer.append({"role": "user", "content": user_text})
        reply = chat(self.messages_for_api())  # summary + buffer + new message
        self.buffer.append({"role": "assistant", "content": reply})

        evicted = []
        while message_tokens(self.buffer) > self.max_buffer_tokens and len(self.buffer) > 2:
            evicted.extend(self.buffer[:2])  # oldest turn first
            del self.buffer[:2]
        if evicted:
            self.merge_into_summary(evicted)  # evicted messages are not wasted
        return reply
```

## Between the sessions

The Friday session ends at 4:04 after the first four techniques. Monday's
opens with a point about the job market: with today's coding assistants
anyone can build a basic agentic workflow. The opportunity for the next
three to five years is the **infrastructure** around it: memory, security
and governance. He also mentions context engineering and, newer, "loop
engineering", which he hasn't explored yet. Then he recaps techniques 1 to
4 (4:07 to 4:20), which is folded into the sections above.

## 5. Token buffer memory

Token buffer memory keeps a buffer that **never exceeds a fixed number of
tokens** (4:20 to 4:24). When a new message would breach the limit, the
oldest messages are dropped, one at a time, until it fits. The notebook's
image is a ticket tape of fixed length: new text prints on the right, the
left end is torn off.

<Infographic
  src="/img/ai-security/m3-token-buffer.svg"
  alt="Token buffer: count tokens with tiktoken and evict the oldest message until the history fits max_token_limit."
  caption="Redrawn from the mentor's architecture notes, 4:05."
/>

Input tokens per call are fixed by a simple formula:

$$
\text{input tokens} = \text{system prompt tokens} + \min(\text{history tokens},\ \text{max buffer tokens})
$$

How it differs from its neighbours:

| | Sliding window (2) | Token buffer (5) |
| --- | --- | --- |
| Control unit | Turns | Tokens |
| Eviction granularity | Whole turns (2 messages) | One message at a time |
| Budget precision | Approximate | Exact |
| Turn-pair integrity | Always preserved | May be broken |
| Extra calls or latency | None | None |

| | Summary buffer (4) | Token buffer (5) |
| --- | --- | --- |
| On overflow | Compress into the summary | Drop, no recovery |
| Information loss | Partial (lossy compression) | Total (hard eviction) |
| Extra LLM calls | Yes, summarisation | None |
| Latency spike on overflow | Yes | None |
| Cost certainty | Budget plus summary overhead | Exact budget |

Point one from the notebook: it is the simplest and most predictable
short-term memory, with no compression calls, no latency spikes and no
external dependencies. It's what LangChain's `ConversationTokenBufferMemory`
implements. Paired with external retrieval it becomes, to an extent,
production-ready. In a hybrid, the long-term half is a separate choice; the
short-term half is one of these five, and choosing is your call as an AI
engineer.

:::note Check the worked example
The notebook's example uses a 500-token budget with 475 tokens in history
when a 65-token message arrives: 540, over. It drops the first user message
(60 tokens) and calls the result, 480, "still over", then drops another.
But 480 fits under 500, so only one message needs to go.
:::

### Complete file: `t05_token_buffer.py`

*Not from the session.* Trimming pops single messages, so a reply can
survive without the question that prompted it: the turn-pair integrity
concern above.

```python
from memory_common import SYSTEM_PROMPT, chat, message_tokens
from t01_buffer import ConversationBufferMemory

class TokenBufferMemory(ConversationBufferMemory):
    """Keep the history under a hard token budget by dropping the oldest messages."""

    def __init__(self, max_token_limit: int = 500, system_prompt: str = SYSTEM_PROMPT):
        super().__init__(system_prompt)
        self.max_token_limit = max_token_limit

    def _trim(self) -> None:
        # One message at a time, so a user/assistant pair can be split.
        while self.buffer and message_tokens(self.buffer) > self.max_token_limit:
            self.buffer.pop(0)

    def ask(self, user_text: str) -> str:
        self.buffer.append({"role": "user", "content": user_text})
        self._trim()
        reply = chat(self.messages_for_api())
        self.buffer.append({"role": "assistant", "content": reply})
        self._trim()
        return reply
```

### Choosing a short-term memory

*A summary of the five techniques above.*

| Technique | Cost per call | What it loses | Use it when |
| --- | --- | --- | --- |
| Buffer | Grows every turn | Nothing, until the window overflows | Short sessions, debugging |
| Sliding window | Constant | Everything older than *k* turns | Messages of similar length, recency matters most |
| Summary | Bounded, plus summariser calls | Detail, through lossy compression | Long sessions where facts matter more than wording |
| Summary buffer | Bounded, plus summariser calls | Old detail only | The default for long chats and support agents |
| Token buffer | Exact budget | Everything past the budget | Strict cost control, paired with retrieval |

## 6. Vector store memory

With the vector store the module crosses into **long-term memory** (4:24 to
4:41). Techniques 1 to 5 live in RAM and reset when the session ends, which
suits temporary chatbots and Q&A bots, as the class answers. They also
retrieve context by **position**, meaning recency; recommendation systems,
someone suggests, are a case where recent context should weigh more.

| Techniques 1–5 | Technique 6 onwards |
| --- | --- |
| Memory lives in RAM | Memory lives in a **database** |
| Resets when the session ends | **Survives** across sessions |
| Retrieved by position (recency) | Retrieved by **semantic similarity** |
| One user, one buffer | **Multi-tenant**: thousands of users, one store |
| Storage: a Python list | Storage: a vector database (ChromaDB) |

The idea: convert every turn into a vector, store it persistently, and at
the start of each new turn retrieve the most **semantically relevant** past
messages, not just the most recent. The notebook's image: techniques 1 to 5
were a notepad on the advisor's desk; this is a searchable filing cabinet.

<Infographic
  src="/img/ai-security/m3-vector-store.svg"
  alt="Vector store memory: RAM versus database, and the worked flow from a portfolio question through embedding and ChromaDB search to personalised advice; the stale fact problem."
  caption="Redrawn from the notebook's tables and worked flow, 4:24 to 4:41."
/>

The search is an **approximate nearest neighbour** (ANN) lookup that returns
the top *k*, three here. Along the way he revisits the basics:

- **What a vector is.** In physics, anything with magnitude and direction.
  In two dimensions, a line from the origin to a point on the plane.
- **Do more dimensions capture more?** Not always, says Priyang, and
  Chirantan agrees: past a point, more dimensions can mean over-fitting, and
  what they capture isn't visible to us anyway.
- **Learn from books and good engineering blogs first**, then use Claude,
  and only then YouTube. A YouTuber often distils ten other videos into an
  eleventh.
- **What is stored.** A document object has `page_content`, which is what
  gets embedded, and `metadata`. A stored record keeps four things: an
  **ID**, the **chunk** text, its **embedding** and its **metadata**.
- **Vector store or vector database?** A thin line: a database adds CRUD
  operations, authentication and concurrency. ChromaDB is an open-source
  vector *database*. Read up on where FAISS sits.

About 80 to 90% of the enterprise projects he sees are RAG applications of
some kind, so this is the most familiar technique in the list.

| Strength | Weakness |
| --- | --- |
| Persists across sessions and restarts | No concept of time; stale facts can surface |
| Finds relevant context by meaning | Only as good as the embedding model |
| Multi-tenant through metadata filtering | Extra embedding call per turn |
| Scales to millions of messages | No relationships between facts |
| Works across all sessions | Can't tell "current" from "historical" |

**Production verdict.** The production standard for long-term memory, with
one serious limit: the **stale fact problem**. When facts change, such as a
salary, a job or a diagnosis, both the old and new values are stored and
both can come back. The notebook's fix is a temporal knowledge graph such as
Graphiti, which records *when* each fact was true.

He closes by asking which hybrids make sense. Buffer plus vector store, the
chat argues, keeps the bloat and adds noise. Sliding window plus a vector
store of the *evicted* messages makes sense. Token buffer plus vector store
looks good. His advice is to experiment: you'll build some absolute
bloopers, and you'll learn.

:::note
Two statements need correcting. ChromaDB's local storage is an embedded
SQLite database managed by Chroma; you work with it through Chroma's client,
not through MySQL or another SQL client. And when he asks which algorithm
"vectorises" a query, the answer is the **embedding model**, a neural
network. HNSW and IVF, suggested in the chat, are **index structures** that
make the nearest-neighbour search over stored vectors fast.
:::

He also shares a January 2026 paper, *Memory is Reconstructed, Not
Retrieved: Graph Memory for LLM Agents*, as the direction he's reading.

### Complete file: `t06_vector_store.py`

*Not from the session.* One Chroma collection holds every user's turns;
`user_id` metadata keeps them apart, and the current session is excluded
from recall because it's already in the small verbatim window.

```python
import time
import uuid

from memory_common import SYSTEM_PROMPT, chat, get_collection

turns = get_collection("fincoach_turns")  # persists on disk across sessions

class VectorStoreMemory:
    """Store every turn as a vector; before each call, retrieve the most similar past turns."""

    def __init__(self, user_id: str, session_id: str, k: int = 3, recent_turns: int = 2):
        self.user_id, self.session_id = user_id, session_id
        self.k = k
        self.recent: list[dict] = []  # a small verbatim window for the current session
        self.recent_turns = recent_turns

    def recall(self, query: str) -> list[str]:
        if turns.count() == 0:
            return []
        result = turns.query(
            query_texts=[query],
            n_results=self.k,
            # One store, many users: filter by metadata. This session is already in self.recent.
            where={"$and": [{"user_id": self.user_id}, {"session_id": {"$ne": self.session_id}}]},
        )
        return result["documents"][0]

    def remember(self, user_text: str, reply: str) -> None:
        turns.add(
            ids=[str(uuid.uuid4())],
            documents=[f"User: {user_text}\nFinCoach: {reply}"],
            metadatas=[{"user_id": self.user_id, "session_id": self.session_id, "ts": time.time()}],
        )

    def ask(self, user_text: str) -> str:
        memories = self.recall(user_text)
        system = SYSTEM_PROMPT
        if memories:
            system += "\n\nRelevant things from earlier conversations:\n" + "\n".join(
                f"- {m}" for m in memories
            )
        messages = [{"role": "system", "content": system}, *self.recent,
                    {"role": "user", "content": user_text}]
        reply = chat(messages)
        self.recent += [{"role": "user", "content": user_text},
                        {"role": "assistant", "content": reply}]
        self.recent = self.recent[-2 * self.recent_turns:]
        self.remember(user_text, reply)
        return reply
```

This is the "sliding window plus vector store" hybrid from the discussion:
a two-turn window for the current session, semantic recall for everything
before it.

## 7. Entity memory

Entity memory comes from NLP's **named entity recognition** (NER), which
Chirantan says should be learnt before AI engineering, alongside bag of
words and n-grams (4:41 to 4:56). How does a model know "Tesla" means the
car company and not Nikola Tesla? Every token is classified: currency,
date, place, person. In "Barack Obama, the 44th President of the USA, was
born in Honolulu, Hawaii", Barack Obama is a person, 44 a number, and the
USA and Hawaii are **GPEs** (geopolitical entities). "Amazon is expanding
rapidly" is the company; "the Amazon is the largest rainforest" is not.
"Jordan won the MVP award" is a person; Jordan in the Middle East is a
country.

Entity memory uses the same idea to keep **one structured record per
entity**, updated as the conversation goes. The image is a personal
assistant with a contact card for every person you mention: say "Sarah got
promoted" and they update Sarah's card, knowing Sarah is a person.

<Infographic
  src="/img/ai-security/m3-entity.svg"
  alt="Entity memory: an extractor turns each message into structured facts in a key-value store that updates in place, used to build the prompt; compared with vector store memory."
  caption="Redrawn from the mentor's architecture notes, 4:05, and the notebook's comparison table."
/>

The store is a plain JSON dictionary. "Chirantan earns ₹1,20,000 per month"
becomes two fields: a name and a salary. From an earnings-call transcript it
would be the CFO's name, the organisation, the profit and the loss. The
mental model is a **CRM record that updates itself**: when a customer says
"I changed jobs", the old employer is replaced and the record always shows
the current state.

| | Vector store (6) | Entity memory (7) |
| --- | --- | --- |
| Storage unit | Raw message text | Structured key-value facts |
| Retrieval | Semantic similarity | Direct key lookup |
| Update model | Append-only | **In place**: the old value is replaced |
| Stale facts | Old and new both retrieved | Only the current value exists |
| Query | "find messages about salary" | `profile["salary"]` |
| Token cost | Varies with results | Fixed (profile size) |
| Best for | Fuzzy, open-ended recall | Structured facts that change |

Direct lookup is metadata filtering by another name; a vector store can do
it too, which is why it has a metadata field.

### Hot path and background

Entity memory introduces a distinction from the LangMem documentation.
Memory can be updated **in the hot path**, as each message arrives, before
the reply, or **in the background** (a cold path), by a separate process
later.

<Infographic
  src="/img/ai-security/m3-hot-background.svg"
  alt="Hot path versus background memory updates: update before replying, or reply first and update later in a separate process."
  caption="Redrawn from the LangMem documentation as shown in the session, 4:50 to 4:52."
/>

His example: tell the agent "from now on, call me Alex". On the hot path the
memory updates first, taking a second, and the reply is "Hi Alex". In the
background the reply still says "Chirantan", and only some time later does
the agent switch to "Alex". The hot path costs latency, which large systems
take very seriously; entity memory usually runs on the hot path.

| Strength | Weakness |
| --- | --- |
| Always current: in-place updates, no stale facts | Extraction can mislabel or hallucinate facts |
| Direct lookup, no similarity search | Captures no relationships between entities |
| Fixed, predictable output | Adds an extraction call to every turn |

Use it where tracking an entity's progress matters, such as an HR engine
that follows each employee's salary history, or an automation that keeps a
small knowledge base or spreadsheet up to date.

### Doubts · Can entity memory be the only memory layer? · 4:58

**Chirantan:** Would a standalone entity layer be a good engineering
decision?

**Priyank:** No, it captures no relations.

**Chirantan:** Right; for relationships, graph memory is the better tool.

He adds an aside on NVIDIA's paper
[*Small Language Models are the Future of Agentic AI*](https://arxiv.org/abs/2506.02153):
the argument that small models are powerful enough, more operationally
suitable and more economical for agent calls. It matches what he sees in
banks and manufacturers, which fine-tune small models and run them on
premises on their own GPUs; his next project routes between small and
medium models.

### Complete file: `t07_entity.py`

*Not from the session.* Extraction runs on the hot path, before the reply.

```python
import json
from pathlib import Path

from memory_common import SYSTEM_PROMPT, chat, chat_json

EXTRACT_PROMPT = """Read the user's message and extract facts about the user.
Return a JSON object containing only fields that were explicitly mentioned, for example
{"name": "Chiru", "monthly_salary_inr": 120000, "employer": "TCS", "risk_profile": "averse"}.
Return {} if the message contains no facts about the user."""

class EntityMemory:
    """One structured profile per user, updated in place on the hot path."""

    def __init__(self, user_id: str, path: str = "entities.json"):
        self.path = Path(path)
        self.store = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.profile = self.store.setdefault(user_id, {})

    def update(self, user_text: str) -> dict:
        facts = chat_json(EXTRACT_PROMPT, user_text)
        self.profile.update(facts)  # in-place: a new value replaces the old one
        self.path.write_text(json.dumps(self.store, indent=2, ensure_ascii=False))
        return facts

    def ask(self, user_text: str) -> str:
        self.update(user_text)  # update memory first, then respond
        system = (SYSTEM_PROMPT + "\n\nCurrent facts about this user:\n"
                  + json.dumps(self.profile, ensure_ascii=False))
        return chat([{"role": "system", "content": system},
                     {"role": "user", "content": user_text}])
```

## 8. Episodic memory

The last six techniques are inspired by human memory, and episodic memory
is the first (4:56 to 5:15). Chirantan notes that LangMem, from the LangChain
ecosystem, implements the three MOSAIC types, and that **every memory
structure here is still experimental**: none is a sure-shot production
pattern, some degrade or become impossible to scale, and choosing is trial
and error.

You don't remember your life as a flat list of facts; you remember
**episodes**, bounded by time and place. Episodic memory gives agents the
same: each conversation is kept as a discrete, timestamped episode. His
example is a coding assistant. You ask for code, run it, get an error, paste
the error back, and the fix works. That whole exchange becomes an episode:
the user had this issue, this solution worked. When someone in New Zealand
hits the same error, the solution is already there. Coaching agents need to
recall last week's goals, and project agents what was decided three days
ago.

<Infographic
  src="/img/ai-security/m3-episodic.svg"
  alt="Episodic memory: a boundary detector closes an episode, a packager summarises it, and the episode store is searched by time and topic at query time."
  caption="Redrawn from the mentor's architecture notes, 4:05."
/>

The hard engineering problem, he says, is the **episode boundary**: where
one episode ends and the next begins. It can be **session-based** (one
session, one episode) or **topic-based** (a type error, then an index error,
then a request timeout: each topic shift starts a new episode).

Each episode is a structured JSON record, a diary entry, capturing what was
discussed, what was decided, what advice was given and the emotional
context. The foundation is Endel Tulving's 1972 distinction: **episodic**
memory is specific events in time; **semantic** memory is general knowledge,
independent of when it was learnt.

| | Vector store (6) | Entity (7) | Episodic (8) |
| --- | --- | --- | --- |
| Unit stored | Individual messages | Individual facts | Complete sessions |
| Granularity | Message | Field | Session |
| Time awareness | Timestamp metadata | Last-updated time | Episode date, duration, sequence |
| Retrieval | Semantic search | Direct lookup | Semantic + temporal + type filters |
| Answers | "Find messages about X" | "What is X now?" | "What happened in session Y?" |
| Mutability | Append-only | Updated in place | **Immutable** |

Two points from the notebook. Episodes are **generated at session end, not
turn by turn**: after the session closes, a structured summarisation
produces the record, asynchronously, so the user never waits. That saves
tokens, but it's summarisation, so lossy compression applies again. And
**retrieval mixes strategies**: semantic, temporal, by type, or a hybrid.

| Strength | Weakness |
| --- | --- |
| Session-level narrative context | Adds post-session latency |
| Compliance and audit trail | Quality depends on the generation prompt |
| Case-based reasoning from past sessions | More expensive than raw message storage |
| Temporal reasoning: "what happened in April?" | Doesn't replace entity or vector memory |
| Immutable: history can't be rewritten | Retrieved episodes add tokens to each call |

**Production verdict.** Essential for regulated domains and long-running
agent relationships, as a complement to entity and vector memory.

### Doubts · Doesn't storing episodes exceed the context window? · 5:09

**Santosh:** Won't all these episodes bloat the context?

**Chirantan:** Anything can exceed the context window, and yes, episodes
add up. That's why context management exists as a field. "Expensive" here
means context as well as money. Retrieve a few relevant episode summaries,
never every transcript.

### Complete file: `t08_episodic.py`

*Not from the session.* Session-based boundaries: `close_session` runs once
when a session ends. The date is stored as a number so it can be
range-filtered.

```python
from datetime import date

from memory_common import chat_json, get_collection, transcript

episodes = get_collection("fincoach_episodes")

EPISODE_PROMPT = """Turn this finished advice session into an episode record.
Return JSON with the keys: title, topics (a list), summary (2-3 sentences),
decisions (a list), advice_given (a list), emotional_context (one phrase)."""

def close_session(user_id: str, session_id: str, messages: list[dict]) -> dict:
    """Run once, when the session ends: package the whole session as one episode."""
    episode = chat_json(EPISODE_PROMPT, transcript(messages))
    today = date.today()
    episodes.add(  # add, never update: episodes are immutable
        ids=[session_id],
        documents=[episode["summary"]],
        metadatas=[{
            "user_id": user_id,
            "day": int(today.strftime("%Y%m%d")),  # numeric, so it can be range-filtered
            "title": episode["title"],
            "topics": ", ".join(episode["topics"]),
        }],
    )
    return episode

def recall_episodes(user_id: str, query: str, since_day: int | None = None, k: int = 2) -> list[str]:
    """Semantic match on the summary, optionally limited to a time range."""
    where = {"user_id": user_id}
    if since_day is not None:
        where = {"$and": [{"user_id": user_id}, {"day": {"$gte": since_day}}]}
    result = episodes.query(query_texts=[query], n_results=k, where=where)
    return [f"{m['day']} · {m['title']}: {doc}"
            for doc, m in zip(result["documents"][0], result["metadatas"][0])]
```

In production, call `close_session` from a background worker, not the
request path, so the user never waits for it.

## 9. Semantic memory

Semantic memory is for **facts** (5:15 to 5:20). You know Paris is the
capital of France, but not the moment you learnt it. If a user mentions
their favourite language is Python in session one, their team size in
session two and their deployment target in session three, an agent without
semantic memory starts every session from zero. With it, the agent builds a
growing profile, stops asking the same questions, and anticipates needs.
He asks whether Instagram and TikTok knowing what you like is semantic
memory in action.

<Infographic
  src="/img/ai-security/m3-semantic.svg"
  alt="Semantic memory: a fact extractor feeds a dedup and conflict check that inserts, merges or resolves facts into a knowledge base; episodic versus semantic."
  caption="Redrawn from the mentor's architecture notes, 4:05, and the notebook's comparison table."
/>

The difference from episodic memory is the question "when was it true?".

| Type | What it stores | When was it true? | Like |
| --- | --- | --- | --- |
| Episodic | "On June 12, Chiru was anxious about markets and chose debt funds" | On a specific date | A diary entry |
| **Semantic** | "Chiru consistently panics during market volatility and needs reassurance before deciding" | Always, independent of when learnt | An encyclopedia entry |

Episodic memory records what happened; semantic memory distils what it
**means**: general, reusable facts and behavioural patterns, with the "when"
and "where" stripped out. Repeated mentions raise a fact's confidence, and
the retrieved facts go into the system prompt as bullets.

### Doubts · Isn't episodic memory hard to scale? Does semantic memory update? · 5:17

**Rishabh:** Handling episodic memory at scale is a challenge, isn't it?

**Chirantan:** Absolutely, and that's why enterprise systems never use a
standalone memory layer. Tokens and episodes compound as you scale. And yes,
semantic memory updates facts when they change.

### Complete file: `t09_semantic.py`

*Not from the session.* Facts are distilled from episode summaries, then
checked against the nearest existing fact: duplicates raise confidence,
contradictions replace the old fact.

```python
import time
import uuid

from memory_common import chat, chat_json, get_collection

facts = get_collection("fincoach_facts")

FACTS_PROMPT = """From these session summaries, extract durable facts and behaviour
patterns about the user. Drop dates and one-off details; keep what stays true.
Return JSON: {"facts": ["...", "..."]}"""

JUDGE_PROMPT = """Compare a NEW fact with an EXISTING fact about the same user.
Reply with exactly one word: DUPLICATE if they say the same thing,
CONTRADICTION if they cannot both be true, or DIFFERENT otherwise."""

def learn_facts(user_id: str, episode_summaries: list[str]) -> None:
    extracted = chat_json(FACTS_PROMPT, "\n".join(episode_summaries))["facts"]
    for fact in extracted:
        nearest = (facts.query(query_texts=[fact], n_results=1, where={"user_id": user_id})
                   if facts.count() else None)
        if nearest and nearest["ids"][0]:
            old_id = nearest["ids"][0][0]
            old_fact = nearest["documents"][0][0]
            old_meta = nearest["metadatas"][0][0]
            verdict = chat([{"role": "system", "content": JUDGE_PROMPT},
                            {"role": "user", "content": f"NEW: {fact}\nEXISTING: {old_fact}"}],
                           temperature=0).strip().upper()
            if verdict.startswith("DUPLICATE"):  # merge: bump confidence
                facts.update(ids=[old_id], metadatas=[
                    {**old_meta, "confidence": old_meta["confidence"] + 1, "ts": time.time()}])
                continue
            if verdict.startswith("CONTRADICTION"):  # resolve: prefer the recent fact
                facts.delete(ids=[old_id])
        facts.add(ids=[str(uuid.uuid4())], documents=[fact],
                  metadatas=[{"user_id": user_id, "confidence": 1, "ts": time.time()}])

def known_facts(user_id: str, query: str, k: int = 5) -> str:
    """Top-k facts as bullets, ready to put inside the system prompt."""
    if facts.count() == 0:
        return ""
    result = facts.query(query_texts=[query], n_results=k, where={"user_id": user_id})
    return "\n".join(f"- {doc}" for doc in result["documents"][0])
```

## 10. Procedural memory

Procedural memory changes **how the agent itself behaves** (5:20 to 5:25).
The agent learns from what happens and updates its own system instructions,
which is where its core behaviour lives. Human procedural memory is why you
can ride a bicycle without thinking: you don't recall the session where you
learnt to balance, you just know how.

<Infographic
  src="/img/ai-security/m3-procedural.svg"
  alt="Procedural memory: successful runs become parameterised workflow templates in a skill library, retrieved and adapted for new tasks; semantic versus procedural."
  caption="Redrawn from the mentor's architecture notes, 4:05 (again at 4:11)."
/>

A **skill library** of step-by-step workflows lets the agent retrieve a
proven procedure instead of reasoning from scratch.

| | Semantic (9) | Procedural (10) |
| --- | --- | --- |
| Stores | Facts about the user | Rules for how the agent should act |
| Example | "Chiru is risk averse" | "Always quantify the worst case before the user asks" |
| Subject | The user | The agent's own behaviour |
| Updates when | The user's behaviour changes | A workflow succeeds or fails |
| Source | Distilled from the user's episodes | Learned from outcome feedback |

Semantic memory is knowledge *about* the user; procedural memory is
knowledge *for* the agent. It takes three forms: **workflows** ("for an FD
maturity: check risk profile, present three options, quantify the risk of
each"), **rules** ("if the user shows anxiety, reassure before
recommending"), and **interaction patterns** ("this user prefers numbered
lists").

The key production insight is **where it lives**. Episodic memory goes in as
a context block and semantic memory as user facts, but procedural memory
extends the **system prompt** itself, so the model reads it as directives,
not background. Procedures are learned from outcomes, not just stated,
which makes procedural memory the bridge to **self-improving agents**: as
the agent works in its ReAct loop, its instructions improve alongside. The
notebook cites Voyager and Agent Workflow Memory as the landmark papers.

| Strength | Weakness |
| --- | --- |
| Improves with experience, no retraining | Bad procedures, once reinforced, cause systematic errors |
| Reuses proven paths, less reasoning | Needs an outcome feedback signal |
| Personalises *how* it advises | Procedures can conflict: which one wins? |
| Extends the system prompt with learned knowledge | Too many procedures bloat the prompt |
| Enables self-improvement | Hard to audit why the agent behaved as it did |

Bad or conflicting procedures are forms of what context engineering calls
**context poisoning** and **context clash**. Guardrails around what may
become a rule are worth building. He argues any serious coding assistant
must have procedural memory: it should be learning from your conversations
and updating how it works.

### Complete file: `t10_procedural.py`

*Not from the session.* Feedback becomes one imperative rule, stored and
appended to the system prompt; a cap keeps the prompt from bloating.

```python
import json
from pathlib import Path

from memory_common import SYSTEM_PROMPT, chat, transcript

RULE_PROMPT = """You improve an advisor agent's standing instructions.
From the conversation and the user's feedback, write ONE short, general rule
the agent must follow from now on, as an imperative sentence.
If no new rule is needed, reply NONE."""

class ProceduralMemory:
    """Learned rules live in the system prompt and are read as instructions."""

    def __init__(self, path: str = "procedures.json", max_rules: int = 15):
        self.path = Path(path)
        self.rules: list[str] = json.loads(self.path.read_text()) if self.path.exists() else []
        self.max_rules = max_rules  # too many rules bloat the prompt

    def system_prompt(self) -> str:
        if not self.rules:
            return SYSTEM_PROMPT
        return SYSTEM_PROMPT + "\n\nRules you have learned. Follow them:\n" + "\n".join(
            f"{i}. {rule}" for i, rule in enumerate(self.rules, start=1))

    def learn(self, messages: list[dict], feedback: str) -> str | None:
        rule = chat([{"role": "system", "content": RULE_PROMPT},
                     {"role": "user", "content": f"{transcript(messages)}\n\nFeedback: {feedback}"}],
                    temperature=0).strip()
        if rule.upper() == "NONE" or rule in self.rules:
            return None
        self.rules = (self.rules + [rule])[-self.max_rules:]
        self.path.write_text(json.dumps(self.rules, indent=2))
        return rule
```

:::warning Review learned rules before they go live
*Not from the session.* A single bad piece of feedback becomes a standing
instruction for every future session. In MOSAIC a human approves the
rejection that creates a rule; do the same, or at least log every new rule
with the conversation that produced it.
:::

## 11. Self-reflection memory

Self-reflection has the agent **analyse its own performance** after a task,
extract the lessons and store them for next time (5:25 to 5:33). The class
suggests trading bots, code generation and planning as uses, and he agrees.

His analogy is a thoughtful doctor. After a difficult consultation they
don't just call the next patient; they spend five minutes asking: did I ask
the right questions, miss a symptom, communicate clearly? The notes go in a
personal journal, read before the next similar case. Practice improves
through self-critique, not formal training, as it does for people.

<Infographic
  src="/img/ai-security/m3-self-reflection.svg"
  alt="Self-reflection memory: attempt the task, evaluate the outcome, extract a short insight into a reflection store, and retrieve it for the next task of the same type."
  caption="Redrawn from the mentor's architecture notes, 4:05."
/>

The research foundation is
[Reflexion](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023):
"language agents with verbal reinforcement learning". Instead of updating
the network's weights, the agent reflects in words on feedback, whether the
code ran, whether the task succeeded, and keeps its reflective text in an
**episodic memory buffer** to decide better next time. So self-reflection
runs on episodic memory under the hood. ExpeL and Self-Refine follow the
same idea.

| | Procedural | Self-reflection |
| --- | --- | --- |
| Source | Patterns extracted from session content | The agent's critique of its own performance |
| Perspective | Third-party observation | First-person self-assessment |
| Content | "How to handle X" | "I should have done Y instead of Z" |
| Mechanism | Extraction from the transcript | Deliberate reflection after the session |

For FinCoach the notebook reviews five dimensions: **consistency**,
**completeness**, **constraint compliance** ("never equity"),
**communication quality**, and **missed signals** such as anxiety or
frustration. High-severity notes are injected on every call; low-severity
notes only when a similar situation comes up.

Is it token hungry? It depends on the use case, the documents and the
number of agents; no technique is simply cheap or expensive. Token
budgeting frameworks exist for exactly this.

:::note
In the Monday recap (4:05) Chirantan says coding assistants such as Claude
Code "primarily use" self-reflection memory. Treat that as his opinion. What
Claude Code documents is compaction, summary memory, and project memory
files of standing instructions, which are closer to procedural memory.
:::

### Complete file: `t11_self_reflection.py`

*Not from the session.* After answering, the agent reviews itself against
the five dimensions and stores a short insight with a severity.

```python
import json
import time
from pathlib import Path

from memory_common import SYSTEM_PROMPT, chat, chat_json

REFLECT_PROMPT = """You are FinCoach reviewing your own answer after the task.
Check five things: consistency with what the user said, completeness,
compliance with the user's constraints, clarity, and missed emotional signals.
Return JSON: {"outcome": "success" or "partial" or "fail",
"insight": "1-3 sentences you should remember next time",
"severity": "high" or "low"}"""

class ReflectionMemory:
    """After each task the agent critiques itself and stores a short, reusable insight."""

    def __init__(self, path: str = "reflections.json"):
        self.path = Path(path)
        self.notes: list[dict] = json.loads(self.path.read_text()) if self.path.exists() else []

    def relevant(self, task_type: str, limit: int = 5) -> list[str]:
        # High-severity notes ride along on every call; low ones only for the same task type.
        picked = [n for n in self.notes
                  if n["severity"] == "high" or n["task_type"] == task_type]
        return [n["insight"] for n in picked[-limit:]]

    def ask(self, task_type: str, question: str, user_context: str = "") -> str:
        lessons = self.relevant(task_type)
        system = SYSTEM_PROMPT + (f"\n\nWhat you know about the user:\n{user_context}"
                                  if user_context else "")
        if lessons:
            system += "\n\nLessons from your past reviews:\n" + "\n".join(f"- {x}" for x in lessons)
        answer = chat([{"role": "system", "content": system},
                       {"role": "user", "content": question}])
        review = chat_json(REFLECT_PROMPT,
                           f"User context: {user_context}\nQuestion: {question}\nYour answer: {answer}")
        self.notes.append({"task_type": task_type, "ts": time.time(), **review})
        self.path.write_text(json.dumps(self.notes, indent=2))
        return answer
```

## 12. Memory routing

With half a dozen stores, something has to decide which one each message
touches (5:33 to 5:40). **Memory routing** classifies the intent of every
incoming message and dispatches it to the right store, for reading, writing
or both. The notebook's image is an air-traffic controller: it doesn't send
a plane to every runway, it picks one based on type, destination, size and
traffic.

<Infographic
  src="/img/ai-security/m3-routing.svg"
  alt="Memory routing: a router classifies each message and sends it to the entity, episodic, semantic, procedural or vector store, with the notebook's routing examples."
  caption="Redrawn from the mentor's architecture notes, 4:05 to 4:06, and the notebook's routing table."
/>

| Message | Routed to |
| --- | --- |
| "What is my current salary?" | Entity store: a structured current fact |
| "What did we decide last April?" | Episodic store: a past-session query |
| "How does an SIP work?" | Vector store: general knowledge |
| "I just changed jobs to TCS" | Entity store (update) + vector store (write) |
| "Never recommend equity to me again" | Procedural store: a hard constraint |
| "I'm worried about market volatility" | Semantic store: a behavioural pattern |

Without routing, every turn queries every store: 150 tokens from the entity
store, 200 from vectors, 250 from episodes, 200 from semantic facts and 150
from reflections, **950 tokens on every turn** whatever the question. MOSAIC
needs routing for the same reason: a thousand concurrent users asking about
Pfizer, GlaxoSmithKline, or whether vitamin D3 should be taken with K2
can't all hit one memory layer.

In the demo, "What is my current monthly salary?" is classified as a fact
query with 90% confidence and answered from the entity profile, in three
context blocks and 220 tokens. "Never suggest cryptocurrency to me under
any circumstances" becomes a procedural constraint, and the agent confirms
it won't. "I changed jobs to TCS and my new salary is 150" is a **fan-out**:
a fact update that rewrites the entity profile's salary from 120 to 150 and
is also stored in the vector store.

### Doubts · Is an LLM used for routing? · 5:36

**Chirantan:** Not in the notebook: the routing is rule-based code, in a
`RoutedMemory` class. Frameworks for memory routing exist too.

### Complete file: `t12_routing.py`

*Not from the session.* A rule-based router with a confidence per rule and
a fan-out table. It runs without an API key.

```python
import re

# (route, pattern, confidence). Checked in order; several can match (fan-out).
RULES = [
    ("procedural.write", r"\b(never|always|stop)\b.*\b(suggest|recommend|mention|advise)", 0.90),
    ("entity.update",    r"\b(i (just )?(changed|switched|moved)|my new (salary|job|employer)|i got a raise)", 0.90),
    ("episodic.read",    r"\b(last (week|month|session|time|january|february|march|april|may|june|july"
                         r"|august|september|october|november|december)|did we decide|we discussed)", 0.85),
    ("entity.read",      r"\bmy (current )?(salary|income|expenses|age|employer)\b", 0.90),
    ("semantic.read",    r"\b(worried|anxious|nervous|scared|panic)", 0.80),
]
FAN_OUT = {"entity.update": ["vector.write"]}  # a changed fact is also stored as history

def route(message: str) -> list[tuple[str, float]]:
    text = message.lower()
    hits = [(name, conf) for name, pattern, conf in RULES if re.search(pattern, text)]
    if not hits:
        return [("vector.read", 0.60)]  # general knowledge: semantic search
    extra = [(target, conf) for name, conf in hits for target in FAN_OUT.get(name, [])]
    return hits + extra

if __name__ == "__main__":
    for q in [
        "What is my current salary?",
        "What did we decide last April?",
        "How does an SIP work?",
        "I just changed jobs to TCS",
        "Never recommend equity to me again",
        "I'm worried about market volatility",
        "I changed jobs to TCS and my new salary is 1,50,000",
    ]:
        print(f"{q:55} -> {route(q)}")
```

Output:

```text
What is my current salary?                              -> [('entity.read', 0.9)]
What did we decide last April?                          -> [('episodic.read', 0.85)]
How does an SIP work?                                   -> [('vector.read', 0.6)]
I just changed jobs to TCS                              -> [('entity.update', 0.9), ('vector.write', 0.9)]
Never recommend equity to me again                      -> [('procedural.write', 0.9)]
I'm worried about market volatility                     -> [('semantic.read', 0.8)]
I changed jobs to TCS and my new salary is 1,50,000     -> [('entity.update', 0.9), ('vector.write', 0.9)]
```

Rules are cheap, fast and predictable, but brittle: "What's my pay these
days?" falls through to vector search. An LLM or a small trained classifier
generalises better at the cost of a call per message; a common compromise
is rules first, a model only when no rule fires.

## 13. Forgetting and decay

The last technique is the one Chirantan is most excited about, though he
hasn't yet used it in a product (5:40 to 5:50). **More memory is not always
better.** An agent that remembers everything becomes slower to search, noisier
in its answers, and unable to tell what matters now from what mattered two
years ago. Forgetting on purpose, systematically pruning low-value, stale
and redundant memories, keeps a long-running store fast, relevant and
trustworthy. It's the same idea as pruning a decision tree, applied to
stale facts.

It rests on the **Ebbinghaus forgetting curve**: memory fades exponentially
unless it is reinforced. Without review, people lose most of what they
learnt within a day: roughly 60% is left after 20 minutes, 35% after nine
hours and 20 to 25% after a month, as the page he shows puts it.

$$
R(t) = e^{-t/S}
$$

*R* is retention after time *t*, and *S* is the memory's **stability**. The
tuning knob is the **half-life**, the time for a memory's strength to halve:
a 24-hour half-life means an unused memory loses half its strength every
day. Each memory has a half-life score; as it approaches zero, the memory
is due for eviction. Reading a memory reinforces it: a boost of 0.3 takes a
memory at 0.5 to 0.8, the idea behind spaced repetition.

<Infographic
  src="/img/ai-security/m3-forgetting.svg"
  alt="Forgetting and decay: a decay engine lowers memory strength, retrieval boosts it, a pruning engine archives or deletes memories below 0.10; the four forgetting strategies."
  caption="Redrawn from the mentor's architecture notes, 4:11, and the notebook's strategy table."
/>

The notes' caption says it well: every read is a vote to keep, silence is a
vote to forget, and storage pressure speeds the process up.

| Strategy | Rule | Pro | Con | Best for |
| --- | --- | --- | --- | --- |
| 1. TTL (time to live) | Every memory has a fixed expiry; then it's deleted or archived | Simple, deterministic, bounded storage | Rare but critical facts (allergies, hard constraints) age out on schedule | High-churn data: market news, session notes, search results |
| 2. LRU (least recently used) | When full, evict what hasn't been accessed for longest; each access resets the clock | Keeps what's actually used; how OS caches work | Rare but critical facts are pruned just for not being retrieved lately | High-frequency assistants where context shifts quickly |
| 3. Importance-weighted | Score = f(recency, access count, semantic relevance, category weight); remove the lowest | Context-aware | More complex; the function needs tuning | Long-running agents with diverse memory types |
| 4. Budget-constrained | Hard cap on store size; run importance eviction when exceeded | Bounded storage whatever the session length | Needs a budget decision at design time | Any store that must not grow without limit |

Strategy 3 draws on research into "controlled pruning rather than heuristic
removal". A related idea, **temporal memory**, keeps old memories but lowers
their retrieval score by blending similarity with recency; forgetting goes
further and removes them, so the store stays bounded.

:::note Not from the session
Half-life and stability are the same knob in two units. From
$R = e^{-t/S}$, retention halves when $t = S \ln 2$, so a 24-hour half-life
means $S \approx 34.6$ hours, and the curve can be written as
$R = 0.5^{\,t/h}$ for half-life $h$. That's the form the code below uses.
:::

### Complete file: `t13_decay.py`

*Not from the session.* No API calls: strength decays on a half-life,
access reinforces it, the four strategies choose what to drop, and a
pruning pass archives memories below 0.10. Hard constraints are pinned and
never decay, which is the fix for the "rare but critical fact" problem
every strategy shares.

```python
import time
from dataclasses import dataclass, field

HOUR = 3600.0
# How much each kind of memory matters, whatever its age.
CATEGORY_WEIGHT = {"constraint": 1.0, "profile": 0.8, "preference": 0.6, "chit_chat": 0.2}

@dataclass
class Memory:
    text: str
    category: str
    strength: float = 1.0
    half_life_h: float = 24.0  # an unused memory loses half its strength every 24 h
    created: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    pinned: bool = False  # hard constraints ("allergic to equity") never decay

    def decay(self, now: float) -> None:
        if self.pinned:
            return
        hours = (now - self.last_update) / HOUR
        # Ebbinghaus: R = e^(-t/S). With half-life h, S = h / ln 2, so R = 0.5 ** (t / h).
        self.strength *= 0.5 ** (hours / self.half_life_h)
        self.last_update = now

    def reinforce(self, now: float, boost: float = 0.3) -> None:
        # Every read is a vote to keep: a memory at 0.5 jumps to 0.8.
        self.decay(now)
        self.strength = min(1.0, self.strength + boost)
        self.last_access = now
        self.access_count += 1

def importance(m: Memory, relevance: float = 0.5) -> float:
    frequency = min(m.access_count / 10, 1.0)
    strength = 1.0 if m.pinned else m.strength
    return 0.4 * strength + 0.2 * frequency + 0.2 * relevance + 0.2 * CATEGORY_WEIGHT[m.category]

# Strategy 1: TTL. Everything older than the TTL goes, however important.
def ttl_expired(memories, now, ttl_h=72):
    return [m for m in memories if not m.pinned and (now - m.created) / HOUR > ttl_h]

# Strategy 2: LRU. When full, drop whatever was used least recently.
def lru_evict(memories, capacity):
    oldest_first = sorted((m for m in memories if not m.pinned), key=lambda m: m.last_access)
    return oldest_first[: max(0, len(memories) - capacity)]

# Strategy 3: importance-weighted eviction.
def importance_evict(memories, threshold=0.35):
    return [m for m in memories if not m.pinned and importance(m) < threshold]

# Strategy 4: budget-constrained pruning. Keep the N most important.
def budget_prune(memories, max_items):
    ranked = sorted(memories, key=lambda m: (m.pinned, importance(m)), reverse=True)
    return ranked[max_items:]

def run_pruning(memories, archive, now, threshold=0.10, capacity=None):
    """Decay engine + pruning engine. Storage pressure raises the threshold."""
    for m in memories:
        m.decay(now)
    if capacity is not None and len(memories) > capacity:
        threshold *= 2  # storage pressure: prune harder
    weak = [m for m in memories if not m.pinned and m.strength < threshold]
    for m in weak:
        memories.remove(m)
        archive.append(m)  # soft delete; drop the archive later for a hard delete
    return weak

if __name__ == "__main__":
    t0 = 1_800_000_000.0  # a fixed clock so the output is repeatable
    store = [
        Memory("User is allergic to equity instruments", "constraint", pinned=True,
               created=t0, last_update=t0, last_access=t0),
        Memory("Monthly take-home is ₹1,20,000", "profile",
               created=t0, last_update=t0, last_access=t0, access_count=6),
        Memory("Prefers answers as numbered lists", "preference",
               created=t0, last_update=t0, last_access=t0, access_count=2),
        Memory("Said the weather in Mumbai is hot", "chit_chat",
               created=t0, last_update=t0, last_access=t0),
    ]
    now = t0 + 24 * HOUR
    store[1].reinforce(now)  # the salary was used today
    now = t0 + 96 * HOUR  # four days after they were created
    for m in store:
        m.decay(now)
        print(f"{m.text:40} strength={1.0 if m.pinned else m.strength:.2f} importance={importance(m):.2f}")
    print("TTL 72 h      ->", [m.text for m in ttl_expired(store, now)])
    print("LRU, keep 3   ->", [m.text for m in lru_evict(store, 3)])
    print("importance    ->", [m.text for m in importance_evict(store)])
    print("budget, top 2 ->", [m.text for m in budget_prune(store, 2)])
    archive = []
    print("pruned < 0.10 ->", [m.text for m in run_pruning(store, archive, now)])
```

Output:

```text
User is allergic to equity instruments   strength=1.00 importance=0.70
Monthly take-home is ₹1,20,000           strength=0.10 importance=0.44
Prefers answers as numbered lists        strength=0.06 importance=0.29
Said the weather in Mumbai is hot        strength=0.06 importance=0.17
TTL 72 h      -> ['Monthly take-home is ₹1,20,000', 'Prefers answers as numbered lists', 'Said the weather in Mumbai is hot']
LRU, keep 3   -> ['Prefers answers as numbered lists']
importance    -> ['Prefers answers as numbered lists', 'Said the weather in Mumbai is hot']
budget, top 2 -> ['Prefers answers as numbered lists', 'Said the weather in Mumbai is hot']
pruned < 0.10 -> ['Prefers answers as numbered lists', 'Said the weather in Mumbai is hot']
```

Read the output against the table. TTL would delete the salary, a profile
fact still in use, just because it's old. Importance and budget pruning keep
it, because its access count and category outweigh its faded strength. The
pinned allergy survives everything. Tune the weights, the half-life and the
boost on your own data; these values are only a starting point.

### Doubts · What decides that a memory is low value? · 5:44

**Manpreet:** What counts as "low value"?

**Chirantan:** The half-life score: how strong the memory still is, given
its age and how often it's been used.

## What comes next

Chirantan closes by pointing to what's left for later sessions: context
engineering, hierarchical and graph memory with Graphiti, and temporal
memory. The course moves on to running all of this in production:
[Module 4: AgentOps & Production Deployment](/docs/projects/ai-security/agentops).

## Checklist

- [ ] I can explain why LLMs are stateless and how a buffer works around
      it, and show why its cost grows every turn.
- [ ] I can compare sliding window, summary, summary buffer and token buffer
      memory on cost, what each loses, and when each fits.
- [ ] I can explain lossy compression and why a rare, critical fact vanishes
      by the third summary, and write a summarisation prompt that protects
      it.
- [ ] I can say what changes when memory moves from RAM to a vector
      database, and name the stale fact problem.
- [ ] I can tell entity, episodic, semantic and procedural memory apart by
      what they store, when it was true, and where it enters the prompt.
- [ ] I can explain hot-path and background memory updates and the latency
      trade-off between them.
- [ ] I can describe how self-reflection builds on episodic memory, with
      Reflexion as the reference.
- [ ] I can route a message to the right memory store, and justify rules
      versus a classifier.
- [ ] I can apply the Ebbinghaus curve and a half-life to decide what an
      agent should forget, and protect hard constraints from it.
- [ ] I can design a hybrid memory: one short-term technique, the long-term
      stores it needs, and routing between them.
