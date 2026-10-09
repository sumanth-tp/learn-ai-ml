---
id: agentic-course-vectorless-rag
title: "05. Vectorless RAG with PageIndex: No Vector DB, No Chunking (Complete Agentic AI Course in 10 Hours)"
sidebar_label: "5 - Vectorless RAG"
sidebar_position: 5
slug: /projects/agentic-ai-complete-course/vectorless-rag
description: "Why vector RAG struggles on long professional documents, how PageIndex builds a table-of-contents tree that an LLM reasons over, the full notebook code, and when to choose vectorless, vector or hybrid retrieval."
tags: [agentic-ai, rag, vectorless-rag, pageindex, tree-index, openai, retrieval]
---

import Infographic from '@site/src/components/Infographic';

> **Part 5 of 9** ·
> [Watch on YouTube](https://www.youtube.com/watch?v=rV3HJ4LEZ7k) ·
> Notebook:
> `RAG-Tutorials/PageIndex_Vectorless_RAG_CrashCourse (1).ipynb`. Notes follow
> the video in order.

This chapter teaches a way to do retrieval-augmented generation without any
vector database: the document becomes a tree of sections, and the LLM walks that
tree to find the answer, the way you would use a book's table of contents.

By the end you will understand why chunk-and-embed retrieval breaks down on long,
structured documents, how PageIndex turns a PDF into a hierarchical JSON index,
how the retrieval loop works, every line of the notebook the instructor runs, and
how to choose between vectorless, vector and hybrid retrieval for a real project.

:::note What this part of the video is
The section opens mid-way through a longer lesson (the instructor has just
finished the RAG section and is introducing a bonus topic). The instructor wraps up
the notebook demo and says goodbye, then the recording carries on with a second,
separately recorded segment (he is in a different shirt) that recaps the idea and
walks through a slide deck comparing the two approaches. The chapter keeps the
video's order, so you will meet the same ideas twice: first on a whiteboard and
in code, then on slides. The second pass adds the storage question, the strengths
and weaknesses, the decision guide and the hybrid advice.
:::

## Meet PageIndex

The instructor opens by promising to show, line by line, how **vectorless RAG**
works, and says he has built some practical applications around it that are worth
following. He also announces his live cohorts (AI for Everyone, a full-stack
generative and agentic AI bootcamp, and a data science and GenAI bootcamp). That
is promotion, not course content, so the notes skip it.

The tool for this lesson is **PageIndex**, from a company called Vectify (its
homepage is `pageindex.ai`, and `vectify.ai` leads to the same place). Its
headline, as he reads it from the page, is *Human-like Document AI*: it aims to
give precise, verifiable answers over complex documents. On its GitHub page the
project describes itself as *Vectorless, Reasoning-based RAG* with four promises:
reasoning-based RAG, no vector DB, no chunking, and human-like retrieval.

Three offerings sit under that one name, and it helps to keep them apart because
the video moves between them:

| Offering | What it is | Where it shows up in this chapter |
| --- | --- | --- |
| The open-source repository | The tree-building and retrieval code you can run yourself with your own OpenAI key | Mentioned as the self-hosted option; not run on camera |
| The hosted API and Python SDK (`pageindex` package) | You upload a PDF, PageIndex builds the tree in its cloud, you fetch the JSON | The notebook demo |
| PageIndex Chat | A ready-made chat page that answers questions over uploaded PDFs | A short live demo |

:::note Open source versus hosted
He says it is "not completely open source, but you get some free requests". That
is a fair summary once you separate the pieces: the repository is public (the
README he shows carries an MIT licence badge), while the cloud API and chat
platform are a hosted service with a free allowance and paid tiers. Limits and
prices change, so check the dashboard rather than trusting a number from a video.
:::

## Recap: how traditional vector RAG works

Before showing the new idea he rebuilds the old one on a whiteboard, for anyone
who has not watched his earlier RAG videos (he has a whole playlist on them on his channel). The whiteboard page he uses is a
two-column diagram: traditional vector RAG on the left, PageIndex on the right,
with matching rows for input, indexing, storage, query time, retrieval, retrieved
content and generation. He draws over it in red and green as he talks.

**The left-hand column, step by step.** You start with one or more long PDFs.

1. **Chunking.** The text is split into pieces, because an LLM has a limited
   context window and embedding models work best on short passages.
2. **Embedding.** Each chunk is converted into a vector by an embedding model.
   He notes that you can pick whichever provider suits you, for example OpenAI or
   Google.
3. **Vector database.** The vectors are stored in a database built for similarity
   search (the board names Pinecone, FAISS and ChromaDB). Later this database is
   connected to the LLM.
4. **Query time.** When a user asks something, the question is embedded with the
   same model, and the database is searched for the stored vectors closest to it.
   This is the "similarity match" he keeps returning to.
5. **Retrieved content.** What comes back is a set of *flat text chunks*, with no
   structure around them. That text is called the context.
6. **Generation.** The chunks and the question go into a prompt, and the LLM
   writes the answer. Because the chunks carry no page or section information,
   there is normally no citation to point back to.

<Infographic
  src="/img/agentic-course/05-compare-pipelines.svg"
  alt="Two pipelines side by side: traditional vector RAG (chunk, embed, vector database, ANN search, flat chunks) and PageIndex (LLM tree builder, JSON tree index, LLM tree search, named sections)."
  caption="Redrawn from the whiteboard page, including his red marks (the query going into the vector database, the context coming out, and the cross over the vector database on the vectorless side)."
/>

His summary of the left-hand pipeline is: it is a *similarity search*, it finds
the nearest vectors, and nearest does not always mean best. Keep that last point
in mind, because the rest of the chapter is a response to it.

## Why vector RAG struggles on long documents

The instructor spreads this argument across several moments. It is easier to learn in one place, so here it is, with a worked example
for each failure. The numbers in the second example are illustrative, not
measurements.

**Failure 1: chunking cuts through meaning.** A fixed-size splitter does not know
where a section starts or ends. A single section can be cut into three pieces, and
those pieces get embedded and stored separately. At question time only the chunks
that score highest come back. If the top chunk holds the rule but the next chunk
holds the exception ("the supplier is liable for delays, as defined in Section
3.2 ... except where the delay is caused by force majeure"), the model sees half a
rule and answers confidently from it. The instructor's own phrasing is that if the
important information sits in chunks three and four but only one of them matches,
the other is missed, and the LLM never gets the full context it needed.

**Failure 2: similarity is not relevance.** An embedding places text close to
other text that is about a similar topic. It does not know whether a passage
answers *this* question. The notebook's introduction puts it concretely: a chunk
about "market conditions" can outrank the section that actually answers an
EBITDA question simply because it shares more vocabulary with the query. In the
instructor's words, cosine similarity is a similarity search, not a relevance
search: it says nothing about how one chunk relates to another or in what order
they should be read.

<Infographic
  src="/img/agentic-course/05-vector-failures.svg"
  alt="Two worked failures of vector retrieval: a contract rule split across two chunks, and a cosine ranking that puts a market-conditions chunk above the MD&A section."
  caption="Explanatory board (not shown in the video). The scores are illustrative."
/>

There are three more weaknesses that he lists on a slide and explains afterwards. They are summarised here and drawn in the slide redraw later in the chapter.

| Failure mode | What it means in practice |
| --- | --- |
| No cross-section reasoning | A question like "compare the risks with the mitigations" needs two separate sections read together. Nearest-neighbour search returns whatever is closest to the question, not a deliberate pair of sections. |
| Hard to explain | When a chunk is chosen, the only justification is a cosine score. A compliance reviewer cannot be shown why that chunk was picked. |
| Embedding drift | The vectors belong to one embedding model. If you switch to a newer or different model, every document must be embedded again. |

:::note How strong is this argument?
The failures are real, but they are not unavoidable. Production vector systems
reduce them with overlapping chunks, parent-document retrieval, rerankers, hybrid
keyword plus vector search and metadata filters. The instructor's framing is the
worst case of naive fixed-size chunking. Treat "chunking destroys context" as
"naive chunking risks destroying context", and remember that the cure is not
always to abandon vectors, as the hybrid section below shows.
:::

## Vectorless RAG: the LLM tree builder

Now he switches to the right-hand column. The first thing he stresses is the
headline benefit: **there is no vector database at all**. Start again from the
same PDF.

### The idea, with a table of contents

Imagine the PDF has a table of contents. A table of contents is an index: chapter
1 is *Introduction* on page 1, chapter 2 is *AI*, with sub-sections *2.1 Machine
learning* and *2.2 Deep learning*, each with a page number (he writes them as P1,
P2, P3 and P4). With such a list, finding anything in the document is easy: look
it up, go to the page, read the section.

An **LLM tree builder** turns that list into a tree of **nodes**:

- *Introduction* is one node.
- *AI* is another node, and inside it sit two child nodes, *ML* and *DL*.
- Further sections become further nodes, nested to match the document.

So far this is just the table of contents drawn as a tree. The important addition
is what each node *holds*. For every node, the LLM reads the pages of that section
and writes a **summary** of them. The node for *AI* (page 2) stores a summary of
the AI section, the *ML* node stores a summary of its page, and so on. Each node
is therefore a small, labelled, summarised pointer into the document.

That whole structure is then stored as a **JSON tree index**: a nested JSON
document that lists every node with its title, page and summary. JSON is simply
the format that lets software walk the tree.

### Using the tree to answer a question

At query time the roles reverse. The user's question goes to an LLM, and, this is
the key move, the LLM is also handed the **entire JSON tree index as context**.
He draws a box labelled LLM with two arrows into it: the query, and the tree.

Suppose the question is *what is deep learning?* Because the model can see every
node's title and summary, it can reason its way to the *DL* node. It picks that
node, takes its title, page number and summary, and that content becomes the
context from which the final answer is written. He describes the model as
"traversing" the tree; in the notebook you will see that this is really one
reasoning call that returns a list of node ids, not a crawl over the nodes.

<Infographic
  src="/img/agentic-course/05-tree-builder.svg"
  alt="A table of contents becomes a tree of nodes with summaries, which is stored as a JSON tree index; at question time the LLM receives the tree as context, walks to the DL node and answers."
  caption="Redrawn from the whiteboard scribble (the TOC list, the AI/ML/DL tree, the JSON tree index and the LLM box)."
/>

Two things follow from this design, and he points out both:

- **No vector database setup.** For any number of documents you build a JSON tree
  index per document, and hand the relevant one to the LLM as context. There are
  no embeddings to compute, no index to host and nothing to re-embed when models
  change.
- **Reasoning-based retrieval.** The LLM's choice of section comes from reading
  titles and summaries, not from a numeric distance, so the system can explain
  itself.

### The human-expert analogy

His analogy is a book. If you are given a book and need one topic, you do not
read it from the front or compare every paragraph to your question. You open the
table of contents, find the chapter and page, and read that part. The LLM does the
same: it reads the tree like a table of contents, decides which sections matter,
and reads them. What the LLM has on top of a human is a ready-made summary on
every node. This is why he describes the approach as retrieval that "navigates the
document like a human expert".

## What if the PDF has no table of contents?

Many real PDFs have no table of contents. The instructor's second whiteboard page
shows what PageIndex does then. Read it as a flowchart.

1. **Start with a raw PDF**, any long structured document.
2. **TOC detection.** PageIndex scans the first N pages (the open-source tool's
   option is called `--toc-check-pages`, default 20) looking for an existing table
   of contents or headings.
3. **Two branches.** If a TOC exists, the tool *parses it* to extract the chapter
   structure. If not, **the LLM reads the pages itself and infers the headings
   and structure**. The notebook's PDF is a good example: it has a list of
   modules at the front but no page numbers beside them.
4. **Section-aware splitting.** The document is divided at logical boundaries,
   *not* at a token count. This is the heart of the contrast with vector RAG.
5. **Summarise each section.** The LLM produces, for each section, a node id, a
   title, a page and a summary.
6. **Assemble the hierarchical tree.** Parent, child and grandchild nodes are
   linked together.

The output looks like the small JSON tree at the bottom of the board. It is an
in-context tree for a financial-stability report: a parent node *Financial
stability* (node id 0006, page 21) with two children, *Monitoring vulnerabilities*
(node id 0007, pages 22 to 28) and *International cooperation* (node id 0008,
pages 28 to 31). Each of those children carries a summary of its own pages. The
page ranges show that sections have natural, uneven lengths, which a fixed token
splitter would never respect.

<Infographic
  src="/img/agentic-course/05-toc-flow.svg"
  alt="Flowchart: raw PDF, TOC detection, parse existing TOC or LLM reads pages, section-aware splitting, LLM summarises each section, assemble hierarchical tree, with a financial-stability node and two child nodes as output."
  caption="Redrawn from the slide. The orange note on the right is his own margin scribble about even splitting."
/>

### Evenly split versus split by section

While walking down this chart he draws a second scribble, in orange, to make the
difference with chunking visible. Under chunking, the document is cut into evenly
sized pieces, so one section can be spread across three chunks. Under
section-aware splitting, each box is a whole section: one section about DL, one
about ML, one about AI, one about something else. His argument is that the LLM
gets proper context only when the section is whole. If just one fragment of a
section were handed over and the rest withheld, the model could not produce the
answer. So the rule to remember, which he repeats and underlines, is: **respect
logical boundaries, not token counts.**

<Infographic
  src="/img/agentic-course/05-chunk-vs-section.svg"
  alt="The same three-section document cut at fixed token intervals, tearing the AI section across four chunks, versus split at section boundaries into three whole nodes."
  caption="Redrawn from the orange evenly-split scribble, with a worked example added."
/>

:::note Sections are whole, within limits
"No chunking" is a simplification. PageIndex still has to cap how large a node can
get: the self-hosted runner in the notebook has a `--max-pages-per-node` option
(default 10), so a very long chapter is split into child nodes rather than kept as
one giant block. The difference from vector RAG is *where* the cuts fall (at
headings and logical boundaries) and *what is kept* (title, page and a summary
around every piece), not that cutting never happens.
:::

## The retrieval process

The third board is the retrieval loop. It is the most useful picture in the
section, because it shows that retrieval here is an *iterative reasoning process*,
not a single lookup.

| Step | What happens |
| --- | --- |
| Query | The user asks a question. |
| 1. Read the tree index | The LLM scans titles, page numbers and summaries held in context. |
| 2. Reason and select node | The LLM returns its **thinking** plus a **node list** as JSON. |
| 3. Extract section content | The raw pages for the selected node ids are fetched. |
| Sufficient to answer? | The LLM judges whether what it has read is complete enough. |
| If no | Loop back to step 2 and choose more nodes. |
| If yes | Step 4: generate the answer, cited by section title and page. |

The board also shows a dashed side branch for **cross-reference following**: if
the text says something like "see Appendix G", the LLM can navigate the tree to
that node. This is something a similarity search cannot do, because the reference
is an instruction to go somewhere, not a phrase that resembles the question.

<Infographic
  src="/img/agentic-course/05-retrieval-loop.svg"
  alt="Retrieval loop: user query, read tree index, reason and select node, extract section content, sufficiency check that loops back or proceeds to generate a cited answer, with a dashed cross-reference branch."
  caption="Redrawn from the slide, titled Retrieval Process."
/>

He closes the idea with the human-expert point once more: reasoning-based
retrieval follows how a person navigates a book, and the answer comes back with
**section-level citations** because every piece of context is a named section with
a page number. The notebook you are about to read implements steps 1 to 3 and 4
but not the "loop back" arrow: it makes a single selection pass. Treat the loop as
what the full product can do, and a good extension if you build this yourself.

## PageIndex Chat: a live demo

Before the code, he opens `chat.pageindex.ai` to show what the technology looks
like as a product. He selects an uploaded PDF (a long pattern-recognition
textbook), asks it to summarise the book, and then asks *what are the challenges
in pattern recognition?*

What the screen shows is the point of the demo:

1. A short "thought for a few seconds" step, where the agent decides what to look
   at.
2. A tool call that **gets the document structure**: the node-by-node JSON tree,
   created and returned very quickly.
3. A second tool call that **gets page content**: it passes the document name and
   a set of page ranges, and receives the raw page text for just those pages.
4. A streamed answer built from those pages.

He is impressed by the speed and by the absence of any vector database. Notice
that the two tool calls are the same two operations the notebook performs as
`get_tree` and "fetch the nodes' text". The chat page is the whole idea wrapped in
a UI.

<Infographic
  src="/img/agentic-course/05-pageindex-chat.svg"
  alt="PageIndex Chat flow: the user's question, a get-document-structure tool call, a get-page-content tool call with page ranges, raw page text, and a streamed answer."
  caption="Redrawn from the PageIndex Chat screen. The exact page ranges in the tool call are too small to read in the 360p video, so they are described rather than copied."
/>

He adds a disclaimer: this is not a sponsored video. You can use the PageIndex
APIs, or, if you are comfortable in Python, build the same thing yourself with a
coding assistant such as Claude once you understand the concept. His advice is to
understand the concepts first and then try it. The notebook is exactly that: a
small amount of Python on top of the hosted tree builder.

## The crash-course notebook

He then opens a notebook he prepared, titled **PageIndex Vectorless RAG Crash
Course**, in his editor (a notebook in a `.venv` with Python 3.13.2). Its opening
cell lists seven things you will learn:

1. Why vector RAG fails on professional documents.
2. How PageIndex builds a tree index from a PDF.
3. LLM tree search: reasoning over structure.
4. The full end-to-end vectorless RAG pipeline.
5. Expert-guided retrieval (injecting domain knowledge).
6. A chat API with zero LLM setup.
7. A self-hosted open-source option.

The key-concept box repeats the contrast in one line each. Traditional RAG is
**chunk, embed, cosine similarity, retrieve**. PageIndex RAG is **build a tree,
let the LLM reason over the tree, retrieve the exact sections**. The best part, he
says, is that every node already carries a correct summary of its section, which
is plenty of context for the LLM. On camera he works through items 1 to 4 and then
the multi-query test. Items 5 to 7 are in the notebook but he does not run them;
they are covered after the demo, clearly marked.

The whole code flow, in one picture, before the cells:

<Infographic
  src="/img/agentic-course/05-code-flow.svg"
  alt="Seven steps: set up clients, upload the PDF, poll until processed, fetch the tree (once per document); then llm_tree_search, find_nodes_by_ids and generate_answer (per question)."
  caption="Explanatory board (not shown in the video). Steps 1 to 4 run once per document; steps 5 to 7 run for every question."
/>

:::danger Do not copy a key from the screen
In the cell that loads keys, the notebook he shares (and the one on screen) has a
PageIndex key typed straight into the code. He says it will be deleted after the
video. Never do this: a key committed in a notebook is a leaked key. The code
below reads it from the environment instead, which is the one deliberate change to
the cell. If you ever paste a real key into a notebook, revoke it.
:::

### Section 1: install and set up

**Get the two API keys.** The PageIndex key comes from the PageIndex dashboard
(the notebook links `https://dash.pageindex.ai/api-keys`): sign in, open API keys
and create a secret key. He says the free tier covers about a thousand documents,
which is more than enough for learning, though you should confirm current limits.
The OpenAI key comes from the OpenAI platform. If you would rather use a different
provider (he mentions Groq), that is fine, but you would then have to change the
two LLM calls later in the notebook.

**Install the packages.** Three are needed: `pageindex` (the SDK that talks to the
hosted tree builder), `openai` (the LLM used for reasoning and answering) and
`python-dotenv` (to load keys from a `.env` file).

```python
# Install required packages
!pip install -U pageindex openai python-dotenv
```

The `!` is notebook syntax for running a shell command; in a terminal drop it.
`-U` upgrades the packages if they are already installed. The notebook does not
pin versions, so you get the latest release, and the SDK's method names could
change over time.

**The `.env` file.** The next cell in the notebook is commented out. It is a
convenience that writes a `.env` file with two lines, which is how the environment
variables get created. You can run that cell once, or simply create the file by
hand next to the notebook:

```text
PAGEINDEX_API_KEY=your_pageindex_key_here
OPENAI_API_KEY=your_openai_key_here
```

Keep `.env` out of version control (add it to `.gitignore`).

**Load the keys.** This is the notebook's next cell, with the key line changed to
read from the environment as explained above.

```python
import os, json, time
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

print("PageIndex key loaded:", "OK" if PAGEINDEX_API_KEY else "MISSING!")
print("OpenAI key loaded:   ", "OK" if OPENAI_API_KEY    else "MISSING!")
```

Line by line: the imports bring in `os` (environment access), `json` (used later
to print and send the tree) and `time` (used when waiting for the tree to build).
`load_dotenv()` reads the `.env` file and puts its entries into the process
environment. Then `os.getenv` fetches each key, and the two `print` lines confirm
that both are present, which saves a confusing authentication failure later.

Output:

```text
PageIndex key loaded: OK
OpenAI key loaded:    OK
```

**Create the two clients.**

```python
from pageindex import PageIndexClient
from openai import OpenAI

pi_client     = PageIndexClient(api_key=PAGEINDEX_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

print("PageIndex client ready")
print("OpenAI client ready")
```

`PageIndexClient` is your handle to the hosted service: it builds and stores trees.
`OpenAI` is the LLM client used in your own retrieval and answer code. Keeping the
two separate is a useful mental model. PageIndex does the *indexing*, once per
document. Your OpenAI client does the *reasoning*, once per question.

Output:

```text
PageIndex client ready
OpenAI client ready
```

### Section 2: upload and index a PDF

The instructor's sample document is an advanced course syllabus: the "Advanced Route
of Learning AI" programme his academy is about to launch for working professionals who
want to upskill to enterprise level (21 modules, 38 sections and 481 topics). He opens the
PDF to show it: it has a table of contents but **no page numbers beside the
entries**, the sort of document that exercises the "no TOC" branch from earlier.
He saves it next to the notebook as `sample_document.pdf`, and suggests that you
try a PDF with more text if you have one.

```python
# ── Upload your PDF ─────────────────────────────────────────────────────────
# Replace with the path to your PDF file
# Great candidates: Annual reports, research papers, legal docs, textbooks

PDF_PATH = "./sample_document.pdf"   # ← change this

print(f"Uploading: {PDF_PATH}")
result = pi_client.submit_document(PDF_PATH)
doc_id = result["doc_id"]

print(f"Uploaded!")
print(f"Document ID: {doc_id}")
print("   (Save this ID — you'll use it throughout the notebook)")
```

How it works: `PDF_PATH` points at the file. `pi_client.submit_document(PDF_PATH)`
uploads it to PageIndex and starts the tree build, returning a dictionary. The
`doc_id` inside it is the identifier of your document in the PageIndex cloud, and
every later call needs it, which is why the notebook asks you to save it. A fresh
run gives you a different id.

Output (your id will differ):

```text
Uploading: ./sample_document.pdf
Uploaded!
Document ID: pi-cmnj5e1b801f801qpozge1pvx
   (Save this ID - you'll use it throughout the notebook)
```

The tree is built **asynchronously**: the upload call returns straight away while
PageIndex reads the pages in the background. The notebook's comment says a
50-page PDF takes 30 to 90 seconds. So the next cell *polls*:

```python
# ── Poll until processing is complete ───────────────────────────────────────
# PageIndex builds the tree asynchronously.
# For a 50-page PDF this typically takes 30–90 seconds.

print("⏳ Building tree index...")
print("   (This runs once per document — the index is cached for reuse)")

while True:
    status_result = pi_client.get_document(doc_id)
    status = status_result.get("status")
    print(f"   Status: {status}")
    
    if status == "completed":
        print("\nTree index ready!")
        break
    elif status == "failed":
        print("\nProcessing failed. Check your PDF format.")
        break
    
    time.sleep(5)
```

The loop asks `get_document` for the document's status every five seconds. When
the status is `"completed"` the tree is ready and the loop stops. If it is
`"failed"`, the PDF could not be processed and the loop stops with a message.
Anything else (for example "processing") means keep waiting. The note that the
index is cached matters: the build happens once per document, and later
questions reuse it.

Output, as it appeared on camera (the run took about 14 seconds in the video):

```text
Building tree index...
   (This runs once per document - the index is cached for reuse)
   Status: completed

Tree index ready!
```

### Section 3: inspect the tree

Before using the tree, the notebook explains what is in it, with a small example
tree and a list of the fields on every node:

```text
Document
├── Introduction (pages 1-3)
│   └── Background (pages 1-2)
├── Financial Stability (pages 21-31)
│   ├── Monitoring Vulnerabilities (pages 22-28)
│   └── International Cooperation (pages 28-31)
└── Conclusion (pages 45-47)
```

Each node has a `node_id` (the handle used in retrieval), a `title`, a
`page_index` (its page in the PDF), some text, and `nodes` (its child sections,
nested). This is the structure the LLM will reason over.

```python
# ── Fetch the full tree ─────────────────────────────────────────────────────
tree_result  = pi_client.get_tree(doc_id, node_summary=True)
pageindex_tree = tree_result.get("result", [])

print(f"Top-level sections: {len(pageindex_tree)}")
print("\nRaw tree (first node):")
print(json.dumps(pageindex_tree[0] if pageindex_tree else {}, indent=2))
```

`get_tree(doc_id, node_summary=True)` downloads the finished tree and asks for
summaries to be included. The SDK wraps the payload in a dictionary, so
`tree_result.get("result", [])` unwraps it into `pageindex_tree`, a Python list of
top-level nodes. The prints then show how many top-level sections there are and the
first node as indented JSON.

Output (the long strings are trimmed here):

```text
Top-level sections: 24

Raw tree (first node):
{
  "title": "Preface",
  "node_id": "0000",
  "page_index": 1,
  "summary": "This document outlines the comprehensive syllabus for the 'Advanced Route of Learning AI' course offered by Krish Naik Academy for the 2025-2026 cohort. The curriculum spans 21 modules ...",
  "text": "ADVANCED ROUTE OF\nLEARNING AI\nComprehensive Syllabus\nFrom Neural Networks &amp; Transformers through Scaling Laws, MoE, ..."
}
```

Read that output carefully, because it matters for the next section. Each node has
**both** a `summary` (written by the LLM) and a `text` field (the section's own
text). The instructor highlights both on screen.

:::warning The notebook's own description of `text` is wrong
The explanatory cell above the code says `text` is the "section summary". The real
output shows the opposite: `summary` holds the LLM summary and `text` holds the
extracted section text. This matters because two of the functions below read
`text`. Keep it in mind when you reach the compress step in `llm_tree_search`.
:::

Next, a helper that prints the whole structure as an outline, which is how you
check that the tree matches the document:

```python
# ── Pretty-print the full tree ───────────────────────────────────────────────
def print_tree(nodes, indent=0):
    """Recursively print tree titles for a visual overview."""
    for node in nodes:
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        page   = node.get("page_index", "?")
        print(f"{prefix}[{node['node_id']}] {node['title']}  (p.{page})")
        if node.get("nodes"):
            print_tree(node["nodes"], indent + 1)

print("Full Document Structure:\n")
print_tree(pageindex_tree)
```

`print_tree` is **recursive**: it prints one node, then, if the node has children
under `nodes`, calls itself on them with a deeper `indent`. Children get a
`└─` marker and two spaces of indentation per level. Each line shows the node id,
the title and the page.

Output (all 40 nodes are in the real output; the top-level run and the one big
parent are shown in full):

```text
[0000] Preface  (p.1)
[0001] MODULE 1  (p.4)
[0002] Neural Network Refresher  (p.4)
[0003] Hardware  (p.5)
[0004] Transformers 101  (p.6)
[0005] Tokenization Deep Dive  (p.7)
[0006] Finetuning Transformer Architectures  (p.8)
[0007] KV Cache &amp; Attention Variants &amp; Positional Encodings  (p.9)
[0008] Scaling Laws  (p.10)
[0009] Mixture of Experts  (p.11)
[0010] Modern LLMs Finetuning  (p.12)
  └─ [0011] The LLM Development Lifecycle  (p.12)
  └─ [0012] Pre-Training Deep Dive  (p.12)
  └─ [0013] Data Preparation for Fine-Tuning  (p.12)
  └─ [0014] Parameter-Efficient Fine-Tuning (PEFT)  (p.13)
  └─ [0015] Supervised Fine-Tuning (SFT)  (p.13)
  └─ [0016] Preference Alignment  (p.13)
  └─ [0017] The Modern Post-Training Stack  (p.14)
  └─ [0018] Evaluation  (p.14)
  └─ [0019] Quantization &amp; Deployment Prep  (p.14)
  └─ [0020] Tooling &amp; Frameworks  (p.14)
  └─ [0021] Synthetic Dataset Generation  (p.15)
  └─ [0022] Reasoning Models  (p.15)
[0023] SLM  (p.16)
[0024] Knowledge Distillation  (p.17)
...
[0036] Agents  (p.30)
  └─ [0037] MODULE 21  (p.32)
[0038] RL  (p.32)
[0039] Programme Summary  (p.34)
```

The picture below shows the same tree and the anatomy of one node. Two small
oddities are worth noticing and are normal: titles contain `&amp;` because the
text was HTML-escaped (you can run `html.unescape` on titles if you display them),
and a few nodes are titled "MODULE 13", "MODULE 16" and so on, probably because the
syllabus PDF puts those module labels on their own lines and the tree builder
treated them as sections. The last node is on page 34, so this PDF is shorter than the
"48 to 45 pages" he guesses aloud.

<Infographic
  src="/img/agentic-course/05-tree-anatomy.svg"
  alt="An excerpt of the syllabus PDF's tree with Modern LLMs Finetuning expanded into twelve child nodes, and the fields of one node explained."
  caption="Explanatory board (not shown in the video), built from the notebook's printed tree."
/>

To finish the inspection, a counter:

```python
# ── Count total nodes ────────────────────────────────────────────────────────
def count_nodes(nodes):
    total = len(nodes)
    for n in nodes:
        if n.get("nodes"):
            total += count_nodes(n["nodes"])
    return total

total = count_nodes(pageindex_tree)
print(f"Total nodes in tree: {total}")
print("   Each node = one retrievable section of the document")
```

Another small recursion: `len(nodes)` counts this level, then each child list is
counted the same way and added on. There are 24 top-level sections but the whole
tree has 40 nodes, because the children are nodes too.

Output:

```text
Total nodes in tree: 40
   Each node = one retrievable section of the document
```

Each node is one retrievable unit. He then says, in effect, that he is not here to
teach Python: read the code and you will follow it, since it is plain recursion.

### Section 4: LLM tree search, the core of PageIndex

The next cell of text states the contrast the whole lesson turns on. Vector RAG
retrieval is `query -> embed -> cosine similarity against all chunk vectors ->
top-k chunks`, and its weakness is that it finds what is similar rather than what
is relevant. PageIndex retrieval is `query + tree -> LLM reasons -> "node 0007 and
0008 contain the answer"`, and its advantage is that the model understands
structure, context and intent. The picture is a human expert scanning a table of
contents.

```python
# ── LLM Tree Search Function ─────────────────────────────────────────────────

def llm_tree_search(query: str, tree: list, model: str = "gpt-4o") -> dict:
    """
    Core PageIndex retrieval:
    Sends the query + document tree to an LLM.
    LLM reasons over the structure and returns relevant node_ids.
    
    Returns: dict with 'thinking' (reasoning) and 'node_list' (node IDs)
    """
    
    # Compress tree to save tokens — only send titles + short summaries
    def compress(nodes):
        out = []
        for n in nodes:
            entry = {
                "node_id": n["node_id"],
                "title":   n["title"],
                "page":    n.get("page_index", "?"),
                "summary": n.get("text", "")[:150]  # first 150 chars
            }
            if n.get("nodes"):
                entry["children"] = compress(n["nodes"])
            out.append(entry)
        return out
    
    compressed_tree = compress(tree)
    
    prompt = f"""You are given a query and a document's tree structure (like a Table of Contents).
Your task: identify which node IDs most likely contain the answer to the query.
Think step-by-step about which sections are relevant.

Query: {query}

Document Tree:
{json.dumps(compressed_tree, indent=2)}

Reply ONLY in this exact JSON format:
{{
  "thinking": "<your step-by-step reasoning>",
  "node_list": ["node_id1", "node_id2"]
}}"""

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)
```

What this function does, in order:

1. **Signature.** `llm_tree_search(query, tree, model="gpt-4o")` takes the question
   and the tree and returns a dictionary. The default model is `gpt-4o`; swap in
   whichever current chat model you prefer that supports JSON mode.
2. **`compress`.** Sending the entire tree with every node's full text would be
   huge and expensive, so `compress` builds a slim copy: for each node only its id,
   title, page and a short summary field, recursing into `nodes` and storing the
   children under `children`. This is what shrinks the tree to fit comfortably in
   the prompt.
3. **The prompt.** It tells the model it is looking at a document's tree structure
   like a table of contents, asks it to identify which node ids most likely contain
   the answer, and to "think step by step". It then pastes the question and the
   compressed tree as JSON, and demands a reply in a fixed JSON shape with two
   keys: `thinking` (the reasoning) and `node_list` (the chosen ids). The doubled
   curly braces are how an f-string prints a literal brace.
4. **The call.** `chat.completions.create` sends the prompt. The setting
   `response_format={"type": "json_object"}` switches on JSON mode, which forces the
   model to return valid JSON (JSON mode requires the word "JSON" in the prompt,
   which it has).
5. **The return.** `json.loads` converts the reply into a Python dictionary.

This is the "reason and select node" step from the retrieval board, implemented as
one call.

:::warning The compress step reads `text`, not `summary`
Look at the line that builds `"summary"`. It takes `n.get("text", "")[:150]`, the
first 150 characters of the section's raw text, even though the node also carries
a proper LLM-written `summary`. For the syllabus PDF this works because titles and
opening lines are descriptive, and the video's results are good. But the whole
design promise is that the model reasons over *summaries*. If you build on this,
change that line to prefer the summary, for example
`n.get("summary") or n.get("text", "")`, and consider a longer cut-off than 150
characters. This is a suggested improvement, not something shown in the video.
:::

Now he tests it with a question:

```python
# ── Test with a sample query ─────────────────────────────────────────────────
query = "What is the syllabus covered in Modern LLM finetuning?"

print(f"Query: {query}\n")
result = llm_tree_search(query, pageindex_tree)

print("LLM Reasoning:")
print(result.get("thinking", "N/A"))
print()
print("Selected Node IDs:", result.get("node_list", []))
```

The query is stored in `query`, the function runs, and the three prints show the
model's reasoning, then the node ids it chose.

Output on camera (reasoning trimmed; the exact list varies between runs, because the
LLM's choice is not deterministic):

```text
Query: What is the syllabus covered in Modern LLM finetuning?

LLM Reasoning:
To find nodes relevant to the query about the syllabus for Modern LLM finetuning, I first identify the section titles that directly mention 'Finetuning' and 'LLMs'. The node labeled 'Modern LLMs Finetuning' (node_id: 0010) appears to be highly relevant ... These sections collectively comprise the syllabus for modern LLM finetuning ...

Selected Node IDs: ['0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020', '0021']
```

He compares the ids with the printed tree and shows they line up: `0010` is
*Modern LLMs Finetuning* and `0011` onward are its children. He had asked about
modern LLM fine-tuning, and the model returned exactly that branch. The
instructor's point, which he stresses, is that he set up no vector database and no
embeddings. Also notice the model's own explanation: this is the "thinking" that
makes the retrieval explainable, which a cosine score cannot be.

### Section 5: the full end-to-end pipeline

Choosing nodes is only half the job. The model still has to be given the *content*
of those nodes and asked to answer. The notebook splits the rest into three steps:
tree search (done), retrieve the section content, and generate a grounded answer
with page citations.

**Step A: turn ids back into nodes.**

```python
# ── Helper: Find nodes by ID ─────────────────────────────────────────────────

def find_nodes_by_ids(tree: list, target_ids: list) -> list:
    """Recursively walk the tree and collect nodes matching target_ids."""
    found = []
    for node in tree:
        if node["node_id"] in target_ids:
            found.append(node)
        if node.get("nodes"):
            found.extend(find_nodes_by_ids(node["nodes"], target_ids))
    return found
```

This helper takes the original tree and the list of ids the model returned and
collects the matching node dictionaries. It walks every node, adds it to `found` if
its `node_id` is in `target_ids`, and calls itself on the children (using
`extend` to merge the children's matches into the list). A node can match
whether it is a parent or a child, which is why choosing `0010` and also its
children gives you both.

**Step B: generate the cited answer.**

```python
# ── Generate answer from retrieved nodes ─────────────────────────────────────

def generate_answer(query: str, nodes: list, model: str = "gpt-4o") -> str:
    """
    Takes retrieved nodes as context and generates a grounded answer.
    Instructs the LLM to cite section titles and page numbers.
    """
    if not nodes:
        return "No relevant sections found in the document."
    
    # Build context string from retrieved nodes
    context_parts = []
    for node in nodes:
        context_parts.append(
            f"[Section: '{node['title']}' | Page {node.get('page_index', '?')}]\n"
            f"{node.get('text', 'Content not available.')}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""You are an expert document analyst.
Answer the question using ONLY the provided context.
For every claim you make, cite the section title and page number in parentheses.
Be concise and precise.

Question: {query}

Context:
{context}

Answer:"""
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

Walking through it:

- **Guard clause.** If no nodes were found it returns a warning message instead of
  asking the LLM to invent something. This is the "no relevant section found"
  behaviour he points out.
- **Building the context.** For each node it creates a block that starts with a
  header, `[Section: 'title' | Page N]`, followed by the node's `text`, falling back
  to "Content not available." if the field is missing. The blocks are joined with a
  separator line. The header is what lets the model quote section and page.
- **The prompt.** It casts the model as an expert document analyst and gives three
  rules: answer using *only* the provided context, cite the section title and page
  number in parentheses for every claim, and be concise and precise. The question
  and the assembled context follow.
- **The call and return.** An ordinary chat completion returns the answer text.

Note that this step sends the nodes' `text`, so the LLM answers from the real
section content, not only from a summary. Navigation uses the compact tree; answering
uses the full text of the chosen sections.

**Step C: wrap it in one function.**

```python
# ── The complete Vectorless RAG function ─────────────────────────────────────

def vectorless_rag(query: str, tree: list, verbose: bool = True) -> str:
    """
    Full end-to-end PageIndex RAG pipeline:
    
    Step 1: LLM Tree Search  → finds relevant node_ids
    Step 2: Node Retrieval   → fetches section content
    Step 3: Answer Generation → produces cited answer
    """
    if verbose:
        print(f"{'='*55}")
        print(f"Query: {query}")
        print(f"{'='*55}")
    
    # Step 1: Tree Search
    search_result  = llm_tree_search(query, tree)
    node_ids       = search_result.get("node_list", [])
    
    if verbose:
        print(f"\nReasoning: {search_result.get('thinking', '')[:200]}...")
        print(f"Retrieved node IDs: {node_ids}")
    
    # Step 2: Retrieve nodes
    nodes = find_nodes_by_ids(tree, node_ids)
    
    if verbose:
        print(f"Sections found: {[n['title'] for n in nodes]}")
    
    # Step 3: Generate answer
    answer = generate_answer(query, nodes)
    
    if verbose:
        print(f"\nAnswer:\n{answer}")
    
    return answer
```

`vectorless_rag` is the whole pipeline in three labelled steps. It calls
`llm_tree_search` for the node ids, `find_nodes_by_ids` to fetch the nodes, and
`generate_answer` to write the response. With `verbose=True` it prints the first
200 characters of the reasoning, the node ids, the section titles it found, and the
final answer. It returns the answer string so you can use it in other code.

**Run it.**

```python
# ── Run the full pipeline ────────────────────────────────────────────────────
answer = vectorless_rag(
    query="What are the syllabus covered in modern llm finetuning?",
    tree=pageindex_tree
)
```

Output (trimmed):

```text
=======================================================
Query: What are the syllabus covered in modern llm finetuning?
=======================================================

Reasoning: The query asks about the syllabus covered in modern LLM fine-tuning. Looking at the document tree, the most relevant section under 'Modern LLMs Finetuning' is node_id '0010'. This node likely covers a...
Retrieved node IDs: ['0010', '0013', '0014', '0015', '0016', '0017', '0018']
Sections found: ['Modern LLMs Finetuning', 'Data Preparation for Fine-Tuning', 'Parameter-Efficient Fine-Tuning (PEFT)', 'Supervised Fine-Tuning (SFT)', 'Preference Alignment', 'The Modern Post-Training Stack', 'Evaluation']

Answer:
The syllabus covered in modern LLM fine-tuning includes:

1. **Pre-training**
2. **Data Preparation**: Topics include dataset formats, chat templates, loss masking, quality vs. quantity tradeoff, deduplication, and filtering pipelines (Section: 'Data Preparation for Fine-Tuning', Page 12).
3. **Parameter-Efficient Fine-Tuning (PEFT)**: Covers various techniques like LoRA, QLoRA, DoRA, AdaLoRA, and others (Section: 'Parameter-Efficient Fine-Tuning (PEFT)', Page 13).
4. **Supervised Fine-Tuning (SFT)**: ...
5. **Preference Alignment**: Includes methods like RLHF with PPO, DPO, ORPO, SimPO, and others (Section: 'Preference Alignment', Page 13).
6. **Evaluation**: ...
7. **Quantization**
8. **Tooling**
9. **Synthetic Data Generation** (Section: 'Modern LLMs Finetuning', Page 12).

These sections collectively form the fine-tuning stack outlined in the document.
```

Read the output as a lesson in what citations buy you. Every claim ends in a
section name and a page number, so a reader can open the PDF at that page and check
it. Two other things are visible. First, this run picked *different* ids from the
previous test (it left out `0011`, `0012` and `0019` to `0021`), which is normal
variation between LLM calls. Second, a few items in the list (quantization,
tooling) have no citation, probably because they came from the parent node's
overview. That is a reminder that "cite every claim" is an instruction the model
follows imperfectly, so spot-check answers where it matters.

:::note Why a missed child is not always a failure
Because the parent node `0010` was selected, its own `text` already outlines the
whole fine-tuning stack, so the answer stayed complete even though some children
were missing. Selecting a parent plus a handful of children is a good outcome:
broad enough for context, narrow enough to stay within the model's window.
:::

**Test more questions.** He says he prepared three more and invites you to try
your own.

```python
# ── Test with multiple queries ───────────────────────────────────────────────
test_queries = [
    "What are the syllabus covered in modern llm finetuning?",
    "What are the syllabus covered in RAG?",
    "Summarize the syllabus of Tokenization Deep Dive?",
]

for q in test_queries:
    print()
    ans = vectorless_rag(q, pageindex_tree, verbose=False)
    print(f"Q: {q}")
    print(f"A: {ans[:300]}...")
    print("-" * 55)
```

The loop runs the whole pipeline for each question with `verbose=False` (so it
prints nothing along the way) and shows the first 300 characters of each answer (he says "300 words" aloud, but the slice `ans[:300]` is characters),
with a divider line of 55 hyphens. Together the three questions covered modern LLM
fine-tuning again, the RAG module, and a different module, tokenization. On camera
the cell took about 20 seconds for all three, which is the cost of several LLM calls
per question.

Output (trimmed):

```text
Q: What are the syllabus covered in modern llm finetuning?
A: The syllabus covered in modern LLM fine-tuning includes the following key areas:

1. **Pre-training:** Focuses on preparing base models and the objectives like CLM, MLM, and Prefix-LM (Section: 'The LLM Development Lifecycle', Page 12).

2. **Data Preparation:** Involves dataset formats, chat templa...
-------------------------------------------------------

Q: What are the syllabus covered in RAG?
A: The syllabus covered in RAG includes:

- Vanilla RAG
- Embedding models
- Chunking Strategies
- BM25, SPLADE & Multi Vector COLBERT
- Hybrid RAG
- Meta Hybrid RAG
- Query Transformations
- RAG Evaluations
- Rerankers
- Self RAG
- Corrective RAG
- Adaptive RAG
- Contextual retrieval
- Agentic RAG
- V...
-------------------------------------------------------

Q: Summarize the syllabus of Tokenization Deep Dive?
A: The syllabus for "Tokenization Deep Dive" covers a comprehensive exploration of tokenization strategies, including classical methods like byte pair encoding (BPE), WordPiece, SentencePiece, and advanced approaches involving Byte Latent Transformers (BLT) that avoid traditional tokenization (Section:...
-------------------------------------------------------
```

His closing remarks on the notebook are worth taking seriously. He calls it a very
trending topic and says he does not know how many companies have adopted it yet, but
he has been suggesting to managers and architects that they try it, because far less
setup is needed. The benefit he highlights is that you no longer carry the burden of setting up and maintaining a
vector database. The caveat he raises himself is the flip side: **if the LLM tree
becomes big, where do you store it, and how do you cope?** He promises to cover
storage, and does in the second segment, below. Also think about the other limit:
the tree itself goes into every prompt, so a single huge document (or a corpus of
thousands) can outgrow the model's context window. That is one reason the slides
later say the approach suits tens to thousands of documents, not millions.

## Where do you save the JSON tree?

The video now moves into the second recording. After a short "see my earlier
video" aside (he points viewers to his previous PageIndex tutorial, titled
"Vectorless RAG Tutorial With PageIndex - No VectorDB And Chunking Required", for
the code walk-through), he rebuilds the comparison on the same board as before and
adds one clarification, because many viewers asked about it in the comments:
**where do you save the tree?**

The answer is that the JSON tree index is just JSON, so you can store it anywhere
that holds JSON or key-value data:

| Where | When it fits |
| --- | --- |
| A file on the file system | Prototypes and a handful of documents; the self-hosted runner writes `<pdf name>_pageindex.json` next to the PDF |
| An S3 (object storage) bucket | Many documents, shared access across services |
| MongoDB or another document or key-value database | When you want to query, version or update trees alongside other data |

From there you load the tree and give it to the LLM at question time. He adds a
promise to discuss, with the decision guide, how large a JSON tree can get. The
practical limit is set by your LLM's context window and by how much the compressed
tree costs to send on every question. Note also that the hosted API already
stores the tree for you under a `doc_id`, so a database is only your job when you
build and host the index yourself.

He then re-runs the pipeline in words, which doubles as a good revision of the
whole chapter: the PDF goes through TOC detection and scanning; with a TOC it
parses chapters and does section-aware splitting that respects logical boundaries,
not token counts; each section is summarised; a hierarchical tree is assembled; at
query time the LLM walks the tree and returns names, pages and summaries that
become the context for the answer. Whenever you are unsure of a step, re-read the
retrieval loop above.

## Slides: vectorless RAG, reasoning through structure

He then switches to a PowerPoint deck (titled *Vectorless RAG versus Traditional
RAG*). Its first relevant slide, labelled "Approach 2", restates the idea with a
concrete example: an **Annual Report 2024** tree whose chapters are *1. Business*,
*2. Risks* and *3. Financials*, and where *Risks* has sub-sections *Market*,
*Credit* and *Operational*. For a question about credit risk the LLM navigates
**root, chapter, section, answer**.

The right-hand side lists how the LLM navigates, in five steps:

1. **Build the tree.** Parse the document structure (headings, sections) and
   generate summaries at each node. This is done once, offline.
2. **LLM reads the root summary.** It asks itself which chapter is most likely to
   contain the answer.
3. **Descend the tree.** Repeat at each level until a leaf section is reached.
4. **Read the full section.** No chunking: the full context is preserved.
5. **Answer and cite the path.** The response includes the answer *and* the
   navigation path that led to it.

<Infographic
  src="/img/agentic-course/05-reasoning-slide.svg"
  alt="An Annual Report 2024 tree with Business, Risks and Financials, Risks expanded into Market, Credit and Operational with Credit highlighted, beside five numbered steps of how the LLM navigates."
  caption="Redrawn from the slide, Vectorless RAG: Reasoning Through Structure."
/>

Notice how this differs slightly from the notebook: the slide describes a *top-down
descent*, level by level, whereas the notebook gives the LLM the whole tree and lets
it select nodes in one go. Both are reasoning over the same structure. The slide's
version needs more LLM calls on a deep tree (which is the "higher latency" cost
below), while the notebook's version needs one big prompt (which is the "tree must
fit in context" cost).

## Traditional RAG: the real picture

The next slide is called "The Real Picture", with the subtitle "powerful, but it has
known failure modes". It has two panels, strengths and weaknesses. He goes through
every item, and several of his explanations add useful detail.

<Infographic
  src="/img/agentic-course/05-traditional-real-picture.svg"
  alt="Traditional RAG strengths: massive scale, mature ecosystem, cheap retrieval, great for factoids, domain agnostic. Weaknesses: chunking destroys context, similarity is not relevance, no cross-section reasoning, hard to explain, embedding drift."
  caption="Redrawn from the slide (Traditional RAG: The Real Picture)."
/>

### Strengths

- **Massive scale.** With millions of documents, a company can look things up and
  get context in milliseconds. This is the main reason traditional RAG is still
  what you use at scale.
- **Mature ecosystem.** There are purpose-built vector databases: Chroma, FAISS,
  Pinecone, Qdrant, Weaviate, so you do not have to build the storage layer.
- **Cheap retrieval.** One embedding call plus one vector search per query. When you
  have huge amounts of data, cheap retrieval is a sound design goal.
- **Great for factoids.** Short, lookup-style questions, for example "what is the
  revenue of the company?", where one chunk holds the answer.
- **Domain agnostic.** It works on any text: blogs, tickets, PDFs. If you have a
  pile of unrelated information and want a chatbot assistant over it quickly, this
  is the way.

He adds a practical remark: the choice between traditional and vectorless RAG will
come up for you in a real job, when a problem statement arrives, so these
questions should be in your head.

### Weaknesses

- **Chunking destroys context.** The text of one concept can end up in chunks 1, 2
  and 3, while chunk 4 holds more of it, and only some of them match the query. The
  slide's example is a sentence like "as defined in Section 3.2 ...", which means
  nothing once it is retrieved alone. He notes why chunking exists at all: the LLM
  has a limited context, so you cannot pass the whole huge document, and chunking
  also lets you store the pieces in a vector database. The cost of that convenience
  is lost context.
- **Similarity is not relevance.** The embedding model can confidently match the
  wrong thing.
- **No cross-section reasoning.** It cannot answer "compare risk versus mitigation",
  because that needs two sections together.
- **Hard to explain.** "Why was this chunk picked?" has only a cosine score as
  an answer, and a cosine score is a similarity search, not a statement of
  relevance.
- **Embedding drift.** If you change the embedding model (it may have been trained on
  different information), you must embed everything again with the new model before
  you can use it.

## Vectorless RAG: the real picture

The matching slide has the subtitle "Different tradeoffs, better for some workloads,
worse for others". He introduces the idea once more: here the LLM navigates the
document like a human, the way you flip through books and pages.

<Infographic
  src="/img/agentic-course/05-vectorless-real-picture.svg"
  alt="Vectorless RAG strengths: preserves document context, cross-section reasoning, explainable retrieval, no embedding pipeline, plays well with structure. Weaknesses: higher per-query cost, higher latency, does not scale to millions, needs structured docs, less mature tooling."
  caption="Redrawn from the slide (Vectorless RAG: The Real Picture)."
/>

### Strengths

- **Preserves document context.** Because the structure comes from a table of
  contents and each node holds only its own section, the section stays whole, with
  no broken references. Information that belongs to one section lives in that node,
  as a summary, and is not scattered across other nodes.
- **Cross-section reasoning.** The LLM can compare, contrast and synthesise across
  sections, because it can pick several nodes and read them together.
- **Explainable retrieval.** The output carries the navigation path, not a cosine
  source. You can see which nodes were chosen and why.
- **No embedding pipeline.** One of the biggest costs disappears: there is nothing
  to embed, index or refresh, and nothing to re-embed when a model changes.
- **Plays well with structure.** Reports, contracts, filings and textbooks suit it.

His one-line takeaway on relevance: with vectorless RAG, it is *relevance* that
the retrieval captures (rather than cosine similarity), so the context you get is
usually better than from the traditional approach on structured material. This is
why he says use traditional RAG if the data is unstructured, and use vectorless if
the data has structure or belongs to a specific domain.

### Weaknesses

- **Higher per-query cost.** Several LLM calls are needed to traverse the tree. Even
  building the tree requires LLM calls, because the LLM writes the node summaries
  (the cost of that is paid once per document).
- **Higher latency.** Several hundred milliseconds to a few seconds per query. In
  his example the tree is tiny; a real document produces a very big tree, and the
  query has to travel down it. Whenever you design an inference-time system, check
  the inference performance first.
- **Does not scale to millions.** It works for tens to thousands of documents, not
  internet scale, because with millions of documents the trees would be enormous to
  build and to traverse.
- **Needs structured documents.** A random blog post has no headings to build a
  tree from, so the tree adds little value. He says "it is no use to use vector rag"
  for unstructured content, which is a slip: he means vectorless RAG.
- **Less mature tooling.** PageIndex and a few others exist, but the ecosystem is
  far younger than the vector database world. He expects it to improve and sees
  it being very handy for domain-specific use cases.

:::note A slip in the video
He says that for unstructured documents "it is no use to use vector
rag". The slide beside it says the tree "adds little value" for random blog posts,
so the intended word is *vectorless*. The correct rule is: unstructured, mixed
material belongs with vector RAG.
:::

## Side-by-side: the honest comparison

The comparison slide has eight rows, which he reads out to settle the decision. It
is the centrepiece of the section, so it is reproduced both as a picture and as a
table.

<Infographic
  src="/img/agentic-course/05-side-by-side.svg"
  alt="A table of eight dimensions comparing traditional RAG and vectorless RAG: scale, latency, cost, cross-section reasoning, explainability, best for, setup complexity, ecosystem maturity."
  caption="Redrawn from the slide (Side-by-Side: The Honest Comparison)."
/>

| Dimension | Traditional RAG | Vectorless RAG |
| --- | --- | --- |
| Scale | Millions of docs (strong) | Tens to thousands |
| Latency per query | Milliseconds (strong) | Hundreds of ms to seconds |
| Cost per query | Cheap, one embedding lookup (strong) | Higher, multiple LLM calls |
| Cross-section reasoning | Weak | Strong |
| Explainability | Cosine score (opaque) | Navigation path (strong) |
| Best for | Factoid Q&A, mixed corpora | Long structured documents |
| Setup complexity | Embedding pipeline plus a database | Tree builder, no database |
| Ecosystem maturity | Very mature (strong) | Emerging |

He walks down the rows with a few remarks: with millions of documents go to
traditional RAG, with tens to thousands go vectorless; the query cost is cheap on
the vector side and higher on the tree side because of the several LLM calls; on
cross-section reasoning, chunking can miss context between sections whereas the tree
approach summarises each section; for the "best for" row, think of the finances of a
company or its legal contracts as the typical vectorless use; and on setup
complexity, the vector side needs an embedding pipeline *and* a database, while the
tree side needs just the tree builder.

### Extra rows from the notebook (an addition)

The notebook's own comparison table, which the video does not show, adds a few
dimensions that complete the picture.

| Dimension | Traditional vector RAG | PageIndex (vectorless) |
| --- | --- | --- |
| Document preparation | Chunk into fixed pieces | Build a hierarchical tree |
| Indexing | Embed each chunk | LLM reads the structure |
| Storage | A vector database | A JSON file |
| Query processing | Embed the query, approximate nearest-neighbour search | LLM reasons over the tree |
| What is retrieved | Flat, anonymous chunks | Named sections with page references |
| Domain expertise | Needs the embedding model fine-tuned | Add rules to the prompt |
| Infrastructure | Pinecone, FAISS or Chroma | No vector DB needed |

:::note Not from the video
The notebook also claims accuracy figures (about 80 percent for vector RAG against
98.7 percent for PageIndex on the FinanceBench question-answering benchmark). Those
numbers are vendor-reported for the vendor's own system and are not independently
verified, and the instructor never mentions them. Treat them as marketing, and run a
small evaluation on your own documents before deciding. The cell also claims there is
"no hallucination from irrelevant chunks", which is too strong: a model can still
hallucinate over perfectly relevant sections.
:::

## When to use which

Two more slides turn the trade-offs into a decision guide.

<Infographic
  src="/img/agentic-course/05-when-to-use.svg"
  alt="Two panels of four cards each: use traditional RAG for massive heterogeneous corpora, latency-critical apps, short factoid queries and cost-sensitive scale; use vectorless RAG for long structured documents, reasoning over similarity, explainability and when chunking destroys meaning."
  caption="Redrawn from the slides "Use Traditional RAG when" and "Use Vectorless RAG when"."
/>

**Use traditional RAG when:**

| Situation | Why |
| --- | --- |
| Massive, heterogeneous corpora | Millions of mixed-format items: blogs, tickets, transcripts, knowledge-base articles. |
| Latency-critical apps | Chatbots, search-as-you-type and voice assistants where every millisecond counts and you need the output quickly. |
| Short factoid queries | "What is the warranty period?" or "Who is the CEO?", where the answer lives in one chunk. |
| Cost-sensitive at scale | Thousands of queries per minute: embedding lookups cost pennies, while an LLM tree walk would not be affordable. |

**Use vectorless RAG when:**

| Situation | Why |
| --- | --- |
| Long, structured documents | Annual reports, 10-Ks (the US annual company filings), legal contracts, regulatory filings, research papers and textbooks. |
| Reasoning matters more than similarity | His emphasis is that relevance is worth more than similarity: "compare the risk factors in section 7 with the mitigations in section 12" is something pure similarity cannot do. |
| Explainability is required | Compliance, audit, legal and financial advisory: show your work, not just the answer. |
| Chunking destroys meaning | When cross-references such as "as defined in section 3.2" lose their meaning once the chunk is retrieved alone, do not use plain traditional RAG; use vectorless. |

He also gives a two-question rule of thumb while discussing the weaknesses: first,
is the document **structured**? Second, **how many documents** are there (tens of
thousands, say) and is the use case **domain-specific**? Together with latency and
cost, those are the factors to decide on.

## Hybrid RAG and the key takeaways

His closing point is where the field is heading. People are starting to build
**hybrid RAG**, which combines the strongest feature of vectorless RAG with the
strongest of traditional vector RAG, so both kinds of search happen. The summary
line: traditional RAG is *scale* and vectorless RAG is *reasoning plus structure*.
They are **not competitors but complementary**. Pure vector search and pure tree
navigation are both extremes. The right choice depends on the document, not on the
hype. Production systems are going hybrid, and many companies already use both.

The final slide, "Key Takeaways", lists five things to remember:

| # | Takeaway | Detail from the slide |
| --- | --- | --- |
| 1 | Traditional RAG is scale plus speed | Vector similarity is unbeatable for millions of documents and millisecond lookups. |
| 2 | Vectorless RAG is reasoning plus structure | LLM-driven tree navigation preserves context and explains itself. |
| 3 | They are not competitors | They are complementary; pure vector search and pure tree navigation are both extremes. |
| 4 | The right pick depends on the document, not the hype | Long structured filings: vectorless. Mixed knowledge base: vector. Big system: hybrid. |
| 5 | Production systems are going hybrid | Vectors narrow the search space; trees do the precise reasoning inside it. |

The video does not draw how a hybrid pipeline fits together, so the board below
spells out the pattern in the slide's own words: use vectors to narrow millions of
documents to a few, then let the tree do precise reasoning inside them, then answer
with citations.

<Infographic
  src="/img/agentic-course/05-hybrid-pattern.svg"
  alt="A hybrid pipeline: vector search narrows the corpus to a few documents, tree reasoning picks sections inside them, and a cited answer is produced; below, a guide to picking by document type."
  caption="Explanatory board (not shown in the video), drawn from his takeaways 4 and 5."
/>

He signs off here, and the recording moves straight into the next topic, Deep
Agents.

## Notebook cells he did not run (an addition)

The notebook contains four more sections that the instructor mentions in the opening
list but does not run on camera. They are included here so the notes cover the whole
file, and are clearly an addition. Everything in this part builds on the
`pageindex_tree`, `llm_tree_search`, `find_nodes_by_ids` and `generate_answer`
definitions above.

:::note Not run in the video
None of the cells in this part appear on screen. The outputs shown are from the
notebook file's saved outputs and are trimmed. Code is unchanged apart from emoji
removed from print strings.
:::

### Section 6: expert-guided retrieval

The notebook's pitch: with vector RAG, injecting domain expertise means
fine-tuning the embedding model, which is expensive. With PageIndex you add
**routing rules to the prompt**, such as "if the query mentions EBITDA, prioritise
the MD&A section" or "if it is about risks, check Part I, Item 1A". The model uses
the rules to decide where to look, with no training at all.

The notebook first defines a set of finance rules, then overwrites the same
variable with rules for the syllabus document (it is still called
`FINANCIAL_EXPERT_RULES`, a leftover name). Only the second definition is live when
the later cells run.

```python

FINANCIAL_EXPERT_RULES = """
Expert routing rules for financial documents (10-K, annual reports):
- EBITDA, profitability queries    → MD&A section (Management Discussion & Analysis)
- Liquidity, cash flow queries     → Cash Flow Statement + liquidity footnotes
- Risk factor queries              → Part I, Item 1A (Risk Factors)  
- Revenue breakdown queries        → Segment reporting or Item 7
- Forward-looking / strategy       → CEO letter, Outlook, Strategy section
- Debt, credit, leverage queries   → Balance Sheet + debt footnotes
- Regulatory / compliance queries  → Legal Proceedings or regulatory filings
"""

print("Expert rules defined")
print("   These get injected into the retrieval prompt at query time.")
```

```python
# ── Expert Routing Rules — Advanced Route of Learning AI ─────────────────────
# Krish Naik Academy | 21 Modules | 38 Sections | 481 Topics
FINANCIAL_EXPERT_RULES = """
Route queries to the correct module using these rules:
 
M1  Neural Network Refresher   → backprop, activations, optimizers, PyTorch basics
M2  Hardware                   → GPU, TPU, Apple Silicon, compute infrastructure
M3  Transformers 101           → attention, self-attention, encoder-decoder, MHA
M4  Tokenization               → BPE, WordPiece, SentencePiece, Byte Latent Transformers
M5  Finetuning Architectures   → hands-on BERT/GPT/T5 finetuning, Hugging Face
M6  KV Cache & Attention       → KV cache, Flash Attention, MQA, GQA, RoPE, vLLM
M7  Scaling Laws               → Kaplan, Chinchilla, compute-optimal training
M8  Mixture of Experts         → MoE, sparse computation, Mixture of Depths
M9  Modern LLM Finetuning      → LoRA, QLoRA, SFT, DPO, PPO, RLHF, GRPO, ORPO,
                                  quantization, TRL, Unsloth, synthetic data,
                                  reasoning models, evaluation, deployment
M10 SLM                        → small language models, pruning, when SLM vs LLM
M11 Knowledge Distillation     → student-teacher, soft labels, DistilBERT, DeepSeek-R1
M12 Hybrid Architectures       → Mamba, RWKV, SSMs, Jamba, Nemotron, beyond Transformers
M13 Vision Foundations         → ViT, patch embeddings, CLIP, SigLIP, DINOv2
M14 Visual Language Models     → VLM architecture, aligner, multimodal reasoning
M15 Stable Diffusion & DiT     → DDPM, latent diffusion, FLUX.1, ControlNet, DreamBooth
M16 Embedding Models           → dense, sparse, binary, Matryoshka, MRL, fine-tuning
M17 RAG                        → chunking, BM25, ColBERT, hybrid RAG, rerankers,
                                  self/corrective/adaptive/agentic RAG, Graph RAG,
                                  multi-modal RAG, ColPali, RAG security
M18 Context Engineering        → prompt vs context engineering, memory architecture,
                                  context compression, KV cache, agent context lifecycle
M19 DSPy                       → signatures, modules, MIPROv2, self-optimizing RAG
M20 Agents                     → ReAct, MCP, LangGraph, CrewAI, browser agents,
                                  A2A, guardrails, observability, evaluation
M21 RL                         → PPO, GRPO, DAPO, GSPO, CISPO, reward models,
                                  RLHF vs RLVR, policy gradient, DeepSeek-R1 training
 
Cross-cutting rules:
- "learning path / where to start"     → M1 → M2 → M3 in order
- "production / deployment / serving"  → M9 (quantization) + M20 (agents)
- "fine-tuning vs RAG"                 → M9 + M17 + M18
- "multimodal / vision + language"     → M13 + M14 + M17 (multi-modal RAG)
- "reasoning models / test-time RL"    → M9 (reasoning) + M21 (GRPO/DAPO)
"""
```

The second cell, the one in force, maps each of the 21 course modules to the topics
it covers, then adds cross-cutting rules such as "production or deployment means
module 9 plus module 20". It is a table of "where to look" that a senior person on
the team would know.

```python
# ── Expert-guided tree search ────────────────────────────────────────────────

def llm_tree_search_with_expert(
    query: str,
    tree: list,
    expert_rules: str,
    model: str = "gpt-4o"
) -> dict:
    """
    Same as llm_tree_search() but with domain expert rules injected.
    The LLM uses these rules to guide its reasoning.
    """
    
    def compress(nodes):
        out = []
        for n in nodes:
            entry = {"node_id": n["node_id"], "title": n["title"],
                     "page": n.get("page_index", "?"),
                     "summary": n.get("text", "")[:150]}
            if n.get("nodes"):
                entry["children"] = compress(n["nodes"])
            out.append(entry)
        return out

    prompt = f"""You are a domain expert analyzing a document.
Find all node IDs that most likely contain the answer to the query.
Use the expert routing rules below to guide your reasoning.

Query: {query}

Document Tree:
{json.dumps(compress(tree), indent=2)}

Expert Routing Rules (follow these carefully):
{expert_rules}

Reply ONLY in this JSON format:
{{
  "thinking": "<your reasoning, referencing the expert rules>",
  "node_list": ["node_id1", "node_id2"]
}}"""

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

`llm_tree_search_with_expert` is the earlier search function plus one block in the
prompt: a persona ("domain expert"), the rules text, and an instruction to follow
them and reference them in the reasoning.

```python
# ── Test expert-guided retrieval ─────────────────────────────────────────────
query = "Details of the modern llm finetuning?"

print(f"Query: {query}\n")

# Without expert rules
print("── Without Expert Rules ──")
basic   = llm_tree_search(query, pageindex_tree)
print("Nodes:", basic.get("node_list"))

print()

# With expert rules
print("── With Expert Rules ──")
guided  = llm_tree_search_with_expert(query, pageindex_tree, FINANCIAL_EXPERT_RULES)
print("Nodes:", guided.get("node_list"))
print("Reasoning:", guided.get("thinking", "")[:300])
```

This compares the same question with and without rules.

```text
Query: Details of the modern llm finetuning?

-- Without Expert Rules --
Nodes: ['0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020', '0021']

-- With Expert Rules --
Nodes: ['0010', '0014', '0015', '0016', '0018', '0020']
Reasoning: Based on the expert routing rules, the query 'Details of the modern llm finetuning?' aligns with M9, which covers Modern LLM Finetuning. This module includes topics like LoRA, QLoRA, SFT, DPO, PPO, RLHF, GRPO, ORPO, quantization, evaluation, and deployment, all of which are part of modern LLM fine-t
```

With the rules, the model picks fewer, more targeted nodes and quotes the rule in its
reasoning.

```python
# ── Full expert-guided RAG ───────────────────────────────────────────────────

def expert_rag(query: str, tree: list, rules: str) -> str:
    """Expert-guided end-to-end RAG pipeline."""
    result  = llm_tree_search_with_expert(query, tree, rules)
    nodes   = find_nodes_by_ids(tree, result.get("node_list", []))
    return generate_answer(query, nodes)

# Run it
answer = expert_rag(
    query="Details of the syllabus of modern llm finetuning",
    tree=pageindex_tree,
    rules=FINANCIAL_EXPERT_RULES
)
print(answer)
```

`expert_rag` is the full pipeline with the guided search in the first step. Its answer
(trimmed) opens: "The syllabus for modern LLM fine-tuning includes several key
components and techniques: 1. Comprehensive Fine-Tuning Stack ... (Section: 'Modern
LLMs Finetuning' | Page 12) ...", with a numbered, cited list that goes on to
pre-training, data preparation, PEFT, SFT, preference alignment, the post-training
pipeline, evaluation and quantization.

### Section 7: the PageIndex Chat API

If you do not want to manage OpenAI calls yourself, PageIndex has its own LLM and an
OpenAI-style chat endpoint: you pass the question and the `doc_id`, with no OpenAI
key. This is what powers the hosted chat page he showed.

```python
# ── Single question with Chat API ────────────────────────────────────────────
# No OpenAI key needed — PageIndex runs the LLM internally

question = "What are the key findings in this document?"

response = pi_client.chat_completions(
    messages=[{"role": "user", "content": question}],
    doc_id=doc_id
)

answer = response["choices"][0]["message"]["content"]
print("Chat API Answer:")
print(answer)
```

The call is `pi_client.chat_completions(messages=[...], doc_id=doc_id)`. The reply
has the usual `choices[0].message.content` shape. The saved output is a long,
well-structured markdown summary of the syllabus ("Key Findings": scope and scale,
foundations, advanced architectures, LLM training and fine-tuning, specialised models
and production systems), which begins "I'll analyze the document structure and
extract the key findings from this AI course syllabus."

For a multi-turn conversation you resend the history on each turn:

```python
# ── Multi-turn conversation ───────────────────────────────────────────────────
# Keep the full message history for context across turns

conversation_history = []

def chat_with_doc(user_message: str, doc_id: str) -> str:
    """Chat with a document, maintaining conversation history."""
    global conversation_history
    
    conversation_history.append({"role": "user", "content": user_message})
    
    response = pi_client.chat_completions(
        messages=conversation_history,
        doc_id=doc_id
    )
    
    assistant_reply = response["choices"][0]["message"]["content"]
    conversation_history.append({"role": "assistant", "content": assistant_reply})
    
    return assistant_reply


# Simulate a 3-turn conversation
questions = [
    "What were the main revenue sources last year?",
    "How does that compare to the year before?",
    "What factors drove that change?"
]

for q in questions:
    print(f"\nUser: {q}")
    reply = chat_with_doc(q, doc_id)
    print(f"Assistant: {reply[:400]}...")
    print("-" * 55)
```

`conversation_history` accumulates the user and assistant messages, so each new
question carries the earlier ones, and "How does that compare to the year before?"
can be understood. Be careful with the sample questions: they ask about revenue
and last year, which makes sense for a financial report but not for the syllabus
PDF, so on this document the answers will be about nothing in particular. Replace
them with questions about your document. The function also uses `global`, which is
fine in a demo but better replaced by passing the history in explicitly in real
code.

### Section 8: the self-hosted open-source option

Use this when documents cannot leave your network, when you need on-prem deployment,
or when you want to change the tree-building logic. The open-source runner reads
your PDF, detects any table of contents, uses an OpenAI model to build the tree and
saves `<pdf name>_pageindex.json` next to the PDF.

```python
# ── Clone the open-source repo ───────────────────────────────────────────────
!git clone https://github.com/VectifyAI/PageIndex.git
%cd PageIndex
!pip install -r requirements.txt
```

Clone the repository, change into it and install its requirements.

```python
# ── Create .env for self-hosted mode ─────────────────────────────────────────
# The local runner uses CHATGPT_API_KEY (not OPENAI_API_KEY)

import os
openai_key = os.getenv("OPENAI_API_KEY", "your_key_here")

with open(".env", "w") as f:
    f.write(f"CHATGPT_API_KEY={openai_key}\n")

print(".env created for self-hosted mode")
```

The local runner expects the key under the name `CHATGPT_API_KEY`, not
`OPENAI_API_KEY`, so this cell writes a small `.env` for it.

```python
# ── Run PageIndex locally on a PDF ───────────────────────────────────────────
# Optional parameters you can customize:
#   --model                  OpenAI model (default: gpt-4o-2024-11-20)
#   --toc-check-pages        Pages to scan for existing TOC (default: 20)
#   --max-pages-per-node     Max pages per tree node (default: 10)
#   --if-add-node-summary    Include summaries in output (yes/no)

PDF_PATH = "/path/to/your/document.pdf"   # ← change this

!python run_pageindex.py \
    --pdf_path {PDF_PATH} \
    --model gpt-4o-2024-11-20 \
    --toc-check-pages 20 \
    --max-pages-per-node 10 \
    --if-add-node-summary yes
```

The options map back to the tree-building flow you saw earlier: `--toc-check-pages`
is how many pages to scan for an existing table of contents (default 20),
`--max-pages-per-node` caps how many pages one node may hold (default 10, which is
why "no chunking" is not absolute), `--if-add-node-summary` includes the summaries,
and `--model` chooses the OpenAI model.

```python
# ── Load locally generated tree ──────────────────────────────────────────────
# Output is saved as: <your_pdf_name>_pageindex.json

import json

TREE_JSON_PATH = "/path/to/your/document_pageindex.json"  # ← change this

with open(TREE_JSON_PATH, "r") as f:
    local_tree = json.load(f)

print(f"Local tree loaded: {count_nodes(local_tree)} total nodes")
print_tree(local_tree)
```

```python
# ── Run the same RAG pipeline on the local tree ──────────────────────────────
# Everything from Sections 4–6 works identically with local trees

query  = "Summarize the executive summary section."
answer = vectorless_rag(query, local_tree)
```

Because the local tree has the same shape, every function from the video works on it
unchanged.

### Section 9 and 10: comparison demo and clean-up

The notebook's comparison cell only prints text describing the two approaches
(already summarised above), and the final cell, which deletes the document from the
PageIndex cloud, is commented out so that you do not lose your index by accident:

```python
# ── Delete document from cloud ───────────────────────────────────────────────
# WARNING: This permanently deletes the tree index.
# Comment this out if you want to reuse the doc_id later.

# pi_client.delete_document(doc_id)
# print(f"Deleted document: {doc_id}")
print("ℹDeletion commented out — uncomment when you're done with this doc_id")
```

:::warning Deleting is permanent
`delete_document` permanently removes the tree index. Leave it commented out until
you are finished with that `doc_id`.
:::

## Putting it together: choosing a retrieval approach

Here is the decision as a short checklist, in the order the instructor reasons about
it.

1. **Is the material structured?** Reports, contracts, filings, textbooks and
   manuals have headings and a table of contents. Blogs, tickets and transcripts do
   not. Unstructured material points to vector RAG.
2. **How many documents?** Tens to thousands is comfortable for a tree. Millions
   points to vector RAG, because the cost and latency of tree work grows with the
   corpus.
3. **Does the question need two sections read together, or a reason to be shown?**
   Compare-and-contrast questions, audit and compliance use cases favour the tree.
4. **Is every millisecond or every penny a constraint?** Chatbots and high query
   rates favour vector lookups.
5. **Do you have both kinds of content or both kinds of need?** Go hybrid: vectors to
   narrow, trees to reason.

## What you can now do

- I can explain, step by step, how a traditional vector RAG pipeline works (chunk,
  embed, store, embed the query, similarity search, flat chunks, generate) and
  where the answer's context comes from.
- I can name the weaknesses of naive vector RAG (chunking destroys context,
  similarity is not relevance, no cross-section reasoning, hard to explain,
  embedding drift) and give an example of the first two.
- I can describe what a PageIndex tree is: nodes with an id, title, page, summary and
  children, stored as a JSON tree index, with no vector database.
- I can explain how PageIndex builds the tree with a table of contents and without
  one (TOC detection, parse or infer headings, section-aware splitting, summaries,
  hierarchical assembly) and why splitting at section boundaries matters.
- I can draw the retrieval loop (read the tree, reason and select nodes, extract
  section content, check sufficiency, answer with citations) and explain how
  cross-references can be followed.
- I can set up the SDK and clients, upload a PDF, poll until the tree is ready, fetch
  it with `get_tree`, and print and count its nodes.
- I can write and explain `llm_tree_search`, `find_nodes_by_ids`, `generate_answer`
  and `vectorless_rag`, including why the tree is compressed and how JSON mode
  returns reasoning plus node ids.
- I can spot the two issues in the notebook (the `text` field is not a summary, and
  the compress step reads the first 150 characters of `text`) and avoid hard-coding an
  API key.
- I can say where a JSON tree index can be stored (file system, S3, MongoDB) and
  what limits its size.
- I can choose between vector, vectorless and hybrid retrieval for a given project
  using the comparison table, the decision checklist and the document-structure test.
