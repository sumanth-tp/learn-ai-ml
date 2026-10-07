---
id: rag-adv-graphrag
title: "GraphRAG and Knowledge Graphs"
sidebar_label: "1 · GraphRAG"
sidebar_position: 1
slug: /genai/rag-advanced/graphrag-and-knowledge-graphs
description: "Build an entity graph and community summaries from a corpus, then use it for the two question types that plain vector retrieval handles badly: multi-hop lookups and whole-corpus themes."
tags: [graphrag, knowledge-graph, networkx, community-detection, multi-hop, rag]
---

import Infographic from '@site/src/components/Infographic';
import GraphRetrievalLab from '@site/src/components/viz/GraphRetrievalLab';

**In one line.** Vector RAG finds passages that sound like the question; GraphRAG first turns the corpus into entities, relations and communities, so it can follow a chain of facts or summarise the whole collection, at the price of an expensive index.

:::note Not from a lecture
Written for this site from the sources under Further reading. The corpus, the extractor and every number below are produced by the code in this chapter, not taken from a paper.
:::

## The idea in plain words

Ordinary RAG ([chapter 16 of this course](/docs/genai/rag), and the retrieval stage in [neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking)) cuts documents into chunks, embeds them and returns the chunks nearest to the question. That works when the answer sits inside one or two chunks that share words and meaning with the question.

Two kinds of question break it.

1. **Multi-hop questions.** "Which regulator oversees the company that owns the truck lessor of Delmar Freight?" The facts live in three different sentences: Delmar leases trucks from Quarry Fleet, Quarry Fleet is owned by Helion Group, Nordic Safety Board regulates Helion Group. No single sentence resembles the question, and the sentence that holds the final answer never mentions Delmar at all.
2. **Global questions.** "What are the main themes across these documents?" There is no passage to match. The answer is a property of the whole collection.

GraphRAG attacks both with one build step. A language model reads every chunk and writes down **entities** (people, companies, products) and **relations** between them. Those become a graph. A community-detection algorithm then splits the graph into clusters of tightly connected entities, and a model writes a short **report** for each cluster. At question time a **local** query starts from the entities named in the question and walks outward; a **global** query reads the community reports and merges them.

The graph is not free. Every chunk costs at least one generation call at index time, and every edit to the corpus means re-extracting. Section "When it is worth the build cost" turns that into numbers.

<Infographic src="/img/rag-adv/graphrag-and-knowledge-graphs-pipeline.svg" alt="Documents are extracted into an entity graph, split into communities and summarised; local questions walk relations from a seed entity, global questions read every community report" caption="One index, two query paths. The counts are printed by code block 1." />

## How it works

### Extraction: text to triples

A triple is `(head, relation, tail)`, for example `(Helion Group, owns, Voltro)`. In a real pipeline a model with a structured-output prompt produces them, usually after a second "did you miss anything?" pass over each chunk, often called gleaning. Two failure modes matter more than any other: the model names the same entity two ways ("Helion", "Helion Group", "the Helion group"), and it invents a relation the text does not state. Both silently corrupt the graph, so production systems merge duplicate entities and keep, on every edge, a pointer to the source chunk so an answer can always be traced back.

The code below uses a regular expression per sentence shape instead of a model. This is a **deterministic stand-in**, chosen so the chapter runs offline and the numbers are exact. It teaches the graph mechanics; it says nothing about how accurate a model's extraction would be.

### The graph and its provenance

Entities become nodes and relations become labelled edges. Each edge carries the id of the document it came from, which is what lets a graph answer cite evidence. Because several sentences can add edges around one entity, hubs such as `Helion Group` appear, and hubs are where naive traversal goes wrong: a two-hop neighbourhood of a hub is most of the graph.

### Communities and reports

Community detection groups nodes so that edges inside a group are denser than edges between groups. The Louvain method, available in networkx, greedily moves nodes between groups to raise a score called modularity. Real systems also build a **hierarchy**: communities of communities, so a report exists at several zoom levels. Each community gets a report describing its entities and what they do. Here the report is a template stand-in; with a model it is a paragraph.

### Local and global queries

| | Local query | Global query |
| --- | --- | --- |
| Starts from | entities named in the question | no entity, the whole corpus |
| Reads | the neighbourhood: relations, nearby text | every community report (map), then merges partial answers (reduce) |
| Good for | "who owns the owner of X", multi-hop lookups | "what are the main themes", comparisons across the corpus |
| Cost per question | about one vector query plus some graph context | one call per report, so it grows with the number of communities |

In the code, "path following" is the local method in its purest form. Someone, normally a model, maps the question to a path of relation types such as owns, then regulates; the graph executes the path and returns both the answer and the documents it used. Here the mapping is written by hand, so the 8 out of 8 below is an **upper bound**: it assumes the question was understood perfectly. The honest cost of the method is how often a model gets that mapping wrong.

### When it is worth the build cost

Use a graph when at least one of these is true:

- Users ask relational questions whose evidence is spread across documents (ownership chains, dependencies, "who reports to whom").
- Users ask corpus-level questions: themes, trends, "summarise everything about X".
- Auditability matters and every claim must point to the edge and document that support it.

Skip it when questions are mostly lookups inside a single document, when the corpus changes hourly, or when you have not yet tried the cheaper levers: better chunking, hybrid search, reranking and query rewriting (see [contextual retrieval and reranking](/docs/genai/rag-advanced/contextual-retrieval-and-reranking)). Knowledge graphs as a data model, and how to build and query them, are covered in [knowledge graphs in the NLP track](/docs/theory/nlp/knowledge-graphs).

## A real system that works this way

**Microsoft's GraphRAG.** The paper "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Edge and colleagues, first posted 24 April 2024, revised 19 February 2025) describes a two-stage index: an LLM extracts an entity knowledge graph from the source documents, then pre-generates summaries for clusters of closely related entities. At question time each community summary yields a partial response and the partial responses are combined into the final answer. The authors argue that conventional RAG fails on questions such as "What are the main themes in the dataset?" because they are query-focused summarisation rather than retrieval, and report improvements over a conventional RAG baseline in the comprehensiveness and diversity of answers on corpora of about one million tokens. Microsoft Research's announcement post of 13 February 2024 makes the same argument and describes the index as an LLM-generated knowledge graph plus bottom-up hierarchical clustering with pre-generated summaries. It says nothing about cost; the paper and post I opened give no price figure, which is why the cost model below is parameterised and synthetic.

**LightRAG** (Guo and colleagues, arXiv 2410.05779, October 2024, revised April 2025) takes a lighter route: graph structures are built into text indexing, retrieval is dual-level (specific entities and broader themes), and an incremental update algorithm lets new documents join the index without rebuilding it. That targets exactly the cost problem above. I read only the abstract, so I make no claim about its measured quality.

## Code you can run

Two blocks. Block 1 builds a 24-sentence corpus, extracts a graph, compares vector search with path following on eight multi-hop questions, then finds communities and tries a global question. Block 2 is a cost model. Libraries used: networkx 3.6.1, sentence-transformers 6.1.0 (model `all-MiniLM-L6-v2`), numpy 2.5.3, run on CPU with Python 3.14.

If the Hub is slow once the model is cached, run with `HF_HUB_OFFLINE=1`.

### 1. Graph, local questions and global questions

The reader is not simulated. For each question the test is whether **every document a reader would need** is in the context it is given: with `k` documents from vector search, or exactly the documents a path walk touched.

```python
import re
from collections import Counter

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS = [
    "Kestrel Motors builds the Aster Scooter in Lisbon.",
    "Voltro supplies battery cells to Kestrel Motors.",
    "Helion Group owns Voltro.",
    "Nordic Safety Board regulates Helion Group.",
    "Mara Ellis is chief executive of Kestrel Motors.",
    "Kestrel Motors sells the Aster Scooter to Cobalt Rentals.",
    "Calder Labs develops the drug Zerafil.",
    "Lumen Therapeutics licenses Zerafil from Calder Labs.",
    "Pharma Standards Agency approved Zerafil.",
    "Pharma Standards Agency regulates Lumen Therapeutics.",
    "Tomas Brandt is chief executive of Calder Labs.",
    "Lumen Therapeutics sells Zerafil to Harbour Pharmacies.",
    "Delmar Freight ships goods for Harbour Pharmacies.",
    "Delmar Freight leases trucks from Quarry Fleet.",
    "Quarry Fleet is owned by Helion Group.",
    "Customs Directorate regulates Delmar Freight.",
    "Ines Valdo is chief executive of Delmar Freight.",
    "Cobalt Rentals partners with Delmar Freight.",
    "Paylink processes payments for Cobalt Rentals.",
    "Paylink is owned by Tidewater Capital.",
    "Financial Conduct Office regulates Paylink.",
    "Tidewater Capital funds Calder Labs.",
    "Rhea Okafor is chief executive of Paylink.",
    "Financial Conduct Office regulates Tidewater Capital.",
]

N = r"([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*)"
PATTERNS = [
    (rf"^{N} builds the {N} in {N}\.$", lambda m: [(m[1], "builds", m[2]), (m[1], "located_in", m[3])]),
    (rf"^{N} supplies battery cells to {N}\.$", lambda m: [(m[1], "supplies", m[2])]),
    (rf"^{N} owns {N}\.$", lambda m: [(m[1], "owns", m[2])]),
    (rf"^{N} is owned by {N}\.$", lambda m: [(m[2], "owns", m[1])]),
    (rf"^{N} regulates {N}\.$", lambda m: [(m[1], "regulates", m[2])]),
    (rf"^{N} is chief executive of {N}\.$", lambda m: [(m[1], "leads", m[2])]),
    (rf"^{N} sells the {N} to {N}\.$", lambda m: [(m[1], "sells_to", m[3]), (m[3], "buys", m[2])]),
    (rf"^{N} sells {N} to {N}\.$", lambda m: [(m[1], "sells_to", m[3]), (m[3], "buys", m[2])]),
    (rf"^{N} develops the drug {N}\.$", lambda m: [(m[1], "develops", m[2])]),
    (rf"^{N} licenses {N} from {N}\.$", lambda m: [(m[1], "licenses", m[2]), (m[1], "licenses_from", m[3])]),
    (rf"^{N} approved {N}\.$", lambda m: [(m[1], "approved", m[2])]),
    (rf"^{N} ships goods for {N}\.$", lambda m: [(m[1], "serves", m[2])]),
    (rf"^{N} leases trucks from {N}\.$", lambda m: [(m[1], "leases_from", m[2])]),
    (rf"^{N} partners with {N}\.$", lambda m: [(m[1], "partners_with", m[2])]),
    (rf"^{N} processes payments for {N}\.$", lambda m: [(m[1], "serves", m[2])]),
    (rf"^{N} funds {N}\.$", lambda m: [(m[1], "funds", m[2])]),
]


def extract(sentence):
    for pattern, build in PATTERNS:
        match = re.match(pattern, sentence)
        if match:
            return build(match)
    raise ValueError(sentence)


graph = nx.MultiDiGraph()
for doc_id, text in enumerate(DOCS):
    for head, relation, tail in extract(text):
        graph.add_edge(head, tail, key=(relation, doc_id), relation=relation, doc=doc_id)
print(f"{len(DOCS)} documents -> {graph.number_of_nodes()} entities, {graph.number_of_edges()} relations")

QUESTIONS = [
    ("Which regulator oversees the owner of Voltro?", {2, 3}, "Voltro", [("owns", "in"), ("regulates", "in")]),
    ("Which regulator oversees the company that owns the truck lessor of Delmar Freight?", {13, 14, 3}, "Delmar Freight", [("leases_from", "out"), ("owns", "in"), ("regulates", "in")]),
    ("Who funds the developer of Zerafil, and which regulator oversees that funder?", {6, 21, 23}, "Zerafil", [("develops", "in"), ("funds", "in"), ("regulates", "in")]),
    ("Which regulator oversees the parent of the supplier of Kestrel Motors?", {1, 2, 3}, "Kestrel Motors", [("supplies", "in"), ("owns", "in"), ("regulates", "in")]),
    ("Who leads the carrier that serves the pharmacy chain buying Zerafil?", {11, 12, 16}, "Zerafil", [("buys", "in"), ("serves", "in"), ("leads", "in")]),
    ("Which regulator oversees the payment processor used by the rental firm that buys the Aster Scooter?", {5, 18, 20}, "Aster Scooter", [("buys", "in"), ("serves", "in"), ("regulates", "in")]),
    ("Which agency regulates the company that licenses Zerafil?", {7, 9}, "Zerafil", [("licenses", "in"), ("regulates", "in")]),
    ("Who leads the developer of the drug that Harbour Pharmacies buys?", {11, 6, 10}, "Harbour Pharmacies", [("buys", "out"), ("develops", "in"), ("leads", "in")]),
]


def follow(start, plan):
    frontier, docs = {start}, set()
    for relation, direction in plan:
        nxt = set()
        for node in frontier:
            edges = graph.in_edges(node, data=True) if direction == "in" else graph.out_edges(node, data=True)
            for head, tail, data in edges:
                if data["relation"] == relation:
                    nxt.add(head if direction == "in" else tail)
                    docs.add(data["doc"])
        frontier = nxt
    return frontier, docs


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
doc_vecs = encoder.encode(DOCS, normalize_embeddings=True)
question_vecs = encoder.encode([q[0] for q in QUESTIONS], normalize_embeddings=True)

print(f"{'documents given to the reader (k)':<36}" + "".join(f"{k:>5}" for k in range(2, 9)))
row = []
for k in range(2, 9):
    row.append(sum(q[1] <= set(np.argsort(-(doc_vecs @ question_vecs[i]))[:k].tolist()) for i, q in enumerate(QUESTIONS)))
print(f"{'questions with every needed document':<36}" + "".join(f"{v:>5}" for v in row) + f"   (of {len(QUESTIONS)})")

correct = 0
for question, gold, start, plan in QUESTIONS:
    answers, docs = follow(start, plan)
    correct += docs == gold
    print(f"{start:<20} {len(plan)} hops -> {sorted(answers)[0]:<26} documents {sorted(docs)}")
print(f"path following found exactly the needed documents for {correct}/{len(QUESTIONS)} questions")

undirected = nx.Graph()
for head_node, tail_node, data in graph.edges(data=True):
    if undirected.has_edge(head_node, tail_node):
        undirected[head_node][tail_node]["weight"] += 1
    else:
        undirected.add_edge(head_node, tail_node, weight=1)

communities = sorted(nx.community.louvain_communities(undirected, weight="weight", seed=0), key=lambda c: (-len(c), sorted(c)[0]))
member_of = {node: i for i, c in enumerate(communities) for node in c}
print(f"Louvain found {len(communities)} communities (sizes {[len(c) for c in communities]})")

doc_community = {}
for head_node, tail_node, data in graph.edges(data=True):
    doc_community[data["doc"]] = member_of[head_node]


def report(index):
    members = communities[index]
    sub = undirected.subgraph(members)
    hub = max(sorted(members), key=lambda n: undirected.degree(n))
    relations = Counter(d["relation"] for h, t, d in graph.edges(data=True) if h in members and t in members)
    docs = sorted(d for d, c in doc_community.items() if c == index)
    top = ", ".join(f"{name} x{count}" for name, count in relations.most_common(3))
    return {"hub": hub, "members": sorted(members), "docs": docs, "text": f"Community around {hub}: {len(members)} entities, {sub.number_of_edges()} links; main relations {top}."}


reports = [report(i) for i in range(len(communities))]
for i, r in enumerate(reports):
    print(f"C{i}: {r['text']}")

question = "What are the main themes across these documents?"
q_vec = encoder.encode([question], normalize_embeddings=True)[0]
print(f"\nglobal question: {question}")
for k in (3, 5, 8):
    top = np.argsort(-(doc_vecs @ q_vec))[:k]
    covered = {doc_community[int(d)] for d in top}
    print(f"vector search, top {k} documents: touches {len(covered)} of {len(communities)} communities")
print(f"map-reduce over community reports: {len(reports)} reports read, touches {len(communities)} of {len(communities)} communities, documents behind them {sum(len(r['docs']) for r in reports)} of {len(DOCS)}")
```

Read the first table. At `k = 4`, vector search puts every needed document in front of the reader for only 2 of the 8 questions; even at `k = 8`, a third of the whole corpus, it manages 6. Path following needs 2 or 3 documents per question and finds exactly the right ones for all 8, because the graph keeps the chain `Delmar Freight -> Quarry Fleet -> Helion Group -> Nordic Safety Board` as edges rather than hoping three sentences rank together. The second table shows the global side: top-5 vector search touches 2 of the 5 communities, reading all five reports touches all five.

Try it in the lab. The default shows the Delmar question with vector search at `k = 4`; the line under the plot reproduces the 2 of 8 above.

<GraphRetrievalLab />

<Infographic src="/img/rag-adv/graphrag-and-knowledge-graphs-vector-vs-graph.svg" alt="Bars comparing vector search at k from 2 to 8 with path following on 8 questions, community coverage for a global question, and a cost table" caption="What the graph buys and what it costs. Counts from block 1, costs from block 2." />

### 2. The cost model

Every price and size here is a named parameter on **synthetic** numbers (1.0 per million input tokens, 4.0 per million output tokens). Replace them with your provider's real prices and your corpus before deciding anything.

```python
PRICE_IN, PRICE_OUT = 1.0, 4.0

CORPUS_TOKENS = 1_000_000
CHUNK_TOKENS = 600
PROMPT_OVERHEAD = 400
EXTRACT_OUT = 300
GLEANINGS = 1
COMMUNITIES = 400
REPORT_IN, REPORT_OUT = 3000, 500
ANSWER_CONTEXT, ANSWER_OUT = 4000, 400


def cost(tokens_in, tokens_out):
    return (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6


chunks = CORPUS_TOKENS // CHUNK_TOKENS
extract_calls = chunks * (1 + GLEANINGS)
extract_in = extract_calls * (CHUNK_TOKENS + PROMPT_OVERHEAD)
extract_out = extract_calls * EXTRACT_OUT
extract_cost = cost(extract_in, extract_out)
summary_cost = cost(COMMUNITIES * REPORT_IN, COMMUNITIES * REPORT_OUT)
graph_index = extract_cost + summary_cost
print(f"chunks {chunks}, extraction calls {extract_calls}, summary calls {COMMUNITIES}")
print(f"graph index: extraction {extract_cost:.2f} + summaries {summary_cost:.2f} = {graph_index:.2f} units")
print("vector index: 0.00 units of generation (embeddings only)")

vector_query = cost(ANSWER_CONTEXT, ANSWER_OUT)
local_query = cost(ANSWER_CONTEXT + 1500, ANSWER_OUT)
global_query = cost(COMMUNITIES * (REPORT_OUT + 200), COMMUNITIES * 150) + cost(COMMUNITIES * 150 + 500, ANSWER_OUT)
print(f"\nper query: vector {vector_query:.4f}, graph local {local_query:.4f}, graph global {global_query:.4f} units")
print(f"one global query costs as much as {global_query / vector_query:.0f} vector queries")

for queries in (1_000, 100_000):
    print(f"\nafter {queries:,} queries: vector {queries * vector_query:,.0f}, graph (local only) {graph_index + queries * local_query:,.0f}, "
          f"graph (1 in 100 global) {graph_index + queries * (0.99 * local_query + 0.01 * global_query):,.0f}")
print(f"\nindex premium is {graph_index / vector_query:,.0f} vector queries; a 10 percent monthly edit of the corpus re-extracts {extract_cost * 0.1:.2f} units")
```

The shape is the lesson, not the units. The index is a fixed charge of roughly 1,700 vector queries; a local query costs only a little more than a vector query; one global query costs about as much as 104 vector queries because it reads every report. Edits are the hidden recurring cost: if 10 percent of the corpus changes each month, the extraction share is paid again for that tenth.

## Production snippets (not run here)

An extraction prompt for a real model. Not run in this environment, because it needs a hosted or local model of at least a few billion parameters; SmolLM2-135M is far too small to follow it.

```text
Read the passage. Return JSON with two lists.
entities: objects with name, type (person, company, product, agency, place).
relations: objects with head, relation, tail, evidence (a short quote).
Use only facts the passage states. Use one canonical name per entity.
Return nothing for facts you are unsure of.
```

Not run in this environment. A second pass ("list any entities or relations you missed") roughly doubles extraction calls, which is the `GLEANINGS` parameter in block 2.

## Designing with it

- **Start with questions, not graphs.** Collect twenty real user questions and label each local, global or ordinary lookup. If fewer than a fifth are local-multi-hop or global, a graph will not pay back.
- **Evaluate extraction before everything else.** Sample fifty chunks, extract by hand, and score the model's entities and relations for precision and recall. Wrong edges produce confident wrong answers, which are worse than a missed passage.
- **Resolve entities.** Decide a canonical name policy and test it with a list of known aliases. Unmerged duplicates split a hub in two and cut the path.
- **Keep provenance on every edge.** An answer that cannot point to a document is not auditable, and you cannot debug it.
- **Combine, do not replace.** Practical systems route: ordinary lookups to hybrid search ([the enterprise document-QA design](/docs/senior/design-enterprise-document-qa) shows the funnel), relational and corpus-level questions to the graph.
- **Budget the global query.** It scales with the number of communities. Cache answers to recurring questions, and use higher levels of the hierarchy for coarse questions.
- **Plan updates.** Decide up front whether you rebuild nightly, patch incrementally, or accept staleness; incremental update is the explicit selling point of LightRAG, and it is also where bugs hide.

If the surrounding system is an agent that decides when to retrieve, see [agents in LangChain v1](/docs/genai/langchain-advanced/create-agent): a graph query is just another tool the agent can call.

## Where this stands in 2026

:::info Industry view
The two papers above give the pattern (extract, cluster, summarise, query locally or globally), and LightRAG's abstract names the cost of rebuilding as the problem to attack. I opened only the sources listed below, so I cannot tell you which libraries or services implement it best today, or how widely it is deployed; check the current documentation of whichever you choose and run your own twenty-question test. What does not depend on the tooling is the trade: you pay at index time to answer relational and corpus-level questions, and you pay again whenever the corpus moves.
:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why can a vector search over short documents fail a three-hop question even when every needed document is in the corpus?</summary>

The ranking compares each document to the question separately. The question shares words with the first hop only, so the second and third hop sentences rank lower than unrelated documents that mention the same entity. In block 1, vector search at `k = 4` completes 2 of 8 questions.

</details>

<details>
<summary><strong>Q2.</strong> Path following scored 8 of 8. Why is that not evidence that GraphRAG is 100 percent accurate?</summary>

The mapping from question to relation path was written by hand, standing in for a model. A real pipeline must also extract the graph correctly and map the question correctly. The 8 of 8 is an upper bound that isolates what the graph structure can do.

</details>

<details>
<summary><strong>Q3.</strong> A user asks for the main complaints across 40,000 support tickets. Why does top-k vector search give a poor answer, and what does a community report add?</summary>

The question has no matching passage, so top-k returns the few tickets most similar to the word "complaints", a biased sample. A report per community summarises each cluster in advance, so a map-reduce over reports covers the whole collection. In block 1, top-5 vector search touched 2 of 5 communities and the reports touched all 5.

</details>

<details>
<summary><strong>Q4.</strong> Your corpus is 10 percent re-written every month. What happens to the cost, and which design choice reduces it?</summary>

The extraction share of the index is paid again for the changed tenth, and affected communities need new reports. Incremental updating (as in LightRAG) or splitting the corpus into stable and volatile parts reduces it; a plain vector index would only re-embed the changed chunks.

</details>

<details>
<summary><strong>Q5.</strong> Name two failure modes of LLM entity extraction that damage a graph silently.</summary>

Duplicate entities under different names, which split a hub and cut paths, and invented relations the text never stated, which create false paths. Mitigations: a canonical-name policy with alias tests, evidence quotes on every relation, and a sampled precision and recall check.

</details>

<details>
<summary><strong>Q6.</strong> With the block 2 parameters, roughly how many vector queries equal the index premium, and how many equal one global query?</summary>

About 1,666 for the index (9.33 against 0.0056 per query) and about 104 for one global query (0.5821 against 0.0056). If you change any parameter, rerun the block rather than trusting these.

</details>

## Further reading

- Edge et al., [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130), arXiv 2404.16130, v1 24 April 2024, v2 19 February 2025. Opened 5 October 2026 (abstract page).
- Microsoft Research, [GraphRAG: Unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/), 13 February 2024.
- Guo et al., [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779), arXiv 2410.05779, October 2024, revised April 2025 (abstract page only).
- On this site: [RAG basics](/docs/genai/rag), [knowledge graphs](/docs/theory/nlp/knowledge-graphs), [neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking), [enterprise document QA](/docs/senior/design-enterprise-document-qa), [the Enterprise RAG project](/docs/projects/enterprise-rag/improvements).

## Check yourself

- I can explain why multi-hop and whole-corpus questions defeat top-k vector retrieval.
- I can describe the GraphRAG index in five steps: extract, build graph, detect communities, summarise, store provenance.
- I can say when a local query beats a global query and what each one reads.
- I can read the cost model and name which parameter dominates the index charge and which dominates a global query.
- I can explain why 8 of 8 on hand-written paths is an upper bound.
- I can list the checks I would run on extraction quality before trusting a graph.
