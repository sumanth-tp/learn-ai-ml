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

:::tip Before you start
**You should already know**

- What RAG is: cut documents into chunks, embed them, return the nearest chunks to the model ([RAG basics](/docs/genai/rag)).
- What an embedding and cosine similarity are ([neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking)).
- What a graph is: nodes joined by edges ([knowledge graphs](/docs/theory/nlp/knowledge-graphs)).

**Reading time:** about 30 minutes, plus about 5 minutes to run the code.

**After this chapter you can**

- Say why a three-hop question defeats top-k vector search, with numbers.
- Build a small entity graph with `networkx`, follow relations through it and split it into communities.
- Decide, from a cost model, whether a graph index is worth its price for your questions.
:::

:::note Not from a lecture
Written for this site from the sources under Go deeper. The corpus, the extractor and every number below are produced by the code in this chapter, not taken from a paper.
:::

## In 30 seconds

Imagine a detective board with photos and red string. Ordinary RAG hands the model the few photos that look most like the question. GraphRAG first draws the string: who owns whom, who regulates whom. Then a question like "who oversees the owner of this company?" is answered by following the string, not by hoping that three separate sentences all look like the question. The price is that someone must read every document and draw the string first, and that costs real money.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Entity | A thing the text talks about | `Voltro`, `Helion Group` |
| Relation | A labelled link between two entities | `Helion Group` owns `Voltro` |
| Triple | One relation written as (head, relation, tail) | (`Helion Group`, `owns`, `Voltro`) |
| Hop | One step along a relation | Voltro to its owner is one hop |
| Multi-hop question | A question whose answer needs two or more hops | "Which regulator oversees the owner of Voltro?" |
| Community | A cluster of entities linked more tightly to each other than to the rest | The Calder Labs, Zerafil and Lumen group |
| Community report | A short summary written for one community | "Community around Calder Labs: 6 entities..." |
| Local query | A question that starts at named entities and walks outward | "Who owns Voltro?" |
| Global query | A question about the whole corpus, with no entity to start from | "What are the main themes?" |
| Provenance | The document an edge came from, kept so answers can be traced | Edge owns, from document 2 |

## The idea in plain words

Ordinary RAG cuts documents into chunks, embeds them and returns the chunks nearest to the question. That works when the answer sits inside one or two chunks that share words and meaning with the question.

Two kinds of question break it.

1. **Multi-hop questions.** "Which regulator oversees the company that owns the truck lessor of Delmar Freight?" The facts live in three sentences: Delmar leases trucks from Quarry Fleet, Quarry Fleet is owned by Helion Group, Nordic Safety Board regulates Helion Group. No single sentence resembles the question, and the sentence with the final answer never mentions Delmar.
2. **Global questions.** "What are the main themes across these documents?" There is no passage to match. The answer is a property of the whole collection.

GraphRAG attacks both with one build step. A language model reads every chunk and writes down entities and relations. Those become a graph. A community-detection algorithm splits the graph into clusters, and a model writes a short report for each cluster. At question time a local query starts from the entities named in the question and walks outward. A global query reads the community reports and merges what they say.

The graph is not free. Every chunk costs at least one generation call at index time, and every edit to the corpus means extracting again. The section "When is it worth the cost?" turns that into numbers.

<Infographic src="/img/rag-adv/graphrag-and-knowledge-graphs-pipeline.svg" alt="Documents are extracted into an entity graph, split into communities and summarised; local questions walk relations from a seed entity, global questions read every community report" caption="Look at the top row first: everything there happens once, before any question. The two boxes below it are the two ways to ask. The counts are printed by the code blocks." />

## Worked example, step by step

Take only three of the 24 sentences and one question. Documents are numbered from 0.

| Doc | Sentence |
| --- | --- |
| 2 | Helion Group owns Voltro. |
| 3 | Nordic Safety Board regulates Helion Group. |
| 1 | Voltro supplies battery cells to Kestrel Motors. |

Question: *Which regulator oversees the owner of Voltro?*

1. **Extract triples.** Doc 2 gives (`Helion Group`, `owns`, `Voltro`). Doc 3 gives (`Nordic Safety Board`, `regulates`, `Helion Group`). Doc 1 gives (`Voltro`, `supplies`, `Kestrel Motors`). Each triple remembers its document number.
2. **Find the seed.** The question names Voltro, so start at the node `Voltro`.
3. **Hop 1, who owns it?** Look for `owns` edges that point into Voltro. One exists, from Helion Group. We used doc 2.
4. **Hop 2, who regulates that owner?** Look for `regulates` edges pointing into Helion Group. One exists, from Nordic Safety Board. We used doc 3.
5. **Answer with evidence.** The answer is Nordic Safety Board, and the evidence is exactly documents 2 and 3. Nothing else was read.

A vector search has no step 3. It ranks all 24 sentences by similarity to the whole question and takes the top few. For this question the two needed documents rank 1st and 3rd, so asking for the top 2 misses doc 3. Block 3 below reproduces that.

<Infographic src="/img/rag-adv/graphrag-and-knowledge-graphs-worked-example.svg" alt="Three triples drawn as a small graph, with the question walked from Voltro to Helion Group to Nordic Safety Board and the documents used at each step" caption="Follow the numbered arrows from the left. Each hop uses one relation type and one document, so the answer comes with its evidence." />

Now the cost side, by hand, because the cost is the reason not to build a graph by default. A corpus of 1,000,000 tokens cut into 600-token chunks gives 1,666 chunks. Extraction plus one "did you miss anything?" pass is 2 calls per chunk, so 3,332 calls. Each call reads about 1,000 tokens (chunk plus instructions) and writes about 300. With the synthetic prices used in this chapter (1.0 per million tokens read, 4.0 per million written), that is 3.33 for reading and 4.00 for writing, so 7.33 units. Summaries for 400 communities add 2.00. The index costs 9.33 units. One ordinary vector query costs 0.0056. The index is a fixed charge worth about 1,666 vector queries. Block 7 prints the same numbers.

## How it works

### Extraction: text to triples

A triple is `(head, relation, tail)`. In a real pipeline a model with a structured-output prompt produces them, usually after a second pass that asks what it missed, often called gleaning. Two failure modes matter more than any other. The model names the same entity two ways ("Helion", "Helion Group"), and it invents a relation the text does not state. Both silently corrupt the graph. So production systems merge duplicate entities and keep, on every edge, a pointer to the source chunk.

The code below uses one regular expression per sentence shape instead of a model. This is a **deterministic stand-in**, chosen so the chapter runs offline and the numbers are exact. It teaches the graph mechanics. It says nothing about how accurate a model's extraction would be.

### The graph and its provenance

Entities become nodes and relations become labelled edges. Each edge carries the id of the document it came from, which is what lets an answer cite evidence. Several sentences can add edges around one entity, so hubs appear. Hubs are where careless traversal goes wrong, as block 4 shows.

### Communities and reports

Community detection groups nodes so that edges inside a group are denser than edges between groups. The Louvain method, in `networkx`, moves nodes between groups to raise a score called modularity. In words: modularity is high when many more edges fall inside groups than a random wiring of the same nodes would put there. Real systems also build a hierarchy, communities of communities, so a report exists at several zoom levels. Each community gets a report. Here the report is a template stand-in; with a model it is a paragraph.

### Local and global queries

| | Local query | Global query |
| --- | --- | --- |
| Starts from | entities named in the question | no entity, the whole corpus |
| Reads | the neighbourhood: relations and nearby text | every community report (map), then merges partial answers (reduce) |
| Good for | "who owns the owner of X", multi-hop lookups | "what are the main themes", comparisons across the corpus |
| Cost per question | about one vector query plus some graph context | one call per report, so it grows with the number of communities |

In the code, "path following" is the local method in its purest form. Someone, normally a model, maps the question to a path of relation types such as owns then regulates. The graph executes the path and returns the answer and the documents it used. Here the mapping is written by hand, so the 8 out of 8 below is an **upper bound**: it assumes the question was understood perfectly. The honest cost of the method is how often a model gets that mapping wrong.

### When is it worth the cost?

Use a graph when at least one of these is true:

- Users ask relational questions whose evidence is spread across documents: ownership chains, dependencies, "who reports to whom".
- Users ask corpus-level questions: themes, trends, "summarise everything about X".
- Auditability matters and every claim must point to the edge and document that support it.

Skip it when questions are mostly lookups inside one document, when the corpus changes hourly, or when you have not yet tried the cheaper levers: better chunking, hybrid search, reranking and query rewriting (see [contextual retrieval and reranking](/docs/genai/rag-advanced/contextual-retrieval-and-reranking)).

## A real system that works this way

**Microsoft's GraphRAG.** The paper "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Edge and colleagues, first posted 24 April 2024, revised 19 February 2025) describes a two-stage index. A language model extracts an entity knowledge graph from the source documents, then pre-generates summaries for clusters of closely related entities. At question time each community summary yields a partial response and the partial responses are combined into the final answer. The authors argue that conventional RAG fails on questions such as "what are the main themes in the dataset?" because they are query-focused summarisation, not retrieval. For global questions over datasets in the 1 million token range they report substantial improvements over a conventional RAG baseline in the comprehensiveness and diversity of answers. The Microsoft Research announcement of 13 February 2024 makes the same argument and gives no cost figure, which is why the cost model here uses named, synthetic parameters.

**LazyGraphRAG.** A Microsoft Research post of 25 November 2024 attacks the cost problem directly. It states that LazyGraphRAG indexing costs are identical to vector RAG and 0.1 percent of full GraphRAG, and that on global queries it reaches comparable answer quality to GraphRAG Global Search at more than 700 times lower query cost. The trick is to defer language-model use from index time to query time. These are the vendor's own figures; I did not reproduce them.

**LightRAG** (Guo and colleagues, arXiv 2410.05779, October 2024, revised April 2025) builds graph structures into text indexing, retrieves at two levels (specific entities and broader themes), and updates incrementally so that new documents join the index without a rebuild. I read the abstract only and make no claim about its measured quality.

## Code you can run

Seven blocks, run in order. Blocks 1 to 6 build and query a 24-sentence corpus. Block 7 is a cost model. Blocks 1 to 5 hand a small file to the next block through your temporary folder, so run them in order. Libraries: `networkx` 3.6.1, `sentence-transformers` 6.1.0 (model `all-MiniLM-L6-v2`), `numpy` 2.5.3, Python 3.14, CPU only. If the Hugging Face Hub is slow once the model is cached, run with `HF_HUB_OFFLINE=1`.

The reader is not simulated. For each question the test is whether **every document a reader would need** is in the context it is given: either the top `k` from vector search, or exactly the documents a path walk touched.

### 1. The corpus

First we write the 24 one-sentence documents to a file the later blocks read. The corpus has four business areas (scooters, a drug, freight, payments) that touch each other through shared companies.

```python
import json
import tempfile
from pathlib import Path

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

Path(tempfile.gettempdir(), "ragadv_docs.json").write_text(json.dumps(DOCS))
print(len(DOCS), "one-sentence documents saved")
words = sum(len(d.split()) for d in DOCS)
print(f"{words} words in total, about {words / len(DOCS):.1f} words per document")
```

**Reading the output.** 24 documents of about 6.5 words each, 156 words in all. The corpus is tiny on purpose, so every number can be checked by hand.

**Line by line.**

- `DOCS` is the whole corpus; the list index is the document id used everywhere below.
- The JSON file is the hand-off to block 2.

### 2. Extract triples and build the graph

Next we turn each sentence into triples with one pattern per sentence shape, and add them to a `networkx` multigraph. A multigraph allows two different relations between the same pair, and every edge keeps its document id.

```python
import json
import re
import tempfile
from pathlib import Path

import networkx as nx

DOCS = json.loads(Path(tempfile.gettempdir(), "ragadv_docs.json").read_text())
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
for doc_id in (13, 14):
    print(doc_id, DOCS[doc_id], "->", extract(DOCS[doc_id]))
edges = [[h, t, d["relation"], d["doc"]] for h, t, d in graph.edges(data=True)]
Path(tempfile.gettempdir(), "ragadv_edges.json").write_text(json.dumps(edges))
```

**Reading the output.** 24 documents become 22 entities and 28 relations. Some sentences give two relations: "Kestrel Motors builds the Aster Scooter in Lisbon" gives `builds` and `located_in`. The two printed examples are the second and third hop of the Delmar question.

**Line by line.**

- `N` matches a capitalised name of one or more words, such as `Nordic Safety Board`.
- "is owned by" swaps head and tail so that `owns` always points from owner to owned.
- `key=(relation, doc_id)` lets two documents state the same relation without overwriting each other.

### 3. The vector baseline

Now the question set. Each question carries the documents a reader needs, the entity to start from and the relation path. First we measure plain vector search: for each `k`, how many questions have every needed document among the top `k`?

```python
import json
import tempfile
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DOCS = json.loads(Path(tempfile.gettempdir(), "ragadv_docs.json").read_text())
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
Path(tempfile.gettempdir(), "ragadv_questions.json").write_text(json.dumps([[q, sorted(g), s, p] for q, g, s, p in QUESTIONS]))

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
doc_vecs = encoder.encode(DOCS, normalize_embeddings=True)
question_vecs = encoder.encode([q[0] for q in QUESTIONS], normalize_embeddings=True)

print(f"{'documents given to the reader (k)':<36}" + "".join(f"{k:>5}" for k in range(2, 9)))
row = []
for k in range(2, 9):
    row.append(sum(q[1] <= set(np.argsort(-(doc_vecs @ question_vecs[i]))[:k].tolist()) for i, q in enumerate(QUESTIONS)))
print(f"{'questions with every needed document':<36}" + "".join(f"{v:>5}" for v in row) + f"   (of {len(QUESTIONS)})")

second = QUESTIONS[1]
order = np.argsort(-(doc_vecs @ question_vecs[1]))
print("\nDelmar question, the 6 documents vector search ranks first:")
for rank, doc in enumerate(order[:6], 1):
    print(f"  {rank}. doc {doc:>2} {'NEEDED' if doc in second[1] else '      '} {DOCS[doc]}")
```

**Reading the output.** At `k = 4` only 2 of 8 questions are fully supported. Even at `k = 8`, a third of the whole corpus, only 6 are. The Delmar question is the worst case. Its needed documents rank 3rd, 16th and 17th, so you must read 17 of 24 documents to get all three. Worse, the top result is "Customs Directorate regulates Delmar Freight", a regulator, but the regulator of the wrong company. A reader who sees only the top few documents will answer confidently and wrongly.

**Line by line.**

- `q[1] <= set(...)` is a subset test: every needed document must be present.
- Vectors are normalised, so the dot product `doc_vecs @ question_vecs[i]` is cosine similarity.

### 4. Follow the relations, and see why hubs need care

Here we execute each relation path against the graph and compare the documents touched with the documents needed. Then we try the lazy alternative: take everything within two steps of the seed, ignoring relation types.

```python
import json
import tempfile
from pathlib import Path

import networkx as nx

tmp = Path(tempfile.gettempdir())
DOCS = json.loads((tmp / "ragadv_docs.json").read_text())
QUESTIONS = json.loads((tmp / "ragadv_questions.json").read_text())
graph = nx.MultiDiGraph()
for head, tail, relation, doc in json.loads((tmp / "ragadv_edges.json").read_text()):
    graph.add_edge(head, tail, key=(relation, doc), relation=relation, doc=doc)


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


correct = 0
for question, gold, start, plan in QUESTIONS:
    answers, docs = follow(start, [tuple(step) for step in plan])
    correct += docs == set(gold)
    print(f"{start:<20} {len(plan)} hops -> {sorted(answers)[0]:<26} documents {sorted(docs)}")
print(f"path following found exactly the needed documents for {correct}/{len(QUESTIONS)} questions")

undirected = graph.to_undirected()
for seed in ("Delmar Freight", "Helion Group"):
    near = nx.single_source_shortest_path_length(undirected, seed, cutoff=2)
    docs = {d["doc"] for h, t, d in graph.edges(data=True) if h in near and t in near}
    print(f"blind 2-hop neighbourhood of {seed}: {len(near)} of {graph.number_of_nodes()} entities, {len(docs)} of {len(DOCS)} documents")
```

**Reading the output.** Path following touches exactly the needed documents for all 8 questions, and always 2 or 3 of them, because the graph keeps the chain as edges. The blind two-step neighbourhood is a different story. Around `Delmar Freight` it pulls in 12 of 22 entities and 11 of 24 documents, about half the corpus, for a question that needs 3. Around `Helion Group` it is smaller (6 entities, 5 documents) in this tiny graph, but in a real corpus with thousands of entities a hub's neighbourhood explodes. Following a typed path is what keeps the context small.

**Line by line.**

- `in_edges` follows a relation backwards (who points at this node), `out_edges` forwards.
- `plan` is the relation path a model would produce from the question; here it is hand-written, so 8 out of 8 is an upper bound.
- `to_undirected` plus `cutoff=2` is the blind neighbourhood.

### 5. Communities and reports

Now the global side. We merge parallel edges into weights, run Louvain with a fixed seed, and write a template report for each community.

```python
import json
import tempfile
from collections import Counter
from pathlib import Path

import networkx as nx

tmp = Path(tempfile.gettempdir())
graph = nx.MultiDiGraph()
for head, tail, relation, doc in json.loads((tmp / "ragadv_edges.json").read_text()):
    graph.add_edge(head, tail, key=(relation, doc), relation=relation, doc=doc)

undirected = nx.Graph()
for head, tail, data in graph.edges(data=True):
    if undirected.has_edge(head, tail):
        undirected[head][tail]["weight"] += 1
    else:
        undirected.add_edge(head, tail, weight=1)

communities = sorted(nx.community.louvain_communities(undirected, weight="weight", seed=0), key=lambda c: (-len(c), sorted(c)[0]))
member_of = {node: i for i, c in enumerate(communities) for node in c}
print(f"Louvain found {len(communities)} communities (sizes {[len(c) for c in communities]})")
print(f"modularity {nx.community.modularity(undirected, communities, weight='weight'):.3f}")

doc_community = {data["doc"]: member_of[head] for head, tail, data in graph.edges(data=True)}


def report(index):
    members = communities[index]
    hub = max(sorted(members), key=lambda n: undirected.degree(n))
    relations = Counter(d["relation"] for h, t, d in graph.edges(data=True) if h in members and t in members)
    docs = sorted(d for d, c in doc_community.items() if c == index)
    top = ", ".join(f"{name} x{count}" for name, count in relations.most_common(3))
    return {"hub": hub, "docs": docs, "text": f"Community around {hub}: {len(members)} entities; main relations {top}."}


reports = [report(i) for i in range(len(communities))]
for i, r in enumerate(reports):
    print(f"C{i}: {r['text']}  members {sorted(communities[i])}")
(tmp / "ragadv_communities.json").write_text(json.dumps({"doc_community": {str(k): v for k, v in doc_community.items()}, "docs": [r["docs"] for r in reports]}))
```

**Reading the output.** Louvain finds 5 communities of sizes 6, 5, 4, 4 and 3, with modularity 0.561 (a rule of thumb reads values above about 0.3 as real structure). The groups follow the story: the Calder Labs group holds the drug business, the Kestrel group holds scooters, the Paylink group holds payments, the Helion group holds ownership, and a small Delmar group holds freight. The template reports are thin ("main relations develops x1, licenses x1"); a model would write a paragraph. The point here is the structure the reports sit on.

**Line by line.**

- `weight` counts how many documents link the same pair, so repeated facts pull entities together.
- `seed=0` fixes Louvain's random node order so the result is repeatable.
- `doc_community` gives each document the community of the entity at the head of its edge, so we can ask which communities a set of documents touches.

### 6. A global question

Last we ask a question with no entity in it, and count how many communities each method touches.

```python
import json
import tempfile
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

tmp = Path(tempfile.gettempdir())
DOCS = json.loads((tmp / "ragadv_docs.json").read_text())
saved = json.loads((tmp / "ragadv_communities.json").read_text())
doc_community = {int(k): v for k, v in saved["doc_community"].items()}
communities = len(saved["docs"])

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
doc_vecs = encoder.encode(DOCS, normalize_embeddings=True)
question = "What are the main themes across these documents?"
q_vec = encoder.encode([question], normalize_embeddings=True)[0]
print(f"global question: {question}")
for k in (3, 5, 8):
    top = np.argsort(-(doc_vecs @ q_vec))[:k]
    covered = {doc_community[int(d)] for d in top}
    print(f"vector search, top {k} documents: touches {len(covered)} of {communities} communities")
reads = sum(len(d) for d in saved["docs"])
print(f"map-reduce over community reports: {communities} reports read, touches {communities} of {communities} communities, documents behind them {reads} of {len(DOCS)}")
```

**Reading the output.** Top-3 and top-5 vector search both touch 2 of 5 communities, and top-8 touches 3. Reading all 5 reports touches 5 of 5, and the reports sit on all 24 documents. A "themes" question has no passage to match, so top-k returns the sentences that look most like the word "themes", a biased sample of whatever happens to be nearby.

**Line by line.**

- `covered` is the set of community ids behind the retrieved documents.
- `reads` adds up the documents behind all reports, which is the whole corpus.

### Try the first six blocks in the lab

Pick a question, give vector search 2 to 8 documents, then switch to path following or to community reports. Edges whose document was given are drawn thick with the relation name. Needed documents that are missing are dashed red.

<GraphRetrievalLab />

**What each control does.**

- **question**: one of the eight multi-hop questions or the global question.
- **method**: vector search (top `k`), path following (exactly the needed documents) or community reports (all documents).
- **k**: how many documents vector search hands over, 2 to 8. It does nothing for the other methods.
- **show data**: the table of all 24 documents with the vector rank for the chosen question.

**Try it yourself.**

1. Leave question 2 (the Delmar question) on vector search with `k = 4`. The status shows 1 of 3 needed documents found, and the line under the plot reads "completes 2 of 8", the same as block 3. Raise `k` to 8: still not complete, because two needed documents rank 16th and 17th.
2. Switch method to path following. The three needed documents light up and nothing else does. This is why the answer can cite its evidence.
3. Pick the global question and set vector search to `k = 5`. It touches 2 of 5 communities. Switch to community reports and all 5 light up.

<Infographic src="/img/rag-adv/graphrag-and-knowledge-graphs-vector-vs-graph.svg" alt="Bars comparing vector search at k from 2 to 8 with path following on 8 questions, community coverage for a global question, and a cost table" caption="Left: how many of 8 multi-hop questions each method fully supports. Right: communities touched for the global question. Bottom: what the graph costs. Counts come from blocks 3 to 7." />

### 7. The cost model

Every price and size here is a named parameter on **synthetic** numbers (1.0 per million tokens read, 4.0 per million written). Replace them with your provider's real prices and your corpus before deciding anything.

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

**Reading the output.** The index costs 9.33 units: 7.33 for extraction and 2.00 for summaries. A vector index costs no generation at all. A local graph query costs 0.0071 against 0.0056 for a vector query, a small premium. One global query costs 0.5821, as much as 104 vector queries, because it reads every report. After 1,000 queries vector RAG has cost 6 units and a graph with local queries only 16. If one query in a hundred is global, the graph reaches 22. If 10 percent of the corpus changes each month, re-extraction costs 0.73 units a month.

**Line by line.**

- `GLEANINGS = 1` adds one extra extraction pass per chunk, which doubles `extract_calls`.
- `global_query` has two parts: reading every report (the map step), then merging the partial answers (the reduce step).
- The last lines show that edits, not queries, can dominate the bill when the corpus is volatile.

## Production snippets (not run here)

An extraction prompt for a real model. Not run in this environment, because it needs a hosted or local model of at least a few billion parameters; SmolLM2-135M is far too small to follow it. The `graphrag` package itself (version 3.2.0 on PyPI, uploaded 23 September 2026) declares `Python >=3.11,<3.14`, and this chapter's environment is Python 3.14, so I did not install or run it.

```text
Read the passage. Return JSON with two lists.
entities: objects with name, type (person, company, product, agency, place).
relations: objects with head, relation, tail, evidence (a short quote).
Use only facts the passage states. Use one canonical name per entity.
Return nothing for facts you are unsure of.
```

Not run in this environment. A second pass ("list any entities or relations you missed") roughly doubles extraction calls, which is the `GLEANINGS` parameter in block 7.

## Designing with it

- **Start with questions, not graphs.** Collect twenty real user questions and label each local, global or ordinary lookup. If fewer than a fifth are multi-hop or global, a graph will not pay back.
- **Evaluate extraction first.** Sample fifty chunks, extract by hand, and score the model's entities and relations for precision and recall. Wrong edges produce confident wrong answers, which are worse than a missed passage.
- **Resolve entities.** Decide a canonical name policy and test it with a list of known aliases. Unmerged duplicates split a hub in two and cut the path.
- **Keep provenance on every edge.** An answer that cannot point to a document is not auditable, and you cannot debug it.
- **Combine, do not replace.** Practical systems route: ordinary lookups to hybrid search ([the enterprise document-QA design](/docs/senior/design-enterprise-document-qa) shows the funnel), relational and corpus-level questions to the graph.
- **Budget the global query.** It scales with the number of communities. Cache answers to recurring questions and use higher levels of the hierarchy for coarse questions.
- **Plan updates.** Decide up front whether you rebuild nightly, patch incrementally, or accept staleness.

If the surrounding system is an agent that decides when to retrieve, see [agents in LangChain v1](/docs/genai/langchain-advanced/create-agent): a graph query is just another tool the agent can call.

## Where this stands in 2026

:::info Industry view
The pattern is stable: extract, cluster, summarise, query locally or globally. What moved is cost. Microsoft's own LazyGraphRAG post (25 November 2024) moves the language-model work from index time to query time and reports indexing at the cost of vector RAG; LightRAG's abstract names incremental update as its fix for rebuilding. The reference `graphrag` package was at version 3.2.0 on 7 October 2026. I checked the papers, the two Microsoft posts and the package index, not independent benchmarks, so I cannot tell you which implementation is best today or how widely graphs are deployed. Run your own twenty-question test. What does not depend on tooling is the trade: you pay at index time (or query time) to answer relational and corpus-level questions, and you pay again whenever the corpus moves.
:::

## Common mistakes

- **Building the graph before looking at the questions.** It feels like the advanced option, so it must be better. If most questions are single-document lookups, a graph adds cost and failure modes for no gain. Label twenty real questions first.
- **Reading 8 out of 8 as accuracy.** The relation path was written by hand. In production a model must map every question to a path, and that step fails. Measure it separately.
- **Skipping entity resolution.** "Helion" and "Helion Group" look like the same thing to a person. To the graph they are two nodes and the chain breaks between them. Keep an alias list and test it.
- **Expanding the neighbourhood blindly.** "Give the model everything within two steps" is easy to code. Around a hub it pulls in half the corpus (block 4). Follow typed relations instead.
- **Running a global query on every request.** It answers the hard question, so it feels like the safe default. It costs about 104 vector queries here. Route by question type and cache.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> Why can a vector search over short documents fail a three-hop question even when every needed document is in the corpus?</summary>

The ranking compares each document to the question separately. The question shares words with the first hop only, so later-hop sentences rank below unrelated documents that mention the same entity. In block 3, vector search at `k = 4` completes 2 of 8 questions, and the Delmar question needs `k = 17` of 24 documents.

</details>

<details>
<summary><strong>Q2 (Medium).</strong> Path following scored 8 of 8. Why is that not evidence that GraphRAG is 100 percent accurate?</summary>

The mapping from question to relation path was written by hand, standing in for a model. A real pipeline must also extract the graph correctly and map the question correctly. The 8 of 8 is an upper bound that isolates what the graph structure can do.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> A user asks for the main complaints across 40,000 support tickets. Why does top-k vector search give a poor answer, and what does a community report add?</summary>

The question has no matching passage, so top-k returns the few tickets most similar to the word "complaints", a biased sample. A report per community summarises each cluster in advance, so a map-reduce over reports covers the whole collection. In block 6, top-5 vector search touched 2 of 5 communities and the reports touched all 5.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> Your corpus is 10 percent re-written every month. What happens to the cost, and which design choice reduces it?</summary>

The extraction share of the index is paid again for the changed tenth, and affected communities need new reports. Incremental updating (as in LightRAG) or splitting the corpus into stable and volatile parts reduces it. A plain vector index only re-embeds the changed chunks. With the block 7 parameters the monthly re-extraction is 0.73 units.

</details>

<details>
<summary><strong>Q5 (Easy).</strong> Name two failure modes of language-model entity extraction that damage a graph silently.</summary>

Duplicate entities under different names, which split a hub and cut paths, and invented relations the text never stated, which create false paths. Mitigations: a canonical-name policy with alias tests, an evidence quote on every relation, and a sampled precision and recall check.

</details>

<details>
<summary><strong>Q6 (Medium).</strong> With the block 7 parameters, roughly how many vector queries equal the index cost, and how many equal one global query?</summary>

About 1,666 for the index (9.33 against 0.0056 per query) and about 104 for one global query (0.5821 against 0.0056). If you change any parameter, rerun the block rather than trusting these.

</details>

<details>
<summary><strong>Q7 (Stretch).</strong> In block 4 the blind two-step neighbourhood of Delmar Freight touches 11 of 24 documents, but that of Helion Group only 5. Without running anything, which would you expect to be larger in a corpus of a million documents, and what rule would you add?</summary>

Hubs, entities with very high degree, grow fastest, and a two-step neighbourhood of a hub is roughly the sum of its neighbours' neighbourhoods, so it can reach a large fraction of the graph. Rules that help: follow only the relation types the question implies (as path following does), cap the number of neighbours per step by a relevance score, and skip or down-weight nodes above a degree threshold.

</details>

## Go deeper

All sources opened on 7 October 2026.

- Edge et al., [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130), arXiv 2404.16130, v1 24 April 2024, v2 19 February 2025 (abstract page).
- Microsoft Research, [GraphRAG: Unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/), 13 February 2024.
- Microsoft Research, [LazyGraphRAG: Setting a new standard for quality and cost](https://www.microsoft.com/en-us/research/blog/lazygraphrag/), 25 November 2024. Vendor figures, not reproduced here.
- Guo et al., [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779), arXiv 2410.05779, submitted 8 October 2024, v3 28 April 2025 (abstract page only).
- Blondel et al., [Fast unfolding of communities in large networks](https://arxiv.org/abs/0803.0476), arXiv 0803.0476, 2008 (the Louvain method).
- NetworkX documentation, [`louvain_communities`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html). The docs show 3.7; the code here ran on 3.6.1.
- [`graphrag` on PyPI](https://pypi.org/project/graphrag/), version 3.2.0, uploaded 23 September 2026, requires Python 3.11 to 3.13.

## Check yourself

- I can explain why multi-hop and whole-corpus questions defeat top-k vector retrieval, using the Delmar numbers.
- I can describe the GraphRAG index in five steps: extract, build graph, detect communities, summarise, store provenance.
- I can walk a three-hop question through a graph by hand and name the documents it uses.
- I can say when a local query beats a global query and what each one reads.
- I can read the cost model and name which parameter dominates the index charge and which dominates a global query.
- I can explain why 8 of 8 on hand-written paths is an upper bound.
- I can list the checks I would run on extraction quality before trusting a graph.

## Where to go next

Next in this series: [long context versus RAG](/docs/genai/rag-advanced/long-context-vs-rag), which asks when you can skip retrieval altogether and put the whole corpus in the prompt. For the data model behind graphs, see [knowledge graphs, ontologies and GraphRAG](/docs/theory/nlp/knowledge-graphs).
