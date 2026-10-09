---
id: rag-adv-contextual
title: "Contextual Retrieval and Reranking"
sidebar_label: "4 · Contextual retrieval and reranking"
sidebar_position: 4
slug: /genai/rag-advanced/contextual-retrieval-and-reranking
description: "Two repairs for a retrieval pipeline that returns plausible but wrong chunks: give every chunk its missing context before indexing, and rescore a short list with a cross-encoder, including what happens when you fine-tune one."
tags: [contextual-retrieval, reranking, cross-encoder, late-chunking, colbert, bm25, rag]
---

import Infographic from '@site/src/components/Infographic';
import RerankLab from '@site/src/components/viz/RerankLab';

**In one line.** Retrieval fails in two separate places, so it has two separate repairs: chunks that lost their context are fixed before indexing, and a shortlist that is in the wrong order is fixed after retrieval by a slower model that reads the query and each candidate together.

:::tip Before you start
**You should already know**

- How BM25 and dense retrieval work and how their rankings are fused ([neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking)).
- How a RAG pipeline cuts documents into chunks ([RAG basics](/docs/genai/rag), [text splitters](/docs/genai/text-splitters)).
- How reciprocal rank fusion merges two result lists, and how it looks in a working project ([the Research Copilot capstone](/docs/genai/capstone)).

**Reading time:** about 35 minutes, plus about 20 minutes to run the code (the real-data blocks embed 25,000 chunks and train a small model).

**After this chapter you can**

- Explain, with measured numbers, when adding context to a chunk helps and when it does nothing.
- Choose a reranking depth and say what it costs in latency and what it can never fix.
- Fine-tune a small cross-encoder on labelled pairs and test whether it beat the general model.
:::

:::note Not from a lecture
Written for this site from the sources under Go deeper. Every number below is printed by the code in this chapter. The funnel of BM25, dense retrieval, fusion and reranking is built in the [enterprise document-QA design](/docs/senior/design-enterprise-document-qa); this chapter does not repeat it, and instead asks what to add around it.
:::

## In 30 seconds

You ask a librarian for "the report on revenue growth at Brightwell Energy in the third quarter". Every page in the archive says "revenue grew by 12 percent over the previous quarter", and none says which company. The librarian cannot know which page you mean. That is the first problem, and the fix is to stamp each page with its company and quarter before filing. The second problem is order. The librarian hands you twenty pages, quickly sorted, and the best one is seventh. A careful reader who reads your request and each page side by side will sort them better, but is too slow for the whole archive, so you let them read only the twenty.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Chunk | A piece of a document stored and searched on its own | Two sentences from a report |
| Contextual retrieval | Prefixing each chunk with a short note on where it came from before indexing | "Brightwell Energy, Q3 2025 report." |
| Late chunking | Reading the whole document through the embedding model first, then cutting the token vectors into chunk vectors | One pass over an abstract, three chunk vectors out |
| Multi-vector retrieval | Keeping one vector per token instead of one per passage, and scoring by best token matches | ColBERT-style MaxSim |
| First stage | The cheap search over the whole corpus | BM25 plus dense plus fusion |
| Reranker | A slower model that rescores a short list | A cross-encoder |
| Cross-encoder | A model that reads the query and one document together and outputs a relevance score | MS MARCO MiniLM, 22.7 million parameters |
| Shortlist depth | How many first-stage results the reranker sees | 20 |
| nDCG@10 | A score for the top ten that rewards relevant documents near the top | 1.0 is a perfect order |
| Hard negative | A wrong document that looks right, used as a training example | A BM25 top-3 result that is not relevant |

## The idea in plain words

A retrieval pipeline can fail in two places. Either the right chunk never reaches the top of the first-stage results, or it reaches the shortlist but sits below wrong ones. Check which one is happening before you touch anything.

**Failure 1: the chunk lost its context.** Cutting a document into chunks removes the title, the company, the date, the section. A chunk that says "the company raised its guidance by 3 percent" is a fine sentence and a useless search target. Anthropic's September 2024 write-up on contextual retrieval proposes asking a model to write one or two sentences placing each chunk in its document, and prepending that to the chunk before embedding and before BM25 indexing. This costs one model call per chunk, once.

**Failure 2: the shortlist is in the wrong order.** First-stage retrievers compare a query with a document that was encoded without ever seeing the query. A cross-encoder reads both together, so it can notice that the query says "decreases" and the document says "increases". It is far too slow to run over the whole corpus, so you run it over the top 20 or so.

<Infographic src="/img/rag-adv/contextual-retrieval-and-reranking-idea.svg" alt="Two repairs: add a context note to each chunk at index time, and rescore a shortlist with a cross-encoder at query time, with the measured scores for each" caption="Read the top band first: it happens once, at index time. The bottom band happens on every query. The shortlist is a ceiling: the reranker can reorder what it is given and nothing else." />

## Worked example, step by step

A corpus of annual-report chunks, 24 of which are about revenue. Question: *How fast did revenue grow at Brightwell Energy in Q3 2025?* We count how many query words appear in a chunk, which is the heart of BM25 without the weighting.

1. **Bare chunk.** "Revenue grew by 12% over the previous quarter." It shares one query word with the question: *revenue*. So do the other 23 revenue chunks. Twenty-four chunks tie at 1.
2. **Pick the winner among ties.** With 24 tied candidates, the right chunk is first with chance 1 in 24, which is 0.042.
3. **Add context.** Prefix each chunk with its company and quarter: "Brightwell Energy, Q3 2025 report. Revenue grew by 12% over the previous quarter."
4. **Count again.** The right chunk now shares five words: *brightwell, energy, q3, 2025, revenue*. The same company in Q2 shares four, another company in Q3 shares three, and an unrelated one shares two.
5. **Read off the ranking.** The right chunk wins with no tie. In block 1 below, BM25 hit@1 goes from 0.042 to 1.000.

<Infographic src="/img/rag-adv/contextual-retrieval-and-reranking-worked-example.svg" alt="A query about Brightwell Energy in Q3 2025 scored against 24 identical-looking plain chunks that all tie, and against context-prefixed chunks where the right one scores 5 and wins" caption="Left: nothing separates the 24 chunks. Right: five shared words against four, three and two. This is a synthetic corpus, built to show the mechanism; block 2 shows how much of it survives on real text." />

## How it works

### Contextual retrieval: what to write and where to put it

The note goes in front of the chunk text, and the combined string is what you embed and what BM25 indexes. Anthropic's published prompt asks the model to "give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk", and answer with only that. The note must be specific (names, dates, section) and short, 50 to 100 tokens.

Anthropic reports that contextual embeddings reduced the top-20 failed-retrieval rate by 35 percent (5.7 percent to 3.7 percent), contextual embeddings plus contextual BM25 by 49 percent (to 2.9 percent), and adding reranking by 67 percent (to 1.9 percent). Those are the vendor's numbers on its own mix of datasets. I did not reproduce them; blocks 1 and 2 measure the mechanism on data you can rerun. The same post puts the one-time cost of generating the notes at 1.02 dollars per million document tokens when prompt caching is used, and advises that a knowledge base under 200,000 tokens can simply go into the prompt, which is the subject of [the previous chapter](/docs/genai/rag-advanced/long-context-vs-rag).

A cheaper variant uses only metadata you already have: title, section heading, author, date. Block 2 tests exactly that on real text.

### Late chunking: context from the model instead of from a prompt

Late chunking (Günther and colleagues, arXiv 2409.04701, current version July 2025) keeps the document whole while the embedding model reads it. Every token's vector is then computed with attention over the full document, and each chunk vector is the mean of its own tokens' vectors. Chunks therefore carry information from their neighbours without any extra model calls. It needs an embedding model that can read the whole document, which means a long-context embedding model. The small model used in this chapter reads at most 512 tokens, so block 3 is a mechanism test, not a fair test of the method.

### Multi-vector retrieval: scoring token by token

ColBERT (Khattab and Zaharia, SIGIR 2020, arXiv 2004.12832) keeps one vector per token of every document and scores a query by taking, for each query token, its best match among the document tokens, and summing. This is called MaxSim. In words: a document scores high when every word of the query finds a close partner somewhere in it, even if the document as a whole sounds different. The abstract reports effectiveness competitive with BERT rerankers while being two orders of magnitude faster. The price is storage: one vector per token. Block 4 shows both sides on three tiny documents.

### Reranking: what a cross-encoder adds and what it cannot

A first-stage retriever encodes the query and the document separately and compares the results. A cross-encoder feeds both into one model, so every query word can attend to every document word. That is more accurate and much slower: you must run the model once per query-document pair. So rerankers work on a shortlist, and the shortlist sets a ceiling. If the relevant document is not among the top 20, no reranker can put it first.

Three decisions follow.

- **Depth.** More depth raises the ceiling and the latency together. Block 6 measures both.
- **Model.** A general model trained on web search (here MS MARCO) may not match your domain. Block 8 fine-tunes one and block 9 tests it.
- **Training signal.** Rerankers are trained with one of three losses. Pointwise: score each pair as relevant or not (binary cross-entropy, used below). Pairwise: the relevant document should outscore a wrong one by a margin. Listwise: the relevant document should win a softmax over the whole list. Block 4 prints a pairwise and a listwise loss for one list so you can see how differently they weigh the same scores.

## A real system that works this way

**Anthropic, Contextual Retrieval** (19 September 2024) is the source of the context-prefix recipe, the prompt and the cost and quality figures above.

**Qwen3 Embedding and Reranker** (Alibaba, arXiv 2506.05176, submitted 5 June 2025) is an open example of the other repair, trained rerankers: its abstract describes embedding and reranking models in three sizes, 0.6B, 4B and 8B parameters, trained with a mix of large-scale unsupervised pre-training and supervised fine-tuning, with the Qwen3 models themselves used to synthesise training data. I read the abstract only and make no claim about its scores.

**BEIR** (Thakur and colleagues, arXiv 2104.08663) is the reason to be careful: across 18 datasets it found BM25 a robust baseline and re-ranking and late-interaction models best on average, at high computational cost. "On average" hides datasets where they do not win, and block 6 lands on one of them.

## Code you can run

Nine blocks. Blocks 1 to 4 are about chunks, blocks 5 to 9 are about reranking. Blocks 2 and 3 share chunk vectors through a file in your temporary folder, and blocks 5 to 9 share candidate lists, scores and a trained model the same way, so run them in order. The real-data blocks use SciFact from the BEIR collection (5,183 scientific abstracts, 300 test claims with labelled evidence abstracts; Wadden and colleagues, 2020), `sentence-transformers` 6.1.0 with `all-MiniLM-L6-v2`, and the cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2`, on CPU, with `transformers` 5.18.0 and `torch` 2.14.1. With the models cached, run with `HF_HUB_OFFLINE=1`. Blocks 2, 3, 5, 6 and 9 take a few minutes each and block 8 about three.

### 1. Context helps when chunks cannot name their parent

First a synthetic corpus built to show the mechanism: six invented companies, four quarters, five report topics, one chunk per combination, none of them naming the company. We search with BM25 (written with a sparse matrix in a few lines) and with dense vectors, once on the bare chunks and once with a template note in front. The template is a deterministic stand-in for the note a model would write.

```python
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

COMPANIES = ["Aldermoor Foods", "Brightwell Energy", "Corvane Pharma", "Delmere Logistics", "Eskarn Telecom", "Fenwick Retail"]
QUARTERS = ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]
TOPICS = [
    ("Revenue grew by {a}% over the previous quarter, helped by steady demand.", "How fast did revenue grow"),
    ("The company added {a} employees during the quarter and closed two offices.", "How many employees were added"),
    ("Capital spending reached {a} million, mostly on new equipment.", "How much capital spending was there"),
    ("Management expects growth of about {a}% next quarter, with no change to guidance.", "What growth does management expect next"),
    ("The main risk is supplier concentration, with {a}% of purchases from one vendor.", "What is the main supplier risk"),
]


def bm25_scorer(docs, k1=1.5, b=0.75):
    vec = CountVectorizer(token_pattern=r"[a-z0-9]+")
    tf = vec.fit_transform(docs).tocoo().astype(float)
    df = np.bincount(tf.col, minlength=tf.shape[1])
    idf = np.log(1 + (tf.shape[0] - df + 0.5) / (df + 0.5))
    dl = np.bincount(tf.row, weights=tf.data, minlength=tf.shape[0])
    w = idf[tf.col] * tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / dl.mean()))
    matrix = csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)
    return lambda query: matrix @ (vec.transform([query]).toarray().ravel() > 0).astype(float)


rng = np.random.default_rng(3)
chunks, queries = [], []
for company in COMPANIES:
    for quarter in QUARTERS:
        for template, question in TOPICS:
            chunks.append((company, quarter, template.format(a=int(rng.integers(2, 40)))))
            queries.append((f"{question} at {company} in {quarter}?", len(chunks) - 1))
print(f"{len(chunks)} chunks, {len(queries)} questions, no chunk text names a company")

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
q_vecs = encoder.encode([q for q, _ in queries], normalize_embeddings=True)
print(f"{'index built from':<22}{'method':<8}{'hit@1':>7}{'hit@5':>7}")
for label, docs in (("plain chunk", [c[2] for c in chunks]), ("context + chunk", [f"{c[0]}, {c[1]} report. {c[2]}" for c in chunks])):
    d_vecs, score_bm25 = encoder.encode(docs, normalize_embeddings=True), bm25_scorer(docs)
    for method in ("BM25", "dense"):
        orders = [np.argsort(-(score_bm25(q) if method == "BM25" else d_vecs @ q_vecs[i])) for i, (q, _) in enumerate(queries)]
        h1 = np.mean([o[0] == gold for o, (_, gold) in zip(orders, queries)])
        h5 = np.mean([gold in o[:5] for o, (_, gold) in zip(orders, queries)])
        print(f"{label:<22}{method:<8}{h1:7.3f}{h5:7.3f}")
```

**Reading the output.** On bare chunks BM25 finds the right chunk first 4.2 percent of the time, which is chance among 24 look-alikes, and dense retrieval 5.0 percent. With the context note, BM25 gets every question right and dense retrieval 74.2 percent, with the right chunk always inside the top 5. Dense retrieval is weaker here because it blends the note into a general meaning; BM25 matches the exact company and quarter tokens.

**Line by line.**

- `bm25_scorer` builds the weight matrix once; calling the returned function with a query returns one score per document.
- The queries are generated from the same template as the chunks, so the right chunk for each is known (`gold`).
- `hit@1` is the share of queries whose first result is the right chunk; `hit@5` allows the top five.

### 2. On real text, a title prefix helps a little

Real abstracts are not look-alikes, so the test is harder and the answer less dramatic. We cut all 5,183 SciFact abstracts into 25,378 two-sentence chunks, embed them bare and with the abstract's title in front (a free, deterministic context note), score each document by its best chunk, and compare.

```python
import math
import pickle
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

rows = {r["_id"]: (r["title"], r["text"]) for r in load_dataset("BeIR/scifact", "corpus")["corpus"]}
queries = {r["_id"]: r["text"] for r in load_dataset("BeIR/scifact", "queries")["queries"]}
qrels = defaultdict(set)
for r in load_dataset("BeIR/scifact-qrels")["test"]:
    if r["score"] > 0:
        qrels[str(r["query-id"])].add(str(r["corpus-id"]))
qids, doc_ids = sorted(qrels, key=int), sorted(rows, key=int)
chunks = []
for d in doc_ids:
    sentences = re.split(r"(?<=[.!?])\s+", rows[d][1].strip())
    chunks += [(d, rows[d][0], " ".join(sentences[s:s + 2])) for s in range(0, len(sentences), 2)]
starts = np.array([i for i in range(len(chunks)) if i == 0 or chunks[i][0] != chunks[i - 1][0]])
print(f"{len(doc_ids)} documents, {len(chunks)} chunks of two sentences, {len(qids)} test queries")

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
plain = encoder.encode([c[2] for c in chunks], batch_size=128, normalize_embeddings=True)
prefixed = encoder.encode([f"{c[1]}. {c[2]}" for c in chunks], batch_size=128, normalize_embeddings=True)
q_vecs = encoder.encode([queries[q] for q in qids], normalize_embeddings=True)
pickle.dump((plain, prefixed, q_vecs), open(Path(tempfile.gettempdir()) / "ragadv_chunk_vectors.pkl", "wb"))


def report(label, matrix):
    best = np.maximum.reduceat(matrix @ q_vecs.T, starts, axis=0)
    ndcg, hit1, rec = [], [], []
    for qi, q in enumerate(qids):
        ranked = [doc_ids[i] for i in np.argsort(-best[:, qi])[:10]]
        hits = [d in qrels[q] for d in ranked]
        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
        ndcg.append(dcg / sum(1 / math.log2(i + 2) for i in range(min(len(qrels[q]), 10))))
        hit1.append(hits[0])
        rec.append(sum(hits) / len(qrels[q]))
    print(f"{label:<26}{np.mean(ndcg):9.3f}{np.mean(hit1):8.3f}{np.mean(rec):11.3f}")


print(f"{'chunk vector built from':<26}{'nDCG@10':>9}{'hit@1':>8}{'recall@10':>11}")
report("the chunk alone", plain)
report("title + chunk", prefixed)
```

**Reading the output.** The title prefix improves nDCG@10 from 0.670 to 0.687 and recall@10 from 0.804 to 0.820. That is a small gain, nowhere near the synthetic jump, because most of these chunks already contain their own subject. Context pays when chunks are ambiguous. Measure ambiguity in your own corpus before paying a model call per chunk.

**Line by line.**

- `starts` marks where each document's chunks begin, so `np.maximum.reduceat` takes the best chunk score per document in one call.
- Scoring a document by its best chunk is the usual way to turn chunk retrieval into document retrieval.
- The vectors are saved for block 3.

### 3. Late chunking, tried honestly

Block 3 reads each whole abstract through the model once and builds each chunk vector from its own tokens' outputs. It then scores the three variants against the same questions.

```python
import math
import pickle
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

rows = {r["_id"]: (r["title"], r["text"]) for r in load_dataset("BeIR/scifact", "corpus")["corpus"]}
qrels = defaultdict(set)
for r in load_dataset("BeIR/scifact-qrels")["test"]:
    if r["score"] > 0:
        qrels[str(r["query-id"])].add(str(r["corpus-id"]))
qids, doc_ids = sorted(qrels, key=int), sorted(rows, key=int)
plain, prefixed, q_vecs = pickle.load(open(Path(tempfile.gettempdir()) / "ragadv_chunk_vectors.pkl", "rb"))
name = "sentence-transformers/all-MiniLM-L6-v2"
tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
late, starts = [], []
for d in doc_ids:
    title, text = rows[d]
    full = f"{title}. {text}"
    enc = tok(full, truncation=True, max_length=512, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state[0]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    cursor, starts = len(title) + 2, starts + [len(late)]
    for s in range(0, len(sentences), 2):
        piece = " ".join(sentences[s:s + 2])
        begin = full.find(piece[:25], cursor)
        cursor = max(cursor, begin)
        picked = [i for i, (a, b) in enumerate(offsets) if b > a and a >= begin and b <= begin + len(piece)]
        late.append(torch.nn.functional.normalize(hidden[picked].mean(0) if picked else hidden.mean(0), dim=-1).numpy())
late, starts = np.vstack(late), np.array(starts)
print(f"{len(late)} late-chunking vectors, same count as the {len(plain)} ordinary ones: {len(late) == len(plain)}")

print(f"{'chunk vector built from':<26}{'nDCG@10':>9}{'recall@10':>11}")
for label, matrix in (("the chunk alone", plain), ("title + chunk", prefixed), ("late chunking", late)):
    best = np.maximum.reduceat(matrix @ q_vecs.T, starts, axis=0)
    scores = []
    for qi, q in enumerate(qids):
        ranked = [doc_ids[i] for i in np.argsort(-best[:, qi])[:10]]
        hits = [d in qrels[q] for d in ranked]
        scores.append((sum(h / math.log2(i + 2) for i, h in enumerate(hits)) / sum(1 / math.log2(i + 2) for i in range(min(len(qrels[q]), 10))), sum(hits) / len(qrels[q])))
    print(f"{label:<26}{np.mean([s[0] for s in scores]):9.3f}{np.mean([s[1] for s in scores]):11.3f}")
```

**Reading the output.** Late chunking scores 0.659 nDCG@10 against 0.670 for the bare chunk: a little worse than the bare chunk, and below the title prefix (0.687); recall@10 is 0.796 against 0.804. The reason to expect this: `all-MiniLM-L6-v2` was trained on short texts, so its long-range attention is not tuned to carry document context into token vectors. The method is built for long-context embedding models. This result says nothing about the method with such a model, only that it is not a free upgrade for any model.

**Line by line.**

- `offsets` maps each model token to character positions, so `picked` selects exactly the tokens inside a chunk.
- `hidden[picked].mean(0)` is the late-chunking step: pooling after the model has read the whole text.
- `starts` is rebuilt here because the vectors are counted in the same order as block 2's chunks.

### 4. Multi-vector scoring and the three losses

First, ColBERT-style MaxSim with MiniLM token vectors (not a trained ColBERT model, so this shows the arithmetic only). Then the pairwise and listwise losses for one list of scores.

```python
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

name = "sentence-transformers/all-MiniLM-L6-v2"
tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()


def token_vectors(text):
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state[0]
    return tok.convert_ids_to_tokens(enc["input_ids"][0])[1:-1], torch.nn.functional.normalize(hidden, dim=-1)[1:-1].numpy()


query = "refund window for damaged items"
docs = {
    "returns": "Damaged goods can be returned for a full refund within thirty days of delivery.",
    "shipping": "Standard delivery takes five working days and tracking is sent by email.",
    "warranty": "The warranty covers manufacturing faults for twelve months from purchase.",
}
q_tokens, q_vecs = token_vectors(query)
print(f"{'document':<10}{'one vector':>12}{'MaxSim sum':>12}")
for label, text in docs.items():
    d_tokens, d_vecs = token_vectors(text)
    a, b = q_vecs.mean(0), d_vecs.mean(0)
    sim = q_vecs @ d_vecs.T
    print(f"{label:<10}{a @ b / np.linalg.norm(a) / np.linalg.norm(b):12.3f}{sim.max(axis=1).sum():12.3f}")
    if label == "returns":
        for token, row in zip(q_tokens, sim):
            print(f"   query token {token!r:<10} best match {d_tokens[int(row.argmax())]!r:<10} {row.max():.3f}")
length = len(token_vectors(docs["returns"])[0])
print(f"storage for the returns sentence: single vector 384 floats, multi-vector {length * 384} floats ({length} tokens)")

scores = np.array([2.0, 1.4, 0.3, -0.2])
relevant = 1
others = np.delete(scores, relevant)
pairwise = np.mean(np.log1p(np.exp(-(scores[relevant] - others))))
listwise = -np.log(np.exp(scores[relevant]) / np.exp(scores).sum())
print(f"scores {scores.tolist()}, relevant item {relevant}: pairwise loss {pairwise:.3f}, listwise loss {listwise:.3f}")
```

**Reading the output.** The returns sentence scores 0.615 as a single averaged vector and 4.068 under MaxSim, against 0.010 and 0.573 for the shipping sentence. Both methods pick the right document here. The per-token table shows why MaxSim is explainable: "damaged" finds "damaged" at 0.922, "items" finds "goods" at 0.622, and "window" finds nothing close (0.169). The storage line shows the price: 15 token vectors of 384 numbers instead of one. On the losses: for scores 2.0, 1.4, 0.3 and -0.2 with the relevant item second, the pairwise loss is 0.503 and the listwise loss is 1.211, because the listwise loss also punishes the 2.0 that sits above it.

**Line by line.**

- `sim.max(axis=1).sum()` is MaxSim: best document token for each query token, summed.
- `np.log1p(np.exp(-(a - b)))` is the logistic loss on a score gap `a - b`; it is small when the relevant item outscores the wrong one by a lot.
- The listwise loss is the negative log of the relevant item's softmax probability.

### 5. First-stage candidates

Now reranking. We build the three first stages over the whole corpus and keep the hybrid top 100 for each of the 300 test queries.

```python
import math
import pickle
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer


def bm25_scorer(docs, k1=1.5, b=0.75):
    vec = CountVectorizer(token_pattern=r"[a-z0-9]+")
    tf = vec.fit_transform(docs).tocoo().astype(float)
    df = np.bincount(tf.col, minlength=tf.shape[1])
    idf = np.log(1 + (tf.shape[0] - df + 0.5) / (df + 0.5))
    dl = np.bincount(tf.row, weights=tf.data, minlength=tf.shape[0])
    w = idf[tf.col] * tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / dl.mean()))
    matrix = csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)
    return lambda query: matrix @ (vec.transform([query]).toarray().ravel() > 0).astype(float)


corpus = {r["_id"]: (r["title"] + ". " + r["text"]).strip() for r in load_dataset("BeIR/scifact", "corpus")["corpus"]}
queries = {r["_id"]: r["text"] for r in load_dataset("BeIR/scifact", "queries")["queries"]}
qrels = defaultdict(set)
for r in load_dataset("BeIR/scifact-qrels")["test"]:
    if r["score"] > 0:
        qrels[str(r["query-id"])].add(str(r["corpus-id"]))
qids, doc_ids = sorted(qrels, key=int), sorted(corpus, key=int)
texts = [corpus[d] for d in doc_ids]
score_bm25 = bm25_scorer(texts)
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
doc_vecs = encoder.encode(texts, batch_size=64, normalize_embeddings=True)
q_vecs = encoder.encode([queries[q] for q in qids], normalize_embeddings=True)

runs = {"bm25": [], "dense": [], "hybrid": []}
for qi, q in enumerate(qids):
    by_bm25, by_dense = np.argsort(-score_bm25(queries[q]))[:100], np.argsort(-(doc_vecs @ q_vecs[qi]))[:100]
    fused = Counter()
    for ranking in (by_bm25, by_dense):
        for rank, i in enumerate(ranking):
            fused[int(i)] += 1 / (60 + rank + 1)
    runs["bm25"].append([int(i) for i in by_bm25])
    runs["dense"].append([int(i) for i in by_dense])
    runs["hybrid"].append([i for i, _ in fused.most_common(100)])
pickle.dump({"runs": runs, "qids": qids, "doc_ids": doc_ids}, open(Path(tempfile.gettempdir()) / "ragadv_candidates.pkl", "wb"))

print(f"{len(doc_ids)} documents, {len(qids)} test queries, {sum(len(v) for v in qrels.values())} relevance labels")
print(f"{'first stage':<12}{'nDCG@10':>9}{'recall@20':>11}{'recall@50':>11}{'recall@100':>12}")
for name, ranked in runs.items():
    ndcg = np.mean([sum(1 / math.log2(i + 2) for i, d in enumerate(r[:10]) if doc_ids[d] in qrels[q]) / sum(1 / math.log2(i + 2) for i in range(min(len(qrels[q]), 10))) for r, q in zip(ranked, qids)])
    rec = [np.mean([len({doc_ids[i] for i in r[:k]} & qrels[q]) / len(qrels[q]) for r, q in zip(ranked, qids)]) for k in (20, 50, 100)]
    print(f"{name:<12}{ndcg:9.3f}{rec[0]:11.3f}{rec[1]:11.3f}{rec[2]:12.3f}")
```

**Reading the output.** Hybrid fusion gets nDCG@10 0.691, against 0.664 for BM25 and 0.648 for dense retrieval. Recall rises with depth: 0.879 of relevant documents are in the top 20 hybrid results, 0.937 in the top 50 and 0.955 in the top 100. Those recall figures are the ceilings for any reranker working on that shortlist.

**Line by line.**

- `fused[int(i)] += 1 / (60 + rank + 1)` is reciprocal rank fusion with the usual constant 60.
- Ranks are saved as corpus positions, so later blocks can score exactly these candidates.

### 6. How deep should the reranker look?

The general MS MARCO cross-encoder scores the top 50 hybrid candidates of every query once (15,000 pairs). We then measure what happens if we rerank only the top 5, 10, 20, 30 or 50.

```python
import math
import pickle
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import CrossEncoder

tmp = Path(tempfile.gettempdir())
saved = pickle.load(open(tmp / "ragadv_candidates.pkl", "rb"))
qids, doc_ids, hybrid = saved["qids"], saved["doc_ids"], saved["runs"]["hybrid"]
corpus = {r["_id"]: (r["title"] + ". " + r["text"]).strip() for r in load_dataset("BeIR/scifact", "corpus")["corpus"]}
queries = {r["_id"]: r["text"] for r in load_dataset("BeIR/scifact", "queries")["queries"]}
qrels = defaultdict(set)
for r in load_dataset("BeIR/scifact-qrels")["test"]:
    if r["score"] > 0:
        qrels[str(r["query-id"])].add(str(r["corpus-id"]))

cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu", max_length=256)
start = time.perf_counter()
scores = [cross.predict([(queries[q], corpus[doc_ids[i]]) for i in ranking[:50]], batch_size=25, show_progress_bar=False) for q, ranking in zip(qids, hybrid)]
ms_per_pair = 1000 * (time.perf_counter() - start) / (len(qids) * 50)
pickle.dump(scores, open(tmp / "ragadv_scores.pkl", "wb"))
print(f"scored {len(qids) * 50} pairs, {ms_per_pair:.1f} ms per pair")


def ndcg10(ranked, rel):
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if doc_ids[d] in rel)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))


def rerank(depth):
    return [[r[j] for j in np.argsort(-s[:depth])] + list(r[depth:]) for r, s in zip(hybrid, scores)]


base = np.array([ndcg10(r, qrels[q]) for r, q in zip(hybrid, qids)])
print(f"{'depth':>5}{'nDCG@10':>9}{'change':>8}{'relevant in shortlist':>23}{'ms/query':>10}")
print(f"{'none':>5}{base.mean():9.3f}")
for depth in (5, 10, 20, 30, 50):
    after = np.array([ndcg10(r, qrels[q]) for r, q in zip(rerank(depth), qids)])
    inside = np.mean([len({doc_ids[i] for i in r[:depth]} & qrels[q]) / len(qrels[q]) for r, q in zip(hybrid, qids)])
    print(f"{depth:>5}{after.mean():9.3f}{after.mean() - base.mean():+8.3f}{inside:23.3f}{depth * ms_per_pair:10.0f}")
after = np.array([ndcg10(r, qrels[q]) for r, q in zip(rerank(20), qids)])
up, down = int((after > base + 1e-9).sum()), int((after < base - 1e-9).sum())
print(f"at depth 20 the reranker improved {up} queries, hurt {down}, left {len(qids) - up - down} unchanged")
```

**Reading the output.** The result is not what the model card suggests. The model scores 74.30 nDCG@10 on TREC DL 19 and 39.01 MRR@10 on MS MARCO dev according to its card, but on SciFact reranking the hybrid shortlist does not help: nDCG@10 is 0.691 for the hybrid ranking alone and 0.691, 0.684, 0.682, 0.675 and 0.684 after reranking the top 5, 10, 20, 30 and 50. At depth 20 it improves 56 queries and hurts 60, leaving the rest unchanged. Latency grows linearly (about 9.3 ms per pair on this CPU, so 186 ms per query at depth 20) while the share of relevant documents in the shortlist rises with depth. A deeper shortlist raises the ceiling but this model cannot exploit it.

**Why a good reranker can hurt.** The model was trained on web questions with short passages. SciFact queries are scientific claims ("Arterioles have a larger lumen diameter than venules") and the documents are abstracts. A model that learned "answers look like a sentence that restates the question" will push topically close abstracts that do not settle the claim. The hybrid first stage already captures most of what is cheaply captured.

**Line by line.**

- The scores are saved so block 9 can compare against them.
- `rerank(depth)` reorders only the first `depth` candidates and keeps the rest in first-stage order.
- "Relevant in shortlist" is recall at that depth: the ceiling.

### 7. Build training pairs with hard negatives

A reranker can be taught your domain. We turn SciFact's training claims (809 claims, disjoint from the 300 test claims) into labelled pairs: every known evidence abstract is a positive, and three BM25 hard negatives per claim are negatives. Documents that are relevant to any test claim are never used as negatives, so training cannot touch the test labels.

```python
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


def bm25_scorer(docs, k1=1.5, b=0.75):
    vec = CountVectorizer(token_pattern=r"[a-z0-9]+")
    tf = vec.fit_transform(docs).tocoo().astype(float)
    df = np.bincount(tf.col, minlength=tf.shape[1])
    idf = np.log(1 + (tf.shape[0] - df + 0.5) / (df + 0.5))
    dl = np.bincount(tf.row, weights=tf.data, minlength=tf.shape[0])
    w = idf[tf.col] * tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / dl.mean()))
    matrix = csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)
    return lambda query: matrix @ (vec.transform([query]).toarray().ravel() > 0).astype(float)


corpus = {r["_id"]: (r["title"] + ". " + r["text"]).strip() for r in load_dataset("BeIR/scifact", "corpus")["corpus"]}
queries = {r["_id"]: r["text"] for r in load_dataset("BeIR/scifact", "queries")["queries"]}
rels = {split: defaultdict(set) for split in ("train", "test")}
for split in rels:
    for r in load_dataset("BeIR/scifact-qrels")[split]:
        if r["score"] > 0:
            rels[split][str(r["query-id"])].add(str(r["corpus-id"]))
held_out = set().union(*rels["test"].values())
doc_ids = sorted(corpus, key=int)
score_bm25 = bm25_scorer([corpus[d] for d in doc_ids])

pairs = []
for q, positives in rels["train"].items():
    pairs += [(queries[q], corpus[d], 1.0) for d in positives]
    ranked = [doc_ids[i] for i in np.argsort(-score_bm25(queries[q]))[:30]]
    hard = [d for d in ranked if d not in positives and d not in held_out][:3]
    pairs += [(queries[q], corpus[d], 0.0) for d in hard]
Path(tempfile.gettempdir(), "ragadv_pairs.json").write_text(json.dumps(pairs))
print(f"{len(rels['train'])} training queries, {len(rels['test'])} test queries, no query in both")
print(f"{len(pairs)} pairs: {sum(p[2] for p in pairs):.0f} positive, {sum(1 - p[2] for p in pairs):.0f} hard negative")
print("example hard negative for:", pairs[1][0][:70])
print("   ", pairs[1][1][:90])
```

**Reading the output.** 809 training claims give 3,346 pairs, 919 positive and 2,427 hard negative, and none of the 300 test claims is among the training claims. A hard negative shares vocabulary with the claim but does not settle it, which is what a first stage will hand a reranker in production. Random negatives would be too easy to teach anything.

**Line by line.**

- `held_out` is every document relevant to a test claim.
- `ranked[:30]` takes BM25's top 30 for the training claim, then drops positives and held-out documents, then keeps three.

### 8. Fine-tune the cross-encoder

One pass over the pairs with the binary cross-entropy loss, batches of 16, learning rate 2e-5. This is deliberately small: 22.7 million parameters, CPU only.

```python
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.manual_seed(0)
tmp = Path(tempfile.gettempdir())
pairs = json.loads((tmp / "ragadv_pairs.json").read_text())
name = "cross-encoder/ms-marco-MiniLM-L6-v2"
tok, model = AutoTokenizer.from_pretrained(name), AutoModelForSequenceClassification.from_pretrained(name)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
order = np.random.default_rng(0).permutation(len(pairs))
model.train()
start, losses = time.time(), []
for step, lo in enumerate(range(0, len(order), 16)):
    batch = [pairs[i] for i in order[lo:lo + 16]]
    enc = tok([b[0] for b in batch], [b[1] for b in batch], truncation=True, max_length=256, padding=True, return_tensors="pt")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(**enc).logits.squeeze(-1), torch.tensor([b[2] for b in batch]))
    loss.backward()
    opt.step()
    opt.zero_grad()
    losses.append(loss.item())
model.eval()
model.save_pretrained(tmp / "ragadv_reranker")
tok.save_pretrained(tmp / "ragadv_reranker")
print(f"{step + 1} steps of 16 pairs in {time.time() - start:.0f} s")
print(f"mean loss, first 20 steps {np.mean(losses[:20]):.3f}, last 20 steps {np.mean(losses[-20:]):.3f}")
```

**Reading the output.** 210 steps of 16 pairs take 163 seconds on this CPU. The mean loss falls from 0.851 over the first 20 steps to 0.421 over the last 20. The first value is high because the starting model's scores were tuned for a different notion of relevance. The loss is noisy because each batch is only 16 pairs, which is why the block prints the mean of the first and last 20 steps instead of single values.

**Line by line.**

- `AutoModelForSequenceClassification` with one output label is how a cross-encoder is stored: the logit is the relevance score.
- `binary_cross_entropy_with_logits` is the pointwise loss from the explanation above.
- The model and tokenizer are saved for block 9.

### 9. Did fine-tuning help? Test, do not hope

We score the top 20 hybrid candidates with the general and the fine-tuned model and compare per query, with a bootstrap interval on the difference.

```python
import math
import pickle
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tmp = Path(tempfile.gettempdir())
saved = pickle.load(open(tmp / "ragadv_candidates.pkl", "rb"))
qids, doc_ids, hybrid = saved["qids"], saved["doc_ids"], saved["runs"]["hybrid"]
corpus = {r["_id"]: (r["title"] + ". " + r["text"]).strip() for r in load_dataset("BeIR/scifact", "corpus")["corpus"]}
queries = {r["_id"]: r["text"] for r in load_dataset("BeIR/scifact", "queries")["queries"]}
qrels = defaultdict(set)
for r in load_dataset("BeIR/scifact-qrels")["test"]:
    if r["score"] > 0:
        qrels[str(r["query-id"])].add(str(r["corpus-id"]))
DEPTH = 20


def ndcg10(ranked, rel):
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if doc_ids[d] in rel)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))


def scores_for(path):
    tok, model = AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path).eval()
    out = []
    for q, ranking in zip(qids, hybrid):
        enc = tok([queries[q]] * DEPTH, [corpus[doc_ids[i]] for i in ranking[:DEPTH]], truncation=True, max_length=256, padding=True, return_tensors="pt")
        with torch.no_grad():
            out.append(model(**enc).logits.squeeze(-1).numpy())
    return out


def evaluate(scores):
    ranked = [[r[j] for j in np.argsort(-s)] + list(r[DEPTH:]) for r, s in zip(hybrid, scores)]
    return np.array([ndcg10(r, qrels[q]) for r, q in zip(ranked, qids)])


base = np.array([ndcg10(r, qrels[q]) for r, q in zip(hybrid, qids)])
before = evaluate(scores_for("cross-encoder/ms-marco-MiniLM-L6-v2"))
after = evaluate(scores_for(str(tmp / "ragadv_reranker")))
print(f"{'ranking':<34}{'nDCG@10':>9}")
for label, values in (("hybrid first stage", base), ("rerank top-20, general model", before), ("rerank top-20, fine-tuned", after)):
    print(f"{label:<34}{values.mean():9.3f}")
better, worse = int((after > before + 1e-9).sum()), int((after < before - 1e-9).sum())
print(f"fine-tuned vs general: better on {better} queries, worse on {worse}, same on {len(qids) - better - worse}")
gap = after - before
rng = np.random.default_rng(0)
boot = [gap[rng.integers(0, len(gap), len(gap))].mean() for _ in range(2000)]
print(f"mean gain {gap.mean():+.3f}, 95 percent bootstrap interval [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]")
```

**Reading the output.** The hybrid first stage scores 0.691, the general reranker 0.682 and the fine-tuned reranker 0.702. Against the general model the fine-tuned one is better on 50 queries and worse on 29; the mean gain is +0.020, with a 95 percent bootstrap interval of +0.002 to +0.039. The interval excludes zero, so the gain over the general model is probably real; the gain over the plain hybrid ranking is +0.011 and is within what 300 queries can resolve poorly, so the honest summary is: fine-tuning recovered the loss and gave a modest gain, not a transformation. A real project should also hold out a validation set for choosing the learning rate and epochs, which this block skips.

**Line by line.**

- `evaluate` reorders only the top 20 and leaves the rest in first-stage order, as in block 6.
- The bootstrap resamples queries with replacement 2,000 times and reads the 2.5th and 97.5th percentiles of the mean difference.

### Try it in the lab

The lab holds the same 300 queries, the general model's scores for the top 50, and the fine-tuned model's scores for the top 20. The summary line averages all 300. Pick one of six example queries to see the top ten before and after.

<RerankLab />

**What each control does.**

- **query**: one of six example claims, chosen automatically: two where the general model helps most, two where it hurts most, two where fine-tuning helps most.
- **reranker**: the general MS MARCO model or the one fine-tuned in block 8.
- **depth**: how many first-stage results are rescored. The fine-tuned model was scored to depth 20 only.
- **show data**: the table of the reranked top ten with first-stage rank and relevance label.

**Try it yourself.**

1. Leave the defaults (general model, depth 20). The summary reads 0.691 to 0.682 (56 better, 60 worse), the same as block 6. Pick the first example: its nDCG@10 rises on the right. Pick the third: it falls.
2. Set depth to 5, then 50 (general model). Watch the all-queries average: the value at depth 5 is almost unchanged, at depth 50 it is still below the hybrid score. A deeper shortlist did not rescue this model.
3. Switch the reranker to fine-tuned at depth 20. The summary becomes 0.691 to 0.702, the same as block 9. Look at what changed for the third example.

<Infographic src="/img/rag-adv/contextual-retrieval-and-reranking-results.svg" alt="Bars for nDCG at 10 of the hybrid first stage, the general reranker at several depths and the fine-tuned reranker, and the effect of context on a synthetic and a real corpus" caption="Left: the three rankers on SciFact. Right: how much context helped on a synthetic corpus built to need it, and on real abstracts. Every number is printed by a code block." />

## Designing with it

- **Diagnose before repairing.** For each failed query, ask whether the right chunk was in the first-stage top 100 (a context or coverage problem) or in the top 100 but not in the top 5 (an ordering problem). They have different fixes.
- **Try the free context first.** Titles, headings, authors, dates and section paths cost nothing to prepend. Pay a model call per chunk only if this is not enough and your chunks are genuinely ambiguous.
- **Prepend, then embed and index.** Add the note to both the embedding text and the BM25 text. In block 1, BM25 did most of the work.
- **Set reranking depth from recall.** Choose the smallest depth whose first-stage recall is close to the recall at 100, then check the latency budget. Block 6 gave about 9.3 ms per pair; measure on your own hardware.
- **Never ship a reranker you have not measured on your own queries.** A model that tops a public benchmark can lose to your first stage on your corpus, as block 6 shows.
- **Log the shortlist.** Keep the first-stage ranks next to the reranked ones. When quality drops, you can tell which stage moved.
- **Pair this with evaluation.** Build the 50-query judged set described in [testing RAG retrievers](/docs/llm-evals/testing-rag-retrievers) before you tune anything here.

## Where this stands in 2026

:::info Industry view
Context notes and rerankers are standard parts of production retrieval stacks, and the open model families keep expanding: Qwen3 released embedding and reranking models at 0.6B, 4B and 8B parameters in June 2025. I checked the papers and model cards named under Go deeper, not independent leaderboards, so I cannot tell you today's best reranker for your language or domain. What stays true whatever the model: the shortlist is a ceiling, a general reranker is a hypothesis to test on your data, and context added at index time is paid for once while reranking is paid for on every query.
:::

## Common mistakes

- **Reranking a shortlist that does not contain the answer.** It feels like the model is not good enough, so people swap rerankers. First check recall at the shortlist depth (block 5). If the answer is not there, fix the first stage or raise the depth.
- **Assuming a benchmark winner wins on your data.** A model with a strong public score feels safe. Block 6 shows a score of 74.30 nDCG@10 on TREC DL 19 and a loss on SciFact. Always run your own judged queries.
- **Paying for LLM-written context before trying metadata.** The prompt-based method feels more sophisticated. If the title and section path already name the subject, as in block 2, the gain is small. Test the free prefix first.
- **Evaluating on the training claims.** Training and testing on the same queries gives a flattering number. Block 7 excludes every test-relevant document from the negatives for exactly this reason.
- **Trusting one number from 300 queries.** The fine-tuned gain over the hybrid ranking is small; without a bootstrap interval it is easy to over-read. Report intervals, and per-query wins and losses.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> In block 1, why does BM25 get hit@1 of 0.042 on bare chunks?</summary>

Each question has 24 look-alike chunks that share the same query words. They tie, so the right one is first by chance, 1 in 24, which is 0.042.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> What can a reranker never fix?</summary>

A relevant document that is not in the shortlist. A reranker reorders the candidates it is given. The recall of the first stage at that depth is the ceiling.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> Block 2 shows a small gain from a title prefix and block 1 a huge one. What explains the difference, and what should you measure in your own corpus?</summary>

In block 1 the chunks were designed to be ambiguous: none named its company, so nothing but the context separated 24 look-alikes. Real abstracts already contain their subject in nearly every chunk. Measure ambiguity: for a sample of chunks, ask whether a person could tell which document and entity it belongs to without the title. If most chunks pass, expect a gain like block 2's.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> Why can the general MS MARCO reranker lower nDCG@10 on SciFact even though it is a strong model?</summary>

It was trained on web questions and short passages, a different distribution from scientific claims and abstracts. A first stage that already fuses BM25 and dense signals leaves little that the model can add, and where the model's idea of relevance differs from the labels, it moves good results down. The BEIR paper's "best on average" is an average over datasets, not a promise for each.

</details>

<details>
<summary><strong>Q5 (Medium).</strong> Late chunking did not beat the bare chunk in block 3. Does this show that late chunking does not work?</summary>

No. The embedding model reads at most 512 tokens and was trained on short text, so it was not built to carry document context into token vectors. The method was proposed for long-context embedding models. The result shows only that the technique is not a free upgrade for an arbitrary model.

</details>

<details>
<summary><strong>Q6 (Medium).</strong> In block 6 the share of relevant documents in the shortlist rises as depth rises, yet nDCG@10 does not. Why?</summary>

Depth only raises the ceiling. nDCG@10 depends on whether the reranker puts those documents in the top ten, and this reranker does not do so reliably on this data. More candidates also give a weak scorer more wrong documents to promote.

</details>

<details>
<summary><strong>Q7 (Stretch).</strong> You fine-tuned a reranker and got +0.014 nDCG@10 over the hybrid baseline on 300 queries. List three things you would check before shipping it.</summary>

A confidence interval on the difference between the two rankings, per query. A held-out validation set used to pick the learning rate and number of epochs, separate from the test set. A check that no training positive or negative overlaps the test labels. Latency at the chosen depth against the budget, since you pay it on every query. A regression check on query types the training data did not cover.

</details>

<details>
<summary><strong>Q8 (Stretch).</strong> Multi-vector retrieval stores one vector per token. For a corpus of 5 million passages of 100 tokens with 128-dimensional vectors stored as 4-byte floats, how much storage is that compared with one 384-dimensional vector per passage?</summary>

Multi-vector: 5,000,000 x 100 x 128 x 4 bytes = 256,000,000,000 bytes, about 256 GB. Single vector: 5,000,000 x 384 x 4 = 7,680,000,000 bytes, about 7.7 GB. The ratio is about 33 to 1, which is why multi-vector indexes use compression and why many teams use ColBERT-style scoring only on a shortlist.

</details>

## Go deeper

All sources opened on 7 October 2026.

- Anthropic, [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), 19 September 2024. Figures are the vendor's own.
- Günther et al., [Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models](https://arxiv.org/abs/2409.04701), arXiv 2409.04701, v3 July 2025.
- Khattab and Zaharia, [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832), arXiv 2004.12832, SIGIR 2020.
- Zhang et al., [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://arxiv.org/abs/2506.05176), arXiv 2506.05176, June 2025 (abstract only).
- Thakur et al., [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663), arXiv 2104.08663, 2021.
- Wadden et al., [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974), arXiv 2004.14974, EMNLP 2020 (the SciFact dataset).
- Model cards: [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) (22.7M parameters, 74.30 nDCG@10 on TREC DL 19, 39.01 MRR@10 on MS MARCO dev) and [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache 2.0, 384 dimensions, 256 word-piece limit).
- On this site: [neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking), [enterprise document QA](/docs/senior/design-enterprise-document-qa), [RAG evaluation](/docs/llm-evals/testing-rag-retrievers).

## Check yourself

- I can say whether a retrieval failure is a context problem or an ordering problem by looking at the first-stage top 100.
- I can explain why prefixing a chunk with its company and quarter lifted BM25 hit@1 from 0.042 to 1.000, and why the same trick gained little on real abstracts.
- I can describe late chunking and ColBERT-style MaxSim in two sentences each and name the cost of each.
- I can choose a reranking depth from first-stage recall and a latency budget.
- I can build hard-negative training pairs without leaking the test labels.
- I can test a fine-tuned reranker per query with a bootstrap interval instead of one average.

## Where to go next

This is the last chapter of the advanced RAG series. To go back to the start, see [GraphRAG and knowledge graphs](/docs/genai/rag-advanced/graphrag-and-knowledge-graphs), or see the previous one on [text-to-SQL and structured RAG](/docs/genai/rag-advanced/text-to-sql-and-structured-rag). To evaluate everything you build here, go to [testing RAG retrievers](/docs/llm-evals/testing-rag-retrievers).
