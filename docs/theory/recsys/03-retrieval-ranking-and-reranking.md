---
title: "Recommenders · Retrieval, ranking and reranking"
sidebar_label: "Retrieval and ranking"
sidebar_position: 3
slug: /theory/recsys/retrieval-ranking-and-reranking
description: "Two-tower candidate generation, approximate-neighbour indexes and final slate decisions."
tags: [recommender-systems, retrieval, ranking]
---

import Infographic from '@site/src/components/Infographic';
import CandidateRecallLab from '@site/src/components/viz/CandidateRecallLab';

**In one line.** Retrieval finds a broad, affordable shortlist; ranking scores it with richer context; reranking turns those scores into an eligible and useful displayed slate.

:::tip Before you start
**You should already know**

- What a user-item score is and why popularity is the baseline to beat: [collaborative filtering](/docs/theory/recsys/collaborative-filtering).
- Precision and recall on a ranked list: [evaluating ranked retrieval](/docs/theory/ir/evaluating-ranked-retrieval).
- What a neural network layer and a softmax do, at the level of [neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking).

**Reading time.** About 40 minutes, plus about a minute to run the code.

**After this chapter you can**

- explain why a recommender is built as retrieve, rank, rerank, and what each stage can and cannot fix,
- train a small two-tower retriever in PyTorch and say why in-batch negatives need a frequency correction,
- measure how candidate depth, approximate search and a re-ranking rule change the final list.
:::

## In 30 seconds

A librarian cannot read every book to answer you. The first step is to pull forty plausible books from the shelves, quickly. The second is to read the blurbs and order those forty. The last is to check that the five on your desk are not all by the same author. Recommenders work the same way: a fast, rough stage pulls candidates, a slower, smarter stage orders them, and a last stage fixes the list as a whole. The rough stage sets a ceiling. If it never pulled the book you wanted, nothing downstream can bring it back.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Candidate generation | A fast first stage that shortlists items | 100 films from 1,682 |
| Two-tower model | Separate networks embed the user and the item, and a dot product scores them | User vector (0.6, 0.8) times film vector (0.8, 0.6) |
| Embedding | A vector of numbers standing for a user or an item | 32 numbers per film |
| In-batch negative | Other items in the same training batch used as wrong answers | In a batch of 1,024 pairs, 1,023 negatives per user |
| logQ correction | Subtract the log of each item's sampling frequency from its score during training | Popular film X gets a lower training logit |
| Approximate nearest neighbours (ANN) | Search that returns most, not all, of the true top items, faster | IVF or HNSW indexes |
| Recall at K | Share of the user's liked items found in the top K | 0.519 at K = 100 |
| Re-ranker | A last pass that adjusts the list for diversity, freshness or rules | Prefer a slate with different genres |

## The idea in plain words

A large catalogue cannot always be scored item by item with an expensive model for every page request. A staged system divides the work. **Candidate generation** quickly finds hundreds or thousands of plausible items from a much larger catalogue. **Ranking** applies a richer model to that shortlist. **Reranking** applies list-level constraints and adjustments such as diversity, freshness, deduplication and eligibility. The stages solve different problems, so each needs its own metric and failure diagnosis.

If retrieval drops a useful item, the ranker cannot restore it. If retrieval returns relevant items but ranking puts them below weak ones, candidate recall is not the problem. If ranking is good but the final slate repeats the same topic or includes an unavailable item, the reranker or policy layer needs work. Trace a request through all stages, preserving candidate IDs, source names and scores so that an offline investigation can identify which stage lost an item.

<Infographic src="/img/recsys/retrieval-ranking.svg" alt="Four cards summarise a query tower, precomputed item embeddings and index, ranking and reranking, and toy top-two candidate recall of 0.5." caption="Candidate recall limits every later stage's ability to select a relevant item." />

## Worked example, step by step

Why does a two-tower model trained on in-batch negatives need a correction? Take one training example whose positive film is X, and two other films Y and Z in the same batch. The first block under "Code you can run" reproduces these numbers.

1. The model's scores (logits) for this user are X = 2.0, Y = 1.0, Z = 0.5. The share of training pairs each film accounts for is X = 0.5, Y = 0.3, Z = 0.2. Popular X is in a lot of batches.
2. Plain softmax: $e^{2.0} = 7.389$, $e^{1.0} = 2.718$, $e^{0.5} = 1.649$. The total is 11.756, so X gets $7.389/11.756 = 0.629$, Y gets 0.231 and Z gets 0.140. The loss is $-\ln 0.629 = 0.464$.
3. Because the negatives are drawn by popularity, popular films show up as negatives more often than they would in a uniform sample. The model is pushed to score them down, even when users like them.
4. The logQ fix subtracts $\ln(\text{share})$ from each logit: $2.0 - \ln 0.5 = 2.693$, $1.0 - \ln 0.3 = 2.204$, $0.5 - \ln 0.2 = 2.109$.
5. Corrected softmax: $e^{2.693} = 14.78$, $e^{2.204} = 9.06$, $e^{2.109} = 8.24$, total 32.08. X now gets $14.78/32.08 = 0.461$, Y 0.282, Z 0.257. The loss on the popular positive rises to about 0.775, so training pushes its raw score back up.

In words: sampling items by popularity makes the training signal anti-popular, and the correction removes that tilt.

## How it works

### Generate candidates from several routes

One candidate source may use collaborative factors, another content similarity, another session context, and another a safe popularity fallback. Union the candidate IDs, deduplicate them, then apply hard eligibility rules at a defined point. Keeping source attribution matters: a ranking improvement may actually come from a new source that covers fresh items. Different sources have scores on different scales, so do not sort their raw scores together as if they were calibrated. A common ranker can score the merged pool with one objective.

For the lab's exact toy list, retrieval scores A 0.9, B 0.8, C 0.6 and D 0.4. Only B and D have been labelled relevant. At cutoff two, retrieval returns A and B, so it finds one of two relevant items: recall is $1/2=0.5$. No downstream reranker of A and B can select D. Increasing the cutoff to four yields recall one on this tiny labelled set, but in a real system a larger shortlist raises ranker cost and may lower precision. The measured trade-off includes latency and relevance at several cutoffs, by new-item and user segment.

### Use a two-tower retrieval model

A two-tower design maps a user or request context to a query vector $q$ and each item to a candidate vector $v_i$. A lightweight similarity, often a dot product, scores the pair. Item vectors can be precomputed and placed in a vector index; the request computes the query vector and looks up near neighbours. [TensorFlow Recommenders' retrieval task](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval) uses this factorised structure. It is attractive because the item tower does not need to run for every item on each request. The downside is that only interactions expressible through the separate towers and lightweight final score affect retrieval directly.

Training commonly uses observed positive pairs and sampled alternatives. In-batch negatives are efficient, but an item sampled as a negative for one user could in fact interest that user. Popular items can appear often and create sampling bias; duplicate positives can become accidental negatives. The loss, temperature and sampling distribution determine what the vectors learn. A high batch metric based on easy negatives does not guarantee recall among a million catalogue items. Measure full-catalogue or representative-index recall on held-out interactions when feasible.

### Index and refresh candidate vectors

An exact nearest-neighbour search scores every item and can be a useful small-catalogue baseline. An approximate nearest-neighbour (ANN) index trades some exactness for latency and scale. The [TensorFlow Recommenders retrieval tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval) illustrates a query model paired with exact or approximate candidate lookup. For an ANN deployment, compare retrieved neighbours against an exact-search reference on held-out queries, then measure end-to-end recommendation quality. An index that is fast but omits many strong candidates can cap the whole system's quality.

Index freshness is as important as search speed. A newly published item is invisible until its vector and metadata reach the index. A removed item must be filtered even if its vector remains temporarily present. Query and item tower versions must match. If a new query tower is deployed against an old item index, the shared embedding geometry may shift. Use atomic or controlled index swaps, version tags and rollback. Monitor empty retrievals, stale embeddings, catalogue coverage, latency and approximate recall.

### Rank with richer context

The ranker sees a manageable candidate pool and can use features too costly or pair-specific for retrieval: user-item crosses, current session, price, location, item quality, freshness and source attribution. Its score should correspond to a specified outcome and horizon, such as completion after an impression. A score can be calibrated into an estimated probability only if calibration is checked on relevant data; a ranking model's raw logit or dot product is not automatically a probability. A high offline AUC is not a guarantee of a useful top ten because the displayed region is a narrow part of the score distribution.

Ranking data are generated by earlier retrieval and display policies. If an item was never a candidate, the ranker cannot learn its effect from ordinary impression logs. Position affects clicks, and current rankers change what gets labelled. Record source and position, use controlled exploration or appropriate causal evaluation where justified, and keep an independent online test. The [Google scoring guide](https://developers.google.com/machine-learning/recommendation/dnn/scoring) emphasises that the common scoring stage can compare candidates from different sources and use richer features than retrieval.

### Rerank the displayed set

The final slate may need to suppress duplicate creators, avoid showing five near-identical lessons, ensure at least one fresh item or enforce inventory rules. Some are hard constraints; others are preferences. Hard constraints should not be represented as a tiny score penalty that can be overwhelmed by another feature. Diversity can be a list-level objective: the second item may be chosen partly for how it complements the first, not just its individual score. Recheck eligibility immediately before display if inventory or permissions change quickly. The [Google reranking guide](https://developers.google.com/machine-learning/recommendation/dnn/re-ranking) discusses freshness, diversity and fairness in this final stage.

## A real system that works this way

The [2016 Google Research paper on YouTube recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) documents a historical two-stage system with a candidate-generation model and a separate ranking model. It illustrates why a single model is not forced to scan an enormous catalogue and optimise every display decision at once. It does not describe YouTube's exact 2026 implementation. A smaller course platform can use the same separation with content and collaborative candidate routes, a ranking model for completion, and a prerequisite-aware final slate. The architecture is useful even before neural towers are justified.

## Code you can run

```python
candidates = [('A', 0.9, False), ('B', 0.8, True), ('C', 0.6, False), ('D', 0.4, True)]
k = 2
retrieved = candidates[:k]
relevant_total = sum(relevant for _, _, relevant in candidates)
relevant_found = sum(relevant for _, _, relevant in retrieved)
print('retrieved:', [item for item, _, _ in retrieved])
print('candidate recall:', relevant_found / relevant_total)
```

This prints `['A', 'B']` and candidate recall `0.5`. The labels are complete only within this four-item illustration.

<CandidateRecallLab />

Move the cutoff to see which labelled relevant items enter the shortlist. At two candidates, the lab reproduces one of two. It is an exact toy ranking, not a performance measurement of ANN.

```python
query = (1.0, 0.5)
item_vectors = {'A': (0.9, 0.0), 'B': (0.6, 0.4), 'C': (0.1, 1.0)}
scores = {item: sum(a * b for a, b in zip(query, vector)) for item, vector in item_vectors.items()}
print({item: round(score, 2) for item, score in scores.items()})
print('top two:', sorted(scores, key=scores.get, reverse=True)[:2])
```

This exact dot-product scan returns A and B. At catalogue scale, an index may approximate the same search; any quality or speed claim needs a measured comparison.

### Experiment: the whole funnel on MovieLens

This experiment uses the MovieLens 100K download from [the first chapter](/docs/theory/recsys/feedback-and-objectives) (licence: research use, acknowledgement, no redistribution, no commercial use without permission; the code downloads it at run time). Each user's history is split by time into 70% fit, 10% validation and 20% test. The towers learn from the fit part. The ranker learns from validation labels. Recall and precision are measured on the test part, counting films rated 4 or 5, with every film the user already has excluded.

Block one is the by-hand arithmetic.

```python
import numpy as np

items = ['X', 'Y', 'Z']
logits = np.array([2.0, 1.0, 0.5])
share = np.array([0.5, 0.3, 0.2])

def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()

plain = softmax(logits)
corrected = softmax(logits - np.log(share))
for name, a, b in zip(items, plain, corrected):
    print(f'{name}: plain probability {a:.3f}, corrected {b:.3f}')
print('loss when X is the positive, plain:', round(-np.log(plain[0]), 3), '| corrected:', round(-np.log(corrected[0]), 3))
```

It prints plain probabilities 0.629, 0.231, 0.140, corrected 0.461, 0.282, 0.257, and losses 0.464 and 0.775.

The second block trains two two-tower models, one with and one without the logQ correction, compares their candidate recall with popularity, then trains a gradient-boosted ranker on the logQ candidates and measures the final top 10 at different candidate depths. A last loop re-ranks with a genre-similarity penalty.

```python
import os
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier

URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
CACHE = os.path.join(tempfile.gettempdir(), 'ml-100k.zip')
if not os.path.exists(CACHE):
    urllib.request.urlretrieve(URL, CACHE)
with zipfile.ZipFile(CACHE) as z:
    df = pd.read_csv(z.open('ml-100k/u.data'), sep='\t', names=['u', 'i', 'r', 't'])
    people = pd.read_csv(z.open('ml-100k/u.user'), sep='|', names=['u', 'age', 'sex', 'job', 'zip'])
    films = pd.read_csv(z.open('ml-100k/u.item'), sep='|', header=None, encoding='latin-1')
df['u'] -= 1
df['i'] -= 1
n_users, n_items = df.u.max() + 1, df.i.max() + 1
genres = films.iloc[:, 5:].values.astype(np.float32)
profile = torch.tensor(np.stack([(people.age // 10).clip(0, 6), (people.sex == 'M').astype(int), pd.factorize(people.job)[0]], 1))

df = df.sort_values(['u', 't'], kind='stable')
frac = (df.groupby('u').cumcount() + 0.5) / df.groupby('u').u.transform('size')
fit, val, test = df[frac < 0.7], df[(frac >= 0.7) & (frac < 0.8)], df[frac >= 0.8]
def grid(part, minimum=0):
    g = np.zeros((n_users, n_items), bool)
    keep = part[part.r >= minimum]
    g[keep.u, keep.i] = True
    return g
seen_fit, seen_all = grid(fit), grid(fit) | grid(val)
liked_val, liked_test = grid(val, 4), grid(test, 4)
users = np.flatnonzero(liked_test.any(1))

class Tower(torch.nn.Module):
    def __init__(self, n_id, side, d=32):
        super().__init__()
        self.emb, self.side = torch.nn.Embedding(n_id, d), side
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d + side(torch.arange(2)).shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, d))
    def forward(self, ids):
        return F.normalize(self.mlp(torch.cat([self.emb(ids), self.side(ids)], 1)), dim=1)

def train_towers(logq, epochs=15):
    torch.manual_seed(0)
    tabs = torch.nn.ModuleList([torch.nn.Embedding(7, 8), torch.nn.Embedding(2, 8), torch.nn.Embedding(21, 8)])
    user_side = lambda ids: torch.cat([t(profile[ids, c]) for c, t in enumerate(tabs)], 1)
    proj, table = torch.nn.Linear(19, 8), torch.tensor(genres)
    ut, it = Tower(n_users, user_side), Tower(n_items, lambda ids: proj(table[ids]))
    params = list(ut.parameters()) + list(it.parameters()) + list(tabs.parameters()) + list(proj.parameters())
    opt = torch.optim.Adam(params, lr=0.01)
    pu, pi = torch.tensor(fit.u.values), torch.tensor(fit.i.values)
    freq = torch.tensor(np.bincount(fit.i.values, minlength=n_items) / len(fit), dtype=torch.float32)
    for _ in range(epochs):
        order = torch.randperm(len(pu))
        for s in range(0, len(pu), 1024):
            b = order[s:s + 1024]
            logits = ut(pu[b]) @ it(pi[b]).T / 0.07
            if logq:
                logits = logits - torch.log(freq[pi[b]])[None, :]
            clash = (pi[b][:, None] == pi[b][None, :]) & ~torch.eye(len(b), dtype=torch.bool)
            loss = F.cross_entropy(logits.masked_fill(clash, -1e9), torch.arange(len(b)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return (ut(torch.arange(n_users)) @ it(torch.arange(n_items)).T).numpy()

def order_for(S, seen):
    return np.argsort(-np.where(seen, -np.inf, S), axis=1)

pop = np.tile(seen_fit.sum(0).astype(float), (n_users, 1))
S_plain, S_logq = train_towers(False), train_towers(True)
orders = {'popularity': order_for(pop, seen_all), 'two-tower plain': order_for(S_plain, seen_all), 'two-tower logQ': order_for(S_logq, seen_all)}
print('candidate recall at depth K')
for K in (10, 50, 100, 200, 400):
    row = {n: np.mean([liked_test[u, o[u, :K]].sum() / liked_test[u].sum() for u in users]) for n, o in orders.items()}
    print(f'  K={K:3d}  ' + '  '.join(f'{n} {v:.3f}' for n, v in row.items()))

S = S_logq
log_pop = np.log1p(seen_fit.sum(0))
shrunk_mean = fit.groupby('i').r.sum().reindex(range(n_items)).fillna(0).values / (seen_fit.sum(0) + 5)
taste = seen_fit.astype(np.float32) @ genres
taste = taste / np.maximum(taste.sum(1, keepdims=True), 1)
hist_len = np.log1p(seen_fit.sum(1))

def features(cand, rows):
    c, u = cand[rows], np.asarray(rows)[:, None]
    cols = [S[u, c], np.broadcast_to(np.arange(c.shape[1]), c.shape), log_pop[c], shrunk_mean[c], np.einsum('ug,ukg->uk', taste[rows], genres[c]), np.broadcast_to(hist_len[u], c.shape)]
    return np.stack(cols, axis=2).reshape(-1, len(cols))

train_users = np.flatnonzero(liked_val.any(1))
cand_fit = order_for(S, seen_fit)[:, :100]
y = np.concatenate([liked_val[u, cand_fit[u]] for u in train_users])
ranker = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=0).fit(features(cand_fit, train_users), y)
print('depth  candidate recall  P@10 retrieval order  P@10 after ranker')
for K in (10, 25, 50, 100, 200, 400):
    cand = order_for(S, seen_all)[:, :K]
    prob = ranker.predict_proba(features(cand, users))[:, 1].reshape(len(users), K)
    final = np.take_along_axis(cand[users], np.argsort(-prob, axis=1)[:, :10], axis=1)
    recall = np.mean([liked_test[u, cand[u]].sum() / liked_test[u].sum() for u in users])
    before = np.mean([liked_test[u, cand[u, :10]].mean() for u in users])
    after = np.mean([liked_test[u, final[j]].mean() for j, u in enumerate(users)])
    print(f'{K:5d}  {recall:16.3f}  {before:20.3f}  {after:17.3f}')
    if K == 50:
        pool, pool_prob = cand[users], prob

unit = genres / np.maximum(np.linalg.norm(genres, axis=1, keepdims=True), 1e-9)

def mmr(c, p, lam):
    order = list(np.argsort(-p))
    chosen = [order.pop(0)]
    while len(chosen) < 10:
        gain = [p[j] - lam * max(unit[c[j]] @ unit[c[k]] for k in chosen) for j in order]
        chosen.append(order.pop(int(np.argmax(gain))))
    return c[chosen]

print('lambda  P@10   genre distance  genres per slate')
for lam in (0.0, 0.02, 0.05, 0.1):
    slates = [mmr(pool[j], pool_prob[j], lam) for j in range(len(users))]
    p10 = np.mean([liked_test[u, s].mean() for u, s in zip(users, slates)])
    dist = np.mean([1 - np.mean([unit[a] @ unit[b] for x, a in enumerate(s) for b in s[x + 1:]]) for s in slates])
    print(f'{lam:6.2f}  {p10:.3f}  {dist:14.3f}  {np.mean([(genres[s].sum(0) > 0).sum() for s in slates]):.2f}')
```

**Reading the output.** `candidate recall at depth K` is the share of a user's liked test films that appear in the first K retrieved. In the second table `P@10 retrieval order` is precision of the top 10 straight from the towers, and `P@10 after ranker` is precision after the ranker reorders K candidates. In the last table `genre distance` is one minus the mean pairwise cosine of genre vectors in the slate, and `genres per slate` counts distinct genres.

**Line by line.**

- `logits - torch.log(freq[pi[b]])[None, :]` is the correction: it subtracts the log frequency of each column's item.
- `clash` masks cases where another row in the batch holds the same film as the positive, so a true positive is never used as its own negative.
- `features` gives the ranker the tower score, the candidate's depth, log popularity, a shrunk mean rating, the match between the user's past genres and the film's genres, and the user's history length.
- The ranker is fitted on depth-100 pools from the validation period, so depth 100 is the pool shape it knows.
- `mmr` chooses each next film by its ranker probability minus `lam` times its largest genre similarity to the films already chosen.

The printed output, with PyTorch 2.14.1 and scikit-learn 1.9.1, was:

```text
candidate recall at depth K
  K= 10  popularity 0.081  two-tower plain 0.031  two-tower logQ 0.124
  K= 50  popularity 0.257  two-tower plain 0.163  two-tower logQ 0.356
  K=100  popularity 0.376  two-tower plain 0.302  two-tower logQ 0.519
  K=200  popularity 0.553  two-tower plain 0.486  two-tower logQ 0.698
  K=400  popularity 0.770  two-tower plain 0.703  two-tower logQ 0.866
depth  candidate recall  P@10 retrieval order  P@10 after ranker
   10             0.124                 0.099              0.099
   25             0.233                 0.099              0.112
   50             0.356                 0.099              0.115
  100             0.519                 0.099              0.113
  200             0.698                 0.099              0.109
  400             0.866                 0.099              0.099
lambda  P@10   genre distance  genres per slate
  0.00  0.115           0.648  9.57
  0.02  0.114           0.720  10.78
  0.05  0.107           0.793  12.04
  0.10  0.100           0.833  12.76
```

**What the numbers say.** The first surprise is that the plain two-tower model loses to popularity at every depth: at depth 10 its recall is 0.031 against 0.081, and at depth 100 it is 0.302 against 0.376. With the logQ correction the same architecture, data and seed reach 0.124 and 0.519. A retrieval model with an uncorrected sampling bias is worse than a counter. The second surprise is in the funnel table. Candidate recall climbs from 0.124 to 0.866 as depth grows, but precision at 10 after the ranker peaks at 0.115 at depth 50 and falls back to 0.099 at depth 400, the same as retrieval order. The ranker was fitted on depth-100 pools, and a much deeper pool is a different distribution, so extra recall did not turn into a better slate. The re-ranking loop shows the usual trade: raising `lam` from 0 to 0.10 lowers precision from 0.115 to 0.100 and raises distinct genres per slate from 9.57 to 12.76.

Limits: one seed, one split, a six-feature ranker that I did not tune, genre as the only diversity signal, and test relevance defined by what users chose to rate. A stronger ranker trained on varied depths might use deeper pools well. Read the shape of each curve, not the third decimal.

The third block compares exact search with two approximate indexes. MovieLens has only 1,682 films, which is too small to show the point, so the block also builds a synthetic catalogue of 100,000 unit vectors in 64 dimensions, drawn around 200 centres with noise. Faiss runs on one thread, each query is compared to the exact top 10, and the timings are per query over a batch of 500.

```python
import time

import faiss
import numpy as np

faiss.omp_set_num_threads(1)
rng = np.random.default_rng(0)
d, n_queries = 64, 500

def catalogue(n):
    centres = rng.normal(0, 1, (200, d)).astype('float32')
    x = centres[rng.integers(0, 200, n)] + 0.7 * rng.normal(0, 1, (n, d)).astype('float32')
    faiss.normalize_L2(x)
    return x

def timed(index, q, k=10):
    start = time.perf_counter()
    _, ids = index.search(q, k)
    return ids, 1000 * (time.perf_counter() - start) / len(q)

for n in (1682, 100000):
    items = catalogue(n)
    queries = catalogue(n_queries)
    exact = faiss.IndexFlatIP(d)
    exact.add(items)
    truth, t_exact = timed(exact, queries)
    print(f'catalogue {n}: exact search {t_exact:.3f} ms per query')
    nlist = 40 if n < 5000 else 512
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(items)
    ivf.add(items)
    for nprobe in (1, 4, 16):
        ivf.nprobe = nprobe
        ids, t = timed(ivf, queries)
        rec = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)])
        print(f'  IVF nlist={nlist} nprobe={nprobe:2d}: recall@10 {rec:.3f}, {t:.3f} ms per query')
    hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 80
    hnsw.add(items)
    for ef in (8, 32, 128):
        hnsw.hnsw.efSearch = ef
        ids, t = timed(hnsw, queries)
        rec = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)])
        print(f'  HNSW M=32 efSearch={ef:3d}: recall@10 {rec:.3f}, {t:.3f} ms per query')
```

**Reading the output.** `recall@10` is the share of the exact top 10 that the index returned. The `nprobe` setting is how many of the IVF index's clusters are scanned, and `efSearch` is how many graph nodes HNSW explores. Both trade speed for recall.

**Line by line.**

- `IndexFlatIP` is exact inner-product search and serves as ground truth.
- `IndexIVFFlat` first clusters the vectors into `nlist` cells, then scans only the `nprobe` closest cells.
- `IndexHNSWFlat` builds a layered neighbour graph and walks it greedily. `efConstruction` is the build-time search width.

A run with faiss-cpu 1.15.1 printed:

```text
catalogue 1682: exact search 0.009 ms per query
  IVF nlist=40 nprobe= 1: recall@10 0.241, 0.002 ms per query
  IVF nlist=40 nprobe= 4: recall@10 0.544, 0.002 ms per query
  IVF nlist=40 nprobe=16: recall@10 0.901, 0.006 ms per query
  HNSW M=32 efSearch=  8: recall@10 0.678, 0.007 ms per query
  HNSW M=32 efSearch= 32: recall@10 0.944, 0.017 ms per query
  HNSW M=32 efSearch=128: recall@10 1.000, 0.066 ms per query
catalogue 100000: exact search 0.510 ms per query
  IVF nlist=512 nprobe= 1: recall@10 0.340, 0.005 ms per query
  IVF nlist=512 nprobe= 4: recall@10 0.697, 0.011 ms per query
  IVF nlist=512 nprobe=16: recall@10 0.934, 0.029 ms per query
  HNSW M=32 efSearch=  8: recall@10 0.216, 0.016 ms per query
  HNSW M=32 efSearch= 32: recall@10 0.421, 0.042 ms per query
  HNSW M=32 efSearch=128: recall@10 0.728, 0.144 ms per query
```

At 1,682 items, exact search already takes under 10 microseconds a query, so approximation saves nothing you would notice and costs recall (0.544 for IVF at `nprobe=4`). At 100,000 items the exact scan takes between 0.4 and 0.5 milliseconds a query in my runs, and IVF with `nprobe=16` is roughly 15 to 18 times faster at recall 0.934. HNSW with the same settings did worse here, 0.728 at its widest, which should not be read as a ranking of the two methods: these vectors are isotropic noise around centres in 64 dimensions, a hard case for graph search. Timings are single-threaded, the exact scan is a batched matrix product, and run-to-run timings moved by up to about 30% in my runs, so quote recall, not milliseconds.

<Infographic src="/img/recsys-enrich/funnel-depth.svg" alt="Cards and a table show retrieval recall with and without the logQ correction, candidate recall against final precision at each depth, approximate search recall at 100,000 items and the cost of genre re-ranking." caption="Look first at the table: recall keeps rising with depth while precision after the ranker peaks at depth 50." />

## Designing with it

### Trace one request across stages

Suppose a learner opens a page after finishing a basic statistics lesson. The request contains the current lesson, language, completed prerequisites and a recent interaction sequence. A content source nominates related probability lessons, a collaborative source nominates lessons often completed next, and a popularity fallback adds broadly useful material. Candidate IDs are merged and deduplicated. A hard filter removes lessons already completed or inaccessible to the learner. A ranker then scores the remaining pairs using short-term context and lesson quality. The reranker limits repeated topics and selects the final few tiles. A request trace should record enough detail to answer whether a missing lesson was never retrieved, filtered, ranked low or displaced by a slate rule.

Each stage has a different denominator. Candidate recall asks whether relevant labelled items appeared in a broad shortlist. Ranker quality asks whether useful candidates were ordered well within the pool it received. Final-slate metrics ask what the user actually saw. A system can improve ranker AUC while reducing user outcomes if retrieval changed, a filter removed more items or a new layout shifted attention. Compare stage metrics on matched request cohorts and inspect sample traces. When training a ranker, include source and eligibility conditions that match serving; otherwise its input distribution shifts when candidate sources change.

### Train and test the towers as a pair

The query tower can use user history and current context; the item tower can use ID, text or metadata. If both are pure IDs, a new item still needs an embedding learned from interactions. Content inputs can help it enter the index sooner, but only if the item encoder was trained to use those features meaningfully. A new user may still require a default query representation or a session-based query. Two towers make candidate scoring efficient because item vectors are precomputed, but that factorisation limits cross features: a rule about one particular user's current situation and one particular item's price may be better handled by the ranker.

Training with sampled negatives can make loss values look good while the served retrieval is weak. If there are four items and one positive, a random negative may be easy. If there are millions of items, the nearest wrong item is a harder competitor. Evaluate with the real index or an exact full-catalogue reference on a representative sample, and report recall at the cutoff the ranker can afford. Check accidental positives in the negative set and how duplicate items or variants are treated. The training objective should reflect the similarity used by the index, including any vector normalisation.

### Make index freshness observable

An item can be created, updated, withdrawn or made unavailable between index builds. Define how quickly each state must reach candidate retrieval and how the final eligibility filter protects users during propagation. A newly published item may have a content vector before its interaction history exists. If its embedding enters the index only overnight, the “fresh item” path has a measurable delay. Log item publication and index-entry times, then measure that delay. If a withdrawn item still appears in ANN results, a final filter can block display, but a high filtered-item rate can shrink the candidate pool and hurt recall.

An ANN index must be assessed against an exact search reference on the same vectors. Measure approximate recall at the same cutoff and latency under realistic concurrency. Index settings can favour speed or quality, and a quality loss can vary across dense popular regions and sparse long-tail regions of the embedding space. Monitor both average and tail request latency. When rolling out new towers, create a matching index, verify offline scores and switch query and index versions together. Keep the old pair ready for rollback.

### Treat reranking as a policy

Some post-ranking decisions are obligations: suppress blocked items, respect age or access rules, enforce availability. Others are preferences: diversity, freshness, creator exposure or novelty. Encode the former as hard constraints and the latter as measurable objectives. A diversity rule that always forces one item from a weak category can lower user value; a rule that never takes a chance on a new category can narrow the slate. Compare full lists, not only scores, and log which rule changed which position. For a learning path, prerequisite fit may be a hard rule while topic variety is a soft one.

If multiple candidate sources produce the same item, preserve all source attributions. The item can be useful because it matches both content and collaborative signals. A ranker can learn from those signals, but source attribution can also leak a previous policy or become unavailable after a source is retired. Version source definitions. A source-level ablation can then show whether a route genuinely adds unique relevant items or merely duplicates items already retrieved by another route.

Write a stage contract: catalogue and eligibility snapshot, retrieval sources and cutoffs, index version, ranker input and output, reranking policy, final display positions and log schema. Define separate service-level budgets. Retrieval should be broad enough to preserve high-value items, while ranking must fit the latency budget of the chosen cutoff. Test the complete pipeline because a strong tower evaluated in isolation may degrade when its index is stale or when the ranker was trained on a different candidate mixture.

Maintain fallbacks for empty queries, new users, index outage and missing ranker features. A safe fallback should still obey eligibility. When adding a new candidate source, compare both source-level recall and final-slate outcomes; more candidates can confuse a ranker or increase latency. When adding a reranking rule, report the relevance it sacrifices and the diversity or policy benefit it produces. A list-level constraint is a product decision that deserves a measured objective, not an invisible patch.

## Where this stands in 2026

The three-stage pattern remains a useful way to reason about large recommenders. Modern embedding and ANN systems make retrieval fast, but availability, sampling and index-version problems remain. Richer rankers and policy layers are increasingly important where user satisfaction, freshness and catalogue health matter together. The stable engineering principle is to measure each stage and the final slate at the same request context.

## Common mistakes

1. **Trusting a low training loss on in-batch negatives.** The loss falls whatever the bias. The uncorrected tower here was worse than popularity. Evaluate full-catalogue recall on held-out interactions and compare it with a popularity counter.
2. **Assuming a deeper candidate list helps the ranker.** More recall looks free. Precision after the ranker peaked at depth 50 and returned to its starting value at depth 400. Train the ranker on the depths you serve.
3. **Reporting ANN speed without recall.** A fast index that misses the neighbours is a smaller catalogue. Always print recall against exact search, and test on your own embeddings.
4. **Using an ANN index on a tiny catalogue.** With 1,682 items exact search took under 10 microseconds. Approximation added error and nothing else.
5. **Encoding a hard rule as a soft penalty.** A diversity weight can be outvoted by a high score. Filter ineligible items outright and keep the penalty for preferences.

## Practice questions

<details>
<summary>Why can a stronger ranker fail to improve a recommendation?</summary>

The desired item may never enter its candidate pool. Rankers cannot select what retrieval omitted, and eligibility can remove an otherwise strong item.

</details>

<details>
<summary>What is the toy list's candidate recall at cutoff two?</summary>

B is retrieved and D is not, so one of two labelled relevant items is retrieved: 0.5.

</details>

<details>
<summary>What can break if a query tower and item index use different versions?</summary>

Their vector spaces may no longer align, making similarity scores and neighbours unreliable. Version them together and support rollback.

</details>

<details>
<summary>Why might a hard eligibility rule belong outside a soft rank score?</summary>

A soft penalty can be outweighed by other score terms. A truly ineligible item must be filtered regardless of predicted preference.

</details>

<details>
<summary><strong>Q5. (Easy)</strong> Retrieval returns 30 of a user's 60 liked films in its top 200. What is candidate recall at 200, and what is the best precision at 10 any ranker could reach?</summary>

Recall is 30/60 = 0.5. A ranker can only reorder what it receives, so it can place at most 10 of those 30 liked films in the top 10, which would be precision 1.0 for a user with enough liked candidates. For a user with fewer than 10 liked films in the pool the ceiling is lower.

</details>

<details>
<summary><strong>Q6. (Medium)</strong> A colleague says "recall rose from 0.356 to 0.866 when we deepened the pool, so the product must improve". What would you check?</summary>

Check precision of the final top 10 at each depth. In the experiment it peaked at 0.115 for depth 50 and fell to 0.099 at depth 400 because the ranker saw a different pool from the one it was trained on. Also check latency, since deeper pools cost the ranker more time.

</details>

<details>
<summary><strong>Q7. (Stretch)</strong> In the worked example the corrected loss on the popular positive rose from 0.464 to 0.775. Does that mean the model got worse?</summary>

No. The loss was recomputed with a different formula, so the two numbers are not comparable as quality. What changes is the gradient: the model can no longer lower its loss by scoring popular films down, so it stops treating popularity as a negative signal.

</details>

## Further reading

- [Google recommendation stages](https://developers.google.com/machine-learning/recommendation/overview/types) separates candidates, scoring and reranking.
- [TensorFlow Recommenders retrieval task](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval) documents the two-tower factorisation.
- [TensorFlow Recommenders retrieval tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval) shows exact and approximate serving indexes.
- [Google Research's historical two-stage system](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) is a published production example.
- [Yi, Yang, Hong and others, "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/) (RecSys 2019, opened 2026-10-09) states that in-batch loss is subject to sampling bias under power-law item distributions and proposes a frequency-based correction.
- [Malkov and Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"](https://arxiv.org/abs/1603.09320) (arXiv 2016, revised 2018, opened 2026-10-09) introduces HNSW.
- [Johnson, Douze and Jegou, "Billion-scale similarity search with GPUs"](https://arxiv.org/abs/1702.08734) (arXiv 2017, opened 2026-10-09) covers brute-force, approximate and product-quantised search.
- [GroupLens MovieLens 100K README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt) (opened 2026-10-08) for the data licence and citation.

## Check yourself

- I can compute candidate recall and explain its effect on downstream ranking.
- I can describe why two towers permit precomputed item vectors.
- I can identify negative-sampling and index-version risks.
- I can separate hard eligibility rules from soft slate preferences.
- I can explain why in-batch negatives bias a two-tower model toward anti-popularity, and apply the logQ correction by hand.
- I can measure candidate recall against a popularity baseline instead of trusting the training loss.
- I can show that deeper candidate lists raise recall but not necessarily final precision, and name why.
- I can compare an approximate index with exact search by recall and decide whether a catalogue is big enough to need one.

## Where to go next

Next is [evaluation and feedback loops](/docs/theory/recsys/evaluation-and-feedback-loops), which asks what happens when these lists feed their own training data. For the text-retrieval version of the same two stages see [neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking).
