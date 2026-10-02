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

## The idea in plain words

A large catalogue cannot always be scored item by item with an expensive model for every page request. A staged system divides the work. **Candidate generation** quickly finds hundreds or thousands of plausible items from a much larger catalogue. **Ranking** applies a richer model to that shortlist. **Reranking** applies list-level constraints and adjustments such as diversity, freshness, deduplication and eligibility. The stages solve different problems, so each needs its own metric and failure diagnosis.

If retrieval drops a useful item, the ranker cannot restore it. If retrieval returns relevant items but ranking puts them below weak ones, candidate recall is not the problem. If ranking is good but the final slate repeats the same topic or includes an unavailable item, the reranker or policy layer needs work. Trace a request through all stages, preserving candidate IDs, source names and scores so that an offline investigation can identify which stage lost an item.

<Infographic src="/img/recsys/retrieval-ranking.svg" alt="A query tower produces a user embedding; precomputed item embeddings in a retrieval index yield candidates, which a richer ranker and eligibility-aware reranker turn into a slate." caption="Candidate recall limits every later stage's ability to select a relevant item." />

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

## Designing with it

Write a stage contract: catalogue and eligibility snapshot, retrieval sources and cutoffs, index version, ranker input and output, reranking policy, final display positions and log schema. Define separate service-level budgets. Retrieval should be broad enough to preserve high-value items, while ranking must fit the latency budget of the chosen cutoff. Test the complete pipeline because a strong tower evaluated in isolation may degrade when its index is stale or when the ranker was trained on a different candidate mixture.

Maintain fallbacks for empty queries, new users, index outage and missing ranker features. A safe fallback should still obey eligibility. When adding a new candidate source, compare both source-level recall and final-slate outcomes; more candidates can confuse a ranker or increase latency. When adding a reranking rule, report the relevance it sacrifices and the diversity or policy benefit it produces. A list-level constraint is a product decision that deserves a measured objective, not an invisible patch.

## Where this stands in 2026

The three-stage pattern remains a useful way to reason about large recommenders. Modern embedding and ANN systems make retrieval fast, but availability, sampling and index-version problems remain. Richer rankers and policy layers are increasingly important where user satisfaction, freshness and catalogue health matter together. The stable engineering principle is to measure each stage and the final slate at the same request context.

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

## Further reading

- [Google recommendation stages](https://developers.google.com/machine-learning/recommendation/overview/types) separates candidates, scoring and reranking.
- [TensorFlow Recommenders retrieval task](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval) documents the two-tower factorisation.
- [TensorFlow Recommenders retrieval tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval) shows exact and approximate serving indexes.
- [Google Research's historical two-stage system](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) is a published production example.

## Check yourself

- I can compute candidate recall and explain its effect on downstream ranking.
- I can describe why two towers permit precomputed item vectors.
- I can identify negative-sampling and index-version risks.
- I can separate hard eligibility rules from soft slate preferences.
