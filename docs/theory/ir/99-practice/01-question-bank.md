---
id: ir-question-bank
title: "Information Retrieval — Question Bank"
sidebar_label: "1 · Question bank"
sidebar_position: 1
slug: /theory/ir/question-bank
description: "Thirty-six IR questions grouped by topic, with worked answers and corrections to ambiguous source calculations."
tags: [information-retrieval, practice, question-bank]
---

import Infographic from '@site/src/components/Infographic';
import RankingMetricsLab from '@site/src/components/viz/RankingMetricsLab';

**In one line.** Work through the classic and modern IR arc as 36 questions, checking each answer's assumptions before opening the solution.

## How to use this bank

Attempt a group without revealing answers, then open the `<details>` panels and compare reasoning, not just the final number. The six groups follow the lecture sequence, from Boolean retrieval to neural and multimodal search. The comprehensive bank repeats Q22 to Q36 of the main bank exactly, including answers, so each question appears once here. Both source banks are credited below; no questions are silently dropped.

Numerical questions deserve an explicit denominator or comparison set. Q11 distinguishes a linear worst-case bound from the six comparisons performed by the displayed postings. Q20 gives AP 0.756 only when three relevant documents exist in the collection. Q29 gives a cosine score but cannot establish a rank without other image candidates. These are corrections to the source answers, carried into the relevant chapters as well.

<Infographic src="/img/ir/question-bank-map.svg" alt="The 36 IR questions are grouped into six topic ranges, with the comprehensive bank's 15 repeated questions appearing only once." caption="Use the groups as a revision path and open each answer after an attempt." />

## Questions by topic

### Boolean Retrieval & Preprocessing

<details>
<summary><strong>Q1.</strong> What is an inverted index, and why is it central to IR?</summary>

A map from each term to the list (postings) of documents containing it. It lets a search engine find matching documents in time proportional to the query terms, not the corpus size; the core data structure of IR.

</details>

<details>
<summary><strong>Q2.</strong> How does the Boolean retrieval model answer a query?</summary>

Terms are combined with AND/OR/NOT; the engine intersects/unions/complements the terms' postings lists. It is precise and predictable but returns unranked sets and can be brittle (too many or too few results).

</details>

<details>
<summary><strong>Q3.</strong> Name the main text-preprocessing steps before indexing.</summary>

Tokenisation, case-folding, stop-word removal, and normalisation via stemming (crude suffix stripping) or lemmatisation (dictionary base form); reducing surface variation so related terms match.

</details>

<details>
<summary><strong>Q4.</strong> Contrast stemming and lemmatisation.</summary>

Stemming chops affixes with rules (fast, can over/under-stem: 'studies'→'studi'); lemmatisation uses vocabulary and morphology to return a valid base form ('studies'→'study'); slower but cleaner.

</details>

### Tolerant Retrieval, Index & Vector Space

<details>
<summary><strong>Q5.</strong> What is tolerant retrieval, and name two techniques.</summary>

Handling queries that don't match exactly. Techniques: wildcard queries (via permuterm or k-gram indexes) and spelling correction (edit distance, k-gram overlap) to match near-miss terms.

</details>

<details>
<summary><strong>Q6.</strong> What is index compression and why do it?</summary>

Storing the dictionary and postings in fewer bytes (e.g. gap encoding of doc IDs with variable-byte or gamma codes). It saves disk/memory and, by improving cache use, often speeds queries.

</details>

<details>
<summary><strong>Q7.</strong> Write the tf-idf weight and explain each part.</summary>

w = tf × idf, where tf is the term frequency in the document (local importance) and idf = log(N/df) down-weights terms common across the corpus. Rare, frequent-in-doc terms score highest.

</details>

<details>
<summary><strong>Q8.</strong> How does the Vector Space Model rank documents?</summary>

Documents and the query are tf-idf vectors; rank by cosine similarity cos(q,d) = (q·d)/(‖q‖‖d‖). Cosine normalises for length, so long and short documents compare fairly.

</details>

<details>
<summary><strong>Q9.</strong> For q·d = 12, ‖q‖ = 3, ‖d‖ = 8, compute the cosine similarity.</summary>

cosine = 12/(3·8) = 0.5. This is a geometric score in the stated vector space, not a universal probability of relevance; compare it with other candidates for the same query.

</details>

<details>
<summary><strong>Q10.</strong> A query retrieves 50 docs, 40 relevant, with 60 relevant in the corpus. Compute precision, recall and F1.</summary>

Precision = 40/50 = 0.80; recall = 40/60 = 0.667; F1 = 2PR/(P+R) = 0.727.

</details>

<details>
<summary><strong>Q11.</strong> Intersect the postings [1,2,4,11,31] AND [1,2,4,5,31]. What is the cost?</summary>

Intersection = [1, 2, 4, 31]. A two-pointer scan of these particular lists uses six document-ID comparisons. O(x+y) = O(10) is the worst-case linear bound for lists of length five, not the observed step count. See the Boolean retrieval chapter for the pointer trace.

</details>

<details>
<summary><strong>Q12.</strong> Give the edit distance from 'cat' to 'cart' and from 'cat' to 'dog'.</summary>

cat→cart = 1 (insert 'r'); cat→dog = 3 (substitute all three characters).

</details>

<details>
<summary><strong>Q13.</strong> Estimate the 10th most frequent term's count under Zipf if the top term occurs 100,000 times.</summary>

Zipf: freq ≈ K/r = 100,000/10 = 10,000 (a few terms dominate; hence stop-words).

</details>

<details>
<summary><strong>Q14.</strong> Estimate the vocabulary for 10⁶ tokens using Heaps' law (k=44, b=0.49).</summary>

M = 44·(10⁶)^0.49 ≈ 44×871 ≈ 38,000 distinct terms; vocabulary grows sub-linearly.

</details>

<details>
<summary><strong>Q15.</strong> Encode postings [5, 130, 132] with gaps + variable-byte. How many bytes?</summary>

Gaps = [5, 125, 2], each &lt; 128 → 1 byte each → 3 bytes (vs 12 for three 4-byte ints).

</details>

<details>
<summary><strong>Q16.</strong> N=1000 docs, a term in df=100. Compute idf and tf-idf for tf=3.</summary>

idf = log₁₀(1000/100) = 1.0; tf-idf = 3 × 1.0 = 3.0. A term in only 10 docs would have idf = 2.0.

</details>

### Classification, Clustering & Evaluation (S6-7)

<details>
<summary><strong>Q17.</strong> How does Rocchio classification work?</summary>

Compute a centroid (mean tf-idf vector) per class from labelled data and assign each document to the nearest centroid (largest cosine). Fast, but assumes single spherical classes.

</details>

<details>
<summary><strong>Q18.</strong> Describe the k-means clustering loop and the cluster hypothesis.</summary>

Assign each document to the nearest of k centroids → recompute centroids as members' means → repeat until stable. The cluster hypothesis: documents in the same cluster tend to be relevant to the same queries.

</details>

<details>
<summary><strong>Q19.</strong> Why do ranked-retrieval metrics beat plain precision/recall?</summary>

Precision/recall ignore order; ranked metrics (P@k, MAP, NDCG, MRR) reward putting relevant documents higher.

</details>

<details>
<summary><strong>Q20.</strong> Relevant docs at ranks 1, 3, 5 of 5. Compute Average Precision.</summary>

P@1=1, P@3=2/3 and P@5=3/5. If these are all three relevant documents in the collection, standard AP = (1 + 2/3 + 3/5)/3 = 0.756. If more relevant documents exist below the returned list, divide by that larger known relevant count; for four relevant documents AP is 0.567. MAP averages AP across queries.

</details>

<details>
<summary><strong>Q21.</strong> What does NDCG add over MAP?</summary>

Graded relevance and a rank discount (a hit at rank 1 beats rank 10), normalised by the ideal ranking.

</details>

### Web Search, Crawling & Link Analysis (S9-11)

<details>
<summary><strong>Q22.</strong> How does web search differ from classic IR?</summary>

Web scale (billions of pages), spam, hyperlinks and duplication; ranking fuses content (tf-idf/BM25), link authority (PageRank) and usage signals, resisting spam.

</details>

<details>
<summary><strong>Q23.</strong> A random A-indexed page is in B with prob 0.4; a random B page is in A with 0.5. Estimate |A|/|B|.</summary>

|A|/|B| = p_B/p_A = 0.5/0.4 = 1.25; A's index is 25% larger (sampling/overlap estimation).

</details>

<details>
<summary><strong>Q24.</strong> How does a crawler stay polite while scaling?</summary>

It limits the request rate per host (per-host delay) and obeys robots.txt, scaling by crawling many hosts in parallel (e.g. 500 hosts × 1/s ≈ 500 pages/s), not by raising the per-host rate.

</details>

<details>
<summary><strong>Q25.</strong> Write the PageRank formula and compute one update.</summary>

PR(p) = (1−d)/N + d·Σ_\{q→p\} PR(q)/L(q). With N=3, d=0.85, two in-links of PR 1/3 (out-degree 1): PR = 0.15/3 + 0.85·(1/3+1/3) = 0.05 + 0.567 = 0.617.

</details>

<details>
<summary><strong>Q26.</strong> Contrast PageRank and HITS.</summary>

PageRank is a query-independent global authority (random-surfer stationary distribution); HITS computes query-dependent hub and authority scores that reinforce each other.

</details>

### Cross-Language, Multimodal & Recommenders (S12-14)

<details>
<summary><strong>Q27.</strong> Contrast multilingual IR and cross-language IR (CLIR), and name three bridging strategies.</summary>

Multilingual IR indexes documents in several languages; CLIR retrieves documents in a language different from the query. Bridge by translating the query, translating documents, or shared cross-lingual embeddings (mBERT/LaBSE).

</details>

<details>
<summary><strong>Q28.</strong> How are CLIP/ALIGN trained, and how do they retrieve?</summary>

Image and text encoders are trained with a contrastive objective on image–caption pairs (matching pairs close, mismatched far); retrieve across modalities by cosine similarity in the shared space.

</details>

<details>
<summary><strong>Q29.</strong> Rank an image for a text query with text q=[1,0,1,0] and image d=[1,1,1,0].</summary>

The cosine is 2/(√2·√3) = 0.816. This is one similarity score between illustrative vectors, not actual CLIP output. Its rank cannot be known without scores for other images under the same query and encoder.

</details>

<details>
<summary><strong>Q30.</strong> Two neighbours rated an item 4 and 5 with similarities 0.8 and 0.6. Predict the rating (collaborative filtering).</summary>

Similarity-weighted average = (0.8·4 + 0.6·5)/(0.8+0.6) = 6.2/1.4 = 4.43.

</details>

<details>
<summary><strong>Q31.</strong> What is the cold-start problem, and how do content-based/hybrid recommenders help?</summary>

A new user lacks interaction history; a new item lacks co-interaction evidence. Content-based matching can score a new item if useful features exist, while a hybrid combines that route with collaborative evidence. A new user still needs an onboarding, context or exploration fallback.

</details>

### Neural IR & Synthesis (S15-16)

<details>
<summary><strong>Q32.</strong> Contrast dual-encoder (Siamese) and cross-encoder neural IR models.</summary>

Dual-encoders encode query and document separately → dot product → fast dense retrieval with ANN; cross-encoders (BERT over query+doc) are accurate but slow; used for re-ranking.

</details>

<details>
<summary><strong>Q33.</strong> Contrast lexical and semantic matching, and the standard pipeline.</summary>

Lexical (BM25) is precise for exact terms; semantic (dense embeddings) handles synonyms/paraphrase but can drift. Standard pipeline: cheap lexical/dense retrieval → neural cross-encoder re-ranker.

</details>

<details>
<summary><strong>Q34.</strong> What is RAG?</summary>

Retrieval-Augmented Generation: retrieve relevant passages and feed them to an LLM to generate a grounded, cited answer; the basis of conversational/QA-based search.

</details>

<details>
<summary><strong>Q35.</strong> What single idea underlies tf-idf, CLIP and dense retrieval?</summary>

Representing queries and items as vectors and ranking by cosine similarity in an embedding space; tf-idf, CLIP and neural encoders all share this.

</details>

<details>
<summary><strong>Q36.</strong> Which formulas should you carry into the comprehensive exam?</summary>

tf-idf = tf·log(N/df); cosine = q·d/(‖q‖‖d‖); PageRank = (1−d)/N + d·Σ PR/L; AP = mean precision at relevant ranks.

</details>

## Work three calculations before revealing an answer

### Postings intersection

For Q11, start one pointer at each sorted list's first document ID. Compare IDs; emit a match when equal, otherwise advance the pointer on the smaller ID. The visible lists have length five each, so the usual upper bound is proportional to ten postings. The actual pointer trace stops after six comparisons because both pointers reach the end shortly after the last match. This distinction matters when explaining an algorithm's asymptotic cost: a bound is not a measurement of this input.

### AP needs a relevance denominator

For Q20, relevant hits at ranks 1, 3 and 5 contribute precisions 1, 2/3 and 3/5. Their sum is about 2.267. If exactly three relevant documents exist in the judged collection, divide by three for AP 0.756. If four exist, the unseen fourth contributes zero and the denominator four gives AP 0.567. State the collection relevance assumption. Precision at five remains 3/5 in either case, which shows why AP and set-level precision answer different questions.

### Cosine is a comparison score

For Q29, the text and image vectors share two positive coordinates. Their dot product is two, the lengths are √2 and √3, and cosine is 0.816. That value is not a percentage of the image that matches the query, and it is not a rank. It may be first, last or in between depending on the other images scored by the same system. The chapter's cross-modal lab makes the extra coordinate visible, while the evaluation chapter explains how to judge a whole ranking.

## Code you can run

These short checks reproduce three bank answers without external packages. The first counts actual document-ID comparisons for Q11.

```python
left = [1, 2, 4, 11, 31]
right = [1, 2, 4, 5, 31]
i = j = comparisons = 0
intersection = []
while i < len(left) and j < len(right):
    comparisons += 1
    if left[i] == right[j]:
        intersection.append(left[i])
        i += 1
        j += 1
    elif left[i] < right[j]:
        i += 1
    else:
        j += 1
print(intersection, comparisons)
assert intersection == [1, 2, 4, 31] and comparisons == 6
```

The second makes the AP denominator explicit for Q20.

```python
precision_at_relevant = [1, 2 / 3, 3 / 5]
ap_three_known = sum(precision_at_relevant) / 3
ap_four_known = sum(precision_at_relevant) / 4
print(f"three relevant: {ap_three_known:.3f}; four relevant: {ap_four_known:.3f}")
assert round(ap_three_known, 3) == 0.756
assert round(ap_four_known, 3) == 0.567
```

Move a result in the lab and watch AP change. Its default has three known relevant documents at ranks 1, 3 and 5, matching Q20's 0.756 assumption. The data view shows precision at each rank.

<RankingMetricsLab />

## Study the cross-topic links

Many questions are connected. Q1's inverted index is the structure used by Q11's Boolean intersection and Q15's compressed postings. Q7's tf-idf weights feed Q8's cosine model, which shares its vector comparison idea with Q29's cross-modal example and Q35's synthesis. Q19's ranked metrics are needed before claiming the pipelines in Q22, Q32 or Q34 are better. Q24's crawler determines whether a relevant web page enters the collection at all; no ranking method can return a page that was never indexed.

Use those links when a question asks for a design choice rather than a formula. Name the user's task, candidate source, scoring method and evaluation evidence. For a product-code query, exact lexical matching may be more important than broad semantic similarity. For a paraphrased policy question, dense candidates can help recall. For an answer generator, the passages must contain sufficient evidence and the answer must use them faithfully. The question bank is a compact way to practise that reasoning before the solved paper.

Q23's index-size ratio also illustrates a recurring habit: write the probability's sampling denominator before manipulating the formula. A random page from A lies in B with probability $|A∩B|/|A|$; a random page from B lies in A with probability $|A∩B|/|B|$. The intersection cancels only when both measurements describe the same snapshot and matching rule. Q25's PageRank update has a different denominator, the linking page's out-degree. Confusing the two may still yield a plausible-looking number. In each numeric answer, label the collection, sample or graph to which a denominator belongs, and record how each quantity was measured.

## Go deeper

- [Stanford IR book](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) explains the classical indexes, ranking and evaluation behind the numeric questions.
- [Hybrid search ranking documentation](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking) provides a current example for the later pipeline questions.
- Built from the course lectures "ir-question-bank" and "ir-comprehensive-question-bank" (Lecture Library series). The comprehensive bank's 15 repeated questions were de-duplicated after comparing both questions and answers.

## Check your understanding

- [ ] I can work the postings, tf-idf, cosine, PageRank and AP calculations while stating assumptions.
- [ ] I can explain how crawling, indexing, ranking and evaluation connect across the course.
- [ ] I can distinguish a score from a rank and a worst-case bound from observed steps.
- [ ] I can choose a retrieval method for a concrete query and describe how to test it.
