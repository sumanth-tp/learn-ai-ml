---
id: ir-dictionaries-tolerant-search
title: "Information Retrieval · Session 3 — Dictionaries and Tolerant Search"
sidebar_label: "3 · Tolerant search"
sidebar_position: 3
slug: /theory/ir/dictionaries-and-tolerant-search
description: "Term dictionaries, normalisation, wildcards, k-grams, edit distance and phonetic matching for queries that do not exactly match the index."
tags: [information-retrieval, normalisation, edit-distance, spelling-correction]
---

import Infographic from '@site/src/components/Infographic';
import EditDistanceLab from '@site/src/components/viz/EditDistanceLab';

**In one line.** A good term dictionary makes exact lookup fast and gives spelling mistakes a bounded path back to useful documents.

:::tip Before you start

**You should already know**

- How a term dictionary points to postings lists ([Session 2](/docs/theory/ir/boolean-retrieval)).
- What a Python dictionary and a set are.

**Reading time:** about 40 minutes, plus under a minute to run the code.

**After this chapter you can**

- Compute Levenshtein and Damerau-Levenshtein distance by hand.
- Build a k-gram filter that narrows 30,000 dictionary terms to a handful before the distance check.
- Set a correction policy that does not damage valid words.

:::

## In 30 seconds

You type "baggege" and the search box finds nothing. A helpful friend would guess "baggage". To guess well, the system needs two tools. The first is a quick, rough filter that picks a few dictionary words that look similar. The second is an exact count of how many keystrokes separate your word from each candidate.

Rough first, exact second: that way the system never has to compare your word with every word in the dictionary. Think of finding a friend in a crowd by first looking only at people in a red coat, then checking faces.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Vocabulary (dictionary) | The list of distinct terms the index knows | 30,058 terms in the SciFact abstracts |
| Edit distance (Levenshtein) | Fewest insertions, deletions and substitutions to turn one word into another | `cat` -> `cart` = 1 |
| Damerau-Levenshtein | Edit distance that also counts swapping two neighbouring letters as one edit | `form` -> `from` = 1 (Levenshtein gives 2) |
| k-gram (bigram when k = 2) | A run of k characters, with `$` marking the word's start and end | `$cart$` -> `$c`, `ca`, `ar`, `rt`, `t$` |
| Jaccard overlap | Shared grams divided by all grams in either word | 3 shared / 7 in total = 0.429 |
| Candidate set | The small group of terms that passes the cheap filter | 9 terms out of 30,058 |
| Normalisation | Making different spellings of the same term agree | `CART` -> `cart` |


## The idea in plain words

An inverted index works when the analysed query term matches a dictionary term. A user, however, may type `CART` while the index stores `cart`, type `car` when seeking `cars`, or type `cart` when a document says `cat`. These are different problems. Case and spelling normalisation should make equivalent forms agree at index and query time. Tolerant matching should propose *nearby* terms only when exact lookup is insufficient or the product deliberately offers expansion.

The **term dictionary** is the vocabulary of distinct searchable terms. A hash table gives fast exact lookup. An ordered structure helps prefix lookup and range scans. A full search index also points from dictionary terms to postings, so a term candidate becomes a document candidate. The dictionary is smaller than the document collection but can still be very large; broad wildcard expansion or comparing every term by edit distance is too expensive for each keystroke.

<Infographic src="/img/ir/tolerant-terms.svg" alt="A query is normalised, narrowed to candidate terms and checked by edit distance before retrieving documents." caption="Tolerant search has two stages: find plausible terms, then verify how close they are." />

Two edit-distance checks show the idea: `cat` to `cart` costs one insertion, while `cat` to `dog` costs three substitutions. Those are **Levenshtein distances**, in which insertion, deletion and substitution each cost one. A small distance is only evidence of similar spelling. It is not proof that two words mean the same thing: `form` and `from` are both valid words with different meanings.

:::note Added for this site

The production choices below extend the vocabulary survey: Unicode handling, candidate limits, when to offer a correction, and how to evaluate whether expansion helps users.

:::

The dynamic-programming grid starts at **cat → cart = 1**. Change the second input to `dog` and its lower-right cell becomes **3**, exactly as in the Python block.

<EditDistanceLab />

## Worked example, step by step

Take the query `cart` and one dictionary word, `cort`, and ask whether the cheap filter keeps it.

1. Pad each word with `$` and list its bigrams. `$cart$` gives `$c`, `ca`, `ar`, `rt`, `t$` (5 grams). `$cort$` gives `$c`, `co`, `or`, `rt`, `t$` (5 grams).
2. Shared grams: `$c`, `rt` and `t$`, so 3. All grams in either word: 5 + 5 - 3 = 7.
3. Jaccard overlap is 3/7 = 0.429. A threshold of 0.4 keeps `cort`; a threshold of 0.5 would drop it. Short words are fragile, because one wrong letter changes two of only five grams.
4. The exact check follows. `cart` to `cort` is one substitution, so the distance is 1.

Now the typing slip `form` for `from`. Under plain Levenshtein the two letters `o` and `r` are both wrong, so two substitutions: distance 2. Damerau-Levenshtein counts the swap of two neighbouring letters as one edit: distance 1.

In words: the cheap filter measures shared character pieces, the exact check counts edits, and which edits you allow decides whether a common slip is within reach. The first block below prints these numbers.

## How it works

### Terms & dictionary

Tokenise → case-fold/normalise → stop-words → stem (studi) or lemmatise (study). Store the dictionary in hash tables (exact) or B-trees (prefix/range).

### Wildcards & correction

Wildcards via permuterm/k-gram indexes; spelling correction via edit distance (+ k-gram pre-filter); Soundex for sound-alikes.

:::tip

**Worked.** cat→cart = 1 (insert r); cat→dog = 3 (substitute all).

:::


## A real system that works this way

**Apache Lucene** exposes separate query types for prefix, wildcard and fuzzy matching. Its [search API](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/package-summary.html) describes a fuzzy query as matching terms similar under Levenshtein distance, and warns that broad wildcard patterns can be slow. That is a useful design signal: these operations are not one generic "flexible search" switch. Each expands the vocabulary differently and has a different cost.

Imagine an employee directory. A search for `Smyth` may need to find `Smith` because names have common spelling variants; phonetic coding can propose that pair. An employee ID such as `SMY-4207`, however, should usually be treated as an exact identifier, not stemmed or phonetically expanded. The same application therefore needs field-specific analysis: tolerant person-name search and exact keyword lookup for IDs. Applying one analyser to every field can make exact data unreliable.

**Soundex** is a traditional phonetic code. It can group some English names that sound alike, but it loses information and is language dependent. It is better used to propose candidates than to make a final identity decision. A person search should show the actual matched name and other identifying fields, never silently assume a phonetic match is the same person.

## Code you can run

The dynamic program below computes the minimum edits for every pair of prefixes. Row zero is the cost of inserting all characters of the candidate; column zero is the cost of deleting all characters of the original. Each interior cell takes the cheapest of delete, insert and match or substitute.

```python
def edit_distance(source, target):
    costs = list(range(len(target) + 1))
    for row, source_char in enumerate(source, 1):
        next_costs = [row]
        for column, target_char in enumerate(target, 1):
            delete = costs[column] + 1
            insert = next_costs[column - 1] + 1
            replace = costs[column - 1] + (source_char != target_char)
            next_costs.append(min(delete, insert, replace))
        costs = next_costs
    return costs[-1]

print("cat → cart:", edit_distance("cat", "cart"))
print("cat → dog:", edit_distance("cat", "dog"))
assert edit_distance("cat", "cart") == 1
assert edit_distance("cat", "dog") == 3
```

The block prints distances 1 and 3. It uses $O(mn)$ time and $O(n)$ memory for source length $m$ and target length $n$. On a large dictionary, running it against every term would still be wasteful.

A k-gram index can cheaply propose candidates. Here bigrams from a padded term provide a small overlap filter; edit distance then verifies the survivors. This is an illustration, not a full spelling-correction engine.

```python
def edit_distance(source, target):
    costs = list(range(len(target) + 1))
    for row, source_char in enumerate(source, 1):
        next_costs = [row]
        for column, target_char in enumerate(target, 1):
            next_costs.append(min(costs[column] + 1, next_costs[column - 1] + 1,
                                  costs[column - 1] + (source_char != target_char)))
        costs = next_costs
    return costs[-1]

def bigrams(term):
    padded = "$" + term.lower() + "$"
    return {padded[i:i + 2] for i in range(len(padded) - 1)}

vocabulary = ["cat", "cart", "car", "coat", "dog", "care", "card"]
query = "cart"
overlap = sorted(vocabulary, key=lambda term: (-len(bigrams(query) & bigrams(term)), term))
candidates = overlap[:5]
ranked = sorted(((edit_distance(query, term), term) for term in candidates))
print("candidate terms:", candidates)
print("closest:", ranked[:3])
assert ranked[0] == (0, "cart")
```

The `$` markers make start and end grams distinct. A real implementation stores the reverse map from gram to terms so it can avoid scanning the whole vocabulary, sets an expansion cap, and uses term frequency or context when several candidates have the same edit distance.

### The worked example in code

This block reproduces the bigram overlap and the two distances from the steps above.

```python
from rapidfuzz.distance import DamerauLevenshtein, Levenshtein

def grams(term):
    padded = "$" + term + "$"
    return {padded[i:i + 2] for i in range(len(padded) - 1)}

a, b = grams("cart"), grams("cort")
print(sorted(a), sorted(b))
print("shared", len(a & b), "union", len(a | b), "Jaccard", round(len(a & b) / len(a | b), 3))
print("Levenshtein form -> from:", Levenshtein.distance("form", "from"))
print("Damerau-Levenshtein form -> from:", DamerauLevenshtein.distance("form", "from"))
```

**Reading the output.** The two gram sets match steps 1 and 2, the overlap prints `shared 3 union 7 Jaccard 0.429`, and the two distances print 2 and 1.

### An experiment on a real vocabulary

A spelling corrector can be judged by a direct test. This block takes the SciFact vocabulary (alphabetic words of three or more letters, 30,058 of them, 13,287 occurring in only one abstract), samples 200 long, reasonably common words, and damages each in four ways: a deleted letter, an inserted letter, a substituted letter and a swap of two neighbours. It keeps the damaged strings that are not themselves dictionary words (797 of 800). For each it asks the corrector to recover the original.

The corrector picks the nearest dictionary term within a distance budget, breaking ties by how many abstracts contain the term. A second stage tests the k-gram filter: how many candidates it keeps, whether the right word survives, and how long a query takes.

Versions used: Python 3.14.6, rapidfuzz 3.14.6, scikit-learn 1.9.1, NumPy 2.5.3. The run takes about 30 seconds.

```python
import random
from collections import Counter, defaultdict
from time import perf_counter

import numpy as np
from datasets import load_dataset
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein, Levenshtein
from sklearn.feature_extraction.text import CountVectorizer

corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
texts = [d["title"] + " " + d["text"] for d in corpus]
vectoriser = CountVectorizer(binary=True, token_pattern=r"[a-z]{3,}")
matrix = vectoriser.fit_transform(texts)
vocab = list(vectoriser.get_feature_names_out())
df = np.asarray(matrix.sum(axis=0)).ravel()
known = set(vocab)

def grams(term):
    padded = "$" + term + "$"
    return {padded[i:i + 2] for i in range(len(padded) - 1)}

index = defaultdict(list)
vocab_grams = [grams(t) for t in vocab]
for position, gs in enumerate(vocab_grams):
    for g in gs:
        index[g].append(position)

rng = random.Random(0)
letters = "abcdefghijklmnopqrstuvwxyz"
def corrupt(term, kind):
    i = rng.randrange(1, len(term) - 2)
    if kind == "delete":
        return term[:i] + term[i + 1:]
    if kind == "insert":
        return term[:i] + rng.choice(letters) + term[i:]
    if kind == "substitute":
        return term[:i] + rng.choice([c for c in letters if c != term[i]]) + term[i + 1:]
    return term[:i] + term[i + 1] + term[i] + term[i + 2:]

kinds = ["delete", "insert", "substitute", "swap"]
pool = [i for i, t in enumerate(vocab) if len(t) >= 6 and df[i] >= 2 and t[3] != t[4] and t[2] != t[3]]
cases = [(t, k, corrupt(vocab[t], k)) for t in rng.sample(pool, 200) for k in kinds]
cases = [(t, k, q) for t, k, q in cases if q not in known]
print(f"{len(vocab)} terms, {int((df == 1).sum())} in one document only, {len(cases)} misspelt queries")

def correct(query, scorer, budget, candidates=None):
    pool_ids = range(len(vocab)) if candidates is None else candidates
    scored = [(scorer(query, vocab[p]), p) for p in pool_ids]
    scored = [s for s in scored if s[0] <= budget]
    if not scored:
        return -1
    best = min(s[0] for s in scored)
    return max((p for d, p in scored if d == best), key=lambda p: df[p])

for name, scorer, budget in (("Levenshtein", Levenshtein.distance, 1), ("Levenshtein", Levenshtein.distance, 2),
                             ("Damerau", DamerauLevenshtein.distance, 1)):
    accuracy = {k: np.mean([correct(q, scorer, budget) == t for t, kk, q in cases if kk == k]) for k in kinds}
    print(f"{name:12} max distance {budget}: " + "  ".join(f"{k} {v:.2f}" for k, v in accuracy.items()))

def candidates(query, threshold):
    qg = grams(query)
    shared = Counter(p for g in qg for p in index.get(g, ()))
    return [p for p, c in shared.items() if c / (len(qg) + len(vocab_grams[p]) - c) >= threshold]

print(f"{'filter':16}{'candidates':>11}{'recall':>8}{'ms/query':>10}{'accuracy':>10}")
start = perf_counter()
full = [correct(q, DamerauLevenshtein.distance, 2) == t for t, k, q in cases[:300]]
print(f"{'all terms':16}{len(vocab):11d}{1.0:8.2f}{(perf_counter() - start) / 300 * 1000:10.2f}{np.mean(full):10.2f}")
for threshold in (0.2, 0.3, 0.4):
    start = perf_counter()
    found = [candidates(q, threshold) for t, k, q in cases[:300]]
    picks = [correct(q, DamerauLevenshtein.distance, 2, c) for (t, k, q), c in zip(cases[:300], found)]
    elapsed = (perf_counter() - start) / 300 * 1000
    recall = np.mean([t in c for (t, k, q), c in zip(cases[:300], found)])
    print(f"{'k-gram J>=' + str(threshold):16}{np.mean([len(c) for c in found]):11.0f}{recall:8.2f}{elapsed:10.2f}{np.mean([p == t for (t, k, q), p in zip(cases[:300], picks)]):10.2f}")

sample = rng.sample([t for t in vocab if len(t) >= 5], 1000)
near = sum(any(d == 1 for _, d, _ in process.extract(w, vocab, scorer=Levenshtein.distance, score_cutoff=1, limit=3) if d) for w in sample)
print(near, "of 1000 valid terms have another dictionary term one edit away")
```

The output of the run:

```text
30058 terms, 13287 in one document only, 797 misspelt queries
Levenshtein  max distance 1: delete 0.90  insert 1.00  substitute 0.98  swap 0.00
Levenshtein  max distance 2: delete 0.90  insert 1.00  substitute 0.98  swap 0.78
Damerau      max distance 1: delete 0.90  insert 1.00  substitute 0.98  swap 0.99
filter           candidates  recall  ms/query  accuracy
all terms             30058    1.00     10.65      0.97
k-gram J>=0.2           347    1.00      2.04      0.97
k-gram J>=0.3            42    1.00      1.93      0.97
k-gram J>=0.4             9    1.00      1.85      0.97
431 of 1000 valid terms have another dictionary term one edit away
```

**Reading the output.** The first three rows give the share of damaged words restored to the exact original, by kind of damage. In the filter table, "candidates" is the mean number of terms the filter keeps, "recall" is the share of queries where the right term is among them, and "accuracy" is the share corrected correctly after the distance check. The last line counts valid words that have another valid word one edit away.

**Line by line.**

- `grams` pads with `$`, as in the worked example, so word starts and ends count as grams.
- `index` maps each gram to the dictionary terms containing it. `candidates` counts, for each term, how many grams it shares with the query, and keeps terms above the Jaccard threshold. Only those terms are ever compared by edit distance.
- `max(..., key=lambda p: df[p])` is the tie-break: among terms at the best distance, prefer the one found in the most abstracts.
- `corrupt` keeps the first letter and the last two letters untouched, so the damage always falls inside the word.

### What the numbers say

Plain Levenshtein with a budget of one edit repaired none of the swapped-letter slips (0.00), because a swap costs two edits. Raising the budget to two recovered 0.78 of them. Damerau-Levenshtein with a budget of one recovered 0.99. Insertions (1.00) and substitutions (0.98) were easy under every setting.

Deletions stayed at 0.90 under all three settings. I inspected the 21 failures: each had a competing dictionary word at the same distance, such as `causes` losing its `a` to become `cuses`: both `causes` and `cases` are one edit away, and the tie went to `cases`. No distance rule can fix a tie like that. Context from the rest of the query is the usual remedy.

The filter was the practical win. With a Jaccard threshold of 0.4 only 9 candidates remained out of 30,058, the right word survived every time, accuracy stayed at 0.97, and a query took about 1.9 milliseconds instead of 10.7, roughly five times faster. The surprise is the last line: 431 of 1,000 valid words have another valid word one edit away, many of them plurals such as `cell` and `cells`. A corrector that always replaces the nearest word would damage them.

Limits: one vocabulary of scientific English, targets of six or more letters (short words are harder, as the worked example shows), artificial typos instead of real user logs, pure Python timings from a single run, and tie-breaking by frequency in this corpus.

<Infographic src="/img/ir-enrich/ir1-tolerant-search.svg" alt="Left, accuracy of three distance settings for deletion, insertion, substitution and swap. Right, how many candidates each k-gram threshold keeps." caption="Look first at the adjacent swap row: 0.00 under plain Levenshtein with one edit, 0.99 under Damerau-Levenshtein." />

## Designing with it

### Keep analysis consistent

Tokenisation, case folding and Unicode normalisation should be a deliberate contract between indexing and querying. If the index turns `Café` into `cafe` but the query path retains the accent, an exact lookup misses. The reverse can also happen. Normalisation choices are field- and language-dependent: accent folding may help casual article search but may blur distinct names; stripping punctuation may destroy a product code. Test real queries against the exact analysis chain, including non-ASCII names, numbers, hyphens and apostrophes.

**Stemming** removes affixes by rules and may produce a nonword such as `studi` from `studies`. **Lemmatisation** aims for a dictionary form such as `study`, often using linguistic context. Neither is automatically correct for every corpus. Aggressive stemming can conflate terms with different meanings; no stemming can split related forms into separate postings. Use a small judged query set and compare the result lists before changing a live analyser, because rebuilding the index may be necessary.

### Choose a candidate generator

| Method | What it can retrieve | Main cost or failure |
| --- | --- | --- |
| Exact dictionary lookup | Identical analysed term | Misses typos and variants |
| Prefix lookup | Terms beginning with known letters | Can expand too broadly for a short prefix |
| Permuterm index | General wildcard represented as a rotation ending in a marker | Additional dictionary storage |
| k-gram index | Terms sharing character fragments | Needs verification; common grams produce many candidates |
| Edit distance | Terms within a chosen edit budget | Expensive over an unrestricted vocabulary |
| Phonetic code | Some sound-alike names | Language bias and false matches |

For the permuterm method, append a terminal marker to each term and store its rotations. A wildcard expression can be rotated so the unknown span is at the end, turning it into a prefix lookup. This is a teaching-friendly construction; a production engine may use automata or another terms-dictionary representation. Whatever the implementation, cap the number of expanded terms and measure the worst patterns, especially those beginning with a wildcard.

### Decide when to correct

Auto-correction can help an obvious typo such as `retrival`, but it can damage a valid product name or a newly coined term. A safer interface often searches the exact term first, offers "Did you mean ...?", and lets the user choose. If the exact query has zero results, expansion is easier to justify; when it has good results, silently changing it is risky. For an identifier field, refuse fuzzy expansion entirely unless the product has a specific reviewed use case.

Context can disambiguate equally close candidates. If a user types `apple stock`, a correction mechanism should consider the surrounding word and corpus, not just the edit distance from `apple` to some other dictionary term. Frequency alone can also be misleading: common terms may swamp a rare but correct name. Keep spelling correction separate from final document ranking so you can diagnose which stage caused a bad result.

### Evaluate the change as retrieval, not just spelling

An edit-distance unit test proves that the algorithm counts edits correctly. It does not prove that suggested terms improve search. Build a query set containing genuine misspellings, valid rare terms, names and exact IDs. Measure how often a correction leads to a useful document near the top, how often a valid query is wrongly altered, and how latency changes for broad wildcards. Review failed cases by language and field. The right expansion budget is a product choice constrained by both search quality and response time.

## Work through a tolerant query

Suppose a user types `reimbursment` into a policy search. The exact dictionary lookup finds no term. A tolerant system does not immediately compare that string with every word in the corpus. It first proposes a small vocabulary set using character fragments, a prefix or another dictionary traversal. It then computes an edit distance for the candidates and may rank them using frequency and the rest of the query. `reimbursement` is a plausible candidate, but the system should still distinguish between a *suggestion* and a changed query.

Now suppose the same user types `RMB-2026`, an internal form code. Removing punctuation, stemming or fuzzy matching the code could retrieve a different form. The safe path is to recognise an identifier field or pattern and preserve an exact lookup. This illustrates why tolerant retrieval is policy as well as an algorithm. The distance threshold can vary by field, term length and context. A one-edit expansion for a three-letter acronym can create many false candidates; the same threshold for a long word may be conservative.

The `cat` to `cart` worked example has a simple path: align `c`, `a`, insert `r`, then align `t`. The dynamic-programming grid checks all possible prefix alignments rather than committing to that path in advance. In the `cat` to `dog` example, replacing each of the three letters costs three. Another path may use insertions and deletions, but cannot do better under the stated unit costs. This is why the lower-right cell is the minimum edit cost rather than a count of mismatched positions.

### Understand what k-grams buy

Character bigrams for `$cart$` include `$c`, `ca`, `ar`, `rt` and `t$`. The padded markers encode the beginning and end of a term. Terms sharing several grams are worth checking with the more expensive distance calculation. A reverse index from each gram to vocabulary terms makes that proposal step efficient. Gram overlap is an approximation: two terms can be close in edit distance but share fewer grams than a threshold expects, especially when they are short. Measure the candidate recall before tightening the filter.

After candidate generation, the system may use a maximum edit distance, a prefix constraint or a cap on expanded terms. Each limit trades recall for bounded work. A broad wildcard can create a similar expansion problem: the pattern `*tion` may match a huge portion of an English vocabulary. A query that looks short to the user can therefore be computationally large. The index should reject or constrain pathological patterns, especially in public endpoints where one request can consume shared resources.

### Treat language as a first-class input

English-oriented stemming and Soundex rules are not universal. A multilingual corpus may need language detection, separate field analysers or a conservative common normalisation that avoids destructive transformations. Unicode case folding, composed and decomposed accents, non-Latin scripts and right-to-left text need tests with actual user queries. Transliteration can help some name searches but may also create collisions. The right rule depends on what a mistaken match costs.

For names, a phonetic match should be shown as an alternative with enough context to disambiguate the person. For medical or legal records, a false identity match can be serious. For a casual article search, broader recall may be acceptable. In both settings, the evaluation set should include valid rare terms so that a "correction" feature is penalised when it changes a correct query.

### Observe correction quality after launch

Track how often users accept a suggestion, rephrase immediately, or click a result after expansion. Those signals are imperfect: a click is not necessarily satisfaction, and popular terms attract more observations. Review failures by language, field and query length rather than relying on one overall acceptance rate. Keep the exact original query in the diagnostic trace under the product's retention rules so engineers can tell whether the candidate generator, distance threshold or ranking stage caused a miss.

## Where this stands in 2026

:::info Industry view

- Tolerant search is usually a bounded candidate-generation step followed by ranking or an explicit suggestion. Edit distance alone cannot establish relevance or identity.
- Lucene's documented prefix, wildcard and fuzzy query types remain separate tools; broad wildcard patterns can be costly, while fuzzy matching targets spelling similarity.
- Multilingual and identifier-heavy corpora need field-specific analysis. A single global normalisation rule is easy to build but can erase distinctions that matter to users.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Setting the budget to one Levenshtein edit and calling typing slips handled | One edit sounds like one typo | A swapped pair costs two edits. Use Damerau-Levenshtein or allow two edits |
| Always replacing a word with its nearest neighbour | The nearest word is the most likely | Run the exact term first. Correct only when it finds nothing or when you offer "Did you mean?" (431 of 1,000 valid words have a neighbour one edit away) |
| Breaking ties by distance alone | The distance is the score | Add term frequency and the other query words. 21 deletion failures were all ties |
| Tuning the gram threshold on long words | It gave recall 1.00 | Check short words separately: `cart` against `cort` scores 0.429, only just above a threshold of 0.4 |
| Applying fuzzy matching to identifier fields | It worked on names | Keep identifiers exact. A one-edit neighbour of `SMY-4207` is a different record |

## Practice questions

<details>
<summary><strong>Q1.</strong> Name the steps that build the term vocabulary.</summary>

Tokenisation, case-folding/normalisation, stop-word removal, and stemming or lemmatisation.<br /><em>Session 3 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Contrast stemming and lemmatisation.</summary>

Stemming rule-strips affixes (fast, crude: studies→studi); lemmatisation returns a dictionary base form (studies→study).<br /><em>Session 3 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> How does a permuterm index handle a general wildcard?</summary>

It stores every rotation of each term with a $ marker, turning any wildcard into a prefix search on the rotations.<br /><em>Session 3 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> Give the edit distance from 'cat' to 'cart' and 'cat' to 'dog'.</summary>

cat→cart = 1 (insert 'r'); cat→dog = 3 (substitute all three).<br /><em>Session 3 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> What does Soundex do?</summary>

Maps words that sound alike to the same phonetic code (Smith/Smyth), useful for name search.<br /><em>Session 3 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> (Medium) Why did Levenshtein with a budget of one correct 0.00 of the swapped-letter typos while Damerau-Levenshtein got 0.99?</summary>

Swapping two neighbouring letters changes two positions. Plain Levenshtein needs two substitutions, so the correct word is at distance 2, outside a budget of 1. Damerau-Levenshtein counts the swap as a single edit, so the correct word is at distance 1.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) The k-gram filter at Jaccard 0.4 kept 9 candidates and never lost the right word, yet a colleague proposes 0.6 to save more time. What should you check first?</summary>

Check recall for short words. A one-letter error in a four-letter word like `cart` (5 grams) leaves a Jaccard of 3/7 = 0.429, so a threshold of 0.6 would drop the right word. The experiment sampled words of six or more letters, so its recall of 1.00 says nothing about short ones. Measure recall separately by word length before changing the threshold.

</details>

## Go deeper

- [Stanford IR book: dictionaries and tolerant retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/dictionaries-and-tolerant-retrieval-1.html); term structures, wildcard expansion and correction.
- [Apache Lucene search API](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/package-summary.html); current query types for prefix, wildcard and fuzzy search.
- [Stanford IR book: dictionaries and tolerant retrieval, spelling correction](https://nlp.stanford.edu/IR-book/html/htmledition/dictionaries-and-tolerant-retrieval-1.html); the k-gram and edit-distance method used here.
- The `rapidfuzz` Python package (version 3.14.6 installed for the run) computes the Levenshtein and Damerau-Levenshtein distances used in the experiment.
- Built from the course lecture "ir-s3-dictionary-tolerant" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check yourself

- [ ] I can explain what a term dictionary stores and why query and index analysis must agree.
- [ ] I can calculate Levenshtein distance and describe how k-grams narrow candidate terms.
- [ ] I can compare stemming, lemmatisation, wildcards and phonetic matching for a concrete field.
- [ ] I can set a correction policy that protects valid names and exact identifiers.
- [ ] I can compute the bigram Jaccard overlap of two short words and say whether a threshold keeps the pair.
- [ ] I can explain why a budget of one Levenshtein edit misses swapped letters and what fixes it.
- [ ] I can describe a correction policy that tries the exact term first and breaks ties with frequency or context.

## Where to go next

Next: [Session 4, index construction and compression](/docs/theory/ir/index-construction-and-compression), which stores the dictionary and postings that this filter points into. Related: [Boolean retrieval](/docs/theory/ir/boolean-retrieval), where the corrected term is looked up.
