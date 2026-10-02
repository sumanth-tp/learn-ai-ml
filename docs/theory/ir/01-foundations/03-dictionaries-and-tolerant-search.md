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

## The idea in plain words

An inverted index works when the analysed query term matches a dictionary term. A user, however, may type `CART` while the index stores `cart`, type `car` when seeking `cars`, or type `cart` when a document says `cat`. These are different problems. Case and spelling normalisation should make equivalent forms agree at index and query time. Tolerant matching should propose *nearby* terms only when exact lookup is insufficient or the product deliberately offers expansion.

The **term dictionary** is the vocabulary of distinct searchable terms. A hash table gives fast exact lookup. An ordered structure helps prefix lookup and range scans. A full search index also points from dictionary terms to postings, so a term candidate becomes a document candidate. The dictionary is smaller than the document collection but can still be very large; broad wildcard expansion or comparing every term by edit distance is too expensive for each keystroke.

<Infographic src="/img/ir/tolerant-terms.svg" alt="A query is normalised, narrowed to candidate terms and checked by edit distance before retrieving documents." caption="Tolerant search has two stages: find plausible terms, then verify how close they are." />

The lecture gives two edit-distance checks: `cat` to `cart` costs one insertion, while `cat` to `dog` costs three substitutions. Those are **Levenshtein distances**, in which insertion, deletion and substitution each cost one. A small distance is only evidence of similar spelling. It is not proof that two words mean the same thing: `form` and `from` are both valid words with different meanings.

:::note Beyond the lecture

The production choices below extend the lecture's vocabulary survey: Unicode handling, candidate limits, when to offer a correction, and how to evaluate whether expansion helps users.

:::

The dynamic-programming grid starts at **cat → cart = 1**. Change the second input to `dog` and its lower-right cell becomes **3**, exactly as in the Python block.

<EditDistanceLab />

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

The source lecture also names **Soundex**, a traditional phonetic code. It can group some English names that sound alike, but it loses information and is language dependent. It is better used to propose candidates than to make a final identity decision. A person search should show the actual matched name and other identifying fields, never silently assume a phonetic match is the same person.

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

For the lecture's permuterm method, append a terminal marker to each term and store its rotations. A wildcard expression can be rotated so the unknown span is at the end, turning it into a prefix lookup. This is a teaching-friendly construction; a production engine may use automata or another terms-dictionary representation. Whatever the implementation, cap the number of expanded terms and measure the worst patterns, especially those beginning with a wildcard.

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

## Go deeper

- [Stanford IR book: dictionaries and tolerant retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/dictionaries-and-tolerant-retrieval-1.html); term structures, wildcard expansion and correction.
- [Apache Lucene search API](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/package-summary.html); current query types for prefix, wildcard and fuzzy search.
- Built from the course lecture "ir-s3-dictionary-tolerant" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check your understanding

- [ ] I can explain what a term dictionary stores and why query and index analysis must agree.
- [ ] I can calculate Levenshtein distance and describe how k-grams narrow candidate terms.
- [ ] I can compare stemming, lemmatisation, wildcards and phonetic matching for a concrete field.
- [ ] I can set a correction policy that protects valid names and exact identifiers.
