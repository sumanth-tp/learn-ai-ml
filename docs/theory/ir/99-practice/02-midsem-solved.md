---
id: ir-midsem-solved
title: "Information Retrieval — 2026 Mid-Semester Paper, Solved"
sidebar_label: "2 · Solved mid-sem"
sidebar_position: 2
slug: /theory/ir/midsem-solved
description: "The six-question, 30-mark IR mid-semester paper with the source's worked answers and independent numeric checks."
tags: [information-retrieval, practice, mid-semester]
---

import Infographic from '@site/src/components/Infographic';

**In one line.** Work the six original questions under exam conditions, then inspect the supplied solutions and verify their numeric assumptions.

## The paper at a glance

The 2026 regular mid-semester paper has six questions worth **30 marks**: skip pointers (4), edit distance (5), a Boolean query (3), a wildcard dictionary (5), cosine ranking (6) and variable-byte plus dictionary storage (7). Try each question before expanding its worked answer. The question wording and the supplied solution are retained in the panels below. The board is an original redraw of the mark allocation.

<Infographic src="/img/ir/midsem-marks.svg" alt="Six IR exam questions allocate 4, 5, 3, 5, 6 and 7 marks, totalling 30 across postings, edit distance, Boolean logic, wildcard lookup, cosine and compression." caption="Use the mark distribution to budget time, then verify the worked calculations." />

:::note Beyond the lecture

The independent Python checks and the cautions after the paper are added to help readers test the supplied answers. The source's questions and answer panels remain visible.

:::

## Original questions and worked answers

<details>
<summary><strong>Q1.</strong> Q1 · Boolean intersection with skip pointers [4 M] cancer=[3,8,13,21,34,55,72,89,105,130,160,190,225,260,310,370], immunotherapy=[8,21,30,34,60,72,100,105,160,200,225,310,330,370,400]. (i) How many skip pointers (rule of thumb) and place them. (ii) Process 'cancer AND immunotherapy' with skips. (iii) Are skip pointers always useful?</summary>

(i) Rule of thumb: place √L evenly-spaced skips. √16=4 skips for cancer, √15≈4 for immunotherapy (e.g. cancer skips at positions 0,4,8,12 → docIDs 3,34,105,225). (ii) Intersection (walk both lists, use a skip only when the skip target ≤ the other list's current docID): common docs = **\{8, 21, 34, 72, 105, 160, 225, 310, 370\}** (9 documents). (iii) **Not always.** Skips help on long lists with few matches, but cost extra storage and are useless when the lists are short, when almost everything matches (you can't skip), or when posting lists change often (skips must be rebuilt).

</details>

<details>
<summary><strong>Q2.</strong> Q2 · Minimum edit distance KITTEN→SITTING [5 M] Compute the edit distance between KITTEN and SITTING and back-track the operations.</summary>

Levenshtein DP gives edit distance = **3**. Operations: K→S (substitute), E→I (substitute), insert G at the end → 'KITTEN'→'SITTEN'→'SITTIN'→'SITTING'. Back-tracking the DP matrix from cell (6,7) along the minimal path recovers exactly these 3 edits (2 substitutions + 1 insertion).

</details>

<details>
<summary><strong>Q3.</strong> Q3 · Boolean query on inverted index [3 M] Index: laptop=D1,D2,D3,D5,D7; gaming=D2,D3,D6,D7; lightweight=D1,D4,D5; battery=D1,D2,D5,D6; touchscreen=D3,D4,D7; refurbished=D4,D6. Process (laptop AND gaming) OR (lightweight AND battery) AND NOT refurbished.</summary>

AND binds tighter than OR. **laptop ∩ gaming** = \{D2,D3,D7\}. **lightweight ∩ battery** = \{D1,D5\}. **… AND NOT refurbished** (\{D4,D6\}) leaves \{D1,D5\}. Final OR: \{D2,D3,D7\} ∪ \{D1,D5\} = **\{D1, D2, D3, D5, D7\}**.

</details>

<details>
<summary><strong>Q4.</strong> Q4 · Tree dictionary + wildcard data* [5 M] Dictionary \{database, datascience, dataanalytics, datamining, datavisualization, design, development, deepmind\}; wildcard query data*. (i) B-Tree vs Binary Tree; optimal? (ii) How does it process data*? (iii) Matching terms. (iv) Terms examined: sequential vs tree search.</summary>

(i) **B-Tree** is optimal: it is balanced and keeps keys in sorted order with high fan-out, so a *prefix range* can be found in O(log n) and scanned contiguously (a binary tree is taller and not cache-friendly). (ii) A prefix query data* becomes the **range [data, datb)**; descend to the first key ≥ 'data', then read sequentially until the prefix no longer matches. (iii) Matching terms: **database, datascience, dataanalytics, datamining, datavisualization** (5 terms). (iv) Sequential search examines all **8** terms; tree search examines ≈ log₂(8)=**3** to locate the prefix start, then only the 5 matches.

</details>

<details>
<summary><strong>Q5.</strong> Q5 · Cosine similarity; faculty assignment [6 M] Query (machinelearning, artificialintelligence). F1=\{ML,datamining\}, F2=\{AI,robotics\}, F3=\{ML,AI,deeplearning\}. (1) Build vectors. (2) Cosine scores. (3) Best faculty.</summary>

(1) Vocabulary \{ML, DM, AI, robotics, DL\}; query q=[1,0,1,0,0]; F1=[1,1,0,0,0], F2=[0,0,1,1,0], F3=[1,0,1,0,1]. (2) cos(q,F)=q·F/(‖q‖‖F‖), ‖q‖=√2: **cos(q,F1)=1/2=0.5**; **cos(q,F2)=0.5**; cos(q,F3)=2/(√2·√3)=**0.816**. (3) **F3** is the best match (highest cosine 0.816); it shares both ML and AI.

</details>

<details>
<summary><strong>Q6.</strong> Q6 · Variable-byte encoding + dictionary storage [7 M] (A) VB-encode docIDs 1024, 1032, 250000; total bytes. (B) 500,000-term dictionary, 4 MB term string; each entry: frequency 4 B, posting pointer 4 B, term pointer variable. Bits for the term pointer? Total dictionary storage with minimum-size term pointers.</summary>

(A) VB encodes **gaps** (1024, 8, 248968): 1024 needs 11 bits→**2 B**, 8→**1 B**, 248968 needs 18 bits→**3 B** ⇒ total **6 bytes** (if raw docIDs are encoded instead: 2+2+3 = 7 bytes). (B) Term pointer must index into a 4 MB string: ⌈log₂(4·1024·1024)⌉ = **22 bits → 3 bytes**. Each entry = 4+4+3 = 11 B; 500,000 entries = **5.5 MB** of dictionary records, plus the 4 MB term string = **≈9.5 MB** total.

</details>

## Code you can run

### Check the calculations independently

The first block checks the Q1 intersection and Q3 Boolean expression from the lists in the paper. It uses sets to verify the final answer, while the worked Q1 panel discusses where a skip pointer can save comparisons. A set result cannot tell you whether a particular skip pointer was helpful; that depends on the traversal.

```python
cancer = {3, 8, 13, 21, 34, 55, 72, 89, 105, 130, 160, 190, 225, 260, 310, 370}
immunotherapy = {8, 21, 30, 34, 60, 72, 100, 105, 160, 200, 225, 310, 330, 370, 400}
common = sorted(cancer & immunotherapy)
print("Q1 intersection:", common)
assert common == [8, 21, 34, 72, 105, 160, 225, 310, 370]

laptop = {1, 2, 3, 5, 7}
gaming = {2, 3, 6, 7}
lightweight = {1, 4, 5}
battery = {1, 2, 5, 6}
refurbished = {4, 6}
answer = sorted((laptop & gaming) | ((lightweight & battery) - refurbished))
print("Q3 Boolean answer:", answer)
assert answer == [1, 2, 3, 5, 7]
```

The second block computes Q2's Levenshtein distance by dynamic programming. It verifies the minimum **three edits**; the answer panel gives one valid operation sequence.

```python
source, target = "KITTEN", "SITTING"
previous = list(range(len(target) + 1))
for source_position, source_character in enumerate(source, 1):
    current = [source_position]
    for target_position, target_character in enumerate(target, 1):
        current.append(min(
            previous[target_position] + 1,
            current[target_position - 1] + 1,
            previous[target_position - 1] + (source_character != target_character),
        ))
    previous = current
print("Q2 edit distance:", previous[-1])
assert previous[-1] == 3
```

The third block checks Q5's cosine scores. It assumes `machinelearning` and `artificialintelligence` are mapped to the abbreviations ML and AI used in the faculty profiles. Without that normalisation, the literal terms would not overlap and the stated scores would not follow.

```python
from math import sqrt

query = [1, 0, 1, 0, 0]
faculty = {"F1": [1, 1, 0, 0, 0], "F2": [0, 0, 1, 1, 0],
           "F3": [1, 0, 1, 0, 1]}

def cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    return dot / sqrt(sum(a * a for a in left) * sum(b * b for b in right))

scores = {name: cosine(query, vector) for name, vector in faculty.items()}
print({name: round(score, 3) for name, score in scores.items()})
assert round(scores["F1"], 3) == round(scores["F2"], 3) == 0.5
assert round(scores["F3"], 3) == 0.816
```

The fourth block checks Q6's gap and variable-byte count. It uses seven data bits per byte and sets the high bit only on the final byte of each encoded integer, matching the lecture's convention. The paper's three document IDs become gaps 1024, 8 and 248968, which use two, one and three bytes respectively.

```python
ids = [1024, 1032, 250000]
gaps = [ids[0]] + [later - earlier for earlier, later in zip(ids, ids[1:])]

def variable_byte(value):
    pieces = [value & 0x7F]
    value >>= 7
    while value:
        pieces.insert(0, value & 0x7F)
        value >>= 7
    pieces[-1] |= 0x80
    return pieces

encoded = [variable_byte(gap) for gap in gaps]
print("Q6 gaps:", gaps, "bytes per gap:", [len(part) for part in encoded])
assert gaps == [1024, 8, 248968]
assert [len(part) for part in encoded] == [2, 1, 3]
assert sum(map(len, encoded)) == 6
```

## Read the modelling assumptions

**Skip pointers.** A rule of thumb places skip links roughly every square root of a postings-list length. It does not guarantee that a skip is used on this particular intersection. The paper's set of nine common IDs is independently verified above; an efficient traversal depends on where the skipped target falls relative to the other pointer. The list-length bound and actual comparison count are different ideas, as in the Boolean practice bank.

**Wildcard dictionary.** A balanced B-tree is a reasonable disk-oriented answer for prefix-range lookup, but it is not uniquely "optimal" in every environment. A sorted array, trie or finite-state term dictionary can also support prefix enumeration with different memory and I/O costs. The source solution's approximate three comparisons comes from $\log_2(8)$ for a balanced binary search abstraction; B-tree height depends on node fan-out. Its five matching `data` terms are the stable part of the answer.

**Cosine aliases.** Q5 writes full query terms but abbreviations in the faculty profiles. The supplied vectors implicitly normalise `machinelearning` to ML and `artificialintelligence` to AI. That mapping must be stated before calculating the reported 0.5, 0.5 and 0.816 scores. This is the same analyser-consistency problem that affects a real search index.

**Dictionary pointer size.** Q6's 4 MiB string has $2^{22}$ byte addresses, so 22 bits identify a byte offset and the minimum whole-byte pointer occupies three bytes. With 500,000 entries, 4-byte frequency and 4-byte posting pointers, records use $500000(4+4+3)=5.5$ million bytes, plus roughly 4 MiB of string data. The supplied answer calls the total about 9.5 MB by treating its MB units approximately; exact binary and decimal units should be stated if precision matters.

## Go deeper

- [Stanford IR book: postings and vocabulary](https://nlp.stanford.edu/IR-book/html/htmledition/the-term-vocabulary-and-postings-lists-1.html) supports Q1, Q4 and Q6's index structures.
- [Stanford IR book: ranked retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html) supports the cosine method in Q5.
- Built from the course lecture "ir-midsem-2026" (Lecture Library series). The original six question and answer panels are retained; the numeric checks and cautions are additional.

## Check your understanding

- [ ] I can work each of the six questions before opening its answer.
- [ ] I can verify set operations, edit distance, cosine and variable-byte counts in code.
- [ ] I can identify where a worked answer depends on an unstated analyser or data-structure assumption.
