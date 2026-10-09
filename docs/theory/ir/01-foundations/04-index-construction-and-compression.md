---
id: ir-index-construction-compression
title: "Information Retrieval · Session 4 — Index Construction and Compression"
sidebar_label: "4 · Build the index"
sidebar_position: 4
slug: /theory/ir/index-construction-and-compression
description: "How BSBI and SPIMI build large indexes, why dynamic indexes merge segments, and how gap and variable-byte encoding shrink postings."
tags: [information-retrieval, index-construction, compression, postings]
---

import Infographic from '@site/src/components/Infographic';
import GapEncodingLab from '@site/src/components/viz/GapEncodingLab';

**In one line.** Build the index in memory-sized pieces, then compress sorted postings so lookup reads fewer bytes.

:::tip Before you start

**You should already know**

- What a postings list is and why it is sorted ([Session 2](/docs/theory/ir/boolean-retrieval)).
- How integers are stored as bits and bytes (a byte holds 8 bits and values 0 to 255).

**Reading time:** about 45 minutes, plus a minute to run the code.

**After this chapter you can**

- Turn a postings list into gaps and count its size in variable-byte and gamma codes by hand.
- Measure real compression ratios and decode times on a real index.
- Say which code suits rare terms and which suits frequent ones.

:::

## In 30 seconds

A postings list is a sorted list of document numbers. Writing each number in full wastes space, because neighbours are close together. Store the differences between neighbours (gaps) and the numbers shrink, and small numbers need fewer bits.

Think of directions: "go 3 blocks, then 7, then 1" is shorter than quoting the full address of every corner. The price is that to learn where you are, you must add the steps up from the start.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Gap | The difference between a document ID and the one before it | `[3, 10, 11]` -> `[3, 7, 1]` |
| Variable-byte code | Each number uses as many bytes as it needs, 7 bits of value per byte | 187 needs 2 bytes, 3 needs 1 |
| Gamma code | A bit-level code: a unary length, then the binary digits | 3 -> `011`, 7 -> `00111` |
| Compression ratio | Compressed size divided by original size | 6 bytes / 20 bytes = 30% |
| Heaps' law | Vocabulary size grows with the text as $M = kT^b$ | 1.17 million tokens, 35,734 terms |
| Block / segment | A memory-sized slice of the index, written to disk and merged later | 10 GB / 1 GB = 10 blocks |
| Document frequency (df) | How many documents contain a term | `the` in 5,000 documents, a rare gene in 1 |


## The idea in plain words

The inverted index from Session 2 is easy to build when all term-document pairs fit in memory. A real document collection may generate more pairs than RAM can hold. Index construction therefore becomes an external-memory problem: process a bounded chunk, write a sorted or directly indexed block, then merge the blocks into searchable structures. Compression keeps the resulting dictionary and postings small enough to store and fast enough to read.

Two classic approaches stand out. **Blocked sort-based indexing (BSBI)** gathers term-ID and document-ID pairs until a block is full, sorts that block and writes it out. An external merge combines the sorted blocks. **Single-pass in-memory indexing (SPIMI)** instead builds a dictionary from terms to postings inside each block, writes the block when memory fills, and merges blocks later. The final search index may consist of multiple searchable segments rather than one giant immutable file.

In a capacity example, 10 GB of pairs with a 1 GB block budget makes **10 blocks**. They can be merged in one pass if the merge process can keep an input buffer for every run plus output space; otherwise it needs several merge levels. That is the precise meaning of a "one merge pass" claim.

<Infographic src="/img/ir/index-compression.svg" alt="Ten gigabytes of pairs become ten sorted blocks, while postings 5, 130 and 132 become gaps 5, 125 and 2 encoded in three bytes." caption="Two scales of index engineering: construct in bounded blocks, then compress each sorted postings list." />

The postings `[5, 130, 132]` become gaps `[5, 125, 2]`: store the first ID, then each difference from its predecessor. Under the variable-byte convention used below, every gap fits in one byte, so the three values need **3 bytes**. Three unsigned 32-bit absolute IDs would need **12 bytes**. This comparison covers those IDs alone; a full postings list also needs lengths, frequencies, positions or other metadata according to the index design.

:::note Added for this site

The operational details below extend the BSBI, SPIMI and gap-coding outline. They connect those algorithms to current segment-based indexes and explain the conditions behind the worked byte counts.

:::

The defaults show these values: document IDs **5, 130, 132**, gaps **5, 125, 2**, and **3** variable bytes versus **12** bytes for three raw 32-bit IDs. Increase the middle gap above 127 to see its encoding require two bytes.

<GapEncodingLab />

## Worked example, step by step

Take the postings list `[3, 10, 11, 13, 200]` and compare the raw size with two codes.

1. **Raw.** Five document IDs at 4 bytes each: 20 bytes.
2. **Gaps.** Keep the first ID, then subtract neighbours: 3, 10 - 3 = 7, 11 - 10 = 1, 13 - 11 = 2, 200 - 13 = 187. The gaps are `[3, 7, 1, 2, 187]`.
3. **Variable byte.** A byte carries 7 bits of value, so any gap below 128 fits in one byte. 3, 7, 1 and 2 take 1 byte each. 187 is above 127, so it takes 2. Total: 6 bytes, which is 6/20 = 30% of the raw size.
4. **Gamma.** A gap n costs 2 x floor(log2 n) + 1 bits: a unary length followed by the binary digits. So 3 costs 3 bits, 7 costs 5, 1 costs 1, 2 costs 3 and 187 costs 15. Total: 27 bits, which rounds up to 4 bytes, 20% of the raw size.

In words: gamma pays little for small gaps and a lot for big ones; variable byte pays a fixed minimum of one byte but grows slowly. The first block below prints these numbers.

## How it works

### BSBI & SPIMI

BSBI sorts (termID,docID) blocks then external-merges; SPIMI uses per-block dictionaries. Distributed = MapReduce; dynamic = main + auxiliary index.

:::tip

**Worked.** 10 GB / 1 GB per block = 10 sorted blocks → one merge pass.

:::

### Heaps & gaps

Dictionary: dictionary-as-a-string + front-coding; sized by Heaps M=kT^b. Postings: store gaps then variable-byte/gamma codes.

:::tip

**Worked.** Heaps: 44·(10⁶)^0.49 ≈ 38,000 terms. Postings [5,130,132] → gaps [5,125,2] → 3 bytes (vs 12).

:::


## A real system that works this way

**Apache Lucene** uses searchable index segments. Its [index API](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/index/package-summary.html) describes how new documents create segments and how a writer merges smaller segments to keep search efficient and reclaim dead space after deletions. A reader sees a consistent point-in-time view and refreshes to include later writes. This is the operational descendant of the block-and-merge idea, though Lucene's exact file formats and merge policy are more sophisticated than our classroom encoder.

Lucene's internal document IDs are assigned within segments and may change when segments merge. An application should use its own stable identifier for a policy, product or page. This matters if a search result links into another service or if evaluation labels refer to a document across reindexing. The stable application ID can be stored as a field, while internal IDs remain an index implementation detail.

Suppose a knowledge base imports thousands of updated policy PDFs each night. The writer can add them to new segments and make them searchable after refresh. Over time, many small segments make queries touch more structures, so merges consolidate them. Deletes are recorded and eventually reclaimed by merges. A team therefore watches ingestion throughput, segment count, merge work, index size and refresh lag together; optimising only raw byte count can make updates or queries worse.

## Code you can run

The first block reproduces the capacity and vocabulary estimates. **Heaps' law** is empirical: its constants depend on the collection, tokenisation and language, so this calculation is a planning estimate rather than a guarantee.

```python
from math import ceil

pair_bytes = 10_000_000_000
block_bytes = 1_000_000_000
blocks = ceil(pair_bytes / block_bytes)
tokens = 1_000_000
k, exponent = 44, 0.49
estimated_terms = k * tokens ** exponent

print("sorted blocks:", blocks)
print("estimated distinct terms:", round(estimated_terms))
assert blocks == 10
assert 37_000 < estimated_terms < 39_000
```

The estimate is about 38,000 terms. A one-pass merge additionally needs sufficient fan-in and buffers; the arithmetic alone only tells us the number of blocks.

The second block gap-encodes the posting IDs and writes variable-byte integers. In this convention, the high bit marks the **last** byte of a value. The representation is unambiguous because every preceding byte has a zero high bit.

```python
def variable_byte(value):
    if value < 0:
        raise ValueError("Only non-negative integers are supported")
    chunks = []
    while True:
        chunks.insert(0, value % 128)
        value //= 128
        if value == 0:
            break
    chunks[-1] |= 128
    return bytes(chunks)

postings = [5, 130, 132]
gaps = [postings[0]] + [current - previous for previous, current in zip(postings, postings[1:])]
encoded = b"".join(variable_byte(gap) for gap in gaps)
print("gaps:", gaps)
print("encoded hex:", encoded.hex(" "))
print("compressed bytes:", len(encoded), "raw 32-bit bytes:", 4 * len(postings))
assert gaps == [5, 125, 2]
assert encoded == bytes.fromhex("85 fd 82")
assert len(encoded) == 3
```

The code prints `85 fd 82`. This is a teaching encoder for non-negative gaps, not Lucene's current on-disk codec. The comparison excludes dictionary storage, offsets and other postings fields.

### The worked example in code

This block repeats the five-ID list with NumPy and counts bytes and bits.

```python
import numpy as np

postings = np.array([3, 10, 11, 13, 200])
gaps = np.diff(postings, prepend=0)
vbyte = [1 if g < 128 else 2 for g in gaps]
gamma = [2 * int(g).bit_length() - 1 for g in gaps]
print("gaps", gaps.tolist())
print("variable byte bytes", vbyte, "total", sum(vbyte))
print("gamma bits", gamma, "total", sum(gamma), "bytes", -(-sum(gamma) // 8))
print("raw 32-bit bytes", 4 * len(postings))
```

**Reading the output.** The gaps match step 2, the variable-byte sizes `[1, 1, 1, 1, 2]` add to 6, the gamma sizes `[3, 5, 1, 3, 15]` add to 27 bits (4 bytes after rounding up), and the raw size is 20 bytes.

### An experiment on a real index

Now the same comparison runs on a real index: every term of the 5,183 SciFact abstracts (lower-cased runs of letters and digits, no stop list), giving 35,734 postings lists and 633,514 postings. The block writes each list as gaps, encodes it with a variable-byte code and a gamma code, checks that decoding returns the original list, and also tries `zlib` at level 6 on the gap bytes as a general-purpose baseline. It then times decoding and splits the size by list length.

Versions used: Python 3.14.6, scikit-learn 1.9.1, NumPy 2.5.3. The run takes about 5 seconds.

```python
import zlib
from time import perf_counter

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer

corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
texts = [d["title"] + " " + d["text"] for d in corpus]
vectoriser = CountVectorizer(binary=True, token_pattern=r"[a-z0-9]+")
matrix = vectoriser.fit_transform(texts).tocsc()
lists = [matrix.indices[matrix.indptr[c]:matrix.indptr[c + 1]] + 1 for c in range(matrix.shape[1])]
gaps = [np.diff(p, prepend=0) for p in lists]
print(len(lists), "postings lists,", sum(len(p) for p in lists), "postings in total")

def vbyte_encode(values):
    out = bytearray()
    for v in values:
        chunk = [v & 127]
        v >>= 7
        while v:
            chunk.append(v & 127)
            v >>= 7
        chunk[0] |= 128
        out += bytes(reversed(chunk))
    return bytes(out)

def vbyte_decode_numpy(data):
    raw = np.frombuffer(data, dtype=np.uint8)
    ends = np.flatnonzero(raw >= 128)
    starts = np.concatenate(([0], ends[:-1] + 1))
    owner = np.repeat(np.arange(len(ends)), ends - starts + 1)
    power = ends[owner] - np.arange(len(raw))
    return np.bincount(owner, weights=(raw & 127) * 128.0 ** power, minlength=len(ends)).astype(np.int64)

def gamma_encode(values):
    return "".join("0" * (int(v).bit_length() - 1) + format(int(v), "b") for v in values)

def gamma_decode(bits):
    out, i = [], 0
    while i < len(bits):
        zeros = 0
        while bits[i + zeros] == "0":
            zeros += 1
        out.append(int(bits[i + zeros:i + 2 * zeros + 1], 2))
        i += 2 * zeros + 1
    return out

vb = [vbyte_encode(g.tolist()) for g in gaps]
gm = [gamma_encode(g) for g in gaps]
assert all(vbyte_decode_numpy(b).tolist() == g.tolist() for b, g in zip(vb, gaps))
assert all(gamma_decode(b) == g.tolist() for b, g in zip(gm, gaps))

raw_bytes = 4 * sum(len(p) for p in lists)
sizes = {
    "raw 32-bit ids": raw_bytes,
    "gaps + variable byte": sum(len(b) for b in vb),
    "gaps + gamma": sum((len(b) + 7) // 8 for b in gm),
    "gaps + zlib level 6": sum(len(zlib.compress(g.astype(np.int32).tobytes(), 6)) for g in gaps),
}
for name, size in sizes.items():
    print(f"{name:22}{size:10d} bytes  {size / raw_bytes:6.1%} of raw")

start = perf_counter()
[vbyte_decode_numpy(b) for b in vb]
t_vb = perf_counter() - start
start = perf_counter()
[gamma_decode(b) for b in gm]
t_gm = perf_counter() - start
print(f"decode all lists: variable byte (numpy) {t_vb * 1000:.0f} ms, gamma (python loop) {t_gm * 1000:.0f} ms")

df = np.array([len(p) for p in lists])
for lo, hi in ((1, 1), (2, 9), (10, 99), (100, 10 ** 6)):
    idx = np.flatnonzero((df >= lo) & (df <= hi))
    avg_vb = sum(len(vb[i]) for i in idx) / df[idx].sum()
    avg_gm = sum(len(gm[i]) for i in idx) / df[idx].sum()
    label = f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
    print(f"df {label:>7} lists={len(idx):6d}  vbyte {avg_vb:5.2f} bytes/posting  gamma {avg_gm / 8:5.2f} bytes/posting")
```

The output of the run:

```text
35734 postings lists, 633514 postings in total
raw 32-bit ids           2534056 bytes  100.0% of raw
gaps + variable byte      754571 bytes   29.8% of raw
gaps + gamma              701524 bytes   27.7% of raw
gaps + zlib level 6      1331155 bytes   52.5% of raw
decode all lists: variable byte (numpy) 407 ms, gamma (python loop) 212 ms
df     1-1 lists= 17001  vbyte  1.97 bytes/posting  gamma  2.74 bytes/posting
df     2-9 lists= 12307  vbyte  1.88 bytes/posting  gamma  2.30 bytes/posting
df   10-99 lists=  5272  vbyte  1.38 bytes/posting  gamma  1.56 bytes/posting
df    100+ lists=  1154  vbyte  1.01 bytes/posting  gamma  0.69 bytes/posting
```

**Reading the output.** The first rows give total bytes and the share of the raw 32-bit size. The last four rows split the average cost per posting by how many documents contain the term, in four bands.

**Line by line.**

- `matrix.indices[matrix.indptr[c]:matrix.indptr[c + 1]] + 1` reads one column of the sparse matrix as a sorted list of row numbers. The `+ 1` makes document numbers start at 1, which gamma needs because it cannot code 0.
- `np.diff(p, prepend=0)` produces the gaps, with the first ID kept as its own gap.
- `vbyte_encode` sets the top bit of the last byte of each number, the convention used in the chapter. `vbyte_decode_numpy` finds those last bytes and rebuilds each value with a weighted sum.
- `gamma_encode` writes `bit_length - 1` zeros and then the binary digits, as in step 4. The `assert` lines decode every list and compare it with the original.

A second, short block measures Heaps' law on the same abstracts.

```python
import re

import numpy as np
from datasets import load_dataset

corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
seen, tokens, checkpoints = set(), 0, []
for doc in corpus:
    words = re.findall(r"[a-z0-9]+", (doc["title"] + " " + doc["text"]).lower())
    tokens += len(words)
    seen.update(words)
    checkpoints.append((tokens, len(seen)))

t, m = np.array(checkpoints, dtype=float).T
b, log_k = np.polyfit(np.log(t[50:]), np.log(m[50:]), 1)
print(f"{int(t[-1])} tokens, {int(m[-1])} distinct terms")
print(f"fitted Heaps law: M = {np.exp(log_k):.1f} * T^{b:.3f}")
print(f"constants k=44, b=0.49 predict {44 * t[-1] ** 0.49:.0f} terms")
print(f"fitted constants predict {np.exp(log_k) * t[-1] ** b:.0f} terms")
```

```text
1167322 tokens, 35734 distinct terms
fitted Heaps law: M = 30.2 * T^0.506
constants k=44, b=0.49 predict 41341 terms
fitted constants predict 35707 terms
```

The fit uses `np.polyfit` on the logarithms of the running token and term counts, skipping the first 50 documents where the curve is still noisy.

### What the numbers say

Gap coding was worth it: variable byte reached 29.8 per cent of the raw size and gamma 27.7 per cent. Gamma won overall, but it lost on the short lists, which are the majority. Of the 35,734 lists, 29,308 (82 per cent) appear in nine documents or fewer, and there variable byte used 1.97 and 1.88 bytes per posting against gamma's 2.74 and 2.30. A single-document list stores its first ID (up to 5,183) as the gap, and gamma pays about twice the bit length for it. Gamma overtook variable byte only for lists in 100 or more documents (0.69 against 1.01 bytes), where gaps are tiny.

The general-purpose `zlib` baseline reached 52.5 per cent, far worse than either purpose-built code. One reason may be per-list overhead on thousands of tiny lists, but I did not test that.

The surprise is in the decode timings. NumPy variable-byte decoding took 407 milliseconds against 212 for a plain Python gamma loop, and the ratio stayed between 1.7 and 1.9 across three runs. NumPy's fixed cost per call dominates on the 17,001 one-posting lists. That says nothing about production codecs, which decode long blocks with vector instructions, but it warns against benchmarking a codec on the wrong list lengths.

For Heaps' law the constants k = 44 and b = 0.49 predicted 41,341 terms for these 1,167,322 tokens, which is 15.7 per cent above the actual 35,734. A fit to this collection gave k = 30.2 and b = 0.506. The IR book reports k = 44 and b = 0.49 for Reuters-RCV1 (predicting 38,323 terms against 38,365 actual), so the constants belong to a collection. Limits: one small collection, document IDs in arbitrary order (renumbering similar documents together would shrink gaps, which I did not test), and Python-level timings.

<Infographic src="/img/ir-enrich/ir1-compression.svg" alt="Bars of index size as a share of raw for variable byte, gamma and zlib, and a table of bytes per posting by list length." caption="Look first at the bottom table: variable byte wins for rare terms, gamma only for terms in 100 or more documents." />

## Designing with it

### Keep memory use bounded

An indexing job does not need to keep the entire corpus in RAM. Set a memory budget for term buffers, flush blocks when it is reached and use sequential disk I/O for large runs. Reserve memory for sorting or per-block dictionaries, input buffers for the merge and output buffers. If a 1 GB machine devotes all 1 GB to a block, the merge has no room to run. The 10-block calculation is an order-of-magnitude model, not a complete resource plan.

BSBI and SPIMI both trade additional disk work for bounded memory. BSBI presumes a mapping from terms to term IDs while it sorts pairs. SPIMI can build a term dictionary independently in each block, avoiding a global term-ID map during the first pass. Both require a later merge or a query layer that can search multiple blocks. The right choice depends on corpus size, available RAM, update frequency and the implementation's data structures.

### Compress in a way the query can read

Sorted postings make gaps non-negative and often small. Variable-byte encoding uses one byte for a gap below 128, two for a larger range, and so on. It is simple to decode and skip through, but not always the smallest representation. Gamma codes use bit-level coding; block codecs can compress groups of integers more tightly or decode faster with vectorised instructions. Compare compressed size, decompression CPU, random-access behaviour and query latency on the *same* representative collection.

Dictionary compression solves a different problem. A dictionary-as-a-string stores term text contiguously rather than repeating object overhead. Front coding stores a shared prefix once for nearby sorted terms. These methods help when many terms share prefixes, but they must preserve efficient seeking. A codec that saves bytes while making every term lookup scan a long chain may lose at query time.

### Plan for updates and deletion

| Choice | Query consequence | Write consequence |
| --- | --- | --- |
| One large merged index | Fewer structures to search | Expensive rebuild or merge on every update |
| Many small segments | New data becomes searchable quickly | Queries touch more segments; background merges grow |
| Aggressive compression | Less storage and I/O | More encoding and decoding work |
| Frequent refresh | New documents appear sooner | More reader turnover and resource cost |

The correct balance depends on freshness requirements. A static archive can spend more time on construction and compression. A news or policy index may need frequent small writes and deletes. Monitor the queue from source change to searchable result, not just the time a writer acknowledges a document.

### Treat estimates as assumptions

Heaps' law expresses sublinear vocabulary growth, $M=kT^b$, but the constants change with tokenisation and corpus. A code repository, a multilingual legal archive and a news site have different vocabularies. Use the formula to make an initial memory estimate, then measure actual unique terms, term lengths, postings bytes and segment growth. The ratio of 3 to 12 bytes in the worked example is likewise local: a rare term with wide document-ID gaps may use more variable bytes, and a real index carries metadata beyond the integer list.

## Follow one document through the index lifecycle

A policy PDF arrives with an application ID and a new version. The ingestion job extracts text, normalises it and emits term occurrences. A bounded in-memory buffer accumulates those occurrences with document and position data. When the buffer reaches its limit, the writer flushes a searchable segment. The application can refresh its reader to make the segment visible, but an existing reader remains a point-in-time view until refreshed. These steps explain why "written" and "searchable" are two different timestamps.

The new segment has a term dictionary and postings for its own documents. A query that spans the whole collection can search several segments and combine their results. Each additional segment brings overhead: more dictionaries to consult, more per-segment scoring work and more files to manage. A merge groups small segments into a larger one in the background. It may rewrite document IDs, consolidate postings and reclaim space from deleted documents. The application's stable external ID remains attached to the logical PDF even though the index's internal IDs change.

An update to the PDF is often implemented as deleting the old version and inserting the new one. Search freshness depends on when the new segment becomes visible and when the old version stops appearing. A merge is not necessarily required for a deletion to be logically hidden; it is needed to reclaim some dead space. Monitor both visibility and physical storage. A dashboard that only shows disk utilisation misses stale-result risk, while a dashboard that only shows query latency misses runaway merge work.

### Understand the byte-count boundary

The worked three-ID posting shows why gaps help, but it intentionally strips the problem down. A term's complete postings may include document frequency, per-document term frequency, positions for phrase queries, offsets for highlighting and skip information for fast traversal. The dictionary itself needs term bytes and pointers into the postings. A search product may also store the original title and body snippets, doc values for sorting and filters, and vector data. Saying "the index is 3 bytes" would therefore be false; saying "these three gap-coded IDs need 3 bytes under this variable-byte convention" is precise.

Compression can help latency because fewer bytes must be read from storage or memory, but decoding takes CPU. If the working set already fits in RAM, a smaller representation can improve cache hit rates. If a query needs only the highest-scoring few documents, skip data can avoid decoding blocks that cannot compete. These effects interact: the best codec for a sequential full scan may not be the best one for top-ten search. Benchmark representative queries rather than extrapolating from a micro-example.

### Make the merge plan explicit

For ten sorted blocks, a one-pass ten-way merge needs a cursor and input buffer for each block, plus an output buffer. If file-descriptor limits or available memory permit fewer than ten streams, use multiple passes. The number of passes changes I/O cost because each pass reads and writes data again. This is why external-memory algorithms count disk passes, not only CPU comparisons. In a distributed build, blocks may live on different machines; data transfer and recovery matter as much as local sorting.

For a continuously changing corpus, merging never fully stops. A policy that merges too aggressively can consume write bandwidth and slow ingestion; one that barely merges can leave too many segments for every query. Measure segment counts, merge backlog and refresh age under the real update pattern. Then pick limits that keep both freshness and search latency within the product's targets.

## Where this stands in 2026

:::info Industry view

- Current Lucene documentation describes immutable segment cores, refreshable readers and background merging. Segment management remains central to indexing even when a product also stores vectors.
- Compressed postings reduce I/O, but a production codec balances storage, decoding speed and the ability to skip low-value blocks during top-result retrieval.
- An application should persist stable external document IDs. Lucene's internal doc IDs can be reassigned by segment merges and are unsuitable as permanent public identifiers.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Quoting "3 bytes against 12" as the size of an index | It is a clean ratio | Say what it covers: three gap-coded IDs only, without frequencies, positions or the dictionary |
| Choosing a code by size alone | Smaller is better | Measure decode time on representative lists as well. A code that saves 2 points but decodes twice as slowly can lose at query time |
| Benchmarking on a tiny corpus or on the wrong list lengths | The numbers look convincing | Report results by list length, as the df table does. Here 82 per cent of lists were short |
| Copying Heaps' constants from a textbook | They fit a famous collection | Fit k and b on your own tokens. The book's constants overshot this collection by 15.7 per cent |
| Gap-coding unsorted or duplicated IDs | The code runs | Sort and deduplicate first. A negative gap breaks both codes |

## Practice questions

<details>
<summary><strong>Q1.</strong> Contrast BSBI and SPIMI.</summary>

BSBI sorts blocks of (termID,docID) then external-merges; SPIMI builds per-block term→postings dictionaries directly (no global sort, no termID map).<br /><em>Session 4 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> 10 GB of pairs, 1 GB memory per block. How many BSBI blocks?</summary>

10/1 = 10 sorted blocks, then one external merge pass.<br /><em>Session 4 · numeric</em>

</details>

<details>
<summary><strong>Q3.</strong> State Heaps' law and estimate M for T=10⁶ (k=44, b=0.49).</summary>

M = kT^b; 44·(10⁶)^0.49 ≈ 38,000 terms (sub-linear growth).<br /><em>Session 4 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Encode postings [5,130,132] as gaps with variable-byte; how many bytes?</summary>

Gaps = [5,125,2], each &lt; 128 → 1 byte each → 3 bytes (vs 12 for 4-byte ints).<br /><em>Session 4 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> Why store gaps instead of absolute docIDs?</summary>

Postings are sorted, so gaps are small (especially for frequent terms) and compress to far fewer bits with variable-byte/gamma codes.<br /><em>Session 4 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> (Medium) Gap-encode `[2, 5, 6, 9]` and count the bytes in variable-byte code and the bits in gamma code.</summary>

The gaps are 2, 3, 1 and 3. Every gap is below 128, so variable byte uses 4 bytes. Gamma costs 3, 3, 1 and 3 bits: 10 bits, which is 2 bytes. Gamma wins here because every gap is tiny.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) In the SciFact index, variable byte beat gamma on one-document lists (1.97 against 2.74 bytes per posting) but lost on lists in 100 or more documents (1.01 against 0.69). Explain both results.</summary>

A one-document list stores one number, the document ID itself, which can be as large as 5,183 (13 bits). Gamma spends 2 x 13 - 1 = 25 bits on the largest values, about 3 bytes, while variable byte spends 2. In a frequent term's list the gaps are mostly 1, 2 or 3, and gamma codes those in 1 to 3 bits, below the one-byte minimum of variable byte.

</details>

## Go deeper

- [Stanford IR book: index construction](https://nlp.stanford.edu/IR-book/html/htmledition/index-construction-1.html); BSBI, SPIMI and dynamic indexing.
- [Stanford IR book: index compression](https://nlp.stanford.edu/IR-book/html/htmledition/index-compression-1.html); vocabulary growth, dictionary storage and postings codes.
- [Apache Lucene index API](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/index/package-summary.html); current segment, reader and postings concepts.
- [Stanford IR book: Heaps' law, estimating the number of terms](https://nlp.stanford.edu/IR-book/html/htmledition/heaps-law-estimating-the-number-of-terms-1.html); the constants k = 44 and b = 0.49 for Reuters-RCV1 and the prediction of 38,323 terms against 38,365 actual.
- [Stanford IR book: postings file compression](https://nlp.stanford.edu/IR-book/html/htmledition/postings-file-compression-1.html); why gaps between postings are short and need far less space than full document IDs.
- [SciFact in the BEIR collection](https://huggingface.co/datasets/BeIR/scifact); the 5,183 abstracts used for the index.
- Built from the course lecture "ir-s4-index-compression" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check yourself

- [ ] I can explain why BSBI and SPIMI build an index in memory-sized blocks.
- [ ] I can state when ten blocks can be merged in one pass and what resources the merge needs.
- [ ] I can turn sorted document IDs into gaps and encode the example values in three variable bytes.
- [ ] I can explain why segment merges change internal IDs while external IDs should stay stable.
- [ ] I can gap-encode a short postings list and count its size in variable-byte and gamma codes by hand.
- [ ] I can explain why gamma beat variable byte overall on SciFact but lost on rare terms.
- [ ] I can say why a codec benchmark must be split by list length and include decode time.
- [ ] I can fit Heaps' law constants to a corpus and say how far a borrowed set of constants is off.

## Where to go next

Next: [Session 5, vector space and term weighting](/docs/theory/ir/vector-space-and-term-weighting), which scores the documents that the compressed index returns. Related: [Boolean retrieval](/docs/theory/ir/boolean-retrieval), where the sorted postings are merged.
