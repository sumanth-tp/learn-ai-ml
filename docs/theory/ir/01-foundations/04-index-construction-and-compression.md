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

## The idea in plain words

The inverted index from Session 2 is easy to build when all term-document pairs fit in memory. A real document collection may generate more pairs than RAM can hold. Index construction therefore becomes an external-memory problem: process a bounded chunk, write a sorted or directly indexed block, then merge the blocks into searchable structures. Compression keeps the resulting dictionary and postings small enough to store and fast enough to read.

The lecture names two classic approaches. **Blocked sort-based indexing (BSBI)** gathers term-ID and document-ID pairs until a block is full, sorts that block and writes it out. An external merge combines the sorted blocks. **Single-pass in-memory indexing (SPIMI)** instead builds a dictionary from terms to postings inside each block, writes the block when memory fills, and merges blocks later. The final search index may consist of multiple searchable segments rather than one giant immutable file.

In the lecture's capacity example, 10 GB of pairs with a 1 GB block budget makes **10 blocks**. They can be merged in one pass if the merge process can keep an input buffer for every run plus output space; otherwise it needs several merge levels. That is the precise meaning of the lecture's "one merge pass" statement.

<Infographic src="/img/ir/index-compression.svg" alt="Ten gigabytes of pairs become ten sorted blocks, while postings 5, 130 and 132 become gaps 5, 125 and 2 encoded in three bytes." caption="Two scales of index engineering: construct in bounded blocks, then compress each sorted postings list." />

The lecture's postings `[5, 130, 132]` become gaps `[5, 125, 2]`: store the first ID, then each difference from its predecessor. Under the variable-byte convention used below, every gap fits in one byte, so the three values need **3 bytes**. Three unsigned 32-bit absolute IDs would need **12 bytes**. This comparison covers those IDs alone; a full postings list also needs lengths, frequencies, positions or other metadata according to the index design.

:::note Beyond the lecture

The operational details below extend the lecture's BSBI, SPIMI and gap-coding outline. They connect those algorithms to current segment-based indexes and explain the conditions behind the worked byte counts.

:::

The defaults show the lecture's values: document IDs **5, 130, 132**, gaps **5, 125, 2**, and **3** variable bytes versus **12** bytes for three raw 32-bit IDs. Increase the middle gap above 127 to see its encoding require two bytes.

<GapEncodingLab />

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

**Apache Lucene** uses searchable index segments. Its [index API](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/index/package-summary.html) describes how new documents create segments and how a writer merges smaller segments to keep search efficient and reclaim dead space after deletions. A reader sees a consistent point-in-time view and refreshes to include later writes. This is the operational descendant of the lecture's block-and-merge idea, though Lucene's exact file formats and merge policy are more sophisticated than our classroom encoder.

Lucene's internal document IDs are assigned within segments and may change when segments merge. An application should use its own stable identifier for a policy, product or page. This matters if a search result links into another service or if evaluation labels refer to a document across reindexing. The stable application ID can be stored as a field, while internal IDs remain an index implementation detail.

Suppose a knowledge base imports thousands of updated policy PDFs each night. The writer can add them to new segments and make them searchable after refresh. Over time, many small segments make queries touch more structures, so merges consolidate them. Deletes are recorded and eventually reclaimed by merges. A team therefore watches ingestion throughput, segment count, merge work, index size and refresh lag together; optimising only raw byte count can make updates or queries worse.

## Code you can run

The first block reproduces the lecture's capacity and vocabulary estimates. **Heaps' law** is empirical: its constants depend on the collection, tokenisation and language, so this calculation is a planning estimate rather than a guarantee.

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

The second block gap-encodes the lecture's posting IDs and writes variable-byte integers. In this convention, the high bit marks the **last** byte of a value. The representation is unambiguous because every preceding byte has a zero high bit.

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

## Designing with it

### Keep memory use bounded

An indexing job does not need to keep the entire corpus in RAM. Set a memory budget for term buffers, flush blocks when it is reached and use sequential disk I/O for large runs. Reserve memory for sorting or per-block dictionaries, input buffers for the merge and output buffers. If a 1 GB machine devotes all 1 GB to a block, the merge has no room to run. The lecture's 10-block calculation is an order-of-magnitude model, not a complete resource plan.

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

## Go deeper

- [Stanford IR book: index construction](https://nlp.stanford.edu/IR-book/html/htmledition/index-construction-1.html); BSBI, SPIMI and dynamic indexing.
- [Stanford IR book: index compression](https://nlp.stanford.edu/IR-book/html/htmledition/index-compression-1.html); vocabulary growth, dictionary storage and postings codes.
- [Apache Lucene index API](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/index/package-summary.html); current segment, reader and postings concepts.
- Built from the course lecture "ir-s4-index-compression" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check your understanding

- [ ] I can explain why BSBI and SPIMI build an index in memory-sized blocks.
- [ ] I can state when ten blocks can be merged in one pass and what resources the merge needs.
- [ ] I can turn sorted document IDs into gaps and encode the lecture's values in three variable bytes.
- [ ] I can explain why segment merges change internal IDs while external IDs should stay stable.
