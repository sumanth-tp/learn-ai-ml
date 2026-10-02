---
id: ir-cross-language-retrieval
title: "Information Retrieval · Session 12 — Cross-Language Retrieval"
sidebar_label: "12 · Cross languages"
sidebar_position: 4
slug: /theory/ir/cross-language-retrieval
description: "How query translation, document translation and shared embeddings bridge a language mismatch, with language-specific analysis and evaluation."
tags: [information-retrieval, cross-language-retrieval, multilingual-search]
---

import Infographic from '@site/src/components/Infographic';
import CrossLanguageLab from '@site/src/components/viz/CrossLanguageLab';

**In one line.** Cross-language retrieval must bridge the query and document languages while preserving the meaning, names and evidence the user needs.

## The idea in plain words

A collection can be **multilingual** simply because it contains documents in several languages. **Cross-language information retrieval** goes further: a query in one language must find a useful document in another. An English query for a Spanish-language refund policy might have no shared surface term, even if the policy answers it exactly. Ordinary lexical matching fails unless the two sides are brought into a compatible representation.

The lecture gives three routes. Translate the short **query** into the document language, translate **documents** into a query language in advance, or encode query and documents with a **shared multilingual embedding** model. None is free of ambiguity. Query translation handles little text but receives little context; document translation has context but may cost much more at ingestion. A shared encoder avoids explicit text translation for candidate matching, yet requires a model validated for the languages and domain. It does not translate the evidence that a user must read.

Even after the language bridge, per-language analysis matters. Chinese text may need word segmentation without spaces. German compounds can split or join concepts in ways an English analyser does not. Inflected word forms, stop words, accents and Unicode normalisation vary. A proper name may need **transliteration** between scripts rather than semantic translation. Language detection can also be uncertain for short or mixed-language text.

<Infographic src="/img/ir/cross-language.svg" alt="Cross-language search can translate one query, translate one thousand documents, or encode documents and query in a shared space; every route still needs language-aware processing." caption="The three bridges trade processing work against ambiguity and model dependence. The counts are illustrative work units, not prices." />

:::note Beyond the lecture

The `banco` ambiguity and work-unit comparison below are teaching examples. The unit counts compare texts processed, not measured runtime or model cost. The production field design and LaBSE details are grounded in the linked documentation and paper.

:::

Switch among the three routes and change the collection size. With **1,000 documents**, translating one query processes **1 text**, translating the collection processes **1,000**, and encoding every document plus the query counts **1,001** text-encoding units. Change the context to see why Spanish `banco` needs more than a one-word dictionary.

<CrossLanguageLab />

## How it works

### Bridging languages

- **Translate query**; Cheap; short queries ambiguous.
- **Translate docs**; Expensive; higher quality.
- **Shared embeddings**; mBERT/LaBSE map both to one space.

### Per-language issues

Each language needs its own tokenisation (no spaces in Chinese; German compounds), stemming/morphology, stop-words and Unicode encoding; transliteration handles cross-script names.


## A real system that works this way

**Azure AI Search** documents a [multi-language index design](https://learn.microsoft.com/en-us/azure/search/search-language-support) with separate fields such as an English description and a French description, each assigned an appropriate language analyser. A query can target the language-specific fields through `searchFields`; translated text can be created during indexing. The same documentation describes multilingual vector representations as another route. This is a concrete product example of both language-aware lexical processing and a shared-representation option, rather than a claim that a default analyser solves every language pair.

**LaBSE** is a research example of a shared sentence-embedding route. The [original paper](https://arxiv.org/abs/2007.01852) presents a model trained to put translations near one another in a shared space. The lecture also names multilingual BERT, but a generic multilingual language model is not automatically an effective sentence-retrieval encoder. Training objective, pooling and evaluation matter. The code below does not download or run LaBSE; it demonstrates the translation decision and processing counts with a tiny fixed example.

Imagine an employee asking in English whether a damaged parcel's delivery fee is refundable. The relevant policy may be in Spanish. A translated query can search a Spanish lexical field; a translated English field can be indexed; a multilingual encoder can retrieve a Spanish passage from the English query. In every route, the final result must retain the original document, its language, permission context and a trustworthy way to show the evidence to the employee.

## Code you can run

The first block reproduces the lab's text-processing counts for a collection of 1,000 documents. It deliberately does not compare quality or price: translation and embedding are different operations, and indexed documents can be reused across many queries.

```python
documents = 1000
query_translation_units = 1
document_translation_units = documents
shared_embedding_units = documents + 1
print("query translation:", query_translation_units)
print("document translation:", document_translation_units)
print("shared-space encoding:", shared_embedding_units)
assert (query_translation_units, document_translation_units,
        shared_embedding_units) == (1, 1000, 1001)
```

The second block shows a translation ambiguity, not an automatic translator. Spanish `banco` may mean a financial bank or a bench. A context label determines the toy mapping. A real system should infer or ask for context, retain alternative translations when useful and evaluate errors with fluent assessors.

```python
translations = {"finance": "bank", "furniture": "bench"}
for context, english in translations.items():
    print(f"banco + {context} context → {english}")
assert translations["finance"] != translations["furniture"]
```

For an actual evaluation, collect query-document relevance judgements across languages. Include exact names, ambiguous short queries, code-switched phrases and low-resource language pairs. The toy dictionary proves only that one word can have two readings; it does not estimate recall for any search model.

## Designing with it

### Pick a bridge from the workload

Query translation is attractive when queries are far fewer than documents and the document language is known. It can be done at request time, so it adds latency and may make a short ambiguous query worse. Translating documents once can make query-time lexical matching fast, and the longer text may give a translator more context, but it expands storage and maintenance. When the source changes, all derived translations must be refreshed. Shared embeddings can search across languages without a separate translation of every query, but encoding the corpus and maintaining a vector index have their own cost and quality risks.

| Route | Processing point | Useful when | Failure to inspect |
| --- | --- | --- | --- |
| Query translation | Each query | Few queries, clear target language | Ambiguous short terms and latency |
| Document translation | Ingestion | Stable corpus, many queries in one language | Stale or misleading translated fields |
| Shared embeddings | Ingestion and query | Cross-language semantic candidate recall | Weak alignment for a domain or language pair |

A hybrid design can preserve lexical precision for names and codes while using a multilingual encoder for paraphrases. Fuse or rerank the candidate lists only after measuring whether each path adds relevant documents. The Session 5 comparison shows the structure of this decision, but its hand-designed toy vectors are not cross-language embeddings.

### Preserve language and evidence fields

Store the original text, language tag, translated fields if any, model version and the span that supports the result. A translated snippet can help a user decide whether to open a document, but it should be clear that it is a translation. If a policy's wording is legally or operationally important, show the original alongside the translation and allow a qualified reviewer to resolve ambiguity. Do not replace an original source with a synthetic translation in the index without a way back.

Use language-appropriate analysis on lexical fields. An English stemmer applied to a Turkish or Hindi field will not repair morphology and can distort tokens. A language-agnostic analyser may be a baseline, but test it against native-language queries. For a blended index, choose fields deliberately rather than sending every query through every analyser and hoping the score will be comparable.

### Handle names and mixed scripts

People often search for a name as pronounced or written in their script. Transliteration maps sounds or spellings across scripts; translation maps meaning. A company name or person's name should usually not be semantically translated. Keep original-script and common transliterated forms where justified, with a clear policy for aliases. Unicode normalisation can make canonically equivalent spellings match, but over-normalisation may erase distinctions that matter in a language. Test with real names from the corpus.

Mixed-language documents and code-switched queries challenge a single language tag. A document can contain an English title, Hindi body and product code. Analyse fields or passages at the right granularity. Very short queries may not contain enough evidence for reliable language detection; allow a user-selected language or search multiple plausible language paths with bounded cost.

### Evaluate by language pair and task

An aggregate multilingual score can conceal poor performance for a low-resource language. Report recall and rank-aware measures by query language, document language, direction, topic and query type. Include human judgements from people who understand the source text. Machine translation can help build candidates but should not silently supply the only ground truth for sensitive cases. Check that retrieved passages truly answer the query, not merely share a broad topic.

Translation quality and retrieval quality are related but not identical. A fluent translation can omit the decisive term, while an awkward one may preserve enough keywords to retrieve the right document. A shared encoder may find semantically related passages yet miss a number, negation or proper name. Inspect those errors in the original language and compare against a language-specific lexical baseline.

## Follow one query through the three routes

Assume the English query is `refund delivery fee for damaged goods`, and the relevant Spanish page describes returned shipping charges for an item arriving broken. In query translation, translate the user's words into Spanish, then run lexical retrieval over a Spanish field. This can preserve the search engine's existing inverted index and highlight matches, but the short query may leave `refund` or `damaged` ambiguous. A translation that chooses the wrong sense will miss the passage. Generating several bounded alternatives can improve recall while adding noise and cost.

In document translation, translate the Spanish policy into an English field at ingestion. Then the English query searches that field. The fuller policy offers translation context, and one translation can serve many later queries. The cost is paid across the corpus, even for documents never searched. If the policy changes, the translated field must be regenerated before the index can accurately answer. Store the original and translation version together so stale derived text is detectable.

In a shared embedding route, encode both the English query and Spanish passage into a common vector space and retrieve nearby vectors. This can find a paraphrase without exact translated terms. It can also rank a passage about general refunds ahead of the required delivery-fee exception. Use a multilingual encoder trained for retrieval, validate its language coverage and consider a reranker or lexical path for exact conditions. The model's cosine score is not proof of the policy's answer.

### See why `banco` needs context

The word `banco` can refer to a financial bank or a bench. A one-word query translator must guess or return both. In a finance policy collection, `bank` is plausible; in furniture instructions, `bench` may be right. A full sentence or domain filter helps. The lab's context switch is intentionally explicit so readers see the choice rather than mistaking a tiny dictionary for a language model. A production system can preserve alternatives and use document context to decide, but should assess whether extra alternatives add irrelevant results.

The same issue appears with names, acronyms and specialised vocabulary. A product code should often be copied unchanged; a local institution may have an official name in two scripts; a medical or legal term may have several translations with different scope. A cross-language test collection should include these cases rather than only straightforward parallel sentences.

### Keep processing counts in perspective

For 1,000 documents, translating one query processes one text while translating all documents processes 1,000. Encoding all documents and one query processes 1,001 texts initially. These are counts of input items, not cost estimates. Documents may be long and require chunking, while queries are short. An embedding can be computed once per document version, then reused for many queries. A translation can also be cached. Real cost depends on text length, model, update rate, query volume and latency requirements. The lab isolates the basic scale difference before those variables are added.

The document count also affects maintenance. If 10% of a 1,000-document collection changes each day, a document-translation or embedding pipeline must refresh roughly 100 derived representations daily. A query-translation pipeline has no document-derived field to refresh, but it performs work for every query and may need language detection each time. Measure total lifecycle cost, not only first-build cost.

### Evaluate retrieval and answer presentation separately

A cross-language candidate may be relevant even when the user cannot read it. Search can present a translated snippet or full translation, with a route to the original. If a generated answer uses a Spanish policy, evaluate both whether the right passage was retrieved and whether the answer accurately conveyed its qualification. A wrong answer may originate in retrieval, translation, summarisation or an outdated source. Preserve stage-level traces.

Judges should know the source language or have a reliable review process. Automatic translation of judgements can introduce the same ambiguity the system is being tested on. Report confidence and examples for each language pair, especially where training data is sparse. If the system serves multilingual users, measure whether their own language is respected in result presentation, not merely whether the internal vector search finds a passage.

### Connect back to earlier IR choices

Session 3's term dictionary and tolerant matching still matter within each language. Session 5's lexical-versus-dense trade-off becomes sharper across languages: an exact term may disappear through translation, while a shared vector may miss a strict code or negation. Session 7's evaluation measures still apply, but relevance labels must reflect the cross-language task. Session 9's web corpus raises a further challenge because language, script and content freshness vary from page to page. The bridge is one part of a retrieval pipeline, not a complete multilingual product.

## Where this stands in 2026

:::info Industry view

- Azure AI Search currently documents language-specific analysers, translated fields and vector options for multi-language indexes; each approach still requires explicit field design.
- LaBSE is one research model for aligned multilingual sentence embeddings, but model coverage and domain quality must be tested per language pair.
- Cross-language retrieval for an answer system needs source-language evidence and a trustworthy presentation path, not only a multilingual candidate score.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Contrast multilingual IR and cross-language IR.</summary>

Multilingual IR indexes documents in several languages; CLIR retrieves documents in a language different from the query.<br /><em>Session 12 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Give three strategies to bridge the language gap in CLIR.</summary>

Translate the query, translate the documents, or map both into a shared cross-lingual embedding space (mBERT/LaBSE).<br /><em>Session 12 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Why is query translation harder than document translation in quality?</summary>

Queries are short and ambiguous, giving little context to disambiguate; documents provide richer context (but are costlier to translate).<br /><em>Session 12 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> Name two per-language processing challenges.</summary>

Tokenisation (no spaces in Chinese, German compounding), language-specific stemming/morphology, stop-words, and Unicode encoding.<br /><em>Session 12 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What is transliteration and why is it needed?</summary>

Converting names/terms across scripts (e.g. Latin↔Devanagari) so cross-script name queries match.<br /><em>Session 12 · conceptual</em>

</details>

## Go deeper

- [Azure AI Search: multi-language indexing](https://learn.microsoft.com/en-us/azure/search/search-language-support); current language-aware field design.
- [LaBSE paper](https://arxiv.org/abs/2007.01852); a shared multilingual sentence-embedding model.
- [Stanford IR book: tokenisation and normalisation](https://nlp.stanford.edu/IR-book/html/htmledition/the-term-vocabulary-and-postings-lists-1.html); language-dependent lexical preparation.
- Built from the course lecture "ir-s12-cross-language" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check your understanding

- [ ] I can distinguish a multilingual collection from a cross-language query task.
- [ ] I can compare query translation, document translation and shared embeddings without treating work units as measured cost.
- [ ] I can explain why tokenisation, transliteration and short-query ambiguity matter.
- [ ] I can design an evaluation slice for one query-document language pair and preserve original evidence.
