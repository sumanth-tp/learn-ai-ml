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

:::tip Before you start

**You should already know**

- How a text query is matched against an inverted index ([Session 2](/docs/theory/ir/boolean-retrieval)) and why tokenisation matters ([Session 3](/docs/theory/ir/dictionaries-and-tolerant-search)).
- What cosine similarity measures ([Session 5](/docs/theory/ir/vector-space-and-term-weighting)).

**Reading time:** about 50 minutes, plus about 2 minutes to run the code (the first run downloads a 450 MB multilingual model and the sentence files).

**After this chapter you can**

- Choose between translating the query, translating the documents and using a shared embedding space.
- Measure cross-language retrieval accuracy by language, and read the result without trusting one average.
- Explain why a cosine score cannot be compared across languages.

:::

## In 30 seconds

Suppose you ask in English for a refund policy that is written in Spanish. Your words share nothing with the page, so a keyword index finds nothing. There are three ways out: translate your question, translate every page in advance, or turn both into numbers in a shared space where a sentence and its translation land close together.

The third route is the cheapest to run and the easiest to over-trust. The space is only as good as the languages the model learned, and for a language it barely saw, the right translation can score lower than a wrong sentence.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Cross-language retrieval | The query and the answer are in different languages | English query, Spanish page |
| Parallel sentences | The same sentence in two languages | "Are you in favour?" and "Teklifin lehine misiniz?" |
| Shared embedding space | One vector space for sentences of many languages | A sentence and its translation have cosine near 0.9 |
| Query translation | Translate the question at search time | `refund` becomes `reembolso` |
| Document translation | Translate pages at ingestion | A Spanish policy stored with an English copy |
| Transliteration | Writing a name in another script, by sound | Latin to Devanagari |
| Accuracy at 1 | Share of queries whose true translation is ranked first | 975 of 1,000 is 0.975 |
| Recall at 10 | Share of queries whose true translation is in the top 10 | 1.000 |

## The idea in plain words

A collection can be **multilingual** simply because it contains documents in several languages. **Cross-language information retrieval** goes further: a query in one language must find a useful document in another. An English query for a Spanish-language refund policy might have no shared surface term, even if the policy answers it exactly. Ordinary lexical matching fails unless the two sides are brought into a compatible representation.

There are three routes. Translate the short **query** into the document language, translate **documents** into a query language in advance, or encode query and documents with a **shared multilingual embedding** model. None is free of ambiguity. Query translation handles little text but receives little context; document translation has context but may cost much more at ingestion. A shared encoder avoids explicit text translation for candidate matching, yet requires a model validated for the languages and domain. It does not translate the evidence that a user must read.

Even after the language bridge, per-language analysis matters. Chinese text may need word segmentation without spaces. German compounds can split or join concepts in ways an English analyser does not. Inflected word forms, stop words, accents and Unicode normalisation vary. A proper name may need **transliteration** between scripts rather than semantic translation. Language detection can also be uncertain for short or mixed-language text.

<Infographic src="/img/ir/cross-language.svg" alt="Cross-language search can translate one query, translate one thousand documents, or encode documents and query in a shared space; every route still needs language-aware processing." caption="The three bridges trade processing work against ambiguity and model dependence. The counts are illustrative work units, not prices." />

:::note Added for this site

The `banco` ambiguity and work-unit comparison below are teaching examples. The unit counts compare texts processed, not measured runtime or model cost. The production field design and LaBSE details are grounded in the linked documentation and paper.

:::

Switch among the three routes and change the collection size. With **1,000 documents**, translating one query processes **1 text**, translating the collection processes **1,000**, and encoding every document plus the query counts **1,001** text-encoding units. Change the context to see why Spanish `banco` needs more than a one-word dictionary.

<CrossLanguageLab />

## Worked example, step by step

Imagine a shared space with just two dimensions, so we can compute by hand. Real models use hundreds. An English query and three Spanish sentences have been embedded and scaled to length 1, so cosine is just the dot product.

1. The query "refund for a broken parcel" is $q = (0.6, 0.8)$.
2. The Spanish sentences are $d_1$ "reembolso por un paquete roto" at $(0.8, 0.6)$, $d_2$ "reembolso por un paquete perdido" (a lost parcel) at $(0, 1)$, and $d_3$ "horario de la biblioteca" at $(1, 0)$.
3. Cosines: $q \cdot d_1 = 0.48 + 0.48 = 0.96$. $q \cdot d_2 = 0 + 0.8 = 0.80$. $q \cdot d_3 = 0.6 + 0 = 0.60$.
4. Ranking: $d_1$, then $d_2$, then $d_3$. The right sentence is first. The near-miss is second, and its score of 0.80 is close.
5. Score a set of four queries where the right sentence ranked 1, 1, 2 and 1. Accuracy at 1 is $3/4 = 0.75$. Recall at 2 is $4/4 = 1.0$. Mean reciprocal rank is $(1 + 1 + 0.5 + 1)/4 = 0.875$.

In words: each query is a vote for the single translation it should find. Accuracy at 1 counts first places, and recall at 10 forgives near-misses.

<Infographic src="/img/ir-enrich/ir2-cross-language.svg" alt="Bars of accuracy at 1 for three methods across eight languages, with Swahili standing out as low, and a small table of the mean cosine of the right translation against the best wrong sentence for four languages." caption="Look first at Swahili: the multilingual bar collapses to 0.233, and the table shows its right translation scoring 0.386 against 0.520 for the best wrong sentence." />

## How it works

### Bridging languages

- **Translate query**; Cheap; short queries ambiguous.
- **Translate docs**; Expensive; higher quality.
- **Shared embeddings**; mBERT/LaBSE map both to one space.

### Per-language issues

Each language needs its own tokenisation (no spaces in Chinese; German compounds), stemming/morphology, stop-words and Unicode encoding; transliteration handles cross-script names.


## A real system that works this way

**Azure AI Search** documents a [multi-language index design](https://learn.microsoft.com/en-us/azure/search/search-language-support) with separate fields such as an English description and a French description, each assigned an appropriate language analyser. A query can target the language-specific fields through `searchFields`; translated text can be created during indexing. The same documentation describes multilingual vector representations as another route. This is a concrete product example of both language-aware lexical processing and a shared-representation option, rather than a claim that a default analyser solves every language pair.

**LaBSE** is a research example of a shared sentence-embedding route. The [original paper](https://arxiv.org/abs/2007.01852) presents a model trained to put translations near one another in a shared space. Multilingual BERT is also often named, but a generic multilingual language model is not automatically an effective sentence-retrieval encoder. Training objective, pooling and evaluation matter. The code below does not download or run LaBSE; it demonstrates the translation decision and processing counts with a tiny fixed example.

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

### The worked example in code

This block reproduces the three cosines and the metrics of the four-query example.

```python
import numpy as np

query = np.array([0.6, 0.8])
documents = np.array([[0.8, 0.6], [0.0, 1.0], [1.0, 0.0]])
print((documents @ query).round(2), (-(documents @ query)).argsort())
ranks = np.array([1, 1, 2, 1])
print((ranks == 1).mean(), (ranks <= 2).mean(), (1 / ranks).mean())
```

**Reading the output.** The cosines print as `[0.96 0.8  0.6 ]` and the order as `[0 1 2]`. The metrics print as 0.75, 1.0 and 0.875.

### An experiment: how well does a shared space bridge real languages?

The block uses Tatoeba sentence pairs, 1,000 per language except Swahili, which has only 390. The query is the English sentence and the candidates are every sentence in the other language, so chance is 1 in 1,000. Three systems compete. The first is a lexical baseline with no translation at all: character 3-gram tf-idf, which can only match shared spellings such as names and cognates. The second is an English-only sentence encoder, `all-MiniLM-L6-v2`. The third is `paraphrase-multilingual-MiniLM-L12-v2`, a model trained so that translations share a vector. The model card (Apache 2.0) lists about 50 languages, not including Swahili. The data is the Tatoeba bitext set (CC BY 2.0). Versions used: Python 3.14.6, sentence-transformers 6.1.0, scikit-learn 1.9.1. The run takes about a minute.

```python
import gzip
import json

import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def load(language):
    path = hf_hub_download("mteb/tatoeba-bitext-mining", f"test/{language}-eng.jsonl.gz", repo_type="dataset")
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    return [r["sentence1"].strip() for r in rows], [r["sentence2"].strip() for r in rows]

def accuracy_at_1(scores):
    return float((scores.argmax(1) == np.arange(len(scores))).mean())

def recall_at_10(scores):
    top = np.argsort(-scores, axis=1)[:, :10]
    return float((top == np.arange(len(scores))[:, None]).any(1).mean())

multilingual = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
english_only = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

def embed(model, texts):
    return model.encode(texts, batch_size=64, normalize_embeddings=True)

print(f"{'language':<9}{'pairs':>6}{'char 3-gram':>12}{'English-only':>14}{'multilingual':>14}{'  (R@10)':>9}{'xx to en':>10}")
for language in ("spa", "fra", "deu", "tur", "swh", "ara", "cmn", "hin"):
    foreign, english = load(language)
    grams = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True).fit(foreign + english)
    lexical = (grams.transform(english) @ grams.transform(foreign).T).toarray()
    scores = {}
    for name, model in (("english", english_only), ("multi", multilingual)):
        scores[name] = embed(model, english) @ embed(model, foreign).T
    print(f"{language:<9}{len(foreign):>6}{accuracy_at_1(lexical):>12.3f}{accuracy_at_1(scores['english']):>14.3f}{accuracy_at_1(scores['multi']):>14.3f}{recall_at_10(scores['multi']):>9.3f}{accuracy_at_1(scores['multi'].T):>10.3f}")

foreign, english = load("tur")
scores = embed(multilingual, english) @ embed(multilingual, foreign).T
wrong = np.flatnonzero(scores.argmax(1) != np.arange(len(scores)))[:2]
for i in wrong:
    print(f"EN: {english[i]!r}\n   wanted TR {foreign[i]!r}\n   got    TR {foreign[scores[i].argmax()]!r}")
```

The output of the run:

```text


language  pairs char 3-gram  English-only  multilingual   (R@10)  xx to en
spa        1000       0.208         0.147         0.975    1.000     0.965
fra        1000       0.222         0.188         0.944    0.994     0.935
deu        1000       0.249         0.167         0.984    0.996     0.977
tur        1000       0.085         0.044         0.971    0.999     0.962
swh         390       0.126         0.100         0.233    0.444     0.203
ara        1000       0.008         0.011         0.915    0.991     0.906
cmn        1000       0.021         0.065         0.964    0.999     0.961
hin        1000       0.003         0.004         0.981    0.999     0.981
EN: 'I hear that you are a good tennis player.'
   wanted TR 'Ben, iyi bir tenis oyuncusu olduğunu duyuyorum.'
   got    TR 'Sen teniste iyisin.'
EN: 'Are you in favor of the proposal?'
   wanted TR 'Teklifin lehine misiniz?'
   got    TR 'Planı destekliyorsun, değil mi?'
```

**Reading the output.** Columns two to four are accuracy at 1 for English queries against the foreign pool. `(R@10)` is recall at 10 for the multilingual model. `xx to en` runs the other direction, foreign queries against the English pool. Then three wrong Turkish retrievals are printed.

**Line by line.**

- `normalize_embeddings=True` makes every vector length 1, so the matrix product is a matrix of cosines.
- `analyzer="char_wb"` with `ngram_range=(3, 3)` splits words into 3-letter pieces, so `banco` and `bank` share almost nothing but `deutsch` and `Deutsch` share a lot.
- `scores.argmax(1) == np.arange(len(scores))` works because sentence $i$ in one list is the translation of sentence $i$ in the other.

### A second look: pool size and score scale

Does the accuracy hold up when the pool grows, and can a fixed cosine threshold work across languages? The second block runs the multilingual model on four languages with the first 100, the first 300 and all sentences, and compares the cosine of the right translation with the best wrong one.

```python
import gzip
import json

import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

def load(language):
    path = hf_hub_download("mteb/tatoeba-bitext-mining", f"test/{language}-eng.jsonl.gz", repo_type="dataset")
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    return [r["sentence1"].strip() for r in rows], [r["sentence2"].strip() for r in rows]

print(f"{'language':<9}{'pairs':>6}{'pool 100':>9}{'pool 300':>9}{'full pool':>10}{'mean cosine, right':>20}{'mean cosine, best wrong':>25}")
for language in ("deu", "tur", "hin", "swh"):
    foreign, english = load(language)
    scores = model.encode(english, normalize_embeddings=True) @ model.encode(foreign, normalize_embeddings=True).T
    row = [(scores[:n, :n].argmax(1) == np.arange(n)).mean() for n in (100, 300, len(foreign))]
    right = np.diag(scores)
    wrong = np.where(np.eye(len(scores), dtype=bool), -1, scores).max(1)
    print(f"{language:<9}{len(foreign):>6}{row[0]:>9.3f}{row[1]:>9.3f}{row[2]:>10.3f}{right.mean():>20.3f}{wrong.mean():>25.3f}")
```

```text

language  pairs pool 100 pool 300 full pool  mean cosine, right  mean cosine, best wrong
deu        1000    1.000    0.983     0.984               0.906                    0.569
tur        1000    1.000    0.970     0.971               0.910                    0.645
hin        1000    0.980    0.987     0.981               0.942                    0.614
swh         390    0.330    0.257     0.233               0.386                    0.520
```

**Reading the output.** `mean cosine, right` is the average cosine between an English sentence and its true translation. `mean cosine, best wrong` is the average of the highest cosine to any other sentence.

### What the numbers say

The shared space worked well where the model had seen the language. English queries found their exact translation first in 0.975 of cases for Spanish, 0.984 for German, 0.971 for Turkish, 0.981 for Hindi, 0.964 for Chinese (Mandarin), 0.944 for French and 0.915 for Arabic. Recall at 10 was 0.991 or higher for all of them. The English-only encoder scored between 0.004 (Hindi) and 0.188 (French), and the character 3-gram baseline between 0.003 (Hindi) and 0.249 (German). Cognates and names in Latin script carry some signal, and a script change removes it.

The surprise is Swahili. The multilingual model scored 0.233 at accuracy 1 and 0.444 at recall 10, on a pool of only 390 sentences. Its correct-pair cosine averaged 0.386, lower than the 0.520 of the best wrong sentence. For German the same figures were 0.906 and 0.569. A cosine is therefore meaningful only inside a language pair the model knows, and one global threshold of, say, 0.7 would accept almost every German match and almost no Swahili one.

The Turkish errors are not clear failures. Both printed misses return a close relative of the target: "Are you in favor of the proposal?" retrieves a sentence that reads "You support the plan, don't you?", and the tennis sentence retrieves a shorter one that drops the clause "I hear that". Tatoeba holds one gold translation per sentence, so near-paraphrases count as misses. The pool-size check moved accuracy little for the supported languages (German 1.000, 0.983, 0.984 for 100, 300 and 1,000 sentences) but monotonically for Swahili (0.330, 0.257, 0.233).

Limits: short everyday sentences, not documents or queries; eight language pairs; one model per category; exact-match gold labels; and one run with no resampling interval, so differences of a point or two between languages are not claims.

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

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Trusting one multilingual average | The model says "multilingual" | Report accuracy per language pair. Swahili was 0.233 while the other seven were 0.915 to 0.984 |
| Using one cosine threshold for every language | Cosine looks like a probability | The right Swahili translation averaged 0.386 and the best wrong one 0.520. Calibrate per language pair, or rank without a threshold |
| Counting near-paraphrases as failures without looking | The label says one gold answer | Read the misses. Both printed Turkish misses were close paraphrases of the target |
| Using a character match as a free baseline | It needs no model | It reached 0.249 for German and 0.003 for Hindi. It tells you about shared spelling, not meaning |
| Using an English-only encoder on foreign text | It runs without error | It returned the right answer first in 0.004 of Hindi queries. A model that does not fail loudly can still be useless |

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

<details>
<summary><strong>Q6.</strong> (Medium) The multilingual model scored 0.981 for Hindi and the English-only model 0.004. What does the second number tell you about running a model outside the languages it was trained on?</summary>

It tells you that a model can run without error and return confident-looking vectors that carry no cross-language meaning. For Hindi the English-only encoder sees unfamiliar script as rare word pieces, so its vectors reflect surface form, not meaning. Check any model's language list and measure on your own language pair before relying on it.

</details>

<details>
<summary><strong>Q7.</strong> (Medium) For Swahili the correct translation scored a mean cosine of 0.386 and the best wrong sentence 0.520. What happens to a rule that accepts any match above 0.5?</summary>

On average the best wrong sentence passes the rule while the correct translation fails it, so the rule accepts the wrong answers and rejects the right ones. The same rule works for German, where the figures are 0.906 and 0.569. The cosine scale is not shared across languages, so a threshold has to be set and tested for each pair.

</details>

<details>
<summary><strong>Q8.</strong> (Stretch) Design a fair test of query translation against the shared space for English queries over Spanish policy pages. What would you avoid?</summary>

Collect real English queries and judge, with a Spanish reader, which policy pages answer each. Run both routes over the same pages and report recall at 10 and nDCG by query type, including short ambiguous queries such as `bank` and queries with names or codes. Avoid scoring against machine translations of the labels, which repeats the ambiguity under test, and avoid parallel sentence benchmarks alone, which reward exact matches and ignore the near-miss page that still answers the question.

</details>

## Go deeper

- [Azure AI Search: multi-language indexing](https://learn.microsoft.com/en-us/azure/search/search-language-support); current language-aware field design.
- [LaBSE paper](https://arxiv.org/abs/2007.01852); a shared multilingual sentence-embedding model.
- [Stanford IR book: tokenisation and normalisation](https://nlp.stanford.edu/IR-book/html/htmledition/the-term-vocabulary-and-postings-lists-1.html); language-dependent lexical preparation.
- [paraphrase-multilingual-MiniLM-L12-v2 model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) (opened 2026-10-09); the language list and the Apache 2.0 licence.
- [Reimers and Gurevych: Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813) (opened 2026-10-09); the training idea that a translated sentence should map to the same place as the original.
- [Tatoeba bitext mining dataset (MTEB)](https://huggingface.co/datasets/mteb/tatoeba-bitext-mining) (opened 2026-10-09); the sentence pairs, CC BY 2.0.
- Built from the course lecture "ir-s12-cross-language" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check yourself

- [ ] I can distinguish a multilingual collection from a cross-language query task.
- [ ] I can compare query translation, document translation and shared embeddings without treating work units as measured cost.
- [ ] I can explain why tokenisation, transliteration and short-query ambiguity matter.
- [ ] I can design an evaluation slice for one query-document language pair and preserve original evidence.
- [ ] I can compute cosine, accuracy at 1, recall at k and reciprocal rank by hand for a small cross-language example.
- [ ] I can report retrieval accuracy by language pair and say what the Swahili result in this chapter shows.
- [ ] I can explain why a cosine threshold cannot be shared across language pairs.
- [ ] I can say what a character 3-gram baseline does and does not tell me.

## Where to go next

Next: [Session 13, multimodal retrieval and CLIP](/docs/theory/ir/multimodal-retrieval-and-clip), which also relies on a shared space, between images and text. Related: [Neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking), where the same kind of encoder is compared with BM25.
