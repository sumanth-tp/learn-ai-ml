---
title: spaCy NLP Toolkit Master Cheatsheet
sidebar_position: 19
---

# spaCy NLP Toolkit Master Cheatsheet

## Loading pipelines and docs

| Method | Description | Code example |
|---|---|---|
| `spacy.load()` | `spacy.load(name, disable=None, exclude=None)` loads a trained pipeline. | `import spacy`<br/>`nlp = spacy.load("en_core_web_sm")`<br/>`doc = nlp("Apple is looking at buying a startup.")` |
| Blank pipeline | Starts an empty language pipeline. | `nlp = spacy.blank("en")` |
| `nlp.pipe()` | Efficient batch processing. | `texts = ["Hello world", "spaCy is fast"]`<br/>`docs = list(nlp.pipe(texts, batch_size=64))` |
| Disable components | Speeds up processing when only some annotations are needed. | `nlp = spacy.load("en_core_web_sm", disable=["ner"])` |
| Pipeline names | Inspect pipeline components. | `print(nlp.pipe_names)` |

## Tokens, spans, and docs

| Method | Description | Code example |
|---|---|---|
| Token attributes | Access text, lemma, POS, shape, stopword flags. | `for token in doc:`<br/>`    print(token.text, token.lemma_, token.pos_, token.is_stop)` |
| Sentence boundaries | Iterate detected sentences. | `for sent in doc.sents:`<br/>`    print(sent.text)` |
| Named entities | Access entity spans from NER. | `for ent in doc.ents:`<br/>`    print(ent.text, ent.label_)` |
| Noun chunks | Base noun phrase extraction. | `for chunk in doc.noun_chunks:`<br/>`    print(chunk.text)` |
| `Span` slicing | Slice docs into spans. | `span = doc[0:3]`<br/>`print(span.text)` |
| Custom extension | Attach custom fields to tokens/docs/spans. | `from spacy.tokens import Doc`<br/>`Doc.set_extension("source", default=None, force=True)`<br/>`doc._.source = "support-ticket"` |

## Dependency parsing and matching

| Method | Description | Code example |
|---|---|---|
| Dependency attributes | Inspect syntactic head and dependency labels. | `for token in doc:`<br/>`    print(token.text, token.dep_, token.head.text)` |
| Subtree | Extract syntactic subtree. | `token = doc[3]`<br/>`print(" ".join(t.text for t in token.subtree))` |
| `Matcher` | Rule-based token pattern matching. | `from spacy.matcher import Matcher`<br/>`matcher = Matcher(nlp.vocab)`<br/>`matcher.add("EMAIL_HELP", [[{"LOWER": "reset"}, {"LOWER": "password"}]])`<br/>`matches = matcher(doc)` |
| `PhraseMatcher` | Efficient exact phrase matching. | `from spacy.matcher import PhraseMatcher`<br/>`matcher = PhraseMatcher(nlp.vocab)`<br/>`patterns = [nlp.make_doc(text) for text in ["machine learning", "deep learning"]]`<br/>`matcher.add("SKILL", patterns)` |
| Entity ruler | Adds rule-based entities before or after NER. | `ruler = nlp.add_pipe("entity_ruler", before="ner")`<br/>`ruler.add_patterns([{"label": "ORG", "pattern": "OpenAI"}])` |

## Custom components and training

| Method | Description | Code example |
|---|---|---|
| Custom component | Add deterministic logic to the pipeline. | `@spacy.Language.component("lowercase_doc")`<br/>`def lowercase_doc(doc):`<br/>`    doc._.source = "processed"`<br/>`    return doc`<br/>`nlp.add_pipe("lowercase_doc", last=True)` |
| Train config | spaCy training is config-driven. | `python -m spacy init config config.cfg --lang en --pipeline ner` |
| Train command | Trains pipeline from config and data. | `python -m spacy train config.cfg --output ./output --paths.train train.spacy --paths.dev dev.spacy` |
| Convert data | Converts JSON/CoNLL formats to spaCy binary format. | `python -m spacy convert train.json ./corpus --converter json` |
| Evaluate | Evaluate trained pipeline. | `python -m spacy evaluate ./output/model-best dev.spacy` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| Fast text preprocessing | Batch process texts and keep lemmas. | `docs = nlp.pipe(texts, disable=["ner", "parser"])`<br/>`tokens = [[t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha] for doc in docs]` |
| Extract organizations | Use NER entity labels. | `orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]` |
| Rule-based product tags | PhraseMatcher for controlled vocabulary. | `terms = ["refund", "password reset", "billing"]`<br/>`patterns = [nlp.make_doc(term) for term in terms]`<br/>`matcher.add("TOPIC", patterns)` |
| Combine rules and ML | Entity ruler improves domain entities. | `ruler = nlp.add_pipe("entity_ruler", before="ner")`<br/>`ruler.add_patterns([{"label": "PRODUCT", "pattern": "Pro Plan"}])` |
| Process large file | Stream texts instead of loading all docs. | `with open("tickets.txt", encoding="utf-8") as file:`<br/>`    for doc in nlp.pipe(file, batch_size=128):`<br/>`        handle(doc)` |
| Save pipeline | Persist custom pipeline. | `nlp.to_disk("./model")`<br/>`nlp2 = spacy.load("./model")` |
| Similarity | Requires vectors for meaningful similarity. | `nlp = spacy.load("en_core_web_md")`<br/>`score = nlp("cat").similarity(nlp("dog"))` |
| Debug pipeline | Inspect component order and disabled pipes. | `print(nlp.pipe_names)`<br/>`print(nlp.disabled)` |

## Senior NLP pipeline design

| Method | Description | Code example |
|---|---|---|
| Component contract | Custom components should be deterministic and return the same `Doc`. | `@spacy.Language.component("ticket_router")`<br/>`def ticket_router(doc):`<br/>`    doc._.route = classify_route(doc)`<br/>`    return doc` |
| Batch and stream | Process large corpora without loading all texts into memory. | `def read_lines(path):`<br/>`    with open(path, encoding="utf-8") as file:`<br/>`        yield from file`<br/>`for doc in nlp.pipe(read_lines("corpus.txt"), batch_size=256):`<br/>`    write_features(doc)` |
| Disable expensive pipes | Use only the components needed for the task. | `with nlp.select_pipes(disable=["parser", "ner"]):`<br/>`    docs = list(nlp.pipe(texts))` |
| Entity overlap policy | Resolve rule-based and model entities consistently. | `from spacy.util import filter_spans`<br/>`doc.ents = filter_spans(list(doc.ents) + custom_ents)` |
| Label governance | Keep entity labels stable and documented. | `LABELS = {"PRODUCT": "Paid product name", "PLAN": "Billing plan"}` |
| Error analysis | Store false positives and false negatives by label. | `errors.append({"text": doc.text, "gold": gold_ents, "pred": [(e.text, e.label_) for e in doc.ents]})` |
| Data versioning | Record corpus version with model package. | `meta = nlp.meta`<br/>`meta["corpus_version"] = "tickets-v12"` |
| Pipeline packaging | Package trained model for deployment. | `python -m spacy package ./output/model-best ./packages --name ticket_ner --version 1.2.0` |

## Training and evaluation depth

| Method | Description | Code example |
|---|---|---|
| Train/dev/test split | Keep test set untouched until final evaluation. | `# train.spacy for optimization, dev.spacy for tuning, test.spacy for final report.` |
| Span-level metrics | Evaluate exact span and label quality, not just token accuracy. | `python -m spacy evaluate ./model test.spacy --output metrics.json` |
| Confusion by label | Find systematically confused entity types. | `for gold, pred in zip(gold_labels, pred_labels):`<br/>`    matrix[gold][pred] += 1` |
| Weak supervision | Use rules to bootstrap labels, then manually review. | `ruler.add_patterns(seed_patterns)`<br/>`docs = list(nlp.pipe(unlabeled_texts))` |
| Active learning | Prioritize uncertain examples for annotation. | `uncertain = sorted(examples, key=lambda ex: ex.score)[:100]` |
| Regression suite | Protect domain rules during model updates. | `assert extract_entities("Upgrade Pro Plan") == [("Pro Plan", "PRODUCT")]` |
| Throughput benchmark | Measure docs/sec before deployment. | `start = time.perf_counter()`<br/>`docs = list(nlp.pipe(texts, batch_size=512))`<br/>`print(len(docs) / (time.perf_counter() - start))` |
| Memory controls | Avoid keeping full docs when only features are needed. | `for doc in nlp.pipe(texts):`<br/>`    yield [(t.lemma_, t.pos_) for t in doc]` |
