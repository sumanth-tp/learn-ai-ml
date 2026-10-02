---
id: llme-embedding-tuning
title: "Tuning Embedding Models and Rerankers"
sidebar_label: "7 · Tuning embeddings"
sidebar_position: 7
slug: /llm-engineering/tuning-embedding-models-and-rerankers
description: "Fine-tune a bi-encoder with contrastive loss and in-batch negatives, mine hard negatives, measure recall and nDCG before and after, and see when a cross-encoder reranker helps and when it undoes your tuning."
tags: [embeddings, sentence-transformers, contrastive-loss, infonce, hard-negatives, reranker, cross-encoder, retrieval]
---

import Infographic from '@site/src/components/Infographic';
import ContrastiveLossLab from '@site/src/components/viz/ContrastiveLossLab';

**In one line.** An embedding model finds the right document if its training taught it your vocabulary, so tuning means showing it pairs of queries and documents from your domain and punishing it whenever a wrong document in the batch scores higher.

:::note Not from a lecture
This chapter is written for this site from the sources under Further reading. The data is a **toy**: an invented set of 128 internal-documentation pages with queries written from templates, built so that a general model has something concrete to learn. The numbers show the mechanism on a small CPU run, not what you should expect on your own corpus.
:::

## The idea in plain words

A general embedding model has read a vast amount of text and turns sentences into vectors where similar meaning means a high cosine. It does not know that, in your company, "the piggy" is the expense system and the documentation never uses that word. Ask it to find the page for "I cannot log in to the piggy" and it has nothing to go on.

Fine-tuning fixes this with examples. Give it a few hundred (query, right document) pairs. For each pair the training loss treats the other documents in the same batch as wrong answers and asks the right one to win a softmax over them. After a few passes the vectors for "piggy" queries sit near the expense pages.

Retrieval is usually two stages. A fast **bi-encoder** embeds the query and every document separately, so documents are embedded once and compared by cosine. A slower **cross-encoder** reads the query and one candidate together and scores the pair, so it is used only to reorder the few candidates the first stage returned.

<Infographic src="/img/llme/tuning-embedding-models-and-rerankers-contrastive.svg" alt="An 8 by 8 cosine matrix for eight queries and eight documents from the base model with the right pairs outlined, two tables of loss against temperature and negatives, and two cards on the library check and fine-tuning." caption="One batch seen through the loss. The matrix, the loss values and the library check are printed by block 1; the tuned values by block 2." />

<Infographic src="/img/llme/tuning-embedding-models-and-rerankers-results.svg" alt="A table of recall before and after tuning on seen and unseen documents, a bi-encoder against cross-encoder panel, and a table of two-stage retrieval with an off-the-shelf cross-encoder." caption="What tuning bought and what it did not. Every figure is printed by block 2." />

## How it works

### Bi-encoder, cross-encoder and why there are two

Sentence-BERT's abstract states the cost: finding the most similar pair among 10,000 sentences takes about 50 million inference computations, roughly 65 hours, with a BERT cross-encoder, against about 5 seconds with sentence embeddings. A cross-encoder cannot be precomputed because its score depends on the pair, so it cannot search a corpus. A bi-encoder compresses each text to one vector independently, which lets you store the vectors and search them quickly. The price is accuracy: query and document never see each other. The usual design keeps both: retrieve a few dozen candidates with the bi-encoder, then reorder them with the cross-encoder. See [neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking) for the pipeline and [the RAG project](/docs/projects/enterprise-rag/improvements) for how it is used.

### The contrastive loss

For a batch of $B$ pairs, let $s_{ij}$ be the cosine between query $i$ and document $j$, and $\tau$ the temperature. The loss for the batch is

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(s_{ii} / \tau)}{\sum_{j=1}^{B} \exp(s_{ij} / \tau)}$$

That is a cross-entropy over the row of scores, with the diagonal as the correct answer: **in-batch negatives**, every other document in the batch is a wrong answer for free. The form is called InfoNCE in the Contrastive Predictive Coding paper, which names Noise-Contrastive Estimation as its basis. Dense Passage Retrieval used exactly this trick: with $B$ questions in a batch the similarity matrix covers $B^2$ question and passage pairs, giving each question $B - 1$ negatives.

Two knobs matter. **Temperature** controls how sharply the softmax separates: sentence-transformers' `MultipleNegativesRankingLoss` multiplies the cosine by a `scale` of 20.0 by default, a temperature of 0.05. A smaller temperature punishes a close wrong document harder. **Negatives per query** set the difficulty: more of them make the right answer harder to win. One trap: if the same document appears twice in a batch, one copy is treated as a wrong answer for the other. The library offers a `NO_DUPLICATES` batch sampler for this reason.

### Hard negatives

Random in-batch negatives are mostly easy. A **hard negative** is a wrong document that scores high: a plausible-looking page that does not answer the query. Dense Passage Retrieval's best model used in-batch negatives with one extra BM25-mined negative per question, and reported that adding it helped. The usual recipe is to rank the corpus with the current model and take the highest-ranked wrong document. Two cautions: mined negatives can be unlabelled right answers (a *false negative*), and mining is only as informative as the model doing it. Block 2 shows the second one.

### Building pairs and measuring

Real pairs come from three places: logged queries with the document the user clicked, queries written by people for a sample of documents, and queries generated by a language model for each document (see [synthetic data generation](/docs/llm-engineering/synthetic-data-generation) for the filters). Split by **document** for at least one test, otherwise the model can pass by memorising your corpus.

Measure with **recall@k** (the share of queries whose right document is in the top $k$) and **nDCG@10**. With one right document per query the gain of a result at zero-based rank $r$ is $1/\log_2(r + 2)$ if $r < 10$ and zero otherwise. For the metrics in depth see [evaluating ranked retrieval](/docs/theory/ir/evaluating-ranked-retrieval).

## A real system that works this way

**`all-MiniLM-L6-v2`** is the model tuned in this chapter. Its model card describes a base of `nreimers/MiniLM-L6-H384-uncased`, 384-dimensional embeddings and a contrastive objective, fine-tuned on over a billion sentence pairs (1,170,060,424 in the card's table), under the Apache 2.0 licence. **`cross-encoder/ms-marco-MiniLM-L6-v2`**, the reranker used below, is trained on the MS MARCO passage ranking task; its card quotes NDCG@10 of 74.30 on TREC DL 19, MRR@10 of 39.01 on the MS MARCO dev set and a throughput of 1,800 documents per second on a V100 GPU, and says it is intended for reranking candidates from a retriever. Both are general-purpose models trained on web-style data, which is exactly why the toy below has a gap to close.

## Code you can run

CPU only, seeded. Environment: Python 3.14, sentence-transformers 6.1.0, torch 2.14.1, datasets 5.0.1. The block downloads `all-MiniLM-L6-v2` and the cross-encoder from the Hugging Face Hub on first run. Note that sentence-transformers 6.1.0 moved its classes under `sentence_transformers.sentence_transformer`, and the older import paths print a deprecation warning.

### 1. The loss from scratch, and temperature and negatives

Eight invented internal tools, one query each, one document each. The queries use the staff nickname ("the compass") and the documents use the tool's name, so the base model has no way to match them.

```python
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

TOOLS = [("Wenvon", "the meeting room system", "compass"), ("Gilkora", "the on-call rota system", "mailbox"),
         ("Fentirx", "the payroll query system", "harbour"), ("Imbrin", "the security incident system", "pebble"),
         ("Salvo", "the bug triage system", "lighthouse"), ("Moreplix", "the recruiting pipeline system", "meadow"),
         ("Plibrax", "the cloud budget system", "hammock"), ("Gilquan", "the leave booking portal", "sunbed")]
TAIL = ("This page covers how to get an account and sign in. Accounts are created from the staff directory after a manager "
        "confirms the role. Sign in with the company login; if the account is missing, ask the service desk to create it.")
queries = [f"I cannot log in to the {nick}" for _, _, nick in TOOLS]
docs = [f"{tool} is {what}. {TAIL}" for tool, what, _ in TOOLS]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").eval()
with torch.no_grad():
    q = F.normalize(model.encode(queries, convert_to_tensor=True), dim=-1)
    d = F.normalize(model.encode(docs, convert_to_tensor=True), dim=-1)
S = q @ d.T
print("cosine similarity, queries (rows) against documents (columns):")
print(np.array2string(S.numpy(), precision=2, suppress_small=True))
print(f"mean cosine to the right document {S.diag().mean():.3f}, to the wrong ones {(S.sum() - S.diag().sum()).item() / 56:.3f}")

def info_nce(sim, scale, n=None):
    sim = sim if n is None else sim[:n, :n]
    return F.cross_entropy(sim * scale, torch.arange(len(sim))).item()

library = MultipleNegativesRankingLoss(model)
with torch.no_grad():
    value = library([model.preprocess(queries), model.preprocess(docs)], None).item()
print(f"\nlibrary default scale {library.scale}; library loss {value:.6f}; from scratch {info_nce(S, library.scale):.6f}")

print("\ntemperature (scale = 1 / temperature), all 8 queries")
for scale in (100, 20, 10, 2):
    print(f"temperature {1 / scale:.2f}   scale {scale:3d}   loss {info_nce(S, scale):.4f}")
print("\nnumber of in-batch negatives per query, temperature 0.05")
for n in (2, 4, 8):
    print(f"{n - 1} negatives   loss {info_nce(S, 20, n):.4f}")
p = F.softmax(S * 20, dim=1)
print(f"\nprobability the loss gives the right document at temperature 0.05: mean {p.diag().mean():.3f}, lowest {p.diag().min():.3f}")
```

The base model's cosine to the right document averages 0.197 and to the wrong ones 0.196: indistinguishable. The from-scratch loss equals the library's to six decimals (2.586478), so the loss is nothing more than that softmax cross-entropy. Then the two knobs. Dropping the temperature from 0.05 to 0.01 raises the loss from 2.5865 to 8.2145, because the softmax becomes razor sharp and a wrong document with a cosine just above the right one is punished heavily; raising it to 0.50 flattens everything to 2.0839, close to the 2.0794 of random guessing among 8. Reducing the batch to two queries cuts the loss to 0.6904 and to four gives 1.9741, because there are fewer wrong documents to beat.

The lab replays this batch for the base and the two tuned models. Its defaults (base model, temperature 0.05, 7 negatives) give the 2.5865 printed above; switch the model to see 1.0621 and 1.3016, printed by block 2.

<ContrastiveLossLab />

### 2. Tuning, hard negatives and a reranker

The domain: 32 invented tools, 4 documentation pages each (access, limits, approval, errors), 128 documents. Each page has six queries. Four use the staff nickname and are used for training; two are held out (one nickname, one using the tool name). Six tools are never trained on, as a test that the model has not just memorised the corpus. Three models are compared: the base model, one fine-tuned on 416 (query, document) pairs with in-batch negatives, and one fine-tuned on the same pairs plus one mined hard negative each. Then a pretrained cross-encoder reranks the top 10 of the base and tuned retrievers.

```python
import random
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.sentence_transformer import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

SUBJECTS = ["expense claim", "leave booking", "hardware request", "customer ticket", "meeting room", "software release",
            "staff directory", "vendor invoice", "on-call rota", "training course", "file transfer", "travel booking",
            "payroll query", "onboarding checklist", "building badge", "parking permit", "security incident",
            "contract review", "purchase order", "performance review", "bug triage", "data access", "license renewal",
            "event planning", "recruiting pipeline", "office supplies", "visitor registration", "policy library",
            "cloud budget", "dashboard sharing", "mail alias", "device repair"]
NICKS = ["piggy", "sunbed", "lantern", "ferry", "compass", "beehive", "kettle", "anchor", "mailbox", "teapot", "bridge",
         "windmill", "harbour", "orchard", "locker", "scooter", "pebble", "ladder", "umbrella", "marble", "lighthouse",
         "tractor", "puzzle", "canoe", "meadow", "trolley", "gadget", "boulder", "hammock", "cupboard", "pigeon", "rocket"]
KINDS = ["system", "portal", "tool", "desk"]
SYLLABLES = ["zor", "vex", "pli", "met", "quan", "dro", "bri", "xel", "ta", "vo", "lo", "mir", "ost", "rav", "dun", "more",
             "fen", "nick", "hal", "vard", "im", "bra", "jas", "pel", "kor", "wen", "sal", "tir", "ube", "ran", "gil", "nox"]

def make_tools():
    rng = random.Random(3)
    names, tools = set(), {}
    for i, subject in enumerate(SUBJECTS):
        while True:
            name = (rng.choice(SYLLABLES) + rng.choice(SYLLABLES) + rng.choice(["", "n", "x", "a"])).capitalize()
            if name not in names:
                names.add(name)
                break
        tools[name] = (f"the {subject} {KINDS[i % 4]}", NICKS[i])
    return tools

TOOLS = make_tools()
ASPECTS = {
    "access": ("how to get an account and sign in", [
        "I cannot log in to {t}", "how do I get access to {t}", "{t} says my account does not exist",
        "request an account for {t}", "locked out of {t}", "who grants permission to use {t}"]),
    "limits": ("the limits and quotas that apply", [
        "what is the maximum I can submit in {t}", "{t} refuses my request because it is too large",
        "is there a cap on {t}", "why does {t} reject big submissions", "size limit in {t}", "{t} quota exceeded"]),
    "approval": ("who approves requests and how long it takes", [
        "my request in {t} is stuck waiting", "who signs off in {t}", "how long does approval take in {t}",
        "{t} has not been approved for days", "escalate a pending item in {t}", "approver missing in {t}"]),
    "errors": ("what common error messages mean and how to fix them", [
        "{t} shows an error when I save", "red banner in {t} after submit", "{t} throws a timeout",
        "what does the failure message in {t} mean", "{t} will not let me finish", "fix a broken submission in {t}"]),
}
BODY = {
    "access": "Accounts are created from the staff directory after a manager confirms the role. Sign in with the company login; if the account is missing, ask the service desk to create it.",
    "limits": "Each submission is capped. Anything above the cap is rejected until it is split into smaller parts. The cap is reviewed every quarter.",
    "approval": "Every request goes to the line manager first and then to the owning team. Items older than three working days can be escalated to the team lead.",
    "errors": "Most failures come from expired sessions or missing mandatory fields. Sign in again, complete the highlighted fields and resubmit; persistent failures go to the service desk.",
}

def build():
    docs, queries = [], []
    for tool, (what, nick) in TOOLS.items():
        for aspect, (summary, templates) in ASPECTS.items():
            doc_id = len(docs)
            docs.append({"id": doc_id, "tool": tool, "aspect": aspect,
                         "text": f"{tool} is {what}. This page covers {summary}. {BODY[aspect]}"})
            for k, tpl in enumerate(templates):
                queries.append({"doc": doc_id, "variant": k, "tool": tool, "text": tpl.format(t=tool if k == 5 else "the " + nick)})
    return docs, queries

BASE = "sentence-transformers/all-MiniLM-L6-v2"
docs, queries = build()
doc_text = [d["text"] for d in docs]
tools = sorted({d["tool"] for d in docs})
unseen_tools = set(tools[::6][:6])
train_q = [q for q in queries if q["tool"] not in unseen_tools and q["variant"] < 4]
test_seen = [q for q in queries if q["tool"] not in unseen_tools and q["variant"] >= 4]
test_unseen = [q for q in queries if q["tool"] in unseen_tools]
print(f"{len(docs)} documents, {len(train_q)} training pairs; test: {len(test_seen)} new phrasings of seen documents, "
      f"{len(test_unseen)} queries on {len(unseen_tools)} tools never trained on")

def load():
    return SentenceTransformer(BASE, device="cpu")

def scores(model, qs):
    D = model.encode(doc_text, normalize_embeddings=True)
    Q = model.encode([q["text"] for q in qs], normalize_embeddings=True)
    return Q @ D.T

def ranks(S, qs):
    gold = np.array([q["doc"] for q in qs])
    return (S > S[np.arange(len(qs)), gold][:, None]).sum(1)

def summary(rank):
    ndcg = np.where(rank < 10, 1 / np.log2(rank + 2), 0.0).mean()
    return f"recall@1 {np.mean(rank < 1):.3f}  recall@5 {np.mean(rank < 5):.3f}  recall@10 {np.mean(rank < 10):.3f}  nDCG@10 {ndcg:.3f}"

def report(label, model):
    for split, qs in (("seen docs", test_seen), ("unseen docs", test_unseen)):
        print(f"{label:22s} {split:12s} {summary(ranks(scores(model, qs), qs))}")

def finetune(train):
    model = load()
    args = SentenceTransformerTrainingArguments(
        output_dir="/tmp/e2-embedding", num_train_epochs=3, per_device_train_batch_size=32, learning_rate=2e-5,
        warmup_steps=5, batch_sampler=BatchSamplers.NO_DUPLICATES, seed=0, report_to="none", save_strategy="no",
        logging_steps=1000, use_cpu=True, disable_tqdm=True)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=train, loss=MultipleNegativesRankingLoss(model)).train()
    return model

batch_tools = ["Wenvon", "Gilkora", "Fentirx", "Imbrin", "Salvo", "Moreplix", "Plibrax", "Gilquan"]
batch = [(next(q for q in queries if q["doc"] == d["id"] and q["variant"] == 0), d)
         for t in batch_tools for d in docs if d["tool"] == t and d["aspect"] == "access"]

def batch_matrix(model):
    q = model.encode([b[0]["text"] for b in batch], normalize_embeddings=True)
    d = model.encode([b[1]["text"] for b in batch], normalize_embeddings=True)
    return torch.tensor(q @ d.T)

def batch_loss(model, scale=20):
    return F.cross_entropy(batch_matrix(model) * scale, torch.arange(len(batch))).item()

base = load()
report("base model", base)

pairs = Dataset.from_dict({"anchor": [q["text"] for q in train_q], "positive": [doc_text[q["doc"]] for q in train_q]})
tuned = finetune(pairs)
report("in-batch negatives", tuned)
print(f"loss on the 8-pair batch of block 1 at temperature 0.05: base {batch_loss(base):.4f}, tuned {batch_loss(tuned):.4f}")

S_train = scores(base, train_q)
hard = []
for row, q in zip(S_train, train_q):
    row = row.copy()
    row[q["doc"]] = -1
    hard.append(doc_text[int(row.argmax())])
print(f"\nmined {len(hard)} hard negatives with the base model; "
      f"{np.mean([docs[int(np.argmax(r))]['tool'] == q['tool'] for r, q in zip(S_train, train_q)]):.3f} share of the time the top wrong document is another aspect of the same tool")
triplets = Dataset.from_dict({"anchor": pairs["anchor"], "positive": pairs["positive"], "negative": hard})
tuned_hard = finetune(triplets)
report("plus hard negatives", tuned_hard)

print("\ntwo-stage retrieval: pretrained cross-encoder reranks the bi-encoder's top 10")
cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
for label, model in (("base bi-encoder", base), ("tuned bi-encoder", tuned)):
    for split, qs in (("seen docs", test_seen), ("unseen docs", test_unseen)):
        S = scores(model, qs)
        before = ranks(S, qs)
        top = np.argsort(-S, axis=1)[:, :10]
        reranked = []
        for qi, q in enumerate(qs):
            ce = cross.predict([(q["text"], doc_text[j]) for j in top[qi]], show_progress_bar=False)
            order = top[qi][np.argsort(-ce)]
            reranked.append(list(order).index(q["doc"]) if q["doc"] in order else 10)
        reranked = np.array(reranked)
        print(f"{label:17s} {split:12s} top 10 holds the answer {np.mean(before < 10):.3f}; recall@1 {np.mean(before < 1):.3f} -> after rerank {np.mean(reranked < 1):.3f}")
print(f"cost per query: bi-encoder 1 forward pass (documents encoded once), cross-encoder {10} passes over query-document pairs")
```

**Tuning worked where it could.** On new phrasings of documents it trained on, recall@1 rises from 0.404 to 0.635 and nDCG@10 from 0.501 to 0.771, and recall@10 from 0.615 to 0.923. The loss on the block 1 batch falls from 2.5865 to 1.0621. **It did not teach what it was not shown.** On the six tools never trained on, recall@1 moves only from 0.132 to 0.174, because their nicknames never appeared in training. A tuned embedding model learns your corpus's vocabulary, not a general ability to read nicknames.

**Hard negatives did not help here.** Seen-document recall@1 is 0.606 against 0.635 for in-batch negatives alone, and unseen recall@10 drops from 0.333 to 0.222. The reason is visible in the printed share: only 6.0% of the base model's top-ranked wrong documents were another page of the same tool. The base model cannot read the nicknames, so what it ranks highest among wrong documents is close to arbitrary and teaches little. Hard-negative mining needs a first-stage model whose errors are informative. Dense Passage Retrieval found a BM25 negative helpful; this toy shows it is not automatic.

**The off-the-shelf reranker hurt.** For the tuned retriever the right answer is in the top 10 for 92.3% of seen-document queries, yet reordering them with the cross-encoder cuts recall@1 from 0.635 to 0.394. For the base retriever the reranker also lowers recall@1 (0.404 to 0.385). The cross-encoder never saw the nicknames either, so it overrides what the tuned stage learned. A reranker can only choose among what stage one returned (when the top 10 holds the answer 29.2% of the time, as for the base model on unseen tools, nothing can fix the rest), and it reorders by its own training. If you rerank, evaluate with and without it, and train or fuse the reranker on domain pairs too.

## Designing with it

1. **Fix retrieval before generation.** If the right passage is not in the top $k$, no prompt or reranker repairs it; measure recall@k first.
2. **Tune the bi-encoder when the gap is vocabulary.** Domain jargon, product names and nicknames are the classic case, as in the toy.
3. **Choose negatives deliberately.** Start with in-batch negatives and a large enough batch; add mined hard negatives only after checking that the mining model's errors are meaningful, and review a sample for false negatives.
4. **Split by document** for one evaluation, and keep real user queries for the final test; templated or LLM-written queries are narrower than real ones.
5. **Rerank last, and test it.** A reranker adds latency per candidate. Keep it only if recall@1 or nDCG improves on your data.
6. **Re-embed after tuning.** A tuned model produces new vectors, so the whole index must be rebuilt, and the old and new vectors cannot be mixed.

**Failure modes to name**

- *Memorising the corpus:* high recall on seen documents and no lift on unseen ones.
- *False negatives:* duplicates or near-duplicates in a batch treated as wrong answers.
- *Reranker disagreement:* a general reranker undoing a domain-tuned first stage.
- *Stale index:* queries embedded with the new model against documents embedded with the old.

## Where this stands in 2026

:::info Industry view

- **Retrieve then rerank is the standard shape,** and the two stages are tuned and evaluated separately. Measure each stage on its own metric: recall@k for the first, nDCG or recall@1 for the second.
- **sentence-transformers 6.1.0 is the version used here.** Classes live under `sentence_transformers.sentence_transformer` and the older paths still work with a deprecation warning. `MultipleNegativesRankingLoss` now accepts `hardness_mode` and `hardness_strength` options alongside `scale`, so hard-negative weighting can be built into the loss.
- **Check the card before you tune.** The two model cards used here state their training data and the Apache 2.0 licence; confirm both for any embedding model you fine-tune.
- **Synthetic queries are one way to get pairs.** Generating queries per document with a language model and filtering them helps when logs are thin; the filters in the previous chapter apply.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What does in-batch negatives mean, and what goes wrong if the same document appears twice in a batch?</summary>

Every other document in the batch counts as a wrong answer for a given query, so a batch of $B$ pairs gives each query $B - 1$ negatives at no extra encoding cost. If a document appears twice, the second copy is treated as a wrong answer for the first query although it is the right one, which pushes the model away from correct matches. A no-duplicates batch sampler avoids it.<br /><em>Authored · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> In block 1, lowering the temperature from 0.05 to 0.01 raises the loss from 2.5865 to 8.2145. Why?</summary>

Dividing cosines by a smaller temperature makes the softmax much sharper, so the largest cosine in each row takes almost all the probability. For the base model that largest cosine is often a wrong document, so the probability on the right one collapses and the cross-entropy rises. The same sharpness is useful once the model separates right from wrong, because it then punishes the remaining close wrong documents hard.<br /><em>Authored · interpretation</em>

</details>

<details>
<summary><strong>Q3.</strong> After tuning, recall@1 on seen documents rose from 0.404 to 0.635 but on unseen tools only from 0.132 to 0.174. What does that tell you about what was learned?</summary>

The model learned the mapping from the nicknames it was trained on to their pages, which does not transfer to nicknames it never saw. A small tuned embedding model absorbs the vocabulary and structure of its training corpus. For new vocabulary you need new pairs, not a cleverer loss.<br /><em>Authored · interpretation</em>

</details>

<details>
<summary><strong>Q4.</strong> Hard negatives lowered recall in block 2. Name two possible reasons in general, and the one shown here.</summary>

In general: mined negatives can be unlabelled right answers (false negatives), and they can be too hard for the model's capacity early in training. Here: the base model could not read the nicknames, so its top-ranked wrong documents were close to arbitrary and carried no useful signal; only 6.0% were pages of the same tool.<br /><em>Authored · applied</em>

</details>

<details>
<summary><strong>Q5.</strong> The tuned retriever has the right document in its top 10 for 92.3% of queries, but adding a cross-encoder lowers recall@1. What do you do?</summary>

Check what the reranker was trained on: here it was trained on web passages and never saw the domain vocabulary. Options are to fine-tune the cross-encoder on domain pairs, to blend its score with the bi-encoder's rather than replace it, or to drop it. In every case compare recall@1 and nDCG with and without the reranker on held-out queries.<br /><em>Authored · applied</em>

</details>

## Further reading

- [Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering" (EMNLP 2020)](https://arxiv.org/abs/2004.04906): in-batch negatives, BM25 negatives and the comparison with BM25.
- [van den Oord et al., "Representation Learning with Contrastive Predictive Coding" (2018)](https://arxiv.org/abs/1807.03748): the InfoNCE loss.
- [Reimers and Gurevych, "Sentence-BERT" (EMNLP 2019)](https://arxiv.org/abs/1908.10084): bi-encoders against cross-encoders and the cost argument.
- [sentence-transformers: MultipleNegativesRankingLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) and [training overview](https://sbert.net/docs/sentence_transformer/training_overview.html): the trainer, dataset format and losses used here.
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [cross-encoder/ms-marco-MiniLM-L6-v2 model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).
- Related chapters on this site: [neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking), [evaluating ranked retrieval](/docs/theory/ir/evaluating-ranked-retrieval), [the enterprise RAG project](/docs/projects/enterprise-rag/improvements), [synthetic data generation](/docs/llm-engineering/synthetic-data-generation).

## Check yourself

- I can write the in-batch contrastive loss as a cross-entropy over a similarity matrix and say what the temperature and the number of negatives do.
- I can explain why a bi-encoder can search a corpus and a cross-encoder cannot, and why the two are combined.
- I can build query and document pairs, split them by document, and evaluate recall@k and nDCG before and after tuning.
- I can say what fine-tuning an embedding model can and cannot teach, and read the seen-against-unseen gap.
- I can explain when hard-negative mining helps and when it does not, and what a false negative is.
- I can decide whether a reranker is helping by comparing recall@1 with and without it.
