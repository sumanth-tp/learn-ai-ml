---
id: gov-regulation
title: "Regulation and Model Documentation"
sidebar_label: "Regulation and documentation"
sidebar_position: 4
slug: /governance/regulation-and-model-documentation
description: "The EU AI Act's risk tiers and application dates as of October 2026, read from the Official Journal texts including the Digital Omnibus amendments, with NIST AI RMF, ISO/IEC 42001, model cards, datasheets and tamper-evident audit logs."
tags: [eu-ai-act, regulation, nist-ai-rmf, iso-42001, model-cards, datasheets, audit-trail, compliance]
---

import Infographic from '@site/src/components/Infographic';
import RiskTierLab from '@site/src/components/viz/RiskTierLab';

**In one line.** The EU AI Act sorts AI systems into tiers by what they do, attaches obligations and dates to each tier, and expects you to prove compliance with documentation and logs, which NIST's framework, ISO/IEC 42001 and model cards help you produce.

:::warning This is not legal advice
This chapter is engineering education. It reports what the published texts say as of 2 October 2026, and a teaching triage that sorts example systems into tiers. Classifying a real system is a legal judgement that depends on facts, on national implementation and on guidance that is still being written. Ask counsel before relying on any of it.
:::

:::note Not from a lecture
Written for this site from the sources under Further reading. The Act's dates and obligations below were read from the Official Journal texts of Regulation (EU) 2024/1689 and Regulation (EU) 2026/1744, retrieved from the EU Publications Office in October 2026.
:::

## The idea in plain words

The AI Act regulates by risk. A spam filter and a hiring tool are both "AI", but the Act does not treat them alike. Four ideas carry most of the structure.

1. **Prohibited practices** (Article 5) are banned outright: for example social scoring, inferring emotions at work or in education (with medical and safety exceptions), and untargeted scraping of faces to build recognition databases.
2. **High-risk systems** (Article 6) are allowed but heavily regulated. They are either safety components of products covered by EU product legislation (Annex I), or systems in the eight areas of Annex III: biometrics, critical infrastructure, education, employment, essential services (including credit scoring and life and health insurance pricing), law enforcement, migration and border control, and justice and democratic processes.
3. **Transparency duties** (Article 50) apply to systems that talk to people, generate synthetic content, recognise emotions or categorise by biometrics.
4. **General-purpose AI models** (Chapter V) carry their own obligations, with extra duties for models presumed to have systemic risk once training compute exceeds 10^25 floating point operations.

Everything else is minimal-risk: no specific obligation beyond AI literacy for providers and deployers.

Three voluntary frameworks sit beside the law and give you a way to organise the work: the **NIST AI Risk Management Framework**, **ISO/IEC 42001** and the documentation practices of **model cards** and **datasheets**.

<Infographic src="/img/gov/regulation-and-model-documentation-timeline.svg" alt="A timeline of EU AI Act application dates from 1 August 2024 to 2 August 2028, with fines and later dates summarised below." caption="The application dates in the text as amended, with Article 99 fines. Every date was read from the Official Journal texts." />

## How it works

### The dates, as amended

The Act entered into force on 1 August 2024 (Official Journal of 12 July 2024). Its general date of application is 2 August 2026, with earlier and later exceptions. In July 2026 the **Digital Omnibus on AI**, Regulation (EU) 2026/1744 of 8 July 2026, was published in the Official Journal on 24 July 2026 and, under its Article 4, entered into force on the third day after publication, 27 July 2026. It amends the Act, and the text shows these dates for the provisions that matter most to engineers.

| Date | What applies | Source in the text |
| --- | --- | --- |
| 2 February 2025 | Chapters I and II: general provisions including AI literacy, and the prohibited practices of Article 5 | Art 113(a) |
| 2 August 2025 | Chapter III Section 4 (notified bodies), Chapter V (general-purpose models), Chapter VII (governance), Chapter XII (penalties) and Article 78, except Article 101 | Art 113(b) |
| 27 July 2026 | the Omnibus enters into force; Articles 102 to 110 apply from this date | Omnibus Art 4; Art 113(d) as inserted |
| 2 August 2026 | the Act's general date of application: Article 50 transparency duties, and Article 101 (fines for general-purpose model providers) | Art 113, first paragraph |
| 2 December 2026 | new prohibitions on certain sexual deepfake and child-abuse-material generators, Article 5(1)(ba), (bb), (1a), (1b); and Article 50(2) marking for generative systems placed on the market before 2 August 2026 | Art 113(a) as amended; Art 111(4) as inserted |
| 2 December 2027 | Chapter III Sections 1, 2 and 3 (high-risk requirements) for systems classified under Article 6(2) and **Annex III** | Art 113(c)(i) as replaced |
| 2 August 2028 | the same obligations for systems classified under Article 6(1) and **Annex I** (products) | Art 113(c)(ii) as replaced |

Three more dates from the text are worth knowing. General-purpose models placed on the market before 2 August 2025 must comply by 2 August 2027 (Article 111(3)). High-risk systems placed on the market before the relevant date are caught only if they later undergo significant design changes, but providers and deployers of high-risk systems intended for use by public authorities must comply by 2 August 2030 (Article 111(2), as replaced). And Article 6(5), the Commission's duty to publish classification guidelines with examples, is excluded from the delay in Article 113(c); the original text set 2 February 2026 for it, and I did not check whether that guidance has been published.

The **Omnibus did not delay everything.** It moved the Annex III high-risk date from 2 August 2026 to 2 December 2027 and the Annex I date to 2 August 2028; the prohibitions, the general-purpose model duties, the penalties chapter and the Article 50 transparency duties keep their dates, and it added new prohibitions from 2 December 2026. The Omnibus's own recitals give the reason for the delay: the delayed availability of standards, common specifications and guidance, and of national competent authorities.

### What the tiers require

**High-risk providers** must meet the requirements of Chapter III Section 2: a risk management system (Article 9), data and data governance (Article 10), technical documentation drawn up before placing on the market and kept up to date (Article 11, with its contents in Annex IV), automatic event logging over the system's lifetime (Article 12), transparency to deployers (Article 13), human oversight (Article 14), and accuracy, robustness and cybersecurity (Article 15). Around those sit the provider obligations of Article 16, a quality management system (Article 17), conformity assessment (Article 43), registration for Annex III systems (Article 49) and post-market monitoring (Article 72).

**Deployers** of high-risk systems must use them according to the instructions, assign competent human oversight, and keep the logs the system generates for a period appropriate to the purpose and at least six months unless other law says otherwise (Article 26). Certain deployers, namely public bodies, private entities providing public services and deployers of the creditworthiness and life and health insurance systems in Annex III point 5(b) and (c), must carry out a fundamental rights impact assessment first (Article 27).

**Serious incidents** involving high-risk systems must be reported by the provider: immediately once a causal link is established and in any event within 15 days of becoming aware, within 10 days for a death, and within two days for a widespread infringement or a serious and irreversible disruption of critical infrastructure (Article 73). Chapter 5 returns to this.

**Transparency** (Article 50): people must be told they are interacting with an AI system unless that is obvious; providers of generative systems must mark outputs in a machine-readable format; deployers must disclose deep fakes, and AI-generated text published to inform the public on matters of public interest unless it has had human editorial review.

**General-purpose models** (Articles 53 and 55): providers keep technical documentation, give downstream providers the information they need, adopt a copyright policy and publish a summary of training content. Providers of models with systemic risk, presumed above 10^25 floating point operations of training compute, must also evaluate the model including adversarial testing, assess and mitigate systemic risks, report serious incidents and secure the model (see [red-teaming](/docs/governance/red-teaming-llm-systems)).

**Penalties** (Article 99, as amended): up to EUR 35 million or 7% of worldwide annual turnover, whichever is higher, for prohibited practices; up to EUR 15 million or 3% for most other operator obligations including Article 50; up to EUR 7.5 million or 1% for supplying incorrect, incomplete or misleading information. For SMEs each fine is capped at the lower of the two figures, and the Omnibus extends that rule to small mid-cap enterprises. Fines for general-purpose model providers under Article 101 are up to 3% or EUR 15 million.

### The voluntary frameworks

| Framework | What it is | Verified facts |
| --- | --- | --- |
| NIST AI RMF 1.0 (NIST AI 100-1) | a voluntary framework with four functions, GOVERN, MAP, MEASURE and MANAGE; GOVERN is cross-cutting | released January 2023; characteristics of trustworthy AI: valid and reliable, safe, secure and resilient, accountable and transparent, explainable and interpretable, privacy-enhanced, fair with harmful bias managed; NIST's page says it is being revised under the White House AI Action Plan, and records an April 2026 concept note for a critical-infrastructure profile |
| NIST AI 600-1, Generative AI Profile | a companion profile naming twelve risks unique to or exacerbated by generative AI | July 2024; the risks are CBRN information or capabilities, confabulation, dangerous, violent or hateful content, data privacy, environmental impacts, harmful bias and homogenization, human-AI configuration, information integrity, information security, intellectual property, obscene, degrading or abusive content, and value chain and component integration |
| ISO/IEC 42001:2023 | requirements for an AI management system, from ISO/IEC JTC 1/SC 42 | published 18 December 2023, 51 pages, edition 1; I could not read the paywalled text, so I make no claim about its clauses or controls |

### Documentation that carries the evidence

A **model card** (Mitchell and colleagues, FAT* 2019) is a short document that accompanies a trained model. The paper's template has nine sections: model details, intended use, factors, metrics, evaluation data, training data, quantitative analyses, ethical considerations, and caveats and recommendations. Its distinctive demand is that performance be reported **disaggregated** by the relevant groups and conditions. A **datasheet for datasets** (Gebru and colleagues, 2018, published in Communications of the ACM in December 2021) does the same for data: motivation, composition, collection process and recommended uses. A **system card** extends the idea to a deployed application with several components; the Act's technical documentation (Annex IV) even asks for datasheets where relevant. An **audit trail** is the log that lets you reconstruct who or what decided, with which model and input.

## A real system that works this way

The Act is the case: Regulation (EU) 2024/1689 is in force, and the Omnibus recitals show how the timetable responded to missing standards and national authorities. The model-card paper demonstrates its template on a smile detector and a toxic-comment classifier; I did not review any vendor's published cards.

## Code you can run

Everything is CPU only, seeded and deterministic. The triage reference date is fixed at 2 October 2026 so the output does not change.

#### 1. A model-card generator with a completeness check

The card is built from a real evaluation: a logistic regression on scikit-learn's bundled breast cancer dataset, evaluated overall and **by tumour-size slice**, with an interval on each accuracy. The generator refuses a card with an empty section.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X, y = data.data, data.target
area = X[:, list(data.feature_names).index("mean area")]
bands = np.digitize(area, np.quantile(area, [1 / 3, 2 / 3]))
names = ["smaller tumours", "middle tumours", "larger tumours"]
Xtr, Xte, ytr, yte, btr, bte = train_test_split(X, y, bands, test_size=0.3, random_state=0, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
pred = model.predict(Xte)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


rows = []
for label, mask in [("all test rows", np.ones(len(yte), bool))] + [(names[i], bte == i) for i in range(3)]:
    correct = int((pred[mask] == yte[mask]).sum())
    low, high = wilson(correct, int(mask.sum()))
    rows.append((label, int(mask.sum()), accuracy_score(yte[mask], pred[mask]), recall_score(yte[mask], pred[mask], zero_division=0),
                 precision_score(yte[mask], pred[mask], zero_division=0), low, high))

CARD = {
    "Model details": "Logistic regression with standard scaling, scikit-learn, trained on the 569-row breast cancer diagnostic dataset bundled with scikit-learn. Version 1, teaching example.",
    "Intended use": "Demonstrating model documentation. Not a medical device and not for clinical decisions.",
    "Factors": "Tumour size band, from terciles of the mean area feature.",
    "Metrics": "Accuracy, recall and precision for the benign class, with a 95% Wilson interval on accuracy.",
    "Evaluation data": f"{len(yte)} held-out rows, stratified 70/30 split with random_state 0.",
    "Training data": f"{len(ytr)} rows from the same dataset; no external data.",
    "Quantitative analyses": None,
    "Ethical considerations": "The data come from one source and carry no patient demographics, so no demographic fairness claim can be made.",
    "Caveats and recommendations": "Slices are small; intervals are wide. Re-evaluate on data from the deployment site before any use.",
}
table = ["| slice | n | accuracy | recall | precision | accuracy 95% interval |", "| --- | --- | --- | --- | --- | --- |"]
for label, n, acc, rec, prec, low, high in rows:
    table.append(f"| {label} | {n} | {acc:.3f} | {rec:.3f} | {prec:.3f} | {low:.3f} to {high:.3f} |")
CARD["Quantitative analyses"] = "\n".join(table)

REQUIRED = list(CARD)


def check(card):
    return [section for section in REQUIRED if not str(card.get(section) or "").strip()]


def render(card):
    return "# Model card\n\n" + "\n\n".join(f"## {k}\n{v}" for k, v in card.items())


print(render(CARD))
print("\nmissing sections:", check(CARD) or "none")
broken = dict(CARD, **{"Ethical considerations": ""})
print("with the ethics section blanked, the check reports:", check(broken))
```

```text
# Model card

## Model details
Logistic regression with standard scaling, scikit-learn, trained on the 569-row breast cancer diagnostic dataset bundled with scikit-learn. Version 1, teaching example.

## Intended use
Demonstrating model documentation. Not a medical device and not for clinical decisions.

## Factors
Tumour size band, from terciles of the mean area feature.

## Metrics
Accuracy, recall and precision for the benign class, with a 95% Wilson interval on accuracy.

## Evaluation data
171 held-out rows, stratified 70/30 split with random_state 0.

## Training data
398 rows from the same dataset; no external data.

## Quantitative analyses
| slice | n | accuracy | recall | precision | accuracy 95% interval |
| --- | --- | --- | --- | --- | --- |
| all test rows | 171 | 0.959 | 0.963 | 0.972 | 0.918 to 0.980 |
| smaller tumours | 50 | 1.000 | 1.000 | 1.000 | 0.929 to 1.000 |
| middle tumours | 64 | 0.922 | 0.961 | 0.942 | 0.830 to 0.966 |
| larger tumours | 57 | 0.965 | 0.750 | 1.000 | 0.881 to 0.990 |

## Ethical considerations
The data come from one source and carry no patient demographics, so no demographic fairness claim can be made.

## Caveats and recommendations
Slices are small; intervals are wide. Re-evaluate on data from the deployment site before any use.

missing sections: none
with the ethics section blanked, the check reports: ['Ethical considerations']
```

Three things to read from the output. The overall accuracy of 0.959 hides a slice at 0.922 and, in the larger-tumour slice, a benign-class recall of 0.750 (a handful of rows), which is the kind of disaggregation the model-card paper asks for. The intervals are wide (0.830 to 0.966 for the middle slice), so the "Caveats" section is doing real work rather than ceremony. And the check catches a blanked "Ethical considerations" section, which is how you stop a card from shipping half-empty. This is a teaching example, not a medical device, and the card says so.

#### 2. A toy risk-tier triage with dates

A rule table keyed to the Act's structure, labelled a **teaching aid**. It encodes the prohibited list, the eight Annex III areas, the Article 6(3) narrow-task derogation (never available when the system profiles people), the Article 50 triggers and the general-purpose threshold, and attaches the verified application dates.

```python
from datetime import date

REFERENCE = date(2026, 10, 2)

PROHIBITED = {
    "manipulative techniques": ("Art 5(1)(a)", date(2025, 2, 2)),
    "exploiting vulnerabilities": ("Art 5(1)(b)", date(2025, 2, 2)),
    "social scoring": ("Art 5(1)(c)", date(2025, 2, 2)),
    "crime prediction from profiling alone": ("Art 5(1)(d)", date(2025, 2, 2)),
    "untargeted face scraping": ("Art 5(1)(e)", date(2025, 2, 2)),
    "emotion inference at work or school": ("Art 5(1)(f)", date(2025, 2, 2)),
    "biometric categorisation of sensitive traits": ("Art 5(1)(g)", date(2025, 2, 2)),
    "real-time remote biometric identification for law enforcement": ("Art 5(1)(h)", date(2025, 2, 2)),
    "non-consensual sexual deepfakes": ("Art 5(1)(ba), as inserted", date(2026, 12, 2)),
}
ANNEX_III = {
    "biometrics": "1", "critical infrastructure": "2", "education": "3", "employment": "4",
    "essential services": "5", "law enforcement": "6", "migration and border control": "7",
    "justice and democratic processes": "8",
}
GPAI_FLOP = 1e25


def triage(system):
    out = []
    practice = system.get("practice")
    if practice in PROHIBITED:
        article, start = PROHIBITED[practice]
        return [("prohibited", article, start)]
    if system.get("annex_i_product_with_third_party_assessment"):
        out.append(("high-risk (Annex I product)", "Art 6(1), Chapter III Sections 1 to 3", date(2028, 8, 2)))
    area = system.get("annex_iii_area")
    if area in ANNEX_III:
        derogation = system.get("narrow_task") and not system.get("profiles_people")
        if derogation:
            out.append(("not high-risk if the assessment is documented", "Art 6(3) and 6(4)", date(2027, 12, 2)))
        else:
            out.append((f"high-risk (Annex III point {ANNEX_III[area]})", "Art 6(2), Chapter III Sections 1 to 3", date(2027, 12, 2)))
    if system.get("talks_to_people"):
        out.append(("transparency duty", "Art 50(1)", date(2026, 8, 2)))
    if system.get("generates_synthetic_content"):
        out.append(("transparency duty", "Art 50(2) marking, 50(4) deepfakes", date(2026, 8, 2)))
    if system.get("gpai_model"):
        systemic = system.get("training_flop", 0) > GPAI_FLOP
        out.append(("GPAI model with systemic risk" if systemic else "GPAI model",
                    "Arts 53 and 55" if systemic else "Art 53", date(2025, 8, 2)))
    return out or [("minimal risk", "Art 4 AI literacy only", date(2025, 2, 2))]


SYSTEMS = {
    "CV screening tool": dict(annex_iii_area="employment"),
    "credit scoring model": dict(annex_iii_area="essential services"),
    "exam proctoring": dict(annex_iii_area="education"),
    "customer support chatbot": dict(talks_to_people=True),
    "image generator": dict(generates_synthetic_content=True),
    "workplace emotion inference": dict(practice="emotion inference at work or school"),
    "nudification app": dict(practice="non-consensual sexual deepfakes"),
    "payslip date extractor in HR": dict(annex_iii_area="employment", narrow_task=True),
    "spam filter": dict(),
    "3e25 FLOP foundation model": dict(gpai_model=True, training_flop=3e25),
    "AI safety component in a medical device": dict(annex_i_product_with_third_party_assessment=True),
}

print(f"teaching triage as of {REFERENCE.isoformat()} (not legal advice)\n")
print(f"{'system':42s} {'tier':46s} {'applies from':13s} status")
for name, system in SYSTEMS.items():
    for tier, article, start in triage(system):
        status = "in application" if start <= REFERENCE else f"in {(start - REFERENCE).days} days"
        print(f"{name:42s} {tier:46s} {start.isoformat():13s} {status}   [{article}]")

print("\nmaximum fine headline figures in Article 99 (the higher of the two for an undertaking, the lower for an SME):")
for what, amount, share in (("prohibited practices", 35_000_000, 7), ("most operator obligations, including Article 50", 15_000_000, 3),
                            ("incorrect information to authorities", 7_500_000, 1)):
    print(f"  {what:50s} EUR {amount:>10,d} or {share}% of worldwide annual turnover")
```

```text
teaching triage as of 2026-10-02 (not legal advice)

system                                     tier                                           applies from  status
CV screening tool                          high-risk (Annex III point 4)                  2027-12-02    in 426 days   [Art 6(2), Chapter III Sections 1 to 3]
credit scoring model                       high-risk (Annex III point 5)                  2027-12-02    in 426 days   [Art 6(2), Chapter III Sections 1 to 3]
exam proctoring                            high-risk (Annex III point 3)                  2027-12-02    in 426 days   [Art 6(2), Chapter III Sections 1 to 3]
customer support chatbot                   transparency duty                              2026-08-02    in application   [Art 50(1)]
image generator                            transparency duty                              2026-08-02    in application   [Art 50(2) marking, 50(4) deepfakes]
workplace emotion inference                prohibited                                     2025-02-02    in application   [Art 5(1)(f)]
nudification app                           prohibited                                     2026-12-02    in 61 days   [Art 5(1)(ba), as inserted]
payslip date extractor in HR               not high-risk if the assessment is documented  2027-12-02    in 426 days   [Art 6(3) and 6(4)]
spam filter                                minimal risk                                   2025-02-02    in application   [Art 4 AI literacy only]
3e25 FLOP foundation model                 GPAI model with systemic risk                  2025-08-02    in application   [Arts 53 and 55]
AI safety component in a medical device    high-risk (Annex I product)                    2028-08-02    in 670 days   [Art 6(1), Chapter III Sections 1 to 3]

maximum fine headline figures in Article 99 (the higher of the two for an undertaking, the lower for an SME):
  prohibited practices                               EUR 35,000,000 or 7% of worldwide annual turnover
  most operator obligations, including Article 50    EUR 15,000,000 or 3% of worldwide annual turnover
  incorrect information to authorities               EUR  7,500,000 or 1% of worldwide annual turnover
```

The output shows how dates and tiers interact. A CV screening tool, a credit scoring model and exam proctoring are high-risk under Annex III points 4, 5 and 3 and apply from 2 December 2027, 426 days after the reference date; a medical-device safety component falls under Annex I with 2 August 2028. The customer support chatbot and image generator are already inside Article 50, since 2 August 2026. The nudification app is prohibited from 2 December 2026, in 61 days, because of the new Article 5(1)(ba). A payslip date extractor in HR is in an Annex III area but performs a narrow task, so it is not high-risk **provided the provider documents the assessment** (Article 6(4)). The foundation model above 10^25 floating point operations is a general-purpose model with systemic risk. The triage cannot decide real cases: it does not see context, exceptions, Union product law, national rules or the Commission's guidelines.

The lab runs the same rules. Its defaults are the CV screening tool and the reference date of 2 October 2026, showing high-risk under Annex III point 4, applying from 2 December 2027, in 426 days. Switch the date to 2027-12-02 to see it come into application.

<RiskTierLab />

#### 3. A tamper-evident audit trail

Article 12 requires automatic event logging for high-risk systems; the log is only evidence if edits are detectable. This block chains each record to the previous one with a SHA-256 hash.

```python
import hashlib
import json

GENESIS = "0" * 64


def entry_hash(previous, record):
    body = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((previous + body).encode()).hexdigest()


def append(log, record):
    previous = log[-1]["hash"] if log else GENESIS
    log.append({"record": record, "hash": entry_hash(previous, record)})


def verify(log):
    previous = GENESIS
    for position, item in enumerate(log):
        if item["hash"] != entry_hash(previous, item["record"]):
            return position
        previous = item["hash"]
    return None


log = []
events = [
    {"t": "2026-10-02T09:00:01Z", "model": "credit-v3.2", "input_id": "app-1041", "score": 0.31, "decision": "approve", "reviewer": None},
    {"t": "2026-10-02T09:00:07Z", "model": "credit-v3.2", "input_id": "app-1042", "score": 0.72, "decision": "refer", "reviewer": "j.ortiz"},
    {"t": "2026-10-02T09:01:15Z", "model": "credit-v3.2", "input_id": "app-1043", "score": 0.55, "decision": "decline", "reviewer": "j.ortiz"},
    {"t": "2026-10-02T09:02:40Z", "model": "credit-v3.2", "input_id": "app-1044", "score": 0.12, "decision": "approve", "reviewer": None},
]
for event in events:
    append(log, event)

print(f"{len(log)} entries, chain ends in {log[-1]['hash'][:16]}")
print("verify untouched log:", verify(log))

log[2]["record"]["decision"] = "approve"
print("verify after editing entry 2 (decline became approve): first bad position", verify(log))

log[2]["record"]["decision"] = "decline"
removed = log.pop(1)
print("verify after deleting entry 1: first bad position", verify(log))
log.insert(1, removed)
print("verify after restoring it:", verify(log))

tail = log[-1]
log[-1] = {"record": dict(tail["record"], decision="decline"), "hash": entry_hash(log[-2]["hash"], dict(tail["record"], decision="decline"))}
print("verify after rewriting the LAST entry and recomputing its hash:", verify(log))
print("the final hash changed, so an external copy of the previous head would expose it:", log[-1]["hash"] != tail["hash"])
```

```text
4 entries, chain ends in 43a19c2ff86cb6d8
verify untouched log: None
verify after editing entry 2 (decline became approve): first bad position 2
verify after deleting entry 1: first bad position 1
verify after restoring it: None
verify after rewriting the LAST entry and recomputing its hash: None
the final hash changed, so an external copy of the previous head would expose it: True
```

Editing entry 2 is caught at position 2, deleting entry 1 at position 1, and restoring it makes the log verify again. The last line is the limit: someone who rewrites the **final** entry and recomputes its hash passes verification, because nothing after it commits to the old value. The fix is operational. Periodically publish or escrow the head hash somewhere the log writer cannot edit, and a rewrite shows as a different head.

<Infographic src="/img/gov/regulation-and-model-documentation-tiers.svg" alt="A tier ladder from prohibited to minimal, three frameworks with verified facts, the nine model card sections and the audit trail results." caption="The tiers, the voluntary frameworks, the nine model card sections and what the chained log can and cannot prove." />

## Designing with it

- **Classify before you build.** Decide the tier at the design stage, because a high-risk tier changes the architecture: logging, oversight, documentation and data governance are cheaper designed in than retrofitted.
- **Treat dates as data.** Keep application dates in one table with their legal source, as the triage does, and review it when the law changes, as it did in July 2026.
- **Generate documents from the pipeline**, so the card and the evaluation cannot drift apart, and report per slice with intervals.
- **Log for reconstruction.** Record the model version, input identifier, output, decision and reviewer, keep the logs for at least the period the text sets for deployers, and anchor the chain head.
- **Pick a framework and map to it.** NIST AI RMF and ISO/IEC 42001 are voluntary; the Act is not. A management system gives you a place to hold the evidence either way.

## Where this stands in 2026

:::info Industry view

- **The high-risk timetable moved in July 2026.** Regulation (EU) 2026/1744 delays the Annex III obligations to 2 December 2027 and the Annex I obligations to 2 August 2028, adds new prohibitions from 2 December 2026, and brings Articles 102 to 110 into application from 27 July 2026. Practitioners reading older material that says 2 August 2026 for high-risk systems are reading a superseded date.
- **Transparency and general-purpose model rules are already running.** Article 50 applies from 2 August 2026, with an extension to 2 December 2026 for older generative systems, and the general-purpose model duties since 2 August 2025.
- **NIST's framework is in revision.** NIST says AI RMF 1.0 is being revised as part of the White House AI Action Plan, and it published a concept note on a critical-infrastructure profile on 7 April 2026.
- **ISO/IEC 42001:2023 remains edition 1** on the IEC webstore, and is the certifiable standard for an AI management system.
- **Not verified here:** the status of harmonised standards, the Commission's classification guidelines, national authorities and any general-purpose code of practice. Check them before you plan a compliance schedule.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> A colleague says high-risk obligations apply from 2 August 2026. Is that right in October 2026?</summary>

No. The Digital Omnibus on AI (Regulation (EU) 2026/1744, in force 27 July 2026) replaced the date: Chapter III Sections 1 to 3 apply from 2 December 2027 for Article 6(2) and Annex III systems and from 2 August 2028 for Article 6(1) and Annex I systems.

</details>

<details>
<summary><strong>Q2.</strong> Which obligations already apply to a customer support chatbot in October 2026?</summary>

Article 50(1): it must be designed so that people are informed they are interacting with an AI system, unless that is obvious. The general date of application, 2 August 2026, has passed. It is not high-risk unless it falls in an Annex III area, and AI literacy (Article 4) applies to its provider and deployer.

</details>

<details>
<summary><strong>Q3.</strong> Why can the narrow-task derogation not rescue a CV-ranking tool that profiles candidates?</summary>

Article 6(3) says an Annex III system that performs profiling of natural persons is always high-risk. The derogations for narrow procedural tasks and similar cases do not apply to it.

</details>

<details>
<summary><strong>Q4.</strong> What does a hash-chained log prove, and what can still be rewritten undetected?</summary>

It proves that earlier entries have not been edited or removed, since each hash covers its predecessor. It cannot detect someone rewriting the last entry and recomputing its hash. Anchor the head hash outside the writer's control.

</details>

<details>
<summary><strong>Q5.</strong> Why does a model card report accuracy by slice, and what did the example show?</summary>

An overall figure hides subgroup behaviour. Overall accuracy was 0.959 while the larger-tumour slice had a benign recall of 0.750, and the intervals for 50 to 64 rows were wide, so the card must say how little the slices support.

</details>

<details>
<summary><strong>Q6.</strong> When must a serious incident with a high-risk system be reported, and by whom?</summary>

By the provider to the market surveillance authorities of the Member State where it occurred, immediately once a causal link is established or reasonably likely and in any event within 15 days of becoming aware; within 10 days for a death; within two days for a widespread infringement or a serious and irreversible disruption of critical infrastructure (Article 73).

</details>

## Further reading

Official texts retrieved from the EU Publications Office in October 2026; other pages opened the same month.

- [Regulation (EU) 2024/1689 (the AI Act)](http://data.europa.eu/eli/reg/2024/1689/oj), Official Journal, 12 July 2024.
- [Regulation (EU) 2026/1744 (Digital Omnibus on AI)](http://data.europa.eu/eli/reg/2026/1744/oj), Official Journal, 24 July 2026.
- NIST, [AI Risk Management Framework 1.0, AI 100-1](https://doi.org/10.6028/NIST.AI.100-1), January 2023, and the [AI RMF page](https://www.nist.gov/itl/ai-risk-management-framework).
- NIST, [AI 600-1, Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf), July 2024.
- IEC webstore, [ISO/IEC 42001:2023](https://webstore.iec.ch/en/publication/90574).
- Mitchell and colleagues, [Model cards for model reporting](https://arxiv.org/abs/1810.03993), 2019.
- Gebru and colleagues, [Datasheets for datasets](https://arxiv.org/abs/1803.09010), 2018, revised 2021.

## Check yourself

- I can name the Act's tiers and say which obligations attach to each.
- I can state the application dates that matter, and say which of them the July 2026 Omnibus changed.
- I can explain why high-risk obligations apply from 2 December 2027 for Annex III systems, and what has applied since 2 August 2026.
- I can name the four NIST AI RMF functions and describe what ISO/IEC 42001 is, and what I could not verify about it.
- I can generate a model card from an evaluation, report slices with intervals and check completeness.
- I can build a tamper-evident log, and explain what it cannot detect without external anchoring.
- I can say plainly that a teaching triage is not legal advice.
