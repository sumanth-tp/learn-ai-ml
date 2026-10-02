---
id: dm-data-privacy-and-governance
title: "Data Management · Lecture 15 — Data Privacy and Governance"
sidebar_label: "15 · Privacy and governance"
sidebar_position: 2
slug: /mlops/data/privacy-and-governance
description: "Govern personal data through purpose, access, retention, k-anonymity limits and differential privacy."
tags: [data-management, privacy, governance, differential-privacy]
---

import Infographic from '@site/src/components/Infographic';
import KAnonymityLab from '@site/src/components/viz/KAnonymityLab';

**In one line.** A privacy control works only when the team can state what data it protects, from whom and for how long.

## The idea in plain words

A data team can make a table technically accurate and still use it irresponsibly. Customer records may be collected for one purpose and copied into a model dataset for another. A supposedly anonymous export may retain combinations of attributes that point back to individuals. A training snapshot may outlive the retention rule of its source. **Data governance** names the ownership, policies and evidence needed to keep data use within an approved purpose; privacy controls implement those decisions in collection, storage, analysis and release.

The lecture lists catalogues, access control, retention and audit as governance practices. It also names data minimisation, masking, tokenisation, encryption, anonymisation and pseudonymisation as protection techniques. These are not interchangeable. Encryption protects data while keys are controlled, but authorised users can still see plaintext. Tokenisation replaces a direct identifier with a reference, but the mapping service can reconnect it. Pseudonymised data can remain personal data when re-identification is possible. A genuinely anonymous release requires an assessment of what recipients can link with other information.

<Infographic src="/img/dm/privacy-governance.svg" alt="The data lifecycle moves from purpose-limited collection through protected transformation to reviewed release; a smallest age and ZIP group of four gives k equals four, while differential privacy needs a unit and epsilon budget." caption="A group count and a noise parameter answer different privacy questions; neither substitutes for governance of the whole data flow." />

The source's worked table groups people by age and ZIP code. If the smallest released group contains **four** people, the table is **4-anonymous** with respect to those chosen quasi-identifiers. The source writes a re-identification probability bound of 1/4. That interpretation needs strong assumptions, such as an attacker knowing only the group and treating all four members as equally likely. It is not a general upper bound: outside knowledge, other columns and group homogeneity may reveal identity or a sensitive attribute. A group of one clearly fails a k≥2 goal and needs suppression or generalisation before release.

:::note Beyond the lecture

The source equates k-anonymity with a 1/k re-identification bound and describes differential privacy as adding epsilon-calibrated noise. The sections below correct those shortcuts. K-anonymity controls indistinguishability on selected quasi-identifiers but has known inference limits. Differential privacy requires a defined neighbouring-dataset relationship, sensitivity, mechanism and composed privacy budget; arbitrary noise does not establish its guarantee.

:::

The lab begins with equivalence groups of sizes **4, 5 and 7**, so **k = 4**. It shows **1/4 = 0.25** only as a uniform-guess illustration within a known group. Move the first group's size to one to see why a unique combination fails even a modest k target. A higher k does not guarantee that the group's sensitive values are diverse.

<KAnonymityLab />

## How it works

### Policies and protection

Governance = ownership, catalog, access control, retention, audit. Protect PII (GDPR) via minimisation, masking/tokenisation, encryption, anonymisation.

### k-anonymity & DP

K-anonymity groups records on selected quasi-identifiers; its k value alone does not bound all disclosure risks. Differential privacy needs a defined neighbouring-dataset unit, sensitivity, mechanism and privacy budget.

:::tip

**Worked.** The smallest age-and-ZIP group has four records, so k = 4 on those fields. One divided by four is only a uniform-guess illustration, not a general risk bound. A group of one needs suppression or generalisation for a k≥2 goal.

:::


## A real system that works this way

A health research organisation may want to publish a patient-level table for a study. The **UK Information Commissioner's Office** describes k-anonymity as a dataset property on selected attributes and explicitly warns about homogeneity and background-knowledge attacks. An analyst can generalise exact age into an age band and ZIP code into a broader area until every released combination occurs at least k times. This reduces simple singling-out, but the release team still checks whether an outside dataset or a uniform diagnosis within a group can reveal sensitive information.

The **European Commission** explains that data minimisation limits collection to what is necessary for a stated purpose. It also distinguishes anonymous data from data that remains identifiable after encryption or pseudonymisation. This affects a model-training workflow: removing names from a table does not necessarily remove obligations when account IDs, rare events or location histories can be linked to people. A catalogue and lineage graph can show which training snapshots received the fields and who can access them.

For aggregate publication, **NIST's differential privacy guidance** offers a different framework. A properly designed mechanism limits how much a release can change when one person's data is added or removed, under a stated neighbouring-dataset definition. Epsilon is a privacy-loss parameter, and repeated releases consume a composed budget. A team can use a vetted mechanism to publish counts or train a model under a formal guarantee, but must specify the unit of privacy, clipping or sensitivity and implementation details. The label "DP" on a dashboard does not prove the calculation is correct.

## Code you can run

Compute k from the released combinations of quasi-identifiers. This is a property of these columns and this release, not a probability of full re-identification.

```python
from collections import Counter

groups = Counter({("20-29", "AB1"): 4, ("30-39", "AB1"): 5, ("40-49", "CD2"): 7})
k = min(groups.values())
uniform_guess = 1 / k
print(f"k={k}; uniform-within-group illustration={uniform_guess:.2f}")
assert k == 4
assert uniform_guess == 0.25
groups[("50-59", "EF3")] = 1
assert min(groups.values()) == 1
```

For a count query where one person's contribution is bounded to one, the Laplace mechanism uses noise scale sensitivity divided by epsilon. The following arithmetic calculates scale and a simple sequential budget. It does not publish a noisy answer and is not a proof that a larger pipeline is differentially private.

```python
def laplace_scale(sensitivity, epsilon):
    if sensitivity <= 0 or epsilon <= 0:
        raise ValueError("sensitivity and epsilon must be positive")
    return sensitivity / epsilon

unit_sensitivity = 1
epsilon_per_release = 0.5
scale = laplace_scale(unit_sensitivity, epsilon_per_release)
releases = 3
basic_composed_epsilon = releases * epsilon_per_release
print(f"scale={scale:.1f}; basic composed epsilon={basic_composed_epsilon:.1f}")
assert scale == 2.0
assert basic_composed_epsilon == 1.5
```

The sensitivity assumption breaks if one person can contribute many rows. A real release must bound contributions, use a vetted random mechanism, track all disclosures and review utility and privacy together. A fixed or public random seed would not be an appropriate shortcut for a private release.

## Designing with it

### Start with purpose and data inventory

For each dataset, record the collection purpose, lawful basis where relevant, owner, data subjects, sensitive fields, retention period and authorised consumers. A catalogue entry is useful only if it describes the actual table and refresh process. Identify copied datasets in notebooks, feature stores, indexes and model-training artifacts; privacy rules can be defeated by an untracked export. Minimise columns at the source or earliest practical stage rather than copying everything and hoping a later filter removes it.

Use role- or attribute-based access at storage and query boundaries. A pipeline service account should have only the access it needs. Separate access to raw identifiers, token mappings and curated features. Log reads and administrative changes in an auditable form without placing sensitive values in the audit log itself. Review access periodically and when a person changes role. A successful encryption check does not address an overly broad read permission.

### Choose the right transformation

Masking may hide data in a development display but leave the original available elsewhere. Tokenisation replaces an identifier with a stable token, useful for joins when a protected mapping service controls reversal. Pseudonymisation reduces direct identification but may retain linkability and remain within personal-data rules. Encryption at rest and in transit protects against some unauthorised access paths, subject to key management. Anonymisation aims to remove reasonable identification paths for a specified release context, which requires a risk assessment rather than a single transform name.

K-anonymity groups records by selected **quasi-identifiers**. The minimum group size is k after generalisation or suppression. It does not protect against a group in which every member has the same sensitive diagnosis. It also does not account automatically for an attacker who knows a precise location or a rare event outside the release. Changing the selected quasi-identifiers changes k. Evaluate utility loss: broad age bands may protect identity while making a medical study unusable. State the intended recipient and likely auxiliary data when deciding whether to release.

### Treat differential privacy as a system property

Define neighbouring datasets and the **unit of privacy**: one row, one person or a household can lead to different guarantees. Bound each unit's contribution and calculate the query's sensitivity. Choose an appropriate proven mechanism and parameters, then track composition across releases. Lower epsilon generally means stronger formal protection and more noise for a given sensitivity; the meaningful value depends on context and implementation. A value without its unit, delta where applicable, clipping rule and composition history is incomplete.

Differential privacy protects the effect of one unit's data on a published output under the formal model. It does not grant permission to collect unnecessary data or remove the need for secure raw storage. Debug logs, intermediate artifacts, repeated unaccounted queries or a flawed random-number implementation can violate the intended protection. Use vetted tools, test implementation assumptions and have a privacy expert review a consequential release.

### Retain and delete deliberately

Retention is a workflow, not a date written in a policy document. Track where a source record can propagate: raw landing, validated tables, features, search indexes, training snapshots, model artifacts and backups. Some derived artifacts may require different handling; record the decision and authority. When a deletion request or source correction arrives, route it through lineage to affected copies. Define completion evidence and a realistic backup expiry. Audit the exception path as carefully as the normal path.

## Review a patient-level research release

A researcher requests a dataset of visits with age, ZIP code and diagnosis. The data steward first confirms the purpose and whether patient-level detail is necessary. Perhaps a region-level count would answer the question with less risk. If rows are needed, remove names, account numbers and unnecessary dates before creating a candidate export. Identify which other columns could act as quasi-identifiers: a rare hospital, exact visit date or unusual treatment may narrow a person far more than age and ZIP alone.

The candidate has three age/ZIP equivalence groups of sizes four, five and seven. Its k on those columns is four. That computation is reproducible, but the steward does not write "risk ≤25%" in the approval record. If one group of four contains only patients with the same diagnosis, membership in that group reveals the diagnosis even without naming the person. If an attacker has a public story about a patient in a small town on a specific day, outside knowledge may distinguish a row. The team tests those scenarios and may generalise the geography or suppress the rare group.

The resulting dataset has lower detail. The researcher checks whether the study still has enough utility: are the needed age effects visible with broader bands? Are suppressed rows concentrated in a marginalised population, biasing the result? Privacy and analytical validity must be assessed together. A release that is safe but systematically erases a subgroup may be unsuitable for the intended analysis. The owner documents the chosen release and its limitations.

Suppose the organisation instead publishes aggregate counts each month. A differential privacy mechanism may be appropriate. The owner defines whether one person's multiple visits count as one protected unit or several and bounds contribution accordingly. For a simple count with unit sensitivity one and epsilon 0.5, Laplace scale is two. If the same population is queried repeatedly, the privacy budget composes; three epsilon-0.5 releases have a basic bound of epsilon 1.5. More advanced accounting can differ, so the release process records the actual mechanism and accountant used.

The published aggregate is only one output path. A notebook export, chart tooltip showing raw rows or an unprotected error log can disclose the same data regardless of the DP count. The steward reviews the pipeline and access controls end to end. NIST guidance emphasises the unit of privacy, algorithm correctness and practical hazards precisely because a formal parameter cannot compensate for unexamined surrounding systems.

### Govern an ML training copy

A model team copies the visits into a training lake and pseudonymises patient IDs. The copied dataset is still linkable across visits and may remain personal data. The owner records the training purpose, authorised team, snapshot version and retention. Access to the token mapping is separated from access to model features. Logs use job and snapshot IDs, not patient names. A lineage edge links the source and transform to every training run so a later correction can find affected artifacts.

The team tests whether the model memorises rare free-text notes. K-anonymity of structured demographic columns does not cover those notes. It may remove or redact free text, apply access restrictions, and evaluate the model's output for unintended disclosure. A DP training claim would need its own documented algorithm and privacy accounting; the DP count example above does not automatically protect the trained model.

When a source record is deleted, the pipeline removes it from active tables and updates the next training snapshot. Whether a previously trained model must be retrained or retired depends on the applicable policy and technical evidence; it is not decided by one SQL delete. The governance record states what changed, which downstream artifacts were reviewed, and which versions remain available under controlled retention. This turns a vague promise of responsible use into an auditable process.

### Audit the controls as a chain

An audit begins at collection: was the purpose communicated, and was every retained field necessary? It follows each transformation and permission change. A table can be masked in one analytics view while a raw copy remains widely readable. A token map can be tightly controlled while verbose logs print the original ID. A training run can use a cleaned snapshot but upload raw sample rows as an artifact. The audit therefore selects a real record and traces where its values and derivatives went, using lineage and access logs as evidence.

Test the negative path. Remove a user's permission, request a dataset export and confirm the denial occurs before data leaves the query service. Submit a deletion and verify it is reflected in active stores and the next approved training snapshot. Create a record with a unique quasi-identifier combination and confirm the release pipeline suppresses or generalises it. None of those checks alone proves anonymity or compliance, but they test whether stated controls actually execute.

Review the system after a new data source, model feature or release recipient is added. The threat model changes when an outside party gains more auxiliary data or a dataset is linked with another collection. An approved release from last year may be inappropriate under a new linkage environment. Treat the risk decision as versioned, with an owner, scope, evidence and next review date.

## Where this stands in 2026

:::info Industry view

- European Commission guidance treats re-identifiable pseudonymised data as personal data and requires purpose-limited minimisation.
- The ICO identifies homogeneity and background-knowledge attacks as limits of k-anonymity.
- NIST differential privacy guidance stresses unit definition, mechanism correctness and composed releases, beyond a lone epsilon value.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What is data governance?</summary>

The framework of policies, roles and controls (ownership/stewardship, catalog, access control, retention, audit) ensuring data is accurate, secure, compliant and responsibly used.<br /><em>Lecture 15 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Name four ways to protect PII.</summary>

Data minimisation, masking/tokenisation, encryption (rest + transit), and anonymisation/pseudonymisation (per GDPR).<br /><em>Lecture 15 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Define k-anonymity.</summary>

Each released quasi-identifier combination occurs in at least k records. This prevents simple singling-out on those fields alone, but other knowledge or sensitive-value homogeneity can still disclose information.<br /><em>Lecture 15 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> The smallest \{age,ZIP\} group has 4 people. Give k and the re-identification bound.</summary>

k = 4 on the selected quasi-identifiers. One divided by four is only a uniform-guess illustration under restrictive assumptions, not a general re-identification bound. A group of size one fails a k≥2 release goal.<br /><em>Lecture 15 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> What guarantee does differential privacy provide?</summary>

A correctly implemented mechanism limits the output change between defined neighbouring datasets. It needs a specified privacy unit, sensitivity, parameters and composed budget; arbitrary noise does not confer the guarantee.<br /><em>Lecture 15 · conceptual</em>

</details>

## Go deeper

- [European Commission on personal data](https://commission.europa.eu/law/law-topic/data-protection/information-business-and-organisations/application-gdpr_en) distinguishes anonymous and re-identifiable data.
- [European Commission on data minimisation](https://commission.europa.eu/law/law-topic/data-protection/reform/rules-business-and-organisations/principles-gdpr/overview-principles/what-data-can-we-process-and-under-which-conditions_en) states purpose and necessity principles.
- [ICO anonymisation guidance](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/data-sharing/anonymisation/how-do-we-ensure-anonymisation-is-effective/) discusses k-anonymity and its limits.
- [NIST SP 800-226](https://csrc.nist.gov/pubs/sp/800/226/final) evaluates differential privacy guarantees and implementation hazards.
- Built from the course lecture "dm-l15-privacy-governance" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can identify a dataset's purpose, owner, access boundary and retention path.
- [ ] I can calculate k=4 for the lecture's minimum group and explain why 1/4 is not a universal risk bound.
- [ ] I can distinguish masking, tokenisation, pseudonymisation, encryption and anonymisation.
- [ ] I can state the unit, sensitivity, mechanism and budget needed for a differential privacy claim.
