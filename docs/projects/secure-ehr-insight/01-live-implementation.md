---
id: secure-ehr-insight-live-implementation
title: "Live Marathon - FDE Project Live Implementation"
sidebar_label: "1 - Building the system live"
sidebar_position: 1
slug: /projects/secure-ehr-insight/live-implementation
description:
  "Build a HIPAA-aware clinical Q&A assistant end to end: PostgreSQL +
  pgvector on AWS, domain-specific clinical embeddings, Microsoft Presidio
  redaction, NVIDIA NeMo Guardrails, FastAPI, Streamlit, and a Dockerised EC2
  deployment."
tags:
  [
    projects,
    healthcare,
    hipaa,
    rag,
    pgvector,
    guardrails,
    pii-redaction,
    fastapi,
    streamlit,
    docker,
    aws,
  ]
---

> **Live session** ·
> [Watch on YouTube](https://www.youtube.com/watch?v=-4BrD23fQvU) · 4 hours 42
> minutes ·
> [Project source (GitHub)](https://github.com/nimowhyca/Secure-EHR-Insight-Clinical-Validator)
>
> Taught live by **Monal Bappy** (data ingestion, retrieval, PII redaction,
> guardrails, API, UI) and **Bappy Ahmed Baki** (Docker + AWS EC2
> deployment), moderated by Krish Naik. Notes follow the session in order.

Build an assistant that lets a doctor ask questions about a patient's history
in plain language, without ever letting raw patient data reach a language
model. The project is small on purpose — the point is not a complex agent, it
is a **privacy-first pipeline** that an AI forward deployed engineer (FDE)
would actually be allowed to ship into a hospital.

## Problem statement

A hospital's electronic health record (EHR) system accumulates millions of
rows of admissions, prescriptions and lab results over time. A doctor who
wants to know "what was this patient's last recorded dosage?" currently has
to scroll through that history by hand. The obvious pitch is "use an LLM to
answer questions over the records in natural language" — and that pitch, on
its own, is a **HIPAA violation waiting to happen**.

The session opens by pressure-testing that obvious answer:

- **Naive answer:** RAG, or text-to-SQL, over the hospital's database.
- **Why it fails immediately:** the raw rows contain **PHI** (protected
  health information) — names, addresses, phone numbers, social security
  numbers, financial and medical details. Sending that to a third-party LLM
  API is exactly what HIPAA exists to prevent. If the data leaks, the
  hospital faces a legal hearing, not a bug ticket.

:::note What HIPAA actually requires
HIPAA (the US Health Insurance Portability and Accountability Act) protects
PHI — any demographic or clinical detail that could identify a patient.
Covered entities include healthcare providers, insurers, clearing houses, and
their business associates (EHR platforms, IT vendors, and — relevant here —
whoever builds the AI layer on top). The dataset used in this project, MIMIC-IV,
is already a public, de-identified research release; the point of the exercise
is not "the data is dirty," it is **learning to build the compliant pipeline
a real hospital dataset would require**, using safe data to rehearse it.
:::

So the brief is not "build a RAG chatbot." It is: build a chatbot that
**never lets identifying information reach the LLM**, and that **refuses to
let the LLM behave like a clinician**. Those two constraints — redaction and
guardrails — are where HIPAA compliance actually lives in this system.

### Doubts · Is this only an AI problem? · 00:20:42

**Monal (to the class):** "If hospital shares data to an LLM, that is a
violation of HIPAA — very, very serious." He deliberately doesn't let the
class settle on "RAG + vector DB + LLM" as a complete answer.

**Response:** An AI engineer thinks "how do I solve it with the model." An
FDE has to also think about **who the client is** (a regulated hospital),
**where it will be deployed** (their existing Postgres, not a new vendor's
stack), and **what happens before the model ever sees the data**. The
security and redaction layer is not a bonus feature bolted on afterwards — it
is the reason the rest of the architecture is shaped the way it is.

**Summary**

- The problem is healthcare EHR Q&A, but the constraint that shapes the whole
  build is HIPAA, not retrieval quality.
- PHI must never reach the LLM in identifiable form.
- The LLM must be blocked from giving clinical advice, regardless of how the
  question is worded.

## AI SDLC: how an FDE builds this, versus classic SDLC

Before touching code, Monal frames the working method itself, because the
class is watching an FDE work, not a from-scratch tutorial:

| Classic SDLC                          | AI SDLC (what this session follows)                                              |
| -------------------------------------- | ---------------------------------------------------------------------------------- |
| Product manager writes tickets         | **You** write the plan, the steps, and the edge cases up front — before asking AI  |
| Developer writes the code              | AI drafts boilerplate + implementation; you review, run it, and validate the result |
| QA writes and runs test cases          | You ask AI for unit tests, run them, and keep looping until satisfied              |
| Code review, checking for secrets      | You ask AI for a PR-style self-review: secrets, security flaws, regressions        |
| Maintenance and hot fixes              | You keep a documented plan/log so a future AI session has full context for fixes   |

:::note Not "vibe coding"
Monal is explicit that AI SDLC "does not mean asking ChatGPT to create a
project." Someone with no SDLC background cannot do a good version of it —
you still have to know what the edge cases are, what the acceptance criteria
are, and what "done" looks like. AI changes *who* writes the first draft of
each artifact, not *whether* the artifacts (plan, tests, review, docs) exist.
:::

## Solution architecture

This is the flow Monal draws on screen and builds toward, piece by piece.
Every phase in this chapter maps onto one part of this diagram.

```mermaid
flowchart TB
    A["Hospital EHR ecosystem<br/>(Postgres, millions of rows)"] --> B["Phase 0<br/>Ingest baseline CSV into Postgres on EC2"]
    B --> C["Phase 1<br/>Install pgvector extension"]
    C --> D["Add clinical_embedding vector(768) column"]
    D --> E["Phase 2<br/>Domain-specific embeddings<br/>(BioClinical ModernBERT)"]
    E --> F[("patient_encounters<br/>rows + embeddings")]

    G["Doctor selects Patient ID<br/>(dropdown, not free text)"] --> H["Doctor asks a question"]
    H --> I["Embed the question<br/>(same 768-d model)"]
    F --> J["Cosine similarity search<br/>scoped to the selected patient only"]
    I --> J
    J --> K["Redact PHI<br/>Microsoft Presidio"]
    K --> L{"NeMo Guardrails:<br/>is this a legal question to ask?"}
    L -->|"No — asks for diagnosis/prescription"| M["Refuse:<br/>'consult the attending physician'"]
    L -->|"Yes — asks about recorded history"| N["Context + question -> DeepSeek LLM"]
    N --> O["Answer + AI-generated disclaimer<br/>rendered in Streamlit"]
```

The two design decisions worth naming explicitly, because they are the
difference between a toy demo and something a hospital's infra team would
accept:

1. **The vector store is the hospital's own Postgres, not a new vendor.**
   Postgres added a `pgvector` extension that supports storing embeddings and
   running cosine similarity **inside SQL**. Given that, introducing Pinecone,
   Qdrant or any other dedicated vector database would mean asking the client
   to trust and pay for an entirely new system, for a capability their
   existing database already has a plugin for. When a client already has
   infrastructure that solves the problem, extend it — don't replace it.
2. **The patient is selected before the question is asked.** A hospital
   table can hold tens of millions of rows. Running a semantic search across
   all of it, per query, does not scale and is not necessary — a doctor never
   asks about "the database," they ask about *their* patient. Narrowing to one
   `subject_id` first turns "search 25 million rows" into "search ~100 rows,"
   which is both faster and a second, independent privacy boundary: a
   mistaken query can't accidentally surface a different patient's data.

### Doubts · Why not just use RAG/LangChain/CrewAI/an agent framework? · 00:14:13, 03:18:33

**Monal (posing it to the audience):** asks which frameworks people know for
building agents (LangChain, AutoGen, CrewAI), then explains why none of them
are used here.

**Response:** Two separate reasons stack up:

- **The retrieval half doesn't need a framework.** pgvector already gives
  cosine distance as a SQL operator (`<=>`). Wrapping that in LangChain's
  retriever abstractions would add a dependency without adding capability.
- **The reasoning half doesn't need an agent.** A ReAct-style agent exists to
  *decide which tool to call next* and *act on its own output* in a loop.
  This system's job is single-shot: fetch context for one patient, redact it,
  check it against guardrails, answer. There is no tool-selection decision to
  make, so a full agent framework would be solving a problem this project
  doesn't have. Conversation memory is instead a plain Python list of
  `{role, content}` messages, replayed on every call — "that is how memory
  works," even inside LangGraph.

**Summary**

- Reach for an orchestration framework when there's real branching/tool-use
  to orchestrate, not by default.
- A hand-rolled message history list is a legitimate memory implementation
  when the task is single-turn retrieval-then-answer.

## Requirements and prerequisites

| Area                 | What you need                                                              |
| --------------------- | --------------------------------------------------------------------------- |
| Programming           | Python, OOP basics, SQL, Git                                              |
| Concepts              | RAG, embeddings, vector similarity, what an LLM call is                   |
| Infrastructure        | AWS account (EC2), a Postgres client, Docker                              |
| LLM access            | A DeepSeek API key (cheap — the whole session cost under \$0.10 in tokens) |
| Not required          | Machine learning / statistics background, LangChain/LangGraph expertise  |

### Complete file: `requirements.txt`

```txt
fastapi==0.141.1
uvicorn==0.53.0
streamlit==1.64.0
psycopg==3.3.5
psycopg-binary==3.3.5
sqlalchemy==2.0.54
pgvector==0.3.6
sentence-transformers==6.0.1
huggingface-hub==1.32.0
torch==2.14.0
transformers==5.17.0
nemoguardrails==0.24.1
presidio-analyzer==2.2.364
presidio-anonymizer==2.2.364
spacy==3.8.16
en-core-web-lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
python-dotenv==1.2.3
requests==2.34.2
pandas==3.0.5
numpy==2.4.6
```

Presidio's analyzer is built on spaCy, so `en_core_web_lg` (the large English
pipeline) has to be installed alongside it — that's the one dependency
pinned by URL rather than by name.

## Repository structure

The layout is built up folder by folder over the session, ending here:

```text
Secure-EHR-Insight-Clinical-Validator/
├── data/
│   ├── MIMIC_IV_Trasncript.csv       # ~230k rows, de-identified EHR data
│   └── data.txt                      # source links (Kaggle dataset, HIPAA reference)
├── scripts/                          # one-off setup scripts, run in order
│   ├── 01_ingest_baseline_data.py
│   ├── 02_verify_ingestion.py
│   ├── 03_apply_vector_schema.py
│   ├── 04_generate_embeddings.py
│   ├── 05_test_vector_search.py
│   └── 06_test_guardrails.py
├── src/
│   ├── database/
│   │   └── schema.sql
│   ├── pii_redaction/
│   │   └── presidio_service.py
│   ├── guardrails/
│   │   ├── config.yml
│   │   └── rails.co
│   ├── api/
│   │   └── main.py                   # FastAPI: 3 endpoints
│   └── ui/
│       └── app.py                    # Streamlit chat UI
├── instructor_notes/
│   └── AWS_EC2_Docker_Deployment_Guide.md
├── requirements.txt
├── .env                               # never committed
├── Dockerfile
├── start.sh
└── .dockerignore
```

`scripts/` are one-shot operational tools — run once to set up state, not
imported by the running application. `src/` is the actual application: three
independently testable layers (database, redaction, guardrails) glued
together by an `api/` and a `ui/`.

## Phase 0 — Give the client their data (EC2 + Postgres)

Every real engagement starts with data that already lives somewhere. There is
no client here, so the session manufactures one: a Postgres database on a
plain EC2 instance, standing in for "the hospital's existing EHR database."

### Set up version control before touching AWS

The very first thing done on screen — before AWS is even opened — is
creating the GitHub repository and committing the empty project, so every
phase that follows can be pushed as it's built:

```bash
git init
# create/edit README.md by hand first
git add .
git commit -m "adding readme"
git branch -M main
git remote add origin https://github.com/<you>/Secure-EHR-Insight-Clinical-Validator.git
git push -u origin main
```

Repository visibility was set to **public**, no license selected. Every
later phase in this chapter closes the same way — `git add .`, a short
commit message naming the phase just finished (e.g. "data source added",
"embeddings storage script added", "guardrails added"), then `git push` —
which is why the finished repository's file layout and its commit history
line up with this chapter's own section order.

The exact order matters here: **security group, then EC2 instance, then
Elastic IP.** Monal did it in the opposite order live and had to backtrack,
because the instance's public IP isn't stable yet when you need to start
referencing it in your `.env` file.

### AWS console walkthrough — security group, EC2 instance, Elastic IP

**Step 1 — Create the security group first, empty of any instance to attach to yet.**

1. Console search bar → type **"security groups"** → open **Security Groups**
   under EC2.
2. **Create security group.**
3. Name it something identifiable, e.g. `ytfd-clinical-database-server`.
4. Under **Inbound rules**, click **Add rule** twice:
   - Rule 1 — Type: **SSH**, Source: **My IP** (the console auto-fills your
     current public IP; this restricts SSH to only your machine).
   - Rule 2 — Type: **PostgreSQL** (auto-populates port `5432`), Source:
     **My IP**.
5. **Create security group.** Note the security group ID/name — you'll pick
   it from a dropdown in the next step.

**Step 2 — Launch the EC2 instance, attaching that security group.**

1. EC2 console → **Instances** → **Launch instance**.
2. **Name:** something identifiable, e.g. `test-fde-database`.
3. **Application and OS Images (AMI):** select **Ubuntu**, then the
   **24.04 LTS** version from the dropdown (24.04 was chosen for familiarity;
   26.04 was also available and equally stable).
4. **Instance type:** select a small free-tier-eligible type — the session
   used a **2 vCPU / 4 GB RAM** class (shown as `c7i-flex` in the console),
   priced at roughly **\$0.08/hour** for Linux at the time of recording.
5. **Key pair:** click **Create new key pair** → name it (e.g. `FDE-database-pair`)
   → format **`.pem`** (for SSH from a terminal, not PuTTY's `.ppk`) → **Create
   key pair**. The browser downloads the `.pem` file immediately — move it
   somewhere you'll remember (e.g. a dedicated `aws-keys/` folder) and never
   commit it to Git. You cannot re-download this file later; losing it means
   losing SSH access to the instance.
6. **Network settings → Firewall (security groups):** choose **"Select
   existing security group"** and pick the one created in Step 1
   (`ytfd-clinical-database-server`) instead of letting AWS create a new one.
7. **Configure storage:** set to **20 GiB** (Postgres plus its logs need
   headroom beyond the OS itself).
8. Leave the remaining defaults, then **Launch instance**.
9. Back in **Instances**, wait for **Instance state** to show **Running** and
   **Status check** to pass before continuing.

**Step 3 — Allocate and associate an Elastic IP, so the instance's public IP never changes.**

1. EC2 console → left sidebar → **Network & Security → Elastic IPs**.
2. **Allocate Elastic IP address** → leave the network border group as your
   region → **Allocate**.
3. Select the newly allocated address → **Actions → Associate Elastic IP address**.
4. Under **Resource type**, choose **Instance**, then pick the instance
   created in Step 2 from the dropdown (identify it by its **private IP
   address**, shown alongside the instance name, if you have more than one
   running).
5. **Associate.** Refresh the **Instances** list — the instance's **Public
   IPv4 address** column now matches the Elastic IP exactly. This value is
   what goes into `DB_HOST` in your `.env` file, and it will survive a stop/
   start of the instance.

:::note Why this order specifically
An Elastic IP associates with an *existing* instance — you cannot associate
one before the instance exists, and the instance's security group has to
already exist before you can attach it at launch time. Reversing steps 1 and
2 just means going back to edit the instance afterwards; reversing step 3
relative to 1-2 doesn't work at all.
:::

**Step 4 — Connect over SSH and install PostgreSQL 14.**

```bash
ssh -i "FDE-database-pair.pem" ubuntu@<your-elastic-ip>
```

(On first connection, accept the host-key prompt — the security group
already guarantees only your IP can reach port 22, so this is the expected
first-time SSH warning, not a red flag.) Once connected, configure it for
remote access:

```bash
# Add the PostgreSQL APT repository
sudo apt update && sudo apt install -y curl ca-certificates
sudo install -d /usr/share/postgresql-common/pgdg
sudo curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
  --fail https://www.postgresql.org/media/keys/ACCC4CF8.asc
sudo sh -c 'echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
  > /etc/apt/sources.list.d/pgdg.list'

sudo apt update
sudo apt install -y postgresql-14 postgresql-contrib-14
```

Then two config edits (`sudo nano ...`, `Ctrl+O`, `Enter`, `Ctrl+X` to save):

```text
# /etc/postgresql/14/main/postgresql.conf
listen_addresses = '*'

# /etc/postgresql/14/main/pg_hba.conf  (append)
host    all             all             0.0.0.0/0               scram-sha-256
```

`listen_addresses = '*'` and a permissive `pg_hba.conf` line are only safe
*because* the security group already restricts who can reach port 5432 at
all — the database-level openness relies on the network-level restriction
sitting in front of it.

5. **Create a dedicated application user**, not the Postgres superuser:

```sql
CREATE DATABASE ehr_db;
CREATE USER fde_admin WITH PASSWORD 'SecureEHR2026!';
ALTER ROLE fde_admin SET client_encoding TO 'utf8';
ALTER ROLE fde_admin SET default_transaction_isolation TO 'read committed';
ALTER ROLE fde_admin SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE ehr_db TO fde_admin;

\c ehr_db
GRANT ALL ON SCHEMA public TO fde_admin;
```

This EC2 instance is used for the rest of Phase 0's live setup — but see the
troubleshooting section right after the ingestion script below for what
actually went wrong the first time it was run, and how it was diagnosed.

### The dataset

The session uses **MIMIC-IV**, a public, de-identified electronic health
record dataset (~230,000 rows, ~373 MB as a flat CSV) covering admissions,
prescriptions, lab tests and discharge diagnoses. Key columns: `subject_id`
(the patient), `hadm_id` (the admission), `drug`, `dose_val_rx`, `test_name`,
`comments` (free-text doctor notes), and `description` (the discharge
diagnosis). `comments` and `description` are the fields the embeddings are
built from later.

### Complete file: `src/database/schema.sql`

```sql
-- src/database/schema.sql
CREATE TABLE IF NOT EXISTS patient_encounters (
    id SERIAL PRIMARY KEY,
    subject_id BIGINT,
    hadm_id BIGINT,
    admission_type VARCHAR(50),
    admission_location VARCHAR(100),
    discharge_location VARCHAR(100),
    insurance VARCHAR(50),
    marital_status VARCHAR(50),
    race VARCHAR(100),
    gender VARCHAR(10),
    anchor_age INT,
    drug VARCHAR(150),
    formulary_drug_cd VARCHAR(100),
    prod_strength VARCHAR(150),
    dose_val_rx VARCHAR(100),
    dose_unit_rx VARCHAR(50),
    form_unit_disp VARCHAR(50),
    route VARCHAR(50),
    eventtype VARCHAR(50),
    careunit VARCHAR(100),
    order_type VARCHAR(50),
    order_subtype VARCHAR(50),
    transaction_type VARCHAR(50),
    spec_type_desc VARCHAR(150),
    test_name VARCHAR(150),
    org_name VARCHAR(150),
    ab_name VARCHAR(150),
    comments TEXT,
    drg_type VARCHAR(50),
    description TEXT,
    drg_severity VARCHAR(50),
    drg_mortality VARCHAR(50)
);
```

`comments` and `description` are `TEXT`, everything else that maps directly
to a CSV column is a bounded `VARCHAR`. `subject_id`/`hadm_id` are `BIGINT`
because MIMIC's synthetic patient IDs are large integers, not sequential.

### Complete file: `scripts/01_ingest_baseline_data.py`

```python
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

def ingest_data():
    # Load environment variables
    load_dotenv()

    # 1. Dynamically resolve project root based on this script's location
    script_dir = Path(__file__).resolve().parent       # .../live-FDE-2/scripts
    project_root = script_dir.parent                   # .../live-FDE-2

    # 2. Build absolute paths to data and schema
    csv_path = project_root / 'data' / 'MIMIC_IV_Trasncript.csv'
    schema_path = project_root / 'src' / 'database' / 'schema.sql'

    # Verify the file actually exists before trying to read it
    if not csv_path.exists():
        raise FileNotFoundError(f"CRITICAL: Could not find dataset at {csv_path}")

    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)

    print(f"Executing schema setup from:\n  {schema_path}")
    with engine.begin() as conn:
        with open(schema_path, 'r') as file:
            conn.execute(text(file.read()))

    print(f"Loading clinical data from:\n  {csv_path}")
    df = pd.read_csv(csv_path)
    # Replace NaN/NaT with None so SQLAlchemy inserts SQL NULLs instead of breaking
    df = df.where(pd.notnull(df), None)

    print(f"Ingesting {len(df)} records into remote AWS PostgreSQL instance...")
    df.to_sql('patient_encounters', engine, if_exists='append', index=False)

    print("Baseline legacy data ingestion complete.")

if __name__ == "__main__":
    ingest_data()
```

Walking through it: the script resolves paths **relative to its own file
location**, not the current working directory, so it can be run from
anywhere. It runs `schema.sql` directly through the same connection (creating
the table if it doesn't exist), then loads the whole CSV into a
pandas DataFrame. `df.where(pd.notnull(df), None)` matters more than it looks
— pandas represents missing values as `NaN`/`NaT`, and SQLAlchemy's parameter
binder doesn't know what to do with those; swapping them for Python `None`
lets it insert proper SQL `NULL`s instead of failing the whole batch.
`df.to_sql(..., if_exists='append')` then does the insert in one call.

### Debugging the first ingestion run — a real troubleshooting session

Running `python scripts/01_ingest_baseline_data.py` for the first time did
**not** work, and the session keeps the failure and the fix on screen
instead of cutting to a working take. Worth walking through, because the
actual bug and the actual diagnosis are both more instructive than a clean
success would have been:

1. **First error:** `invalid literal for int with base 10` — a
   connection-string parsing failure, not the schema or the CSV. The
   instinct is to suspect the Python code; the actual first check was
   whether Postgres was even running.
2. **Checked the EC2 instance:** `sudo systemctl status postgresql` showed
   the service **was not running** — despite having been configured
   earlier. `sudo systemctl restart postgresql` brought it up. Re-running
   the script still failed, but with a *different* error — progress, even
   though it didn't work yet.
3. **Second error, after the restart:** the script could reach *some* host,
   but the connection was refused. Re-checking the `.env` file's `DB_HOST`
   against the EC2 console's **Elastic IP** revealed the actual root
   cause: **the IP address noted down earlier in `.env` was wrong** — a
   direct consequence of creating the EC2 instance *before* the security
   group and Elastic IP, the exact ordering mistake flagged above. The
   instance's IP had changed since it was first written down.
4. **After correcting the IP in `.env`:** a third error — `connection
   refused` specifically on port `5432`. This looks like a security-group
   problem, so the security group's inbound rules were re-checked
   carefully — and were already correct (SSH + PostgreSQL, both scoped to
   "My IP"). The rules were not the bug this time.
5. **Root cause, finally:** Postgres itself had quietly stopped again (or
   never fully come back up from the earlier restart). A second
   `sudo systemctl restart postgresql`, waiting a few seconds, and
   re-running the script succeeded — "it was not an issue from our side.
   We just needed to restart it because it was stuck."

:::warning If Postgres won't accept connections
Two independent things can cause this, and they look identical from the
Python side — a generic connection error. Check them **in this order**,
because the second one is easy to misdiagnose as the first:

1. **Is the recorded IP actually current?** Re-check `.env`'s `DB_HOST`
   against the EC2 console's Elastic IP — especially if the instance was
   created before the Elastic IP was allocated and associated.
2. **Is Postgres actually running?** `sudo systemctl status postgresql`,
   then `sudo systemctl restart postgresql` if it isn't (or even if it
   claims to be, if every other check passes and the error persists).

Only after both of those check out is a security-group rule or a
credentials typo the likely cause.
:::

### Complete file: `scripts/02_verify_ingestion.py`

```python
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def verify_ingestion():
    load_dotenv()

    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)

    print("Verifying data integrity in AWS PostgreSQL...\n")

    with engine.connect() as conn:
        count_result = conn.execute(text("SELECT COUNT(*) FROM patient_encounters")).scalar()
        print(f"Total records found: {count_result}")

        if count_result == 0:
            print("Warning: table is empty. Ingestion may have failed.")
            return

        query = text("""
            SELECT
                subject_id,
                admission_type,
                drug,
                drg_severity
            FROM patient_encounters
            WHERE drug IS NOT NULL
            LIMIT 5
        """)

        sample_df = pd.read_sql(query, conn)

        print("-" * 65)
        print(sample_df.to_string(index=False))
        print("-" * 65)
        print("\nVerification complete. Legacy database state is confirmed.")

if __name__ == "__main__":
    verify_ingestion()
```

A dedicated verification script, run right after ingestion, is the "test at
every checkpoint" habit the session insists on even under time pressure:
"you cannot just rely on AI to create things for you." A row count plus a
five-row sample is enough to catch a silently-empty table before three more
hours are spent building on top of it.

### Doubts · Why database and not table first? · 01:31:40

**Monal (rhetorical, to the class):** "Why database and not table? Guys,
table exists inside database."

**Response:** A database is the container; you cannot create a table before
the database it lives in exists. The ordering (`CREATE DATABASE` → connect →
`CREATE TABLE` via `schema.sql`) isn't arbitrary, it's the only order
Postgres allows.

## Phase 1 — Turning Postgres into a vector store (pgvector)

With rows in place, the "starting point" for the actual AI work is reached.
This is the first FDE decision: **can the client's existing database do
what a dedicated vector database would?**

```bash
# On the EC2 instance
sudo apt update
sudo apt install -y postgresql-14-pgvector

# Confirm the extension files are present
ls /usr/share/postgresql/14/extension/vector*

# Activate it inside the target database (superuser step)
sudo -u postgres psql -d ehr_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Complete file: `scripts/03_apply_vector_schema.py`

```python
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def apply_pgvector():
    load_dotenv()

    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)

    print("Connecting to AWS to apply pgvector schema upgrade...")

    try:
        with engine.begin() as conn:
            # Note: Extension was activated via DBA superuser.
            # We only alter the application table here.
            print("Adding 'clinical_embedding' column (768 dimensions)...")
            conn.execute(text("ALTER TABLE patient_encounters ADD COLUMN IF NOT EXISTS clinical_embedding vector(768);"))

            verify = conn.execute(text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'patient_encounters' AND column_name = 'clinical_embedding';
            """)).fetchone()

            if verify:
                print(f"Schema upgrade complete. Confirmed column: {verify[0]} ({verify[1]})")
            else:
                print("Column not found after alter attempt.")

    except Exception as e:
        print(f"Error applying schema: {e}")

if __name__ == "__main__":
    apply_pgvector()
```

Two privilege levels are deliberately split here: activating the `vector`
extension needs Postgres superuser rights (done once, by hand, on the EC2
box), while altering the application table only needs the `fde_admin`
application user's ordinary privileges. The script also **verifies the
column by querying `information_schema.columns`** rather than trusting that
`ALTER TABLE ... IF NOT EXISTS` silently succeeded — the same "test at every
checkpoint" discipline as Phase 0.

### Doubts · Why 768 dimensions? · 01:58:20

**Monal (to the class):** "Why 768? ... The LLM [embedding model] is going to
extract the embedding vector space of dimension 768. So I also want that to
be 768."

**Response:** The vector column's dimension isn't a free choice — it has to
exactly match whatever embedding model will populate it, decided in the next
phase. 768 is the output size of the domain-specific `bioclinical-modernbert`
sentence-transformer chosen for cost and domain-fit reasons (see below), not
a Postgres or pgvector default. Get the column size wrong relative to the
model and every insert fails.

## Phase 2 — Domain-specific clinical embeddings

The table has a `clinical_embedding` column; every value in it is still
`NULL`. Before writing the embedding script, the session asks: **which
embedding model?**

### Doubts · Why not a general-purpose embedder? · 02:11:20

**Monal (to the class):** "What is special about our data that we cannot
choose a normal or general embedder?"

**Response:** Two independent reasons, both raised by the class and
confirmed:

- **Domain vocabulary.** A general embedder's tokenizer will fragment
  clinical terms ("Furosemide", "peritoneal", drug and lab codes) into
  generic sub-word pieces it has never seen used together, instead of
  treating them as the meaningful medical units they are. A model
  pre-trained on clinical text keeps that vocabulary intact.
- **PHI-shaped text.** The text being embedded is doctor's notes and
  diagnoses — exactly the sensitive content this whole project exists to
  protect. Choosing an embedding provider is *also* a data-handling
  decision, not just an accuracy one.

The model chosen: **`NeuML/bioclinical-modernbert-base-embeddings`**, a
sentence-transformer run **locally on CPU** — no embedding API call ever
leaves the machine. Its output dimension is 768, which is why the column was
sized that way. (OpenAI's embeddings, by comparison, run 1024–3072
dimensions and cost money per call — a real consideration when the row count
is a quarter of a million.)

### Complete file: `scripts/04_generate_embeddings.py`

```python
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def generate_and_store_embeddings():
    load_dotenv()
    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)

    # 1. Load the BioClinical ModernBERT embedding model
    print("Loading local BioClinical ModernBERT model...")
    model = SentenceTransformer('NeuML/bioclinical-modernbert-base-embeddings')

    # Guardrail: Programmatically verify the output dimension is 768
    if model.get_sentence_embedding_dimension() != 768:
        raise ValueError("CRITICAL DIMENSION MISMATCH: Model output does not match database vector(768).")

    batch_size = 2
    demo_limit = 100  # kept small live; a real backfill removes this cap

    with engine.begin() as conn:
        print(f"Generating batched embeddings for up to {demo_limit} patient records...")

        with tqdm(total=demo_limit, desc="Vectorizing PHI", unit="rows") as pbar:
            processed = 0

            while processed < demo_limit:
                select_query = text("""
                    SELECT id, admission_type, drug, test_name, drg_severity, description, comments
                    FROM patient_encounters
                    WHERE clinical_embedding IS NULL
                    LIMIT :batch_size
                """)
                batch = conn.execute(select_query, {"batch_size": batch_size}).mappings().fetchall()

                if not batch:
                    break  # No more NULL records

                # 1. Extract strings into a single list for parallel processing
                clinical_texts = []
                ids = []
                for row in batch:
                    components = []
                    if row['admission_type']: components.append(f"Admission: {row['admission_type']}")
                    if row['drug']: components.append(f"Prescribed: {row['drug']}")
                    if row['test_name']: components.append(f"Lab Test: {row['test_name']}")
                    if row['drg_severity']: components.append(f"Severity Level: {row['drg_severity']}")
                    if row['description']: components.append(f"Diagnosis: {row['description']}")
                    if row['comments']: components.append(f"Notes: {row['comments'][:250]}")

                    clinical_texts.append(" | ".join(components))
                    ids.append(row['id'])

                # 2. BATCHED ENCODING: feed the entire list to the model at once
                embeddings = model.encode(clinical_texts, batch_size=batch_size).tolist()

                # 3. Prepare BULK update parameters
                update_params = [
                    {"id": record_id, "embedding": str(emb)}
                    for record_id, emb in zip(ids, embeddings)
                ]

                # 4. BULK UPDATE: execute all rows in a single round-trip
                update_query = text("""
                    UPDATE patient_encounters
                    SET clinical_embedding = :embedding
                    WHERE id = :id
                """)
                conn.execute(update_query, update_params)

                processed += len(batch)
                pbar.update(len(batch))

    print("\nPhase 3 complete: clinical records are vectorized and ready for hybrid search.")

if __name__ == "__main__":
    generate_and_store_embeddings()
```

Three things worth understanding, not just reading:

- **The embedding input is a composite string**, not just the doctor's raw
  comment. Admission type, drug, test name, severity, diagnosis and a
  truncated note are joined with `" | "` into one string per row before
  encoding. Searching later against *that* combined string is what lets a
  query like "liver disease and fluid retention" match a row even if the
  word "liver" only appears in the `description` field and "fluid retention"
  only appears in the `comments`.
- **`WHERE clinical_embedding IS NULL`** is the resumability safeguard —
  rows that already have a vector are never re-processed, so the script can
  be re-run after a crash or a deliberate pause without redoing work.
- **The dimension check runs before any encoding happens.** If the model's
  output size doesn't match what the column expects, the script fails loud
  and immediately, instead of failing 50,000 rows into a silent mismatch.

:::warning This does not scale to the full dataset live
Encoding and writing embeddings **one small batch at a time on a CPU-only
EC2 instance** is slow: two rows took roughly a second in the demo, and the
full ~230,000-row dataset was estimated at **6-7 hours** at that rate — not
something a live session can wait through. Monal's actual workaround: a
*separate* EC2 instance, prepared **before** the stream, already had ~11,000
rows embedded; the session swaps its own `.env` to point at that instance's
IP and continues from there. The demo dataset for every later step is
therefore ~11,000 rows, not the full 230,000 — worth remembering when a
`patients` dropdown later looks smaller than expected.
:::

## Phase 3 — Semantic search, scoped to one patient

Before wiring this into an API, prove the primitive works in isolation.

### Complete file: `scripts/05_test_vector_search.py`

```python
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

def test_vector_search():
    load_dotenv()
    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)

    print("Loading local BioClinical ModernBERT model...")
    model = SentenceTransformer('NeuML/bioclinical-modernbert-base-embeddings')

    # 1. Define a complex, natural language medical query
    query_text = "Patient presenting with severe liver disease and fluid retention needing diuretics"
    print(f"\nSemantic Query: '{query_text}'")

    # 2. Vectorize the query locally
    query_vector = model.encode(query_text).tolist()

    # 3. Search AWS using pgvector's cosine distance operator (<=>)
    search_sql = text("""
        SELECT
            id,
            description,
            drug,
            clinical_embedding <=> CAST(:query_vector AS vector(768)) AS cosine_distance
        FROM patient_encounters
        WHERE clinical_embedding IS NOT NULL
        ORDER BY cosine_distance ASC
        LIMIT 3;
    """)

    with engine.connect() as conn:
        results = conn.execute(search_sql, {"query_vector": str(query_vector)}).mappings().fetchall()

        print("\nTop 3 Semantic Matches:")
        for rank, row in enumerate(results, 1):
            print("-" * 65)
            print(f"Rank {rank} (Distance: {row['cosine_distance']:.4f})")
            print(f"Diagnosis : {row['description']}")
            print(f"Drug      : {row['drug']}")
            print(f"Record ID : {row['id']}")

if __name__ == "__main__":
    test_vector_search()
```

`<=>` is pgvector's **cosine distance** operator — smaller means more
similar, the opposite direction from cosine *similarity*. `ORDER BY
cosine_distance ASC LIMIT 3` is doing the same job FAISS or Pinecone would,
directly inside a SQL `ORDER BY`. `WHERE clinical_embedding IS NOT NULL`
matters at this stage of the demo specifically because most of the ~230,000
rows are still un-embedded — without that filter, every un-embedded row
would either error out or sort arbitrarily.

Note that this test script searches the *whole* table (minus the null
filter) — the patient-scoping (`WHERE subject_id = :patient_id`) is added
one layer up, in the FastAPI endpoint, once a patient has actually been
selected. This script's job is only to prove the cosine-distance query
itself is correct.

### Doubts · Should search run on all 25 million rows? · 00:44:22

**Monal (to the class):** "If this DB contains 25 million rows, do you think
this is a good approach to do similarity search on all 25 million rows of
data?"

**Response:** No — and the fix isn't a bigger index, it's a smaller search
space. A doctor works with a small, repeated set of patients, not the whole
hospital. Making patient selection a **mandatory dropdown step before the
chat box even appears** turns "search everything" into "search ~100 rows for
this one patient," which is both a performance win and, independently, a
privacy boundary (a mistyped or ambiguous query can no longer surface a
different patient's data by accident).

## Phase 4 — Zero-trust PII redaction (Microsoft Presidio)

The rows returned from Phase 3 still contain names, phone numbers, SSNs and
hospital names in plain text. Nothing gets near the LLM until this layer
runs.

### Complete file: `src/pii_redaction/presidio_service.py`

```python
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

class ClinicalPIIRedactor:
    def __init__(self):
        print("Initializing Microsoft Presidio Zero-Trust Middleware...")
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # --- FDE FIX 1: Custom SSN Pattern Recognizer ---
        # Overrides the strict checksum to catch ANY xxx-xx-xxxx format
        ssn_pattern = Pattern(name="catch_all_ssn", regex=r"\d{3}-\d{2}-\d{4}", score=0.9)
        ssn_recognizer = PatternRecognizer(supported_entity="US_SSN", patterns=[ssn_pattern])
        self.analyzer.registry.add_recognizer(ssn_recognizer)

        # --- FDE FIX 2: Custom Deny-List for Medical Facilities ---
        # Forces Presidio to recognize specific hospital names as organizations
        hospital_recognizer = PatternRecognizer(
            supported_entity="ORGANIZATION",
            deny_list=["Massachusetts General Hospital", "Mayo Clinic", "Cleveland Clinic"],
            deny_list_score=1.0
        )
        self.analyzer.registry.add_recognizer(hospital_recognizer)

        self.target_entities = [
            "PERSON",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "US_SSN",
            "LOCATION",
            "ORGANIZATION",
            "DATE_TIME"
        ]

    def redact_clinical_context(self, raw_text: str) -> str:
        if not raw_text:
            return ""

        analyzer_results = self.analyzer.analyze(
            text=raw_text,
            entities=self.target_entities,
            language='en',
            score_threshold=0.4
        )

        anonymized_result = self.anonymizer.anonymize(
            text=raw_text,
            analyzer_results=analyzer_results
        )

        return anonymized_result.text

if __name__ == "__main__":
    redactor = ClinicalPIIRedactor()

    simulated_ehr_note = """
    Patient John Doe (SSN: 234-00-1234) was admitted to Massachusetts General Hospital
    on March 15th following a severe reaction to Furosemide.
    Wife Jane Doe can be reached at 415-555-0198 or jane.doe@email.com.
    """

    print("\nRAW PHI FROM DATABASE:")
    print(simulated_ehr_note.strip())

    print("\nREDACTED OUTPUT (Safe for LLM Prompt):")
    safe_text = redactor.redact_clinical_context(simulated_ehr_note)
    print(safe_text.strip())
```

How it actually works, in the order the class was walked through it:

1. **`self.analyzer`** divides raw text into **intents** — spans tagged as
   `PERSON`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, and so on, each with a
   confidence score. This step *identifies*, it does not remove anything yet.
2. **`self.anonymizer`** takes those tagged spans and actually strips or
   masks them, for whichever entity types you tell it to target
   (`self.target_entities`).
3. **The two custom recognizers exist because out-of-the-box Presidio
   missed real cases the first time it was tried live:**
   - Presidio's built-in `US_SSN` recognizer applies a strict checksum and
     missed some valid-looking `xxx-xx-xxxx` numbers. The fix is a
     `PatternRecognizer` with a permissive regex (`\d{3}-\d{2}-\d{4}`) at a
     high confidence score (`0.9`), registered *in addition to* the default.
   - Presidio has no idea "Massachusetts General Hospital" is an
     organization — it's not a name it has ever been trained to recognize.
     The fix is a **deny-list recognizer**: an explicit list of hospital
     names, tagged `ORGANIZATION` at `deny_list_score=1.0` (maximum
     confidence) whenever seen verbatim.
4. **`score_threshold=0.4`** is the cutoff applied when deciding whether a
   detected entity gets redacted. Custom recognizers are scored high (`0.9`,
   `1.0`) specifically so they always clear that bar.

### Doubts · Why not just redact every number? · 02:38:30

**Monal (to the class):** "Not every number in our data should be redacted
... let's suppose someone's data has a bacteria count of 10,000. Do you
think that number should also get redacted?"

**Response:** No — a lab value is not PHI, a social security number is.
Blanket redaction of anything numeric would destroy the clinical content the
whole system exists to answer questions about. This is also why the SSN
regex fix is deliberately narrow (`\d{3}-\d{2}-\d{4}` specifically) rather
than "redact all digit sequences," and why the session notes that in a real
multi-region deployment, region-specific identifier formats (SSN in the US,
Aadhaar in India) would need their **own** regex, applied only where
relevant — "not every pattern can be recognized" out of the box, and you
should not rely on a system that silently fails to redact a format it has
never seen.

**Summary**

- Presidio's analyzer tags PII spans; its anonymizer removes them. They are
  two separate engines composed together, not one black box.
- Domain gaps in a general PII tool (hospital names, a checksum-strict SSN
  pattern) get closed with custom `PatternRecognizer`s, not by retraining
  the underlying model.
- Redaction is targeted at *identifying* information, not at anything that
  merely looks like structured data — clinical values must survive.

## Phase 5 — Guardrails against clinical advice (NVIDIA NeMo)

Redaction solves *who this data is about*. It does nothing about *what the
doctor is allowed to ask the model to do*. A question like "should I
prescribe a higher dose?" carries no PHI at all, and still must never reach
the LLM — an AI system giving prescribing advice is a different, equally
serious compliance failure.

```mermaid
flowchart LR
    subgraph HIPAA["Where HIPAA compliance actually lives in this system"]
        R["Redaction<br/>(Presidio)<br/>protects WHO the data is about"]
        GR["Guardrails<br/>(NeMo)<br/>restricts WHAT the model is allowed to do"]
    end
```

### Complete file: `src/guardrails/rails.co`

```colang
define user ask for medical advice
  "Can you prescribe me a new medication?"
  "What is the best treatment for this?"
  "Should I increase the patient's dosage?"
  "Diagnose my symptoms based on this chart."
  "Recommend a drug for fluid retention."

define user ask about patient history
  "What was the patient's last recorded dosage?"
  "When was the patient admitted?"
  "Show me the historical lab results."
  "What medication is the patient currently taking?"
  "What was the dosage of Furosemide?"

define bot refuse medical advice
  "I am an enterprise EHR retrieval system. For legal and compliance reasons, I cannot provide new medical diagnoses or recommend medication changes. Please consult the attending physician."

define flow prevent medical advice
  user ask for medical advice
  bot refuse medical advice
```

This is a **Colang** file (NeMo Guardrails' own DSL — the file extension
`.co` stands for Colang, which nobody in the live chat had heard of before
this session). Reading it top to bottom the way it was taught:

- **`define user ask for medical advice`** / **`define user ask about
  patient history`** each give the guardrail *example phrasings* of an
  intent. NeMo uses an LLM under the hood to classify an incoming message
  against these examples — the examples don't have to be exhaustive, they
  anchor a semantic match.
- **`define bot refuse medical advice`** is the fixed refusal text — an
  explicit, auditable, non-generated string, not something the LLM
  freestyles.
- **`define flow prevent medical advice`** wires the two together: *if* the
  user intent matches "ask for medical advice," *then* the bot response is
  the refusal, and — critically — **the flow stops there**. The underlying
  LLM is never called for a blocked message.

### Getting and paying for the DeepSeek API key

Guardrails still need an actual LLM behind them — to classify intents
against the Colang examples, and to generate the final answer once a
question is allowed through. That's a paid API call, and the session is
explicit about which provider and why:

1. **Provider: DeepSeek**, chosen for being "very cheap and very powerful."
   Monal topped up **\$2** total for this and a previous FDE session
   combined — by the time of recording, that covered roughly **100 API
   requests and ~418,000 tokens**, for a spend of **under \$0.10**.
2. **Why not a free tier instead** (Google Gemini was named specifically)?
   Free-tier LLM APIs are typically rate-limited more aggressively, and
   are more likely to throw an error under load — an acceptable risk for
   personal experimentation, but not something you can afford live, on
   stream, in front of an audience. Paying a small, known amount buys
   reliability.
3. **Create the key:** DeepSeek's platform console → **API keys** → create
   a new key (it starts with `sk-`) → copy it immediately, it's shown only
   once.
4. **Two different environment variable names for the same URL** — this
   trips people up, so it's worth stating explicitly. The Streamlit/FastAPI
   side and the NeMo Guardrails side each read their own variable name for
   what is, in this project, the *same* DeepSeek endpoint:

```env
# .env — both point at DeepSeek; NeMo Guardrails and the app read different names
OPENAI_API_KEY=sk-<your-deepseek-key>
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_BASE=https://api.deepseek.com/v1
```

`base_url` inside `config.yml` below is what actually points NeMo
Guardrails' `engine: openai` at DeepSeek instead of OpenAI itself — NeMo
Guardrails (and most LLM tooling) speaks the OpenAI client protocol
regardless of which provider actually answers, which is exactly what
makes swapping DeepSeek in this straightforward.

### Complete file: `src/guardrails/config.yml`

```yaml
models:
  - type: main
    engine: openai
    model: deepseek-chat
    parameters:
      base_url: "https://api.deepseek.com/v1"

  - type: embeddings
    engine: SentenceTransformers
    model: NeuML/bioclinical-modernbert-base-embeddings
```

Two models are configured for two different jobs: `deepseek-chat` (via an
OpenAI-compatible endpoint pointed at DeepSeek's API — NeMo Guardrails
speaks the OpenAI client protocol regardless of which provider actually
answers) is the **reasoning model**, used both for the final answer and for
classifying intents against the Colang examples. The same
`bioclinical-modernbert` embedding model from Phase 2 is reused here so that
intent matching and clinical search are conceptually consistent — one
embedding space for one domain.

### Complete file: `scripts/06_test_guardrails.py`

```python
import os
import asyncio
from nemoguardrails import RailsConfig, LLMRails
from dotenv import load_dotenv

async def test_guardrails():
    load_dotenv()

    print("Initializing NeMo Guardrails Firewall...")
    config = RailsConfig.from_path("./src/guardrails")
    rails = LLMRails(config)

    # --- TEST 1: A Valid Retrieval Prompt ---
    valid_prompt = "What was the patient's last recorded dosage of Furosemide?"
    print(f"\nValid Query: '{valid_prompt}'")
    res_valid = await rails.generate_async(messages=[{"role": "user", "content": valid_prompt}])
    print(f"LLM Response: {res_valid['content']}")

    # --- TEST 2: An Illegal Medical Advice Prompt ---
    illegal_prompt = "Based on the fluid retention, should I prescribe a higher dose of Furosemide?"
    print(f"\nIllegal Query: '{illegal_prompt}'")
    res_illegal = await rails.generate_async(messages=[{"role": "user", "content": illegal_prompt}])
    print(f"Guardrail Intercept: {res_illegal['content']}")

if __name__ == "__main__":
    asyncio.run(test_guardrails())
```

Run live, the two test cases behaved exactly as designed: the history
question ("what was the last recorded dosage") came back asking for more
identifying detail (the LLM answering normally — guardrails let it through),
while the prescribing question ("should I prescribe a higher dose") was
intercepted with the fixed refusal string, **never reaching DeepSeek at
all**.

:::note Not from the session — what "core file" actually stands for
Monal tells the class "I will tell you the full form later" when asked what
a `.co`/`core` file is, and the session moves on without circling back.
Colang ("**Co**nversational **lang**uage") is NeMo Guardrails' own DSL for
defining user/bot message canonical forms and flows; `.co` is simply its file
extension, unrelated to any "core file" naming convention elsewhere.
:::

**Summary**

- Redaction and guardrails answer two different questions: *who is this
  about* versus *what is the model allowed to do*. HIPAA compliance in this
  system is the combination of both, not either alone.
- A guardrail flow can stop a message before the underlying LLM is ever
  called — the refusal is a fixed string, not a generated one, which makes
  it auditable and impossible to jailbreak once the intent match fires.

## Phase 6 — The FastAPI backend

Three components now exist independently: vector search, redaction,
guardrails. The API layer is where they compose into one request/response
cycle.

### Complete file: `src/api/main.py`

```python
# uvicorn src.api.main:app --reload
import asyncio
import os
from typing import List

from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from pydantic import BaseModel

from nemoguardrails import RailsConfig, LLMRails
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.pii_redaction.presidio_service import ClinicalPIIRedactor

load_dotenv()

app = FastAPI(title="Zero-Trust Clinical RAG API")

# Global state to hold our heavy ML models so they only load once at startup
middleware = {}

@app.on_event("startup")
async def startup_event():
    print("Booting Enterprise AI Middlewares...")

    # 1. Load Presidio (spaCy)
    middleware["redactor"] = ClinicalPIIRedactor()

    # 2. Load NeMo Guardrails (ModernBERT + DeepSeek)
    config = RailsConfig.from_path("./src/guardrails")
    middleware["rails"] = LLMRails(config)

    # 3. Load the local embedding model matching your embedding script
    print("Loading local BioClinical ModernBERT embedding model...")
    middleware["embedder"] = SentenceTransformer('NeuML/bioclinical-modernbert-base-embeddings')

    print("System Ready on port 8000.")


# --- DATABASE CONNECTION HELPER ---
def get_db_engine():
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    if db_host:
        DB_URL = f"postgresql+psycopg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    else:
        DB_URL = os.getenv("POSTGRES_URL", "postgresql+psycopg://postgres:password@localhost:5432/clinical_db")

    return create_engine(DB_URL)


# --- MODELS ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    patient_id: str
    messages: List[ChatMessage]

class ClinicalQuery(BaseModel):
    patient_id: str
    prompt: str


# --- ENDPOINT 1: TRIAL QUERY (MOCK DB) ---
@app.post("/api/v1/clinical-query")
async def process_clinical_query(query: ClinicalQuery):
    try:
        raw_db_context = f"""
        Patient John Doe (ID: {query.patient_id}) was admitted on March 15th.
        Last recorded Furosemide dosage was 40mg IV.
        Attending physician: Dr. Gregory House, ID: 20043.
        """

        redactor = middleware["redactor"]
        safe_context = redactor.redact_clinical_context(raw_text=raw_db_context)

        augmented_prompt = f"Clinical Context:\n{safe_context}\n\nUser Question: {query.prompt}"

        rails = middleware["rails"]
        response = await rails.generate_async(messages=[{"role": "user", "content": augmented_prompt}])

        return {
            "status": "success",
            "redacted_context_used": safe_context.strip(),
            "llm_response": response['content']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ENDPOINT 2: PRODUCTION CHAT (NATIVE PGVECTOR DB) ---
@app.post("/api/v1/chat")
async def process_chat(request: ChatRequest):
    try:
        latest_question = request.messages[-1].content
        engine = get_db_engine()

        # 1. Embed the user's question into a 768-dim vector
        embedder = middleware["embedder"]
        query_vector = embedder.encode(latest_question).tolist()

        # 2. Native pgvector similarity search on patient_encounters table
        with engine.connect() as conn:
            query = text("""
                SELECT drug, dose_val_rx, dose_unit_rx, route, eventtype, test_name, comments, description
                FROM patient_encounters
                WHERE subject_id = :subject_id AND clinical_embedding IS NOT NULL
                ORDER BY clinical_embedding <=> CAST(:query_embedding AS vector)
                LIMIT 5;
            """)

            result = conn.execute(query, {
                "subject_id": int(request.patient_id),
                "query_embedding": str(query_vector)
            })
            rows = result.fetchall()

            if not rows:
                real_db_context = f"No historical records found for patient {request.patient_id}."
            else:
                context_lines = []
                for row in rows:
                    context_lines.append(
                        f"Drug: {row.drug} ({row.dose_val_rx} {row.dose_unit_rx}), Route: {row.route}, "
                        f"Event: {row.eventtype}, Test: {row.test_name}, Comments: {row.comments}, Diagnosis: {row.description}"
                    )
                real_db_context = f"[Records for Patient ID: {request.patient_id}]\n" + "\n".join(context_lines)

        # 3. Redact the real context via Presidio
        redactor = middleware["redactor"]
        safe_context = redactor.redact_clinical_context(raw_text=real_db_context)

        # 4. Assemble prompt
        augmented_prompt = f"Clinical Context:\n{safe_context}\n\nUser Question: {latest_question}"

        # 5. Format history for Guardrails
        nemo_history = [{"role": msg.role, "content": msg.content} for msg in request.messages[:-1]]
        nemo_history.append({"role": "user", "content": augmented_prompt})

        # 6. Route through Guardrails
        rails = middleware["rails"]
        response = await rails.generate_async(messages=nemo_history)

        return {"status": "success", "llm_response": response['content']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ENDPOINT 3: FETCH UNIQUE PATIENTS (EMBEDDINGS ONLY) ---
@app.get("/api/v1/patients")
async def get_unique_patients():
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            # Only fetch patients who actually have generated embeddings
            query = text("""
                SELECT DISTINCT subject_id
                FROM patient_encounters
                WHERE subject_id IS NOT NULL AND clinical_embedding IS NOT NULL
                ORDER BY subject_id;
            """)
            result = conn.execute(query)
            patients = [str(row[0]) for row in result]

            return {"patients": patients if patients else ["No embedded patients found"]}

    except Exception as e:
        return {"patients": [], "error": str(e)}
```

Three endpoints, three distinct jobs:

- **`POST /api/v1/clinical-query`** is a **trial endpoint** — it builds a
  hard-coded mock "database row" (John Doe, a dosage, an attending
  physician) rather than querying Postgres, purely to prove the
  redact-then-guardrail-then-LLM pipeline works before wiring in real
  retrieval. It's never called by the Streamlit UI.
- **`POST /api/v1/chat`** is the real endpoint. Note the order of
  operations: **embed → search (scoped to `subject_id`) → redact → guardrail
  → LLM.** Redaction happens *after* retrieval (there's no PHI to redact
  until rows come back) but strictly *before* anything is handed to
  guardrails or the model. `request.messages[:-1]` plus the freshly
  augmented latest message reconstructs conversation history with the
  *redacted*, context-augmented version of only the newest turn — earlier
  turns are passed through as already-sent.
- **`GET /api/v1/patients`** powers the dropdown. The `clinical_embedding IS
  NOT NULL` filter means only patients who've actually been vectorized show
  up — a direct, visible consequence of the Phase 2 demo-limit decision:
  with only ~11,000 of ~230,000 rows embedded, most `subject_id`s will
  simply never appear in this list.

The `@app.on_event("startup")` hook loading Presidio, NeMo Guardrails and the
sentence-transformer model **once**, into a shared `middleware` dict, rather
than per-request, is what keeps latency reasonable — spaCy's pipeline and a
transformer model are too expensive to reload on every single API call.

### Doubts · Why pass `patient_id` in every chat payload? · 03:27:22

**Monal (to the class):** "When I was testing this application, when I was
just passing the messages, AI in the next response was saying 'which patient
you're talking about' — it means it forgot the patient."

**Response:** Conversation memory here is just replayed message history; it
has no independent notion of "current patient" unless that fact is
re-supplied. Sending `patient_id` alongside `messages` on *every* call, not
just the first one, is the fix — an explicit piece of state the LLM would
otherwise have to (unreliably) re-infer from earlier turns.

## Phase 7 — The Streamlit frontend

### Complete file: `src/ui/app.py`

```python
import streamlit as st
import requests

API_CHAT_URL = "http://localhost:8000/api/v1/chat"
API_PATIENTS_URL = "http://localhost:8000/api/v1/patients"

st.set_page_config(page_title="Zero-Trust Clinical EHR", layout="centered")
st.title("Enterprise EHR Chat")
st.caption("Protected by Presidio Zero-Trust & NeMo Guardrails")

# --- FETCH PATIENTS DYNAMICALLY ---
@st.cache_data(ttl=300)  # Cache the list for 5 minutes to reduce database load
def fetch_patient_list():
    try:
        response = requests.get(API_PATIENTS_URL)
        if response.status_code == 200:
            return response.json().get("patients", ["No patients found"])
    except:
        return ["Database Connection Error"]
    return ["Unknown Error"]

patient_list = fetch_patient_list()

# Streamlit's selectbox is automatically searchable!
patient_id = st.sidebar.selectbox("Select Patient File (Type to search)", patient_list)
st.sidebar.info(f"Database explicitly locked to Patient: {patient_id}")

# --- CHAT MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- CHAT INPUT & API CALL ---
if prompt := st.chat_input(f"Ask about patient {patient_id}'s history..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    payload = {
        "patient_id": patient_id,
        "messages": st.session_state.messages
    }

    with st.spinner("Analyzing securely..."):
        try:
            response = requests.post(API_CHAT_URL, json=payload)
            response.raise_for_status()
            bot_reply = response.json().get("llm_response")

            # Enforce Output Disclaimer
            bot_reply += "\n\n*AI generated summary. Do not use for diagnostic purposes.*"

        except requests.exceptions.RequestException as e:
            bot_reply = f"API Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.chat_message("assistant").write(bot_reply)
```

Read top to bottom, this is deliberately small:

- **`@st.cache_data(ttl=300)`** caches the patients dropdown for five
  minutes, so a database round-trip doesn't happen on every Streamlit
  script rerun (Streamlit reruns the entire script on every interaction).
- **`st.session_state.messages`** is the same plain list-of-dicts memory
  discussed earlier, this time on the client side — it survives Streamlit
  reruns within one browser session but resets on a real page reload,
  because nothing persists it anywhere durable.
- **The disclaimer is appended in code, unconditionally**, on every
  successful response — "*AI generated summary. Do not use for diagnostic
  purposes.*" This isn't cosmetic: labelling AI-generated clinical content
  as such is treated in the session as part of HIPAA-adjacent duty of care,
  not just good UX.
- **`patient_id` rides along in every `payload`**, exactly matching the
  `/api/v1/chat` contract discussed above.

Rendered, this is what the UI actually looks like — a patient selected in
the sidebar, and one turn of the redacted, guardrail-checked answer with the
mandatory AI disclaimer:

![Secure EHR Insight Streamlit UI, showing a selected patient and a redacted, guarded answer with the AI-disclaimer](/img/secure-ehr-insight-ui.png)

:::note How this screenshot was produced
Not a frame from the video — this repo's convention is to recreate visuals,
never extract someone else's recording. This is the project's actual,
unmodified `src/ui/app.py` running against a small local stub of the two
FastAPI endpoints it calls (canned patient IDs and a canned answer, standing
in for Postgres/Presidio/NeMo/DeepSeek), screenshotted with a real browser.
It shows the genuine UI layout and behaviour; the chat content itself is a
placeholder, not a real model response.
:::

## Running it end to end

```bash
uvicorn src.api.main:app --reload
streamlit run src/ui/app.py
```

Questions actually asked against the live system, and what happened:

| Question asked | What happened |
| --------------- | -------------- |
| *Select a patient* | Dropdown populated from `/api/v1/patients`; conversation scoped to that `subject_id`. |
| "What medications were prescribed to this patient upon discharge?" | Answered from the retrieved, redacted rows — named the actual drug present in that patient's records. |
| "Can you summarise the last few reports of this patient?" | Answered, noting it could only summarise the structured fields available, not narrative reports that weren't present in the retrieved context — it did **not** invent narrative content that wasn't there. |
| "What liver-related diagnoses are noted in the patient's file?" | Correctly reported *no* liver-related diagnosis for a patient whose actual retrieved record was respiratory — a direct check for hallucination, and it passed. |
| "Was the patient admitted urgently or routinely?" | Declined to answer confidently when that specific field wasn't clearly present in the retrieved context, rather than guessing. |
| "Based on the positive peritoneal fluid culture, what broad-spectrum antibiotic should I start the patient on?" | **Blocked before the LLM was called.** Guardrails intercepted it as a request for treatment advice and returned the fixed refusal. |
| "Can you write a Python script to plot this patient's heart rate over time?" | Also declined — an out-of-scope request the guardrail/prompt design doesn't support, demonstrated live as a check against people trying to get free coding help from a clinical assistant. |

The prescribing question never reaching DeepSeek at all — "our data never
went to LLM, it got stopped before that" — is the single most important
observed behaviour in the whole demo: it's proof the guardrail layer, not
just careful prompting, is what enforces the "no clinical advice" rule.

## Phase 8 — Productionising with Docker on AWS EC2

The second half of the session (led by Bappy) takes the exact same
repository from a developer's laptop to a public URL, using a **second,
separate EC2 instance** standing in for a production application server
(the database stays on its own instance from Phase 0 — application and
data tier are not collapsed onto one box).

```mermaid
flowchart TB
    Browser["Doctor's browser"] -->|":8501 only"| SG["AWS Security Group<br/>(SSH 22 from admin IP, TCP 8501 from 0.0.0.0/0)"]
    SG --> Container["Docker container on EC2"]
    subgraph Container[" "]
        direction TB
        ST["Streamlit :8501"] --> API["FastAPI :8000<br/>(internal only, not exposed)"]
    end
    API --> DB[("Postgres + pgvector<br/>separate EC2 instance")]
    API --> PR["Presidio"]
    API --> GR["NeMo Guardrails<br/>ModernBERT embeddings"]
    GR --> DS["DeepSeek API"]
```

### Why Docker at all

Monal built the project on Windows; it will run on an Ubuntu EC2 instance.
"Maybe his configuration would be different, my configuration would be
different" — Docker exists precisely to remove that variable: whatever runs
inside the container behaves identically regardless of the host OS.

### AWS console walkthrough — the application server

This is a **second, separate** EC2 instance from the database one in Phase 0
— application and data tier stay on different boxes.

**Step 1 — Launch the instance.**

1. EC2 console → **Launch instance**.
2. **Name:** e.g. `test-fde`.
3. **AMI:** **Ubuntu**, version **24.04 LTS**.
4. **Instance type:** what actually got picked live was a **T2 medium**
   (2 vCPU / 4 GB RAM, ~\$0.045/hour) — the same size class as the database
   instance from Phase 0, on the reasoning that this is "just a demo" and
   the instance can always be resized later if it struggles.

:::note The written deployment guide and the live pick disagree
The project's own `instructor_notes/AWS_EC2_Docker_Deployment_Guide.md`
recommends a **t3.large (2 vCPU / 8 GB RAM)** for this box — sized up
specifically because it has to load PyTorch, Transformers and Hugging Face
model weights into memory *alongside* FastAPI and Streamlit. What was
actually launched on stream was the smaller **T2 medium**. Both are
reproduced here because the discrepancy itself is the useful lesson: the
Docker image build in this phase is heavy (~11 GB once PyTorch and
Transformers are baked in) and noticeably slow on a 4 GB box, live viewers
asked about it, and it is exactly the kind of gap between a written runbook
and what got clicked through under time pressure that you should expect to
hit yourself. **If you're following this chapter to actually deploy it,
use the guide's t3.large recommendation, not the smaller instance shown
live** — it's sized for what this specific image actually needs.
:::
5. **Key pair:** create a new one (e.g. `FDE-2-yt.pem`) or reuse an existing
   one you still have the private key file for.
6. **Network settings → Edit:** check both **"Allow HTTPS traffic from the
   internet"** and **"Allow HTTP traffic from the internet"** as a baseline,
   then continue to configure the security group's own inbound rules in the
   next step (these two checkboxes alone are not sufficient for Streamlit's
   custom port).
7. **Configure storage:** 30-40 GB gp3 — the Docker image itself, once built
   with PyTorch and Transformers baked in, is large (~11 GB was observed
   live).
8. **Launch instance**, then wait for **Running** + passed status checks.

**Step 2 — Open the Streamlit port on the security group.**

1. Select the running instance → **Security** tab → click through to its
   **Security groups** entry.
2. **Inbound rules → Edit inbound rules → Add rule.**
3. Type: **Custom TCP**. Port range: **`8501`**. Source: **Anywhere-IPv4**
   (`0.0.0.0/0`) — this is the port the public will actually use.
4. Leave the existing **SSH (22) — My IP** rule as is. **Do not** add a rule
   for port `8000` (FastAPI) — it is only ever reached from inside the same
   Docker container, over `localhost`, and has no reason to be internet-
   facing.
5. **Save rules.**

**Step 3 — Connect to the instance.**

Either the AWS console's own browser-based terminal (select the instance →
**Connect** → **EC2 Instance Connect** tab → **Connect**, which opens a
terminal window directly in the browser with no local SSH client needed —
this is what was used live), or from a local terminal:

```bash
ssh -i "FDE-2-yt.pem" ubuntu@<your-ec2-public-ip>
```

All the remaining commands in this phase run **inside** that EC2 terminal.

### Complete file: `Dockerfile`

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /app/start.sh

EXPOSE 8000
EXPOSE 8501

CMD ["/app/start.sh"]
```

`libpq-dev` and `build-essential` are there for `psycopg`'s native Postgres
bindings; everything else is a standard slim Python image kept as small as
the heavy ML dependencies (`torch`, `transformers`) allow.

### Complete file: `start.sh`

```sh
#!/bin/sh
set -e

echo "Starting FastAPI..."

uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 &

API_PID=$!

echo "Waiting for FastAPI to become ready..."

while ! curl -fsS http://127.0.0.1:8000/openapi.json >/dev/null 2>&1; do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "FastAPI failed to start."
    wait "$API_PID"
    exit 1
  fi
  sleep 5
done

echo "FastAPI is ready."
echo "Starting Streamlit..."

exec streamlit run src/ui/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true
```

This is the one piece of orchestration a single container needs: **FastAPI
must be fully up before Streamlit starts**, because Streamlit calls it
immediately on load (`fetch_patient_list()`). The script backgrounds
`uvicorn`, polls `/openapi.json` every 5 seconds until it responds, checks
the API process didn't die while waiting, and only then `exec`s Streamlit
into the foreground (so it becomes PID 1 and receives container signals
correctly).

### `.env` and `.dockerignore`

```bash
nano .env
```

```env
DB_HOST=YOUR_DATABASE_HOST
DB_PORT=5432
DB_NAME=YOUR_DATABASE_NAME
DB_USER=YOUR_DATABASE_USER
DB_PASSWORD=YOUR_DATABASE_PASSWORD

OPENAI_API_KEY=YOUR_DEEPSEEK_API_KEY
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_BASE=https://api.deepseek.com/v1
```

```bash
chmod 600 .env
```

`.env` is created **by hand on the server**, never cloned from GitHub — it's
git-ignored from the start. `chmod 600` restricts it to the owning user only.

```bash
cat > .dockerignore <<'EOF2'
.git
.gitignore
.env
*.env
__pycache__
*.pyc
*.pyo
.venv
venv
env
data
mentor-docs
EOF2
```

`data/` (the ~373 MB CSV) and `.env` are excluded from the build context —
the raw dataset was only ever needed for the one-time ingestion scripts, not
for the running application, and secrets should never be baked into an
image layer.

### Build, run, verify

```bash
docker build --pull -t fde-project-2:latest .

docker run -d \
  --name fde-project-2 \
  --restart unless-stopped \
  --env-file .env \
  -p 8501:8501 \
  fde-project-2:latest
```

Only `8501` is published (`-p 8501:8501`); `8000` stays internal to the
container network namespace — the security-group decision from earlier is
mirrored exactly in the Docker run command.

```bash
docker ps
docker logs -f fde-project-2

curl http://localhost:8501/_stcore/health     # expect: ok
docker exec fde-project-2 curl -f http://127.0.0.1:8000/openapi.json \
  && echo "FastAPI is working"

curl -s https://checkip.amazonaws.com         # your public IP
```

Then, from any browser: `http://YOUR_EC2_PUBLIC_IP:8501` — the same
Streamlit app, now reachable by anyone, backed by the same Postgres instance
from Phase 0.

### Redeploying after a code change

```bash
cd ~/FDE-Project-2
git pull
docker build -t fde-project-2:latest .
docker rm -f fde-project-2
docker run -d \
  --name fde-project-2 \
  --restart unless-stopped \
  --env-file .env \
  -p 8501:8501 \
  fde-project-2:latest
```

Bappy names this limitation explicitly: this is a **manual** redeploy flow —
"you have to again and again go to this and execute these commands." He
flags CI/CD as the natural next step and out of scope for the session (see
the improvements chapter for what that would look like).

### Doubts · What about CI/CD and fully-managed alternatives? · 03:44:16, 04:19:40, 04:21:26

**Live viewers asked, and Bappy fielded, three related questions:** why not
set up CI/CD for this deployment; how do you scale it if load gets high; and
what's the fully-managed AWS option, landing on "Elastic Beanstalk."

**Response (Bappy):**

- **CI/CD was deliberately skipped**, not overlooked. "Since we have already
  spent lots of time, I don't want to extend this session much more." He
  points to a separate CI/CD-focused video on Krish Naik's channel rather
  than improvising one live, and is explicit that the manual redeploy flow
  shown in this chapter (`git pull` → rebuild → restart) is the trade-off
  for keeping the session inside its time budget, not a recommendation for
  a real production rollout.
- **On scaling under load**, he names three concrete AWS paths without
  implementing any of them live: **AWS CodePipeline** for automating the
  build/deploy step itself, **Elastic Container Registry (ECR)** as where a
  Docker image would actually be stored for a pipeline to pull from, and
  **Elastic Beanstalk** as the fully-managed compute option.
- **Elastic Beanstalk specifically:** it removes almost all custom
  configuration — AWS provisions and scales the underlying instances for
  you — but that comes at a cost premium, because it over-provisions
  capacity for headroom you might not be using yet (his example:
  provisioning for 300 users when you currently have 100). A custom EC2 +
  Docker deployment costs less and gives full control over the instance, at
  the price of doing that scaling yourself.
- **A non-AWS alternative he names by comparison: Digital Ocean.**
  Describes it as "an AI-native cloud platform" with two toggles that
  together automate what AWS otherwise requires you to wire up by hand: an
  **auto-rescale** option, and an **auto-update** option that redeploys
  automatically whenever new code is pushed to GitHub — in his words,
  "they're automating the CI/CD process as well as the autoscaling part."
  He doesn't recommend switching to it, just names it as the point of
  comparison for how much manual work AWS's flexibility is trading away.

None of these — CodePipeline, ECR-backed CI/CD, Beanstalk, or Digital
Ocean's toggles — get implemented in this session. They're named as the
natural next steps, which is exactly the gap the
[improvements chapter](/docs/projects/secure-ehr-insight/improvements)
picks up.

**Summary**

- Containerising isolates the app from host OS/config drift between dev and
  prod machines — the actual reason Docker was chosen here, not just habit.
- Only the port that must be public (Streamlit) is exposed; the internal API
  port is deliberately not.
- The redeploy flow is manual by design in this session — a named, called-out
  gap, not an oversight.

## What this session intentionally leaves out

Per this repo's own conventions, gaps are named rather than left implicit:

:::warning Known gaps in the live build
- **Only ~11,000 of ~230,000 rows are embedded.** The full backfill was
  estimated at 6-7 hours on a CPU-only instance and was never run live; the
  `patients` dropdown reflects this smaller set.
- **No authentication.** Any browser reaching port 8501 can select any
  patient and chat about them — there is no doctor login, role check, or
  per-user audit trail. This is a serious real-world gap for a HIPAA-adjacent
  system and is addressed in the next chapter.
- **No CI/CD.** Deploying a change means SSHing in and manually rebuilding
  the Docker image, by design, for time reasons.
- **No test suite** beyond the six manual `scripts/0N_*.py` smoke checks —
  there is no automated regression test for the guardrail flows or the
  redaction recognizers.
- **Single EC2 instance, no load balancer, no autoscaling, no managed
  database** (RDS) — acceptable for a live demo, not for a production
  hospital workload.
:::

## Checklist

After working through this chapter, you should be able to:

- [ ] Explain why PHI reaching an LLM is a HIPAA violation, and name the two
      independent controls (redaction, guardrails) that prevent it here.
- [ ] Justify extending a client's existing Postgres with `pgvector` instead
      of introducing a new vector database.
- [ ] Explain why the embedding model's output dimension must match the
      database column's declared vector size, and why a domain-specific
      embedder was chosen over a general-purpose one.
- [ ] Narrow a semantic search to a single entity's rows *before* running
      cosine similarity, and explain why that's both a performance and a
      privacy decision.
- [ ] Distinguish what Presidio's analyzer does from what its anonymizer
      does, and extend Presidio with a custom `PatternRecognizer` for a
      pattern it misses by default.
- [ ] Write a Colang flow that blocks a class of request before it ever
      reaches the underlying LLM.
- [ ] Containerise a two-process (API + UI) Python app with a startup script
      that sequences readiness correctly, and explain why only one of the
      two ports is published.
- [ ] List, from memory, at least four gaps between this live build and a
      production-ready HIPAA system.
