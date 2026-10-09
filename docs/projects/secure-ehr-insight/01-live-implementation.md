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

import Infographic from '@site/src/components/Infographic';

> **Live session** ·
> [Watch on YouTube](https://www.youtube.com/watch?v=-4BrD23fQvU) · 4 hours 42
> minutes ·
> [Project source (GitHub)](https://github.com/nimowhyca/Secure-EHR-Insight-Clinical-Validator)
>
> Taught live by two mentors: **Monal** (problem framing, data, retrieval, PII
> redaction, guardrails, API and UI) and **Bappy** (Docker and AWS EC2
> deployment), hosted by Krish Naik. Notes follow the session in order.
>
> Diagrams captioned *Redrawn from…* recreate the instructors' whiteboards,
> with timestamps. Monal's originals ship in the repository as
> `instructor_notes/monal-handwritten-notes.pdf`.

Build an assistant that lets a doctor ask questions about a patient's history
in plain language, without ever letting raw patient data reach a language
model. The project is small on purpose. The point is not a complex agent; it
is a **privacy-first pipeline** that an AI forward deployed engineer (FDE)
would actually be allowed to ship into a hospital.

## Problem statement

A hospital's electronic health record (EHR) system accumulates millions of
rows of admissions, prescriptions and lab results over time. A doctor who
wants to know "what was this patient's last recorded dosage?" currently has
to scroll through that history by hand. The obvious pitch is "use an LLM to
answer questions over the records in natural language", and that pitch, on
its own, is a **HIPAA violation waiting to happen**.

The session opens by pressure-testing that obvious answer:

- **Naive answer:** RAG, or text-to-SQL, over the hospital's database.
- **Why it fails immediately:** the raw rows contain **PHI** (protected
  health information): names, addresses, phone numbers, social security
  numbers, financial and medical details. Sending that to a third-party LLM
  API is exactly what HIPAA exists to prevent. If the data leaks, the
  hospital faces a legal hearing, not a bug ticket.

<Infographic
  src="/img/secure-ehr/ehr-problem.svg"
  alt="Hospital EHR data, the privacy risk of sending raw records to an LLM, and the privacy-first pipeline."
  caption="Redrawn from Monal's whiteboard (page 1 of the handwritten notes)."
/>

:::note What HIPAA actually requires
HIPAA (the US Health Insurance Portability and Accountability Act) protects
PHI: any demographic or clinical detail that could identify a patient, such
as a name, address, phone number, social security number, medical record or
photo. It applies to healthcare providers, insurers and clearing houses, and
to their business associates: EHR platforms, IT vendors and, relevant here,
whoever builds the AI layer on top. Monal's point is that these companies
care first about whether a solution is compliant, and only then about what
it does.
:::

So the brief is not "build a RAG chatbot." It is: build a chatbot that
**never lets identifying information reach the LLM**, and that **refuses to
let the LLM behave like a clinician**. Those two constraints, redaction and
guardrails, are where HIPAA compliance actually lives in this system.

### Doubts · Is this only an AI problem?

**Monal (to the class):** "If hospital shares data to an LLM, that is a
violation of HIPAA. Very, very serious." He deliberately doesn't let the
class settle on "RAG + vector DB + LLM" as a complete answer.

**Response:** An AI engineer thinks "how do I solve it with the model." An
FDE has to also think about **who the client is** (a regulated hospital),
**where it will be deployed** (their existing Postgres, not a new vendor's
stack), and **what happens before the model ever sees the data**. The
security and redaction layer is not a bonus feature bolted on afterwards; it
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
| Product manager writes tickets         | **You** write the plan, the steps, and the edge cases up front, before asking AI  |
| Developer writes the code              | AI drafts boilerplate + implementation; you review, run it, and validate the result |
| QA writes and runs test cases          | You ask AI for unit tests, run them, and keep looping until satisfied              |
| Code review, checking for secrets      | You ask AI for a PR-style self-review: secrets, security flaws, regressions        |
| Maintenance and hot fixes              | You keep a documented plan/log so a future AI session has full context for fixes   |

<Infographic
  src="/img/secure-ehr/ehr-ai-sdlc.svg"
  alt="Classic SDLC beside the AI-assisted planning, coding, testing, review and maintenance process."
  caption="Redrawn from Monal's whiteboard (page 2)."
/>

:::note Not "vibe coding"
Monal is explicit that AI SDLC "does not mean asking ChatGPT to create a
project." Someone with no SDLC background cannot do a good version of it:
you still have to know what the edge cases are, what the acceptance criteria
are, and what "done" looks like. AI changes *who* writes the first draft of
each artifact, not *whether* the artifacts (plan, tests, review, docs) exist.
:::

## Solution architecture

This is the flow Monal draws on screen ("In-depth overview";
page 3 of the handwritten notes), redrawn here. Every phase in this chapter
builds one part of it.

<Infographic
  src="/img/secure-ehr/ehr-overview.svg"
  alt="The full EHR architecture: patient-scoped retrieval, Presidio redaction, NeMo guardrails, LLM and Streamlit."
  caption="Redrawn from Monal's in-depth overview (page 3)."
/>

He labels redaction and the guardrail together as the two places HIPAA
compliance lives in this system.

The two design decisions worth naming explicitly, because they are the
difference between a toy demo and something a hospital's infra team would
accept:

1. **The vector store is the hospital's own Postgres, not a new vendor.**
   Postgres added a `pgvector` extension that supports storing embeddings and
   running cosine similarity **inside SQL**. Given that, introducing Pinecone,
   Qdrant or any other dedicated vector database would mean asking the client
   to trust and pay for an entirely new system, for a capability their
   existing database already has a plugin for. When a client already has
   infrastructure that solves the problem, extend it rather than replace it.
2. **The patient is selected before the question is asked.** A hospital
   table can hold tens of millions of rows. Running a semantic search across
   all of it, per query, does not scale and is not necessary. A doctor never
   asks about "the database," they ask about *their* patient. Narrowing to one
   `subject_id` first turns "search 25 million rows" into "search ~100 rows,"
   which is both faster and a second, independent privacy boundary: a
   mistaken query can't accidentally surface a different patient's data.

### Doubts · Why not just use RAG/LangChain/CrewAI/an agent framework?

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
  `{role, content}` messages, replayed on every call; as Monal puts it, "that is how memory
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
| LLM access            | A DeepSeek API key (cheap: the whole session cost under \$0.10 in tokens) |
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
pipeline) has to be installed alongside it. It's the one dependency
pinned by URL rather than by name.

## Repository structure

The repository is built up folder by folder over the session. This is what
it holds at the end:

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
│   ├── instruction.txt               # every command run live, in order
│   ├── AWS_EC2_Docker_Deployment_Guide.md
│   └── monal-handwritten-notes.pdf
├── uv_instructions.txt               # how the Python environment was created
├── requirements.txt
├── readme.md                         # empty
└── .gitignore                        # contains only `.env`
```

Three more files appear only on the deployment server in Phase 8:
`Dockerfile`, `start.sh` and `.dockerignore`. Bappy writes them there with
shell heredocs and says he hasn't pushed them yet, so they are not in the
repository. The `.env` file is never committed.

`scripts/` holds one-shot tools. You run each once to set up state, and the
running application never imports them. `src/` is the application itself:
three layers you can test on their own (database, redaction, guardrails),
joined by an `api/` and a `ui/`.

## Phase 0 — Give the client their data (EC2 + Postgres)

Every real engagement starts with data that already lives somewhere. There is
no client here, so the session manufactures one: a Postgres database on a
plain EC2 instance, standing in for "the hospital's existing EHR database."
Monal calls this phase zero because, on a real engagement, the client hands
you this part.

Phase 0 runs in this order: put the project under version control, look at
the dataset, build the database server on AWS, create the application's
database user, point the project at it with a `.env` file, set up Python,
and load the CSV.

<Infographic
  src="/img/secure-ehr/ehr-phase0.svg"
  alt="AWS security group, EC2, Elastic IP, PostgreSQL setup and local Python data ingestion."
  caption="Redrawn from Monal's Phase 0 board (page 4)."
/>

### Set up version control before touching AWS

The first thing done on screen, before AWS is even opened, is creating the
GitHub repository and committing the empty project. Every phase that follows
can then be pushed as it's built:

```bash
git init
# create/edit README.md by hand first
git add .
git commit -m "adding readme"
git branch -M main
git remote add origin https://github.com/<you>/Secure-EHR-Insight-Clinical-Validator.git
git push -u origin main
```

The repository is **public**, with no licence. Every later phase ends the
same way: `git add .`, a short commit message naming what was just finished
("phase 0 1 completed", "data source added", "embeddings storage script
added"), then `git push`. That is why the repository's history follows the
same order as this chapter.

### The dataset

The session uses **MIMIC-IV**, a public, de-identified electronic health
record dataset, in the cleaned CSV version published on
[Kaggle](https://www.kaggle.com/datasets/isaacritharson/mimic-iv-cleaned-medical-transcripts).
The file is about 373 MB and has roughly 232,000 rows. Each row is one
clinical event, so the same patient appears on many rows.

| Column                                  | What it holds                                                                      |
| --------------------------------------- | ---------------------------------------------------------------------------------- |
| `subject_id`                            | The patient. A synthetic ID, already anonymised, repeated across that patient's rows |
| `hadm_id`                               | One hospital admission                                                             |
| `admission_type`, `admission_location`  | For example `URGENT`, `TRANSFER FROM HOSPITAL`                                     |
| `drug`, `dose_val_rx`, `dose_unit_rx`, `route` | What was prescribed, and how it was given                                   |
| `test_name`, `comments`                 | A lab test and the free-text note about it, for example "No VRE isolated."         |
| `description`, `drg_severity`           | The discharge diagnosis, for example "OTHER DISORDERS OF THE LIVER", and its severity |

Monal singles out the free-text columns, `comments` and `description`, as
the ones a doctor's questions will hit. Phase 2 turns them into embeddings.
Nothing is filtered out at this stage; retrieval decides later which rows
matter for a given question.

:::note Why use data that is already anonymised?
If MIMIC-IV is already de-identified, why build a redaction pipeline around
it? Because no company can publish real patient data, so any public dataset
is already clean. The point of the project is to build the pipeline a real
hospital dataset would need, and rehearse it on safe data.
:::

### AWS console walkthrough — security group, EC2 instance, Elastic IP

Now build the stand-in for the hospital's database server. Do the three
console tasks in this order: **security group, then EC2 instance, then
Elastic IP.** Monal created the instance first live, and the IP he had
written down went stale before he used it. The troubleshooting section after
the ingestion script shows what that cost.

**Step 1 — Create the security group.**

A security group is the instance's firewall: it lists who may connect, and
on which ports.

1. Console search bar → type **"security groups"** → open **Security Groups**
   under EC2.
2. **Create security group.**
3. Name it something identifiable, e.g. `ytfd-clinical-database-server`.
4. Under **Inbound rules**, click **Add rule** twice:
   - Rule 1: type **SSH**, source **My IP** (the console auto-fills your
     current public IP; this restricts SSH to only your machine).
   - Rule 2: type **PostgreSQL** (auto-populates port `5432`), source
     **My IP**.
5. **Create security group.** Note its name; you'll pick
   it from a dropdown in the next step.

**Step 2 — Launch the EC2 instance, attaching that security group.**

1. EC2 console → **Instances** → **Launch instance**.
2. **Name:** something identifiable, e.g. `test-fde-database`.
3. **Application and OS Images (AMI):** select **Ubuntu**, then the
   **24.04 LTS** version from the dropdown (24.04 was chosen for familiarity;
   26.04 was also available and equally stable).
4. **Instance type:** select a small free-tier-eligible type. The session
   used a **2 vCPU / 4 GB RAM** class (shown as `c7i-flex` in the console),
   priced at roughly **\$0.08/hour** for Linux at the time of recording.
5. **Key pair:** click **Create new key pair** → name it (e.g. `FDE-database-pair`)
   → format **`.pem`** (for SSH from a terminal, not PuTTY's `.ppk`) → **Create
   key pair**. The browser downloads the `.pem` file immediately. Move it
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

**Step 3 — Give the instance a fixed IP (Elastic IP).**

Without this, the public IP can change whenever the instance restarts, and
you'd have to update `.env` every time.

1. EC2 console → left sidebar → **Network & Security → Elastic IPs**.
2. **Allocate Elastic IP address** → leave the network border group as your
   region → **Allocate**.
3. Select the newly allocated address → **Actions → Associate Elastic IP address**.
4. Under **Resource type**, choose **Instance**, then pick the instance
   created in Step 2 from the dropdown (identify it by its **private IP
   address**, shown alongside the instance name, if you have more than one
   running).
5. **Associate.** Refresh the **Instances** list: the instance's **Public
   IPv4 address** column now matches the Elastic IP exactly. This value is
   what goes into `DB_HOST` in your `.env` file, and it will survive a stop/
   start of the instance.

:::note Why this order specifically
An Elastic IP associates with an *existing* instance, so you cannot associate
one before the instance exists, and the instance's security group has to
already exist before you can attach it at launch time. Reversing steps 1 and
2 just means going back to edit the instance afterwards; reversing step 3
relative to 1-2 doesn't work at all.
:::

**Step 4 — Connect over SSH and install PostgreSQL 14.**

```bash
ssh -i "FDE-database-pair.pem" ubuntu@<your-elastic-ip>
```

On the first connection, accept the host-key prompt. The security group
already ensures only your IP can reach port 22, so this is the normal
first-time SSH warning. Once connected, add the PostgreSQL package
repository and install PostgreSQL 14:

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

**Step 5 — Let Postgres accept connections from outside the instance.**

By default Postgres only listens on `localhost`. Two config edits change
that. Open each file with `sudo nano <file>`, make the change, then save with
`Ctrl+X`, `Y`, `Enter`:

```text
# /etc/postgresql/14/main/postgresql.conf
listen_addresses = '*'

# /etc/postgresql/14/main/pg_hba.conf  (append)
host    all             all             0.0.0.0/0               scram-sha-256
```

`listen_addresses = '*'` and a permissive `pg_hba.conf` line are only safe
*because* the security group already restricts who can reach port 5432 at
all. The database is open only because the network in front of it isn't.

**Step 6 — Create the database and a dedicated application user.**

The Python code connects as its own user, not as the Postgres superuser.
Create the database, then open `psql` as the `postgres` system user:

```bash
sudo -u postgres psql -c "CREATE DATABASE ehr_db;"
sudo -i -u postgres psql
```

The first command may print a warning that it could not change directory.
That's harmless: the database is still created. Inside `psql`, create the
user and give it this database. Set a unique password through the interactive
`psql` prompt, which keeps it out of the command text and shell history:

```sql
CREATE USER fde_admin;
\password fde_admin
ALTER ROLE fde_admin SET client_encoding TO 'utf8';
ALTER ROLE fde_admin SET default_transaction_isolation TO 'read committed';
ALTER ROLE fde_admin SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE ehr_db TO fde_admin;

\c ehr_db
GRANT ALL ON SCHEMA public TO fde_admin;
\q
```

**Step 7 — Point the project at the database with a `.env` file.**

Back on your own machine, create `.env` in the project root. Monal starts
it with only `DB_HOST` right after the Elastic IP step, but every script
reads all five values:

```env
DB_HOST=<your-elastic-ip>
DB_PORT=5432
DB_NAME=ehr_db
DB_USER=fde_admin
DB_PASSWORD=<the password from step 6>
```

Keep `.env` out of Git. Monal notices it about to be committed at the end of
phase 1 and adds a one-line `.gitignore` containing just `.env`, which is
still the whole of the repository's `.gitignore`.

### Set up the Python environment

Monal manages Python with `uv`. He keeps one Python 3.12 environment outside
the project folder rather than a `.venv` inside it, a personal preference;
a normal project `.venv` works the same.

```bash
uv venv --python 3.12 <path-to-env>
# activate: <path-to-env>\Scripts\activate (Windows) or source <path-to-env>/bin/activate
uv pip install -r requirements.txt
uv pip install pandas psycopg2-binary python-dotenv sqlalchemy
```

He installed everything before the stream: PyTorch and the embedding model
take several minutes to download.

:::warning Install `psycopg2-binary` as well as `requirements.txt`
Scripts 01 to 05 connect with a plain `postgresql://` URL, which SQLAlchemy
serves with the `psycopg2` driver. `requirements.txt` only contains
`psycopg` version 3, which the API uses through `postgresql+psycopg://`. With
just `requirements.txt` installed, every setup script fails with
`ModuleNotFoundError: No module named 'psycopg2'`. It worked live because
the session's `instruction.txt` installs `psycopg2-binary` separately, as in
the last line above.
:::

### Load the CSV into Postgres

With the database reachable and Python ready, ingestion takes two files: a
table definition, and a script that creates the table and pushes the CSV
into it.

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
    
    print("✅ Baseline legacy data ingestion complete.")

if __name__ == "__main__":
    ingest_data()
```

Walking through it: the script resolves paths **relative to its own file
location**, not the current working directory, so it can be run from
anywhere. It runs `schema.sql` directly through the same connection (creating
the table if it doesn't exist), then loads the whole CSV into a
pandas DataFrame. `df.where(pd.notnull(df), None)` matters more than it looks.
Pandas represents missing values as `NaN`/`NaT`, and SQLAlchemy's parameter
binder doesn't know what to do with those; swapping them for Python `None`
lets it insert proper SQL `NULL`s instead of failing the whole batch.
`df.to_sql(..., if_exists='append')` then does the insert in one call.

Expect the insert to take several minutes. There is nothing slow in the
code: roughly 232,000 rows are travelling over the network to a small EC2
instance, and "the database itself takes time to insert all these rows."

### Debugging the first ingestion run

Running `python scripts/01_ingest_baseline_data.py` for the first time did
**not** work. The session keeps the failure on screen instead of cutting to
a working take, and it's worth following, because it mixes a configuration
mistake with a server problem, and each one hides the other:

1. **First error:** `ValueError: invalid literal for int() with base 10`.
   Monal's first check is not the code but the server: `sudo systemctl
   status postgresql` shows the service is **not running**, so he starts it.
   The error stays.
2. **The actual cause of that error was `.env`.** It still held only
   `DB_HOST`, and that IP was the one written down before the Elastic IP was
   attached, so it was stale. With `DB_PORT` unset, the script builds a URL
   containing `:None/`, and SQLAlchemy fails trying to read `None` as a port
   number. That is exactly the message above. Fixing the IP and adding the
   port, database name, user and password makes it go away.
3. **Second error:** `connection refused` on port 5432. It looks like a
   firewall problem, so the security group's inbound rules are re-checked.
   They are correct: SSH and PostgreSQL, both limited to "My IP".
4. **Real cause:** Postgres had got stuck again. One more `sudo systemctl
   restart postgresql`, a few seconds' wait, and the ingestion runs. In
   Monal's words, "it was not an issue from our side. We just needed to
   restart it because it was stuck."

:::warning If the scripts can't reach Postgres
From Python, all of these look like a generic connection error. Check them
in this order:

1. **Is `.env` complete?** All five `DB_` values must be set. A missing
   `DB_PORT` produces `invalid literal for int() with base 10: 'None'`.
2. **Is `DB_HOST` current?** Compare it with the Elastic IP in the EC2
   console, especially if the instance existed before the Elastic IP did.
3. **Is Postgres running?** Run `sudo systemctl status postgresql`, then
   `sudo systemctl restart postgresql`. Restart it even if it claims to be
   running when everything else checks out.

Only then suspect the security group or a mistyped password.
:::

### Complete file: `scripts/02_verify_ingestion.py`

```python
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def verify_ingestion():
    # Load environment variables
    load_dotenv()
    
    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)
    
    print("🔍 Verifying Data Integrity in AWS PostgreSQL...\n")
    
    with engine.connect() as conn:
        # 1. Check Total Row Count
        count_result = conn.execute(text("SELECT COUNT(*) FROM patient_encounters")).scalar()
        print(f"Total records found: {count_result}")
        
        if count_result == 0:
            print("⚠️ Warning: Table is empty. Ingestion may have failed.")
            return
            
        # 2. Retrieve a Sample of Critical Clinical Columns
        # We select specific columns so the terminal output is readable
        print("\nFetching sample clinical records...")
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
        
        # Use Pandas for clean terminal formatting
        sample_df = pd.read_sql(query, conn)
        
        print("-" * 65)
        print(sample_df.to_string(index=False))
        print("-" * 65)
        print("\n✅ Verification complete. Legacy database state is confirmed.")

if __name__ == "__main__":
    verify_ingestion()
```

A dedicated verification script, run right after ingestion, is the "test at
every checkpoint" habit the session insists on even under time pressure:
"you cannot just rely on AI to create things for you." A row count plus a
five-row sample is enough to catch a silently-empty table before three more
hours are spent building on top of it.

### Doubts · Why database and not table first?

**Monal (rhetorical, to the class):** "Why database and not table? Guys,
table exists inside database."

**Response:** A database is the container; you cannot create a table before
the database it lives in exists. The ordering (`CREATE DATABASE` → connect →
`CREATE TABLE` via `schema.sql`) isn't arbitrary, it's the only order
Postgres allows.

## Phase 1 — Turning Postgres into a vector store (pgvector)

With the rows in place, the AI work can start. Monal draws the line clearly:
phase 0 is what the client gives you; from phase 1 on, you are choosing the
solution. The first choice is this: **can the client's existing database do
what a dedicated vector database would?** Postgres can, once the `pgvector`
extension is installed, so the plan is to install it, add an embedding
column to the existing table, and fill it.

<Infographic
  src="/img/secure-ehr/ehr-phase1.svg"
  alt="A pgvector extension adds a clinical embedding column with 768 dimensions to the patient encounters table."
  caption="Redrawn from Monal's Phase 1 board (page 4)."
/>

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
    
    print("🚀 Connecting to AWS to apply pgvector schema upgrade...")
    
    try:
        with engine.begin() as conn:
            # Note: Extension was activated via DBA superuser. 
            # We only alter the application table here.
            print("Adding 'clinical_embedding' column (768 dimensions)...")
            conn.execute(text("ALTER TABLE patient_encounters ADD COLUMN IF NOT EXISTS clinical_embedding vector(768);"))
            
            # Verify the column was added
            verify = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'patient_encounters' AND column_name = 'clinical_embedding';
            """)).fetchone()
            
            if verify:
                print(f"✅ Schema upgrade complete. Confirmed column: {verify[0]} ({verify[1]})")
            else:
                print("❌ Column not found after alter attempt.")
            
    except Exception as e:
        print(f"❌ Error applying schema: {e}")

if __name__ == "__main__":
    apply_pgvector()
```

Two privilege levels are deliberately split here: activating the `vector`
extension needs Postgres superuser rights (done once, by hand, on the EC2
box), while altering the application table only needs the `fde_admin`
application user's ordinary privileges. The script also **verifies the
column by querying `information_schema.columns`** rather than trusting that
`ALTER TABLE ... IF NOT EXISTS` silently succeeded. It's the same "test at every
checkpoint" discipline as Phase 0.

### Doubts · Why 768 dimensions?

**Monal (to the class):** "Why 768? ... The LLM [embedding model] is going to
extract the embedding vector space of dimension 768. So I also want that to
be 768."

**Response:** The vector column's dimension isn't a free choice: it has to
exactly match whatever embedding model will populate it, decided in the next
phase. 768 is the output size of the domain-specific `bioclinical-modernbert`
sentence-transformer chosen for cost and domain-fit reasons (see below), not
a Postgres or pgvector default. Get the column size wrong relative to the
model and every insert fails.

## Phase 2 — Domain-specific clinical embeddings

The table has a `clinical_embedding` column; every value in it is still
`NULL`. Before writing the embedding script, the session asks: **which
embedding model?**

<Infographic
  src="/img/secure-ehr/ehr-phase2.svg"
  alt="Clinical text is embedded with a domain-aware model and fills the previously null vector column."
  caption="Redrawn from Monal's Phase 2 board (page 5)."
/>

### Doubts · Why not a general-purpose embedder?

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
  diagnoses, exactly the sensitive content this whole project exists to
  protect. Choosing an embedding provider is *also* a data-handling
  decision, not just an accuracy one.

The model chosen: **`NeuML/bioclinical-modernbert-base-embeddings`**, a
sentence-transformer run **locally on CPU**, so no embedding API call ever
leaves the machine. Its output dimension is 768, which is why the column was
sized that way. Monal tried another model first and switched to this one for
better results. OpenAI's embedding models, by comparison, return 1536 or
3072 dimensions and charge per call, which adds up across a quarter of a
million rows.

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
    print("⏳ Loading local BioClinical ModernBERT model...")
    model = SentenceTransformer('NeuML/bioclinical-modernbert-base-embeddings')
    
    # Guardrail: Programmatically verify the output dimension is 768
    if model.get_sentence_embedding_dimension() != 768:
        raise ValueError("CRITICAL DIMENSION MISMATCH: Model output does not match database vector(768).")

    # Increase batch size for network efficiency
    batch_size = 2
    # For a demo, 10,000 records is perfect to prove scale without waiting hours
    demo_limit = 100
    
    with engine.begin() as conn:
        print(f"🚀 Generating batched embeddings for up to {demo_limit} patient records...")
        
        with tqdm(total=demo_limit, desc="Vectorizing PHI", unit="rows") as pbar:
            processed = 0
            
            while processed < demo_limit:
                # Fetch a batch of records
                select_query = text("""
                    SELECT id, admission_type, drug, test_name, drg_severity, description, comments 
                    FROM patient_encounters 
                    WHERE clinical_embedding IS NULL 
                    LIMIT :batch_size
                """)
                batch = conn.execute(select_query, {"batch_size": batch_size}).mappings().fetchall()
                
                if not batch:
                    break # No more NULL records
                
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
                
                # 2. BATCHED ENCODING: Feed the entire list to the model at once!
                # This engages PyTorch's parallel processing.
                embeddings = model.encode(clinical_texts, batch_size=batch_size).tolist()
                
                # 3. Prepare BULK update parameters
                update_params = [
                    {"id": record_id, "embedding": str(emb)} 
                    for record_id, emb in zip(ids, embeddings)
                ]
                
                # 4. BULK UPDATE: execute all 250 rows in a single network round-trip
                update_query = text("""
                    UPDATE patient_encounters 
                    SET clinical_embedding = :embedding 
                    WHERE id = :id
                """)
                conn.execute(update_query, update_params)
                
                processed += len(batch)
                pbar.update(len(batch))

    print("\n✅ Phase 3 Complete: Clinical records are vectorized and ready for hybrid search.")

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
- **`WHERE clinical_embedding IS NULL`** is the resumability safeguard:
  rows that already have a vector are never re-processed, so the script can
  be re-run after a crash or a deliberate pause without redoing work.
- **The dimension check runs before any encoding happens.** If the model's
  output size doesn't match what the column expects, the script fails loud
  and immediately, instead of failing 50,000 rows into a silent mismatch.

:::note The comments and the values in this script disagree
The comments describe the script's original settings, a batch of 250 rows
and a cap of 10,000 ("execute all 250 rows in a single network round-trip").
The committed values are `batch_size = 2` and `demo_limit = 100`. Monal
lowered them live, first to 2 and 2, then to a limit of 10, so the class
could watch a run finish. For a real backfill, raise `batch_size` back to
around 250 and set `demo_limit` to the number of rows in the table.
:::

:::warning This does not scale to the full dataset live
Encoding and writing embeddings **one small batch at a time on a CPU-only
EC2 instance** is slow: two rows took roughly a second in the demo, and the
full ~230,000-row dataset was estimated at **6-7 hours** at that rate, far
too long to wait through live. Monal's actual workaround: a
*separate* EC2 instance, prepared **before** the stream, already had ~11,000
rows embedded; the session swaps its own `.env` to point at that instance's
IP and continues from there. The demo dataset for every later step is
therefore ~11,000 rows, not the full 230,000. Remember this when the
`patients` dropdown later looks smaller than expected.
:::

<Infographic
  src="/img/secure-ehr/ehr-recap.svg"
  alt="The database and embedding work so far, the prepared 11000-row demo, and cosine similarity search."
  caption="Redrawn from Monal's recap board (page 5)."
/>

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
    
    print("⏳ Loading local BioClinical ModernBERT model...")
    model = SentenceTransformer('NeuML/bioclinical-modernbert-base-embeddings')
    
    # 1. Define a complex, natural language medical query
    query_text = "Patient presenting with severe liver disease and fluid retention needing diuretics"
    print(f"\n🔍 Semantic Query: '{query_text}'")
    
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
        # Pass the vector as a string-formatted array
        results = conn.execute(search_sql, {"query_vector": str(query_vector)}).mappings().fetchall()
        
        print("\n🏆 Top 3 Semantic Matches:")
        for rank, row in enumerate(results, 1):
            print("-" * 65)
            print(f"Rank {rank} (Distance: {row['cosine_distance']:.4f})")
            print(f"Diagnosis : {row['description']}")
            print(f"Drug      : {row['drug']}")
            print(f"Record ID : {row['id']}")
            
if __name__ == "__main__":
    test_vector_search()
```

`<=>` is pgvector's **cosine distance** operator: smaller means more
similar, the opposite direction from cosine *similarity*. `ORDER BY
cosine_distance ASC LIMIT 3` is doing the same job FAISS or Pinecone would,
directly inside a SQL `ORDER BY`. `WHERE clinical_embedding IS NOT NULL`
matters at this stage because most of the ~230,000 rows have no vector yet.
Their distance would be `NULL`, and the filter keeps them out of the ranking
instead of relying on where Postgres happens to sort nulls.

Run live against the instance with ~11,000 embedded rows, the query
("severe liver disease and fluid retention needing diuretics") returned
three different patients, each with a liver-related diagnosis.

Note that this test script searches the *whole* table (minus the null
filter). The patient-scoping (`WHERE subject_id = :patient_id`) is added
one layer up, in the FastAPI endpoint, once a patient has actually been
selected. This script's job is only to prove the cosine-distance query
itself is correct.

### Doubts · Should search run on all 25 million rows?

**Monal (to the class):** "If this DB contains 25 million rows, do you think
this is a good approach to do similarity search on all 25 million rows of
data?"

**Response:** No, and the fix isn't a bigger index, it's a smaller search
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
        print("⏳ Initializing Microsoft Presidio Zero-Trust Middleware...")
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
            deny_list_score=1.0  # <--- Changed this from 'score'
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
    
    print("\n🚨 RAW PHI FROM DATABASE:")
    print(simulated_ehr_note.strip())
    
    print("\n🛡️ REDACTED OUTPUT (Safe for LLM Prompt):")
    safe_text = redactor.redact_clinical_context(simulated_ehr_note)
    print(safe_text.strip())
```

How it actually works, in the order the class was walked through it:

1. **`self.analyzer`** divides raw text into **intents**: spans tagged as
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
     organisation; it's not a name it has ever been trained to recognise.
     The fix is a **deny-list recognizer**: an explicit list of hospital
     names, tagged `ORGANIZATION` at `deny_list_score=1.0` (maximum
     confidence) whenever seen verbatim.
4. **`score_threshold=0.4`** is the cutoff applied when deciding whether a
   detected entity gets redacted. Custom recognizers are scored high (`0.9`,
   `1.0`) specifically so they always clear that bar.

<Infographic
  src="/img/secure-ehr/ehr-presidio.svg"
  alt="Presidio redaction in the request path, regional ID recognisers, and the analyzer and anonymizer engines."
  caption="Redrawn from Monal's Presidio board (page 6)."
/>

<Infographic
  src="/img/secure-ehr/ehr-scores.svg"
  alt="Entity confidence scores for the sample sentence, with SSN and person spans above the redaction threshold."
  caption="Redrawn from Monal's entity-score example (page 6), with the 0.45 cut-off from page 7."
/>

On the board the cut-off is written as 0.45; the committed code uses
`score_threshold=0.4`. Either way, the custom SSN pattern (0.9) and a
detected name (0.7) clear it, and filler words don't.

### Doubts · Why not just redact every number?

**Monal (to the class):** "Not every number in our data should be redacted
... let's suppose someone's data has a bacteria count of 10,000. Do you
think that number should also get redacted?"

**Response:** No. A lab value is not PHI; a social security number is.
Blanket redaction of anything numeric would destroy the clinical content the
whole system exists to answer questions about. This is also why the SSN
regex fix is deliberately narrow (`\d{3}-\d{2}-\d{4}` specifically) rather
than "redact all digit sequences," and why the session notes that in a real
multi-region deployment, region-specific identifier formats (SSN in the US,
Aadhaar in India) would need their **own** regex, applied only where
relevant. "Not every pattern can be recognized" out of the box, and you
should not rely on a system that silently fails to redact a format it has
never seen.

**Summary**

- Presidio's analyzer tags PII spans; its anonymizer removes them. They are
  two separate engines composed together, not one black box.
- Domain gaps in a general PII tool (hospital names, a checksum-strict SSN
  pattern) get closed with custom `PatternRecognizer`s, not by retraining
  the underlying model.
- Redaction is targeted at *identifying* information, not at anything that
  merely looks like structured data. Clinical values must survive.

## Phase 5 — Guardrails against clinical advice (NVIDIA NeMo)

Redaction solves *who this data is about*. It does nothing about *what the
doctor is allowed to ask the model to do*. A question like "should I
prescribe a higher dose?" carries no PHI at all, and still must never reach
the LLM. An AI system giving prescribing advice is a different, equally
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

This is a **Colang** file, NeMo Guardrails' own language for describing
conversations; `.co` is its extension. Nobody in the live chat had seen one
before. Reading it top to bottom, the way it was taught:

- **`define user ask for medical advice`** / **`define user ask about
  patient history`** each give the guardrail *example phrasings* of an
  intent. NeMo uses an LLM under the hood to classify an incoming message
  against these examples. The examples don't have to be exhaustive; they
  anchor a semantic match.
- **`define bot refuse medical advice`** is the fixed refusal text: an
  explicit, auditable, non-generated string, not something the LLM
  freestyles.
- **`define flow prevent medical advice`** wires the two together: *if* the
  message is classified as "ask for medical advice", *then* the bot replies
  with the refusal, and the flow ends there. The model is never asked to
  *answer* a blocked question. Anything that isn't refused carries on to
  the model as normal.

### Getting and paying for the DeepSeek API key

Guardrails still need an actual LLM behind them, to classify intents
against the Colang examples, and to generate the final answer once a
question is allowed through. That's a paid API call, and the session is
explicit about which provider and why:

<Infographic
  src="/img/secure-ehr/ehr-guardrails.svg"
  alt="Colang flows and the DeepSeek-backed guardrail accept or reject a request."
  caption="Redrawn from Monal's guardrails board (page 7)."
/>

1. **Provider: DeepSeek**, chosen for being "very cheap and very powerful."
   Monal topped up **\$2** total for this and a previous FDE session
   combined. By the time of recording, that covered roughly **100 API
   requests and ~418,000 tokens**, for a spend of **under \$0.10**.
2. **Why not a free tier instead** (Google Gemini was named specifically)?
   Free-tier LLM APIs are typically rate-limited more aggressively, and
   are more likely to throw an error under load. That's an acceptable risk for
   personal experimentation, but not something you can afford live, on
   stream, in front of an audience. Paying a small, known amount buys
   reliability.
3. **Create the key:** DeepSeek's platform console → **API keys** → create
   a new key (it starts with `sk-`) → copy it immediately, it's shown only
   once.
4. **Two different environment variable names for the same URL.** This
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
Guardrails' `engine: openai` at DeepSeek instead of OpenAI itself. NeMo
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

Two models, two jobs. `deepseek-chat` is the **main model**: it classifies
each message against the Colang intents and writes the final answer. The
**embeddings model** turns the incoming message into a vector, so NeMo can
find which example phrasings in `rails.co` it is closest to. Reusing the
Phase 2 clinical model keeps that matching in the same medical vocabulary as
the search.

### Complete file: `scripts/06_test_guardrails.py`

```python
import os
import asyncio
from nemoguardrails import RailsConfig, LLMRails
from dotenv import load_dotenv

async def test_guardrails():
    load_dotenv()
    
    print("⏳ Initializing NeMo Guardrails Firewall...")
    # Point the config loader to our guardrails directory
    config = RailsConfig.from_path("./src/guardrails")
    rails = LLMRails(config)
    
    # --- TEST 1: A Valid Retrieval Prompt ---
    valid_prompt = "What was the patient's last recorded dosage of Furosemide?"
    print(f"\n🟢 Valid Query: '{valid_prompt}'")
    res_valid = await rails.generate_async(messages=[{"role": "user", "content": valid_prompt}])
    print(f"🤖 LLM Response: {res_valid['content']}")
    
    # --- TEST 2: An Illegal Medical Advice Prompt ---
    illegal_prompt = "Based on the fluid retention, should I prescribe a higher dose of Furosemide?"
    print(f"\n🛑 Illegal Query: '{illegal_prompt}'")
    res_illegal = await rails.generate_async(messages=[{"role": "user", "content": illegal_prompt}])
    print(f"🛡️ Guardrail Intercept: {res_illegal['content']}")

if __name__ == "__main__":
    asyncio.run(test_guardrails())
```

Run live, both cases behaved as designed. The history question passed the
guardrail and DeepSeek answered it; this test sends no patient records, so
the answer asked for the patient's name and date of birth. The prescribing
question came back with the fixed refusal string.

On the DeepSeek dashboard, the request count went from 100 to 138 during
this test. Guardrails make several small model calls per message, first to
classify it and then to answer, so they cost requests more than tokens.

:::note A blocked question still reaches DeepSeek once
Later, in the demo, Monal says of the blocked question that "our data never
went to LLM, it got stopped before that." That holds for the *answer*, not
for the *check*. NeMo decides whether a message is medical advice by asking
the main model, DeepSeek, to classify it; Monal says as much when
introducing the rails ("guardrail uses LLM to decide whether something is
right or wrong"). And in `main.py` below, what goes through the guardrail is
the whole augmented prompt: the redacted clinical context plus the question.

So a blocked question sends the *redacted* context to DeepSeek once, for
classification, and DeepSeek never gets to answer it. Raw PHI still never
leaves, because redaction runs first. The whiteboard plan sent only the
question to the guardrail, in parallel with retrieval; the code sends the
full prompt. Classifying the bare question first, then retrieving only if
it's allowed, would match the plan.
:::

**Summary**

- Redaction and guardrails answer two different questions: *who is this
  about* versus *what is the model allowed to do*. HIPAA compliance in this
  system is the combination of both, not either alone.
- A guardrail flow stops the model from *answering*: the refusal is a fixed
  string, not generated text, so it is predictable and auditable.
- The check itself is a model call. Whatever you pass through the guardrail
  reaches the model provider, which is why redaction has to run first.

## Phase 6 — The FastAPI backend

Three components now exist independently: vector search, redaction,
guardrails. The API layer is where they compose into one request/response
cycle.

<Infographic
  src="/img/secure-ehr/ehr-request-apis.svg"
  alt="A query moves through retrieval, redaction, guardrails and generation; three APIs serve the demo, chat and patient list."
  caption="Redrawn from Monal's board (page 7): one request end to end, then the three APIs that serve it."
/>

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
    print("⏳ Booting Enterprise AI Middlewares...")
    
    # 1. Load Presidio (spaCy)
    middleware["redactor"] = ClinicalPIIRedactor()
    
    # 2. Load NeMo Guardrails (ModernBERT + DeepSeek)
    config = RailsConfig.from_path("./src/guardrails")
    middleware["rails"] = LLMRails(config)
    
    # 3. Load the local embedding model matching your embedding script
    print("⏳ Loading local BioClinical ModernBERT embedding model...")
    middleware["embedder"] = SentenceTransformer('NeuML/bioclinical-modernbert-base-embeddings')
    
    print("✅ System Ready on port 8000.")


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


# --- ENDPOINT 1: HEALTH CHECK (MOCK DB) ---
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
        
        # 4. Assemble Prompt
        augmented_prompt = f"Clinical Context:\n{safe_context}\n\nUser Question: {latest_question}"
        
        # 5. Format History for Guardrails
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

- **`POST /api/v1/clinical-query`** is a **trial endpoint**. The code
  comment calls it a health check; Monal calls it "a trial API". It builds a
  hard-coded mock database row (John Doe, a dosage, an attending physician)
  instead of querying Postgres, to prove the redact, guardrail and LLM steps
  work before real retrieval is wired in. The Streamlit UI never calls it.
- **`POST /api/v1/chat`** is the real endpoint. Note the order of
  operations: **embed → search (scoped to `subject_id`) → redact → guardrail
  → LLM.** Redaction happens *after* retrieval (there's no PHI to redact
  until rows come back) but strictly *before* anything is handed to
  guardrails or the model. `request.messages[:-1]` plus the freshly
  augmented latest message reconstructs conversation history with the
  *redacted*, context-augmented version of only the newest turn; earlier
  turns are passed through as already-sent.
- **`GET /api/v1/patients`** powers the dropdown. The `clinical_embedding IS
  NOT NULL` filter means only patients who've actually been vectorized show
  up, a direct consequence of the Phase 2 demo-limit decision:
  with only ~11,000 of ~230,000 rows embedded, most `subject_id`s will
  simply never appear in this list.

The `@app.on_event("startup")` hook loading Presidio, NeMo Guardrails and the
sentence-transformer model **once**, into a shared `middleware` dict, rather
than per-request, is what keeps latency reasonable: spaCy's pipeline and a
transformer model are too expensive to reload on every single API call.

### Doubts · Why pass `patient_id` in every chat payload?

**Monal (to the class):** "When I was testing this application, when I was
just passing the messages, AI in the next response was saying 'which patient
you're talking about'. It means it forgot the patient."

**Response:** Conversation memory here is just replayed message history; it
has no independent notion of "current patient" unless that fact is
re-supplied. Sending `patient_id` alongside `messages` on *every* call, not
just the first one, is the fix: an explicit piece of state the LLM would
otherwise have to (unreliably) re-infer from earlier turns.

## Phase 7 — The Streamlit frontend

### Complete file: `src/ui/app.py`

```python
import streamlit as st
import requests

API_CHAT_URL = "http://localhost:8000/api/v1/chat"
API_PATIENTS_URL = "http://localhost:8000/api/v1/patients"

st.set_page_config(page_title="Zero-Trust Clinical EHR", layout="centered")
st.title("🏥 Enterprise EHR Chat")
st.caption("Protected by Presidio Zero-Trust & NeMo Guardrails")

# --- FETCH PATIENTS DYNAMICALLY ---
@st.cache_data(ttl=300) # Cache the list for 5 minutes to reduce database load
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
st.sidebar.info(f"🔒 Database explicitly locked to Patient: {patient_id}")

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
            bot_reply += "\n\n*⚠️ AI generated summary. Do not use for diagnostic purposes.*"
            
        except requests.exceptions.RequestException as e:
            bot_reply = f"❌ API Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.chat_message("assistant").write(bot_reply)
```

Read top to bottom, this is deliberately small:

- **`@st.cache_data(ttl=300)`** caches the patients dropdown for five
  minutes, so a database round-trip doesn't happen on every Streamlit
  script rerun (Streamlit reruns the entire script on every interaction).
- **`st.session_state.messages`** is the same plain list-of-dicts memory
  discussed earlier, this time on the client side. It survives Streamlit
  reruns within one browser session but resets on a real page reload,
  because nothing persists it anywhere durable.
- **The disclaimer is appended in code, unconditionally**, on every
  successful response: "*AI generated summary. Do not use for diagnostic
  purposes.*" This isn't cosmetic: labelling AI-generated clinical content
  as such is treated in the session as part of HIPAA-adjacent duty of care,
  not just good UX.
- **`patient_id` rides along in every `payload`**, exactly matching the
  `/api/v1/chat` contract discussed above.

<Infographic
  src="/img/secure-ehr/ehr-ui-prompt.svg"
  alt="Streamlit patient selection and chat, with the newest question joined to clinical context before the guardrail."
  caption="Redrawn from Monal's UI and prompt board (page 8)."
/>

<Infographic
  src="/img/secure-ehr/ehr-memory.svg"
  alt="A plain workflow retains the conversation as a growing message list and sends the history to the LLM on each call."
  caption="Redrawn from Monal's workflow and memory boards (pages 8 and 9)."
/>

Rendered, this is what the UI actually looks like: a patient selected in
the sidebar, and one turn of the redacted, guardrail-checked answer with the
mandatory AI disclaimer:

![Secure EHR Insight Streamlit UI, showing a selected patient and a redacted, guarded answer with the AI-disclaimer](/img/secure-ehr-insight-ui.png)

:::note How this screenshot was produced
This is not a frame from the video. This repo recreates visuals rather
than extracting them from someone else's recording. This is the project's actual,
unmodified `src/ui/app.py` running against a small local stub of the two
FastAPI endpoints it calls (canned patient IDs and a canned answer, standing
in for Postgres/Presidio/NeMo/DeepSeek), screenshotted with a real browser.
It shows the genuine UI layout and behaviour; the chat content itself is a
placeholder, not a real model response.
:::

## Running it end to end

The API and the UI are two separate processes, a deliberate design choice:
if Streamlit fails, the API keeps running. Start them in two terminals, API
first, from the project root with the environment activated in both:

```bash
# terminal 1
uvicorn src.api.main:app --reload

# terminal 2, once terminal 1 prints "System Ready on port 8000."
streamlit run src/ui/app.py
```

:::warning "Database Connection Error" in the patient dropdown
Live, Streamlit was started before the API, and the dropdown showed
"Database Connection Error". The database was fine; the API wasn't running
yet. Starting the API alone doesn't clear the message.
`fetch_patient_list()` catches the failure and *returns* the error text as
a list, and `@st.cache_data(ttl=300)` caches that list for five minutes.
Restart Streamlit, as Monal did, or clear its cache from the app menu.
:::

These are the questions asked against the running system, and what
happened:

| Question asked | What happened |
| --------------- | -------------- |
| *Select a patient* | Dropdown populated from `/api/v1/patients`; conversation scoped to that `subject_id`. |
| "What medications were prescribed to this patient upon discharge?" | Answered from the retrieved, redacted rows, naming the actual drug present in that patient's records. |
| "Can you summarise the last few reports of this patient?" | Said the context held no narrative reports, only structured medication records, instead of inventing any. After the follow-up "whatever reports are present, please summarise those", it summarised the structured records. The follow-up only works because the earlier turns are sent as history. |
| "What liver-related diagnoses are noted in the patient's file?" | Correctly said there were none. This patient's retrieved record was a respiratory culture, so this was a direct hallucination check, and it passed. |
| "Was the patient admitted urgently or routinely?" | Replied "I don't know the answer to that." Monal guessed the data wasn't included, and the code confirms it: `admission_type` goes into the embedding text, but `/api/v1/chat` never selects it into the context, so the model can't see it. Adding it to the `SELECT` would fix this. |
| "Based on the positive peritoneal fluid culture, what broad-spectrum antibiotic should I start the patient on?" | **Refused with the fixed guardrail message.** The guardrail classified it as a request for treatment advice, and DeepSeek never answered it. |
| "Can you write a Python script to plot this patient's heart rate over time?" | Also declined, with "I don't know the answer to that", not the EHR refusal text. There is no coding rule in `rails.co`, so this wasn't the refusal flow. Monal asks it to check that nobody can use the clinical assistant as a free coding chatbot. |

The prescribing refusal is the key result of the demo. The rule against
clinical advice is enforced by the guardrail's fixed refusal, not by
careful prompting. As the note in Phase 5 explains, the question was still
*classified* by DeepSeek; it was never *answered*.

## Phase 8 — Productionising with Docker on AWS EC2

In the second half of the session, Bappy takes the same repository from a
developer's laptop to a public URL. He uses a **second, separate EC2
instance** as the application server, while the database stays on its own
instance from Phase 0, so the application and data tiers never share a box.

The plan he sketches is short: clone the project onto the server, build a
Docker image from it, and run that image as a container. CI/CD is
deliberately left out (see the Doubts section at the end of this phase).

<Infographic
  src="/img/secure-ehr/ehr-docker-plan.svg"
  alt="GitHub project to Docker image and container on AWS EC2, with CI/CD named but left out of the session."
  caption="Redrawn from Bappy's Excalidraw sketch."
/>

The result, as deployed:

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
different." Docker exists precisely to remove that variable: whatever runs
inside the container behaves identically regardless of the host OS.

### AWS console walkthrough — the application server

**Step 1 — Launch the instance.**

1. EC2 console → **Launch instance**.
2. **Name:** e.g. `test-fde`.
3. **AMI:** **Ubuntu**. Bappy takes the console's default, **26.04 LTS**.
4. **Instance type:** Bappy picks a **T2 medium** (2 vCPU, 4 GB RAM, about
   \$0.046/hour), reasoning that this is a demo and the instance can be
   resized later.
5. **Key pair:** create a new one (e.g. `FDE-2-yt.pem`) or reuse an existing
   one you still have the private key file for.
6. **Network settings → Edit:** check both **"Allow HTTPS traffic from the
   internet"** and **"Allow HTTP traffic from the internet"** as a baseline,
   then continue to configure the security group's own inbound rules in the
   next step (these two checkboxes alone are not sufficient for Streamlit's
   custom port).
7. **Configure storage:** 30 GB, which is what Bappy uses. The guide says
   30-40 GB gp3. The built image alone is about 11 GB, because PyTorch and
   Transformers are baked into it.
8. **Launch instance**, then wait for **Running** + passed status checks.

:::note The written guide and the live choices differ
The project's `instructor_notes/AWS_EC2_Docker_Deployment_Guide.md` asks for
**Ubuntu 24.04 LTS** on a **t3.large (2 vCPU, 8 GB RAM)**. Live, Bappy used
the 26.04 default on a **T2 medium (4 GB RAM)**. The Ubuntu version doesn't
matter here, but the memory does: this one box loads PyTorch, the
BioClinical ModernBERT model, spaCy's large English model, FastAPI and
Streamlit together. Monal notes at the end that the public demo became
unreachable when many viewers tried it at once, because "we only bought 4 GB
RAM". **If you're deploying this for real, follow the guide's t3.large.**
:::

**Step 2 — Open the Streamlit port on the security group.**

1. Select the running instance → **Security** tab → click through to its
   **Security groups** entry.
2. **Inbound rules → Edit inbound rules → Add rule.**
3. Type: **Custom TCP**. Port range: **`8501`**. Source: **Anywhere-IPv4**
   (`0.0.0.0/0`). This is the port the public will use.
4. Leave the existing **SSH (22), My IP** rule as it is. **Do not** add a rule
   for port `8000` (FastAPI). It is only ever reached from inside the same
   Docker container, over `localhost`, and has no reason to be internet-
   facing.
5. **Save rules.**

**Step 3 — Connect to the instance.**

Either the AWS console's own browser-based terminal (select the instance →
**Connect** → **EC2 Instance Connect** tab → **Connect**, which opens a
terminal window directly in the browser with no local SSH client needed;
this is what was used live), or from a local terminal:

```bash
ssh -i "FDE-2-yt.pem" ubuntu@<your-ec2-public-ip>
```

All the remaining commands in this phase run **inside** that EC2 terminal.

**Step 4 — Update Ubuntu and install Docker.**

A new instance has no Docker and stale packages, so update it first and
install a few utilities:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y ca-certificates curl git nano
```

Then install Docker from its official package repository. These are the
steps from Docker's own Ubuntu install guide, which is where Bappy took
them from:

```bash
# Docker's signing key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Docker's package repository
sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF2
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF2

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Start Docker, let your user run it without `sudo`, and check it works:

```bash
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker

docker --version
docker run --rm hello-world   # prints "Hello from Docker!"
```

**Step 5 — Clone the repository.**

Clone into your home directory, not whatever folder you happen to be in:

```bash
cd ~
git clone https://github.com/nimowhyca/Secure-EHR-Insight-Clinical-Validator.git FDE-Project-2
cd FDE-Project-2
ls   # data  instructor_notes  readme.md  requirements.txt  scripts  src ...
```

Live, Bappy cloned under the repository's default folder name. The guide
uses `FDE-Project-2`, which the redeploy commands below assume.

### Create the server-only files

Four files are created by hand on the server rather than cloned: `.env`,
because it holds secrets and is never in Git, and `.dockerignore`,
`start.sh` and `Dockerfile`. Bappy writes the last three with shell
heredocs (`cat > Dockerfile <<'EOF2' ... EOF2`) and says he hasn't pushed
them yet. Create them in this order, the one used live, because the
`Dockerfile` runs `start.sh`.

### `.env` and `.dockerignore`

Create `.env` with `nano .env`, paste the same values as your local one
(the database settings from Phase 0 and the DeepSeek key from Phase 5), and
save with `Ctrl+O`, `Enter`, `Ctrl+X`:

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

Then restrict it to your own user and check what was saved:

```bash
chmod 600 .env
cat .env
```

:::warning Let the application server reach the database
In Phase 0, port 5432 on the database's security group was limited to
**My IP**, your laptop. The application server is a different machine, so
with those rules alone the deployed API can't reach Postgres, and the
patient dropdown comes up empty. Add an inbound PostgreSQL (5432) rule to
the **database's** security group, in one of two ways:

- **Preferred:** in the server's `.env`, set `DB_HOST` to the database's
  **private** IP (both instances sit in the same default VPC), and allow
  5432 with the application server's **security group** as the source.
  This is what the guide's database networking note recommends for RDS.
- **Or** keep the public Elastic IP, and allow 5432 from the application
  server's public IP.

Don't mix the two. A security-group source only matches traffic arriving
on the private IP, so it has no effect while `DB_HOST` is the public
address. And don't open 5432 to `0.0.0.0/0`.
:::

`.dockerignore` works like `.gitignore` for the image: anything listed stays
out of the build context, which keeps the image smaller and the build
faster.

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

`data/` (the ~373 MB CSV) and `.env` are the important exclusions. The raw
dataset was only ever needed by the one-time ingestion scripts, not by the
running application, and secrets should never be baked into an image layer;
`docker run` passes them in with `--env-file` instead.

:::note `mentor-docs` doesn't exist in this repository
The guide's list excludes `mentor-docs`, but this project's notes folder is
`instructor_notes/`, which still ends up in the image. Bappy leaves the list
to you ("whatever folders and file you want to ignore, you can add"). Add
`instructor_notes` and `uv_instructions.txt` to it.
:::

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

Read top to bottom, it matches Bappy's walkthrough: start from the slim
Python 3.12 image, set a few environment flags, make `/app` the working
directory, install system packages, install the Python requirements, copy
the project in, and run `start.sh`. `build-essential` and `libpq-dev` are
there for packages that compile native code, such as Postgres drivers.

`requirements.txt` is copied and installed *before* the rest of the code.
Docker caches each step, so after a code-only change the slow dependency
layer is reused rather than reinstalled.

### Build, run, verify

From the project folder, build the image. `-t` tags it with a name, and the
final `.` means the `Dockerfile` is in the current directory:

```bash
docker build --pull -t fde-project-2:latest .
docker images
```

Expect the build to take a while. PyTorch, Transformers and the Hugging Face
libraries are large, and `docker images` listed the finished image at about
11 GB.

Then run it as a container. The first line removes any previous container
with the same name, a safe no-op the first time. `-d` runs it detached in
the background, and `--restart unless-stopped` brings it back after a
reboot:

```bash
docker rm -f fde-project-2 2>/dev/null || true

docker run -d \
  --name fde-project-2 \
  --restart unless-stopped \
  --env-file .env \
  -p 8501:8501 \
  fde-project-2:latest
```

Only `8501` is published (`-p 8501:8501`). Port `8000` stays inside the
container, mirroring the security-group rule from Step 2.

Check that the container is up and both services started. `docker logs -f`
follows the log; `Ctrl+C` leaves the log view without stopping the
container:

```bash
docker ps
docker logs -f fde-project-2

curl http://localhost:8501/_stcore/health     # expect: ok
docker exec fde-project-2 curl -f http://127.0.0.1:8000/openapi.json \
  && echo "FastAPI is working"

curl -s https://checkip.amazonaws.com         # your public IP
```

Then open `http://YOUR_EC2_PUBLIC_IP:8501` in any browser. It's the same
Streamlit app, now reachable by anyone, backed by the same Postgres instance
from Phase 0. Live, Bappy selected a patient, asked "What medications were
prescribed to this patient upon discharge?", and got an answer from the
public URL. You could point your own domain at this IP.

### Redeploying after a code change

To ship a change: pull the latest code, rebuild the image, and replace the
container.

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

Bappy calls this the weak point of a manual deployment: "we have to again
and again go to this [server] and execute these commands." CI/CD would
remove it.

### Doubts · What about CI/CD and fully-managed alternatives?

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
  configuration (AWS provisions and scales the underlying instances for
  you), but that comes at a cost premium, because it over-provisions
  capacity for headroom you might not be using yet (his example:
  provisioning for 300 users when you currently have 100). A custom EC2 +
  Docker deployment costs less and gives full control over the instance, at
  the price of doing that scaling yourself.
- **A non-AWS alternative he names by comparison: Digital Ocean.**
  Describes it as "an AI-native cloud platform" with two toggles that
  together automate what AWS otherwise requires you to wire up by hand: an
  **auto-rescale** option, and an **auto-update** option that redeploys
  automatically whenever new code is pushed to GitHub. In his words,
  "they're automating the CI/CD process as well as the autoscaling part."
  He doesn't recommend switching to it, just names it as the point of
  comparison for how much manual work AWS's flexibility is trading away.

None of these (CodePipeline, ECR-backed CI/CD, Beanstalk, or Digital
Ocean's toggles) gets implemented in this session. They're named as the
natural next steps, which is exactly the gap the
[improvements chapter](/docs/projects/secure-ehr-insight/improvements)
picks up.

**Summary**

- Containerising isolates the app from host OS/config drift between dev and
  prod machines. That's the actual reason Docker was chosen here, not just habit.
- Only the port that must be public (Streamlit) is exposed; the internal API
  port is deliberately not.
- The redeploy flow is manual by design in this session: a named, called-out
  gap, not an oversight.

## What this session intentionally leaves out

What the live build leaves out, so nothing is implied that isn't there:

:::warning Known gaps in the live build
- **Only ~11,000 of ~230,000 rows are embedded.** The full backfill was
  estimated at 6-7 hours on a CPU-only instance and never run live, so the
  patient dropdown shows only this smaller set.
- **No authentication.** Anyone who can reach port 8501 can select any
  patient and chat about them. There is no doctor login, no role check and
  no audit trail of who asked what, a serious gap for a HIPAA-adjacent
  system. The [improvements chapter](/docs/projects/secure-ehr-insight/improvements)
  starts there.
- **No CI/CD.** Deploying a change means logging in to the server and
  rebuilding the image by hand, chosen to save time.
- **No automated tests.** The six `scripts/0N_*.py` files are manual smoke
  checks. Nothing guards the guardrail flows or the redaction recognizers
  against regressions.
- **One small instance per tier.** No load balancer, no autoscaling and no
  managed database such as RDS. The 4 GB application server already became
  unreachable when many viewers tried it at once.
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
- [ ] Write a Colang flow that answers a class of request with a fixed
      refusal, and explain why the guardrail check still sends the prompt
      to the model provider.
- [ ] Set up the `.env` file and Python drivers the setup scripts need, and
      diagnose the three causes of a failed database connection in order.
- [ ] Containerise a two-process (API + UI) Python app with a startup script
      that sequences readiness correctly, and explain why only one of the
      two ports is published.
- [ ] List, from memory, at least four gaps between this live build and a
      production-ready HIPAA system.
