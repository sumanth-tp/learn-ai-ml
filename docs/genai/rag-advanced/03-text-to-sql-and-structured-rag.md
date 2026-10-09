---
id: rag-adv-text-to-sql
title: "Text-to-SQL and Structured RAG"
sidebar_label: "3 · Text-to-SQL"
sidebar_position: 3
slug: /genai/rag-advanced/text-to-sql-and-structured-rag
description: "Answer questions about a database by linking the question to the right columns, generating SQL, running it through a read-only guard and scoring by execution, with a real small model and honest numbers."
tags: [text-to-sql, schema-linking, sqlite, structured-rag, execution-accuracy, rag]
---

import Infographic from '@site/src/components/Infographic';
import SchemaLinkingLab from '@site/src/components/viz/SchemaLinkingLab';

**In one line.** When the answer lives in a database table, do not retrieve text: show the model only the columns the question needs, let it write a query, check the query before it touches anything, run it read-only, and judge it by the rows it returns, not by how it is spelled.

:::tip Before you start
**You should already know**

- What a relational table, a `SELECT`, a `JOIN` and a `GROUP BY` are (any SQL tutorial; SQLite is used here).
- What an embedding and cosine similarity are ([neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking)).
- How a RAG pipeline is built ([RAG basics](/docs/genai/rag)).

**Reading time:** about 35 minutes, plus about 15 minutes to run the code (blocks 4 and 6 run a 1.7-billion-parameter model on CPU).

**After this chapter you can**

- Pick the columns that go into the prompt and measure how often that step loses one you needed.
- Build a guard that stops a model from changing, deleting or hanging your database.
- Score generated SQL by execution on two databases, and say why one database is not enough.
:::

:::note Not from a lecture
Written for this site from the sources under Go deeper. The database, the 12 questions and every number below are produced by the code in this chapter. The model is a small open one (SmolLM2-1.7B-Instruct), chosen so the code runs on a laptop; its mistakes are the point, not a flaw to hide.
:::

## In 30 seconds

You ask a colleague, "how many customers live in Lyon?" If they know the shop's database, they open the customers table, look at the city column and count. They do not read every table first. Text-to-SQL does the same: pick the one or two tables that matter, write the query, run it, read off the number. The dangers are in the details: picking the wrong column, writing a query that runs but answers a different question, or, worst, a query that deletes things. So between the model and the database you put a guard, and you grade the model by comparing the answers it gets with the answers you know are right.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Schema | The list of tables and their columns | `customers(customer_id, name, city, signup_date)` |
| Schema linking | Choosing which columns to show the model for this question | "Lyon" points to `customers.city` |
| Text-to-SQL | A model turning a question into a SQL query | `SELECT COUNT(*) FROM customers WHERE city = 'Lyon'` |
| Foreign key | A column that points at another table's id | `orders.customer_id` |
| Read-only | A connection that can read rows but never change them | A SQLite file opened with `mode=ro` |
| Authorizer | A callback that SQLite asks before it compiles each operation | Allows `SELECT`, refuses `DELETE` |
| Execution accuracy | A generated query is right if it returns the same rows as a known-good query | Same 8 customers either way |
| Gold query | The known-good query used for grading | Written by a person |
| Structured RAG | Retrieval-augmented generation whose source is tables, not text | A question answered from rows |

## The idea in plain words

Most company data is not in documents. It is in tables: orders, tickets, accounts, readings. A question like "which cities have a warehouse holding more than 700 pallets?" has an exact answer, and the right tool to get it is a query, not a search over paragraphs. Text-to-SQL asks a language model to write that query.

Four things have to go right, in order.

1. **Link the schema.** A real database has tens of tables and hundreds or thousands of columns. Showing them all costs tokens and confuses the model, so you show the columns the question needs.
2. **Generate.** The model writes SQL from the question and the columns it was shown.
3. **Guard.** Treat the model's output as untrusted input. Reject anything that is not a single read, run it through a connection that physically cannot write, and stop it if it runs too long.
4. **Grade by execution.** Two queries can look different and return the same rows, or look the same and return different rows. Compare rows, not text.

Spider (Yu and colleagues, 2018) and BIRD (Li and colleagues, 2023) are the two benchmarks that taught the field this. Spider has 10,181 questions over 200 databases in 138 domains. BIRD has 12,751 question and SQL pairs over 95 databases totalling 33.4 GB, and was built to look more like real data: messy values, large tables, questions that need outside knowledge. The paper behind Spider 2.0 (2024, ICLR 2025) went further, to 632 enterprise workflow tasks over databases with more than 1,000 columns, and reported that the model it tested, o1-preview, solved 21.3 percent of them against 91.2 percent on Spider 1.0 and 73.0 percent on BIRD. The gap is the lesson: toy schemas are easy.

<Infographic src="/img/rag-adv/text-to-sql-and-structured-rag-pipeline.svg" alt="A question flows through schema linking, the prompt, the model, a three-layer guard, execution and a comparison with the gold rows; failures loop back to the model as error messages" caption="Follow the arrows from the question on the left. The guard sits between the model and the database. The repair loop at the bottom feeds an error message back for one retry." />

## Worked example, step by step

The shop database has 40 customers whose cities cycle through Lyon, Porto, Graz, Leeds and Turin by id. Question: *How many customers live in Lyon?*

1. **Which columns matter?** The word "customers" points at the customers table and "Lyon" at a city. The column `customers.city` has the description "city where the customer lives". That is the only column the question needs.
2. **Show the model the columns.** The prompt lists the linked columns, for example `customers(customer_id, name, city, signup_date)`, then the question.
3. **The model writes a query.** For example: `SELECT COUNT(*) FROM customers WHERE city = 'Lyon'`.
4. **The guard checks it.** It starts with `SELECT`, has no second statement, and every operation in it is a read, so the authorizer allows it.
5. **Run it.** The cities cycle with the customer id, so ids 5, 10, 15, 20, 25, 30, 35 and 40 are Lyon: 8 customers. The result is `[(8,)]`.
6. **Grade it.** The gold query returns the same `[(8,)]`. Execution matches. The text of the two queries does not need to match: `COUNT(DISTINCT customer_id)` would also return 8.

A harder question shows where this breaks. "Names of customers who bought a toys product" needs four tables: `customers` to `orders` to `order_items` to `products`. If linking drops one join key, the model cannot write a correct query no matter how good it is. Block 2 measures how often that happens.

<Infographic src="/img/rag-adv/text-to-sql-and-structured-rag-worked-example.svg" alt="The question about Lyon traced through the linked column, the generated query, the guard check and the result of 8 rows matching the gold query" caption="Left to right, one question. The last step compares rows, not query text." />

## How it works

### Schema linking: show less, but not too little

Linking scores every column against the question and keeps the best ones. Here each column becomes a short text, either its name ("customers city") or its name plus a description ("customers city: city where the customer lives"), embedded with a small sentence model and compared with the question by cosine similarity. Three details decide quality.

- **Description text is not automatically better.** It adds words that the question may share with the wrong column. Block 2 measures this, and the answer is not what you would expect.
- **Join keys never match the question.** "Names of customers who bought a toys product" does not mention `order_id`. So linking often adds the key columns of every chosen table.
- **Missing a needed column is fatal, showing an extra one is cheap.** Measure recall first.

### Generation: ask for SQL only

A good prompt lists the tables compactly, states the dialect (SQLite here), asks for the query only, and decodes greedily so the same question gives the same query. The model's output must still be cleaned: strip code fences, cut at the first semicolon.

### The guard: three layers, because each one alone leaks

| Layer | What it does | What it misses |
| --- | --- | --- |
| Statement check | Accepts one `SELECT` or `WITH` only | A cleverly spelled write the pattern does not recognise |
| Read-only connection | Opens the database file with `mode=ro` so writes fail | Operations that are not writes, such as `ATTACH` |
| Authorizer | SQLite asks a callback before compiling every operation and the callback can refuse | Long-running reads; it does not limit time |
| Time limit | A progress handler aborts the query after a deadline | Nothing that finishes quickly |

SQLite's own documentation states that the authorizer callback is invoked at compile time, while a statement is being prepared, not while it runs, and that it exists to evaluate SQL from untrusted sources. Block 3 shows `ATTACH` slipping through the read-only file alone and being stopped by the authorizer.

On a production database the best guard is outside the code: a database account that has only `SELECT` rights on the tables you chose to expose, ideally on views that already hide sensitive columns.

### Grading by execution, and why one database is not enough

Execution accuracy runs the generated query and the gold query and compares the rows. It is better than comparing text, because many different queries are correct. But a wrong query can return the right rows by luck on one particular database. The distilled test-suite idea (Zhong, Yu and Klein, EMNLP 2020) checks each query against a small set of databases chosen to separate near-misses, and reports that the existing Spider metric had a 2.5 percent false-negative rate on average and 8.1 percent in the worst case. Block 1 builds a second, slightly changed copy of the database, and block 5 requires a query to match on both.

### Repair: feed the error back once

When the database rejects a query ("no such column"), you can append the error to the conversation and ask for a corrected query. That fixes mistakes the database can see. It cannot fix a query that runs and answers the wrong question, because nothing signals the error. Block 6 measures both.

### Structured RAG beyond SQL

Not every table question is a SQL question. If the answer is in a free-text column (support ticket bodies, product descriptions), retrieval over that column is the right tool and SQL only filters (by date, customer, status). A common design is to let the model write the filter as SQL and run vector search inside the filtered rows. If the rows are few, put them in the prompt. If users ask for aggregates ("revenue per category"), SQL is far better than any retrieval. Pick per question type, as in [GraphRAG](/docs/genai/rag-advanced/graphrag-and-knowledge-graphs).

## A real system that works this way

The three papers above are the evidence base for how the field measures this task. Spider (arXiv 1809.08887, submitted 24 September 2018, revised February 2019) reported that the best baseline then reached 12.4 percent exact-match accuracy on unseen databases. BIRD (arXiv 2305.03111, NeurIPS 2023) reported 40.08 percent execution accuracy for ChatGPT, the best model it tested, against 92.96 percent for human data engineers and students. On 7 October 2026 the BIRD leaderboard page showed a top test score of 82.95 percent (the entry GrainSQL, dated 7 September 2026), with the human figure unchanged at 92.96 percent. That is the best published number I could open, not a claim that any system reaches it on your data. Spider 2.0 (arXiv 2411.07763, ICLR 2025 oral) is the reminder that enterprise schemas are far harder than benchmark ones.

## Code you can run

Six blocks, run in order; they hand files to each other through your temporary folder. Block 1 builds a small shop database, block 2 links the schema, block 3 writes the guard as a small module the later blocks import, block 4 generates SQL with a real model, block 5 scores it by execution, and block 6 tries one repair retry. Libraries: Python 3.14, `sqlite3` from the standard library, `sentence-transformers` 6.1.0 (`all-MiniLM-L6-v2`), `transformers` 5.18.0 and `torch` 2.14.1. The model is `HuggingFaceTB/SmolLM2-1.7B-Instruct`, decoded greedily on CPU in 32-bit floats; with the model cached, run with `HF_HUB_OFFLINE=1`. Greedy decoding is repeatable on one machine, but another CPU may differ in a token or two, so expect your counts to be close, not necessarily identical.

### 1. The shop database

We create eight tables with 31 columns and some invented rows. Then we make a second copy in which seven shipped orders are changed to paid, which block 5 uses to catch lucky answers.

```python
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

SCHEMA = {
    "customers": {"customer_id": "unique id of the customer", "name": "customer full name", "city": "city where the customer lives", "signup_date": "date the customer registered"},
    "products": {"product_id": "unique id of the product", "title": "product name shown in the shop", "category": "product category such as books or toys", "unit_price": "price of one unit in euros"},
    "orders": {"order_id": "unique id of the order", "customer_id": "customer who placed the order", "ordered_on": "date the order was placed", "status": "order state: paid, shipped or cancelled"},
    "order_items": {"order_id": "order this line belongs to", "product_id": "product bought on this line", "quantity": "number of units bought"},
    "employees": {"employee_id": "unique id of the staff member", "full_name": "staff member full name", "department": "department the staff member works in", "salary": "yearly salary in euros"},
    "support_tickets": {"ticket_id": "unique id of the ticket", "customer_id": "customer who opened the ticket", "opened_on": "date the ticket was opened", "priority": "ticket urgency: low, medium or high"},
    "warehouses": {"warehouse_id": "unique id of the warehouse", "city": "city where the warehouse stands", "capacity": "number of pallets the warehouse holds"},
    "shipments": {"shipment_id": "unique id of the shipment", "order_id": "order being shipped", "warehouse_id": "warehouse it left from", "shipped_on": "date the parcel left", "carrier": "delivery company"},
}
tmp = Path(tempfile.gettempdir())
path = tmp / "ragadv_shop.db"
path.unlink(missing_ok=True)
db = sqlite3.connect(path)
for table, columns in SCHEMA.items():
    db.execute(f"CREATE TABLE {table} ({', '.join(columns)})")
rng = np.random.default_rng(7)
cities, cats = ["Lyon", "Porto", "Graz", "Leeds", "Turin"], ["books", "toys", "garden", "kitchen"]
for i in range(1, 41):
    db.execute("INSERT INTO customers VALUES (?,?,?,?)", (i, f"Customer {i}", cities[i % 5], f"2025-{1 + i % 12:02d}-{1 + i % 27:02d}"))
for i in range(1, 21):
    db.execute("INSERT INTO products VALUES (?,?,?,?)", (i, f"Product {i}", cats[i % 4], float(5 + (i * 7) % 40)))
for i in range(1, 121):
    day = f"2026-{1 + i % 9:02d}-{1 + i % 27:02d}"
    db.execute("INSERT INTO orders VALUES (?,?,?,?)", (i, int(rng.integers(1, 41)), day, ["paid", "shipped", "cancelled"][i % 3]))
    for _ in range(int(rng.integers(1, 4))):
        db.execute("INSERT INTO order_items VALUES (?,?,?)", (i, int(rng.integers(1, 21)), int(rng.integers(1, 5))))
    db.execute("INSERT INTO shipments VALUES (?,?,?,?,?)", (i, i, 1 + i % 3, day, ["DHL", "UPS", "GLS"][i % 3]))
for i in range(1, 9):
    db.execute("INSERT INTO employees VALUES (?,?,?,?)", (i, f"Staff {i}", ["sales", "support", "ops", "finance"][i % 4], 40000 + 3500 * i))
for i in range(1, 31):
    db.execute("INSERT INTO support_tickets VALUES (?,?,?,?)", (i, int(rng.integers(1, 41)), f"2026-{1 + i % 9:02d}-{3 + i % 25:02d}", ["low", "medium", "high"][i % 3]))
for i in range(1, 4):
    db.execute("INSERT INTO warehouses VALUES (?,?,?)", (i, cities[i], 500 * i))
db.commit()

variant = tmp / "ragadv_shop_variant.db"
shutil.copy(path, variant)
edit = sqlite3.connect(variant)
edit.execute("UPDATE orders SET status = 'paid' WHERE order_id IN (SELECT order_id FROM orders WHERE status = 'shipped' LIMIT 7)")
edit.commit()
(tmp / "ragadv_schema.json").write_text(json.dumps(SCHEMA))
columns = sum(len(c) for c in SCHEMA.values())
print(f"{len(SCHEMA)} tables, {columns} columns")
for table in SCHEMA:
    print(f"  {table:<16}{db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:>5} rows")
```

**Reading the output.** Eight tables, 31 columns. The biggest tables are `order_items` (234 rows) and `orders` and `shipments` (120 each). The sizes are small on purpose so every query returns instantly.

**Line by line.**

- `SCHEMA` keeps a description for every column; these are what linking will embed.
- The customer and product ids cycle through fixed values, which is why the Lyon count in the worked example can be done by hand.
- `variant` is the second database; the seven changed orders make "shipped" counts differ between the two copies.

### 2. Link the schema, and measure what it loses

For each of 12 questions we know which columns a correct query needs. We embed all 31 columns two ways, by name only and by name plus description, and measure recall: the share of needed columns that reach the prompt. "All found" counts questions where nothing needed was lost.

```python
import json
import tempfile
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

tmp = Path(tempfile.gettempdir())
SCHEMA = json.loads((tmp / "ragadv_schema.json").read_text())
COLUMNS = [(t, c) for t, cols in SCHEMA.items() for c in cols]
QUESTIONS = [
    ("How many customers live in Lyon?", ["customers.city"]),
    ("What is the total revenue per product category?", ["products.category", "products.unit_price", "order_items.quantity", "order_items.product_id", "products.product_id"]),
    ("List the cancelled orders placed in March 2026", ["orders.order_id", "orders.status", "orders.ordered_on"]),
    ("What is the average salary in each department?", ["employees.department", "employees.salary"]),
    ("How many parcels did GLS ship?", ["shipments.carrier"]),
    ("How many high priority tickets exist?", ["support_tickets.priority"]),
    ("Which cities have a warehouse holding more than 700 pallets?", ["warehouses.city", "warehouses.capacity"]),
    ("Names of customers who bought a toys product", ["customers.name", "customers.customer_id", "orders.customer_id", "orders.order_id", "order_items.order_id", "order_items.product_id", "products.product_id", "products.category"]),
    ("What is the most expensive product?", ["products.title", "products.unit_price"]),
    ("How many orders came from each customer city?", ["customers.city", "customers.customer_id", "orders.customer_id"]),
    ("How many orders were shipped?", ["orders.status"]),
    ("Delete all cancelled orders", ["orders.status"]),
]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
by_name = model.encode([f"{t} {c.replace('_', ' ')}" for t, c in COLUMNS], normalize_embeddings=True)
by_text = model.encode([f"{t} {c.replace('_', ' ')}: {SCHEMA[t][c]}" for t, c in COLUMNS], normalize_embeddings=True)
question_vecs = model.encode([q for q, _ in QUESTIONS], normalize_embeddings=True)
keys = {t: {f"{t}.{c}" for c in cols if c.endswith("_id")} for t, cols in SCHEMA.items()}


def link(matrix, qi, k, add_keys):
    top = {f"{COLUMNS[i][0]}.{COLUMNS[i][1]}" for i in np.argsort(-(matrix @ question_vecs[qi]))[:k]}
    if add_keys:
        for table in {c.split(".")[0] for c in top}:
            top |= keys[table]
    return top


print(f"{'schema shown to the model':<28}{'recall':>8}{'all found':>11}{'columns':>9}")
for label, matrix, k, add in (("top-5 by name", by_name, 5, False), ("top-5 by description", by_text, 5, False), ("top-8 by description", by_text, 8, False), ("top-5 + key columns", by_text, 5, True), ("top-8 + key columns", by_text, 8, True)):
    got = [link(matrix, i, k, add) for i in range(len(QUESTIONS))]
    recall = [len(g & set(need)) / len(need) for g, (_, need) in zip(got, QUESTIONS)]
    print(f"{label:<28}{np.mean(recall):8.3f}{sum(r == 1 for r in recall):>8}/12{np.mean([len(g) for g in got]):9.1f}")
linked = [sorted(link(by_text, i, 8, True)) for i in range(len(QUESTIONS))]
lost = [(i + 1, sorted(set(need) - set(linked[i]))) for i, (_, need) in enumerate(QUESTIONS) if set(need) - set(linked[i])]
print("questions that lost a needed column with top-8 + keys:", lost)
(tmp / "ragadv_linked.json").write_text(json.dumps({"questions": QUESTIONS, "linked": linked}))
```

**Reading the output.** Name-only linking at `k = 5` reaches mean recall 0.897 and loses nothing for 9 of 12 questions. Adding descriptions at the same `k` lowers recall to 0.761 and complete questions to 6 of 12. Descriptions bring in extra words that match unrelated columns ("customer who placed the order" lands near every question that mentions customers). Raising `k` to 8 repairs most of it (recall 0.931, 10 of 12). Adding key columns at `k = 8` gives recall 0.979 and 11 of 12, at the cost of 12.4 columns shown on average instead of 31. Question 8 still loses `orders.customer_id` and `orders.order_id`: the join path from customers to orders is not in the question's words.

**Line by line.**

- `link` returns the top `k` columns by cosine similarity, then optionally adds every `*_id` column of the tables it touched.
- `keys` holds those id columns per table.
- The last line saves the 8-plus-keys selection for block 4.

### 3. Build the guard

The guard is written to a small module, so blocks 5 and 6 can import it. Then we try writes against one protection layer at a time, and finally the full `run` function.

```python
import sys
import tempfile
from pathlib import Path

tmp = Path(tempfile.gettempdir())
GUARD = r'''
import re
import sqlite3
import time
from pathlib import Path
import tempfile

PATH = Path(tempfile.gettempdir()) / "ragadv_shop.db"
ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


def authorize(action, *_):
    return sqlite3.SQLITE_OK if action in ALLOWED else sqlite3.SQLITE_DENY


def run(sql, db=PATH, limit=100, seconds=0.5):
    if not re.match(r"^\s*(select|with)\b", sql, re.I) or ";" in sql.strip().rstrip(";"):
        return "blocked: not a single SELECT", None
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.set_authorizer(authorize)
    deadline = time.monotonic() + seconds
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
    try:
        return "ok", conn.execute(sql).fetchmany(limit)
    except sqlite3.Error as error:
        return f"rejected: {error}", None
    finally:
        conn.close()
'''
(tmp / "ragadv_guard.py").write_text(GUARD)
sys.path.insert(0, str(tmp))
import sqlite3

from ragadv_guard import PATH, authorize, run

print("each write attempt against one protection layer at a time")
for sql in ("DELETE FROM orders WHERE status = 'cancelled'", "DROP TABLE customers", "ATTACH ':memory:' AS scratch"):
    file_only = sqlite3.connect(f"file:{PATH}?mode=ro", uri=True)
    rule_only = sqlite3.connect(PATH)
    rule_only.set_authorizer(authorize)
    outcome = []
    for label, conn in (("read-only file", file_only), ("authorizer", rule_only)):
        try:
            conn.execute(sql)
            outcome.append(f"{label}: RAN")
        except sqlite3.Error as error:
            outcome.append(f"{label}: {str(error)[:30]}")
    print(f"  {sql[:34]:<36}", " | ".join(outcome))
print("orders still in the table:", sqlite3.connect(PATH).execute("SELECT COUNT(*) FROM orders").fetchone()[0])

for sql in ("SELECT COUNT(*) FROM orders", "DELETE FROM orders", "SELECT 1; DROP TABLE orders", "SELECT * FROM nothing_here", "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r"):
    verdict, rows = run(sql)
    print(f"{sql[:44]:<46}{verdict[:48]:<50}{rows if rows is None else rows[:1]}")
```

**Reading the output.** The read-only file stops `DELETE` and `DROP` ("attempt to write a readonly database") but `ATTACH` runs, because attaching a database is not a write to this file. The authorizer blocks all three. The orders table still has 120 rows. The full `run` function accepts a plain `SELECT`, blocks `DELETE` and the two-statement string at the statement check, reports an unknown table as a database error, and stops the endless recursive query with "interrupted" after half a second.

**Line by line.**

- `authorize` is called by SQLite while it compiles each statement. It allows reads and refuses everything else with `SQLITE_DENY`.
- `?mode=ro` with `uri=True` opens the file read-only.
- `set_progress_handler` calls the lambda every 10,000 virtual-machine steps; returning a non-zero value aborts the query.

### 4. Generate SQL with a real model

Now the model. For each of the 12 questions we generate SQL under three conditions: no schema at all, only the linked columns from block 2, and the full schema. This is the slow block: 36 generations on CPU.

```python
import json
import re
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tmp = Path(tempfile.gettempdir())
saved = json.loads((tmp / "ragadv_linked.json").read_text())
SCHEMA = json.loads((tmp / "ragadv_schema.json").read_text())
FULL = [f"{t}.{c}" for t, cols in SCHEMA.items() for c in cols]
FENCE = chr(96) * 3
name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()


def schema_text(columns):
    if columns is None:
        return "(the table and column names are not shown)"
    by_table = {}
    for column in columns:
        table, col = column.split(".")
        by_table.setdefault(table, []).append(col)
    return "\n".join(f"{t}({', '.join(cs)})" for t, cs in by_table.items())


def ask(question, columns, history=()):
    system = "You write one SQLite query. Reply with the SQL only, no explanation."
    user = f"Tables:\n{schema_text(columns)}\n\nQuestion: {question}\nSQL:"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}, *history]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=100, do_sample=False)
    text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return re.sub(FENCE + "(sql)?", "", text).strip().split(";")[0].strip()


if __name__ == "__main__":
    generated = {}
    for label in ("no schema", "linked schema", "full schema"):
        columns_for = {"no schema": lambda linked: None, "linked schema": lambda linked: linked, "full schema": lambda linked: FULL}[label]
        generated[label] = [ask(question, columns_for(linked)) for (question, _), linked in zip(saved["questions"], saved["linked"])]
        print(f"{label:<14} {len(generated[label])} queries generated")
    (tmp / "ragadv_generated.json").write_text(json.dumps(generated))
    print("question 8, linked schema:", generated["linked schema"][7])
    print("question 11, full schema: ", generated["full schema"][10])
```

**Reading the output.** Each condition generates 12 queries. Two are printed. Question 8 under the linked schema becomes `SELECT name FROM customers WHERE category = 'Toys'`. That is a failure: `category` lives in `products`, three joins away, and the model wrote it as if it were a customer column. Question 11 under the full schema is `SELECT COUNT(order_id) FROM orders WHERE status = 'shipped'`, which is correct. The block takes several minutes because 36 generations run on CPU.

**Line by line.**

- `schema_text` prints each table as `table(column, column)`, which is compact and unambiguous.
- `ask` builds a chat prompt with a system line demanding SQL only, decodes greedily with `do_sample=False`, strips code fences and cuts at the first semicolon.
- The `history` argument is unused here; block 6 uses it for the repair turn in its own copy.

### 5. Grade by execution, on two databases

Each generated query goes through the guard from block 3. The result must equal the gold query's rows on the shop database, and again on the variant database. The delete request has no gold query: the right behaviour is to refuse, so a blocked query counts as correct.

```python
import json
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

tmp = Path(tempfile.gettempdir())
sys.path.insert(0, str(tmp))
from ragadv_guard import PATH, run

VARIANT = tmp / "ragadv_shop_variant.db"
generated = json.loads((tmp / "ragadv_generated.json").read_text())
GOLD = [
    "SELECT COUNT(*) FROM customers WHERE city = 'Lyon'",
    "SELECT category, SUM(quantity * unit_price) FROM order_items JOIN products USING (product_id) GROUP BY category",
    "SELECT order_id FROM orders WHERE status = 'cancelled' AND ordered_on LIKE '2026-03%'",
    "SELECT department, AVG(salary) FROM employees GROUP BY department",
    "SELECT COUNT(*) FROM shipments WHERE carrier = 'GLS'",
    "SELECT COUNT(*) FROM support_tickets WHERE priority = 'high'",
    "SELECT city FROM warehouses WHERE capacity > 700",
    "SELECT DISTINCT name FROM customers JOIN orders USING (customer_id) JOIN order_items USING (order_id) JOIN products USING (product_id) WHERE category = 'toys'",
    "SELECT title FROM products ORDER BY unit_price DESC LIMIT 1",
    "SELECT city, COUNT(*) FROM orders JOIN customers USING (customer_id) GROUP BY city",
    "SELECT COUNT(*) FROM orders WHERE status = 'shipped'",
    None,
]
(tmp / "ragadv_gold.json").write_text(json.dumps(GOLD))


def score(sql, index):
    verdict, rows = run(sql)
    if GOLD[index] is None:
        return verdict, rows is None, rows is None
    same = rows is not None and Counter(rows) == Counter(run(GOLD[index])[1])
    other = run(sql, db=VARIANT)[1]
    both = same and other is not None and Counter(other) == Counter(run(GOLD[index], db=VARIANT)[1])
    return verdict, same, both


print(f"{'schema shown':<15}{'executes':>9}{'matches gold':>14}{'on both dbs':>13}{'delete blocked':>16}")
results = {}
for label, queries in generated.items():
    scored = [score(sql, i) for i, sql in enumerate(queries)]
    results[label] = [s[0] for s in scored]
    answerable = scored[:11]
    print(f"{label:<15}{sum(s[0] == 'ok' for s in answerable):>6}/11{sum(s[1] for s in answerable):>11}/11{sum(s[2] for s in answerable):>10}/11{'yes' if scored[11][1] else 'NO':>16}")
(tmp / "ragadv_results.json").write_text(json.dumps(results))
lucky = "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'"
print("\na wrong query for 'How many orders were shipped?':", lucky)
for label, db in (("shop database", PATH), ("variant database", VARIANT)):
    print(f"  {label:<17} gold {run(GOLD[10], db=db)[1]}, wrong query {run(lucky, db=db)[1]}")
print("\nlinked schema, question by question")
for i, sql in enumerate(generated["linked schema"]):
    verdict, same, both = score(sql, i)
    print(f"{i + 1:>2} {verdict[:34]:<36}{'match' if same else 'WRONG':<7}{'robust' if both else 'fragile':<8}{' '.join(sql.split())[:58]!r}")
```

**Reading the output.** Out of the 11 questions that have a gold query, the model with no schema runs 3 queries and gets 2 right. With the linked schema it runs 7 and gets 5 right. With the full 31-column schema it runs 9 and gets 7 right. The delete request is blocked in all three conditions.

The surprise is that the full schema beat the linked one. With only 31 columns the whole schema is small enough to read, and linking mainly removed columns the model needed (question 8 lost two join keys, as block 2 predicted). Linking pays when the schema is too large to show, not before.

The per-question table lists the failures. Four of the 11 linked-schema queries are rejected by the database (questions 2, 3, 8 and 10). Two more run but return wrong rows (questions 7 and 9), which no error message reveals.

The second database changed no verdict for the model's queries, because none of its wrong answers happened to be right on the first database. The last lines show what it is for. A deliberately wrong query that counts cancelled orders returns 40, the same as the correct count of shipped orders, because the data holds 40 of each. On the variant database the correct answer is 33 and the wrong query still says 40.

**Line by line.**

- `score` returns the guard verdict, whether the rows match on the main database, and whether they also match on the variant.
- `Counter(rows)` compares the result sets as bags, so row order does not matter.
- The per-question table is for the linked-schema condition only.

### 6. One repair retry

Last, the queries the database rejected get one retry with the error message appended. Queries that ran but returned wrong rows are not retried, because no error signals the problem.

```python
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tmp = Path(tempfile.gettempdir())
sys.path.insert(0, str(tmp))
from ragadv_guard import run

saved = json.loads((tmp / "ragadv_linked.json").read_text())
generated = json.loads((tmp / "ragadv_generated.json").read_text())
GOLD = json.loads((tmp / "ragadv_gold.json").read_text())
SCHEMA = json.loads((tmp / "ragadv_schema.json").read_text())
FULL = [f"{t}.{c}" for t, cols in SCHEMA.items() for c in cols]
FENCE = chr(96) * 3
name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()


def repair(question, columns, bad_sql, error):
    tables = {}
    for column in columns:
        tables.setdefault(column.split(".")[0], []).append(column.split(".")[1])
    schema = "\n".join(f"{t}({', '.join(cs)})" for t, cs in tables.items())
    messages = [
        {"role": "system", "content": "You write one SQLite query. Reply with the SQL only, no explanation."},
        {"role": "user", "content": f"Tables:\n{schema}\n\nQuestion: {question}\nSQL:"},
        {"role": "assistant", "content": bad_sql},
        {"role": "user", "content": f"SQLite rejected that query: {error}. Use only the tables and columns listed. Write a corrected query."},
    ]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=100, do_sample=False)
    text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return re.sub(FENCE + "(sql)?", "", text).strip().split(";")[0].strip()


def matches(sql, index):
    return GOLD[index] is not None and run(sql)[1] is not None and Counter(run(sql)[1]) == Counter(run(GOLD[index])[1])


print(f"{'schema shown':<15}{'rejected first':>15}{'fixed by one retry':>20}{'now match gold':>16}")
for label in ("linked schema", "full schema"):
    failed = [i for i, sql in enumerate(generated[label]) if run(sql)[0].startswith("rejected")]
    fixed = right = 0
    for i in failed:
        columns = saved["linked"][i] if label == "linked schema" else FULL
        new = repair(saved["questions"][i][0], columns, generated[label][i], run(generated[label][i])[0])
        fixed += run(new)[0] == "ok"
        right += matches(new, i)
    print(f"{label:<15}{len(failed):>15}{fixed:>20}{right:>16}")
```

**Reading the output.** With the linked schema, 4 queries were rejected and one retry fixed none of them, and with the full schema 2 were rejected and again none was fixed. This is an honest negative result for a 1.7-billion-parameter model.

Question 8 shows why. Told that `category` does not exist, the model replaced it with `title`, another column that does not exist on `customers`. Question 3 shows a second failure type: the error is "incomplete input" because the 100-token output limit cut the query off, so raising the limit, not a retry, is the fix. A larger model or a prompt with worked examples would likely do better; measure before assuming it

**Line by line.**

- A query counts as failed here only if the guard reports `rejected`, meaning SQLite raised an error.
- The repair prompt repeats the schema, the bad query and the error, and asks for a corrected query using only the listed tables and columns.
- `matches` repeats the execution comparison from block 5 for the new query.

### Try the linking step in the lab

The lab holds the 31 columns and the similarity scores for all 12 questions. The summary line recomputes the mean recall over all 12 for whatever setting you choose.

<SchemaLinkingLab />

**What each control does.**

- **question**: one of the 12 questions. The default is question 8, the four-table join.
- **column text**: embed only the column name, or the name plus its description.
- **k**: how many top-scoring columns to keep, 3 to 12.
- **add key columns**: also include every `*_id` column of the tables the top columns belong to.
- **show data**: the ranked table of all 31 columns with similarity, whether shown and whether needed.

**Try it yourself.**

1. Leave the defaults (question 8, description text, `k = 8`, key columns on). The summary reads mean recall 0.979 with 11 of 12 complete, as block 2 prints. The question itself shows two red bars, `orders.customer_id` and `orders.order_id`: needed, not shown.
2. Switch column text to "name only" and set `k = 5`, key columns off. The summary changes to recall 0.897 and 9 of 12. Now switch to description text at the same settings and recall falls to 0.761. The extra words in the descriptions pull in the wrong columns.
3. Pick question 1 ("How many customers live in Lyon?") and drop `k` to 3. Every setting finds `customers.city`. Pick question 2 (revenue per category), needing five columns, and watch how large `k` must be before nothing is missing.

<Infographic src="/img/rag-adv/text-to-sql-and-structured-rag-results.svg" alt="Bars for schema-linking recall under five settings and for the generated SQL's execution match under three schema conditions" caption="Left: how much of the needed schema reaches the prompt. Right: how many of 12 generated queries return the right rows. Counts come from blocks 2, 5 and 6." />

## Designing with it

- **Limit the surface.** Expose a handful of views, not the raw tables. A view can hide personal columns and pre-join the awkward tables, which also makes linking easier.
- **Describe columns in the words your users use.** Descriptions help only when they match the question's vocabulary. Test name-only against description text on your own questions, as block 2 does.
- **Link generously, then add keys.** Missing a needed column is fatal, an extra one is cheap.
- **Guard in layers and in the database.** Use a database account with `SELECT` only. Keep the code-level guard as a second wall. Set time and row limits.
- **Grade by execution on more than one database.** Keep a set of questions with gold queries, run it on every prompt or model change, and add a variant database to catch lucky matches.
- **Show the user the query and the rows.** A model that is right 7 times in 11 should not be trusted silently. Let people see and edit the SQL.
- **Log failures by type.** Rejected, wrong rows, refused. Rejected queries can be retried; wrong rows need better linking, examples or a larger model.

For the retrieval-over-text side of a mixed system see [contextual retrieval and reranking](/docs/genai/rag-advanced/contextual-retrieval-and-reranking), and for choosing between building and buying see [the enterprise document-QA design](/docs/senior/design-enterprise-document-qa).

## Where this stands in 2026

:::info Industry view
Benchmark progress has been fast: BIRD's leaderboard top rose from the 40 percent of the original paper to 82.95 percent on 7 October 2026, against 92.96 percent for people. Spider 2.0 shows that enterprise workflows with very large schemas stay much harder. I checked the papers and the leaderboard page, not independent evaluations, so I cannot say which system or model suits your database. What does not change: the schema step decides what the model can do, execution grading beats text matching, and a guard outside the model is not optional.
:::

## Common mistakes

- **Putting the whole schema in every prompt.** It is simple and it works on a toy database. On a real one it costs tokens, confuses the model and hits context limits. Link first, and measure recall.
- **Trusting that descriptions always help.** They feel like free information. Block 2 shows them lowering recall at the same `k`. Test both ways.
- **Guarding with one check.** A regex for `SELECT` feels sufficient. Block 3 shows a read-only file letting `ATTACH` through. Stack the layers and use a least-privilege account.
- **Grading by query text.** Two correct queries rarely match character for character, and two wrong ones can. Grade by rows.
- **Grading on one database.** A wrong query that happens to return the right rows passes. Add a variant database where the answers differ.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> Why is execution accuracy a better grade than comparing the query text with the gold query?</summary>

Many different queries are correct (`COUNT(*)` against `COUNT(DISTINCT customer_id)`), and a query can look close to the gold text yet return different rows. Execution accuracy compares what the user cares about: the rows returned.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> In block 3, which layer stopped <code>ATTACH</code>, and why did the read-only file not?</summary>

The authorizer stopped it. The read-only flag blocks writes to the opened file, and attaching another database is not a write to that file, so it ran.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> Description text lowered schema-linking recall at <code>k = 5</code>. Give a likely reason and a fix.</summary>

Descriptions add shared words ("customer", "order", "date") that match many columns, so unrelated columns outrank the needed ones. Fixes: raise `k`, add key columns, use descriptions written with the users' vocabulary, or re-rank the linked columns with a model.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> Why does block 5 require a match on a second, slightly different database?</summary>

A wrong query can return the right rows by luck on one database. If seven shipped orders become paid in a copy, "how many orders were shipped" gives a different answer there, and a query that filtered on the wrong status stops matching.

</details>

<details>
<summary><strong>Q5 (Medium).</strong> The repair loop in block 6 retries only rejected queries. What kind of failure does it leave untouched, and how would you catch it?</summary>

Queries that run and return wrong rows. The database sees no error to report. Catch them with execution grading against gold queries offline, and online with a display of the SQL and the rows plus user feedback.

</details>

<details>
<summary><strong>Q6 (Medium).</strong> Why is "delete all cancelled orders" a useful test question for a text-to-SQL system?</summary>

It shows what the system does with a request that should not be a query at all. A model will usually just write the `DELETE`. The guard must block it, and the right behaviour is to refuse or explain, which execution grading with no gold query checks.

</details>

<details>
<summary><strong>Q7 (Stretch).</strong> Your database has 900 columns. Linking with <code>k = 8</code> plus keys gives 97 percent recall on your test questions. Is it ready? What would you check next?</summary>

Not yet. Recall measures only whether needed columns reach the prompt, not whether the model then writes correct SQL. Check execution accuracy on questions with gold queries, the split between rejected and wrong-row failures, performance on multi-table joins, behaviour on questions the schema cannot answer, and how many tokens the linked schema costs.

</details>

## Go deeper

All sources opened on 7 October 2026.

- Yu et al., [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887)
- Li et al., [Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs](https://arxiv.org/abs/2305.03111), arXiv 2305.03111, NeurIPS 2023.
- Lei et al., [Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows](https://arxiv.org/abs/2411.07763), arXiv 2411.07763, ICLR 2025.
- Zhong, Yu and Klein, [Semantic Evaluation for Text-to-SQL with Distilled Test Suites](https://arxiv.org/abs/2010.02840), arXiv 2010.02840, EMNLP 2020.
- SQLite documentation, [`sqlite3_set_authorizer`](https://www.sqlite.org/c3ref/set_authorizer.html).
- BIRD benchmark leaderboard page (the benchmark's own site), read 7 October 2026: top test execution accuracy 82.95 percent, human 92.96 percent.
- [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) model card.
- On this site: [RAG basics](/docs/genai/rag), [neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking), [the enterprise document-QA design](/docs/senior/design-enterprise-document-qa).

## Check yourself

- I can explain the four steps of text-to-SQL: link, generate, guard, grade.
- I can measure schema-linking recall and say what a missing key column does to a join.
- I can explain why a read-only file alone is not a safe guard, and name the layers I would stack.
- I can grade generated SQL by execution on two databases and say what the second one catches.
- I can separate a rejected query from a wrong-rows query and say which one a retry can fix.
- I can say when a table question should be answered by retrieval over a text column instead of SQL.

## Where to go next

Next: [contextual retrieval and reranking](/docs/genai/rag-advanced/contextual-retrieval-and-reranking). For the previous chapter, see [long context versus RAG](/docs/genai/rag-advanced/long-context-vs-rag).
