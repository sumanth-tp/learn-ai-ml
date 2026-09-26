# Practical AI interview labs

Tested on Python 3.12.8 on macOS ARM64, 26 September 2026. No API keys, paid services, network calls or GPU required to execute the labs. Installation downloads packages from your configured package index.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_labs.py
python retrieval.py
python attention.py
python async_workers.py
python safe_action.py
python feature_join.py
python graph.py
python numerical.py
python eval_gate.py
```

On Windows, activate with `.venv\Scripts\activate`. The lock records the tested environment, not a guarantee of compatibility on every platform. NumPy 2.5.3 requires Python 3.12 or newer. pandas and scikit-learn are included for the chapter's inline examples; the eight labs themselves use NumPy and the standard library.

## What to implement and verify

| File | Main exercise | Failure cases |
| --- | --- | --- |
| retrieval.py | Tenant filtering, revision selection, lexical retrieval, RRF and ranking metrics | Cross-tenant results, duplicate IDs, stale revisions, no evidence |
| attention.py | Q/K/V projections, head splitting, stable softmax and masks | Future leakage, padding, fully masked rows, bad shapes |
| async_workers.py | Bounded queue and worker pool | Timeout, malformed response, duplicate IDs, cancellation |
| safe_action.py | Atomic SQLite effect and durable receipt | Crash before/after commit, restart, conflicting payload |
| feature_join.py | Event-time and availability-time SQL join | Late/future features, missing entities, deterministic ties |
| graph.py | Deterministic topological ordering | Disconnected graph, duplicate edges, cycle, missing endpoint |
| numerical.py | Stable logistic loss/gradient and NCHW convolution | Extreme logits, finite-difference comparison, asymmetric kernel, stride/padding/channels |
| eval_gate.py | Complete paired comparison and bootstrap interval | Missing/duplicate/NaN scores, critical failure, regression, inconclusive result |

`python -m unittest -v test_labs.py` runs 32 tests. Tiny synthetic fixtures teach mechanisms; they do not establish production quality, statistical power or resistance to all attacks.

## Explicit limits

- Retrieval uses token overlap, not BM25 or embeddings, and returns evidence rather than a generated answer. Lexical overlap cannot prove answerability.
- Attention is a transparent NumPy reference, not a GPU training/serving kernel. Biases, rotary positions and KV caches are extension exercises.
- Async workers bound queued and active work. Collected results and seen IDs still grow with case count; persist them for large jobs. Blocking CPU work must leave the event loop.
- SQLite commits the simulated effect and receipt in one database transaction. An external API requires provider idempotency or reconciliation; a database row alone cannot make a remote action exactly once. Real abrupt process termination and multi-process contention are not simulated by these exception tests.
- The feature lab uses integer UTC epoch timestamps with a fixed unit. Production corrections and bitemporal history need explicit rules.
- Topological sort requires a DAG and comparable string node IDs. Heap ordering adds log V work; a FIFO Kahn implementation is O(V+E) when lexicographic order is unnecessary.
- Numerical routines are teaching references; their input validation is intentionally narrower than a general numerical library.
- The gate uses a percentile paired bootstrap on independent case deltas. Cluster correlated cases, plan sample size and predeclare margins for real releases. It checks candidate critical failures, not every product release condition.

`package-snapshot.json` contains observed package release metadata. Most listed frameworks were not installed or tested together. `requirements.txt` is the separate, tested lab environment.
