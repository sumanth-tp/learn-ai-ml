---
title: Tools, Versions and Reproducibility Notebook
sidebar_label: Tools and versions
sidebar_position: 99
---

# Tools, Versions and Reproducibility Notebook

Name the mechanism, pin the implementation, and show what you actually tested.

**Snapshot checked: 26 September 2026.** The package table records release metadata observed from PyPI on that date. It is not a recommended all-in-one installation or a tested compatible stack. Model providers, framework APIs and package versions change; revisit the linked release pages before copying an integration.

## Tested coding environment

The [eight coding labs](11-coding-labs.md) and new chapter Python examples were checked using **Python 3.12.8 on macOS ARM64** with NumPy **2.5.3**, pandas **3.0.6**, scikit-learn **1.9.1** and the transitive versions in the [requirements file](/examples/interviews/requirements.txt). The labs make no paid/model/network calls. Framework integrations, GPU kernels and remote services in the broader topic discussions are not certified by those local tests.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_labs.py
```

These commands assume you extracted the [lab archive](/examples/interviews/interview-labs.zip) and entered its directory. NumPy in this environment requires Python 3.12 or later. A Python minimum in package metadata is only one compatibility constraint; wheels, OS, accelerator runtime and transitive dependencies also matter.

## Observed package releases

Release timestamps below come from package metadata. Each version links to its specific package page. [Download the metadata snapshot](/examples/interviews/package-snapshot.json).

| Package | Observed version | Python requirement | Release date (UTC) | Lab status |
| --- | --- | --- | --- | --- |
| numpy | [2.5.3](https://pypi.org/project/numpy/2.5.3/) | `>=3.12` | 2026-09-06 | Installed and used |
| pandas | [3.0.6](https://pypi.org/project/pandas/3.0.6/) | `>=3.11` | 2026-09-17 | Installed and used |
| scikit-learn | [1.9.1](https://pypi.org/project/scikit-learn/1.9.1/) | `>=3.11` | 2026-09-10 | Installed and used |
| torch | [2.14.0](https://pypi.org/project/torch/2.14.0/) | `>=3.10` | 2026-09-02 | Metadata only; not tested here |
| transformers | [5.17.0](https://pypi.org/project/transformers/5.17.0/) | `>=3.10.0` | 2026-09-09 | Metadata only; not tested here |
| peft | [0.21.0](https://pypi.org/project/peft/0.21.0/) | `>=3.10.0` | 2026-09-15 | Metadata only; not tested here |
| langchain | [1.4.2](https://pypi.org/project/langchain/1.4.2/) | `<4.0.0,>=3.10.0` | 2026-09-18 | Metadata only; not tested here |
| langgraph | [1.2.12](https://pypi.org/project/langgraph/1.2.12/) | `>=3.10` | 2026-09-21 | Metadata only; not tested here |
| ragas | [0.4.3](https://pypi.org/project/ragas/0.4.3/) | `>=3.9` | 2026-01-13 | Metadata only; not tested here |
| deepeval | [4.2.6](https://pypi.org/project/deepeval/4.2.6/) | `<4.0,>=3.9` | 2026-09-24 | Metadata only; not tested here |
| mlflow | [3.16.1](https://pypi.org/project/mlflow/3.16.1/) | `>=3.10` | 2026-09-16 | Metadata only; not tested here |
| fastapi | [0.141.1](https://pypi.org/project/fastapi/0.141.1/) | `>=3.10` | 2026-07-29 | Metadata only; not tested here |
| pydantic | [2.13.5](https://pypi.org/project/pydantic/2.13.5/) | `>=3.9` | 2026-08-28 | Metadata only; not tested here |
| vllm | [0.30.0](https://pypi.org/project/vllm/0.30.0/) | `<3.15,>=3.10` | 2026-09-22 | Metadata only; not tested here |
| qdrant-client | [1.19.1](https://pypi.org/project/qdrant-client/1.19.1/) | `>=3.10` | 2026-09-16 | Metadata only; not tested here |
| mcp | [2.2.0](https://pypi.org/project/mcp/2.2.0/) | `>=3.10` | 2026-09-07 | Metadata only; not tested here |
| pytest | [9.1.1](https://pypi.org/project/pytest/9.1.1/) | `>=3.10` | 2026-06-19 | Metadata only; not tested here |

## Differences an interviewer can probe

| Area | Earlier habit or misleading shortcut | Current implementation question |
| --- | --- | --- |
| Python concurrency | Create one task per input and add a semaphore | What bounds task creation, queued payloads, active calls and retained results separately? |
| pandas 3 | Mutate a chained selection and assume the parent changed | Copy-on-Write requires changing the intended object explicitly, such as one `.loc` assignment |
| NumPy 2 | Use `np.array(..., copy=False)` to mean “copy if necessary” | `copy=False` is stricter; use `np.asarray` for ordinary conversion that may copy |
| scikit-learn encoders | Use old `sparse=` examples | Inspect `OneHotEncoder`'s pinned signature; modern code uses `sparse_output=` and a deliberate unknown-category policy |
| Pretrained estimator reuse | Accidentally refit a model while composing a pipeline | `FrozenEstimator` can prevent fitting an already fitted estimator; data used for calibration must still be appropriate |
| Graph persistence | Assume resuming a node resumes after every earlier statement | Code before an interrupt can run again; place side effects behind a safe execution protocol |
| MCP | Treat the SDK package version as the protocol version | Record both the SDK release and negotiated protocol revision; still enforce tool and token policy |
| Model serving | Copy metric names from a tutorial without checking engine/version | Inspect emitted metrics, units, histogram buckets, labels and engine version |
| Evaluation frameworks | Assume identically named metrics mean the same thing | Compare formulas, rubrics, judge prompts, missing-result handling and aggregation |
| Model aliases | Record only a friendly model name | Save a resolvable revision where available and detect behavioural changes through evaluations |

Sources: [Python tasks and cancellation](https://docs.python.org/3/library/asyncio-task.html), [pandas Copy-on-Write](https://pandas.pydata.org/docs/user_guide/copy_on_write.html), [NumPy migration guide](https://numpy.org/doc/2.0/numpy_2_0_migration_guide.html), [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), [FrozenEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.frozen.FrozenEstimator.html), [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts), [MCP 2025-11-25 specification](https://modelcontextprotocol.io/specification/2025-11-25), [vLLM metrics source](https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md). The vLLM `main` source explains design; check the matching release before using exact names.

### A tested pandas mutation example

```python
import pandas as pd

frame = pd.DataFrame({"score": [0.2, 0.8]})
selected = frame[frame["score"] > 0.5]
selected.loc[:, "score"] = 1.0
assert frame.loc[1, "score"] == 0.8
frame.loc[frame["score"] > 0.5, "score"] = 1.0
assert frame.loc[1, "score"] == 1.0
```

### A tested preprocessing and model pipeline

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train = pd.DataFrame({"age": [20, 30, 40, 50, 25, 45],
                      "region": ["N", "S", "N", "S", "N", "S"]})
y = [0, 0, 1, 1, 0, 1]
numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
features = ColumnTransformer([
    ("age", numeric, ["age"]),
    ("region", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["region"]),
])
model = Pipeline([("features", features), ("model", LogisticRegression())])
model.fit(train, y)
# A new region and missing age exercise the declared preprocessing contract.
probability = model.predict_proba(pd.DataFrame({"age": [float("nan")], "region": ["NEW"]}))
assert probability.shape == (1, 2)
assert abs(probability.sum() - 1) < 1e-9
```

This tiny dataset tests API behaviour, not model quality. A real experiment needs suitable train/validation/test boundaries, sufficient data and calibrated uncertainty. Save and serve the pipeline together; check trusted provenance before loading serialised Python objects.

## What belongs in a reproducibility manifest?

The following JSON is a **schema illustration with example identifiers**, not a claim that these model artefacts exist:

```json
{
  "run_id": "eval-demo-017",
  "code_revision": "record-the-actual-git-commit",
  "python": "3.12.8",
  "dependency_lock_sha256": "compute-from-your-lock-file",
  "dataset_revision": "support-eval-v7",
  "split_ids_revision": "split-v3",
  "model_revision": "record-provider-or-artifact-revision",
  "tokenizer_revision": "record-exact-tokenizer",
  "embedding_revision": "record-exact-embedding-model",
  "index_revision": "support-index-v12",
  "prompt_revision": "answer-v9",
  "tool_schema_revision": "tools-v4",
  "judge_and_rubric_revision": "judge-rubric-v6",
  "seed": 7,
  "hardware_and_runtime": "record-device-driver-kernel-runtime",
  "expected_case_ids": ["q1", "q2"]
}
```

Do not log credentials, raw access tokens or unnecessary private prompts in this manifest. A seed controls only the randomness using it; it does not freeze upstream data, provider changes or nondeterministic kernels.

## How to answer a version question

**Weak answer:** “I use the latest LangGraph, Ragas and vLLM.”

**Defensible answer:** “This local lab used Python 3.12.8 and NumPy 2.5.3. I verified causal masking against a reference. For a hosted service I would also record the engine/model/tokenizer revisions and benchmark its supported runtime. I have not run that GPU stack here.”

For an upgrade, identify the changed contract, read that release's migration notes, run your regression fixtures, compare behavioural metrics on frozen data, and preserve a compatible rollback bundle. Release freshness is not evidence that a combination works.

## Summary in simple points

- Separate observed package versions from versions actually tested together.
- Record Python, packages, operating environment and accelerator runtime where relevant.
- Treat pandas mutation, NumPy copy rules and estimator API changes as behaviour to test.
- Framework persistence does not make external side effects automatically safe.
- MCP's protocol revision and its SDK package version are different identifiers.
- Judge, prompt, model, tokenizer, data and index changes can invalidate comparisons.
- Use a small reproducible fixture to explain a version difference in an interview.
- Run an upgrade experiment and keep a compatible rollback path.
