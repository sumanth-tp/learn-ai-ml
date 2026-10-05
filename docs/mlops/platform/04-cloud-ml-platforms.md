---
id: plat-cloud-ml
title: "Cloud ML Platforms"
sidebar_label: "Cloud ML platforms"
sidebar_position: 4
slug: /mlops/platform/cloud-ml-platforms
description: "A map of Amazon SageMaker AI, Google's Vertex AI (now Gemini Enterprise Agent Platform), Azure Machine Learning and Microsoft Foundry, and Amazon Bedrock, built from the providers' own documentation, with a cost model and a capability checker."
tags: [sagemaker, vertex-ai, azure-machine-learning, bedrock, foundry, cloud-ml, ml-platform]
---

import Infographic from '@site/src/components/Infographic';
import PlatformChooserLab from '@site/src/components/viz/PlatformChooserLab';

**In one line.** The big cloud ML platforms all cover the same lifecycle (train, register, serve online, score in batch, monitor) and differ mainly in their managed generative-AI layer, their current names and where your data already lives, so choose by verifying the few features you depend on rather than by comparing brochures.

:::note Not from a lecture

This chapter is written for this site from the providers' official documentation pages named under Further reading, all opened in October 2026; it is not built from a course lecture. Every product claim below names the page it came from.

:::

## The idea in plain words

Suppose your model works on a laptop. A cloud ML platform sells you the rest: somewhere to train it on rented machines, a place to record which version is which, a way to serve it behind an endpoint that scales, a way to score millions of rows overnight, and something that notices when the data it sees stops looking like the data it learned from. You could build each piece from raw virtual machines, as the previous chapters showed, but the platforms package the pieces and charge for the convenience.

Two kinds of product now sit side by side, and confusing them is the most common mistake.

| Kind | You bring | The platform gives you | Examples in this chapter |
| --- | --- | --- | --- |
| **ML platform** | Data, code, a model you train or import | Training jobs, pipelines, a registry, endpoints, monitoring | SageMaker AI, Vertex AI, Azure Machine Learning |
| **Managed model and agent layer** | A prompt, some documents, a policy | Hosted foundation models behind an API, retrieval, guardrails, agent runtimes | Bedrock, Microsoft Foundry, the agent side of Vertex AI |

The classical lifecycle is the first row; the second row is where most new work happens in 2026. The big three vendors now ship both, and the boundary moves every year, which is why this chapter spends as much time on how to verify a claim as on what the claims are.

<Infographic src="/img/plat/plat-cloud-ml-map.svg" alt="A table mapping six lifecycle stages to the feature names used by SageMaker AI, Vertex, Azure ML with Foundry, and Bedrock" caption="One job, four vocabularies. Every cell is a feature name from an official documentation page; cells that say not found mean the pages I read did not show it." />

<Infographic src="/img/plat/plat-cloud-ml-cost-and-names.svg" alt="A cost table for always-on, pay-per-use and batch serving at five traffic levels, and the dated renames of four products" caption="Cost shape and the moving map. The cost table is printed by block 1 below; the renames are quoted from the provider pages cited in the text." />

## How it works

### Training, pipelines and the registry

The classical stack has the same four pieces everywhere.

- **Train.** SageMaker AI describes itself as a fully managed service to build, train and deploy models ([What is SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)); Azure Machine Learning runs training scripts in the cloud with PyTorch, TensorFlow, scikit-learn, XGBoost and LightGBM supported, plus AutoML and hyperparameter tuning ([What is Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-machine-learning)); Google's introduction page lists AutoML and custom training with PyTorch, TensorFlow and distributed approaches ([Google Cloud introduction](https://docs.cloud.google.com/vertex-ai/docs/start/introduction-unified-platform)).
- **Pipelines.** A pipeline turns the steps into a repeatable graph. Google's pipelines page names Kubeflow Pipelines and TFX as supported formats and says the service orchestrates data preparation, training and tuning, batch inference and artifact lineage ([Pipelines](https://docs.cloud.google.com/vertex-ai/docs/pipelines/introduction)); SageMaker lists Model Building Pipelines as a major feature ([SageMaker AI features](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis-features.html)); Azure's MLOps page says pipelines define repeatable steps for data preparation, training and scoring ([MLOps model management](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment?view=azureml-api-2)).
- **Registry.** Azure's page describes versioned registration, where registering a model under an existing name increments the version, and notes that a registered model cannot be deleted while an active deployment uses it. SageMaker lists a Model Registry with versioning, lineage, approval workflow and cross-account deployment; the Google introduction lists a Model Registry for versioning and lifecycle.
- **Features and experiments.** SageMaker's Feature Store has an online store for low-latency inference and an offline store for training and batch inference, and Experiments tracks runs ([features page](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis-features.html)); the Google page lists a Feature Store and Experiments.

### Serving: four ways to answer a request

This is the choice that most affects both latency and the bill. SageMaker's deployment page describes real-time endpoints for low-latency interactive work, serverless endpoints for traffic with idle periods that can tolerate cold starts, and asynchronous endpoints that queue requests for large payloads (up to 1 GB) and long processing (up to one hour) ([Deploy models for inference](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)). Azure's endpoints page has the same split with different words: *standard deployments* for hosted foundation models billed per token without consuming your compute quota, *online endpoints* for low-latency custom models, and *batch endpoints* for long-running asynchronous work over large data ([Endpoints for inference](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints?view=azureml-api-2)). Its comparison table is worth reading once: online endpoints autoscale on resource use and cannot scale to zero, while batch endpoints scale on job count, can scale to zero and can use low-priority compute.

Online endpoints also carry the release machinery you met in [serving and release strategies](/docs/theory/seml/serving-and-release-strategies). Azure documents controlled rollout with traffic percentages across deployments in one endpoint for A/B testing, and SageMaker lists inference shadow tests, which compare new serving infrastructure against the current one. The experimental design behind those traffic splits is in the [A/B testing chapter](/docs/mlops/platform/experimentation-and-ab-testing).

### Batch scoring

When nobody is waiting for the answer, skip the endpoint. SageMaker's batch transform partitions the S3 objects of the input by key and maps objects to instances; the page warns that with one input file and many instances only one instance works. It can split a file into mini-batches (`SplitType` set to `Line`), caps `MaxPayloadInMB` at 100 and writes one output file per input file with an `.out` suffix ([Batch transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)). Google's batch inference reads from Cloud Storage or BigQuery ([Get batch inferences](https://docs.cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions)). Bedrock's batch inference takes JSONL prompts in S3 and returns outputs to S3; the page notes it is not supported for provisioned models and does not support tool calling or structured output ([Batch inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html)). The next chapter, [batch inference pipelines](/docs/mlops/platform/batch-inference-pipelines), covers how to make such jobs restartable.

### Monitoring

SageMaker's Model Monitor watches production endpoints for data drift and quality deviations. Google's Model Monitoring detects drift and training-serving skew, and exists in two versions with v2 the current approach. Azure's MLOps page lists lifecycle events published to Event Grid, including data drift, to trigger alerts and automation. The data-side counterpart is in [profiling, validation and drift](/docs/mlops/data/profiling-validation-drift).

### The generative layer

**Amazon Bedrock** is described as a fully managed service giving access to foundation models from several providers, with more than 100 models listed ([What is Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)). Around the models it offers Knowledge Bases for retrieval-augmented generation, in a managed form or one where you run your own vector store ([Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)), and Guardrails with content filters, denied topics, word filters, sensitive-information filters, contextual grounding checks and Automated Reasoning checks, usable through the `ApplyGuardrail` API without calling a model ([Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)). The agents page now says that Bedrock Agents is Agents Classic and is no longer open to new customers, pointing to Amazon Bedrock AgentCore ([Agents](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html)).

**Microsoft Foundry** unifies agents, models and tools under one management grouping, advertises more than 10,000 models, and lists observability, governance and content filters ([What is Foundry](https://learn.microsoft.com/en-us/azure/foundry/what-is-foundry)). The Azure Machine Learning page says both Machine Learning studio and Foundry can work with LLMs, and carries a warning that prompt flow is retired on 20 April 2027.

**Google's** platform page lists Gemini and partner models and Agent Studio next to the classical services, and the announcement describes the product as the evolution of Vertex AI with Model Garden giving access to more than 200 models ([announcement](https://cloud.google.com/blog/products/ai-machine-learning/introducing-gemini-enterprise-agent-platform)). For retrieval, evaluation and guardrails on any platform, the site's [Enterprise RAG project](/docs/projects/enterprise-rag/session-1) and [LLM evals](/docs/llm-evals/online-evaluation) chapters show what to build or buy.

:::warning Names are moving

Three renames matter when you search documentation. Amazon SageMaker became Amazon SageMaker AI on 3 December 2024, and "SageMaker" now also names a wider unified data and AI platform (the page lists SageMaker Lakehouse, Unified Studio and others). Google's Vertex AI documentation now uses the product name Gemini Enterprise Agent Platform; the Google Cloud blog, dated 23 April 2026, states that all Vertex AI services and roadmap evolutions will be delivered through it. Azure AI Studio and Azure AI Foundry are now Microsoft Foundry, per the Foundry page's own "previous and current" table. Old tutorials use the old names.

:::

### What a feature checklist cannot tell you

Block 2 below encodes this section as a matrix and asks three kinds of team which platforms show every capability they need. For the classical team, three platforms tie at 7 of 7. A checklist does not separate them, and neither will the brochure. What does: where your data already lives, which identity system your company runs, the price of your actual traffic, and the limits on the one or two features you cannot do without.

## Code you can run

Two blocks, executed with Python 3.14. Neither calls a cloud; both are arithmetic and bookkeeping you would otherwise do in a spreadsheet.

### 1. A cost model with named parameters

Prices change, so no price is quoted here. The rate is a placeholder in cost units per instance-hour, and the model has three named parameters you replace with your own: the premium that pay-per-use charges over an owned instance per busy second, the compute time per request, and the utilisation you size always-on capacity for.

```python
import math

HOURS_PER_MONTH = 24 * 365 / 12
SECONDS_PER_MONTH = HOURS_PER_MONTH * 3600

def always_on_cost(requests, seconds_per_request, rate, target_utilisation):
    busy_seconds = requests * seconds_per_request
    instances = max(1, math.ceil(busy_seconds / (SECONDS_PER_MONTH * target_utilisation)))
    return instances, instances * HOURS_PER_MONTH * rate

def pay_per_use_cost(requests, seconds_per_request, rate, premium):
    return requests * seconds_per_request * (rate / 3600) * premium

def batch_cost(requests, seconds_per_request, rate):
    return requests * seconds_per_request / 3600 * rate

rate, premium, seconds, target = 1.0, 2.0, 0.2, 0.6
print(f"placeholder rate {rate} cost unit per instance-hour; pay-per-use premium x{premium:g}; {seconds}s of compute per request; always-on sized for {target:.0%} busy")
one_instance = HOURS_PER_MONTH * rate
break_even = one_instance / (seconds * (rate / 3600) * premium)
print(f"one always-on instance for a month: {one_instance:.0f} cost units")
print(f"pay-per-use matches it at {break_even:,.0f} requests a month, {break_even * seconds / SECONDS_PER_MONTH:.0%} busy time on that instance\n")

print(f"{'requests/month':>16}{'instances':>11}{'always-on':>11}{'pay-per-use':>13}{'batch':>8}{'cheapest online':>17}")
for requests in (100_000, 1_000_000, 10_000_000, 30_000_000, 100_000_000):
    n, a = always_on_cost(requests, seconds, rate, target)
    p = pay_per_use_cost(requests, seconds, rate, premium)
    b = batch_cost(requests, seconds, rate)
    print(f"{requests:>16,}{n:>11}{a:>11.0f}{p:>13.0f}{b:>8.0f}{('pay-per-use' if p < a else 'always-on'):>17}")
print("\nbatch is the compute you actually use, with no idle capacity, but the answers arrive hours later")
```

The table shows three regimes. At low traffic one idle instance dominates, so pay-per-use is far cheaper (11 against 730 at 100,000 requests). Around 6.57 million requests a month a single instance would be 50% busy and the two cost the same. Above that, owned capacity wins (2,920 against 3,333 at 30 million). The column to notice is **batch**: 1,667 at 30 million, because it pays for compute actually used. If answers can wait, batch is the cheapest of the three here, which is why the next chapter exists. The sawtooth at 10 million, where a second instance appears, is real and is why you read a capacity curve, not a single break-even number.

### 2. The capability matrix, with sources

Each cell holds the documentation page behind it, or nothing when the pages I read did not show it. Nothing in this table claims a feature is missing, only that I did not find it.

```python
PLATFORMS = ["SageMaker AI", "Vertex (Agent Platform)", "Azure ML", "Bedrock", "Foundry"]

SOURCES = {
    "sm_whatis": "docs.aws.amazon.com/sagemaker/latest/dg/whatis.html",
    "sm_features": "docs.aws.amazon.com/sagemaker/latest/dg/whatis-features.html",
    "sm_deploy": "docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html",
    "sm_batch": "docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html",
    "bed_overview": "docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html",
    "bed_kb": "docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html",
    "bed_guard": "docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html",
    "bed_batch": "docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html",
    "vtx_intro": "docs.cloud.google.com/vertex-ai/docs/start/introduction-unified-platform",
    "vtx_batch": "docs.cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions",
    "vtx_pipe": "docs.cloud.google.com/vertex-ai/docs/pipelines/introduction",
    "azml_overview": "learn.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-machine-learning",
    "azml_endpoints": "learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints",
    "azml_mlops": "learn.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment",
    "vtx_monitor": "docs.cloud.google.com/vertex-ai/docs/model-monitoring/overview",
    "bed_agents": "docs.aws.amazon.com/bedrock/latest/userguide/agents.html",
    "foundry": "learn.microsoft.com/en-us/azure/foundry/what-is-foundry",
}

CAPABILITIES = [
    ("custom_training", "Train your own models", {"SageMaker AI": "sm_whatis", "Vertex (Agent Platform)": "vtx_intro", "Azure ML": "azml_overview"}),
    ("pipelines", "Managed ML pipelines", {"SageMaker AI": "sm_features", "Vertex (Agent Platform)": "vtx_pipe", "Azure ML": "azml_overview"}),
    ("model_registry", "Model registry", {"SageMaker AI": "sm_features", "Vertex (Agent Platform)": "vtx_intro", "Azure ML": "azml_mlops"}),
    ("feature_store", "Feature store", {"SageMaker AI": "sm_features", "Vertex (Agent Platform)": "vtx_intro"}),
    ("experiment_tracking", "Experiment tracking", {"SageMaker AI": "sm_features", "Vertex (Agent Platform)": "vtx_intro", "Azure ML": "azml_overview"}),
    ("drift_monitoring", "Production drift monitoring", {"SageMaker AI": "sm_features", "Vertex (Agent Platform)": "vtx_monitor", "Azure ML": "azml_mlops"}),
    ("online_endpoint", "Online inference endpoint", {"SageMaker AI": "sm_deploy", "Vertex (Agent Platform)": "vtx_intro", "Azure ML": "azml_endpoints", "Bedrock": "bed_overview", "Foundry": "foundry"}),
    ("batch_inference", "Batch inference", {"SageMaker AI": "sm_batch", "Vertex (Agent Platform)": "vtx_batch", "Azure ML": "azml_endpoints", "Bedrock": "bed_batch"}),
    ("serverless_inference", "Serverless inference", {"SageMaker AI": "sm_deploy", "Azure ML": "azml_endpoints"}),
    ("foundation_models", "Hosted foundation models", {"SageMaker AI": "sm_deploy", "Vertex (Agent Platform)": "vtx_intro", "Azure ML": "azml_overview", "Bedrock": "bed_overview", "Foundry": "foundry"}),
    ("managed_rag", "Managed retrieval (RAG)", {"Bedrock": "bed_kb"}),
    ("guardrails", "Guardrails or content filters", {"Bedrock": "bed_guard", "Foundry": "foundry"}),
    ("agents", "Agent building", {"Vertex (Agent Platform)": "vtx_intro", "Bedrock": "bed_agents", "Foundry": "foundry"}),
    ("kubernetes_compute", "Kubernetes compute", {"SageMaker AI": "sm_features", "Azure ML": "azml_overview"}),
]
KEYS = {key: (label, cells) for key, label, cells in CAPABILITIES}

SCENARIOS = {
    "classical ML team": ["custom_training", "pipelines", "model_registry", "experiment_tracking", "drift_monitoring", "online_endpoint", "batch_inference"],
    "GenAI application team": ["foundation_models", "managed_rag", "guardrails", "agents", "batch_inference", "online_endpoint"],
    "fine-tune and serve": ["custom_training", "foundation_models", "online_endpoint", "serverless_inference"],
}

def coverage(required):
    rows = []
    for platform in PLATFORMS:
        have = [k for k in required if platform in KEYS[k][1]]
        missing = [KEYS[k][0] for k in required if platform not in KEYS[k][1]]
        rows.append((platform, len(have), missing))
    return sorted(rows, key=lambda r: (-r[1], r[0]))

for name, required in SCENARIOS.items():
    print(f"{name}: needs {len(required)} capabilities")
    for platform, count, missing in coverage(required):
        gap = "" if not missing else "  not found on the pages read: " + ", ".join(missing)
        print(f"  {platform:<24}{count} of {len(required)}{gap}")
    print()

cited = sum(len(cells) for _, _, cells in CAPABILITIES)
total = len(CAPABILITIES) * len(PLATFORMS)
print(f"{cited} of {total} cells have a source page; the other {total - cited} read 'not found on the pages read', which is not the same as 'not offered'")
print(f"distinct source pages: {len({s for _, _, c in CAPABILITIES for s in c.values()})}")
```

Three platforms show all seven capabilities for the classical team, so that checklist cannot decide between them. For the generative team, Bedrock shows all six, but the counts also reflect which pages I opened, which is the honest limit of this method: of 70 cells, 41 have a source and 29 do not. Use the lab below to run your own list. Its defaults reproduce block 2 and the matrix is generated from the same data.

<PlatformChooserLab />

## Designing with it

- **Start from the workload, not the vendor.** Is it a model you train, or a model you call? Online with a latency target, or batch with a deadline? The answers pick the serving mode before they pick the platform.
- **Price your own traffic.** Use the provider's current price page with your request rate, compute time per request and idle time. Put the numbers in a model like block 1, and re-run it when the price page changes.
- **Make batch the default for anything that can wait.** It has no idle capacity, scales to zero, and in Azure's table can use low-priority compute.
- **Keep a thin exit.** Containerise the model and keep pipelines and infrastructure in code, as in the [Kubernetes](/docs/mlops/platform/kubernetes-for-ml) and [infrastructure as code](/docs/mlops/platform/infrastructure-as-code) chapters, so the platform is a place you run rather than a place you are trapped. SageMaker's own deployment page recommends infrastructure as code for production at scale.
- **Treat names and dates as data.** Record the product name, the documentation page and the date you read it next to every platform decision. The renames above would have broken a design document written two years ago.
- **Verify the limits that bind you.** Payload size, timeouts, quotas, regions and model availability decide whether a design works, and they live in pages the overview does not summarise.

## Where this stands in 2026

:::info Industry view

- **Rebrands around agents.** The three big clouds all lead their documentation with agents and models: Google's blog of 23 April 2026 announces Gemini Enterprise Agent Platform as the evolution of Vertex AI; Microsoft Foundry unifies agents, models and tools; Bedrock's agents page points new customers from Agents Classic to AgentCore.
- **Platform sprawl is real.** SageMaker's page lists dozens of features, including the 2024 unified platform around it. Learn the lifecycle first, then the product.
- **Retirements arrive on a schedule.** Prompt flow in Azure Machine Learning and Foundry is retired on 20 April 2027 and Microsoft points to Microsoft Agent Framework. Build on stable parts and check the retirement notices.
- **Model catalogues are large and changing.** Counts quoted above (100+, 200+, 10,000+) come from each provider's own pages on the dates read and are counted differently, so they are not comparable.
- **Not verified here.** Prices, quotas, regional availability, GPU instance types and any benchmark between platforms were deliberately not checked; they change too fast to quote.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Your team sees a tutorial about "Vertex AI Pipelines" but the documentation site now says "Gemini Enterprise Agent Platform". Is the tutorial obsolete?</summary>

Probably not the concepts: the pipelines page still describes Kubeflow Pipelines and TFX workflows, and the product is described as the evolution of Vertex AI. Expect names of consoles and some API paths to differ, so check the current page before copying commands, and record the date you checked.

</details>

<details>
<summary><strong>Q2.</strong> Traffic is 1 million requests a month at 0.2 s of compute each, with no latency target beyond a few seconds. Which serving mode do you reach for first, using block 1?</summary>

Pay-per-use or serverless, which cost 111 against 730 for one always-on instance. If the work arrives in bursts and answers can wait, batch costs even less (56). The decision flips above about 6.57 million requests a month at the placeholder premium of 2, so re-run the model with your real premium.

</details>

<details>
<summary><strong>Q3.</strong> A nightly job scores 40 million rows from a single 80 GB CSV on SageMaker batch transform with ten instances, and only one instance seems busy. Why?</summary>

The documentation says batch transform partitions the S3 objects by key across instances, and with one input file and several instances only one instance processes it. Split the input into many objects, so the instances each get files, and consider `SplitType` set to `Line` to form mini-batches within a file.

</details>

<details>
<summary><strong>Q4.</strong> Why does block 2 say "not found on the pages read" instead of "not supported"?</summary>

Because it is a statement about my reading, not about the product. A feature can exist in a page I did not open. A checker that turned absence of evidence into a claim would mislead anyone using it to decide. It is a list of what to verify.

</details>

<details>
<summary><strong>Q5.</strong> A design document from two years ago recommends Bedrock Agents for a new project. What do you check today?</summary>

The current agents page, which states that Bedrock Agents is now Agents Classic and is no longer open to new customers, pointing to AgentCore. Existing customers continue as normal, but a new project should evaluate the replacement.

</details>

<details>
<summary><strong>Q6.</strong> Three platforms tie on a capability checklist. List four tie-breakers that do not appear on feature pages.</summary>

Where the training data and the identity system already live; the price of your own traffic under each platform's rates; the limits on the features you depend on, such as payload size, timeout and quotas; and the team's existing skills and tooling, including how much of the stack is already in code that can move.

</details>

## Further reading

- [Amazon SageMaker AI: what it is](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html), [features](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis-features.html), [deploying models](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html) and [batch transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html): the rename note, the feature list, the inference options and the batch limits.
- [Amazon Bedrock overview](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html), [Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html), [Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html), [batch inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html) and [Agents](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html).
- [Google Cloud: introduction to ML on the platform](https://docs.cloud.google.com/vertex-ai/docs/start/introduction-unified-platform), [Pipelines](https://docs.cloud.google.com/vertex-ai/docs/pipelines/introduction), [batch inference](https://docs.cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions), [Model Monitoring](https://docs.cloud.google.com/vertex-ai/docs/model-monitoring/overview) and the [announcement of Gemini Enterprise Agent Platform](https://cloud.google.com/blog/products/ai-machine-learning/introducing-gemini-enterprise-agent-platform).
- [What is Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-machine-learning), [endpoints for inference](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints?view=azureml-api-2), [MLOps model management](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment?view=azureml-api-2) and [What is Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/what-is-foundry).

## Check yourself

- I can name the stages of the ML lifecycle and find each one in at least two platforms' vocabularies.
- I can choose between real-time, serverless, asynchronous and batch serving for a workload, and build a cost model with named parameters instead of quoted prices.
- I can explain why batch is the cheapest option when answers can wait and where the always-on versus pay-per-use crossover comes from.
- I can separate an ML platform from a managed model and agent layer.
- I can verify a vendor feature claim against its documentation page and record the page and the date.
- I can say which names have changed (SageMaker AI, the Agent Platform, Foundry, Agents Classic) and why dated notes matter.
