---
id: plat-iac
title: "Infrastructure as Code"
sidebar_label: "Infrastructure as code"
sidebar_position: 3
slug: /mlops/platform/infrastructure-as-code
description: "Describe cloud infrastructure in files, preview every change as a plan, track what you own in state, and catch drift, with a plan and apply engine in Python so the mechanics are visible."
tags: [infrastructure-as-code, terraform, pulumi, state, drift, modules, ml-platform]
---

import Infographic from '@site/src/components/Infographic';
import PlanDiffLab from '@site/src/components/viz/PlanDiffLab';

**In one line.** Infrastructure as code writes down what should exist, compares it with what the tool believes exists and with what really exists, and shows you the difference as a plan before anything changes.

:::note Not from a lecture

This chapter is written for this site from the Terraform, Pulumi and SageMaker documentation listed under Further reading; it is not built from a course lecture.

:::

## The idea in plain words

Suppose the GPU cluster behind your model endpoint was built by clicking through a console two years ago. Nobody remembers which node type was chosen, a colleague resized it last month, and the staging copy differs in six unknown ways. Rebuilding it after an outage becomes archaeology.

**Infrastructure as code** (IaC) turns the clicks into text files held in version control. The files are *declarative*: they describe the end state ("a cluster with 2 nodes of this type, an endpoint with 3 replicas"), not the steps. The tool works out the steps. Three things must be kept apart, and every IaC bug is a confusion between two of them.

| Thing | Where it lives | Question it answers |
| --- | --- | --- |
| **Configuration** (desired) | Your files | What do we want? |
| **State** (recorded) | A file or remote store the tool keeps | What does the tool think it created? |
| **Reality** (actual) | The cloud | What is really there? |

The loop is: **plan** compares configuration with state (after refreshing state from reality) and lists the actions; **apply** executes them in dependency order and updates the state. Reading the plan is the review step. It is the reason IaC is safer than a console: you see "this will destroy the bucket" before it happens, in a form a colleague can review in a pull request.

<Infographic src="/img/plat/plat-iac-plan-apply.svg" alt="Configuration, state and reality as three boxes with plan and apply arrows, and the plan the chapter engine produces for a bucket rename and an endpoint change" caption="Plan and apply. The plan lines and summary are printed by block 1 below." />

<Infographic src="/img/plat/plat-iac-state-drift.svg" alt="A console edit causing drift, the plan that reverts it, ignore_changes, and two writers losing an update without a state lock" caption="Drift and locking. The drift report and the lock experiment are printed by blocks 1 and 2 below." />

## How it works

### Plan, apply, and the symbols

Terraform's plan command does three things, according to its documentation: it reads the current state (synchronising it with the remote objects), compares the configuration with it, and proposes the actions that would make the remote objects match. The plan marks each resource with a symbol.

| Symbol | Meaning |
| --- | --- |
| `+` | create |
| `-` | destroy |
| `~` | update in place |
| `-/+` | replace: destroy, then create |

A plan can be saved with `-out` and applied later, which is how pipelines make sure the thing reviewed is the thing applied; the documentation warns that saved plans hold the full configuration and sensitive data in cleartext, so store them carefully. The `-detailed-exitcode` option returns 0 for no changes, 1 for an error and 2 for changes present, which lets a scheduled job fail loudly when reality has drifted.

**Replace versus update** is the distinction that costs money. Some attributes cannot be changed on a live object: a bucket's name, a network's address range. Changing them in code means destroying the object and creating a new one, with whatever data or traffic it held. Read every `-/+` in a plan as "this resource will be deleted".

### State

State is a JSON record that binds each resource in your configuration to a real object, plus metadata and a performance cache, as the documentation puts it. By default it is a local file; the documentation recommends against putting it in version control and recommends a remote backend with **locking and access control**, because state can contain secrets and because two people applying at once can corrupt it. Block 2 shows the second failure: two engineers start from the same state, each writes back their own view, and the second write silently erases the first. Terraform also expects a one-to-one mapping between configuration and remote objects; bringing an existing object under management uses `terraform import`, and removing one from tracking uses `terraform state rm`, after which you maintain that mapping by hand.

### Drift

**Drift** is reality changing behind the tool's back: an engineer scales the cluster in the console during an incident, or a policy deletes an object. Terraform refreshes state from reality before it plans, so drift appears as a diff. Its refresh-only mode (`plan -refresh-only`) inspects the difference without proposing infrastructure changes, and its tutorial shows two honest choices: accept the drift by applying the refresh-only plan, which updates state to match reality, or run a normal apply, which overwrites the manual change with what the code says. The tutorial's recommended habit is to bring manual creations under management with import and reconcile.

Neither choice is wrong. The wrong choice is not choosing: leaving the cluster at 4 nodes while the code says 2 means the next unrelated apply quietly shrinks it. For attributes owned by something else, such as an autoscaler adjusting node count, the `ignore_changes` setting inside `lifecycle` tells the tool to leave them alone.

### Lifecycle rules

The `lifecycle` block adjusts how a resource is changed. From the documentation: `create_before_destroy` builds the replacement before removing the old one, `prevent_destroy` makes Terraform reject any plan that would destroy the object (but not if you delete the resource block entirely), `ignore_changes` disregards named attributes during updates, and `replace_triggered_by` forces replacement when something it references changes. For model registries and data buckets, `prevent_destroy` is cheap insurance: block 1 shows the plan failing rather than replacing the bucket.

### Modules, and the dependency graph

A **module** is a set of resources managed together; the configuration in your working directory is the root module, and it calls child modules, which can be reused with different arguments. Modules come from public or private registries, a local path or other sources, and the documentation shows a call carrying a `source` and a pinned `version`. Pin versions: an unpinned module changes under you.

Inside a plan, references between resources form a graph. The endpoint refers to the cluster, so the cluster is created first and destroyed last; independent resources go in the same wave and run in parallel. Block 1 builds that graph with Python's `graphlib`.

### Terraform and Pulumi

Both are declarative and both have a preview step and state. They differ in how you write the desired state. Pulumi describes its model as programs in general-purpose languages (TypeScript, Python, Go, Java) or YAML that declare resource objects within a *project*, with isolated instances of a program called *stacks* for environments like development and production; its CLI and deployment engine work out the operations. Terraform uses its own configuration language, HCL, organised into modules. Choose on team skills and ecosystem, since the plan and state concepts in this chapter apply to both.

## A real system that works this way

**Amazon SageMaker AI** documents IaC as the route for model deployment at scale. Its "deploy models" page lists three use cases and for the third (managing models at scale in production) recommends the AWS SDK for Python together with CloudFormation and IaC and CI/CD tools to provision and automate resources, "ideal for advanced users who require consistent and repeatable deployments". It also notes the cost: this path needs infrastructure management, organisational resources and familiarity with the tooling.

**Terraform's own plan workflow** is the system the engine below imitates. Its documentation describes the refresh, compare and propose steps above, saved plans for automation and a drift tutorial built on a real instance and security group: a group is attached by hand, `plan -refresh-only` reports the old group removed and the new one added, and the user chooses to accept or overwrite.

## Code you can run

Two blocks, executed with Python 3.14. The resource types and their immutable attributes are a made-up schema; the logic (refresh, compare, classify as create, update, replace or destroy, order by dependency, honour `prevent_destroy` and `ignore_changes`) follows the concepts in the documentation above. The summary line copies the shape of Terraform's, with a replacement counted as one add and one destroy.

### 1. A plan and apply engine

```python
import copy
import re
from graphlib import TopologicalSorter

IMMUTABLE = {"network": {"cidr"}, "bucket": {"name"}, "cluster": set(), "endpoint": set(), "alarm": set()}
REF = re.compile(r"\$\{([a-z]+\.[a-z]+)\.([a-z_]+)\}")

def resolve(config):
    out = {}
    for addr, res in config.items():
        attrs = {}
        for key, value in res["attrs"].items():
            if isinstance(value, str):
                value = REF.sub(lambda m: str(config[m.group(1)]["attrs"][m.group(2)]), value)
            attrs[key] = value
        out[addr] = {"type": res["type"], "attrs": attrs, "ignore": res.get("ignore_changes", []), "keep": res.get("prevent_destroy", False)}
    return out

def depends(config, addr):
    refs = set()
    for value in config[addr]["attrs"].values():
        if isinstance(value, str):
            refs |= {m.group(1) for m in REF.finditer(value)}
    return refs

def plan(config, state):
    want = resolve(config)
    actions = []
    for addr, res in want.items():
        if addr not in state:
            actions.append(("create", addr, {}))
            continue
        changed = {k: (state[addr]["attrs"].get(k), v) for k, v in res["attrs"].items()
                   if k not in res["ignore"] and state[addr]["attrs"].get(k) != v}
        if not changed:
            continue
        forced = [k for k in changed if k in IMMUTABLE[res["type"]]]
        if forced and res["keep"]:
            raise RuntimeError(f"{addr}: prevent_destroy blocks the replacement forced by {forced}")
        actions.append(("replace" if forced else "update", addr, changed))
    for addr in state:
        if addr not in want:
            actions.append(("destroy", addr, {}))
    return actions

def summary(actions):
    add = sum(a[0] in ("create", "replace") for a in actions)
    change = sum(a[0] == "update" for a in actions)
    destroy = sum(a[0] in ("destroy", "replace") for a in actions)
    return "No changes." if not actions else f"Plan: {add} to add, {change} to change, {destroy} to destroy."

SYMBOL = {"create": "+", "update": "~", "destroy": "-", "replace": "-/+"}

def show(actions):
    for kind, addr, changed in actions:
        detail = ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in changed.items())
        print(f"  {SYMBOL[kind]:<4}{addr}  {detail}".rstrip())
    print(" ", summary(actions))

def apply(config, state, actions):
    want = resolve(config)
    graph = {addr: depends(config, addr) for addr in config}
    waves = []
    sorter = TopologicalSorter(graph)
    sorter.prepare()
    todo = {a[1]: a for a in actions}
    while sorter.is_active():
        ready = sorted(sorter.get_ready())
        waves.append([r for r in ready if r in todo])
        sorter.done(*ready)
    for addr in [a[1] for a in actions if a[0] in ("destroy", "replace")]:
        state.pop(addr, None)
    for addr in [a[1] for a in actions if a[0] != "destroy"]:
        state[addr] = {"type": want[addr]["type"], "attrs": dict(want[addr]["attrs"])}
    return [w for w in waves if w]

def refresh(state, real):
    drift = []
    for addr in list(state):
        if addr not in real:
            drift.append((addr, "deleted outside"))
            continue
        for key, value in real[addr]["attrs"].items():
            if state[addr]["attrs"].get(key) != value:
                drift.append((addr, f"{key}: {state[addr]['attrs'][key]} -> {value}"))
    return drift

V1 = {
    "network.main": {"type": "network", "attrs": {"cidr": "10.0.0.0/16"}},
    "bucket.artifacts": {"type": "bucket", "attrs": {"name": "ml-artifacts-prod", "versioning": True}},
    "cluster.gpu": {"type": "cluster", "attrs": {"network": "${network.main.cidr}", "node_count": 2, "machine_type": "gpu-small"}},
    "endpoint.ranker": {"type": "endpoint", "attrs": {"cluster": "${cluster.gpu.machine_type}", "bucket_name": "${bucket.artifacts.name}", "replicas": 3, "image": "ranker:1.0"}},
}
V2 = copy.deepcopy(V1)
V2["bucket.artifacts"]["attrs"]["name"] = "ml-artifacts-prod-eu"
V2["endpoint.ranker"]["attrs"].update(replicas=5, image="ranker:1.1")
V2["alarm.latency"] = {"type": "alarm", "attrs": {"endpoint": "${endpoint.ranker.image}", "threshold_ms": 200}}

state = {}
print("1. first plan against an empty state")
actions = plan(V1, state)
show(actions)
waves = apply(V1, state, actions)
print("  apply order, one wave at a time:", waves)
print("\n2. plan again with nothing changed")
show(plan(V1, state))

print("\n3. the config changes: bigger endpoint, new bucket name, a latency alarm")
actions = plan(V2, state)
show(actions)
apply(V2, state, actions)
print("\n   after apply, planning V2 again:", summary(plan(V2, state)))

print("\n4. someone changes node_count in the console")
real = copy.deepcopy(state)
real["cluster.gpu"]["attrs"]["node_count"] = 4
print("  refresh-only reports:", refresh(state, real))
state["cluster.gpu"]["attrs"]["node_count"] = 4
print("  a normal plan then wants to undo it:")
show(plan(V2, state))
V3 = copy.deepcopy(V2)
V3["cluster.gpu"]["ignore_changes"] = ["node_count"]
print("  with ignore_changes on node_count:", summary(plan(V3, state)))

print("\n5. prevent_destroy on the bucket")
V4 = copy.deepcopy(V2)
V4["bucket.artifacts"]["prevent_destroy"] = True
V4["bucket.artifacts"]["attrs"]["name"] = "ml-artifacts-prod-us"
try:
    plan(V4, state)
except RuntimeError as err:
    print("  error:", err)
```

Read the five steps. The first plan against an empty state is four creations, applied in three waves because the cluster waits for the network and the endpoint waits for the cluster. A second plan reports no changes, which is **idempotence**: applying the same code twice does nothing the second time. Step 3 is the interesting one. Renaming the bucket is a `-/+` because the name is immutable, and because the endpoint reads the bucket's name, the endpoint changes too, together with its replica count and image; the new alarm is a creation. The summary is "2 to add, 1 to change, 1 to destroy". The `-/+` counts once in each of add and destroy, which is why one replaced bucket plus one new alarm makes 2 adds.

Step 4 is drift. The refresh-only report names the one attribute that moved, and a normal plan then wants to put it back (`node_count: 4 -> 2`). With `ignore_changes` the plan is empty. Step 5 shows the guard: the plan itself fails, before any apply could replace the bucket.

The lab below runs this engine. Its defaults reproduce step 3, and the drift boxes reproduce step 4.

<PlanDiffLab />

### 2. Why state needs a lock

```python
import json
import os
import tempfile

class StateStore:
    def __init__(self, directory):
        self.path = os.path.join(directory, "terraform.tfstate.json")
        self.lock_path = self.path + ".lock"
        self.write({"serial": 0, "resources": {}})

    def read(self):
        with open(self.path) as handle:
            return json.load(handle)

    def write(self, state):
        with open(self.path, "w") as handle:
            json.dump(state, handle)

    def acquire(self, who):
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, who.encode())
        os.close(fd)
        return True

    def release(self):
        os.remove(self.lock_path)

def apply_change(snapshot, name, value):
    snapshot = json.loads(json.dumps(snapshot))
    snapshot["resources"][name] = value
    snapshot["serial"] += 1
    return snapshot

with tempfile.TemporaryDirectory() as directory:
    store = StateStore(directory)
    a_view, b_view = store.read(), store.read()
    store.write(apply_change(a_view, "cluster.gpu", {"node_count": 4}))
    store.write(apply_change(b_view, "endpoint.ranker", {"replicas": 5}))
    final = store.read()
    print("no locking, two engineers apply from the same starting state")
    print("  resources recorded:", sorted(final["resources"]), " serial:", final["serial"])
    print("  cluster.gpu is lost from the state, so the next plan wants to create it again")

with tempfile.TemporaryDirectory() as directory:
    store = StateStore(directory)
    print("\nwith a lock file")
    print("  A acquires:", store.acquire("A"))
    print("  B acquires while A holds it:", store.acquire("B"))
    store.write(apply_change(store.read(), "cluster.gpu", {"node_count": 4}))
    store.release()
    print("  A released; B acquires:", store.acquire("B"))
    store.write(apply_change(store.read(), "endpoint.ranker", {"replicas": 5}))
    store.release()
    final = store.read()
    print("  resources recorded:", sorted(final["resources"]), " serial:", final["serial"])
```

Without the lock the second writer overwrites the first: only `endpoint.ranker` survives and the serial number is 1, so `cluster.gpu` has vanished from state and the next plan would try to create it again. With the lock the second engineer is told no, waits, reads the updated state and ends with both resources and serial 2. Real remote backends supply the same guarantee with a lock the whole team shares.

## Production snippets (not run here)

The provider and resource types in these files are placeholders for the shape of the language, not a real provider's schema. The module call follows the form shown in the Terraform modules documentation.

```hcl
variable "replicas" {
  type    = number
  default = 3
}

resource "example_bucket" "artifacts" {
  name       = "ml-artifacts-prod"
  versioning = true

  lifecycle {
    prevent_destroy = true
  }
}

resource "example_cluster" "gpu" {
  node_count   = 2
  machine_type = "gpu-small"

  lifecycle {
    ignore_changes = [node_count]
  }
}

resource "example_endpoint" "ranker" {
  cluster     = example_cluster.gpu.id
  bucket_name = example_bucket.artifacts.name
  replicas    = var.replicas
  image       = "ranker:1.1"
}

module "network" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"
}
```

Not run in this environment.

```bash
terraform init
terraform plan -out=release.plan -detailed-exitcode
terraform apply release.plan
terraform plan -refresh-only
```

Not run in this environment.

## Designing with it

- **Review the plan, not the diff of the code.** The plan is the only artefact that shows what will really be destroyed. Make it a required output of every pull request.
- **Treat `-/+` as a deletion.** For stateful resources (buckets, registries, databases) add `prevent_destroy`, and plan renames explicitly with a migration, never as a casual edit.
- **Use remote state with locking from day one,** and keep it out of version control. State holds secrets and is the team's shared memory.
- **Schedule a drift check.** A nightly `plan -detailed-exitcode` that pages on exit code 2 finds console edits while they are still cheap to understand.
- **Decide who owns each attribute.** If an autoscaler owns node count, say so with `ignore_changes`; if code owns it, forbid console edits.
- **Pin module and provider versions** and upgrade them as their own reviewed change.
- **Keep environments as copies of one definition.** Module inputs or Pulumi stacks make staging a smaller instance of production rather than a cousin of it.

## Where this stands in 2026

:::info Industry view

- **Terraform's current line.** The release index lists Terraform 1.16.4 as the newest stable release at the time of writing (dates not shown on that page).
- **Two models coexist.** Terraform's own configuration language and Pulumi's general-purpose-language programs both remain current, and Pulumi documents an optional managed backend for state, access control and policy.
- **Cloud platforms assume IaC.** The SageMaker AI page above steers production deployments at scale towards CloudFormation and IaC tooling, and the Kubernetes objects in the [previous chapter](/docs/mlops/platform/kubernetes-for-ml) are themselves declarative manifests of the same kind.
- **Plans are for people and machines.** Saved plans, exit codes and refresh-only runs make it straightforward to put the review step in a pipeline.
- **Not verified here.** Licensing changes among IaC tools, OpenTofu, and vendor-specific policy engines were not checked for this chapter.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> A plan shows `-/+ bucket.artifacts (name forces replacement)`. A colleague says it is just a rename. What do you tell them?</summary>

Replacement means the old bucket is destroyed and a new one created. Anything stored in the old bucket is gone unless you migrate it first. Add `prevent_destroy` to stop an accidental replacement, plan the migration, and then make the change deliberately.

</details>

<details>
<summary><strong>Q2.</strong> Explain the three things an IaC tool compares and what each one answers.</summary>

Configuration answers what you want; state answers what the tool believes it created; reality answers what actually exists. Plan refreshes state from reality, then compares configuration with state, which is why drift shows up as a diff.

</details>

<details>
<summary><strong>Q3.</strong> Someone scaled the cluster to 4 nodes in the console during an incident, and the code still says 2. Give two options and when each is right.</summary>

Accept the drift: apply a refresh-only plan to record the 4 in state, then update the code to 4 so they agree. Or overwrite: run a normal apply, which sets it back to 2. Accept when the change was correct and should stay, overwrite when it was a temporary fix. If an autoscaler owns the value, use `ignore_changes` instead.

</details>

<details>
<summary><strong>Q4.</strong> Two engineers run apply at the same time with local state files. What goes wrong, and what is the fix?</summary>

Each starts from the same view and writes back only their own change, so one write erases the other (block 2) and the next plan wants to re-create something that already exists. Use a shared remote backend with locking so the second apply waits.

</details>

<details>
<summary><strong>Q5.</strong> Why does a second apply of unchanged code do nothing, and why does that matter for pipelines?</summary>

Declarative tools compare desired and recorded state and act only on the difference. With no difference the plan is empty (block 1, step 2). That idempotence means a pipeline can run on every merge and a nightly `-detailed-exitcode` check can treat exit code 2 as a signal that something changed outside the code.

</details>

<details>
<summary><strong>Q6.</strong> When would you choose Pulumi over Terraform, and what stays the same?</summary>

Choose Pulumi when the team wants to write infrastructure in a general-purpose language such as Python or TypeScript with loops, functions and tests. Choose Terraform when the team prefers a dedicated configuration language and its ecosystem. In both, you preview changes, keep state, separate environments and review before applying.

</details>

## Further reading

- [Terraform: the plan command](https://developer.hashicorp.com/terraform/cli/commands/plan): the compare steps, saved plans, refresh-only, destroy mode and exit codes.
- [Terraform: state](https://developer.hashicorp.com/terraform/language/state): what state is for, remote backends, locking and secrets.
- [Terraform: how drift is detected and handled](https://developer.hashicorp.com/terraform/tutorials/state/resource-drift): the refresh-only walkthrough.
- [Terraform: modules](https://developer.hashicorp.com/terraform/language/modules): root and child modules, sources and versions.
- [Terraform: the lifecycle meta-argument](https://developer.hashicorp.com/terraform/language/meta-arguments/lifecycle): create_before_destroy, prevent_destroy, ignore_changes and replace_triggered_by.
- [Terraform release index](https://releases.hashicorp.com/terraform/): the versions behind the 2026 note.
- [Pulumi: core concepts](https://www.pulumi.com/docs/iac/concepts/): programs, stacks, resources and the optional managed backend.
- [Amazon SageMaker AI: deploy models for inference](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html): the IaC route for production deployments.

## Check yourself

- I can explain configuration, state and reality, and say which pair plan compares.
- I can read plan symbols and say why `-/+` is a deletion.
- I can explain what state holds, why it needs locking and why it stays out of version control.
- I can handle drift by accepting it, overwriting it or marking the attribute as owned elsewhere.
- I can use `prevent_destroy` and `ignore_changes` for the right resources.
- I can explain how references create a dependency graph and why independent resources apply in parallel.
- I can compare Terraform and Pulumi without confusing how they write the desired state with what a plan does.
