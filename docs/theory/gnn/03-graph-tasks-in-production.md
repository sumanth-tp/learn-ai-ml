---
id: gnn-graph-tasks-production
title: "Graph Tasks in Production: Node, Link and Graph Prediction at Scale"
sidebar_label: "3 · Graph tasks in production"
sidebar_position: 3
slug: /theory/gnn/graph-tasks-in-production
description: "Fraud rings as node classification, recommendation as link prediction, and whole-graph labels, each with its evaluation trap, plus over-smoothing with depth and neighbour sampling for graphs that do not fit in memory, measured on simulated graphs with known structure."
tags: [graph-neural-networks, fraud-detection, link-prediction, over-smoothing, neighbour-sampling, readout, graph-classification]
---

import Infographic from '@site/src/components/Infographic';
import NeighbourSamplingLab from '@site/src/components/viz/NeighbourSamplingLab';

**In one line.** Putting a graph model into production means picking the task (node, link or whole graph), evaluating it in a way that does not leak the answer through the edges, keeping the network shallow enough to avoid over-smoothing, and sampling neighbours so that a batch does not drag in the whole graph.

:::tip Before you start
- **You should already know** the three layers and what the graph does to a classifier ([GCN, GraphSAGE and GAT](/docs/theory/gnn/gcn-graphsage-gat)) and why accuracy misleads on rare classes ([model evaluation](/docs/theory/ml/model-evaluation)).
- **Reading time:** about 55 minutes. The code takes about three minutes to run on a CPU.
- **After this chapter you can** frame a problem as a node, link or graph task, choose a metric that suits it, split edges without leaking, say how deep a graph network can go before its nodes become indistinguishable, and estimate how many nodes a mini-batch needs with and without sampling.
:::

:::note Not from a lecture
This chapter was written for this site from the sources under Go deeper, including the public Stanford CS224W course whose Fall 2026 schedule was read. Every number comes from the code below, on simulated graphs where the truth is known. Environment: Python 3.14, PyTorch 2.14.1 on the CPU, PyTorch Geometric 2.8.0.post1, NetworkX 3.6.1, scikit-learn 1.9.1. Sources were opened on 8 October 2026.
:::

## In 30 seconds

A bank has accounts, and accounts send money to each other. A gang of fraudsters opens a dozen accounts that pay each other in circles. Each account looks ordinary on its own: the amounts are modest and the ages are normal. Together they form a tight cluster that a clerk would spot in a drawing in seconds. A graph model can read that drawing.

Three questions arise in practice. Is this node bad (node task)? Will these two nodes connect, as in a recommendation (link task)? What is this whole network (graph task)? Each has a way to look better in testing than it will be in production, and each runs into a size problem, because a graph of a hundred thousand accounts cannot be loaded for every prediction.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Node task | Predict a label for each node | Is this account in a fraud ring? |
| Link task | Predict whether two nodes are, or will be, joined | Will this user click this item? |
| Graph task | Predict one label for a whole graph | Is this molecule toxic? |
| Average precision | A ranking score for rare positives, between 0 and 1 | 0.852 against 0.05 for a random ranking |
| Negative sampling | Drawing non-edges as examples of "no link" | Random pairs that are not edges |
| Readout | A rule that combines all node vectors into one graph vector | Sum, mean or max |
| Over-smoothing | Many layers make every node's vector the same | Similarity 1.000 at 8 layers |
| Skip connection | Adding a layer's input to its output | h plus step |
| Fan-out | How many neighbours are sampled per node at a hop | 10 then 5 |
| Transductive and inductive | Test nodes are in the training graph, or are new | Same accounts or tomorrow's accounts |

## The idea in plain words

**Node tasks: fraud rings.** The label sits on a node, and the graph supplies the clues. A fraud ring is a set of accounts whose neighbours are mostly each other. Averaging the features of an account's neighbours puts a fraud signal, weak in any one member, into every other member's representation. Fraud is rare, so the metric must be built for rare positives. Average precision (AP) scores a ranking: take the accounts in order of suspicion, and average the precision at the position of each true fraud case.

**Link tasks: recommendation.** The unit is a pair. A graph network gives every node a vector, and a decoder scores a pair, most simply by the dot product of the two vectors. Train on known edges as positives and random non-edges as negatives. The trap is leakage: if the edges you test on are present in the graph the network passes messages over, the network has seen the answer. Hold the test edges out of the graph as well as out of the loss.

**Graph tasks.** One label per graph, such as a molecule. Each node gets a vector from the layers, and a readout turns the table of node vectors into one vector. The choice matters. Sum keeps the size of the graph; mean and max do not. A property that depends on how big the graph is cannot be learned from a mean.

**Depth and scale.** Two engineering limits cut across all three. Stacking layers lets a node see further, but repeated averaging makes nodes alike, so there is a depth beyond which accuracy collapses. And the receptive field grows by about the mean degree at every layer, so the data a batch needs explodes with depth. Sampling a fixed number of neighbours per hop, the idea from GraphSAGE, caps the growth.

<Infographic src="/img/gnn/task-types.svg" alt="Three columns. Node task, fraud rings: average precision 0.228 for features only, 0.603 with degree and clustering, 0.852 for a graph network, 0.176 for the same network with shuffled edges. Link task: AUC 0.604 for common neighbours and for Adamic-Adar, 0.762 for an honest graph network and 0.813 when test edges are left in the graph. Graph task: accuracy 0.842 with sum readout, 0.594 with mean and max, 0.571 for guessing the majority." caption="Read each column's table, then its trap. Node: shuffling the edges drops the model below the features alone. Link: the leaked score looks better than the honest one. Graph: only the sum readout sees size." />

## Worked example, step by step

**Average precision.** Ten accounts, two of them fraudulent. The model ranks the fraud cases first and fourth.

1. **Precision at the first fraud case:** 1 fraud among the top 1, so 1/1 = 1.0.
2. **Precision at the second fraud case:** 2 frauds among the top 4, so 2/4 = 0.5.
3. **Average the two:** (1.0 + 0.5) / 2 = 0.75. A random ranking would score about the base rate, 0.2, here.

**Adamic-Adar for a link.** Two users share two friends. One friend has 2 connections and the other has 10.

4. **Weight each shared friend** by 1 over the natural logarithm of their number of connections: 1/ln 2 = 1.443 and 1/ln 10 = 0.434.
5. **Add:** 1.443 + 0.434 = 1.877. A shared friend with few connections is stronger evidence than a shared hub. Plain common neighbours would just give 2.

**Neighbour sampling.** A batch of 64 target nodes, two layers, fan-outs of 10 then 5, in a graph with a mean degree of 10.

6. **Targets:** 64.
7. **First hop:** 64 x 10 = 640 sampled neighbours.
8. **Second hop:** 640 x 5 = 3,200 sampled neighbours of those.
9. **Total at most:** 64 + 640 + 3,200 = 3,904 nodes. Taking all neighbours it would be 64 + 640 + 6,400 = 7,104 on the same tree formula. Block 4 measures what really happens in a graph with hubs.

<Infographic src="/img/gnn/sampling-footprint.svg" alt="Bars for the number of nodes loaded to embed 64 targets in a 100,000-node graph: 862 for one layer taking all neighbours, 20,871 for two layers, 94,068 for three layers, 2,530 for two layers sampling 10 then 5, and 10,378 for three layers sampling 10, 5 and 5." caption="The red bars are what happens with all neighbours: a third layer pulls in 94 per cent of the graph for 64 targets. The green bars are the same targets with sampling." />

## How it works

### How do I split data for a node task?

For a fixed set of accounts, the transductive setting, you can mask labels: all accounts and all edges are in the graph, but only some accounts have labels the model learns from, and you test on the rest. That is what block 1 does. In production the test is usually inductive: tomorrow's new accounts must be scored with a model trained on yesterday's graph. Then split by time, so that every training node and edge predates every test node. Feature leakage works through edges too: a neighbour's label is a legitimate input only if it was known at prediction time.

### How do I split data for a link task?

Hold out a share of the edges as test positives and remove them from the graph that the layers use. Draw an equal number of non-edges as test negatives. For training, use the remaining edges as positives and sample negatives afresh each step. Block 2 compares the honest split with one where the held-out edges stay in the message-passing graph.

Most random pairs in a sparse graph are far apart, so random negatives are easy. A common refinement is hard negatives (pairs that share some context), with ranking metrics measured on a candidate list rather than AUC on random pairs.

### Why are graph baselines worth building?

Common neighbours and Adamic-Adar need no training and no features. They fail on pairs with no shared neighbour, and in a sparse graph that is most held-out edges: block 2 finds 59.3 per cent of held-out edges whose ends share no neighbour in the training graph. A learned model can score such pairs from node features, which is its advantage here.

### What do the three readouts keep?

With node vectors $h_1, \ldots, h_n$, sum gives $\sum_v h_v$, mean gives that divided by $n$, and max takes the largest entry per feature. Sum is the only one that grows with $n$. If every node in two different-sized graphs looks locally the same, as in regular graphs, only the sum can separate them. This echoes the aggregation lesson of the first chapter, applied to the whole graph.

### Why does depth hurt, and what helps?

Each layer replaces a node's vector by an average over its neighbourhood. Repeated, this converges toward a vector shared by every node in the connected component. A skip connection adds the layer's input back to its output, so each node keeps a copy of its own earlier vector, which slows the convergence. Block 3 measures both. Other remedies are normalising layers, dropping edges during training and concatenating the outputs of all layers.

### How does neighbour sampling work, and what does it cost?

For each target node, draw at most $f_1$ neighbours at hop 1. For each of those, draw at most $f_2$ at hop 2, and so on. The batch then contains the targets, their sampled neighbours and the sampled neighbours of those, and the computation is a small tree per target. Two costs follow. The neighbour average is now an estimate, so predictions are noisier. And nodes with fewer neighbours than the fan-out are unaffected, so sampling bites hardest at the hubs, which are also the nodes that blow up the receptive field.

<Infographic src="/img/gnn/depth-collapse.svg" alt="A table of accuracy and similarity of different-class nodes by number of layers. A plain stack scores 0.948, 0.985 and 0.890 at 1, 2 and 4 layers, then 0.347, 0.345 and 0.318 at 8, 16 and 32 layers with similarity 1.000. With skip connections accuracy is 0.937, 0.965, 0.982, 0.972, 0.917 and then 0.485 at 32 layers." caption="Read the plain accuracy column down: it holds to 4 layers and then falls to chance at 8, where the similarity column hits 1.000. The skip column holds much longer." />

## Code you can run

All five blocks are self-contained and run on the CPU. Blocks 1 and 5 repeat small layer classes so that each runs alone. The whole chapter takes about three minutes.

### 1. Fraud rings: a node task

A preferential-attachment network of 1,900 ordinary accounts plus 12 rings of 8 fraudulent accounts. Ring members are densely linked (probability 0.6) and each also links to two ordinary accounts. Each account has six features, shifted by 0.5 for fraud, so the features alone are weak. We train on a random half of the accounts and score the other half by average precision, over five graphs. A control reruns the network on the same graph with its edges shuffled while the degrees are kept.

```python
import networkx as nx
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

def make_accounts(seed, normal=1900, rings=12, ring_size=8):
    rng = np.random.default_rng(seed)
    g = nx.barabasi_albert_graph(normal, 3, seed=seed)
    labels = [0] * normal
    for r in range(rings):
        members = list(range(len(labels), len(labels) + ring_size))
        labels += [1] * ring_size
        g.add_nodes_from(members)
        for i in members:
            for j in members:
                if i < j and rng.random() < 0.6:
                    g.add_edge(i, j)
            for other in rng.choice(normal, 2, replace=False):
                g.add_edge(i, int(other))
    y = np.array(labels)
    x = rng.normal(size=(len(y), 6)) + 0.5 * y[:, None]
    return g, x, y

class Layer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_self, self.w_nb = nn.Linear(d_in, d_out, bias=False), nn.Linear(d_in, d_out)

    def forward(self, x, adj):
        return self.w_self(x) + self.w_nb(adj @ x)

class Net(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.l1, self.l2 = Layer(d_in, 16), Layer(16, 1)

    def forward(self, x, adj):
        return self.l2(F.dropout(F.relu(self.l1(x, adj)), 0.3, self.training), adj).squeeze(-1)

def gnn_scores(g, x, y, train, seed):
    n = len(y)
    a = torch.tensor(nx.to_numpy_array(g, nodelist=range(n)), dtype=torch.float32)
    a = a / a.sum(1, keepdim=True).clamp(min=1)
    tx, ty = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(seed)
    net = Net(x.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-4)
    weight = torch.tensor(8.0)
    for _ in range(150):
        net.train()
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(net(tx, a)[train], ty[train], pos_weight=weight).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(tx, a).numpy()

rows = {"features only": [], "features + degree and clustering": [], "GraphSAGE-style GNN": [], "same GNN, edges shuffled": []}
for seed in range(5):
    g, x, y = make_accounts(seed)
    rng = np.random.default_rng(100 + seed)
    train = rng.random(len(y)) < 0.5
    test = ~train
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x[train], y[train])
    rows["features only"].append(average_precision_score(y[test], lr.predict_proba(x[test])[:, 1]))
    deg = np.array([g.degree(i) for i in range(len(y))])
    clus = np.array([nx.clustering(g, i) for i in range(len(y))])
    xe = np.column_stack([x, np.log1p(deg), clus])
    lr2 = LogisticRegression(max_iter=1000, class_weight="balanced").fit(xe[train], y[train])
    rows["features + degree and clustering"].append(average_precision_score(y[test], lr2.predict_proba(xe[test])[:, 1]))
    rows["GraphSAGE-style GNN"].append(average_precision_score(y[test], gnn_scores(g, x, y, torch.tensor(train), seed)[test]))
    shuffled = g.copy()
    nx.double_edge_swap(shuffled, nswap=3 * g.number_of_edges(), max_tries=10 ** 6, seed=seed)
    rows["same GNN, edges shuffled"].append(average_precision_score(y[test], gnn_scores(shuffled, x, y, torch.tensor(train), seed)[test]))
g, x, y = make_accounts(0)
print(f"{len(y)} accounts, {g.number_of_edges()} edges, {y.sum()} fraudulent ({y.mean():.3f}) in 12 rings of 8")
print("average precision on the held-out half (a random ranking scores about 0.05); mean and sd over 5 graphs")
for name, v in rows.items():
    print(f"  {name:36s} {np.mean(v):.3f}  +/- {np.std(v):.3f}")
```

**Reading the output.** Of 1,996 accounts, 96 (4.8 per cent) are fraudulent, so a random ranking scores an AP of about 0.05. Logistic regression on the six features gets 0.228. Adding two hand-made structure features, the log degree and the clustering coefficient, lifts it to 0.603. A two-layer graph network that sees neighbours' features gets 0.852, with a spread of 0.072 over five graphs.

The control is the part to remember. The same network on the shuffled graph scores 0.176, below the features alone. The 0.852 does not come from the network's capacity: it comes from the true edges. A shuffled graph is not neutral; it mixes the features of unrelated accounts into every node, which is worse than ignoring the graph.

The comparison also shows that the hand features are not useless: they capture a good share of the ring signal through the clustering coefficient at no training cost. A production team should build that baseline first and ask what the network adds beyond it.

**Line by line.**

- `pos_weight=weight` with 8.0 raises the loss on the rare class so the model does not just predict "ordinary".
- `a / a.sum(1, keepdim=True).clamp(min=1)` row-normalises the adjacency, so the layer averages neighbour features.
- `nx.double_edge_swap` rewires edges while keeping every node's degree, so the control has the same degree distribution but none of the ring structure.
- The held-out mask is `~train`: the random half of the nodes whose labels the loss never saw, although their features and edges are in the graph, the transductive setting.

### 2. Recommendation as link prediction, and the leak

A graph of three communities of 100 nodes. We hold out 20 per cent of edges as test positives, draw an equal number of non-edges as negatives, and compare two training-free scores with a graph encoder that scores a pair by a dot product. The last variant leaves the test edges in the graph the encoder passes messages over.

```python
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

def make_graph(seed, sizes=(100, 100, 100)):
    p = [[0.10 if i == j else 0.01 for j in range(3)] for i in range(3)]
    g = nx.stochastic_block_model(list(sizes), p, seed=seed)
    labels = np.repeat(np.arange(3), sizes)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3, 16))[labels] + 2.0 * rng.normal(size=(len(labels), 16))
    return g, torch.tensor(x, dtype=torch.float32), rng

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_self, self.a_nb = nn.Linear(16, 32, bias=False), nn.Linear(16, 32)
        self.b_self, self.b_nb = nn.Linear(32, 16, bias=False), nn.Linear(32, 16)

    def forward(self, x, adj):
        h = F.relu(self.a_self(x) + self.a_nb(adj @ x))
        return self.b_self(h) + self.b_nb(adj @ h)

def normalised(g, n):
    a = torch.tensor(nx.to_numpy_array(g, nodelist=range(n)), dtype=torch.float32)
    return a / a.sum(1, keepdim=True).clamp(min=1)

def fit_and_score(x, train_edges, message_graph, test_pairs, n, seed):
    torch.manual_seed(seed)
    enc = Encoder()
    opt = torch.optim.Adam(enc.parameters(), lr=0.01)
    adj = normalised(message_graph, n)
    pos = torch.tensor(train_edges)
    for _ in range(100):
        opt.zero_grad()
        z = enc(x, adj)
        neg = torch.randint(0, n, pos.shape)
        logits = torch.cat([(z[pos[:, 0]] * z[pos[:, 1]]).sum(1), (z[neg[:, 0]] * z[neg[:, 1]]).sum(1)])
        target = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
        F.binary_cross_entropy_with_logits(logits, target).backward()
        opt.step()
    with torch.no_grad():
        z = enc(x, adj)
        t = torch.tensor(test_pairs)
        return (z[t[:, 0]] * z[t[:, 1]]).sum(1).numpy()

no_overlap = []
results = {k: [] for k in ("common neighbours", "Adamic-Adar", "GNN encoder, honest", "GNN encoder, test edges in the graph")}
for seed in range(5):
    g, x, rng = make_graph(seed)
    n = g.number_of_nodes()
    edges = list(g.edges())
    order = rng.permutation(len(edges))
    cut = int(0.8 * len(edges))
    train_edges = [edges[i] for i in order[:cut]]
    test_pos = [edges[i] for i in order[cut:]]
    non_edges = [(u, v) for u, v in (rng.integers(0, n, 2) for _ in range(20 * len(test_pos))) if u != v and not g.has_edge(u, v)][: len(test_pos)]
    pairs = test_pos + non_edges
    y = np.array([1] * len(test_pos) + [0] * len(non_edges))
    train_graph = nx.Graph(train_edges)
    train_graph.add_nodes_from(range(n))
    cn = [len(list(nx.common_neighbors(train_graph, u, v))) for u, v in pairs]
    aa = [s for _, _, s in nx.adamic_adar_index(train_graph, pairs)]
    no_overlap.append(np.mean([c == 0 for c, label in zip(cn, y) if label == 1]))
    results["common neighbours"].append(roc_auc_score(y, cn))
    results["Adamic-Adar"].append(roc_auc_score(y, aa))
    results["GNN encoder, honest"].append(roc_auc_score(y, fit_and_score(x, train_edges, train_graph, pairs, n, seed)))
    results["GNN encoder, test edges in the graph"].append(roc_auc_score(y, fit_and_score(x, train_edges, g, pairs, n, seed)))
print(f"{n} nodes, {len(edges)} edges; 20% of edges held out as test positives, an equal number of non-edges as negatives")
print("AUC on the held-out pairs, mean +/- sd over 5 graphs")
for name, v in results.items():
    print(f"  {name:38s} {np.mean(v):.3f} +/- {np.std(v):.3f}")
print(f"held-out edges whose two ends share no neighbour in the training graph: {np.mean(no_overlap):.3f}")
```

**Reading the output.** The graph has 300 nodes and 1,780 edges. Common neighbours and Adamic-Adar both reach an AUC of 0.604, identical to three digits, because in a graph this sparse the shared friends rarely differ in degree enough to change the order. They fail on 59.3 per cent of held-out edges, whose ends share no neighbour in the training graph and so score zero, tied with the negatives.

The encoder scores 0.762 with the honest split. It can place two nodes from the same community close together using their features, even with no shared neighbour. With the test edges left in the message-passing graph it scores 0.813, 0.051 higher, for no better model. That is the cost of the leak at this size. A leak does not need to be large to flatter a result: the number a team reports would have been wrong by five points.

**Line by line.**

- `train_graph = nx.Graph(train_edges)` is the graph the honest model passes messages over; the leaky variant passes `g`, the full graph.
- `torch.randint(0, n, pos.shape)` draws random pairs as negatives afresh at every step. About 4 per cent of the draws are real edges at this density, which adds a little noise.
- `(z[pos[:, 0]] * z[pos[:, 1]]).sum(1)` is the dot-product decoder: a large value means a likely link.

### 3. How deep can a graph network go?

A graph network with a variable number of hidden graph layers on the three-community graph of the previous chapter, 30 labels, with and without skip connections. Besides accuracy we measure how alike nodes of different classes become: the mean cosine similarity between the final vectors of nodes in different classes. A value near 0.3 means the classes are separate; 1.000 means indistinguishable.

```python
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def make_graph(seed, sizes=(80, 80, 80), noise=2.0):
    p = [[0.08 if i == j else 0.01 for j in range(3)] for i in range(3)]
    g = nx.stochastic_block_model(list(sizes), p, seed=seed)
    adj = torch.tensor(nx.to_numpy_array(g), dtype=torch.float32)
    labels = torch.tensor(np.repeat(np.arange(3), sizes))
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3, 16))[labels.numpy()] + noise * rng.normal(size=(len(labels), 16))
    return adj, torch.tensor(x, dtype=torch.float32), labels

class DeepGCN(nn.Module):
    def __init__(self, depth, residual, hidden=16):
        super().__init__()
        self.inp = nn.Linear(16, hidden)
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(depth))
        self.out = nn.Linear(hidden, 3)
        self.residual = residual

    def forward(self, x, s):
        h = F.relu(self.inp(F.dropout(x, 0.5, self.training)))
        for layer in self.layers:
            step = F.relu(s @ layer(h))
            h = h + step if self.residual else step
            h = F.dropout(h, 0.2, self.training)
        return self.out(h), h

def run(depth, residual, seed):
    adj, x, y = make_graph(seed)
    a_hat = adj + torch.eye(len(y))
    d = a_hat.sum(1)
    s = a_hat / torch.sqrt(d[:, None] * d[None, :])
    g = torch.Generator().manual_seed(seed)
    train = torch.cat([torch.nonzero(y == c).flatten()[torch.randperm(80, generator=g)[:10]] for c in range(3)])
    mask = torch.ones(len(y), dtype=torch.bool)
    mask[train] = False
    test = torch.nonzero(mask).flatten()[60:]
    torch.manual_seed(seed)
    net = DeepGCN(depth, residual)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        net.train()
        opt.zero_grad()
        F.cross_entropy(net(x, s)[0][train], y[train]).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        logits, h = net(x, s)
    h = F.normalize(h, dim=1)
    cos = h @ h.T
    other = (y[:, None] != y[None, :])
    return (logits.argmax(1)[test] == y[test]).float().mean().item(), cos[other].mean().item()

print("depth   plain: accuracy  similarity of different-class nodes   with skip connections: accuracy  similarity")
for depth in (1, 2, 4, 8, 16, 32):
    plain = np.array([run(depth, False, s) for s in range(4)])
    skip = np.array([run(depth, True, s) for s in range(4)])
    print(f"{depth:5d} {plain[:, 0].mean():16.3f} {plain[:, 1].mean():30.3f} {skip[:, 0].mean():28.3f} {skip[:, 1].mean():12.3f}")
```

**Reading the output.** The plain stack is accurate at 1 and 2 layers (0.948, 0.985), loses some at 4 (0.890), and then collapses: 0.347 at 8 layers, 0.345 at 16 and 0.318 at 32, which is chance for three classes. At the same time the similarity of nodes in different classes reaches 1.000 at 8 layers: every node has the same vector. The collapse is sudden, not gradual, between 4 and 8.

Skip connections keep the network working: 0.982 at 4 layers, 0.972 at 8 and 0.917 at 16. At 32 layers it too fails (0.485, similarity 0.781). So skip connections move the wall rather than remove it. Notice that the best plain result, 0.985 at 2 layers, is about as good as any deeper model: depth earned nothing here.

**Line by line.**

- `h = h + step if self.residual else step` is the skip connection; nothing else changes between the two columns.
- `F.normalize(h, dim=1)` then `h @ h.T` gives all pairwise cosine similarities; `other` selects pairs from different classes.
- Each depth uses 4 seeds, so the numbers are averages with a visible spread, not single runs.

### 4. Neighbour sampling at scale

A graph of 100,000 nodes generated by preferential attachment, so it has hubs like a real network. For 64 random targets we count how many nodes must be loaded to compute their embeddings, with all neighbours and with sampling, and compare with the tree formula from the worked example.

```python
import time

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

g = nx.barabasi_albert_graph(100_000, 5, seed=0)
indptr = np.zeros(g.number_of_nodes() + 1, dtype=np.int64)
indices = []
for v in range(g.number_of_nodes()):
    nb = list(g[v])
    indices.extend(nb)
    indptr[v + 1] = indptr[v] + len(nb)
indices = np.array(indices)
degree = np.diff(indptr)
print(f"{g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges, mean degree {degree.mean():.1f}, largest degree {degree.max():,}")

def neighbours(v):
    return indices[indptr[v]: indptr[v + 1]]

def tree_estimate(batch, fanouts, mean_degree):
    total, width = batch, batch
    for f in fanouts:
        width *= mean_degree if f is None else f
        total += width
    return min(total, g.number_of_nodes())

def receptive_field(seeds, fanouts, rng):
    layer = set(seeds)
    seen = set(seeds)
    for fanout in fanouts:
        nxt = set()
        for v in layer:
            nb = neighbours(v)
            if fanout is not None and len(nb) > fanout:
                nb = rng.choice(nb, fanout, replace=False)
            nxt.update(int(u) for u in nb)
        seen |= nxt
        layer = nxt
    return len(seen)

rng = np.random.default_rng(0)
seeds = rng.choice(g.number_of_nodes(), 64, replace=False).tolist()
print("\n64 target nodes: how many nodes must be loaded to compute their embeddings")
for label, fan in (("1 layer, all neighbours", [None]), ("2 layers, all neighbours", [None, None]), ("3 layers, all neighbours", [None, None, None]),
                   ("2 layers, sample 10 then 5", [10, 5]), ("3 layers, sample 10, 5, 5", [10, 5, 5])):
    start = time.perf_counter()
    count = receptive_field(seeds, fan, rng)
    estimate = tree_estimate(64, fan, degree.mean())
    print(f"  {label:28s} measured {count:9,d} ({count / g.number_of_nodes():6.1%} of the graph)  tree formula {estimate:9,.0f}  {time.perf_counter() - start:5.2f} s")

edge_index = torch.tensor(np.array(list(g.edges())).T, dtype=torch.long)
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
data = Data(x=torch.randn(g.number_of_nodes(), 16), edge_index=edge_index)
try:
    loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=64, input_nodes=torch.tensor(seeds))
    batch = next(iter(loader))
    print(f"\nPyG NeighborLoader, fan-out [10, 5], batch of 64: {batch.num_nodes:,} nodes, {batch.edge_index.shape[1]:,} edges")
except Exception as error:
    print("\nPyG NeighborLoader unavailable here:", type(error).__name__, str(error)[:90])
```

**Reading the output.** The graph has 499,975 edges, a mean degree of 10 and a hub with 1,141 neighbours. One layer needs 862 nodes (0.9 per cent of the graph), two layers 20,871 (20.9 per cent) and three layers 94,068 (94.1 per cent): for 64 targets, three layers touch nearly the whole graph. This is called the neighbourhood explosion.

Sampling 10 then 5 neighbours loads 2,530 nodes (2.5 per cent) for two layers, and 10, 5, 5 loads 10,378 (10.4 per cent) for three.

The tree formula is a rough guide with two failure directions. For all neighbours it gives 7,103 for two layers against 20,871 measured: it is three times too low, because neighbours of a node are likelier to be hubs than average nodes, so their own neighbour lists are longer than the mean degree. For sampling it overestimates, 3,904 against 2,530, because many nodes have fewer neighbours than the fan-out. Plan memory from measurements on your own graph.

PyTorch Geometric's `NeighborLoader` is the library tool for this. In this environment it raised an error, because it needs the optional `pyg-lib` or `torch-sparse` package, which is not installed. The block catches the error and says so; the sampler written above does the same job for the counting.

**Line by line.**

- `indptr` and `indices` are a compressed sparse row layout: the neighbours of node `v` are `indices[indptr[v]:indptr[v + 1]]`, which is how every graph library stores adjacency.
- `rng.choice(nb, fanout, replace=False)` draws a sample only for nodes with more neighbours than the fan-out.
- `seen |= nxt` counts each node once even if several targets reach it.

### 5. A whole-graph label and the readout

Random 3-regular graphs with 6 to 18 nodes. Every node has three neighbours and starts from the same feature, so locally every node looks identical in every graph. The label is whether the graph has 12 or more nodes. The same small network is trained with three readouts.

```python
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def make_graphs(count, seed):
    rng = np.random.default_rng(seed)
    graphs, labels = [], []
    for _ in range(count):
        n = 2 * int(rng.integers(3, 10))
        g = nx.random_regular_graph(3, n, seed=int(rng.integers(1 << 30)))
        a = torch.tensor(nx.to_numpy_array(g), dtype=torch.float32)
        graphs.append(a / a.sum(1, keepdim=True).clamp(min=1))
        labels.append(int(n >= 12))
    return graphs, torch.tensor(labels, dtype=torch.float32)

class GraphNet(nn.Module):
    def __init__(self, readout):
        super().__init__()
        self.readout = readout
        self.l1, self.l2 = nn.Linear(2, 16), nn.Linear(16, 16)
        self.out = nn.Linear(16, 1)

    def embed(self, mean_nb):
        h = torch.ones(len(mean_nb), 1)
        h = F.relu(self.l1(torch.cat([h, mean_nb @ h], dim=1)))
        h = F.relu(self.l2(h) + mean_nb @ h)
        return {"sum": h.sum(0), "mean": h.mean(0), "max": h.max(0).values}[self.readout]

    def forward(self, graphs):
        return self.out(torch.stack([self.embed(g) for g in graphs])).squeeze(-1)

def accuracy(readout, seed):
    train_g, train_y = make_graphs(300, seed)
    test_g, test_y = make_graphs(300, seed + 1000)
    torch.manual_seed(seed)
    net = GraphNet(readout)
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    for _ in range(300):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(net(train_g), train_y).backward()
        opt.step()
    with torch.no_grad():
        return ((net(test_g) > 0).float() == test_y).float().mean().item()

print("graph classification: is the graph large (12 or more nodes)? Every graph is 3-regular and every node starts with the same feature.")
print("readout   test accuracy, mean over 3 seeds")
for readout in ("sum", "mean", "max"):
    scores = [accuracy(readout, s) for s in range(3)]
    print(f"{readout:8s} {np.mean(scores):.3f}   (seeds: {', '.join(f'{v:.3f}' for v in scores)})")
```

**Reading the output.** Sum readout reaches 0.842. Mean and max both get 0.594, hardly above the 0.571 that guessing the larger class gives. In a regular graph with identical starting features every node's vector is the same, so the mean and the max are the same vector in a 6-node graph and in an 18-node graph, and they carry no size information whatever the network learns. The sum is the node vector times the number of nodes.

The sum is not perfect at 0.842. The input to its classifier is exactly proportional to the size, so a perfect rule exists, and this small network with 300 training steps did not find it. The point of the experiment is the gap between 0.842 and 0.594, not the ceiling.

**Line by line.**

- `nx.random_regular_graph(3, n)` makes every node's degree 3, which is what removes every other clue to size.
- `mean_nb` is the row-normalised adjacency, the same neighbour mean as before. The dictionary in `embed` selects the readout.
- The three readouts differ only in the `{"sum": ..., "mean": ..., "max": ...}` line; the network and data are identical.

## Try it yourself

The lab applies the tree formula of the worked example. Its defaults, 64 targets, mean degree 10 and fan-outs 10 and 5 over two layers, give the 3,904 of the formula and block 4. It is an upper-bound estimate: real graphs with hubs behave differently, as block 4 shows.

<NeighbourSamplingLab />

**What each control does.**

- **Target nodes per batch** is how many nodes you want embeddings for at once.
- **Layers** sets how many hops of neighbours are needed, 1 to 3.
- The three **Fan-out** sliders set how many neighbours are kept at each hop; only the first as many as there are layers are used.
- **Mean degree** is the average number of neighbours; **Feature values per node** converts nodes to megabytes at 4 bytes per value.

**Try it yourself.**

1. Leave the defaults and add a third layer. Taking all neighbours the batch grows from 7,104 to 71,104 nodes, while sampling grows from 3,904 to 19,904. Why: each extra hop multiplies the frontier by the mean degree, but sampling multiplies by only the fan-out.
2. Set the mean degree to 3 with three layers. Both counts are 2,560. Why: a node cannot have more sampled neighbours than it has, so sampling does nothing when the degree is already below the fan-out. Sampling helps at hubs, not in sparse graphs.
3. Set the batch to 512 with all three fan-outs at 30 and the mean degree at 40. Both counts reach the cap of 100,000, the whole graph. Why: the tree is far larger than the graph, so every node is loaded. A fan-out that large is not sampling.

## Designing with it

1. **Pick the unit of prediction first.** Node, pair or graph decides the model head, the metric and the split. Decide whether tomorrow's prediction is transductive or inductive and split by time if it is inductive.
2. **Build the no-learning baseline.** Hand-made structure features for nodes, common neighbours for links, size and degree statistics for graphs. Block 1's baseline is at 0.603 against 0.852.
3. **Run the shuffled-edge control.** If a model with its edges shuffled does as well as the real one, the graph is not the reason.
4. **Keep the network shallow.** Two or three layers is where block 3 is stable; add skip connections before you go further and measure the node similarity as well as accuracy.
5. **Plan for the batch, not the graph.** Measure the receptive field on your graph with your fan-outs, then set memory from that measurement.
6. **Serve embeddings, not graphs, where you can.** Computing node vectors in batch and looking them up is cheap. Sampling at request time costs latency. Freshness is the trade-off: a precomputed vector is as old as the last batch.

## Where this stands in 2026

The CS224W Fall 2026 schedule shows where practice has moved: lectures on graph neural networks for recommender systems, heterogeneous graphs with several node and edge types, knowledge graphs, relational deep learning and foundation models for knowledge graphs. These all keep the loop of this chapter, task, split, baseline, depth and sampling. The libraries have matured: PyTorch Geometric lists sampling support through optional packages, which is why `NeighborLoader` did not run in this environment. The part that has not changed is the evaluation discipline, since fraud adapts, edges leak and AUC on random pairs flatters a recommender.

## Common mistakes

1. **Reporting accuracy for a rare class.** Predicting "ordinary" for every account is 95.2 per cent accurate in block 1. Report average precision, or precision at a review budget.
2. **Leaving test edges in the graph.** It is easy to build one graph and split only the labels. Block 2's leak added 0.051 AUC. Remove held-out edges from the message-passing graph.
3. **Trusting a graph model without a shuffled-edge control.** Block 1's shuffled graph scored 0.176, below the features alone. If the control matches the real graph, the edges are not helping.
4. **Going deep.** Block 3's plain network was at chance from 8 layers. Start with two.
5. **Using mean readout for a size-dependent label.** Block 5: 0.594 against 0.842. Match the readout to the label, or add the node count as a feature.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> A model predicts "not fraud" for every account in block 1. What is its accuracy and its average precision?</summary>

Its accuracy is 1 - 0.048 = 0.952. Its ranking has no information, so its average precision is about the base rate, 0.048. This is why accuracy is the wrong summary for rare positives.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> What is the average precision if the two fraud cases among ten accounts are ranked second and third?</summary>

Precision at rank 2 is 1/2 = 0.5, and at rank 3 it is 2/3 = 0.667. The average is (0.5 + 0.667) / 2 = 0.583.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> Compute the Adamic-Adar score for two users who share three friends with 2, 4 and 8 connections.</summary>

1/ln 2 + 1/ln 4 + 1/ln 8 = 1.443 + 0.721 + 0.481 = 2.645. Plain common neighbours gives 3. The score weights rare friends more.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> Why did the shuffled-edge control in block 1 score below the features-only model?</summary>

The network averages each node's neighbours' features. With the true graph, the neighbours of a ring member are mostly other ring members, so the average carries a fraud signal. With shuffled edges the neighbours are random accounts, and the average mixes unrelated features into every node, adding noise to a feature set that was weak but informative.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> The tree formula gave 7,103 nodes for two layers and measured 20,871. Why is the measurement three times higher, and what does it mean for capacity planning?</summary>

A neighbour of a random node is more likely to be a high-degree node than a random node is, because high-degree nodes have more edges to arrive by. In preferential-attachment graphs this is strong. The second hop therefore expands by more than the mean degree, so a formula built on the mean underestimates. Measure the receptive field on a sample of your own targets and size memory for a high percentile, not the mean.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> In block 5, all node vectors in a 3-regular graph are identical. Show that the mean readout cannot distinguish a 6-node from an 18-node graph, and that the sum can.</summary>

Every node has three neighbours and starts from the same feature, so after any number of layers every node has the same vector $v$. The mean of $n$ copies of $v$ is $v$ for every $n$, so the mean readout is the same for both graphs; the max is the same. The sum is $n v$: $6v$ against $18v$, which differ by a factor of 3 and can be separated by a threshold on its scale.

</details>

## Go deeper

All sources were opened on 8 October 2026.

- [Stanford CS224W, Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/): the Fall 2026 schedule, with lectures on GNN augmentation and training, heterogeneous graphs, knowledge graphs, GNNs for recommender systems and relational deep learning. Slides are public.
- Hamilton WL, Ying R and Leskovec J, "Inductive Representation Learning on Large Graphs", arXiv 1706.02216, latest version v4 of 10 September 2018, NIPS 2017. Neighbour sampling and the inductive setting.
- Li Q, Han Z and Wu X-M, "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning", arXiv 1801.07606, AAAI 2018. Graph convolution as Laplacian smoothing and the over-smoothing warning.
- Xu K, Hu W, Leskovec J and Jegelka S, "How Powerful are Graph Neural Networks?", arXiv 1810.00826, v3 of 22 February 2019. Why the sum aggregation and readout keep more information.
- [PyTorch Geometric on PyPI](https://pypi.org/project/torch-geometric/): release 2.8.0.post1 of 20 July 2026. Its listing names `pyg-lib` as an optional library for graph sampling routines; that package was not installed here, so `NeighborLoader` was not run.
- [NetworkX documentation](https://networkx.org/): `barabasi_albert_graph`, `stochastic_block_model`, `adamic_adar_index`, `double_edge_swap`. Version 3.6.1 was run.

## Check yourself

- I can frame a problem as a node, link or graph task and choose a metric for it.
- I can split edges for link prediction without leaking, and say what the leak cost in block 2.
- I can design a shuffled-edge control and say what it shows.
- I can explain over-smoothing, say where it set in here and what delays it.
- I can estimate how many nodes a batch needs, with and without sampling, and say why the estimate is rough on graphs with hubs.

## Where to go next

This is the last chapter of the graph series. Go back to [graphs and message passing](/docs/theory/gnn/graphs-and-message-passing) for the expressiveness limits that explain block 5, or on to [retrieval, ranking and reranking](/docs/theory/recsys/retrieval-ranking-and-reranking), where candidate lists and hard negatives are treated for recommenders. A related chapter: [collaborative filtering](/docs/theory/recsys/collaborative-filtering), the user-item graph that link prediction generalises.
