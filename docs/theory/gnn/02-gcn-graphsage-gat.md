---
id: gnn-gcn-graphsage-gat
title: "GCN, GraphSAGE and GAT from Scratch"
sidebar_label: "2 · GCN, GraphSAGE and GAT"
sidebar_position: 2
slug: /theory/gnn/gcn-graphsage-gat
description: "Three graph neural network layers written from scratch in torch and numpy, checked against PyTorch Geometric to 1e-7, trained on a small community graph, and tested as the graph gets noisier, with the honest result that attention learned almost nothing here."
tags: [graph-neural-networks, gcn, graphsage, gat, attention, torch, pytorch-geometric, homophily]
---

import Infographic from '@site/src/components/Infographic';
import GatAttentionLab from '@site/src/components/viz/GatAttentionLab';

**In one line.** The three classic graph layers differ in one choice, how much a node listens to each neighbour: GCN fixes the weights by degree, GraphSAGE averages the neighbours and keeps the node's own value apart, and GAT learns a weight for every edge.

:::tip Before you start
- **You should already know** one round of message passing, adjacency matrices and the self-loop normalisation ([graphs and message passing](/docs/theory/gnn/graphs-and-message-passing)), and the attention idea of scoring inputs and normalising with a softmax ([what self-attention is](/docs/theory/dnn/what-self-attention-is)).
- **Reading time:** about 50 minutes, plus about a minute to run the code.
- **After this chapter you can** write each of the three layers in a few lines of torch, say what every parameter does, check a layer against a library, train a small node classifier and predict when the graph will help it and when it will hurt.
:::

:::note Not from a lecture
This chapter was written for this site from the three original papers and the Stanford CS224W course, whose Fall 2026 schedule lists graph neural networks, a general perspective on them and a lecture on their theory. Every number is printed by the code. Environment: Python 3.14, PyTorch 2.14.1 on the CPU, PyTorch Geometric 2.8.0.post1 (released 20 July 2026), NumPy 2.5.3, NetworkX 3.6.1. Sources were opened on 8 October 2026.
:::

## In 30 seconds

Imagine asking your friends for advice. You could treat every friend equally, you could weigh each by how many other people they also advise, so a popular friend counts for less, or you could decide for each friend how much to trust them on this particular question. GCN, GraphSAGE and GAT are those three styles, applied to each node of a graph in turn.

They also differ in what they do about you. GraphSAGE keeps your own opinion in a separate slot, so it can learn to ignore the friends entirely. GCN stirs your opinion into the same pot. That small difference matters a great deal when your friends are not like you, and the experiment in this chapter shows how much.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Layer | One round of message passing with learned weights | Two layers see two hops |
| Weight matrix | Learned numbers that mix a node's features | 8 inputs to 5 outputs |
| Self-loop | A node counts itself as a neighbour | Added by GCN |
| Attention weight | How much a node listens to one neighbour, between 0 and 1 | 0.5876 |
| Head | One independent attention mechanism; several are joined | 4 heads |
| Semi-supervised | Only a few nodes have labels, all nodes take part | 30 labels, 240 nodes |
| Homophily | Linked nodes tend to share a class | 0.812 of edges join the same class |
| Inductive | A model that can embed nodes it never saw in training | GraphSAGE |
| Validation set | Labelled nodes used to pick when to stop | 60 nodes |

## The idea in plain words

Start from the five-node graph of the previous chapter. Node 2 has three neighbours, nodes 0, 1 and 3, and the layer must turn four vectors (its own and theirs) into one. Each of the three layers does it differently.

**GCN** takes a fixed weighted sum. The weight on a neighbour is one over the square root of the product of the two degrees (counting the self-loop). For node 2 it gives 0.25 to itself and 0.289 to each neighbour. Then the sum goes through one shared weight matrix. In symbols, with $\hat S$ the normalised adjacency of the previous chapter, $H$ the table of node vectors and $W$ a weight matrix,

$$H' = \mathrm{relu}(\hat S H W).$$

In words: average the neighbourhood with degree-based weights, then transform.

**GraphSAGE** (Hamilton, Ying and Leskovec) takes the mean of the neighbours, transforms it with one matrix, transforms the node's own vector with another, and adds them:

$$h_v' = \mathrm{relu}(W_{\text{self}}\, h_v + W_{\text{nb}}\, \mathrm{mean}_{u \in N(v)}\, h_u).$$

In words: what I am, plus what my neighbours are like, each with its own transform. The paper's contribution was making this work for graphs that keep growing: it samples a fixed number of neighbours per node and learns functions that apply to nodes unseen in training, which is why it is called inductive. This chapter uses the mean version over all neighbours; sampling is the subject of the third chapter.

**GAT** (Veličković and colleagues) learns the weights. Transform every node with $W$. For a node $i$ and an input $j$ compute a score from the pair, shrink negatives with a LeakyReLU (multiply by a small slope such as 0.2), and turn the scores of all inputs into weights that sum to 1 with a softmax:

$$e_{ij} = \mathrm{leakyrelu}\big(a_{\text{dst}}^{\top} W h_i + a_{\text{src}}^{\top} W h_j\big), \qquad \alpha_{ij} = \frac{\exp e_{ij}}{\sum_k \exp e_{ik}}, \qquad h_i' = \mathrm{relu}\Big(\sum_j \alpha_{ij} W h_j\Big).$$

In words: each node asks how relevant each input is to it, and takes a weighted average by relevance. The vectors $a_{\text{dst}}$ and $a_{\text{src}}$ are the only extra parameters. Several independent copies, called heads, are run and their outputs joined, as in transformer attention.

<Infographic src="/img/gnn/three-layers.svg" alt="Three panels for GCN, GraphSAGE and GAT. GCN uses fixed weights 0.250, 0.289, 0.289, 0.289 with 45 parameters; GraphSAGE uses a mean of 0.333 for each neighbour plus a separate self matrix with 85 parameters; GAT learns weights 0.588, 0.131, 0.065, 0.216 with 110 parameters for two heads. A strip below gives the largest difference from the PyTorch Geometric layers: 1.2e-07, 0.0 and 2.4e-07." caption="Read each panel's weights for node 2, then the parameter counts. The strip at the bottom is the evidence that the from-scratch layers compute the same thing as the library." />

## Worked example, step by step

Node 2 in the five-node graph, with GAT raw scores for itself and nodes 0, 1 and 3 of 2.0, 0.5, -1.0 and 1.0.

1. **GCN weights.** With self-loops, node 2 has degree 4 and nodes 0, 1 and 3 have degree 3. Itself: 1/4 = 0.25. Each neighbour: 1/sqrt(4 x 3) = 0.289. Fixed, whatever the features.
2. **GraphSAGE weights.** The mean over three neighbours gives 1/3 = 0.333 each. The node's own vector is not in the mean: it is multiplied by a separate matrix.
3. **GAT scores.** Apply the LeakyReLU with slope 0.2. The positive scores stay 2.0, 0.5 and 1.0. The negative one becomes 0.2 x (-1.0) = -0.2.
4. **Exponentiate.** exp(2.0) = 7.389, exp(0.5) = 1.649, exp(-0.2) = 0.819, exp(1.0) = 2.718. They sum to 12.575.
5. **Normalise.** Divide each by 12.575: 0.5876, 0.1311, 0.0651 and 0.2162. They sum to 1.
6. **Read the difference.** GCN and GraphSAGE would give the same weights whatever the node vectors. GAT's weights depend on the vectors, because the scores are computed from them. A negative score is shrunk to a fifth, but still gets a weight of 0.0651, not zero.

<Infographic src="/img/gnn/attention-steps.svg" alt="A table with four rows for node 2's inputs: itself with raw score 2.0 and weight 0.5876, node 0 with 0.5 and 0.1311, node 1 with minus 1.0 shrinking to minus 0.2 and weight 0.0651, node 3 with 1.0 and 0.2162, totals 12.575 and 1.0000. Two cards on the right report that after training attention put 0.823 of its weight on same-class neighbours when 0.817 of neighbours were same-class, and 0.576 against 0.567 with noisy edges." caption="Follow one row left to right: raw score, LeakyReLU, exponential, weight. The cards on the right preview the chapter's honest result: trained attention stayed close to uniform." />

## How it works

### What do the matrices cost?

For 8 input features and 5 outputs: a GCN layer has 45 parameters (a weight matrix and a bias), a GraphSAGE layer 85 (two matrices and a bias) and a two-head GAT layer 110 (a matrix with 10 output columns, two scoring vectors per head and a bias). Block 1 prints these. In a sparse implementation the cost of a forward pass is proportional to the number of edges, since each edge carries one message. The dense matrices used here are for clarity.

### How is a layer checked against a library?

Take PyTorch Geometric's layer with the same weights and the same graph, copy the parameters across, and compare outputs. If your layer differs from the library by 1e-7 on a random graph with float32 numbers, it is the same function up to rounding. This catches the usual mistakes: a missing self-loop, a transposed weight, a softmax over the wrong axis. Block 1 does it for all three.

### Why train on 30 labels?

The standard test for these layers is semi-supervised: the model sees the features and edges of every node but the class of only a few. The loss is computed on the labelled nodes; the graph spreads their influence to the others through the layers. A validation set of other labelled nodes picks the epoch to report. Block 2 uses 10 labels per class (30 of 240 nodes), 60 validation nodes and 150 test nodes, the same kind of setting as the citation-graph benchmarks (Cora, Citeseer and Pubmed) named in the GAT paper's abstract, but on a graph we generate, so the truth about its structure is known.

### When does the graph help, and when does it hurt?

All three layers assume neighbours are informative about a node. That is homophily: linked nodes share a class. When most edges join nodes of the same class, averaging a neighbourhood cancels noise in each node's features. When edges mostly join different classes, averaging mixes the classes and blurs the signal. A model with a separate self term can learn to rely on the node's own features. GCN's single normalised sum cannot. Block 2 sweeps from 81 per cent of edges within a class to 40 per cent.

### Does attention learn to pick good neighbours?

In principle, yes: a high score for a same-class neighbour and a low one for others would make GAT robust to bad edges. In practice it needs enough labelled data and signal to learn that. Block 2 measures it: the share of attention going to same-class neighbours after training is almost the share of neighbours that are same-class, so the attention stayed close to uniform. With 30 labels there is not enough signal to learn a better rule.

<Infographic src="/img/gnn/homophily-sweep.svg" alt="Three panels of accuracy bars for an MLP, GCN, GraphSAGE and GAT. With 0.812 of edges within a class: 0.647, 0.974, 0.973, 0.969. With 0.566: 0.647, 0.669, 0.795, 0.679. With 0.400: 0.647, 0.385, 0.572, 0.305." caption="Compare each bar with the grey MLP bar, which never looks at the graph. On the left every graph model wins by a wide margin. On the right GCN and GAT lose to the MLP, and GraphSAGE loses less." />

## Code you can run

Each block is self-contained, so block 2 repeats the layer classes of block 1; that is also what makes each one runnable alone. Everything runs on the CPU.

### 1. The three layers from scratch, checked against PyTorch Geometric

We write the layers as dense torch modules on a 12-node random graph, copy their weights into the library layers, and compare outputs.

```python
import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(d_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        a_hat = adj + torch.eye(adj.shape[0])
        d = a_hat.sum(dim=1)
        norm = a_hat / torch.sqrt(d[:, None] * d[None, :])
        return norm @ (x @ self.weight) + self.bias

class SageLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_self = nn.Linear(d_in, d_out, bias=False)
        self.w_neigh = nn.Linear(d_in, d_out)

    def forward(self, x, adj):
        mean = adj @ x / adj.sum(dim=1, keepdim=True).clamp(min=1)
        return self.w_self(x) + self.w_neigh(mean)

class GATLayer(nn.Module):
    def __init__(self, d_in, d_out, heads=1):
        super().__init__()
        self.heads, self.d_out = heads, d_out
        self.weight = nn.Parameter(torch.empty(d_in, heads * d_out))
        self.att_src = nn.Parameter(torch.empty(1, heads, d_out))
        self.att_dst = nn.Parameter(torch.empty(1, heads, d_out))
        self.bias = nn.Parameter(torch.zeros(heads * d_out))
        for p in (self.weight, self.att_src, self.att_dst):
            nn.init.xavier_uniform_(p)

    def forward(self, x, adj, return_attention=False):
        n = x.shape[0]
        h = (x @ self.weight).view(n, self.heads, self.d_out)
        score_src = (h * self.att_src).sum(-1)
        score_dst = (h * self.att_dst).sum(-1)
        e = torch.nn.functional.leaky_relu(score_dst[:, None, :] + score_src[None, :, :], 0.2)
        mask = (adj + torch.eye(n)) > 0
        e = e.masked_fill(~mask[:, :, None], float("-inf"))
        alpha = torch.softmax(e, dim=1)
        out = torch.einsum("ijh,jhd->ihd", alpha, h).reshape(n, -1) + self.bias
        return (out, alpha) if return_attention else out

torch.manual_seed(0)
n, d_in, d_out = 12, 8, 5
adj = (torch.rand(n, n) < 0.3).float().triu(1)
adj = adj + adj.T
x = torch.randn(n, d_in)
edge_index = adj.nonzero().T.contiguous()

mine, theirs = GCNLayer(d_in, d_out), GCNConv(d_in, d_out)
with torch.no_grad():
    mine.bias.normal_()
    theirs.lin.weight.copy_(mine.weight.T)
    theirs.bias.copy_(mine.bias)
print("GCN   largest difference from torch_geometric GCNConv :", f"{(mine(x, adj) - theirs(x, edge_index)).abs().max().item():.1e}")

mine, theirs = SageLayer(d_in, d_out), SAGEConv(d_in, d_out, aggr="mean")
with torch.no_grad():
    theirs.lin_l.weight.copy_(mine.w_neigh.weight)
    theirs.lin_l.bias.copy_(mine.w_neigh.bias)
    theirs.lin_r.weight.copy_(mine.w_self.weight)
print("SAGE  largest difference from torch_geometric SAGEConv:", f"{(mine(x, adj) - theirs(x, edge_index)).abs().max().item():.1e}")

mine, theirs = GATLayer(d_in, d_out, heads=2), GATConv(d_in, d_out, heads=2, add_self_loops=True)
with torch.no_grad():
    theirs.lin.weight.copy_(mine.weight.T)
    theirs.att_src.copy_(mine.att_src)
    theirs.att_dst.copy_(mine.att_dst)
    theirs.bias.copy_(mine.bias)
print("GAT   largest difference from torch_geometric GATConv :", f"{(mine(x, adj) - theirs(x, edge_index)).abs().max().item():.1e}")
print("parameters: GCN", d_in * d_out + d_out, " SAGE", 2 * d_in * d_out + d_out, " GAT (2 heads)", d_in * 2 * d_out + 2 * 2 * d_out + 2 * d_out)
_, alpha = mine(x, adj, return_attention=True)
seen = (alpha[0, :, 0] > 0).nonzero().flatten().tolist()
print("node 0 attends to nodes", seen, "with head-0 weights", [round(alpha[0, j, 0].item(), 3) for j in seen], "summing to", round(alpha[0, :, 0].sum().item(), 3))
```

**Reading the output.** The largest differences from PyTorch Geometric are 1.2e-07 for GCN, 0.0 for GraphSAGE and 2.4e-07 for GAT. These are the size of float32 rounding, which is the evidence that each layer computes the same function. A missing self-loop or a wrong normalisation would show as differences far larger than rounding.

The parameter counts are 45, 85 and 110 as in the board. The last line prints node 0's attention: it attends to itself and nodes 2 and 3, with weights 0.432, 0.285 and 0.282 that sum to 1. A random, untrained GAT already gives unequal weights, because the scores come from random vectors.

**Line by line.**

- `a_hat / torch.sqrt(d[:, None] * d[None, :])` is the GCN normalisation in one line: row $i$, column $j$ divided by the square root of degree $i$ times degree $j$.
- `adj @ x / adj.sum(dim=1, keepdim=True).clamp(min=1)` is the neighbour mean; `clamp` protects isolated nodes.
- `e.masked_fill(~mask[:, :, None], float("-inf"))` makes the softmax ignore non-neighbours: their exponential is zero.
- The `theirs.lin.weight.copy_(mine.weight.T)` lines translate between conventions: PyTorch Geometric stores a linear layer's weight transposed relative to ours.

### 2. Training, with the graph getting noisier

A stochastic block model with three communities of 80 nodes each. Members of a community are linked with probability 0.08, members of different communities with probability `p_out`. Each node has 16 noisy features that weakly reveal its community. We train a feature-only MLP and the three layers with 30 labels, ten seeds each, at three noise levels.

```python
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x, adj):
        a_hat = adj + torch.eye(adj.shape[0])
        d = a_hat.sum(dim=1)
        return (a_hat / torch.sqrt(d[:, None] * d[None, :])) @ self.lin(x)

class SageLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_self = nn.Linear(d_in, d_out, bias=False)
        self.w_neigh = nn.Linear(d_in, d_out)

    def forward(self, x, adj):
        return self.w_self(x) + self.w_neigh(adj @ x / adj.sum(dim=1, keepdim=True).clamp(min=1))

class GATLayer(nn.Module):
    def __init__(self, d_in, d_out, heads):
        super().__init__()
        self.heads, self.d_out = heads, d_out
        self.lin = nn.Linear(d_in, heads * d_out, bias=False)
        self.att_src = nn.Parameter(torch.randn(1, heads, d_out) * 0.1)
        self.att_dst = nn.Parameter(torch.randn(1, heads, d_out) * 0.1)

    def forward(self, x, adj):
        n = x.shape[0]
        h = self.lin(x).view(n, self.heads, self.d_out)
        e = F.leaky_relu((h * self.att_dst).sum(-1)[:, None, :] + (h * self.att_src).sum(-1)[None, :, :], 0.2)
        e = e.masked_fill(((adj + torch.eye(n)) == 0)[:, :, None], float("-inf"))
        self.alpha = torch.softmax(e, dim=1)
        return torch.einsum("ijh,jhd->ihd", self.alpha, h).reshape(n, -1)

class Net(nn.Module):
    def __init__(self, kind, d_in, hidden, classes):
        super().__init__()
        self.kind = kind
        make = {"gcn": lambda a, b: GCNLayer(a, b), "sage": lambda a, b: SageLayer(a, b),
                "gat": lambda a, b: GATLayer(a, b // 4 if b > classes else b, 4 if b > classes else 1),
                "mlp": lambda a, b: nn.Linear(a, b)}[kind]
        self.l1, self.l2 = make(d_in, hidden), make(hidden, classes)

    def forward(self, x, adj):
        x = F.dropout(x, 0.5, self.training)
        h = F.relu(self.l1(x) if self.kind == "mlp" else self.l1(x, adj))
        h = F.dropout(h, 0.5, self.training)
        return self.l2(h) if self.kind == "mlp" else self.l2(h, adj)

def make_graph(seed, p_out, sizes=(80, 80, 80), noise=2.0):
    p = [[0.08 if i == j else p_out for j in range(3)] for i in range(3)]
    g = nx.stochastic_block_model(list(sizes), p, seed=seed)
    adj = torch.tensor(nx.to_numpy_array(g), dtype=torch.float32)
    labels = torch.tensor(np.repeat(np.arange(len(sizes)), sizes))
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(sizes), 16))
    x = torch.tensor(centres[labels.numpy()] + noise * rng.normal(size=(labels.shape[0], 16)), dtype=torch.float32)
    return adj, x, labels

def run(kind, seed, p_out):
    adj, x, y = make_graph(seed, p_out)
    g = torch.Generator().manual_seed(seed)
    train = torch.cat([torch.nonzero(y == c).flatten()[torch.randperm(80, generator=g)[:10]] for c in range(3)])
    rest = torch.tensor([i for i in range(len(y)) if i not in set(train.tolist())])
    val, test = rest[:60], rest[60:]
    torch.manual_seed(seed)
    net = Net(kind, 16, 16, 3)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    best, best_test = 0.0, 0.0
    for _ in range(200):
        net.train()
        opt.zero_grad()
        F.cross_entropy(net(x, adj)[train], y[train]).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(x, adj).argmax(1)
        if (pred[val] == y[val]).float().mean() >= best:
            best = (pred[val] == y[val]).float().mean().item()
            best_test = (pred[test] == y[test]).float().mean().item()
    return best_test, net

print("30 labelled nodes, 10 seeds per cell, test accuracy on 150 unlabelled nodes")
for p_out in (0.01, 0.03, 0.06):
    adj, x, y = make_graph(0, p_out)
    same = ((y[:, None] == y[None, :]).float() * adj).sum() / adj.sum()
    print(f"\nbetween-community edge probability {p_out}: mean degree {adj.sum(1).mean().item():.1f}, "
          f"{same.item():.3f} of edges join nodes of the same class")
    for kind in ("mlp", "gcn", "sage", "gat"):
        scores = np.array([run(kind, s, p_out)[0] for s in range(10)])
        print(f"  {kind:5s} mean {scores.mean():.3f}  sd {scores.std():.3f}  worst {scores.min():.3f}")

for p_out in (0.01, 0.03):
    adj, x, y = make_graph(0, p_out)
    _, net = run("gat", 0, p_out)
    net.eval()
    net(x, adj)
    alpha = net.l1.alpha.mean(dim=2)
    same = (y[:, None] == y[None, :]).float()
    neigh = adj > 0
    share_neighbours = (neigh * same).sum(1) / neigh.sum(1).clamp(min=1)
    mass = (alpha * neigh)
    share_attention = (mass * same).sum(1) / mass.sum(1).clamp(min=1e-9)
    print(f"\np_out {p_out}: neighbours of the same class {share_neighbours.mean().item():.3f}, "
          f"attention on same-class neighbours {share_attention.mean().item():.3f}, "
          f"attention a node gives itself {alpha.diagonal().mean().item():.3f} (uniform would be {(1 / (adj.sum(1) + 1)).mean().item():.3f})")
```

**Reading the output.** With clean edges (81 per cent of edges join the same class) the MLP, which never looks at the graph, gets 0.647 because the features are noisy. All three graph models reach 0.97: GCN 0.974, GraphSAGE 0.973, GAT 0.969, differences far smaller than their spread of 0.02. The graph is worth 0.32 to 0.33 of accuracy.

Make 43 per cent of edges cross communities (`p_out` 0.03, 57 per cent same-class) and the picture changes. GCN gets 0.669, barely above the MLP and with a spread of 0.229 and a worst seed of 0.180. GAT gets 0.679 with a spread of 0.284. GraphSAGE gets 0.795 with a worst seed of 0.600.

At 40 per cent same-class edges, almost what random edges between three classes would give (33 per cent), the graph models fall below the MLP: GCN 0.385, GAT 0.305, GraphSAGE 0.572 against 0.647.

Two lessons. A graph is not free: bad edges are worse than no edges. And GraphSAGE's separate self term helps, since it can learn to lean on the node's own features, though it does not remove the damage.

The attention check at the end settles what GAT did. With clean edges 0.817 of a node's neighbours share its class and the trained attention puts 0.823 of its weight there; a node gives itself 0.128 against a uniform 0.124. With noisy edges the numbers are 0.567, 0.576, 0.089 and 0.085. In both cases the attention stayed within about 0.01 of uniform, so GAT behaved like a GCN with a slightly different normalisation. That is consistent with GAT matching GCN in the first two settings. At the noisiest setting GAT scores lower (0.305 against 0.385), which is more likely the noise of ten seeds with a spread of 0.14 than a real ranking.

**Line by line.**

- `GATLayer.forward` stores `self.alpha` so the attention can be read after training.
- `weight_decay=5e-4` and `dropout` at 0.5 are common regularisers for this setting, needed because there are only 30 labels.
- The loop keeps the test accuracy at the epoch with the best validation accuracy, so the reported number never uses test labels for selection.
- `attention a node gives itself` reads the diagonal of the attention matrix; `1 / (degree + 1)` is what uniform attention would give.

### 3. The same layer in numpy, with a hand-written gradient

A two-layer GCN in plain numpy on the five-node graph, with the gradients derived by hand and compared against torch's automatic differentiation, plus the three weightings of node 2.

```python
import numpy as np
import torch

edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
A = np.zeros((5, 5))
for i, j in edges:
    A[i, j] = A[j, i] = 1
A_hat = A + np.eye(5)
d = A_hat.sum(axis=1)
S = A_hat / np.sqrt(np.outer(d, d))

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 3))
W1 = rng.normal(size=(3, 4)) * 0.5
W2 = rng.normal(size=(4, 2)) * 0.5
target = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)

Z1 = S @ X @ W1
H = np.maximum(Z1, 0)
out = S @ H @ W2
loss = ((out - target) ** 2).sum()
d_out = 2 * (out - target)
grad_W2 = (S @ H).T @ d_out
d_H = S.T @ d_out @ W2.T
grad_W1 = (S @ X).T @ (d_H * (Z1 > 0))
print(f"numpy forward loss {loss:.6f}")

tX, tS = torch.tensor(X), torch.tensor(S)
tW1, tW2 = torch.tensor(W1, requires_grad=True), torch.tensor(W2, requires_grad=True)
t_loss = ((tS @ torch.relu(tS @ tX @ tW1) @ tW2 - torch.tensor(target)) ** 2).sum()
t_loss.backward()
print(f"torch loss {t_loss.item():.6f}; largest gradient difference W1 {np.abs(grad_W1 - tW1.grad.numpy()).max():.1e}, "
      f"W2 {np.abs(grad_W2 - tW2.grad.numpy()).max():.1e}")

print("\nhow node 2 weights its inputs (itself, then nodes 0, 1, 3)")
print("GCN, fixed by degree      :", np.round(S[2][[2, 0, 1, 3]], 3).tolist())
print("GraphSAGE mean over the three neighbours (own value has its own weight matrix):", np.round(np.full(3, 1 / 3), 3).tolist())
scores = np.array([2.0, 0.5, -1.0, 1.0])
leaky = np.where(scores > 0, scores, 0.2 * scores)
alpha = np.exp(leaky) / np.exp(leaky).sum()
print("GAT, raw scores for itself and nodes 0, 1, 3", scores.tolist(), "-> leaky", leaky.tolist(), "-> softmax", np.round(alpha, 4).tolist(), "sum", round(alpha.sum(), 6))
```

**Reading the output.** The numpy loss, 5.475864, equals torch's, and the largest differences in the gradients for the two weight matrices are 2.2e-16 and 0.0, so the hand-derived backward pass is right. This is all a GCN is: two matrix products per layer and a relu, with the normalised adjacency fixed in advance.

The three weightings of node 2 are the ones in the worked example: GCN 0.25, 0.289, 0.289 and 0.289, GraphSAGE one third each, and GAT 0.5876, 0.1311, 0.0651 and 0.2162, summing to 1.

**Line by line.**

- `grad_W2 = (S @ H).T @ d_out` and `d_H = S.T @ d_out @ W2.T` are the chain rule through `S @ H @ W2`; the relu mask `(Z1 > 0)` zeroes the gradient where the unit was off.
- `S.T` equals `S` here because the normalised adjacency is symmetric; it is written out so the rule reads correctly for a directed graph.

## Try it yourself

The lab computes one GAT attention row with the same formula as the numpy block. Its defaults reproduce the 0.5876, 0.1311, 0.0651 and 0.2162. The formula was checked against torch's softmax of the LeakyReLU to 2e-16.

<GatAttentionLab />

**What each control does.**

- The four **Raw score** sliders are the scores node 2 gives itself and nodes 0, 1 and 3 before normalisation.
- **LeakyReLU slope** sets how much a negative score is shrunk: 0 clips it to zero, 1 leaves it unchanged.
- Click **show data** to compare the attention weights with the fixed GCN weights and GraphSAGE's mean.

**Try it yourself.**

1. Set the slope to 1. The weights become 0.6095, 0.136, 0.0303 and 0.2242, and the weight on node 1 halves from 0.0651 to 0.0303. Why: with slope 1 the negative score of -1 stays at -1 instead of being shrunk to -0.2, so node 1 loses more weight.
2. Set all four scores to 0. Every weight becomes 0.25. Why: equal scores give a uniform average, so GAT contains the plain mean as a special case. That is where the trained network in block 2 sat.
3. Set the itself score to 3, the other three to -3, and the slope to 0. The weights are 0.87 for itself and 0.0433 for each of the others, not zero. Why: with slope 0 a negative score clips to 0, which is exp(0) = 1 against exp(3) = 20.1 for itself. Attention can lower a neighbour's weight but cannot switch it off unless another score is far larger.

## Designing with it

1. **Start with GCN or GraphSAGE.** Block 2 gives them the same accuracy on clean data, and both are cheap and easy to debug. Add attention only if you can show it helps on a validation set.
2. **Measure homophily before training.** The share of edges whose end nodes have the same label, on your labelled nodes, predicts whether a graph model can beat the feature-only baseline. Below about one half, expect trouble.
3. **Always train the feature-only baseline.** It costs a minute. Block 2's table shows a graph model below it.
4. **Keep a separate self term** when the neighbours might be unreliable. GraphSAGE's two matrices, or a GCN with a skip connection, give the model a way to ignore them.
5. **Check your layer against a library layer once**, as in block 1, and then trust it.

## Where this stands in 2026

These three layers are the baseline every newer graph architecture is compared with. CS224W's Fall 2026 schedule moves from them to a theory lecture on expressiveness, designs for stronger graph encoders, graph transformers and heterogeneous graphs. Library support is mature: PyTorch Geometric 2.8.0.post1, released 20 July 2026 under the MIT licence, provides all three layers. Its listing marks the extra sampling libraries as optional. Block 2's lesson about homophily is the most reliable piece of practical advice from this literature: the architecture matters less than whether the edges join similar nodes.

## Common mistakes

1. **Comparing graph models without a feature-only baseline.** A graph model feels like it must win. Block 2 has two settings where it does not. Always report the MLP.
2. **Reading attention as an explanation.** Attention weights look interpretable. Here they were within 0.01 of uniform after training, so reading them as "the model trusts node 3" would say nothing. Check how far they are from uniform first.
3. **Selecting the epoch on the test set.** With 150 test nodes, picking the best epoch by test accuracy inflates the score by several points. Use a validation set.
4. **Judging from one seed.** The spread in block 2 is 0.02 on clean data and 0.28 on noisy data. Run ten seeds and report the mean and the worst case.
5. **Transposing a weight matrix when porting.** A silent shape-compatible mistake. Copy weights into a library layer and compare, as in block 1.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> In the worked example, why does GCN give node 2 a weight of 0.25 for itself but 0.289 for each neighbour?</summary>

The weight between two nodes is 1 over the square root of the product of their degrees with self-loops. For node 2 with itself that is 1/sqrt(4 x 4) = 0.25. For a neighbour of degree 3 it is 1/sqrt(4 x 3) = 0.289. Neighbours with fewer connections get slightly more weight.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> Why can GAT's weights not be zero for a neighbour?</summary>

The weights come from a softmax, and the exponential of any number is positive. So every input gets some positive weight. A very negative score makes it tiny but not zero, and with a LeakyReLU slope of 0 a negative score is clipped to 0, which gives a weight comparable to other zero scores.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> Compute GAT's weights for raw scores 1.0 and -2.0 with slope 0.2.</summary>

The LeakyReLU gives 1.0 and -0.4. The exponentials are 2.718 and 0.670. Their sum is 3.388, so the weights are 2.718 / 3.388 = 0.802 and 0.670 / 3.388 = 0.198.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> At 57 per cent same-class edges, GraphSAGE scored 0.795 and GCN 0.669. Give the mechanism.</summary>

GCN mixes a node's own features with its neighbours' in one fixed weighted sum and passes it through one matrix, so when many neighbours are from another class its representation is blurred and the layer cannot undo it. GraphSAGE transforms the node's own vector with a separate matrix, so training can raise the weight on the node's own features and lower the weight on the neighbour mean. It still loses to the clean case, because the neighbour mean carries wrong-class information the model must learn to suppress.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> Block 2 reports GAT's attention on same-class neighbours as 0.823 against 0.817 of neighbours. What would you try to make the attention useful, and what would you measure?</summary>

More labels or an auxiliary loss on edges would give the scores a signal to learn from, and more heads or a longer schedule might help. Measure the same quantity, the share of attention on same-class neighbours against the share of same-class neighbours, and the test accuracy over ten seeds. Attention is useful only if both move: the share well above the base rate, and accuracy above GCN at the noisy settings.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> Explain why the numpy gradient for the first weight matrix needs the mask `Z1 > 0`.</summary>

The layer computes a relu of `Z1`. The derivative of relu is 1 where its input is positive and 0 where it is not, so any gradient arriving at a unit that was switched off must be zeroed. Without the mask, the gradient would pass through switched-off units and would no longer match torch's.

</details>

## Go deeper

All sources were opened on 8 October 2026.

- Kipf TN and Welling M, "Semi-Supervised Classification with Graph Convolutional Networks", arXiv 1609.02907, first submitted 9 September 2016, latest version v4 of 22 February 2017, ICLR 2017. The abstract describes a scalable approach whose cost is linear in the number of edges; the full text was not read beyond the abstract page.
- Hamilton WL, Ying R and Leskovec J, "Inductive Representation Learning on Large Graphs", arXiv 1706.02216, first submitted 7 June 2017, latest version v4 of 10 September 2018, NIPS 2017. GraphSAGE: sampling and aggregating from a node's neighbourhood to embed unseen nodes. This chapter implements the mean aggregator only.
- Veličković P, Cucurull G, Casanova A, Romero A, Liò P and Bengio Y, "Graph Attention Networks", arXiv 1710.10903, first submitted 30 October 2017, latest version v3 of 4 February 2018, ICLR 2018. Masked self-attention over neighbours.
- [PyTorch Geometric on PyPI](https://pypi.org/project/torch-geometric/): release 2.8.0.post1 of 20 July 2026, MIT licence, Python 3.10 to 3.14. Version 2.8.0.post1 was run.
- [Stanford CS224W](https://web.stanford.edu/class/cs224w/): the Fall 2026 schedule, with public slides.

## Check yourself

- I can write the GCN, GraphSAGE and GAT updates and say what each weights.
- I can compute an attention row by hand from raw scores.
- I can check a layer against a library layer and read the size of the difference.
- I can predict from the share of same-class edges whether a graph model will beat a feature-only model.
- I can say what block 2 showed about GAT's attention and why that matters for interpreting it.

## Where to go next

Next chapter: [graph tasks in production](/docs/theory/gnn/graph-tasks-in-production), where the same layers are applied to fraud rings, link prediction and graph-level labels, and where neighbour sampling is needed because the graph does not fit in memory. A related chapter: [self-attention in transformers, with code](/docs/theory/dnn/self-attention-in-transformers-with-code), whose scores and softmax are the same mechanism as GAT's on a graph where every token is a neighbour.
