---
id: gnn-graphs-message-passing
title: "Graphs and Message Passing"
sidebar_label: "1 · Graphs and message passing"
sidebar_position: 1
slug: /theory/gnn/graphs-and-message-passing
description: "What graph data is, why a node's neighbours decide what a graph neural network sees, how one round of message passing works with sum, mean and normalised aggregation, and where it breaks: expressiveness and over-smoothing, shown on Zachary's karate club."
tags: [graph-neural-networks, message-passing, adjacency-matrix, permutation-equivariance, weisfeiler-leman, over-smoothing, networkx]
---

import Infographic from '@site/src/components/Infographic';
import MessagePassingLab from '@site/src/components/viz/MessagePassingLab';

**In one line.** A graph neural network learns about each node by repeatedly collecting information from its neighbours, so what a node knows after k rounds is what lies within k steps of it.

:::tip Before you start
- **You should already know** how a neural network layer turns a vector into a new vector ([what deep learning is](/docs/theory/dnn/what-deep-learning-is-and-how-it-differs-from-machine-learning)) and have seen a graph of links between pages ([link analysis and PageRank](/docs/theory/ir/link-analysis-pagerank-and-hits)).
- **Reading time:** about 40 minutes, plus a few seconds to run the code.
- **After this chapter you can** write a graph as an adjacency matrix, compute one round of message passing by hand with three different summaries, explain why a graph layer must not depend on node order, and name two limits of the idea: what it cannot tell apart and what happens with too many layers.
:::

:::note Not from a lecture
This chapter was written for this site from the sources under Go deeper, which include the public Stanford CS224W course (Fall 2026 offering, whose schedule was read) and the original papers. Every number is printed by the code in the chapter. Environment: Python 3.14, NumPy 2.5.3, NetworkX 3.6.1, PyTorch 2.14.1 on the CPU. Sources were opened on 8 October 2026.
:::

## In 30 seconds

You can tell a lot about a person from their friends. If most of someone's friends like hiking, they probably do too. A graph neural network works the same way: each node (a person, an account, an atom) starts with some facts about itself, then looks at its neighbours, mixes what it sees into its own description, and does that again.

After one round a node knows about its friends. After two rounds it knows about friends of friends. The trick is to do this in a way that does not depend on how you happened to number the nodes, because a graph has no first node. The rest of the chapter is the detail of that idea and its two main limits.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Graph | A set of nodes joined by edges | Accounts and the transfers between them |
| Node feature | The facts we know about a node | Account age, balance |
| Adjacency matrix (A) | A square table with a 1 where two nodes are joined | Row 2 has ones for nodes 0, 1 and 3 |
| Degree | How many neighbours a node has | Node 2 has degree 3 |
| Message passing | Each node collects its neighbours' values and updates its own | Replace my value by the mean of my friends |
| Aggregation | The rule that combines the neighbours' values | Sum, mean or max |
| Receptive field | All nodes that can influence a node after k rounds | 17 of 34 after one round for member 0 |
| Permutation equivariance | Relabelling the nodes only relabels the outputs | Same result, rows in a new order |
| Over-smoothing | Many rounds make all nodes look alike | Every guess becomes the same after 50 rounds |

## The idea in plain words

Most machine learning expects a table (one row per example) or a grid (an image). Many problems are neither. Bank accounts are linked by payments, molecules are atoms linked by bonds, people are linked by friendships, roads by junctions. The links are the information: a payment from a flagged account is a clue about the receiver, however ordinary the receiver's own record looks.

You could flatten the links into a table by listing, say, the number of flagged neighbours. That works for one hand-built feature. A graph neural network (GNN) learns the feature instead, by a loop called message passing. In each round, every node does three things:

1. Gather the current values of its neighbours.
2. Combine them with an aggregation rule: sum, mean, maximum or a weighted mix.
3. Mix the result with its own value and pass it through a small learned function.

In symbols, with $h_v$ the value of node $v$ and $N(v)$ its neighbours,

$$h_v^{(k+1)} = \mathrm{update}\Big(h_v^{(k)},\ \mathrm{aggregate}\big(\{h_u^{(k)} : u \in N(v)\}\big)\Big).$$

In words: the new value of a node is a function of its old value and a summary of what its neighbours currently hold. Stack $k$ rounds and the value of a node depends on everything within $k$ edges of it.

One rule is not optional. A graph has no natural order for its nodes, so the aggregation must give the same answer however you list the neighbours, and a layer must give the same answers if you rename every node. Sum, mean and max all have that property. Concatenating neighbours in the order you stored them does not.

<Infographic src="/img/gnn/message-passing-steps.svg" alt="A graph of five nodes, a triangle of nodes 0, 1 and 2 with a tail through nodes 3 and 4, starting values 1 to 5, a table of the sum, mean and normalised GCN step for each node, and three cards showing how node 2 is computed: sum 7, mean 2.333, GCN step 2.771." caption="Start with the graph on the left and the three rows of the table. Then read the cards on the right: they compute node 2, which has three neighbours, in all three ways." />

## Worked example, step by step

Five nodes joined by the edges 0-1, 0-2, 1-2, 2-3 and 3-4. Their starting values are 1, 2, 3, 4 and 5.

1. **Write the adjacency matrix.** Node 0 touches nodes 1 and 2, so its row has ones in columns 1 and 2. Node 2 touches 0, 1 and 3. The degrees (row sums) are 2, 2, 3, 2, 1.
2. **Sum the neighbours.** Node 0: 2 + 3 = 5. Node 1: 1 + 3 = 4. Node 2: 1 + 2 + 4 = 7. Node 3: 3 + 5 = 8. Node 4 has one neighbour, node 3, so 4.
3. **Take the mean instead.** Divide each sum by the degree: 2.5, 2.0, 2.333, 4.0, 4.0.
4. **Use the GCN rule.** Add a self-loop (each node is its own neighbour, so degrees become 3, 3, 4, 3, 2). The weight of neighbour $j$ for node $i$ is 1 over the square root of the product of the two new degrees. For node 2 with itself it is 1/4 = 0.25; with node 0 it is 1/sqrt(4 x 3) = 0.289. The new value of node 2 is 0.289 x 1 + 0.289 x 2 + 0.25 x 3 + 0.289 x 4 = 2.771.
5. **Compare.** Sum grows with degree, so high-degree nodes get big values. The mean ignores degree. The GCN rule sits between: it averages, but downweights messages from busy neighbours.

The code in block 1 reproduces every one of these numbers.

## How it works

### What does the adjacency matrix do?

For $n$ nodes, the adjacency matrix $A$ is an $n \times n$ table with $A_{ij} = 1$ when nodes $i$ and $j$ are joined. Multiplying $A$ by the column of node values $x$ gives, for each node, the sum of its neighbours' values. That is why message passing is cheap: one round is one multiplication by a very sparse matrix, and its cost grows with the number of edges rather than the square of the number of nodes.

Dividing row $i$ of the sum by the degree $d_i$ gives the mean. The GCN rule uses $\hat A = A + I$ (adding the self-loops) and scales by the new degrees $\hat d$: $\hat S = \hat D^{-1/2} \hat A \hat D^{-1/2}$. In words: each entry is 1 over the square root of the two degrees, which stops a node with 100 neighbours from shouting over the rest.

### How far does a node see?

After $k$ rounds a node's value depends on nodes within $k$ edges. In the karate club, member 0 has 16 neighbours, so with itself the receptive field is 17 of 34 nodes after one layer, 26 after two and all 34 after three. Real social graphs are denser in the middle, so the field often covers a large part of the graph in two or three hops. This matters for cost, treated in the third chapter.

### Why must the layer ignore node order?

If you renumber the nodes with a permutation matrix $P$, the graph is the same graph. A correct layer gives $f(PAP^{\top}, PX) = P f(A, X)$: the output rows are shuffled the same way as the input. This property is called permutation equivariance. A model that reads node 0 as "first" would learn quirks of one numbering. Block 2 checks the property on a random graph to floating-point precision.

For a task about a whole graph, such as classifying a molecule, a final step called a readout (a sum or mean over nodes) must also ignore order.

### What can message passing not tell apart?

A node's value after $k$ rounds is determined by its tree of neighbours, so two nodes with the same tree get the same value, however different the graphs around them. The classical test that mirrors this is Weisfeiler-Leman colour refinement: start with a colour per node, and repeatedly recolour each node by its own colour plus the multiset of its neighbours' colours. Xu and colleagues showed that standard message-passing networks are no more powerful than this test, and that GCN and GraphSAGE fail on some simple pairs of graphs.

A 6-cycle and two separate triangles are the standard example. Every node has two neighbours, each with two neighbours, so every colour round is identical. Yet one graph contains triangles and the other does not. A network built on this idea cannot count them.

The aggregation matters too. Sum keeps the number of neighbours. Mean loses it (the mean of $\{1, 1\}$ equals the mean of $\{1\}$). Max loses more (it cannot tell $\{1, 2\}$ from $\{1, 1, 2\}$).

### Why can more layers hurt?

Each round averages a node with its neighbours, and averaging repeatedly pulls values together. After enough rounds all nodes in a connected graph look alike, so a classifier built on them cannot separate anything. Li and colleagues described graph convolution as a form of Laplacian smoothing and warned that stacking many layers can over-smooth. Block 1 measures it: accuracy holds at 0.971 from 3 to 10 hops and collapses to 0.500 at 50 hops.

<Infographic src="/img/gnn/karate-hops.svg" alt="A table of how accuracy changes with the number of hops when propagating the labels of two members of Zachary's karate club: 0.941 after one and two hops, 0.971 from three to ten hops, 0.941 at twenty hops and 0.500 with a single guess at fifty hops. Bars show the receptive field of member 0: 17, 26 and 34 of 34 nodes." caption="Read the table down the accuracy column: it rises, plateaus and then collapses to a coin flip. The bars on the right show why a few layers are enough to see the whole graph." />

<Infographic src="/img/gnn/wl-limits.svg" alt="A hexagon and two triangles that get identical colour histograms under colour refinement, a table in which a path of six nodes and a double star get different colour lists, and a table showing that sum, mean and max of neighbour values keep different amounts of information." caption="The top left is the limit: two graphs with different structure that message passing cannot separate. The bottom table shows why the choice of aggregation matters." />

## Code you can run

Each block is self-contained and runs on the CPU in seconds. The graph data is the karate club that ships with NetworkX, so nothing is downloaded.

### 1. One round by hand, then propagation on a real graph

The first part repeats the worked example with NumPy. The second takes Zachary's karate club, where the club split into two factions of 17 after a dispute. We reveal only the factions of the two leaders (nodes 0 and 33), spread those two labels over the graph with the GCN rule, and guess each member's faction by which leader's signal is stronger.

```python
import networkx as nx
import numpy as np

edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
graph = nx.Graph(edges)
A = nx.to_numpy_array(graph, nodelist=range(5))
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
degree = A.sum(axis=1)
print("degrees:", degree.astype(int).tolist())
print("sum of neighbours   :", (A @ X).ravel().tolist())
print("mean of neighbours  :", ((A @ X).ravel() / degree).round(3).tolist())

A_hat = A + np.eye(5)
d_hat = A_hat.sum(axis=1)
S = A_hat / np.sqrt(np.outer(d_hat, d_hat))
print("GCN-normalised step :", (S @ X).ravel().round(3).tolist())
print("node 2 weights on itself and its neighbours:", {int(j): round(float(S[2, j]), 3) for j in np.nonzero(S[2])[0]})

club = nx.karate_club_graph()
n = club.number_of_nodes()
A = nx.to_numpy_array(club, nodelist=range(n))
A_hat = A + np.eye(n)
d_hat = A_hat.sum(axis=1)
S = A_hat / np.sqrt(np.outer(d_hat, d_hat))
faction = np.array([club.nodes[i]["club"] == "Mr. Hi" for i in range(n)])
print(f"\nkarate club: {n} nodes, {club.number_of_edges()} edges, factions {faction.sum()} and {(~faction).sum()}")

seeds = np.zeros((n, 2))
seeds[0, 0] = 1.0
seeds[33, 1] = 1.0
checkpoints = (1, 2, 3, 4, 6, 10, 20, 50, 100, 200)
print("hops   nodes reached   accuracy on all nodes   distinct guesses")
signal = seeds.copy()
for hops in range(1, max(checkpoints) + 1):
    signal = S @ signal
    if hops in checkpoints:
        guess = signal[:, 0] > signal[:, 1]
        print(f"{hops:4d} {(signal.sum(axis=1) > 0).sum():16d} {np.mean(guess == faction):20.3f} {len(set(guess.tolist())):15d}")
```

**Reading the output.** The degrees, sums, means and GCN values match the worked example: node 2 gets 7, 2.333 and 2.771. Its GCN weights are 0.289 for each neighbour and 0.25 for itself.

On the karate club, two labels are enough to classify the whole club with high accuracy. After one hop, 31 of 34 nodes have received a signal and the accuracy is 0.941. After three hops it is 0.971, which is 33 of 34 correct, and it stays there through 10 hops. The one wrong member is node 8, who is in the leader-0 faction but has three friends in the other faction and only two in his own, so any rule based on friends votes against him.

Then it breaks. At 20 hops accuracy is 0.941, and at 50 hops it is 0.500 with a single distinct guess. The signals from both leaders converge to the same direction, the dominant eigenvector of the normalised matrix, so the comparison between them gives the same answer for every node. This is over-smoothing, with no learning involved at all. Notice also that the useful range, 3 to 10 hops, is wide: the failure needs far more layers than most networks use.

**Line by line.**

- `A_hat = A + np.eye(5)` adds the self-loops, so a node keeps part of its own value.
- `S = A_hat / np.sqrt(np.outer(d_hat, d_hat))` divides each entry by the square root of the two nodes' degrees, the GCN normalisation.
- `seeds[0, 0] = 1.0` and `seeds[33, 1] = 1.0` put a unit of signal on each leader in its own column, so each column tracks one leader's influence.
- `signal = S @ signal` is one round of message passing for both leaders at once.

### 2. Equivariance and the choice of aggregation

A message-passing layer in PyTorch, written with two weight matrices (one for the node, one for the mean of its neighbours). We relabel the nodes at random and confirm that the layer's output is just relabelled.

```python
import torch

torch.manual_seed(0)
n, features, hidden = 7, 4, 3
A = (torch.rand(n, n) < 0.4).float()
A = torch.triu(A, diagonal=1)
A = A + A.T
X = torch.randn(n, features)
W_self, W_neigh = torch.randn(features, hidden), torch.randn(features, hidden)

def layer(A, X):
    degree = A.sum(dim=1, keepdim=True).clamp(min=1)
    return torch.relu(X @ W_self + (A @ X / degree) @ W_neigh)

out = layer(A, X)
perm = torch.randperm(n)
P = torch.eye(n)[perm]
permuted = layer(P @ A @ P.T, P @ X)
print("permutation order:", perm.tolist())
print("largest difference between layer(permuted graph) and permuted layer(graph):",
      f"{(permuted - P @ out).abs().max().item():.2e}")

pooled = out.sum(dim=0)
pooled_permuted = permuted.sum(dim=0)
print("graph-level sum readout changes by", f"{(pooled - pooled_permuted).abs().max().item():.2e}", "after relabelling the nodes")

multisets = {"{1, 1, 2}": [1.0, 1.0, 2.0], "{1, 2}": [1.0, 2.0], "{1, 1}": [1.0, 1.0], "{1}": [1.0], "{2, 2, 1, 1}": [2.0, 2.0, 1.0, 1.0]}
print("\nneighbour values   sum   mean    max")
for name, values in multisets.items():
    v = torch.tensor(values)
    print(f"{name:16s} {v.sum().item():5.1f} {v.mean().item():6.3f} {v.max().item():6.1f}")
```

**Reading the output.** The largest difference between running the layer on the relabelled graph and relabelling the output is 2.38e-07, which is single-precision rounding, not a real difference. The graph-level sum readout is identical after relabelling. If a layer depended on node order, this difference would be of the same size as the outputs themselves.

The table shows what each aggregation keeps. Sum gives five different numbers for the five multisets. Mean maps $\{1, 1\}$ and $\{1\}$ to the same 1.000, and $\{1, 2\}$ and $\{2, 2, 1, 1\}$ to the same 1.500, so it loses the neighbour count. Max gives 2 for three different multisets. The practical lesson is that the aggregation decides which structural questions a network can even ask.

**Line by line.**

- `torch.triu(A, diagonal=1)` keeps the upper triangle, and `A + A.T` mirrors it, so the random graph is undirected with no self-loops.
- `P = torch.eye(n)[perm]` builds a permutation matrix by shuffling the rows of the identity.
- `P @ A @ P.T` and `P @ X` relabel the graph and its features consistently.
- `clamp(min=1)` stops a division by zero for isolated nodes.

### 3. What colour refinement can and cannot see

NetworkX implements the Weisfeiler-Leman graph hash. We compare four small graphs and write the colour refinement by hand as well, to see what the hash is built from.

```python
import networkx as nx
import numpy as np

hexagon = nx.cycle_graph(6)
two_triangles = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
path = nx.path_graph(6)
star_pair = nx.Graph([(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)])

def colours(graph, rounds):
    colour = {v: graph.degree(v) for v in graph}
    for _ in range(rounds):
        signature = {v: (colour[v], tuple(sorted(colour[u] for u in graph[v]))) for v in graph}
        palette = {sig: i for i, sig in enumerate(sorted(set(signature.values())))}
        colour = {v: palette[signature[v]] for v in graph}
    return sorted(colour.values())

for name, graph in (("hexagon", hexagon), ("two triangles", two_triangles), ("path of 6", path), ("double star", star_pair)):
    print(f"{name:14s} degrees {sorted(d for _, d in graph.degree())}  "
          f"WL hash {nx.weisfeiler_lehman_graph_hash(graph, iterations=3)[:12]}  triangles {sum(nx.triangles(graph).values()) // 3}")
print("hexagon isomorphic to two triangles:", nx.is_isomorphic(hexagon, two_triangles))
print("same colour histogram after 3 rounds:", colours(hexagon, 3) == colours(two_triangles, 3))
print("path of 6 against double star, colours:", colours(path, 3), colours(star_pair, 3))

club = nx.karate_club_graph()
A = nx.to_numpy_array(club)
reach = np.eye(len(A))
print("\nreceptive field of node 0 in the karate club")
for k in range(1, 5):
    reach = (reach @ (A + np.eye(len(A)))) > 0
    print(f"  {k} layers: {int(reach[0].sum())} of {len(A)} nodes")
```

**Reading the output.** The hexagon and the two triangles have the same degrees and the same hash, even though the hexagon has no triangles and the other graph has two. They are not isomorphic (`False`), and their colour lists are equal after three rounds (`True`). A message-passing network that starts from identical node features cannot separate them either.

The path of six nodes and the double star are separated: their colour lists differ. The last lines give the receptive field of node 0: 17 nodes after one layer, 26 after two and 34 after three, the same as in the board.

**Line by line.**

- `colour = {v: graph.degree(v) ...}` starts every node with its degree as its colour, since all nodes begin alike.
- `signature` pairs each node's colour with the sorted colours of its neighbours, the multiset a sum aggregator can see.
- `palette` renames each distinct signature to a small integer, so the colours stay compact after each round.

## Try it yourself

The lab runs the same five-node graph. Its defaults, the GCN rule and one layer, reproduce block 1: 1.866, 1.866, 2.771, 4.241 and 4.133. The arithmetic was checked against NumPy for all four aggregations over eight layers, with differences of about 1e-16.

<MessagePassingLab />

**What each control does.**

- **Aggregation** chooses the summary: the GCN rule, sum, mean or max of the neighbours.
- **Layers** sets how many rounds to run, from 0 (the starting values) to 8.
- Click **show data** to see every step's values side by side.

**Try it yourself.**

1. Choose the GCN rule and move Layers from 1 to 8. The spread between the largest and smallest value falls from 2.375 to 0.571, and the values approach each other (2.694, 2.694, 3.265, 3.162, 2.727). Why: each round averages a node with its neighbours, so differences shrink. This is over-smoothing in miniature.
2. Switch to max and use 8 layers. Every node ends at 5. Why: the largest value in the graph, held by node 4, spreads along the edges and nothing can dilute it. Compare mean at 8 layers, where the spread is 0.248.
3. Switch to sum and use 4 layers. The values are 62, 63, 82, 45 and 25, and at 8 layers they reach 1554, 1555, 1924, 1059 and 503. Why: without normalisation each node adds up its neighbours, so values grow with every round. Practical networks normalise or use a mean.

## Designing with it

Four decisions when you first model a problem as a graph.

1. **What are the nodes, and what are the edges?** A payment network could use accounts as nodes and transfers as edges, or accounts and devices as two kinds of node. The choice decides what the model can see.
2. **How many layers?** Start with two or three. The receptive field grows quickly, and the karate club needed three to cover everyone.
3. **Which aggregation?** Use sum or a learned weighting when the neighbour count matters (for fraud, ten flagged neighbours differ from one), and mean when only the typical neighbour matters.
4. **What are the node features?** If the nodes have no features, the structure alone can be informative, as the karate club example shows, but identical features make the model blind to anything colour refinement cannot see.

## Where this stands in 2026

Message passing remains the base of most graph learning, from molecule models to recommendation. The Stanford CS224W course, whose Fall 2026 schedule lists graph neural networks, a theory lecture on what GNNs can express, graph transformers and relational deep learning, treats it as the starting point for more expressive designs. Those designs add positional information or let nodes attend to non-neighbours to get past the limits shown here. The chapter's two failure modes, indistinguishable structures and over-smoothing, are still the first things to check when a graph model underperforms.

## Common mistakes

1. **Treating the adjacency matrix as an image.** It looks like a grid, so a convolutional network seems natural. But reordering the nodes changes the picture while the graph stays the same. Use a layer that is equivariant to node order.
2. **Stacking many layers because depth helped elsewhere.** Block 1 shows accuracy collapsing to 0.500 at 50 hops. Start with two or three and add layers only if validation accuracy rises.
3. **Using mean aggregation when degree is the signal.** Mean cannot tell one flagged neighbour from ten. Use sum or add the degree as a feature.
4. **Assuming the structure is enough.** Two graphs with the same colour refinement result look identical to a message-passing model. If the task depends on counting triangles or cycles, add that as a feature.
5. **Forgetting the self-loop.** Without it a node's own value never enters its update through the normalised sum. GCN adds it on purpose.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> In the worked example, why is the sum for node 4 equal to 4?</summary>

Node 4 has one neighbour, node 3, whose value is 4. The sum of the neighbours' values is therefore 4. The mean is also 4, because dividing by one neighbour changes nothing.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> After two layers, which nodes can influence node 4?</summary>

After one layer, node 4 sees node 3. After two, it also sees node 3's neighbours, nodes 2 and 4 itself. So nodes 2, 3 and 4 can influence it. Nodes 0 and 1 are three edges away and need a third layer.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> Compute the GCN weight between node 0 and node 2 in the worked example, and explain why the weight from node 3 to node 4 is larger.</summary>

With self-loops the degrees are 3 for node 0 and 4 for node 2, so the weight is 1/sqrt(3 x 4) = 0.289. Node 3 has degree 3 and node 4 degree 2, so the weight is 1/sqrt(3 x 2) = 0.408. The weight is larger because both nodes are lightly connected, so each message carries more of the node's attention.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> Why can the mean aggregator not distinguish a node with neighbours $\{1, 1\}$ from one with neighbours $\{1\}$?</summary>

Both means are 1.0, as block 2 prints. The mean divides by the number of neighbours, so it discards how many there are. The sum gives 2 and 1 and keeps the difference.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> The hexagon and the two triangles get identical colour lists. Why does that stop a message-passing network from counting triangles, even with learned weights?</summary>

After each round a node's value is a function of its previous value and the multiset of its neighbours' values. At the start every node in both graphs has the same value, and every node in both graphs has two neighbours with the same value, so the update gives the same result everywhere in both graphs. The values stay identical in every round, whatever the learned functions are. Any readout of identical values is identical for the two graphs.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> In block 1 the accuracy at 50 hops is exactly 0.500 with one distinct guess. Explain why the leaders' signals end up ordered the same way for every node.</summary>

Repeated multiplication by a symmetric normalised matrix pushes any starting vector towards the matrix's dominant eigenvector, scaled by a constant that depends on the start. Both columns of the signal become multiples of the same vector, with the leaders' columns scaled by the eigenvector's value at node 0 and at node 33. The comparison between the columns then has the same sign at every node, so every node gets the same guess and half are right.

</details>

## Go deeper

All sources were opened on 8 October 2026.

- [Stanford CS224W, Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/): the Fall 2026 offering, whose schedule was read, with public slides. It lists lectures on graph neural networks, a general perspective on GNNs, theory of GNNs, graph transformers, knowledge graphs, recommender systems and relational deep learning. Lecture videos are not public.
- Kipf TN and Welling M, "Semi-Supervised Classification with Graph Convolutional Networks", arXiv 1609.02907, first submitted 9 September 2016, latest version v4 of 22 February 2017, ICLR 2017. The source of the normalised propagation rule used here.
- Gilmer J and colleagues, "Neural Message Passing for Quantum Chemistry", arXiv 1704.01212, v2 of 12 June 2017. Unifies earlier models as one message passing framework.
- Xu K, Hu W, Leskovec J and Jegelka S, "How Powerful are Graph Neural Networks?", arXiv 1810.00826, v3 of 22 February 2019. The expressiveness result and the sum aggregator argument.
- Li Q, Han Z and Wu X-M, "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning", arXiv 1801.07606, AAAI 2018. Graph convolution as Laplacian smoothing and the over-smoothing warning.
- [NetworkX](https://networkx.org/): the karate club graph is Zachary's 1977 data as shipped in NetworkX 3.6.1; I did not open the original 1977 paper.

## Check yourself

- I can write a small graph as an adjacency matrix and compute the sum and mean of each node's neighbours.
- I can state the message-passing update and say what the receptive field is after k layers.
- I can explain why a layer must be equivariant to node order and check it numerically.
- I can give a pair of graphs that message passing cannot tell apart and say why.
- I can explain over-smoothing and say roughly when it starts, with a number from this chapter.

## Where to go next

Next chapter: [GCN, GraphSAGE and GAT from scratch](/docs/theory/gnn/gcn-graphsage-gat), where the fixed rules of this chapter get learned weights and three different ways to weight a neighbour. A related chapter: [link analysis, PageRank and HITS](/docs/theory/ir/link-analysis-pagerank-and-hits), which spreads a score over a graph the same way this chapter spreads a value.
