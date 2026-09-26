"""Deterministic Kahn topological sort; an edge is prerequisite -> dependent."""
from collections import defaultdict
import heapq

def topological_sort(nodes, edges):
    nodes = set(nodes)
    adjacency = defaultdict(set)
    indegree = dict.fromkeys(nodes, 0)
    for before, after in edges:
        if before not in nodes or after not in nodes:
            raise ValueError("edge endpoint not declared")
        if after not in adjacency[before]:
            adjacency[before].add(after)
            indegree[after] += 1
    ready = [node for node in nodes if indegree[node] == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        node = heapq.heappop(ready)
        result.append(node)
        for after in sorted(adjacency[node]):
            indegree[after] -= 1
            if indegree[after] == 0:
                heapq.heappush(ready, after)
    if len(result) != len(nodes):
        raise ValueError("cycle detected")
    return result

if __name__ == "__main__":
    print(topological_sort(["parse", "embed", "index", "audit"],
                           [("parse", "embed"), ("embed", "index")]))
