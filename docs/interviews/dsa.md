---
title: Data Structures & Algorithms Interview Questions (100)
sidebar_position: 37
sidebar_label: "Reference · Dsa"
---

# Data Structures & Algorithms Interview Questions (100)

:::note Foundation reference
This earlier 100-entry bank remains for prerequisite revision. Its questions were not part of the interview-source audit, and some examples are fragments rather than standalone programmes. Start with the [practical topic bank](01-python-data.md) for sourced scenarios, answered follow-ups and tested code. See [tools and versions](99-tools-versions.md) for current behaviour and [coding labs](11-coding-labs.md) for runnable implementations.
:::

## Core Concepts

<details>
<summary><strong>1. What are data structures and why are they important?</strong></summary>

**Answer:**
Data structures organize data efficiently; impact algorithm performance.

```python
# Different structures have different time complexities

# Array/List: O(1) access, O(n) insertion
arr = [1, 2, 3, 4, 5]
print(arr[2])  # O(1)

# Linked List: O(n) access, O(1) insertion if location known
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

# Stack: LIFO (Last In First Out)
stack = []
stack.append(1)  # Push: O(1)
stack.pop()      # Pop: O(1)

# Queue: FIFO (First In First Out)
from collections import deque
queue = deque()
queue.append(1)      # Enqueue: O(1)
queue.popleft()      # Dequeue: O(1)

# Hash Map: O(1) average lookup
hash_map = {'key': 'value'}
print(hash_map['key'])  # O(1)

# Binary Search Tree: O(log n) average search
# Balanced tree is important!

# Common trade-offs:
# - Arrays: Fast access, slow insertion
# - Linked lists: Fast insertion, slow access
# - Hash tables: Fast lookup, no ordering
# - Trees: Balanced access and insertion
```

**Interview Tip**: Know time/space complexity of each structure.
</details>

<details>
<summary><strong>2. What is Big O notation?</strong></summary>

**Answer:**
Describes algorithm efficiency as input size grows.

```python
# O(1) - Constant time
def get_first_element(arr):
    return arr[0]  # Always 1 operation

# O(n) - Linear time
def find_element(arr, target):
    for elem in arr:
        if elem == target:
            return True
    return False

# O(n^2) - Quadratic time (nested loops)
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

# O(log n) - Logarithmic time (binary search)
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return True
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return False

# O(n log n) - Linear logarithmic (efficient sorting)
# Merge sort, Quick sort

# O(2^n) - Exponential (very slow)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Complexity hierarchy (from fast to slow):
# O(1) < O(log n) < O(n) < O(n log n) < O(n^2) < O(2^n) < O(n!)
```

**Interview Tip**: Analyze both time and space complexity.
</details>

<details>
<summary><strong>3. What are arrays and lists?</strong></summary>

**Answer:**
Ordered collections with indexed access.

```python
# Python list (dynamic array)
arr = [1, 2, 3, 4, 5]

# Time complexities:
arr[0]              # O(1) - access by index
arr.append(6)       # O(1) amortized - add to end
arr.insert(0, 0)    # O(n) - insert at beginning
arr.remove(3)       # O(n) - find and remove
arr.pop()           # O(1) - remove from end
arr.pop(0)          # O(n) - remove from beginning

# Space: O(n)

# Common operations in interviews:
# 1. Two pointers
def two_sum(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return None

# 2. Sliding window
def max_subarray_sum(arr, k):
    max_sum = sum(arr[:k])
    current_sum = max_sum
    for i in range(k, len(arr)):
        current_sum = current_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, current_sum)
    return max_sum

# 3. Binary search
```

**Interview Tip**: Know two pointers and sliding window patterns.
</details>

<details>
<summary><strong>4. What are linked lists?</strong></summary>

**Answer:**
Sequential data structure with nodes and pointers.

```python
# Node definition
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

# Linked list operations
class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, value):  # O(n)
        new_node = Node(value)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def insert_at_head(self, value):  # O(1)
        new_node = Node(value)
        new_node.next = self.head
        self.head = new_node
    
    def search(self, value):  # O(n)
        current = self.head
        while current:
            if current.value == value:
                return True
            current = current.next
        return False
    
    def delete(self, value):  # O(n)
        if not self.head:
            return
        if self.head.value == value:
            self.head = self.head.next
            return
        current = self.head
        while current.next:
            if current.next.value == value:
                current.next = current.next.next
                return
            current = current.next

# Time complexities:
# Access: O(n)
# Search: O(n)
# Insertion: O(1) if position known, O(n) otherwise
# Deletion: O(1) if position known, O(n) otherwise

# Space: O(n)

# Common interview problems:
# - Reverse linked list
# - Merge two sorted lists
# - Detect cycle
# - Find middle
```

**Interview Tip**: Practice reverse, merge, and cycle detection problems.
</details>

<details>
<summary><strong>5. What are stacks and queues?</strong></summary>

**Answer:**
Specialized lists with restricted insertion/deletion.

```python
# Stack - LIFO (Last In First Out)
# Use case: Function calls, undo/redo, expression evaluation

class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, value):  # O(1)
        self.items.append(value)
    
    def pop(self):  # O(1)
        return self.items.pop()
    
    def peek(self):  # O(1)
        return self.items[-1]
    
    def is_empty(self):  # O(1)
        return len(self.items) == 0

# Queue - FIFO (First In First Out)
# Use case: BFS, task scheduling, printer queue

from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, value):  # O(1)
        self.items.append(value)
    
    def dequeue(self):  # O(1)
        return self.items.popleft()
    
    def is_empty(self):  # O(1)
        return len(self.items) == 0

# Example: Valid parentheses
def is_valid_parentheses(s):
    stack = Stack()
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in s:
        if char in pairs:
            stack.push(char)
        elif char in pairs.values():
            if stack.is_empty() or pairs[stack.pop()] != char:
                return False
    return stack.is_empty()

# Example: BFS (uses queue)
def bfs(graph, start):
    visited = set()
    queue = Queue()
    queue.enqueue(start)
    visited.add(start)
    
    while not queue.is_empty():
        node = queue.dequeue()
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.enqueue(neighbor)
                visited.add(neighbor)
```

**Interview Tip**: Know when to use each; practice valid parentheses problem.
</details>

<details>
<summary><strong>6. What are hash maps/dictionaries?</strong></summary>

**Answer:**
Key-value pairs with O(1) average lookup.

```python
# Python dictionary (hash map)
d = {'key': 'value', 'name': 'Alice'}

# Time complexities:
d['key']              # O(1) average, O(n) worst case
d['new_key'] = 'val'  # O(1) average
del d['key']          # O(1) average
'key' in d            # O(1) average

# Collision handling strategies:
# 1. Chaining (separate lists)
# 2. Open addressing (find next empty slot)

# Common interview problems:

# Two sum
def two_sum(arr, target):
    seen = {}
    for num in arr:
        complement = target - num
        if complement in seen:
            return [seen[complement], arr.index(num)]
        seen[num] = arr.index(num)
    return None

# First non-repeating character
def first_non_repeating(s):
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    for char in s:
        if char_count[char] == 1:
            return char
    return None

# Group anagrams
def group_anagrams(words):
    groups = {}
    for word in words:
        key = ''.join(sorted(word))
        if key not in groups:
            groups[key] = []
        groups[key].append(word)
    return list(groups.values())

# Load factor (resize when needed)
# Typically resize when load_factor = size / capacity > 0.7
```

**Interview Tip**: Master hash tables for interview problems.
</details>

<details>
<summary><strong>7. What are trees and BSTs?</strong></summary>

**Answer:**
Hierarchical structures with parent-child relationships.

```python
# Binary tree node
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Binary Search Tree (BST)
class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, value):  # O(log n) average, O(n) worst
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):  # O(log n) average
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)

# Tree traversals:

# Inorder (Left-Root-Right): gives sorted sequence for BST
def inorder(node):
    if node:
        inorder(node.left)
        print(node.value)
        inorder(node.right)

# Preorder (Root-Left-Right): useful for copying tree
def preorder(node):
    if node:
        print(node.value)
        preorder(node.left)
        preorder(node.right)

# Postorder (Left-Right-Root): useful for deletion
def postorder(node):
    if node:
        postorder(node.left)
        postorder(node.right)
        print(node.value)

# Level-order (BFS)
def level_order(root):
    if not root:
        return
    queue = [root]
    while queue:
        node = queue.pop(0)
        print(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
```

**Interview Tip**: Know BST properties and all traversal methods.
</details>

<details>
<summary><strong>8. What are heaps?</strong></summary>

**Answer:**
Complete binary trees with heap property (min or max).

```python
import heapq

# Python uses min-heap by default
min_heap = [3, 1, 4, 1, 5, 9]
heapq.heapify(min_heap)  # O(n) - convert to heap

heapq.heappush(min_heap, 2)  # O(log n) - add element
min_element = heapq.heappop(min_heap)  # O(log n) - remove min

# Max-heap (negate values)
max_heap = [-3, -1, -4, -1, -5, -9]
heapq.heapify(max_heap)
heapq.heappush(max_heap, -2)
max_element = -heapq.heappop(max_heap)

# Common use cases:
# - Priority queue
# - Find k largest/smallest
# - Heap sort

# K largest elements
def find_k_largest(arr, k):
    return heapq.nlargest(k, arr)

# Merge K sorted lists
def merge_k_lists(lists):
    heap = []
    result = []
    
    # Add first element from each list to heap
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
    return result

# Time complexities:
# Insert: O(log n)
# Delete min: O(log n)
# Heapify: O(n)
# Space: O(n)
```

**Interview Tip**: Master heap operations and k-largest problem.
</details>

<details>
<summary><strong>9. What are graphs?</strong></summary>

**Answer:**
Networks of nodes connected by edges.

```python
# Adjacency list representation
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}

# DFS (Depth-First Search) - O(V + E)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# BFS (Breadth-First Search) - O(V + E)
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Dijkstra's shortest path - O((V + E) log V)
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        if current_dist > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
```

**Interview Tip**: Know DFS, BFS, and shortest path algorithms.
</details>

<details>
<summary><strong>10. What are sorting algorithms?</strong></summary>

**Answer:**
Arrange elements in order with different efficiency tradeoffs.

```python
# Bubble sort - O(n^2)
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Merge sort - O(n log n), O(n) space
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Quick sort - O(n log n) average, O(n^2) worst
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# Python's built-in (Timsort - O(n log n))
sorted_arr = sorted(arr)
arr.sort()

# Summary of complexities:
# Bubble sort: O(n^2) time, O(1) space
# Merge sort: O(n log n) time, O(n) space
# Quick sort: O(n log n) average, O(n^2) worst, O(log n) space
# Heap sort: O(n log n) time, O(1) space
# Insertion sort: O(n^2) time, O(1) space
```

**Interview Tip**: Know merge sort and quick sort; understand when to use each.
</details>

---

## Linked Lists & Stacks

<details>
<summary><strong>11. How do you implement a linked list and detect a cycle?</strong></summary>

**Answer:**
Linked list is a chain of nodes; cycle detection uses Floyd's two-pointer algorithm.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Build a linked list: 1 -> 2 -> 3 -> 4 -> None
def build_list(values):
    dummy = ListNode(0)
    cur = dummy
    for v in values:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next

# Floyd's cycle detection (tortoise and hare)
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False

# Find cycle start
def detect_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            slow = head
            while slow is not fast:
                slow = slow.next
                fast = fast.next
            return slow  # cycle start node
    return None

# Reverse a linked list
def reverse_list(head):
    prev, cur = None, head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev

# Find middle node
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

head = build_list([1, 2, 3, 4, 5])
print(find_middle(head).val)  # 3
```

**Key insight**: Fast pointer moves 2x; when they meet, reset one to head and advance both 1x to find cycle start.
</details>

<details>
<summary><strong>12. How do you implement a stack that supports getMin() in O(1)?</strong></summary>

**Answer:**
Use an auxiliary min-stack that tracks minimums at each state.

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        min_val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(min_val)

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]

ms = MinStack()
for v in [5, 3, 7, 2, 8]:
    ms.push(v)

print(ms.getMin())   # 2
ms.pop()
print(ms.getMin())   # 2
ms.pop()
print(ms.getMin())   # 3

# Balanced parentheses using stack
def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for ch in s:
        if ch in mapping:
            top = stack.pop() if stack else '#'
            if mapping[ch] != top:
                return False
        else:
            stack.append(ch)
    return not stack

print(is_valid("({[]})"))  # True
print(is_valid("([)]"))    # False
```
</details>

<details>
<summary><strong>13. How do you implement a queue using two stacks?</strong></summary>

**Answer:**
Amortized O(1) enqueue/dequeue by lazy transfer between stacks.

```python
class MyQueue:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x):
        self.in_stack.append(x)

    def _transfer(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())

    def pop(self):
        self._transfer()
        return self.out_stack.pop()

    def peek(self):
        self._transfer()
        return self.out_stack[-1]

    def empty(self):
        return not self.in_stack and not self.out_stack

q = MyQueue()
q.push(1); q.push(2); q.push(3)
print(q.pop())   # 1 (FIFO)
print(q.peek())  # 2
print(q.pop())   # 2
```
</details>

## Trees & Heaps

<details>
<summary><strong>14. How do you traverse a binary tree (all orders)?</strong></summary>

**Answer:**
Inorder, preorder, postorder, and level-order traversal.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Recursive traversals
def inorder(root):    # Left, Root, Right  -> sorted for BST
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def preorder(root):   # Root, Left, Right  -> used to copy tree
    if not root: return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def postorder(root):  # Left, Right, Root  -> used to delete tree
    if not root: return []
    return postorder(root.left) + postorder(root.right) + [root.val]

# Iterative inorder (common interview question)
def inorder_iter(root):
    result, stack, cur = [], [], root
    while cur or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        result.append(cur.val)
        cur = cur.right
    return result

# Level-order (BFS)
from collections import deque
def level_order(root):
    if not root: return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result

# Build tree: 1 -> (2,3) -> (4,5,None,6)
root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3, None, TreeNode(6)))
print(inorder(root))       # [4, 2, 5, 1, 3, 6]
print(level_order(root))   # [[1], [2, 3], [4, 5, 6]]
```
</details>

<details>
<summary><strong>15. How do you find the lowest common ancestor (LCA) of a binary tree?</strong></summary>

**Answer:**
Recursive postorder: if both subtrees return a node, current node is LCA.

```python
def lca(root, p, q):
    if not root or root is p or root is q:
        return root
    left  = lca(root.left, p, q)
    right = lca(root.right, p, q)
    if left and right:
        return root   # p and q are in different subtrees
    return left or right

# For BST: exploit ordering property
def lca_bst(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root

# Max depth of binary tree
def max_depth(root):
    if not root: return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# Check if balanced
def is_balanced(root):
    def height(node):
        if not node: return 0
        lh = height(node.left)
        if lh == -1: return -1
        rh = height(node.right)
        if rh == -1: return -1
        if abs(lh - rh) > 1: return -1
        return 1 + max(lh, rh)
    return height(root) != -1
```
</details>

<details>
<summary><strong>16. How does a heap work and what is heapq in Python?</strong></summary>

**Answer:**
Heap is a complete binary tree satisfying heap property; Python's heapq is a min-heap.

```python
import heapq

# Min-heap operations
heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 1)
heapq.heappush(heap, 3)
print(heap[0])           # 1 (min)
print(heapq.heappop(heap))  # 1

# Max-heap: negate values
max_heap = []
for v in [5, 1, 3]:
    heapq.heappush(max_heap, -v)
print(-heapq.heappop(max_heap))  # 5

# Heapify in O(n)
nums = [3, 1, 4, 1, 5, 9]
heapq.heapify(nums)
print(nums[0])  # 1

# K largest elements
def k_largest(nums, k):
    return heapq.nlargest(k, nums)

# K smallest
def k_smallest(nums, k):
    return heapq.nsmallest(k, nums)

# Merge k sorted lists
def merge_k_sorted(lists):
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    result = []
    while heap:
        val, i, j = heapq.heappop(heap)
        result.append(val)
        if j + 1 < len(lists[i]):
            heapq.heappush(heap, (lists[i][j+1], i, j+1))
    return result

lists = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
print(merge_k_sorted(lists))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

**Time**: push/pop O(log n), heapify O(n), peek O(1).
</details>

## Hashing & Searching

<details>
<summary><strong>17. How does a hash table work and handle collisions?</strong></summary>

**Answer:**
Hash function maps keys to buckets; collisions resolved by chaining or open addressing.

```python
# Simple hash table with chaining
class HashTable:
    def __init__(self, size=16):
        self.size = size
        self.buckets = [[] for _ in range(size)]

    def _hash(self, key):
        return hash(key) % self.size

    def put(self, key, value):
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx][i] = (key, value)
                return
        self.buckets[idx].append((key, value))

    def get(self, key):
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        return None

    def remove(self, key):
        idx = self._hash(key)
        self.buckets[idx] = [(k, v) for k, v in self.buckets[idx] if k != key]

ht = HashTable()
ht.put("name", "Alice")
ht.put("age", 30)
print(ht.get("name"))   # Alice

# Two-sum using hash map: O(n)
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

print(two_sum([2, 7, 11, 15], 9))  # [0, 1]

# Group anagrams
from collections import defaultdict
def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        groups[tuple(sorted(s))].append(s)
    return list(groups.values())
```
</details>

<details>
<summary><strong>18. What is binary search and when do you use it?</strong></summary>

**Answer:**
Binary search runs in O(log n) on sorted arrays; also applies to search spaces.

```python
# Standard binary search
def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# Find first/last occurrence
def find_first(nums, target):
    lo, hi, result = 0, len(nums) - 1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            result = mid
            hi = mid - 1   # keep searching left
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return result

# Search in rotated sorted array
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:   # left half is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                        # right half is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1

# Binary search on answer space: minimum capacity to ship packages in D days
def ship_within_days(weights, D):
    lo, hi = max(weights), sum(weights)
    while lo < hi:
        mid = (lo + hi) // 2
        days, cap = 1, 0
        for w in weights:
            if cap + w > mid:
                days += 1
                cap = 0
            cap += w
        if days <= D:
            hi = mid
        else:
            lo = mid + 1
    return lo

print(binary_search([1, 3, 5, 7, 9], 5))  # 2
```
</details>

## Dynamic Programming

<details>
<summary><strong>19. What is dynamic programming and how do you approach DP problems?</strong></summary>

**Answer:**
DP solves problems by breaking them into overlapping subproblems and storing results.

```python
# Framework: 1) Define state, 2) Recurrence, 3) Base case, 4) Order

# Classic: Fibonacci (top-down memoization)
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo(n):
    if n <= 1: return n
    return fib_memo(n-1) + fib_memo(n-2)

# Bottom-up tabulation (space-optimized)
def fib_dp(n):
    if n <= 1: return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

# Coin change: minimum coins to make amount
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

print(coin_change([1, 5, 6, 9], 11))  # 2 (5+6)

# Longest Increasing Subsequence (LIS) O(n log n)
import bisect
def lis(nums):
    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)

print(lis([10, 9, 2, 5, 3, 7, 101, 18]))  # 4

# House robber
def rob(nums):
    prev2, prev1 = 0, 0
    for n in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + n)
    return prev1

print(rob([2, 7, 9, 3, 1]))  # 12
```
</details>

<details>
<summary><strong>20. How do you solve the 0/1 knapsack problem?</strong></summary>

**Answer:**
Classic DP: dp[i][w] = max value using first i items with capacity w.

```python
def knapsack(weights, values, capacity):
    n = len(weights)
    # dp[i][w] = max value with i items, capacity w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i
            dp[i][w] = dp[i-1][w]
            # Take item i if it fits
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    
    return dp[n][capacity]

# Space optimized (1D)
def knapsack_1d(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for w in range(capacity, weights[i]-1, -1):  # traverse right to left!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]

weights = [2, 3, 4, 5]
values  = [3, 4, 5, 6]
print(knapsack(weights, values, 8))     # 10
print(knapsack_1d(weights, values, 8))  # 10

# Subset sum (special case: values = weights)
def can_partition(nums):
    total = sum(nums)
    if total % 2: return False
    target = total // 2
    dp = {0}
    for n in nums:
        dp |= {x + n for x in dp}
    return target in dp

print(can_partition([1, 5, 11, 5]))  # True (1+5+5=11)
```
</details>

## Graph Algorithms

<details>
<summary><strong>21. How do you implement DFS and BFS on a graph?</strong></summary>

**Answer:**
DFS uses recursion/stack (explores deep); BFS uses queue (explores level by level).

```python
from collections import defaultdict, deque

# Graph as adjacency list
graph = defaultdict(list)
edges = [(0,1),(0,2),(1,3),(2,3),(3,4)]
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

# DFS - recursive
def dfs(graph, node, visited=None):
    if visited is None: visited = set()
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

# DFS - iterative
def dfs_iter(graph, start):
    visited, stack = set(), [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(n for n in graph[node] if n not in visited)
    return visited

# BFS - iterative (find shortest path)
def bfs(graph, start, end):
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

# Number of connected components
def count_components(n, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    visited = set()
    count = 0
    for i in range(n):
        if i not in visited:
            dfs(graph, i, visited)
            count += 1
    return count
```
</details>

<details>
<summary><strong>22. What is topological sort and when do you use it?</strong></summary>

**Answer:**
Linear ordering of DAG nodes such that every edge goes from earlier to later — used for dependency resolution.

```python
from collections import defaultdict, deque

# Kahn's algorithm (BFS-based)
def topo_sort_kahn(n, prerequisites):
    graph = defaultdict(list)
    in_degree = [0] * n
    for a, b in prerequisites:
        graph[b].append(a)
        in_degree[a] += 1
    
    queue = deque(i for i in range(n) if in_degree[i] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return order if len(order) == n else []  # empty = cycle exists

# DFS-based topological sort
def topo_sort_dfs(n, prerequisites):
    graph = defaultdict(list)
    for a, b in prerequisites:
        graph[b].append(a)
    
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    result = []
    
    def dfs(v):
        color[v] = GRAY
        for u in graph[v]:
            if color[u] == GRAY: return False   # cycle
            if color[u] == WHITE:
                if not dfs(u): return False
        color[v] = BLACK
        result.append(v)
        return True
    
    for i in range(n):
        if color[i] == WHITE:
            if not dfs(i): return []
    return result[::-1]

# Course schedule: can finish all courses?
prereqs = [[1,0],[2,0],[3,1],[3,2]]
print(topo_sort_kahn(4, prereqs))  # [0, 1, 2, 3] or similar
```
</details>

<details>
<summary><strong>23. How does Union-Find (Disjoint Set Union) work?</strong></summary>

**Answer:**
DSU efficiently tracks connected components with path compression and union by rank.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.components -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

# Detect cycle in undirected graph
def has_cycle(n, edges):
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return True  # already same component = cycle
    return False

# Kruskal's MST using DSU
def kruskal_mst(n, edges):
    edges.sort(key=lambda x: x[2])  # sort by weight
    uf = UnionFind(n)
    mst_cost, mst_edges = 0, []
    for u, v, w in edges:
        if uf.union(u, v):
            mst_cost += w
            mst_edges.append((u, v, w))
    return mst_cost, mst_edges

edges = [(0,1,4),(0,2,3),(1,2,1),(1,3,2),(2,3,4)]
print(kruskal_mst(4, edges))  # cost=6
```
</details>

## Sliding Window & Two Pointers

<details>
<summary><strong>24. What is the sliding window technique?</strong></summary>

**Answer:**
Maintain a window over a sequence to avoid recomputing from scratch — O(n) instead of O(n²).

```python
# Fixed window: max sum of subarray of size k
def max_sum_subarray(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i-k]
        max_sum = max(max_sum, window_sum)
    return max_sum

# Variable window: longest substring without repeating chars
def length_of_longest_substring(s):
    char_set = set()
    left = max_len = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len

print(length_of_longest_substring("abcabcbb"))  # 3

# Minimum window substring
from collections import Counter
def min_window(s, t):
    need = Counter(t)
    missing = len(t)
    start = result_start = result_end = 0
    for end, ch in enumerate(s, 1):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        if missing == 0:
            while need[s[start]] < 0:
                need[s[start]] += 1
                start += 1
            if not result_end or end - start < result_end - result_start:
                result_start, result_end = start, end
            need[s[start]] += 1
            missing += 1
            start += 1
    return s[result_start:result_end]

print(min_window("ADOBECODEBANC", "ABC"))  # "BANC"
```
</details>

<details>
<summary><strong>25. What is the two-pointer technique?</strong></summary>

**Answer:**
Two pointers move toward each other or in the same direction to solve array/string problems in O(n).

```python
# Two sum in sorted array
def two_sum_sorted(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        s = nums[lo] + nums[hi]
        if s == target:   return [lo, hi]
        elif s < target:  lo += 1
        else:             hi -= 1
    return []

# Container with most water
def max_water(height):
    lo, hi = 0, len(height) - 1
    max_area = 0
    while lo < hi:
        area = min(height[lo], height[hi]) * (hi - lo)
        max_area = max(max_area, area)
        if height[lo] < height[hi]:
            lo += 1
        else:
            hi -= 1
    return max_area

# 3Sum (sort + two pointers)
def three_sum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]: continue
        lo, hi = i+1, len(nums)-1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s == 0:
                result.append([nums[i], nums[lo], nums[hi]])
                while lo < hi and nums[lo] == nums[lo+1]: lo += 1
                while lo < hi and nums[hi] == nums[hi-1]: hi -= 1
                lo += 1; hi -= 1
            elif s < 0: lo += 1
            else:       hi -= 1
    return result

print(three_sum([-1, 0, 1, 2, -1, -4]))  # [[-1,-1,2],[-1,0,1]]
```
</details>

## Recursion & Backtracking

<details>
<summary><strong>26. How do you generate all permutations and subsets?</strong></summary>

**Answer:**
Backtracking explores all choices by making/undoing decisions at each step.

```python
# All permutations
def permutations(nums):
    result = []
    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return
        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current, remaining[:i] + remaining[i+1:])
            current.pop()
    backtrack([], nums)
    return result

print(permutations([1,2,3]))  # 6 permutations

# All subsets (power set)
def subsets(nums):
    result = []
    def backtrack(start, current):
        result.append(current[:])
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    backtrack(0, [])
    return result

print(len(subsets([1,2,3])))  # 8 (2^3)

# N-Queens
def solve_n_queens(n):
    result = []
    cols = set(); diag1 = set(); diag2 = set()
    board = [['.']*n for _ in range(n)]
    
    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        for col in range(n):
            if col in cols or (row-col) in diag1 or (row+col) in diag2:
                continue
            cols.add(col); diag1.add(row-col); diag2.add(row+col)
            board[row][col] = 'Q'
            backtrack(row + 1)
            board[row][col] = '.'
            cols.remove(col); diag1.remove(row-col); diag2.remove(row+col)
    
    backtrack(0)
    return result

print(len(solve_n_queens(8)))  # 92
```
</details>

<details>
<summary><strong>27. How do you solve word search using backtracking?</strong></summary>

**Answer:**
DFS with backtracking on a 2D grid, marking visited cells temporarily.

```python
def word_search(board, word):
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, idx):
        if idx == len(word): return True
        if r < 0 or r >= rows or c < 0 or c >= cols: return False
        if board[r][c] != word[idx]: return False
        
        temp = board[r][c]
        board[r][c] = '#'   # mark visited
        
        found = (dfs(r+1,c,idx+1) or dfs(r-1,c,idx+1) or
                 dfs(r,c+1,idx+1) or dfs(r,c-1,idx+1))
        
        board[r][c] = temp  # restore
        return found
    
    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False

board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
print(word_search(board, "ABCCED"))  # True
print(word_search(board, "ABCB"))    # False

# Combination sum
def combination_sum(candidates, target):
    result = []
    def backtrack(start, current, remaining):
        if remaining == 0:
            result.append(current[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining: break
            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])
            current.pop()
    candidates.sort()
    backtrack(0, [], target)
    return result

print(combination_sum([2,3,6,7], 7))  # [[2,2,3],[7]]
```
</details>

## Advanced DP

<details>
<summary><strong>28. How do you solve string DP problems (LCS, edit distance)?</strong></summary>

**Answer:**
2D DP table where rows/cols represent characters of the two strings.

```python
# Longest Common Subsequence
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

print(lcs("ABCBDAB", "BDCAB"))  # 4 (BCAB)

# Edit distance (Levenshtein)
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # delete
                                   dp[i][j-1],    # insert
                                   dp[i-1][j-1])  # replace
    return dp[m][n]

print(edit_distance("horse", "ros"))  # 3

# Palindrome partitioning: min cuts
def min_cut(s):
    n = len(s)
    is_pal = [[False]*n for _ in range(n)]
    for length in range(1, n+1):
        for i in range(n-length+1):
            j = i + length - 1
            is_pal[i][j] = (s[i]==s[j]) and (length<=2 or is_pal[i+1][j-1])
    
    dp = [float('inf')] * n
    for j in range(n):
        if is_pal[0][j]:
            dp[j] = 0
        else:
            for i in range(1, j+1):
                if is_pal[i][j]:
                    dp[j] = min(dp[j], dp[i-1]+1)
    return dp[n-1]
```
</details>

<details>
<summary><strong>29. How do you solve interval problems (merge intervals, meeting rooms)?</strong></summary>

**Answer:**
Sort by start time, then greedily merge or count overlaps.

```python
# Merge overlapping intervals
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

print(merge_intervals([[1,3],[2,6],[8,10],[15,18]]))
# [[1,6],[8,10],[15,18]]

# Meeting rooms II: min conference rooms needed
import heapq
def min_meeting_rooms(intervals):
    intervals.sort(key=lambda x: x[0])
    heap = []   # tracks end times
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heapreplace(heap, end)
        else:
            heapq.heappush(heap, end)
    return len(heap)

meetings = [[0,30],[5,10],[15,20]]
print(min_meeting_rooms(meetings))  # 2

# Non-overlapping intervals (min removals)
def erase_overlap_intervals(intervals):
    intervals.sort(key=lambda x: x[1])  # sort by end
    count, prev_end = 0, float('-inf')
    for start, end in intervals:
        if start >= prev_end:
            prev_end = end
        else:
            count += 1  # remove this interval
    return count
```
</details>

<details>
<summary><strong>30. How does Dijkstra's algorithm work?</strong></summary>

**Answer:**
Greedy shortest-path using min-heap; works for non-negative weighted graphs in O((V+E) log V).

```python
import heapq
from collections import defaultdict

def dijkstra(graph, start, end):
    # graph: {node: [(neighbor, weight), ...]}
    dist = {start: 0}
    heap = [(0, start)]
    
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, float('inf')): continue
        if node == end: return d
        
        for neighbor, weight in graph[node]:
            new_dist = d + weight
            if new_dist < dist.get(neighbor, float('inf')):
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    
    return dist.get(end, float('inf'))

# Build graph
g = defaultdict(list)
for u, v, w in [(0,1,4),(0,2,1),(2,1,2),(1,3,1),(2,3,5)]:
    g[u].append((v, w))
    g[v].append((u, w))

print(dijkstra(g, 0, 3))  # 4 (0->2->1->3: 1+2+1)

# Network delay time (single source, all destinations)
def network_delay(times, n, k):
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    
    dist = {k: 0}
    heap = [(0, k)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')): continue
        for v, w in graph[u]:
            if d + w < dist.get(v, float('inf')):
                dist[v] = d + w
                heapq.heappush(heap, (d+w, v))
    
    if len(dist) < n: return -1
    return max(dist.values())
```
</details>

## Tries & Advanced Data Structures

<details>
<summary><strong>31. How do you implement a Trie (prefix tree)?</strong></summary>

**Answer:**
Trie stores strings character by character — O(L) insert/search where L is word length.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children: return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children: return False
            node = node.children[ch]
        return True

    def words_with_prefix(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children: return []
            node = node.children[ch]
        result = []
        def dfs(n, path):
            if n.is_end: result.append(prefix[:-len(path)] + path if path else prefix)
            for ch, child in n.children.items():
                dfs(child, path + ch)
        dfs(node, "")
        return result

trie = Trie()
for w in ["apple", "app", "application", "apply"]:
    trie.insert(w)
print(trie.search("app"))        # True
print(trie.starts_with("appl")) # True
```
</details>

<details>
<summary><strong>32. What is a segment tree and when do you use it?</strong></summary>

**Answer:**
Segment tree answers range queries (sum, min, max) in O(log n) with O(log n) updates.

```python
class SegmentTree:
    def __init__(self, nums):
        n = len(nums)
        self.n = n
        self.tree = [0] * (4 * n)
        self._build(nums, 0, 0, n - 1)

    def _build(self, nums, node, start, end):
        if start == end:
            self.tree[node] = nums[start]
        else:
            mid = (start + end) // 2
            self._build(nums, 2*node+1, start, mid)
            self._build(nums, 2*node+2, mid+1, end)
            self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]

    def update(self, idx, val, node=0, start=0, end=None):
        if end is None: end = self.n - 1
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self.update(idx, val, 2*node+1, start, mid)
            else:
                self.update(idx, val, 2*node+2, mid+1, end)
            self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]

    def query(self, l, r, node=0, start=0, end=None):
        if end is None: end = self.n - 1
        if r < start or end < l: return 0
        if l <= start and end <= r: return self.tree[node]
        mid = (start + end) // 2
        return (self.query(l, r, 2*node+1, start, mid) +
                self.query(l, r, 2*node+2, mid+1, end))

st = SegmentTree([1, 3, 5, 7, 9, 11])
print(st.query(1, 3))   # 15 (3+5+7)
st.update(1, 10)
print(st.query(1, 3))   # 22 (10+5+7)
```

**Use cases**: Range sum/min/max queries with point updates; can extend to lazy propagation for range updates.
</details>

<details>
<summary><strong>33. What is a monotonic stack and when do you use it?</strong></summary>

**Answer:**
Stack maintained in monotonically increasing or decreasing order — solves "next greater/smaller element" in O(n).

```python
# Next greater element (monotonic decreasing stack)
def next_greater(nums):
    n = len(nums)
    result = [-1] * n
    stack = []  # indices
    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    return result

print(next_greater([2, 1, 2, 4, 3]))  # [4, 2, 4, -1, -1]

# Largest rectangle in histogram
def largest_rectangle(heights):
    stack = [-1]
    max_area = 0
    for i, h in enumerate(heights):
        while stack[-1] != -1 and heights[stack[-1]] >= h:
            height = heights[stack.pop()]
            width = i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    while stack[-1] != -1:
        height = heights[stack.pop()]
        width = len(heights) - stack[-1] - 1
        max_area = max(max_area, height * width)
    return max_area

print(largest_rectangle([2,1,5,6,2,3]))  # 10

# Daily temperatures
def daily_temperatures(temps):
    result = [0] * len(temps)
    stack = []
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)
    return result
```
</details>

## Bit Manipulation

<details>
<summary><strong>34. What are common bit manipulation tricks?</strong></summary>

**Answer:**
Bitwise operations enable O(1) solutions for many counting and flag problems.

```python
# Core operations
n = 42  # 0b101010

# Check if bit i is set
def is_set(n, i): return (n >> i) & 1

# Set bit i
def set_bit(n, i): return n | (1 << i)

# Clear bit i
def clear_bit(n, i): return n & ~(1 << i)

# Toggle bit i
def toggle_bit(n, i): return n ^ (1 << i)

# Count set bits (Brian Kernighan's)
def count_bits(n):
    count = 0
    while n:
        n &= n - 1   # removes lowest set bit
        count += 1
    return count

print(count_bits(42))  # 3

# Power of 2 check
def is_power_of_two(n): return n > 0 and (n & (n-1)) == 0

# XOR tricks: a ^ a = 0, a ^ 0 = a
# Single number: only element appearing once
def single_number(nums):
    result = 0
    for n in nums: result ^= n
    return result

print(single_number([4, 1, 2, 1, 2]))  # 4

# Subsets using bitmask
def subsets_bitmask(nums):
    n = len(nums)
    return [[nums[j] for j in range(n) if mask & (1<<j)]
            for mask in range(1 << n)]

# Reverse bits
def reverse_bits(n):
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result
```
</details>

## Greedy Algorithms

<details>
<summary><strong>35. What are greedy algorithms and what are classic examples?</strong></summary>

**Answer:**
Greedy makes the locally optimal choice at each step — works when greedy choice property and optimal substructure hold.

```python
# Activity selection (max non-overlapping activities)
def activity_selection(activities):
    activities.sort(key=lambda x: x[1])  # sort by end time
    selected = [activities[0]]
    for start, end in activities[1:]:
        if start >= selected[-1][1]:
            selected.append((start, end))
    return selected

acts = [(1,4),(3,5),(0,6),(5,7),(3,9),(5,9),(6,10),(8,11),(8,12),(2,14),(12,16)]
print(len(activity_selection(acts)))  # 4

# Jump game: can you reach the end?
def can_jump(nums):
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach: return False
        max_reach = max(max_reach, i + jump)
    return True

# Jump game II: minimum jumps
def jump(nums):
    jumps = cur_end = cur_far = 0
    for i in range(len(nums) - 1):
        cur_far = max(cur_far, i + nums[i])
        if i == cur_end:
            jumps += 1
            cur_end = cur_far
    return jumps

print(jump([2, 3, 1, 1, 4]))  # 2

# Gas station: circular route
def can_complete_circuit(gas, cost):
    if sum(gas) < sum(cost): return -1
    tank = start = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start
```
</details>

## Additional Questions (36-100)

<details>
<summary><strong>36. Floyd-Warshall Algorithm (All-Pairs Shortest Path)</strong></summary>

```python

def floyd_warshall(graph, n):
    dist = [[float('inf')]*n for _ in range(n)]
    for i in range(n): dist[i][i] = 0
    for u, v, w in graph: dist[u][v] = w
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist
```
</details>

<details>
<summary><strong>37. Bellman-Ford (handles negative weights)</strong></summary>

```python

def bellman_ford(edges, n, src):
    dist = [float('inf')] * n
    dist[src] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    # Check for negative cycle
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return None  # negative cycle exists
    return dist
```
</details>

<details>
<summary><strong>38. Prim's MST Algorithm</strong></summary>

```python

import heapq
from collections import defaultdict

def prims_mst(n, edges):
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((w, v))
        graph[v].append((w, u))
    
    visited = set([0])
    heap = list(graph[0])
    heapq.heapify(heap)
    cost = 0
    
    while heap and len(visited) < n:
        w, v = heapq.heappop(heap)
        if v not in visited:
            visited.add(v)
            cost += w
            for edge in graph[v]:
                heapq.heappush(heap, edge)
    return cost
```
</details>

<details>
<summary><strong>39. Binary Indexed Tree (Fenwick Tree)</strong></summary>

```python

class BIT:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)   # move to parent
    
    def query(self, i):      # prefix sum [1..i]
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)   # remove lowest set bit
        return s
    
    def range_query(self, l, r):
        return self.query(r) - self.query(l - 1)

bit = BIT(5)
for i, v in enumerate([1,2,3,4,5], 1):
    bit.update(i, v)
print(bit.range_query(2, 4))  # 9 (2+3+4)
```
</details>

<details>
<summary><strong>40. LRU Cache Implementation</strong></summary>

```python

from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = OrderedDict()
    
    def get(self, key):
        if key not in self.cache: return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)

lru = LRUCache(2)
lru.put(1, 1); lru.put(2, 2)
print(lru.get(1))   # 1
lru.put(3, 3)       # evicts key 2
print(lru.get(2))   # -1
```
</details>

<details>
<summary><strong>41. Matrix DFS - Number of Islands</strong></summary>

```python

def num_islands(grid):
    rows, cols = len(grid), len(grid[0])
    count = 0
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            dfs(r+dr, c+dc)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1
    return count
```
</details>

<details>
<summary><strong>42. Longest Palindromic Substring</strong></summary>

```python

def longest_palindrome(s):
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return s[l+1:r]
    
    result = ""
    for i in range(len(s)):
        odd  = expand(i, i)
        even = expand(i, i+1)
        result = max(result, odd, even, key=len)
    return result

print(longest_palindrome("babad"))  # "bab" or "aba"
```
</details>

<details>
<summary><strong>43. Trap Rain Water</strong></summary>

```python

def trap(height):
    lo, hi = 0, len(height) - 1
    left_max = right_max = water = 0
    while lo < hi:
        if height[lo] < height[hi]:
            if height[lo] >= left_max: left_max = height[lo]
            else: water += left_max - height[lo]
            lo += 1
        else:
            if height[hi] >= right_max: right_max = height[hi]
            else: water += right_max - height[hi]
            hi -= 1
    return water

print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))  # 6
```
</details>

<details>
<summary><strong>44. Serialize and Deserialize Binary Tree</strong></summary>

```python

class Codec:
    def serialize(self, root):
        if not root: return "null"
        return f"{root.val},{self.serialize(root.left)},{self.serialize(root.right)}"
    
    def deserialize(self, data):
        vals = iter(data.split(','))
        def build():
            val = next(vals)
            if val == 'null': return None
            node = TreeNode(int(val))
            node.left = build()
            node.right = build()
            return node
        return build()
```
</details>

<details>
<summary><strong>45. Clone Graph</strong></summary>

```python

def clone_graph(node):
    if not node: return None
    visited = {}
    def dfs(n):
        if n in visited: return visited[n]
        clone = Node(n.val)
        visited[n] = clone
        for nb in n.neighbors:
            clone.neighbors.append(dfs(nb))
        return clone
    return dfs(node)
```
</details>

<details>
<summary><strong>46. Decode Ways (DP on strings)</strong></summary>

```python

def num_decodings(s):
    if not s or s[0] == '0': return 0
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1; dp[1] = 1
    for i in range(2, n + 1):
        one = int(s[i-1])
        two = int(s[i-2:i])
        if one != 0: dp[i] += dp[i-1]
        if 10 <= two <= 26: dp[i] += dp[i-2]
    return dp[n]

print(num_decodings("226"))  # 3 (2-2-6, 22-6, 2-26)
```
</details>

<details>
<summary><strong>47. Minimum Path Sum in Grid</strong></summary>

```python

def min_path_sum(grid):
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0: continue
            elif r == 0:    grid[r][c] += grid[r][c-1]
            elif c == 0:    grid[r][c] += grid[r-1][c]
            else:           grid[r][c] += min(grid[r-1][c], grid[r][c-1])
    return grid[-1][-1]
```
</details>

<details>
<summary><strong>48. Spiral Matrix</strong></summary>

```python

def spiral_order(matrix):
    result = []
    while matrix:
        result += matrix.pop(0)
        matrix = list(zip(*matrix))[::-1]
    return result
```
</details>

<details>
<summary><strong>49. Merge K Sorted Arrays</strong></summary>

```python

import heapq
def merge_k_arrays(arrays):
    heap = [(arr[0], i, 0) for i, arr in enumerate(arrays) if arr]
    heapq.heapify(heap)
    result = []
    while heap:
        val, i, j = heapq.heappop(heap)
        result.append(val)
        if j + 1 < len(arrays[i]):
            heapq.heappush(heap, (arrays[i][j+1], i, j+1))
    return result
```
</details>

<details>
<summary><strong>50. Find Median from Data Stream</strong></summary>

```python

import heapq
class MedianFinder:
    def __init__(self):
        self.lo = []  # max heap (negated)
        self.hi = []  # min heap
    
    def addNum(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))
    
    def findMedian(self):
        if len(self.lo) > len(self.hi): return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```
</details>

<details>
<summary><strong>51. Word Ladder (BFS shortest transformation)</strong></summary>

```python

from collections import deque
def word_ladder(beginWord, endWord, wordList):
    wordSet = set(wordList)
    if endWord not in wordSet: return 0
    queue = deque([(beginWord, 1)])
    while queue:
        word, length = queue.popleft()
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                if next_word == endWord: return length + 1
                if next_word in wordSet:
                    wordSet.remove(next_word)
                    queue.append((next_word, length + 1))
    return 0
```
</details>

<details>
<summary><strong>52. Palindrome Linked List</strong></summary>

```python

def is_palindrome_list(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    # Reverse second half
    prev = None
    while slow:
        nxt = slow.next
        slow.next = prev
        prev = slow
        slow = nxt
    # Compare
    left, right = head, prev
    while right:
        if left.val != right.val: return False
        left = left.next; right = right.next
    return True
```
</details>

<details>
<summary><strong>53. Valid BST</strong></summary>

```python

def is_valid_bst(root, lo=float('-inf'), hi=float('inf')):
    if not root: return True
    if not (lo < root.val < hi): return False
    return (is_valid_bst(root.left, lo, root.val) and
            is_valid_bst(root.right, root.val, hi))
```
</details>

<details>
<summary><strong>54. Kth Smallest in BST</strong></summary>

```python

def kth_smallest(root, k):
    stack, count = [], 0
    cur = root
    while cur or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        count += 1
        if count == k: return cur.val
        cur = cur.right
```
</details>

<details>
<summary><strong>55. Binary Tree Maximum Path Sum</strong></summary>

```python

def max_path_sum(root):
    max_sum = [float('-inf')]
    def dfs(node):
        if not node: return 0
        left  = max(0, dfs(node.left))
        right = max(0, dfs(node.right))
        max_sum[0] = max(max_sum[0], node.val + left + right)
        return node.val + max(left, right)
    dfs(root)
    return max_sum[0]
```
</details>

<details>
<summary><strong>56. Course Schedule (cycle detection)</strong></summary>

```python

def can_finish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    for a, b in prerequisites:
        graph[b].append(a)
    # 0=unvisited, 1=visiting, 2=done
    state = [0] * numCourses
    def dfs(v):
        if state[v] == 1: return False  # cycle
        if state[v] == 2: return True
        state[v] = 1
        if not all(dfs(u) for u in graph[v]): return False
        state[v] = 2
        return True
    return all(dfs(i) for i in range(numCourses))
```
</details>

<details>
<summary><strong>57. Pacific Atlantic Water Flow</strong></summary>

```python

def pacific_atlantic(heights):
    rows, cols = len(heights), len(heights[0])
    def bfs(starts):
        visited = set(starts)
        queue = deque(starts)
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if (0<=nr<rows and 0<=nc<cols and
                    (nr,nc) not in visited and
                    heights[nr][nc] >= heights[r][c]):
                    visited.add((nr,nc))
                    queue.append((nr,nc))
        return visited
    pac = bfs([(0,c) for c in range(cols)] + [(r,0) for r in range(rows)])
    atl = bfs([(rows-1,c) for c in range(cols)] + [(r,cols-1) for r in range(rows)])
    return list(pac & atl)
```
</details>

<details>
<summary><strong>58. Alien Dictionary (topological sort on chars)</strong></summary>

```python

from collections import defaultdict, deque
def alien_order(words):
    graph = defaultdict(set)
    in_degree = {c: 0 for w in words for c in w}
    for i in range(len(words)-1):
        w1, w2 = words[i], words[i+1]
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break
    queue = deque(c for c in in_degree if in_degree[c] == 0)
    result = []
    while queue:
        c = queue.popleft()
        result.append(c)
        for nb in graph[c]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)
    return "".join(result) if len(result) == len(in_degree) else ""
```
</details>

<details>
<summary><strong>59. Wildcard Matching (DP)</strong></summary>

```python

def is_match(s, p):
    m, n = len(s), len(p)
    dp = [[False]*(n+1) for _ in range(m+1)]
    dp[0][0] = True
    for j in range(1, n+1):
        if p[j-1] == '*': dp[0][j] = dp[0][j-1]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if p[j-1] == '*':
                dp[i][j] = dp[i-1][j] or dp[i][j-1]
            elif p[j-1] == '?' or s[i-1] == p[j-1]:
                dp[i][j] = dp[i-1][j-1]
    return dp[m][n]
```
</details>

<details>
<summary><strong>60. Regular Expression Matching</strong></summary>

```python

def regex_match(s, p):
    m, n = len(s), len(p)
    dp = [[False]*(n+1) for _ in range(m+1)]
    dp[0][0] = True
    for j in range(2, n+1):
        if p[j-1] == '*': dp[0][j] = dp[0][j-2]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-2]  # zero occurrences
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] |= dp[i-1][j]
            elif p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
    return dp[m][n]
```
</details>

<details>
<summary><strong>61. Burst Balloons (interval DP)</strong></summary>

```python

def max_coins(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0]*n for _ in range(n)]
    for length in range(2, n):
        for left in range(0, n-length):
            right = left + length
            for k in range(left+1, right):
                dp[left][right] = max(dp[left][right],
                    nums[left]*nums[k]*nums[right] + dp[left][k] + dp[k][right])
    return dp[0][n-1]
```
</details>

<details>
<summary><strong>62. Largest Rectangle (stack-based)</strong></summary>

See Q33 (largest_rectangle function).
</details>

<details>
<summary><strong>63. Basic Calculator II</strong></summary>

```python

def calculate(s):
    stack, num, sign = [], 0, '+'
    for i, ch in enumerate(s):
        if ch.isdigit(): num = num * 10 + int(ch)
        if (not ch.isdigit() and ch != ' ') or i == len(s)-1:
            if   sign == '+': stack.append(num)
            elif sign == '-': stack.append(-num)
            elif sign == '*': stack.append(stack.pop() * num)
            elif sign == '/': stack.append(int(stack.pop() / num))
            sign, num = ch, 0
    return sum(stack)
```
</details>

<details>
<summary><strong>64. Task Scheduler (greedy)</strong></summary>

```python

def least_interval(tasks, n):
    from collections import Counter
    counts = Counter(tasks)
    max_count = max(counts.values())
    max_count_tasks = sum(1 for c in counts.values() if c == max_count)
    return max(len(tasks), (max_count - 1) * (n + 1) + max_count_tasks)
```
</details>

<details>
<summary><strong>65. Reconstruct Itinerary (Euler path)</strong></summary>

```python

def find_itinerary(tickets):
    from collections import defaultdict
    graph = defaultdict(list)
    for src, dst in sorted(tickets, reverse=True):
        graph[src].append(dst)
    result = []
    def dfs(airport):
        while graph[airport]:
            dfs(graph[airport].pop())
        result.append(airport)
    dfs("JFK")
    return result[::-1]
```
</details>

<details>
<summary><strong>66. Best Time to Buy/Sell Stock with Cooldown (DP)</strong></summary>

```python

def max_profit_cooldown(prices):
    held = float('-inf')
    sold = cooldown = 0
    for p in prices:
        held, sold, cooldown = max(held, cooldown - p), held + p, max(cooldown, sold)
    return max(sold, cooldown)
```
</details>

<details>
<summary><strong>67. Word Break (DP + Trie)</strong></summary>

```python

def word_break(s, wordDict):
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n+1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[n]

print(word_break("leetcode", ["leet","code"]))  # True
```
</details>

<details>
<summary><strong>68. Find Duplicate in Array (Floyd's cycle)</strong></summary>

```python

def find_duplicate(nums):
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast: break
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow
```
</details>

<details>
<summary><strong>69. Maximum Product Subarray</strong></summary>

```python

def max_product(nums):
    max_p = min_p = result = nums[0]
    for n in nums[1:]:
        candidates = (n, max_p*n, min_p*n)
        max_p, min_p = max(candidates), min(candidates)
        result = max(result, max_p)
    return result
```
</details>

<details>
<summary><strong>70. Rotate Image (in-place)</strong></summary>

```python

def rotate(matrix):
    n = len(matrix)
    # Transpose
    for i in range(n):
        for j in range(i+1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Reverse each row
    for row in matrix:
        row.reverse()
```
</details>

<details>
<summary><strong>71. Set Matrix Zeroes</strong></summary>

```python

def set_zeroes(matrix):
    rows, cols = len(matrix), len(matrix[0])
    zero_rows = {r for r in range(rows) if 0 in matrix[r]}
    zero_cols = {c for c in range(cols) if any(matrix[r][c]==0 for r in range(rows))}
    for r in range(rows):
        for c in range(cols):
            if r in zero_rows or c in zero_cols:
                matrix[r][c] = 0
```
</details>

<details>
<summary><strong>72. Time-based Key-Value Store</strong></summary>

```python

from collections import defaultdict
import bisect
class TimeMap:
    def __init__(self):
        self.store = defaultdict(list)
    
    def set(self, key, value, timestamp):
        self.store[key].append((timestamp, value))
    
    def get(self, key, timestamp):
        entries = self.store[key]
        idx = bisect.bisect_right(entries, (timestamp, chr(127))) - 1
        return entries[idx][1] if idx >= 0 else ""
```
</details>

<details>
<summary><strong>73. Check if Graph is Bipartite (2-coloring)</strong></summary>

```python

def is_bipartite(graph):
    color = {}
    for start in range(len(graph)):
        if start in color: continue
        queue = deque([start])
        color[start] = 0
        while queue:
            node = queue.popleft()
            for nb in graph[node]:
                if nb in color:
                    if color[nb] == color[node]: return False
                else:
                    color[nb] = 1 - color[node]
                    queue.append(nb)
    return True
```
</details>

<details>
<summary><strong>74. Maximum Depth of N-ary Tree</strong></summary>

```python

def max_depth_nary(root):
    if not root: return 0
    if not root.children: return 1
    return 1 + max(max_depth_nary(c) for c in root.children)
```
</details>

<details>
<summary><strong>75. Number of Ways to Decode (with * wildcard)</strong></summary>

```python

def num_decodings_wild(s):
    MOD = 10**9 + 7
    dp = [0] * (len(s)+1)
    dp[0] = 1
    dp[1] = 9 if s[0]=='*' else (0 if s[0]=='0' else 1)
    for i in range(2, len(s)+1):
        c, p = s[i-1], s[i-2]
        if c == '*':  dp[i] = 9 * dp[i-1]
        elif c != '0': dp[i] = dp[i-1]
        # two-digit combinations
        if p == '*':
            dp[i] = (dp[i] + (15 if c=='*' else (1 if c<='6' else 1) if '1'<=c<='9' else 0) * dp[i-2]) % MOD
        elif p == '1': dp[i] = (dp[i] + (9 if c=='*' else 1) * dp[i-2]) % MOD
        elif p == '2': dp[i] = (dp[i] + (6 if c=='*' else 1 if c<='6' else 0) * dp[i-2]) % MOD
    return dp[len(s)]
```
</details>

<details>
<summary><strong>76. Minimum Spanning Tree — When to use Kruskal vs Prim</strong></summary>

- Kruskal: sparse graphs (sort edges + DSU)

- Prim: dense graphs (priority queue from a vertex)
- Both yield O(E log E) or O(E log V) complexity
</details>

<details>
<summary><strong>77. Counting Inversions (merge sort)</strong></summary>

```python

def count_inversions(arr):
    if len(arr) <= 1: return arr, 0
    mid = len(arr) // 2
    left, l_inv = count_inversions(arr[:mid])
    right, r_inv = count_inversions(arr[mid:])
    merged, split_inv = [], 0
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
            split_inv += len(left) - i
    merged += left[i:] + right[j:]
    return merged, l_inv + r_inv + split_inv
```
</details>

<details>
<summary><strong>78. Sqrt(x) using binary search</strong></summary>

```python

def my_sqrt(x):
    lo, hi = 0, x
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid*mid <= x < (mid+1)*(mid+1): return mid
        elif mid*mid > x: hi = mid - 1
        else: lo = mid + 1
    return 0
```
</details>

<details>
<summary><strong>79. Random Pick with Weight</strong></summary>

```python

import random, bisect
class WeightedRandom:
    def __init__(self, w):
        self.prefix = []
        total = 0
        for wi in w:
            total += wi
            self.prefix.append(total)
        self.total = total
    
    def pick_index(self):
        target = random.uniform(0, self.total)
        return bisect.bisect_left(self.prefix, target)
```
</details>

<details>
<summary><strong>80. Reverse Nodes in K-Group</strong></summary>

```python

def reverse_k_group(head, k):
    dummy = ListNode(0)
    dummy.next = head
    prev_group = dummy
    while True:
        kth = prev_group
        for _ in range(k):
            kth = kth.next
            if not kth: return dummy.next
        group_next = kth.next
        prev, cur = group_next, prev_group.next
        while cur != group_next:
            nxt = cur.next
            cur.next = prev
            prev = cur; cur = nxt
        tmp = prev_group.next
        prev_group.next = kth
        prev_group = tmp
    return dummy.next
```
</details>

<details>
<summary><strong>81. Path Sum III (prefix sum in tree)</strong></summary>

```python

def path_sum_iii(root, target):
    prefix = {0: 1}
    count = [0]
    def dfs(node, curr):
        if not node: return
        curr += node.val
        count[0] += prefix.get(curr - target, 0)
        prefix[curr] = prefix.get(curr, 0) + 1
        dfs(node.left, curr)
        dfs(node.right, curr)
        prefix[curr] -= 1
    dfs(root, 0)
    return count[0]
```
</details>

<details>
<summary><strong>82. Maximum Subarray Sum (Kadane's)</strong></summary>

```python

def max_subarray(nums):
    max_sum = cur_sum = nums[0]
    for n in nums[1:]:
        cur_sum = max(n, cur_sum + n)
        max_sum = max(max_sum, cur_sum)
    return max_sum
```
</details>

<details>
<summary><strong>83. Matrix Chain Multiplication (interval DP)</strong></summary>

```python

def matrix_chain(dims):
    n = len(dims) - 1
    dp = [[0]*n for _ in range(n)]
    for length in range(2, n+1):
        for i in range(n-length+1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
                dp[i][j] = min(dp[i][j], cost)
    return dp[0][n-1]
```
</details>

<details>
<summary><strong>84. Shortest Bridge (BFS+DFS)</strong></summary>

```python

def shortest_bridge(grid):
    n = len(grid)
    visited = set()
    queue = deque()
    found = False
    def dfs(r, c):
        if r<0 or r>=n or c<0 or c>=n or grid[r][c]==0 or (r,c) in visited: return
        visited.add((r,c)); queue.append((r,c,0)); dfs(r+1,c); dfs(r-1,c); dfs(r,c+1); dfs(r,c-1)
    for r in range(n):
        if found: break
        for c in range(n):
            if grid[r][c]==1: dfs(r,c); found=True; break
    while queue:
        r, c, dist = queue.popleft()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<n and 0<=nc<n and (nr,nc) not in visited:
                if grid[nr][nc]==1: return dist
                visited.add((nr,nc)); queue.append((nr,nc,dist+1))
```
</details>

<details>
<summary><strong>85. Count Smaller Numbers After Self (merge sort / BIT)</strong></summary>

```python

def count_smaller(nums):
    result = [0] * len(nums)
    def merge_sort(enum):
        half = len(enum) // 2
        if half:
            left = merge_sort(enum[:half])
            right = merge_sort(enum[half:])
            m = k = 0
            merged = []
            while m < len(left) and k < len(right):
                if left[m][1] > right[k][1]:
                    result[left[m][0]] += len(right) - k
                    merged.append(left[m]); m += 1
                else:
                    merged.append(right[k]); k += 1
            return merged + left[m:] + right[k:]
        return enum
    merge_sort(list(enumerate(nums)))
    return result
```
</details>

<details>
<summary><strong>86. Detect and Remove Cycle in Linked List</strong></summary>

```python

def remove_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next; fast = fast.next.next
        if slow is fast: break
    else: return head
    slow = head
    while slow.next is not fast.next:
        slow = slow.next; fast = fast.next
    fast.next = None
    return head
```
</details>

<details>
<summary><strong>87. Construct Binary Tree from Preorder and Inorder</strong></summary>

```python

def build_tree(preorder, inorder):
    if not preorder: return None
    root = TreeNode(preorder[0])
    idx = inorder.index(preorder[0])
    root.left = build_tree(preorder[1:idx+1], inorder[:idx])
    root.right = build_tree(preorder[idx+1:], inorder[idx+1:])
    return root
```
</details>

<details>
<summary><strong>88. Minimum Window to Sort Array</strong></summary>

```python

def find_unsorted_subarray(nums):
    n = len(nums)
    lo, hi = -1, -2
    min_val, max_val = nums[-1], nums[0]
    for i in range(1, n):
        max_val = max(max_val, nums[i])
        if nums[i] < max_val: hi = i
    for i in range(n-2, -1, -1):
        min_val = min(min_val, nums[i])
        if nums[i] > min_val: lo = i
    return hi - lo + 1
```
</details>

<details>
<summary><strong>89. All Nodes Distance K in Binary Tree</strong></summary>

```python

def distance_k(root, target, k):
    graph = defaultdict(list)
    def build(node, parent):
        if not node: return
        if parent: graph[node.val].append(parent.val); graph[parent.val].append(node.val)
        build(node.left, node); build(node.right, node)
    build(root, None)
    visited = {target.val}
    queue = deque([(target.val, 0)])
    result = []
    while queue:
        node, dist = queue.popleft()
        if dist == k: result.append(node)
        elif dist < k:
            for nb in graph[node]:
                if nb not in visited:
                    visited.add(nb); queue.append((nb, dist+1))
    return result
```
</details>

<details>
<summary><strong>90. Maximal Square of 1s in Matrix (DP)</strong></summary>

```python

def maximal_square(matrix):
    rows, cols = len(matrix), len(matrix[0])
    dp = [[0]*cols for _ in range(rows)]
    max_side = 0
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == '1':
                dp[r][c] = 1 + min(
                    dp[r-1][c] if r>0 else 0,
                    dp[r][c-1] if c>0 else 0,
                    dp[r-1][c-1] if r>0 and c>0 else 0)
                max_side = max(max_side, dp[r][c])
    return max_side * max_side
```
</details>

<details>
<summary><strong>91. Stone Game (DP / Math)</strong></summary>

```python

def stone_game(piles):
    # First player always wins (math proof), but DP:
    n = len(piles)
    dp = [list(piles) for _ in range(n)]
    for length in range(2, n+1):
        for i in range(n-length+1):
            j = i + length - 1
            dp[i][j] = max(piles[i] - dp[i+1][j], piles[j] - dp[i][j-1])
    return dp[0][n-1] > 0
```
</details>

<details>
<summary><strong>92. Count of Range Sum (merge sort)</strong></summary>

```python

def count_range_sum(nums, lower, upper):
    prefix = [0]
    for n in nums: prefix.append(prefix[-1] + n)
    count = [0]
    def merge_sort(arr):
        if len(arr) <= 1: return arr
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        j = k = 0
        for l in left:
            while j < len(right) and right[j] - l < lower: j += 1
            while k < len(right) and right[k] - l <= upper: k += 1
            count[0] += k - j
        return sorted(left + right)
    merge_sort(prefix)
    return count[0]
```
</details>

<details>
<summary><strong>93. Implement Trie with Wild Card Search</strong></summary>

```python

class WordDictionary:
    def __init__(self): self.root = {}
    
    def addWord(self, word):
        node = self.root
        for c in word:
            node = node.setdefault(c, {})
        node['#'] = True
    
    def search(self, word):
        def dfs(node, i):
            if i == len(word): return '#' in node
            c = word[i]
            if c == '.':
                return any(dfs(node[k], i+1) for k in node if k != '#')
            return c in node and dfs(node[c], i+1)
        return dfs(self.root, 0)
```
</details>

<details>
<summary><strong>94. Flatten Binary Tree to Linked List</strong></summary>

```python

def flatten(root):
    def dfs(node):
        if not node: return None
        if not node.left and not node.right: return node
        left_tail = dfs(node.left)
        right_tail = dfs(node.right)
        if left_tail:
            left_tail.right = node.right
            node.right = node.left
            node.left = None
        return right_tail or left_tail
    dfs(root)
```
</details>

<details>
<summary><strong>95. Design HashMap (from scratch)</strong></summary>

```python

class MyHashMap:
    def __init__(self):
        self.size = 1000
        self.buckets = [[] for _ in range(self.size)]
    def _hash(self, key): return key % self.size
    def put(self, key, value):
        h = self._hash(key)
        for i, (k,v) in enumerate(self.buckets[h]):
            if k == key: self.buckets[h][i] = (key, value); return
        self.buckets[h].append((key, value))
    def get(self, key):
        for k, v in self.buckets[self._hash(key)]:
            if k == key: return v
        return -1
    def remove(self, key):
        h = self._hash(key)
        self.buckets[h] = [(k,v) for k,v in self.buckets[h] if k != key]
```
</details>

<details>
<summary><strong>96. Count Primes (Sieve of Eratosthenes)</strong></summary>

```python

def count_primes(n):
    if n < 2: return 0
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5)+1):
        if is_prime[i]:
            for j in range(i*i, n, i):
                is_prime[j] = False
    return sum(is_prime)

print(count_primes(10))  # 4 (2,3,5,7)
```
</details>

<details>
<summary><strong>97. Minimum Cost to Connect Sticks (greedy heap)</strong></summary>

```python

def connect_sticks(sticks):
    heapq.heapify(sticks)
    cost = 0
    while len(sticks) > 1:
        a, b = heapq.heappop(sticks), heapq.heappop(sticks)
        cost += a + b
        heapq.heappush(sticks, a + b)
    return cost
```
</details>

<details>
<summary><strong>98. Preimage Size of Factorial Zeroes (binary search)</strong></summary>

```python

def preim_factorial_zeroes(k):
    def trailing_zeroes(n):
        count = 0
        while n >= 5:
            n //= 5; count += n
        return count
    def first_ge(k):
        lo, hi = 0, 5 * (k + 1)
        while lo < hi:
            mid = (lo + hi) // 2
            if trailing_zeroes(mid) >= k: hi = mid
            else: lo = mid + 1
        return lo
    lo = first_ge(k)
    hi = first_ge(k + 1)
    return hi - lo
```
</details>

<details>
<summary><strong>99. Max Points on a Line</strong></summary>

```python

from math import gcd
def max_points(points):
    n = len(points)
    if n <= 2: return n
    result = 2
    for i in range(n):
        slopes = defaultdict(int)
        for j in range(i+1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            g = gcd(abs(dx), abs(dy))
            key = (dx//g, dy//g)
            slopes[key] += 1
            result = max(result, slopes[key] + 1)
    return result
```
</details>

<details>
<summary><strong>100. Design In-Memory File System</strong></summary>

```python

class FileSystem:
    def __init__(self):
        self.root = {'__files__': {}}
    
    def _navigate(self, path):
        parts = [p for p in path.split('/') if p]
        node = self.root
        for part in parts:
            if part not in node:
                node[part] = {'__files__': {}}
            node = node[part]
        return node
    
    def ls(self, path):
        node = self._navigate(path)
        dirs = [k for k in node if k != '__files__']
        files = list(node['__files__'].keys())
        return sorted(dirs + files)
    
    def mkdir(self, path):
        self._navigate(path)
    
    def addContentToFile(self, path, content):
        parts = path.rsplit('/', 1)
        dir_path, filename = parts[0] or '/', parts[1]
        node = self._navigate(dir_path)
        node['__files__'][filename] = node['__files__'].get(filename, '') + content
    
    def readContentFromFile(self, path):
        parts = path.rsplit('/', 1)
        dir_path, filename = parts[0] or '/', parts[1]
        return self._navigate(dir_path)['__files__'][filename]
```
</details>
---

## Complexity Reference

| Algorithm | Time | Space |
|-----------|------|-------|
| Binary Search | O(log n) | O(1) |
| Merge Sort | O(n log n) | O(n) |
| Quick Sort | O(n log n) avg | O(log n) |
| Heap operations | O(log n) | O(1) |
| BFS/DFS | O(V+E) | O(V) |
| Dijkstra | O((V+E)log V) | O(V) |
| Floyd-Warshall | O(V³) | O(V²) |
| Union-Find (amortized) | O(α(n)) | O(n) |
| Trie insert/search | O(L) | O(alphabet × L) |
| Segment Tree query | O(log n) | O(n) |

## Summary in simple points

- Choose structures from required operations and explain time/space complexity, including worst-case or amortised assumptions.
- Arrays, linked lists, stacks, queues, hash maps, heaps and trees trade access, insertion, ordering and memory differently.
- Implement lists and pointer transformations with empty/single-node tests. Floyd's algorithm detects cycles; reversal and grouping require careful boundary links.
- Stacks support parsing, minimum tracking and monotonic boundaries; queues/deques support BFS and sliding windows.
- Binary search requires sorted data or a monotonic predicate. Specify inclusive/exclusive bounds and test absent targets.
- Sorting and merge-based counting support inversion, interval, range-sum and order-statistic problems.
- Heaps support top-k, merging sorted streams, streaming medians and greedy merge costs; preserve tie and duplicate rules.
- Trees need traversal, subtree and path invariants. Validate BST bounds, reconstruct/serialise trees and reason about LCA and distance paths.
- BFS finds shortest unweighted paths; DFS explores connectivity and search state. Clone graphs using identity-aware visited maps.
- Topological ordering requires a DAG. Cycle detection, bipartite colouring, union-find and Euler paths answer different graph questions.
- Dijkstra needs nonnegative weights; Bellman–Ford handles negative edges and detects relevant negative cycles. Floyd–Warshall computes all-pairs distances.
- Prim and Kruskal construct minimum spanning trees under graph assumptions; they do not solve shortest-path problems.
- Tries represent prefixes, while hash maps represent direct keys. Segment/Fenwick trees support structured range updates or queries.
- Two pointers and sliding windows require a maintained invariant. Monotonic stacks solve boundary problems such as histogram rectangles.
- Dynamic programming specifies state, recurrence, base cases and evaluation order. Distinguish sequence, grid, knapsack, interval and game states.
- Backtracking explores choices with restoration and pruning; permutations, subsets, word search and wildcard search need explicit duplicate handling.
- Greedy algorithms need an exchange or ordering argument. A plausible local choice is not proof of global optimality.
- Bit operations, prefix sums, sieves, weighted sampling and numeric search each depend on representation and boundary assumptions.
- Grid flood fills, image transforms, water flow and bridge problems need visited-state and coordinate conventions.
- Caches, time-based maps, file systems and custom hash maps combine structures with API contracts; test updates, missing keys and capacity limits.
