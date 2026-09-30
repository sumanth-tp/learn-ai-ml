---
id: py-types
title: "Types and Data Structures"
sidebar_label: "Types & data structures"
sidebar_position: 1
slug: /code/python/types-and-data-structures
description: "Numbers, strings, lists, tuples, dicts and sets — what each one costs, and when to reach for which."
tags: [python, data-structures, list, dict, set, tuple, mutability]
---

**In one line.** Five containers cover almost everything: list for order, dict for lookup, set for membership, tuple for fixed records, and str for text.

## Start here: values, expressions and containers

> **Video connection:** [numbers and strings, around 1:10](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=4212s), then [containers, around 1:48](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=6480s). These introductory examples prepare you for the deeper material below.

### Numbers, conversion and arithmetic

| Type | Example | Meaning |
| --- | --- | --- |
| `int` | `12` | Whole number |
| `float` | `12.5` | Floating-point number, often an approximation |
| `str` | `"12"` | Text, even when it contains digits |
| `bool` | `True`, `False` | Boolean value; capitals matter |
| `NoneType` | `None` | Absence of a value |

```python
pages = 7
print(pages + 2)   # 9
print(pages - 2)   # 5
print(pages * 2)   # 14
print(pages / 2)   # 3.5: division
print(pages // 2)  # 3: floor division
print(pages % 2)   # 1: remainder
print(pages ** 2)  # 49: exponentiation
print(2 + 3 * 4)   # 14
print((2 + 3) * 4) # 20
pages += 1
print(pages)      # 8

text = "12"
print(int(text) + 3)  # 15
print(float("2.5"))   # 2.5
print(str(12))        # text suitable for string operations
print(type(text))     # <class 'str'>
```

Floor division rounds down: `-7 // 2` is `-4`. `int("twelve")` raises `ValueError`; conversion can fail. `input()` returns a string, so convert numeric input before arithmetic. Floats cannot represent every decimal exactly; use `math.isclose` for approximate comparisons and consider integer minor units or `Decimal` for monetary calculations.

### Text manipulation and f-strings

```python
raw_name = "  Maya Rao  "
name = raw_name.strip()
print(name.lower())          # maya rao
print(name.upper())          # MAYA RAO
print(name.replace("Rao", "Shah"))  # Maya Shah
print(name.startswith("Maya"))     # True
print(name.endswith("Rao"))        # True
print(name.find("Rao"))      # 5; -1 means not found
print(name.count("a"))       # 3
print(name.split())          # ['Maya', 'Rao']
print(" / ".join(name.split()))  # Maya / Rao
print(name[0], name[-1])     # M o
print(name[:4])              # Maya: stop index is excluded
print(len(name))             # 8, including the space
print("-" * 8)               # --------

score = 0.875
message = f"{name}: {score:.1%}"
print(message)              # Maya Rao: 87.5%
print(raw_name)             # still contains the original spaces
```

String methods return new strings; assign the result when you need to retain it. Without the leading `f`, braces and names inside a string are printed literally. F-strings are useful for readable output and building prompts, but do not make interpolated text trustworthy or validate it.

### Booleans and comparisons

```python
score = 82
submitted = True
blocked = False

passed = score >= 60 and submitted and not blocked
print(passed)               # True
print(score == 82)          # True
print(score != 82)          # False
print(60 <= score < 90)     # True
print(bool("False"))        # True: it is a non-empty string
```

`=` assigns, `==` compares, and `!=` means unequal. `and` needs both conditions, `or` needs at least one, and `not` reverses truthiness. With arbitrary objects, `and` and `or` return an operand; they do not always return a `bool`. Use `is None` to check for a missing value. Do not confuse a zero score with a missing score.

### Four containers you can create and change

```python
# List: ordered items, accessed by zero-based position.
names = ["Maya", "Noor", "Leo"]
print(names[0], names[-1])   # Maya Leo
print(names[1:3])           # ['Noor', 'Leo']
names[0] = "Asha"
names.append("Maya")
names.insert(1, "Kai")
names.remove("Noor")
last = names.pop()
print(names, last)          # ['Asha', 'Kai', 'Leo'] Maya

# Dict: associated values, accessed by key.
person = {"name": "Maya", "age": 25}
person["age"] = 26
person["city"] = "Pune"
print(person["name"])               # Maya
print(person.get("language", "en")) # en
del person["city"]
print(list(person.items()))         # [('name', 'Maya'), ('age', 26)]

# Tuple: a fixed sequence. The comma makes a one-item tuple.
point = (3, 5)
one_item = (3,)
x, y = point
print(x, y, len(one_item))  # 3 5 1

# Set: unique hashable values, with no positional indexing.
tags = set(["python", "ai", "python"])
tags.add("data")
tags.discard("missing")
print(sorted(tags))         # ['ai', 'data', 'python']
print("python" in tags)     # True
empty_set = set()           # {} creates an empty dictionary
```

`names[99]` raises `IndexError`; `person["missing"]` raises `KeyError`. `list.remove` removes the first matching value and fails if none exists; `pop` removes and returns an item. In-place methods such as `append`, `sort` and `reverse` return `None`. Use `sorted(names)` when you want a separate sorted list.

:::note Precision about hashability

The shorthand “immutable means hashable” in the original discussion below needs a qualification. A tuple containing a list is immutable as a container but **unhashable**. Dictionary keys and set elements must be hashable, with stable hashes and compatible equality. `hash((1, 2))` works; `hash((1, []))` raises `TypeError`. See the [object model](../02-programs/05-object-model-and-copies.md) for the full contract.

:::

**Added practice:** create a list of three dictionaries containing `name` and `score`. Retrieve the second person's score, append a fourth person, and calculate the unique scores with a set. Explain why the set cannot be indexed by position.

## The idea in plain words

Python gives you a small set of built-in types, and picking the right one is most of the performance work you will ever do at this level.

The distinction that causes the most bugs is **mutable versus immutable**. Lists, dicts and sets can be changed in place; strings, tuples, ints and frozensets cannot. That single fact explains default-argument bugs, surprise aliasing, and why only immutable things can be dictionary keys.

The second thing to internalise is **cost**:

- `list` — ordered, indexable. Append is O(1); searching or inserting at the front is O(n).
- `dict` — hash lookup. Insert, lookup and delete are O(1) on average, and it keeps insertion order.
- `set` — a dict without values. Membership tests are O(1), which is why `x in big_set` beats `x in big_list` by orders of magnitude.
- `tuple` — an immutable list. Slightly smaller, hashable, and signals "this is a fixed record".
- `str` — immutable sequence of characters; every "modification" builds a new string.

```mermaid
flowchart TD
    Q{"What do you need?"} --> ORD["Order matters,<br/>contents change"] --> L["list"]
    Q --> LOOK["Look up by key"] --> D["dict"]
    Q --> MEM["Only 'is it present?'"] --> S["set"]
    Q --> FIX["Fixed record,<br/>hashable"] --> T["tuple"]
    Q --> TXT["Text"] --> STR["str"]
    L -. "append O(1) · search O(n)" .-> COST["Cost drives the choice"]
    D -. "lookup O(1)" .-> COST
    S -. "membership O(1)" .-> COST
```

## How it works

### Mutability, aliasing and the classic trap

Two names can point at the same object. Mutating through one is visible through the other:

```python
a = [1, 2, 3]
b = a          # same object, not a copy
b.append(4)
print(a)       # [1, 2, 3, 4]
```

Use `list(a)`, `a.copy()` or `copy.deepcopy(a)` when you want independence. The most famous version of this bug is a mutable default argument:

```python
def add_item(item, basket=[]):     # WRONG: the list is created once, at def time
    basket.append(item)
    return basket
```

Use `basket=None` and create the list inside the function.

### Strings are immutable, so build them properly

Every `+=` on a string in a loop allocates a new string. For anything longer than a few items, collect into a list and `"".join(...)`, or use an f-string.

f-strings are the default formatting tool: `f"{name:>10} {value:.2%}"` handles alignment, precision and percent formatting in one place. The `=` specifier (`f"{x=}"`) prints both the expression and its value, which is excellent for debugging.

### Dict and set are the performance levers

Turning an `in list` check into an `in set` check is the single most common real speedup in ordinary Python code — O(n) becomes O(1).

Modern dict idioms worth knowing: `dict.get(key, default)`, `dict.setdefault`, `collections.defaultdict`, `collections.Counter`, dict comprehensions, and the merge operators `{**a, **b}` or `a | b`.

## A real system that works this way

**Deduplicating an event stream.** A naive implementation keeps a list of seen ids and does `if event_id not in seen` — which is O(n) per event and quadratic overall. With a `set` it is O(1) per event. On a million events that is the difference between minutes and a fraction of a second, and it is the same three lines of code.

**Config objects** are usually `dict` in transit and a frozen `dataclass` at rest: mutable while you are assembling it, immutable and hashable once it is agreed.

## Code you can run

Measure the difference yourself rather than taking it on faith.

```python
import random, timeit
from collections import Counter

random.seed(0)
ids = [random.randrange(200_000) for _ in range(20_000)]
haystack_list = list(range(200_000))
haystack_set = set(haystack_list)

def with_list():
    return sum(1 for i in ids[:2_000] if i in haystack_list)

def with_set():
    return sum(1 for i in ids if i in haystack_set)

t_list = timeit.timeit(with_list, number=1)
t_set = timeit.timeit(with_set, number=1)
print(f"list membership, 2k lookups : {t_list*1000:8.1f} ms")
print(f"set  membership, 20k lookups: {t_set*1000:8.1f} ms")
print(f"per-lookup speedup ≈ {(t_list/2_000) / (t_set/20_000):,.0f}×\n")

# mutability in one screen
a = [1, 2, 3]
b = a
b.append(4)
print("aliasing:", a)

def bad(item, basket=[]):
    basket.append(item); return basket

def good(item, basket=None):
    basket = [] if basket is None else basket
    basket.append(item); return basket

print("mutable default:", bad("x"), bad("y"))     # ['x'] then ['x', 'y'] — shared!
print("fixed          :", good("x"), good("y"))

# counting without writing a loop
words = "the cat sat on the mat the end".split()
print("counter:", Counter(words).most_common(2))
```

## Designing with it

**Choosing a container**

| Need | Use | Watch out for |
| --- | --- | --- |
| Ordered, growing sequence | `list` | `insert(0, …)` and `in` are O(n) — use `collections.deque` for queues |
| Keyed lookup | `dict` | Keys must be hashable (so immutable) |
| Membership / uniqueness | `set` | Unordered; no indexing |
| Fixed record passed around | `tuple` or `NamedTuple` | Prefer a `dataclass` once it has behaviour |
| Text | `str` | Build with `join`, not `+=` in a loop |

**Rules that prevent most bugs**

- **Never use a mutable default argument.** Use `None` and create inside.
- **Copy at the boundary.** If a function stores a list the caller passed in, copy it, or the caller can mutate your state later.
- **Prefer immutable for anything shared** across threads, cached, or used as a key.
- **Let the structure carry meaning**: a `set` says "uniqueness matters", a `tuple` says "this shape is fixed". That communicates more than a comment.

## Where this stands in 2026

:::info Industry view

- Container choice is the most common source of accidental O(n²) in production Python — the fix is almost always a `set` or a `dict`.
- `collections` (`Counter`, `defaultdict`, `deque`) is standard in professional code; hand-rolled equivalents read as inexperience.
- Immutability is increasingly the default for shared state — frozen dataclasses and tuples avoid a whole class of concurrency bug.
- f-strings are the expected formatting style; `%` and `.format()` survive only in legacy code and logging calls.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why can a tuple be a dictionary key but a list cannot?</summary>

Keys must be hashable, and hashability requires the value never to change — otherwise the hash computed at insert time would stop matching. Tuples are immutable (provided their contents are too), lists are not.

</details>

<details>
<summary><strong>Q2.</strong> What does `a = [1,2]; b = a; b += [3]` leave in `a`, and why?</summary>

`[1, 2, 3]`. `+=` on a list calls `__iadd__`, which extends **in place**, so both names still point at the same mutated list. `b = b + [3]` would instead create a new list and leave `a` unchanged.

</details>

<details>
<summary><strong>Q3.</strong> When is a list comprehension the wrong choice?</summary>

When the result is never used (use a plain loop — a comprehension for side effects is misleading), when it will not fit in memory (use a generator expression), or when the logic needs more than one condition and a transformation and stops being readable.

</details>

## Further reading

- [Python tutorial: data structures](https://docs.python.org/3/tutorial/datastructures.html) — the official walkthrough of the built-in containers.
- [TimeComplexity wiki](https://wiki.python.org/moin/TimeComplexity) — the cost table for every built-in operation. Worth memorising the top half.
- [collections — container datatypes](https://docs.python.org/3/library/collections.html) — `Counter`, `defaultdict`, `deque` and friends.
