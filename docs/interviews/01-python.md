---
title: Python Interview Questions (100)
sidebar_position: 1
sidebar_label: "1 · Python foundations"
---

# Python Interview Questions (100)

:::note Foundation reference
This earlier 100-entry bank remains for prerequisite revision. Its questions were not part of the interview-source audit, and some examples are fragments rather than standalone programmes. After revising these foundations, continue with the [practical topic bank](13-python-data.md) for sourced scenarios, answered follow-ups and tested code. See [tools and versions](23-tools-versions.md) for current behaviour and [coding labs](21-coding-labs.md) for runnable implementations.
:::

## Beginner Questions (1-30)

<details>
<summary><strong>1. What are the key differences between lists and tuples?</strong></summary>

**Answer:**
- **Mutability**: Lists are mutable (modifiable), tuples are immutable (fixed)
- **Performance**: Tuples are faster and use less memory due to immutability
- **Usage**: Use lists for dynamic collections; tuples for fixed data, dictionary keys, or return values
- **Syntax**: Lists use `[]`, tuples use `()`

```python
lst = [1, 2, 3]
lst[0] = 10  # Valid

tpl = (1, 2, 3)
tpl[0] = 10  # TypeError
```

**Interview Tip**: Mention hashability — tuples can be dictionary keys, lists cannot.
</details>

<details>
<summary><strong>2. Explain decorators in Python.</strong></summary>

**Answer:**
Decorators are functions that modify or enhance other functions/classes without permanently changing their source code.

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    return f"Hello {name}"

say_hello("Alice")
# Output:
# Before function call
# Hello Alice
# After function call
```

**Common uses**: logging, timing, authentication, caching (functools.lru_cache)

**Interview Tip**: Write a simple decorator and explain closure concept.
</details>

<details>
<summary><strong>3. What is a lambda function?</strong></summary>

**Answer:**
Lambda is an anonymous function for small, one-time operations. Syntax: `lambda arguments: expression`

```python
# Using lambda
add = lambda x, y: x + y
print(add(2, 3))  # 5

# With map, filter, sorted
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16]
filtered = list(filter(lambda x: x > 2, numbers))  # [3, 4]
sorted_desc = sorted(numbers, key=lambda x: -x)  # [4, 3, 2, 1]
```

**When to use**: Data transformation in map/filter/sorted, simple callbacks

**When NOT to use**: Complex logic, multi-line operations, need for docstrings

**Interview Tip**: Know map/filter/reduce and when to prefer list comprehensions.
</details>

<details>
<summary><strong>4. What are list comprehensions?</strong></summary>

**Answer:**
Concise, Pythonic way to create lists. Faster and more readable than map/filter.

```python
# Basic comprehension
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(10) if x % 2 == 0]

# Nested comprehension
matrix = [[i*j for j in range(3)] for i in range(3)]

# vs traditional loops
result = []
for x in range(10):
    if x % 2 == 0:
        result.append(x**2)

# Comprehension is cleaner and faster
result = [x**2 for x in range(10) if x % 2 == 0]
```

**Advantages**: More readable, faster execution, can include conditions naturally

**Interview Tip**: Show proficiency with basic, conditional, and nested comprehensions.
</details>

<details>
<summary><strong>5. Explain *args and **kwargs.</strong></summary>

**Answer:**
Allow functions to accept variable number of arguments.

```python
def func(*args, **kwargs):
    print(f"args: {args}")      # Tuple of positional args
    print(f"kwargs: {kwargs}")  # Dict of keyword args

func(1, 2, 3, name="Alice", age=25)
# args: (1, 2, 3)
# kwargs: {'name': 'Alice', 'age': 25}

# Unpacking
numbers = [1, 2, 3]
func(*numbers)  # Unpacks as positional args

config = {'name': 'Bob', 'age': 30}
func(**config)  # Unpacks as keyword args
```

**Use cases**: Wrapper functions, decorators, flexible APIs

**Interview Tip**: Explain order: positional args → *args → keyword args → **kwargs.
</details>

<details>
<summary><strong>6. What's the difference between shallow and deep copy?</strong></summary>

**Answer:**
- **Shallow copy**: New object but references same nested objects
- **Deep copy**: New object AND recursively copies all nested objects

```python
import copy

original = [[1, 2], [3, 4]]

shallow = copy.copy(original)
shallow[0][0] = 99
print(original)  # [[99, 2], [3, 4]] - original changed!

deep = copy.deepcopy(original)
deep[0][0] = 99
print(original)  # [[1, 2], [3, 4]] - unchanged
```

**When to use deep copy**: Nested structures (lists of dicts, complex objects)

**Interview Tip**: Demonstrate with nested data structures.
</details>

<details>
<summary><strong>7. Explain the GIL (Global Interpreter Lock).</strong></summary>

**Answer:**
GIL allows only one thread to execute Python bytecode at a time. Affects concurrency.

**Implications**:
- Multi-threading doesn't provide parallelism for CPU-bound tasks
- Works well for I/O-bound tasks (network, files)
- Use multiprocessing for CPU-bound parallel work
- Use async/await for I/O-bound concurrency

```python
import threading

# CPU-bound: Threading is slow due to GIL
def cpu_task():
    total = 0
    for i in range(100000000):
        total += i

# I/O-bound: Threading works fine
import time
def io_task():
    time.sleep(1)  # Simulates I/O
```

**Solutions**: multiprocessing, async, PyPy (no GIL), Cython

**Interview Tip**: Discuss when to use threading vs multiprocessing.
</details>

<details>
<summary><strong>8. What are generators?</strong></summary>

**Answer:**
Generators yield values one at a time instead of storing all in memory. Lazy evaluation.

```python
# Generator function
def make_generator(n):
    for i in range(n):
        yield i

# List comprehension vs generator
lst = [i for i in range(1000000)]  # Uses lots of memory
gen = (i for i in range(1000000))  # Minimal memory

# Consuming generator
for value in gen:
    print(value)
# Can't iterate again - generator is exhausted

# Generator function
def count_up_to(max):
    count = 1
    while count <= max:
        yield count
        count += 1

for num in count_up_to(5):
    print(num)  # 1, 2, 3, 4, 5
```

**Advantages**: Memory efficient, lazy evaluation, can represent infinite sequences

**Interview Tip**: Know generator expressions and yield keyword.
</details>

<details>
<summary><strong>9. What is a context manager?</strong></summary>

**Answer:**
Context managers manage resources (files, connections) with proper setup/teardown using `with` statement.

```python
# Built-in context manager
with open('file.txt') as f:
    data = f.read()
# File automatically closed

# Create custom context manager
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# Using decorator
from contextlib import contextmanager

@contextmanager
def managed_resource(name):
    print(f"Acquiring {name}")
    try:
        yield name
    finally:
        print(f"Releasing {name}")

with managed_resource("database"):
    print("Using resource")
```

**Interview Tip**: Show both `__enter__/__exit__` and @contextmanager approaches.
</details>

<details>
<summary><strong>10. Explain class variables vs instance variables.</strong></summary>

**Answer:**
- **Class variables**: Shared among all instances
- **Instance variables**: Unique to each instance

```python
class Dog:
    species = "Canis familiaris"  # Class variable
    
    def __init__(self, name, age):
        self.name = name      # Instance variable
        self.age = age

dog1 = Dog("Rex", 3)
dog2 = Dog("Max", 5)

print(dog1.species)  # "Canis familiaris"
print(dog2.species)  # "Canis familiaris"
print(dog1.name)     # "Rex"
print(dog2.name)     # "Max"

# Modifying class variable affects all instances
Dog.species = "Modified"
print(dog1.species)  # "Modified"
```

**Interview Tip**: Explain memory implications and when to use each.
</details>

<details>
<summary><strong>11. What is @staticmethod vs @classmethod?</strong></summary>

**Answer:**
- **@staticmethod**: No access to instance or class
- **@classmethod**: Access to class via `cls` parameter

```python
class MyClass:
    counter = 0
    
    @staticmethod
    def static_method(x):
        # Can't access self or cls
        return x * 2
    
    @classmethod
    def from_string(cls, string):
        # Can access class, useful for factories
        return cls(int(string))
    
    @classmethod
    def get_counter(cls):
        return cls.counter

print(MyClass.static_method(5))  # 10
```

**Use cases**:
- @staticmethod: Utility functions
- @classmethod: Factory methods, class-level operations

**Interview Tip**: Show why classmethod is better for inheritance.
</details>

<details>
<summary><strong>12. Explain exception handling.</strong></summary>

**Answer:**
Exception handling manages errors gracefully.

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Caught: {e}")
except (ValueError, TypeError):
    print("Value or Type error")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    print("Cleanup code")

# Custom exceptions
class InsufficientFundsError(Exception):
    def __init__(self, amount, balance):
        super().__init__(f"Need {amount}, have {balance}")

try:
    raise InsufficientFundsError(100, 50)
except InsufficientFundsError as e:
    print(f"Transaction failed: {e}")
```

**Best practices**: Catch specific exceptions, log errors, clean up resources

**Interview Tip**: Show specific exception handling and proper error messages.
</details>

<details>
<summary><strong>13. What are properties?</strong></summary>

**Answer:**
Properties provide computed attributes with getter/setter/deleter.

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value
    
    @property
    def area(self):
        import math
        return math.pi * self._radius ** 2

circle = Circle(5)
print(circle.area)   # 78.53...
circle.radius = 10   # Calls setter
```

**Interview Tip**: Know when to use properties (validation, computed attributes).
</details>

<details>
<summary><strong>14. Explain is vs == operators.</strong></summary>

**Answer:**
- **==**: Compares values (equality)
- **is**: Compares object identity (same memory address)

```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)   # True (same values)
print(a is b)   # False (different objects)
print(a is c)   # True (same object)

# None comparison
value = None
print(value is None)   # Correct way
print(value == None)   # Works but not recommended
```

**Interview Tip**: Always use `is None`, not `== None`.
</details>

<details>
<summary><strong>15. What is MRO (Method Resolution Order)?</strong></summary>

**Answer:**
MRO determines method resolution order in inheritance using C3 linearization.

```python
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")
        super().method()

class C(A):
    def method(self):
        print("C")
        super().method()

class D(B, C):
    def method(self):
        print("D")
        super().method()

d = D()
d.method()
# Output: D → B → C → A

print(D.mro())  # [D, B, C, A, object]
```

**Interview Tip**: Understand diamond inheritance and super() usage.
</details>

<details>
<summary><strong>16. Explain mutable vs immutable objects.</strong></summary>

**Answer:**
- **Immutable**: Cannot change after creation (int, str, tuple, frozenset)
- **Mutable**: Can be modified (list, dict, set)

```python
# Immutable
s = "hello"
s[0] = "H"  # TypeError

# Mutable
lst = [1, 2, 3]
lst[0] = 10  # Works

# Function parameters
def modify(obj):
    obj[0] = 999

lst = [1, 2, 3]
modify(lst)
print(lst)  # [999, 2, 3] - modified!

tpl = (1, 2, 3)
modify(tpl)  # TypeError
```

**Interview Tip**: Discuss implications for function arguments.
</details>

<details>
<summary><strong>17. What is EAFP vs LBYL?</strong></summary>

**Answer:**
- **EAFP**: Easier to Ask for Forgiveness than Permission (use try/except)
- **LBYL**: Look Before You Leap (check conditions first)

Python prefers EAFP:

```python
# LBYL (not Pythonic)
if key in dictionary:
    value = dictionary[key]

# EAFP (Pythonic)
try:
    value = dictionary[key]
except KeyError:
    value = None

# EAFP with iteration
try:
    for item in obj:
        process(item)
except TypeError:
    print("Not iterable")
```

**Interview Tip**: This reflects Python philosophy; use try/except over preconditions.
</details>

<details>
<summary><strong>18. What are type hints?</strong></summary>

**Answer:**
Type hints provide static type information, improving code readability and enabling type checking.

```python
# Function type hints
def add(x: int, y: int) -> int:
    return x + y

# Variable type hints
name: str = "Alice"
age: int = 25
scores: list[int] = [90, 85, 92]

# Complex types
from typing import Optional, Union, List, Dict, Callable

def process(items: List[int]) -> Dict[str, float]:
    return {"sum": sum(items)}

def maybe_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        return None

def handle(value: Union[int, str]) -> None:
    if isinstance(value, int):
        print(value * 2)
    else:
        print(value.upper())
```

**Tools**: Use mypy for static type checking

**Interview Tip**: Know basic typing module, Optional, Union.
</details>

<details>
<summary><strong>19. What are docstrings?</strong></summary>

**Answer:**
Docstrings document functions, classes, modules using triple quotes.

```python
def calculate_mean(numbers):
    """Calculate the arithmetic mean of numbers.
    
    Args:
        numbers: List of numeric values
    
    Returns:
        float: The arithmetic mean
    
    Raises:
        ValueError: If list is empty
    
    Example:
        >>> calculate_mean([1, 2, 3])
        2.0
    """
    if not numbers:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(numbers) / len(numbers)

# Access docstring
print(calculate_mean.__doc__)
help(calculate_mean)
```

**Styles**: Google style, NumPy style, reStructuredText

**Interview Tip**: Show you understand documentation standards.
</details>

<details>
<summary><strong>20. Explain pass, continue, and break.</strong></summary>

**Answer:**
- **pass**: Do nothing, placeholder
- **continue**: Skip to next iteration
- **break**: Exit loop entirely

```python
# pass: placeholder
if True:
    pass

def stub_function():
    pass

class StubClass:
    pass

# continue: skip iteration
for i in range(5):
    if i == 2:
        continue
    print(i)  # 0, 1, 3, 4

# break: exit loop
for i in range(5):
    if i == 3:
        break
    print(i)  # 0, 1, 2

# Real-world usage
while True:
    user_input = input("Enter number (0 to quit): ")
    if user_input == "0":
        break
    try:
        value = int(user_input)
        print(f"Got {value}")
        if value < 0:
            continue
    except ValueError:
        continue
```

**Interview Tip**: Show realistic loop control usage.
</details>

<details>
<summary><strong>21. What's the issue with mutable default arguments?</strong></summary>

**Answer:**
Mutable defaults are created once at definition, not per call. This causes bugs.

```python
# WRONG: Mutable default
def append_to_list(value, target=[]):
    target.append(value)
    return target

result1 = append_to_list(1)
result2 = append_to_list(2)
print(result1)  # [1, 2] - UNEXPECTED!
print(result2)  # [1, 2]

# CORRECT: Use None
def append_to_list_correct(value, target=None):
    if target is None:
        target = []
    target.append(value)
    return target

result1 = append_to_list_correct(1)
result2 = append_to_list_correct(2)
print(result1)  # [1]
print(result2)  # [2]
```

**Interview Tip**: This is a classic Python gotcha; explain the why.
</details>

<details>
<summary><strong>22. What is __name__ == "__main__"?</strong></summary>

**Answer:**
Pattern that allows code to run when executed directly but not when imported.

```python
# my_module.py
def greet(name):
    return f"Hello {name}"

if __name__ == "__main__":
    # This only runs when executed directly
    print(greet("World"))
    # Doesn't run when imported

# Usage:
# python my_module.py → runs if block
# from my_module import greet → skips if block
```

**Benefits**: Reusable modules that also work as scripts

**Interview Tip**: Show in realistic script with both functions and test code.
</details>

<details>
<summary><strong>23. Explain monkey patching.</strong></summary>

**Answer:**
Dynamically modifying code at runtime. Powerful but use cautiously.

```python
# Monkey patching a method
class Dog:
    def bark(self):
        return "Woof!"

Dog.bark = lambda self: "Arf! Arf!"

dog = Dog()
print(dog.bark())  # "Arf! Arf!"

# Patching third-party libraries (dangerous!)
import math
original_pi = math.pi
math.pi = 3.14  # Not recommended!

# Better: create wrapper
def patched_function():
    print("Before")
    original_function()
    print("After")
```

**Use cases**: Testing, hot-patching bugs

**Interview Tip**: Know it exists but emphasize risks; mention mocking as safer alternative.
</details>

<details>
<summary><strong>24. What are __slots__?</strong></summary>

**Answer:**
`__slots__` restricts attributes to fixed set, reducing memory overhead.

```python
# Without slots
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# With slots
class SlottedPoint:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Can't add arbitrary attributes
point = SlottedPoint(1, 2)
point.z = 3  # AttributeError

# Memory savings for many instances
import sys
p1 = Point(1, 2)
p2 = SlottedPoint(1, 2)
# SlottedPoint uses much less memory
```

**Interview Tip**: Mention for performance-critical code with many instances.
</details>

<details>
<summary><strong>25. What is a metaclass?</strong></summary>

**Answer:**
Metaclasses are classes of classes; they define how classes behave.

```python
class CounterMeta(type):
    def __new__(mcs, name, bases, dct):
        dct['count'] = 0
        return super().__new__(mcs, name, bases, dct)
    
    def __call__(cls, *args, **kwargs):
        cls.count += 1
        return super().__call__(*args, **kwargs)

class MyClass(metaclass=CounterMeta):
    pass

obj1 = MyClass()
obj2 = MyClass()
print(MyClass.count)  # 2
```

**Use cases**: ORMs, frameworks, enforcing constraints

**Interview Tip**: This is advanced; mention understanding but focus on practical use.
</details>

<details>
<summary><strong>26. What are iterators?</strong></summary>

**Answer:**
Objects that implement `__iter__()` and `__next__()` methods for iteration.

```python
class CountUp:
    def __init__(self, max):
        self.max = max
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.max:
            self.current += 1
            return self.current
        else:
            raise StopIteration

# Using iterator
for num in CountUp(3):
    print(num)  # 1, 2, 3

# Manual iteration
iterator = iter([1, 2, 3])
print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
# next(iterator)  # StopIteration
```

**Interview Tip**: Know difference between iterator and iterable.
</details>

<details>
<summary><strong>27. What is the difference between range() and xrange()?</strong></summary>

**Answer:**
In Python 3, `xrange` doesn't exist; `range()` behaves like Python 2's `xrange()`.

```python
# Python 3
r = range(10)  # Lazy evaluation, not a list
print(type(r))  # <class 'range'>
print(list(r))  # [0, 1, 2, ..., 9]

# Memory efficient
big_range = range(1000000)  # Minimal memory
big_list = list(range(1000000))  # Uses lots of memory

# Operations
print(5 in range(10))  # True (O(1) check)
print(range(10)[5])    # 5 (O(1) indexing)

# Creating subranges
subset = range(10)[2:8:2]  # range(2, 8, 2)
```

**Interview Tip**: Explain why range() is better for loops than list(range()).
</details>

<details>
<summary><strong>28. What is a descriptor?</strong></summary>

**Answer:**
Objects that customize attribute access by implementing `__get__`, `__set__`, `__delete__`.

```python
class ValidatedString:
    def __init__(self, min_length=0):
        self.min_length = min_length
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get('value', '')
    
    def __set__(self, obj, value):
        if len(value) < self.min_length:
            raise ValueError(f"String too short")
        obj.__dict__['value'] = value

class Person:
    name = ValidatedString(min_length=1)

person = Person()
person.name = "Alice"  # Calls __set__
print(person.name)     # Calls __get__
```

**Use cases**: Properties, lazy loading, validation

**Interview Tip**: Properties are descriptors; know basic protocol.
</details>

<details>
<summary><strong>29. What is the __slots__ dunder attribute used for?</strong></summary>

**Answer:**
`__slots__` defines which attributes an instance can have, saving memory.

```python
class RegularClass:
    def __init__(self, x):
        self.x = x

class SlottedClass:
    __slots__ = ['x']
    def __init__(self, x):
        self.x = x

# SlottedClass can't have arbitrary attributes
obj = SlottedClass(1)
obj.y = 2  # AttributeError: 'SlottedClass' object has no attribute 'y'

# Memory comparison
import sys
r = RegularClass(1)
s = SlottedClass(1)
print(sys.getsizeof(r.__dict__))  # Dict overhead
# SlottedClass uses much less memory
```

**Interview Tip**: Use for performance-critical classes with many instances.
</details>

<details>
<summary><strong>30. What are async and await?</strong></summary>

**Answer:**
`async/await` syntax for writing asynchronous code that's readable and concurrent.

```python
import asyncio

# Define async function
async def fetch_data(url):
    print(f"Fetching {url}")
    await asyncio.sleep(1)  # Simulates I/O wait
    return f"Data from {url}"

# Running async function
async def main():
    result = await fetch_data("http://example.com")
    print(result)

asyncio.run(main())

# Concurrent execution
async def main_concurrent():
    # Both run concurrently, not sequentially
    result1 = asyncio.create_task(fetch_data("url1"))
    result2 = asyncio.create_task(fetch_data("url2"))
    
    data1 = await result1
    data2 = await result2

# Run multiple tasks
asyncio.run(main_concurrent())
```

**Interview Tip**: Know async enables concurrency for I/O-bound tasks, not parallelism.
</details>

---

## Intermediate Questions (31-70)

<details>
<summary><strong>31. What is string formatting with f-strings vs format()?</strong></summary>

**Answer:**
F-strings (Python 3.6+) are faster and more readable than format() method.

```python
name = "Alice"
age = 25
salary = 50000.50

# Old: % formatting
msg = "%s is %d" % (name, age)

# format() method
msg = "{}  is {}".format(name, age)
msg = "{name} is {age}".format(name=name, age=age)

# f-strings (preferred)
msg = f"{name} is {age}"

# With expressions in f-strings
print(f"Next year: {age + 1}")
print(f"Uppercase: {name.upper()}")
print(f"Salary: ${salary:,.2f}")

# Format specifiers
value = 3.14159
print(f"{value:.2f}")      # 3.14 - 2 decimals
print(f"{value:>10}")      # Right align in 10 chars
print(f"{value:<10}")      # Left align
print(f"{value:^10}")      # Center

# Performance (f-strings are fastest)
import timeit
print(timeit.timeit(f"'{name} is {age}'", number=100000))
print(timeit.timeit("'{} is {}'.format('{}', {})".format(name, age), number=100000))
```

**Interview Tip**: Prefer f-strings; mention performance advantage.
</details>

<details>
<summary><strong>32. What are set operations and methods?</strong></summary>

**Answer:**
Sets are unordered collections of unique elements with O(1) average operations.

```python
s1 = {1, 2, 3, 4}
s2 = {3, 4, 5, 6}

# Union: all elements from both
union = s1 | s2  # {1, 2, 3, 4, 5, 6}
union = s1.union(s2)

# Intersection: common elements
inter = s1 & s2  # {3, 4}
inter = s1.intersection(s2)

# Difference: in s1 but not s2
diff = s1 - s2  # {1, 2}
diff = s1.difference(s2)

# Symmetric difference: in either but not both
sym_diff = s1 ^ s2  # {1, 2, 5, 6}

# Subset/superset
print({1, 2}.issubset({1, 2, 3}))       # True
print({1, 2, 3}.issuperset({1, 2}))    # True

# Set methods
s = {1, 2, 3}
s.add(4)                # Add element
s.remove(2)             # Remove (raises KeyError if missing)
s.discard(5)            # Remove (no error if missing)
s.pop()                 # Remove arbitrary element
s.clear()               # Remove all

# Frozenset (immutable)
fs = frozenset({1, 2, 3})
# fs.add(4)  # TypeError
```

**Interview Tip**: Know set operations and time complexity.
</details>

<details>
<summary><strong>33. What are dictionary methods and operations?</strong></summary>

**Answer:**
Dictionaries store key-value pairs with O(1) average lookup.

```python
d = {'a': 1, 'b': 2, 'c': 3}

# Access
print(d['a'])                    # 1
print(d.get('a'))                # 1
print(d.get('z', 'default'))    # 'default'

# Modification
d['d'] = 4                       # Add/update
d.update({'e': 5, 'f': 6})     # Merge dictionaries

# Deletion
del d['a']                       # Delete key (raises KeyError if missing)
value = d.pop('b')              # Remove and return value
d.popitem()                      # Remove arbitrary item

# Iteration
for key in d:                    # Iterate keys
    print(key)

for key, value in d.items():    # Iterate key-value pairs
    print(key, value)

for value in d.values():         # Iterate values
    print(value)

# Dictionary comprehension
squared = {k: v**2 for k, v in {'a': 2, 'b': 3}.items()}

# setdefault
d.setdefault('x', 0)            # Get value or set default if missing

# clear
d.clear()                        # Remove all items
```

**Interview Tip**: Know dictionary operations and when to use methods like setdefault.
</details>

<details>
<summary><strong>34. What is the zip() function?</strong></summary>

**Answer:**
Combines multiple iterables, pairing elements by position.

```python
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
cities = ['NYC', 'LA']

# Zip combines iterables
zipped = list(zip(names, ages, cities))
# [('Alice', 25, 'NYC'), ('Bob', 30, 'LA')]
# Note: Stops at shortest iterable

# Unzip
names, ages = zip(*zipped)

# Iterate over pairs
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# Create dictionary from two lists
d = dict(zip(names, ages))  # {'Alice': 25, 'Bob': 30, 'Charlie': 35}

# With enumerate
for i, (name, age) in enumerate(zip(names, ages)):
    print(f"{i}: {name} is {age}")
```

**Interview Tip**: Know zip's behavior with unequal iterables.
</details>

<details>
<summary><strong>35. What is enumerate()?</strong></summary>

**Answer:**
Iterate with both index and element.

```python
names = ['Alice', 'Bob', 'Charlie']

# Without enumerate
for i in range(len(names)):
    print(i, names[i])

# With enumerate (better)
for i, name in enumerate(names):
    print(i, name)

# With start parameter
for i, name in enumerate(names, start=1):
    print(i, name)  # 1, 2, 3...

# Creating list of tuples
indexed = list(enumerate(names))
# [(0, 'Alice'), (1, 'Bob'), (2, 'Charlie')]

# Nested enumerate
matrix = [[1, 2], [3, 4]]
for i, row in enumerate(matrix):
    for j, val in enumerate(row):
        print(f"[{i}][{j}] = {val}")
```

**Interview Tip**: Prefer enumerate over range(len()).
</details>

<details>
<summary><strong>36. What are any() and all() functions?</strong></summary>

**Answer:**
Check conditions across iterables.

```python
# any() - True if any element is truthy
print(any([False, False, True]))           # True
print(any([0, 0, 1]))                      # True
print(any([]))                             # False

# all() - True if all elements are truthy
print(all([True, True, True]))             # True
print(all([1, 2, 3]))                      # True
print(all([1, 2, 0]))                      # False
print(all([]))                             # True (vacuous truth)

# With generators
numbers = [1, 2, 3, 4, 5]
print(any(x > 4 for x in numbers))        # True
print(all(x > 0 for x in numbers))        # True

# Practical example
def validate_data(data):
    if not data:
        return False
    if not all(isinstance(x, int) for x in data):
        return False
    if not any(x > 100 for x in data):
        return False
    return True
```

**Interview Tip**: Know short-circuit behavior.
</details>

<details>
<summary><strong>37. What is the difference between is and ==?</strong></summary>

**Answer:**
`==` compares values; `is` compares object identity.

```python
# Lists
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)   # True (same values)
print(a is b)   # False (different objects)
print(a is c)   # True (same object)

# Numbers (integer caching)
x = 256
y = 256
print(x is y)   # True (small ints cached)

x = 257
y = 257
print(x is y)   # False (large ints not cached)

# None (always use 'is')
value = None
if value is None:   # Correct
    pass
if value == None:   # Works but not recommended
    pass

# Strings (interning)
s1 = "hello"
s2 = "hello"
print(s1 is s2)     # Likely True (literals interned)

s3 = "hel" + "lo"
print(s3 is s1)     # Might be False (not interned)
```

**Interview Tip**: Always use `is None`, never `== None`.
</details>

<details>
<summary><strong>38. What is type() vs isinstance()?</strong></summary>

**Answer:**
type() returns exact type; isinstance() checks inheritance.

```python
class Animal:
    pass

class Dog(Animal):
    pass

dog = Dog()
animal = Animal()

# type() - exact type
print(type(dog) == Dog)        # True
print(type(dog) == Animal)     # False

# isinstance() - includes inheritance
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True

# With built-ins
print(type(5) == int)          # True
print(isinstance(5, int))      # True

# isinstance is generally preferred
# isinstance handles inheritance correctly
# type() is too strict for most use cases

# isinstance with tuple of types
print(isinstance(5, (int, str, float)))  # True
```

**Interview Tip**: Prefer isinstance() for type checking.
</details>

<details>
<summary><strong>39. What is assert and when to use it?</strong></summary>

**Answer:**
Check conditions during development; raises AssertionError if false.

```python
# Simple assert
x = 10
assert x > 0, "x must be positive"

# In functions
def divide(a, b):
    assert b != 0, "Division by zero"
    return a / b

# Multiple assertions
def process(data):
    assert isinstance(data, list), "data must be list"
    assert len(data) > 0, "data cannot be empty"
    assert all(isinstance(x, int) for x in data), "all elements must be int"
    return sum(data)

# Note: Can be disabled with python -O
# So DON'T use for validation in production

# For validation, use exceptions
def process_data(data):
    if not isinstance(data, list):
        raise TypeError("data must be list")
    if not data:
        raise ValueError("data cannot be empty")

# Assert mainly for debugging and preconditions
def factorial(n):
    assert n >= 0, "n must be non-negative"
    assert isinstance(n, int), "n must be integer"
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

**Interview Tip**: Use assert for debugging, exceptions for validation.
</details>

<details>
<summary><strong>40. What is the iter() and next() functions?</strong></summary>

**Answer:**
Create and consume iterators.

```python
# iter() creates iterator
lst = [1, 2, 3]
iterator = iter(lst)

# next() gets next value
print(next(iterator))   # 1
print(next(iterator))   # 2
print(next(iterator))   # 3
# next(iterator)        # StopIteration

# Useful for reading until condition
def read_until_empty(file):
    for line in iter(file.readline, ''):
        process(line)

# Creating custom iterator
class CountUp:
    def __init__(self, max):
        self.max = max
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.max:
            self.current += 1
            return self.current
        raise StopIteration

for num in CountUp(3):
    print(num)  # 1, 2, 3

# Iterators vs iterables
# Iterable: has __iter__() (lists, strings, tuples)
# Iterator: has __iter__() and __next__()
```

**Interview Tip**: Know difference between iterator and iterable.
</details>

<details>
<summary><strong>41. What is getattr, setattr, and hasattr?</strong></summary>

**Answer:**
Dynamic attribute access.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

person = Person("Alice", 25)

# getattr - get attribute by name
name = getattr(person, 'name')           # "Alice"
email = getattr(person, 'email', 'N/A')  # Default if missing

# setattr - set attribute dynamically
setattr(person, 'age', 26)
setattr(person, 'email', 'alice@example.com')

# hasattr - check if attribute exists
print(hasattr(person, 'name'))    # True
print(hasattr(person, 'email'))   # True
print(hasattr(person, 'phone'))   # False

# Practical example: Dynamic method calling
class Calculator:
    def add(self, a, b):
        return a + b
    def subtract(self, a, b):
        return a - b

calc = Calculator()
operation = 'add'
method = getattr(calc, operation)
print(method(5, 3))  # 8

# Building objects dynamically
def create_object(class_name, **kwargs):
    obj = eval(class_name)()
    for key, value in kwargs.items():
        setattr(obj, key, value)
    return obj
```

**Interview Tip**: Show practical uses for dynamic attribute access.
</details>

<details>
<summary><strong>42. What are *args and **kwargs in more detail?</strong></summary>

**Answer:**
Already covered in Q5, but deeper exploration of advanced patterns.

```python
# Forwarding arguments
def decorator(func):
    def wrapper(*args, **kwargs):
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return wrapper

@decorator
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Multiple unpacking
def func(a, b, c):
    return a + b + c

args1 = (1, 2)
args2 = (3,)
result = func(*args1, *args2)  # Multiple * unpacking

# Keyword-only arguments
def func(a, *, b, c=3):
    return a + b + c

# func(1, 2)           # TypeError - b is keyword-only
func(1, b=2)           # OK
func(1, b=2, c=4)      # OK

# Combining args and keyword-only
def func(a, *args, b, **kwargs):
    print(a, args, b, kwargs)

func(1, 2, 3, b=10, x=20)
# a=1, args=(2,3), b=10, kwargs={'x': 20}
```

**Interview Tip**: Explain advanced patterns with real decorators.
</details>

<details>
<summary><strong>43. What is the difference between list.copy() and list[:]?</strong></summary>

**Answer:**
Both create shallow copies; list[:] is more Pythonic.

```python
lst = [1, 2, 3]

# Shallow copy methods (equivalent)
copy1 = lst.copy()
copy2 = lst[:]
copy3 = list(lst)

# All are shallow copies
lst = [[1, 2], [3, 4]]
shallow = lst[:]
shallow[0][0] = 99
print(lst)  # [[99, 2], [3, 4]] - original changed!

# For deep copy
import copy
deep = copy.deepcopy(lst)
deep[0][0] = 99
print(lst)  # [[1, 2], [3, 4]] - unchanged

# Performance
import timeit
lst = list(range(1000))
print(timeit.timeit(lambda: lst.copy(), number=100000))
print(timeit.timeit(lambda: lst[:], number=100000))
# Similar performance
```

**Interview Tip**: Know difference between shallow and deep copy.
</details>

<details>
<summary><strong>44. What is the difference between map and list comprehension?</strong></summary>

**Answer:**
Both transform data; comprehensions are more Pythonic and faster.

```python
numbers = [1, 2, 3, 4, 5]

# map
mapped = list(map(lambda x: x**2, numbers))

# list comprehension
comprehended = [x**2 for x in numbers]

# Performance
import timeit
data = list(range(1000))

time_map = timeit.timeit(
    lambda: list(map(lambda x: x**2, data)),
    number=10000
)
time_comp = timeit.timeit(
    lambda: [x**2 for x in data],
    number=10000
)

# Comprehensions are ~2x faster

# Readability comparison
# map: Less intuitive for complex operations
result = list(map(lambda x: x**2 if x > 2 else 0, numbers))

# comprehension: More readable
result = [x**2 if x > 2 else 0 for x in numbers]
```

**Interview Tip**: Prefer comprehensions for readability and performance.
</details>

<details>
<summary><strong>45. What is reduce() function?</strong></summary>

**Answer:**
Apply function cumulatively across iterable (from functools).

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# Sum using reduce
total = reduce(lambda x, y: x + y, numbers)  # 15

# Product
product = reduce(lambda x, y: x * y, numbers)  # 120

# With initial value
total = reduce(lambda x, y: x + y, numbers, 100)  # 115

# More readable alternatives exist
total = sum(numbers)                # Better for sum
import math
product = math.prod(numbers)        # Better for product (3.8+)

# When reduce is useful: Complex accumulations
def flatten(nested_list):
    return reduce(lambda a, b: a + b, nested_list, [])

result = flatten([[1, 2], [3, 4], [5]])  # [1, 2, 3, 4, 5]

# Finding max depth
def max_depth(obj):
    if isinstance(obj, dict):
        if not obj:
            return 1
        return 1 + reduce(max, [max_depth(v) for v in obj.values()])
    return 0
```

**Interview Tip**: Know reduce; mention better alternatives when applicable.
</details>

<details>
<summary><strong>46. What is reversed() and how is it different from .reverse()?</strong></summary>

**Answer:**
reversed() returns iterator; .reverse() modifies in-place.

```python
lst = [1, 2, 3, 4, 5]

# reversed() - returns reverse iterator
rev_iter = reversed(lst)
print(list(rev_iter))  # [5, 4, 3, 2, 1]
print(lst)             # [1, 2, 3, 4, 5] - unchanged

# .reverse() - modifies in-place, returns None
lst.reverse()
print(lst)             # [5, 4, 3, 2, 1]

# Slicing (also creates new list)
reversed_lst = lst[::-1]
print(reversed_lst)    # [5, 4, 3, 2, 1]
print(lst)             # Unchanged

# Works with any iterable
s = "hello"
print(list(reversed(s)))  # ['o', 'l', 'l', 'e', 'h']

# Performance
import timeit
lst = list(range(10000))

print(timeit.timeit(lambda: list(reversed(lst)), number=10000))
print(timeit.timeit(lambda: lst[::-1], number=10000))
# reversed() is more memory efficient for large lists
```

**Interview Tip**: Know when each is appropriate.
</details>

<details>
<summary><strong>47. What is sorted() vs .sort()?</strong></summary>

**Answer:**
sorted() returns new list; .sort() modifies in-place.

```python
lst = [3, 1, 4, 1, 5, 9]

# sorted() - returns new sorted list
sorted_lst = sorted(lst)         # [1, 1, 3, 4, 5, 9]
print(lst)                       # [3, 1, 4, 1, 5, 9] unchanged

# .sort() - sorts in-place, returns None
lst.sort()
print(lst)                       # [1, 1, 3, 4, 5, 9]

# Both support key and reverse parameters
data = ['apple', 'pie', 'a', 'longer']
sorted_by_length = sorted(data, key=len)
print(sorted_by_length)  # ['a', 'pie', 'apple', 'longer']

# Sorting with custom key
persons = [('Alice', 25), ('Bob', 20), ('Charlie', 23)]
sorted_persons = sorted(persons, key=lambda x: x[1])
# [('Bob', 20), ('Charlie', 23), ('Alice', 25)]

# Reverse sorting
descending = sorted([3, 1, 4, 1, 5], reverse=True)  # [5, 4, 3, 1, 1]

# sorted() works on any iterable
d = {'c': 3, 'a': 1, 'b': 2}
sorted_keys = sorted(d)  # ['a', 'b', 'c']
sorted_values = sorted(d.values())  # [1, 2, 3]
```

**Interview Tip**: Know differences and when to use each.
</details>

<details>
<summary><strong>48. What is filter() function?</strong></summary>

**Answer:**
Returns iterator of elements where function returns true.

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# filter() - returns iterator
even = filter(lambda x: x % 2 == 0, numbers)
print(list(even))  # [2, 4, 6, 8, 10]

# Equivalent list comprehension (preferred)
even = [x for x in numbers if x % 2 == 0]

# With None (filters out falsy values)
values = [0, 1, 2, False, '', 'hello']
filtered = list(filter(None, values))  # [1, 2, 'hello']

# Equivalent
filtered = [x for x in values if x]

# More complex filtering
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = list(filter(is_prime, range(20)))
# [2, 3, 5, 7, 11, 13, 17, 19]

# List comprehension is usually clearer
primes = [x for x in range(20) if is_prime(x)]
```

**Interview Tip**: Know filter(); prefer comprehensions for readability.
</details>

<details>
<summary><strong>49. What is type hints and when to use them?</strong></summary>

**Answer:**
Provide type information for static checking (covered in Q18 but deeper dive).

```python
from typing import Optional, Union, List, Dict, Tuple, Callable

# Basic type hints
def greet(name: str, age: int) -> str:
    return f"{name} is {age}"

# Function hints
def process_data(items: List[int]) -> Dict[str, float]:
    return {"sum": sum(items), "avg": sum(items) / len(items)}

# Optional (can be None)
def find_user(user_id: int) -> Optional[str]:
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)

# Union (multiple types)
def handle(value: Union[int, str]) -> None:
    if isinstance(value, int):
        print(value * 2)
    else:
        print(value.upper())

# Callable (function type)
def apply_function(fn: Callable[[int], str], x: int) -> str:
    return fn(x)

# Tuple (specific types per position)
def get_coordinates() -> Tuple[float, float]:
    return (1.0, 2.0)

# Variable hints
name: str = "Alice"
scores: List[int] = [90, 85, 92]

# Runtime checking (not automatic)
from typeguard import typechecked

@typechecked
def add(x: int, y: int) -> int:
    return x + y

# Runtime type checking tool
import mypy  # pip install mypy
# Run: mypy script.py
```

**Interview Tip**: Know type hints don't enforce at runtime by default.
</details>

<details>
<summary><strong>50. What is JSON serialization and deserialization?</strong></summary>

**Answer:**
Convert Python objects to/from JSON strings.

```python
import json

# Python to JSON (serialization)
data = {
    'name': 'Alice',
    'age': 25,
    'hobbies': ['reading', 'hiking'],
    'address': {'city': 'NYC', 'zip': '10001'}
}

# Convert to JSON string
json_str = json.dumps(data)
print(json_str)
# '{"name": "Alice", "age": 25, ...}'

# Pretty print
json_str = json.dumps(data, indent=2)

# Write to file
with open('data.json', 'w') as f:
    json.dump(data, f)

# JSON to Python (deserialization)
json_str = '{"name": "Bob", "age": 30}'
parsed = json.loads(json_str)
print(parsed['name'])  # "Bob"

# Read from file
with open('data.json', 'r') as f:
    data = json.load(f)

# Custom serialization (for non-JSON types)
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

data = {'timestamp': datetime.now()}
json_str = json.dumps(data, cls=DateTimeEncoder)
```

**Interview Tip**: Know json module and custom encoders.
</details>

<details>
<summary><strong>51. What is regular expressions (re module)?</strong></summary>

**Answer:**
Pattern matching and text manipulation using regex.

```python
import re

text = "Hello World 123 test@example.com"

# Basic matching
pattern = r'\d+'  # One or more digits
match = re.search(pattern, text)
if match:
    print(match.group())  # "123"

# Find all matches
pattern = r'\w+'  # Word characters
matches = re.findall(pattern, text)
# ['Hello', 'World', '123', 'test', 'example', 'com']

# Email pattern
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
is_valid = bool(re.match(email_pattern, 'test@example.com'))

# Substitution
text = "Hello World"
new_text = re.sub(r'World', 'Python', text)  # "Hello Python"

# Split by pattern
text = "one,two;three"
parts = re.split(r'[,;]', text)  # ['one', 'two', 'three']

# Common patterns
# \d - digit
# \w - word character (letter, digit, underscore)
# \s - whitespace
# ^ - start of string
# $ - end of string
# . - any character
# * - zero or more
# + - one or more
# ? - zero or one
# {n,m} - between n and m occurrences
# [...] - character class
# | - OR
```

**Interview Tip**: Know common patterns and re.search vs re.match.
</details>

<details>
<summary><strong>52. What is Collections.Counter?</strong></summary>

**Answer:**
Count frequency of elements in iterable.

```python
from collections import Counter

# Count elements
text = "hello world"
counter = Counter(text)
print(counter)  # Counter({'l': 3, 'o': 2, 'h': 1, ...})

# Most common
print(counter.most_common(3))  # [('l', 3), ('o', 2), ('h', 1)]

# Count list elements
words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
counter = Counter(words)
print(counter)  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})

# Arithmetic operations
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
print(c1 + c2)         # Counter({'a': 4, 'b': 3})
print(c1 - c2)         # Counter({'a': 2})
print(c1 & c2)         # Counter({'a': 1, 'b': 1}) - intersection
print(c1 | c2)         # Counter({'a': 3, 'b': 2}) - union

# Access counts
counter = Counter('hello')
print(counter['l'])    # 2
print(counter['z'])    # 0 (missing elements return 0, not KeyError)

# Convert to dict
d = dict(counter)
```

**Interview Tip**: Show practical use cases for word frequency, vote counting.
</details>

<details>
<summary><strong>53. What is Collections.defaultdict?</strong></summary>

**Answer:**
Dictionary that returns default value for missing keys.

```python
from collections import defaultdict

# Without defaultdict
d = {}
if 'key' not in d:
    d['key'] = 0
d['key'] += 1

# With defaultdict
d = defaultdict(int)  # Default to 0
d['key'] += 1

# Different defaults
from collections import defaultdict

d_list = defaultdict(list)
d_list['fruits'].append('apple')
d_list['fruits'].append('banana')
# {'fruits': ['apple', 'banana']}

d_int = defaultdict(int)
d_int['count'] += 1  # Works without KeyError

d_set = defaultdict(set)
d_set['tags'].add('python')
d_set['tags'].add('coding')

# Custom default factory
d = defaultdict(lambda: 'N/A')
print(d['missing'])  # 'N/A'

# Grouping elements
words = ['apple', 'apricot', 'banana', 'blueberry']
by_first = defaultdict(list)
for word in words:
    by_first[word[0]].append(word)
# {'a': ['apple', 'apricot'], 'b': ['banana', 'blueberry']}

# Counting
words = ['apple', 'banana', 'apple', 'cherry']
count = defaultdict(int)
for word in words:
    count[word] += 1
```

**Interview Tip**: Show practical examples of grouping and counting.
</details>

<details>
<summary><strong>54. What are string methods?</strong></summary>

**Answer:**
Common string operations.

```python
s = "  Hello World Python  "

# Case conversion
print(s.upper())                   # "  HELLO WORLD PYTHON  "
print(s.lower())                   # "  hello world python  "
print(s.capitalize())              # "  hello world python  " (first char)
print(s.title())                   # "  Hello World Python  "
print(s.swapcase())                # "  hELLO wORLD pYTHON  "

# Whitespace
print(s.strip())                   # "Hello World Python"
print(s.lstrip())                  # "Hello World Python  "
print(s.rstrip())                  # "  Hello World Python"

# Searching
print(s.find('World'))             # 8 (index, -1 if not found)
print(s.index('World'))            # 8 (raises ValueError if not found)
print(s.count('l'))                # 3
print(s.startswith('  Hello'))     # True
print(s.endswith('Python  '))      # True
print('World' in s)                # True

# Replacement
print(s.replace('World', 'Python'))  # "  Hello Python Python  "
print(s.replace('o', '0', 1))        # "  Hell0 World Python  " (1 replacement)

# Splitting and joining
words = s.strip().split()           # ['Hello', 'World', 'Python']
joined = ' '.join(words)            # "Hello World Python"
print(s.split(','))                 # ['  Hello World Python  ']

# Checking
print(s.isalpha())                 # False (has spaces)
print("123".isdigit())             # True
print("hello".islower())           # True
print("HELLO".isupper())           # True
print("HelloWorld".isalnum())       # True
```

**Interview Tip**: Know the most common string methods.
</details>

<details>
<summary><strong>55. What are list methods?</strong></summary>

**Answer:**
Common list operations.

```python
lst = [3, 1, 4, 1, 5, 9, 2, 6]

# Modification
lst.append(5)                      # Add at end: [3, 1, 4, 1, 5, 9, 2, 6, 5]
lst.extend([7, 8])                 # Add multiple: [..., 7, 8]
lst.insert(0, 99)                  # Insert at position: [99, 3, 1, ...]

# Removal
lst.remove(1)                      # Remove first occurrence
lst.pop()                          # Remove and return last: 8
lst.pop(0)                         # Remove and return at index: 99
lst.clear()                        # Remove all

# Searching
print(lst.index(4))                # 2 (raises ValueError if not found)
print(lst.count(1))                # 2 (how many 1s)

# Sorting
lst.sort()                         # Sort in-place
lst.sort(reverse=True)             # Descending
lst.sort(key=len)                  # Sort by length

# Reversing
lst.reverse()                      # Reverse in-place

# Copying
copy = lst.copy()                  # Shallow copy

# Example: Using list methods
tasks = ['buy milk', 'walk dog', 'write code']
tasks.append('read book')
tasks.insert(1, 'exercise')
tasks.remove('walk dog')
print(tasks)  # ['buy milk', 'exercise', 'write code', 'read book']
```

**Interview Tip**: Know difference between methods that modify vs return values.
</details>

<details>
<summary><strong>56. What is File I/O?</strong></summary>

**Answer:**
Reading and writing files.

```python
# Writing
with open('data.txt', 'w') as f:
    f.write('Hello World\n')
    f.writelines(['Line 1\n', 'Line 2\n'])

# Reading
with open('data.txt', 'r') as f:
    content = f.read()           # Entire file
    print(content)

# Read line by line
with open('data.txt', 'r') as f:
    for line in f:
        print(line.rstrip())     # Remove newline

# Readlines
with open('data.txt', 'r') as f:
    lines = f.readlines()        # ['Hello World\n', 'Line 1\n', ...]

# Append
with open('data.txt', 'a') as f:
    f.write('Appended line\n')

# File modes
# 'r'  - read (default)
# 'w'  - write (truncates)
# 'a'  - append
# 'x'  - exclusive creation
# 'b'  - binary
# '+' - read and write

# Binary
with open('image.bin', 'rb') as f:
    data = f.read()

# Check file exists
import os
if os.path.exists('data.txt'):
    print("File exists")

# File properties
print(os.path.getsize('data.txt'))
print(os.path.isfile('data.txt'))
print(os.path.isdir('.'))
```

**Interview Tip**: Show using context managers (with statement).
</details>

<details>
<summary><strong>57. What are dictionary comprehensions?</strong></summary>

**Answer:**
Create dictionaries concisely.

```python
# Basic dict comprehension
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# With condition
evens = {x: x**2 for x in range(10) if x % 2 == 0}
# {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# From key-value pairs
pairs = [('a', 1), ('b', 2), ('c', 3)]
d = {k: v for k, v in pairs}
# {'a': 1, 'b': 2, 'c': 3}

# Inverting dictionaries
d = {'a': 1, 'b': 2, 'c': 3}
inverted = {v: k for k, v in d.items()}
# {1: 'a', 2: 'b', 3: 'c'}

# Transforming values
prices = {'apple': 1.2, 'banana': 0.5, 'orange': 0.8}
discounted = {k: v * 0.9 for k, v in prices.items()}

# Grouping
words = ['apple', 'apricot', 'banana', 'blueberry']
grouped = {word[0]: [w for w in words if w[0] == word[0]] for word in words}
```

**Interview Tip**: Show practical examples of dict comprehensions.
</details>

<details>
<summary><strong>58. Set comprehensions</strong></summary>

Topic placeholder from the original remaining-questions outline.
</details>

<details>
<summary><strong>59. Generator expressions</strong></summary>

Topic placeholder from the original remaining-questions outline.
</details>

<details>
<summary><strong>60. Yield and generators</strong></summary>

Topic placeholder from the original remaining-questions outline.
</details>

<details>
<summary><strong>61. Python import system</strong></summary>

Topic placeholder from the original remaining-questions outline.
</details>

<details>
<summary><strong>62. __init__.py files</strong></summary>

Topic placeholder from the original remaining-questions outline.
</details>

<details>
<summary><strong>63. Relative vs absolute imports</strong></summary>

Topic placeholder from the original remaining-questions outline.
</details>

<details>
<summary><strong>64. Module caching</strong></summary>

Topic placeholder from the original remaining-questions outline.
</details>

<details>
<summary><strong>65. unittest framework</strong></summary>

Topic placeholder from the original remaining-questions outline.
</details>

<details>
<summary><strong>66. pytest framework</strong></summary>

Topic placeholder from the original remaining-questions outline.
</details>

<details>
<summary><strong>67. Mock and patch (unittest.mock)</strong></summary>

Topic placeholder from the original remaining-questions outline.
</details>

<details>
<summary><strong>68. Code coverage (coverage.py)</strong></summary>

Topic placeholder from the original remaining-questions outline.
</details>

<details>
<summary><strong>69. Debugging with pdb</strong></summary>

Topic placeholder from the original remaining-questions outline.
</details>

<details>
<summary><strong>70. Logging module and handlers</strong></summary>

Topic placeholder from the original remaining-questions outline.
</details>

---

## Advanced Questions (71-100)

<details>
<summary><strong>71. How does Python handle memory management?</strong></summary>

**Answer:**
Python uses reference counting and garbage collection.

**Reference counting**: Each object tracks how many references exist
```python
import sys

a = []
print(sys.getrefcount(a))  # References to a

b = a  # Another reference
print(sys.getrefcount(a))  # Increased

del a  # Reference removed
# List still exists because b references it
```

**Garbage collection**: Detects and removes unreachable objects
```python
import gc

# Circular references (reference counting can't detect)
class Node:
    def __init__(self):
        self.ref = self

node = Node()
node.ref = node  # Circular reference
del node
# Garbage collector will clean this up

gc.collect()  # Force garbage collection
print(gc.get_stats())  # Collection statistics
```

**Interview Tip**: Explain both mechanisms and memory leaks.
</details>

<details>
<summary><strong>72. What are weak references?</strong></summary>

**Answer:**
Weak references don't prevent object garbage collection.

```python
import weakref

class MyObject:
    pass

obj = MyObject()
weak_ref = weakref.ref(obj)

print(weak_ref())  # <MyObject object>

del obj
print(weak_ref())  # None - object was garbage collected

# WeakValueDictionary
cache = weakref.WeakValueDictionary()
cache['key'] = MyObject()
# When object is deleted, cache entry is removed automatically
```

**Use cases**: Caches, avoiding circular references

**Interview Tip**: Explain when weak references are needed.
</details>

<details>
<summary><strong>73. How do you profile Python code for performance?</strong></summary>

**Answer:**
Use `cProfile` for function-level profiling, `line_profiler` for line-by-line, `timeit` for micro-benchmarks, and `memory_profiler` for memory usage.

```python
import cProfile
import pstats
import io

def slow_function():
    total = 0
    for i in range(10000):
        total += sum(range(i))
    return total

# Profile with cProfile
pr = cProfile.Profile()
pr.enable()
slow_function()
pr.disable()

# Print top 10 functions by cumulative time
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(10)
print(s.getvalue())

# Quick benchmark with timeit
import timeit

# List comprehension vs loop
t1 = timeit.timeit("[x**2 for x in range(1000)]", number=1000)
t2 = timeit.timeit(
    "result = []\nfor x in range(1000):\n    result.append(x**2)",
    number=1000
)
print(f"Comprehension: {t1:.3f}s, Loop: {t2:.3f}s")
```

**Interview Tip:** Always profile before optimizing. 80/20 rule: 80% of time spent in 20% of code. Mention `snakeviz` for visual profiling, `py-spy` for sampling profilers on production.

</details>

<details>
<summary><strong>74. What is functools.lru_cache and how does it work?</strong></summary>

**Answer:**
`lru_cache` memoizes function results, caching up to `maxsize` calls using a Least Recently Used eviction policy. Avoids recomputing expensive function calls with the same arguments.

```python
import functools
import time

# Without cache
def fib_slow(n):
    if n < 2:
        return n
    return fib_slow(n-1) + fib_slow(n-2)

# With lru_cache
@functools.lru_cache(maxsize=128)
def fib_fast(n):
    if n < 2:
        return n
    return fib_fast(n-1) + fib_fast(n-2)

t1 = time.time(); fib_slow(30); print(f"Slow: {time.time()-t1:.3f}s")
t2 = time.time(); fib_fast(30); print(f"Fast: {time.time()-t2:.5f}s")

# Cache info
print(fib_fast.cache_info())  # CacheInfo(hits=..., misses=..., maxsize=128, currsize=...)
fib_fast.cache_clear()        # Clear cache

# Python 3.9+: use @functools.cache (unbounded)
@functools.cache
def factorial(n):
    return 1 if n <= 1 else n * factorial(n-1)

print(factorial(10))  # 3628800
```

**Interview Tip:** Only works with hashable arguments. Use `maxsize=None` for unlimited cache (same as `@cache`). Mention that it adds memory overhead — not suitable for functions with many unique inputs.

</details>

<details>
<summary><strong>75. What is the difference between `__slots__` and `__dict__`?</strong></summary>

**Answer:**
By default, Python stores instance attributes in a `__dict__` (a dictionary). `__slots__` replaces this with a fixed set of attributes, reducing memory overhead significantly for many instances.

```python
import sys

class WithDict:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class WithSlots:
    __slots__ = ["x", "y"]
    def __init__(self, x, y):
        self.x = x
        self.y = y

d = WithDict(1, 2)
s = WithSlots(1, 2)

print(f"WithDict size:  {sys.getsizeof(d)} bytes")
print(f"WithSlots size: {sys.getsizeof(s)} bytes")
print(f"Has __dict__: WithDict={hasattr(d, '__dict__')}, WithSlots={hasattr(s, '__dict__')}")

# Can't add new attributes with __slots__
try:
    s.z = 3  # AttributeError
except AttributeError as e:
    print(f"Can't add attribute: {e}")

# Memory benchmark with many instances
n = 100_000
dict_objs = [WithDict(i, i) for i in range(n)]
slot_objs = [WithSlots(i, i) for i in range(n)]
print(f"\nWithDict list:  {sys.getsizeof(dict_objs) + sum(sys.getsizeof(o) for o in dict_objs[:100])*1000 // 1024} KB approx")
```

**Interview Tip:** Use `__slots__` when creating thousands of instances of the same class (e.g., data model objects). Trade-off: loses flexibility (can't add arbitrary attributes), can't use `__weakref__` without explicitly adding it.

</details>

<details>
<summary><strong>76. What is the Global Interpreter Lock (GIL)?</strong></summary>

**Answer:**
The GIL is a mutex in CPython that allows only one thread to execute Python bytecode at a time. Protects reference counting but limits true parallelism for CPU-bound tasks. I/O-bound tasks still benefit from threading.

```python
import threading
import time

counter = 0

def increment(n):
    global counter
    for _ in range(n):
        counter += 1  # Not thread-safe despite GIL!

# The GIL doesn't make operations atomic
threads = [threading.Thread(target=increment, args=(100000,)) for _ in range(5)]
t = time.time()
for th in threads: th.start()
for th in threads: th.join()
print(f"Expected: 500000, Got: {counter}")  # Often less due to race conditions

# GIL released during I/O — threading works for I/O-bound tasks
import urllib.request

def fetch(url):
    try:
        urllib.request.urlopen(url, timeout=2)
    except:
        pass

urls = ["http://example.com"] * 5

# Sequential
t = time.time()
for url in urls: fetch(url)
print(f"Sequential: {time.time()-t:.2f}s")

# Concurrent (threads release GIL during network I/O)
t = time.time()
threads = [threading.Thread(target=fetch, args=(url,)) for url in urls]
for th in threads: th.start()
for th in threads: th.join()
print(f"Threaded: {time.time()-t:.2f}s")
```

**Interview Tip:** Use `multiprocessing` for CPU-bound parallelism (bypasses GIL via separate processes). Use `threading` or `asyncio` for I/O-bound tasks. NumPy/Pandas release the GIL during C-level computations.

</details>

<details>
<summary><strong>77. What is multiprocessing in Python and when to use it?</strong></summary>

**Answer:**
`multiprocessing` spawns separate OS processes, each with its own Python interpreter and memory space — bypassing the GIL. Ideal for CPU-bound tasks. Communication via queues, pipes, or shared memory.

```python
import multiprocessing as mp
import time

def cpu_bound(n):
    return sum(i**2 for i in range(n))

if __name__ == "__main__":
    data = [5_000_000] * 4

    # Sequential
    t = time.time()
    results = [cpu_bound(x) for x in data]
    print(f"Sequential: {time.time()-t:.2f}s")

    # Multiprocessing pool
    t = time.time()
    with mp.Pool(processes=4) as pool:
        results = pool.map(cpu_bound, data)
    print(f"Multiprocessing: {time.time()-t:.2f}s")

    # Process with Queue
    def worker(q, n):
        q.put(cpu_bound(n))

    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(q, 1_000_000)) for _ in range(4)]
    for p in procs: p.start()
    for p in procs: p.join()
    results = [q.get() for _ in procs]
    print(f"Results via queue: {results[:2]}")
```

**Interview Tip:** `Pool.map` for embarrassingly parallel tasks. `Pool.starmap` for multiple arguments. Overhead of spawning processes makes multiprocessing inefficient for small tasks. Use `concurrent.futures.ProcessPoolExecutor` for a cleaner API.

</details>

<details>
<summary><strong>78. What is `asyncio` and how does async/await work?</strong></summary>

**Answer:**
`asyncio` is Python's event loop-based concurrency framework for I/O-bound tasks. `async def` defines a coroutine; `await` suspends execution until an awaitable completes, allowing other coroutines to run.

```python
import asyncio
import time

async def fetch_data(url, delay):
    print(f"Starting {url}")
    await asyncio.sleep(delay)  # simulates I/O wait; releases event loop
    print(f"Done {url}")
    return f"data from {url}"

async def main():
    # Sequential (slow)
    t = time.time()
    r1 = await fetch_data("api1", 1)
    r2 = await fetch_data("api2", 1)
    print(f"Sequential: {time.time()-t:.2f}s")

    # Concurrent with gather (fast)
    t = time.time()
    results = await asyncio.gather(
        fetch_data("api1", 1),
        fetch_data("api2", 1),
        fetch_data("api3", 1),
    )
    print(f"Concurrent: {time.time()-t:.2f}s")
    print(results)

asyncio.run(main())

# Tasks for explicit scheduling
async def with_tasks():
    task1 = asyncio.create_task(fetch_data("task1", 0.5))
    task2 = asyncio.create_task(fetch_data("task2", 0.5))
    # Both tasks run concurrently
    r1 = await task1
    r2 = await task2
    return r1, r2
```

**Interview Tip:** `asyncio` is single-threaded — achieves concurrency via cooperative multitasking, not parallelism. Only use `await` with async functions or objects implementing `__await__`. Use `aiohttp` for async HTTP, `asyncpg` for async Postgres.

</details>

<details>
<summary><strong>79. What is the difference between `threading`, `multiprocessing`, and `asyncio`?</strong></summary>

**Answer:**
All three achieve concurrency but with different mechanisms and best use cases.

```python
# Summary comparison
"""
threading:
- Multiple threads in ONE process
- GIL limits CPU parallelism
- Good for: I/O-bound tasks (network, file I/O)
- Shared memory (be careful with race conditions)
- Lighter weight than processes

multiprocessing:
- Multiple SEPARATE processes
- True CPU parallelism (bypasses GIL)
- Good for: CPU-bound tasks (computation, ML inference)
- Separate memory (communicate via Queue/Pipe)
- Higher overhead (process spawn, pickling)

asyncio:
- Single thread, event loop
- No parallelism — cooperative concurrency
- Good for: many concurrent I/O operations (thousands of connections)
- No memory sharing issues
- Lowest overhead
"""

import concurrent.futures
import time

def cpu_task(n):
    return sum(i**2 for i in range(n))

async def async_io_task(delay):
    import asyncio
    await asyncio.sleep(delay)
    return "done"

# ProcessPoolExecutor for CPU-bound
with concurrent.futures.ProcessPoolExecutor() as executor:
    t = time.time()
    futures = [executor.submit(cpu_task, 1_000_000) for _ in range(4)]
    results = [f.result() for f in futures]
    print(f"CPU-bound (4 procs): {time.time()-t:.2f}s")

# ThreadPoolExecutor for I/O-bound
import urllib.request
def fetch(url):
    try: urllib.request.urlopen(url, timeout=3)
    except: pass

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    t = time.time()
    list(executor.map(fetch, ["http://example.com"]*5))
    print(f"I/O-bound (5 threads): {time.time()-t:.2f}s")
```

**Interview Tip:** Rule of thumb: CPU-bound → multiprocessing; I/O-bound with few connections → threading; I/O-bound with many connections → asyncio.

</details>

<details>
<summary><strong>80. What are Python descriptors?</strong></summary>

**Answer:**
Descriptors are objects that define `__get__`, `__set__`, or `__delete__` methods. When assigned as class attributes, they intercept attribute access on instances. `property`, `classmethod`, `staticmethod` are all built using descriptors.

```python
class Validator:
    """Data descriptor: defines both __get__ and __set__"""
    def __set_name__(self, owner, name):
        self.name = name
        self.private = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self  # accessed on class, not instance
        return getattr(obj, self.private, None)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be numeric, got {type(value)}")
        if value < 0:
            raise ValueError(f"{self.name} must be non-negative")
        setattr(obj, self.private, value)

class Circle:
    radius = Validator()
    area_cache = None

    def __init__(self, radius):
        self.radius = radius  # triggers Validator.__set__

    @property
    def area(self):
        import math
        return math.pi * self.radius ** 2

c = Circle(5)
print(f"Radius: {c.radius}, Area: {c.area:.2f}")

try:
    c.radius = -1
except ValueError as e:
    print(f"Caught: {e}")

try:
    c.radius = "hello"
except TypeError as e:
    print(f"Caught: {e}")

# Descriptor lookup order: data descriptor > instance __dict__ > non-data descriptor
print(f"Class-level access: {Circle.radius}")  # returns the Validator itself
```

**Interview Tip:** Data descriptors (both `__get__` + `__set__`) take priority over instance `__dict__`. Non-data descriptors (only `__get__`) can be overridden by instance attributes. `property` is a data descriptor.

</details>

<details>
<summary><strong>81. What is Python's `__init_subclass__` and `__class_getitem__`?</strong></summary>

**Answer:**
`__init_subclass__` is called on the parent class whenever a subclass is created — useful for plugin registration or validation. `__class_getitem__` enables generic syntax like `MyClass[int]`.

```python
class PluginBase:
    _registry = {}

    def __init_subclass__(cls, plugin_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if plugin_name:
            PluginBase._registry[plugin_name] = cls
            print(f"Registered plugin: {plugin_name} -> {cls.__name__}")

class CSVPlugin(PluginBase, plugin_name="csv"):
    def parse(self, data): return data.split(",")

class JSONPlugin(PluginBase, plugin_name="json"):
    def parse(self, data):
        import json; return json.loads(data)

print(f"Registry: {PluginBase._registry}")

# __class_getitem__ for generic-style syntax
class TypedList:
    def __class_getitem__(cls, item):
        return type(f"TypedList[{item.__name__}]", (cls,), {"item_type": item})

IntList = TypedList[int]
print(f"IntList.item_type: {IntList.item_type}")
print(f"TypedList[str] name: {TypedList[str].__name__}")
```

**Interview Tip:** `__init_subclass__` replaces many metaclass use cases more cleanly. Used in Django-style model registration, plugin systems, and ORM frameworks.

</details>

<details>
<summary><strong>82. What are abstract base classes (ABCs)?</strong></summary>

**Answer:**
ABCs define interfaces that subclasses must implement. Using `abc.ABC` and `@abstractmethod`, you prevent instantiation of incomplete subclasses and enforce contracts.

```python
from abc import ABC, abstractmethod
from typing import List

class DataProcessor(ABC):
    @abstractmethod
    def load(self, path: str) -> List:
        """Load data from path."""
        ...

    @abstractmethod
    def process(self, data: List) -> List:
        """Process data."""
        ...

    def run(self, path: str) -> List:  # Template method
        data = self.load(path)
        return self.process(data)

class CSVProcessor(DataProcessor):
    def load(self, path):
        return [line.split(",") for line in open(path)]

    def process(self, data):
        return [row for row in data if len(row) > 1]

# Can't instantiate ABC directly
try:
    dp = DataProcessor()
except TypeError as e:
    print(f"Cannot instantiate ABC: {e}")

# Can check if class implements interface
print(issubclass(CSVProcessor, DataProcessor))  # True

# ABCs also work for virtual subclasses (register without inheriting)
class FakeProcessor:
    def load(self, path): return []
    def process(self, data): return data

DataProcessor.register(FakeProcessor)
print(isinstance(FakeProcessor(), DataProcessor))  # True
```

**Interview Tip:** ABCs are Python's way to define interfaces. `collections.abc` has ABCs for Sequence, Mapping, Iterable, etc. Use `isinstance(x, collections.abc.Mapping)` instead of `isinstance(x, dict)` for duck-typing checks.

</details>

<details>
<summary><strong>83. What is `dataclasses` and when to use it?</strong></summary>

**Answer:**
`@dataclass` auto-generates `__init__`, `__repr__`, `__eq__`, and optionally `__hash__`, `__lt__` for classes that primarily hold data. Less boilerplate than writing them manually.

```python
from dataclasses import dataclass, field, asdict, astuple
from typing import List

@dataclass(order=True, frozen=False)
class Employee:
    name: str
    salary: float
    skills: List[str] = field(default_factory=list)
    department: str = "Engineering"
    # sort_index: float = field(init=False, repr=False)  # computed field

    def __post_init__(self):
        if self.salary < 0:
            raise ValueError("Salary must be non-negative")

    def give_raise(self, pct: float):
        self.salary *= (1 + pct)

emp1 = Employee("Alice", 90000, ["Python", "ML"])
emp2 = Employee("Bob", 75000, ["Java"])

print(emp1)         # Employee(name='Alice', salary=90000, ...)
print(emp1 == Employee("Alice", 90000, ["Python", "ML"]))  # True
emp1.give_raise(0.1)
print(emp1.salary)  # 99000.0

# Serialization
print(asdict(emp1))   # {'name': 'Alice', 'salary': 99000.0, ...}
print(astuple(emp1))  # ('Alice', 99000.0, ...)

# Frozen dataclass (immutable, hashable)
@dataclass(frozen=True)
class Point:
    x: float
    y: float

p = Point(1.0, 2.0)
print(hash(p))  # hashable because frozen
```

**Interview Tip:** Use `frozen=True` for immutable value objects (also makes them hashable). Prefer over `namedtuple` when you need methods or mutable state. `@dataclass(slots=True)` in Python 3.10+ for memory efficiency.

</details>

<details>
<summary><strong>84. What is `typing` module and type hints?</strong></summary>

**Answer:**
Type hints provide static type information for tools like mypy, IDEs, and `pydantic`. They don't enforce types at runtime by default but greatly improve code readability and catch bugs early.

```python
from typing import (Optional, Union, List, Dict, Tuple, Set,
                    Callable, TypeVar, Generic, Any, Literal)

# Basic hints
def process(name: str, count: int = 1) -> List[str]:
    return [name] * count

# Optional (can be None)
def find_user(user_id: int) -> Optional[str]:
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)  # returns str or None

# Union
def parse(value: Union[str, int, float]) -> float:
    return float(value)

# Callable
def apply(func: Callable[[int], int], value: int) -> int:
    return func(value)

print(apply(lambda x: x**2, 5))  # 25

# TypeVar for generics
T = TypeVar("T")

def first(items: List[T]) -> T:
    return items[0]

# Python 3.9+: use built-in generics (no import needed)
def modern(data: list[int], mapping: dict[str, int]) -> tuple[int, ...]:
    return tuple(data)

# Literal for specific values
def set_direction(d: Literal["north", "south", "east", "west"]) -> None:
    print(f"Going {d}")

# Runtime type checking with isinstance (not type hints)
value: Union[str, int] = "hello"
if isinstance(value, str):
    print(value.upper())
```

**Interview Tip:** Run `mypy` for static type checking. `pydantic` uses type hints for runtime validation. `from __future__ import annotations` enables forward references and PEP 563 lazy evaluation. Python 3.10 added `X | Y` syntax for Union.

</details>

<details>
<summary><strong>85. What is `contextlib` and how do you create context managers without classes?</strong></summary>

**Answer:**
`contextlib` provides utilities for creating context managers using generators (`@contextmanager`) or function-based decorators — simpler than implementing `__enter__`/`__exit__`.

```python
from contextlib import contextmanager, suppress, ExitStack
import time

# contextmanager: generator-based context manager
@contextmanager
def timer(label=""):
    t = time.time()
    try:
        yield  # code inside `with` block runs here
    finally:
        print(f"{label}: {time.time()-t:.4f}s")

with timer("sum operation"):
    total = sum(range(10_000_000))

# suppress: swallow specific exceptions
with suppress(FileNotFoundError, PermissionError):
    open("nonexistent_file.txt")
# No exception raised — continues silently

# ExitStack: dynamic context managers
@contextmanager
def temp_env(key, value):
    import os
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            del os.environ[key]
        else:
            os.environ[key] = old

with ExitStack() as stack:
    stack.enter_context(temp_env("DEBUG", "1"))
    stack.enter_context(temp_env("LOG_LEVEL", "INFO"))
    import os
    print(f"DEBUG={os.environ['DEBUG']}, LOG_LEVEL={os.environ['LOG_LEVEL']}")
# Both env vars restored after block

# Reentrant context managers
from contextlib import redirect_stdout
import io

f = io.StringIO()
with redirect_stdout(f):
    print("This goes to StringIO instead of stdout")
print(f"Captured: {f.getvalue()!r}")
```

**Interview Tip:** `@contextmanager` is preferred over `__enter__`/`__exit__` for simple cases. Everything before `yield` is `__enter__`, everything in `finally` is `__exit__`. `suppress` replaces `try/except: pass` patterns.

</details>

<details>
<summary><strong>86. What is `itertools` and when is it useful?</strong></summary>

**Answer:**
`itertools` provides memory-efficient iterator building blocks. Essential for data pipelines, combinatorics, and working with large sequences without loading everything into memory.

```python
import itertools

# chain: concatenate iterables
combined = list(itertools.chain([1, 2], [3, 4], [5]))
print(combined)  # [1, 2, 3, 4, 5]

# islice: lazy slicing of iterables
def infinite_counter():
    n = 0
    while True:
        yield n
        n += 1

first_10 = list(itertools.islice(infinite_counter(), 10))
print(first_10)  # [0, 1, 2, ..., 9]

# groupby: group consecutive elements
data = [("a", 1), ("a", 2), ("b", 3), ("b", 4), ("c", 5)]
for key, group in itertools.groupby(data, key=lambda x: x[0]):
    print(f"{key}: {list(group)}")

# combinations and permutations
print(list(itertools.combinations("ABC", 2)))   # [('A','B'), ('A','C'), ('B','C')]
print(list(itertools.permutations("AB", 2)))    # [('A','B'), ('B','A')]
print(list(itertools.combinations_with_replacement("AB", 2)))  # [('A','A'), ('A','B'), ('B','B')]

# product: cartesian product
print(list(itertools.product([0, 1], repeat=3)))  # all 3-bit binary numbers

# cycle, repeat, count
limited_cycle = list(itertools.islice(itertools.cycle("AB"), 6))
print(limited_cycle)  # ['A', 'B', 'A', 'B', 'A', 'B']

# dropwhile / takewhile
import itertools
data = [1, 2, 5, 3, 1, 6]
print(list(itertools.takewhile(lambda x: x < 5, data)))  # [1, 2]
print(list(itertools.dropwhile(lambda x: x < 5, data)))  # [5, 3, 1, 6]
```

**Interview Tip:** `itertools` functions return iterators — memory efficient for large datasets. `itertools.groupby` requires sorted input to group all matching keys together. Combine with `functools.reduce` and `operator` for functional pipelines.

</details>

<details>
<summary><strong>87. What is `collections` module? Name key data structures.</strong></summary>

**Answer:**
`collections` provides specialized container types: `Counter`, `defaultdict`, `OrderedDict`, `namedtuple`, `deque`, and `ChainMap`.

```python
from collections import Counter, defaultdict, OrderedDict, namedtuple, deque, ChainMap

# Counter: frequency counting
words = "the cat sat on the mat the cat".split()
c = Counter(words)
print(c.most_common(3))    # [('the', 3), ('cat', 2), ('sat', 1)]
print(c["the"] + c["dog"]) # 3 + 0 (no KeyError)

# defaultdict: no KeyError on missing keys
graph = defaultdict(list)
for u, v in [(1,2), (1,3), (2,4)]:
    graph[u].append(v)
print(dict(graph))  # {1: [2, 3], 2: [4]}

# namedtuple: tuple with named fields
Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
print(p.x, p.y, p._asdict())  # 3 4 {'x': 3, 'y': 4}

# deque: O(1) append/pop from both ends
dq = deque([1, 2, 3], maxlen=5)
dq.appendleft(0)
dq.append(4)
print(dq)           # deque([0, 1, 2, 3, 4])
dq.rotate(2)
print(dq)           # deque([3, 4, 0, 1, 2])

# ChainMap: multiple dicts as one view
defaults = {"color": "blue", "size": "M"}
user_prefs = {"color": "red"}
merged = ChainMap(user_prefs, defaults)
print(merged["color"])  # "red" (user_prefs takes priority)
print(merged["size"])   # "M" (falls back to defaults)

# OrderedDict: maintains insertion order (less needed in Python 3.7+)
od = OrderedDict([("b", 2), ("a", 1)])
od.move_to_end("b")
print(list(od.keys()))  # ['a', 'b']
```

**Interview Tip:** `Counter` arithmetic: `+`, `-`, `&`, `|` work element-wise. `defaultdict(int)` is common for frequency counting. `deque` with `maxlen` is a sliding window buffer. In Python 3.7+, regular `dict` maintains insertion order.

</details>

<details>
<summary><strong>88. What is `logging` and how should you use it in production?</strong></summary>

**Answer:**
Python's `logging` module provides a flexible logging system with levels (DEBUG < INFO < WARNING < ERROR < CRITICAL), handlers (file, console, network), and formatters. Always prefer logging over `print` in production.

```python
import logging
import sys

# Basic configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log"),
    ]
)

logger = logging.getLogger(__name__)  # Best practice: use module name

logger.debug("Debug info (not shown at INFO level)")
logger.info("Application started")
logger.warning("Low disk space")
logger.error("Database connection failed")
logger.critical("System shutting down")

# Structured logging with extra fields
logger.info("User logged in", extra={"user_id": 42, "ip": "192.168.1.1"})

# Exception logging
try:
    1 / 0
except ZeroDivisionError:
    logger.exception("Unexpected error occurred")  # includes traceback

# Logger hierarchy
parent_logger = logging.getLogger("myapp")
child_logger = logging.getLogger("myapp.database")

# RotatingFileHandler for log rotation
from logging.handlers import RotatingFileHandler
handler = RotatingFileHandler("app.log", maxBytes=10_000_000, backupCount=5)
parent_logger.addHandler(handler)

# Production pattern: configure once at entry point
def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',  # JSON
    )
```

**Interview Tip:** Use `logger = logging.getLogger(__name__)` in every module. Never use root logger directly. Configure handlers only at the application entry point. Use `logging.exception()` inside `except` blocks to capture traceback.

</details>

<details>
<summary><strong>89. What is `unittest` and how does Python testing work?</strong></summary>

**Answer:**
`unittest` is Python's built-in testing framework (xUnit style). `pytest` is the preferred tool in practice — simpler syntax, better output, fixtures, and rich plugin ecosystem.

```python
import unittest
from unittest.mock import Mock, patch, MagicMock

# Class under test
class Calculator:
    def add(self, a, b): return a + b
    def divide(self, a, b):
        if b == 0: raise ValueError("Cannot divide by zero")
        return a / b
    def fetch_rate(self, currency):
        import urllib.request
        # simulate external API call
        return 1.0

class TestCalculator(unittest.TestCase):
    def setUp(self):
        """Runs before each test"""
        self.calc = Calculator()

    def tearDown(self):
        """Runs after each test"""
        pass

    def test_add(self):
        self.assertEqual(self.calc.add(2, 3), 5)
        self.assertEqual(self.calc.add(-1, 1), 0)

    def test_divide(self):
        self.assertAlmostEqual(self.calc.divide(10, 3), 3.333, places=3)

    def test_divide_by_zero(self):
        with self.assertRaises(ValueError) as ctx:
            self.calc.divide(5, 0)
        self.assertIn("Cannot divide by zero", str(ctx.exception))

    @patch("urllib.request.urlopen")  # mock external dependency
    def test_fetch_rate_with_mock(self, mock_urlopen):
        mock_urlopen.return_value = MagicMock()
        result = self.calc.fetch_rate("USD")
        self.assertEqual(result, 1.0)

    def test_parametrized(self):
        """pytest supports @pytest.mark.parametrize; unittest uses subTest"""
        cases = [(2, 3, 5), (0, 0, 0), (-1, 1, 0)]
        for a, b, expected in cases:
            with self.subTest(a=a, b=b):
                self.assertEqual(self.calc.add(a, b), expected)

if __name__ == "__main__":
    unittest.main()
```

**Interview Tip:** Prefer `pytest` in practice — simpler `assert` statements, better failure messages, fixtures via `@pytest.fixture`, parametrize with `@pytest.mark.parametrize`. Use `unittest.mock.patch` for both frameworks.

</details>

<details>
<summary><strong>90. What is the `pathlib` module?</strong></summary>

**Answer:**
`pathlib.Path` provides an object-oriented interface for filesystem paths, replacing `os.path` string manipulation with cleaner, cross-platform code.

```python
from pathlib import Path

# Create paths (forward slash operator works on all platforms)
base = Path("/tmp")
data_dir = base / "data" / "raw"
config = Path.home() / ".config" / "myapp.json"

# Path introspection
p = Path("/home/user/data/report.csv")
print(p.name)       # "report.csv"
print(p.stem)       # "report"
print(p.suffix)     # ".csv"
print(p.parent)     # /home/user/data
print(p.parts)      # ('/', 'home', 'user', 'data', 'report.csv')
print(p.is_absolute())  # True

# Create directories
data_dir.mkdir(parents=True, exist_ok=True)

# Read and write
config_path = Path("config.txt")
config_path.write_text("key=value\n")
content = config_path.read_text()
print(repr(content))

# Glob patterns
project = Path(".")
python_files = list(project.glob("**/*.py"))
print(f"Python files: {len(python_files)}")

# Check existence
print(config_path.exists())  # True
print(config_path.is_file()) # True
print(data_dir.is_dir())     # True

# Replace / rename
new_path = config_path.with_suffix(".json")
print(new_path)  # config.json

# Iterate directory
for f in Path(".").iterdir():
    if f.is_file():
        print(f.name, f.stat().st_size, "bytes")
```

**Interview Tip:** `pathlib` is preferred over `os.path` in Python 3.6+. The `/` operator constructs paths. Use `Path.cwd()` for current directory, `Path.home()` for home directory. Fully cross-platform.

</details>

<details>
<summary><strong>91. What is `__call__` and callable objects?</strong></summary>

**Answer:**
Implementing `__call__` makes an instance callable like a function. Useful for stateful callables, function-like objects, and decorators implemented as classes.

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
        self.call_count = 0

    def __call__(self, value):
        self.call_count += 1
        return value * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(5))     # 10
print(triple(5))     # 15
print(callable(double))  # True

# Class-based decorator using __call__
class Retry:
    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts

    def __call__(self, func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == self.max_attempts - 1:
                        raise
                    print(f"Attempt {attempt+1} failed: {e}, retrying...")
        return wrapper

@Retry(max_attempts=3)
def unstable_operation():
    import random
    if random.random() < 0.7:
        raise ValueError("Random failure")
    return "success"

try:
    result = unstable_operation()
except ValueError:
    print("All attempts failed")

# Check number of calls
print(f"double called {double.call_count} times")
```

**Interview Tip:** `callable(obj)` returns True if `obj` has `__call__`. Class-based decorators are useful when the decorator needs state or configuration. Functions are callable objects too.

</details>

<details>
<summary><strong>92. What is `__new__` vs `__init__`?</strong></summary>

**Answer:**
`__new__` creates the instance (allocates memory), `__init__` initializes it. `__new__` returns the instance; `__init__` receives it. Override `__new__` for singletons, immutable types, or metaclass patterns.

```python
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, value):
        self.value = value  # Called every time, but same object

s1 = Singleton(1)
s2 = Singleton(2)
print(s1 is s2)    # True — same instance
print(s1.value)    # 2 — __init__ ran twice on same object

# __new__ for immutable type customization
class PositiveInt(int):
    def __new__(cls, value):
        if value <= 0:
            raise ValueError(f"Must be positive, got {value}")
        return super().__new__(cls, value)

n = PositiveInt(5)
print(n, type(n))   # 5 <class '__main__.PositiveInt'>
print(n + 3)        # 8 (int arithmetic works)

try:
    PositiveInt(-1)
except ValueError as e:
    print(e)

# __new__ order: __new__ creates, __init__ initializes
class Tracked:
    instances = []

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        cls.instances.append(instance)
        return instance

    def __init__(self, name):
        self.name = name

t1 = Tracked("a")
t2 = Tracked("b")
print(f"Total instances: {len(Tracked.instances)}")  # 2
```

**Interview Tip:** Rarely need to override `__new__` — only for singleton patterns, customizing immutable types (int, str, tuple subclasses), or metaclass work. `__init__` is sufficient for most cases.

</details>

<details>
<summary><strong>93. What are Python's built-in functions you should know?</strong></summary>

**Answer:**
Key built-ins every Python developer should know: `map`, `filter`, `zip`, `enumerate`, `sorted`, `any`, `all`, `min`/`max` with key, `sum`, `vars`, `dir`, `getattr`/`setattr`, `isinstance`, `issubclass`.

```python
# map and filter (prefer list comprehensions, but know these)
nums = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, nums))
evens = list(filter(lambda x: x % 2 == 0, nums))
print(squared, evens)

# zip (and zip_longest)
names = ["Alice", "Bob", "Charlie"]
scores = [95, 87, 92]
paired = dict(zip(names, scores))
print(paired)

from itertools import zip_longest
print(list(zip_longest([1,2,3], ["a","b"], fillvalue=None)))

# enumerate
for i, name in enumerate(names, start=1):
    print(f"{i}. {name}")

# sorted with key
data = [{"name": "Bob", "age": 30}, {"name": "Alice", "age": 25}]
print(sorted(data, key=lambda x: x["age"]))

# any / all
print(any(x > 4 for x in nums))     # True
print(all(x > 0 for x in nums))     # True
print(all(x > 2 for x in nums))     # False

# min/max with key
print(min(data, key=lambda x: x["age"]))  # Alice

# getattr with default
class Config: debug = True
cfg = Config()
print(getattr(cfg, "debug", False))    # True
print(getattr(cfg, "missing", "N/A")) # "N/A"

# vars and dir
print(vars(cfg))  # {'debug': True} — instance __dict__

# isinstance with tuple of types
value = 42
print(isinstance(value, (int, float)))  # True
```

**Interview Tip:** Use `any()`/`all()` with generator expressions for short-circuit evaluation (stops at first True/False). `zip` stops at shortest iterable — use `zip_longest` when lengths differ.

</details>

<details>
<summary><strong>94. What are common Python anti-patterns to avoid?</strong></summary>

**Answer:**
Common anti-patterns: mutable default arguments, bare `except`, using `type()` instead of `isinstance()`, comparing to `None` with `==`, importing `*`, and catching then silently ignoring exceptions.

```python
# ANTI-PATTERN 1: Mutable default argument
def bad_append(item, lst=[]):    # lst shared across calls!
    lst.append(item)
    return lst

print(bad_append(1))  # [1]
print(bad_append(2))  # [1, 2] -- unexpected!

def good_append(item, lst=None):  # CORRECT
    if lst is None:
        lst = []
    lst.append(item)
    return lst

# ANTI-PATTERN 2: Bare except
try:
    risky()
except:  # catches EVERYTHING including KeyboardInterrupt, SystemExit!
    pass

# CORRECT
try:
    risky()
except Exception as e:  # or specific exception
    logger.exception("Failed")

# ANTI-PATTERN 3: type() instead of isinstance()
x = True
print(type(x) == int)     # False -- bool is a subclass of int
print(isinstance(x, int)) # True  -- correct for duck typing

# ANTI-PATTERN 4: Comparing to None with ==
value = None
if value == None: pass   # works but not Pythonic
if value is None: pass   # CORRECT -- use 'is' for None, True, False

# ANTI-PATTERN 5: Not using context managers
f = open("file.txt", "w")
f.write("data")
# f.close() might not be called if exception occurs!

with open("file.txt", "w") as f:  # CORRECT
    f.write("data")

# ANTI-PATTERN 6: String concatenation in loop
parts = ["a", "b", "c", "d"]
result = ""
for p in parts:
    result += p   # O(n^2) — creates new string each time

result = "".join(parts)  # CORRECT — O(n)
```

**Interview Tip:** The Zen of Python: `import this`. Explicit is better than implicit. Flat is better than nested. Errors should never pass silently. Know PEP 8 style conventions.

</details>

<details>
<summary><strong>95. What is `__repr__` vs `__str__`?</strong></summary>

**Answer:**
`__repr__` is for developers (unambiguous, ideally eval-able to recreate the object). `__str__` is for end users (readable, informal). `repr()` falls back to `__repr__` if `__str__` is not defined.

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector({self.x!r}, {self.y!r})"  # !r uses repr for values

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __format__(self, fmt):
        if fmt == "polar":
            import math
            r = math.sqrt(self.x**2 + self.y**2)
            theta = math.degrees(math.atan2(self.y, self.x))
            return f"|{r:.2f}| ∠{theta:.1f}°"
        return str(self)

v = Vector(3, 4)
print(repr(v))    # Vector(3, 4)  -- developer representation
print(str(v))     # (3, 4)        -- user representation
print(f"{v}")     # (3, 4)        -- uses __str__
print(f"{v!r}")   # Vector(3, 4)  -- forces __repr__
print(f"{v:polar}")  # |5.00| angle45.0deg

# In containers, __repr__ is used
print([v, Vector(1, 2)])  # [Vector(3, 4), Vector(1, 2)]

# Minimal __repr__ recommendation
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):
        # Ideal: eval(repr(p)) == p
        return f"{type(self).__name__}({self.x}, {self.y})"
```

**Interview Tip:** Always implement `__repr__` — it's used in debugging, logging, and containers. Implement `__str__` when user-facing output differs. The `!r` format spec is shorthand for `repr()`.

</details>

<details>
<summary><strong>96. What are Python's comparison and hashing protocols?</strong></summary>

**Answer:**
Objects are comparable via `__eq__`, `__lt__`, `__le__`, `__gt__`, `__ge__`. For use in sets/dicts as keys, `__hash__` must be consistent with `__eq__`. `@functools.total_ordering` fills in missing comparisons.

```python
import functools

@functools.total_ordering  # auto-generates remaining comparisons from __eq__ and one of __lt__/__le__/__gt__/__ge__
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius

    def __eq__(self, other):
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius == other.celsius

    def __lt__(self, other):
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius < other.celsius

    def __hash__(self):
        return hash(self.celsius)  # required for use in sets/dicts

    def __repr__(self):
        return f"Temperature({self.celsius}°C)"

t1 = Temperature(20)
t2 = Temperature(30)
t3 = Temperature(20)

print(t1 < t2)   # True
print(t1 > t2)   # False (from total_ordering)
print(t1 == t3)  # True
print(t1 <= t3)  # True

# Hashable — can be used in sets and as dict keys
temp_set = {t1, t2, t3}
print(len(temp_set))  # 2 (t1 == t3, same hash)

temp_dict = {t1: "comfortable", t2: "warm"}
print(temp_dict[Temperature(20)])  # "comfortable"

# Sorted uses comparisons
temps = [Temperature(30), Temperature(10), Temperature(20)]
print(sorted(temps))  # [Temperature(10), Temperature(20), Temperature(30)]
```

**Interview Tip:** If you define `__eq__`, you should also define `__hash__` (Python makes the class unhashable if you only define `__eq__`). Return `NotImplemented` (not `NotImplementedError`) when comparison is not supported — this lets Python try the reverse operation.

</details>

<details>
<summary><strong>97. What is `sys` module and key attributes?</strong></summary>

**Answer:**
`sys` provides access to Python interpreter state: command-line arguments, module import system, Python path, recursion limits, standard streams, and interpreter info.

```python
import sys

# Command-line arguments
print(f"Script name: {sys.argv[0]}")
print(f"Arguments: {sys.argv[1:]}")

# Python version and platform
print(f"Python: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Max int: {sys.maxsize}")

# Recursion limit
print(f"Recursion limit: {sys.getrecursionlimit()}")
sys.setrecursionlimit(2000)  # increase if needed for deep recursion

# Module import system
print(f"Module search path: {sys.path[:3]}")
print("numpy" in sys.modules)  # True if numpy is imported

# Object size (shallow)
x = [1, 2, 3, 4, 5]
print(f"List size: {sys.getsizeof(x)} bytes")

# Standard streams
import sys
sys.stdout.write("Hello stdout\n")
sys.stderr.write("Error message\n")

# Exit with code
# sys.exit(0)   # Clean exit
# sys.exit(1)   # Error exit

# Reference count
a = []
b = a
print(f"Reference count for a: {sys.getrefcount(a)}")  # 3 (a, b, getrefcount arg)

# Exception info inside except block
try:
    1 / 0
except:
    exc_type, exc_value, exc_tb = sys.exc_info()
    print(f"Exception type: {exc_type.__name__}")
```

**Interview Tip:** `sys.path` manipulation at runtime changes where Python looks for modules (use with caution). `sys.modules` caching means importing the same module twice is free. `sys.getsizeof` is shallow — use `tracemalloc` for deep memory profiling.

</details>

<details>
<summary><strong>98. What is `os` and `subprocess` module?</strong></summary>

**Answer:**
`os` provides OS-level operations (environment, file system, process). `subprocess` runs external commands and captures output.

```python
import os
import subprocess

# Environment variables
path = os.environ.get("PATH", "")
os.environ["MY_VAR"] = "hello"
print(os.getenv("MY_VAR"))

# File system
cwd = os.getcwd()
print(f"CWD: {cwd}")

files = os.listdir(".")
print(f"Files: {files[:5]}")

# os.path (prefer pathlib)
p = os.path.join("folder", "file.txt")
print(os.path.exists(p), os.path.dirname(p))

# Walk directory tree
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if not d.startswith(".")]  # skip hidden
    for f in files:
        if f.endswith(".py"):
            print(os.path.join(root, f))
    break  # only first level for demo

# Process info
print(f"PID: {os.getpid()}, CPU count: {os.cpu_count()}")

# subprocess
result = subprocess.run(
    ["python", "--version"],
    capture_output=True,
    text=True,
    check=True
)
print(f"Python version: {result.stdout.strip()}")

# Shell command (avoid shell=True when possible -- security risk)
output = subprocess.check_output(["ls", "-la"], text=True)
print(output[:200])

# Communicate with subprocess
proc = subprocess.Popen(
    ["sort"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
)
stdout, _ = proc.communicate("banana\napple\ncherry\n")
print(stdout)
```

**Interview Tip:** Use `pathlib` over `os.path` for path operations. Avoid `shell=True` in `subprocess` — security risk (command injection). Use `subprocess.run(..., check=True)` to raise on non-zero exit codes.

</details>

<details>
<summary><strong>99. What is Python packaging? `setup.py`, `pyproject.toml`, and virtual environments.</strong></summary>

**Answer:**
Python packaging bundles code for distribution. Modern projects use `pyproject.toml` (PEP 517/518). Virtual environments isolate project dependencies. `pip`, `poetry`, and `uv` manage packages.

```python
# pyproject.toml (modern standard, replaces setup.py)
"""
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "my-package"
version = "1.0.0"
description = "A sample package"
authors = [{name = "Alice", email = "alice@example.com"}]
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20",
    "pandas>=1.3",
]

[project.optional-dependencies]
dev = ["pytest", "mypy", "black"]
ml = ["scikit-learn>=1.0"]

[project.scripts]
my-cli = "my_package.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
"""

# Virtual environments
import subprocess, sys

# Create venv (shell commands)
shell_commands = """
# Create virtual environment
python -m venv .venv

# Activate (Unix/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Export dependencies
pip freeze > requirements.txt

# deactivate
deactivate
"""
print(shell_commands)

# Checking current environment
print(f"Python executable: {sys.executable}")
print(f"Virtual env: {sys.prefix}")

# import system basics
import importlib

# Programmatic import
module_name = "json"
mod = importlib.import_module(module_name)
print(f"Loaded: {mod.__name__}")

# Reload module (useful in development)
importlib.reload(mod)
```

**Interview Tip:** Use `pyproject.toml` for new projects. `poetry` and `uv` are popular modern package managers. Always use virtual environments — never install to system Python. `src` layout (`src/mypackage/`) prevents accidental imports from project root.

</details>

<details>
<summary><strong>100. What are Python's protocols and structural subtyping (duck typing)?</strong></summary>

**Answer:**
Python uses duck typing — "if it walks like a duck and quacks like a duck, it's a duck." `typing.Protocol` (PEP 544) formalizes structural subtyping without inheritance, checked statically by mypy.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...
    def resize(self, factor: float) -> None: ...

class Circle:
    def __init__(self, radius: float):
        self.radius = radius

    def draw(self) -> None:
        print(f"Drawing circle r={self.radius}")

    def resize(self, factor: float) -> None:
        self.radius *= factor

class Square:
    def __init__(self, side: float):
        self.side = side

    def draw(self) -> None:
        print(f"Drawing square s={self.side}")

    def resize(self, factor: float) -> None:
        self.side *= factor

# No inheritance from Drawable needed!
def render_all(shapes: list[Drawable]) -> None:
    for shape in shapes:
        shape.draw()

shapes = [Circle(5), Square(3)]
render_all(shapes)  # works via structural subtyping

# runtime_checkable enables isinstance checks
print(isinstance(Circle(1), Drawable))  # True
print(isinstance("hello", Drawable))   # False

# Classic duck typing without Protocol
class FakeCircle:
    def draw(self): print("fake draw")
    def resize(self, f): pass

render_all([FakeCircle()])  # works — duck typing!

# Key Python protocols (magic methods)
protocols = {
    "__len__": "Sized — supports len()",
    "__iter__": "Iterable — supports for loop",
    "__getitem__": "Sequence — supports indexing",
    "__contains__": "Container — supports 'in'",
    "__enter__/__exit__": "Context manager — supports 'with'",
    "__await__": "Awaitable — supports 'await'",
    "__add__": "Supports + operator",
}
for method, desc in protocols.items():
    print(f"  {method}: {desc}")
```

**Interview Tip:** `Protocol` with `@runtime_checkable` enables both static (mypy) and runtime (`isinstance`) structural checks. Prefer Protocol over ABCs when you don't control the implementations. Core Python protocols like `Iterable`, `Sized`, `Mapping` are in `collections.abc`.

</details>

---

## Study Tips

1. **Understand, don't memorize**: Focus on "why" not just "what"
2. **Code along**: Run every example and modify it
3. **Ask follow-ups**: "What if we...?"
4. **Mock interviews**: Practice explaining answers verbally
5. **Track weak areas**: Focus extra on topics you struggle with

Good luck! 🚀

## Summary in simple points

- Lists, dictionaries and sets are mutable; tuples are immutable containers whose contents may still reference mutable objects. Choose structures by required operations, not slogans about speed.
- Understand identity versus equality, hashing contracts, shallow versus deep copies, mutable defaults and shared class state.
- Use comprehensions, generators, iterators, range, zip and enumerate with clear ordering, exhaustion and length assumptions.
- Functions use positional/keyword arguments, closures, decorators and lambdas; distinguish lazy map/filter from eager collection creation.
- Classes use properties, descriptors, MRO, abstract bases, dataclasses and protocols. Metaclasses, slots and subclass hooks solve specific construction or storage needs.
- Handle expected errors explicitly. Context managers and contextlib make file, connection and lock cleanup reliable.
- Use JSON for interoperable data, regex for suitable text patterns, and documented schemas for external inputs.
- Understand imports, package boundaries, module caching and main guards. Use isolated environments and pyproject metadata for reproducible packaging.
- Counter, defaultdict, deque, itertools and functools provide reusable counting, queueing, iteration and caching tools.
- Test behaviours with unittest/pytest and controlled mocks. Coverage counts executed code; it does not prove useful assertions.
- Profile before optimising. Understand reference ownership, garbage collection, weak references and the limitations of cache growth.
- Choose threads, processes and asyncio for the actual workload. GIL behaviour depends on interpreter/build and extension code; do not assume every Python runtime has identical constraints.
- Use pathlib and bounded file operations. Pass subprocess arguments safely and avoid logging secrets.
- Type hints and structural protocols support analysis and interfaces; ordinary hints do not validate runtime input.
- Know object construction, callable objects, repr/str, comparison methods and built-in utilities well enough to predict concrete output.
- Explain performance and failure behaviour with runnable fixtures; avoid bare exceptions, mutable defaults, unbounded tasks and accidental shared state.
