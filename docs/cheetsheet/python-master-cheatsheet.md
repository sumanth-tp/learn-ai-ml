---
title: Python Master Cheatsheet
sidebar_position: 1
---

# Python Master Cheatsheet

## Beginner

| Method | Method description | Code example |
|---|---|---|
| `print()` | Writes values to standard output. Useful for quick debugging and scripts. | `print("status:", status)` |
| `type()` | Returns the runtime type of an object. | `type([1, 2, 3])  # list` |
| `isinstance()` | Checks whether an object matches a type or tuple of types. Prefer this over direct `type()` checks for inheritance-aware code. | `isinstance(value, (int, float))` |
| `len()` | Returns the number of items in a sequence, collection, or mapping. | `len(users)` |
| `range()` | Produces a lazy integer sequence, commonly used for loops. | `for i in range(0, 10, 2): print(i)` |
| `enumerate()` | Iterates with both index and value without manually tracking counters. | `for idx, name in enumerate(names, start=1): print(idx, name)` |
| `zip()` | Combines multiple iterables position by position. Stops at the shortest iterable. | `for name, score in zip(names, scores): print(name, score)` |
| `input()` | Reads a string from standard input. Convert it explicitly when needed. | `age = int(input("Age: "))` |
| `str.lower()` | Converts text to lowercase for normalization. | `"Python".lower()  # "python"` |
| `str.upper()` | Converts text to uppercase. | `"ok".upper()  # "OK"` |
| `str.strip()` | Removes leading and trailing whitespace or specified characters. | `"  hello  ".strip()` |
| `str.split()` | Splits a string into a list using whitespace or a delimiter. | `"a,b,c".split(",")` |
| `str.join()` | Joins strings with a separator. More efficient than repeated concatenation. | `", ".join(["red", "blue"])` |
| `str.replace()` | Replaces matching substrings. | `"a-b-c".replace("-", "_")` |
| `str.startswith()` | Checks whether text begins with a prefix. | `path.startswith("/api/")` |
| `str.endswith()` | Checks whether text ends with a suffix. | `filename.endswith(".py")` |
| `list.append()` | Adds one item to the end of a list. | `items.append("new")` |
| `list.extend()` | Adds all items from another iterable to a list. | `items.extend(["a", "b"])` |
| `list.insert()` | Inserts an item at a specific index. | `items.insert(0, "first")` |
| `list.pop()` | Removes and returns an item by index. Defaults to the last item. | `last = items.pop()` |
| `list.remove()` | Removes the first matching value. Raises `ValueError` if absent. | `items.remove("old")` |
| `list.sort()` | Sorts a list in place. Use when mutating the original list is fine. | `users.sort(key=lambda user: user["age"])` |
| `sorted()` | Returns a new sorted list from any iterable. | `top = sorted(scores, reverse=True)[:3]` |
| `dict.get()` | Reads a key with a fallback instead of raising `KeyError`. | `role = user.get("role", "guest")` |
| `dict.keys()` | Returns a dynamic view of dictionary keys. | `if "id" in payload.keys(): ...` |
| `dict.values()` | Returns a dynamic view of dictionary values. | `sum(cart.values())` |
| `dict.items()` | Iterates through key-value pairs. | `for key, value in config.items(): print(key, value)` |
| `set.add()` | Adds an item to a set if it is not already present. | `seen.add(user_id)` |
| `set.remove()` | Removes an item and raises `KeyError` if missing. | `active.remove(user_id)` |
| `set.discard()` | Removes an item if present and does nothing if missing. | `active.discard(user_id)` |
| `in` | Checks membership in lists, strings, sets, tuples, and dict keys. | `if user_id in seen: return` |

## Intermediate

| Method | Method description | Code example |
|---|---|---|
| List comprehension | Builds a list from an iterable in a compact, readable way. | `squares = [n * n for n in nums]` |
| Dict comprehension | Builds dictionaries from iterables. Useful for indexing and transformations. | `by_id = {user["id"]: user for user in users}` |
| Set comprehension | Builds a unique collection while transforming values. | `domains = {email.split("@")[1] for email in emails}` |
| Generator expression | Lazy expression for streaming values without allocating a full list. | `total = sum(order.amount for order in orders)` |
| `any()` | Returns `True` if at least one item is truthy. Short-circuits. | `has_admin = any(user.is_admin for user in users)` |
| `all()` | Returns `True` if every item is truthy. Short-circuits. | `valid = all(field in payload for field in required)` |
| `map()` | Applies a function lazily to each item. Often clearer as a comprehension for simple cases. | `names = list(map(str.title, raw_names))` |
| `filter()` | Keeps items for which a predicate is truthy. Often clearer as a comprehension. | `active = list(filter(lambda user: user.active, users))` |
| `lambda` | Small anonymous function for simple callbacks and sorting keys. | `sorted(users, key=lambda user: user.last_login)` |
| `sum()` | Adds numeric values, often with a generator expression. | `revenue = sum(order.total for order in orders)` |
| `min()` | Finds the smallest item, optionally by key. | `oldest = min(users, key=lambda user: user.created_at)` |
| `max()` | Finds the largest item, optionally by key. | `latest = max(events, key=lambda event: event.timestamp)` |
| `reversed()` | Iterates over items in reverse order without copying when possible. | `for item in reversed(history): print(item)` |
| `slice` syntax | Extracts parts of sequences with start, stop, and step. | `last_five = items[-5:]` |
| Unpacking | Assigns multiple values from an iterable. | `first, *middle, last = values` |
| `pathlib.Path` | Object-oriented filesystem paths. Prefer it over manual string path joins. | `from pathlib import Path; data = Path("data/users.json").read_text()` |
| `open()` | Opens files. Always use a context manager to close safely. | `with open("app.log", encoding="utf-8") as file: lines = file.readlines()` |
| `json.loads()` | Parses a JSON string into Python objects. | `payload = json.loads(raw_body)` |
| `json.dumps()` | Serializes Python objects to a JSON string. | `body = json.dumps(payload, indent=2)` |
| `try` and `except` | Handles expected failures without crashing the whole program. Keep exception scopes small. | `try: value = int(raw); except ValueError: value = 0` |
| `raise` | Throws an exception when a function cannot continue correctly. | `if amount < 0: raise ValueError("amount must be positive")` |
| `with` | Uses a context manager for setup and cleanup. | `with lock: shared_state["count"] += 1` |
| Function defaults | Provide default argument values. Avoid mutable defaults. | `def fetch(limit=100): return limit` |
| Keyword-only arguments | Forces clarity at call sites for options and flags. | `def retry(fn, *, attempts=3, delay=1): ...` |
| `*args` | Accepts extra positional arguments. | `def log(message, *tags): print(message, tags)` |
| `**kwargs` | Accepts extra keyword arguments. | `def build_user(**fields): return User(**fields)` |
| `dataclasses.dataclass` | Generates common methods for plain data objects. | `@dataclass(frozen=True)\nclass Point:\n    x: int\n    y: int` |
| `typing.Optional` | Represents a value that can be present or `None`. In modern Python, `str` unioned with `None` is also common. | `def find_user(id: int) -> Optional[User]: ...` |
| `typing.Callable` | Types a callback function. | `def apply(fn: Callable[[int], int], value: int) -> int: return fn(value)` |
| `collections.Counter` | Counts hashable items. Excellent for frequency problems. | `counts = Counter(words); counts.most_common(3)` |
| `collections.defaultdict` | Supplies default values for missing keys. | `groups = defaultdict(list); groups[user.team].append(user)` |
| `collections.deque` | Fast queue and stack operations from both ends. | `queue = deque([root]); node = queue.popleft()` |

## Advanced

| Method | Method description | Code example |
|---|---|---|
| Decorator | Wraps a function to add behavior such as logging, timing, auth, or caching. | `def timed(fn):\n    @wraps(fn)\n    def wrapper(*args, **kwargs):\n        start = time.perf_counter(); result = fn(*args, **kwargs); print(time.perf_counter() - start); return result\n    return wrapper` |
| `functools.wraps()` | Preserves function metadata when writing decorators. | `@wraps(fn)\ndef wrapper(*args, **kwargs): return fn(*args, **kwargs)` |
| `functools.lru_cache()` | Memoizes pure function results. Useful for dynamic programming and expensive deterministic calls. | `@lru_cache(maxsize=1024)\ndef fib(n): return n if n < 2 else fib(n - 1) + fib(n - 2)` |
| `functools.cached_property` | Computes an instance property once and caches the value. | `@cached_property\ndef profile(self): return load_profile(self.user_id)` |
| `functools.partial()` | Pre-fills some function arguments to create a specialized callable. | `read_json = partial(json.load, parse_float=Decimal)` |
| `itertools.chain()` | Lazily concatenates iterables. | `for row in chain(csv_rows, api_rows): process(row)` |
| `itertools.groupby()` | Groups adjacent items by key. Sort first if you need global grouping. | `for team, members in groupby(sorted(users, key=attrgetter("team")), key=attrgetter("team")): ...` |
| `itertools.product()` | Cartesian product for combinations of parameter choices. | `for lr, batch in product(lrs, batch_sizes): train(lr, batch)` |
| `itertools.islice()` | Lazily slices an iterator. | `first_100 = list(islice(stream, 100))` |
| `operator.itemgetter()` | Fast key function for dict-like or tuple-like objects. | `sorted(rows, key=itemgetter("created_at"))` |
| `operator.attrgetter()` | Fast key function for object attributes. | `sorted(users, key=attrgetter("last_login"))` |
| `@property` | Exposes computed values as attributes while keeping method logic hidden. | `@property\ndef full_name(self): return f"{self.first} {self.last}"` |
| `@classmethod` | Receives the class as `cls`; useful for named constructors. | `@classmethod\ndef from_dict(cls, data): return cls(**data)` |
| `@staticmethod` | Namespaced helper that does not need `self` or `cls`. | `@staticmethod\ndef normalize_email(email): return email.strip().lower()` |
| `__repr__()` | Returns an unambiguous developer-facing representation. | `def __repr__(self): return f"User(id={self.id!r})"` |
| `__str__()` | Returns a user-facing string representation. | `def __str__(self): return self.name` |
| `__eq__()` | Defines equality behavior. Keep it consistent with hashing rules. | `def __eq__(self, other): return isinstance(other, User) and self.id == other.id` |
| `__hash__()` | Allows immutable objects to be used in sets and dict keys. | `def __hash__(self): return hash(self.id)` |
| `__iter__()` | Makes an object iterable. | `def __iter__(self): return iter(self.items)` |
| `__enter__()` and `__exit__()` | Implements a context manager. | `def __enter__(self): self.conn = connect(); return self.conn\n def __exit__(self, exc_type, exc, tb): self.conn.close()` |
| `contextlib.contextmanager` | Creates a context manager from a generator function. | `@contextmanager\ndef timer(): start = time.perf_counter(); yield; print(time.perf_counter() - start)` |
| `yield` | Produces values lazily from a generator. | `def chunks(items, size):\n    for i in range(0, len(items), size): yield items[i:i + size]` |
| `yield from` | Delegates yielding to another iterable or generator. | `def flatten(groups):\n    for group in groups: yield from group` |
| Structural pattern matching | Matches shapes of data with `match` and `case`. Useful for parsers and command handlers. | `match command:\n    case {"type": "create", "name": name}: create(name)\n    case _: raise ValueError("unknown command")` |
| `Protocol` | Defines structural interfaces for type checking without inheritance. | `class Repository(Protocol):\n    def get(self, id: str) -> User: ...` |
| `TypedDict` | Types dictionary-shaped data, often from APIs or JSON. | `class UserPayload(TypedDict):\n    id: str\n    email: str` |
| `Generic` | Builds reusable typed containers and services. | `class Repository(Generic[T]):\n    def get(self, id: str) -> T: ...` |
| `Enum` | Defines named constant values. | `class Status(Enum):\n    PENDING = "pending"\n    DONE = "done"` |
| `slots` dataclass | Reduces instance memory and prevents dynamic attributes. | `@dataclass(slots=True)\nclass User:\n    id: str\n    email: str` |
| `frozen` dataclass | Makes dataclass instances immutable and hash-friendly when fields are hashable. | `@dataclass(frozen=True)\nclass Money:\n    amount: Decimal\n    currency: str` |
| `heapq` | Implements priority queues and top-k algorithms. | `heapq.heappush(heap, (priority, task)); priority, task = heapq.heappop(heap)` |
| `bisect` | Binary-search insertion points in sorted lists. | `idx = bisect_left(sorted_scores, target)` |
| `re.compile()` | Compiles regex once for reuse and clearer validation code. | `EMAIL_RE = re.compile(r"^[^@]+@[^@]+$"); bool(EMAIL_RE.match(email))` |
| `subprocess.run()` | Runs external commands safely when arguments are passed as a list. | `result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, check=True)` |
| `logging.getLogger()` | Uses structured module-level logging instead of `print()` in production. | `logger = logging.getLogger(__name__); logger.info("created user", extra={"user_id": user.id})` |

## Senior Developer

| Method | Method description | Code example |
|---|---|---|
| Dependency injection | Pass dependencies into functions or constructors to make code testable and decoupled. | `def create_user(payload, *, repo: UserRepo, clock: Clock): repo.save(User.from_payload(payload, clock.now()))` |
| Repository pattern | Encapsulates persistence behind a focused interface. Useful when swapping DB implementations or testing services. | `class UserRepo(Protocol):\n    def save(self, user: User) -> None: ...` |
| Unit of work | Groups related repository operations into one transaction boundary. | `with uow.transaction(): user = uow.users.get(id); uow.audit.log(user)` |
| Pure function | Avoids hidden I/O and mutation. Easier to test, cache, and reason about. | `def price_after_discount(price: Decimal, pct: Decimal) -> Decimal: return price * (Decimal("1") - pct)` |
| Idempotent function | Can be safely retried without duplicating side effects. Essential for APIs and jobs. | `def upsert_user(repo, user): repo.upsert(key=user.email, value=user)` |
| Defensive copy | Prevents callers from mutating internal state. | `def tags(self) -> tuple[str, ...]: return tuple(self._tags)` |
| Sentinel object | Distinguishes "argument omitted" from `None` as a valid value. | `_MISSING = object()\ndef update(name=_MISSING):\n    if name is not _MISSING: ...` |
| Custom exception hierarchy | Makes error handling precise and expressive. | `class AppError(Exception): pass\nclass UserNotFound(AppError): pass` |
| Exception chaining | Preserves original failure context when raising domain errors. | `except IntegrityError as exc: raise DuplicateEmail(email) from exc` |
| `contextvars.ContextVar` | Stores request-scoped data safely across async tasks. | `request_id_var = ContextVar("request_id", default="-")` |
| `async def` | Defines coroutine functions for high-concurrency I/O. | `async def fetch_user(client, id): response = await client.get(f"/users/{id}"); return response.json()` |
| `asyncio.gather()` | Runs independent coroutines concurrently and waits for all results. | `users, orders = await asyncio.gather(fetch_users(), fetch_orders())` |
| `asyncio.TaskGroup` | Structured concurrency for related async tasks. Cancels siblings on failure. | `async with asyncio.TaskGroup() as tg:\n    tg.create_task(sync_users())\n    tg.create_task(sync_orders())` |
| `asyncio.Semaphore` | Limits concurrency to protect services or local resources. | `sem = asyncio.Semaphore(10)\nasync with sem: return await call_api()` |
| Async context manager | Handles async setup and cleanup, such as DB sessions or HTTP clients. | `async with httpx.AsyncClient() as client: data = await fetch(client)` |
| Retry with backoff | Retries transient failures while avoiding request storms. | `for attempt in range(5):\n    try: return call()\n    except TimeoutError: time.sleep(2 ** attempt)` |
| Timeout boundary | Prevents a dependency from hanging the whole workflow. | `result = await asyncio.wait_for(fetch(), timeout=3)` |
| `ThreadPoolExecutor` | Runs blocking I/O concurrently. Not for CPU-bound speedups due to the GIL. | `with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(download, urls))` |
| `ProcessPoolExecutor` | Runs CPU-bound work in separate processes. | `with ProcessPoolExecutor() as pool: results = list(pool.map(parse_file, files))` |
| Immutability by design | Uses immutable values to reduce shared-state bugs. | `@dataclass(frozen=True)\nclass Command:\n    user_id: str\n    action: str` |
| Configuration object | Centralizes environment parsing and validation. | `@dataclass(frozen=True)\nclass Settings:\n    db_url: str\n    debug: bool = False` |
| Boundary validation | Validates external input at the edge, then keeps internal code strongly typed. | `def handler(raw): command = CreateUserCommand.validate(raw); service.create(command)` |
| Pagination iterator | Streams pages without exposing pagination mechanics to callers. | `def iter_users(client):\n    cursor = None\n    while page := client.list_users(cursor=cursor):\n        yield from page.items; cursor = page.next_cursor` |
| Batch processing | Processes large inputs in chunks to control memory and transaction size. | `for batch in chunks(records, 500): repo.bulk_insert(batch)` |
| Stable sorting key | Makes ordering deterministic for repeatable tests and APIs. | `users = sorted(users, key=lambda user: (user.last_name, user.id))` |
| `time.perf_counter()` | Measures elapsed time with a high-resolution monotonic clock. | `start = time.perf_counter(); run_job(); elapsed = time.perf_counter() - start` |
| `time.monotonic()` | Measures deadlines safely without wall-clock jumps. | `deadline = time.monotonic() + timeout` |
| `pytest.mark.parametrize` | Tests many cases with one concise test body. | `@pytest.mark.parametrize("raw,expected", [("1", 1), ("2", 2)])\ndef test_parse(raw, expected): assert parse(raw) == expected` |
| Fixture pattern | Shares test setup cleanly without global state. | `@pytest.fixture\ndef user_repo(tmp_path): return SqliteUserRepo(tmp_path / "test.db")` |
| Monkeypatching | Replaces environment or dependencies in tests. | `monkeypatch.setenv("APP_ENV", "test")` |
| Property-based testing | Tests behavior across many generated inputs. | `@given(st.lists(st.integers()))\ndef test_sorted_is_ordered(nums): assert sorted(nums) == sorted(sorted(nums))` |
| Profiling hot paths | Measures before optimizing. Use CPU and memory profilers to find real bottlenecks. | `python -m cProfile -o profile.out app.py` |
| Avoiding mutable defaults | Uses `None` or sentinel defaults to avoid shared mutable state. | `def add_tag(tag, tags=None): tags = [] if tags is None else tags; tags.append(tag); return tags` |
| Lazy import | Delays expensive optional imports until needed. Use sparingly and document why. | `def export_pdf(data): import reportlab; return build_pdf(data)` |
| Module `__all__` | Defines a public import surface for a module. | `__all__ = ["UserService", "UserRepository"]` |
| `if __name__ == "__main__"` | Keeps script entrypoints from running on import. | `if __name__ == "__main__": raise SystemExit(main())` |
| `argparse` | Builds maintainable command-line interfaces. | `parser = argparse.ArgumentParser(); parser.add_argument("--limit", type=int, default=100)` |
| `dataclass.replace()` | Creates modified copies of immutable dataclass objects. | `updated = replace(user, email=new_email)` |
| `zoneinfo.ZoneInfo` | Uses standard-library timezone handling. | `now = datetime.now(ZoneInfo("UTC"))` |
| `decimal.Decimal` | Represents money and exact decimal arithmetic. | `total = Decimal("19.99") * Decimal("1.18")` |
| `uuid.uuid4()` | Generates random unique identifiers. | `order_id = uuid.uuid4()` |
| `secrets.token_urlsafe()` | Generates secure random tokens. Use for auth, not `random`. | `token = secrets.token_urlsafe(32)` |
| `weakref.WeakValueDictionary` | Caches objects without preventing garbage collection. | `cache = weakref.WeakValueDictionary(); cache[key] = obj` |
| `copy.deepcopy()` | Recursively copies nested mutable objects. Use when ownership boundaries require isolation. | `snapshot = copy.deepcopy(config)` |
| `abc.ABC` | Defines nominal abstract base classes when inheritance is intentional. | `class Parser(ABC):\n    @abstractmethod\n    def parse(self, text: str) -> Document: ...` |
| `importlib.import_module()` | Dynamically imports plugins or optional integrations. | `module = importlib.import_module(plugin_path)` |
| `importlib.resources.files()` | Reads package data without relying on filesystem paths. | `template = resources.files("app.templates").joinpath("email.html").read_text()` |
