---
title: Regex Master Cheatsheet
sidebar_position: 25
---

# Regex Master Cheatsheet

## Pattern basics

| Method | Description | Code example |
|---|---|---|
| Literal match | Matches exact characters. | `re.search("error", "fatal error found")` |
| `.` | Matches any character except newline by default. | `re.findall("a.c", "abc axc a-c")` |
| Character class | Matches one character from a set. | `re.findall("[aeiou]", "regex")` |
| Negated class | Matches one character not in the set. | `re.findall("[^0-9]", "a1b2")` |
| Range | Matches characters in a range. | `re.findall("[A-Z]", "Model API")` |
| Escape | Matches special characters literally. | `re.search(r"price\\.json", "price.json")` |

## Quantifiers and anchors

| Method | Description | Code example |
|---|---|---|
| `*` | Zero or more repetitions. | `re.findall(r"ab*", "a ab abb")` |
| `+` | One or more repetitions. | `re.findall(r"ab+", "a ab abb")` |
| `?` | Zero or one repetition. | `re.findall(r"colou?r", "color colour")` |
| `{m,n}` | Bounded repetitions. | `re.findall(r"\\d{2,4}", "id 12 1234 12345")` |
| `^` | Start of string or line with multiline mode. | `re.search(r"^ERROR", line)` |
| `$` | End of string or line with multiline mode. | `re.search(r"\\.csv$", filename)` |

## Groups and lookaround

| Method | Description | Code example |
|---|---|---|
| Capturing group | Captures part of a match. | `m = re.search(r"user:(\\w+)", "user:ada")`<br/>`print(m.group(1))` |
| Named group | Captures with a readable name. | `m = re.search(r"(?P<year>\\d{4})-(?P<month>\\d{2})", "2026-05")`<br/>`print(m.groupdict())` |
| Non-capturing group | Groups without capturing. | `re.findall(r"(?:cat&#124;dog)s?", "cat dogs")` |
| Alternation | Matches one alternative. | `re.findall(r"error&#124;warning", "error then warning")` |
| Positive lookahead | Requires text ahead without consuming it. | `re.findall(r"\\w+(?=@example\\.com)", "a@example.com b@test.com")` |
| Negative lookahead | Rejects text ahead. | `re.findall(r"\\b(?!test\\b)\\w+", "prod test dev")` |

## Python `re` module

| Method | Description | Code example |
|---|---|---|
| `re.compile()` | Compiles pattern for reuse. | `EMAIL_RE = re.compile(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")` |
| `re.search()` | Finds first match anywhere. | `match = re.search(r"loss=(\\d+\\.\\d+)", log_line)` |
| `re.match()` | Matches only at start. | `re.match(r"GET", "GET /health")` |
| `re.fullmatch()` | Requires whole string match. | `ok = re.fullmatch(r"[A-Z]{3}-\\d{4}", ticket_id)` |
| `re.findall()` | Returns all non-overlapping matches. | `numbers = re.findall(r"\\d+", text)` |
| `re.sub()` | Replaces matches. | `clean = re.sub(r"\\s+", " ", raw_text).strip()` |
| Flags | Modify matching behavior. | `pattern = re.compile(r"^error", flags=re.IGNORECASE)` |

## ML and text tasks

| Method | Description | Code example |
|---|---|---|
| Extract emails | Finds simple email-like strings. | `emails = re.findall(r"[^@\\s]+@[^@\\s]+\\.[^@\\s]+", text)` |
| Normalize whitespace | Common preprocessing step. | `text = re.sub(r"\\s+", " ", text).strip()` |
| Remove URLs | Drop links before text modeling. | `text = re.sub(r"https?://\\S+", "", text)` |
| Extract hashtags | Social text feature extraction. | `tags = re.findall(r"#(\\w+)", tweet)` |
| Parse logs | Extract fields from structured logs. | `m = re.search(r"level=(\\w+) latency_ms=(\\d+)", line)`<br/>`level, latency = m.groups()` |
| Mask PII | Replace sensitive patterns. | `masked = re.sub(r"\\b\\d{3}-\\d{2}-\\d{4}\\b", "***-**-****", text)` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| Validate email lightly | Good for basic forms, not full RFC validation. | `EMAIL_RE = re.compile(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")`<br/>`is_valid = EMAIL_RE.fullmatch(email) is not None` |
| Validate ISO date | Captures year, month, day. | `DATE_RE = re.compile(r"^(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})$")` |
| Tokenize simple words | Quick baseline tokenizer. | `tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())` |
| Extract numbers | Capture integers and decimals. | `numbers = [float(x) for x in re.findall(r"-?\\d+(?:\\.\\d+)?", text)]` |
| Split on many delimiters | Split by comma, semicolon, or whitespace. | `parts = re.split(r"[,;\\s]+", raw)` |
| Redact API keys | Remove secret-looking tokens. | `redacted = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-REDACTED", text)` |
| Multiline log blocks | Use DOTALL for stack traces. | `blocks = re.findall(r"ERROR.*?(?=\\n\\d{4}-\\d{2}-\\d{2}&#124;$)", logs, flags=re.DOTALL)` |
| Prefer parser when needed | Regex is poor for nested languages. | `# Use json.loads, BeautifulSoup, or ast instead of regex for nested formats.` |

## Senior regex engineering

| Method | Description | Code example |
|---|---|---|
| Compile at module load | Avoid recompiling hot-path regex patterns. | `EMAIL_RE = re.compile(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")`<br/>`def is_email(value): return EMAIL_RE.fullmatch(value) is not None` |
| Use raw strings | Avoid accidental Python string escapes. | `pattern = re.compile(r"\\b\\d{4}-\\d{2}-\\d{2}\\b")` |
| Bound quantifiers | Avoid runaway patterns on untrusted input. | `SAFE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")` |
| Avoid catastrophic backtracking | Do not nest ambiguous quantifiers. | `# Risky: (a+)+`<br/>`# Safer: use explicit structure or a parser.` |
| Prefer `fullmatch` for validation | Validation should consume the whole string. | `if not SLUG_RE.fullmatch(slug):`<br/>`    raise ValueError("bad slug")` |
| Verbose mode | Document complex patterns. | `DATE_RE = re.compile(r"""`<br/>`^ (?P<year>\\d{4}) - (?P<month>\\d{2}) - (?P<day>\\d{2}) $`<br/>`""", re.VERBOSE)` |
| Normalize before matching | Reduce regex complexity by preprocessing text. | `text = unicodedata.normalize("NFKC", text).strip().lower()`<br/>`match = PATTERN.search(text)` |
| Timeout strategy | Python `re` has no built-in timeout; use safer engines or process isolation for hostile input. | `# For untrusted complex matching, consider the third-party regex module with timeout.` |

## Production text pipelines

| Method | Description | Code example |
|---|---|---|
| PII redaction pipeline | Apply multiple compiled patterns in deterministic order. | `patterns = [(EMAIL_RE, "[EMAIL]"), (PHONE_RE, "[PHONE]")]`<br/>`for pattern, repl in patterns:`<br/>`    text = pattern.sub(repl, text)` |
| Log parser with schema | Convert regex groups to typed records. | `m = LOG_RE.fullmatch(line)`<br/>`record = LogRecord(level=m["level"], latency_ms=int(m["latency"]))` |
| Test regex examples | Lock expected behavior with positive and negative cases. | `assert EMAIL_RE.fullmatch("a@example.com")`<br/>`assert not EMAIL_RE.fullmatch("not an email")` |
| Unicode words | ASCII word classes may fail for multilingual text. | `# For robust multilingual tokenization, prefer NLP tokenizers over simple \\w regex.` |
| Incremental extraction | Stream large files line by line. | `with open("logs.txt") as file:`<br/>`    for line in file:`<br/>`        if match := ERROR_RE.search(line):`<br/>`            yield match.groupdict()` |
| Replacement callback | Use a function for context-aware substitutions. | `def repl(match): return mask(match.group(0))`<br/>`safe = SECRET_RE.sub(repl, text)` |
| Pattern registry | Name patterns so ownership and intent are clear. | `PATTERNS = {"email": EMAIL_RE, "date": DATE_RE, "api_key": API_KEY_RE}` |
| Parser boundary | Escalate to parsers for HTML, JSON, SQL, Python, and nested grammars. | `data = json.loads(raw)`<br/>`soup = BeautifulSoup(html, "html.parser")` |
