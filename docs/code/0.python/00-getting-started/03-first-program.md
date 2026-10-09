---
id: py-first-program
title: "Your First Python Program: Syntax, Variables and Errors"
sidebar_label: "First program & syntax"
sidebar_position: 3
slug: /code/python/first-program
description: "Learn assignment, names, comments, indentation and tracebacks before moving into Python's types and control flow."
tags: [python, beginner, syntax, variables, debugging]
---

> **Video:** [Python for AI - Full Beginner Course](https://www.youtube.com/watch?v=ygXn5nV5qFc), covering programming basics, syntax and variables.

A program stores information, calculates results and chooses which instructions to run.

## Write a small calculation

Save this as `first_program.py` and run `python first_program.py` from your activated environment:

```python
name = "Maya"
pages = 12
minutes_per_page = 2
total_minutes = pages * minutes_per_page

# Leave time to review the notes after reading.
review_minutes = 5
total_minutes += review_minutes

print(name)
print(total_minutes)
```

The output is `Maya`, followed by `29` on a new line. `=` assigns the object on the right to the name on the left. `pages * minutes_per_page` is an expression that computes a value. `print(...)` calls a function that displays a value.

Python executes these statements in order. Change `pages` to `20` and run the file again: the total becomes `45`. Assigning a new value to `pages` later does not automatically recalculate `total_minutes`; execute the calculation again.

## Names and assignment

```python
first_name = "Maya"
First_name = "Different name"
pages = 12
pages = 20

print(first_name)  # Maya
print(First_name)  # Different name
print(pages)      # 20
print(pages == 20)  # True: comparison, not assignment
```

Names are case-sensitive. Use descriptive `snake_case` names for variables and functions. A name cannot begin with a digit, contain a space or hyphen, or be a reserved keyword such as `class`. Avoid overwriting built-ins with names such as `list`, `str` or `print`.

`"20"` is text; `20` is an integer. Python allows rebinding a name to a different type, but the object's type determines which operations make sense. Inspect it with `type(pages)` when uncertain.

## Comments, strings and docstrings

`#` starts a comment through the end of the line. Use it to explain a decision or an assumption that the code alone does not make clear.

```python
# Each page is short enough to read in two minutes.
minutes_per_page = 2

instructions = """Read the page.
Write one question.
Check the answer."""
print(instructions)
```

:::note Clarification to the video

Triple quotes create a string, not a special comment syntax. A string at the beginning of a module, function or class can serve as its **docstring**, available through tools such as `help()`. Use `#` for ordinary comments. An unassigned string is still a string expression.

:::

## Indentation is part of the program

The colon introduces a block. The indentation tells Python which statements belong to it:

```python
pages = 3

if pages > 0:
    print("Start reading")
    print("Take notes")

print("Plan saved")
```

The first two prints run only when the condition is true. The last print is outside the block and runs in either case. Use four spaces per indentation level and avoid mixing tabs with spaces.

Syntax rules determine whether a program is valid. Style rules make valid programs easier to read. [PEP 8](https://peps.python.org/pep-0008/) describes Python's style conventions; projects can choose formatting settings such as line length. The [Ruff workflow](../06-engineering/00-project-workflow.md#format-lint-and-sort-imports) automates much of the formatting.

## Read an error before changing the code

These deliberately broken examples illustrate different failures. Run one at a time:

```python
# Broken example: remove the leading comment marker to try it.
# print("Hello)
```

This causes a `SyntaxError`: the quoted string was never closed. The caret points near the parser's problem; also inspect the preceding line when a quote or bracket is missing.

```python
# Broken example:
# print(total_pages)
```

This causes a `NameError` if `total_pages` has not been assigned in the running process. A variable left over in a notebook can hide this bug until you restart the kernel.

```python
# Broken example:
# print("Pages: " + 3)
```

This causes a `TypeError`: string concatenation requires strings. A working version is `print("Pages:", 3)` or `print(f"Pages: {3}")`.

When reading a traceback:

1. Start with the exception type and message at the bottom.
2. Find the last frame that points to your own file and inspect that line.
3. Check the values and types used on that line.
4. Fix the cause and rerun from a clean start.

An error-free program can still calculate the wrong result. For example, `pages + minutes_per_page` is valid Python but does not calculate the reading time. Compare against a small answer you can work out by hand.

## Added practice: predict, break, repair

For the first example, predict the result for zero pages before running it. Decide whether five minutes of review should still be included; the answer depends on your program's requirements.

Then make each of these mistakes and explain the resulting error:

- Change one use of `pages` to `Pages`.
- Change `pages = 12` to `pages = "12"` and inspect the type and result of multiplying it by `2`.
- Remove one closing parenthesis.
- Remove the indentation from the body of an `if` statement.

Continue with [values and containers](../01-core-language/01-types-and-data-structures.md#start-here-values-expressions-and-containers), [control flow](../01-core-language/02-control-flow-and-comprehensions.md#start-here-choose-a-branch-then-repeat-an-action), and [functions](../01-core-language/03-functions-and-scope.md#start-here-input-work-result).

- [ ] I can distinguish assignment from comparison.
- [ ] I can explain which statements belong to an indented block.
- [ ] I can distinguish a comment from a string and a docstring.
- [ ] I can use a traceback and a hand-calculated result to find different kinds of bug.
