import ast
import math
import operator
import os
from datetime import datetime
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.usage import UsageLimits

NOTES_FILE = Path(__file__).resolve().with_name("notes.txt")
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
}


def get_current_time() -> str:
    """Return the computer's current local date, time and timezone."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        value = OPERATORS[type(node.op)](
            evaluate_node(node.left), evaluate_node(node.right)
        )
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = evaluate_node(node.operand)
        if isinstance(node.op, ast.USub):
            value = -value
    else:
        raise ValueError("Use numbers, parentheses and +, -, *, / or // only.")
    if not math.isfinite(value) or abs(value) > 1_000_000_000_000:
        raise ValueError("Numbers and results must be finite and within 1e12.")
    return value


def calculate(expression: str) -> str:
    """Calculate a short arithmetic expression with numbers and +, -, *, /, //."""
    try:
        if len(expression) > 120:
            raise ValueError("Use at most 120 characters.")
        tree = ast.parse(expression.strip(), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 40:
            raise ValueError("Use a simpler expression.")
        return str(evaluate_node(tree.body))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as error:
        return f"Calculation error: {error}"


def save_note(note: str) -> str:
    """Append one short note to notes.txt beside this script."""
    note = note.strip()
    if not note or len(note) > 2000:
        return "Use a non-empty note of at most 2000 characters."
    text = note + "\n"
    current_size = NOTES_FILE.stat().st_size if NOTES_FILE.exists() else 0
    if current_size + len(text.encode("utf-8")) > 20_000:
        return "The note file is full. Move old notes before adding more."
    with NOTES_FILE.open("a", encoding="utf-8") as file:
        file.write(text)
    return "Note saved."


def read_notes() -> str:
    """Read notes previously saved in notes.txt beside this script."""
    if not NOTES_FILE.exists():
        return "No notes saved yet."
    if NOTES_FILE.stat().st_size > 20_000:
        return "The note file is too large. Move old notes before reading."
    return NOTES_FILE.read_text(encoding="utf-8") or "No notes saved yet."


def build_agent(model=None) -> Agent:
    if model is None:
        model = OllamaModel(
            os.getenv("OLLAMA_MODEL", "qwen3.5:2b"),
            provider=OllamaProvider(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            ),
        )
    return Agent(
        model,
        tools=[get_current_time, calculate, save_note, read_notes],
        instructions=(
            "You are a helpful personal assistant running locally. "
            "Use tools for the current time, arithmetic and saving or reading notes. "
            "Save notes only when the user asks. Report tool errors accurately. "
            "Keep answers short and friendly."
        ),
    )


def main() -> None:
    agent = build_agent()
    history = []
    print("Local agent ready. Type 'quit' or 'exit' to end.")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                break
            if not user_input:
                continue
            result = agent.run_sync(
                user_input,
                message_history=history,
                usage_limits=UsageLimits(request_limit=8, tool_calls_limit=8),
            )
            history = result.all_messages()
            print(f"Agent: {result.output}\n")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        except Exception as error:
            print(f"Request failed ({type(error).__name__}): {error}")


if __name__ == "__main__":
    main()
