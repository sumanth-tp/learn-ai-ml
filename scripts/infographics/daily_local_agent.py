import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board

OUT = Path(__file__).resolve().parents[2] / "static/img/daily/local-python-agent"


def architecture():
    b = Board(1100, 640, "A local agent: model + tools + conversation", "The model chooses an action; Python performs it and sends back the result")
    b.group(20, 100, 1060, 500, "Your computer", "blue")
    prompt = b.card(40, 165, 215, 110, "Terminal", ["You: Save hello world", "Agent: Note saved"], "blue", size=13)
    agent = b.card(305, 165, 220, 110, "Pydantic AI", ["instructions", "tool schemas + history", "tool execution loop"], "purple", size=13)
    model = b.card(585, 165, 450, 110, "Ollama + downloaded model", ["localhost:11434/v1", "model: qwen3.5:2b", "chooses save_note(note=...)"], "orange", size=13)
    tools = b.card(305, 370, 300, 155, "Python tools", ["get_current_time()", "calculate(expression)", "save_note(note)", "read_notes()"], "green", size=14)
    disk = b.card(705, 370, 330, 155, "notes.txt", ["hello world, Tim", "saved beside agent.py", "survives a process restart"], "teal", size=14)
    history = b.card(40, 370, 215, 155, "History in RAM", ["user + model messages", "tool calls + results", "reset on restart"], "grey", size=13)
    b.arrow(prompt.right(), agent.left(), label="input")
    b.arrow(agent.right(), model.left(), label="request")
    b.arrow(model.bottom(), agent.bottom(), via=[(810, 315), (415, 315)], label="tool call / answer")
    b.arrow(agent.bottom(), tools.top(), label="execute")
    b.arrow(tools.right(), disk.left(), label="append / read")
    b.arrow(history.top(), agent.left(0.8), via=[(145, 335), (280, 335), (280, 253)], dashed=True)
    return b


def note_round_trip():
    b = Board(1100, 600, "Save a note, then read it back", "A confirmed tool return connects the model's answer to a real file change")
    cards=[]
    stages=[
        ("1. User request", ["Save a note that says", "hello world, Tim"], "blue"),
        ("2. Model tool call", ["save_note", "note = hello world, Tim"], "orange"),
        ("3. Python executes", ["append UTF-8 text", "to notes.txt"], "green"),
        ("4. Tool returns", ["Note saved.", "added to message history"], "teal"),
    ]
    for i,(title,lines,color) in enumerate(stages):
        cards.append(b.card(25+i*275,115,240,125,title,lines,color,size=13))
    for left,right in zip(cards,cards[1:]):b.arrow(left.right(),right.left())
    b.card(25,290,1050,70,"5. Agent answers using the tool result",["The application has actually written the file before confirming the save."],"purple",size=14)
    b.arrow(cards[-1].bottom(),(965,290))
    b.card(25,405,510,140,"Next question: what is in my note file?",["read_notes() -> hello world, Tim", "A new process can read the same saved file."],"teal",size=14)
    b.card(565,405,510,140,"A separate example: calculate",["23 * 7 + 1 -> 161 + 1 -> 162", "The Python calculator supplies the number."],"yellow",size=14)
    return b


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture().save(OUT / "architecture.svg")
    note_round_trip().save(OUT / "note-round-trip.svg")
