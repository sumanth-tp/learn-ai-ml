import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

import agent


def reply(messages, info):
    names = {tool.name for tool in info.function_tools}
    assert names == {"get_current_time", "calculate", "save_note", "read_notes"}
    returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
    if returns:
        return ModelResponse(parts=[TextPart(str(returns[0].content))])
    prompt = messages[-1].parts[-1].content
    name, arguments = {
        "math": ("calculate", {"expression": "23 * 7 + 1"}),
        "save": ("save_note", {"note": "hello world, Tim"}),
        "read": ("read_notes", {}),
        "clock": ("get_current_time", {}),
    }[prompt]
    return ModelResponse(parts=[ToolCallPart(name, arguments, tool_call_id=name)])


def main():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "notes.txt"
        with patch.object(agent, "NOTES_FILE", path):
            assert agent.calculate("23 * 7 + 1") == "162"
            assert agent.calculate("(10 + 2) / 3") == "4.0"
            assert agent.calculate("-5 + 2") == "-3"
            for expression in ["__import__('os')", "2 ** 1000000", "1/0", "1e309", "True", "1+" * 100]:
                assert agent.calculate(expression).startswith("Calculation error:")
            assert agent.read_notes() == "No notes saved yet."
            assert agent.save_note(" ").startswith("Use a non-empty")
            assert agent.save_note("x" * 2001).startswith("Use a non-empty")
            assert not path.exists()
            local_agent = agent.build_agent(FunctionModel(reply))
            history = []
            outputs = {}
            for prompt in ["math", "save", "read", "clock"]:
                result = local_agent.run_sync(prompt, message_history=history)
                assert len(result.all_messages()) > len(history)
                history = result.all_messages()
                outputs[prompt] = result.output
            assert outputs["math"] == "162"
            assert outputs["save"] == "Note saved."
            assert outputs["read"] == "hello world, Tim\n"
            fresh_agent = agent.build_agent(FunctionModel(reply))
            assert fresh_agent.run_sync("read").output == "hello world, Tim\n"
            assert path.read_text() == "hello world, Tim\n"
            print(json.dumps({"arithmetic": outputs["math"], "saved_note": outputs["read"].strip(), "messages_after_four_runs": len(history), "fresh_agent_reads_file": True}))
    print("Offline checks passed. No Ollama service or model is used.")


if __name__ == "__main__":
    main()
