# Local Python AI agent

A complete local terminal assistant with four tools: local time, arithmetic,
save a note and read notes. Python 3.10 or newer; Python 3.11+ recommended.
This implementation accompanies the second Daily chapter at /docs/daily/local-ai-agent.
The lesson follows Tech With Tim's video, https://www.youtube.com/watch?v=ByWCsa8DbF8.
The publisher's downloadable original requires community membership; this is an
independent implementation of the workflow visible in the video.

## Setup

1. Install Ollama from https://ollama.com/download and start it.
2. Run `ollama pull qwen3.5:2b`, then `ollama list`.
3. From this folder, create an environment with `python3 -m venv .venv`.
4. macOS/Linux: `source .venv/bin/activate`. Windows PowerShell:
   `.\.venv\Scripts\Activate.ps1` (create it with `py -m venv .venv`).
5. Run `python -m pip install -r requirements.txt`.
6. Run `python agent.py`.

Ollama normally serves at http://localhost:11434/v1. If its service is not running,
start `ollama serve` in a separate terminal and keep it open. If it is already
running, use the existing service instead of starting a second one on the same port.

## Try it

- What is 23 * 7 + 1? Please use the calculator.
- Save a note that says hello world, Tim.
- What is currently in my note file?
- What is today's date and time?
- quit

Notes are appended to notes.txt beside agent.py. That file survives restarts;
conversation history does not. Only run the save tool on notes you want to write.
The calculator parses arithmetic without executing Python source code. Numbers
and results are limited to 1e12, expression length to 120 characters and AST size
to 40 nodes. Notes are limited to 2,000 characters and the file to 20,000 bytes.

## Configuration

macOS/Linux:

```bash
export OLLAMA_MODEL=qwen3:14b
export OLLAMA_BASE_URL=http://localhost:11434/v1
python agent.py
```

PowerShell:

```powershell
$env:OLLAMA_MODEL = "qwen3:14b"
$env:OLLAMA_BASE_URL = "http://localhost:11434/v1"
python agent.py
```

Pick an installed, tool-capable model. The defaults need no cloud API key.
Local inference applies when using a downloaded local model and the loopback URL;
changing the URL to a remote service changes where the data is processed.

## Offline checks

```bash
python check_offline.py
```

This uses real Pydantic AI orchestration with a deterministic FunctionModel,
not a language model. It checks arithmetic, rejected expressions, file behaviour,
tool execution, history and reading saved notes from a new agent instance. It
cannot tell you whether your chosen model selects the right tools.
