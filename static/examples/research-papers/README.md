# Research paper teaching implementations

Each Python file is self-contained. These are small educational implementations,
not released author checkpoints or reproductions of published benchmark results.
The corresponding site chapters explain the methods, code and substitutions.

Tested: Python 3.12, PyTorch 2.14.0, CPU. No model downloads are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python attention.py
```

Run another file by replacing attention.py with its filename. Training programs
write checkpoints in the current directory. DDPM writes ddpm-samples/*.pgm.
ReAct writes react-trace.json. Use a separate working directory if you want to
retain outputs from previous runs.

ReAct defaults to a deterministic runner test. For model-driven decisions, set
REACT_ENDPOINT to a full compatible chat-completions URL, REACT_MODEL to your
model name, and REACT_API_KEY if the endpoint requires a key. This optional mode
makes network calls and may incur your provider's normal costs.
