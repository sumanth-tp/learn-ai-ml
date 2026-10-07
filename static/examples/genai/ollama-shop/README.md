# Check the Ollama shop example

This is an added validation experiment for the CampusX Ollama masterclass (video YcAYmIFtA0o). It keeps the video's inventory, tool schemas and 30%-capped loyalty rule, then adds strict argument checks and a bounded tool loop.

Install the Ollama Python SDK, start your local Ollama service, and run `python verify_shop.py`. The default model is `llama3.1:latest`, which was already downloaded on the machine used for the check. Change `model` to an installed model with tool support if needed.

Tested on 7 October 2026 with Ollama server 0.30.8 and SDK 0.6.3. The live run returned string arguments for the discount, then printed a retry as ordinary text without a structured tool call. Only inventory executed. The independent function checks returned 900.0 for five years and 840.0 for ten years.

Read the execution list before trusting the generated answer. This example demonstrates validation and observable failures, rather than guaranteeing that a model completes the workflow. It does not download or delete models or use a cloud account.
