---
id: llme-serving-engines
title: "Serving Engines: vLLM, SGLang, TGI, TensorRT-LLM, llama.cpp, Ollama and Triton"
sidebar_label: "6 · Serving engines"
sidebar_position: 6
slug: /llm-engineering/serving-engines
description: "A feature matrix for seven LLM serving engines checked against each engine's own documentation, an OpenAI-compatible server and benchmark client you can run on a CPU, and a way to choose an engine from requirements instead of reputation."
tags: [serving, vllm, sglang, tgi, tensorrt-llm, llama-cpp, ollama, triton, openai-compatible, benchmarking]
---

import Infographic from '@site/src/components/Infographic';
import ServingEngineChooserLab from '@site/src/components/viz/ServingEngineChooserLab';

**In one line.** A serving engine is the program that turns a model file into an HTTP endpoint with batching, caching and quantisation done for you, and the right one is chosen by writing down your hardware and requirements first, then reading each engine's documentation, then measuring on your own traffic.

:::note Not from a lecture
This chapter is written for this site from the documentation pages and release pages listed under Further reading, all opened on 2 October 2026. The feature matrix is a snapshot of those pages on that date; the engines move monthly, so treat every cell as something to re-check before you commit.
:::

## The idea in plain words

The earlier chapters in this section explained what is slow about decoding ([memory-bound decoding](/docs/llm-engineering/why-decoding-is-memory-bound)), why the KV cache needs careful memory management ([KV cache and paged attention](/docs/llm-engineering/kv-cache-and-paged-attention)) and how requests share a GPU ([continuous batching](/docs/llm-engineering/continuous-batching-and-scheduling)). A **serving engine** packages all of that. You give it model weights and a port; it gives you an endpoint that many clients can hit at once.

Seven names come up in almost every conversation, and they are not all the same kind of thing.

| Engine | What it is, in one line |
| --- | --- |
| **vLLM** | A general-purpose serving engine with an OpenAI-compatible server, many quantisation formats and many hardware targets. |
| **SGLang** | A serving framework built around RadixAttention prefix caching, with structured output and broad hardware support. |
| **TGI** (Text Generation Inference) | Hugging Face's server. Its own documentation now says it is in maintenance mode. |
| **TensorRT-LLM** | NVIDIA's engine, for NVIDIA GPUs only, with its own serving command, `trtllm-serve`. |
| **llama.cpp** (`llama-server`) | A C and C++ engine for GGUF models that runs on laptops, phones and servers alike. |
| **Ollama** | A local model runner with a friendly CLI and an OpenAI-compatible endpoint. |
| **Triton** (now Dynamo-Triton) | A model host that runs other engines as backends; it is not an LLM engine by itself. |

Two families are hiding in that list. **Datacentre engines** (vLLM, SGLang, TGI, TensorRT-LLM) are designed for many concurrent users on GPUs. **Local engines** (llama.cpp, Ollama) are designed for one machine and a few users, with the widest hardware range. Triton sits across both as a host.

<Infographic src="/img/llme/serving-engines-benchmark.svg" alt="A flow from a client to an OpenAI-compatible FastAPI server around SmolLM2, and a table of measured time to first token and throughput at concurrency 1, 2 and 4." caption="The server and benchmark client from the code below, with ranges across five runs: one client waits 0.03 to 0.06 s for its first token, four clients wait 0.47 to 0.75 s on average, and throughput has not moved." />

The picture above is the first lesson of serving, learned on a CPU: a server that handles requests one at a time has **flat throughput and rising waiting time** as concurrency grows. Everything these engines do (continuous batching, paged KV memory) exists to make throughput rise with concurrency instead.

## How it works

### The common surface

Almost every engine here speaks the **OpenAI chat-completions protocol**: a `POST` to `/v1/chat/completions` with a list of messages, optionally with `stream` set so tokens arrive as server-sent events. That is why the same client works against all of them, and why the first code block builds exactly that surface around a small model. Three metrics matter to a client:

- **TTFT** (time to first token): request sent to first token received. It includes queueing and prefill.
- **TPOT** (time per output token): the gap between successive tokens once streaming starts.
- **Throughput**: tokens per second across all clients.

### The feature matrix

Every cell below links to the page that supports it. A **yes** means the page I opened documents the feature. **Not established** means the pages I opened did not settle the question, which is different from "no". **No** is used only where a page states the limit. Opened 2 October 2026.

| Engine | OpenAI API | Structured output | Multi-LoRA | Speculative decoding | Prefix reuse | Multi-GPU serving |
| --- | --- | --- | --- | --- | --- | --- |
| vLLM | [yes](https://docs.vllm.ai/en/latest/serving/online_serving/) | [yes](https://docs.vllm.ai/en/latest/features/structured_outputs/) | [yes](https://docs.vllm.ai/en/latest/features/lora/) | [yes](https://docs.vllm.ai/en/latest/features/speculative_decoding/) | [yes](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | [yes](https://docs.vllm.ai/en/latest/deployment/k8s/) |
| SGLang | [yes](https://docs.sglang.io/) | [yes](https://docs.sglang.io/advanced_features/structured_outputs.html) | [yes](https://docs.sglang.io/advanced_features/lora.html) | [yes](https://docs.sglang.io/advanced_features/speculative_decoding.html) | [yes](https://docs.sglang.io/) | [yes](https://docs.sglang.io/) |
| TGI | [yes](https://huggingface.co/docs/text-generation-inference/messages_api) | [yes](https://huggingface.co/docs/text-generation-inference/conceptual/guidance) | [yes](https://huggingface.co/docs/text-generation-inference/conceptual/lora) | [yes](https://huggingface.co/docs/text-generation-inference/conceptual/speculation) | [yes](https://huggingface.co/docs/text-generation-inference/conceptual/chunking) | [yes](https://huggingface.co/docs/text-generation-inference/index) |
| TensorRT-LLM | [yes](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html) | [yes](https://nvidia.github.io/TensorRT-LLM/features/guided-decoding.html) | [yes](https://nvidia.github.io/TensorRT-LLM/features/lora.html) | [yes](https://nvidia.github.io/TensorRT-LLM/features/speculative-decoding.html) | [yes](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html) | [yes](https://pypi.org/project/tensorrt-llm/) |
| llama.cpp | yes, server README | yes, server README | yes, server README | yes, server README | yes, server README | not established |
| Ollama | [yes](https://docs.ollama.com/api/openai-compatibility) | [yes](https://docs.ollama.com/capabilities/structured-outputs) | not established | not established | not established | not established |
| Triton | not established | depends on backend | depends on backend | depends on backend | depends on backend | depends on backend |

The llama.cpp cells come from the `llama-server` README in the llama.cpp project repository, which I opened directly; it has no other documentation site I could cite, and this site does not link to code hosting pages.

What each column hides, taken from the same pages:

- **Hardware.** vLLM lists NVIDIA CUDA, AMD ROCm, Intel XPU, Apple silicon through vLLM-Metal, and x86 and ARM CPUs ([installation page](https://docs.vllm.ai/en/latest/getting_started/installation/)). SGLang lists NVIDIA, AMD, Intel Xeon, Google TPU and Ascend NPU ([docs home](https://docs.sglang.io/)). TensorRT-LLM lists only NVIDIA architectures, from Ampere (A100) to Blackwell ([supported hardware](https://nvidia.github.io/TensorRT-LLM/supported-hardware.html)). TGI has guides for NVIDIA GPUs, AMD GPUs, Intel Gaudi, AWS Trainium and Inferentia, Google TPUs and Intel GPUs ([installation page](https://huggingface.co/docs/text-generation-inference/installation)); the chooser code below keeps its Intel GPU cell at "not established", which is conservative. Ollama lists NVIDIA, AMD through ROCm, Apple through Metal, and Vulkan for Intel and AMD ([GPU page](https://docs.ollama.com/gpu)). llama.cpp lists Apple silicon, CUDA, HIP, Vulkan and SYCL, plus CPU and GPU hybrid inference for models larger than VRAM (project README).
- **Quantisation.** The pages name very different sets. vLLM documents AutoAWQ, BitsAndBytes, GPTQModel, LLM Compressor variants, TorchAO and a quantised KV cache ([quantisation page](https://docs.vllm.ai/en/latest/features/quantization/)). SGLang's page lists FP8, MXFP4, AWQ, GPTQ, GGUF and more, each with its hardware ([quantisation page](https://docs.sglang.io/advanced_features/quantization.html)). TGI lists GPTQ, AWQ, bitsandbytes, EETQ, Marlin, EXL2 and fp8 ([quantisation page](https://huggingface.co/docs/text-generation-inference/conceptual/quantization)). Ollama says it does not quantise GGUF models during import ([import page](https://docs.ollama.com/import)). Chapter 4 on [quantisation for inference](/docs/llm-engineering/quantisation-for-inference) explains the formats.
- **Status.** TGI's documentation opens with a caution that it is in maintenance mode, accepting only minor bug fixes and documentation work, and points readers to vLLM, SGLang, llama.cpp and MLX ([TGI home](https://huggingface.co/docs/text-generation-inference/index)). NVIDIA's pages show Triton Inference Server as "formerly" the product now called Dynamo-Triton ([Dynamo-Triton page](https://developer.nvidia.com/dynamo-triton)).
- **Versions checked.** vLLM 0.30.0 and SGLang 0.5.21 on the Python package index; TensorRT-LLM 1.2.1 on the package index (released 20 April 2026) while its documentation was built on 26 September 2026, so the docs may describe newer behaviour than that package.

Some cells deserve a caveat the matrix cannot hold.

| Cell | Caveat from the page itself |
| --- | --- |
| Ollama OpenAI API | The page lists what is not supported: log probabilities, `tool_choice`, `logit_bias` and `n`. |
| llama.cpp multi-LoRA | A request can carry its own adapter list, but requests with different LoRA configurations are not batched together. |
| TensorRT-LLM speculative decoding | The page lists MTP, Eagle3, NGram, DraftTarget, PARD, DFlash and SA, and notes that with `trtllm-serve` and `trtllm-bench` the PyTorch backend supports only Eagle3. It does not say which backend supports the others. |
| vLLM prefix reuse | The prefix caching page says to set `enable_prefix_caching=True` to enable it. Other material says newer versions switch it on by default. The page I opened does not say so, so I make no claim. |
| Triton | Its page lists backends (TensorRT, PyTorch, ONNX Runtime, OpenVINO, Python, vLLM, TensorRT-LLM and others); LLM features come from the backend you pick. |

<Infographic src="/img/llme/serving-engines-matrix.svg" alt="The feature matrix as a grid of ticks, crosses and question marks for seven engines, with the rule for how requirements filter it." caption="The matrix as the chooser sees it. A question mark is an honest gap in what I could verify, so the chooser never turns it into a yes." />

### Structured output and multi-LoRA, concretely

**Structured output** forces the model to produce text that matches a schema by masking tokens that the grammar forbids. vLLM and SGLang both accept a JSON schema, regular expression or EBNF grammar, with xgrammar as a backend; TensorRT-LLM names xgrammar and llguidance; TGI calls the feature Guidance; Ollama takes a JSON schema in a `format` field. Every engine implements the mask differently, so the same schema can cost different latency. Test it on your schema.

**Multi-LoRA serving** keeps one copy of the base weights and swaps small adapters per request. vLLM takes `--enable-lora` and `--lora-modules name=path`, with a runtime load endpoint when an environment variable allows it; SGLang takes `--enable-lora` and `--lora-paths`; TGI reads a `LORA_ADAPTERS` environment variable and a request names its `adapter_id`. The idea is explained in the LoRA chapters of the deep-learning theory section; the serving consequence is that fifty fine-tunes cost far less than fifty deployments.

## A real system that works this way

TGI's own documentation is the cleanest example of how fast this field moves. Its home page describes continuous batching, tensor parallelism, quantisation and guidance, then opens with a caution box saying that the project is in maintenance mode and that the authors recommend vLLM and SGLang, as well as local engines such as llama.cpp, going forward. A project that was the default answer in 2023 is now pointing its readers elsewhere, and its stated reason is that optimised engines now build on `transformers` model architectures. The practical lesson: **a serving engine is a dependency with a lifetime**. Record which engine and version you deploy, and keep the OpenAI-compatible interface as the boundary so that swapping it is a configuration change, not a rewrite.

A second small point from the same documentation: the TGI v3 write-up reports its benchmark from the second run, because the first run has a cold prefix cache. If you benchmark an engine with prefix caching on, decide deliberately whether you are measuring the cold or the warm case.

## Code you can run

The first block is a complete OpenAI-compatible server around `HuggingFaceTB/SmolLM2-135M-Instruct`, plus a benchmark client that measures TTFT, TPOT and throughput at concurrency 1, 2 and 4. It runs on a CPU. The server streams one chunk per token, and a lock allows one generation at a time, which is what makes it a good model of a server without batching.

```python
import json
import queue
import threading
import time

import httpx
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
torch.manual_seed(0)
torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
gpu_lock = threading.Lock()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    max_tokens: int = 32
    stream: bool = False


def chunk(delta, finish=None):
    body = {
        "object": "chat.completion.chunk",
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return "data: " + json.dumps(body) + "\n\n"


class TokenStreamer(BaseStreamer):
    def __init__(self):
        self.q = queue.Queue()
        self.skipped_prompt = False

    def put(self, value):
        if not self.skipped_prompt:
            self.skipped_prompt = True
            return
        for token_id in value.tolist():
            self.q.put(token_id)

    def end(self):
        self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            yield item


def generate_stream(prompt_ids, max_tokens):
    streamer = TokenStreamer()

    def work():
        with gpu_lock:
            model.generate(
                prompt_ids,
                max_new_tokens=max_tokens,
                min_new_tokens=max_tokens,
                do_sample=False,
                streamer=streamer,
                pad_token_id=tok.eos_token_id,
            )

    threading.Thread(target=work, daemon=True).start()
    yield chunk({"role": "assistant"})
    for token_id in streamer:
        yield chunk({"content": tok.decode([token_id])})
    yield chunk({}, "length")
    yield "data: [DONE]\n\n"


app = FastAPI()


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    text = tok.apply_chat_template(
        [m.model_dump() for m in req.messages], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tok(text, return_tensors="pt").input_ids
    if req.stream:
        return StreamingResponse(generate_stream(prompt_ids, req.max_tokens), media_type="text/event-stream")
    with gpu_lock:
        out = model.generate(
            prompt_ids, max_new_tokens=req.max_tokens, min_new_tokens=req.max_tokens,
            do_sample=False, pad_token_id=tok.eos_token_id,
        )
    new_ids = out[0, prompt_ids.shape[1]:]
    return JSONResponse({
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": tok.decode(new_ids, skip_special_tokens=True)},
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": prompt_ids.shape[1], "completion_tokens": len(new_ids),
                  "total_tokens": prompt_ids.shape[1] + len(new_ids)},
    })


def one_request(client, max_tokens):
    payload = {"messages": [{"role": "user", "content": "Explain why the sky is blue."}],
               "max_tokens": max_tokens, "stream": True}
    start = time.perf_counter()
    stamps = []
    with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                delta = json.loads(line[6:])["choices"][0]["delta"]
                if "content" in delta:
                    stamps.append(time.perf_counter())
    return start, stamps


def benchmark(client, concurrency, max_tokens=24):
    results = [None] * concurrency
    t0 = time.perf_counter()

    def run(i):
        results[i] = one_request(client, max_tokens)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    ttft = [r[1][0] - r[0] for r in results]
    tpot = [(r[1][-1] - r[1][0]) / (len(r[1]) - 1) for r in results]
    tokens = sum(len(r[1]) for r in results)
    return sum(ttft) / len(ttft), max(ttft), sum(tpot) / len(tpot), tokens / wall


if __name__ == "__main__":
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    client = httpx.Client(base_url="http://127.0.0.1:8765", timeout=120)
    models = client.get("/v1/models").json()
    print("models:", [m["id"] for m in models["data"]])
    reply = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8}).json()
    print("usage:", reply["usage"], "finish:", reply["choices"][0]["finish_reason"])
    benchmark(client, 1, 4)
    print("concurrency  mean TTFT s  max TTFT s  mean TPOT ms  tokens/s")
    for c in (1, 2, 4):
        mean_ttft, max_ttft, tpot, tput = benchmark(client, c)
        print(f"{c:>11}  {mean_ttft:>11.2f}  {max_ttft:>10.2f}  {tpot * 1000:>12.1f}  {tput:>8.1f}")
    server.should_exit = True
```

Read the printed table the way a serving engineer would. I ran this block five times on one laptop, and the machine's speed moved by up to a factor of 1.7 between runs, so quote ranges, not a single figure. Across the five runs the mean TTFT with one client was 0.03 to 0.06 s, with two clients 0.18 to 0.32 s and with four clients 0.47 to 0.75 s; the worst of the four clients waited 0.89 to 1.5 s. That is roughly 12 to 18 times the single-client TTFT. TPOT, the gap between tokens once a request is streaming, stayed between 10.8 and 19.7 ms and did not rise with concurrency within any run. Throughput was between 48 and 86 tokens per second across runs, and within each run the four-client figure was within about 11 per cent of the one-client figure. Nothing is slower per token; requests are simply **queueing behind the lock**. Trust the shape: TTFT grows with the queue, TPOT does not, throughput is flat. A batching engine changes the third fact.

The second block turns the matrix into data and runs the same filter the lab uses. It never ranks; it sorts engines into **fits**, **unverified** and **out**.

```python
Y, N, U, V = "yes", "no", "not established", "depends on backend"

ENGINES = {
    "vLLM": {
        "openai": Y, "structured": Y, "multilora": Y, "spec": Y, "prefix": Y, "tp": Y,
        "hardware": {"nvidia": Y, "amd": Y, "apple": Y, "cpu": Y, "intel_gpu": Y},
        "maintenance": False,
    },
    "SGLang": {
        "openai": Y, "structured": Y, "multilora": Y, "spec": Y, "prefix": Y, "tp": Y,
        "hardware": {"nvidia": Y, "amd": Y, "apple": U, "cpu": Y, "intel_gpu": U},
        "maintenance": False,
    },
    "TGI": {
        "openai": Y, "structured": Y, "multilora": Y, "spec": Y, "prefix": Y, "tp": Y,
        "hardware": {"nvidia": Y, "amd": Y, "apple": U, "cpu": U, "intel_gpu": U},
        "maintenance": True,
    },
    "TensorRT-LLM": {
        "openai": Y, "structured": Y, "multilora": Y, "spec": Y, "prefix": Y, "tp": Y,
        "hardware": {"nvidia": Y, "amd": N, "apple": N, "cpu": N, "intel_gpu": N},
        "maintenance": False,
    },
    "llama.cpp": {
        "openai": Y, "structured": Y, "multilora": Y, "spec": Y, "prefix": Y, "tp": U,
        "hardware": {"nvidia": Y, "amd": Y, "apple": Y, "cpu": Y, "intel_gpu": Y},
        "maintenance": False,
    },
    "Ollama": {
        "openai": Y, "structured": Y, "multilora": U, "spec": U, "prefix": U, "tp": U,
        "hardware": {"nvidia": Y, "amd": Y, "apple": Y, "cpu": U, "intel_gpu": Y},
        "maintenance": False,
    },
    "Triton": {
        "openai": U, "structured": V, "multilora": V, "spec": V, "prefix": V, "tp": V,
        "hardware": {"nvidia": U, "amd": U, "apple": U, "cpu": Y, "intel_gpu": U},
        "maintenance": False,
    },
}


def choose(hardware, needs, avoid_maintenance):
    fits, open_, out = [], [], []
    for name, row in ENGINES.items():
        cells = [row["hardware"][hardware]] + [row[n] for n in needs]
        if avoid_maintenance and row["maintenance"]:
            out.append((name, "maintenance mode"))
        elif N in cells:
            out.append((name, "a requirement is documented as unsupported"))
        elif all(c == Y for c in cells):
            fits.append(name)
        else:
            open_.append(name)
    return fits, open_, out


SCENARIOS = {
    "laptop": ("apple", ["openai", "structured"], False),
    "nvidia fleet": ("nvidia", ["openai", "structured", "multilora", "spec", "prefix", "tp"], True),
    "nvidia fleet, TGI allowed": ("nvidia", ["openai", "structured", "multilora", "spec", "prefix", "tp"], False),
    "amd cluster": ("amd", ["openai", "multilora", "tp"], False),
    "cpu only": ("cpu", ["openai", "structured"], False),
}

for label, (hw, needs, avoid) in SCENARIOS.items():
    fits, open_, out = choose(hw, needs, avoid)
    print(f"{label}: fits {len(fits)} {fits} | unverified {len(open_)} {open_} | out {len(out)}")
```

The `nvidia fleet` line prints `fits 3 ['vLLM', 'SGLang', 'TensorRT-LLM'] | unverified 3 ['llama.cpp', 'Ollama', 'Triton'] | out 1`: the lab's default. With TGI allowed, four engines fit. On the laptop scenario three fit (vLLM through vLLM-Metal, llama.cpp and Ollama). On the CPU-only scenario three fit as well (vLLM, SGLang, llama.cpp).

<ServingEngineChooserLab />

Notice what the filter cannot tell you. On the NVIDIA fleet three engines pass every documented requirement, and the filter has nothing to say about which one is fastest, easiest to operate, or best for your model. That comes from the next step: pick two, serve your own model with your own prompts, and run a benchmark client like the one above against each.

## Production snippets (not run here)

None of these were run, because they need a GPU, a container runtime or a cluster. The flags and manifest elements come from the pages cited above.

:::warning Not run in this environment
vLLM's container, with the OpenAI-compatible server on port 8000. The image name, flags and port mapping follow vLLM's Docker deployment page; substitute your own model.
:::

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-0.6B
```

:::warning Not run in this environment
A Kubernetes Deployment and Service assembled from the elements vLLM's Kubernetes page shows: one GPU requested through `nvidia.com/gpu`, a readiness probe on `/health` at port 8000 with a 60 second initial delay and 5 second period, and a memory-backed volume mounted at `/dev/shm`.
:::

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-server
  template:
    metadata:
      labels:
        app: llm-server
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args: ["--model", "Qwen/Qwen3-0.6B"]
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "1"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 5
          volumeMounts:
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 2Gi
---
apiVersion: v1
kind: Service
metadata:
  name: llm-server
spec:
  selector:
    app: llm-server
  ports:
    - port: 80
      targetPort: 8000
```

:::warning Not run in this environment
Multi-LoRA in vLLM and SGLang, and a structured-output request through the OpenAI client. Flag names follow the vLLM and SGLang LoRA pages; the `response_format` request shape follows the SGLang structured-output page, and vLLM's structured-output page accepts the same family of constraints.
:::

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-lora \
    --lora-modules sql-lora=/adapters/sql support-lora=/adapters/support --max-loras 4

python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora --lora-paths sql=/adapters/sql support=/adapters/support
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
schema = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["city", "population"],
}
reply = client.chat.completions.create(
    model="sql-lora",
    messages=[{"role": "user", "content": "Largest city in Japan, as JSON."}],
    response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": schema}},
)
print(reply.choices[0].message.content)
```

:::warning Not run in this environment
TensorRT-LLM's server with guided decoding switched on through a YAML file, and llama.cpp's server with a draft model for speculative decoding and a LoRA adapter. Both follow the flags named on the pages cited under Further reading.
:::

```yaml
guided_decoding_backend: xgrammar
```

```bash
trtllm-serve nvidia/Llama-3.1-8B-Instruct-FP8 --config config.yaml

llama-server -m model-q4_k_m.gguf -md draft-q4_k_m.gguf --lora adapter.gguf --port 8080
```

## Designing with it

1. **Write the requirements first**: hardware, the OpenAI features you call, whether you need structured output, adapters, speculative decoding, and the traffic shape. Run the filter. Distrust any engine that wins before you have written the list.
2. **Treat every question mark as homework.** A cell the documentation does not settle is a thing to test, not assume. Ollama's multi-LoRA and Triton's LLM features are examples in the matrix above.
3. **Keep the OpenAI-compatible endpoint as the contract.** Clients, benchmark scripts and gateways written against it survive an engine swap.
4. **Benchmark your own prompts, warm and cold.** Report TTFT, TPOT and throughput at several concurrency levels, as the benchmark above does, and say whether prefix caches were warm.
5. **Pin versions and record the date.** Features land and disappear between releases; the matrix in this chapter will be stale within months.
6. **Do not read a datacentre engine's feature list as a local engine's.** llama.cpp and Ollama win on hardware range and simplicity, not on fleet throughput.

## Where this stands in 2026

:::info Industry view
The surface has converged. Four datacentre engines document an OpenAI-compatible server, structured output, multi-LoRA, speculative decoding and prefix reuse, and they now share building blocks such as xgrammar for structured output. So choosing is increasingly about hardware coverage (TensorRT-LLM is NVIDIA only; vLLM and SGLang span several vendors), operational maturity, how quickly a new model architecture is supported, and measured throughput on your workload. The loudest sign of consolidation is TGI's own maintenance-mode notice. Benchmarks published by an engine's authors, including the TGI v3 comparison against vLLM, are useful for method but are not independent evidence.
:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why did throughput stay flat in the benchmark while TTFT grew with concurrency?</summary>

The server holds a lock so only one generation runs at a time; TPOT per request is unchanged, but later requests wait their turn, which shows up as TTFT. A continuous-batching engine instead runs several requests in the same decode step, so throughput rises with concurrency.

</details>

<details>
<summary><strong>Q2.</strong> A cell in the matrix says "not established". Is the feature missing?</summary>

No. It means the pages I opened did not settle the question. Check the engine's current documentation and test it. Only a page that states a limit justifies a "no", as with TensorRT-LLM's hardware list naming only NVIDIA architectures.

</details>

<details>
<summary><strong>Q3.</strong> Your team wants multi-LoRA on AMD GPUs with an OpenAI-compatible API. Which engines in the matrix fit?</summary>

vLLM, SGLang and TGI fit the three requirements (the code's `amd cluster` line also requires multi-GPU serving, which all three document). TensorRT-LLM is out because its hardware page lists only NVIDIA architectures. llama.cpp, Ollama and Triton remain unverified for at least one requirement.

</details>

<details>
<summary><strong>Q4.</strong> What does a model host like Triton add that an LLM engine does not?</summary>

It hosts many kinds of models behind one server, with backends for TensorRT, PyTorch, ONNX Runtime, Python and others, and it can run vLLM or TensorRT-LLM as a backend. It adds a uniform deployment layer, not LLM-specific scheduling of its own.

</details>

<details>
<summary><strong>Q5.</strong> Why is it risky to rely on one engine's own benchmark against a rival?</summary>

The author chooses the models, hardware, prompt lengths and whether prefix caches are warm. TGI's v3 write-up, for example, reports the second run with prefix caching on. Run both engines on your own traffic with the same client.

</details>

<details>
<summary><strong>Q6.</strong> In the Kubernetes manifest, why mount a memory-backed volume at `/dev/shm`?</summary>

vLLM's Kubernetes page includes it so tensor-parallel inference has shared memory between processes; the container default is small. The Docker page makes the same point with `--ipc=host`.

</details>

## Further reading

All opened on 2 October 2026.

- vLLM documentation: [online serving](https://docs.vllm.ai/en/latest/serving/online_serving/), [structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/), [LoRA](https://docs.vllm.ai/en/latest/features/lora/), [speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/), [quantisation](https://docs.vllm.ai/en/latest/features/quantization/), [automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/), [installation](https://docs.vllm.ai/en/latest/getting_started/installation/), [Docker](https://docs.vllm.ai/en/latest/deployment/docker/), [Kubernetes](https://docs.vllm.ai/en/latest/deployment/k8s/); package version on the [Python package index](https://pypi.org/project/vllm/).
- SGLang documentation: [home](https://docs.sglang.io/), [structured outputs](https://docs.sglang.io/advanced_features/structured_outputs.html), [LoRA](https://docs.sglang.io/advanced_features/lora.html), [quantisation](https://docs.sglang.io/advanced_features/quantization.html), [speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html); [package page](https://pypi.org/project/sglang/).
- Text Generation Inference: [home and status notice](https://huggingface.co/docs/text-generation-inference/index), [Messages API](https://huggingface.co/docs/text-generation-inference/messages_api), [guidance](https://huggingface.co/docs/text-generation-inference/conceptual/guidance), [LoRA](https://huggingface.co/docs/text-generation-inference/conceptual/lora), [speculation](https://huggingface.co/docs/text-generation-inference/conceptual/speculation), [v3 overview](https://huggingface.co/docs/text-generation-inference/conceptual/chunking), [quantisation](https://huggingface.co/docs/text-generation-inference/conceptual/quantization), [installation](https://huggingface.co/docs/text-generation-inference/installation).
- TensorRT-LLM: [documentation home](https://nvidia.github.io/TensorRT-LLM/), [supported hardware](https://nvidia.github.io/TensorRT-LLM/supported-hardware.html), [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html), [guided decoding](https://nvidia.github.io/TensorRT-LLM/features/guided-decoding.html), [LoRA](https://nvidia.github.io/TensorRT-LLM/features/lora.html), [speculative decoding](https://nvidia.github.io/TensorRT-LLM/features/speculative-decoding.html), [KV cache reuse](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html), [in-flight batching](https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html); [package page](https://pypi.org/project/tensorrt-llm/).
- Ollama: [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility), [structured outputs](https://docs.ollama.com/capabilities/structured-outputs), [GPU support](https://docs.ollama.com/gpu), [import](https://docs.ollama.com/import).
- NVIDIA Dynamo-Triton: [product page](https://developer.nvidia.com/dynamo-triton) and the [Triton backend guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/backend/README.html).
- llama.cpp: the `llama-server` README and the project README in its repository, read directly.

## Check yourself

- I can explain why a server that runs one request at a time has flat throughput and rising TTFT.
- I can measure TTFT, TPOT and throughput against any OpenAI-compatible endpoint.
- I can name which of the seven engines are datacentre engines, which are local engines and which is a host.
- I can read the feature matrix and say what "not established" means and does not mean.
- I can turn hardware and requirements into a short list of engines, and say what the list cannot tell me.
- I can explain why an engine's maintenance-mode notice should change a deployment plan.
