---
title: Bash and Linux Master Cheatsheet
sidebar_position: 15
---

# Bash and Linux Master Cheatsheet

## File operations

| Method | Description | Code example |
|---|---|---|
| `pwd` | Prints current working directory. | `pwd` |
| `ls` | Lists files. Use `-lah` for detailed human-readable output. | `ls -lah` |
| `cd` | Changes directories. | `cd ~/projects/learn-ai-ml` |
| `cp` | Copies files or directories. | `cp config.example.yaml config.yaml`<br/>`cp -r data data_backup` |
| `mv` | Moves or renames files. | `mv old_name.md new_name.md` |
| `mkdir` | Creates directories. Use `-p` for nested paths. | `mkdir -p logs/2026/05` |
| `rm` | Removes files. Use carefully, especially with `-r`. | `rm old.log` |

## Inspecting text and finding files

| Method | Description | Code example |
|---|---|---|
| `cat` | Prints whole files. Best for small files. | `cat README.md` |
| `less` | Opens a scrollable file viewer. | `less app.log` |
| `head` and `tail` | Shows first or last lines. | `head -20 app.log`<br/>`tail -50 app.log` |
| `grep` | Searches text patterns. | `grep -n "ERROR" app.log` |
| `rg` | Fast recursive search. Prefer over grep when available. | `rg -n "train_test_split" docs` |
| `find` | Finds files by name, type, or metadata. | `find . -name "*.py" -type f` |
| `wc` | Counts lines, words, and bytes. | `wc -l docs/cheetsheet/*.md` |

## Processes and system resources

| Method | Description | Code example |
|---|---|---|
| `ps` | Lists processes. | `ps aux` |
| `top` | Interactive process monitor. | `top` |
| `kill` | Sends a signal to a process. | `kill 12345`<br/>`kill -9 12345` |
| `jobs` | Lists background jobs in current shell. | `jobs` |
| `nohup` | Runs a command that survives terminal disconnect. | `nohup python train.py > train.log 2>&1 &` |
| `df` | Shows filesystem disk usage. | `df -h` |
| `du` | Shows directory/file size. | `du -sh data models` |

## Environment, networking, and SSH

| Method | Description | Code example |
|---|---|---|
| `export` | Sets environment variables for child processes. | `export CUDA_VISIBLE_DEVICES=0`<br/>`export MODEL_PATH=/models/model.pt` |
| `env` | Prints environment variables. | `env` |
| `which` | Shows executable path. | `which python` |
| `curl` | Sends HTTP requests from the terminal. | `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"hello"}'` |
| `ssh` | Connects to remote machines. | `ssh user@host.example.com` |
| `scp` | Copies files over SSH. | `scp model.pt user@host:/models/model.pt` |
| `rsync` | Efficiently syncs directories. | `rsync -av --progress data/ user@host:/data/` |

## ML and GPU operations

| Method | Description | Code example |
|---|---|---|
| `nvidia-smi` | Shows GPU utilization, memory, driver, and processes. | `nvidia-smi` |
| Watch GPU | Refreshes GPU stats periodically. | `watch -n 1 nvidia-smi` |
| CUDA device selection | Restricts visible GPUs for a process. | `CUDA_VISIBLE_DEVICES=1 python train.py` |
| Virtual env | Creates isolated Python environment. | `python -m venv .venv`<br/>`source .venv/bin/activate` |
| Install requirements | Installs pinned dependencies. | `pip install -r requirements.txt` |
| Run module | Runs Python module by import path. | `python -m pytest tests` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| Safe script header | Makes Bash scripts safer and easier to debug. | `#!/usr/bin/env bash`<br/>`set -euo pipefail` |
| Redirect output | Writes stdout and stderr to a log file. | `python train.py > train.log 2>&1` |
| Background process | Runs long jobs in background. | `python train.py > train.log 2>&1 &`<br/>`echo $!` |
| Create timestamped dir | Useful for experiment outputs. | `run_dir="runs/$(date +%Y%m%d-%H%M%S)"`<br/>`mkdir -p "$run_dir"` |
| Find large files | Locate files consuming disk. | `find . -type f -size +100M -print` |
| Remove caches | Clean Python cache directories. | `find . -type d -name "__pycache__" -prune -exec rm -rf {} +` |
| Check port | See what process owns a port. | `lsof -i :8000` |
| JSON pretty print | Format JSON from a file. | `python -m json.tool response.json` |

## Senior shell scripting

| Method | Description | Code example |
|---|---|---|
| Strict mode with traps | Fail fast and report the failing line. | `set -euo pipefail`<br/>`trap 'echo "failed at line $LINENO" >&2' ERR` |
| Parse flags | Lightweight CLI parsing for scripts. | `while [[ $# -gt 0 ]]; do`<br/>`  case "$1" in`<br/>`    --env) env="$2"; shift 2 ;;`<br/>`    *) echo "unknown arg $1"; exit 1 ;;`<br/>`  esac`<br/>`done` |
| Safe temp dir | Create and clean temporary workspace. | `tmp="$(mktemp -d)"`<br/>`trap 'rm -rf "$tmp"' EXIT` |
| Quoting discipline | Quote variables to preserve spaces and avoid globbing. | `cp "$source_path" "$target_path"` |
| Arrays | Store argument lists safely. | `cmd=(python train.py --epochs 10 --lr 0.001)`<br/>`"${cmd[@]}"` |
| Retry loop | Retry flaky network commands. | `for attempt in {1..5}; do`<br/>`  curl -fsS "$url" && break`<br/>`  sleep "$attempt"`<br/>`done` |
| Lock file | Prevent overlapping scheduled jobs. | `exec 9>/tmp/train.lock`<br/>`flock -n 9 &#124;&#124; exit 0` |
| Parallel jobs | Run independent commands concurrently and wait. | `python job_a.py &`<br/>`python job_b.py &`<br/>`wait` |

## Linux debugging for production ML

| Method | Description | Code example |
|---|---|---|
| Open files | See files and sockets used by a process. | `lsof -p "$PID"` |
| Process tree | Understand parent/child process structure. | `pstree -ap "$PID"` |
| Memory map | Inspect memory usage by mapping. | `pmap -x "$PID" &#124; tail -20` |
| Network listeners | Show listening ports and owning processes. | `ss -ltnp` |
| Disk hot spots | Find largest directories. | `du -xh . &#124; sort -h &#124; tail -20` |
| File descriptors | Diagnose descriptor leaks. | `ls /proc/$PID/fd &#124; wc -l` |
| Strace | Trace syscalls for stuck processes. | `strace -p "$PID" -f -tt` |
| Journal logs | Read systemd service logs. | `journalctl -u ml-api.service -f` |

## Data and ML one-liners

| Method | Description | Code example |
|---|---|---|
| CSV row count | Count rows excluding header. | `tail -n +2 data.csv &#124; wc -l` |
| Sample lines | Randomly sample a text dataset. | `shuf -n 1000 train.jsonl > sample.jsonl` |
| Split file | Split large file by line count. | `split -l 100000 train.jsonl shard_` |
| Validate JSONL | Fail on malformed JSON lines. | `python -c 'import json,sys; [json.loads(line) for line in sys.stdin]' < data.jsonl` |
| Compress artifacts | Compress large outputs. | `tar -czf run-artifacts.tar.gz runs/2026-05-16` |
| Check checksums | Verify downloaded artifacts. | `sha256sum model.safetensors` |
| Monitor training log | Follow selected metrics. | `tail -f train.log &#124; grep --line-buffered "val_loss"` |
| GPU process cleanup | Find GPU processes before stopping them. | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` |
