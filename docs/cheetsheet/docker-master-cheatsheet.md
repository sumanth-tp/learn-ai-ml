---
title: Docker Master Cheatsheet
sidebar_position: 13
---

# Docker Master Cheatsheet

## Images and containers

| Method | Description | Code example |
|---|---|---|
| `docker build` | `docker build -t name:tag path` builds an image from a Dockerfile. | `docker build -t ml-api:latest .` |
| `docker run` | `docker run [options] image command` starts a container. | `docker run --rm -p 8000:8000 ml-api:latest` |
| `docker ps` | Lists running containers. Use `-a` for all containers. | `docker ps`<br/>`docker ps -a` |
| `docker logs` | Shows container logs. Use `-f` to stream. | `docker logs -f ml-api` |
| `docker exec` | Runs a command inside a running container. | `docker exec -it ml-api bash` |
| `docker stop` | Stops a running container gracefully. | `docker stop ml-api` |

## Dockerfile basics

| Method | Description | Code example |
|---|---|---|
| `FROM` | Chooses the base image. Prefer slim images for Python APIs. | `FROM python:3.11-slim` |
| `WORKDIR` | Sets the working directory for following commands. | `WORKDIR /app` |
| `COPY` | Copies files from build context into the image. | `COPY requirements.txt .`<br/>`COPY app ./app` |
| `RUN` | Executes build-time commands and creates image layers. | `RUN pip install --no-cache-dir -r requirements.txt` |
| `ENV` | Sets environment variables inside the image. | `ENV PYTHONUNBUFFERED=1`<br/>`ENV MODEL_PATH=/models/model.joblib` |
| `EXPOSE` | Documents the port the app listens on. | `EXPOSE 8000` |
| `CMD` | Default command when container starts. | `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]` |

## Python and ML images

| Method | Description | Code example |
|---|---|---|
| Minimal Python API | Good default for FastAPI model serving. | `FROM python:3.11-slim`<br/>`WORKDIR /app`<br/>`COPY requirements.txt .`<br/>`RUN pip install --no-cache-dir -r requirements.txt`<br/>`COPY app ./app`<br/>`CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]` |
| Layer caching | Copy dependency files before source to reuse install layers. | `COPY requirements.txt .`<br/>`RUN pip install --no-cache-dir -r requirements.txt`<br/>`COPY . .` |
| CUDA image | Use NVIDIA CUDA base images for GPU inference/training. | `FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`<br/>`RUN apt-get update && apt-get install -y python3 python3-pip` |
| Non-root user | Avoid running production containers as root. | `RUN useradd -m appuser`<br/>`USER appuser` |
| Model files | Prefer mounting large models instead of baking them into every image. | `docker run --rm -v ./models:/models -e MODEL_PATH=/models/model.joblib ml-api` |

## Volumes, networks, and compose

| Method | Description | Code example |
|---|---|---|
| Volume mount | `-v host_path:container_path` mounts files into a container. | `docker run -v ./data:/app/data ml-job:latest` |
| Named volume | Persistent Docker-managed storage. | `docker volume create ml-cache`<br/>`docker run -v ml-cache:/cache ml-job` |
| Port mapping | `-p host:container` exposes container ports. | `docker run -p 8000:8000 ml-api` |
| Network | User-defined networks let services resolve each other by name. | `docker network create ml-net`<br/>`docker run --network ml-net --name api ml-api` |
| `docker compose up` | Starts multiple services from `compose.yaml`. | `docker compose up --build` |
| Compose service | Defines API plus Redis/Postgres dependencies. | `services:`<br/>`  api:`<br/>`    build: .`<br/>`    ports: ["8000:8000"]`<br/>`  redis:`<br/>`    image: redis:7` |

## Debugging and cleanup

| Method | Description | Code example |
|---|---|---|
| Inspect image | Shows image metadata and config. | `docker image inspect ml-api:latest` |
| Inspect container | Shows mounts, env, network, and runtime metadata. | `docker inspect ml-api` |
| Shell into image | Overrides command for debugging. | `docker run --rm -it ml-api:latest bash` |
| Check size | Lists image sizes. | `docker images` |
| Remove stopped containers | Cleanup stopped containers. | `docker container prune` |
| Remove unused images | Cleanup dangling and unused images. | `docker image prune -a` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| `.dockerignore` | Reduces build context and avoids leaking local files. | `.git`<br/>`__pycache__/`<br/>`.venv/`<br/>`data/`<br/>`models/` |
| Multi-stage build | Keeps runtime image smaller. | `FROM python:3.11 AS builder`<br/>`RUN pip wheel -r requirements.txt -w /wheels`<br/>`FROM python:3.11-slim`<br/>`COPY --from=builder /wheels /wheels` |
| GPU run | Run GPU containers with NVIDIA runtime support. | `docker run --gpus all --rm nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` |
| Healthcheck | Lets orchestrators know if service is unhealthy. | `HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"` |
| Env file | Load environment variables from a file. | `docker run --env-file .env ml-api` |
| Reproducible tag | Tag with git SHA or version, not only `latest`. | `docker build -t ml-api:1.4.2 .` |
| Local compose dev | Mount source for fast development. | `volumes:`<br/>`  - ./app:/app/app` |
| Push image | Publish image to registry. | `docker tag ml-api registry.example.com/ml-api:1.4.2`<br/>`docker push registry.example.com/ml-api:1.4.2` |

## Senior image engineering

| Method | Description | Code example |
|---|---|---|
| Pin base digest | Prevent silent base image drift in regulated or reproducible builds. | `FROM python:3.11-slim@sha256:abc123...` |
| Build args | Parameterize image builds without baking runtime secrets into layers. | `ARG APP_VERSION`<br/>`LABEL org.opencontainers.image.version=$APP_VERSION` |
| BuildKit cache mount | Speed dependency installs in CI without persisting cache in final image. | `RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt` |
| Wheelhouse build | Build wheels once, install from local wheelhouse in runtime stage. | `RUN pip wheel --wheel-dir /wheels -r requirements.txt`<br/>`RUN pip install --no-index --find-links=/wheels -r requirements.txt` |
| Distroless runtime | Reduce shell/package attack surface when app does not need OS tools. | `FROM gcr.io/distroless/python3-debian12`<br/>`COPY --from=builder /app /app` |
| Read-only filesystem | Prevent runtime writes except explicit temp/cache mounts. | `docker run --read-only --tmpfs /tmp ml-api:1.0.0` |
| Drop capabilities | Remove unnecessary Linux capabilities. | `docker run --cap-drop ALL --security-opt no-new-privileges ml-api` |
| SBOM | Produce software bill of materials for supply-chain review. | `syft packages ml-api:1.0.0 -o spdx-json > sbom.json` |
| Vulnerability scan | Scan image layers before deployment. | `trivy image --severity HIGH,CRITICAL ml-api:1.0.0` |
| Multi-arch build | Build images for multiple CPU architectures. | `docker buildx build --platform linux/amd64,linux/arm64 -t registry/ml-api:1.0.0 --push .` |

## ML container production patterns

| Method | Description | Code example |
|---|---|---|
| Separate train and serve images | Training images need compilers and notebooks; serving images should be small and stable. | `docker build -f Dockerfile.train -t trainer:1.0.0 .`<br/>`docker build -f Dockerfile.serve -t ml-api:1.0.0 .` |
| Model as mounted artifact | Keep image immutable and promote models independently. | `docker run -v /models/churn/v7:/models:ro -e MODEL_PATH=/models/model.joblib ml-api` |
| CUDA compatibility | Match CUDA runtime to PyTorch/TensorFlow wheel support. | `python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"` |
| Deterministic dependency install | Pin direct and transitive dependencies for reproducibility. | `pip-compile requirements.in --generate-hashes`<br/>`pip install --require-hashes -r requirements.txt` |
| Runtime user permissions | Make model files readable by non-root app user. | `RUN chown -R appuser:appuser /app`<br/>`USER appuser` |
| Compose health dependency | Start API after dependent services are healthy. | `depends_on:`<br/>`  redis:`<br/>`    condition: service_healthy` |
| Container resource limits | Catch memory leaks and size production capacity. | `docker run --memory 4g --cpus 2 ml-api` |
| Log to stdout | Let orchestrators collect logs; avoid writing app logs inside container. | `ENV PYTHONUNBUFFERED=1`<br/>`CMD ["python", "-m", "app"]` |
