---
title: FastAPI Master Cheatsheet
sidebar_position: 12
---

# FastAPI Master Cheatsheet

## App setup and routes

| Method | Description | Code example |
|---|---|---|
| `FastAPI()` | `FastAPI(title=None, version=None, lifespan=None, docs_url="/docs")` creates the ASGI app. | `from fastapi import FastAPI`<br/>`app = FastAPI(title="ML API", version="1.0.0")` |
| `@app.get()` | `app.get(path, response_model=None, status_code=200, tags=None)` registers a GET route. | `@app.get("/health")`<br/>`def health():`<br/>`    return {"status": "ok"}` |
| `@app.post()` | Registers a POST route for creating resources or running inference. | `@app.post("/predict")`<br/>`def predict(payload: PredictRequest):`<br/>`    return {"label": "positive"}` |
| Path parameters | Typed function args bind to URL placeholders. | `@app.get("/users/{user_id}")`<br/>`def get_user(user_id: int):`<br/>`    return {"user_id": user_id}` |
| Query parameters | Function args not in the path become query params. | `@app.get("/search")`<br/>`def search(q: str, limit: int = 10):`<br/>`    return {"q": q, "limit": limit}` |
| `APIRouter()` | `APIRouter(prefix="", tags=None, dependencies=None)` splits large apps into modules. | `router = APIRouter(prefix="/v1", tags=["v1"])`<br/>`@router.get("/items")`<br/>`def items(): return []`<br/>`app.include_router(router)` |

## Pydantic models and validation

| Method | Description | Code example |
|---|---|---|
| `BaseModel` | Defines request and response schemas with validation. | `from pydantic import BaseModel`<br/>`class PredictRequest(BaseModel):`<br/>`    text: str`<br/>`    top_k: int = 3` |
| `Field()` | `Field(default, gt=None, ge=None, max_length=None, description=None)` adds constraints and docs. | `class PredictRequest(BaseModel):`<br/>`    text: str = Field(min_length=1, max_length=5000)`<br/>`    top_k: int = Field(default=3, ge=1, le=10)` |
| `response_model` | Validates and filters response output. | `class Prediction(BaseModel):`<br/>`    label: str`<br/>`    score: float`<br/>`@app.post("/predict", response_model=Prediction)`<br/>`def predict(req: PredictRequest): return run_model(req.text)` |
| `model_dump()` | Converts Pydantic model to a dictionary. | `payload = req.model_dump()`<br/>`logger.info("request", extra=payload)` |
| `model_validate()` | Validates raw data into a model instance. | `raw = {"text": "hello", "top_k": 2}`<br/>`req = PredictRequest.model_validate(raw)` |

## Dependencies and configuration

| Method | Description | Code example |
|---|---|---|
| `Depends()` | `Depends(dependency=None, use_cache=True)` injects reusable dependencies. | `def get_repo(): return UserRepo()`<br/>`@app.get("/users")`<br/>`def users(repo = Depends(get_repo)):`<br/>`    return repo.list()` |
| Settings dependency | Centralizes environment config. | `class Settings(BaseModel):`<br/>`    model_path: str`<br/>`def get_settings():`<br/>`    return Settings(model_path=os.environ["MODEL_PATH"])` |
| Auth dependency | Rejects requests before route logic runs. | `def require_token(authorization: str = Header("")):`<br/>`    if authorization != "Bearer secret":`<br/>`        raise HTTPException(status_code=401)` |
| App state | Store long-lived resources on `app.state`. | `@app.on_event("startup")`<br/>`def load_model():`<br/>`    app.state.model = joblib.load("model.joblib")` |
| Lifespan | Preferred startup/shutdown hook for modern FastAPI. | `@asynccontextmanager`<br/>`async def lifespan(app):`<br/>`    app.state.model = load_model()`<br/>`    yield`<br/>`app = FastAPI(lifespan=lifespan)` |

## Async, files, middleware, and errors

| Method | Description | Code example |
|---|---|---|
| `async def` route | Use for async I/O such as HTTP calls or async DB drivers. | `@app.get("/remote")`<br/>`async def remote():`<br/>`    data = await client.get("/data")`<br/>`    return data.json()` |
| `UploadFile` | Efficient file upload API with async file methods. | `@app.post("/upload")`<br/>`async def upload(file: UploadFile):`<br/>`    content = await file.read()`<br/>`    return {"name": file.filename, "bytes": len(content)}` |
| `File()` | Declares file inputs in multipart requests. | `@app.post("/images")`<br/>`async def image(file: UploadFile = File(...)):`<br/>`    return {"content_type": file.content_type}` |
| `HTTPException` | Raises structured HTTP errors. | `if user is None:`<br/>`    raise HTTPException(status_code=404, detail="user not found")` |
| Exception handler | Converts custom exceptions to API responses. | `@app.exception_handler(ModelError)`<br/>`async def model_error_handler(request, exc):`<br/>`    return JSONResponse(status_code=422, content={"detail": str(exc)})` |
| Middleware | Wraps every request for logging, timing, CORS, auth, etc. | `@app.middleware("http")`<br/>`async def add_timing(request, call_next):`<br/>`    start = time.perf_counter()`<br/>`    response = await call_next(request)`<br/>`    response.headers["X-Time"] = str(time.perf_counter() - start)`<br/>`    return response` |
| `CORSMiddleware` | Allows browser clients from selected origins. | `app.add_middleware(CORSMiddleware, allow_origins=["https://app.example.com"], allow_methods=["*"], allow_headers=["*"])` |

## Serving ML models

| Method | Description | Code example |
|---|---|---|
| Load once | Load model at startup, not per request. | `@app.on_event("startup")`<br/>`def startup():`<br/>`    app.state.model = joblib.load("classifier.joblib")` |
| Inference endpoint | Convert validated request to model input and return typed output. | `@app.post("/predict", response_model=Prediction)`<br/>`def predict(req: PredictRequest):`<br/>`    pred = app.state.model.predict([req.text])[0]`<br/>`    return Prediction(label=str(pred), score=1.0)` |
| Batch endpoint | Accept multiple items to improve throughput. | `class BatchRequest(BaseModel):`<br/>`    texts: list[str]`<br/>`@app.post("/predict-batch")`<br/>`def predict_batch(req: BatchRequest):`<br/>`    return {"labels": app.state.model.predict(req.texts).tolist()}` |
| Background tasks | Run non-critical work after response is sent. | `@app.post("/events")`<br/>`def event(payload: dict, tasks: BackgroundTasks):`<br/>`    tasks.add_task(write_audit_log, payload)`<br/>`    return {"accepted": True}` |
| Uvicorn command | Run FastAPI with an ASGI server. | `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| Project layout | Keep API, schemas, services, and model loading separate. | `app/`<br/>`  main.py`<br/>`  schemas.py`<br/>`  services.py`<br/>`  model.py` |
| Health and readiness | Separate "process alive" from "model ready". | `@app.get("/ready")`<br/>`def ready():`<br/>`    return {"model_loaded": hasattr(app.state, "model")}` |
| Request ID logging | Add trace IDs for production debugging. | `request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))`<br/>`response.headers["X-Request-ID"] = request_id` |
| Pagination | Use `limit` and `offset` for list endpoints. | `@app.get("/items")`<br/>`def list_items(limit: int = 50, offset: int = 0):`<br/>`    return repo.list(limit=limit, offset=offset)` |
| Versioned API | Prefix routers by version. | `v1 = APIRouter(prefix="/api/v1")`<br/>`app.include_router(v1)` |
| Test client | Test routes without running a server. | `from fastapi.testclient import TestClient`<br/>`client = TestClient(app)`<br/>`assert client.get("/health").json()["status"] == "ok"` |
| Override dependency in tests | Swap real DB/model dependencies in tests. | `app.dependency_overrides[get_repo] = lambda: FakeRepo()` |
| Docker run | Serve with Uvicorn inside a container. | `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]` |

## Senior production patterns

| Method | Description | Code example |
|---|---|---|
| Lifespan resource boundary | Load expensive shared resources once and release them deterministically. Prefer this over scattered startup globals. | `@asynccontextmanager`<br/>`async def lifespan(app):`<br/>`    app.state.model = load_model(settings.model_path)`<br/>`    app.state.http = httpx.AsyncClient(timeout=5)`<br/>`    yield`<br/>`    await app.state.http.aclose()`<br/>`app = FastAPI(lifespan=lifespan)` |
| Service layer boundary | Keep route handlers thin: validate request, call service, return schema. | `@app.post("/predict", response_model=Prediction)`<br/>`def predict(req: PredictRequest, svc: Predictor = Depends(get_predictor)):`<br/>`    return svc.predict(req)` |
| Typed settings | Centralize env parsing so bad config fails during startup, not mid-request. | `from pydantic_settings import BaseSettings`<br/>`class Settings(BaseSettings):`<br/>`    model_path: str`<br/>`    max_batch_size: int = 32`<br/>`settings = Settings()` |
| Request body limits | Protect inference endpoints from oversized payloads. | `async def enforce_size(request: Request):`<br/>`    size = int(request.headers.get("content-length", 0))`<br/>`    if size > 2_000_000:`<br/>`        raise HTTPException(413, "payload too large")` |
| Idempotency key | Prevent duplicate side effects when clients retry POST requests. | `key = request.headers.get("Idempotency-Key")`<br/>`if key and cache.exists(key):`<br/>`    return cache.get(key)`<br/>`result = service.create(payload)`<br/>`cache.set(key, result)` |
| Rate limit dependency | Apply cheap request rejection before expensive model execution. | `def rate_limit(request: Request):`<br/>`    key = request.client.host`<br/>`    if limiter.too_many(key):`<br/>`        raise HTTPException(429, "too many requests")` |
| Model warmup | Run dummy inference after load to allocate kernels and reduce first-request latency. | `@app.on_event("startup")`<br/>`def warmup():`<br/>`    app.state.model.predict(["warmup request"])` |
| Microbatch queue | Batch concurrent inference requests for throughput-sensitive models. | `class Pending(BaseModel):`<br/>`    text: str`<br/>`    future: asyncio.Future`<br/>`queue: asyncio.Queue[Pending] = asyncio.Queue()` |
| Timeout wrapper | Bound downstream calls and model calls so workers do not hang forever. | `async def bounded_predict(req):`<br/>`    return await asyncio.wait_for(run_predict(req), timeout=3.0)` |
| Structured logging | Log machine-readable events with request ID, route, status, and latency. | `logger.info("request_done", extra={"request_id": rid, "path": request.url.path, "status": response.status_code, "latency_ms": elapsed})` |
| OpenAPI tags | Organize large APIs by domain. | `router = APIRouter(prefix="/models", tags=["models"])`<br/>`app.include_router(router)` |
| Custom response headers | Surface model version, request ID, and cache status. | `response.headers["X-Model-Version"] = settings.model_version`<br/>`response.headers["X-Request-ID"] = request_id` |

## Testing, security, and observability

| Method | Description | Code example |
|---|---|---|
| Contract test | Assert response schema and status for public API behavior. | `resp = client.post("/predict", json={"text": "great"})`<br/>`assert resp.status_code == 200`<br/>`assert set(resp.json()) == {"label", "score"}` |
| Async route test | Use an async client for true async behavior. | `async with AsyncClient(app=app, base_url="http://test") as ac:`<br/>`    resp = await ac.get("/health")`<br/>`assert resp.status_code == 200` |
| Dependency override cleanup | Reset overrides so tests do not leak state. | `app.dependency_overrides[get_repo] = lambda: fake_repo`<br/>`yield`<br/>`app.dependency_overrides.clear()` |
| Security headers | Add defensive response headers at middleware boundary. | `response.headers["X-Content-Type-Options"] = "nosniff"`<br/>`response.headers["Referrer-Policy"] = "no-referrer"` |
| Prometheus metrics | Export counters and histograms for API behavior. | `REQUESTS.labels(route="/predict", status=response.status_code).inc()`<br/>`LATENCY.observe(elapsed)` |
| Graceful degradation | Return a clear fallback when optional services fail. | `try:`<br/>`    features = feature_store.fetch(user_id)`<br/>`except FeatureStoreError:`<br/>`    features = default_features(user_id)` |
| Backpressure | Reject or queue requests when inference workers are saturated. | `if queue.qsize() > settings.max_queue_size:`<br/>`    raise HTTPException(503, "server busy")` |
| API deprecation | Signal clients before removing old endpoints. | `response.headers["Deprecation"] = "true"`<br/>`response.headers["Sunset"] = "2026-12-31"` |
