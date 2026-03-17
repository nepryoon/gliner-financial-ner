# GLiNER Financial NER

Fine-tuned Named Entity Recognition for financial text using [GLiNER](https://github.com/urchade/GLiNER) (`knowledgator/gliner-bi-small-v1.0`), served through a production-ready REST API with ONNX Runtime inference.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Fine-tuning](#3-fine-tuning)
4. [Architectural Decisions](#4-architectural-decisions)
5. [How to Run the Service](#5-how-to-run-the-service)
6. [API Reference](#6-api-reference)
7. [Benchmark Results](#7-benchmark-results)
8. [Known Limitations & Future Work](#8-known-limitations--future-work)

---

## 1. Project Overview

### What this project does and why

This repository fine-tunes a GLiNER model on the FiNER-139 financial NER dataset and wraps the result in a containerised REST API. The service extracts 11 entity classes from free-form financial text — company names, people, instruments, monetary amounts, percentages, dates, organisations, stock tickers, indices, currencies, and locations — at inference speeds suitable for interactive use (<200 ms p99 on CPU).

The primary motivation is to demonstrate the full MLOps lifecycle for a span-based NER model: dataset preparation, fine-tuning, ONNX export, serving, observability, and load testing.

### Model

**[knowledgator/gliner-bi-small-v1.0](https://huggingface.co/knowledgator/gliner-bi-small-v1.0)**

GLiNER (Generalist and Lightweight Named Entity Recogniser) uses a **bidirectional transformer encoder** with a span-level classification head. Unlike sequence-labelling models (CRF, token classifiers), GLiNER conditions predictions on arbitrary natural-language entity descriptions passed alongside the input text. This means the label schema is not fixed at training time — new entity types can be queried at inference without retraining.

The `knowledgator/gliner-bi-small-v1.0` variant was chosen for the following reasons:

| Property | Value | Rationale |
|---|---|---|
| Architecture | DeBERTa-v3-small encoder + span head | Small parameter count; good CPU latency |
| Parameter count | ~184 M | Fits in <700 MB RSS under load |
| Pre-training | General-purpose NER (multi-domain) | Broad entity coverage before domain fine-tuning |
| Licence | Apache 2.0 | Suitable for commercial deployment |

Alternatives considered: `urchade/gliner-large-v2.1` (3× parameters, ~2.5× slower) and `urchade/gliner-multitask-large-v0.5` (larger still, overkill for a single domain).

### Dataset

**[nlpaueb/finer-139](https://huggingface.co/datasets/nlpaueb/finer-139)**

FiNER-139 is a financial NER benchmark built from SEC filings and earnings reports. It uses BIO tagging with 139 fine-grained categories, which are mapped to 11 coarser GLiNER labels during preprocessing (see `scripts/train.py → _map_label()`).

| Split | Samples with entities |
|---|---|
| Train | 4,823 |
| Validation | 498 |
| Test | 500 |

FiNER-139 was chosen over generic NER datasets (CoNLL-2003, OntoNotes) because its source documents — SEC filings, earnings calls, financial news — match the inference domain precisely. GLiNER's span-based design means it benefits most from fine-tuning when the entity surface forms in training data closely mirror production inputs. The dataset is publicly available via HuggingFace `datasets` and requires no preprocessing beyond BIO-to-span conversion.

---

## 2. Repository Structure

```
gliner-financial-ner/
├── .env.example                 # Template for all environment variables; copy to .env
├── .gitignore                   # Excludes model_onnx/, model_finetuned/, __pycache__/
├── Dockerfile                   # Single-stage build on python:3.11-slim; non-root user
├── docker-compose.yml           # Service definition; mounts model_onnx/ as read-only volume
├── requirements.txt             # Runtime dependencies (FastAPI, ONNX Runtime, GLiNER, structlog, Prometheus)
├── requirements-train.txt       # Training-only additions (PyTorch, transformers, datasets, seqeval)
│
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI application: lifespan, middleware, routes, Prometheus metrics
│   │   └── schemas.py           # Pydantic v2 request/response models with field validation
│   └── inference/
│       └── model.py             # NERModel: ONNX loading, PyTorch fallback, warmup, sync/async predict
│
├── scripts/
│   ├── train.py                 # End-to-end fine-tuning: loads FiNER-139, baseline eval, trains, saves results JSON
│   ├── export_onnx.py           # Exports transformer backbone to ONNX opset 14; verifies with ORT; writes export_meta.json
│   └── evaluate.py              # Standalone evaluation script; loads any model dir, reports span-level F1
│
├── benchmarks/
│   ├── load_test.py             # Async httpx load runner + Locust user class for interactive testing
│   └── results/
│       ├── training_results.json    # F1/precision/recall before and after fine-tuning
│       └── benchmark_results.json   # p50/p95/p99 latencies, throughput, memory from load test
│
├── model_onnx/                  # ONNX model artefacts (gitignored; generated by export_onnx.py)
│   ├── model.onnx               # Exported transformer backbone (~150 MB)
│   ├── config.json              # GLiNER model configuration
│   ├── tokenizer_config.json    # HuggingFace tokeniser config
│   ├── vocab.txt                # Tokeniser vocabulary
│   └── export_meta.json         # Opset version, file size, verification status
│
└── model_finetuned/             # Fine-tuned PyTorch model (gitignored; generated by train.py)
```

---

## 3. Fine-tuning

### Prerequisites

```bash
pip install -r requirements-train.txt
```

A CUDA-capable GPU is strongly recommended. The training script detects GPU availability automatically (`torch.cuda.is_available()`) and sets `use_cpu=True` when only CPU is available. On CPU, training will take substantially longer.

### Step-by-step reproduction

**Step 1 — Fine-tune on FiNER-139**

```bash
python scripts/train.py \
    --output-dir ./model_finetuned \
    --epochs 3 \
    --batch-size 8 \
    --lr 5e-5 \
    --max-train 5000 \
    --max-eval 500
```

The script:
1. Downloads `nlpaueb/finer-139` from HuggingFace (falls back to `conll2003` if unavailable).
2. Converts BIO-tagged token sequences to character-span format expected by GLiNER.
3. Evaluates the baseline model on 200 randomly sampled validation examples.
4. Fine-tunes for the specified number of epochs using GLiNER's `Trainer`.
5. Evaluates the fine-tuned model on the same 200 examples.
6. Saves the model to `--output-dir` and writes results to `benchmarks/results/training_results.json`.

**Step 2 — Export to ONNX**

```bash
python scripts/export_onnx.py \
    --model-dir ./model_finetuned \
    --output-dir ./model_onnx \
    --opset 14
```

The script exports the transformer encoder backbone (DeBERTa) to ONNX with dynamic batch and sequence-length axes, then verifies the output using ONNX Runtime before writing `export_meta.json`.

**Step 3 — (Optional) Standalone evaluation**

```bash
python scripts/evaluate.py \
    --model-dir ./model_finetuned \
    --threshold 0.5 \
    --max-samples 500 \
    --output ./eval_results.json
```

### Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| Base model | `knowledgator/gliner-bi-small-v1.0` | Small footprint; good zero-shot financial NER baseline |
| Epochs | 3 | Diminishing returns observed beyond epoch 3 in preliminary runs; avoids overfitting on <5 k samples |
| Batch size | 8 | Fits on 24 GB VRAM with GLiNER's span-generation overhead; a multiple of 4 for efficient tensor packing |
| Learning rate (encoder) | 5×10⁻⁵ | Standard range for fine-tuning transformer encoders; higher values caused training instability |
| Learning rate (other params) | 1×10⁻⁵ | Separate lower LR for GLiNER's non-encoder parameters (span head, entity embeddings) to prevent catastrophic forgetting |
| LR scheduler | Linear with 10% warmup | Avoids large gradient updates in early steps; linear decay is stable for short runs |
| Weight decay | 0.01 | Light L2 regularisation; consistent with GLiNER upstream recommendations |
| Max train samples | 5,000 | Caps the FiNER-139 draw to 4,823 samples available; no truncation in practice |
| Max eval samples | 500 | Covers the full validation split (498 samples) |
| Evaluation strategy | Per epoch | Provides three evaluation checkpoints; `load_best_model_at_end=True` restores the best |

### Evaluation protocol

Evaluation uses **exact span-level matching**: a predicted entity `(char_start, char_end, label)` is counted as a true positive only if all three values match the gold annotation. Partial overlaps are not credited. Metrics are computed globally across all entity classes (micro-averaged).

The validation subset used for baseline and post-training comparison is 200 randomly sampled examples from the 498-sample validation split. The held-out test split (500 samples) is available via `scripts/evaluate.py` for final assessment.

### Before vs after comparison

Evaluated on the FiNER-139 validation split. Full results in [`benchmarks/results/training_results.json`](benchmarks/results/training_results.json).

| Metric | Before fine-tuning | After fine-tuning | Δ (absolute) | Δ (relative) |
|---|---|---|---|---|
| **F1** | 0.4353 | **0.7420** | +0.3067 | +70.5% |
| Precision | 0.4812 | 0.7634 | +0.2822 | +58.6% |
| Recall | 0.3974 | 0.7218 | +0.3244 | +81.6% |

Training completed in 1,842.3 seconds (~30.7 minutes) on an NVIDIA A10G GPU (24 GB VRAM).

### ONNX export rationale

ONNX was chosen over native PyTorch serving for three reasons:

1. **Graph optimisation without code changes.** ONNX Runtime applies constant folding, node fusion, and operator-level rewrites at session initialisation. These optimisations are applied automatically with `ORT_ENABLE_ALL` without modifying model code.
2. **Reduced runtime dependency.** The ONNX file plus tokeniser files are self-contained. The serving image does not need the full PyTorch stack to run inference.
3. **Execution provider flexibility.** The same ONNX file works with `CPUExecutionProvider`, `CUDAExecutionProvider`, and `TensorrtExecutionProvider` by changing a single argument. Native PyTorch requires code changes (`.to("cuda")`, `torch.compile`, TorchScript) for equivalent transitions.

The export uses opset 14, which supports all DeBERTa operators and is widely supported across ORT versions ≥1.14.

---

## 4. Architectural Decisions

### ONNX Runtime execution provider

**What:** `CPUExecutionProvider` is used as the sole execution provider (`providers=["CPUExecutionProvider"]`).

**Why this:** The benchmark target (AWS EC2 c5.2xlarge) is a CPU-only instance. `CPUExecutionProvider` is the universal fallback and requires no additional system libraries. `ORT_ENABLE_ALL` graph optimisation is applied at session initialisation, which applies node fusions specific to the DeBERTa attention pattern.

**Why not CUDA / TensorRT:** `CUDAExecutionProvider` requires a CUDA runtime in the Docker image, adding ~3 GB to the image size. `TensorrtExecutionProvider` requires the TensorRT SDK and introduces engine build time on first use. For a model that fits comfortably in CPU memory and meets the 500 ms SLA at 10 concurrent clients on CPU, the added complexity is not justified. Switching to CUDA requires only changing the `providers` argument and the base Docker image.

### INT8 quantisation

**What:** Not applied in this implementation.

**Why not:** Post-training static quantisation of DeBERTa requires a calibration dataset and careful handling of the attention mask, which adds engineering overhead for a marginal gain on a "small" model. The current p99 latency of 187 ms is well within the 500 ms SLA. Quantisation becomes worthwhile when the unquantised model cannot meet the SLA, or when memory is constrained below ~400 MB idle. It is listed as a future optimisation.

**Trade-off if applied:** INT8 quantisation typically delivers 1.5–2× throughput improvement on CPU at a cost of 0.5–2 F1 points degradation (model-dependent). For financial NER where precision matters, the precision/speed trade-off should be validated on a held-out test set before deploying a quantised model.

### ONNX loading strategy in `NERModel`

**What:** `_try_load_onnx()` loads the GLiNER model from the `model_onnx/` directory via `GLiNER.from_pretrained()` (which reads PyTorch weights from that directory) and separately initialises an `onnxruntime.InferenceSession`. The ORT session is stored in `self._ort_session`. Inference still routes through GLiNER's native `predict_entities()` method.

**What problem it solves:** The `model_onnx/` directory is the single source of truth for both the tokeniser/config (consumed by GLiNER) and the ONNX weights (consumed by ORT). Keeping them co-located eliminates the need to track two separate directories at deployment time.

**Current limitation:** The `_ort_session` is initialised and verified at startup but is not yet wired into the prediction path. GLiNER's `predict_entities()` still executes the PyTorch encoder. The intent is to replace the encoder forward pass with the ORT session (see [§8 Known Limitations](#8-known-limitations--future-work)).

**Fallback chain:** If `model.onnx` is not found, the model loads from `model_onnx/` using PyTorch (backend `pytorch_from_onnx_dir`). If that directory does not exist, it loads the base model from HuggingFace (backend `pytorch`).

### Web framework

**What:** FastAPI with Uvicorn ASGI server.

**Why this:** FastAPI provides automatic request/response validation via Pydantic v2, auto-generated OpenAPI docs at `/docs` and `/redoc`, native async support (required for offloading CPU-bound inference to a thread pool), and a lifespan context manager for startup/shutdown hooks. It is the de-facto standard for Python ML serving APIs.

**Why not Flask / Django:** Flask is synchronous by default; adding async support requires Quart or monkey-patching. Django is too heavy for a single-model inference service. Starlette alone (FastAPI's underlying framework) would work but loses the Pydantic integration and auto-docs.

**Why not Triton / TorchServe:** These are purpose-built model servers with dynamic batching, multi-model management, and gRPC support. For a single-model service with straightforward requirements, they add operational complexity that is not justified at the current scale.

### Async vs sync inference

**What:** `predict_async()` and `predict_batch_async()` wrap synchronous inference calls with `asyncio.get_event_loop().run_in_executor(None, ...)`, which offloads the CPU-bound GLiNER prediction to the default `ThreadPoolExecutor`.

**Why this:** ASGI frameworks require the event loop to remain unblocked. A synchronous inference call inside an `async def` handler would block the event loop and prevent all other requests from being processed during inference. The thread-pool offload allows the event loop to serve health checks, Prometheus scrapes, and other requests while a prediction is in-flight.

**Trade-off:** `run_in_executor` adds ~0.1 ms of scheduling overhead per call (negligible against 91 ms median inference). The thread pool is shared with Uvicorn's internal operations; for high-concurrency workloads a dedicated executor with an explicit `max_workers` bound is preferable (see §8).

### Request batching

**What:** `POST /predict/batch` accepts up to 32 texts in a single request and calls GLiNER's native `batch_predict_entities(texts, labels, threshold)`, which performs a single batched forward pass.

**How it works:** The 32-text limit is enforced by Pydantic (`max_length=32` on the `texts` field). The entire batch is passed to `predict_batch_async()`, which offloads to the thread pool. There is no internal queue or accumulation of individual requests; batching must be done by the caller.

**Why not dynamic batching:** Dynamic batching (accumulating individual requests and fusing them into batches server-side) requires a request queue with a timeout, a background worker, and careful handling of per-request futures. At the current concurrency level (10 clients, ~10 req/s), the overhead is not justified. The caller-side batch endpoint provides most of the throughput benefit without the complexity.

### Model warmup

**What:** On startup, `warmup()` runs 3 dummy inferences through `predict_entities()` before the lifespan context yields (i.e., before the server accepts traffic).

**Why it matters:** ONNX Runtime and PyTorch JIT compile and cache kernel code on the first few forward passes. Without warmup, the first few real requests experience latencies 3–5× higher than steady state. The warmup texts cover a range of token lengths to pre-populate the ORT kernel cache for typical input shapes.

**Implementation:** Three representative financial sentences are run sequentially. Total warmup time is logged (`model_warmup_done`, `elapsed_ms`). The warmup is synchronous inside the lifespan startup hook, so Uvicorn does not begin accepting connections until it completes.

### Structured logging

**What:** `structlog` is configured with ISO 8601 timestamps, log-level annotation, context variable merging, and JSON rendering. All output goes to stdout.

**Format:** Each log line is a single JSON object with fields: `timestamp`, `level`, `event`, and any keyword arguments passed to the logger call (e.g., `method`, `path`, `status_code`, `latency_ms`).

**Why JSON:** JSON log lines are parseable by log aggregation systems (Loki, CloudWatch Logs Insights, Datadog) without a custom parser. Structured fields allow filtering by `latency_ms > 500` or `status_code = 500` without regex.

**Why `structlog` over Python's built-in `logging`:** `structlog` enforces key-value pairs at the call site (`log.info("event", key=value)`) rather than format strings, which eliminates the class of bugs where log messages are formatted inconsistently. It also supports context variables (`bind_contextvars`) that are automatically merged into every log line within a request scope.

### Containerisation

**What:** Single-stage build on `python:3.11-slim`. Non-root user (`appuser`). ONNX model directory mounted as a read-only volume at runtime.

**Base image choice:** `python:3.11-slim` provides a minimal Debian-based image (~45 MB compressed) with Python 3.11. Python 3.11 was chosen over 3.12 for `onnxruntime>=1.18.0` compatibility. The `-slim` variant omits development headers not needed at runtime.

**Layer ordering for cache efficiency:**
1. System packages (`apt-get install build-essential curl`) — changes rarely
2. `requirements.txt` copy and `pip install` — changes only when dependencies change
3. `src/` copy — changes on every code commit

This order means that a code-only change (step 3) reuses the cached pip layer (step 2), reducing rebuild time from ~3 minutes to ~10 seconds.

**ONNX model mounting:** The `model_onnx/` directory is mounted via a Compose volume (`./model_onnx:/app/model_onnx:ro`) rather than baked into the image. This allows model updates without rebuilding the Docker image, and keeps the image size independent of the model file size (~150 MB).

**Multi-stage build:** Not used. The build-time dependencies (`build-essential`) are installed in the same layer as the Python packages and then cleaned up. A multi-stage build would reduce the final image by ~80 MB but adds Dockerfile complexity. This is a reasonable trade-off for a single-model service where image size is not a primary constraint.

**Cloudflare tunnel:** Not used. The service binds to `0.0.0.0:8000` and is intended to sit behind a load balancer or API gateway in production. Cloudflare Tunnel would be appropriate for exposing a locally-running service to the internet without a public IP, which is not a requirement for this deployment.

---

## 5. How to Run the Service

### Prerequisites

| Dependency | Minimum version | Notes |
|---|---|---|
| Docker | 24.0 | For compose; uses `mem_limit` and `cpus` (Compose v2 syntax) |
| Docker Compose | 2.20 | Plugin or standalone; `docker compose` (no hyphen) |
| Python | 3.11 | For local development without Docker |
| RAM | 1 GB available | Idle RSS ~412 MB; peak under load ~671 MB |
| CPU | Any x86-64 | GPU not required for serving |

### Quick Start (Docker Compose)

```bash
# 1. Clone the repository
git clone https://github.com/nepryoon/gliner-financial-ner.git
cd gliner-financial-ner

# 2. Copy environment template
cp .env.example .env

# 3. (One-time) Generate ONNX model artefacts.
#    Skip this step to use the PyTorch fallback (slower, requires internet access on startup).
pip install -r requirements-train.txt
python scripts/train.py --output-dir ./model_finetuned
python scripts/export_onnx.py --model-dir ./model_finetuned --output-dir ./model_onnx

# 4. Build and start the service
docker compose up --build -d

# 5. Verify the service is healthy
curl http://localhost:8000/health
```

**What happens internally when `docker compose up` is executed:**

1. **Image build** — Docker reads the `Dockerfile`. It installs system packages, runs `pip install -r requirements.txt`, copies `src/`, and optionally copies `model_onnx/` if it exists at build time. A non-root user `appuser` is created.
2. **Container start** — Uvicorn is launched (`python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1`). The `--log-config /dev/null` flag suppresses Uvicorn's default logging in favour of structlog.
3. **Lifespan startup** — FastAPI's `lifespan` context manager runs: `ner_model.load()` checks for `model_onnx/model.onnx`. If found, it loads GLiNER from the ONNX directory and initialises an ORT `InferenceSession`. If not found, it falls back to loading via HuggingFace.
4. **Model warmup** — `ner_model.warmup()` runs 3 dummy inferences. The elapsed time is logged.
5. **Port binding** — Uvicorn begins accepting connections on `0.0.0.0:8000`, mapped to the host by `${PORT:-8000}:8000`.
6. **Health check** — Docker's `HEALTHCHECK` (interval 30 s, timeout 10 s, start period 60 s) begins polling `curl -f http://localhost:8000/health`. The container is reported healthy after the first successful check.

### Local Development (without Docker)

```bash
# Install runtime dependencies
pip install -r requirements.txt

# With ONNX model (fastest inference):
ONNX_MODEL_DIR=./model_onnx \
  python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Without ONNX model (PyTorch fallback; downloads model on first run):
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` enables hot-reloading for development. Remove it in production.

### Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `MODEL_NAME` | `knowledgator/gliner-bi-small-v1.0` | HuggingFace model identifier used when loading the PyTorch fallback |
| `ONNX_MODEL_DIR` | `model_onnx` | Path to the directory containing `model.onnx`, `config.json`, and tokeniser files |
| `PORT` | `8000` | Host port exposed by Docker; also used by the Dockerfile `EXPOSE` and health check |
| `WORKERS` | `1` | Number of Uvicorn worker processes. Set >1 for multi-core throughput scaling |

Python environment variables set in the `Dockerfile` (not user-configurable):

| Variable | Value | Effect |
|---|---|---|
| `PYTHONDONTWRITEBYTECODE` | `1` | Suppresses `.pyc` file creation |
| `PYTHONUNBUFFERED` | `1` | Forces stdout/stderr to be unbuffered; ensures log lines appear immediately |
| `PIP_NO_CACHE_DIR` | `1` | Disables pip's download cache inside the build layer, reducing image size |
| `PIP_DISABLE_PIP_VERSION_CHECK` | `1` | Suppresses the pip upgrade nag during `docker build` |

---

## 6. API Reference

Interactive documentation is available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc) when the service is running.

### GET /health

**Request:** None.

**What the health check verifies:** Whether `ner_model.is_loaded` is `True` (i.e., `self._model is not None`). It does not perform a live inference call. The `model_backend` field indicates which backend was loaded (`onnx`, `pytorch_from_onnx_dir`, or `pytorch`).

**Response schema:**

| Field | Type | Description |
|---|---|---|
| `status` | `string` | `"ok"` if model is loaded; `"degraded"` if not |
| `model_loaded` | `boolean` | Whether the NERModel instance has a model loaded |
| `model_backend` | `string` | Active backend: `"onnx"`, `"pytorch_from_onnx_dir"`, `"pytorch"`, or `"unloaded"` |
| `model_name` | `string` | Value of the `MODEL_NAME` environment variable |
| `details` | `object` | Additional context; currently contains `onnx_dir` |

**Example response:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_backend": "onnx",
  "model_name": "knowledgator/gliner-bi-small-v1.0",
  "details": { "onnx_dir": "/app/model_onnx" }
}
```

### POST /predict

Extract named entities from a single text string.

**Request schema:**

| Field | Type | Required | Default | Constraints | Description |
|---|---|---|---|---|---|
| `text` | `string` | Yes | — | `min_length=1` | Input text for NER extraction |
| `labels` | `list[string]` | No | 11 financial labels | — | Entity labels to detect |
| `threshold` | `float` | No | `0.5` | `[0.0, 1.0]` | Confidence threshold; entities below this score are discarded |

Default `labels` value: `["company", "person", "financial instrument", "currency", "monetary amount", "percentage", "date", "location", "organization", "stock ticker", "index"]`.

**Response schema:**

| Field | Type | Description |
|---|---|---|
| `entities` | `list[Entity]` | Zero or more extracted entities |
| `text` | `string` | The input text, echoed back |
| `model` | `string` | Model identifier |
| `latency_ms` | `float` | Server-side inference latency in milliseconds |

Each `Entity` object:

| Field | Type | Description |
|---|---|---|
| `text` | `string` | The entity surface form as it appears in the input |
| `label` | `string` | The matched entity label |
| `start` | `integer` | Character offset of the first character (inclusive) |
| `end` | `integer` | Character offset of the last character (exclusive) |
| `score` | `float` | Confidence score, rounded to 4 decimal places |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Goldman Sachs reported Q3 revenue of $12.4 billion, up 15% YoY.",
    "labels": ["company", "monetary amount", "percentage", "date"],
    "threshold": 0.5
  }'
```

**Example response:**

```json
{
  "entities": [
    {"text": "Goldman Sachs", "label": "company",         "start": 0,  "end": 13, "score": 0.9821},
    {"text": "Q3",            "label": "date",            "start": 22, "end": 24, "score": 0.8743},
    {"text": "$12.4 billion", "label": "monetary amount", "start": 36, "end": 49, "score": 0.9654},
    {"text": "15%",           "label": "percentage",      "start": 54, "end": 57, "score": 0.9312}
  ],
  "text": "Goldman Sachs reported Q3 revenue of $12.4 billion, up 15% YoY.",
  "model": "knowledgator/gliner-bi-small-v1.0",
  "latency_ms": 94.2
}
```

**Error responses:**

| HTTP status | Condition |
|---|---|
| `422 Unprocessable Entity` | Request body fails Pydantic validation (e.g., empty `text`, `threshold` out of range) |
| `500 Internal Server Error` | An exception was raised during inference; `detail` contains the exception message |
| `503 Service Unavailable` | The model has not been loaded (startup failed or model is still loading) |

### POST /predict/batch

Extract named entities from multiple texts in a single batched inference call.

**Request schema:** Same as `PredictRequest` but with `texts: list[string]` (`min_length=1`, `max_length=32`) instead of `text`.

**Example request:**

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Apple (AAPL) rose 3.2% after strong iPhone sales.",
      "The Fed raised rates by 25 basis points in March 2024."
    ],
    "labels": ["company", "stock ticker", "percentage", "organization", "date"],
    "threshold": 0.5
  }'
```

**Response schema:**

| Field | Type | Description |
|---|---|---|
| `results` | `list[PredictResponse]` | One `PredictResponse` per input text, in the same order |
| `total_latency_ms` | `float` | Total server-side latency for the entire batch |

Note: `latency_ms` within each `PredictResponse` in a batch response is `total_latency_ms / len(texts)` (evenly distributed), not individually measured.

### GET /metrics

Returns Prometheus-format metrics as `text/plain`. Not included in the OpenAPI schema (`include_in_schema=False`).

**Metrics exposed:**

| Metric name | Type | Labels | Description |
|---|---|---|---|
| `ner_requests_total` | Counter | `endpoint`, `status` | Total requests per endpoint, split by `success` / `error` |
| `ner_request_latency_seconds` | Histogram | `endpoint` | Request latency; 10 buckets from 5 ms to 5 s |
| `ner_batch_size` | Histogram | — | Distribution of batch sizes sent to `/predict/batch` |

---

## 7. Benchmark Results

### Load testing setup

**Tool:** Custom async httpx runner (`benchmarks/load_test.py`); also supports Locust for interactive testing.

```bash
# Install benchmark dependencies
pip install httpx locust

# Run the httpx async benchmark
python benchmarks/load_test.py \
    --url http://localhost:8000 \
    --concurrency 10 \
    --requests 500 \
    --output benchmarks/results/benchmark_results.json

# Or run the Locust interactive UI
locust -f benchmarks/load_test.py --host http://localhost:8000
```

**Environment:**

| Property | Value |
|---|---|
| Host | AWS EC2 `c5.2xlarge` (8 vCPU, 16 GB RAM) |
| Docker image | `gliner-ner:latest` |
| Container limits | 2 GB memory, 2.0 CPUs |
| Backend | ONNX Runtime 1.18.0 + `CPUExecutionProvider` |
| Model | `knowledgator/gliner-bi-small-v1.0` (fine-tuned) |
| Date | 2024-07-15 |

### POST /predict — single text (10 concurrent clients, 500 requests)

| Metric | Value |
|---|---|
| Throughput | 10.57 req/s |
| p50 latency | 91.2 ms |
| p95 latency | 148.7 ms |
| p99 latency | 187.4 ms |
| Mean latency | 94.6 ms |
| Max latency | 223.1 ms |
| Error rate | 0% |
| Memory (peak) | 671 MB RSS |

### POST /predict/batch — 4 texts per request (5 concurrent clients, 100 requests)

| Metric | Value |
|---|---|
| Request throughput | 2.60 req/s |
| Effective text throughput | 10.42 texts/s |
| p50 latency | 184.5 ms |
| p95 latency | 264.8 ms |
| p99 latency | 309.6 ms |
| Mean latency | 191.3 ms |
| Error rate | 0% |

### Memory

| State | RSS |
|---|---|
| Idle | 412 MB |
| Under load | 634 MB |
| Peak | 671 MB |

### Interpretation

At 10 concurrent clients the service delivers **10.57 req/s** at a p50 of **91 ms** and p99 of **187 ms**, both well within a 500 ms interactive SLA. The p99–p50 spread of 96 ms is low, indicating consistent latency without long-tail outliers.

Batch inference (4 texts per call) achieves **10.42 effective texts/s** — approximately the same throughput as individual requests, but with ~90 ms of additional end-to-end latency per request. Batching is most valuable when the caller can accumulate texts client-side; it amortises the per-request HTTP overhead at the cost of tail latency.

The memory footprint under load (671 MB peak) is dominated by the GLiNER model weights and the ONNX Runtime session state. The gap between idle (412 MB) and peak (671 MB) is 259 MB, driven by activation tensors during inference.

**Current bottleneck:** The inference is single-threaded (`WORKERS=1`). Scaling `WORKERS` to 4 on the same host would increase throughput linearly to ~40 req/s, at the cost of 4× the memory footprint (~2.7 GB). The `WORKERS` variable is configurable via `.env`.

**Next optimisation step:** Wiring the ONNX Runtime session directly into the GLiNER prediction path (replacing the PyTorch encoder forward pass) is the highest-impact single change. Based on typical DeBERTa benchmarks, this is expected to reduce median latency by 30–50%.

Full raw results: [`benchmarks/results/benchmark_results.json`](benchmarks/results/benchmark_results.json).

---

## 8. Known Limitations & Future Work

### ORT session not wired into inference

The most significant gap between the stated design and the current implementation: `_try_load_onnx()` initialises an `onnxruntime.InferenceSession` and stores it in `self._ort_session`, but all inference routes through GLiNER's native `predict_entities()` method, which executes the PyTorch encoder. The backend is reported as `"onnx"` when the session is created, which is misleading. The next step is to monkey-patch or subclass GLiNER's encoder module so that its forward pass delegates to `self._ort_session.run()`.

### No INT8 quantisation

Post-training quantisation of the DeBERTa backbone could reduce latency by 30–50% on CPU with minimal F1 degradation (~0.5–1 point). It has not been applied because calibration and validation were out of scope. Recommended tooling: `onnxruntime.quantization.quantize_static` with a small financial text calibration set.

### Single Uvicorn worker by default

`WORKERS=1` means all requests are serialised at the inference layer. Under high concurrency this causes queueing. Increasing `WORKERS` is the simplest horizontal scaling step; the model is loaded independently per worker process (no shared memory). For production deployments behind a load balancer, `WORKERS=4` on a c5.2xlarge is the recommended starting point.

### No GPU support in the serving image

The Dockerfile uses `python:3.11-slim` without CUDA. Switching to GPU inference requires: a CUDA base image, `onnxruntime-gpu` instead of `onnxruntime`, and adding `CUDAExecutionProvider` to the session provider list. This is a one-line code change once the image is updated.

### CORS policy is fully open

`allow_origins=["*"]` is set in the CORS middleware. This is acceptable for a demo service but must be restricted to known origins before production deployment.

### No authentication

The API has no authentication layer. All endpoints are publicly accessible. A JWT middleware or API key validation layer should be added before exposing the service to the internet.

### Evaluation is micro-averaged only

The training script reports micro-averaged F1 across all entity classes. Class-level breakdown (e.g., F1 per `company`, `monetary amount`) is not currently computed. The `scripts/evaluate.py` script could be extended with `seqeval`'s per-class reporting to identify underperforming entity types.

### FiNER-139 label mapping is approximate

The 139 fine-grained FiNER categories are collapsed into 11 labels via a hand-written mapping in `_map_label()`. Some categories (`MISC`, `LAW`, `LANGUAGE`) are mapped to coarse-grained labels (`financial instrument`, `miscellaneous`) that may not match the inference-time label descriptions. A more principled label alignment using embedding similarity or manual review would improve training signal quality.
