# GLiNER Financial NER

Fine-tuned Named Entity Recognition for financial text using [GLiNER](https://github.com/urchade/GLiNER) (`knowledgator/gliner-bi-small-v1.0`), served through a production-ready REST API with ONNX Runtime inference.

---

## Table of Contents

1. [Overview](#overview)
2. [Part 1 — Fine-tuning](#part-1--fine-tuning)
   - [Model Selection](#model-selection)
   - [Dataset](#dataset)
   - [Training](#training)
   - [Results (F1 / Precision / Recall)](#results)
   - [ONNX Export](#onnx-export)
3. [Part 2 — Deployment](#part-2--deployment)
   - [Quick Start (Docker Compose)](#quick-start-docker-compose)
   - [API Endpoints](#api-endpoints)
   - [Running Locally (without Docker)](#running-locally-without-docker)
4. [Performance & Observability](#performance--observability)
5. [Benchmarking](#benchmarking)
6. [Project Structure](#project-structure)

---

## Overview

This project fine-tunes a GLiNER model from [knowledgator](https://huggingface.co/knowledgator) on the publicly available [FiNER-139](https://huggingface.co/datasets/nlpaueb/finer-139) financial NER dataset and exposes the resulting model as a containerised REST API using ONNX Runtime for fast CPU inference.

Supported entity types:

| Label | Examples |
|---|---|
| `company` | Apple Inc., Goldman Sachs |
| `person` | David Solomon, Warren Buffett |
| `financial instrument` | S&P 500 options, T-bonds |
| `monetary amount` | $90.1 billion, €1.2M |
| `percentage` | 2.4%, 25 basis points |
| `date` | Q3 2024, March 2025 |
| `organization` | Federal Reserve, SEC |
| `stock ticker` | TSLA, AAPL |
| `index` | S&P 500, NASDAQ |
| `currency` | USD, EUR |
| `location` | New York, Hong Kong |

---

## Part 1 — Fine-tuning

### Model Selection

Base model: **[knowledgator/gliner-bi-small-v1.0](https://huggingface.co/knowledgator/gliner-bi-small-v1.0)**

GLiNER (Generalist and Lightweight NER) uses a bidirectional encoder and span-level entity prediction with arbitrary natural-language entity labels. The `knowledgator` variants are optimised for production use.

### Dataset

**[nlpaueb/finer-139](https://huggingface.co/datasets/nlpaueb/finer-139)** — Financial Named Entity Recognition with 139 entity categories derived from SEC filings, earnings reports and financial news. Used via HuggingFace `datasets`.

```
Train : 4,823 sentences with entities
Eval  :   498 sentences
Test  :   500 sentences
```

### Training

Install training dependencies:

```bash
pip install -r requirements-train.txt
```

Run fine-tuning (GPU recommended):

```bash
python scripts/train.py \
    --output-dir ./model_finetuned \
    --epochs 3 \
    --batch-size 8 \
    --lr 5e-5 \
    --max-train 5000 \
    --max-eval 500
```

Key hyperparameters:

| Hyperparameter | Value |
|---|---|
| Base model | `knowledgator/gliner-bi-small-v1.0` |
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 5×10⁻⁵ |
| LR schedule | Linear with 10% warmup |
| Weight decay | 0.01 |

### Results

Evaluated on the FiNER-139 validation split using **exact span-level matching**.

| Metric | Before fine-tuning | After fine-tuning | Δ |
|---|---|---|---|
| **F1** | 0.4353 | **0.7420** | +0.3067 |
| Precision | 0.4812 | 0.7634 | +0.2822 |
| Recall | 0.3974 | 0.7218 | +0.3244 |

> Full results: [`benchmarks/results/training_results.json`](benchmarks/results/training_results.json)

### ONNX Export

After fine-tuning, export the model to ONNX for runtime-efficient CPU inference:

```bash
python scripts/export_onnx.py \
    --model-dir ./model_finetuned \
    --output-dir ./model_onnx \
    --opset 14
```

The `model_onnx/` directory is then mounted into the Docker container.

---

## Part 2 — Deployment

### Quick Start (Docker Compose)

The entire service starts with **a single command** and requires no manual steps afterwards:

```bash
# 1. (One-time) Run fine-tuning and ONNX export, or skip to use PyTorch fallback
python scripts/train.py --output-dir ./model_finetuned
python scripts/export_onnx.py --model-dir ./model_finetuned --output-dir ./model_onnx

# 2. Start the service
docker compose up --build -d

# 3. Check health
curl http://localhost:8000/health
```

> **Without an ONNX model**: the service automatically falls back to loading `knowledgator/gliner-bi-small-v1.0` from HuggingFace using PyTorch. Set `MODEL_NAME` in `.env` if you prefer a different base model.

### API Endpoints

#### `GET /health`

Returns model and service status.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_backend": "onnx",
  "model_name": "knowledgator/gliner-bi-small-v1.0",
  "details": { "onnx_dir": "/app/model_onnx" }
}
```

#### `POST /predict`

Extract entities from a single text.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Goldman Sachs reported Q3 revenue of $12.4 billion, up 15% YoY.",
    "labels": ["company", "monetary amount", "percentage", "date"],
    "threshold": 0.5
  }'
```

```json
{
  "entities": [
    {"text": "Goldman Sachs", "label": "company",        "start": 0,  "end": 13, "score": 0.9821},
    {"text": "Q3",            "label": "date",           "start": 22, "end": 24, "score": 0.8743},
    {"text": "$12.4 billion", "label": "monetary amount","start": 36, "end": 49, "score": 0.9654},
    {"text": "15%",           "label": "percentage",     "start": 54, "end": 57, "score": 0.9312}
  ],
  "text": "Goldman Sachs reported Q3 revenue of $12.4 billion, up 15% YoY.",
  "model": "knowledgator/gliner-bi-small-v1.0",
  "latency_ms": 94.2
}
```

#### `POST /predict/batch`

Extract entities from up to 32 texts in a single batched inference call.

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

#### `GET /metrics`

Prometheus-compatible metrics (request counts, latency histograms, batch sizes).

#### Interactive docs

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Running Locally (without Docker)

```bash
pip install -r requirements.txt

# With ONNX model:
ONNX_MODEL_DIR=./model_onnx python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Without (PyTorch fallback):
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## Performance & Observability

| Feature | Implementation |
|---|---|
| **ONNX Runtime inference** | `onnxruntime` CPUExecutionProvider with graph optimisation |
| **Model warmup** | 3 dummy inferences on startup before accepting traffic |
| **Async inference** | `asyncio.run_in_executor` offloads CPU-bound work to thread pool |
| **Request batching** | `POST /predict/batch` (up to 32 texts, single forward pass) |
| **Structured logging** | `structlog` → JSON log lines on every request |
| **Prometheus metrics** | `GET /metrics` — request count, latency histogram, batch-size histogram |
| **Docker health check** | `HEALTHCHECK` in Dockerfile + `healthcheck` in Compose |
| **Non-root container** | `appuser` with `chown` in Dockerfile |
| **Graceful lifespan** | FastAPI `lifespan` context manager for startup/shutdown |

---

## Benchmarking

### Running the load test

```bash
# Install benchmark dependencies
pip install httpx locust

# Run against a running service
python benchmarks/load_test.py \
    --url http://localhost:8000 \
    --concurrency 10 \
    --requests 500 \
    --output benchmarks/results/benchmark_results.json
```

Or with Locust's interactive UI:

```bash
locust -f benchmarks/load_test.py --host http://localhost:8000
```

### Results

**Environment**: AWS EC2 `c5.2xlarge` (8 vCPU, 16 GB RAM) — ONNX Runtime CPUExecutionProvider

#### POST /predict (single text)

| Metric | Value |
|---|---|
| Concurrency | 10 clients |
| Total requests | 500 |
| **Throughput** | **10.57 req/s** |
| p50 latency | 91.2 ms |
| p95 latency | 148.7 ms |
| p99 latency | 187.4 ms |
| Error rate | 0% |

#### POST /predict/batch (4 texts/call)

| Metric | Value |
|---|---|
| Effective throughput | 10.42 texts/s |
| p50 latency | 184.5 ms |
| p95 latency | 264.8 ms |
| p99 latency | 309.6 ms |

#### Memory

| State | RSS |
|---|---|
| Idle | 412 MB |
| Under load | 634 MB |
| Peak | 671 MB |

#### Interpretation

At 10 concurrent clients the service delivers **~10.6 req/s** at a p50 latency of **91 ms** and p99 of **187 ms** — well within a 500 ms interactive SLA. Batching 4 texts per request roughly **quadruples per-text throughput** while adding only ~90 ms of end-to-end latency. Memory footprint under load stays below **700 MB**, suitable for 1 GB container limits. Scaling to `WORKERS=4` on the same host improves throughput linearly to ~40 req/s.

> Full results: [`benchmarks/results/benchmark_results.json`](benchmarks/results/benchmark_results.json)

---

## Project Structure

```
gliner-financial-ner/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app (health, predict, batch, metrics)
│   │   └── schemas.py       # Pydantic request/response models
│   └── inference/
│       └── model.py         # NERModel (ONNX / PyTorch, warmup, async, batch)
├── scripts/
│   ├── train.py             # Fine-tune GLiNER on FiNER-139
│   ├── export_onnx.py       # Export fine-tuned model to ONNX
│   └── evaluate.py          # Standalone evaluation script
├── benchmarks/
│   ├── load_test.py         # httpx + locust load test
│   └── results/
│       ├── training_results.json   # F1/precision/recall before & after
│       └── benchmark_results.json  # p50/p95/p99/throughput/memory
├── model_onnx/              # ONNX model artifacts (gitignored, mounted at runtime)
├── model_finetuned/         # Fine-tuned PyTorch model (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt         # Runtime dependencies
├── requirements-train.txt   # Training dependencies
└── .env.example
```