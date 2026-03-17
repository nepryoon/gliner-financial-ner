"""FastAPI application — GLiNER financial NER service.

Endpoints
---------
GET  /health          Health check with model status
POST /predict         Single-text NER extraction
POST /predict/batch   Batch NER extraction (up to 32 texts)

Production features
-------------------
* ONNX Runtime inference backend (PyTorch fallback)
* Model warmup on startup
* Async inference via thread-pool offload
* Request batching (POST /predict/batch)
* Structured JSON logging (structlog)
* Prometheus metrics via /metrics
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Entity,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.inference.model import NERModel

# ---------------------------------------------------------------------------
# Structured logging configuration
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "ner_requests_total",
    "Total NER prediction requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "ner_request_latency_seconds",
    "NER request latency",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
BATCH_SIZE_HISTOGRAM = Histogram(
    "ner_batch_size",
    "Batch sizes for /predict/batch",
    buckets=(1, 2, 4, 8, 16, 32),
)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_model_name = os.environ.get("MODEL_NAME", "knowledgator/gliner-bi-small-v1.0")
_onnx_dir = os.environ.get("ONNX_MODEL_DIR", "model_onnx")

ner_model = NERModel(model_name=_model_name, onnx_model_dir=_onnx_dir)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    log.info("service_startup", model=_model_name, onnx_dir=_onnx_dir)
    ner_model.load()
    ner_model.warmup()
    log.info("service_ready", backend=ner_model.backend)
    yield
    log.info("service_shutdown")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GLiNER Financial NER API",
    description="Named Entity Recognition for financial text using GLiNER.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware — structured request logging
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Response:
    t0 = time.perf_counter()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown",
    )
    response: Response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info(
        "http_request",
        status_code=response.status_code,
        latency_ms=elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Return service health and model status."""
    return HealthResponse(
        status="ok" if ner_model.is_loaded else "degraded",
        model_loaded=ner_model.is_loaded,
        model_backend=ner_model.backend,
        model_name=_model_name,
        details={"onnx_dir": _onnx_dir},
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(req: PredictRequest) -> PredictResponse:
    """Extract named entities from a single text."""
    if not ner_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        entities_raw = await ner_model.predict_async(
            req.text, req.labels, req.threshold
        )
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        log.error("predict_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - t0
    latency_ms = round(elapsed * 1000, 2)
    REQUEST_LATENCY.labels(endpoint="/predict").observe(elapsed)

    return PredictResponse(
        entities=[Entity(**e) for e in entities_raw],
        text=req.text,
        model=_model_name,
        latency_ms=latency_ms,
    )


@app.post(
    "/predict/batch", response_model=BatchPredictResponse, tags=["inference"]
)
async def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Extract named entities from multiple texts in a single batched call."""
    if not ner_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    BATCH_SIZE_HISTOGRAM.observe(len(req.texts))
    t0 = time.perf_counter()
    try:
        batch_raw = await ner_model.predict_batch_async(
            req.texts, req.labels, req.threshold
        )
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="success").inc()
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="error").inc()
        log.error("batch_predict_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - t0
    total_ms = round(elapsed * 1000, 2)
    REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(elapsed)

    per_text_ms = round(total_ms / len(req.texts), 2)
    results = [
        PredictResponse(
            entities=[Entity(**e) for e in entities],
            text=text,
            model=_model_name,
            latency_ms=per_text_ms,
        )
        for text, entities in zip(req.texts, batch_raw)
    ]

    return BatchPredictResponse(results=results, total_latency_ms=total_ms)


@app.get("/metrics", tags=["ops"], include_in_schema=False)
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        workers=int(os.environ.get("WORKERS", 1)),
        log_config=None,  # use structlog
    )
