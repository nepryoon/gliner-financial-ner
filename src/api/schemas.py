"""Pydantic v2 request and response schemas for the GLiNER NER API.

This module defines all data models used for input validation and response
serialisation across the three main endpoints: ``/predict``, ``/predict/batch``,
and ``/health``.  Field-level constraints (``min_length``, ``ge``/``le``) are
enforced automatically by FastAPI before the handler is invoked, so handlers
can assume all incoming data is already valid.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for ``POST /predict``.

    Encapsulates a single text string together with the entity labels the
    caller wants to detect and the minimum confidence threshold to apply.
    All fields except ``text`` are optional; sensible defaults covering the
    eleven standard financial entity types are provided.
    """

    text: str = Field(..., min_length=1, description="Input text for NER extraction")
    labels: list[str] = Field(
        default=[
            "company",
            "person",
            "financial instrument",
            "currency",
            "monetary amount",
            "percentage",
            "date",
            "location",
            "organization",
            "stock ticker",
            "index",
        ],
        description="Entity labels to detect",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for entity extraction",
    )


class Entity(BaseModel):
    """A single named entity extracted from input text.

    Character offsets follow Python slice conventions: ``text[start:end]``
    reproduces the exact surface form stored in ``text``.  ``score`` is the
    raw GLiNER span confidence value rounded to four decimal places.
    """

    text: str = Field(..., description="Surface form of the entity as it appears in the input text")
    label: str = Field(..., description="Entity type label assigned by the model")
    start: int = Field(..., description="Character offset of the first character (inclusive)")
    end: int = Field(..., description="Character offset of the character after the last (exclusive)")
    score: float = Field(..., description="Model confidence score in [0, 1], rounded to 4 d.p.")


class PredictResponse(BaseModel):
    """Response body for ``POST /predict``.

    Returns all entities found in the input text together with metadata
    that allows the caller to correlate the response with the original
    request and to monitor per-request latency.
    """

    entities: list[Entity] = Field(..., description="Zero or more extracted entities, in order of appearance")
    text: str = Field(..., description="The input text echoed back for reference")
    model: str = Field(..., description="Identifier of the model that produced the predictions")
    latency_ms: float = Field(..., description="Server-side inference latency in milliseconds")


class BatchPredictRequest(BaseModel):
    """Request body for ``POST /predict/batch``.

    Accepts between 1 and 32 texts that are forwarded to GLiNER's native
    ``batch_predict_entities`` method as a single batched forward pass.
    The ``labels`` and ``threshold`` fields apply uniformly to every text
    in the batch.
    """

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of input texts (1–32 items)",
    )
    labels: list[str] = Field(
        default=[
            "company",
            "person",
            "financial instrument",
            "currency",
            "monetary amount",
            "percentage",
            "date",
            "location",
            "organization",
            "stock ticker",
            "index",
        ],
        description="Entity labels to detect; applied to every text in the batch",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold applied uniformly across all texts",
    )


class BatchPredictResponse(BaseModel):
    """Response body for ``POST /predict/batch``.

    Contains one ``PredictResponse`` per input text in the same order as
    the request ``texts`` list, plus the aggregate latency for the entire
    batch.  The ``latency_ms`` field within each nested ``PredictResponse``
    is the total batch latency divided evenly across all texts; it is not
    individually measured.
    """

    results: list[PredictResponse] = Field(
        ...,
        description="One PredictResponse per input text, preserving request order",
    )
    total_latency_ms: float = Field(
        ...,
        description="Total server-side latency for the entire batch in milliseconds",
    )


class HealthResponse(BaseModel):
    """Response body for ``GET /health``.

    Reflects the current operational state of the service without executing
    a live inference call.  The ``model_backend`` field distinguishes between
    the three possible loading modes so that operators can verify whether the
    ONNX session or the PyTorch fallback is active.
    """

    status: str = Field(
        ...,
        description='"ok" when the model is loaded; "degraded" otherwise',
    )
    model_loaded: bool = Field(
        ...,
        description="True when NERModel holds a live model instance",
    )
    model_backend: str = Field(
        ...,
        description=(
            'Active inference backend: "onnx", "pytorch_from_onnx_dir", '
            '"pytorch", or "unloaded"'
        ),
    )
    model_name: str = Field(
        ...,
        description="Value of the MODEL_NAME environment variable",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic context; currently contains onnx_dir",
    )
