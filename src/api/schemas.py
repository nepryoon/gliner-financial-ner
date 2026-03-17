"""Pydantic schemas for request and response validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
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
    text: str
    label: str
    start: int
    end: int
    score: float


class PredictResponse(BaseModel):
    entities: list[Entity]
    text: str
    model: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=32)
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
        ]
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backend: str
    model_name: str
    details: dict[str, Any] = Field(default_factory=dict)
