"""ONNX-backed GLiNER inference with optional PyTorch fallback."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Default model identifiers
DEFAULT_MODEL_NAME = "knowledgator/gliner-bi-small-v1.0"
ONNX_MODEL_DIR = os.environ.get("ONNX_MODEL_DIR", "model_onnx")


class NERModel:
    """Wrapper around GLiNER that uses ONNX Runtime when available."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        onnx_model_dir: str = ONNX_MODEL_DIR,
    ) -> None:
        self.model_name = model_name
        self.onnx_model_dir = onnx_model_dir
        self._model: Any = None
        self._backend: str = "unloaded"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model (ONNX if available, otherwise PyTorch)."""
        if self._try_load_onnx():
            return
        self._load_pytorch()

    def _try_load_onnx(self) -> bool:
        onnx_path = os.path.join(self.onnx_model_dir, "model.onnx")
        if not os.path.isfile(onnx_path):
            logger.info(
                "onnx_model_not_found",
                path=onnx_path,
                fallback="pytorch",
            )
            return False

        try:
            from gliner import GLiNER  # type: ignore[import]

            logger.info("loading_onnx_model", path=self.onnx_model_dir)
            self._model = GLiNER.from_pretrained(
                self.onnx_model_dir,
                load_tokenizer=True,
            )
            # Switch the model's inference to ONNX Runtime
            self._model.set_sampling_params(
                max_width=12,
                max_neg_type_ratio=1,
            )
            try:
                import onnxruntime as ort  # type: ignore[import]

                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                self._ort_session = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
                self._backend = "onnx"
                logger.info("onnx_model_loaded", path=onnx_path)
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("onnxruntime_init_failed", error=str(exc))
                # Still use the GLiNER model but with PyTorch
                self._backend = "pytorch_from_onnx_dir"
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("onnx_load_failed", error=str(exc))
            return False

    def _load_pytorch(self) -> None:
        from gliner import GLiNER  # type: ignore[import]

        logger.info("loading_pytorch_model", model_name=self.model_name)
        self._model = GLiNER.from_pretrained(self.model_name)
        self._backend = "pytorch"
        logger.info("pytorch_model_loaded", model_name=self.model_name)

    def warmup(self, labels: list[str] | None = None) -> None:
        """Run a few dummy inferences to warm up the model/JIT."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        warmup_labels = labels or ["company", "person", "monetary amount"]
        warmup_texts = [
            "Apple Inc. reported revenues of $90 billion in Q3 2024.",
            "Goldman Sachs CEO David Solomon discussed interest rates.",
            "The S&P 500 index fell 2% amid rising inflation concerns.",
        ]
        logger.info("model_warmup_start", num_texts=len(warmup_texts))
        t0 = time.perf_counter()
        for text in warmup_texts:
            self._model.predict_entities(text, warmup_labels, threshold=0.5)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("model_warmup_done", elapsed_ms=round(elapsed, 2))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Synchronous NER prediction."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        raw = self._model.predict_entities(text, labels, threshold=threshold)
        return [
            {
                "text": e["text"],
                "label": e["label"],
                "start": e["start"],
                "end": e["end"],
                "score": round(float(e["score"]), 4),
            }
            for e in raw
        ]

    async def predict_async(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Async wrapper — offloads CPU-bound work to a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.predict, text, labels, threshold
        )

    def predict_batch(
        self,
        texts: list[str],
        labels: list[str],
        threshold: float = 0.5,
    ) -> list[list[dict[str, Any]]]:
        """Batch inference (single forward pass when backend supports it)."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        # GLiNER supports batch prediction natively
        batch_results = self._model.batch_predict_entities(
            texts, labels, threshold=threshold
        )
        return [
            [
                {
                    "text": e["text"],
                    "label": e["label"],
                    "start": e["start"],
                    "end": e["end"],
                    "score": round(float(e["score"]), 4),
                }
                for e in entities
            ]
            for entities in batch_results
        ]

    async def predict_batch_async(
        self,
        texts: list[str],
        labels: list[str],
        threshold: float = 0.5,
    ) -> list[list[dict[str, Any]]]:
        """Async batch inference."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.predict_batch, texts, labels, threshold
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def backend(self) -> str:
        return self._backend
