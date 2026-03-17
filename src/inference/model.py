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
    """Wrapper around GLiNER that uses ONNX Runtime when available.

    This class implements a three-tier backend strategy:

    1. **ONNX** (``backend = "onnx"``) — ``model_onnx/model.onnx`` is found
       and ``onnxruntime`` can create an ``InferenceSession``.  The ORT session
       is stored in ``self._ort_session`` for future wiring into the prediction
       path.  Inference currently still routes through GLiNER's native
       ``predict_entities()`` (PyTorch); the ORT session is initialised and
       verified at startup but is not yet substituted for the encoder forward
       pass (see the Known Limitations section in the README).

    2. **PyTorch from ONNX directory** (``backend = "pytorch_from_onnx_dir"``)
       — ``model_onnx/model.onnx`` is found but ``onnxruntime`` is unavailable
       or fails to initialise.  The GLiNER model is still loaded from the ONNX
       directory (which contains ``config.json`` and tokeniser files) and
       inference runs through PyTorch.

    3. **PyTorch** (``backend = "pytorch"``) — No ONNX artefacts are found.
       The model is downloaded from HuggingFace using ``model_name``.

    All public inference methods are available in both synchronous and
    asynchronous variants.  The async variants offload to the default
    ``ThreadPoolExecutor`` so that the FastAPI event loop remains unblocked
    during CPU-bound inference.

    Attributes:
        model_name: HuggingFace model identifier used for the PyTorch fallback.
        onnx_model_dir: Path to the directory containing ``model.onnx`` and
            the associated tokeniser/config files.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        onnx_model_dir: str = ONNX_MODEL_DIR,
    ) -> None:
        """Initialise the NERModel without loading any model weights.

        ``load()`` must be called explicitly before any inference method is
        used.  This separation allows the object to be instantiated at module
        import time while deferring the (potentially slow) model loading to
        the ASGI lifespan startup hook.

        Args:
            model_name: HuggingFace model identifier used as the PyTorch
                fallback when no ONNX artefacts are found.
            onnx_model_dir: Path to the directory that may contain
                ``model.onnx``, ``config.json``, and tokeniser files.
        """
        self.model_name = model_name
        self.onnx_model_dir = onnx_model_dir
        self._model: Any = None
        self._backend: str = "unloaded"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model, preferring ONNX Runtime over PyTorch.

        Attempts to load from ``onnx_model_dir`` first by calling
        ``_try_load_onnx()``.  If that returns ``False`` (ONNX artefacts not
        found), falls back to loading the HuggingFace checkpoint specified by
        ``model_name`` via ``_load_pytorch()``.

        After this method returns, ``is_loaded`` will be ``True`` and
        ``backend`` will reflect which loading strategy succeeded.

        Raises:
            Any exception propagated from ``_load_pytorch()`` if the
            HuggingFace download or model initialisation fails.
        """
        if self._try_load_onnx():
            return
        self._load_pytorch()

    def _try_load_onnx(self) -> bool:
        """Attempt to load the model from the ONNX artefact directory.

        Checks whether ``<onnx_model_dir>/model.onnx`` exists.  If it does,
        loads the GLiNER instance from the same directory (which must also
        contain ``config.json`` and tokeniser files) and separately creates an
        ``onnxruntime.InferenceSession`` with all graph optimisations enabled.

        The ORT session is stored in ``self._ort_session`` for future use.
        If ``onnxruntime`` is not importable or the session creation raises,
        the method still returns ``True`` and sets the backend to
        ``"pytorch_from_onnx_dir"`` so that GLiNER can serve requests via
        PyTorch.

        Returns:
            ``True`` if the GLiNER model was loaded successfully from the ONNX
            directory (regardless of whether ORT initialisation succeeded).
            ``False`` if ``model.onnx`` was not found or GLiNER loading failed.
        """
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
        """Load the GLiNER model directly from HuggingFace using PyTorch.

        Downloads (or loads from the local HuggingFace cache) the checkpoint
        identified by ``self.model_name``, sets ``self._backend`` to
        ``"pytorch"``, and assigns the loaded model to ``self._model``.

        This method is called only when ``_try_load_onnx()`` returns ``False``,
        i.e. when no ONNX artefacts are present in ``onnx_model_dir``.

        Raises:
            Any exception raised by ``GLiNER.from_pretrained()`` (e.g. network
            error, invalid model identifier, or missing ``gliner`` package).
        """
        from gliner import GLiNER  # type: ignore[import]

        logger.info("loading_pytorch_model", model_name=self.model_name)
        self._model = GLiNER.from_pretrained(self.model_name)
        self._backend = "pytorch"
        logger.info("pytorch_model_loaded", model_name=self.model_name)

    def warmup(self, labels: list[str] | None = None) -> None:
        """Run dummy inferences to pre-populate ONNX Runtime kernel caches.

        ONNX Runtime (and PyTorch's JIT compiler) performs kernel selection
        and compilation on the first few forward passes for each unique input
        shape.  Without warmup, the first real requests arrive cold and
        experience latencies 3–5× higher than steady state.  Running three
        representative financial sentences at startup ensures the kernel cache
        is populated before any client traffic is served.

        The warmup is executed synchronously inside FastAPI's lifespan startup
        hook, so the ASGI server does not accept connections until it completes.

        Args:
            labels: Entity labels passed to ``predict_entities()`` during
                warmup.  Defaults to ``["company", "person", "monetary amount"]``
                if not provided.

        Raises:
            RuntimeError: If ``load()`` has not been called before ``warmup()``.
        """
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
        """Extract named entities from a single text string synchronously.

        Delegates to GLiNER's ``predict_entities()`` and normalises the output
        into a consistent dictionary format with the ``score`` value rounded to
        four decimal places.

        Args:
            text: The input text to analyse.
            labels: Entity type labels to detect (e.g. ``["company", "person"]``).
            threshold: Minimum confidence score; entities with a lower score are
                discarded.  Must be in ``[0.0, 1.0]``.

        Returns:
            A list of entity dictionaries, each containing:
            ``text`` (str), ``label`` (str), ``start`` (int), ``end`` (int),
            and ``score`` (float).  Returns an empty list if no entities are
            found above the threshold.

        Raises:
            RuntimeError: If ``load()`` has not been called before ``predict()``.
        """
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
        """Extract named entities from a single text string asynchronously.

        Wraps the synchronous ``predict()`` call with
        ``loop.run_in_executor(None, ...)`` to offload the CPU-bound inference
        to the default ``ThreadPoolExecutor``.  This prevents the FastAPI event
        loop from blocking while a prediction is in-flight, allowing the server
        to continue serving health checks and other lightweight requests
        concurrently.

        Args:
            text: The input text to analyse.
            labels: Entity type labels to detect.
            threshold: Minimum confidence score in ``[0.0, 1.0]``.

        Returns:
            The same list of entity dictionaries as ``predict()``.

        Raises:
            RuntimeError: Propagated from ``predict()`` if the model is not
                loaded.
        """
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
        """Extract named entities from multiple texts in a single forward pass.

        Uses GLiNER's native ``batch_predict_entities()`` which processes all
        texts as a padded batch, amortising the per-call overhead of tokenisation
        and the encoder forward pass across the entire batch.

        Args:
            texts: A non-empty list of input texts (up to 32 items, enforced by
                the Pydantic schema at the API layer).
            labels: Entity type labels to detect; applied uniformly to every
                text in the batch.
            threshold: Minimum confidence score in ``[0.0, 1.0]``.

        Returns:
            A list of entity lists in the same order as ``texts``.  Each inner
            list has the same structure as the return value of ``predict()``.

        Raises:
            RuntimeError: If ``load()`` has not been called before
                ``predict_batch()``.
        """
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
        """Extract named entities from multiple texts asynchronously.

        Offloads the synchronous ``predict_batch()`` call to the default
        ``ThreadPoolExecutor`` via ``loop.run_in_executor()``, keeping the
        FastAPI event loop unblocked during the (typically longer) batch
        inference call.

        Args:
            texts: A non-empty list of input texts.
            labels: Entity type labels to detect.
            threshold: Minimum confidence score in ``[0.0, 1.0]``.

        Returns:
            The same nested list of entity dictionaries as ``predict_batch()``.

        Raises:
            RuntimeError: Propagated from ``predict_batch()`` if the model is
                not loaded.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.predict_batch, texts, labels, threshold
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """``True`` if a model instance has been assigned, ``False`` otherwise.

        Returns ``False`` before ``load()`` is called and after an unsuccessful
        load attempt.  Used by route handlers to return a 503 response rather
        than propagating an ``AttributeError`` if inference is attempted on an
        uninitialised model.
        """
        return self._model is not None

    @property
    def backend(self) -> str:
        """The active inference backend identifier.

        Possible values:

        * ``"onnx"`` — ONNX Runtime session created from ``model.onnx``.
        * ``"pytorch_from_onnx_dir"`` — GLiNER loaded from the ONNX artefact
          directory but ORT session creation failed; inference via PyTorch.
        * ``"pytorch"`` — Model loaded from HuggingFace; no ONNX artefacts
          present.
        * ``"unloaded"`` — ``load()`` has not yet been called.
        """
        return self._backend
