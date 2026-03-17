#!/usr/bin/env python3
"""Export a fine-tuned GLiNER model to ONNX format for fast CPU inference.

Usage
-----
    python scripts/export_onnx.py \
        --model-dir ./model_finetuned \
        --output-dir ./model_onnx

The exported directory contains:
    model.onnx       — ONNX model weights
    tokenizer/       — HuggingFace tokenizer files
    config.json      — GLiNER model config
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch


def export_onnx(model_dir: str, output_dir: str, opset: int = 14) -> None:
    """Export a fine-tuned GLiNER model's transformer backbone to ONNX format.

    Performs the following steps:

    1. Loads the GLiNER model from ``model_dir`` (including the tokeniser).
    2. Copies the tokeniser files and ``config.json`` to ``output_dir`` so that
       the inference service can load the full GLiNER instance from a single
       directory.
    3. Exports the DeBERTa encoder (``model.model``) to an ONNX file with
       dynamic ``batch`` and ``seq_len`` axes, enabling the session to accept
       inputs of any length at runtime.
    4. Verifies the ONNX output using ONNX Runtime with a dummy forward pass.
    5. Writes ``export_meta.json`` recording the opset version, file size, and
       verification status.

    The exported ``model.onnx`` file contains only the transformer encoder
    backbone, not the GLiNER span classification head.  At inference time,
    GLiNER still manages tokenisation and span scoring; only the encoder
    forward pass is intended to be replaced by the ORT session.

    Args:
        model_dir: Path to the fine-tuned GLiNER model directory (as produced
            by ``scripts/train.py``).
        output_dir: Destination directory for ONNX artefacts.  Created if it
            does not exist.
        opset: ONNX opset version to target.  Default 14 supports all DeBERTa
            operators and is compatible with ONNX Runtime ≥ 1.14.

    Raises:
        FileNotFoundError: If ``model_dir`` does not contain a valid GLiNER
            checkpoint.
        RuntimeError: If ONNX Runtime fails to verify the exported model.
    """
    from gliner import GLiNER  # type: ignore[import]

    print(f"Loading GLiNER model from: {model_dir}")
    model = GLiNER.from_pretrained(model_dir, load_tokenizer=True)
    model.eval()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save tokenizer and config alongside the ONNX model so the inference
    # service can load them without the original model directory.
    # ------------------------------------------------------------------
    print("Saving tokenizer and config …")
    model.save_pretrained(str(out))

    # ------------------------------------------------------------------
    # Export the transformer backbone (ONNX)
    # ------------------------------------------------------------------
    print(f"Exporting ONNX model (opset={opset}) …")

    tokenizer = model.data_processor.transformer_tokenizer
    device = torch.device("cpu")
    model.model.to(device)

    # Build a representative dummy input
    sample_text = (
        "Goldman Sachs reported quarterly revenue of $12.4 billion, "
        "driven by its investment banking division."
    )
    sample_labels = ["company", "monetary amount", "organization"]

    # Tokenize through GLiNER's tokeniser
    tokenized = tokenizer(
        [sample_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    onnx_path = out / "model.onnx"

    # Export only the encoder (transformer backbone)
    with torch.no_grad():
        torch.onnx.export(
            model.model,
            args=(input_ids, attention_mask),
            f=str(onnx_path),
            opset_version=opset,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "last_hidden_state": {0: "batch", 1: "seq_len"},
            },
            do_constant_folding=True,
        )

    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"ONNX model saved: {onnx_path}  ({file_size_mb:.1f} MB)")

    # ------------------------------------------------------------------
    # Verify with ONNX Runtime
    # ------------------------------------------------------------------
    print("Verifying with ONNX Runtime …")
    import onnxruntime as ort  # type: ignore[import]
    import numpy as np

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    feed = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
    outputs = sess.run(None, feed)
    print(f"  ✓ ONNX output shape: {outputs[0].shape}")

    # ------------------------------------------------------------------
    # Save export metadata
    # ------------------------------------------------------------------
    meta = {
        "source_model_dir": str(model_dir),
        "opset_version": opset,
        "onnx_file": "model.onnx",
        "file_size_mb": round(file_size_mb, 2),
        "verified": True,
    }
    with open(out / "export_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExport complete → {output_dir}")
    print("Contents:")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments for the ONNX export script.

    Returns:
        An ``argparse.Namespace`` with attributes ``model_dir``, ``output_dir``,
        and ``opset``.
    """
    p = argparse.ArgumentParser(description="Export fine-tuned GLiNER to ONNX")
    p.add_argument(
        "--model-dir",
        default="./model_finetuned",
        help="Directory with fine-tuned GLiNER model",
    )
    p.add_argument(
        "--output-dir",
        default="./model_onnx",
        help="Directory to write ONNX artifacts",
    )
    p.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(args.model_dir, args.output_dir, args.opset)
