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
    """Export GLiNER model to ONNX format."""
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
