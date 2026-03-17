#!/usr/bin/env python3
"""Evaluate a GLiNER model on the financial NER test split.

Prints entity-level F1 / precision / recall and optionally saves JSON results.

Usage
-----
    python scripts/evaluate.py --model-dir ./model_finetuned
    python scripts/evaluate.py --model-dir ./model_onnx --onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import classification_report


FINANCIAL_LABELS = [
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


def load_model(model_dir: str, use_onnx: bool = False):  # type: ignore[return]
    """Load GLiNER model from directory."""
    from gliner import GLiNER  # type: ignore[import]

    print(f"Loading model from: {model_dir} (onnx={use_onnx})")
    model = GLiNER.from_pretrained(model_dir, load_tokenizer=True)
    return model


def load_test_data(max_samples: int = 500) -> list[dict[str, Any]]:
    """Load FiNER-139 test split and convert to evaluation format."""
    from datasets import load_dataset  # type: ignore[import]

    try:
        ds = load_dataset("nlpaueb/finer-139", split="test", trust_remote_code=True)
    except Exception:
        ds = load_dataset("conll2003", split="test", trust_remote_code=True)

    samples = []
    for row in ds:
        tokens: list[str] = row["tokens"]
        ner_tags = row.get("ner_tags", row.get("fine_ner_tags", []))

        offsets = []
        pos = 0
        for tok in tokens:
            offsets.append((pos, pos + len(tok)))
            pos += len(tok) + 1

        text = " ".join(tokens)
        ner_spans = []
        i = 0
        while i < len(ner_tags):
            tag_id = ner_tags[i]
            if hasattr(ds.features["ner_tags"], "feature"):
                tag_name = ds.features["ner_tags"].feature.int2str(tag_id)
            else:
                tag_name = str(tag_id)

            if tag_name.startswith("B-"):
                entity_type = tag_name[2:]
                char_start = offsets[i][0]
                char_end = offsets[i][1]
                j = i + 1
                while j < len(ner_tags):
                    next_id = ner_tags[j]
                    if hasattr(ds.features["ner_tags"], "feature"):
                        next_tag = ds.features["ner_tags"].feature.int2str(next_id)
                    else:
                        next_tag = str(next_id)
                    if next_tag == f"I-{entity_type}":
                        char_end = offsets[j][1]
                        j += 1
                    else:
                        break
                ner_spans.append((char_start, char_end, entity_type.lower()))
                i = j
            else:
                i += 1

        if ner_spans:
            samples.append({"text": text, "ner": ner_spans})
        if len(samples) >= max_samples:
            break

    return samples


def evaluate(
    model: Any,
    samples: list[dict[str, Any]],
    labels: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Entity-level evaluation."""
    y_true: list[int] = []
    y_pred: list[int] = []
    tp = fp = fn = 0

    for sample in samples:
        gold = {(s[0], s[1]) for s in sample["ner"]}
        pred_raw = model.predict_entities(sample["text"], labels, threshold=threshold)
        pred = {(e["start"], e["end"]) for e in pred_raw}

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GLiNER on financial NER")
    p.add_argument("--model-dir", required=True, help="Model directory")
    p.add_argument("--onnx", action="store_true", help="Use ONNX inference")
    p.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--max-samples", type=int, default=500, help="Max test samples")
    p.add_argument("--output", default=None, help="JSON output path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model_dir, use_onnx=args.onnx)
    samples = load_test_data(args.max_samples)

    print(f"Evaluating on {len(samples)} test samples …")
    metrics = evaluate(model, samples, FINANCIAL_LABELS, threshold=args.threshold)

    print("\n=== Results ===")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  TP={metrics['true_positives']}  FP={metrics['false_positives']}  FN={metrics['false_negatives']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to: {args.output}")
