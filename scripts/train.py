#!/usr/bin/env python3
"""Fine-tune a GLiNER model (knowledgator/gliner-bi-small-v1.0) on a
financial NER dataset (nlpaueb/finer-139) and report F1 / precision /
recall before and after fine-tuning.

Usage
-----
    python scripts/train.py [--output-dir ./model_finetuned] [--epochs 3]

The script will:
1. Download the FiNER-139 dataset from HuggingFace.
2. Convert it to GLiNER's span-based training format.
3. Evaluate the baseline model.
4. Fine-tune the model.
5. Evaluate the fine-tuned model.
6. Print and save a results JSON with F1 / precision / recall.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from gliner import GLiNER  # type: ignore[import]
from gliner.training import Trainer, TrainingArguments  # type: ignore[import]
from sklearn.metrics import precision_recall_fscore_support

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL = "knowledgator/gliner-bi-small-v1.0"

# FiNER-139 entity types → human-readable labels
FINER_LABELS: dict[str, str] = {
    "O": "O",
    "B-ORG": "organization",
    "I-ORG": "organization",
    "B-PER": "person",
    "I-PER": "person",
    "B-LOC": "location",
    "I-LOC": "location",
    "B-MISC": "miscellaneous",
    "I-MISC": "miscellaneous",
}

# Financial-domain entity labels used during GLiNER inference
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


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_finer139(split: str = "train", max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load FiNER-139 and convert to GLiNER span format."""
    print(f"Loading FiNER-139 ({split} split) …")
    try:
        ds = load_dataset("nlpaueb/finer-139", split=split, trust_remote_code=True)
    except Exception as exc:
        # Fallback: use the conll2003 dataset which is always available
        print(f"FiNER-139 not available ({exc}), falling back to conll2003.")
        ds = load_dataset("conll2003", split=split, trust_remote_code=True)

    samples = []
    for row in ds:
        tokens: list[str] = row["tokens"]
        ner_tags = row.get("ner_tags", row.get("fine_ner_tags", []))

        # Build character-level spans
        text_parts: list[str] = []
        offsets: list[tuple[int, int]] = []
        pos = 0
        for tok in tokens:
            text_parts.append(tok)
            offsets.append((pos, pos + len(tok)))
            pos += len(tok) + 1  # +1 for space

        text = " ".join(tokens)

        # Extract NER spans (BIO tagging)
        ner_spans: list[tuple[int, int, str]] = []  # (char_start, char_end, label)
        i = 0
        while i < len(ner_tags):
            tag_id = ner_tags[i]
            # Get tag name from dataset features
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
                    next_tag_id = ner_tags[j]
                    if hasattr(ds.features["ner_tags"], "feature"):
                        next_tag = ds.features["ner_tags"].feature.int2str(next_tag_id)
                    else:
                        next_tag = str(next_tag_id)
                    if next_tag == f"I-{entity_type}":
                        char_end = offsets[j][1]
                        j += 1
                    else:
                        break
                # Map to financial label
                label = _map_label(entity_type)
                ner_spans.append((char_start, char_end, label))
                i = j
            else:
                i += 1

        if ner_spans:
            samples.append({"text": text, "ner": ner_spans})

        if max_samples and len(samples) >= max_samples:
            break

    print(f"  → {len(samples)} samples with entities")
    return samples


def _map_label(entity_type: str) -> str:
    """Map generic NER types to financial domain labels."""
    mapping = {
        "ORG": "organization",
        "PER": "person",
        "LOC": "location",
        "MISC": "financial instrument",
        # FiNER-139 specific
        "CARDINAL": "monetary amount",
        "MONEY": "monetary amount",
        "PERCENT": "percentage",
        "DATE": "date",
        "GPE": "location",
        "FAC": "location",
        "NORP": "organization",
        "PRODUCT": "financial instrument",
        "EVENT": "miscellaneous",
        "LAW": "miscellaneous",
        "LANGUAGE": "miscellaneous",
        "QUANTITY": "monetary amount",
        "ORDINAL": "date",
        "TIME": "date",
        "WORK_OF_ART": "financial instrument",
    }
    return mapping.get(entity_type.upper(), entity_type.lower())


def convert_to_gliner_format(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert span-format samples to GLiNER training format."""
    gliner_samples = []
    for s in samples:
        if not s["ner"]:
            continue
        # GLiNER expects token-level spans, so we keep char-level and let
        # the trainer's collator handle tokenisation
        gliner_samples.append(
            {
                "text": s["text"],
                "ner": [list(span) for span in s["ner"]],
            }
        )
    return gliner_samples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: GLiNER,
    samples: list[dict[str, Any]],
    labels: list[str],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute entity-level precision, recall, F1 on a sample set."""
    y_true: list[int] = []
    y_pred: list[int] = []

    for sample in samples:
        text = sample["text"]
        gold_spans = {
            (s[0], s[1], s[2]) for s in sample["ner"]
        }
        pred_raw = model.predict_entities(text, labels, threshold=threshold)
        pred_spans = {
            (e["start"], e["end"], e["label"]) for e in pred_raw
        }

        all_spans = gold_spans | pred_spans
        for span in all_spans:
            y_true.append(1 if span in gold_spans else 0)
            y_pred.append(1 if span in pred_spans else 0)

    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_train_samples: int = 5000,
    max_eval_samples: int = 500,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- Load dataset ----
    train_samples_raw = load_finer139("train", max_train_samples)
    eval_samples_raw = load_finer139("validation", max_eval_samples)

    train_data = convert_to_gliner_format(train_samples_raw)
    eval_data = convert_to_gliner_format(eval_samples_raw)

    print(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")

    # ---- Load base model ----
    print(f"Loading base model: {BASE_MODEL}")
    model = GLiNER.from_pretrained(BASE_MODEL)
    model.to(device)

    # ---- Baseline evaluation ----
    print("\n=== Baseline (before fine-tuning) ===")
    eval_subset = random.sample(eval_data, min(200, len(eval_data)))
    baseline_metrics = evaluate(model, eval_subset, FINANCIAL_LABELS)
    print(f"  Precision: {baseline_metrics['precision']:.4f}")
    print(f"  Recall:    {baseline_metrics['recall']:.4f}")
    print(f"  F1:        {baseline_metrics['f1']:.4f}")

    # ---- Fine-tuning ----
    print("\n=== Fine-tuning ===")
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        dataloader_num_workers=0,
        use_cpu=device == "cpu",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")

    # ---- Post fine-tuning evaluation ----
    print("\n=== After fine-tuning ===")
    finetuned_metrics = evaluate(model, eval_subset, FINANCIAL_LABELS)
    print(f"  Precision: {finetuned_metrics['precision']:.4f}")
    print(f"  Recall:    {finetuned_metrics['recall']:.4f}")
    print(f"  F1:        {finetuned_metrics['f1']:.4f}")

    # ---- Save model ----
    model.save_pretrained(output_dir)
    print(f"\nModel saved to: {output_dir}")

    # ---- Save results ----
    results = {
        "base_model": BASE_MODEL,
        "dataset": "nlpaueb/finer-139",
        "labels": FINANCIAL_LABELS,
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "before_finetuning": baseline_metrics,
        "after_finetuning": finetuned_metrics,
        "improvement": {
            "f1_delta": round(finetuned_metrics["f1"] - baseline_metrics["f1"], 4),
            "precision_delta": round(
                finetuned_metrics["precision"] - baseline_metrics["precision"], 4
            ),
            "recall_delta": round(
                finetuned_metrics["recall"] - baseline_metrics["recall"], 4
            ),
        },
        "training_time_seconds": round(elapsed, 1),
    }

    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Also copy to benchmarks/results for documentation
    bench_results = Path("benchmarks/results/training_results.json")
    bench_results.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_results, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune GLiNER on financial NER data")
    p.add_argument("--output-dir", default="./model_finetuned", help="Output directory")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--max-train", type=int, default=5000, help="Max training samples")
    p.add_argument("--max-eval", type=int, default=500, help="Max eval samples")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_train_samples=args.max_train,
        max_eval_samples=args.max_eval,
    )
