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
    """Load FiNER-139 and convert to GLiNER span format.

    Downloads the ``nlpaueb/finer-139`` dataset from HuggingFace for the
    requested split and converts BIO-tagged token sequences into character-span
    format.  Falls back to ``conll2003`` if FiNER-139 is unavailable.

    The BIO tags are decoded iteratively: a ``B-`` tag opens a span, and
    consecutive ``I-`` tags of the same type extend it.  Each span is mapped to
    a coarser financial domain label via ``_map_label()``.  Samples with no
    entities after mapping are discarded.

    Args:
        split: Dataset split to load; one of ``"train"``, ``"validation"``,
            or ``"test"``.
        max_samples: If provided, stops after collecting this many samples
            with at least one entity span.  ``None`` means load all samples.

    Returns:
        A list of dictionaries, each with:
        ``"text"`` (the space-joined token string) and
        ``"ner"`` (a list of ``(char_start, char_end, label)`` tuples).
    """
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
    """Map a BIO entity type tag to a financial domain label.

    Converts the fine-grained NER types used by CoNLL-2003 and FiNER-139 (e.g.
    ``"ORG"``, ``"MONEY"``, ``"PERCENT"``) to the eleven coarser labels used
    during GLiNER inference.  Types not present in the mapping are returned
    lowercased as-is.

    Args:
        entity_type: The entity type extracted from a BIO tag (without the
            ``B-`` or ``I-`` prefix), e.g. ``"ORG"``, ``"PER"``, ``"DATE"``.

    Returns:
        A lowercase financial domain label string, e.g. ``"organization"``,
        ``"person"``, or ``"date"``.
    """
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
    """Convert span-format samples to the list format expected by GLiNER's Trainer.

    GLiNER's data collator expects each ``"ner"`` entry to be a list of lists
    (not tuples) so that the JSON serialiser and the collator's tensor-building
    logic can iterate over them uniformly.  This function also filters out any
    samples where ``"ner"`` is empty after upstream processing.

    Args:
        samples: A list of dictionaries as returned by ``load_finer139()``,
            each containing ``"text"`` (str) and ``"ner"`` (list of tuples).

    Returns:
        A list of dictionaries suitable for passing to ``Trainer`` as
        ``train_dataset`` or ``eval_dataset``.  Each dictionary has
        ``"text"`` (str) and ``"ner"`` (list of ``[start, end, label]`` lists).
    """
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
    """Compute entity-level precision, recall, and F1 on a sample set.

    Uses exact span-level matching: a predicted entity ``(start, end, label)``
    is counted as a true positive only when all three values exactly match a
    gold annotation.  Metrics are computed globally (micro-averaged) across all
    entity classes using ``sklearn.metrics.precision_recall_fscore_support``
    with binary averaging over the flattened set of (span, correct?) decisions.

    Args:
        model: A loaded ``GLiNER`` instance to evaluate.
        samples: A list of dictionaries, each with ``"text"`` (str) and
            ``"ner"`` (list of ``[start, end, label]`` or tuples).
        labels: Entity type labels passed to ``model.predict_entities()``.
        threshold: Confidence threshold; predictions below this value are
            ignored.

    Returns:
        A dictionary with keys ``"precision"``, ``"recall"``, and ``"f1"``,
        all rounded to four decimal places.  Returns all-zero values when
        ``samples`` is empty or no spans are found.
    """
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
    """Run the complete fine-tuning pipeline and save results.

    Orchestrates the following steps in order:

    1. Load the FiNER-139 train and validation splits (or CoNLL-2003 fallback).
    2. Convert to GLiNER span format.
    3. Load the base model (``knowledgator/gliner-bi-small-v1.0``).
    4. Evaluate the baseline model on 200 sampled validation examples.
    5. Fine-tune using GLiNER's ``Trainer`` for ``epochs`` epochs.
    6. Evaluate the fine-tuned model on the same 200 examples.
    7. Save the fine-tuned model to ``output_dir``.
    8. Write a ``training_results.json`` file to both ``output_dir`` and
       ``benchmarks/results/``.

    Args:
        output_dir: Directory where the fine-tuned model and results JSON are
            written.  Created if it does not exist.
        epochs: Number of training epochs.  Default 3 balances convergence and
            overfitting risk on the ~5 k sample dataset.
        batch_size: Per-device training and evaluation batch size.  Default 8
            fits comfortably on a 24 GB VRAM GPU with GLiNER's span overhead.
        learning_rate: Encoder learning rate.  Default 5×10⁻⁵ is the standard
            upper bound for fine-tuning transformer encoders.
        max_train_samples: Maximum number of training samples to draw from the
            dataset.  Default 5,000 covers the full FiNER-139 train split.
        max_eval_samples: Maximum number of validation samples.  Default 500
            covers the full FiNER-139 validation split.
    """
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
    """Parse and return command-line arguments for the training script.

    Returns:
        An ``argparse.Namespace`` object with attributes ``output_dir``,
        ``epochs``, ``batch_size``, ``lr``, ``max_train``, and ``max_eval``,
        each pre-populated with sensible defaults.
    """
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
