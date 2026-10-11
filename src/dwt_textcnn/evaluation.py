"""Classification prediction and metric computation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn


@torch.inference_mode()
def predict(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Return gold labels and model predictions in dataset order."""

    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    for batch in batches:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        logits = model(input_ids)
        labels.extend(batch["labels"].tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return labels, predictions


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Evaluate loss, accuracy, and binary macro-F1."""

    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    sample_count = 0
    for batch in batches:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        batch_labels = batch["labels"].to(device, non_blocking=True)
        logits = model(input_ids)
        if criterion is not None:
            total_loss += float(criterion(logits, batch_labels).item()) * len(batch_labels)
        labels.extend(batch_labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        sample_count += len(batch_labels)

    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return {
        "loss": total_loss / sample_count if criterion is not None else None,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)
        ),
        "sample_count": sample_count,
    }
