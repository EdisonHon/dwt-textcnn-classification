"""MCCNN training, checkpoint selection, and saved-run evaluation."""

from __future__ import annotations

import csv
import json
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Adadelta
from torch.utils.data import DataLoader

from dwt_textcnn.data import EncodedSSTDataset, SSTBatchCollator, sha256_file
from dwt_textcnn.embeddings import load_original_embedding_cache
from dwt_textcnn.evaluation import evaluate_model
from dwt_textcnn.models.mccnn import MCCNN
from dwt_textcnn.reproducibility import capture_environment, seed_everything
from dwt_textcnn.vocabulary import Vocabulary


class TrainingError(ValueError):
    """Raised when training inputs or saved runs are inconsistent."""


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but is not available")
    return device


def build_mccnn(
    config: Mapping[str, Any], pretrained_matrix: np.ndarray | torch.Tensor, pad_id: int
) -> MCCNN:
    weights = torch.as_tensor(np.asarray(pretrained_matrix).copy(), dtype=torch.float32)
    model_config = config["model"]
    return MCCNN(
        weights,
        pad_id=pad_id,
        kernel_sizes=model_config["kernel_sizes"],
        filters_per_size=model_config["filters_per_size"],
        dropout=model_config["dropout"],
        num_classes=model_config["num_classes"],
        initialization_std=model_config["initialization"]["std"],
    )


def _make_loader(
    dataset: EncodedSSTDataset,
    *,
    batch_size: int,
    collator: SSTBatchCollator,
    shuffle: bool,
    seed: int,
    workers: int,
    pin_memory: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=workers,
        generator=generator,
        pin_memory=pin_memory,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _save_checkpoint(
    path: Path,
    model: MCCNN,
    *,
    seed: int,
    epoch: int,
    dev_accuracy: float,
    config: Mapping[str, Any],
) -> None:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(
        {
            "schema_version": 1,
            "model_state": state,
            "seed": seed,
            "epoch": epoch,
            "dev_accuracy": dev_accuracy,
            "method": config["embedding"]["method"],
        },
        path,
    )


def load_checkpoint(path: str | Path, model: MCCNN, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise TrainingError(f"Invalid checkpoint {path}: {exc}") from exc
    model.enforce_padding_zero()
    return checkpoint


def _parameter_profile(model: MCCNN) -> dict[str, int]:
    def count(module: nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters())

    return {
        "pretrained_embedding_parameters": count(model.pretrained_embedding),
        "random_embedding_parameters": count(model.random_embedding),
        "convolution_parameters": count(model.convolutions),
        "classifier_parameters": count(model.classifier),
        "total_parameters": count(model),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise TrainingError(
            "run_id must contain only letters, digits, dots, underscores, and hyphens"
        )


def train_experiment(
    config: Mapping[str, Any],
    *,
    seed: int,
    prepared_data_dir: str | Path,
    embedding_cache_dir: str | Path,
    output_root: str | Path,
    device_name: str = "auto",
    run_id: str | None = None,
    max_epochs: int | None = None,
    limit_train_batches: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Train one Original experiment and evaluate its best checkpoint."""

    if config["embedding"]["method"] != "original":
        raise TrainingError("Phase 4 training supports the Original method only")
    if seed not in config["experiment"]["seeds"]:
        raise TrainingError(f"Seed {seed} is not in the configured formal seed set")
    if max_epochs is not None and max_epochs < 1:
        raise TrainingError("max_epochs must be positive")
    if limit_train_batches is not None and limit_train_batches < 1:
        raise TrainingError("limit_train_batches must be positive")

    data_dir = Path(prepared_data_dir).expanduser().resolve()
    embedding_dir = Path(embedding_cache_dir).expanduser().resolve()
    vocabulary = Vocabulary.load(data_dir / "vocabulary.json")
    data_manifest = json.loads((data_dir / "data_manifest.json").read_text(encoding="utf-8"))
    if data_manifest["vocabulary"]["sha256"] != vocabulary.sha256:
        raise TrainingError("Prepared vocabulary does not match its data manifest")
    pretrained_matrix, _, embedding_manifest = load_original_embedding_cache(
        embedding_dir, expected_vocabulary_sha256=vocabulary.sha256
    )
    if embedding_manifest["seed"] != seed:
        raise TrainingError(
            f"Embedding seed is {embedding_manifest['seed']}, requested training seed is {seed}"
        )

    datasets = {
        split: EncodedSSTDataset.from_jsonl(data_dir / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    for split, dataset in datasets.items():
        expected = config["data"]["expected_binary_counts"][split]
        if len(dataset) != expected:
            raise TrainingError(
                f"{split} dataset count mismatch: expected {expected}, got {len(dataset)}"
            )

    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _validate_run_id(resolved_run_id)
    run_dir = (
        Path(output_root).expanduser().resolve()
        / config["project"]["dataset"]
        / config["embedding"]["method"]
        / str(seed)
        / resolved_run_id
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise TrainingError(f"Run directory already contains files: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    effective_config = deepcopy(dict(config))
    effective_epochs = max_epochs or config["training"]["max_epochs"]
    effective_config["training"]["max_epochs"] = effective_epochs
    run_kind = (
        "smoke"
        if max_epochs is not None or limit_train_batches is not None
        else "formal"
    )
    effective_config["run"] = {
        "seed": seed,
        "kind": run_kind,
        "run_id": resolved_run_id,
        "prepared_data_dir": str(data_dir),
        "embedding_cache_dir": str(embedding_dir),
        "device_requested": device_name,
        "limit_train_batches": limit_train_batches,
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(effective_config, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "data_manifest.json").write_bytes(data_dir.joinpath("data_manifest.json").read_bytes())
    (run_dir / "embedding_manifest.json").write_bytes(
        embedding_dir.joinpath("embedding_manifest.json").read_bytes()
    )

    deterministic = config["training"]["deterministic"]
    seed_everything(seed, deterministic=deterministic)
    device = resolve_device(device_name)
    collator = SSTBatchCollator(
        pad_id=vocabulary.pad_id,
        minimum_length=config["data"]["minimum_sequence_length"],
    )
    loader_arguments = {
        "batch_size": config["training"]["batch_size"],
        "collator": collator,
        "seed": seed,
        "workers": config["training"]["dataloader_workers"],
        "pin_memory": device.type == "cuda",
    }
    train_loader = _make_loader(datasets["train"], shuffle=True, **loader_arguments)
    dev_loader = _make_loader(datasets["dev"], shuffle=False, **loader_arguments)
    test_loader = _make_loader(datasets["test"], shuffle=False, **loader_arguments)

    model = build_mccnn(config, pretrained_matrix, vocabulary.pad_id).to(device)
    del pretrained_matrix
    optimizer_config = config["training"]["optimizer"]
    optimizer = Adadelta(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        rho=optimizer_config["rho"],
        eps=optimizer_config["eps"],
    )
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    patience = config["training"]["early_stopping"]["patience"]
    history: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "best_model.pt"
    training_started = time.perf_counter()

    for epoch in range(1, effective_epochs + 1):
        model.train()
        _synchronize(device)
        epoch_started = time.perf_counter()
        total_loss = 0.0
        sample_count = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            model.enforce_padding_zero()
            total_loss += float(loss.item()) * len(labels)
            sample_count += len(labels)
            if limit_train_batches is not None and batch_index >= limit_train_batches:
                break
        _synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        dev_metrics = evaluate_model(model, dev_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / sample_count,
            "train_samples": sample_count,
            "dev_loss": dev_metrics["loss"],
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)

        if dev_metrics["accuracy"] > best_accuracy:
            best_accuracy = dev_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoint_path,
                model,
                seed=seed,
                epoch=epoch,
                dev_accuracy=best_accuracy,
                config=config,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    _synchronize(device)
    training_seconds = time.perf_counter() - training_started
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    dev_metrics = evaluate_model(model, dev_loader, device, criterion)
    test_metrics = evaluate_model(model, test_loader, device, criterion)

    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "schema_version": 1,
        "method": config["embedding"]["method"],
        "seed": seed,
        "run_kind": run_kind,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "early_stopped": len(history) < effective_epochs,
        "dev": dev_metrics,
        "test": test_metrics,
        "paper_mccnn_accuracy": 0.881,
        "test_accuracy_difference_from_paper": test_metrics["accuracy"] - 0.881,
        "training_seconds": training_seconds,
        "checkpoint_dev_accuracy": checkpoint["dev_accuracy"],
    }
    _write_json(run_dir / "metrics.json", metrics)

    profile = {
        "schema_version": 1,
        "status": "phase-4-model-size-only",
        "device": str(device),
        "parameters": _parameter_profile(model),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    _write_json(run_dir / "profile.json", profile)
    environment = capture_environment()
    environment.update(
        {
            "device_selected": str(device),
            "deterministic_algorithms": deterministic,
        }
    )
    _write_json(run_dir / "environment.json", environment)
    return run_dir, metrics


def evaluate_saved_run(
    run_dir: str | Path, *, device_name: str = "auto"
) -> dict[str, Any]:
    """Re-evaluate the test split from a self-contained saved run record."""

    run = Path(run_dir).expanduser().resolve()
    try:
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        context = config["run"]
        seed = context["seed"]
        data_dir = Path(context["prepared_data_dir"])
        embedding_dir = Path(context["embedding_cache_dir"])
    except (FileNotFoundError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise TrainingError(f"Invalid saved run configuration in {run}") from exc

    provenance_pairs = (
        (run / "data_manifest.json", data_dir / "data_manifest.json"),
        (
            run / "embedding_manifest.json",
            embedding_dir / "embedding_manifest.json",
        ),
    )
    for saved_manifest, current_manifest in provenance_pairs:
        if sha256_file(saved_manifest) != sha256_file(current_manifest):
            raise TrainingError(
                f"Current input manifest differs from the saved run: {current_manifest}"
            )

    seed_everything(seed, deterministic=config["training"]["deterministic"])
    device = resolve_device(device_name)
    vocabulary = Vocabulary.load(data_dir / "vocabulary.json")
    pretrained_matrix, _, _ = load_original_embedding_cache(
        embedding_dir, expected_vocabulary_sha256=vocabulary.sha256
    )
    model = build_mccnn(config, pretrained_matrix, vocabulary.pad_id).to(device)
    checkpoint = load_checkpoint(run / "best_model.pt", model, device)
    dataset = EncodedSSTDataset.from_jsonl(data_dir / "test.jsonl")
    loader = _make_loader(
        dataset,
        batch_size=config["training"]["batch_size"],
        collator=SSTBatchCollator(
            vocabulary.pad_id, config["data"]["minimum_sequence_length"]
        ),
        shuffle=False,
        seed=seed,
        workers=config["training"]["dataloader_workers"],
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_model(model, loader, device, nn.CrossEntropyLoss())
    return {
        "seed": seed,
        "best_epoch": checkpoint["epoch"],
        "test": metrics,
        "checkpoint_sha256": sha256_file(run / "best_model.pt"),
    }
