"""Validation and aggregation of the complete formal experiment matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


class SummaryError(ValueError):
    """Raised when the formal experiment matrix is incomplete or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise SummaryError(f"Invalid or missing result file: {path}") from exc
    if not isinstance(payload, dict):
        raise SummaryError(f"Result file must contain an object: {path}")
    return payload


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _mean_epoch_seconds(path: Path) -> float:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            values = [float(row["epoch_seconds"]) for row in csv.DictReader(stream)]
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise SummaryError(f"Invalid training history: {path}") from exc
    if not values:
        raise SummaryError(f"Training history is empty: {path}")
    return float(np.mean(values))


def summarize_experiments(
    config: Mapping[str, Any],
    *,
    output_root: str | Path,
    run_id: str,
    summary_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Aggregate one exact run ID for all four methods and five formal seeds."""

    methods = ("original", "dwt_ca", "dwt_cd", "pca")
    seeds = tuple(config["experiment"]["seeds"])
    root = Path(output_root).expanduser().resolve() / config["project"]["dataset"]
    destination = Path(summary_dir).expanduser().resolve()
    output_files = (
        destination / "runs.csv",
        destination / "summary.csv",
        destination / "summary.json",
    )
    existing = [path for path in output_files if path.exists()]
    if existing and not force:
        raise SummaryError(
            f"Summary already exists at {existing[0]}; use --force to replace it"
        )

    rows: list[dict[str, Any]] = []
    data_manifest_hash: str | None = None
    training_protocol: dict[str, Any] | None = None
    for method in methods:
        for seed in seeds:
            run = root / method / str(seed) / run_id
            metrics = _read_json(run / "metrics.json")
            profile = _read_json(run / "profile.json")
            try:
                saved_config = yaml.safe_load(
                    (run / "config.yaml").read_text(encoding="utf-8")
                )
                current_data_hash = _read_json(run / "data_manifest.json")[
                    "vocabulary"
                ]["sha256"]
                benchmark = profile["benchmark"]
            except (FileNotFoundError, KeyError, TypeError, yaml.YAMLError) as exc:
                raise SummaryError(f"Incomplete run record: {run}") from exc
            if (
                metrics.get("run_kind") != "formal"
                or metrics.get("method") != method
                or metrics.get("seed") != seed
            ):
                raise SummaryError(f"Run identity mismatch: {run}")
            if benchmark.get("method") != method or benchmark.get("seed") != seed:
                raise SummaryError(f"Benchmark identity mismatch: {run}")
            if data_manifest_hash is None:
                data_manifest_hash = current_data_hash
                training_protocol = saved_config["training"]
            elif (
                current_data_hash != data_manifest_hash
                or saved_config["training"] != training_protocol
            ):
                raise SummaryError("Formal runs do not share data and training protocol")
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "best_epoch": metrics["best_epoch"],
                    "epochs_completed": metrics["epochs_completed"],
                    "dev_accuracy": metrics["dev"]["accuracy"],
                    "test_accuracy": metrics["test"]["accuracy"],
                    "test_macro_f1": metrics["test"]["macro_f1"],
                    "training_seconds": metrics["training_seconds"],
                    "epoch_seconds_mean": _mean_epoch_seconds(run / "history.csv"),
                    "embedding_dimension": profile["embedding_dimension"],
                    "embedding_compression_ratio": profile[
                        "embedding_compression_ratio_from_300d"
                    ],
                    "pretrained_embedding_parameters": profile["parameters"][
                        "pretrained_embedding_parameters"
                    ],
                    "random_embedding_parameters": profile["parameters"][
                        "random_embedding_parameters"
                    ],
                    "convolution_parameters": profile["parameters"][
                        "convolution_parameters"
                    ],
                    "classifier_parameters": profile["parameters"][
                        "classifier_parameters"
                    ],
                    "total_parameters": profile["parameters"]["total_parameters"],
                    "parameter_bytes": profile["parameter_bytes"][
                        "total_parameter_bytes"
                    ],
                    "embedding_cache_bytes": profile["embedding_cache_size_bytes"],
                    "checkpoint_bytes": profile["checkpoint_size_bytes"],
                    "latency_mean_ms": benchmark["latency_ms"]["mean"],
                    "throughput_samples_per_second": benchmark[
                        "throughput_samples_per_second"
                    ],
                    "cpu_sampled_peak_rss_bytes": benchmark["cpu_process_rss"][
                        "sampled_peak_bytes"
                    ],
                    "cuda_peak_allocated_bytes": (
                        benchmark["cuda_memory"]["peak_allocated_bytes"]
                        if benchmark["cuda_memory"] is not None
                        else None
                    ),
                    "run_dir": str(run),
                }
            )

    aggregate_rows: list[dict[str, Any]] = []
    numeric_metrics = (
        "test_accuracy",
        "test_macro_f1",
        "training_seconds",
        "epoch_seconds_mean",
        "latency_mean_ms",
        "throughput_samples_per_second",
        "checkpoint_bytes",
        "cpu_sampled_peak_rss_bytes",
    )
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        aggregate: dict[str, Any] = {
            "method": method,
            "seed_count": len(method_rows),
            "embedding_dimension": method_rows[0]["embedding_dimension"],
            "embedding_compression_ratio": method_rows[0][
                "embedding_compression_ratio"
            ],
            "pretrained_embedding_parameters": method_rows[0][
                "pretrained_embedding_parameters"
            ],
            "random_embedding_parameters": method_rows[0][
                "random_embedding_parameters"
            ],
            "convolution_parameters": method_rows[0]["convolution_parameters"],
            "classifier_parameters": method_rows[0]["classifier_parameters"],
            "total_parameters": method_rows[0]["total_parameters"],
            "parameter_bytes": method_rows[0]["parameter_bytes"],
            "embedding_cache_bytes": method_rows[0]["embedding_cache_bytes"],
        }
        for metric in numeric_metrics:
            mean, std = _mean_std([float(row[metric]) for row in method_rows])
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
        cuda_values = [row["cuda_peak_allocated_bytes"] for row in method_rows]
        if all(value is not None for value in cuda_values):
            mean, std = _mean_std([float(value) for value in cuda_values])
            aggregate["cuda_peak_allocated_bytes_mean"] = mean
            aggregate["cuda_peak_allocated_bytes_std"] = std
        else:
            aggregate["cuda_peak_allocated_bytes_mean"] = None
            aggregate["cuda_peak_allocated_bytes_std"] = None
        aggregate_rows.append(aggregate)

    original = aggregate_rows[0]
    for aggregate in aggregate_rows:
        aggregate["test_accuracy_difference_from_original"] = (
            aggregate["test_accuracy_mean"] - original["test_accuracy_mean"]
        )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "methods": list(methods),
        "seeds": list(seeds),
        "standard_deviation": "sample (ddof=1)",
        "paper_mccnn_accuracy": 0.881,
        "original_accuracy_difference_from_paper": original[
            "test_accuracy_mean"
        ]
        - 0.881,
        "data_vocabulary_sha256": data_manifest_hash,
        "aggregates": aggregate_rows,
        "runs": rows,
    }
    destination.mkdir(parents=True, exist_ok=True)
    for path, records in (
        (destination / "runs.csv", rows),
        (destination / "summary.csv", aggregate_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
