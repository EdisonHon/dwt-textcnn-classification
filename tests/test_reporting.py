from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dwt_textcnn.config import load_config
from dwt_textcnn.reporting import SummaryError, summarize_experiments


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_formal_matrix(root: Path, run_id: str) -> dict:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    methods = ("original", "dwt_ca", "dwt_cd", "pca")
    for method_index, method in enumerate(methods):
        for seed in config["experiment"]["seeds"]:
            run = root / "sst2" / method / str(seed) / run_id
            run.mkdir(parents=True)
            metrics = {
                "run_kind": "formal",
                "method": method,
                "seed": seed,
                "best_epoch": 2,
                "epochs_completed": 4,
                "dev": {"accuracy": 0.7},
                "test": {
                    "accuracy": 0.8 + method_index / 100 + seed / 100_000,
                    "macro_f1": 0.79 + method_index / 100,
                },
                "training_seconds": 10.0 + method_index,
            }
            profile = {
                "embedding_dimension": 300 if method == "original" else 150,
                "embedding_compression_ratio_from_300d": (
                    0.0 if method == "original" else 0.5
                ),
                "parameters": {
                    "pretrained_embedding_parameters": 200,
                    "random_embedding_parameters": 200,
                    "convolution_parameters": 500,
                    "classifier_parameters": 100,
                    "total_parameters": 1000 - method_index * 100,
                },
                "parameter_bytes": {"total_parameter_bytes": 4000 - method_index * 400},
                "embedding_cache_size_bytes": 100,
                "checkpoint_size_bytes": 200,
                "benchmark": {
                    "method": method,
                    "seed": seed,
                    "latency_ms": {"mean": 1.0 + method_index},
                    "throughput_samples_per_second": 100.0,
                    "cpu_process_rss": {"sampled_peak_bytes": 1000},
                    "cuda_memory": None,
                },
            }
            saved_config = {"training": config["training"]}
            (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (run / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
            (run / "config.yaml").write_text(
                yaml.safe_dump(saved_config), encoding="utf-8"
            )
            (run / "data_manifest.json").write_text(
                json.dumps({"vocabulary": {"sha256": "same-vocabulary"}}),
                encoding="utf-8",
            )
            (run / "history.csv").write_text(
                "epoch,epoch_seconds\n1,1.0\n2,2.0\n", encoding="utf-8"
            )
    return config


def test_complete_matrix_is_summarized_with_sample_standard_deviation(
    tmp_path: Path,
) -> None:
    config = _write_formal_matrix(tmp_path / "outputs", "formal")
    destination = tmp_path / "summary"

    result = summarize_experiments(
        config,
        output_root=tmp_path / "outputs",
        run_id="formal",
        summary_dir=destination,
    )

    assert len(result["runs"]) == 20
    assert len(result["aggregates"]) == 4
    assert result["standard_deviation"] == "sample (ddof=1)"
    assert result["aggregates"][0]["test_accuracy_difference_from_original"] == 0
    assert (destination / "runs.csv").is_file()
    assert (destination / "summary.csv").is_file()
    assert (destination / "summary.json").is_file()


def test_incomplete_matrix_is_rejected(tmp_path: Path) -> None:
    config = _write_formal_matrix(tmp_path / "outputs", "formal")
    missing = tmp_path / "outputs" / "sst2" / "pca" / "100" / "formal" / "metrics.json"
    missing.unlink()

    with pytest.raises(SummaryError, match="missing result"):
        summarize_experiments(
            config,
            output_root=tmp_path / "outputs",
            run_id="formal",
            summary_dir=tmp_path / "summary",
        )
