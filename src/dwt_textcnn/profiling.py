"""Reproducible inference benchmarking for saved MCCNN runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml
from torch.utils.data import DataLoader

from dwt_textcnn.data import EncodedSSTDataset, SSTBatchCollator, sha256_file
from dwt_textcnn.embeddings import load_embedding_cache
from dwt_textcnn.reproducibility import seed_everything
from dwt_textcnn.training import TrainingError, build_mccnn, load_checkpoint, resolve_device
from dwt_textcnn.vocabulary import Vocabulary


class BenchmarkError(ValueError):
    """Raised when a saved run cannot be benchmarked consistently."""


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_saved_run(
    run_dir: str | Path,
    *,
    device_name: str = "auto",
    warmup_iterations: int = 20,
    measured_iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark repeated inference on one fixed full-size test batch."""

    if warmup_iterations < 0 or measured_iterations < 1:
        raise BenchmarkError("Warmup must be non-negative and measured iterations positive")
    run = Path(run_dir).expanduser().resolve()
    try:
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        context = config["run"]
        seed = int(context["seed"])
        data_dir = Path(context["prepared_data_dir"])
        embedding_dir = Path(context["embedding_cache_dir"])
        method = config["embedding"]["method"]
    except (FileNotFoundError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise BenchmarkError(f"Invalid saved run configuration in {run}") from exc

    for saved, current in (
        (run / "data_manifest.json", data_dir / "data_manifest.json"),
        (run / "embedding_manifest.json", embedding_dir / "embedding_manifest.json"),
    ):
        if sha256_file(saved) != sha256_file(current):
            raise BenchmarkError(f"Current input manifest differs from saved run: {current}")

    seed_everything(seed, deterministic=config["training"]["deterministic"])
    device = resolve_device(device_name)
    vocabulary = Vocabulary.load(data_dir / "vocabulary.json")
    matrix, _, manifest = load_embedding_cache(
        embedding_dir,
        expected_method=method,
        expected_vocabulary_sha256=vocabulary.sha256,
    )
    if manifest["seed"] != seed:
        raise BenchmarkError("Embedding seed does not match saved run")
    model = build_mccnn(config, matrix, vocabulary.pad_id).to(device)
    del matrix
    try:
        load_checkpoint(
            run / "best_model.pt",
            model,
            device,
            expected_method=method,
            expected_seed=seed,
        )
    except TrainingError as exc:
        raise BenchmarkError(str(exc)) from exc
    model.eval()

    dataset = EncodedSSTDataset.from_jsonl(data_dir / "test.jsonl")
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=SSTBatchCollator(
            vocabulary.pad_id, config["data"]["minimum_sequence_length"]
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    batch_size = int(input_ids.shape[0])
    sequence_length = int(input_ids.shape[1])

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(input_ids)
        _synchronize(device)

        process = psutil.Process()
        baseline_rss = process.memory_info().rss
        peak_rss = baseline_rss
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            cuda_baseline = torch.cuda.memory_allocated(device)
        else:
            cuda_baseline = None

        latencies: list[float] = []
        for _ in range(measured_iterations):
            _synchronize(device)
            started = time.perf_counter()
            model(input_ids)
            _synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0)
            peak_rss = max(peak_rss, process.memory_info().rss)

    latency_values = np.asarray(latencies, dtype=np.float64)
    mean_latency_ms = float(latency_values.mean())
    result: dict[str, Any] = {
        "schema_version": 1,
        "method": method,
        "seed": seed,
        "device": str(device),
        "protocol": {
            "split": "test",
            "fixed_batch": "first test batch",
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "dataloader_workers": 0,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "cuda_synchronized": device.type == "cuda",
        },
        "latency_ms": {
            "mean": mean_latency_ms,
            "std": float(latency_values.std(ddof=1))
            if measured_iterations > 1
            else 0.0,
            "median": float(np.median(latency_values)),
            "p95": float(np.percentile(latency_values, 95)),
            "minimum": float(latency_values.min()),
            "maximum": float(latency_values.max()),
        },
        "throughput_samples_per_second": batch_size / (mean_latency_ms / 1000.0),
        "cpu_process_rss": {
            "baseline_bytes": baseline_rss,
            "sampled_peak_bytes": peak_rss,
            "sampled_peak_delta_bytes": peak_rss - baseline_rss,
        },
        "cuda_memory": None,
    }
    if device.type == "cuda":
        result["cuda_memory"] = {
            "baseline_allocated_bytes": cuda_baseline,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }

    profile_path = run / "profile.json"
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"Invalid model profile: {profile_path}") from exc
    profile["status"] = "benchmarked"
    profile["benchmark"] = result
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
