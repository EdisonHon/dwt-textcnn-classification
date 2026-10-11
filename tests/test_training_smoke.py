from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from dwt_textcnn.config import load_config
from dwt_textcnn.data import EncodedExample, sha256_file
from dwt_textcnn.embeddings import (
    prepare_compressed_embeddings,
    prepare_original_embeddings,
)
from dwt_textcnn.profiling import benchmark_saved_run
from dwt_textcnn.training import evaluate_saved_run, train_experiment
from dwt_textcnn.vocabulary import Vocabulary


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyVectors:
    vector_size = 300

    def __init__(self, tokens: tuple[str, ...]) -> None:
        self.vectors = {
            token: np.random.default_rng(index).normal(size=300).astype(np.float32)
            for index, token in enumerate(tokens, start=1)
        }

    def __contains__(self, token: str) -> bool:
        return token in self.vectors

    def __getitem__(self, token: str) -> np.ndarray:
        return self.vectors[token]

    def __len__(self) -> int:
        return len(self.vectors)


def _write_jsonl(path: Path, examples: list[EncodedExample]) -> None:
    lines = [
        json.dumps({"input_ids": list(example.input_ids), "label": example.label})
        for example in examples
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _smoke_inputs(tmp_path: Path, method: str) -> tuple[dict, Path, Path]:
    config = deepcopy(load_config(PROJECT_ROOT / "configs" / f"{method}.yaml"))
    config["data"]["expected_binary_counts"] = {"train": 4, "dev": 2, "test": 2}
    config["model"]["kernel_sizes"] = [2, 3]
    config["model"]["filters_per_size"] = 2
    config["training"]["batch_size"] = 2
    config["training"]["early_stopping"]["patience"] = 1

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    lexical_tokens = tuple(f"token-{index}" for index in range(151))
    vocabulary = Vocabulary(("<pad>", "<unk>", *lexical_tokens))
    vocabulary.save(prepared / "vocabulary.json")
    _write_jsonl(
        prepared / "train.jsonl",
        [
            EncodedExample(tuple(range(2, 40)), 1),
            EncodedExample(tuple(range(40, 78)), 0),
            EncodedExample(tuple(range(78, 115)), 1),
            EncodedExample(tuple(range(115, 152)), 0),
        ],
    )
    _write_jsonl(
        prepared / "dev.jsonl",
        [EncodedExample((2, 4), 1), EncodedExample((3, 5), 0)],
    )
    _write_jsonl(
        prepared / "test.jsonl",
        [EncodedExample((2, 5), 1), EncodedExample((3, 4), 0)],
    )
    data_manifest = {
        "vocabulary": {"size": len(vocabulary), "sha256": vocabulary.sha256},
        "artifacts": {
            "vocabulary.json": {
                "sha256": sha256_file(prepared / "vocabulary.json")
            }
        },
    }
    (prepared / "data_manifest.json").write_text(
        json.dumps(data_manifest), encoding="utf-8"
    )

    word2vec = tmp_path / "word2vec.bin"
    word2vec.write_bytes(b"tiny-word2vec")
    content = word2vec.read_bytes()
    original_config = deepcopy(
        load_config(PROJECT_ROOT / "configs" / "original.yaml")
    )
    original_config["embedding"]["word2vec_md5"] = hashlib.md5(
        content, usedforsecurity=False
    ).hexdigest()
    original_config["embedding"]["word2vec_sha256"] = hashlib.sha256(
        content
    ).hexdigest()
    embedding_root = tmp_path / "embeddings" / "original"
    prepare_original_embeddings(
        word2vec,
        prepared,
        embedding_root,
        [13],
        original_config,
        loader=lambda _: TinyVectors(lexical_tokens),
    )
    if method == "original":
        cache = embedding_root / "seed_13"
    else:
        method_root = tmp_path / "embeddings" / method
        prepare_compressed_embeddings(
            embedding_root, prepared, method_root, [13], config
        )
        cache = method_root / "seed_13"
    return config, prepared, cache


@pytest.mark.parametrize("method", ["original", "dwt_ca", "dwt_cd", "pca"])
def test_training_evaluation_checkpoint_and_result_files(
    tmp_path: Path, method: str
) -> None:
    config, prepared, embedding_cache = _smoke_inputs(tmp_path, method)
    run_dir, metrics = train_experiment(
        config,
        seed=13,
        prepared_data_dir=prepared,
        embedding_cache_dir=embedding_cache,
        output_root=tmp_path / "outputs",
        device_name="cpu",
        run_id=f"smoke-test-{method}",
        max_epochs=2,
        limit_train_batches=1,
    )
    reevaluated = evaluate_saved_run(run_dir, device_name="cpu")
    benchmark = benchmark_saved_run(
        run_dir,
        device_name="cpu",
        warmup_iterations=0,
        measured_iterations=2,
    )

    expected_files = {
        "config.yaml",
        "environment.json",
        "data_manifest.json",
        "embedding_manifest.json",
        "metrics.json",
        "profile.json",
        "history.csv",
        "best_model.pt",
    }
    assert expected_files <= {path.name for path in run_dir.iterdir()}
    assert metrics["run_kind"] == "smoke"
    assert metrics["method"] == method
    assert metrics["test"] == reevaluated["test"]
    assert metrics["best_epoch"] == reevaluated["best_epoch"]
    assert 0.0 <= metrics["test"]["accuracy"] <= 1.0
    profile = json.loads((run_dir / "profile.json").read_text(encoding="utf-8"))
    assert profile["embedding_dimension"] == config["embedding"]["dimension"]
    assert profile["parameters"]["total_parameters"] > 0
    assert profile["parameter_bytes"]["total_parameter_bytes"] > 0
    assert profile["status"] == "benchmarked"
    assert benchmark["protocol"]["batch_size"] == 2
    assert benchmark["throughput_samples_per_second"] > 0
