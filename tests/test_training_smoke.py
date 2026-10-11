from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from dwt_textcnn.config import load_config
from dwt_textcnn.data import EncodedExample, sha256_file
from dwt_textcnn.embeddings import prepare_original_embeddings
from dwt_textcnn.training import evaluate_saved_run, train_experiment
from dwt_textcnn.vocabulary import Vocabulary


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyVectors:
    vector_size = 300

    def __init__(self, tokens: tuple[str, ...]) -> None:
        self.vectors = {
            token: np.full(300, index / 10, dtype=np.float32)
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


def _smoke_inputs(tmp_path: Path) -> tuple[dict, Path, Path]:
    config = deepcopy(load_config(PROJECT_ROOT / "configs" / "original.yaml"))
    config["data"]["expected_binary_counts"] = {"train": 4, "dev": 2, "test": 2}
    config["model"]["kernel_sizes"] = [2, 3]
    config["model"]["filters_per_size"] = 2
    config["training"]["batch_size"] = 2
    config["training"]["early_stopping"]["patience"] = 1

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    vocabulary = Vocabulary(("<pad>", "<unk>", "good", "bad", "movie", "film"))
    vocabulary.save(prepared / "vocabulary.json")
    _write_jsonl(
        prepared / "train.jsonl",
        [
            EncodedExample((2, 4), 1),
            EncodedExample((3, 5), 0),
            EncodedExample((2, 5), 1),
            EncodedExample((3, 4), 0),
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
    config["embedding"]["word2vec_md5"] = hashlib.md5(
        content, usedforsecurity=False
    ).hexdigest()
    config["embedding"]["word2vec_sha256"] = hashlib.sha256(content).hexdigest()
    embedding_root = tmp_path / "embeddings"
    prepare_original_embeddings(
        word2vec,
        prepared,
        embedding_root,
        [13],
        config,
        loader=lambda _: TinyVectors(("good", "bad", "movie", "film")),
    )
    return config, prepared, embedding_root / "seed_13"


def test_training_evaluation_checkpoint_and_result_files(tmp_path: Path) -> None:
    config, prepared, embedding_cache = _smoke_inputs(tmp_path)
    run_dir, metrics = train_experiment(
        config,
        seed=13,
        prepared_data_dir=prepared,
        embedding_cache_dir=embedding_cache,
        output_root=tmp_path / "outputs",
        device_name="cpu",
        run_id="smoke-test",
        max_epochs=2,
        limit_train_batches=1,
    )
    reevaluated = evaluate_saved_run(run_dir, device_name="cpu")

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
    assert metrics["test"] == reevaluated["test"]
    assert metrics["best_epoch"] == reevaluated["best_epoch"]
    assert 0.0 <= metrics["test"]["accuracy"] <= 1.0
