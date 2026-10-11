from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dwt_textcnn.data import sha256_file
from dwt_textcnn.embeddings import (
    EmbeddingPreparationError,
    load_original_embedding_cache,
    prepare_original_embeddings,
)
from dwt_textcnn.vocabulary import Vocabulary


class FakeKeyedVectors:
    vector_size = 300

    def __init__(self) -> None:
        self.vectors = {
            "known": np.linspace(-1.0, 1.0, 300, dtype=np.float32),
        }

    def __contains__(self, token: str) -> bool:
        return token in self.vectors

    def __getitem__(self, token: str) -> np.ndarray:
        return self.vectors[token]

    def __len__(self) -> int:
        return 3_000_000


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    vocabulary = Vocabulary(("<pad>", "<unk>", "known", "missing"))
    vocabulary.save(prepared / "vocabulary.json")
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

    source = tmp_path / "word2vec.bin"
    source.write_bytes(b"fake-word2vec-source")
    content = source.read_bytes()
    config = {
        "embedding": {
            "method": "original",
            "dimension": 300,
            "word2vec_md5": hashlib.md5(
                content, usedforsecurity=False
            ).hexdigest(),
            "word2vec_sha256": hashlib.sha256(content).hexdigest(),
            "oov_initialization": {
                "distribution": "uniform",
                "low": -0.25,
                "high": 0.25,
            },
        }
    }
    return source, prepared, config


def _loader(_: Path) -> FakeKeyedVectors:
    return FakeKeyedVectors()


def test_original_caches_are_seeded_validated_and_repeatable(tmp_path: Path) -> None:
    source, prepared, config = _fixture(tmp_path)
    output = tmp_path / "embeddings"

    manifests = prepare_original_embeddings(
        source, prepared, output, [13, 21], config, loader=_loader
    )
    first_matrix, first_mask, first_manifest = load_original_embedding_cache(
        output / "seed_13",
        expected_vocabulary_sha256=manifests[0]["vocabulary"]["sha256"],
    )
    second_matrix, _, _ = load_original_embedding_cache(output / "seed_21")
    original_hash = first_manifest["artifacts"]["pretrained.npy"]["sha256"]

    assert first_matrix.shape == (4, 300)
    assert first_matrix.dtype == np.float32
    assert np.all(first_matrix[0] == 0.0)
    assert np.array_equal(
        first_matrix[2], np.linspace(-1.0, 1.0, 300, dtype=np.float32)
    )
    assert first_mask.tolist() == [False, False, True, False]
    assert np.all(first_matrix[[1, 3]] >= -0.25)
    assert np.all(first_matrix[[1, 3]] < 0.25)
    assert not np.array_equal(first_matrix[1], second_matrix[1])
    assert manifests[0]["coverage"]["matched_lexical_token_count"] == 1
    assert manifests[0]["coverage"]["oov_lexical_token_count"] == 1

    del first_matrix, second_matrix, first_mask
    repeated = prepare_original_embeddings(
        source, prepared, output, [13], config, force=True, loader=_loader
    )
    assert repeated[0]["artifacts"]["pretrained.npy"]["sha256"] == original_hash


def test_existing_cache_is_protected(tmp_path: Path) -> None:
    source, prepared, config = _fixture(tmp_path)
    output = tmp_path / "embeddings"
    prepare_original_embeddings(source, prepared, output, [13], config, loader=_loader)

    with pytest.raises(EmbeddingPreparationError, match="--force"):
        prepare_original_embeddings(
            source, prepared, output, [13], config, loader=_loader
        )


def test_word2vec_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    source, prepared, config = _fixture(tmp_path)
    config["embedding"]["word2vec_sha256"] = "0" * 64

    with pytest.raises(EmbeddingPreparationError, match="SHA256 mismatch"):
        prepare_original_embeddings(
            source, prepared, tmp_path / "output", [13], config, loader=_loader
        )


def test_cache_tampering_is_rejected(tmp_path: Path) -> None:
    source, prepared, config = _fixture(tmp_path)
    output = tmp_path / "embeddings"
    prepare_original_embeddings(source, prepared, output, [13], config, loader=_loader)
    matrix_path = output / "seed_13" / "pretrained.npy"
    with matrix_path.open("ab") as destination:
        destination.write(b"tampered")

    with pytest.raises(EmbeddingPreparationError, match="checksum mismatch"):
        load_original_embedding_cache(output / "seed_13")
