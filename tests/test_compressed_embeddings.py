from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dwt_textcnn.data import sha256_file
from dwt_textcnn.embeddings import (
    EmbeddingPreparationError,
    load_embedding_cache,
    prepare_compressed_embeddings,
    prepare_original_embeddings,
)
from dwt_textcnn.vocabulary import Vocabulary


class SyntheticVectors:
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


def _artifacts(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    lexical_tokens = tuple(f"token-{index}" for index in range(151))
    vocabulary = Vocabulary(("<pad>", "<unk>", *lexical_tokens))
    vocabulary.save(prepared / "vocabulary.json")
    data_manifest = {
        "vocabulary": {"size": len(vocabulary), "sha256": vocabulary.sha256},
        "artifacts": {
            "vocabulary.json": {"sha256": sha256_file(prepared / "vocabulary.json")}
        },
    }
    (prepared / "data_manifest.json").write_text(
        json.dumps(data_manifest), encoding="utf-8"
    )
    # Exactly 150 matched IDs occur in Train; the final matched token is held out.
    (prepared / "train.jsonl").write_text(
        json.dumps({"input_ids": list(range(2, 152)), "label": 1}) + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "word2vec.bin"
    source.write_bytes(b"synthetic")
    return prepared, source, lexical_tokens


def _original_config(source: Path) -> dict:
    content = source.read_bytes()
    return {
        "embedding": {
            "method": "original",
            "dimension": 300,
            "word2vec_md5": hashlib.md5(content, usedforsecurity=False).hexdigest(),
            "word2vec_sha256": hashlib.sha256(content).hexdigest(),
            "oov_initialization": {
                "distribution": "uniform",
                "low": -0.25,
                "high": 0.25,
            },
        }
    }


def _compressed_config(method: str) -> dict:
    return {
        "embedding": {
            "method": method,
            "dimension": 150,
            "dwt": {
                "wavelet": "coif1",
                "level": 1,
                "mode": "periodization",
                "coefficient": "cA" if method == "dwt_ca" else "cD",
            },
            "pca": {"n_components": 150, "whiten": False, "svd_solver": "full"},
        }
    }


def test_dwt_and_pca_caches_preserve_provenance_and_fit_scope(tmp_path: Path) -> None:
    prepared, source, tokens = _artifacts(tmp_path)
    original_root = tmp_path / "original"
    prepare_original_embeddings(
        source,
        prepared,
        original_root,
        [13, 21],
        _original_config(source),
        loader=lambda _: SyntheticVectors(tokens),
    )

    dwt_output = tmp_path / "dwt_ca"
    dwt_manifests = prepare_compressed_embeddings(
        original_root, prepared, dwt_output, [13, 21], _compressed_config("dwt_ca")
    )
    dwt_matrix, dwt_mask, _ = load_embedding_cache(
        dwt_output / "seed_13", expected_method="dwt_ca"
    )
    assert dwt_matrix.shape == (153, 150)
    assert dwt_mask.sum() == 151
    assert np.all(dwt_matrix[0] == 0.0)
    assert dwt_manifests[0]["source"]["seed"] == 13

    pca_output = tmp_path / "pca"
    pca_manifests = prepare_compressed_embeddings(
        original_root, prepared, pca_output, [13, 21], _compressed_config("pca")
    )
    pca_matrix, pca_mask, _ = load_embedding_cache(
        pca_output / "seed_13", expected_method="pca"
    )
    model_manifest = json.loads(
        (pca_output / "pca_model_manifest.json").read_text(encoding="utf-8")
    )
    assert pca_matrix.shape == (153, 150)
    assert np.all(pca_matrix[0] == 0.0)
    assert np.array_equal(pca_mask, dwt_mask)
    assert model_manifest["fit_sample_count"] == 150
    assert pca_manifests[0]["transform"]["fit_scope"].startswith("unique train")

    original_hash = pca_manifests[0]["artifacts"]["pretrained.npy"]["sha256"]
    del pca_matrix, pca_mask
    repeated = prepare_compressed_embeddings(
        original_root,
        prepared,
        pca_output,
        [13, 21],
        _compressed_config("pca"),
        force=True,
    )
    assert repeated[0]["artifacts"]["pretrained.npy"]["sha256"] == original_hash

    with (pca_output / "pca_components.npy").open("ab") as destination:
        destination.write(b"tampered")
    with pytest.raises(EmbeddingPreparationError, match="model artifact checksum"):
        load_embedding_cache(pca_output / "seed_13", expected_method="pca")


def test_generic_loader_rejects_wrong_method(tmp_path: Path) -> None:
    prepared, source, tokens = _artifacts(tmp_path)
    original_root = tmp_path / "original"
    prepare_original_embeddings(
        source,
        prepared,
        original_root,
        [13],
        _original_config(source),
        loader=lambda _: SyntheticVectors(tokens),
    )

    with pytest.raises(EmbeddingPreparationError, match="method mismatch"):
        load_embedding_cache(original_root / "seed_13", expected_method="pca")
