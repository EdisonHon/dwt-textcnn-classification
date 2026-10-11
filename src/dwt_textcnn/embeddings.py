"""Google News Word2Vec extraction and deterministic Original caches."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dwt_textcnn.data import sha256_file
from dwt_textcnn.vocabulary import Vocabulary


class EmbeddingPreparationError(ValueError):
    """Raised when an embedding source or cache violates the protocol."""


Word2VecLoader = Callable[[Path], Any]


def _file_digests(path: Path) -> dict[str, Any]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return {"size_bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _load_gensim_word2vec(path: Path) -> Any:
    from gensim.models import KeyedVectors

    return KeyedVectors.load_word2vec_format(str(path), binary=True)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return f"{text}\n".encode("utf-8")


def _load_verified_vocabulary(prepared_data_dir: Path) -> tuple[Vocabulary, dict[str, Any]]:
    vocabulary_path = prepared_data_dir / "vocabulary.json"
    manifest_path = prepared_data_dir / "data_manifest.json"
    try:
        data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        vocabulary = Vocabulary.load(vocabulary_path)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EmbeddingPreparationError(
            f"Invalid prepared data artifacts in {prepared_data_dir}: {exc}"
        ) from exc

    try:
        expected_hash = data_manifest["vocabulary"]["sha256"]
        artifact_hash = data_manifest["artifacts"]["vocabulary.json"]["sha256"]
        expected_size = data_manifest["vocabulary"]["size"]
    except (KeyError, TypeError) as exc:
        raise EmbeddingPreparationError(
            f"Invalid data manifest structure: {manifest_path}"
        ) from exc
    actual_hash = sha256_file(vocabulary_path)
    if actual_hash != expected_hash or actual_hash != artifact_hash:
        raise EmbeddingPreparationError(
            "Vocabulary checksum does not match data_manifest.json"
        )
    if len(vocabulary) != expected_size:
        raise EmbeddingPreparationError(
            "Vocabulary size does not match data_manifest.json"
        )
    return vocabulary, data_manifest


def prepare_original_embeddings(
    word2vec_path: str | Path,
    prepared_data_dir: str | Path,
    output_dir: str | Path,
    seeds: Sequence[int],
    config: Mapping[str, Any],
    *,
    force: bool = False,
    loader: Word2VecLoader | None = None,
) -> list[dict[str, Any]]:
    """Extract vocabulary vectors once and materialize one Original cache per seed."""

    source = Path(word2vec_path).expanduser().resolve()
    if not source.is_file():
        raise EmbeddingPreparationError(f"Word2Vec source does not exist: {source}")
    prepared = Path(prepared_data_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    normalized_seeds = tuple(dict.fromkeys(seeds))
    if not normalized_seeds:
        raise EmbeddingPreparationError("At least one experiment seed is required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in normalized_seeds):
        raise EmbeddingPreparationError("Seeds must be non-negative integers")

    target_directories = [output / f"seed_{seed}" for seed in normalized_seeds]
    target_names = ("pretrained.npy", "matched_mask.npy", "embedding_manifest.json")
    existing = [
        target / name
        for target in target_directories
        for name in target_names
        if (target / name).exists()
    ]
    if existing and not force:
        raise EmbeddingPreparationError(
            f"Embedding artifacts already exist (for example {existing[0]}). "
            "Use --force to replace them."
        )

    vocabulary, data_manifest = _load_verified_vocabulary(prepared)
    embedding_config = config["embedding"]
    dimension = embedding_config["dimension"]
    if embedding_config["method"] != "original" or dimension != 300:
        raise EmbeddingPreparationError(
            "Original preparation requires embedding.method=original and dimension=300"
        )

    source_digests = _file_digests(source)
    for algorithm in ("md5", "sha256"):
        expected = embedding_config[f"word2vec_{algorithm}"]
        if source_digests[algorithm] != expected:
            raise EmbeddingPreparationError(
                f"Word2Vec {algorithm.upper()} mismatch: expected {expected}, "
                f"got {source_digests[algorithm]}"
            )

    load_vectors = loader or _load_gensim_word2vec
    keyed_vectors = load_vectors(source)
    if getattr(keyed_vectors, "vector_size", None) != dimension:
        raise EmbeddingPreparationError(
            f"Word2Vec dimension must be {dimension}, got "
            f"{getattr(keyed_vectors, 'vector_size', None)}"
        )

    base_matrix = np.zeros((len(vocabulary), dimension), dtype=np.float32)
    matched_mask = np.zeros(len(vocabulary), dtype=np.bool_)
    for token_id, token in enumerate(vocabulary.tokens):
        if token_id in (vocabulary.pad_id, vocabulary.unk_id):
            continue
        if token in keyed_vectors:
            vector = np.asarray(keyed_vectors[token], dtype=np.float32)
            if vector.shape != (dimension,) or not np.isfinite(vector).all():
                raise EmbeddingPreparationError(
                    f"Invalid Word2Vec vector for token {token!r}: shape={vector.shape}"
                )
            base_matrix[token_id] = vector
            matched_mask[token_id] = True

    source_record_count = len(keyed_vectors)
    del keyed_vectors
    gc.collect()

    oov_config = embedding_config["oov_initialization"]
    if oov_config["distribution"] != "uniform":
        raise EmbeddingPreparationError("Only uniform OOV initialization is supported")
    low = float(oov_config["low"])
    high = float(oov_config["high"])
    if not low < high:
        raise EmbeddingPreparationError("OOV uniform bounds must satisfy low < high")

    oov_indices = np.flatnonzero(~matched_mask)
    oov_indices = oov_indices[oov_indices != vocabulary.pad_id]
    lexical_count = len(vocabulary) - 2
    matched_lexical_count = int(matched_mask.sum())
    oov_lexical_count = lexical_count - matched_lexical_count
    manifests: list[dict[str, Any]] = []

    for seed, target in zip(normalized_seeds, target_directories, strict=True):
        matrix = base_matrix.copy()
        generator = np.random.default_rng(seed)
        matrix[oov_indices] = generator.uniform(
            low, high, size=(len(oov_indices), dimension)
        ).astype(np.float32)
        matrix[vocabulary.pad_id] = 0.0
        if not np.isfinite(matrix).all():
            raise EmbeddingPreparationError("Prepared embedding matrix contains non-finite values")

        target.mkdir(parents=True, exist_ok=True)
        matrix_path = target / "pretrained.npy"
        mask_path = target / "matched_mask.npy"
        np.save(matrix_path, matrix, allow_pickle=False)
        np.save(mask_path, matched_mask, allow_pickle=False)
        artifact_metadata = {
            "pretrained.npy": {
                "sha256": sha256_file(matrix_path),
                "size_bytes": matrix_path.stat().st_size,
            },
            "matched_mask.npy": {
                "sha256": sha256_file(mask_path),
                "size_bytes": mask_path.stat().st_size,
            },
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "method": "original",
            "seed": seed,
            "dimension": dimension,
            "dtype": "float32",
            "shape": [len(vocabulary), dimension],
            "vocabulary": {
                "size": len(vocabulary),
                "sha256": vocabulary.sha256,
                "data_manifest_sha256": sha256_file(prepared / "data_manifest.json"),
            },
            "word2vec": {
                "path": str(source),
                **source_digests,
                "record_count": source_record_count,
                "vector_size": dimension,
                "matching": "exact-token",
            },
            "coverage": {
                "lexical_token_count": lexical_count,
                "matched_lexical_token_count": matched_lexical_count,
                "oov_lexical_token_count": oov_lexical_count,
                "matched_lexical_rate": matched_lexical_count / lexical_count,
                "special_tokens_excluded": [vocabulary.pad_token, vocabulary.unk_token],
            },
            "oov_initialization": {
                "distribution": "uniform",
                "low": low,
                "high": high,
                "seed": seed,
                "includes_unk_token": True,
            },
            "padding": {"token": vocabulary.pad_token, "id": vocabulary.pad_id, "all_zero": True},
            "artifacts": artifact_metadata,
        }
        (target / "embedding_manifest.json").write_bytes(_json_bytes(manifest))
        manifests.append(manifest)

    return manifests


def load_original_embedding_cache(
    cache_dir: str | Path,
    *,
    expected_vocabulary_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load an Original cache only after validating its manifest and checksums."""

    cache = Path(cache_dir)
    manifest_path = cache / "embedding_manifest.json"
    matrix_path = cache / "pretrained.npy"
    mask_path = cache / "matched_mask.npy"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise EmbeddingPreparationError(f"Invalid embedding manifest: {manifest_path}") from exc

    if manifest.get("method") != "original" or manifest.get("schema_version") != 1:
        raise EmbeddingPreparationError("Unsupported embedding cache manifest")
    if expected_vocabulary_sha256 is not None:
        actual = manifest["vocabulary"]["sha256"]
        if actual != expected_vocabulary_sha256:
            raise EmbeddingPreparationError(
                f"Embedding vocabulary mismatch: expected {expected_vocabulary_sha256}, got {actual}"
            )
    for name, path in (("pretrained.npy", matrix_path), ("matched_mask.npy", mask_path)):
        if sha256_file(path) != manifest["artifacts"][name]["sha256"]:
            raise EmbeddingPreparationError(f"Embedding artifact checksum mismatch: {path}")

    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    matched_mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(manifest["shape"])
    if matrix.shape != expected_shape or matrix.dtype != np.float32:
        raise EmbeddingPreparationError("Embedding matrix shape or dtype mismatch")
    if matched_mask.shape != (expected_shape[0],) or matched_mask.dtype != np.bool_:
        raise EmbeddingPreparationError("Matched mask shape or dtype mismatch")
    pad_id = manifest["padding"]["id"]
    if not np.all(matrix[pad_id] == 0.0):
        raise EmbeddingPreparationError("Padding embedding is not all zero")
    return matrix, matched_mask, manifest
