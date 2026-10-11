"""Google News Word2Vec extraction and deterministic Original caches."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dwt_textcnn.compression.dwt import compress_dwt
from dwt_textcnn.compression.pca import fit_pca, transform_pca
from dwt_textcnn.data import load_prepared_split, sha256_file
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


def load_verified_vocabulary(prepared_data_dir: Path) -> tuple[Vocabulary, dict[str, Any]]:
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

    vocabulary, data_manifest = load_verified_vocabulary(prepared)
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


def load_embedding_cache(
    cache_dir: str | Path,
    *,
    expected_method: str | None = None,
    expected_vocabulary_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load any supported embedding cache after validating metadata and checksums."""

    cache = Path(cache_dir)
    manifest_path = cache / "embedding_manifest.json"
    matrix_path = cache / "pretrained.npy"
    mask_path = cache / "matched_mask.npy"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise EmbeddingPreparationError(f"Invalid embedding manifest: {manifest_path}") from exc

    supported_methods = {"original", "dwt_ca", "dwt_cd", "pca"}
    if manifest.get("method") not in supported_methods or manifest.get("schema_version") != 1:
        raise EmbeddingPreparationError("Unsupported embedding cache manifest")
    if expected_method is not None and manifest.get("method") != expected_method:
        raise EmbeddingPreparationError(
            f"Embedding method mismatch: expected {expected_method}, got {manifest.get('method')}"
        )
    if expected_vocabulary_sha256 is not None:
        actual = manifest["vocabulary"]["sha256"]
        if actual != expected_vocabulary_sha256:
            raise EmbeddingPreparationError(
                f"Embedding vocabulary mismatch: expected {expected_vocabulary_sha256}, got {actual}"
            )
    method = manifest["method"]
    try:
        if method == "original":
            if manifest["word2vec"]["vector_size"] != 300:
                raise EmbeddingPreparationError("Invalid Original source metadata")
        else:
            source = manifest["source"]
            if source["method"] != "original" or source["dimension"] != 300:
                raise EmbeddingPreparationError("Invalid compressed-cache source metadata")
            if len(source["matrix_sha256"]) != 64 or len(source["manifest_sha256"]) != 64:
                raise EmbeddingPreparationError("Invalid compressed-cache source checksums")
            transform = manifest["transform"]
            if method.startswith("dwt_") and transform["type"] != "dwt":
                raise EmbeddingPreparationError("Invalid DWT transform metadata")
            if method == "pca":
                if transform["type"] != "pca":
                    raise EmbeddingPreparationError("Invalid PCA transform metadata")
                model_manifest_path = cache.parent / "pca_model_manifest.json"
                if sha256_file(model_manifest_path) != transform["model_manifest_sha256"]:
                    raise EmbeddingPreparationError("PCA model manifest checksum mismatch")
                model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
                for name, metadata in model_manifest["artifacts"].items():
                    if sha256_file(cache.parent / name) != metadata["sha256"]:
                        raise EmbeddingPreparationError(
                            f"PCA model artifact checksum mismatch: {cache.parent / name}"
                        )
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise EmbeddingPreparationError("Invalid embedding source metadata") from exc
    for name, path in (("pretrained.npy", matrix_path), ("matched_mask.npy", mask_path)):
        if sha256_file(path) != manifest["artifacts"][name]["sha256"]:
            raise EmbeddingPreparationError(f"Embedding artifact checksum mismatch: {path}")

    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    matched_mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    try:
        expected_shape = tuple(manifest["shape"])
        expected_dimension = int(manifest["dimension"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingPreparationError("Invalid embedding shape metadata") from exc
    if len(expected_shape) != 2 or expected_shape[1] != expected_dimension:
        raise EmbeddingPreparationError("Embedding dimension metadata is inconsistent")
    if matrix.shape != expected_shape or matrix.dtype != np.float32:
        raise EmbeddingPreparationError("Embedding matrix shape or dtype mismatch")
    if matched_mask.shape != (expected_shape[0],) or matched_mask.dtype != np.bool_:
        raise EmbeddingPreparationError("Matched mask shape or dtype mismatch")
    pad_id = manifest["padding"]["id"]
    if not np.all(matrix[pad_id] == 0.0):
        raise EmbeddingPreparationError("Padding embedding is not all zero")
    return matrix, matched_mask, manifest


def load_original_embedding_cache(
    cache_dir: str | Path,
    *,
    expected_vocabulary_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and validate an Original embedding cache."""

    return load_embedding_cache(
        cache_dir,
        expected_method="original",
        expected_vocabulary_sha256=expected_vocabulary_sha256,
    )


def _normalize_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(dict.fromkeys(seeds))
    if not normalized:
        raise EmbeddingPreparationError("At least one experiment seed is required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in normalized):
        raise EmbeddingPreparationError("Seeds must be non-negative integers")
    return normalized


def _save_array(path: Path, values: np.ndarray) -> dict[str, Any]:
    np.save(path, values, allow_pickle=False)
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _training_matched_ids(
    prepared: Path,
    matched_mask: np.ndarray,
    vocabulary_size: int,
) -> np.ndarray:
    train_path = prepared / "train.jsonl"
    try:
        examples = load_prepared_split(train_path)
    except (OSError, ValueError) as exc:
        raise EmbeddingPreparationError(f"Invalid prepared training split: {exc}") from exc
    token_ids = sorted({token_id for example in examples for token_id in example.input_ids})
    if not token_ids or token_ids[-1] >= vocabulary_size:
        raise EmbeddingPreparationError("Training split contains invalid vocabulary IDs")
    return np.asarray([token_id for token_id in token_ids if matched_mask[token_id]], dtype=np.int64)


def prepare_compressed_embeddings(
    original_cache_root: str | Path,
    prepared_data_dir: str | Path,
    output_dir: str | Path,
    seeds: Sequence[int],
    config: Mapping[str, Any],
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Create seed-aligned DWT or PCA caches from verified Original matrices."""

    original_root = Path(original_cache_root).expanduser().resolve()
    prepared = Path(prepared_data_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    normalized_seeds = _normalize_seeds(seeds)
    method = config["embedding"]["method"]
    dimension = config["embedding"]["dimension"]
    if method not in {"dwt_ca", "dwt_cd", "pca"} or dimension != 150:
        raise EmbeddingPreparationError(
            "Compressed preparation requires dwt_ca, dwt_cd, or pca at dimension 150"
        )

    vocabulary, _ = load_verified_vocabulary(prepared)
    target_directories = [output / f"seed_{seed}" for seed in normalized_seeds]
    target_names = ("pretrained.npy", "matched_mask.npy", "embedding_manifest.json")
    existing = [
        target / name
        for target in target_directories
        for name in target_names
        if (target / name).exists()
    ]
    pca_names = (
        "pca_mean.npy",
        "pca_components.npy",
        "pca_explained_variance.npy",
        "pca_explained_variance_ratio.npy",
        "pca_singular_values.npy",
        "pca_model_manifest.json",
    )
    if method == "pca":
        existing.extend(output / name for name in pca_names if (output / name).exists())
    if existing and not force:
        raise EmbeddingPreparationError(
            f"Embedding artifacts already exist (for example {existing[0]}). "
            "Use --force to replace them."
        )

    originals: list[tuple[np.ndarray, np.ndarray, dict[str, Any], Path]] = []
    reference_mask: np.ndarray | None = None
    for seed in normalized_seeds:
        parent_dir = original_root / f"seed_{seed}"
        matrix, mask, manifest = load_original_embedding_cache(
            parent_dir, expected_vocabulary_sha256=vocabulary.sha256
        )
        if manifest.get("seed") != seed or matrix.shape != (len(vocabulary), 300):
            raise EmbeddingPreparationError(f"Original cache does not match seed {seed}")
        mask_values = np.asarray(mask)
        if reference_mask is None:
            reference_mask = mask_values.copy()
        elif not np.array_equal(reference_mask, mask_values):
            raise EmbeddingPreparationError("Original caches have inconsistent Word2Vec masks")
        originals.append((matrix, mask, manifest, parent_dir))

    transform_model: Any = None
    transform_record: dict[str, Any]
    if method.startswith("dwt_"):
        dwt_config = config["embedding"]["dwt"]
        coefficient = dwt_config["coefficient"]
        transform_record = {
            "type": "dwt",
            "library": "PyWavelets",
            "wavelet": dwt_config["wavelet"],
            "level": dwt_config["level"],
            "mode": dwt_config["mode"],
            "coefficient": coefficient,
        }
    else:
        assert reference_mask is not None
        fit_ids = _training_matched_ids(prepared, reference_mask, len(vocabulary))
        pca_config = config["embedding"]["pca"]
        if len(fit_ids) < dimension:
            raise EmbeddingPreparationError(
                f"PCA needs at least {dimension} matched training tokens, got {len(fit_ids)}"
            )
        reference_matrix = np.asarray(originals[0][0])
        for matrix, _, _, _ in originals[1:]:
            if not np.array_equal(reference_matrix[fit_ids], np.asarray(matrix)[fit_ids]):
                raise EmbeddingPreparationError(
                    "Matched Word2Vec rows differ between Original seed caches"
                )
        try:
            transform_model = fit_pca(
                reference_matrix[fit_ids],
                n_components=pca_config["n_components"],
                whiten=pca_config["whiten"],
                svd_solver=pca_config["svd_solver"],
            )
        except ValueError as exc:
            raise EmbeddingPreparationError(str(exc)) from exc
        output.mkdir(parents=True, exist_ok=True)
        model_arrays = {
            "pca_mean.npy": np.asarray(transform_model.mean_, dtype=np.float32),
            "pca_components.npy": np.asarray(transform_model.components_, dtype=np.float32),
            "pca_explained_variance.npy": np.asarray(transform_model.explained_variance_, dtype=np.float32),
            "pca_explained_variance_ratio.npy": np.asarray(transform_model.explained_variance_ratio_, dtype=np.float32),
            "pca_singular_values.npy": np.asarray(transform_model.singular_values_, dtype=np.float32),
        }
        model_artifacts = {
            name: _save_array(output / name, values)
            for name, values in model_arrays.items()
        }
        model_manifest: dict[str, Any] = {
            "schema_version": 1,
            "type": "pca_model",
            "n_components": pca_config["n_components"],
            "whiten": pca_config["whiten"],
            "svd_solver": pca_config["svd_solver"],
            "fit_sample_count": int(len(fit_ids)),
            "fit_scope": "unique train token IDs matched in Word2Vec",
            "fit_token_ids_sha256": hashlib.sha256(fit_ids.tobytes()).hexdigest(),
            "train_split_sha256": sha256_file(prepared / "train.jsonl"),
            "vocabulary_sha256": vocabulary.sha256,
            "source_seed": normalized_seeds[0],
            "source_original_matrix_sha256": originals[0][2]["artifacts"]["pretrained.npy"]["sha256"],
            "artifacts": model_artifacts,
        }
        model_manifest_path = output / "pca_model_manifest.json"
        model_manifest_path.write_bytes(_json_bytes(model_manifest))
        transform_record = {
            "type": "pca",
            "library": "scikit-learn",
            "n_components": pca_config["n_components"],
            "whiten": pca_config["whiten"],
            "svd_solver": pca_config["svd_solver"],
            "model_manifest_sha256": sha256_file(model_manifest_path),
            "fit_sample_count": int(len(fit_ids)),
            "fit_scope": model_manifest["fit_scope"],
        }

    manifests: list[dict[str, Any]] = []
    for (matrix, mask, parent_manifest, parent_dir), seed, target in zip(
        originals, normalized_seeds, target_directories, strict=True
    ):
        try:
            if method.startswith("dwt_"):
                compressed = compress_dwt(
                    np.asarray(matrix),
                    wavelet=config["embedding"]["dwt"]["wavelet"],
                    mode=config["embedding"]["dwt"]["mode"],
                    coefficient=config["embedding"]["dwt"]["coefficient"],
                )
            else:
                compressed = transform_pca(transform_model, np.asarray(matrix))
        except ValueError as exc:
            raise EmbeddingPreparationError(str(exc)) from exc
        if compressed.shape != (len(vocabulary), dimension):
            raise EmbeddingPreparationError(
                f"Compressed matrix has unexpected shape {compressed.shape}"
            )
        compressed[vocabulary.pad_id] = 0.0
        target.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "pretrained.npy": _save_array(target / "pretrained.npy", compressed),
            "matched_mask.npy": _save_array(target / "matched_mask.npy", np.asarray(mask)),
        }
        manifest = {
            "schema_version": 1,
            "method": method,
            "seed": seed,
            "dimension": dimension,
            "dtype": "float32",
            "shape": [len(vocabulary), dimension],
            "vocabulary": {
                "size": len(vocabulary),
                "sha256": vocabulary.sha256,
                "data_manifest_sha256": sha256_file(prepared / "data_manifest.json"),
            },
            "source": {
                "method": "original",
                "cache_dir": str(parent_dir),
                "seed": seed,
                "dimension": parent_manifest["dimension"],
                "matrix_sha256": parent_manifest["artifacts"]["pretrained.npy"]["sha256"],
                "manifest_sha256": sha256_file(parent_dir / "embedding_manifest.json"),
            },
            "coverage": parent_manifest["coverage"],
            "transform": transform_record,
            "padding": {
                "token": vocabulary.pad_token,
                "id": vocabulary.pad_id,
                "all_zero": True,
                "reset_after_transform": True,
            },
            "artifacts": artifacts,
        }
        (target / "embedding_manifest.json").write_bytes(_json_bytes(manifest))
        manifests.append(manifest)
    return manifests
