"""Configuration loading, inheritance, and experiment invariant validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


METHOD_DIMENSIONS = {
    "original": 300,
    "dwt_ca": 150,
    "dwt_cd": 150,
    "pca": 150,
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return content


def _load_with_inheritance(path: Path, active: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in active:
        chain = " -> ".join(str(item) for item in (*active, resolved))
        raise ConfigError(f"Cyclic configuration inheritance: {chain}")

    current = _read_yaml(resolved)
    parent_name = current.pop("extends", None)
    if parent_name is None:
        return current
    if not isinstance(parent_name, str) or not parent_name.strip():
        raise ConfigError(f"'extends' must be a non-empty path in {resolved}")

    parent = (resolved.parent / parent_name).resolve()
    return _deep_merge(
        _load_with_inheritance(parent, (*active, resolved)),
        current,
    )


def _get(config: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigError(f"Missing required configuration key: {dotted_key}")
        value = value[key]
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate fixed protocol values that must remain comparable across runs."""

    method = _get(config, "embedding.method")
    if method not in METHOD_DIMENSIONS:
        choices = ", ".join(METHOD_DIMENSIONS)
        raise ConfigError(f"embedding.method must be one of: {choices}")

    dimension = _get(config, "embedding.dimension")
    expected_dimension = METHOD_DIMENSIONS[method]
    if dimension != expected_dimension:
        raise ConfigError(
            f"embedding.dimension must be {expected_dimension} for {method}, got {dimension}"
        )

    checks = {
        "data.minimum_sequence_length": 7,
        "model.kernel_sizes": [5, 6, 7],
        "model.filters_per_size": 100,
        "model.dropout": 0.5,
        "model.num_classes": 2,
        "training.batch_size": 50,
        "training.max_epochs": 30,
        "training.optimizer.name": "adadelta",
        "training.optimizer.learning_rate": 1.0,
        "training.optimizer.rho": 0.95,
        "training.optimizer.eps": 1.0e-6,
        "training.early_stopping.metric": "accuracy",
        "training.early_stopping.patience": 5,
        "experiment.seeds": [13, 21, 42, 87, 100],
    }
    for key, expected in checks.items():
        actual = _get(config, key)
        if actual != expected:
            raise ConfigError(f"{key} must be {expected!r}, got {actual!r}")

    if method.startswith("dwt_"):
        if _get(config, "embedding.dwt.wavelet") != "coif1":
            raise ConfigError("embedding.dwt.wavelet must be 'coif1'")
        if _get(config, "embedding.dwt.level") != 1:
            raise ConfigError("embedding.dwt.level must be 1")
        if _get(config, "embedding.dwt.mode") != "periodization":
            raise ConfigError("embedding.dwt.mode must be 'periodization'")
        coefficient = _get(config, "embedding.dwt.coefficient")
        expected = "cA" if method == "dwt_ca" else "cD"
        if coefficient != expected:
            raise ConfigError(
                f"embedding.dwt.coefficient must be {expected!r} for {method}"
            )

    if method == "pca":
        if _get(config, "embedding.pca.n_components") != 150:
            raise ConfigError("embedding.pca.n_components must be 150")
        if _get(config, "embedding.pca.whiten") is not False:
            raise ConfigError("embedding.pca.whiten must be false")
        if _get(config, "embedding.pca.svd_solver") != "full":
            raise ConfigError("embedding.pca.svd_solver must be 'full'")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration, resolve local inheritance, and validate it."""

    config_path = Path(path).expanduser()
    config = _load_with_inheritance(config_path, ())
    validate_config(config)
    return config
