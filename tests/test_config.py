from __future__ import annotations

from pathlib import Path

import pytest

from dwt_textcnn.config import ConfigError, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "method", "dimension"),
    [
        ("original.yaml", "original", 300),
        ("dwt_ca.yaml", "dwt_ca", 150),
        ("dwt_cd.yaml", "dwt_cd", 150),
        ("pca.yaml", "pca", 150),
    ],
)
def test_method_configs_resolve_and_validate(
    filename: str, method: str, dimension: int
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / filename)

    assert config["embedding"]["method"] == method
    assert config["embedding"]["dimension"] == dimension
    assert config["model"]["kernel_sizes"] == [5, 6, 7]
    assert config["experiment"]["seeds"] == [13, 21, 42, 87, 100]
    assert config["data"]["vocabulary_scope"] == "train-only"
    assert config["embedding"]["oov_initialization"] == {
        "distribution": "uniform",
        "low": -0.25,
        "high": 0.25,
    }
    assert config["model"]["shared_channel_filters"] is True
    assert config["model"]["channel_merge"] == "sum-after-relu"
    assert config["model"]["initialization"] == {
        "distribution": "truncated-normal",
        "mean": 0.0,
        "std": 0.1,
        "bias": 0.0,
    }
    assert "extends" not in config


def test_invalid_method_dimension_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        f"extends: {PROJECT_ROOT / 'configs' / 'base.yaml'}\n"
        "embedding:\n  method: dwt_ca\n  dimension: 300\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="must be 150"):
        load_config(config_path)


def test_cyclic_inheritance_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Cyclic configuration inheritance"):
        load_config(first)
