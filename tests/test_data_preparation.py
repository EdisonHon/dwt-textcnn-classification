from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from dwt_textcnn.config import load_config
from dwt_textcnn.data import (
    DataPreparationError,
    prepare_sst_data,
    sha256_bytes,
)
from dwt_textcnn.vocabulary import Vocabulary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TREE_CONTENT = {
    "train": b"(3 (2 Good) (2 movie))\n(1 (2 Bad) (2 film))\n(2 (2 Neutral))\n",
    "dev": b"(4 (2 Great))\n",
    "test": b"(0 (2 Awful))\n",
}


def _tiny_source_and_config(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "trees"
    source.mkdir()
    for split, content in TREE_CONTENT.items():
        (source / f"{split}.txt").write_bytes(content)

    config = deepcopy(load_config(PROJECT_ROOT / "configs" / "base.yaml"))
    config["data"]["split_sha256"] = {
        split: sha256_bytes(content) for split, content in TREE_CONTENT.items()
    }
    config["data"]["expected_binary_counts"] = {
        "train": 2,
        "dev": 1,
        "test": 1,
    }
    return source, config


def test_preparation_writes_deterministic_artifacts(tmp_path: Path) -> None:
    source, config = _tiny_source_and_config(tmp_path)
    output = tmp_path / "prepared"

    first = prepare_sst_data(source, output, config)
    first_bytes = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    second = prepare_sst_data(source, output, config, force=True)
    second_bytes = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }

    assert first == second
    assert first_bytes == second_bytes
    assert first["splits"]["train"]["binary_examples"] == 2
    assert first["splits"]["train"]["neutral_dropped"] == 1
    assert first["splits"]["train"]["unknown_token_count"] == 0
    assert first["splits"]["dev"]["unknown_token_count"] == 1
    assert first["vocabulary"]["scope"] == "train-only"
    assert first["vocabulary"]["source_splits"] == ["train"]
    assert first["batching"]["minimum_sequence_length"] == 7
    vocabulary = Vocabulary.load(output / "vocabulary.json")
    assert "Great" not in vocabulary.token_to_id
    assert "Awful" not in vocabulary.token_to_id
    dev_record = json.loads((output / "dev.jsonl").read_text(encoding="utf-8"))
    test_record = json.loads((output / "test.jsonl").read_text(encoding="utf-8"))
    assert dev_record["input_ids"] == [vocabulary.unk_id]
    assert test_record["input_ids"] == [vocabulary.unk_id]
    assert len((output / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads((output / "data_manifest.json").read_text(encoding="utf-8")) == first


def test_preparation_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    source, config = _tiny_source_and_config(tmp_path)
    output = tmp_path / "prepared"
    prepare_sst_data(source, output, config)

    with pytest.raises(DataPreparationError, match="--force"):
        prepare_sst_data(source, output, config)


def test_preparation_rejects_changed_source_file(tmp_path: Path) -> None:
    source, config = _tiny_source_and_config(tmp_path)
    (source / "train.txt").write_text("(3 changed)\n", encoding="utf-8")

    with pytest.raises(DataPreparationError, match="SHA-256 mismatch"):
        prepare_sst_data(source, tmp_path / "prepared", config)
