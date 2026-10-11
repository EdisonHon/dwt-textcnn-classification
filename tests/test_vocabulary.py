from __future__ import annotations

from pathlib import Path

import pytest

from dwt_textcnn.vocabulary import Vocabulary, build_vocabulary


def test_vocabulary_uses_stable_first_occurrence_order() -> None:
    vocabulary = build_vocabulary(
        [("A", "movie", "."), ("movie", "B"), ("test-only",)]
    )

    assert vocabulary.tokens == (
        "<pad>",
        "<unk>",
        "A",
        "movie",
        ".",
        "B",
        "test-only",
    )
    assert vocabulary.encode(["movie", "missing"]) == [3, 1]
    assert vocabulary.decode([0, 2, 6]) == ["<pad>", "A", "test-only"]


def test_vocabulary_round_trip_and_hash_are_stable(tmp_path: Path) -> None:
    vocabulary = build_vocabulary([("café", "film")])
    path = tmp_path / "vocabulary.json"
    vocabulary.save(path)
    loaded = Vocabulary.load(path)

    assert loaded == vocabulary
    assert loaded.sha256 == vocabulary.sha256
    assert path.read_bytes() == vocabulary.to_json_bytes()


@pytest.mark.parametrize("token_id", [-1, 10])
def test_vocabulary_rejects_out_of_range_ids(token_id: int) -> None:
    vocabulary = build_vocabulary([("token",)])

    with pytest.raises(ValueError, match="outside"):
        vocabulary.decode([token_id])


def test_special_tokens_must_be_distinct() -> None:
    with pytest.raises(ValueError, match="different"):
        build_vocabulary([], pad_token="same", unk_token="same")
