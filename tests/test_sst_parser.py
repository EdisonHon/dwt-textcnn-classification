from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dwt_textcnn.data import (
    DataPreparationError,
    EncodedExample,
    EncodedSSTDataset,
    SSTBatchCollator,
    load_prepared_split,
    parse_sst_trees,
)


def test_parser_uses_only_root_label_and_complete_leaf_sequence() -> None:
    lines = [
        "(3 (1 A) (2 surprisingly) (4 good) (0 movie) (2 .))",
        "(1 (4 Not) (3 enjoyable))",
        "(2 (4 Ambivalent) (0 sentence))",
    ]

    examples, statistics = parse_sst_trees(lines, source_name="tiny.txt")

    assert [example.label for example in examples] == [1, 0]
    assert examples[0].tokens == ("A", "surprisingly", "good", "movie", ".")
    assert examples[1].tokens == ("Not", "enjoyable")
    assert statistics.raw_examples == 3
    assert statistics.binary_examples == 2
    assert statistics.neutral_dropped == 1
    assert statistics.negative_examples == 1
    assert statistics.positive_examples == 1


@pytest.mark.parametrize(
    "tree",
    ["(5 (2 invalid))", "not a tree"],
)
def test_parser_rejects_invalid_trees(tree: str) -> None:
    with pytest.raises(DataPreparationError):
        parse_sst_trees([tree], source_name="bad.txt")


def test_dataset_and_collator_apply_dynamic_minimum_padding(tmp_path: Path) -> None:
    prepared = tmp_path / "train.jsonl"
    prepared.write_text(
        '{"input_ids":[2,3],"label":1}\n'
        '{"input_ids":[4,5,6],"label":0}\n',
        encoding="utf-8",
    )
    loaded = load_prepared_split(prepared)
    dataset = EncodedSSTDataset(loaded)

    batch = SSTBatchCollator(pad_id=0, minimum_length=7)(
        [dataset[0], dataset[1]]
    )

    assert batch["input_ids"].shape == (2, 7)
    assert batch["labels"].tolist() == [1, 0]
    assert batch["lengths"].tolist() == [2, 3]
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["attention_mask"].sum(dim=1).tolist() == [2, 3]
    assert torch.all(batch["input_ids"][:, 3:] == 0)


def test_collator_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        SSTBatchCollator()([])


def test_dataset_accepts_in_memory_examples() -> None:
    dataset = EncodedSSTDataset([EncodedExample((2, 3), 1)])

    assert dataset[0]["input_ids"].tolist() == [2, 3]
    assert dataset[0]["label"].item() == 1
