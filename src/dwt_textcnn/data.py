"""Stanford Sentiment Treebank parsing, preparation, and batching utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zipfile import BadZipFile, ZipFile

import torch
from nltk import Tree
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from dwt_textcnn.vocabulary import Vocabulary, build_vocabulary


SPLIT_FILES = {"train": "train.txt", "dev": "dev.txt", "test": "test.txt"}
NEGATIVE_ROOT_LABELS = {0, 1}
NEUTRAL_ROOT_LABEL = 2
POSITIVE_ROOT_LABELS = {3, 4}


class DataPreparationError(ValueError):
    """Raised when SST source data or prepared artifacts violate the protocol."""


@dataclass(frozen=True)
class SSTExample:
    tokens: tuple[str, ...]
    label: int


@dataclass(frozen=True)
class SplitStatistics:
    raw_examples: int
    binary_examples: int
    neutral_dropped: int
    negative_examples: int
    positive_examples: int

    def to_dict(self) -> dict[str, int]:
        return {
            "raw_examples": self.raw_examples,
            "binary_examples": self.binary_examples,
            "neutral_dropped": self.neutral_dropped,
            "negative_examples": self.negative_examples,
            "positive_examples": self.positive_examples,
        }


@dataclass(frozen=True)
class EncodedExample:
    input_ids: tuple[int, ...]
    label: int


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sst_trees(
    lines: Iterable[str], *, source_name: str = "<memory>"
) -> tuple[list[SSTExample], SplitStatistics]:
    """Parse SST trees, retaining only binary-labeled root sentences."""

    examples: list[SSTExample] = []
    raw_examples = neutral_dropped = negative = positive = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        raw_examples += 1
        try:
            tree = Tree.fromstring(line)
            root_label = int(tree.label())
        except (TypeError, ValueError) as exc:
            raise DataPreparationError(
                f"Invalid SST tree at {source_name}:{line_number}: {exc}"
            ) from exc

        tokens = tuple(str(token) for token in tree.leaves())
        if not tokens:
            raise DataPreparationError(
                f"SST tree has no leaf tokens at {source_name}:{line_number}"
            )
        if root_label in NEGATIVE_ROOT_LABELS:
            examples.append(SSTExample(tokens=tokens, label=0))
            negative += 1
        elif root_label in POSITIVE_ROOT_LABELS:
            examples.append(SSTExample(tokens=tokens, label=1))
            positive += 1
        elif root_label == NEUTRAL_ROOT_LABEL:
            neutral_dropped += 1
        else:
            raise DataPreparationError(
                f"Unexpected root label {root_label} at {source_name}:{line_number}"
            )

    statistics = SplitStatistics(
        raw_examples=raw_examples,
        binary_examples=len(examples),
        neutral_dropped=neutral_dropped,
        negative_examples=negative,
        positive_examples=positive,
    )
    return examples, statistics


def _read_source_files(source: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise DataPreparationError(f"SST source file must be a ZIP archive: {source}")
        try:
            with ZipFile(source) as archive:
                files = {
                    split: archive.read(f"trees/{filename}")
                    for split, filename in SPLIT_FILES.items()
                }
        except (BadZipFile, KeyError) as exc:
            raise DataPreparationError(
                f"ZIP archive does not contain the expected trees/*.txt files: {source}"
            ) from exc
        return files, {
            "kind": "zip",
            "path": str(source),
            "archive_sha256": sha256_file(source),
        }

    if source.is_dir():
        candidates = (source, source / "trees", source / "trees" / "trees")
        tree_dir = next(
            (
                candidate
                for candidate in candidates
                if all((candidate / filename).is_file() for filename in SPLIT_FILES.values())
            ),
            None,
        )
        if tree_dir is None:
            raise DataPreparationError(
                f"Directory does not contain train.txt, dev.txt, and test.txt: {source}"
            )
        return {
            split: (tree_dir / filename).read_bytes()
            for split, filename in SPLIT_FILES.items()
        }, {"kind": "directory", "path": str(tree_dir.resolve())}

    raise DataPreparationError(f"SST source does not exist: {source}")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return f"{text}\n".encode("utf-8")


def _jsonl_bytes(examples: Sequence[EncodedExample]) -> bytes:
    lines = [
        json.dumps(
            {"input_ids": list(example.input_ids), "label": example.label},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for example in examples
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_artifact(path: Path, content: bytes) -> None:
    path.write_bytes(content)


def prepare_sst_data(
    source: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Validate the official SST source and write deterministic prepared artifacts."""

    source_files, source_metadata = _read_source_files(Path(source))
    data_config = config["data"]
    expected_hashes = data_config["split_sha256"]
    expected_counts = data_config["expected_binary_counts"]

    actual_archive_hash = source_metadata.get("archive_sha256")
    if actual_archive_hash is not None and actual_archive_hash != data_config["archive_sha256"]:
        raise DataPreparationError(
            "SST archive SHA-256 mismatch: "
            f"expected {data_config['archive_sha256']}, got {actual_archive_hash}"
        )

    split_examples: dict[str, list[SSTExample]] = {}
    split_metadata: dict[str, dict[str, Any]] = {}
    for split, filename in SPLIT_FILES.items():
        content = source_files[split]
        actual_hash = sha256_bytes(content)
        expected_hash = expected_hashes[split]
        if actual_hash != expected_hash:
            raise DataPreparationError(
                f"{filename} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DataPreparationError(f"{filename} is not valid UTF-8") from exc
        examples, statistics = parse_sst_trees(
            text.splitlines(), source_name=filename
        )
        if statistics.binary_examples != expected_counts[split]:
            raise DataPreparationError(
                f"{split} binary count mismatch: expected {expected_counts[split]}, "
                f"got {statistics.binary_examples}"
            )
        split_examples[split] = examples
        split_metadata[split] = {
            "filename": filename,
            "sha256": actual_hash,
            **statistics.to_dict(),
        }

    special_tokens = data_config["special_tokens"]
    vocabulary = build_vocabulary(
        (
            example.tokens
            for example in split_examples["train"]
        ),
        pad_token=special_tokens["pad"],
        unk_token=special_tokens["unk"],
    )
    encoded = {
        split: [
            EncodedExample(tuple(vocabulary.encode(example.tokens)), example.label)
            for example in examples
        ]
        for split, examples in split_examples.items()
    }
    for split, examples in encoded.items():
        token_count = sum(len(example.input_ids) for example in examples)
        unknown_count = sum(
            token_id == vocabulary.unk_id
            for example in examples
            for token_id in example.input_ids
        )
        split_metadata[split].update(
            {
                "token_count": token_count,
                "unknown_token_count": unknown_count,
                "unknown_token_rate": unknown_count / token_count if token_count else 0.0,
            }
        )

    artifacts: dict[str, bytes] = {
        f"{split}.jsonl": _jsonl_bytes(examples)
        for split, examples in encoded.items()
    }
    artifacts["vocabulary.json"] = vocabulary.to_json_bytes()
    artifact_metadata = {
        name: {"sha256": sha256_bytes(content), "size_bytes": len(content)}
        for name, content in artifacts.items()
    }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "sst2",
        "source": {
            **source_metadata,
            "url": data_config["source_url"],
            "declared_archive_sha256": data_config["archive_sha256"],
        },
        "label_mapping": {
            "negative_root_labels": sorted(NEGATIVE_ROOT_LABELS),
            "neutral_root_label_dropped": NEUTRAL_ROOT_LABEL,
            "positive_root_labels": sorted(POSITIVE_ROOT_LABELS),
        },
        "splits": split_metadata,
        "vocabulary": {
            "scope": data_config["vocabulary_scope"],
            "source_splits": ["train"],
            "ordering": "first-occurrence",
            "size": len(vocabulary),
            "pad_token": vocabulary.pad_token,
            "pad_id": vocabulary.pad_id,
            "unk_token": vocabulary.unk_token,
            "unk_id": vocabulary.unk_id,
            "sha256": vocabulary.sha256,
        },
        "batching": {
            "padding": "dynamic",
            "minimum_sequence_length": data_config["minimum_sequence_length"],
        },
        "artifacts": artifact_metadata,
    }
    artifacts["data_manifest.json"] = _json_bytes(manifest)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = [name for name in artifacts if (output / name).exists()]
    if existing and not force:
        names = ", ".join(existing)
        raise DataPreparationError(
            f"Prepared files already exist in {output}: {names}. Use --force to replace them."
        )
    for name, content in artifacts.items():
        _write_artifact(output / name, content)
    return manifest


def load_prepared_split(path: str | Path) -> list[EncodedExample]:
    examples: list[EncodedExample] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            input_ids = tuple(int(value) for value in payload["input_ids"])
            label = int(payload["label"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DataPreparationError(
                f"Invalid prepared example at {path}:{line_number}"
            ) from exc
        if not input_ids or label not in (0, 1) or any(value < 0 for value in input_ids):
            raise DataPreparationError(
                f"Invalid prepared example values at {path}:{line_number}"
            )
        examples.append(EncodedExample(input_ids=input_ids, label=label))
    return examples


class EncodedSSTDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset backed by deterministic integer-encoded examples."""

    def __init__(self, examples: Sequence[EncodedExample]) -> None:
        self.examples = tuple(examples)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "EncodedSSTDataset":
        return cls(load_prepared_split(path))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        return {
            "input_ids": torch.tensor(example.input_ids, dtype=torch.long),
            "label": torch.tensor(example.label, dtype=torch.long),
        }


@dataclass(frozen=True)
class SSTBatchCollator:
    """Dynamically pad a batch while respecting the widest convolution kernel."""

    pad_id: int = 0
    minimum_length: int = 7

    def __post_init__(self) -> None:
        if self.pad_id < 0:
            raise ValueError("pad_id must be non-negative")
        if self.minimum_length < 1:
            raise ValueError("minimum_length must be positive")

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        sequences = [item["input_ids"] for item in batch]
        lengths = torch.tensor([sequence.numel() for sequence in sequences], dtype=torch.long)
        target_length = max(int(lengths.max().item()), self.minimum_length)
        padded = pad_sequence(sequences, batch_first=True, padding_value=self.pad_id)
        if padded.shape[1] < target_length:
            extension = padded.new_full(
                (padded.shape[0], target_length - padded.shape[1]), self.pad_id
            )
            padded = torch.cat((padded, extension), dim=1)
        positions = torch.arange(target_length).unsqueeze(0)
        attention_mask = positions < lengths.unsqueeze(1)
        labels = torch.stack([item["label"] for item in batch])
        return {
            "input_ids": padded,
            "labels": labels,
            "lengths": lengths,
            "attention_mask": attention_mask,
        }
