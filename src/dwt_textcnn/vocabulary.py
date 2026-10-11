"""Deterministic shared vocabulary for all embedding methods."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Vocabulary:
    """A contiguous token-to-ID mapping with fixed special-token positions."""

    tokens: tuple[str, ...]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    def __post_init__(self) -> None:
        if len(self.tokens) != len(set(self.tokens)):
            raise ValueError("Vocabulary tokens must be unique")
        if len(self.tokens) < 2:
            raise ValueError("Vocabulary must contain pad and unknown tokens")
        if self.tokens[0] != self.pad_token:
            raise ValueError("The pad token must have ID 0")
        if self.tokens[1] != self.unk_token:
            raise ValueError("The unknown token must have ID 1")

    @cached_property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        mapping = self.token_to_id
        return [mapping.get(token, self.unk_id) for token in tokens]

    def decode(self, token_ids: Sequence[int]) -> list[str]:
        if any(token_id < 0 or token_id >= len(self.tokens) for token_id in token_ids):
            raise ValueError("Token ID is outside the vocabulary")
        try:
            return [self.tokens[token_id] for token_id in token_ids]
        except IndexError as exc:
            raise ValueError("Token ID is outside the vocabulary") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "tokens": list(self.tokens),
        }

    def to_json_bytes(self) -> bytes:
        text = json.dumps(
            self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        )
        return f"{text}\n".encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_json_bytes())

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported vocabulary schema version")
        return cls(
            tokens=tuple(payload["tokens"]),
            pad_token=payload["pad_token"],
            unk_token=payload["unk_token"],
        )


def build_vocabulary(
    token_sequences: Iterable[Sequence[str]],
    *,
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
) -> Vocabulary:
    """Build a stable vocabulary in first-occurrence order."""

    if pad_token == unk_token:
        raise ValueError("pad_token and unk_token must be different")
    tokens = [pad_token, unk_token]
    seen = set(tokens)
    for sequence in token_sequences:
        for token in sequence:
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return Vocabulary(tuple(tokens), pad_token=pad_token, unk_token=unk_token)
