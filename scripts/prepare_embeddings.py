"""Compatibility wrapper for the unified embedding-preparation command."""

from __future__ import annotations

import sys

from dwt_textcnn.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["prepare-embeddings", *sys.argv[1:]]))
