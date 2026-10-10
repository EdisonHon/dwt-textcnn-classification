"""Compatibility wrapper for the unified training command."""

from __future__ import annotations

import sys

from dwt_textcnn.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
