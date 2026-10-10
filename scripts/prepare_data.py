"""Compatibility wrapper for the unified data-preparation command."""

from __future__ import annotations

import sys

from dwt_textcnn.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["prepare-data", *sys.argv[1:]]))
