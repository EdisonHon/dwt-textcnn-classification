"""Compatibility wrapper for the unified result-summary command."""

from __future__ import annotations

import sys

from dwt_textcnn.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["summarize", *sys.argv[1:]]))
