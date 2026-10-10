"""Unified command line interface for all project stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dwt_textcnn.config import ConfigError, load_config


COMMAND_PHASES = {
    "prepare-data": "phase 2",
    "prepare-embeddings": "phases 3 and 5",
    "train": "phase 4",
    "evaluate": "phase 4",
    "benchmark": "phase 6",
    "summarize": "phase 6",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dwt-textcnn",
        description="Reproducible SST-2 MCCNN experiments with embedding compression.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, phase in COMMAND_PHASES.items():
        subparser = subparsers.add_parser(
            command,
            help=f"Project operation implemented in {phase}.",
            description=f"This operation is scheduled for {phase}.",
        )
        default = "configs/original.yaml" if command in {"train", "evaluate", "benchmark"} else "configs/base.yaml"
        subparser.add_argument(
            "--config",
            type=Path,
            default=Path(default),
            help=f"Experiment YAML configuration (default: {default}).",
        )
        subparser.add_argument(
            "--check-config",
            action="store_true",
            help="Validate and print the resolved configuration without running the operation.",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))

    if args.check_config:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0

    phase = COMMAND_PHASES[args.command]
    print(
        f"'{args.command}' is registered but its execution is intentionally deferred to {phase}. "
        "Use --check-config to validate configuration now."
    )
    return 2
