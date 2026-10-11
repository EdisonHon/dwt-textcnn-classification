"""Unified command line interface for all project stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dwt_textcnn.config import ConfigError, load_config
from dwt_textcnn.data import DataPreparationError, prepare_sst_data


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
        command_help = (
            "Prepare and validate the fixed 2013 SST-2 dataset."
            if command == "prepare-data"
            else f"Project operation implemented in {phase}."
        )
        subparser = subparsers.add_parser(
            command,
            help=command_help,
            description=command_help,
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
        if command == "prepare-data":
            subparser.add_argument(
                "--source",
                type=Path,
                help="Original trainDevTestTrees_PTB.zip or an extracted tree directory.",
            )
            subparser.add_argument(
                "--output-dir",
                type=Path,
                help="Prepared artifact directory (defaults to paths.prepared_data_dir).",
            )
            subparser.add_argument(
                "--force",
                action="store_true",
                help="Replace existing prepared artifacts after all validation succeeds.",
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

    if args.command == "prepare-data":
        source = args.source or config["data"].get("source_path")
        if source is None:
            parser.error(
                "prepare-data requires --source or a non-null data.source_path"
            )
        output_dir = args.output_dir or Path(config["paths"]["prepared_data_dir"])
        try:
            manifest = prepare_sst_data(
                source=source,
                output_dir=output_dir,
                config=config,
                force=args.force,
            )
        except (DataPreparationError, OSError) as exc:
            parser.error(str(exc))
        summary = {
            "output_dir": str(Path(output_dir).expanduser().resolve()),
            "split_counts": {
                split: details["binary_examples"]
                for split, details in manifest["splits"].items()
            },
            "vocabulary_size": manifest["vocabulary"]["size"],
            "vocabulary_sha256": manifest["vocabulary"]["sha256"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    phase = COMMAND_PHASES[args.command]
    print(
        f"'{args.command}' is registered but its execution is intentionally deferred to {phase}. "
        "Use --check-config to validate configuration now."
    )
    return 2
