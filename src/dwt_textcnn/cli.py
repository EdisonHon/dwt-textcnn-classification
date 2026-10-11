"""Unified command line interface for all project stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dwt_textcnn.config import ConfigError, load_config
from dwt_textcnn.data import DataPreparationError, prepare_sst_data
from dwt_textcnn.embeddings import (
    EmbeddingPreparationError,
    prepare_compressed_embeddings,
    prepare_original_embeddings,
)
from dwt_textcnn.profiling import BenchmarkError, benchmark_saved_run
from dwt_textcnn.reporting import SummaryError, summarize_experiments
from dwt_textcnn.training import TrainingError, evaluate_saved_run, train_experiment


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
        if command == "prepare-data":
            command_help = "Prepare and validate the fixed 2013 SST-2 dataset."
        elif command == "prepare-embeddings":
            command_help = "Prepare deterministic embedding caches."
        elif command == "train":
            command_help = "Train one MCCNN experiment and save its best checkpoint."
        elif command == "evaluate":
            command_help = "Re-evaluate the test split for a saved run."
        else:
            command_help = f"Project operation implemented in {phase}."
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
        if command == "prepare-embeddings":
            subparser.add_argument(
                "--word2vec",
                type=Path,
                help="Google News Word2Vec binary or compressed binary path.",
            )
            subparser.add_argument(
                "--prepared-data-dir",
                type=Path,
                help="Directory containing vocabulary.json and data_manifest.json.",
            )
            subparser.add_argument(
                "--output-dir",
                type=Path,
                help="Method cache directory (defaults under paths.prepared_embedding_dir).",
            )
            subparser.add_argument(
                "--original-cache-root",
                type=Path,
                help="Original seed cache root required by DWT and PCA methods.",
            )
            subparser.add_argument(
                "--seed",
                type=int,
                action="append",
                help="Seed to prepare; repeat for multiple seeds (default: all configured seeds).",
            )
            subparser.add_argument(
                "--force",
                action="store_true",
                help="Replace existing caches after validation succeeds.",
            )
        if command == "train":
            subparser.add_argument("--seed", type=int, help="Experiment seed.")
            subparser.add_argument("--prepared-data-dir", type=Path)
            subparser.add_argument("--embedding-cache-dir", type=Path)
            subparser.add_argument(
                "--output-root",
                type=Path,
                help="Root for outputs/sst2/<method>/<seed>/<run-id>.",
            )
            subparser.add_argument(
                "--device", default="auto", help="PyTorch device (default: auto)."
            )
            subparser.add_argument("--run-id", help="Optional unique run directory name.")
            subparser.add_argument(
                "--max-epochs", type=int, help="Smoke-test override for maximum epochs."
            )
            subparser.add_argument(
                "--limit-train-batches",
                type=int,
                help="Smoke-test override for train batches per epoch.",
            )
        if command == "evaluate":
            subparser.add_argument(
                "--run-dir", type=Path, required=True, help="Saved run directory."
            )
            subparser.add_argument(
                "--device", default="auto", help="PyTorch device (default: auto)."
            )
        if command == "benchmark":
            subparser.add_argument(
                "--run-dir", type=Path, required=True, help="Saved run directory."
            )
            subparser.add_argument(
                "--device", default="auto", help="PyTorch device (default: auto)."
            )
            subparser.add_argument(
                "--warmup-iterations", type=int, default=20
            )
            subparser.add_argument(
                "--measured-iterations", type=int, default=100
            )
        if command == "summarize":
            subparser.add_argument(
                "--run-id",
                required=True,
                help="Exact run ID shared by all methods and seeds.",
            )
            subparser.add_argument(
                "--output-root",
                type=Path,
                help="Experiment output root (defaults to paths.output_dir).",
            )
            subparser.add_argument(
                "--summary-dir",
                type=Path,
                help="Destination (defaults to outputs/sst2/summary/<run-id>).",
            )
            subparser.add_argument("--force", action="store_true")
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

    if args.command == "prepare-embeddings":
        method = config["embedding"]["method"]
        prepared_data_dir = args.prepared_data_dir or Path(
            config["paths"]["prepared_data_dir"]
        )
        output_dir = args.output_dir or (
            Path(config["paths"]["prepared_embedding_dir"]) / method
        )
        seeds = args.seed or config["experiment"]["seeds"]
        try:
            if method == "original":
                word2vec_path = args.word2vec or config["embedding"].get("word2vec_path")
                if word2vec_path is None:
                    parser.error(
                        "Original preparation requires --word2vec or "
                        "a non-null embedding.word2vec_path"
                    )
                manifests = prepare_original_embeddings(
                    word2vec_path=word2vec_path,
                    prepared_data_dir=prepared_data_dir,
                    output_dir=output_dir,
                    seeds=seeds,
                    config=config,
                    force=args.force,
                )
            else:
                original_root = args.original_cache_root or (
                    Path(config["paths"]["prepared_embedding_dir"]) / "original"
                )
                manifests = prepare_compressed_embeddings(
                    original_cache_root=original_root,
                    prepared_data_dir=prepared_data_dir,
                    output_dir=output_dir,
                    seeds=seeds,
                    config=config,
                    force=args.force,
                )
        except (EmbeddingPreparationError, OSError) as exc:
            parser.error(str(exc))
        first = manifests[0]
        summary = {
            "output_dir": str(Path(output_dir).expanduser().resolve()),
            "seeds": [manifest["seed"] for manifest in manifests],
            "shape": first["shape"],
            "matched_lexical_token_count": first["coverage"][
                "matched_lexical_token_count"
            ],
            "oov_lexical_token_count": first["coverage"]["oov_lexical_token_count"],
            "matched_lexical_rate": first["coverage"]["matched_lexical_rate"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "train":
        method = config["embedding"]["method"]
        seed = args.seed if args.seed is not None else config["experiment"]["seeds"][0]
        prepared_data_dir = args.prepared_data_dir or Path(
            config["paths"]["prepared_data_dir"]
        )
        embedding_cache_dir = args.embedding_cache_dir or (
            Path(config["paths"]["prepared_embedding_dir"])
            / method
            / f"seed_{seed}"
        )
        output_root = args.output_root or Path(config["paths"]["output_dir"])
        try:
            run_dir, metrics = train_experiment(
                config,
                seed=seed,
                prepared_data_dir=prepared_data_dir,
                embedding_cache_dir=embedding_cache_dir,
                output_root=output_root,
                device_name=args.device,
                run_id=args.run_id,
                max_epochs=args.max_epochs,
                limit_train_batches=args.limit_train_batches,
            )
        except (TrainingError, OSError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "best_epoch": metrics["best_epoch"],
                    "dev_accuracy": metrics["dev"]["accuracy"],
                    "test_accuracy": metrics["test"]["accuracy"],
                    "test_macro_f1": metrics["test"]["macro_f1"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "evaluate":
        try:
            result = evaluate_saved_run(args.run_dir, device_name=args.device)
        except (TrainingError, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "benchmark":
        try:
            result = benchmark_saved_run(
                args.run_dir,
                device_name=args.device,
                warmup_iterations=args.warmup_iterations,
                measured_iterations=args.measured_iterations,
            )
        except (BenchmarkError, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "summarize":
        output_root = args.output_root or Path(config["paths"]["output_dir"])
        summary_dir = args.summary_dir or (
            Path(output_root)
            / config["project"]["dataset"]
            / "summary"
            / args.run_id
        )
        try:
            result = summarize_experiments(
                config,
                output_root=output_root,
                run_id=args.run_id,
                summary_dir=summary_dir,
                force=args.force,
            )
        except (SummaryError, OSError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "summary_dir": str(Path(summary_dir).expanduser().resolve()),
                    "run_count": len(result["runs"]),
                    "methods": result["methods"],
                    "seeds": result["seeds"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2
