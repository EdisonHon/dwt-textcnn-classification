# DWT-MCCNN Text Classification

This repository will reproduce the ordinary multichannel CNN (MCCNN) SST-2
baseline described in *Transformable Convolutional Neural Network for Text
Classification* and compare original Google News Word2Vec embeddings with
one-level DWT `cA`, one-level DWT `cD`, and PCA compression.

The authoritative scope, experimental protocol, and acceptance criteria are
defined in [`PROJECT_DESIGN.md`](PROJECT_DESIGN.md). Implementation is being
built from scratch; TF-MCCNN and transformable convolution/pooling are outside
the initial project scope.

## Environment

- Python 3.11
- PyTorch 2.x
- Windows development environment with optional CUDA acceleration

Create and activate an isolated environment before installing the project:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The checked development machine uses an NVIDIA RTX 4080 SUPER. To reproduce
its CUDA environment after the editable install, replace the default CPU wheel
with the official CUDA 13.0 build:

```powershell
python -m pip install --force-reinstall --no-deps "torch==2.14.1+cu130" `
  --index-url https://download.pytorch.org/whl/cu130
```

`requirements-lock.txt` records the exact preparation environment. Run
experiments on a different PyTorch build only when the build and device are
captured in the generated environment manifest.

## Command line interface

The project exposes one command with six staged operations:

```powershell
dwt-textcnn --help
dwt-textcnn prepare-data --help
dwt-textcnn prepare-embeddings --help
dwt-textcnn train --help
dwt-textcnn evaluate --help
dwt-textcnn benchmark --help
dwt-textcnn summarize --help
```

Method configurations inherit the fixed experiment protocol from
`configs/base.yaml`. They can be resolved and validated independently:

```powershell
dwt-textcnn train --config configs/original.yaml --check-config
dwt-textcnn train --config configs/dwt_ca.yaml --check-config
```

All six operations are implemented. Benchmarking updates the saved run's
profile, while summarization requires one exact run ID shared by the complete
four-method, five-seed matrix.

Prepare SST-2 from either the untouched 2013 archive or its extracted `trees`
directory:

```powershell
dwt-textcnn prepare-data `
  --source C:\path\to\trainDevTestTrees_PTB.zip `
  --output-dir data\processed\sst2
```

The command verifies the archive and split checksums, retains only binary root
sentences, builds the 16,284-entry train-only vocabulary, maps unseen Dev/Test
tokens to `<unk>`, and writes deterministic JSONL files plus
`data_manifest.json`. Existing artifacts are protected unless `--force` is
provided.

After data preparation, extract the train vocabulary from Google News Word2Vec
and create one reproducible Original cache for each configured experiment seed:

```powershell
dwt-textcnn prepare-embeddings `
  --config configs/original.yaml `
  --word2vec C:\path\to\word2vec-google-news-300.gz
```

The command verifies both MD5 and SHA-256 before loading the vectors. Exact
token matches retain their pretrained vectors; lexical OOV tokens and `<unk>`
use seed-controlled `Uniform(-0.25, 0.25)` vectors, while `<pad>` remains zero.
Use one or more `--seed` arguments to prepare a subset of the configured seeds.
Original caches are written below `data/processed/embeddings/original` by
default and are excluded from Git.

Create the 150-dimensional DWT and PCA caches from those verified Original
caches (the Word2Vec source is not loaded again):

```powershell
dwt-textcnn prepare-embeddings --config configs/dwt_ca.yaml
dwt-textcnn prepare-embeddings --config configs/dwt_cd.yaml
dwt-textcnn prepare-embeddings --config configs/pca.yaml
```

Each compressed seed inherits its exact Original matrix, Word2Vec match mask,
and vocabulary identity. DWT uses one-level `coif1` with `periodization`; PCA
uses full SVD and is fitted once on unique Train token IDs that matched
Word2Vec. Use `--original-cache-root` when the Original caches are not in their
default location. Existing outputs remain protected unless `--force` is given.

Train any one MCCNN method and save the best Dev Accuracy checkpoint:

```powershell
dwt-textcnn train --config configs/original.yaml --seed 13 --device auto
dwt-textcnn train --config configs/dwt_ca.yaml --seed 13 --device auto
dwt-textcnn train --config configs/dwt_cd.yaml --seed 13 --device auto
dwt-textcnn train --config configs/pca.yaml --seed 13 --device auto
```

For an explicit development smoke run, `--max-epochs` and
`--limit-train-batches` are available and the saved metrics are marked as
`run_kind: smoke`. A normal run writes the resolved configuration, environment,
data and embedding manifests, training history, metrics, model-size profile,
and best checkpoint under `outputs/sst2/<method>/<seed>/<run-id>`.

Re-evaluate a saved best checkpoint without retraining:

```powershell
dwt-textcnn evaluate `
  --config configs/original.yaml `
  --run-dir outputs\sst2\original\13\<run-id> `
  --device auto
```

The completed five-seed experiment matrix produced the following test results:

| Method | Test accuracy (mean ± sample std) | Macro-F1 | Parameters |
| --- | ---: | ---: | ---: |
| Original | 81.68% ± 0.90% | 81.66% ± 0.92% | 10,311,302 |
| DWT-cA | 80.65% ± 1.29% | 80.63% ± 1.31% | 5,156,102 |
| DWT-cD | 81.04% ± 1.46% | 81.01% ± 1.47% | 5,156,102 |
| PCA | 81.33% ± 0.83% | 81.25% ± 0.89% | 5,156,102 |

See [`RESULTS.md`](RESULTS.md) for the full classification, model-size,
training-time, inference, memory, and reproducibility report.

## Development checks

Run the test suite from the repository root:

```powershell
python -m pytest
```

## External data

SST-2 data, pretrained embeddings, generated caches, checkpoints, and experiment
outputs are intentionally excluded from Git. The project uses Stanford NLP's
original 2013 `trainDevTestTrees_PTB.zip` release because its binary root counts
(6,920/872/1,821) match the reference paper. Google News Word2Vec must be
obtained separately and supplied to the preparation command through a local
path. The project must record source versions and checksums in run manifests.

The Gensim-data catalog identifies the expected model as
`word2vec-google-news-300` (3 million records, 300 dimensions) with catalog MD5
`a5e5354d40acb95f9ec66d5977d140ef`. Its catalog does not identify a license,
so the model must not be redistributed through this repository.
