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

## External data

SST-2 data, pretrained embeddings, generated caches, checkpoints, and experiment
outputs are intentionally excluded from Git. Google News Word2Vec must be
obtained separately and supplied to the preparation command through a local
path. The project must record source versions and checksums in run manifests.

The Gensim-data catalog identifies the expected model as
`word2vec-google-news-300` (3 million records, 300 dimensions) with catalog MD5
`a5e5354d40acb95f9ec66d5977d140ef`. Its catalog does not identify a license,
so the model must not be redistributed through this repository.
