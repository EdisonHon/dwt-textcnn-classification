# External data

This directory is reserved for local datasets, pretrained embeddings, and
generated embedding caches. Its contents are ignored by Git except for this
file.

Expected external inputs include:

- Stanford NLP's original 2013 `trainDevTestTrees_PTB.zip` SST tree release,
  fixed by its SHA-256 checksum. The later Git repository is not used because
  its split counts differ from the reference paper.
- Google News Word2Vec 300-dimensional vectors, obtained separately and never
  committed to this repository.

Preparation commands must record the source, version or commit, file size, and
checksum of every external input.

`prepare-data` writes deterministic encoded splits, the train-only vocabulary,
and `data_manifest.json` under `data/processed/sst2` by default. These generated
artifacts remain local and are not committed.

`prepare-embeddings` writes seed-specific Original matrices, Word2Vec match
masks, and embedding manifests under `data/processed/embeddings/original`.
