# External data

This directory is reserved for local datasets, pretrained embeddings, and
generated embedding caches. Its contents are ignored by Git except for this
file.

Expected external inputs include:

- Stanford `stanfordnlp/sentiment-treebank` binary SST tree files from a fixed
  Git commit.
- Google News Word2Vec 300-dimensional vectors, obtained separately and never
  committed to this repository.

Preparation commands must record the source, version or commit, file size, and
checksum of every external input.
