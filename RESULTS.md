# Formal SST-2 Results

This report records the fixed five-seed experiment matrix with run ID
`formal-5seed`. Seeds are 13, 21, 42, 87, and 100. Values after `±` are sample
standard deviations (`ddof=1`). The complete per-run records and generated CSV
tables remain under `outputs/sst2` and are intentionally excluded from Git.

## Classification

| Method | Dimension | Test accuracy | Test Macro-F1 | Accuracy vs. Original |
| --- | ---: | ---: | ---: | ---: |
| Original | 300 | 81.68% ± 0.90% | 81.66% ± 0.92% | — |
| DWT-cA | 150 | 80.65% ± 1.29% | 80.63% ± 1.31% | -1.03 pp |
| DWT-cD | 150 | 81.04% ± 1.46% | 81.01% ± 1.47% | -0.64 pp |
| PCA | 150 | 81.33% ± 0.83% | 81.25% ± 0.89% | -0.35 pp |

The reference paper reports 88.1% for MCCNN. This implementation's Original
mean is 6.42 percentage points lower. The design document already identifies
missing optimization and initialization details in the reference paper, so the
gap must be reported rather than hidden by selecting the best seed.

Within this fixed protocol, Original has the highest mean accuracy. PCA retains
the most accuracy among the 150-dimensional methods, followed by DWT-cD and
DWT-cA. The five-seed results do not support claiming that either DWT variant
improves classification accuracy over Original or PCA.

## Model and storage size

| Method | Pretrained emb. params | Random emb. params | Conv. params | Classifier params | Total params | Cache bytes | Checkpoint bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 4,885,200 | 4,885,200 | 540,300 | 602 | 10,311,302 | 19,540,928 | 41,249,333 |
| DWT-cA | 2,442,600 | 2,442,600 | 270,300 | 602 | 5,156,102 | 9,770,528 | 20,628,405 |
| DWT-cD | 2,442,600 | 2,442,600 | 270,300 | 602 | 5,156,102 | 9,770,528 | 20,628,405 |
| PCA | 2,442,600 | 2,442,600 | 270,300 | 602 | 5,156,102 | 9,770,528 | 20,628,405 |

Reducing the embedding dimension from 300 to 150 reduces total model
parameters by 50.00%, cache storage by 50.00%, and checkpoint storage by
49.99%. The classifier is unchanged; both embedding channels and convolution
weights shrink with the embedding dimension.

## Runtime profile

Training used an NVIDIA GeForce RTX 4080 SUPER with deterministic algorithms.
Inference used the first 50-example test batch, 20 warm-up iterations, and 100
measured iterations. CUDA was synchronized before and after every measurement.

| Method | Total training seconds | Mean epoch seconds | Latency per batch | Throughput |
| --- | ---: | ---: | ---: | ---: |
| Original | 8.76 ± 1.65 | 0.778 ± 0.066 | 1.247 ± 0.067 ms | 40,177 ± 2,057 samples/s |
| DWT-cA | 7.22 ± 1.89 | 0.732 ± 0.060 | 0.933 ± 0.027 ms | 53,643 ± 1,548 samples/s |
| DWT-cD | 7.11 ± 0.65 | 0.674 ± 0.025 | 1.145 ± 0.238 ms | 45,224 ± 9,376 samples/s |
| PCA | 6.38 ± 0.52 | 0.643 ± 0.037 | 1.118 ± 0.196 ms | 45,732 ± 7,282 samples/s |

Total training time includes per-epoch Dev evaluation and varies with early
stopping, so mean epoch time is the cleaner compute comparison. GPU inference
measurements are short and show noticeable system-level variance; they support
the general efficiency benefit of 150 dimensions but not a strong ranking
among DWT-cA, DWT-cD, and PCA.

Peak CUDA allocation was 60,389,888 bytes for Original and 36,159,488 bytes for
each compressed method, a 40.12% reduction. Sampled process RSS includes the
Python runtime, loaded libraries, and memory-mapped artifacts; it did not fall
for compressed runs and should not be interpreted as model-only CPU memory.

## Reproduction commands

Each method and seed was trained with:

```powershell
dwt-textcnn train --config configs\<method>.yaml `
  --seed <seed> --device auto --run-id formal-5seed
```

Each saved run was benchmarked with:

```powershell
dwt-textcnn benchmark --config configs\<method>.yaml `
  --run-dir outputs\sst2\<method>\<seed>\formal-5seed `
  --device auto --warmup-iterations 20 --measured-iterations 100
```

The complete matrix was validated and summarized with:

```powershell
dwt-textcnn summarize --config configs\base.yaml --run-id formal-5seed
```
