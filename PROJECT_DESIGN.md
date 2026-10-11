# DWT-MCCNN 文本分类项目设计

## 1. 项目目标

本项目首先独立复现《Transformable Convolutional Neural Network for Text Classification》中的普通多通道卷积神经网络（MCCNN）在 SST-2 上的文本分类实验，并将复现结果与论文报告的准确率进行比较。在此基础上，将《Semantic Word and Sentence Embeddings Compression using Discrete Wavelet Transform》中的 DWT 嵌入压缩方法应用到 MCCNN，并与原始词嵌入和 PCA 降维词嵌入进行公平比较。

项目回答以下问题：

1. 独立实现的 MCCNN 能否取得接近参考论文的 SST-2 分类结果？
2. 将 300 维 Word2Vec 压缩为 150 维后，DWT 能否维持 MCCNN 的分类性能？
3. 在相同输出维度下，DWT 的近似系数 `cA`、细节系数 `cD` 与 PCA 存在什么性能差异？
4. 嵌入压缩能否减少模型参数、磁盘占用、内存消耗以及训练和推理时间？

本阶段以完成可靠、可复现的基础项目为目标，不预设论文级创新。Transformable Convolution、Transformable Pooling 以及进一步的研究问题留到基础项目完成后再决定是否扩展。

## 2. 参考论文与复现范围

### 2.1 CNN 参考论文

参考论文：*Transformable Convolutional Neural Network for Text Classification*（IJCAI 2018）。

本项目只复现论文作为基线使用的普通 MCCNN，不实现论文提出的 TF-MCCNN 和 TF-DCNN。论文在 SST-2 上报告：

| 模型 | SST-2 Accuracy |
| --- | ---: |
| MCCNN | 88.1% |
| TF-MCCNN | 88.5% |

本项目以 88.1% 作为 MCCNN 复现的参考值，而不是要求单次运行必须达到该数字。

### 2.2 DWT 参考论文

参考论文：*Semantic Word and Sentence Embeddings Compression using Discrete Wavelet Transform*（Findings of ACL 2024）。

本项目采用其中以下核心思想：

- 将每个词嵌入视为一维信号并独立执行 DWT；
- 分别使用近似系数 `cA` 和细节系数 `cD` 作为压缩表示；
- 使用一级 DWT 将嵌入维度减少约 50%；
- 使用 PCA 作为同维度降维对照；
- 在下游分类任务上评估压缩嵌入。

DWT 论文没有公开每项实验使用的具体母小波。本文档选择 `coif1` 是基于论文关于 Coiflets 整体表现较好的讨论作出的固定实验选择，不声称它是原论文某项实验可以确认的参数。

## 3. 数据集

### 3.1 数据来源

使用 Stanford NLP 官方发布的 2013 年原始 PTB 划分压缩包 `trainDevTestTrees_PTB.zip`：

- 来源：`https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip`
- 固定压缩包 SHA-256：`5c613a4f673fc74097d523a2c83f38e0cc462984d847b82c7aaf36b01cbbbfcc`
- 使用压缩包 `trees/` 目录中的树文件：

  - `train.txt`
  - `dev.txt`
  - `test.txt`

不使用 Hugging Face/GLUE 中包含 67,349 个短语级训练样本且隐藏测试标签的版本，也不重新划分数据。也不使用后续维护的 `stanfordnlp/sentiment-treebank` Git 仓库版本：其当前固定提交的二分类根节点计数为 6,921/872/1,818，与参考论文的 6,920/872/1,821 不一致。

### 3.2 数据解析

- 每棵树只提取根节点，得到完整句子；
- 叶节点序列直接作为已分词的 token 序列；
- 根节点原始标签为 `0` 或 `1` 时映射为负面标签 0；
- 根节点原始标签为 `3` 或 `4` 时映射为正面标签 1；
- 忽略根节点原始标签为 `2` 的中性句子；
- 不进行额外 tokenization、stemming、lemmatization 或停用词删除；
- 保留原始大小写和标点。

数据准备完成后必须验证：

| Split | 样本数 |
| --- | ---: |
| Train | 6,920 |
| Dev | 872 |
| Test | 1,821 |

这与 CNN 参考论文表 1 中的 SST-2 数量一致。数据来源 URL、原始压缩包 SHA-256 和各拆分文件 SHA-256 必须写入运行记录。

### 3.3 词表

为了匹配参考论文约 16K 的词表设置，主实验只使用 train 文本建立词表。原始 2013 SST-2 的 train 文本产生 16,282 个普通 token，加上 `<pad>` 和 `<unk>` 后词表大小为 16,284。dev/test 中未在 train 出现的 token 映射为 `<unk>`，不使用 dev/test 文本或标签建立词表、调参或拟合嵌入变换。该设置在结果中明确标记为 `train-only vocabulary`。

- 不设置最低词频；
- 特殊 token 为 `<pad>` 和 `<unk>`；
- 所有实验共享完全相同的 token 到 ID 映射；
- `<pad>` 的嵌入始终保持为零；
- batch 内动态 padding；
- 序列长度至少为最大卷积核宽度 7。

## 4. 原始词嵌入

使用 Google News Word2Vec 300d，即参考 CNN 论文使用的 100B Google News corpus 预训练词向量。

- Word2Vec 二进制文件由用户通过本地路径提供；
- 大型词向量文件不提交到仓库；
- 训练程序不在运行时自动下载该文件；
- 词表中的 token 优先按原始形式匹配 Word2Vec；
- 未匹配词使用由实验随机种子控制的 300 维 `Uniform(-0.25, 0.25)` 随机向量；该范围是论文未提供可复现标准差时采用的明确复现假设；
- `<pad>` 使用 300 维零向量。

每个正式实验 seed 构造一份固定的原始 300 维预训练通道矩阵，再从同一 seed 的矩阵产生 DWT 和 PCA 版本。四种方法在同一 seed 下共享相同的词表、OOV 判定和基础随机初始化结果。

## 5. MCCNN 模型

### 5.1 两个嵌入通道

MCCNN 使用两个相同维度的 lookup table：

1. 预训练通道：根据实验使用 Original、DWT 或 PCA 词向量初始化；
2. 随机通道：使用相同维度的随机向量初始化。

两个通道都允许在训练中更新。对于压缩实验，随机通道直接在目标维度初始化，不先构造 300 维随机向量再压缩。

### 5.2 网络结构

```text
token ids
  -> pretrained embedding channel
  -> random embedding channel
  -> shared convolutions applied to each channel
     kernel widths: 5, 6, 7
     filters per width: 100
  -> ReLU on each channel feature map
  -> sum corresponding channel feature maps
  -> max-over-time pooling
  -> concatenate 300 pooled features
  -> dropout
  -> fully connected binary classifier
```

固定设置：

- 卷积核宽度：`[5, 6, 7]`；
- 每种宽度 100 个 filters；
- 池化：max-over-time pooling；
- Dropout：0.5；
- 输出类别数：2；
- 损失函数：交叉熵。
- 共享卷积 filter 按论文公式分别作用于两个通道，每个通道独立经过 bias 和 ReLU 后再求和；
- 随机嵌入通道、卷积权重和分类器权重使用零均值截断正态初始化；由于论文遗漏标准差数值，本项目固定 `std=0.1`，bias 初始化为零；
- 两个嵌入通道在每次参数更新后都将 `<pad>` 行强制重设为零。

嵌入维度从 300 降到 150 时，卷积层的参数量也会随之减少。这是压缩带来的实际模型规模收益，报告时必须把嵌入矩阵、卷积分类器和完整模型的参数量分别列出。

## 6. 嵌入压缩

### 6.1 DWT

DWT 是训练前的一次性嵌入预处理，不在每次前向传播中重复执行。

```text
Word2Vec vector (300d)
  -> one-level 1D DWT
  -> cA (150d) and cD (150d)
```

固定设置：

- 库：PyWavelets；
- 母小波：`coif1`；
- 层级：Level 1；
- 边界模式：`periodization`；
- `dwt_ca` 实验只保留 `cA`；
- `dwt_cd` 实验只保留 `cD`；
- 目标维度：150；
- `<pad>` 变换后强制重设为零向量；
- DWT 输出转换为与模型一致的浮点类型并缓存。

本阶段不搜索最佳母小波，不组合 `cA` 与 `cD`，也不执行多级 DWT。

### 6.2 PCA

- 库：scikit-learn；
- `n_components=150`；
- 不使用 whitening；
- 使用完整 SVD，避免随机近似带来的波动；
- 只使用训练集中实际出现、且能在 Word2Vec 中匹配的 token 向量拟合 PCA；
- 不使用分类标签；
- 已拟合 PCA 用于变换完整固定词表中的预训练通道向量；
- OOV 的 300 维随机向量也通过同一个已拟合 PCA 投影；
- `<pad>` 投影后强制重设为零向量；
- 保存 PCA 的均值、分量和相关元数据。

## 7. 核心实验矩阵

| 实验 | 预训练通道 | 随机通道 | 维度 | 两通道是否训练 |
| --- | --- | --- | ---: | --- |
| Original | Word2Vec | Random | 300 | 是 |
| DWT-cA | Word2Vec → DWT `cA` | Random | 150 | 是 |
| DWT-cD | Word2Vec → DWT `cD` | Random | 150 | 是 |
| PCA | Word2Vec → PCA | Random | 150 | 是 |

公平性约束：

- 四组实验使用相同 SST-2 划分和词表；
- 使用相同的网络结构，只有嵌入维度随压缩变化；
- DWT 与 PCA 输出维度严格一致；
- 使用相同训练协议和随机种子集合；
- 不针对某种嵌入单独调参；
- 开发集用于选择 checkpoint，测试集不参与调参；
- 不以最佳 seed 代替多次运行统计结果。

## 8. 训练协议

固定设置：

- Batch size：50；
- 最大 epochs：30；
- 优化器：Adadelta；
- 学习率：1.0；
- `rho=0.95`；
- `eps=1e-6`；
- Early stopping 指标：Dev Accuracy；
- Early stopping patience：5；
- 每个 epoch 后评估开发集；
- 保存开发集 Accuracy 最优的 checkpoint；
- 最终只用该 checkpoint 评估测试集。

参考论文对学习率、训练轮数、停止规则和部分正则化参数的说明不完整，且其 `1e-3 → 1e-4` 表述与常见 Adadelta 实现存在歧义。因此本项目采用以上固定且可复现的现代实现协议，并将其记录为复现假设。如果 MCCNN 基准与论文结果存在明显差距，先分析数据、模型和训练日志，再决定是否单独开展学习率敏感性实验。

### 8.1 随机种子

开发时可使用一个种子完成冒烟测试。正式实验使用以下 5 个固定种子：

```text
13, 21, 42, 87, 100
```

种子控制 Python、NumPy、PyTorch、DataLoader、随机通道和 OOV 向量初始化。运行记录必须说明是否启用确定性算法以及硬件限制导致的潜在非确定性。

## 9. 评价指标

### 9.1 分类性能

- Test Accuracy；
- Test Macro-F1；
- 最佳 epoch；
- 5 个种子的均值和标准差；
- MCCNN 平均 Accuracy 与论文 88.1% 的绝对差值。

结果表同时展示论文值和本项目统计值：

```text
Paper MCCNN: 88.1%
Our MCCNN: mean ± std
Difference: our mean - 88.1%
```

### 9.2 模型规模与压缩

- 嵌入维度；
- 压缩率：`1 - compressed_dim / 300`；
- 预训练嵌入通道参数量与字节数；
- 随机嵌入通道参数量与字节数；
- 卷积与分类器参数量；
- 完整模型参数量；
- checkpoint 文件大小。

### 9.3 运行效率

- 每个 epoch 的训练时间；
- 总训练时间；
- 固定 batch size 下的推理延迟；
- 推理吞吐量（samples/s）；
- CPU 峰值内存；
- 使用 CUDA 时的峰值显存；
- 测量设备、线程数、batch size、预热次数和重复次数。

推理计时先预热并重复多轮。CUDA 测量前后执行同步，避免异步执行导致计时错误。不同嵌入方法必须在同一硬件和同一测量协议下比较。

## 10. 技术栈与命令设计

### 10.1 技术栈

- Python 3.11；
- PyTorch 2.x：模型、训练与推理；
- NumPy：数值与嵌入处理；
- PyWavelets：DWT；
- scikit-learn：PCA、Accuracy 和 Macro-F1；
- Gensim：读取 Google News Word2Vec 二进制文件；
- NLTK：解析 SST 树结构；
- psutil：CPU 内存测量；
- pandas：汇总实验；
- PyYAML：配置文件；
- pytest：测试。

不依赖 TorchText 或 Hugging Face 获取 SST-2。

### 10.2 命令职责

项目提供统一命令入口，至少覆盖：

```text
prepare-data        下载或解析固定的 2013 SST PTB 划分，并验证样本数量
prepare-embeddings  加载 Word2Vec，生成 Original、DWT 和 PCA 缓存
train               按配置训练一个实验和一个 seed
evaluate            加载最佳 checkpoint 评估分类指标
benchmark           测量模型大小、内存、延迟和吞吐量
summarize           汇总所有 seed 和嵌入方法的结果
```

Original、DWT-cA、DWT-cD 和 PCA 共享训练代码，通过配置选择嵌入方法，禁止复制成四套独立训练实现。

## 11. 建议目录结构

```text
dwt-textcnn-classification/
├── configs/
│   ├── base.yaml
│   ├── original.yaml
│   ├── dwt_ca.yaml
│   ├── dwt_cd.yaml
│   └── pca.yaml
├── data/                         # 不提交大型数据和词向量
├── doc/                          # 两篇参考论文
├── outputs/                      # 运行结果和 checkpoints
├── scripts/
│   ├── prepare_data.py
│   ├── prepare_embeddings.py
│   ├── run_experiment.py
│   └── summarize_results.py
├── src/dwt_textcnn/
│   ├── config.py
│   ├── data.py
│   ├── vocabulary.py
│   ├── embeddings.py
│   ├── compression/
│   │   ├── dwt.py
│   │   └── pca.py
│   ├── models/
│   │   └── mccnn.py
│   ├── training.py
│   ├── evaluation.py
│   ├── profiling.py
│   └── reproducibility.py
├── tests/
│   ├── test_sst_parser.py
│   ├── test_vocabulary.py
│   ├── test_dwt.py
│   ├── test_pca.py
│   ├── test_mccnn.py
│   └── test_experiment_smoke.py
├── PROJECT_DESIGN.md
├── README.md
└── pyproject.toml
```

## 12. 结果文件规范

每次运行生成独立目录：

```text
outputs/sst2/<method>/<seed>/<run-id>/
├── config.yaml
├── environment.json
├── data_manifest.json
├── embedding_manifest.json
├── metrics.json
├── profile.json
├── history.csv
└── best_model.pt
```

运行记录至少包含：

- 数据来源和版本；
- 词表大小与 OOV 数量；
- Word2Vec 文件标识；
- DWT 或 PCA 参数；
- 嵌入维度；
- 随机种子；
- 软件版本和硬件信息；
- Accuracy、Macro-F1 和最佳 epoch；
- 参数量、文件大小、内存和时间指标。

## 13. 测试与验收标准

### 13.1 数据测试

- SST 解析结果严格为 6,920/872/1,821；
- 每条样本只使用树根标签和完整叶子序列；
- 四组实验的词表和 token ID 完全一致；
- 动态 padding 后的最短序列长度不小于 7。

### 13.2 嵌入测试

- Original 维度为 300；
- DWT-cA 和 DWT-cD 维度均为 150；
- PCA 维度为 150；
- 相同输入和设置生成相同的嵌入缓存；
- PCA 只在训练文本对应的已匹配 Word2Vec token 上拟合；
- 所有方法的 `<pad>` 向量严格为零；
- 缓存元数据能验证方法、维度和来源。

### 13.3 模型与训练测试

- MCCNN 能接受 300 维和 150 维双通道输入；
- 三种卷积宽度的输出均能正确池化和拼接；
- 前向和反向传播可完成；
- 保存并重新加载 checkpoint 后预测一致；
- 小规模冒烟实验能完成训练、验证、测试和结果保存。

### 13.4 项目验收

- 一条命令可准备并验证 SST-2；
- 一条命令可为指定方法准备嵌入；
- 一条命令可运行指定方法和 seed；
- 能完成 MCCNN Original 基准复现；
- 能完成 Original、DWT-cA、DWT-cD 和 PCA 的 5-seed 正式实验；
- 能自动生成分类性能、模型规模、内存和速度对比表；
- 所有正式结果可通过保存的配置、manifest 和 seed 追溯。

## 14. 实施顺序

1. 建立 Python 包、配置、测试和命令入口；
2. 实现并验证 Stanford 2013 SST PTB 树解析与固定词表；
3. 实现 Word2Vec 加载和 Original 嵌入缓存；
4. 实现 MCCNN，并用单 seed 完成基准冒烟实验；
5. 调试 MCCNN 基准并与论文 88.1% 进行初步比较；
6. 实现 DWT-cA、DWT-cD 和 PCA；
7. 完成四种方法的单 seed 端到端验证；
8. 运行 5-seed 正式实验；
9. 实现并执行统一效率测量；
10. 汇总结果、分析差异并完成项目报告；
11. 基础项目完成后，再决定是否扩展 TF-MCCNN 或论文研究问题。

## 15. 已知限制与风险

- CNN 论文没有完整公开所有训练和初始化细节，无法保证数值完全一致；
- CNN 论文未提供可直接复现的随机初始化标准差；本项目将 OOV 初始化固定为 `Uniform(-0.25, 0.25)` 并记录到 manifest；
- 对论文未给出的截断正态标准差，本项目固定使用 `std=0.1` 并记录到配置和运行结果；
- DWT 论文没有公开每项实验的具体母小波，`coif1` 是本项目固定选择；
- 后续维护的 Stanford SST Git 仓库与 2013 原始发布包存在样本差异；本项目固定使用原始发布包，不自动跟随仓库更新；
- 两个嵌入通道均训练可能逐步改变初始压缩表示，但所有方法保持一致；
- 参考论文只报告约 16K 的词表大小，没有明确公开词表构建算法；本项目采用与该数量吻合且避免 dev/test 文本信息泄漏的 train-only 词表；
- PCA 的中心化会使零向量投影为非零，因此必须显式重置 `<pad>`；
- 300 维与 150 维模型的卷积参数量不同，准确率和效率必须同时报告；
- 单次运行结果受随机性影响，正式结论必须基于 5 个种子；
- 硬件和软件版本会影响计时与内存结果，必须保存环境信息；
- Google News Word2Vec 文件较大且不纳入仓库，复现者需要自行提供。

## 16. 当前阶段明确不做的内容

- TF-MCCNN、TF-DCNN；
- Transformable Convolution 或 Transformable Pooling；
- 多级 DWT；
- 母小波搜索；
- `cA` 与 `cD` 融合；
- 除 SST-2 外的其他数据集；
- 现代 Transformer 基线；
- 可训练小波、维度重排或论文级新方法。

这些内容只有在基础项目完成并分析结果后，才重新讨论是否加入。
