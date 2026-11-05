---
title: "Surveying-the-Dead-Minds-Historical-Psychological-Text-Analy"
source: https://aclanthology.org/2024.emnlp-main.151.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:29:23"
field: "历史心理学与低资源文本分析"
keywords: ["historical psychology", "classical Chinese NLP", "contextualized construct representation", "indirect supervised learning", "contrastive learning", "cross-lingual questionnaire conversion"]
innovations: ["提出CCR管道，将心理测量问卷与上下文语言模型结合，无需预设词典即可量化文言文中心理构造", "设计CQC跨语言转换流程，利用QuoteR将英文问卷映射为文言文引文", "构建首个中文历史心理学语料库C-HI-PSY并采用标题伪标签的间接监督对比学习微调"]
benchmarks: ["C-HI-PSY (STS/Easy, STS/Hard, QIC, PM)", "Song Dynasty officials' attitudes toward Wang Anshi reforms (traditionalism & authority correlation)"]
---

# 论文速读：Surveying-the-Dead-Minds-Historical-Psychological-Text-Analy

## 一句话总结
论文开发了名为 **Contextualized Construct Representation (CCR)** 的分析管道，将心理测量问卷与 Transformer 上下文语言模型相结合，从文言文中自动提取传统主义、规范强度、集体主义等心理构造；同时构建了首个中文历史心理学语料库 **C-HI-PSY**（21,539 段落 / 667 文章），并通过间接监督对比学习微调模型，在语义相似度、问卷项分类与心理测量三项任务上全面优于词嵌入基线（DDR）与 GPT-4 prompt 方法。

## 研究问题与动机
1. **语言与语料偏斜**：现有人工计算心理学/历史心理学文献几乎全部面向英语，古典中文等非英语历史语料的心理量化方法严重匮乏。
2. **方法层级的不足**：社会科学与 NLP 交叉研究长期依赖 bag-of-words / 词嵌入（如 Word2Vec、FastText、GloVe），无法捕捉句子级及以上的语境化语义，难以准确衡量复杂的心理构造。
3. **跨语言对齐困难**：成熟心理学量表（Collectivism、Norm Tightness/Looseness、Traditionalism 等）以英文自陈式问卷形式存在，与文言文在词汇、句法、文化语境上存在巨大鸿沟。
4. **低资源模型稀缺**：针对文言文的专用 Transformer 模型（如 `bert-ancient-chinese`、`guwenbert`）尚未被引入心理文本分析，且缺乏高质量的心理学标注数据。

## 核心贡献（创新点）
1. **提出 CCR 管道**：以心理测量问卷的"声明性句子"作为心理构造的表征锚点，用上下文语言模型计算目标文本与问卷项的余弦相似度（因子载荷），无需预先人工挑选词典词汇。
2. **设计跨语言问卷转换（CQC）流程**：利用多语言引文推荐模型 QuoteR 为每个英文问卷项匹配语义等价的文言文古引文，经人工过滤后构建古典中文版问卷。
3. **构建 C-HI-PSY 语料库**：收录 21,539 个段落、667 篇古文（《韩非子》《论语》《世说新语》《宋文鉴》等），按"忠孝""节义""栖逸""廉耻"等主题标注伪标题，填补中文历史心理学语料空白。
4. **提出间接监督对比学习方案**：以标题相似度为伪标签，通过阈值划分正/负样本对并构造 triplet，用 triplet loss 微调多种 Pre-trained Transformer，显著提升 CCR 在低资源古文上的表征能力。

## 方法详解
1. **跨语言问卷转换（CQC）**
   - 输入：英文心理学量表题项 $q_i$。
   - 使用 QuoteR（多语言引文推荐模型，训练语料含英文/现代中文/文言文）检索语义最接近的文言引文集合 $\{\tilde{q}_i\}$。
   - 人工筛选剔除质量低或偏离心理构造的引文，最终得到古典问卷 $\tilde{\mathcal{Q}}$。

2. **C-HI-PSY 语料构建与伪标签生成**
   - 筛选标准：篇章主题与心理学相关，且标题可明确反映核心道德/心理主题。
   - 使用在 1.04B token 古文语料上训练的 5 种词向量模型（Word2Vec/FastText/GloVe）中最优者 $E_T(\cdot)$ 对标题嵌入，计算余弦相似度 $\text{sim}(t_i, t_j)$ 作为段落相似度 $\text{sim}(s_i, s_j)$ 的**伪地面真值**。

3. **正负样本划分**
   - 阈值策略：以标题相似度分布的百分位数划分，最优设置为 **10th / 90th** percentile。
   - 正样本对：$\{(s_i, s_j)^+ \mid \text{cos}(E_T(t_i), E_T(t_j)) > \delta^+\}$ 或标题完全相同。
   - 负样本对：$\{(s_i, s_j)^- \mid \text{cos}(E_T(t_i), E_T(t_j)) < \delta^-\}$。

4. **Triplet 采样与对比学习微调**
   - 随机采样（Random）：从正/负对中随机抽取；Hard 采样：用预训练模型 $f_\theta(\cdot)$ 计算相似度，取最难区分样本。实验发现随机采样优于 hard sampling（间接监督噪声导致 hard 实例不可靠）。
   - Triples：$(s_A, s_A^+, s_A^-)$，每段作为 anchor 仅用一次。
   - 损失函数（Triplet Loss）：
     $$
     L_{triplet}(\theta) = \sum_{s_A \in S} \max\{\mathcal{D}^+ - \mathcal{D}^- + \alpha,\; 0\}
     $$
     其中 $\mathcal{D}^+ = \|f_\theta(s_A) - f_\theta(s_A^+)\|_2$，$\mathcal{D}^- = \|f_\theta(s_A) - f_\theta(s_A^-)\|_2$，$\alpha$ 为 margin。
   - 超参网格：Batch Size $\{16, 32\}$、Epochs $\{3\}$、Warmup $\{1,2,3\}$、LR $\{1e{-6}, 1e{-5}, 2e{-5}\}$、Optimizer Adam，最终最佳配置见 Table 6。

5. **CCR 推理流程**
   - 将目标段落与古典问卷项分别通过微调后的 Sentence Transformer 嵌入。
   - 计算每条段落与所有问卷项的平均余弦相似度，作为该心理构造的 **loading score**（心理测量得分）。

## 实验与结果
- **数据集与划分**：C-HI-PSY 共 21,539 段落，60% 训练 / 20% 验证 / 20% 测试，段落长度均值 195 字（<50 字合并、>500 字截断）。
- **三组任务**：
  - **STS（语义文本相似度）**：分 Easy（阈值抽样）与 Hard（随机抽样）两类。
  - **QIC（问卷项分类）**：4 套量表（Collectivism、Individualism、Norm Tightness/Looseness），各 15 项，SVM 10-fold CV。
  - **PM（心理测量）**：计算段落对 4 套问卷的平均 loading score。
- **最佳结果**：
  - QIC：`LERT+CoSENT` 微调后达 **97±0.7%** 准确率。
  - STS（Easy）：`BERT`（bert-ancient-chinese）微调后 Pearson **0.53±0.07** / Spearman **0.55±0.07**。
  - PM：`BERT` 微调后 Pearson **0.30±0.04** / Spearman **0.30±0.04**。
- **对比优势**：CCR 在所有任务上均**全面超越 DDR**（Word2Vec/FastText/GloVe）与 **GPT-4-turbo-2024-04-09 prompt**；GPT-4 在 STS Easy 上达 0.63 Pearson，但在 PM 任务（0.23）与 QIC（0.77）上落后于 CCR 最佳模型。
- **外部有效性验证**：以宋代官员王安石变法态度为基准，CCR 测得的 Traditionalism（r=-0.441***）与 Authority（r=-0.472***）与官员支持改革的程度呈显著负相关；OLS 回归含官阶控制与籍贯固定效应，$R^2$ 达 0.54–0.57。

## 相关工作脉络
1. **DDR（Distributed Dictionary Representation, Garten et al., 2018）**：用专家词表均值向量表征心理构造；本文 CCR 以问卷句子为锚点、用上下文模型替代非上下文词向量，捕获更丰富的语义。
2. **Sentence Transformer / SimCSE / CoSENT 系列**：用于句子级语义表示；本文将其与心理测量问卷结合，解决"如何用句子表示抽象心理构造"的问题。
3. **QuoteR（Qi et al., 2022）**：多语言引文推荐；本文首创性将其用于"英文问卷→文言引文"的跨语言转换，避免生硬翻译。
4. **历史语言 NLP**（Latin BERT、Ancient Greek、Ancient Hebrew 处理）：本文聚焦古典中文，首次将心理构造量化引入该语言。
5. **LIWC / 心理词典方法**：依赖手工词典；本文 CCR 无需预设词汇列表，直接利用百年心理学量表的声明性句子。
6. **间接监督（Indirect Supervision, He et al. 2021; Yin et al. 2023）**：本文以标题相似度作为段落相似度的伪标签，延续该范式并拓展至历史文本。

## 局限性与未来方向
- **间接监督噪声**：标题虽概括主旨，但段落中常含与标题无关内容，导致 pseudo ground truth 存在偏差；实验中 hard sampling 反而劣于 random sampling 即源于此。
- **幸存者偏差**：现存古籍以精英/士人阶层书写为主，所测心理指标可能无法代表当时普通民众的真实心理状态。
- **缺乏细粒度人工标注**：目前依赖伪标签训练，未来需构建专家标注的高质量心理构造数据集。
- **未来方向**：收集更多标注数据、扩展至其他历史语言、开源更多 benchmark 数据集以推动跨学科研究。

## 研究启发与可借鉴点
1. **"问卷句子→心理构造表征"的范式**：可将成熟心理学量表直接转化为锚点句子，用于任何具备上下文模型的低资源语言心理分析，无需重新设计词典。
2. **标题/摘要作为伪标签的间接监督策略**：适用于缺乏标注但存在结构性元数据（章节标题、小标题、关键词）的领域，可作为通用低资源表示学习方法。
3. **跨语言引文推荐（CQC）的思路**：对"现代外语→历史文献"的转换具有通用性，可迁移至拉丁文、古希腊文、梵文等历史语料的心理/主题分析。
4. **三重任务评估框架（STS+QIC+PM）**：从语义理解、分类区分到连续心理测量形成完整闭环，可为其他历史文本分析任务提供评估模板。
5. **与历史学/社会科学的结合**：本文已验证 CCR 结果与客观历史事件（王安石变法态度）的相关性，表明可复用该管道进行大尺度历史心理趋势追踪研究。

## 关键术语表
- **CCR（Contextualized Construct Representation）**：用上下文语言模型将心理构造表示为问卷句子的嵌入，通过余弦相似度衡量目标文本与该构造的匹配程度。
- **CQC（Cross-lingual Questionnaire Conversion）**：利用多语言引文推荐模型将英文心理问卷转换为文言文引文的流程。
- **C-HI-PSY**：Chinese Historical Psychology Corpus，首个面向文言文的心理学语料库，含 21,539 段落。
- **DDR（Distributed Dictionary Representation）**：基于词嵌入的词典表征方法，通过词向量均值定义心理构造。
- **间接监督（Indirect Supervision）**：利用文本内部结构（如标题相似度）作为伪标签，避免人工标注的低成本训练范式。
- **Triplet Loss**：对比学习损失，通过 margin 拉近正样本对、推远负样本对。
- **Loading Score**：段落与心理构造问卷项的平均余弦相似度，用于量化段落在该心理维度上的得分。
- **QuoteR**：多语言引文推荐模型，可用于在目标语料库中检索语义相近的历史引文。

## 可复现要素
- **数据集**：C-HI-PSY（21,539 段落 / 667 文章），论文中说明已构建，**是否开源未明确声明**。
- **代码**：**论文未提及**是否开源。
- **关键超参**：正负阈值 10th/90th percentile；Batch Size 16/32；Epochs 3；Warmup 1–3；Learning Rate 1e-6 ~ 2e-5；Optimizer Adam；Margin α 为 triplet loss 默认值。
- **基线模型**：`bert-ancient-chinese`、`guwenbert-base/large`、`paraphrase-multilingual-MiniLM-L12-v2`、`text2vec` 系列（MacBERT/ERNIE/LERT + CoSENT）。
- **外部验证数据**：Wang (2022) 手工标注的 137 位宋代官员对新法态度；来自《宋文鉴》（Zeng & Liu, 2006）。
