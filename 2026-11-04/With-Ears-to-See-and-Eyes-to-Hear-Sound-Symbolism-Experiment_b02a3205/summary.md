---
title: "With-Ears-to-See-and-Eyes-to-Hear-Sound-Symbolism-Experiment"
source: https://aclanthology.org/2024.emnlp-main.167.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:30:44"
field: "多模态大语言模型评估"
keywords: ["sound symbolism", "Kiki-Bouba effect", "magnitude symbolism", "multimodal LLMs", "iconicity", "psycholinguistics", "VLM evaluation"]
innovations: ["首次在真实实体上扩展Kiki-Bouba形状象征实验并引入幅度象征(Mil-Mal)", "系统性评估多种规模LLMs的象似性理解能力并揭示非线性缩放关系", "提出informed prompt引导策略显著改善VLMs声音象征表现"]
benchmarks: ["Winter et al. (2023) Iconicity Ratings", "Kiki-Bouba Shape Symbolism", "Mil-Mal Magnitude Symbolism"]
---

# 论文速读：With-Ears-to-See-and-Eyes-to-Hear-Sound-Symbolism-Experiment

## 一句话总结
本研究通过三项经典心理语言学实验（Kiki-Bouba形状象征、Mil-Mal幅度象征、象似性评分），系统评估了多模态大语言模型（VLMs）和纯文本大语言模型（LLMs）仅基于视觉和文本模态时，能否隐式理解声音象征现象，并探索其在类人感知研究中的潜力与局限。

## 研究问题与动机
- 现有研究已证明LLMs/VLMs可作为人类被试的潜在替代品用于心理语言学实验，但一个未被充分研究的问题是：仅具有视觉和文本模态访问权限的模型，能否仅通过正字法和图像信息的抽象推理来隐式理解基于声音的现象？
- 传统Kiki-Bouba和Mil-Mal实验均依赖人类听觉感知，而模型无法直接"听到"语音，需探究模型能否通过拼写模式（grapheme combinations）和训练数据中的元级文本讨论间接习得声音象征关联。
- 若模型能展现声音象征能力，将对NLP下游任务（如情感分析、情感识别、创意生成）产生积极影响，同时为产品命名等营销场景提供低成本预测试方案。
- 不同模型架构（开放vs闭源）、不同规模（7B/13B/34B/70B）在声音象征任务上的表现差异尚不明确，需系统对比分析。

## 核心贡献（创新点）
1. **首次在真实世界实体上扩展Kiki-Bouba实验**：不同于Alper & Averbuch-Elor (2023)在抽象形状上的研究，本文使用DALL-E 3生成真实实体图像，测试模型将伪词分配给具体视觉刺激的能力，提高了生态效度。
2. **系统性验证幅度象征（Mil-Mal效应）**：首次将声音象征实验扩展至幅度象征领域，通过创建最小对立体（minimal pairs如Dil/Dal、Zil/Zal等）避免英语现有词汇的干扰，全面评估模型对声音-大小关联的理解。
3. **大规模象似性评级对比分析**：基于Winter et al. (2023)的14,772词数据集，系统评估从250M到70B参数的多种开源/闭源LLMs的象似性判断能力，揭示模型规模与象似性理解之间的非线性关系。
4. **引入"informed prompt"实验设计**：通过在提示词中明确告知模型任务相关的声音象征知识，探索是否能引导模型展现更接近人类的感知偏好，为后续提示工程提供方法参考。

## 方法详解
- **形状象征实验（Kiki-Bouba）**：使用DALL-E 3生成50张图像（25张spiky、25张rounded），包含alien、bed、bird等26个实体名词。从既有文献选取6对伪词：Kalika-Mabobe、Zaki-Umbu、Tiki-Giba、Kitaki-Gugagu、Hatiha-Bodubo、Penape-Gunogu，以及原始Kiki-Bouba作为基准。采用zero-shot提示："Look at the [ENTITY] in the provided image. Out of the following two options, which name would you most likely assign to the [ENTITY]: '[KIKI-WORD]' or '[BOUBA-WORD]'. Respond with only your decision"。额外设置"informed"版本，在提示开头添加声音象征现象的定义说明。每对伪词正负位置各呈现一次以消除位置偏差。
- **幅度象征实验（Mil-Mal）**：使用相同实体列表，提示词改为"Generate an image of a/an [ENTITY] in isolation, with something else to help judge scale/perspective"。创建6对最小对立体：Dil/Dal、Zil/Zal、Geel/Gaal（利用/I/-/a/对比）和Beel/Baal、Weel/Waal、Leel/Laal（利用/i/-/A/对比），避免真实英语词汇的语义干扰。实验设置与形状象征相同。
- **象似性评分实验**：采用Trott (2024)的提示策略，在1-7分量表上要求LLMs对Winter et al. (2023)的单词进行象似性评级（1=完全不象似，7=高度象似）。评估模型包括GPT-4、GPT-3.5-Turbo、LLaMA-2（7B/13B/70B）、FLAN-T5（base/XL）、Mistral-7B。使用Spearman和Pearson相关系数与人类评级对比。
- **辅助实验-声音景观描述**：额外测试GPT-4根据图像描述声音景观并使用拟声词的能力，使用ImageNet真实图像和DALL-E 3生成图像两种条件，由5名人类评估者进行1-5分量级评价。

## 实验与结果
- **数据集与模型**：DALL-E 3生成的100张实体图像；评测模型包括GPT-4、Gemini Pro、LLaVA（7B/13B/34B）、GPT-3.5-Turbo、LLaMA-2（7B/13B/70B）、FLAN-T5（base/XL）、Mistral-7B；人类对照组为10名母语英语水平评估者。
- **形状象征结果**：整体 agreement 水平较低，多数条件处于或接近 Chance（50%）。GPT-4在原始Kiki-Bouba和Kalika-Mabobe条件下表现最佳（86%/66%），Gemini在Zaki-Umbu和Hatiha-Bodubo上最优（73%/78%）。"Informed"提示普遍提升performance（如Gemini在Zaki-Umbu从64%提升至73%），但部分条件出现下降（如Penape-Gunogu对Gemini从38%降至26%）。LLaVA 13B在Tiki-Giba上显著优于7B和34B版本，34B整体表现最佳。
- **幅度象征结果**：整体表现优于形状象征。GPT-4在多个条件下达到90%+ agreement（Zil-Zal: 93%, Weel-Waal: 90%）。"Informed"提示带来更显著的提升（如GPT-4在Zil-Zal从58%跃升至93%，κ从0.149增至0.860），表明模型理解声音-大小关系但需明确指引。LLaVA 13B同样在多数条件下优于7B和34B。
- **象似性评分结果**：Spearman相关系数显示模型能力高度依赖规模。FLAN-T5（250M/3B）和LLaMA-2 7B几乎无相关性（≈0）；Mistral-7B-Instruct达0.382；LLaMA-2 13B达0.379；GPT-3.5-Turbo达0.420；GPT-4（Nov '23）达0.537，低于Trott (2024)报告的早期版本0.575。值得注意的是LLaMA-2 70B（0.304）低于13B（0.379），呈现非线性缩放关系。
- **声音景观描述**：GPT-4在真实图像和生成图像的visual/soundscape/onomatopoeia三个维度上均获得人类评分4.0+（满分5），证明其能将视觉信息转换为 convincingly 的听觉描述。

## 相关工作脉络
1. **Usnadze (1924)**：最早系统研究声音象征的心理语言学实验，向10名参与者展示伪词与绘图配对，发现高于随机水平的跨被试一致性，奠定实验范式基础。
2. **Ramachandran & Hubbard (2001)**：提出著名Kiki-Bouba效应，证明尖刺形状被命名为"Kiki/Takete"、圆润形状被命名为"Bouba/Maluma"，揭示声音-形状的非任意关联。
3. **Sapir (1929)**：发现Mil-Mal效应，高前元音/I/与小型实体关联、低后元音/A/与大型实体关联，建立幅度象征的理论框架。
4. **Alper & Averbuch-Elor (2023)**：使用CLIP和Stable Diffusion在抽象形状上验证Kiki/Bouba效应，本文在其基础上扩展至真实实体并引入幅度象征和象似性评分。
5. **Trott (2024)**：使用GPT-4评估象似性评级，报告Spearman相关0.59；本文复现时发现数值差异（0.537 vs 0.575），揭示模型版本迭代对评估结果的影响。
6. **Winter et al. (2023)**：收集14,772个英语单词的大规模象似性人类评级数据集，为本文象似性实验提供评估基准。

## 局限性与未来方向
- **样本量有限**：伪词对数量较少（6-7对），作为proof-of-concept研究，需要更大规模验证。
- **语言单一性**：尽管声音象征被认为 largely language agnostic，但研究仅使用英语母语者和符合英语音系规则的伪词。
- **数据污染风险**：部分伪词来自既有文献，虽通过互联网搜索检查其 prevalence，但仍可能存在训练数据污染（尽管低performance降低了此担忧）。
- **人类评估限制**：向人类参与者呈现的是正字法形式而非语音形式，可能导致发音变异影响结果。
- **模型规模非线性**：LLaMA-2 70B象似性表现低于13B版本，提示单纯扩大参数并非唯一解决方案。
- **未来方向**：显式在训练中融入声音象征数据或任务、通过提示生成文本解释以理解模型决策机制、分析VLMs的视觉注意力是否聚焦于目标特征（如尖刺/圆润）。

## 研究启发与可借鉴点
1. **"Informed prompt"设计范式**：通过提供任务背景的元知识显著提升模型performance（幅度象征提升尤为明显），为后续研究提供了"知识引导"的提示策略参考。
2. **最小对立体（minimal pairs）的伪词构建**：为避免真实词汇语义干扰，本文系统创建 phonetically similar 的最小对立体（如Dil/Dal、Zil/Zal），该方法可迁移至其他语音象征研究中。
3. **DALL-E 3生成图像的生态效度提升**：使用生成模型创建新颖刺激材料而非依赖传统抽象形状，既减少记忆偏差又提高现实场景适用性，为实验材料构建提供新路径。
4. **多模型规模对比揭示非线性规律**：LLaMA-2系列（7B/13B/70B）的象似性表现呈现非线性趋势，提示模型评估需关注规模与任务能力的复杂关系，而非简单假设线性缩放。
5. **类人评估的统计学严谨性**：使用Fleiss' κ衡量人类间一致性、Cohen's κ衡量模型-人类 agreement，结合chance-level（50%）基准，为计算心理学评估提供了标准化方法框架。

## 关键术语表
**Sound Symbolism（声音象征）**：语音形式与概念意义之间的非任意关联现象，如拟声词"bang"模仿 loud noise。
**Kiki-Bouba Effect（Kiki-Bouba效应）**：经典形状象征现象，尖刺形状被命名为"Kiki/Takete"、圆润形状被命名为"Bouba/Maluma"。
**Magnitude Symbolism（幅度象征）**：元音频率与实体感知大小之间的关联，高前元音/I/关联小型实体，低后元音/A/关联大型实体。
**Iconicity（象似性）**：词语形式与其所指概念之间的相似程度，如拟声词具有高象似性。
**Informed Prompt（知情提示）**：在标准提示前添加任务背景知识的增强提示，如说明声音象征的定义以引导模型关注特定特征。
**Minimal Pairs（最小对立体）**：仅在一个音素上不同的词对，如Dil/Dal，用于控制实验中分离特定语音特征的影响。
**Grapheme-to-Phoneme**：从正字法到音位的转换过程，文本模型借此通过拼写模式间接习得语音关联。
**Fleiss' κ / Cohen's κ**：衡量多人标注者间一致性（Fleiss' κ）或模型与人类多数投票间一致性（Cohen's κ）的统计指标。

## 可复现要素
- **数据集**：DALL-E 3生成的实体图像（附录A.1列出完整名词列表）；Winter et al. (2023)的14,772词象似性数据集；ImageNet子集用于声音景观实验。
- **代码/权重**：LLaVA模型通过Hugging Face公开（llava-v1.6-vicuna-7b/13b/34b-hf）；LLaMA-2、FLAN-T5、Mistral-7B均公开；GPT-4、Gemini Pro通过API访问。完整实验细节见附录A.1。
- **关键超参**：max_tokens=10（VLMs输出）；所有模型使用默认超参数测试"off-the-shelf"能力；LLaVA额外添加A/B choice labels以获取单token概率。
- **提示模板**：象似性评分完整prompt见附录C.2；声音景观描述prompt见正文B.1节。
