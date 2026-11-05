---
title: "Table-Question-Answering-for-Low-resourced-Indic-Languages"
source: https://aclanthology.org/2024.emnlp-main.5.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:30:00"
field: "低资源自然语言处理"
keywords: ["低资源表格问答", "数据自动生成", "跨语言零样本迁移", "印地语", "孟加拉语", "SQL2NQ", "质量控制系统"]
innovations: ["面向低资源语言的TableQA自动化数据生成与质量控制流水线", "基于微调语义相似度模型与直方图拐点的合成SQL-NQ对质检方法", "在孟加拉语与印地语上训练的开源自监督模型超越GPT-4等闭源LLM基线"]
benchmarks: ["BanglaTabQA", "HindiTabQA"]
---

# 论文速读：Table Question Answering for Low-resourced Indic Languages

## 一句话总结
本文首次面向低资源语言提出了可扩展的表格问答（TableQA）数据自动生成与质量控制流水线，并在孟加拉语（BanglaTabQA）和印地语（HindiTabQA）上构建大规模数据集与模型；训练后的开源模型在表格精确匹配上超越GPT-4等最强闭源LLM基线，并展现出跨语言零样本迁移能力。

## 研究问题与动机
1. **数据稀缺与文化代表性不足**：低资源语言缺乏高质量的TableQA标注数据；直接翻译英文表格（如WikiTableQuestions等）会丢失文化特有实体与地域性事实。
2. **神经网络模型对齐差**：现有主流模型（如OdiaG、Encoder-Decoder翻译微调模型）对低资源语言脚本与数字体系缺乏良好对齐，无法理解半结构化表格。
3. **表格翻译不可靠**：Minhas et al.(2022)指出，表格并非规范句子，机器翻译错误会在下游QA任务中产生误差传播；因此将英文TableQA数据直译不可行。
4. **缺乏针对低资源语言的系统评估基准**：此前低资源NLP工作主要集中在文本QA，缺少面向半结构化表格的数学推理、集合/关系运算等评测维度。

## 核心贡献（创新点）
1. **首次定义低资源TableQA任务并以印欧语系语言为对象**：提出在孟加拉语和印地语上开展系统性的TableQA研究与评测，区别于以往以高资源语言为主的工作。
2. **设计低成本、可扩展的自动数据生成流程**：以"表格抽取 → 代码混合SQL生成 → 自然语言问题生成 → 答案表格抽取 → 自动质量控制"为核心，通过SQUALL模板与native-speaker映射实现可复用。
3. **提出针对低资源代码混合SQL-NQ对的语义相似度质量控制方案**：发现预训练LaBSE对低资源SQL-NQ配对判别力不足（均值0.55），据此微调出SQL2NQSim模型并基于直方图拐点设定阈值（0.74），显著优于现成句子相似度模型。
4. **发布大规模数据集与模型并验证跨语言泛化**：发布BanglaTabQA（约2M训练/2K验证/165测试）与HindiTabQA（约643K训练/645验证/125测试），展示跨语言零样本迁移与不同算子类别上的数算推理能力。
5. **证明开源小模型可通过高质量低资源数据超越闭源LLM**：BnTQA-mBart在孟加拉语测试集上Table EM达35.88%，高于GPT-4的26.83%；HiTQA-BnTQA在印地语上达到41.32%。

## 方法详解
1. **表格抽取**：直接从孟加拉语/Wikipedia dumps与印地语Wikipedia dumps解析表格，避免英文→低资源语言的机翻失真与文化缺失。
2. **代码混合SQL生成**：使用SQUALL模板库实例化占位符，SQL关键字保持英文、表和列名/值使用低资源语言（孟加拉语/印地语），生成可执行代码混合SQL；孟加拉语侧共生成13,345,000条可执行SQL。
3. **SQL到自然语言问题（SQL2NQ）生成**：
   - 先由精通SQL的母语者建立27个SQL关键字到低资源语言的一对一映射，将代码混合SQL转为单语SQL-like查询（去掉FROM与表名后为良好句子）。
   - 以mbart-50-large为骨干，使用从Spider、WikiSQL、ATIS、Geoquery翻译得到的68,512训练/9,996验证样本微调SQL2NQ模型，再把单语SQL-like转为NL问题。
4. **答案表格抽取**：将低资源语言Wikipedia表格导入关系数据库，使用SQL编译器执行生成的SQL得到答案表，确保答案与输入严格对应。
5. **自动质量控制**：
   - 丢弃编译报错的SQL及样本。
   - 对SQL-NQ对使用SQL2NQSim（LaBSE微调）计算语义相似度；以10,000正负样本直方图的拐点选阈，孟加拉语阈值为0.74，最终保留约2,050,296训练/2,053验证样本。
6. **任务定义（序列到序列）**：输入为$Q$拼接线性化表$T$，使用特殊哨兵token分隔列/行；输出为线性化答案表$T_{out}$，包含列头与单元格值。
7. **质量评估与算子分类**：按SQL关键词数量划分难度；按算子分为arithmetic、sorting、group by、filtering、set、logical六类，并以此分析数学推理能力。

## 实验与结果
1. **数据集规模**：
   - BanglaTabQA：19K Wikipedia表格；训练2,050,296；验证2,053；测试165（人工标注，Cohen's/Fleiss' Kappa=0.82，平均流畅度4.3/5）。
   - HindiTabQA：1,921 Wikipedia表格；训练643,434；验证645；测试121–125。
2. **基线模型**：
   - En2Bn/En2Hi：mbart-50-large在从MultiTabQA译来的25K样本上微调。
   - OdiaG：Llama-7b + LoRA指令微调基线。
   - OpHathi：OpenHathi-7B-Hi-v0.1 2-shot ICL。
   - GPT-3.5/4：2-shot ICL。
3. **主要结果（测试集，Table EM/Row F1/Col F1/Cell F1）**：
   - **孟加拉语**：GPT-4最优基线（Tab=26.83，Row=38.67，Col=26.74，Cell=36.51）；**BnTQA-mBart**达Tab=35.88，显著超过GPT-4；BnTQA-M2M Tab=28.05；**BnTQA-llama**过拟合验证集，测试仅Tab=9.41。
   - **印地语**：**HiTQA-BnTQA**（以BnTQA-mBart冷启动再微调）达Tab=41.32，为最高；HiTQA-mBart Tab=33.06；HiTQA-M2M Tab=28.93；HiTQA-llama Tab=7.29。所有HiTQA模型均超2-shot GPT/OpHathi基线。
4. **跨语言零样本迁移**：BnTQA-mBart在Hindi测试上未微调时Tab仅19.01；经后处理（字典替换数字与SQL关键字、Google翻译表头/值）提升至Tab≈19.01/Row 20.74/Col 19.01/Cell 20.74，表明表结构与推理能力可跨脚本迁移。
5. **按算子类别分析**：BnTQA-mBart在groupBy上Tab=50.00%最高；HiTQA-mBart在sorting上Tab=39.05%最高；说明数据集能有效提升低资源语言下的算术与表操作推理。

## 相关工作脉络
1. **TaPas (Herzig et al., 2020) / TaBERT (Yin et al., 2020)**：高资源英文抽取式/联合式TableQA，侧重英文弱监督预训练；本文聚焦低资源语种与自生成大规模数据。
2. **MultiTabQA (Pal et al., 2023)**：英文多表生成式TableQA与SQUALL模板；本文将其思想迁移到低资源语言并用母语数据库执行SQL以确保答案正确性。
3. **Spider (Yu et al., 2018) / WikiSQL (Zhong et al., 2017)**：英文text-to-SQL基准；本文利用其翻译样本微调SQL2NQ模型，而不是直接复用英文SQL。
4. **TAPEX (Liu et al., 2021) / TAT-QA (Zhu et al., 2021)**：英文表格预训练与金融领域混合数据；本文强调文化本体与本地脚本/数字的对齐。
5. **OMNITAB (Jiang et al., 2022) / FeTaQA (Nan et al., 2022)**：英文开放域与自由表单TableQA；本文在低资源语境下引入算子分析与跨语言迁移评测维度。
6. **IndicBART / OpenHathi / IndicTrans2**：低资源印度语言文本预训练与翻译模型；本文指出这些模型在结构化表格理解与数字算子上仍需大规模任务适配数据。

## 局限性与未来方向
1. **依赖在线可用无标注表格**：方法适用于在网络上有Wikipedia等表格来源的低资源语言；对几乎无网络存在的极弱势语言效果受限。
2. **SQUALL模板不支持多表连接与复杂查询**：当前只覆盖单表SQL操作；跨表联结与复杂嵌套查询需扩展模板库或引入更强text-to-SQL模块。
3. **大模型（Llama-7b）在PEFT设置下泛化较弱**：BnTQA-llama在测试集上过拟合严重，需进一步优化低资源非拉丁脚本的分词策略与参数对齐。
4. **测试集规模相对训练规模偏小**：由于人工标注成本限制，测试样本数量有限，难以全面覆盖长尾问题类型。
5. **未来可探索多语言联合预训练、自动扩展模板至多表/时间表推理，以及与更多低资源语言（如阿拉伯语、斯瓦希里语等）的迁移验证**。

## 研究启发与可借鉴点
1. **"SQL骨架 + 模板实例化 + 真实数据库执行"可复用于其他低资源语言的数据工厂**：先实例化可执行SQL、再从DB拉取真实答案，避免端到端模型生成的幻觉与答案漂移。
2. **用语义相似度做合成数据质量控制：fine-tune LaBSE-style模型+直方图拐点阈值**是一种可扩展且无需额外人工标注的质量筛选范式，适合其他语言的SQL-NQ或代码-自然语言配对数据清洗。
3. **算子级分组评测（arithmetic/sort/group/filter/set/logical）**可作为TableQA数据集/模型的诊断工具，帮助定位模型在特定推理类型上的短板。
4. **以高资源语言模型为冷启动做跨语言微调（如BnTQA → HiTQA-BnTQA）**取得显著提升（Tab 33.06 → 41.32），提示在多语表结构表示上可复用表征，降低单一语言从零起步的算力开销。
5. **后处理管道（字典替换数字/关键字 + MT翻译表头）可有效提升跨语言zero-shot性能**：为低资源跨语言表格任务提供了一条成本低、效果稳的提升路径。

## 关键术语表
**Table Question Answering (TableQA)**：在半结构化表格输入上回答自然语言问题，输出单元格或答案表格的任务。
**Low-resource language**：标注数据少、语料匮乏的语言，本文聚焦孟加拉语与印地语。
**Code-mixed SQL**：SQL关键字为英文、表/列/值为低资源语言的分号混合查询形式。
**SQL2NQSim**：在低资源SQL-NQ对上调优的LaBSE相似度模型，用于自动化数据质检。
**Table Exact Match (Tab EM)**：生成答案表与目标表完全一致的百分比指标。
**Operator class**：按SQL算子将问题分为arithmetic、sorting、group by、filtering、set、logical六类，用于诊断推理能力。
**Zero-shot cross-lingual transfer**：在未用目标语言微调的情况下，将一种低资源语言训练的模型直接应用于另一种语言测试。
**Parameter-efficient fine-tuning (PEFT/LoRA)**：冻结主干参数、仅训练低秩适配器以节约显存与训练成本。

## 可复现要素
- **数据集**：BanglaTabQA与HindiTabQA均已开源（论文声明提供数据集、模型与代码）。
- **代码/权重**：论文声明发布数据集、模型与代码；开源权重可在项目发布页获取（论文未给出具体URL，建议从作者主页/ACLAnthology补充链接检索）。
- **关键超参**：mbart-50-large与m2m100_418M训练batch=128；Llama-7b PEFT batch=16、4-bit量化；学习率1e-4；单次A6000 48GB GPU训练5个epoch；最大序列长度1024。
- **模板来源**：SQUALL数据集模板；翻译训练样本来自Spider/WikiSQL/ATIS/Geoquery（各68,512/9,996对）。
- **阈值选择**：SQL2NQSim采用直方图拐点定阈值，孟加拉语阈值为0.74。
