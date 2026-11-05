---
title: "QUITE-Quantifying-Uncertainty-in-Natural-Language-Text-in-Ba"
source: https://aclanthology.org/2024.emnlp-main.153.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:46:49"
---

# 论文速读：QUITE-Quantifying-Uncertainty-in-Natural-Language-Text-in-Ba

## 一句话总结
本文提出了QUITE数据集，用于评估语言模型在真实世界贝叶斯网络中的不确定性量化与概率推理能力；实验表明，通过将微调后的LLM作为语义解析器并将计算外包给概率逻辑求解器ProbLog的神经符号方法，在因果、证据及解释消除三种推理类型上均显著优于直接使用零样本或CoT提示的现成大语言模型。

## 研究问题与动机
- 现有贝叶斯推理评测数据集（如CLADDER、BLInD）多局限于二元随机变量、高度模板化的文本，且问题形式多为yes/no或排序，无法有效评估模型输出的概率估计与真实贝叶斯后验概率的相关性。
- 大语言模型（LLMs）在复杂高阶因果网络上的贝叶斯推断表现堪忧，缺乏原则性推理机制，且数学计算能力在面对未见分布或复杂结构时易出现系统性下降。
- 人类常使用模糊量词（Words of Estimative Probability, WEPs）表达不确定性，现有工作多聚焦于NLI语境下的蕴含判断，缺乏结合真实领域贝叶斯网络并要求输出细粒度数值概率的评测设置。
- 需要验证将神经符号架构（结构化概率逻辑编程 + LLM语义解析）应用于语言变体更丰富、非模板化的真实因果场景时，是否具备可扩展性与推理鲁棒性。

## 核心贡献（创新点）
- 提出QUITE数据集：首个涵盖真实世界贝叶斯网络、支持类别型随机变量、提供数值型与WEPs双版本前提的自然语言问答基准，要求模型输出连续概率估计；与CLADDER/BLInD的本质区别在于摆脱了二元变量与玩具网络限制，并首次显式区分因果、证据与解释消除三种贝叶斯推理路径。
- 首次明确微调SOTA LLM（Mistral-7B）执行从自然语言到概率一阶逻辑编程语言ProbLog的语义解析（ProbLog-FT），而非依赖手工设计Prompt；与仅需Prompt生成ProbLog代码的既有工作相比，监督微调大幅降低了语法错误与谓词未定义问题。
- 系统验证了神经符号方法在不同推理类型上的鲁棒性：纯LLM提示方法在证据推理与解释消除任务上全面失败，而微调后的ProbLog-FT模型能稳定完成全部三种推理，证明结构化因果模型与LLM结合是提升复杂逻辑推理的有效路径。
- 构建了半自动高语言多样性数据生成流水线：利用Mixtral-8x7B-Instruct将结构化CPT与ProbLog表示转化为贴近人类自然语言的表述，并通过Grammarly客观打分与人工抽样校验确保概率语义保真度。

## 方法详解
- **数据集构建流程**：从bnlearn与BNMA仓库选取14个真实贝叶斯网络，切割为30个自包含子网络（对根节点进行边缘化处理）。以Mixtral-8x7B-Instruct为LLM backbone，遍历每个CPT条目生成模板化前提`If[Conditions], then [Probabilities]`，再经few-shot提示重表达为自然语言；数值版保留百分比，WEPs版依据Fagen-Ulmschneider (2015) 实证调查映射至对应模糊量词（如70%→likely/probably，10%→improbable），并在10%情况下替换为次近邻词以模拟主观性。
- **ProbLog符号表示**：手动定义所有节点与类别的谓词，通过规则将CPT转换为带注释析取（annotated disjunctions）的ProbLog代码，例如`0.02::risk_aversion(car_owner, psychopathic); 0.58::risk_aversion(car_owner, adventurous); ...`。每个实例由概率事实（前提）、证据语句`evidence(X=v, true)`与查询语句`query(X=v)`组成，由外部ProbLog求解器计算精确后验概率作为Ground Truth。
- **建模方法**：
  - *LLM提示基线*：零样本与CAUSALCOT提示（要求模型逐步构建图、识别问题类型并计算），输入GPT-4-Turbo、Llama-3-8B、Mixtral-8x7B。
  - *神经符号ProbLog-FT*：使用长上下文Mistral-7B，通过LoRA微调使其将自然语言前提与QE对解析为合法ProbLog程序，再交由求解器执行精确推理；解析分两阶段，先确定谓词词汇再解析QE，以降低Prompt-based LLM的未定义谓词错误。
  - *纯生成/回归基线*：50%固定回答基线、QE-only基线（GPT-4仅回答问题与证据）、Regression-FT（Llama2-7B接sigmoid回归层）、LLM-FT（Mistral-7B直接以文本输出概率值）。
- **评估指标**：正确率、错误率、无效回答率、RMSE_50%（无效输出强制回退至50%计算整体误差）与RMSE_nonError（仅统计有效数值预测的偏差）。

## 实验与结果
- **数据集规模与分布**：训练集930条数值/WEP前提+812条证据+347条查询；测试集273条前提+578条证据+230条查询（因果92、证据62、解释消除26实例）。平均每个随机变量含3.3个状态，每个网络含5.9个随机变量；语法质量评分显著优于CLADDER与BLInD
