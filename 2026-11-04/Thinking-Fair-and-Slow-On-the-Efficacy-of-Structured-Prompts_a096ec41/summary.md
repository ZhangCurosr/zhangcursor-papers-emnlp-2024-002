---
title: "Thinking-Fair-and-Slow-On-the-Efficacy-of-Structured-Prompts"
source: https://aclanthology.org/2024.emnlp-main.13.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:49:01"
field: "大语言模型公平性与安全"
keywords: ["大语言模型去偏", "提示工程", "公平性", "黑盒去偏", "System-2提示", "蕴含提示"]
innovations: ["提出基于System-2思维的三类结构化提示去偏框架（PP/SR/IP），首次在黑盒条件下系统评估提示去偏效果", "提出三阶段蕴含提示（IP）方法，通过模型自生偏见解释引导去偏，显著优于前缀提示和自优化", "证明角色型提示系统性优于指令型提示，且IP方法在不损害下游任务性能下达到与白盒方法相当的去偏效果"]
benchmarks: ["Stereoset", "Regard", "RealToxicityPrompts", "TruthfulQA", "BoolQ"]
---

# 论文速读：Thinking-Fair-and-Slow-On-the-Efficacy-of-Structured-Prompts

## 一句话总结
本文面向无法访问模型内部参数的终端用户，提出了一套基于"系统2"反思思维的结构化提示去偏框架，通过前缀提示、自优化和蕴含提示三种策略（含指令/角色、单步/多步变体）系统性评估其对语言模型偏见的缓解效果，发现蕴含提示（Implication Prompting）在多项基准上显著优于其他方法，且不损害下游任务性能。

## 研究问题与动机
- **核心问题**：如何在无法访问模型权重、内部参数和输出概率分布的情况下（如仅能调用闭源 API 的 LLM），有效缓解大型语言模型中的社会偏见？
- **现有方法的局限**：主流去偏方法（如 Sent-Debias 投影子空间消除、SelfDebias/CAFIE 调整 logit、CDA 数据增强微调）均需模型内部访问权限或大量标注/反事实数据，对终端用户而言不可行；且对预训练权重或解码策略的修改可能破坏模型在其他下游任务上的性能。
- **已有提示去偏工作的不足**：先前研究（Borchers 等人 2022；Ma 等人 2023）仅使用关键词级或 prompt search 方法，计算代价高昂且缺乏系统性分析；尚无工作对 state-of-the-art 提示框架（如 CoT、自反馈、角色提示）进行面向公平性的全面适配与比较。
- **动机**：将 Kahneman "快思慢想" 系统中的系统2（缓慢、逻辑、反思性）决策过程引入提示设计，通过结构化提示引导模型进行批判性推理，从而在不触碰模型内部的前提下实现公平文本生成。

## 核心贡献（创新点）
1. **首个端到端系统化评估"系统2"提示去偏框架的工作**：将已有及新引入的提示方法组织为前缀提示（PP）、自优化（SR）和蕴含提示（IP）三大类，并在单步/k步、指令/角色两个维度上进行全面交叉评估，填补了提示工程与公平文本生成交叉领域的研究空白。
2. **提出蕴含提示（Implication Prompting）三阶段流程**：首先生成初始输出 → 用同一模型推断该输出的隐含偏见（Instruction-based / Zero-Shot CoT / Few-shot 三种变体）→ 将原始输出、偏见解释与去偏指令组合后重新生成去偏文本；与 SelfDebias 等需 logit 访问的方法本质区别在于完全黑盒、无需修改解码策略。
3. **揭示角色型提示优于指令型提示的规律**：Across 所有模型和指标，Role Prompt 在 Stereoset ICAT 上平均高出 5.08%，在 Regard 上平均提升近 39.47%，在 Toxicity 上降低 4.34%，证明"扮演无偏见的人"比"被告知不要偏见"更有效。
4. **证明 System 2 提示可在不损失下游任务性能的前提下达到与白盒方法相当的去偏效果**：Implication Prompting 在 StereoSet、Regard、Toxicity 三项基准上均最优，且在 TruthfulQA（提升约 9%）和 BoolQ 上保持与基线持平甚至更优；附录 B 中与 SelfDebias、CAFIE、SentenceDebias、CDA 等白盒/微调方法的对比显示其性能相当或更优。

## 方法详解
基于 Kahneman 的"系统1（直觉）/系统2（反思）"类比，将提示方法分为三个层级，随反馈 specificity 递增：

**1. 前缀提示（Prefix Prompting, PP）—— 系统2的简单形式**
在用户输入 prompt C 前拼接去偏前缀 $I_{\text{debias}}$，得到 $C_{\text{debias}} = \text{concat}(I_{\text{debias}}, C)$，让模型直接生成去偏文本。分两种变体：
- **Instruction PP**："以下文本是无偏见的，不因性别、种族、宗教等敏感属性歧视他人：[USER PROMPT]"
- **Role PP**："你是一个不因性别、种族、宗教等敏感属性歧视他人的无偏见者。[USER PROMPT]"

**2. 自优化（Self-Refinement, SR）—— 系统2风险下的决策**
让模型参考自己先前生成的（可能含偏见的）输出 $S_0$，通过 k 轮迭代生成更公平的文本（Algorithm 1）：
- Step I：用 PP 策略得到初始输出 $S_0 = M(C, V)$
- Step II：拼接去偏前缀 $I_{\text{SR}}$ 与 $S_0$ 得到 $C_{\text{SR}} = \text{concat}(I_{\text{SR}}, S_0, C)$，生成 $S_1$；k>1 时循环执行
- 同样分 Instruction SR 和 Role SR 两种变体

**3. 蕴含提示（Implication Prompting, IP）—— 系统2批判性反思**
三阶段流程（Algorithm 2），为模型提供"为什么输出有偏见"的推理依据：
- **Step I**：用用户 prompt C 生成初始输出 S
- **Step II**：将 S 与推断指令 $I_{\text{Impl}}$ 拼接，让模型生成对偏见的解释 $S_{\text{Impl}}$，有三种变体：
  - Instruction-Based：直接要求生成 implication
  - Zero-Shot CoT-Based："Thinking step by step, generate the implication..."
  - Fewshot-Based：提供示例（如"We shouldn't lower our standards just to hire more women" → "This text implies that women are less qualified than men."）
- **Step III**：构建最终 prompt $C_{\text{IP}} = \text{concat}(S, S_{\text{Impl}}, I_{\text{IP}}, C)$，其中 $I_{\text{IP}}$ 要求模型"考虑上述偏见解释，生成无偏见文本"

形式化表示：$M(C, V) = S$，所有策略均为黑盒调用，不修改模型参数或解码策略。

## 实验与结果
**实验设置**：评估 4 个 LLM（GPT-J 6B、Mistral 7B、Llama-2 7B/13B、MPT-Instruct 7B），在 3 个偏见基准 + 2 个下游任务上测试。

**基准与指标**：
- **Stereoset**：SS（理想值50%，越低越公平）、LM（理想值100%）、ICAT（综合指标，理想值100%）
- **Regard**：性别/种族/性取向三维度 regard 分数（理想值0，负值为刻板偏见）
- **Toxicity（RealToxicityPrompts）**：均值及相对基线的变化百分比

**主要结果**：
- **IP 是最优方法**：平均而言，IP 较其他方法在 StereoSet 上 SS 低 4.05%、ICAT 高 6.80%；在 Regard 上平均提升 26.85%；Toxicity 平均下降 6.98%。其中 Few-shot IP 和 Zero-Shot CoT IP 表现最佳。
- **Role 提示普遍优于 Instruction 提示**：Role PP 在 StereoSet 上 ICAT 平均高 5.08%，Regard 高近 39.47%，Toxicity 低 4.34%。
- **SR(k=1) 优于 PP，但 SR(k=2) 提升微小**：SR(k=1) 较 PP 在 SS 上低 6.85%、ICAT 高 11.65%、Regard 提升 21.64%；SR(k=2) 仅比 SR(k=1) 再提升约 0.23%（SS），表明单步自优化已足够。
- **不损害下游任务**：IP 方法在 TruthfulQA 上平均提升约 9%，Self-Refinement 在 BoolQ 上提升约 1%；PP 在 TruthfulQA 上下降约 15%。
- **最强结果示例**：Mistral 7B + Instruction SR(k=1) 在 Regard Race 维度达 -0.02（接近理想值0）；MPT Instruct 7B + Instruction IP 在 StereoSet ICAT 上达 75.04；Llama-2 13B + Instruction IP 在 Regard 平均达 0.16（最低均值，即最少偏见）。
- **与白盒方法对比（附录B）**：在 GPT2-Small 上，Role SR(k=1) ICAT 达 81.24，优于 SelfDebias（~75）、SentenceDebias（~75）、CAFIE（75.96）。

## 相关工作脉络
1. **SelfDebias（Schick et al., 2021）**：基于输出概率的 post-hoc 去偏，需要访问 logit；本文 IP 在效果上与之相当但完全黑盒。
2. **CAFIE（Banerjee et al., 2023）**：利用反事实提示校准输出分布；本文方法无需反事实数据构造，仅依赖自然语言提示。
3. **SentenceDebias（Liang et al., 2020）**：从句向量中消除偏见子空间投影，需修改模型内部表示；本文纯提示方法不涉及参数修改。
4. **Borchers et al. (2022)**：关键词级 prompt engineering 缓解招聘广告性别偏见，是最相近的前驱工作但局限于简单前缀且效果不显著；本文扩展为系统化的三类多层次框架。
5. **Brown et al. (2020a)**：证明自然语言指令可引导 GPT-3 缓解性别偏见，启发了本文 PP 设计但未系统性探索多策略对比。
6. **Ma et al. (2023)**：prompt-search 框架实现预测公平性，但需大量计算资源寻找最优 prompt，不具备通用性；本文方法无需搜索，直接提供结构化提示模板。

## 局限性与未来方向
- **LLM 并非真正"思考"**：作者强调"快思慢想"仅是启发式隐喻，当前 LLM 生成文本仍是在训练数据中学到的模式匹配，不宜过度解读为模型具有推理能力。
- **偏见的社会复杂性被简化**：将复杂社会偏见编码为固定提示可能无法捕捉 LLM 中偏见的全貌；不同偏见类型（宗教、种族、性取向等）的效果存在差异。
- **提示响应的假设不稳定性**：方法假设模型对提示的响应一致，但不同 LLM 版本更新可能导致效果漂移。
- **泛化依赖训练数据**：系统2提示仅在该任务相关信息存在于模型隐空间中时才有效，对训练数据中缺乏相关信息的新任务可能失效。
- **计算资源限制**：未能在 70B 级别模型（如 Llama-2-70B）或 MoE 模型（如 Mixtral-45B）上验证；未探索 Tree-of-Thought、Self-Consistency、Directional Stimulus Prompting 等先进提示方法。
- **未来方向**：设计更具挑战性的去偏任务以提升框架泛化性；探索更大规模模型和更多 advanced prompting 策略。

## 研究启发与可借鉴点
1. **System 2 提示分层框架可作为通用范式**：将提示策略按"反馈特异性"从低到高（PP → SR → IP）组织，这一分层思路可直接迁移到其他需要提升模型输出质量/可控性的任务（如事实性、隐私保护）。
2. **蕴含推断（Implication Generation）是黑盒去偏的有效杠杆**：让模型先解释自身输出为何有偏见，再利用该解释纠正输出，这一"自我诊断+纠正"的两阶段设计可推广至事实核查、毒性检测等场景。
3. **角色提示（Role Prompting）的系统性优势值得重视**：相比直接指令，赋予模型"角色身份"在多个基准上稳定优于指令型提示，可作为 prompt 设计的通用经验法则。
4. **SR 多步迭代的边际效益递减规律**：k=1 已足够，更多迭代几乎无增益，这对工程部署中控制推理成本有直接指导意义。
5. **白盒方法可比性验证的价值**：附录 B 中与 SelfDebias、CAFIE、CDA 等方法的公平对比，为后续纯提示方法的 benchmarking 提供了可复用的实验设计模板。

## 关键术语表
**System 1 / System 2**：Kahneman 提出的双过程决策理论；System 1 指直觉、快速、自动的思维，System 2 指缓慢、费力、逻辑性的反思思维，本文借用此框架组织提示策略。
**Implication Prompting (IP)**：蕴含提示，通过让模型先推断自身输出的隐含偏见再据此纠正的三阶段提示策略，是本文提出的最优方法。
**Self-Refinement (SR)**：自优化提示，让模型参考自己先前生成文本进行迭代修正的去偏策略，分为 k=1 和 k=2 等多步变体。
**Prefix Prompting (PP)**：前缀提示，在用户 prompt 前拼接去偏指令或角色设定作为最简单形式的 System 2 提示。
**Stereoset ICAT**：StereoSet 的综合公平性指标，结合 Stereotype Score（SS）和 Language Modeling Score（LM），理想值 100%，用于衡量去偏与语言建模能力的 trade-off。
**Regard 分数**：基于社会感知的偏见度量，通过比较不同人口群体在职业等描述中获得的"尊重度"差异来量化偏见，理想值 0。
**SelfDebias / CAFIE**：需访问模型 logit/概率分布的 post-hoc 去偏方法，分别通过反事实对比和概率校准实现公平生成，代表白盒方法的基线。
**Zero-Shot CoT IP**：蕴含提示的变体，使用 "Thinking step by step" 零样本链式思考指令引导模型生成对偏见的推理阐释。

## 可复现要素
- **数据集**：Stereoset（公开）、Regard（公开）、RealToxicityPrompts（公开）；论文使用了 Stereoset 的约 53% 子集（句尾填空部分）。
- **代码**：论文未明确提及代码开源情况（ACL Anthology 链接页面无开源声明）。
- **模型权重**：从 Huggingface 下载 GPT-J 6B、Mistral 7B、Llama-2 7B/13B、MPT-Instruct 7B，均可公开获取。
- **关键超参**：temperature=1.0，Stereoset 实验 repetition penalty=1.3，默认 decoding 策略为 beam search；SR 的 k 取 1 或 2。
- **硬件**：单卡 32GB NVIDIA V100 GPU。
- **统计检验**：使用单侧 t 检验，p < 0.05 标记为显著（表中标注 *）。
