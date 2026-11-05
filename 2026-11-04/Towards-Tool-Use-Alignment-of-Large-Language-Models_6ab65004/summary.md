---
title: "Towards-Tool-Use-Alignment-of-Large-Language-Models"
source: https://aclanthology.org/2024.emnlp-main.82.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:49:33"
field: "大语言模型工具使用对齐"
keywords: ["Tool Alignment", "Harmlessness", "Autonomy", "Preference Optimization", "DPO", "Tool Learning", "LLM Safety"]
innovations: ["提出H2A对齐原则，首次将工具使用场景下的LLM对齐目标定义为帮助性、无害性和自主性三维度", "构建ToolAlign数据集，包含46k指令微调和10k偏好数据，覆盖三维度", "验证SFT+DPO分阶段训练范式，SFT建立基础安全/自主能力，DPO进一步提升帮助性"]
benchmarks: ["ToolAlign", "ToolBench", "ToolSword", "MetaTool"]
---

# 论文速读：Towards-Tool-Use-Alignment-of-Large-Language-Models

## 一句话总结
本文提出 H2A 原则（Helpfulness, Harmlessness, Autonomy），构建了包含指令微调数据与偏好数据的 ToolAlign 数据集，通过 SFT 和 DPO 对开源 LLM 进行工具使用对齐训练，使模型在工具调用能力、拒绝有害内容与自主回答三个方面达到平衡。

## 研究问题与动机
1. **现有工具学习工作主要聚焦于提升 LLM 的工具调用能力（Helpfulness），忽视了对齐价值观的需求**——如 ChatGPT 等开源模型在面对恶意指令或恶意工具响应时缺乏拒绝能力。
2. **工具使用场景中存在安全隐患**——外部工具可能被攻击或拦截，返回不安全响应（如钓鱼链接、恶意脚本），LLM 需要识别并拒绝这些有害内容。
3. **工具调用带来额外成本**——对于可直接回答的问题，LLM 应自主回答而非调用工具，以节省时间和费用。
4. **缺少同时涵盖三维度对齐的综合数据集**——现有基准（如 ToolSword、MetaTool）仅评估安全性或自主性，未同时覆盖 Helpfulness、Harmlessness 和 Autonomy。

## 核心贡献（创新点）
1. **提出 H2A 对齐原则**——首次将工具使用场景下的 LLM 对齐目标明确定义为帮助性（Helpfulness）、无害性（Harmlessness）和自主性（Autonomy）三维度，为后续研究提供理论框架。
2. **构建 ToolAlign 数据集**——包含 46k 指令微调数据和 10k 偏好数据，覆盖三个维度，弥补了现有数据集在多维对齐方面的空白。
3. **设计兼顾三者的数据构造流程**——有害指令通过改写 ToolBench 数据和使用 ARTD 生成，工具响应注入四类恶意内容；自主性数据通过改写 Alpaca 并调用 ToolBench 检索器获取工具上下文。
4. **验证 SFT + DPO 的协同训练范式**——SFT 赋予模型基础的安全性和自主性，DPO 进一步提升帮助性，消融实验证实二者缺一不可。

## 方法详解
1. **H2A 原则定义**：
   - **Helpfulness**：正确理解用户指令，调用外部工具提供信息丰富的回答。
   - **Harmlessness**：识别并拒绝有害指令和工具返回的恶意响应，生成带有解释的拒绝回复。
   - **Autonomy**：对可直接回答的问题不调用工具，直接给出响应。

2. **指令微调数据集构造**（共 46k 条）：
   - **Helpfulness**：从 ToolBench 采样 40,000 条指令-响应对。
   - **Harmlessness**：两类数据（2,841 条）——①有害指令：通过改写 ToolBench 添加隐私/非法内容，以及从 Anthropic Red Teaming Dataset 采样并用 ChatGPT 改写，再由 ChatGPT 生成三段式拒绝响应（声明有害→解释风险→建议安全请求）；②有害工具响应：注入四类恶意内容（明显有害、钓鱼网站、恶意脚本、敏感信息索取），使用模板填充工具名和有害类型。
   - **Autonomy**：从 Alpaca 采样 3,881 条，用 ChatGPT 改写以匹配 ToolBench 格式，并通过 ToolBench 检索器获取相关 API，再由 ChatGPT 生成直接回答。

3. **偏好数据集构造**（共 10k 条）：
   - **Helpfulness**：每指令从 ChatGPT 和 ToolLLaMA/AlignToolLLaMA-SFT 各采样一个响应，用 ChatGPT 判断完成质量，优先选 ChatGPT 响应作为 chosen。
   - **Harmlessness**：有害指令的 chosen 为 ChatGPT 拒绝响应，rejected 为 ToolLLaMA 尝试响应的版本；有害工具响应的手工构造拒绝模板作为 chosen。
   - **Autonomy**：对比 ChatGPT 和 ToolLLaMA 的响应，若 ToolLLaMA 调用工具而 ChatGPT 直接回答则选 ChatGPT；若两者均直接回答则由 GPT-4 评分决定 chosen。

4. **模型训练流程**：
   - **SFT**：在 ToolAlign 指令微调数据集上训练 ToolLLaMA 2 个 epoch，全局 batch size 64，峰值学习率 5e-5，4% warmup。
   - **DPO**：在偏好数据集上训练 AlignToolLLaMA-SFT 1 个 epoch，学习率 1e-6，warmup 5 步，batch size 8，β=0.05。
   - 全部在 4×A100 40GB GPU 上使用 bfloat16 精度训练。

## 实验与结果
1. **评估指标**：
   - Helpfulness：Pass Rate (PR)、Win Rate (WR)
   - Harmlessness：Refusal Response Rate (3R)
   - Autonomy：Direct Response Rate (DR2)

2. **主要结果**：
   - **AlignToolLLaMA-DPO** 在 Helpfulness 测试集上平均 PR 达 **49.8%**，显著优于 AlignToolLLaMA-SFT 的 27.3%。
   - **Harmlessness**：AlignToolLLaMA-SFT/DPO 在 HI 测试集上 3R 达 **96.4%/97.4%**，HTR 测试集上达 **100%**，而 ToolLLaMA 均为 0%。
   - **Autonomy**：AlignToolLLaMA-SFT/DPO 在 AU 测试集上 DR2 均为 **100%**，而 ToolLLaMA 仅 22%。
   - 最强模型 AlignToolLLaMA-DPO 在三个维度上均达到优异水平，Helpfulness 接近 GPT-4（57.2%），同时安全性和自主性显著超越 GPT-4（GPT-4 在 HTR 上 76.5%，AU 上仅 11.0%）。

3. **泛化能力**：在 ToolSword 和 MetaTool 上，AlignToolLLaMA-SFT/DPO 在恶意查询拒绝、越狱攻击检测和自主使用工具方面表现优异，部分指标超过 GPT-4。

4. **消融实验**：
   - 跳过 SFT 直接用 DPO 训练 ToolLLaMA（+DPO with Preference Data）在 Harmlessness 上仍为 0%，证明 SFT 是获得基本安全/自主能力的前提。
   - 仅在偏好数据上继续 SFT（+SFT with Preference Data）反而降低帮助性（PR 从 27.3% 降至 12.9%），证明 DPO 对偏好学习的必要性。

5. **人工评估一致性**：GPT-4 评分与人工评分的 Pearson 相关系数在 HI 测试集为 **0.921**，在 AU 测试集为 **0.822**，验证了自动化评估的有效性。

## 相关工作脉络
1. **ToolLLaMA（Qin et al., 2023b）**：本文的基础模型，通过大规模工具调用数据训练提升 Open-source LLM 的工具使用能力，但缺乏对齐考量，本文在其基础上增加 Harmlessness 和 Autonomy 训练。
2. **ToolSword（Ye et al., 2024）**：评估工具学习中安全性问题的基准，覆盖三个阶段的安全威胁，本文与其在 Harmlessness 维度形成对照。
3. **MetaTool（Huang et al., 2023）**：评估 LLM 工具使用决策能力的基准，本文在 Autonomy 维度与其形成对照。
4. **Constitutional AI（Bai et al., 2022b）**：基于原则的自我对齐方法，本文的 H2A 原则与其理念相近，但聚焦于工具使用场景。
5. **Safety-Tuned LLaMAS（Bianchi et al., 2023）**：通过安全指令微调提升 LLM 安全性，本文指出通用安全对齐在工具场景中泛化有限，需专门的数据构建策略。
6. **DPO（Rafailov et al., 2024）**：直接偏好优化方法，本文将其应用于工具使用对齐，验证了偏好学习对提升 Helpfulness 的重要性。

## 局限性与未来方向
1. **单轮交互假设**：当前实验和数据处理均基于单轮对话，未考虑多轮交互中历史对话记录对工具调用决策的影响。
2. **数据规模有限**：偏好数据集仅 10k 条，可能不足以充分捕捉复杂的偏好模式，尤其是极端边缘案例。
3. **模型尺寸限制**：基于 7B 参数的 ToolLLaMA，与 GPT-4 等大模型相比在知识和推理能力上仍有差距，导致某些自主回答质量不及闭源模型。
4. **工具覆盖范围**：基于 ToolBench 的 3,000+ 工具，与实际生产环境中海量异构工具相比仍存在差距。
5. **未来方向**：扩展到多轮对话场景、探索更大规模偏好数据集、结合更多元的安全对齐原则、研究动态工具选择与自主决策的平衡机制。

## 研究启发与可借鉴点
1. **三维度对齐框架具有通用性**——H2A 原则可迁移至其他交互场景（如代码生成、Agent 系统），为多维度对齐提供可复用的设计范式。
2. **数据格式一致性的重要性**——通过将 ARTD 和 Alpaca 的数据改写为与 ToolBench 一致的格式，避免了模型学到数据集分布的捷径，这一策略可用于其他跨数据集融合任务。
3. **SFT + DPO 的分阶段训练策略**——SFT 负责建立基础能力边界，DPO 负责精细化偏好学习，消融实验验证了二者协同的必要性，这一范式值得在其他对齐任务中验证。
4. **三段式拒绝响应设计**——"声明有害→解释风险→建议安全请求"的结构化拒绝模板，既保证安全性又维持帮助性，可作为安全对齐中拒绝响应的参考设计。
5. **自动化偏好标注的效率**——利用 ChatGPT 和 GPT-4 作为评委进行响应质量评估和偏好标注，在缺少人工标注的情况下高效构建了高质量偏好数据集，可为资源受限场景提供参考。

## 关键术语表
**H2A**：Helpfulness, Harmlessness, Autonomy 的缩写，本文提出的工具使用场景下 LLM 对齐三原则。
**ToolAlign**：本文构建的指令微调与偏好数据集，包含 46k 指令数据和 10k 偏好数据，覆盖 H2A 三个维度。
**SFT (Supervised Fine-Tuning)**：监督微调，使用标注数据对预训练模型进行指令调整的训练方式。
**DPO (Direct Preference Optimization)**：直接偏好优化，一种无需显式奖励模型的偏好学习方法，通过对比选择与拒绝样本优化模型。
**Pass Rate (PR)**：衡量模型响应是否完成指令的指标，是工具调用任务的核心评估标准。
**Refusal Response Rate (3R)**：衡量模型拒绝有害指令或有害工具响应的比率，评估 Harmlessness 的关键指标。
**Direct Response Rate (DR2)**：衡量模型在不调用工具的情况下直接回答问题的比率，评估 Autonomy 的关键指标。
**ToolBench**：包含 3,000+ 真实工具和 126k 指令的大规模工具学习数据集，本文的主要数据源。

## 可复现要素
- **数据集**：ToolAlign 已公开，可从 https://github.com/zhiyuanc2001/ToolAlign 获取
- **代码**：开源
- **权重**：论文未提及模型权重公开情况
- **基座模型**：ToolLLaMA（基于 LLaMA-2-7B）
- **关键超参**：SFT 学习率 5e-5，batch size 64，2 epochs；DPO 学习率 1e-6，batch size 8，1 epoch，β=0.05，warmup 5 steps
- **硬件**：4× Nvidia A100 40GB GPU，bfloat16 精度
