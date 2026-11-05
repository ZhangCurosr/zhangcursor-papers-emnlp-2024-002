---
title: "Self-Refine-Instruction-Tuning-for-Aligning-Reasoning-in-Lan"
source: https://aclanthology.org/2024.emnlp-main.139.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:28:44"
field: "语言模型对齐与推理能力提升"
keywords: ["Chain-of-Thought", "Instruction-tuning", "Direct Preference Optimization", "Teacher-Student Alignment", "Small Language Models", "Reasoning Transfer", "DPO"]
innovations: ["提出 Instruction-tuning + DPO 两阶段 Self-refine 框架实现师生推理能力对齐", "学生自生成偏好数据，无需额外训练奖励模型", "系统性对比 in-family 与 out-family 教师对齐并验证低资源鲁棒性"]
benchmarks: ["GSM8K", "MultiArith", "CommonSenseQA", "OpenBookQA", "PIQA", "SIQA", "MATH", "MMLU"]
---

# 论文速读：Self-Refine-Instruction-Tuning-for-Aligning-Reasoning-in-Lan

## 一句话总结
提出了一种两阶段 **Self-refine Instruction-tuning** 方法，通过大语言模型（LLM）生成的 CoT 演示数据对小型语言模型（SLM）进行指令微调（Instruction-tuning），再借助 **Direct Preference Optimization (DPO)** 让学生模型基于自采样推理路径进行自我优化，从而实现师生模型推理能力的对齐，在常识和数学推理任务上均显著超越仅做 Instruction-tuning 的基线。

## 研究问题与动机
1. **现有方法泛化能力不足**：现有 SLM 推理能力对齐工作主要依赖 LLM 生成的 CoT 演示数据进行监督微调（SFT/Instruction-tuning），但仅靠演示数据导致模型泛化性不强，师生之间仍存在性能差距。
2. **单一教师模型的局限**：许多相关工作以 GPT-3.5 等单一教师生成演示，缺乏对不同教师来源及同源/异源教师对齐效果的系统性分析。
3. **偏好优化未与 Instruction-tuning 充分结合**：虽已有使用 RL/DPO 增强推理的方法，但在 Instruction-tuning 基础上引入学生自生成偏好的 self-refine 阶段尚未被系统探索。
4. **低资源场景下的鲁棒性需求**：当演示数据有限时，如何保持并进一步提升 SLM 的推理能力是一个关键挑战。

## 核心贡献（创新点）
1. **提出 Self-refine Instruction-tuning 两阶段框架**：将 Instruction-tuning（LLM 演示数据初始化）与 DPO 自优化相结合，本质区别在于引入了学生模型自采样推理路径的偏好学习，而非仅依赖静态演示数据。
2. **学生自生成偏好，无需额外奖励模型**：与 PPO/RLOO 等方法需训练独立 reward model 不同，本文直接利用 DPO 和学生自身生成交替答案实现偏好优化，简化了训练流程。
3. **系统性对比同源（in-family）与异源（out-family）教师对齐效果**：对比 Llama2-70、Mixtral、GPT-3.5 作为教师时 SLM 的性能差异，发现 out-family 对齐能产生更鲁棒的学生模型。
4. **验证了 low-resource 场景下的有效性与 cross-task generalization**：减少演示数据量及跨任务优化设置下，Self-refine 阶段仍能有效提升模型泛化能力。

## 方法详解
框架分两阶段：

**阶段一：Instruction-tuning（指令微调）**
- LLM 教师以 zero-shot CoT prompt 生成合成演示数据，数据三元组为 $(i, q, a_i)$，其中 $i$ 为指令、$q$ 为问题、$a_i$ 为教师生成的 CoT 推理答案。
- 损失函数：$\mathcal{L}_{\mathrm{inst}}(\theta) = -\mathbb{E}_{(i,q,a_i)\sim D}[\sum_{t=1}^{L}\log \pi_\theta(w_t|s_t, i)]$，模型同时条件于状态 $s_t$ 和指令 $i$，确保指令遵从性。
- 使用 QLoRA 高效微调，4 个 epoch，学习率 $2\times10^{-5}$，weight decay $10^{-4}$，cosine scheduler，warmup ratio 0.03。

**阶段二：Self-refine（DPO 偏好优化）**
- 对每个演示样本，分别以 CoT prompt 和非 CoT prompt 提示学生模型，生成 $y_w$（CoT 答案，标记为偏好/positive）和 $y_l$（非 CoT 答案，标记为不利/negative）。
- 使用 $\mathsf{DPO}_{CoT}$ 损失：
$$\mathcal{L}_{\mathsf{DPO}_{CoT}}(\pi_\theta;\pi_{inst}) = -\mathbb{E}[\log\sigma(M(x, y_w, y_l))]$$
其中 $M(x,y_w,y_l) = \beta\log\frac{\pi_\theta(y_w|x)}{\pi_{inst}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{inst}(y_l|x)}$，$\beta=0.1$。
- DPO 参数：学习率 $10^{-6}$，warmup 100 步，batch size 128，最大 1000 步，保存验证集最低 loss 的 checkpoint。

## 实验与结果
**数据集**：
- 常识推理：CommonSenseQA (CSQA)、OpenBookQA (OBQA)、PIQA、SIQA
- 数学推理：GSM8K、MultiArith
- 额外验证：MATH、MMLU

**模型**：
- 教师（LLMs）：Llama2-70b、Mixtral-8x7B、GPT-3.5
- 学生（SLMs）：Llama2-7b、Llama2-13b、Mistral-7b

**主要结果**（Llama2-7b 学生 + Llama2-70b 教师，in-family）：
| 任务 | Baseline CoT | Instruction-tuning (IT) | Self-refine IT |
|---|---|---|---|
| OBQA | 49.5 | 65.3 | **70.8** |
| CSQA | 55.8 | 71.4 | **79.8** |
| PIQA | 63.8 | 80.2 | **84.6** |
| SIQA | 51.3 | 62.8 | **66.2** |
| GSM8K | 71.3 | 76.9 | **81.9** |
| MultiArith | 72.6 | 85.8 | **89.1** |

**关键结论**：
- Self-refine 后学生模型在 in-domain 和 out-domain 任务上均持续优于仅 IT 的基线。
- Out-family（如 GPT-3.5 教师 → Llama2 学生）对齐效果同样显著，甚至在多数任务上超过 in-family，说明多样化教师来源有助于提升鲁棒性。
- Low-resource 实验（Table 1）显示，减少训练数据量时 Self-refine 依然有效；跨任务 Self-refine（Cross Self-refine）也能获得与域内优化相近的性能。
- Perplexity 分析（Appendix I）表明 IT+DPO 显著降低了 out-domain 上的困惑度。

## 相关工作脉络
1. **CoT Prompting（Wei et al., 2022）**：开创链式思考方法，揭示大模型涌现推理能力；本文关注如何将此能力迁移到小模型。
2. **Teacher-Student Distillation（Magister et al., 2023; Li et al., 2023a; Shridhar et al., 2023）**：通过 SFT 从 LLM 向 SLM 蒸馏推理能力；本文在此基础上增加 DPO 自优化阶段。
3. **Instruction-tuning 对齐（Ranaldi & Freitas, 2024）**：同一团队前期工作，仅用 Instruction-tuning 做师生对齐；本文证明引入 DPO 可进一步消除性能差距。
4. **RL-based Reasoning Alignment（Uesato et al., 2022; Luo et al., 2023; Wang et al., 2023a）**：使用 PPO/过程奖励进行推理优化；本文不使用独立 reward model，改用 DPO 实现偏好优化。
5. **DPO（Rafailov et al., 2023）**：本文引入 DPO 作为 self-refine 的核心算法，将其应用于师生对齐场景，偏好数据由学生自己生成。
6. **Math Instruction-tuning（Yue et al., 2023 - Mammoth; Wang et al., 2023d）**：使用 Alpacalike 风格演示或 IT+RL 做数学推理微调；本文强调多样化教师和跨任务泛化。

## 局限性与未来方向
1. **仅限英语**：所有 prompt 和任务均为英语，跨语言迁移能力待验证。
2. **依赖 LLM 教师**：训练数据质量取决于教师模型输出，且教师模型的预训练数据差异难以完全刻画。
3. **CoT 可能产生虚假自信**：即使最终答案正确，模型生成的多步推理不一定反映真实推理过程，用户可能过度信任 CoT 输出。
4. **安全与 OOD 鲁棒性**：作者指出仍需发展安全机制和分布外泛化能力，才能将方法应用于更关键领域。
5. **未来方向**：扩展到非英语语言，结合课程学习等策略进一步提升跨语言推理迁移能力。

## 研究启发与可借鉴点
1. **Instruction-tuning + DPO 的两阶段范式具有高度可迁移性**：对于任何需要将大模型能力迁移到小模型的任务（如代码生成、摘要、对话），可复制"先 SFT/IT 初始化，再 DPO 自优化"的流程。
2. **Out-family 教师可能优于 In-family**：实验表明来自不同架构的教师（如 GPT-3.5 教 Llama2）反而产生了更鲁棒的模型，这对选择教师模型有指导意义——应优先考虑教师能力多样性而非同源性。
3. **自生成偏好数据可替代人工标注**：利用学生自身生成的 CoT/非 CoT 答案对构建 DPO 偏好对，避免了昂贵的标注成本，这一策略可推广到其他 preference-based 对齐任务。
4. **Cross-task Self-refine 增强泛化**：在任务 A 上 Instruction-tune 后在任务 B 上做 DPO 自优化，仍可获得较好性能，说明 DPO 阶段有较强的跨任务迁移能力，值得在低资源场景下探索。
5. **低资源鲁棒性验证设计**：通过系统性减少训练数据比例（100%/75%/50%/25%）验证方法鲁棒性，是论证方法实用价值的有效实验设计。

## 关键术语表
- **Self-refine Instruction-tuning**：一种两阶段训练方法，先通过 LLM 演示数据进行指令微调，再通过 DPO 让学生模型自我优化推理能力。
- **Chain-of-Thought (CoT)**：将复杂推理任务分解为多个中间步骤的链式推理方式，以提升模型的逐步推理能力。
- **Direct Preference Optimization (DPO)**：一种无需显式训练 reward model 的偏好对齐方法，直接通过对偏好对的概率比优化策略模型。
- **Instruction-tuning**：面向特定任务的监督微调变体，通过在输入中显式加入指令来引导模型生成符合要求的输出。
- **In-family / Out-family 对齐**：分别指教师与学生模型来自同一架构家族（如 Llama2→Llama2）与不同家族（如 GPT-3.5→Llama2）的师生对齐设置。
- **Small Language Model (SLM)**：参数规模较小的语言模型（如 7B 以下），推理成本低但推理能力弱于大模型。
- **QLoRA**：一种高效微调技术，通过量化权重和使用低秩适配器（LoRA）降低大模型微调的显存需求。
- **Cross Self-refine**：在学生模型于任务 A 上完成 Instruction-tuning 后，切换到任务 B 的训练集上进行 DPO 自优化，用于评估跨任务泛化能力。

## 可复现要素
- **数据集**：CSQA、OBQA、PIQA、SIQA、GSM8K、MultiArith、MATH、MMLU 均可从 HuggingFace 公开获取（论文 Table 9 列出了具体仓库地址）。
- **代码**：论文声明代码随 submission 共享（"we shared our code along with the submission"），完整链接见论文附录。
- **模型权重**：所有模型（Llama2-7b/13b/70b-chat、Mistral-7B-Instruct-v0.1、Mixtral-8x7B-v0.1）均来自 HuggingFace 公开版本。
- **关键超参**：
  - Instruction-tuning：4 epochs，lr $2\times10^{-5}$，weight decay $10^{-4}$，cosine scheduler，warmup ratio 0.03，QLoRA
  - DPO：lr $10^{-6}$，$\beta=0.1$，warmup 100 步，batch size 128，max 1000 steps
  - Temperature：教师生成 $0.7$（GPT-3.5）/ $0.1$（Llama2-70）；学生评估 $0.1$
- **硬件**：4× NVIDIA RTX A6000，每张 48GB VRAM
