---
title: "RoseLoRA-Row-and-Column-wise-Sparse-Low-rank-Adaptation-of-P"
source: https://aclanthology.org/2024.emnlp-main.57.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:28:38"
field: "大语言模型参数高效微调"
keywords: ["参数高效微调", "低秩适应", "知识编辑", "稀疏更新", "LLM微调"]
innovations: ["提出行/列方向稀疏低秩适应方法RoseLoRA，实现选择性参数更新", "将乘积稀疏约束转化为行/列稀疏约束并给出理论下界", "基于灵敏度感知的迭代剪枝策略配合立方退火保障训练稳定"]
benchmarks: ["WikiData_recent", "WikiData_counterfact", "ZsRE", "WikiBio", "GLUE", "Commonsense170K", "Math10K", "Alpaca-Eval v1.0"]
---

# 论文速读：RoseLoRA-Row-and-Column-wise-Sparse-Low-rank-Adaptation-of-P

## 一句话总结
本文提出了一种新颖的参数高效微调方法 RoseLoRA，通过对低秩矩阵 B 和 A 分别施加行/列方向稀疏约束，实现对预训练语言模型关键参数的选择性更新；该方法在知识编辑任务上显著优于现有基线，同时在通用微调任务（常识推理、算术推理、指令遵循、NLU）上也能匹配或超越 LoRA 等 SOTA PEFT 方法，且参数量大幅减少。

## 研究问题与动机
- **LoRA 的粗粒度更新问题**：LoRA 家族通过在预训练权重上叠加低秩矩阵乘积 $BA$ 进行微调，但乘积结果是稠密矩阵，推理时会更新所有预训练参数，无法实现选择性编辑。
- **知识编辑任务的需求**：知识编辑要求模型吸收新知识同时保留旧知识，需要精细的参数子集更新，而 LoRA 因全局更新导致"记忆残留"和知识干扰严重。
- **直接 $\ell_0$ 约束不可行**：在 $BA$ 乘积上直接施加 $\ell_0$ 稀疏约束是 NP-hard 问题，且单纯对 A、B 各自做稀疏控制并不能保证乘积稀疏（Example 1 展示了反例：两矩阵行/列稀疏但乘积为稠密矩阵）。
- **计算效率与参数经济的平衡**：如何在极少参数更新下实现高质量任务适配，同时缓解微调带来的知识遗忘问题。

## 核心贡献（创新点）
1. **提出 RoseLoRA 稀疏低秩适应框架**：将 LoRA 的低秩结构与行/列方向稀疏约束相结合，使模型仅更新与任务最相关的关键参数；与 AdaLoRA、SoRA 等仅控制秩但更新仍为稠密的剪枝方法本质不同，RoseLoRA 实现了真正的稀疏更新。
2. **设计了可理论保证的稀疏转换机制**：将难以优化的 $\ell_0$ 乘积稀疏约束转化为对 A 的行和 B 的列分别施加稀疏约束，并给出了乘积矩阵稀疏性的理论下界（Proposition 1），确保控制行/列稀疏即可间接控制最终更新的稀疏程度。
3. **基于灵敏度感知的迭代剪枝优化策略**：利用灵敏度重要性得分 $I(W_{ij}) = |W_{ij} \cdot \nabla_{W_{ij}} \mathcal{L}|$ 配合指数移动平均平滑，按迭代预算 $\tau^{(t)}$ 逐步裁剪 A 的行和 B 的列；配合立方退火策略保障训练稳定性。
4. **系统性的多任务基准验证**：在五个基准（知识编辑、常识推理、算术推理、指令遵循、NLU）共二十余个数据集上评估，证明 RoseLoRA 在知识编辑和通用微调上均优于现有 PEFT 方法。

## 方法详解
- **基础形式**：基于 LoRA 的权重更新 $W = W^o + BA$，其中 $A \in \mathbb{R}^{r \times d_2}$、$B \in \mathbb{R}^{d_1 \times r}$、$r \ll \min\{d_1, d_2\}$。
- **原始优化目标**：$\min_{A,B} \mathcal{L}(\mathcal{D}; W^o + BA)$ s.t. $\frac{\|BA\|_0}{d_1 d_2} \leq \tau$（难以直接优化）。
- **稀疏转换**：等价转换为对每行 $A_{i*}$ 和每列 $B_{*i}$ 施加稀疏约束：$\frac{\|A_{i*}\|_0}{d_2} \leq \tau, \frac{\|B_{*i}\|_0}{d_1} \leq \tau, \forall i=1,...,r$。
- **理论保证**（Proposition 1）：$s(BA) \geq \max\left\{0, 1 + \sum_{i=1}^{r}\left(s(A_{i*}) + s(B_{*i}) - s(A_{i*})s(B_{*i})\right) - r\right\}$，即通过控制行/列稀疏性可保证乘积矩阵的稀疏性下界。
- **迭代剪枝过程**：每轮先做梯度更新 $\tilde{A}^{(t)} = A^{(t)} - \nabla_A \mathcal{L}, \tilde{B}^{(t)} = B^{(t)} - \nabla_B \mathcal{L}$，再按灵敏度重要性得分保留每行/列 top-$\tau^{(t)}$ 参数，其余置零；$\tau^{(t)}$ 按立方策略逐渐从 1 衰减至目标稀疏度 $\tau$。
- **知识编辑附加约束**：额外引入 Frobenius 范数约束 $\|A\|_F^2 \leq \alpha, \|B\|_F^2 \leq \alpha$，防止参数更新幅度过大导致知识遗忘。

## 实验与结果
- **数据集**：知识编辑（WikiData_recent、WikiData_counterfact、ZsRE、WikiBio）、常识推理（BoolQ、PIQA、SIQA、HellaSwag、Wino-Grande、ARC-e、ARC-c、OBQA）、算术推理（AQuA、GSM8K、MAWPS、SVAMP）、指令遵循（Ultrafeedback/Alpaca-Eval v1.0）、NLU（GLUE 8个数据集）。
- **模型**：LLaMA-7B / LLaMA-2 7B / RoBERTa-large。
- **知识编辑**：RoseLoRA 在 WikiData_recent 编辑成功率达 98.4%（ROME 85.1%、MEMIT 85.3%），平均指标 AVG 达 73.7（最优）；WikiData_counterfact AVG 76.7、ZsRE AVG 75.2、WikiBio AVG 84.6 均大幅领先。
- **常识推理**：RoseLoRA（0.03% 参数）AVG 80.7%，超越 LoReFT（80.2%）及其他所有基线，且在 BoolQ、PIQA、ARC-e、ARC-c、OBQA 五个数据集上排名第一。
- **算术推理**：RoseLoRA（0.03% 参数）AVG 45.9%，以 1/22 的 LoRA 参数量达到 LoRA（AVG 46.9%）的 97% 水平，超越 LoReFT 约 6.3%。
- **指令遵循**：RoseLoRA（0.0037% 参数）在 Alpaca-Eval v1.0 获得 85.77% win-rate，略超 LoReFT（85.60%）。
- **NLU（GLUE）**：RoseLoRA（0.015% 参数）AVG 89.0%，超过 LoRA（88.1%）和全参数微调（88.6%）；RTE 数据集提升 3.4%。
- **知识遗忘测试**：微调后 RoseLoRA 在 TriviaQA、MMLU 上保留的知识显著多于 LoRA（如 Commonsense170K 后 TriviaQA 47.8 vs 9.0）。
- **小数据优势**：当训练数据降至 12.5% 时，RoseLoRA 在 GSM8K 上已超越 LoRA，表明其在小样本场景更具优势。

## 相关工作脉络
- **LoRA 及变体（Hu et al., 2021; Zhang et al., 2023; Dettmers et al., 2024）**：LoRA 是 PEFT 主流方法，AdaLoRA/SoRA 对其做剪枝以降低计算开销，但这些方法仍产生稠密更新；RoseLoRA 的核心区别在于通过行/列稀疏约束实现了真正的稀疏参数子集更新。
- **Adapter / Prefix Tuning（Houlsby et al., 2019; Li & Liang, 2021）**：通过在 Transformer 中插入瓶颈模块或前缀软 token 实现 PEFT；与 RoseLoRA 的思路不同，后者直接修改 weight matrix 的局部稀疏子集。
- **ReFT（Wu et al., 2024）**：学习表示空间上的低秩更新而非权重空间；RoseLoRA 在权重空间操作，且避免了对线性表示假设的依赖。
- **知识编辑方法：ROME / MEMIT（Meng et al., 2022a,b）**：定位-编辑范式，需复杂因果追踪来定位知识存储层；RoseLoRA 无需定位，通过稀疏自适应全局微调完成编辑。
- **AdaLoRA（Zhang et al., 2023）**：自适应分配预算到低秩矩阵的奇异值；控制的是秩而非稀疏性，更新仍为稠密，无法支持精确知识编辑。
- **敏感度剪枝（Sanh et al., 2020; Li et al., 2023c; Sun et al., 2023）**：已有模型剪枝工作，但多为针对全量权重的稀疏化；RoseLoRA 的创新在于将稀疏约束巧妙嫁接到 LoRA 的低秩参数化结构上，并通过理论证明了乘积稀疏性下界。

## 局限性与未来方向
- **超参数 $\beta$ 调优**：平滑灵敏度估计的 $\beta$ 虽对性能不敏感，但仍需额外调节；论文采用固定值 0.8。
- **立方退火策略的超参 $t_i, t_f$**：稀疏预算衰减策略需要设定这两个阶段划分超参。
- **未充分探索更复杂的稀疏模式**：当前仅约束每行/列整体稀疏，未考虑更细粒度的结构化稀疏（如块稀疏）。
- **知识编辑中的 $\ell_2$ 范数约束**：额外引入的 $\alpha$ 超参可能影响编辑精度与泛化性，需进一步研究最佳配置。

## 研究启发与可借鉴点
- **稀疏约束的"转化"思路**：将难以优化的 $\ell_0$ 乘积稀疏直接转化为行/列稀疏这一技术路线，可迁移到其他低秩参数化场景（如 QLoRA、DoRA）中，实现可控稀疏更新。
- **理论下界的分析方法**：Proposition 1 的证明手法（利用 Lemma 1 向量稀疏性公式 $s(ba) = s_a + s_b - s_a s_b$）为分析矩阵乘积稀疏性提供了通用工具，可推广至更深层次的乘积分析。
- **小数据场景下稀疏方法的优越性**：实验表明数据稀缺时 RoseLoRA 优于 LoRA，这一现象值得深入探究——稀疏更新本质上具有更强的正则化效果，可作为小样本 PEFT 的通用设计原则。
- **知识编辑与 PEFT 的结合**：将稀疏 PEFT 应用于知识编辑是一个较新的交叉方向，未来可探索稀疏 LoRA 在多跳事实编辑、持续学习等场景的应用。

## 关键术语表
**RoseLoRA**：Row and Column-wise Sparse Low-rank Adaptation，一种通过行/列方向稀疏约束实现选择性参数更新的 PEFT 方法。
**PEFT（Parameter-Efficient Fine-Tuning）**：参数高效微调，通过仅更新少量参数来适配大语言模型，避免全量微调的资源消耗。
**Sensitivity-based Importance Score**：基于灵敏度/importance 的剪枝重要性度量，定义为 $|W_{ij} \cdot \nabla_{W_{ij}} \mathcal{L}|$，衡量参数被置零后对损失函数的近似影响。
**Knowledge Editing**：知识编辑，旨在精确修改预训练语言模型中的特定事实性知识，同时保留其他知识的任务。
**Edit Success / Locality / Portability**：知识编辑三大约评估指标，分别衡量编辑成功率、局部知识保持性和更新知识的可迁移推理能力。
**Cubic Decay Strategy**：立方退火策略，使稀疏预算 $\tau^{(t)}$ 在训练后期以立方速率逐渐衰减至目标稀疏度，以保障训练稳定性。
**$\ell_0$ 稀疏约束**：直接限制矩阵中非零元素数量（比例）的约束，是衡量稀疏性的最自然指标，但优化困难（NP-hard）。
**FFN Layers**：Feed-Forward Network 层，已有研究发现事实性知识多存储于这些层，部分知识编辑方法仅在此层施加更新。

## 可复现要素
- **代码**：论文未明确声明代码开源，但提到使用 LLM-Adapters、EasyEdit、lm-evaluation-harness 等开源库。
- **数据集**：WikiData_recent、WikiData_counterfact、ZsRE、WikiBio、Commonsense170K、Math10K、Ultrafeedback、GLUE 均为公开基准。
- **关键超参**：
  - 知识编辑：lr=2e-4，rank=4，batch_size=1，sparsity=0.95，$\beta$=0.8，$\alpha$=3e-3，target modules="up_proj", "down_proj", "gate_proj"
  - 通用微调：lr=2e-4~3e-4，rank=32，batch_size=8~32，sparsity=0.85~0.865，target modules="q_proj", "v_proj"
  - GLUE：lr=1e-4~6e-4，rank=6，sparsity=0.95，$\beta$=0.8，target modules 因数据集而异
- **基座模型**：LLaMA-7B-chat / LLaMA-2 7B / RoBERTa-large
