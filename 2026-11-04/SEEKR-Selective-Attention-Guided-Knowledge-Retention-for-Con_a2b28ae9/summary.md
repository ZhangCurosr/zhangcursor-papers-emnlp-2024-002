---
title: "SEEKR-Selective-Attention-Guided-Knowledge-Retention-for-Con"
source: https://aclanthology.org/2024.emnlp-main.190.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:28:37"
field: "大语言模型持续学习"
keywords: ["持续学习", "大语言模型", "知识蒸馏", "注意力机制", "灾难性遗忘", "数据效率"]
innovations: ["提出attention head重要性度量（遗忘度×任务敏感性）筛选关键头进行蒸馏", "设计层次化预算分配机制控制训练成本", "仅需1%回放数据即可达到现有方法10%数据的性能"]
benchmarks: ["TRACE", "SuperNI"]
---

# 论文速读：SEEKR-Selective-Attention-Guided-Knowledge-Retention-for-Con

## 一句话总结
本文提出SEEKR方法，通过对大语言模型中选择性重要attention heads进行attention蒸馏，实现更高效的知识保持；仅需1%的回放数据即可达到或超越现有方法使用10%回放数据的效果。

## 研究问题与动机
- **核心问题**：持续学习（CL）中，现有基于数据回放+知识蒸馏的方法未能充分利用旧模型内部嵌入的细粒度知识，导致需要较大比例的回放数据才能达到较好效果。
- **现有方法不足**：
  1. DER++等输出logits蒸馏方法仅关注网络层输出，忽略了LLM内部精妙的attention机制。
  2. 全量attention蒸馏会带来过高计算开销，且不同attention head的重要性存在显著差异。
  3. 缺乏针对attention head的"遗忘敏感度"评估机制，无法高效筛选需保留的关键head。

## 核心贡献（创新点）
1. **探索并验证了attention权重在知识保持中的关键作用**：通过实验证明graft旧模型attention权重可维持旧任务性能，揭示了现有方法未充分利用内部机制知识的局限。
2. **提出知识保持导向的attention head重要性度量**：融合forgettability（遗忘度）和task sensitivity（任务敏感性）两个互补维度，精准识别需蒸馏的关键head。
3. **设计层次化预算分配机制**：通过层预算、head预算、query预算三级控制，实现训练成本的可控调节，兼顾效率与性能。
4. **实现数据高效的持续学习方法**：仅需1%回放数据即可达到或超越对比方法使用10%数据的性能，显著提升replay-based方法的效率。

## 方法详解
**整体框架**：SEEKR在replay-based distillation基础上，新增attention distillation损失项，针对筛选出的重要attention heads进行蒸馏。

**Attention Distillation Loss**：
$$L_{ad}(A, A^k) = \sum_{(l,h)\in \mathcal{H}}\sum_{t\in \mathcal{T}} D_{KL}(A_{l,h,t}^k || A_{l,h,t})$$
其中$\mathcal{H}$为选中的重要heads集合，$\mathcal{T}$为随机采样的queries集合。

**Head重要性度量**（两项相乘）：
- **Task Sensitivity**（任务敏感性）：基于泰勒展开，衡量attention权重变化对任务loss的影响上界
  $$S_{l,h}^k = \mathbb{E}_{(x,y)\in R_k}||\frac{\partial L(x,y)}{\partial A_{l,h}^k}||_F$$
- **Forgettability**（遗忘度）：衡量持续学习过程中attention权重的累积变化量
  $$F_{l,h} = \sum_{k=1}^{i-1}\mathbb{E}_{(x,y)\in R_k}||A_{l,h}^k - A_{l,h}^{k-1}||_F$$
- **综合重要性**：$I_{l,h} = S_{l,h} \cdot F_{l,h}$

**层次化预算分配**：
- 层预算$B_L$：选择重要性得分最高的$B_L$层
- Head预算$B_H$：在各选中层中选择top-$B_H$个heads
- Query预算$B_T$：随机采样$B_T$个queries减少$O(n^2)$计算开销

**总损失函数**：
$$L = L_{task} + \lambda_1 L_{replay} + (1-\lambda_1) L_{ld} + \lambda_2 L_{seekr}$$

## 实验与结果
**数据集与模型**：
- TRACE基准（LLM持续学习）：8个任务，覆盖领域知识、多语言、代码、数学推理
- SuperNI基准（传统NLP任务）：12个任务，覆盖信息抽取、QA、摘要、情感分析
- 模型：LLaMA-2-7B-Chat、Vicuna-7B-v1.5、Vicuna-13B-v1.5

**主要结果**（TRACE基准，LLaMA-2-7B-Chat, Order1）：
| 方法 | OP (BWT) |
|------|----------|
| Replay (1%) | 48.47 (-9.69) |
| DER++ (1%) | 49.22 (-8.32) |
| **SEEKR (1%)** | **54.99 (-2.61)** |
| DER++ (10%) | 55.01 (-3.50) |
| **SEEKR (10%)** | **58.27 (+0.11)** |
| MTL（上界） | 59.38 |

**关键结论**：
- SEEKR(1%)优于DER++(10%)，证明数据效率优势
- SEEKR(10%)的BWT为正(0.11)，无遗忘甚至正迁移
- 通用能力保持：SEEKR(1%)的DeltaGA为-1.05，显著优于SeqFT(-4.27)和Replay(-4.33)
- SuperNI基准同样取得最佳性能

## 相关工作脉络
1. **DER++**：输出logits蒸馏+回放，本文在此基础上引入attention级蒸馏，利用更细粒度知识
2. **EWC/LwF**：正则化/输出蒸馏类方法，本文聚焦attention内部机制的保留
3. **LFPT5**：软prompt生成伪样本，本文直接回放真实数据并增强蒸馏信号
4. **O-LoRA**：参数高效微调+正交约束，本文不依赖PEFT模块，适用于任意transformer模型
5. **MiniLM**：将更大模型的attention scores蒸馏到更小模型，目标不同（知识压缩vs知识保持）
6. **Zhao et al. (2023)**：发现持续学习中仅少数模块剧烈变化，本文据此假设稳定heads编码通用知识

## 局限性与未来方向
1. **数据隐私限制**：作为replay-based方法，历史数据涉及隐私时无法直接使用；可用训练好的LLM生成伪样本替代，但需进一步探索
2. **未在大模型上验证**：受计算资源限制，未在LLaMA-2-70B等更大模型上实验
3. **未扩展到多模态**：尚未探索SEEKR在多模态大语言模型持续学习中的应用

## 研究启发与可借鉴点
1. **细粒度蒸馏信号设计**：从输出logits扩展到attention权重等内部机制，可启发其他任务（如知识压缩、模型编辑）中蒸馏信号的选择
2. **双重度量筛选机制**：forgettability（历史变化）+ task sensitivity（当前影响）的组合思路，可用于其他模块重要性评估
3. **层次化预算控制**：层→head→query的三级预算分配，为计算敏感场景提供可调控的资源分配范式
4. **1%数据高效利用**：验证了高质量蒸馏信号可大幅降低replay数据需求，值得在资源受限场景复用

## 关键术语表
**Catastrophic Forgetting（灾难性遗忘）**：持续学习中新任务训练导致旧任务性能急剧下降的现象。
**Attention Distillation（注意力蒸馏）**：通过KL散度对齐新旧模型attention权重分布的知识迁移方法。
**Task Sensitivity（任务敏感性）**：基于梯度范数衡量attention权重变化对任务loss影响的指标。
**Forgettability（遗忘度）**：attention权重在持续学习过程中的累积变化量，反映知识的稳定性。
**Hierarchical Budget Allocation（层次化预算分配）**：通过层预算、head预算、query预算三级控制蒸馏计算开销的策略。
**OP/BWT（Overall Performance/Backward Transfer）**：整体性能与后向迁移率，分别衡量最终平均表现和对旧任务的性能保持。

## 可复现要素
- **数据集**：TRACE（开源）、SuperNI（开源）；数据公开
- **代码**：https://github.com/jinghan1he/SEEKR（已开源）
- **关键超参**：$\lambda_1=0.5$，$\lambda_2=10^3$(1% replay)/$10^2$(10% replay)，$B_H=128$，$B_L=24$（7B）/8（13B或10% replay），$B_T=100$
- **硬件**：8×NVIDIA Tesla A800，batch size=128，learning rate=1e-5
- **复现状态**：代码已开源，实验结果可复现
