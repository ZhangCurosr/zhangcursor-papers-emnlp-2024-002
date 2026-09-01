---
title: "Predicate-Debiasing-in-Vision-Language-Models-Integration-fo"
source: https://aclanthology.org/2024.emnlp-main.97.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:26:55"
---

# 论文速读：Predicate-Debiasing-in-Vision-Language-Models-Integration-fo

## 一句话总结
针对场景图生成（SGG）中三元组组合爆炸导致的表示不足问题，本文提出将预训练视觉语言模型（VLM）集成至SGG框架，并通过无训练的Lagrange乘子估计（LM Estimation）反推预训练谓词分布，经事后logits调整后以确信度感知策略动态融合双分支，显著提升长尾与未见关系的预测性能。

## 研究问题与动机
- SGG的三元组空间呈指数级增长，训练集无法覆盖所有subject-object-predicate组合，导致测试集中大量罕见甚至完全未见（unseen）样本，模型预测质量严重下降。
- 直接利用预训练VLM的零样本能力可弥补知识缺口，但预训练语料中谓词频率极度不平衡，直接推理会引入强烈的谓词偏好偏差。
- 现有去偏方法均依赖训练阶段的显式分布信息（如重加权、损失设计），而预训练VLM的数据保密且目标函数与SGG不同，无法建立直接标签映射，导致传统去偏手段失效。
- 如何在无训练开销、无预训练数据访问权限的前提下，消除预训练VLM的谓词偏差并将其与SGG模型有效融合，是本论文的核心动机。

## 核心贡献（创新点）
1. **率先引入预训练VLM缓解SGG表示不足**；与以往专注监督架构优化的工作不同，本文首次将VLM的泛化知识作为补偿源，从数据层面而非网络结构层面破解长尾三元组预测瓶颈。
2. **提出LM Estimation估计不可达的预训练分布**；与依赖训练期重采样或代价函数修正的去偏方法本质不同，该方法将分布估计转化为约束优化问题，仅利用SGG验证集即可反推预训练谓词先验，完全无需访问原始语料。
3. **设计零训练开销的确定性动态集成框架**；与需要全量微调或复杂prompt tuning的VLM适配工作不同，本文仅通过后处理logits偏移与sigmoid权重分配实现双分支融合，保持推理阶段即插即用。
4. **在Visual Genome上实现多项指标SOTA并大幅拉升未见样本性能**；与现有单一目标分布优化的方法不同，本文框架可同时适配训练对齐与均匀分布目标，且在unseen三元组上的增益远超平均水平。

## 方法详解
- **双分支架构**：包含固定零样本VLM分支 $f_{\mathrm{zs}}$（采用MLM提示模板“{z_i} is [MASK] {z_j}”提取K类关系logits）与可训练SGG专用分支 $f_{\mathrm{sg}}$（采用VQA提示配合[CLS] token MLP分类头）。背景类（无关系）因预训练阶段未接触定义，仅由 $f_{\mathrm{sg}}$ 输出。
- **事后Logits调整（Post-hoc Logits Adjustment）**：基于标签迁移（Label Shift）假设 $P(\text{feat}|r)$ 跨分布不变，推导得调整公式为 $\hat{\mathbf{o}}^k(r) = \mathbf{o}^k(r) - \log P_{\mathrm{tr}}(r) + \log P_{\mathrm{ta}}(r)$。SGG分支的 $P_{\mathrm{tr}}$ 为训练集分布 $\pi_{\mathrm{sg}}$；零样本VLM分支的 $P_{\mathrm{tr}}$ 为未知的预训练分布 $\pi_{\mathrm{pt}}$。
- **LM Estimation（拉格朗日乘子估计）**：将 $\pi_{\mathrm{pt}}$ 的求解构建为约束优化问题：在SGG验证集上最小化调整后logits与真实标签的交叉熵，约束 $\pi_{\mathrm{pt}}(r)\geq 0$ 且 $\sum_r \pi_{\mathrm{pt}}(r)=1$，通过引入拉格朗日乘子 $\lambda_r$ 与等式乘子 $v$ 求解最优隐式分布。
- **温度校准与确信度集成**：对去偏后的logits施加 $\tau$-calibration 平滑概率避免过置信；计算单样本确信度 $\mathrm{conf}=\max_r P(r)$，通过 $W_{\mathrm{cer}}\propto \mathrm{sigmoid}(\mathrm{conf}_{\mathrm{sg}}-\mathrm{conf}_{\mathrm{zs}})$ 生成动态权重，加权融合非背景关系概率，最终拼接背景类概率得到全集 $\mathcal{C}_e$ 输出。

## 实验与结果
- **数据集**：Visual Genome（VG），108,077张图像，70%训练/30%测试，聚焦150个物体类别与50个谓词类别。
- **评估设置**：PredCls与SGCls两个子任务，主指标为 Recall@K 与 mRecall@K，另补充 Top-1 Acc 与 mAcc 评估关系分类纯度。
- **主要结果（mRecall@100）**：ViLT-ft-la + Ours 达 46.5（+2.0），Oscar-ft-la + Ours 达 42.7（+1.4），PENET + Ours 达 42.3（+1.6），均超越PCPL、Motifs-GCL、VCTree-GCL等去偏SOTA及传统SGG模型。
- **最强结果**：PENET + Ours 在 Recall@100 达到 71.1（+1.0），ViLT-ft-la + Ours 在 mRecall@100 达到 46.5，均为当前最优。
- **未见三元组表现**：去偏集成后 ViLT 与 Oscar 的 unseen mAcc 分别提升 +5.09 与 +5.71，证实方法对极端长尾/训练零样本的补偿效果显著。
- **消融结论**：直接集成未去偏零样本VLM几乎无增益甚至轻微下降，验证LM Estimation去偏的必要性与有效性。

## 相关工作脉络
- **监督SGG架构派**（VTransE、BGNN、PENET等）：专注网络结构设计与消息传递机制，未系统处理预训练知识引入带来的分布偏移，本文将其作为 $f_{\mathrm{sg}}$ 基座并证明兼容提升。
- **无偏去偏派**（Tang et al. 2020、Dong et al. 2022、Li et al. 2021）：依赖训练期重加权或损失修正，必须访问训练分布，无法迁移至已固化的预训练VLM，本文以事后调整绕过该限制。
- **VLM适配SGG派**（He et al. 2022、Zhang et al. 2023、Yu et al. 2023）：通过prompt-tuning或全量微调将VLM适配下游，计算成本高且依赖大量标注，本文提供零训练成本的即插即用范式。
- **标签迁移/长尾校正派**（Menon et al. 2020等）：提出logits adjustment理论，但前提是可显式统计训练分布；本文将其延伸至“预训练分布不可达”场景，通过约束优化间接求解，填补方法论空白。

## 局限性与未来方向
- **推理计算开销**：零样本VLM采用逐对（pair-wise）转发模式，当图像中对象提案数量庞大（如SGDet任务）时，推理成本显著上升。
- **依赖预训练质量**：最终集成性能高度依赖 $f_{\mathrm{zs}}
