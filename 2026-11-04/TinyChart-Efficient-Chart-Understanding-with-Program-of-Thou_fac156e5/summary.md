---
title: "TinyChart-Efficient-Chart-Understanding-with-Program-of-Thou"
source: https://aclanthology.org/2024.emnlp-main.112.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:49:13"
field: "多模态图表理解与高效视觉编码"
keywords: ["图表理解", "多模态大语言模型", "Program-of-Thoughts", "Visual Token Merging", "高效推理", "低资源部署"]
innovations: ["提出仅 3B 参数的 TinyChart 在多个图表基准上超越 13B 模型和 GPT-4V", "引入 Program-of-Thoughts 学习策略训练模型生成 Python 程序解决数值计算问题", "采用 Visual Token Merging 压缩高分辨率图表的视觉 token 序列实现高效编码"]
benchmarks: ["ChartQA", "Chart-to-Text (Pew)", "Chart-to-Table", "OpenCQA", "ChartX-Cognition"]
---

# 论文速读：TinyChart-Efficient-Chart-Understanding-with-Program-of-Thou

## 一句话总结
TinyChart 是一个仅 3B 参数的高效图表理解多模态大语言模型，通过 Program-of-Thoughts (PoT) 学习策略提升数值计算能力，并通过 Visual Token Merging 模块压缩视觉 token 序列长度，在多个图表理解基准上达到 SOTA，同时实现更高的推理吞吐量。

## 研究问题与动机
- **模型参数过大，难以部署**：现有图表理解模型（如 ChartLlama 13B）需要 ≥26GB VRAM 才能在单 GPU 上运行，限制了在资源受限场景的应用。
- **数值计算易出错**：现有 MLLM 直接生成短答案时，面对涉及数值计算的图表问题容易产生错误，缺乏中间推理步骤。
- **高分辨率编码效率低**：标准 Vision Transformer 处理高分辨率图表图像时会产生冗长的视觉特征序列，导致计算开销大、推理速度慢。

## 核心贡献（创新点）
- **提出 TinyChart 高效图表理解模型**：仅用 3B 参数在多个基准上超越 13B 模型和 GPT-4V，推理吞吐量显著提升。
- **提出 Program-of-Thoughts (PoT) 学习策略**：训练模型生成 Python 程序进行逐步计算，而非直接输出短答案，与已有工作使用固定模板/执行命令（如 ChartAst 的 JSON）相比，Python 更具通用性。
- **采用 Visual Token Merging 高效视觉编码**：在每个 ViT 层内插入无参数合并模块，按相似度聚合 token，减少序列长度，使模型能在有限资源下处理高分辨率图表。
- **构建 ChartQA-PoT 数据集**：包含 140,584 条 (问题, PoT 答案) 对，由 Template-based PoT（119,281 条）和 GPT-based PoT（21,303 条）两部分组成，支持 PoT 学习。

## 方法详解
- **模型架构**：TinyChart 由 Vision Transformer 编码器、Vision-Language Connector（单层 MLP + GeLU）和 LLM（Phi-2）组成，训练目标为标准语言建模损失，仅对响应部分计算损失。
- **Visual Token Merging**：在 ViT 每层内，通过二分图匹配找到 top-r 最相似 token 对，使用平均池化合并，逐层压缩视觉序列长度；相似性度量基于 self-attention 的 Key 余弦距离。
- **Proportional Attention**：为补偿合并后 token 在注意力中的比例下降，在 softmax 内添加 log s 项（s 为该 token 代表的原始 patch 数量），使合并 token 的权重被放大 s 倍：`Attention = softmax(QK^T / √d + log s)V`。
- **Program-of-Thoughts (PoT) 学习**：训练模型生成带自然语言注释的多行 Python 代码，代码通过 Python 解释器执行后得到最终答案；Template-based PoT 使用 40 个 PlotQA 模板问题及手写的 NumPy 代码模板，GPT-based PoT 使用 gpt-3.5-turbo 为真实问题生成代码，并用解释器验证答案正确性。
- **多任务学习**：使用 1.36M 样本的图表理解数据集进行监督微调，涵盖 QA、chart-to-text、chart-to-table 和指令遵循等任务，通过任务特定指令区分不同任务。

## 实验与结果
- **评测基准**：ChartQA（Aug/Hum/Avg）、Chart-to-Text（Pew, BLEU4）、Chart-to-Table（RMS_F1）、OpenCQA（BLEU4）、ChartX-Cognition（GPT-Acc/GPTscore）。
- **ChartQA 最强结果**：TinyChart@768 在 Avg 上达 83.60%，超越 GPT-4V（78.50%）、Gemini-Ultra（80.80%）、Qwen-VL-Max（79.80%）及 ChartAst（79.90%）；在人类问答（ChartQA-human）上达 73.34%，较 ChartAst 提升 7.44 个百分点。
- **其他基准**：Chart-to-Text BLEU4 = 17.18；Chart-to-Table RMS_F1 = 93.78；OpenCQA BLEU4 = 20.39。
- **ChartX-Cognition**：QA 任务 GPT-Acc = 33.35，超越 GPT-4V（33.04）；在 Summ/Desc/Redraw 任务上优于开源模型 ChartLlama 和 ChartAst。
- **推理吞吐量**：TinyChart@768 达 3.14 it/s（V100 32G），约为 ChartLlama（1.94 it/s）的 1.6 倍，ChartAst（1.47 it/s）的 2.1 倍。
- **消融**：PoT 学习将计算问题准确率从 56.64 提升至 78.98（Direct vs PoT）；Token Merging 在 512×512 分辨率下将推理吞吐从 2.38 it/s 恢复到 3.65 it/s（r=20），并使 768×768 输入在 32GB V100 上可训练。

## 相关工作脉络
- **ChartLlama (Han et al., 2023)**：13B 开源图表理解模型，不支持高分辨率（336×336），数值计算能力弱；TinyChart 以更小参数和 PoT+Token Merging 实现更强性能。
- **ChartAst (Meng et al., 2024)**：13B 开源 SOTA，使用 executable JSON 回答计算问题，受限于模板；TinyChart 使用 Python 更通用，且参数仅为其 1/4。
- **DePlot+Codex (Liu et al., 2023a)**：将图表转为表格后用 Codex（175B）推理，精度高但参数巨大；TinyChart 端到端 3B 模型，兼顾效率。
- **LLaVA-1.5 (Liu et al., 2023d)**：通用 MLLM，图表理解性能较低（ChartQA Avg 55.32%）；TinyChart 针对图表任务微调，显著提升。
- **mPLUG-DocOwl 1.5 (Hu et al., 2024b)**：文档理解模型，使用 9× 切片处理高分辨率；TinyChart 用 Token Merging 压缩序列，无需切片，更高效。
- **Token Merging (Bolya et al., 2023)**：ViT 加速方法，按注意力相似度合并 token；TinyChart 将其引入多层并加入 Proportional Attention 修正。

## 局限性与未来方向
- **幻觉问题**：在 chart-to-text 摘要生成中仍存在 MLLM 常见幻觉，可能生成与图表内容不符的信息。
- **无 OCR 文字的数据点估计困难**：即使提高分辨率，模型也难以准确估计没有周围文字标注的数据点数值，缺乏对跨大范围空间关系的理解。
- **未见图表类型**：如图 12(d) 所示，对 3D 柱状图等未见过的图表类型重绘能力不足，需扩展训练数据覆盖。
- **未来方向**：缓解摘要幻觉；提升无 OCR 场景下的数值估计能力；通过自动数据构建技术扩展图表类型覆盖。

## 研究启发与可借鉴点
- **PoT 学习策略可迁移至其他数值密集型任务**：如科学图表、财务数据、数学应用题理解等场景，可借鉴"生成可执行代码 + 解释器执行"的训练范式。
- **Visual Token Merging + Proportional Attention 的组合值得复用**：对于任何需要高分辨率输入的视觉任务（文档理解、医学影像、遥感图），该模块可在不显著损失性能的前提下大幅压缩序列长度。
- **模板 + LLM 混合的数据构建策略**：ChartQA-PoT 中 Template-based（保证覆盖率）与 GPT-based（增加多样性）的结合方式，可用于构建其他领域的指令微调数据集。
- **Combine/Auto/Oracle 多设置评估框架**：论文提出的 Direct/PoT/Combine/Auto/Oracle 五种评估设置，为衡量"代码辅助推理"的有效性提供了系统化评估方案，可复用于其他模型的能力评测。

## 关键术语表
- **Program-of-Thoughts (PoT)**：一种训练 LLM 生成 Python 程序以完成数值计算任务的策略，程序经解释器执行后得到答案，替代直接生成数值短答案。
- **Visual Token Merging**：在 ViT 每层中将相似度最高的 token 对通过平均池化合并，逐层压缩视觉序列长度，减少计算开销。
- **Proportional Attention**：在 self-attention 的 softmax 中输入合并 token 所代表的原始 patch 数量的对数修正项，补偿合并导致的注意力权重偏差。
- **ChartQA-PoT**：基于 ChartQA 训练集构建的包含 140,584 条 (问题, Python 程序) 对的专项数据集，支持 PoT 学习。
- **Relaxed Accuracy**：ChartQA 评测指标，允许模型答案与标准答案之间存在 5% 内的数值误差即视为正确。
- **RMS_F1**：Chart-to-Table 任务评测指标，衡量模型提取的数据表格与真实表格之间的 F1 分数（均方根形式）。

## 可复现要素
- **数据集**：ChartQA-PoT 基于 ChartQA 训练集构建，ChartQA、PlotQA、DVQA、OpenCQA、Pew、Statista、Vistext、ChartSumm 等均为公开或可申请的基准数据集。
- **代码/权重**：论文未提及开源声明，需关注作者后续发布计划。
- **关键超参**：训练 3 个 epoch，batch size=512，学习率=1e-4，warmup 3% steps，总训练耗时 3 天/32×V100 32G；Token Merging 参数 r=20（512×512）/ r=84（768×768）。
