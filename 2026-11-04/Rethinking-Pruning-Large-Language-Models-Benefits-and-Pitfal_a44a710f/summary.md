---
title: "Rethinking-Pruning-Large-Language-Models-Benefits-and-Pitfal"
source: https://aclanthology.org/2024.emnlp-main.68.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:47:10"
---

# 论文速读：Rethinking-Pruning-Large-Language-Models-Benefits-and-Pitfal

## 一句话总结
本文系统重审了LLM后训练剪枝中“最小化重构误差”的常规范式，提出BR、GP、CR三种分块重构技术可将序列剪枝的累积误差降低87%–94%，但同时揭示该目标在小规模校准数据上极易引发过拟合，反而推高perplexity并损害zero-shot性能；作者进一步提出利用原始密集模型自生成校准数据，有效缓解了这一重构精度与泛化能力之间的权衡困境。

## 研究问题与动机
- 现有LLM后训练剪枝通常将问题形式化为在校准数据上最小化密集模型与稀疏模型间的重构误差，但直接对整个模型求解该优化问题需要极高显存，难以落地。
- 近期工作采用“分而治之”策略将模型拆分为子序列逐个剪枝再拼接以节省显存，但各子问题求解产生的非零重构误差会逐级累积，导致最终模型质量显著劣化。
- 盲目追求极致重构误差最小化可能在有限校准数据上发生过拟合，这一“误差下降≠泛化提升”的悖论在既往研究中未被明确识别与量化。
- 现有校准集（如C4随机采样）在数量上与待优化参数规模极不匹配，且分布具有任意性，难以充分表征原始预训练数据，限制了重构策略的实际收益。

## 核心贡献（创新点）
1. 提出BR、GP、CR三种重构优化技术，将分块剪枝的累积误差降低87%–94%；与SparseGPT等单层逐层最小二乘求解方法本质不同，本文通过扩大优化单元与切断误差传播链从结构上压制累积误差。
2. 首次系统揭示“重构误差最小化”在大模型剪枝中的过拟合双刃剑效应；区别于以往将低校准误差直接等同于高剪枝质量的经验假设，本文证明过度优化会在LLaMA-7B等大容量模型上损害perplexity与下游任务准确率。
3. 提出基于密集模型的自生成校准数据策略以平衡重构精度与泛化；与前人依赖固定外部小样本校准集的做法不同，本文利用模型自身生成能力扩充分布更贴近的训练信号，为后续压缩研究提供新调控维度。

## 方法详解
- **Block-wise Reconstruction (BR)**：将优化单元从单层扩展至一整块（如一个完整Transformer Block），联合优化块内所有层参数以最小化扩展后的重构误差 $\min_{w_1,\dots,w_B} \sum_{i=1}^B \|g_i(\bar{w}_i; x_i) - g_i(\bar{m}_i \odot w_i; x_i)\|_2^2$，通过标准梯度下降迭代求解，减少单层求解的子优性。
- **Global Propagation (GP)**：在局部重构某一层/块时，不再使用前一稀疏子模型传播过来的次优输入，而是始终使用原始密集模型的输出作为输入（$x_i = g_{i-1}(\bar{w}_{i-1}; x_{i-1})$），切断误差级联放大链。
- **Cross-Block Reconstruction (CR)**：进一步将重构单元扩展至相邻两个块，使块间表示存在重叠（$h_i = g_i \circ g_{i-1}$, $h_{i+1} = g_{i+1} \circ g_i$），通过共享重叠块参数实现子解之间的“缝合”，进一步压降块边界误差。
- **优化配置**：目标为元素乘积掩码约束下的L2重构误差最小化；采用Adam优化器，学习率0.0002线性衰减，batch size 8，训练10 epoch，无weight decay与gradient clipping；归一化误差按 $\frac{1}{NHT}\|g(\bar{w};\bar{x}) - g(m\odot w;x)\|_2^2$ 计算。

## 实验与结果
- **设置**：在LLaMA-7B与OPT-125M上进行无结构50%剪枝，对比SparseGPT、Wanda、Magnitude三类one-shot剪枝器；校准数据为C4中随机采样的256条1024 token样本；评估涵盖Wiki/PTB/C4 perplexity及BoolQ、RTE、HellaSwag、Winogrande、ARC-e/c、OpenbookQA七项zero-shot准确率。
- **误差降低**：BR+GP+CR较基线LR在LLaMA-7B与OPT-125M的最终block累积误差平均下降
