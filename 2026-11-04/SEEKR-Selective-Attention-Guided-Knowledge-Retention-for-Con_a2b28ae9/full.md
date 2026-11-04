# SEEKR: Selective Attention-Guided Knowledge Retention for Continual Learning of Large Language Models

Jinghan He<sup>1,2</sup>, Haiyun Guo<sup>1,2</sup>∗, Kuan Zhu<sup>1,2</sup>∗, Zihan Zhao<sup>5</sup>, Ming Tang<sup>1</sup>, Jinqiao Wang<sup>1,2,3,4</sup>∗

<sup>1</sup>Foundation Model Research Center, Institute of Automation, Chinese Academy of Sciences <sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences <sup>3</sup>Peng Cheng Laboratory, <sup>4</sup>Wuhan AI Research, <sup>5</sup>Chongqing University hejinghan2022@ia.ac.cn, {kuan.zhu, haiyun.guo, jqwang}@nlpr.ia.ac.cn

## Abstract

Continual learning (CL) is crucial for language models to dynamically adapt to the evolving real-world demands. To mitigate the catastrophic forgetting problem in CL, data replay has been proven a simple and effective strat egy, and the subsequent data-replay-based distillation can further enhance the performance. However, existing methods fail to fully exploit the knowledge embedded in models from previous tasks, resulting in the need for a relatively large number of replay samples to achieve good results. In this work, we first explore and emphasize the importance of attention weights in knowledge retention, and then pro pose a SElective attEntion-guided Knowledge Retention method (SEEKR) for data-efficient replay-based continual learning of large language models (LLMs). Specifically, SEEKR performs attention distillation on the selected attention heads for finer-grained knowledge retention, where the proposed forgettability based and task-sensitivity-based measures are used to identify the most valuable attention heads. Experimental results on two continual learning benchmarks for LLMs demonstrate the superiority of SEEKR over the existing methods on both performance and efficiency. Explicitly, SEEKR achieves comparable or even better performance with only 1/10 of the replayed data used by other methods, and reduces the proportion of replayed data to 1%. The code is available at https: //github.com/jinghan1he/SEEKR.

## 1 Introduction

Enabling large language models (Achiam et al., 2023; Touvron et al., 2023; Zheng et al., 2024) with human-like continual learning ability is crucial for the long-term practical deployment. It allows for constant knowledge accumulation on new tasks without the need for costly retraining. However, sequentially finetuning the LLMs with new data can lead to catastrophic forgetting (McCloskey and Cohen, 1989), impairing the general ability of the model and its performance on previous tasks.

![](images/0210627ef8d320474c8b071824b556fac3564a7da4be9b8e41d4a37424c08a3d.jpg)  
Figure 1: Demonstration of the critical role of attention weights in knowledge retention. We apply DER++ (Buzzega et al., 2020) for continual learning on the TRACE benchmark (Wang et al., 2023c) to obtain multiple old task models and the final model. Grafting the attention weights of the old models onto the final model at inference can maintain better performance on the old tasks. Moreover, the final model obtained by our continual learning method, SEEKR, achieves similar results.

Among the array of continual learning methods (Ke and Liu, 2022), data replay stands out as the most widely adopted strategy in practice due to its simplicity and efficacy (Wang et al., 2024). Based on it, replay-based distillation methods, including DER++ (Buzzega et al., 2020) and subsequent techniques (Qin and Joty, 2021; Kang et al., 2022; Gu et al., 2023), further boost the performance by utilizing memories from both data and model perspectives. Specifically, Buzzega et al., 2020; Qin and Joty, 2021; Gu et al., 2023 distill the output logits of old models for knowledge transfer, and Kang et al., 2022 restrict the changes in important feature maps in the image encoders. However, these works have not fully exploited the potential of knowledge distillation in continual learning for LLMs. They focus on the outputs of network layers while neglecting the preservation of intricate internal functions. Consequently, a relatively large amount of replay data is required by these methods to achieve good results.

Recently, many studies have investigated the attention weights of different heads to analyze the interpretability of the internal mechanisms in LLMs (Vig and Belinkov, 2019; Wang et al., 2023a). Inspired by this, we explore whether attention weights play a critical role in knowledge retention during continual learning in LLMs. As shown in Figure 1, grafting the attention weights from the LLM of the old tasks to the final LLM after continual learning can maintain better performance on old tasks, which suggests that the attention weights could be crucial to alleviate the catastrophic forgetting problem and achieve more comprehensive knowledge retention<sup>1</sup>. However, naively preserving the attention weights of all heads in the LLM by distillation introduces significant computational costs. Previous studies have observed a functional specialization phenomenon among attention heads in LLMs (Vig and Belinkov, 2019; Jo and Myaeng, 2020; Li et al., 2023), which indicates the susceptibility of attention heads to forgetting and their importance to previous tasks vary. This property allows us to selectively focus on the valuable attention heads for efficient knowledge retention.

To this end, we propose a finer-grained model distillation method called SElective attEntionguided Knowledge Retention (SEEKR) for continual learning of large language models, which employs attention distillation on the most valuable heads in LLMs to achieve efficient knowledge retention. Specifically, we develop knowledgeretention-oriented head importance measures, which consider both forgettability and task sensitivity, to identify the most valuable heads for distillation. The forgettability, measured by the cumulative changes in attention weights during continual learning, indicates the generality of knowledge and the necessity of distillation. An attention head with higher forgettability indicates a greater need for knowledge retention. The task sensitivity, calculated as the first-order derivative of the task loss, evaluates the importance of maintaining the attention weights of an attention head for a given task. An attention head with greater sensitivity should be prioritized to restrict variations in its attention weights. Using the above two importance scores, SEEKR designs a hierarchical budget allocation mechanism to adaptively select the most valuable attention heads for distillation in a controllable way, which can efficiently regulate the training cost. By using SEEKR, the performance of old tasks can be further maintained as shown in Figure 1.

Extensive experiments are conducted on the recently developed continual learning benchmark for LLMs (Wang et al., 2023c) and the continual learning benchmark on traditional NLP tasks (Wang et al., 2022a). The results consistently demonstrate the superiority of SEEKR in mitigating catastrophic forgetting and maintaining the general capabilities of LLMs. Moreover, as a replay-based method, SEEKR exhibits excellent data efficiency, achieving comparable or better performance with just 1/10 of the replayed data used by the existing methods, reducing the replayed data proportion to only 1%.

Our main contributions are summarized as follows:

• We explore and emphasize the importance of attention weights for knowledge retention, and devise knowledge-retention-oriented measures to identify important attention heads for distillation. The proposed method, SEEKR, can efficiently preserve the finer-grained knowledge in the selected attention heads.

• Extensive experiments validate the superiority of SEEKR, showcasing its data efficiency by using just 1% of replay samples to achieve the comparable or better performance that other methods reach with 10% of replay samples.

## 2 Preliminary

## 2.1 Continual Learning for LLMs

Continual learning algorithms aim to accumulate knowledge across sequential tasks. Suppose there are N tasks with the corresponding datasets $\{ \mathcal { D } _ { 1 } , \cdot \cdot \cdot , \mathcal { D } _ { N } \}$ . An LLM, parameterized by θ, are instruction-tuned on each dataset $\mathcal { D } _ { i }$ sequentially to optimize the following objective:

$$
L _ { t a s k } = \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \in \mathcal { D } _ { i } } \big [ - \log p _ { \theta } ( \pmb { y } | \pmb { x } ) \big ]\tag{1}
$$

where x, y are the instruction and true answer, respectively. Hereafter, we assume the current task is i and omit the corresponding subscript. In this paper, we study a more common scenario in practice where a small amount of data from the old tasks $\{ R _ { 1 } , . . . , R _ { N } \}$ can be stored in the memory buffer to aid the continual learning process. During training on the current task, replay data are acquired from the memory buffer, and the model is optimized for their previous tasks:

$$
L _ { r e p l a y } = \sum _ { k = 1 } ^ { i - 1 } \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \in \mathcal { R } _ { k } } \big [ - \log p _ { \theta } ( \pmb { y } | \pmb { x } ) \big ]\tag{2}
$$

## 2.2 Knowledge Distillation for CL

Knowledge distillation (Hinton et al., 2015) is a technique to train a student model to replicate the teacher model’s behavior for efficient knowledge transfer. To mitigate forgetting on previous tasks in CL, knowledge distillation is performed between each old model $p _ { \theta _ { k } }$ and the current model $p _ { \theta }$ using replay samples from $R _ { k }$ (Buzzega et al., 2020):

$$
L _ { l d } = \sum _ { k = 1 } ^ { i - 1 } \mathbb { E } _ { ( { \pmb x } , { \pmb y } ) \in { \mathcal { R } } _ { k } } \left[ D _ { K L } \big ( p _ { \theta _ { k } } ( { \pmb y } | { \pmb x } ) \| p _ { \theta } ( { \pmb y } | { \pmb x } ) \big ) \right]\tag{3}
$$

The predicted logits from the old model $p _ { \theta _ { k } } ( \pmb { y } | \pmb { x } )$ are saved in the memory buffer along with the replay samples and loaded during training as auxiliary supervision signals.

## 3 Method

In this section, we introduce SEEKR, an efficient replay-based distillation method that identifies valuable attention heads and performs attention distillation for finer-grained knowledge retention.

## 3.1 Attention-guided Knowledge Retention

To achieve more comprehensive knowledge retention by using less replay data, we perform an elaborate distillation on the key mechanism of LLMs, i.e. the attention weights. Specifically, the outputted attention weights of the h-th head in the l-th layer are denoted as $A _ { l , h }$

$$
A _ { l , h } = \mathrm { s o f t m a x } ( \frac { Q _ { l , h } K _ { l , h } ^ { T } } { \sqrt { d _ { k } } } + M _ { c a u s a l } )\tag{4}
$$

where Q and K represent the query vectors and the key vectors in the self-attention operation, respectively. $M _ { c a u s a l }$ is the casual attention mask in LLMs. We use t to index the attention distribution of the t-th query in $A _ { l , h }$ and denote it as $A _ { l , h , t }$ The attention distributions of query t from each old task model $A _ { l , h , t } ^ { k }$ and the current model $A _ { l , h , t }$ are aligned through the KL divergence loss:

$$
L _ { a d } ( A , A ^ { k } ) = \sum _ { ( l , h ) \in \cal U } \sum _ { t = 1 } ^ { | x \oplus y | } D _ { K L } ( A _ { l , h , t } ^ { k } | | A _ { l , h , t } )\tag{5}
$$

where $U$ stands for the set of all attention heads in all layers. x y is the concatenated sequence of x and y, and $| { \pmb x } \oplus { \pmb y } |$ means the length of the whole sequence. In SEEKR, the knowledge distillation is performed at the head level, which can offer more direct and refined regulation on the intricate internal functions of LLMs, achieving a more comprehensive and efficient utilization of the limited replay data.

## 3.2 Important Head Identification

In practice, distilling all the attention heads in an LLM is costly and unnecessary, as different heads exhibit varying levels of task sensitivity and forgettability. Therefore, we propose a two-dimensional measure to identify the most valuable attention heads for knowledge retention.

## 3.2.1 Task Sensitivity Measure

For a model adapted to task k, we assess to which extent changes in the attention weights of each head affect the task performance. Following common practice, we resort to Taylor expansion to formalize this influence (Kang et al., 2022):

$$
\begin{array} { l } { \displaystyle \Delta L ( \pmb { x } , \pmb { y } ) \approx \left. \frac { \partial L ( \pmb { x } , \pmb { y } ) } { \partial A _ { l , h } } , \Delta A _ { l , h } \right. _ { F } } \\ { \displaystyle \qquad \leq \vert \vert \frac { \partial L ( \pmb { x } , \pmb { y } ) } { \partial A _ { l , h } } \vert \vert _ { F } \cdot \vert \vert \Delta A _ { l , h } \vert \vert _ { F } } \end{array}\tag{6}
$$

where $\langle \cdot , \cdot \rangle _ { F }$ and $\| \cdot \| _ { F }$ denote the Frobenius inner product and Frobenius norm, respectively. This inequality demonstrates the upper bound on the increase in task loss due to changes in the attention weights, i.e. $\Delta A _ { l , h }$ . A larger coefficient indicates a higher upper bound for the same changes in $A _ { l , h }$ This implies that changes in these attention weights are more likely to increase task loss or degrade task performance, making it crucial to keep them unchanged. Therefore, we take the coefficient to estimate the sensitivity of the task k to $A _ { l , h } ^ { k }$ , which is formulated as:

$$
S _ { l , h } ^ { k } = \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \in R _ { k } } | | \frac { \partial L ( \pmb { x } , \pmb { y } ) } { \partial A _ { l , h } ^ { k } } | | _ { F }\tag{7}
$$

The importance scores are then normalized within each layer to obtain $\widetilde { S } _ { l , h } ^ { k } .$ , thereby mitigating the impact of varying gradient magnitudes across different layers. During training on the new task, the importance of all previous tasks should be considered. Therefore, the task sensitivity measure for each attention head is defined as:

$$
S _ { l , h } = \sum _ { k = 1 } ^ { i - 1 } \widetilde { S } _ { l , h } ^ { k }\tag{8}
$$

## 3.2.2 Forgettability Measure

The second measure assesses the necessity for performing attention distillation on each attention head. We hypothesize that there exist some attention heads whose attention weights remain relatively stable during continual training on new tasks, suggesting that they are less sensitive to task-specific details and focus more on general or shared knowledge. This hypothesis aligns with prior research (Zhao et al., 2023), which revealed that only a few modules change drastically during continual learning, while others stay relatively stable and may be shared across tasks as common knowledge. Based on this, we propose that stable attention heads may encode general knowledge that is less prone to forgetting, and thus distillation of such heads should be minimized. To this end, we leverage the variability of the attention weights during continual learning to measure the forgettability of the attention head:

$$
F _ { l , h } = \sum _ { k = 1 } ^ { i - 1 } \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \in R _ { k } } | | A _ { l , h } ^ { k } - A _ { l , h } ^ { k - 1 } | | _ { F }\tag{9}
$$

Higher forgettability scores indicate a greater necessity for distilling these attention heads.

## 3.2.3 Overall Importance Measure

To identify valuable heads for attention-guided knowledge retention, we fuse the two complementary measures through multiplication, ultimately forming a holistic metric:

$$
I _ { l , h } = S _ { l , h } \cdot F _ { l , h }\tag{10}
$$

After each task, $S _ { l , h }$ and $F _ { l , h }$ of each attention head are updated according to Equation 8 and 9, and the overall importance $I _ { l , h }$ is re-calculated accordingly.

## 3.3 Hierarchical Budget Allocation

Based on the above head importance measure, we propose a hierarchical budget allocation strategy to manage the training cost. We define the group of selected layers and heads as L and H, with budgets

Algorithm 1 SEEKR   
Input Initial model $\theta _ { 0 } ,$ , Datasets $\{ \mathcal { D } _ { i } \} _ { i = 1 } ^ { N }$ , Hyper  
parameters $\lambda _ { 1 } , \lambda _ { 2 } , B _ { L } , B _ { H } , B _ { T }$   
1: Initialize $L , H  U ; S _ { l , h } , F _ { l , h } , I _ { l , h }  0 ;$   
2: for task $i \gets 1$ to N do   
3: for epoch $e \gets 1$ to epochs do   
4: for batch in $( \mathsf { U } _ { k = 1 } ^ { i - 1 } R _ { k } ) \mathsf { U } \mathscr { D } _ { i }$ do   
5: Minimize L in Eq. 13;   
6: end for   
7: end for   
8: $R _ { i } \gets \mathrm { R }$ andom $\left( \mathcal { D } _ { i } \right)$   
9: Update $S _ { l , h } , F _ { l , h } , I _ { l , h }$ using Eq. 8-10;   
10: Update $L , H$ using Eq. 11;   
11: Randomly select $T ;$   
12: end for

$B _ { L }$ and $B _ { H }$ . Our strategy involves two steps: (1) Select the top- $B _ { L }$ layers that maximize the layerwise importance scores $\sum _ { h } I _ { l , h }$ . (2) Among all the attention heads in all these layers, activate the top- $B _ { H }$ heads for attention distillation. Based on the above process, the set H of the selected heads can be expressed as:

$$
\begin{array} { l } { { \displaystyle H = \arg \mathrm { t o p k } \{ I _ { l , h } \mid l \in L \} } } \\ { { \displaystyle \qquad ( l , h ) } } \\ { { \displaystyle L = \arg \mathrm { t o p k } \sum _ { l } I _ { l , h } } } \end{array}\tag{11}
$$

where arg topk denotes the set of z that achieves the k largest values. k is $B _ { H }$ for H and $B _ { L }$ for $L .$ Additionally, to reduce the $O ( n ^ { 2 } )$ cost of distilling the entire attention map, we introduce a query budget $B _ { T }$ and randomly select the queries $T$ for distillation. After determining H and $T$ , we can rewrite Equation 5 as follows:

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { a d } ( A , A ^ { k } ) = \sum _ { ( l , h ) \in { \cal H } } \sum _ { t \in { \cal T } } D _ { K L } ( A _ { l , h , t } ^ { k } | | A _ { l , h , t } ) } } \\ { { \displaystyle \qquad \ } } \\ { { \displaystyle { \cal L } _ { s e e k r } = \sum _ { k = 1 } ^ { i - 1 } \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { y } ) \in { \mathcal R } _ { k } } \left[ { \cal L } _ { a d } ( A , A ^ { k } ) \right] } } \end{array}\tag{12}
$$

Overall, SEEKR sets three types of budgets to allow flexible control over training costs. First, the layer budget adjusts the number of layers for attention-accelerating algorithms or our distillation strategy. Second, the head budget filters out less essential heads and reduces training costs. Lastly, the query budget specifically targets at reducing the costs associated with distilling long texts.

<table><tr><td rowspan="2"></td><td colspan="2">LLaMA-2-7B-Chat</td><td colspan="2">Vicuna-7B-v1.5</td></tr><tr><td>Order1</td><td>Order2</td><td>Order1</td><td>Order2</td></tr><tr><td>SeqFT</td><td>47.63 (-11.45)</td><td>45.12 (-12.27)</td><td>41.91 (-15.29)</td><td>45.70 (-12.01)</td></tr><tr><td>EWC LwF</td><td>48.20 (-9.48)</td><td>44.54 (-12.00)</td><td>41.88 (-15.57)</td><td>49.32 (-8.62)</td></tr><tr><td>LFPT5</td><td>41.86 (-6.50)</td><td>40.25 (-5.96)</td><td>41.19 (-5.54)</td><td>42.99 (-4.72)</td></tr><tr><td>L2P</td><td>38.67 (-11.43)</td><td>42.26 (-7.43)</td><td>41.79 (-8.10)</td><td>39.22 (-10.70)</td></tr><tr><td></td><td>35.23 (-15.96)</td><td>34.63 (-16.86)</td><td>32.26 (-16.58)</td><td>35.14 (-15.88)</td></tr><tr><td>PP O-LoRA</td><td>29.41 (-5.79)</td><td>21.58 (-8.83)</td><td>26.64 (-6.10)</td><td>24.88 (-11.54)</td></tr><tr><td></td><td>44.64 (-4.20)</td><td>42.83 (-9.11)</td><td>43.42 (-6.27)</td><td>43.87 (-6.37)</td></tr><tr><td>Replay (1%)</td><td>48.47 (-9.69)</td><td>47.04 (-10.24)</td><td>48.43 (-9.23)</td><td>49.46 (-9.43)</td></tr><tr><td>DER++ (1%)</td><td>49.22 (-8.32)</td><td>46.59 (-10.91)</td><td>49.01 (-9.04)</td><td>51.09 (-7.85)</td></tr><tr><td>SEEKR (1%)</td><td>54.99 (-2.61)</td><td>54.69 (-2.53)</td><td>55.78 (-2.64)</td><td>54.91 (-3.40)</td></tr><tr><td>Replay (10%)</td><td>55.67 (-3.96)</td><td>53.39 (-4.15)</td><td>55.62 (-2.15)</td><td>54.57 (-3.41)</td></tr><tr><td>DER++ (10%)</td><td>55.01 (-3.50)</td><td>54.05 (-2.94)</td><td>56.06 (-1.17)</td><td>55.14 (-3.77)</td></tr><tr><td>SEEKR (10%)</td><td>58.27 (0.11)</td><td>57.27 (-0.47)</td><td>57.54 (0.47)</td><td>56.86 (-1.01)</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MTL</td><td colspan="2">59.38</td><td colspan="2">58.18</td></tr></table>

Table 1: Comparison with the state-of-the-art methods on TRACE benchmark. The results are obtained by using two popular LLMs with two transfer orders, and are presented in the format of OP (BWT).

## 3.4 Overall Objective

Combining the above objectives, the overall loss for the new and replay data is formalized as:

$$
L = L _ { t a s k } + \lambda _ { 1 } L _ { r e p l a y } + ( 1 - \lambda _ { 1 } ) L _ { l d } + \lambda _ { 2 } L _ { s e e k r }\tag{13}
$$

where $\lambda _ { 1 }$ is a coefficient to balance the text generation loss supervised by true labels and teacher models, and $\lambda _ { 2 }$ is a weighting factor to adjust the magnitude of attention distillation loss. The overall process of SEEKR is shown in Algorithm 1.

## 4 Experiments

## 4.1 Experimental Setup

## 4.1.1 Datasets

CL Benchmark for LLMs. We evaluate our method on TRACE (Wang et al., 2023c), a continual learning benchmark for LLMs that includes eight datasets covering domain-specific knowledge, multilingual capabilities, code generation, and mathematical reasoning. We use the reasoningaugmented version of datasets and conduct experiments under two task orders following the original paper. After continual learning, we assess the performance of the continually learned tasks and the changes in the general ability of LLMs.

CL on Traditional NLP Tasks. SuperNI (Wang et al., 2022a) contains a variety of traditional NLP tasks and can serve as a practical benchmark for continual learning of large language models. Similar to Zhao et al., 2024, we select three datasets for each of the four types of tasks, i.e. information extraction, question answering, summarization, and sentiment analysis, to examine the effectiveness of continual learning methods. For each dataset, 1000 samples and 100 samples are randomly sampled for training and testing, respectively.

## 4.1.2 Metrics

Let $a _ { i , j }$ denote the testing performance on the i-th task after training on the j-th task. We report the overall performance (OP) (Chaudhry et al., 2018) and the backward transfer (BWT) (Lopez-Paz and Ranzato, 2017) after training on the last task:

$$
O P = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } a _ { i , T }\tag{14}
$$

$$
B W T = \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } ( a _ { i , T } - a _ { i , i } )\tag{15}
$$

Moreover, we also report the general ability (GA) and the delta general ability (DeltaGA) (Wang et al., 2023c) after continual learning. GA is the average performance across evaluation datasets in Table 2 and DeltaGA shows the change in GA compared to the initial model.

<table><tr><td></td><td>MMLU</td><td>GSM</td><td>BBH</td><td>TydiQA</td><td>BoolQ</td><td>PIQA</td><td>GA (DeltaGA)</td></tr><tr><td>LLaMA-2-7B-Chat</td><td>46.89</td><td>27.14</td><td>39.73</td><td>16.76</td><td>79.79</td><td>76.33</td><td>47.77</td></tr><tr><td>SeqFT</td><td>45.16</td><td>14.03</td><td>32.50</td><td>14.84</td><td>79.00</td><td>75.49</td><td>43.50 (-4.27)</td></tr><tr><td>Replay (1%)</td><td>45.49</td><td>12.70</td><td>33.46</td><td>14.65</td><td>78.69</td><td>75.65</td><td>43.44 (-4.33)</td></tr><tr><td>SEEKR (1%)</td><td>46.32</td><td>20.85</td><td>38.52</td><td>18.22</td><td>80.64</td><td>75.79</td><td>46.72 (-1.05)</td></tr><tr><td>Vicuna-7B-v1.5</td><td>49.39</td><td>23.43</td><td>41.12</td><td>15.01</td><td>81.41</td><td>76.77</td><td>47.86</td></tr><tr><td>SeqFT</td><td>46.26</td><td>11.68</td><td>33.09</td><td>13.44</td><td>79.97</td><td>76.72</td><td>43.52 (-4.34)</td></tr><tr><td>Replay (1%)</td><td>47.14</td><td>15.77</td><td>33.51</td><td>14.14</td><td>80.57</td><td>76.39</td><td>44.59 (-3.27)</td></tr><tr><td>SEEKR (1%)</td><td>48.83</td><td>17.55</td><td>38.17</td><td>16.32</td><td>81.96</td><td>77.23</td><td>46.68 (-1.18)</td></tr></table>

Table 2: Changes in general language understanding and reasoning abilities after continual learning with different methods. The reported results of all continual learning models are averaged over two task orders.

## 4.2 Baselines

We compare SEEKR with nine baseline methods: (1) SeqFT sequentially finetunes the model without continual learning strategies. (2) EWC (Kirkpatrick et al., 2017) regularizes parameter variations based on parameter importance scores. (3) LwF (Li and Hoiem, 2017) distills the model of the last task using the current task data. (4) Replay finetunes the model with the current task data and a small number of replay samples. (5) DER++ (Buzzega et al., 2020) saves the logits of the replay samples from the old models for distillation, and combines distillation and replay to reduce forgetting. (6) LFPT5 (Qin and Joty, 2021) learns a soft prompt to generate pseudo samples of previous tasks for replaying. (7) O-LoRA (Wang et al., 2023b) imposes orthogonal constraints on the LoRA matrices for all tasks. (8) L2P (Wang et al., 2022b) instantiates a prompt pool for adaptive prompt selection and prompt tuning for individual samples. (9) PP (Razdaibiedina et al., 2023) tunes a set of prompts for each task and concatenates them together. In addition, the results of the multi-task trained models are reported as MTL and serve as the upper-bound reference.

## 4.3 Implementation Details

SEEKR is a versatile continual learning method compatible with any transformer-based model. Following Wang et al., 2023c, we conduct our main experiments on two popular LLMs, i.e. LLaMA-2-7B-chat (Touvron et al., 2023) and Vicuna-7Bv1.5 (Zheng et al., 2024). We also scale to a larger model Vicuna-13B-v1.5 to validate the effectiveness of SEEKR. All models are trained on 8 NVIDIA Tesla A800 using the DeepSpeed library. The training batch size is 128. For methods not involving parameter-efficient tuning modules, the learning rate is 1e-5. For replay-based methods, the default replay ratio is 1%. For SEEKR, $\lambda _ { 1 }$ in Equation 13 is set to 0.5. $\lambda _ { 2 }$ is 1e3 for a replay ratio of 1% and 1e2 for 10%. The head budget $B _ { H }$ is 128, and the layer budget $B _ { L }$ is 24 by default and 8 for 13B models or a replay ratio of 10%. The query budget $B _ { T }$ is 100. All experimental results were averaged over 3 runs. More implementation details can be found in Appendix B.

<table><tr><td></td><td>Order3</td><td>Order4</td></tr><tr><td>SeqFT LwF</td><td>42.62 (-18.12)</td><td>50.52 (-9.88)</td></tr><tr><td>LFPT5</td><td>43.29 (-15.47) 42.05 (-16.26)</td><td>47.35 (-12.57) 46.09 (-14.16)</td></tr><tr><td>L2P PP</td><td>32.71 (-22.34)</td><td>31.00 (-23.82)</td></tr><tr><td>O-LoRA</td><td>17.96 (-21.27)</td><td>12.19 (-29.08)</td></tr><tr><td>Replay (1%)</td><td>30.07 (-24.47)</td><td>26.70 (-33.82)</td></tr><tr><td>DER++ (1%)</td><td>55.00 (-4.27)</td><td>54.78 (-5.31)</td></tr><tr><td>SEEKR (1%)</td><td>55.89 (-4.51) 57.04 (-3.15)</td><td>53.48 (-5.01) 58.26 (-2.52)</td></tr></table>

Table 3: Comparison with the state-of-the-art methods on SuperNI benchmark. The experiments are conducted on LLaMA-2-7B.

## 4.4 Main Results

Table 1 compares the overall continual learning performance of SEEKR with other baselines on TRACE benchmark. Following Wang et al., 2023c, we also report the changes in the general ability of LLMs after continual learning in Table 2. Similar experiments on the SuperNI benchmark are displayed in Table 3.

SEEKR effectively mitigates catastrophic forgetting of continually learned tasks. Compared to traditional and state-of-the-art continual learning approaches, SEEKR consistently achieves the highest OP and the lowest magnitude of BWT in all settings. Note that the BWT metric specifically captures the resistance of methods to catastrophic forgetting, thus the results demonstrate SEEKR’s superiority in maintaining performance on newly learned tasks. Additionally, on the SuperNI benchmark, we achieve the best performance using only a small proportion of replay samples, likely because the benchmark consists of traditional NLP tasks, which are less challenging.

![](images/c392c732dd44de0b52827c9df223afc11fe754c8a0b969b21baedc97156750ac.jpg)

![](images/734995f39094bb326ca59c4528a9b5c6cddd328f210c4ce1cad47d41b1f963eb.jpg)  
(a) Effect of distillation budget

![](images/f175f0bcef8741d5f329a662119870e72ef3dd211c5756b2ea5ca20c7f6f74c7.jpg)

![](images/afe0f1573cd278d756cabf8a9bc799d062f621c3388d963e6015a708430e0d3f.jpg)  
(b) Effect of replay data ratio  
Figure 2: Results of SEEKR across different distillation budgets and different replay data ratios.

SEEKR fully exploits the small amount of replay data and exhibits excellent data efficiency. Among all replay-based methods, SEEKR stands out with a distinct advantage. On the TRACE benchmark, both Replay and DER++ show limited benefits with a lower ratio of replay data. In contrast, SEEKR demonstrates remarkable performance with just 1% of the samples replayed, achieving comparable or even better results than other methods that replay 10% of the samples. This underscores the ability of SEEKR to maximize the use of a small number of old samples and the inherent knowledge in the old models.

SEEKR is effective in maintaining the general ability of the original LLM. Table 2 exhibits the changes in LLMs’ general ability after continual learning. LLMs that are continually trained on new tasks show a decline in general task performance, demonstrating the catastrophic forgetting of their original capabilities. Results validated that SEEKR, which elaborately distills multiple finetuned LLMs with a variety of data, helps to maintain the general capabilities of the model. This could benefit from the fact that our approach preserves the knowledge of the intricate internal functions in LLMs at the attention head level.

<table><tr><td></td><td>Order1</td><td>Order2</td></tr><tr><td>random</td><td>53.25 (-4.63)</td><td>52.62 (-5.11)</td></tr><tr><td>task-sensitivity-only</td><td>53.91 (-4.29)</td><td>53.56 (-2.84)</td></tr><tr><td>forgettability-only</td><td>54.06 (-3.31)</td><td>53.63 (-3.11)</td></tr><tr><td>both</td><td>54.99 (-2.61)</td><td>54.69 (-2.53)</td></tr></table>

Table 4: Ablation study on the head importance measure. The experiments are conducted on LLaMA-2-7B.

![](images/f9a3770c2a45620a96b0f901f7d3a3b099b5d40b174832b9505b7ec926fe38a4.jpg)  
Figure 3: The continual learning performance and the changes of general ability with Vicuna-13B-v1.5.

## 4.5 Ablation Studies

Effect of distillation budget. Figure 2 (a) exhibits the performance of our method under different budgets. With a fixed layer budget of 24, a larger head budget can lead to better results, but this improvement tends to plateau at a budget of 128. Similarly, the performance improves with an increasing layer budget and reaches its optimum at 24. These results further emphasize the significance of distilling the right attention heads. Distilling less essential attention heads may lead to ineffective work.

Effect of more replay samples. To further explore the potential of SEEKR, we experiment with an increased ratio of replay samples. Meanwhile, we compare SEEKR with Replay to demonstrate its data efficiency. As shown in Figure 2 (b), SEEKR steadily improves performance as the number of replay samples grows. At a replay ratio of 10%, the

![](images/f0d8fe9c4de163c85d451e5a0892bb4b356eceb7943ef1ceb3960bee49291c7d.jpg)  
Figure 4: Histogram of the cumulative variation in the attention weights of the attention heads in the model during sequential finetuning.

![](images/9bc439f69a92f5e537e4452596bf114430139031e57a3b02098d67cc5b6dd7b7.jpg)  
Figure 5: Visualization of the importance scores of all heads in the model.

BWT score exceeds 0, indicating no forgetting or even a positive transfer has been achieved, and the overall performance approximates the upper bound of multi-task training. Moreover, compared with Replay, SEEKR is very data efficient by utilizing only 1% of the old data to achieve the performance of replaying ten times that amount.

Effectiveness of our head importance measure. We present the results of the ablation study on the proposed head importance measure in Table 4 . The results show that the random selection of distilled attention heads noticeably resulted in a higher forgetting indicator, while using either sensitivity-based or variation-based measures helps identify important heads for knowledge retention. Finally, combining both of the above measures produces the best results.

## 4.6 Discussions

Scale to larger models. To validate the generalizability of SEEKR across different model scales, we conducted additional experiments on a larger model, Vicuna-13B-v1.5. Figure 3 shows that our approach still effectively preserves both the performance of newly learned tasks and the general capabilities of the original model.

Variation in attention weights. To further confirm our hypothesis in Section 3.2.2, we examine the cumulative changes in attention weights of each attention head during sequential finetuning. The results in Figure 4 reveal that most attention heads remain stable throughout the process, while a small proportion undergo significant changes. This observation is similar to prior findings (Zhao et al., 2023) and supports our hypothesis that these stable attention heads do exist, making it reasonable to identify them and avoid unnecessary attention distillation.

Analysis of selected important heads. Figure 5 illustrates that important attention heads are mainly distributed in the middle and deep layers of the model, while almost none are observed in the shallow layers. This aligns with the idea that the shallow layers encode more generalized knowledge and are less susceptible to forgetting. A closer look at Figure 5 further reveals that the importance scores for the deeper layers are concentrated in a few heads, while those for the middle layers are more evenly spread over a larger number of heads. This may be because the heads in the deeper layers are more thoroughly function-specialized.

## 5 Related Works

## 5.1 Continual Learning for LLMs

Existing continual learning methods are typically classified into three broad categories: regularization-based methods, replay-based methods, and architectural-based methods. (1) Regularization-based methods restrict model variations to alleviate forgetting. Some works penalize changes to important parameters for previously learned tasks (Kirkpatrick et al., 2017; Wang et al., 2023b; He et al., 2023), while others resort to knowledge distillation to maintain the old models’ predictions (Li and Hoiem, 2017; Buzzega et al., 2020; Kang et al., 2022). (2) Replay-based methods replay data from the old tasks during training on the new task. Experience replay methods (Rebuffi et al., 2017; Wang et al., 2024) design data selection strategies of previous samples, and generative replay (Shin et al., 2017; Qin and Joty, 2021) uses generative models to produce synthetic data from previous tasks. Other methods (Yang et al., 2023) retain old tasks by storing statistical information of the old tasks instead of the original data. (3) Architecture-based methods alter the model structure to accommodate different tasks. Recently, this type of methods on LLMs (Wang et al., 2022b; Razdaibiedina et al., 2023) often add parameterefficient tuning modules for new tasks.

SEEKR falls into the category of replay-based distillation methods and focuses on the preservation of important attention mechanisms in LLMs. Unlike existing output or parameter importance measures (Kirkpatrick et al., 2017; Kang et al., 2022), which focus solely on task loss sensitivity, our head importance measure includes a forgettability aspect. This reflects the susceptibility to forgetting and the generality of knowledge in different heads, thereby determining the necessity for distillation.

## 5.2 Knowledge Distillation

Knowledge distillation aims to leverage the teacher model’s performance and generalize it to the student model (Hinton et al., 2015; Park et al., 2019; Guo et al., 2023). For language models, Sanh et al., 2019 uses the teacher model’s generation distribution for each token as a supervision signal for the student model, and some other works (Wang et al., 2020b,a) distill the attention scores of one layer to transfer the knowledge of larger LMs into smaller models. Unlike their objectives of transferring knowledge between models of different sizes, we use attention distillation for knowledge retention. Both our teacher and student models share a similar architecture and are derived from the same pre-trained LLM, which enables head-by-head and layer-by-layer distillation.

## 6 Conclusion

In this paper, we propose SEEKR, an efficient replay-based distillation method for continual learning in LLMs. SEEKR resorts to attention distillation of important heads for finer-grained knowledge retention, which identifies valuable heads through the proposed knowledge-retention-oriented importance measures. Combined with a hierarchical budget allocation mechanism, SEEKR can ensure its utility across various resource levels. Extensive experiments consistently validated the effectiveness of our method in preserving the performance of newly learned tasks and the original ability of the initial LLMs.

## Limitations

Despite the potential benefits of SEEKR, several limitations need to be considered. First, SEEKR is inherently a replay-based approach, which may not be applicable in scenarios where historical data involves privacy concerns. A potential solution is to use SEEKR with pseudo-samples generated by the trained LLM, but this approach requires further exploration. Second, due to computational resource limitations, we did not experiment with larger-scale LLMs like LLaMA-2-70B. Additionally, the application of SEEKR to continual learning with multimodal large language models remains to be explored in the future.

## Acknowledgements

This work was supported by National Key R&D Program of China under Grant No.2021ZD0110400, also partly supported by Beijing Natural Science Foundation under Grant 4244099, National Natural Science Foundation of China under Grant No.62276260, Postdoctoral Fellowship Program of CPSF under Grant GZC20232996, China Postdoctoral Science Foundation under Gant 2024M753498.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. 2020. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439.

Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. 2020. Dark experience for general continual learning: a strong, simple baseline. Advances in neural information processing systems, 33:15920–15930.

Arslan Chaudhry, Puneet K Dokania, Thalaiyasingam Ajanthan, and Philip HS Torr. 2018. Riemannian walk for incremental learning: Understanding forgetting and intransigence. In Proceedings ofthe European conference on computer vision (ECCV), pages 532–547.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044.

Jonathan H Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. 2020. Tydi qa: A benchmark for information-seeking question answering in ty pologically di verse languages. Transactions ofthe Associationfor Computational Linguistics, 8:454–470.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Ahmad Ghazal, Tilmann Rabl, Minqing Hu, Francois Raab, Meikel Poess, Alain Crolotte, and Hans-Arno Jacobsen. 2013. Bigbench: Towards an industry standard benchmark for big data analytics. In Proceedings ofthe 2013 ACM SIGMOD international conference on Management ofdata, pages 1197–1208.

Qiao Gu, Dongsub Shim, and Florian Shkurti. 2023. Preserving linear separability in continual learning by backward feature projection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24286–24295.

Guangyu Guo, Longfei Han, Le Wang, Dingwen Zhang, and Junwei Han. 2023. Semantic-aware knowledge distillation with parameter-free feature uniformization. Visual Intelligence, 1(1):6.

Jinghan He, Haiyun Guo, Ming Tang, and Jinqiao Wang. 2023. Continual instruction tuning for large multimodal models. arXiv preprint arXiv:2311.16206.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

Jae-young Jo and Sung-Hyon Myaeng. 2020. Roles and utilization of attention heads in transformer-based neural language models. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 3404–3417.

Minsoo Kang, Jaeyoo Park, and Bohyung Han. 2022. Class-incremental learning by knowledge distillation with adaptive feature consolidation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 16071–16080.

Zixuan Ke and Bing Liu. 2022. Continual learning of natural language processing tasks: A survey. arXiv preprint arXiv:2211.12701.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526.

Chong Li, Shaonan Wang, Yunhao Zhang, Jiajun Zhang, and Chengqing Zong. 2023. Interpreting and exploiting functional specialization in multi-head attention under multi-task learning. arXiv preprint arXiv:2310.10318.

Zhizhong Li and Derek Hoiem. 2017. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947.

David Lopez-Paz and Marc’Aurelio Ranzato. 2017. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30.

Michael McCloskey and Neal J Cohen. 1989. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology oflearning and motivation, volume 24, pages 109–165. Elsevier.

Wonpyo Park, Dongju Kim, Yan Lu, and Minsu Cho. 2019. Relational knowledge distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3967–3976.

Chengwei Qin and Shafiq Joty. 2021. Lfpt5: A unified framework for lifelong few-shot language learning based on prompt tuning of t5. arXiv preprint arXiv:2110.07298.

Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Madian Khabsa, Mike Lewis, and Amjad Almahairi. 2023. Progressive prompts: Continual learning for language models. arXiv preprint arXiv:2301.12314.

Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. 2017. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 2001–2010.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim. 2017. Continual learning with deep generative replay. Advances in neural information processing systems, 30.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Jesse Vig and Yonatan Belinkov. 2019. Analyzing the structure of attention in a transformer language model. arXiv preprint arXiv:1906.04284.

Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. 2023a. Label words are anchors: An information flow perspective for understanding in-context learning. arXiv preprint arXiv:2305.14160.

Wenhui Wang, Hangbo Bao, Shaohan Huang, Li Dong, and Furu Wei. 2020a. Minilmv2: Multi-head self-attention relation distillation for compressing pretrained transformers. arXiv preprint arXiv:2012.15828.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020b. Minilm: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. Advances in Neural Information Processing Systems, 33:5776–5788.

Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuanjing Huang. 2023b. Orthogonal subspace learning for language model continual learning. arXiv preprint arXiv:2310.14152.

Xiao Wang, Yuansen Zhang, Tianze Chen, Songyang Gao, Senjie Jin, Xianjun Yang, Zhiheng Xi, Rui Zheng, Yicheng Zou, Tao Gui, et al. 2023c. Trace: A comprehensive benchmark for continual learning in large language models. arXiv preprint arXiv:2310.06762.

Yifan Wang, Yafei Liu, Chufan Shi, Haoling Li, Chen Chen, Haonan Lu, and Yujiu Yang. 2024. Inscl: A data-efficient continual learning paradigm for fine-tuning large language models with instructions. arXiv preprint arXiv:2403.11435.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. 2022a. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. arXiv preprint arXiv:2204.07705.

Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. 2022b. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 139–149.

Yang Yang, Zhiying Cui, Junjie Xu, Changhong Zhong, Wei-Shi Zheng, and Ruixuan Wang. 2023. Continual learning with bayesian model based on a fixed pretrained feature extractor. Visual Intelligence, 1(1):5.

Haiyan Zhao, Tianyi Zhou, Guodong Long, Jing Jiang, and Chengqi Zhang. 2023. Does continual learning equally forget all parameters? In International Conference on Machine Learning, pages 42280–42303. PMLR.

Weixiang Zhao, Shilong Wang, Yulin Hu, Yanyan Zhao, Bing Qin, Xuanyu Zhang, Qing Yang, Dongliang Xu, and Wanxiang Che. 2024. Dapt: A dual attention framework for parameter-efficient continual learning of large language models. arXiv preprint arXiv:2401.08295.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,

Zhuohan Li, Dacheng Li, Eric Xing, et al. 2024. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36.

## A Datasets

For the TRACE benchmark (Wang et al., 2023c), we conduct experiments on the reasoningaugmented datasets as such high-quality training data is more suitable for the LLM learning paradigm. The task order is consistent with the two orders provided by the benchmark, which are also displayed in Table 5. For evaluation on the changes in the general ability, we test the LLMs on the datasets (Hendrycks et al., 2020; Ghazal et al., 2013; Clark et al., 2020; Bisk et al., 2020; Clark et al., 2019; Cobbe et al., 2021) included in this benchmark.

For the SuperNI benchmark (Wang et al., 2022a), we choose four types of tasks and three dataset each for continual learning, containing a total of 12 traditional NLP tasks similar to Zhao et al., 2024. The two task orders can be found in Table 5.

## B Implementation Details

For methods not involving parameter-efficient tuning (PET) modules, we finetuning the LLMs on the task sequence in order1 for 5, 5, 5, 5, 5, 5, 10, 5 epochs, order2 for 10, 10, 10, 5, 5, 5, 5, 5 epochs, and order3 and order4 for 10 epochs each. For the compared baseline methods involving PET modules, the training epochs vary from 5 to 15 epochs for better performance. The hyperparameters of the compared baseline methods were kept the same as in the original repositories. If they did not perform well, we conducted additional searches for the optimal learning rate.

For all the replay-based methods, we randomly selected the indicated proportion of replay samples from the full training set and kept the replay samples utilized by each method consistent for fairness. For the replay-based distillation methods, the distillation signals, i.e. output logits and attention weights, of each old teacher model are saved in the memory buffer along with the original replay samples and loaded from the buffer during training on the new task. When replaying the old data, samples from the memory buffer and the current task are sampled in an evenly interleaved manner according to the ratio of their volumes.

Table 5: Task sequence of different task orders.
<table><tr><td rowspan=1 colspan=2>Order|    Benchmark</td><td rowspan=1 colspan=1>Task Sequence</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>TRACE benchmark</td><td rowspan=1 colspan=1> $\mathrm { C } \mathrm { - } \mathrm { S T A N C E }  \mathrm { F O M C }  \mathrm { M e e t i n g B a n k }  \mathrm { P y } 1 5 0 $  $\mathrm { S c i e n c e Q A }  \mathrm { N u m G L U E - c m }  \mathrm { N u m G L U E - d s }  2 0 \mathrm { M i n u t e n }$ </td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>TRACE benchmark</td><td rowspan=1 colspan=1> $\mathrm { N u m G L U E \mathrm { - } c m }  \mathrm { N u m G L U E \mathrm { - } d s }  \mathrm { F O M C }  2 0 \mathrm { M i n u t e n } $  $\mathrm { C } \mathrm { - } \mathrm { S T A N C E }  \mathrm { P y } 1 5 0  \mathrm { M e e t i n g B a n k }  \mathrm { S c i e n c e Q A }$ </td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>SuperNI benchmark</td><td rowspan=1 colspan=1> $\mathrm { t a s k } 1 5 7 2  \mathrm { t a s k } 3 6 3  \mathrm { t a s k } 1 2 9 0  \mathrm { t a s k } 1 8 1  \mathrm { t a s k } 0 0 2  \mathrm { t a s k } 1 5 1 0 $  $\mathrm { t a s k 0 7 3  t a s k 7 4 8  t a s k 5 1 1  t a s k 5 9 1  t a s k 1 9 5  t a s k 8 7 5 }$ </td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>SuperNI benchmark</td><td rowspan=1 colspan=1> $\mathrm { t a s k 7 4 8 \to t a s k 0 7 3 \to t a s k 1 5 7 2 \to t a s k 1 9 5 \to t a s k 5 9 1 \to t a s k 3 6 3 \to }$  $\mathrm { t a s k } 1 5 1 0  \mathrm { t a s k } 1 8 1  \mathrm { t a s k } 5 1 1  \mathrm { t a s k } 0 0 2  \mathrm { t a s k } 1 2 9 0  \mathrm { t a s k } 8 7 5$ </td></tr></table>