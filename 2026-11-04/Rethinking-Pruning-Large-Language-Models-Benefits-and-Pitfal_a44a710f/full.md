# Rethinking Pruning Large Language Models: Benefits and Pitfalls of Reconstruction Error Minimization

Sungbin Shin<sup>1\*</sup> Wonpyo Park<sup>2</sup> Jaeho Lee<sup>1,2,3</sup> Namhoon Lee<sup>1,2,3</sup>

<sup>1</sup>POSTECH <sup>2</sup>Google <sup>3</sup>Yonsei University

{ssbin4,jaeho.lee,namhoonlee}@postech.ac.kr wppark@google.com

## Abstract

This work suggests fundamentally rethinking the current practice of pruning large language models (LLMs). The way it is done is by divide and conquer: split the model into submodels, sequentially prune them, and reconstruct predictions of the dense counterparts on small calibration data one at a time; the final model is obtained simply by putting the resulting sparse submodels together. While this approach enables pruning under memory constraints, it generates high reconstruction errors. In this work, we first present an array of reconstruction techniques that can significantly reduce this error by more than 90%. Unwittingly, however, we discover that minimizing reconstruction error is not always ideal and can overfit the given calibration data, resulting in rather increased language perplexity and poor performance at downstream tasks. We find out that a strategy of self-generating calibration data can mitigate this trade-off between reconstruction and generalization, suggesting new directions in the presence of both benefits and pitfalls of reconstruction for pruning LLMs.<sup>1</sup>

## 1 Overview

Large language models (LLMs) have shown remarkable potential and achieved tremendous successes in various domains (Brown et al., 2020; Singhal et al., 2023; Roziere et al., 2023). Nevertheless, running them requires a significant amount of computations and memory, raising concerns about accessibility, sustainability, and scalability (Strubell et al., 2019; Bender et al., 2021). Neural network pruning holds great promise for mitigating this issue (LeCun et al., 1989; Hoefler et al., 2021). A complication here is that the standard approach is not quite feasible since it usually involves an exten-

![](images/c1fb04b18c6db095547375b3050da6c89e65f5c315361ac276ebb41c5365e13e.jpg)

(a) Effects of reconstruction techniques on reducing the error  
![](images/48163b76b22aa8277fcbb046c8353711a09ff9a35f08042fc24172beee5c49f6.jpg)  
(b) Effects of self-generated data on mitigating overfitting

Figure 1: (a) Reconstruction techniques significantly reduce the compounding errors and lead to a substantial reduction of error in the final block. Reconstruction O and X refer to the results with and without the proposed reconstruction techniques (BR, GP, CR) respectively. (b) Minimizing reconstruction error may not always be ideal since models can overfit calibration data (we show this in Section 3.2). Using our self-generated calibration data in the reconstruction process mitigates this issue quite effectively by decreasing test error, perplexity, and error rates for downstream tasks.

sive training process (and training data) which is challenging to carry out for LLMs.

To address this issue, LLM pruning is done post training. Specifically, it could be formulated as a reconstruction problem as follows:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { w , m } } & { \| f ( \bar { w } ; \mathcal { D } ) - f ( m \odot w ; \mathcal { D } ) \| _ { 2 } ^ { 2 } } \\ { \mathrm { s . t . ~ } } & { \| m \| _ { 0 } \leqslant k , } \end{array}\tag{1}
$$

i.e., given a pre-trained model w¯, the goal is to find a pruning mask m such that the resulting sparse model m  w reconstructs the predictions of the original dense model $f ( \bar { w } ; \cdot )$ on some calibration data $\mathcal { D } ;$ here, $\odot$ denotes element-wise product for vectorized representations, and m needs to satisfy a given sparsity constraint k. If the objective criterion—reconstruction error—is minimized to zero, then we achieve the perfect reconstruction and thereby pruning results.

While one could now avoid training LLMs from scratch with (1), it still requires as much memory as of the given LLM, hindering development under memory constraints. To circumvent this issue, many recent works take a divide-and-conquer approach: $i . e .$ , split the model into a sequence of smaller submodels, prune and reconstruct each submodel individually, and simply put all resulting sparse submodels together (Frantar and Alistarh, 2023; Sun et al., 2024; Zhang et al., 2024). Albeit fairly effective, we find that this can easily create critically high compounding errors. This is because solutions for each subproblem yield non-zero reconstruction errors.

In this work, we address the reconstruction error minimization for pruning LLMs with the following three major pillars. First, we focus on developing various engineering techniques to reduce this error. These are inspired to lessen the suboptimality of subsolutions by incorporating different levels of extension schemes. Second, we suggest that reducing this error is not necessarily favorable, however. Our extensive experimental results indicate that it is possibly due to overfitting, given limited calibration data and high problem complexity. Third, we present useful strategies to potentially mitigate the risk of reconstruction and improve generalization. This is based on what we call the self-generation of calibration data.

Briefly, this work investigates the benefits and pitfalls of the reconstruction error minimization scheme for pruning LLMs. To our best knowledge, this trade-off has not been explicitly identified or studied before, thereby suggesting rethinking the current practice. Our initial investigations may shed light on some potential future research directions. We summarize our main results in Figure 1.

## 2 Reconstruction Techniques

This section explains three optimization schemes we use to reduce reconstruction errors in this work.

Block-wise reconstruction (BR) The seminal work of Frantar and Alistarh (2023) proposes to reconstruct predictions layer-wise based on least squares. By removing non-linearity this approach yields a closed-form solution. However, we find that this can create a high reconstruction error since the system is highly underdetermined $( i . e .$ , there are much more parameters than calibration data). To reduce compounding errors, we first consider extending the unit of optimization target from a layer to a block of layers. Specifically, this means a block-wise reconstruction (BR) which can be formulated as follows:

$$
\operatorname* { m i n } _ { w _ { 1 } , \dots , w _ { B } } \sum _ { i = 1 } ^ { B } \| g _ { i } ( \bar { w } _ { i } ; x _ { i } ) - g _ { i } ( \bar { m } _ { i } \odot w _ { i } ; x _ { i } ) \| _ { 2 } ^ { 2 }\tag{2}
$$

where $g _ { i }$ refers to the i-th block of layers $( e . g .$ , a Transformer block) in which we have the optimization variables $w _ { i }$ , and $x _ { i }$ denotes the inputs to the i-th block which originally come from calibration data; here, the pruning mask $\bar { m }$ is fixed assuming that it is already obtained from an arbitrary pruning method. $I . e .$ , the goal is to update variables in each block to minimize the extended reconstruction errors. We solve this problem iteratively using the standard gradient-based method. Notably a similar approach is also proposed in the concurrent work of Guo et al. (2024), and we find in our experiments that BR is extremely effective in reducing the reconstruction errors in Section 3.1. We illustrate the idea of BR in Figure 2.

Global propagation (GP) While the general divide-and-conquer principle is quite functional, we identify a potential issue therein: by sequentially solving the subproblem, it is constantly fitting practically suboptimal solutions obtained from the previous step (which become gradually worse), as with $x _ { i } = g _ { i - 1 } ( \bar { m } _ { i - 1 } \odot w _ { i - 1 } ; x _ { i - 1 } )$ . We realize that this is another source of compounding errors, and thus, suggest that when we locally reconstruct a model, at least we use global propagation (GP) from the original dense model as input to the target reconstruction; i.e., $x _ { i } = g _ { i - 1 } \big ( \bar { w } _ { i - 1 } ; x _ { i - 1 } \big )$ . We show that GP improves the reconstruction results quite significantly in Section 3.1. We further note that a similar principle is found in various applications including low-rank approximation (Zhang et al., 2015), channel pruning (He et al., 2017), and quantization (Nagel et al., 2020; Hubara et al., 2021). We illustrate the idea of GP in Figure 2.

Cross-block reconstruction (CR) Another way we consider to further reduce reconstruction errors is to extend the reconstruction unit from a block to multiple blocks and stitch the solutions in between by connecting via the adjacent blocks. Specifically, this means that now $g$ in (2) becomes a composite of multiple blocks, say $h ,$ and we ensure h overlaps; more precisely, $h _ { i } = g _ { i } \circ g _ { i - 1 }$ and $h _ { i + 1 } = g _ { i + 1 } \circ g _ { i }$ for two blocks, and so on for all blocks. This way, namely cross-block reconstruction or CR (Ding et al., 2023), we can potentially bridge between subsolutions by taking into account some interaction between adjacent blocks, and hence, reduce the compounding errors. We illustrate the idea of CR in Figure 2.

![](images/e9fcdacde64863e7168a9abc66c22d1ffda42184c0f5ae85a686b38c2deba926.jpg)  
Figure 2: An illustration of reconstruction techniques for pruning large language models. Here, we want the sparse model $f ( m \odot w ; \cdot )$ to reconstruct the prediction of the dense model on some calibration data . LR, BR, GP, and CR each correspond to layer-wise reconstruction, block-wise reconstruction, global propagation, and cross-block reconstruction. Here, solid and dashed arrows each represent the inputs coming from sparse and dense models.

To elaborate further, the difference between BR and CR is that while BR is about updating parameters within a block (thus it is not concerned with how to combine subsolutions), CR takes a step further and is about stitching the subsolutions; i.e., CR updates parameters within two adjacent blocks, and when it comes to reconstructing the next block, it includes the overlapping block so that it has the effect of “stitching”. This method is found to be quite effective for reducing the error, however, we find that this method can often lead to overfitting. We discuss this in detail in Section 3.2.

## 3 Experiments

## 3.1 Reconstruction error

We first evaluate the effectiveness of the suggested techniques in reducing the reconstruction error. Here, we focus on pruning LLaMA-7B (Touvron et al., 2023) and OPT-125M (Zhang et al., 2022) to unstructured 50% sparsity with three pruning methods: SparseGPT (Frantar and Alistarh, 2023), Wanda (Sun et al., 2024), and Magnitude (Han et al., 2015). For each pruning method, we examine four reconstruction strategies: layer-wise reconstruction (LR), block-wise reconstruction (BR), block-wise reconstruction with global propagation (BR+GP), and cross-block reconstruction with global propagation (BR+GP+CR). Following the convention, we use 256 calibration data randomly sampled from C4 (Raffel et al., 2020) each containing 1024 tokens. We run the Adam optimizer for 10 epochs (see Appendix A for details). The results are presented in Figure 3.

![](images/e93cbb2627ab1f66b1dcffcb967a4d784df9633772f2234c8688ac53f592425c.jpg)  
(a) SparseGPT

![](images/132205d9ec7916c189cbe71a4fd765ff8972275295d57b230cf322f12e5d8ca4.jpg)  
(b) Wanda  
Figure 3: Results of reconstruction techniques for LLaMA-7B. They constantly reduce the compounding errors, achieving a significant decrease at the final block ( 90%). We find this trend is consistent across different settings. See Figures 5 and 6 of Appendix B for more results.

We can see that all the reconstruction techniques reduce the compounding errors quite significantly, yielding a substantial reduction at the final block. Specifically, BR first reduces the final error by at least 50% across all pruning methods compared to LR, BR+GP further reduces the error by at least 60% compared to BR, and finally, BR+GP+CR reduces the error by at least 20% compared to BR+GP. Consequently, we observe that the error is reduced from 87% to 94% with BR+GP+CR compared to the baseline LR.

## 3.2 Generalization performance

We now evaluate the generalization performances of the reconstruction results. Specifically, we measure the perplexity of the pruned model on three different datasets: raw-Wikitext2 (Merity et al., 2017), PTB (Marcus et al., 1994), and validation data of C4. We also measure its zero-shot task performance in accuracy on seven downstream tasks: BoolQ (Clark et al., 2019), RTE (Wang et al., 2019),

<table><tr><td rowspan="2">Pruner</td><td rowspan="2">Reconstruction</td><td rowspan="2">Error (normalized)</td><td colspan="4">Perplexity PTB</td><td colspan="8">Zero-shot accuracy</td></tr><tr><td>Wiki</td><td></td><td>C4</td><td>Mean</td><td>BoolQ</td><td>RTE</td><td>HellaSwag</td><td>WinoGrande</td><td>ARC-e</td><td>ARC-c</td><td>OpenbookQA</td><td>Mean</td></tr><tr><td>Dense</td><td></td><td></td><td>5.68</td><td>10.12</td><td>7.34</td><td>7.71</td><td>75.11</td><td>66.43</td><td>56.96</td><td>70.00</td><td>75.29</td><td>41.81</td><td>34.40</td><td>60.00</td></tr><tr><td rowspan="4">SparseGPT</td><td>LR</td><td>2.86</td><td>7.24</td><td>12.61</td><td>9.17</td><td>9.67</td><td>73.36</td><td>58.12</td><td>51.86</td><td>68.90</td><td>70.62</td><td>36.95</td><td>28.60</td><td>55.49</td></tr><tr><td>BR</td><td>1.24</td><td>6.82</td><td>11.69</td><td>8.66</td><td>9.06</td><td>71.71</td><td>54.51</td><td>52.54</td><td>68.27</td><td>71.68</td><td>36.18</td><td>28.40</td><td>54.76</td></tr><tr><td>BR+GP</td><td>0.48</td><td>6.72</td><td>11.32</td><td>8.55</td><td>8.86</td><td>71.22</td><td>53.79</td><td>53.57</td><td>68.90</td><td>71.76</td><td>37.54</td><td>27.80</td><td>54.94</td></tr><tr><td>BR+GP+CR</td><td>0.37</td><td>6.83</td><td>11.41</td><td>8.71</td><td>8.99</td><td>72.91</td><td>55.60</td><td>53.24</td><td>68.51</td><td>71.21</td><td>36.26</td><td>27.80</td><td>55.07</td></tr><tr><td rowspan="4">Wanda</td><td>LR</td><td>3.56</td><td>7.25</td><td>12.77</td><td>9.28</td><td>9.77</td><td>71.28</td><td>55.23</td><td>52.04</td><td>66.46</td><td>69.36</td><td>36.52</td><td>28.80</td><td>54.24</td></tr><tr><td>BR</td><td>1.33</td><td>6.82</td><td>11.54</td><td>8.70</td><td>9.02</td><td>72.02</td><td>57.04</td><td>52.45</td><td>67.09</td><td>72.18</td><td>36.60</td><td>28.60</td><td>55.14</td></tr><tr><td>BR+GP</td><td>0.51</td><td>6.68</td><td>11.25</td><td>8.56</td><td>8.83</td><td>72.66</td><td>60.29</td><td>53.25</td><td>68.43</td><td>71.46</td><td>37.63</td><td>29.80</td><td>56.22</td></tr><tr><td>BR+GP+CR</td><td>0.38</td><td>6.79</td><td>12.01</td><td>8.72</td><td>9.18</td><td>73.00</td><td>59.93</td><td>53.18</td><td>68.27</td><td>71.13</td><td>37.29</td><td>28.80</td><td>55.94</td></tr><tr><td rowspan="4">Magnitude</td><td>LR</td><td>8.08</td><td>17.29</td><td>49.67</td><td>23.78</td><td>30.25</td><td>54.65</td><td>54.15</td><td>45.47</td><td>59.43</td><td>58.75</td><td>33.45</td><td>22.60</td><td>46.93</td></tr><tr><td>BR</td><td>2.37</td><td>7.83</td><td>15.73</td><td>9.66</td><td>11.07</td><td>68.90</td><td>49.82</td><td>47.85</td><td>66.38</td><td>70.29</td><td>36.77</td><td>27.00</td><td>52.43</td></tr><tr><td>BR+GP</td><td>0.63</td><td>6.88</td><td>11.77</td><td>8.77</td><td>9.14</td><td>71.65</td><td>52.35</td><td>53.00</td><td>68.19</td><td>70.75</td><td>37.63</td><td>29.00</td><td>54.65</td></tr><tr><td>BR+GP+CR</td><td>0.46</td><td>6.98</td><td>11.96</td><td>8.85</td><td>9.27</td><td>72.23</td><td>48.74</td><td>53.20</td><td>67.09</td><td>70.54</td><td>36.95</td><td>28.20</td><td>53.85</td></tr></table>

Table 1: Effects of different reconstruction techniques on error, perplexity, and zero-shot accuracy for LLaMA-7B. Bold and underline refer to best in general and task-specific. See Table 3 of Appendix B for the OPT-125M results.

<table><tr><td rowspan="2">Pruner</td><td rowspan="2">CR</td><td colspan="2">Error (normalized)</td></tr><tr><td>Calib</td><td>Test</td></tr><tr><td rowspan="2">SparseGPT</td><td>X</td><td>0.006</td><td>0.0083</td></tr><tr><td>0</td><td>0.004</td><td>0.0078</td></tr><tr><td rowspan="2">Wanda</td><td>X</td><td>0.006</td><td>0.0080</td></tr><tr><td>0</td><td>0.004</td><td>0.0076</td></tr><tr><td rowspan="2">Magnitude</td><td>X</td><td>0.008</td><td>0.0109</td></tr><tr><td>0</td><td>0.005</td><td>0.0102</td></tr></table>

(a) OPT-125M

<table><tr><td rowspan="2">Pruner</td><td rowspan="2">CR</td><td colspan="2">Error (normalized)</td></tr><tr><td>Calib</td><td>Test</td></tr><tr><td rowspan="2">SparseGPT</td><td>X</td><td>0.48</td><td>2.30</td></tr><tr><td>0</td><td>0.37</td><td>2.53</td></tr><tr><td rowspan="2">Wanda</td><td>X</td><td>0.51</td><td>2.23</td></tr><tr><td>0</td><td>0.38</td><td>2.48</td></tr><tr><td rowspan="2">Magnitude</td><td>X</td><td>0.63</td><td>2.42</td></tr><tr><td>0</td><td>0.46</td><td>2.55</td></tr></table>

(b) LLaMA-7B  
Table 2: Reconstruction errors of OPT-125M and LLaMA-7B on test data (raw-Wikitext2) as well as calibration data. Overfitting by CR is only observed for the larger LLaMA-7B model. We find that larger models in general are more susceptible to overfitting. See Tables 3 and 4 of Appendix B for more results.

HellaSwag (Zellers et al., 2019), Winogrande (Sakaguchi et al., 2020), ARC Easy and Challenge (Clark et al., 2018), and OpenbookQA (Mihaylov et al., 2018). The results are presented in Table 1.

At first, we find that the perplexity effectively decreases with BR and GP; the value reduces across all test cases including different models, pruning methods, and datasets. Unexpectedly, however, the perplexity rather increases when we add CR despite the reduced reconstruction errors. We also observe a similar trend in zero-shot performance for Wanda and Magnitude pruning, with mean accuracy increasing by a large margin with BR and GP but decreasing with CR. Interestingly, for SparseGPT, reconstruction techniques do not generally help zero-shot performance. We hypothesize that it is because SparseGPT already conducts fairly heavy optimization compared to other methods, and applying further reconstruction on particular calibration data may not help improve zero-shot performance since it is more sensitive to distribution shift. Furthermore, we find that such overfiting tends to occur more for LLaMA-7B than OPT-125M (see Table 2). This is possibly due to model size; i.e., given the same amount of (limited) calibration data, over-optimizing can make large models more likely to overfit and lead to poor generalization.

We can summarize our findings are as follows.

• BR and GP are found to be very effective in reducing perplexity in all cases; on the other hand, CR often leads to overfitting, especially for large models.

• This holds true for zero-shot performance as well, with only exception of SparseGPT, for which BR and GP do not help much in improving zero-shot performance; this is possibly due to the fact that SparseGPT already conducted fairly heavy optimization of remaining weights. It is also possible that adapting to downstream task is more prone to overfitting. This certainly requires more investigations.

In short, we can attempt to say without much loss of generality that “BR and GP can generally help for pruning LLMs in terms of reducing perplexity”.

## 4 Further Exploration

We have seen that reconstruction techniques are useful but they can lead to undesirable overfitting. Here we explore potential ways to alleviate this risk. In particular, we identify that the calibration data is highly limited in two aspects: it is too little (compared to optimization variables)<sup>2</sup> and does not represent the training data (as it is arbitrarily given); the former is related to the general representationgeneralization complexity trade-off, and the latter is about whether the reconstruction can mimic the behavior of the original model.

![](images/9886a5e9dd056e0c1b4fa80461de5bb803ddd14531fb79dda2a221b000783398.jpg)  
(a) Test error

![](images/5d3a1a9009223ada9c449c60a468503e7d9a3f3dbf0c4c87208cd804d5e51bfa.jpg)  
(b) Perplexity  
Figure 4: Effects of self-generated calibration data on (a) reconstruction error for test data (raw-Wikitext2) and (b) perplexity for LLaMA-7B; they both improve with more self-generation. See Figure 7 of Appendix B for more results.

To this end, we reflect on the fact that what we are dealing with is a generative (language) model, meaning that we can create calibration data that is potentially much bigger in size and closer to the original distribution. We find that this selfgeneration technique has recently been proposed in other contexts (Meng et al., 2022; Ye et al., 2022; Liu et al., 2023; Li et al., 2024), and thus, follow the process therein to produce high-quality text data. Using that, we perform reconstruction again, and the results are reported in Figure 4. We observe that making use of more self-generated calibration data (without unfairly violating the given setting) reduces both test error and perplexity, mitigating overfitting quite effectively.

## 5 Conclusion

In this work, we take a close look at the current practice of minimizing reconstruction errors for pruning LLMs. We first find that with various reconstruction techniques, one can reduce the error quite significantly and improve quality of pruning results on both language perplexity and zero-shot accuracy. Nevertheless, it turns out that decreasing error as it is now is not always desirable since it may cause overfitting calibration data. We present initial results that this issue can be potentially mitigated by self-generating calibration data. There are many remaining possibilities, and we believe our findings suggest opportunities for future work.

## 6 Limitations

There remain several limitations in our experiments and we plan to address these in future work. First, our main experiments are limited to LLaMA-7B and OPT-125M. We intend to scale up our experiments to much larger models of up to 70B parameters and different architectures including Mixtral (Jiang et al., 2024) or Gemma (Team et al., 2024). Next, reconstruction techniques BR, GP, and CR require additional memory compared to LR, although they still use much less memory compared to model-level reconstruction of solving (1) (see Appendix B for the details). We plan to introduce parameter-efficient optimization (Hu et al., 2022) to alleviate this increased memory burden.

Although the self-generation of calibration data effectively mitigates overfitting, it requires more computation for reconstruction. Finally, we find that some portions of the generated texts are far from plain English texts and thus may not serve as good calibration data (see Table 5 of Appendix C for the examples). In this regard, we believe that reducing the number of these irrelevant examples and generating only a few number of high-quality texts can be a potential way to improve performance and increase efficiency.

## Acknowledgements

This work was partly supported by the Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT) (RS-2019-II191906, Artificial Intelligence Graduate School Program (POSTECH); RS-2022-II220959/No.2022-0- 00959, (part2) Few-Shot learning of Causal Inference in Vision and Language for Decision Making; RS-2024-00338140, Development of Learning and Utilization Technology to Reflect Sustainability of Generative Language Models and Up-to-Dateness over Time) and the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) (RS-2023-00210466, RS-2023- 00265444, RS2023-0021371). Sungbin Shin was supported by Kwanjeong Educational Foundation Scholarship.

## References

Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the dangers of stochastic parrots: Can language models be too big? FAccT.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. NeurIPS.

Christopher Clark, Kenton Lee, Ming-Wei Chang,

Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. Boolq: Exploring the surprising difficulty of natural yes/no questions. NAACL.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Xin Ding, Xiaoyu Liu, Yun Zhang, Zhijun Tu, Wei Li, Jie Hu, Hanting Chen, Yehui Tang, Zhiwei Xiong, Baoqun Yin, et al. 2023. Cbq: Cross-block quantization for large language models. arXiv preprint arXiv:2312.07950.

Elias Frantar and Dan Alistarh. 2023. SparseGPT: Massive language models can be accurately pruned in one-shot. ICML.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2023. A framework for few-shot language model evaluation.

Song Guo, Fan Wu, Lei Zhang, Xiawu Zheng, Shengchuan Zhang, Fei Chao, Yiyu Shi, and Rongrong Ji. 2024. Ebft: Effective and blockwise fine-tuning for sparse llms. arXiv preprint arXiv:2402.12419.

Song Han, Jeff Pool, John Tran, and William Dally. 2015. Learning both weights and connections for efficient neural network. NeurIPS.

Yihui He, Xiangyu Zhang, and Jian Sun. 2017. Channel pruning for accelerating very deep neural networks. ICCV.

Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. 2021. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. JMLR.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. ICLR.

Itay Hubara, Yury Nahshan, Yair Hanani, Ron Banner, and Daniel Soudry. 2021. Accurate post training quantization with small calibration sets. ICML.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088.

Yann LeCun, John Denker, and Sara Solla. 1989. Optimal brain damage. NeurIPS.

Liang Li, Qingyuan Li, Bo Zhang, and Xiangxiang Chu. 2024. Norm tweaking: High-performance lowbit quantization of large language models. AAAI.

Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. 2023. Llm-qat: Data-free quantization aware training for large language models. arXiv preprint arXiv:2305.17888.

Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon, and Tie-Yan Liu. 2022. Biogpt: generative pre-trained transformer for biomedical text generation and mining. Briefings in bioinformatics.

Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D Lee, Danqi Chen, and Sanjeev Arora. 2023. Fine-tuning language models with just forward passes. NeurIPS.

Mitch Marcus, Grace Kim, Mary Ann Marcinkiewicz, Robert MacIntyre, Ann Bies, Mark Ferguson, Karen Katz, and Britta Schasberger. 1994. The penn treebank: Annotating predicate argument structure. HLT.

Yu Meng, Jiaxin Huang, Yu Zhang, and Jiawei Han. 2022. Generating training data with language models: Towards zero-shot language understanding. NeurIPS.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017. Pointer sentinel mixture models. ICLR.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. EMNLP.

Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort. 2020. Up or down? adaptive rounding for post-training quantization. ICML.

Satya Sai Srinath Namburi, Makesh Sreedhar, Srinath Srinivasan, and Frederic Sala. 2023. The cost of compression: Investigating the impact of compression on parametric knowledge in language models. EMNLP 2023 Findings.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR.

Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. 2023. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. Winogrande: An adversarial winograd schema challenge at scale. AAAI.

Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. 2023. Large language models encode clinical knowledge. Nature.

Emma Strubell, Ananya Ganesh, and Andrew McCallum. 2019. Energy and policy considerations for deep learning in nlp. ACL.

Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. 2024. A simple and effective pruning approach for large language models. ICLR.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2019. Glue: A multi-task benchmark and analysis platform for natural language understanding. ICLR.

Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. 2023. Bloomberggpt: A large language model for finance. arXiv preprint arXiv:2303.17564.

Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang. 2023. Fingpt: Open-source financial large language models. arXiv preprint arXiv:2306.06031.

Jiacheng Ye, Jiahui Gao, Qintong Li, Hang Xu, Jiangtao Feng, Zhiyong Wu, Tao Yu, and Lingpeng Kong. 2022. Zerogen: Efficient zero-shot learning via dataset generation. EMNLP.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? ACL.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

Xiangyu Zhang, Jianhua Zou, Kaiming He, and Jian Sun. 2015. Accelerating very deep convolutional networks for classification and detection. TPAMI.

Yuxin Zhang, Lirui Zhao, Mingbao Lin, Yunyun Sun, Yiwu Yao, Xingjia Han, Jared Tanner, Shiwei Liu, and Rongrong Ji. 2024. Dynamic sparse no training: Training-free fine-tuning for sparse llms. ICLR.

## A Experimental Details

Experiment configurations We run our experiments with a single A100 GPU having 80GB of memory. For BR and CR, we run the Adam optimizer for 10 epochs with a batch size of 8, without weight decay or gradient clipping. The learning rate is set to 0.0002 and decays linearly following Guo et al. (2024). For evaluating the performance on downstream tasks, we use the EleutherAIevalharness framework (Gao et al., 2023).

Calculation of normalized reconstruction error The reconstruction error for i-th block is calculated as $\begin{array} { r } { \frac { 1 } { N H T } \lVert g _ { i } ( \bar { w } _ { i } ; \bar { x } _ { i } ) - g _ { i } ( m _ { i } \odot w _ { i } ; x _ { i } ) \rVert _ { 2 } ^ { 2 } } \end{array}$ where $N , H , T$ each represent the number of calibration data, hidden dimension, and the token length. ${ \bar { x } } _ { i } , x _ { i }$ represent the inputs coming from dense and sparse blocks respectively.

Licenses and uses of models and datasets LLaMA (Touvron et al., 2023) and OPT (Zhang et al., 2022) are released under non-commercial bespoke licenses. raw-Wikitext2 (Merity et al., 2017), PTB (Marcus et al., 1994), and C4 (Raffel et al., 2020) are released under CC BY-SA 4.0, LDC user agreement, and ODC-By. BoolQ (Clark et al., 2019), RTE (Wang et al., 2019), HellaSwag (Zellers et al., 2019), Winogrande (Sakaguchi et al., 2020), ARC (Clark et al., 2018), and OpeenbookQA (Mihaylov et al., 2018) are released under CC BY-SA 3.0, Apache 2.0, MIT License, Apache 2.0, CC BY-SA 4.0, and Apache 2.0 respectively. We confirm that these models and datasets are used for their intended use and the data does not contain personal information. EleutherAIevalharness framework is released under the MIT License.

## B Additional Results

More results on the reconstruction techniques Effects of reconstruction techniques on reducing the error for LLaMA-7B and OPT-125M are presented in Figures 5 and 6 respectively. It is clearly observed that different reconstruction techniques significantly reduce the error for all cases.

Effects of reconstruction techniques on performance for OPT-125M are presented in Table 3.

<table><tr><td rowspan="2">Pruner</td><td rowspan="2">Reconstruction</td><td rowspan="2">Error (normalized)</td><td colspan="4">Perplexity</td><td colspan="8">Zero-shot accuracy</td></tr><tr><td>Wiki</td><td>PTB</td><td>C4</td><td>Mean</td><td>BoolQ</td><td>RTE</td><td>HellaSwag</td><td>WinoGrande</td><td>ARC-e</td><td>ARC-c</td><td>OpenbookQA</td><td>Mean</td></tr><tr><td>Dense</td><td>一</td><td></td><td>27.66</td><td>38.99</td><td>26.56</td><td>31.07</td><td>55.44</td><td>50.18</td><td>29.19</td><td>50.20</td><td>43.60</td><td>19.03</td><td>16.6</td><td>37.75</td></tr><tr><td rowspan="4">SparseGPT</td><td>LR</td><td>0.019</td><td>36.35</td><td>54.93</td><td>33.12</td><td>41.47</td><td>61.31</td><td>48.01</td><td>28.29</td><td>53.28</td><td>40.19</td><td>19.28</td><td>15.60</td><td>38.00</td></tr><tr><td>BR</td><td>0.008</td><td>31.94</td><td>45.75</td><td>29.91</td><td>35.87</td><td>60.49</td><td>47.65</td><td>28.44</td><td>51.38</td><td>42.17</td><td>19.88</td><td>14.60</td><td>37.80</td></tr><tr><td>BR+GP</td><td>0.006</td><td>31.57</td><td>45.52</td><td>29.81</td><td>35.63</td><td>60.18</td><td>45.13</td><td>28.53</td><td>52.17</td><td>42.63</td><td>19.62</td><td>14.80</td><td>37.58</td></tr><tr><td>BR+GP+CR</td><td>0.004</td><td>30.86</td><td>44.61</td><td>29.45</td><td>34.97</td><td>60.31</td><td>46.21</td><td>28.64</td><td>51.07</td><td>42.63</td><td>19.71</td><td>15.80</td><td>37.77</td></tr><tr><td rowspan="4">Wanda</td><td>LR</td><td>0.032</td><td>39.00</td><td>56.27</td><td>34.62</td><td>43.30</td><td>62.05</td><td>48.38</td><td>28.31</td><td>52.01</td><td>39.56</td><td>19.62</td><td>14.20</td><td>37.73</td></tr><tr><td>BR</td><td>0.008</td><td>31.55</td><td>46.17</td><td>29.89</td><td>35.87</td><td>60.24</td><td>47.65</td><td>28.34</td><td>50.20</td><td>41.50</td><td>19.54</td><td>15.00</td><td>37.50</td></tr><tr><td>BR+GP</td><td>0.006</td><td>31.18</td><td>45.47</td><td>29.67</td><td>35.44</td><td>59.85</td><td>48.01</td><td>28.66</td><td>51.54</td><td>41.71</td><td>19.28</td><td>16.20</td><td>37.89</td></tr><tr><td>BR+GP+CR</td><td>0.004</td><td>30.59</td><td>44.80</td><td>29.33</td><td>34.91</td><td>58.81</td><td>45.85</td><td>28.68</td><td>50.99</td><td>42.34</td><td>19.03</td><td>15.00</td><td>37.24</td></tr><tr><td rowspan="4">Magnitude</td><td>LR</td><td>0.121</td><td>193.36</td><td>276.15</td><td>141.01</td><td>203.5</td><td>60.55</td><td>53.43</td><td>27.32</td><td>52.57</td><td>33.04</td><td>19.97</td><td>14.20</td><td>37.30</td></tr><tr><td>BR</td><td>0.010</td><td>36.06</td><td>49.15</td><td>31.63</td><td>38.95</td><td>58.99</td><td>48.38</td><td>28.35</td><td>51.22</td><td>41.20</td><td>19.88</td><td>15.80</td><td>37.69</td></tr><tr><td>BR+GP</td><td>0.008</td><td>35.56</td><td>48.17</td><td>31.75</td><td>38.50</td><td>58.20</td><td>49.46</td><td>28.44</td><td>51.54</td><td>42.26</td><td>19.88</td><td>15.20</td><td>37.85</td></tr><tr><td>BR+GP+CR</td><td>0.005</td><td>33.76</td><td>46.84</td><td>30.88</td><td>37.16</td><td>57.28</td><td>45.49</td><td>28.53</td><td>51.93</td><td>42.00</td><td>19.97</td><td>15.60</td><td>37.26</td></tr></table>

Table 3: Effects of different reconstruction techniques on error, perplexity, and zero-shot accuracy for OPT-125M. Bold and underline refer to best in general and task-specific.

<table><tr><td rowspan="2">Pruner</td><td rowspan="2">CR</td><td colspan="4">Error (normalized)</td></tr><tr><td>Calib</td><td>Test (Wiki)</td><td>Test (PTB)</td><td>Tets (C4)</td></tr><tr><td rowspan="2">SparseGPT</td><td>X</td><td>0.006</td><td>0.0083</td><td>0.009</td><td>0.0065</td></tr><tr><td>0</td><td>0.004</td><td>0.0078</td><td>0.0083</td><td>0.0061</td></tr><tr><td rowspan="2">Wanda</td><td>X</td><td>0.006</td><td>0.008</td><td>0.0088</td><td>0.0061</td></tr><tr><td>0</td><td>0.004</td><td>0.0076</td><td>0.0082</td><td>0.0058</td></tr><tr><td rowspan="2">Magnitude</td><td>X</td><td>0.008</td><td>0.0109</td><td>0.0115</td><td>0.0125</td></tr><tr><td>0</td><td>0.005</td><td>0.0102</td><td>0.0111</td><td>0.0099</td></tr></table>

(a) OPT-125M

<table><tr><td rowspan="2">Pruner</td><td rowspan="2">CR</td><td colspan="4">Error (normalized)</td></tr><tr><td>Calib</td><td>Test (Wiki)</td><td>Test (PTB)</td><td>Tets (C4)</td></tr><tr><td rowspan="2">SparseGPT</td><td>X</td><td>0.48</td><td>2.30</td><td>2.29</td><td>1.99</td></tr><tr><td>0</td><td>0.37</td><td>2.53</td><td>2.60</td><td>2.31</td></tr><tr><td rowspan="2">Wanda</td><td>X</td><td>0.51</td><td>2.23</td><td>2.29</td><td>1.98</td></tr><tr><td>0</td><td>0.38</td><td>2.48</td><td>2.86</td><td>2.31</td></tr><tr><td rowspan="2">Magnitude</td><td>X</td><td>0.63</td><td>2.42</td><td>2.72</td><td>2.21</td></tr><tr><td>0</td><td>0.46</td><td>2.55</td><td>3.03</td><td>2.40</td></tr></table>

(b) LLaMA-7B

Table 4: Reconstruction errors of OPT-125M and LLaMA-7B on test data (raw-Wikitext2) as well as calibration data. Overfitting by CR is only observed for the larger LLaMA-7B model.
<table><tr><td>Example number</td><td>Text</td></tr><tr><td>1</td><td>Americas, and the U.K., while 18 other countries have legalized the medical use of cannabis. The latest announcement is a win for Canadians ...</td></tr><tr><td>2</td><td>apprehension of the inevitability of death? And, therefore, how could such a person come to believe ..</td></tr><tr><td>3</td><td>&#x27;#&#x27; + this.currentID + .&quot;n };n\n return {n next: next,\n previous: previous,n}...</td></tr><tr><td>4</td><td>Picker.setSelected(false);n \n actionPhrasesTableModel.fireTableDataChanged();n ...</td></tr></table>

Table 5: Examples of self-generated data.

![](images/b5b725fddadd1f057e643432246f90ce009a9192af25a6e0ee4cc11dc4c37f5f.jpg)  
(a) SparseGPT

![](images/30024c96daa397dbb0079cfdc4ac75dee450e0b0842d141ab67c8139964e14b2.jpg)  
(b) Wanda

![](images/c57bad6783a3a4abb9bc7a3e32b7bea63088707d9bfb19a683a6b2c09edad184.jpg)  
(c) Magnitude

![](images/2aac4b6950a0a1e3e7b9db484d508ac6566cdb54ae8c4a0b5294b68d8a2f2317.jpg)  
(a) SparseGPT

![](images/b0261f3277a6935750aceb9e27e7a537818575508fc88616b47c020a4cc869f1.jpg)  
(b) Wanda

![](images/432f32e23df12f79ab9801fdd1a7794fdd0a71ba8d608af4f1cda2cdfee3254f.jpg)  
(c) Magnitude  
Figure 5: Results of reconstruction techniques for LLaMA-7B. They constantly reduce the compounding errors, achieving a significant decrease at the final block (87% 94%).  
Figure 6: Results of reconstruction techniques for OPT-125M. They constantly reduce the compounding errors, achieving a significant decrease at the final block (79% 96%).

Different techniques effectively improve the performance on perplexity and downstream tasks, with the exception of overfitting for CR on downstream tasks.

More results on self-generated data Reconstruction error on calibration data and test data for OPT-125M and LLaMA-7B are presented in Table 4. Decreased error for calibration data leads to decreased error for test data for OPT-125M, but leads to increased test error for LLaMA-7B.

Effects of self-generated calibration data are presented in Figure 7. In most cases, more number of self-generated data leads to decreased test error and perplexity.

Memory consumption of reconstruction techniques Solving (1) directly can be memoryintensive, thus many recent work suggest divideand-conquer such as LR and BR. In the work of Frantar and Alistarh (2023), the authors show that for the 175B parameter OPT model it requires at least five A100 GPUs of 80GB, whereas by using LR it reduces down to a single A100 GPU of 80GB. In our experiments, for Llama-7B, both LR and BR+GP+CR can all be done on a commodity 3090 GPU of 24GB memory; it requires more than 100GB to perform full fine-tuning of LLaMA-7B (Malladi et al., 2023). In theory, optimizing more parameters can incur more memory footprints, and thus, in the order of $\mathrm { L R } = \mathrm { G P } < \mathrm { B R } < \mathrm { C R }$ , there will be more memory usage.

![](images/d4e1134ea0d559661ede355a44774ae7de11327cf4062a56d471a40732324160.jpg)

![](images/b6dbf77099b0defd2e0e315924c178831f4b636d95a4c2de01171882e9c9dcf3.jpg)  
(a) SparseGPT

![](images/9a047abf558300751537910ad764a47b8fc51b660d1c5767a74d1ddb2120f5ae.jpg)

![](images/89a55a8097dcd140727981e05278db5778fb3c4597152a21b782a7d519454de4.jpg)  
(b) Wanda

![](images/d0fda82d61cad5e3930f0e200967760643d05c424cde482e99c4aa1944ee8400.jpg)

![](images/b2800ab0d056965de28923252e6f6ec588275a39a102c744adeb9abeca4d404a.jpg)  
(c) Magnitude

Figure 7: Effects of self-generated calibration data on reconstruction error for test data and perplexity for LLaMA-7B; they both improve with more self-generation.
<table><tr><td></td><td>LR</td><td>BR</td><td>BR+GP</td><td>BR+GP+CR</td><td>Full fine-tuning</td></tr><tr><td>peak memory (GB)</td><td>3.9</td><td>5.7</td><td>5.7</td><td>10.6</td><td>&gt; 100</td></tr></table>

Table 6: Peak GPU memory for LLaMA-7B and sparseGPT. Compared to LR, reconstruction techniques incur additional GPU memory but it is quite marginal compared to fine-tuning the full model. The results are obtained with the batch size of 8 and gradient accumulation. For full fine-tuning, the results are from Malladi et al. (2023).

The exact amount depends on the specific model. To provide solid evidence, we ran profiling peak GPU memory for LLaMA-7B with the batch size of 8 (see Table 6 for the results). Compared to LR, reconstruction techniques surely incur additional GPU memory, however, (i) it is quite marginal compared to fine-tuning the full model, and (ii) it could be reduced further by introducing memory reduction techniques in practice such as CPU offloading and gradient checkpointing.

Pruning attention vs. feed-forward We also investigated the effects of only pruning attention vs. feed-forward blocks for different reconstruction techniques. Here, we conducted experiments for OPT-125m and SparseGPT by pruning either attention or feed-forward blocks to 50% sparsity and measuring the perplexity on raw-Wikitext2. The results are provided in Table 7. We first observe that pruning both attention and feed-forward yields the largest performance drop. Also, we find that pruning only the attention block leads to worse performance compared to pruning only the feedforward block, which is consistent with the findings in the previous work (Namburi et al., 2023). Interestingly, we find that reconstruction techniques can be more effective for cases with poor performance; i.e., in the order of pruning all blocks > pruning attention > pruning feed-forward, BR, GP, CR reconstruction techniques yield more reduction in perplexity (which is good by itself).

## C Details on Self-generation of Calibration Data

We generate additional calibration data from the original dense model. Here, we sample 10240 number of English texts each containing 2048 tokens. Specifically, we first randomly choose the initial token and generate four subsequent tokens by deterministically selecting top-1 predictions, similar to Liu et al. (2023). Here, we resample the tokens if the generated texts are not detected as English. Then, we stochastically generate the remaining tokens until the <EOS> token is produced or the sequence length exceeds 2048. Finally, the additional calibration data can be obtained by sampling a subset of generated texts and randomly selecting the intermediate 1024 tokens for each text.

Examples of self-generated texts are presented in Table 5. Examples 1 and 2 are plain English texts and can serve as good calibration data. However, we observe that programming codes such as examples 3 and 4 are often generated, which might not serve as good calibration data for improving the perplexity for English texts or accuracy for downstream tasks which are not related to code generation. In this regard, we believe that generating only a few number of high-quality texts can lead to improved performance while reducing computational costs.

<table><tr><td>Pruning block</td><td>LR</td><td>BR</td><td>BR+GP</td><td>BR+GP+CR</td></tr><tr><td>Attention</td><td>32.82</td><td>30.15</td><td>29.97</td><td>29.64</td></tr><tr><td>Feed-forward</td><td>30.69</td><td>29.23</td><td>28.89</td><td>28.73</td></tr><tr><td>All</td><td>36.35</td><td>31.94</td><td>31.57</td><td>30.86</td></tr></table>

Table 7: Effects of pruning block for different reconstruction techniques. Here, we prune either attention or feedforward block to 50% sparsity and measure the perplexity on raw-Wikitext2. Pruning only the attention block leads to worse performance compared to pruning only the feed-forward block. The results are for OPT-125m with sparseGPT.

Here, the generated data do not contain personal information or offensive content.