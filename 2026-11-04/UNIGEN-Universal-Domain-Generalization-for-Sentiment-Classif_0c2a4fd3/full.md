# UNIGEN: Universal Domain Generalization for Sentiment Classification via Zero-shot Dataset Generation

Juhwan Choi<sup>1</sup>, Yeonghwa Kim<sup>1</sup>, Seunguk Yu<sup>1</sup>, Jungmin Yun<sup>1</sup> and YoungBin Kim<sup>1,2</sup>

<sup>1</sup>Department of Artificial Intelligence, Chung-Ang University

<sup>2</sup>Graduate School of Advanced Imaging Sciences, Multimedia and Film, Chung-Ang University {gold5230, movie112, bokju128, cocoro357, ybkim85}@cau.ac.kr

## Abstract

Although pre-trained language models have exhibited great flexibility and versatility with prompt-based few-shot learning, they suffer from the extensive parameter size and limited applicability for inference. Recent studies have suggested that PLMs be used as dataset generators and a tiny task-specific model be trained to achieve efficient inference. However, their applicability to various domains is limited because they tend to generate domain-specific datasets. In this work, we propose a novel approach to universal domain generalization that generates a dataset regardless of the target domain. This allows for generalization of the tiny task model to any domain that shares the label space, thus enhancing the real-world applicability of the dataset generation paradigm. Our experiments indicate that the proposed method accomplishes generalizability across various domains while using a parameter set that is orders of magnitude smaller than PLMs.

## 1 Introduction

As the size and performance of pre-trained language models (PLMs) increase, generation of new data by using PLMs has attracted the attention of many researchers (Anaby-Tavor et al., 2020; Kumar et al., 2020; Yoo et al., 2021). While scholars have applied this method to solve data augmentation problems, in recent studies, they have started to explore zero-shot dataset generation settings (Meng et al., 2022; Ye et al., 2022a, 2023). This novel approach first generates training data from a PLM based on a specific prompt and trains a tiny task model (TAM) by using the dataset generated in the first step. This strategy facilitates effective distillation of the knowledge pertaining to the desired task from the PLM and helps train the TAM without the need for guidance from human-annotated data, thereby enabling zero-shot learning and achieving low-cost inference compared to the case in which PLMs are used directly for inference.

However, the approaches proposed thus far have relied on domain-specific prompts, for example, “The movie review in positive sentiment is:.” Because the data generated using this prompt are related only to the domain of movie reviews, the TAM trained on these data has limited generalization ability across other domains. This is the primary limitation of the TAM-based approach compared to prompt-based zero-shot learning that directly uses PLMs (PROMPTING), which allows for generalizability across diverse domains. This restricts the real-world applicability of the TAMbased approach because it requires many separately trained TAMs for various domains. Moreover, as the costs of dataset generation and TAM training increase, the cost-efficiency of the TAM-based approach may decrease. Hence, a novel strategy is desired to effectively distill the domain generalizability of large-scale PLMs into TAMs while maintaining the cost-efficiency of TAMs.

Meanwhile, the existing approaches to domain generalization often require multiple source domains (Wang et al., 2022; Zhou et al., 2022). This requirement limits the application of these methods because it is difficult to gather the required data from multiple domains. Although the concept of single-domain generalization, which achieves domain generalizability by using data from only one source domain, has been proposed in recent computer vision studies, such a concept is yet to be explored for natural language processing (Qiao et al., 2020; Wang et al., 2021).

In this study, we propose a simple but effective method called UNIGEN to solve the problem of domain generalizability between PLMs and TAMs. Table 1 presents a comparison between UNIGEN and the existing approaches. UNIGEN first focuses on generating a domain-invariant training dataset that is not restricted to specific domains. This allows TAMs to achieve domain generalizability without the need for multiple source domains.

<table><tr><td></td><td>Learning without Human-annotated Data</td><td>Domain Generalizability</td><td>Light Inference</td><td>Handling Noise of Generated Data</td></tr><tr><td>Task-specific Fine-tuning</td><td>X</td><td>x</td><td>√</td><td></td></tr><tr><td>Previous Domain Generalization (Tan et al., 2022)</td><td>x</td><td>J</td><td>¬</td><td></td></tr><tr><td>PROMPTING</td><td>√</td><td>√</td><td>X</td><td></td></tr><tr><td>ZEROGEN (Ye et al., 2022a)</td><td>√</td><td>X</td><td>√</td><td>x</td></tr><tr><td>PROGEN &amp; SUNGEN (Ye et al., 2022b; Gao et al., 2023)</td><td>¬</td><td>X</td><td>J</td><td>L</td></tr><tr><td>UNIGEN (Ours)</td><td>1</td><td>J</td><td>J</td><td>1</td></tr></table>

Table 1: Comparison between previous approaches and UNIGEN.

We extend domain generalization strategies based on supervised contrastive learning (Khosla et al., 2020), as suggested in a previous work (Tan et al., 2022). Moreover, we employ additional tactics such as momentum encoder (He et al., 2020) and denoised memory bank, in addition to the method suggested by the previous work (Tan et al., 2022). Furthermore, because the PLM-based dataset generation method can generate noisy data (Ye et al., 2022b; Gao et al., 2023; Zou et al., 2024), we propose a pseudo-relabeling-based additional denoising method.

Our experiments show that UNIGEN achieves generalizability across various domains and outperforms PROMPTING. This indicates that smaller TAMs can be used universally in various domains, thereby reducing the costs of PROMPTING, dataset generation, and TAM training.

Our contributions are summarized as follows:

• We propose UNIGEN, a universal domain generalization strategy by using zero-shot dataset generation.

• We develop a pseudo-relabeling-based method for denoising the generated data.

• Our extensive experiment reveals that the TAM trained using UNIGEN has domain generalizability, and it can outperform the PLM with considerably fewer parameters.

## 2 Related Work

## 2.1 Dataset Generation for Efficient Zero-shot Learning

The evolution of PLMs in terms of parameter size and performance has facilitated zero-shot learning through the use of well-designed prompts (Radford et al., 2019; Brown et al., 2020). However, it is expensive to directly deploy these massive models into daily services because the process requires numerous rounds of inference. Dataset generation mitigates this problem through the generation of training datasets by using PLMs and training a small TAM on the generated datasets (Meng et al., 2022; Ye et al., 2022a). This TAM is deployed in downstream tasks to reduce inference costs and improve performance compared to PROMPTING.

However, mere generation, that is, ZEROGEN, yields noisy data, such as incorrectly labeled data or irrelevant data (Ye et al., 2022b; Gao et al., 2023). PROGEN (Ye et al., 2022b) proposed to alleviate this problem by adding examples based on in-context feedback. Meanwhile, SUNGEN (Gao et al., 2023) proposed to re-weigh the generated samples during training using noise-robust loss. Additionally, a concurrent study suggested to leverage multiple PLMs as data generator and assign weight to generated samples in single training procedure, different from SUNGEN (Zou et al., 2024).

In this work, we propose a novel approach to extend dataset generation for universal domain generalization that is not restricted to specific training source data, as well as a pseudo-relabeling-based method to denoise the generated dataset.

## 2.2 Methods for Learning from Noisy Data

Researchers have explored various methods to mitigate noisy label data, which is wrongly labeled from ground-truth labels (Song et al., 2023). A relevant study in this field defined two types of noisy labels and evaluated the effectiveness of various methods with respect to BERT model (Agro and Aldarmaki, 2023). Another study proposed to leverage GPT-4 to provide the guidance to noisy labeled data (Wang et al., 2023). However, they suffer from the necessity of massive LLMs that demand cost. Moreover, these studies primarily focused on the human-crafted noisy label, rather than the noisy label of data generated by PLMs.

In this work, we suggest a straightforward method to handle noisy data based on pseudorelabeling, particularly designed for synthetic data.

## 2.3 Domain Generalization for Text Classification

Domain generalization aims to improve the generalization ability in the target domain by employing source data from multiple domains to mitigate the domain shift problem (Wang et al., 2022; Zhou et al., 2022). This domain shift can be observed in natural language processing tasks, such as restaurant reviews and reviews of consumer electronics. For example, long waiting time in a restaurant’s reviews can represent a negative sentiment about the restaurant, while long battery life in a laptop’s reviews can represent a positive sentiment of the laptop (Tan et al., 2022).

Previous studies to alleviate domain shift in text classification have focused primarily on domain adaptation setting, for which training data are needed in the target domain (Chen and Cardie, 2018; Ye et al., 2020; Guo et al., 2020). Recently, researchers have explored the application of domain generalization to natural language processing tasks. A representative study applied supervised contrastive learning (Khosla et al., 2020) to achieve domain generalizability in text classification tasks (Tan et al., 2022).

In this work, we extend an existing method for domain generalization to generate datasets, including the adoption of momentum encoder (He et al., 2020), in addition to proposing a denoising memory bank to further enhance its effectiveness and handle noisy data.

## 3 Method

## 3.1 Preliminaries

## 3.1.1 Dataset Generation

First, we briefly explain the concept and notation of the preliminary dataset generation method, that is, ZEROGEN (Ye et al., 2022a). ZEROGEN aims to create a synthetic dataset $\boldsymbol { S _ { s y n } } = ( \mathcal { X } _ { s y n } , \mathcal { Y } _ { s y n } )$ by using a large-scale PLM $\mathcal { P }$ and task-specific prompt $\mathcal { T } _ { t a s k }$ . For a text classification problem, a desired pseudo-label $y _ { s y n }$ is first sampled from the uniform distribution across every class. Next, $y _ { s y n }$ is passed to the prompt $\mathcal { T } _ { t a s k }$ to construct $\mathcal { T } _ { t a s k } ( y _ { s y n } )$ , that is, the final prompt for $\mathcal { P }$ . Thereafter, synthesized input data $x _ { \mathit { s y n } }$ are generated using $\mathbf { x } _ { s y n } \sim \mathcal { P } ( \cdot | \mathcal { T } _ { t a s k } ( y _ { s y n } ) )$ . Finally, $\boldsymbol { S _ { s y n } }$ is composed of these pairs of generated $\left( \mathbf { x } _ { s y n } , y _ { s y n } \right)$ . Notably, the domain of $\boldsymbol { S _ { s y n } }$ is defined by the structure of $\mathcal { T } _ { t a s k }$ . For example, a $\mathcal { T } _ { b o o k } = \mathrm { \mathrm { ~ ‰ ~ } }$ book review in $< y >$ sentiment is: ” would harness $\mathcal { P }$ to generate ${ \bf x } _ { s y n }$ about book reviews. The TAM is trained on the generated $\boldsymbol { \mathcal { S } } _ { s y n }$ and deployed for inference instead of directly using PLMs with PROMPTING.

## 3.1.2 Supervised Contrastive Learning

Supervised contrastive learning (Khosla et $\mathrm { { a l . } }$ 2020) is a variant of contrastive learning (Chen et al., 2020) that utilizes label values. It allows for explicit pulling of the representation of positive (i.e., same class) samples to the anchor representation while pushing negative representations away from the anchor. Studies have reported that this characteristic is valuable for domain generalization, which aims to group the representations of different domains (Kim et al., 2021; Tan et al., 2022). The supervised contrastive loss is expressed as follows:

$$
\begin{array} { r } { \mathcal { L } _ { S C L } = - \sum _ { \mathbf { z } _ { i } \in B } \frac { 1 } { | P ( i ) | } l o g \frac { e x p ( \mathbf { z } _ { i } \cdot \mathbf { z } _ { p } / \tau _ { S C L } ) } { \sum _ { \mathbf { z } _ { a } \in A ( i ) } e x p ( \mathbf { z } _ { i } \cdot \mathbf { z } _ { a } / \tau _ { S C L } ) } } \end{array}\tag{1}
$$

where z denotes an encoded representation, and $\mathbf { z _ { i } }$ is an anchor. $P ( i ) \equiv \mathbf { z } _ { j } \in B , y _ { j } = y _ { i }$ is the set of positive samples for each anchor $i ,$ and $\mathbf { z } _ { p }$ symbolizes a positive representation from $P ( i )$ $A ( i ) \equiv \mathbf { z } _ { j } \in B , j \neq i$ refers to the union of every sample, except the anchor, including positive and negative samples. ${ \bf z } _ { a }$ indicates each representation from $A ( i )$ . B denotes a mini-batch, and $\tau _ { S C L }$ is the temperature of supervised contrastive learning.

Although supervised contrastive learning is effective, the introduction of a memory bank and momentum encoder may augment the advantages of the method (Wu et al., 2018; He et al., 2020). The potency of contrastive learning is often influenced by the size of B because a larger B may introduce more diverse negative samples. However, increasing the size of B can introduce concerns related to memory consumption. A memory bank is a mechanism that fulfills this demand for a greater number of negative samples by storing previously processed samples within the dictionary M. Memory-efficient contrastive learning can be achieved using this dictionary with the current batch, that is, establishing a union of B and M instead of solely using B to construct $P ( i )$ and $A ( i )$ . Momentum encoder is another technique that smooths the process of updating the representations stored in M. The momentum encoder $\theta _ { k }$ is trained by momentum update, $\theta _ { k } \gets m \theta _ { k } + ( 1 - m ) \theta _ { q } ,$ where m is a coefficient for momentum update, and $\theta _ { q }$ is a normal encoder that is updated through backpropagation. By using the momentum encoder, the representations in M are processed by $\theta _ { k }$

![](images/9c7a9258c16b49e40200710733e1a7d9b63c1e03f24a10e0dc34f211d38621ed.jpg)  
Figure 1: Overall framework for generating a dataset and training a TAM using UNIGEN.

## 3.2 UNIGEN

To build a TAM that can be applied universally to various target domains, UNIGEN generates a domain-invariant dataset by using the universal prompt $\mathcal { T } _ { u n i } .$ , instead of task-specific $\mathcal { T } _ { t a s k }$ . Consider “The text in $< y >$ sentiment $i s . ^ { \prime \prime }$ as an example of $\mathcal { T } _ { u n i }$ . Next, the final input prompt for $\mathcal { P }$ is constructed as $\mathcal { T } _ { u n i } ( y _ { s y n } )$ . The synthesized input data $x _ { s y n }$ are generated by following the same process as that of ZEROGEN:

$$
\mathbf { x } _ { s y n } \sim \mathcal { P } ( \cdot | \mathcal { T } _ { u n i } ( y _ { s y n } ) )\tag{2}
$$

This configuration of prompt design allows us to generate a sentence with the desired label without being restricted to any specific domain. Therefore, it steers $\mathcal { P }$ to generate various sentences within a predefined label space. This domain-invariant data generation allows the TAM trained using UNIGEN to learn the domain-invariant characteristics of the desired label space, thereby resulting in generalizability across the domains that share the label space. Supervised contrastive loss is applied along with conventional cross entropy loss to aid this process. The training loss is defined as follows:

$$
\mathcal { L } = \mathcal { L } _ { C E } + \alpha \mathcal { L } _ { S C L }\tag{3}
$$

where α is a hyperparameter that balances the ratio between the two losses.

## 3.3 Handling Noisy Data through Relabeling

However, the application of $\mathcal { T } _ { u n i }$ instead of $\mathcal { T } _ { t a s k }$ might lead to the generation of noisy sentences, which was noted as a drawback of ZEROGEN. This is because $\mathcal { T } _ { u n i }$ does not have a specific topic to guide the generation process. Furthermore, a previously developed approach to effectively mitigate this problem is applied in the training phase but not the generation phase. Therefore, there is scope to improve the quality of $\boldsymbol { S _ { s y n } }$ (Gao et al., 2023). This problem highlights the necessity to use a denoising scheme in the generation procedure. In the present work, we propose a pseudo-relabeling-based denoising process for dataset generation. In a previous study, the approach of relabeling the generated data and assigning soft labels for data augmentation was proposed (Yoo et al., 2021). Herein, we first perform pseudo-relabeling by using :

$$
\ell ( y _ { i } | \mathbf { x } _ { s y n } ) = \mathcal { P } ( \mathcal { M } ( y _ { i } ) | \mathcal { T } _ { u n i } ( \mathbf { x } _ { s y n } ) )\tag{4}
$$

where $\mathcal { M } ( \cdot )$ denotes a verbalizer that transforms each label $y _ { i }$ into a word. We share $\mathcal { T } _ { u n i }$ between this process and the generation process. These logit values yielded by are normalized using the softmax function with the temperature τ<sub>RE</sub> :

$$
\hat { y } _ { i } = p ( y _ { i } | \mathbf { x } _ { s y n } ) = \frac { e x p ( \ell ( y _ { i } | \mathbf { x } _ { s y n } ) / \tau _ { R E } ) } { \sum _ { j } e x p ( \ell ( y _ { j } | \mathbf { x } _ { s y n } ) / \tau _ { R E } ) }\tag{5}
$$

Finally, we assign $\hat { y } _ { i }$ instead of the predefined $y _ { s y n }$ to the generated $\mathbf { x } _ { s y n }$ . This provides two distinct advantages: (1) because $\hat { y } _ { i }$ is a soft label rather than a hard label, it contains richer information about ${ \bf x } _ { s y n } .$ , such as the degree of the desired label, which enhances the effectiveness of training (Szegedy et al., 2016). (2) Because it relabels the generated $\mathbf { x } _ { s y n }$ and replaces the predefined $y _ { s y n }$ , it can solve the noisy label issue, which results in the generation of $\mathbf { x } _ { s y n }$ that does not correspond to the designated $y _ { s y n } ,$ as pointed out in previous work (Gao et al., 2023). We validate the effectiveness of this relabeling strategy in the ablation study described in Section 4.5.1.

Furthermore, we discard ${ \bf x } _ { s y n }$ if its pseudo-label $\hat { y } _ { i }$ does not exceed the threshold $T _ { R E }$ to enhance the quality of $\boldsymbol { \mathcal { S } } _ { s y n }$ . This guarantees that only those data that have the desired degree of each label are maintained.

## 3.4 Denoising Memory Bank

In addition to the relabeling strategy, we propose a denoising memory bank mechanism to further alleviate the issue of noisy data. We first use SUNGEN (Gao et al., 2023) that learns weights of each training sample w for loss function within the training process to assign small weights to noisy data by employing a noise-robust loss function. We aim to ensure that the memory bank M contains clean samples, rather than noisy samples. We utilize the weights w learned from the noise-robust loss function for this purpose. In the process of updating M, we store only those samples whose weights are larger than the threshold $T _ { M B }$ . This organization of the memory bank ensures the exclusion of noisy samples from the comparison, resulting in higherquality negative and positive samples (Robinson et al., 2021).

## 4 Experiment

## 4.1 Experimental Setup

In this section, we briefly explain the experimental setup used herein to validate the effectiveness of UNIGEN. We employ seven different sentiment classification datasets in our main experiment. Among them, IMDB (Maas et al., 2011), SST-2 (Socher et al., 2013), and Rotten Tomatoes (Pang and Lee, 2005) are datasets comprising movie reviews. Meanwhile, the Amazon (McAuley and Leskovec, 2013) dataset consists of customer reviews of various products, and the Yelp (Zhang et al., 2015) dataset is composed of restaurant reviews. CR (Ding et al., 2008) is another customer review dataset focusing on consumer electronics. Lastly, Tweet (Rosenthal et al., 2017) is composed of messages from Twitter. This configuration allows us to evaluate the ability of UNIGEN, which can be applied to various domains without providing any prior information or domain-specific training. Following the previous study, we adapted long short-term memory (LSTM) (Hochreiter and Schmidhuber, 1997) and DistilBERT (Sanh et al., 2019), and we included RoBERTa (Liu et al., 2019) as our TAM. We compared our approach to ZE-ROGEN and SUNGEN, as well as to PROMPTING using GPT2-XL (Radford et al., 2019), to ensure a fair comparison. We did not include other larger PLMs in the experiments because the previous work discovered that larger PLMs did not offer performance gains (Ye et al., 2022a). We report the average of the performance results obtained across five different random seeds.

## 4.2 Comparison with Task-specific TAMs

Table 2 presents a comparison between the experimental results of UNIGEN and PROMPTING and task-specific TAMs trained by ZEROGEN and SUN-GEN. The comparison results suggest that UNI-GEN can generalize across various domains using a single model without requiring any prior information about the test domain. Nonetheless, UNI-GEN underperformed compared to the task-specific baselines in each domain. However, the primary benefit of UNIGEN lies in its unique domain generalizability while using orders-of-magnitude fewer parameters than PLMs. Additionally, its training procedure is more efficient than those of other TAM training strategies. As can be inferred from Table 3, SUNGEN generates and synthesizes 1,000k data for each task domain. This means that 5,000k data would be required for our experiment, which involves five different domains, in addition to individual denoising processes for finding the best weights of the samples in each of these domains. By contrast, UNIGEN is not limited by such restrictions and requires only a single data generation and denoising process, as well as a single training process. This is extremely beneficial when a novel test domain is introduced, where ZEROGEN and SUN-GEN necessitate a separate procedure for the new domain, but UNIGEN directly reuses the already trained TAM.

<table><tr><td>Model Test Domain</td><td>#Param</td><td>Training Domain</td><td>Setup</td><td>SST-2</td><td>IMDB Movie</td><td>Rotten</td><td>Amazon Products</td><td>Yelp Restaurant</td><td>CR Electronics</td><td>Tweet Tweet</td><td>Average</td></tr><tr><td>GPT2-XL</td><td>1.5B</td><td>-</td><td>PROMPTING</td><td>82.15</td><td>70.26</td><td>77.56</td><td>79.06</td><td>78.04</td><td>80.30</td><td>80.38</td><td>78.25</td></tr><tr><td rowspan="8">LSTM</td><td rowspan="8">7M</td><td rowspan="2">Movie</td><td>ZEROGEN</td><td>75.11</td><td>66.39 69.97</td><td>69.85 73.76</td><td>67.24 72.15</td><td>70.25 73.21</td><td>69.32</td><td>63.43</td><td>68.80</td></tr><tr><td>SUNGEN</td><td>78.79</td><td></td><td></td><td></td><td></td><td>70.39</td><td>66.84</td><td>72.16</td></tr><tr><td rowspan="2">Products</td><td>ZEROGEN</td><td>64.26</td><td>61.82</td><td>60.13</td><td>70.32</td><td>67.78</td><td>69.46</td><td>62.29</td><td>65.15</td></tr><tr><td>SUNGEN</td><td>67.83</td><td>63.87</td><td>63.46</td><td>74.43</td><td>73.71</td><td>73.35</td><td>63.51</td><td>68.59</td></tr><tr><td rowspan="2">Restaurant</td><td>ZEROGEN</td><td>67.41</td><td>63.01</td><td>62.74</td><td>68.73</td><td>75.51</td><td>69.23</td><td>66.35</td><td>63.28</td></tr><tr><td>SUNGEN</td><td>69.15</td><td>66.62</td><td>64.56</td><td>73.22</td><td>79.56</td><td>70.12</td><td>67.43</td><td>70.09</td></tr><tr><td rowspan="2">Electronics</td><td>ZEROGEN</td><td>64.69</td><td>59.13</td><td>60.20</td><td>66.34</td><td>67.72</td><td>72.50</td><td>60.25</td><td>64.40</td></tr><tr><td>SUNGEN</td><td>68.38</td><td>64.33</td><td>63.25</td><td>72.61</td><td>73.01</td><td>76.18</td><td>66.78</td><td>69.22</td></tr><tr><td rowspan="2"></td><td rowspan="2">Tweet</td><td>ZEROGEN</td><td>61.84</td><td>60.17</td><td>59.43</td><td>64.13</td><td>63.68</td><td>65.02</td><td>74.10</td><td>64.05</td></tr><tr><td>SUNGEN</td><td>66.57</td><td>63.96</td><td>64.21</td><td>69.36</td><td>71.68</td><td>72.57</td><td>81.29</td><td>69.95</td></tr><tr><td rowspan="9">DistilBERT 66M</td><td rowspan="2"></td><td>UNIGEN</td><td>64.15</td><td>60.02</td><td>60.51</td><td>63.82</td><td>63.20</td><td>69.61</td><td>70.32</td><td>64.52</td></tr><tr><td>ZEROGEN</td><td>80.06</td><td>69.13</td><td>74.73</td><td>73.02</td><td>72.77</td><td>73.59</td><td>74.83</td><td>74.02</td></tr><tr><td rowspan="2">Movie</td><td>SUNGEN</td><td>82.43</td><td>70.59</td><td>76.37</td><td>74.13</td><td>73.56</td><td>75.14</td><td>75.96</td><td>75.45</td></tr><tr><td>ZEROGEN</td><td>71.04</td><td>64.99</td><td>65.57</td><td>74.54</td><td>71.89</td><td>74.57</td><td>71.93</td><td>70.65</td></tr><tr><td rowspan="2">Products</td><td>SUNGEN</td><td>72.35</td><td>65.95</td><td>66.84</td><td>76.92</td><td>74.98</td><td>75.84</td><td>73.01</td><td>72.27</td></tr><tr><td>ZEROGEN</td><td>77.32</td><td>65.47</td><td>68.86</td><td>74.01</td><td>77.94</td><td>74.89</td><td>73.74</td><td>73.18</td></tr><tr><td rowspan="2">Restaurant</td><td>SUNGEN</td><td>78.93</td><td>67.12</td><td>69.92</td><td>74.93</td><td>80.67</td><td>76.06</td><td>75.28</td><td>74.70</td></tr><tr><td>ZEROGEN</td><td>73.77</td><td>66.14</td><td>66.78</td><td>72.38</td><td>73.21</td><td>78.82</td><td>74.58</td><td>72.24</td></tr><tr><td rowspan="2">Electronics Tweet</td><td>SUNGEN</td><td>74.49</td><td>67.19</td><td>68.29</td><td>73.49</td><td>75.34</td><td>80.49</td><td>75.37</td><td>73.52</td></tr><tr><td>ZEROGEN</td><td>73.98</td><td>66.58</td><td>67.43</td><td>72.88</td><td>71.86</td><td>75.68</td><td>80.86</td><td>72.75</td></tr><tr><td rowspan="9"></td><td rowspan="2">=</td><td>SUNGEN</td><td>75.12</td><td>67.53</td><td>69.06</td><td>73.64</td><td>72.73</td><td>78.17</td><td>82.46</td><td>74.10</td></tr><tr><td>UNIGEN</td><td>77.67</td><td>67.81</td><td>73.16</td><td>75.06</td><td>74.81</td><td>79.86</td><td>81.41</td><td>75.68</td></tr><tr><td rowspan="2">Movie</td><td>ZEROGEN</td><td>84.38</td><td>73.03</td><td>78.38</td><td>77.38</td><td>76.83</td><td>77.36</td><td>77.94</td><td>77.90</td></tr><tr><td>SUNGEN</td><td>85.24</td><td>74.09</td><td>79.19</td><td>78.56</td><td>77.61</td><td>78.21</td><td>79.72</td><td>78.95</td></tr><tr><td rowspan="2">Products</td><td>ZEROGEN</td><td>79.14</td><td>71.16</td><td>70.92</td><td>79.94</td><td>75.79</td><td>76.35</td><td>80.17</td><td>76.21</td></tr><tr><td>SUNGEN</td><td>81.51</td><td>71.28</td><td>72.67</td><td>81.50</td><td>77.76</td><td>78.55</td><td>81.94</td><td>77.87</td></tr><tr><td rowspan="2">Restaurant</td><td>ZEROGEN</td><td>82.87</td><td>70.71</td><td>69.58</td><td>78.61</td><td>81.47</td><td>76.43</td><td>79.51</td><td>77.03</td></tr><tr><td>SUNGEN</td><td>83.65</td><td>71.40</td><td>71.05</td><td>79.42</td><td>82.72</td><td>77.60</td><td>80.92</td><td>78.11</td></tr><tr><td rowspan="2">Electronics</td><td>ZEROGEN SUNGEN</td><td>76.82</td><td>69.42</td><td>67.89</td><td>75.02</td><td>76.53</td><td>81.24</td><td>76.51</td><td>74.78</td></tr><tr><td></td><td>77.51</td><td>71.23</td><td>68.77</td><td>76.91</td><td>78.33</td><td>83.49</td><td>79.03</td><td>76.47</td></tr><tr><td rowspan="2">Tweet</td><td>ZEROGEN</td><td>78.43</td><td>68.31</td><td>72.25</td><td>78.09</td><td>74.61</td><td>79.08</td><td>82.96</td><td></td><td>76.25</td></tr><tr><td></td><td>SUNGEN</td><td>82.19</td><td>70.62 73.21</td><td></td><td>79.84</td><td>76.27</td><td>81.46</td><td>83.25</td><td>78.12</td></tr><tr><td></td><td></td><td>UNIGEN</td><td>84.86</td><td>72.24 78.82</td><td></td><td>80.79</td><td>79.15</td><td>86.37</td><td>87.89</td><td>81.45</td></tr></table>

Table 2: Experimental results of UNIGEN and baselines across various datasets and training domains. The performance of TAM, which is superior to that of PROMPTING, is underlined, and the best result in each test dataset within the group for each TAM is presented in boldface.

<table><tr><td></td><td>Amount of generated data I</td><td>Number of trained TAMs</td></tr><tr><td>ZEROGEN</td><td>1,000k</td><td>5</td></tr><tr><td>SUNGEN</td><td>5,000k</td><td>5</td></tr><tr><td>UNIGEN</td><td>1,000k</td><td>1</td></tr></table>

Table 3: Amount of data generated for training TAMs by using each method, and number of trained TAMs per method.

Notably, the performance of the LSTM-based TAM trained using UNIGEN was significantly lower than that of ZEROGEN and SUNGEN. This implies that while a small-sized TAM can be trained effectively for a single, specific domain, but suffers from generalizing to a universal domain that requires a broad understanding of generated data, as evidenced by detailed study in Appendix E.

Accordingly, the performance of the TAM trained using UNIGEN improves significantly as the model size increases. For instance, the DistilBERT-based TAM trained using UNIGEN exhibited the best average performance against each task-specific baseline. This is particularly remarkable as it outperformed the SUNGEN baseline in the movie domain, which has three in-domain datasets, giving it an inherent advantage for average performance. Moreover, the RoBERTa-based TAM trained using UNIGEN not only yielded the best average performance against these baselines but also outperformed PROMPTING in every domain. This result indicates that it can surpass the zero-shot performance of its PLM counterpart (e.g., GPT2-XL) while using less than 10% of the number of parameters and securing the domain generalizability of the PLM, extending the achievement of the previous study that leveraged small TAMs in single domain (Ye et al., 2022a).

<table><tr><td>RoBERTa</td><td>DVD</td><td>Electronics</td><td>Kitchen</td><td>Book</td><td>Average</td></tr><tr><td>PROMPTING w/ GPT2-XL</td><td>77.73</td><td>78.71</td><td>81.64</td><td>80.27</td><td>79.59</td></tr><tr><td>UNIGEN</td><td>78.14</td><td>80.68</td><td>82.31</td><td>80.93</td><td>80.52</td></tr><tr><td>SUPERVISED (Tan et al., 2022)</td><td>91.40</td><td>95.10</td><td>95.05</td><td>93.25</td><td>93.70</td></tr></table>

Table 4: Experiments conducted using multi-domain review dataset. The experimental result of SUPERVISED was reported in a previous study (Tan et al., 2022) with the memory bank size of 64.

## 4.3 Comparison with Supervised Domain Generalization Method

Next, we analyzed the performance of UNIGEN against that of a domain generalization method that uses human-annotated data (Tan et al., 2022). For this purpose, we used a multi-domain review dataset comprising four domains: DVD, books, kitchen and housewares, and consumer electronics (Blitzer et al., 2007). Following the previous study, we split the dataset into 1,600 training data and 400 testing data for each domain. Table 4 presents the comparison results. These results suggest that UNIGEN can be applied to various domains, and its performance is superior to that of its PLM counterpart. Notably, the SUPERVISED baseline relies on three source domains with human-annotated data to generalize to a target domain, while UNIGEN is based on zero-shot dataset generation and does not require any human-annotated data, which greatly improves its real-world applicability.

## 4.4 Domain Generalizability of UNIGEN

To intuitively examine the domain generalizability of UNIGEN, we plotted the T-SNE (Van der Maaten and Hinton, 2008) visualization of the features interpreted by the RoBERTa-based TAM trained using UNIGEN. Figure 2 depicts the visualization results. These results suggest that the single TAM classified the given data from every domain without explicit training or prior information about the domains, thus demonstrating the unique efficiency of UNIGEN.

Table 5 presents examples of the sentences generated using UNIGEN. These examples showcase that UNIGEN can generate domain-invariant sentences with the designated labels. By training TAMs on these data, it is possible to distill the domain generalizability of PLMs into TAMs.

![](images/780f0ec818a83b16ced93bf46f61c1483af3809aa531bcf7e27ba8b68f152dc0.jpg)  
Figure 2: T-SNE visualization of the encoded representation of the RoBERTa model trained using UNIGEN. The model was trained only on the data generated using UNIGEN, which is shown in gray color. We used the test set of the multi-domain review dataset.

## 4.5 Ablation Study

This section describes the ablation studies conducted to offer rationales for the engineering choices made in this study. We used the DistilBERT-based TAM for these experiments.

## 4.5.1 Effectiveness of Relabeling Strategy

First, we performed an ablation study to validate the effectiveness of the relabeling strategy discussed in Section 3.3. We compared the basic approach that uses soft labels to the two other options. The first option utilizes the pseudo-relabeling process, but it assigns hard labels instead of soft labels. In other words, it only reflects the decision emanating from the PLM, not the probability. The second option completely excludes the relabeling process. While this option would generate the dataset faster than the other options, it might generate text with noisy labels, as already discussed in previous works (Ye et al., 2022a,b; Gao et al., 2023).

The experimental results are presented in the second and third rows of Table 6. They suggest that the use of soft labels offers practical benefits in terms of performance. This finding is consistent with that of a previous study in which the strength of soft labels was demonstrated (Yoo et al., 2021; Fang et al., 2024). Therefore, according to the results of this ablation study, relabeling the generated data with the assignment of soft labels is effective for mitigating the issue of noisy labels.

<table><tr><td>Positive Examples</td><td>Labels</td></tr><tr><td>You are a person who is hardworking, honest, and reliable. You have a good sense of humor, and you love being in charge.</td><td>[0.19, 0.81]</td></tr><tr><td>You are beautiful, you are powerful, you are amazing.</td><td>[0.29, 0.71]</td></tr><tr><td>In a city full of great ideas and creativity, I&#x27;ve met a few people who have done things you wouldn&#x27;t believe.</td><td>[0.26, 0.74]</td></tr><tr><td>The American Dream is alive in this great city. As a new generation of American heroes begins to realize their own American Dream.</td><td>[0.24, 0.76]</td></tr><tr><td>Negative Examples</td><td>Labels</td></tr><tr><td>No one likes it. Nobody wants it. It is a disgrace.</td><td>[0.7,0.3]</td></tr><tr><td>The company is no longer in business and has ceased operations.</td><td>[0.71, 0.29]</td></tr><tr><td>Please don&#x27;t use this feature to communicate with customers</td><td>[0.74, 0.26]</td></tr><tr><td>Do not buy from this seller.</td><td>[0.79, 0.21]</td></tr></table>

Table 5: Examples of the data generated using UNIGEN.

<table><tr><td>DistilBERT</td><td>SST-2</td><td>IMDB</td><td>Rotten</td><td>Amazon</td><td>Yelp</td><td>CR</td><td>Tweet</td><td>Average</td></tr><tr><td>UNIGEN</td><td>77.67</td><td>67.81</td><td>73.16</td><td>75.06</td><td>74.81</td><td>79.86</td><td>81.41</td><td>75.68</td></tr><tr><td>UNIGEN w/ Hard Relabeling</td><td>77.18</td><td>67.18</td><td>72.37</td><td>72.91</td><td>72.95</td><td>78.14</td><td>80.39</td><td>74.45</td></tr><tr><td>UNIGEN w/o Relabeling</td><td>76.34</td><td>66.58</td><td>71.78</td><td>70.63</td><td>70.97</td><td>76.59</td><td>79.62</td><td>73.22</td></tr><tr><td>UNIGEN w/o Denoising MB</td><td>77.06</td><td>67.13</td><td>72.04</td><td>74.69</td><td>73.66</td><td>78.47</td><td>80.84</td><td>74.84</td></tr><tr><td>UNIGEN w/o SCL</td><td>75.53</td><td>66.10</td><td>69.63</td><td>71.43</td><td>69.58</td><td>77.22</td><td>79.31</td><td>72.69</td></tr><tr><td>Combined Prompts</td><td>74.19</td><td>63.16</td><td>71.08</td><td>73.62</td><td>72.93</td><td>78.05</td><td>78.02</td><td>73.01</td></tr></table>

Table 6: Results of ablation studies on methodological choices in Section 4.5.1, 4.5.2, and 4.5.3.

<table><tr><td>DistilBERT</td><td>SST-2</td><td>IMDB</td><td>Rotten</td><td>Amazon</td><td>Yelp</td><td>CR</td><td>Tweet</td><td>Average</td></tr><tr><td>UNIGEN w/ GPT2-XL UNIGEN</td><td>77.67</td><td>67.81</td><td>73.16</td><td>75.06</td><td>74.81</td><td>79.86</td><td>81.41</td><td>75.68</td></tr><tr><td>w/ Gemma-2b UNIGEN</td><td>71.50</td><td>69.40</td><td>67.04</td><td>76.48</td><td>76.89</td><td>77.24</td><td>52.03</td><td>70.08</td></tr><tr><td>w/ Qwen2-1.5B UNIGEN</td><td>66.37</td><td>63.19</td><td>63.76</td><td>71.69</td><td>72.44</td><td>66.06</td><td>63.49</td><td>66.71</td></tr><tr><td>w/ Phi-1.5</td><td>74.98</td><td>68.35</td><td>70.82</td><td>73.86</td><td>75.11</td><td>71.82</td><td>84.01</td><td>74.13</td></tr></table>

Table 7: Results of ablation studies on comparison between various PLMs in Section 4.5.4.

## 4.5.2 Effectiveness of Supervised Contrastive Learning and Denoising Memory Bank

Second, we conducted a comparison to investigate the effectiveness of supervised contrastive learning, which was discussed in Section 3.1.2, and denoising memory bank, which was discussed in Section 3.4. The results of the comparison are presented in fourth and fifth rows of Table 6. Intuitively, if the quality of each of the data in the dataset is given as a weight, it would be effective to employ only high-quality samples for comparing contrastive learning rather than utilizing all data, regardless of their quality. The experimental result in the fourth row demonstrated that the use of a denoising memory bank yielded a performance gain, which was consistent with our intuition. Similarly, the result in the fifth row suggests that supervised contrastive learning plays a crucial role in UNI-GEN.

## 4.5.3 Comparison with Combined Domain-specific Datasets

Third, we compared the performance of the TAMs trained with two different synthetic datasets. The first uses the synthetic dataset generated with the prompt of UNIGEN, and the second uses the concatenation of datasets generated with five different domain-specific prompts used in the other experiments. For this experiment, we only differentiated the synthetic dataset used for training and set every other configuration identical, such as the usage of pseudo-relabeling and denoised memory bank, as well as other hyperparameters. The result of the ablation study is presented in the last row of Table 6. The result indicates that the model trained with the dataset generated by the universal prompt in UNIGEN demonstrated better average performance. This suggests that the broad understanding of the label space offered by the synthetic dataset generated by UNIGEN plays an important role in domain generalization.

## 4.5.4 Comparison between PLMs for Data Generation

Lastly, we evaluated the performance of TAMs trained using various PLMs. Initially, we utilized GPT2-XL as the PLM for data generation. In this experiment, we extended the evaluation by incorporating more recent models as data generators. Specifically, we compared the performance of TAMs trained with UNIGEN using Gemma-2b (Team et al., 2024), Qwen2-1.5B (Yang et al., 2024), and Phi-1.5 (Li et al., 2023), which are more recent models with parameter sizes comparable to GPT2-XL. All other configurations, aside from the PLM used for data generation, were kept consistent with the original GPT2-XL-based TAM.

Table 7 presents the results of this experiment. Interestingly, the findings suggest that employing more recent PLMs does not necessarily lead to better performance in UNIGEN. The TAM trained with GPT2-XL, our original choice for data generation, achieved the highest average performance. This aligns with previous studies, which indicate that using larger PLM does not always result in superior outcomes (Ye et al., 2022a). However, despite using identical hyperparameters and prompts to ensure a fair comparison, it is important to recognize that optimal hyperparameters, such as top-k, top-p, and τ<sub>RE</sub>, as well as the prompt configurations, may vary for each PLM. Future research could focus on developing a unified framework to optimize hyperparameters and prompts for each PLMs, akin to methods like AutoAugment (Cubuk et al., 2019; Ren et al., 2021).

## 5 Conclusion

In this study, we proposed UNIGEN in an attempt to achieve universal domain generalization. UNI-GEN successfully transferred the domain generalizability of PLMs into orders-of-magnitude smaller TAMs. Moreover, human annotation was not required for UNIGEN, which significantly reduced the burden of acquiring labeled data from multiple source domains. Our relabeling method and denoising memory bank offered additional performance gains. Furthermore, our extensive experiments demonstrated that UNIGEN outperformed PROMPTING, facilitating light inference while preserving the domain generalizability of PLMs.

Although we explored an interesting framework for zero-shot, lightweight domain generalization, the performance of UNIGEN appears weaker than those of baseline models that are trained on each domain in several cases. It is desirable to achieve a higher level of performance than those of the indomain baselines, which we will attempt in future work. To this end, the generation of small taskspecific data for additional training of the TAM trained using UNIGEN is a possible approach, especially when a downstream task domain is introduced. By employing TAMs that are pre-trained using UNIGEN as a warm start, high performance could be achieved in the target domain with a small amount of task-specific data, which would reduce the total amount of data generated compared to that when individually training each TAM by using ZEROGEN or SUNGEN from scratch. Another possible approach may involve combining UNIGEN with the concept of test-time learning (Jeong et al., 2023). Similar to the first strategy, it may generate small amounts of test domain-specific data given test-time data as in-context examples. We are committed to exploring these possible strategies, which will enhance the effectiveness of UNIGEN.

## Limitations

The primary limitation of UNIGEN is its relatively weaker in-domain performance than those of baselines that are trained with domain-specific datasets. While it is beneficial for its smaller parameter set and lower inference cost while maintaining the domain generalizability of PLMs, there exists a tradeoff between in-domain performance and efficiency, unlike ZEROGEN and SUNGEN. Therefore, a method for further enhancing the performance of UNIGEN should be explored, as stated in the Conclusion section. A possible solution is a proper prompt designed for UNIGEN because the quality of the generated sentences is affected by prompt design. Even though we adapted an effective prompt designed in a previous work (Ye et al., 2022a), a more effective prompt for UNIGEN that aims to generate diverse and general expressions could exist.

## Ethics Statement

The data generated by the PLM may contain biased sentences, which may offend the readers. This can be attributed to the potential bias of PLMs (Liu et al., 2022). These generated biased sentences do not reflect the views of the authors.

## Acknowledgements

This research was supported by Basic Science Research Program through the National Research Foundation of Korea(NRF) funded by the Ministry of Education(NRF-2022R1C1C1008534), and Institute for Information & communications Technology Planning & Evaluation (IITP) through the Korea government (MSIT) under Grant No. 2021- 0-01341 (Artificial Intelligence Graduate School Program, Chung-Ang University).

## References

Maha Agro and Hanan Aldarmaki. 2023. Handling realistic label noise in bert text classification. In Proceedings ofICNLSP, pages 11–20.

Ateret Anaby-Tavor, Boaz Carmeli, Esther Goldbraich, Amir Kantor, George Kour, Segev Shlomov, Naama Tepper, and Naama Zwerdling. 2020. Do not have enough data? deep learning to the rescue! In Proceedings ofAAAI, pages 7383–7390.

John Blitzer, Mark Dredze, and Fernando Pereira. 2007. Biographies, bollywood, boom-boxes and blenders: Domain adaptation for sentiment classification. In Proceedings ofACL, pages 440–447.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. In Proceedings of NeurIPS, pages 1877– 1901.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In Proceedings ofICML, pages 1597–1607.

Xilun Chen and Claire Cardie. 2018. Multinomial adversarial networks for multi-domain text classification. In Proceedings ofNAACL, pages 1226–1240.

Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V Le. 2019. Autoaugment: Learning augmentation strategies from data. In Proceedings ofCVPR, pages 113–123.

Xiaowen Ding, Bing Liu, and Philip S Yu. 2008. A holistic lexicon-based approach to opinion mining. In Proceedings ofWSDM, pages 231–240.

Tianqing Fang, Wenxuan Zhou, Fangyu Liu, Hongming Zhang, Yangqiu Song, and Muhao Chen. 2024. Onthe-fly denoising for data augmentation in natural language understanding. In Findings ofEACL, pages 766–781.

Jiahui Gao, Renjie Pi, Lin Yong, Hang Xu, Jiacheng Ye, Zhiyong Wu, Weizhong Zhang, Xiaodan Liang, Zhenguo Li, and Lingpeng Kong. 2023. Self-guided noise-free data generation for efficient zero-shot learning. In Proceedings of ICLR.

Han Guo, Ramakanth Pasunuru, and Mohit Bansal. 2020. Multi-source domain adaptation for text classification via distancenet-bandits. In Proceedings of AAAI, pages 7830–7838.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. 2020. Momentum contrast for unsupervised visual representation learning. In Proceedings ofCVPR, pages 9729–9738.

Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation, 9(8):1735– 1780.

Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Hwang, and Jong Park. 2023. Test-time self-adaptive small language models for question answering. In Findings ofEMNLP, pages 15459–15469.

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. 2020. Tinybert: Distilling bert for natural language understanding. In Findings ofEMNLP, pages 4163–4174.

Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2020. Supervised contrastive learning. In Proceedings of NeurIPS, pages 18661–18673.

Daehee Kim, Youngjun Yoo, Seunghyun Park, Jinkyu Kim, and Jaekoo Lee. 2021. Selfreg: Self-supervised contrastive regularization for domain generalization. In Proceedings ofICCV, pages 9619–9628.

Yoon Kim. 2014. Convolutional neural networks for sentence classification. In Proceedings of EMNLP, pages 1746–1751.

Diederik P Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In Proceedings of ICLR.

Varun Kumar, Ashutosh Choudhary, and Eunah Cho. 2020. Data augmentation using pre-trained transformer models. In Proceedings AACL 2020 Workshop on Life-long Learning for Spoken Language Systems, pages 18–26.

Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. 2023. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463.

Ruibo Liu, Chenyan Jia, Jason Wei, Guangxuan Xu, and Soroush Vosoughi. 2022. Quantifying and alleviating political bias in language models. Artificial Intelligence, 304:103654.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Andrew Maas, Raymond E Daly, Peter T Pham, Dan Huang, Andrew Y Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings ofACL, pages 142–150.

Julian McAuley and Jure Leskovec. 2013. Hidden factors and hidden topics: understanding rating dimensions with review text. In Proceedings of RecSys, pages 165–172.

Yu Meng, Jiaxin Huang, Yu Zhang, and Jiawei Han. 2022. Generating training data with language models: Towards zero-shot language understanding. In Proceedings ofNeurIPS, pages 462–477.

Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of EMNLP, pages 188–197.

Bo Pang and Lillian Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of ACL, pages 115–124.

Fengchun Qiao, Long Zhao, and Xi Peng. 2020. Learning to learn single domain generalization. In Proceedings ofCVPR, pages 12556–12565.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Shuhuai Ren, Jinchao Zhang, Lei Li, Xu Sun, and Jie Zhou. 2021. Text autoaugment: Learning compositional augmentation policy for text classification. In Proceedings ofEMNLP, pages 9029–9043.

Joshua David Robinson, Ching-Yao Chuang, Suvrit Sra, and Stefanie Jegelka. 2021. Contrastive learning with hard negative samples. In Proceedings ofICLR.

Sara Rosenthal, Noura Farra, and Preslav Nakov. 2017. Semeval-2017 task 4: Sentiment analysis in twitter. In Proceedings of SemEval, pages 502–518.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings ofEMNLP, pages 1631–1642.

Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. 2023. Learning from noisy labels with deep neural networks: A survey. IEEE Transactions on Neural Networks and Learning Systems, 34(11):8135–8153.

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. 2016. Rethinking the inception architecture for computer vision. In Proceedings ofCVPR, pages 2818–2826.

Qingyu Tan, Ruidan He, Lidong Bing, and Hwee Tou Ng. 2022. Domain generalization for text classification with memory-based supervised contrastive learning. In Proceedings ofCOLING, pages 6916–6926.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295.

Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-sne. Journal of machine learning research, 9(86):2579–2605.

Jindong Wang, Cuiling Lan, Chang Liu, Yidong Ouyang, Tao Qin, Wang Lu, Yiqiang Chen, Wenjun Zeng, and Philip Yu. 2022. Generalizing to unseen domains: A survey on domain generalization. IEEE Transactions on Knowledge and Data Engineering, 35(8):8052–8072.

Song Wang, Zhen Tan, Ruocheng Guo, and Jundong Li. 2023. Noise-robust fine-tuning of pretrained language models via external guidance. In Findings of EMNLP, pages 12528–12540.

Zijian Wang, Yadan Luo, Ruihong Qiu, Zi Huang, and Mahsa Baktashmotlagh. 2021. Learning to diversify for single domain generalization. In Proceedings of ICCV, pages 834–843.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2020. Transformers: State-of-the-art natural language processing. In Proceedings ofEMNLP (Demo Track), pages 38–45.

Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. 2018. Unsupervised feature learning via nonparametric instance discrimination. In Proceedings of CVPR, pages 3733–3742.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. 2024. Qwen2 technical report. arXiv preprint arXiv:2407.10671.

Hai Ye, Qingyu Tan, Ruidan He, Juntao Li, Hwee Tou Ng, and Lidong Bing. 2020. Feature adaptation of pre-trained language models across languages and domains with robust self-training. In Proceedings of EMNLP, pages 7386–7399.

Jiacheng Ye, Jiahui Gao, Qintong Li, Hang Xu, Jiangtao Feng, Zhiyong Wu, Tao Yu, and Lingpeng Kong. 2022a. Zerogen: Efficient zero-shot learning via dataset generation. In Proceedings ofEMNLP, pages 11653–11669.

Jiacheng Ye, Jiahui Gao, Zhiyong Wu, Jiangtao Feng, Tao Yu, and Lingpeng Kong. 2022b. Progen: Progressive zero-shot dataset generation via in-context feedback. In Findings ofEMNLP, pages 3671–3683.

Jiacheng Ye, Chengzu Li, Lingpeng Kong, and Tao Yu. 2023. Generating data for symbolic language with large language models. In Proceedings of EMNLP, pages 8418–8443.

Kang Min Yoo, Dongju Park, Jaewook Kang, Sang-Woo Lee, and Woomyoung Park. 2021. Gpt3mix: Leveraging large-scale language models for text augmentation. In Findings ofEMNLP, pages 2225–2239.

Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Proceedings ofNeurIPS.

Kaiyang Zhou, Ziwei Liu, Yu Qiao, Tao Xiang, and Chen Change Loy. 2022. Domain generalization: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4396–4415.

Tianyuan Zou, Yang Liu, Peng Li, Jianqing Zhang, Jingjing Liu, and Ya-Qin Zhang. 2024. Fusegen: Plm fusion for data-generation based zero-shot learning. arXiv preprint arXiv:2406.12527.

## A Prompt for Each Domain

<table><tr><td>Domain</td><td>Prompt</td></tr><tr><td>Movie Products</td><td>The movie review in [positive/negative] sentiment is:</td></tr><tr><td>Restaurant</td><td>The product review in [positive/negative] sentiment is: The restaurant review in [positive/negative] sentiment is:</td></tr><tr><td>Electronics</td><td>The electronics product review in [positive/negative] sentiment is:</td></tr><tr><td>Tweet UNIGEN &amp;</td><td>The tweet in [positive/negative] sentiment is:</td></tr><tr><td>PROMPTING</td><td>The text in [positive/negative] sentiment is:</td></tr></table>

Table 8: The prompt used for each domain in ZEROGEN and SUNGEN, as well as the prompt used for UNIGEN and PROMPTING.

## B Implementation Detail

For UNIGEN, we first generated 1,000k data from the 1.5B GPT2-XL model as by using the prompt $\mathcal { T } _ { u n i }$ “The text in positive/negative sentiment is: ”, which is a slightly modified version of the best prompt suggested in a previous study. Top-k and top-p were set to 40 and 0.9 during the generation procedure, respectively. The soft relabeling process was performed using a $\tau _ { R E }$ of 0.1. After obtaining the soft labels of each of the generated samples, we filtered them using $T _ { R E }$ of 0.2. This required the largest value from the soft labels to be larger than the sum of the uniform distribution and $T _ { R E }$ , for instance, 0.7 in binary classification with $T _ { R E }$ of 0.2. As an example, the sentence corresponding to the soft label [0.64, 0.36] was discarded because it did not exceed the threshold.

After generation, we followed the bi-level optimization approach proposed in SUNGEN to cleanse the generated dataset and find the sample weights for 50 epochs. The outer learning rate was set to 5e-2, and we randomly sampled 50k data for each outer validation process. Then, we selected 200k data with high weights, which represent highquality data, to train the TAMs.

We used a one-layer bi-LSTM model for the LSTM-based TAM and the distilbert-baseuncased and roberta-base from Transformers (Wolf et al., 2020) for the DistilBERT-based TAM and RoBERTa-based TAM, respectively. We trained the LSTM-based TAM for 5 epochs with the learning rate of 1e-3 by using the Adam (Kingma and Ba, 2015) optimizer. The DistilBERT-based TAM was trained for 3 epochs with a learning rate of 2e-5 by using the Adam optimizer. The RoBERTa-based TAM was trained for 3 epochs with a learning rate of 2e-5 by using the Adam optimizer. During the training process, α for supervised contrastive learning loss was set to 0.5, with a projection size of

256. The temperature $\tau _ { S C L }$ was set to 0.2, and the memory bank size M was set to 64. The coefficient m for updating the momentum encoder was set to 0.999, and the threshold of the denoising memory bank $T _ { M B }$ was set to 0.8. The dataset generation and training procedures were executed using on a single NVIDIA A100 40GB GPU. Please refer to attached source code for further details.<sup>1</sup>

## C Extensibility of Relabeling Strategy

<table><tr><td>DistilBERT</td><td>SST-2</td><td>IMDB</td><td>Rotten</td><td>Amazon</td><td>Yelp</td><td>CR</td><td>Tweet</td><td>Average</td></tr><tr><td>ZEROGEN</td><td>80.06</td><td>69.13</td><td>74.73</td><td>73.02</td><td>72.77</td><td>73.59</td><td>74.83</td><td>74.02</td></tr><tr><td>ZEROGEN w/ Hard Relabeling</td><td>80.72</td><td>69.25</td><td>73.98</td><td>73.41</td><td>73.18</td><td>73.76</td><td>74.91</td><td>74.17</td></tr><tr><td>ZEROGEN w/ Soft Relabeling</td><td>81.79</td><td>70.40</td><td>75.32</td><td>73.65</td><td>73.31</td><td>74.72</td><td>75.14</td><td>74.90</td></tr></table>

Table 9: Experimental result on the extensibility of relabeling strategy. We trained the TAM using ZEROGEN based on the movie domain.

We examined the extensibility of the relabeling strategy discussed in Section 3.3. We applied two different options for relabeling, namely assigning hard labels and soft labels to ZEROGEN. Table 9 summarizes the results. These results suggest that the relabeling strategy is beneficial for the performance of the TAM trained using ZEROGEN. Therefore, filtering the generated data through the relabeling strategy is an extensive strategy for enhancing zero-shot learning methods based on dataset generation. Furthermore, the assignment of soft labels was more beneficial compared to the assignment of hard labels, which is consistent with the results of the ablation study described in Section 4.5.1. We will further investigate the relabeling-based approach to enhance ZEROGEN and SUNGEN in future works.

## D Additional Experiment on Domain Generalizability

To further reveal the domain generalizability of UNIGEN, we conducted an additional experiment on Amazon Review dataset (Ni et al., 2019). We used 5-core data for 29 domains and reported the performance of PROMPTING using GPT2-XL (Radford et al., 2019) and RoBERTa-based TAM trained by UNIGEN. The result in Table 10 demonstrates the performance of UNIGEN that is comparable with PROMPTING, with parameters less than 10%. Additionally, this experiment showcases the universality of UNIGEN, the characteristics that distinguish UNIGEN from previous ZEROGEN and SUN-GEN. Compared to previous methods that would require 29 separately trained TAMs to conduct this experiment, UNIGEN only used one single TAM to perform the experiment, which exhibits the realworld applicability of UNIGEN.

<table><tr><td>Domain</td><td>PROMPTING</td><td>UNIGEN</td></tr><tr><td>Fashion</td><td>93.29</td><td>91.16</td></tr><tr><td>Beauty</td><td>95.63</td><td>92.87</td></tr><tr><td>Appliances</td><td>68.27</td><td>79.10</td></tr><tr><td>Arts, Crafts and Sewing</td><td>91.05</td><td>92.08</td></tr><tr><td>Automotive</td><td>91.07</td><td>88.23</td></tr><tr><td>Books</td><td>89.18</td><td>91.26</td></tr><tr><td>CDs and Vinyl</td><td>82.44</td><td>86.42</td></tr><tr><td>Cell Phones and Accessories</td><td>90.47</td><td>88.65</td></tr><tr><td>Clothing, Shoes and Jewelry</td><td>91.83</td><td>90.80</td></tr><tr><td>Digital Music</td><td>93.72</td><td>90.62</td></tr><tr><td>Electronics</td><td>88.68</td><td>88.34</td></tr><tr><td>Gift Cards</td><td>94.03</td><td>92.50</td></tr><tr><td>Grocery and Gourmet Food</td><td>92.31</td><td>91.09</td></tr><tr><td>Home and Kitchen</td><td>92.11</td><td>91.53</td></tr><tr><td>Industrial and Scientific</td><td>91.07</td><td>92.34</td></tr><tr><td>Kindle Store</td><td>89.49</td><td>92.76</td></tr><tr><td>Luxury Beauty</td><td>90.03</td><td>91.82</td></tr><tr><td>Magazine Subscriptions</td><td>85.97</td><td>89.64</td></tr><tr><td>Movies and TV</td><td>86.39</td><td>88.19</td></tr><tr><td>Musical Instruments</td><td>90.72</td><td>90.20</td></tr><tr><td>Office Products</td><td>91.74</td><td>89.60</td></tr><tr><td>Patio, Lawn and Garden</td><td>89.96</td><td>87.87</td></tr><tr><td>Pet Supplies</td><td>90.60</td><td>89.91</td></tr><tr><td>Prime Pantry</td><td>93.64</td><td>88.15</td></tr><tr><td>Software</td><td>82.55</td><td>83.39</td></tr><tr><td>Sports and Outdoors</td><td>88.63</td><td>90.36</td></tr><tr><td>Tools and Home Improvement</td><td>87.41</td><td>88.90</td></tr><tr><td>Toys and Games</td><td>91.54</td><td>92.02</td></tr><tr><td>Video Games</td><td>85.79</td><td>86.07</td></tr><tr><td>Average</td><td>89.30</td><td>89.51</td></tr></table>

Table 10: The result of the experiment on the Amazon Review dataset.

## E Additional Study on the Performance of UNIGEN on Small-sized TAMs

We found that UNIGEN suffers to exhibit its performance on the LSTM model from the experiment in Table 2. To further investigate this phenomenon, we expand our experiment into two different smallsized TAMs: TextCNN (Kim, 2014) and TinyBERT (Jiao et al., 2020). Table 11 showcases the result of the additional experiment. In the case of TextCNNbased TAM, baseline methods such as ZEROGEN and SUNGEN demonstrated comparable or slightly higher performance compared to that of LSTMbased TAM. Nonetheless, TextCNN-based TAM trained on UNIGEN reported slightly worse performance compared to LSTM-based TAM despite increased parameter size. We hypothesize that this phenomenon is owing to the architecture of TextCNN, which leverages CNN layers that have fixed window size, leading to limited ability to understand the context of diverse expression generated by UNIGEN. On the contrary, TinyBERTbased TAM trained on UNIGEN exhibited the best average performance among the baselines. Furthermore, its average performance is comparable to DistilBERT-based TAM despite a much smaller parameter size. It is noteworthy that TinyBERT is also a model that has a general understanding of the language through knowledge distillation from BERT. Through this investigation, we reveal that the pre-trained knowledge of the TAM aids the successful training of the TAM through UNIGEN.

<table><tr><td>Model Test Domain</td><td>#Param</td><td>Training Domain</td><td>Setup</td><td>SST-2</td><td>IMDB Movie</td><td>Rotten</td><td>Amazon Products</td><td>Yelp Restaurant</td><td>CR Electronics</td><td>Tweet Tweet</td><td>Average</td></tr><tr><td>GPT2-XL</td><td>1.5B</td><td>-</td><td>PROMPTING</td><td>82.15</td><td>70.26</td><td>77.56</td><td>79.06</td><td>78.04</td><td>80.30</td><td>80.38</td><td>78.25</td></tr><tr><td rowspan="9">LSTM</td><td rowspan="9"></td><td rowspan="2">Movie</td><td>ZEROGEN</td><td>75.11</td><td>66.39</td><td>69.85</td><td>67.24</td><td>70.25</td><td>69.32</td><td>63.43</td><td>68.80</td></tr><tr><td>SUNGEN</td><td>78.79</td><td>69.97</td><td>73.76</td><td>72.15</td><td>73.21</td><td>70.39</td><td>66.84</td><td>72.16</td></tr><tr><td rowspan="2">Products</td><td>ZEROGEN</td><td>64.26</td><td>61.82</td><td>60.13</td><td>70.32</td><td>67.78</td><td>69.46</td><td>62.29</td><td>65.15</td></tr><tr><td>SUNGEN</td><td>67.83</td><td>63.87</td><td>63.46</td><td>74.43</td><td>73.71</td><td>73.35</td><td>63.51</td><td>68.59</td></tr><tr><td rowspan="2">Restaurant</td><td>ZEROGEN</td><td>67.41</td><td>63.01</td><td>62.74</td><td>68.73</td><td>75.51</td><td>69.23</td><td>66.35</td><td>63.28</td></tr><tr><td>SUNGEN</td><td>69.15</td><td>66.62</td><td>64.56</td><td>73.22</td><td>79.56</td><td>70.12</td><td>67.43</td><td>70.09</td></tr><tr><td rowspan="2">Electronics</td><td>ZEROGEN</td><td>64.69</td><td>59.13</td><td>60.20</td><td>66.34</td><td>67.72</td><td>72.50</td><td>60.25</td><td>64.40</td></tr><tr><td>SUNGEN</td><td>68.38</td><td>64.33</td><td>63.25</td><td>72.61</td><td>73.01</td><td>76.18</td><td>66.78</td><td>69.22</td></tr><tr><td rowspan="2">Tweet</td><td>ZEROGEN</td><td>61.84</td><td>60.17</td><td>59.43</td><td>64.13</td><td>63.68</td><td>65.02</td><td>74.10</td><td>64.05</td></tr><tr><td>SUNGEN</td><td>66.57</td><td>63.96</td><td>64.21</td><td>69.36</td><td>71.68</td><td>72.57</td><td>81.29</td><td>69.95</td></tr><tr><td rowspan="9">CNN 10M</td><td rowspan="2"></td><td>UNIGEN</td><td>64.15</td><td>60.02</td><td>60.51</td><td>63.82</td><td>63.20</td><td>69.61</td><td>70.32</td><td>64.52</td></tr><tr><td>ZEROGEN</td><td>74.34</td><td>67.91</td><td>70.22</td><td>68.69</td><td>71.03</td><td>70.89</td><td>64.77</td><td>69.69</td></tr><tr><td rowspan="2">Movie</td><td>SUNGEN</td><td>76.98</td><td>68.97</td><td>73.49</td><td>73.04</td><td>73.97</td><td>71.55</td><td>69.43</td><td>72.49</td></tr><tr><td>ZEROGEN</td><td>63.46</td><td>62.13</td><td>60.35</td><td>70.94</td><td>68.34</td><td>72.34</td><td>65.71</td><td>66.18</td></tr><tr><td rowspan="2">Products</td><td>SUNGEN</td><td>65.89</td><td>63.27</td><td>61.97</td><td>73.98</td><td>72.81</td><td>74.02</td><td>67.38</td><td>68.47</td></tr><tr><td>ZEROGEN</td><td>67.76</td><td>64.18</td><td>62.16</td><td>70.17</td><td>76.65</td><td>71.27</td><td>65.43</td><td>68.23</td></tr><tr><td rowspan="2">Restaurant</td><td>SUNGEN</td><td>68.86</td><td>65.62</td><td>64.96</td><td>73.20</td><td>77.87</td><td>72.43</td><td>68.36</td><td>70.19</td></tr><tr><td>ZEROGEN</td><td>65.05</td><td>63.04</td><td>62.13</td><td>67.19</td><td>69.50</td><td>73.66</td><td>63.23</td><td>66.26</td></tr><tr><td rowspan="2">Electronics</td><td>SUNGEN</td><td>67.43</td><td>65.13</td><td>63.25</td><td>70.82</td><td>72.79</td><td>77.42</td><td>67.19</td><td>69.15</td></tr><tr><td>ZEROGEN</td><td>60.56</td><td>60.68</td><td>61.33</td><td>64.91</td><td>64.37</td><td>66.86</td><td>75.62</td><td>64.90</td></tr><tr><td rowspan="9">TinyBERT</td><td rowspan="2">Tweet</td><td>SUNGEN</td><td>65.12</td><td>61.56</td><td>63.42</td><td>66.45</td><td>68.46</td><td>68.71</td><td>80.17</td><td>67.70</td></tr><tr><td>UNIGEN</td><td>62.31</td><td>60.48</td><td>61.82</td><td>61.08</td><td>61.63</td><td>68.24</td><td>65.95</td><td>63.07</td></tr><tr><td rowspan="2">Movie</td><td>ZEROGEN</td><td>78.95</td><td>68.37</td><td>71.34</td><td>70.59</td><td>71.35</td><td>71.18</td><td>68.94</td><td>71.53</td></tr><tr><td>SUNGEN</td><td>80.78</td><td>69.86</td><td>73.47</td><td>72.36</td><td>72.42</td><td>73.75</td><td>70.81</td><td>73.35</td></tr><tr><td rowspan="2">Products</td><td>ZEROGEN</td><td>69.22</td><td>62.79</td><td>63.44</td><td>72.57</td><td>69.70</td><td>73.22</td><td>71.21</td><td>68.88</td></tr><tr><td>SUNGEN</td><td>71.74</td><td>64.38</td><td>64.51</td><td>75.81</td><td>73.76</td><td>74.17</td><td>72.86</td><td>71.03</td></tr><tr><td rowspan="2">Restaurant</td><td>ZEROGEN</td><td>75.79</td><td>64.62</td><td>65.53</td><td>71.33</td><td>77.10</td><td>73.52</td><td>70.84</td><td>71.25</td></tr><tr><td>SUNGEN</td><td>77.45</td><td>67.41</td><td>68.01</td><td>74.41</td><td>79.16</td><td>75.86</td><td>72.11</td><td>73.49</td></tr><tr><td rowspan="2">Electronics</td><td>ZEROGEN</td><td>71.22</td><td>64.37</td><td>63.06</td><td>69.51</td><td>70.75</td><td>75.71</td><td>69.49</td><td>69.16</td></tr><tr><td>SUNGEN</td><td>73.10</td><td>65.81</td><td>66.71</td><td>71.33</td><td>74.86</td><td>78.43</td><td>73.88</td><td>72.02</td></tr><tr><td rowspan="2">Tweet</td><td>ZEROGEN</td><td>70.76</td><td>63.40</td><td>64.43</td><td>68.74</td><td>70.44</td><td>73.72</td><td>78.14</td><td></td><td>69.95</td></tr><tr><td></td><td>SUNGEN</td><td>73.94</td><td>64.87 66.31</td><td></td><td>71.39</td><td>72.21</td><td>78.16</td><td>81.23</td><td>72.59</td></tr><tr><td></td><td></td><td>UNIGEN</td><td>76.74</td><td>66.88 69.63</td><td></td><td>73.29</td><td>72.10</td><td>78.64</td><td>80.52</td><td>73.97</td></tr></table>

Table 11: Result of ablation study that examines the performance of UNIGEN and baselines on small-sized TAMs. The performance of TAM, which is superior to that of PROMPTING, is underlined, and the best result in each test dataset within the group for each TAM is presented in boldface.