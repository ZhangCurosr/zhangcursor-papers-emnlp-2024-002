# UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation

Xiangyu Zhao<sup>1</sup>, Yuehan Zhang<sup>2</sup>, Wenlong Zhang<sup>1,3</sup>, Xiao-Ming Wu<sup>1</sup> <sup>1</sup>Department of Computing, The Hong Kong Polytechnic University <sup>2</sup>Wuhan University, <sup>3</sup>Shanghai AI Laboratory xiang-yu.zhao@connect.polyu.hk, xiao-ming.wu@polyu.edu.hk

## Abstract

The fashion domain includes a range of realworld multimodal tasks, such as multimodal retrieval and generation. Recent advancements in AI-generated content, particularly large language models for text and diffusion models for visuals, have spurred significant research interest in applying these multimodal models to fashion. However, fashion models must also effectively handle embedding tasks, like imageto-text and text-to-image retrieval. Moreover, current unified fashion models often lack the capability for image generation. In this work, we present UniFashion, a unified framework that tackles the challenges of multimodal generation and retrieval tasks in the fashion domain, by integrating image and text generation with retrieval tasks. UniFashion unifies embedding and generative processes through the use of a diffusion model and LLM, enabling controllable and high-fidelity generation. Our model significantly outperforms previous state-of-the-art models focused on single tasks across various fashion-related challenges and can be easily adapted to manage complex vision-language tasks. This study highlights the synergistic potential between multimodal generation and retrieval, offering a promising avenue for future research in the fashion domain. The source code is available at https: //github.com/xiangyu-mm/UniFashion.

## 1 Introduction

The fashion domain presents a range of real-world multimodal tasks, encompassing multimodal retrieval (Gao et al., 2020; Wu et al., 2021; Bai et al., 2023; Liu et al., 2024b) and multimodal generation (Yang et al., 2020) tasks. Such tasks have been utilized in diverse e-commerce scenarios to enhance product discoverability, seller-buyer interaction, and customer conversion rates after catalog browsing (Han et al., 2023; Zhuge et al., 2021). The remarkable progress in the field of artificial intelligence generated content (AIGC), particularly in technologies like large language models (LLMs) (Chiang et al., 2023; Touvron et al., 2023; Brown et al., 2020) for text generation and diffusion models (Rombach et al., 2022; Nichol et al., 2022; Saharia et al., 2022) for visual generation, yielding significant advancements in numerous downstream tasks (Feng et al., 2023; Zhang et al., 2022) and sparking widespread research interest in applying these multimodal models to the fashion domain.

Instruction-tuned multimodal large language models (Liu et al., 2023a; Dai et al., 2023; Dong et al., 2023; Zhao et al., 2024) (MLLMs) have emerged as a promising direction for developing a single multi-task model (Shi et al., 2023). However, due to the heterogeneous nature of multimodal fashion tasks (Han et al., 2023), most existing MLLMs struggle to be directly applicable in the fashion domain. For example, in the fashion domain, retrieval tasks that rely on embedding ability, such as imageto-text or text-to-image retrieval, have largely been overlooked. Furthermore, existing MLLMs lack the ability to solve the composed image retrieval (CIR) (Liu et al., 2021; Baldrati et al., 2022) task, which composes the reference image and related caption in a joint embedding to calculate similarities with candidate images and is particularly relevant in recommender systems (Han et al., 2017; Liu et al., 2022, 2024a).

Drawing inspiration from GRIT (Muennighoff et al., 2024), which successfully combined generative and embedding tasks into a unified model for text-centric applications and enhanced embedding performance by incorporating a generative objective, it is evident that exploring task correlations and integrating embedding with generative models in the fashion domain is promising.

While previous works (Han et al., 2023; Zhuge et al., 2021) in the fashion domain have also proposed using a single model for solving multiple tasks, they ignore image generation tasks. Besides, for fashion tasks such as try-on (Choi et al., 2021) and fashion design (Baldrati et al., 2023b), it is generally required to generate target images based on multimodal input. However, previous works (Baldrati et al., 2023b) in fashion image generation typically adopt the CLIP text encoder for encoding text information. This approach may not effectively capture the textual context due to the limitations of the text encoder, as noted by Saharia et al. (2022). Hence, we posit that current studies have yet to fully explore the potential synergy between generation and retrieval.

![](images/846130530d749c82e534c7fce7527e4f68e81f8e8ea94964d3a55bf6fa0478c8.jpg)  
Figure 1: Illustration of the fashion tasks encompassed in our UniFashion framework: cross-modal retrieval, text-guided image retrieval, fashion image captioning, and fashion image generation. Model inputs highlighted with a light yellow background and outputs denoted by a light blue background.

In this work, we propose UniFashion, which unifies retrieval and generation tasks by integrating LLMs and diffusion models, as illustrated in Figure 2. UniFashion consists of three parts: The Q-Former is crucial for amalgamating text and image input, creating multimodal learnable queries. These queries, once refined through task-specific adapters, enable the LLM module to utilize them as soft prompts for generating captions for target images. Simultaneously, the diffusion module utilizes the learnable queries as conditions to guide the latent diffusion model in image synthesis and editing tasks. To enable controllable and high-fidelity generation, we propose a two-phase training strategy. In the first phase, we perform multimodal representation learning on image-text pairs datasets. We freeze Q-Former and fine-tune the LLM and diffusion modules, ensuring they develop the capability to comprehend the multimodal representations provided by Q-Former. Subsequently, in the second phase, we proceed to fine-tune UniFashion on datasets with multimodal inputs, such as Fashion-IQ, where we freeze the LLM and diffusion modules, only tuning Q-Former. This strategy ensures that Q-Former is adept at crafting multimodal representations that effectively integrate both reference images and text inputs.

UniFashion holds three significant advantages that address the challenges in multimodal fashion retrieval and generation:

• For the first time, we conduct an in-depth study of the synergistic modeling of multimodal retrieval and generation tasks within the fashion domain, thoroughly exploiting the inter-task relatedness. Further, we introduce UniFashion, a versatile, unified model that can handle all fashion tasks.

• Secondly, our model enhances performance via mutual task reinforcement. Specifically, the caption generative module aids the CIR task, while jointly training the generation and retrieval tasks improves the multimodal encoder for the diffusion module.

• Thirdly, extensive experiments on diverse fashion tasks—including cross-modal retrieval, composed image retrieval, and multimodal generation—demonstrate that our unified model significantly surpasses previous state-of-the-art methods.

## 2 Preliminaries and Related Works

## 2.1 Fashion Tasks

Fashion tasks encompass a range of image and language manipulations, including cross-modal retrieval, composed image retrieval, fashion image captioning and generation, etc. The representative tasks can be briefly divided into the following two groups.

Fashion Retrieval. It generally consists of Cross-Modal Retrieval (CMR) (Ma et al., 2022; Rostamzadeh et al., 2018) and composed image retrieval (CIR) tasks (Baldrati et al., 2023a; Bai et al., 2023). CMR requests to efficiently retrieve the most matched image/sentence from a large candidate pool given a text/image query. CIR is a special type of image retrieval with a multimodal query (a combination of a reference image and a modifying text) matched against a set of images. It retrieves a target image from a vast image database based on a reference image and a text description detailing changes to be applied to the reference image. In this scenario, a query pair $p = \{ I _ { R } , t \}$ is provided, where $I _ { R }$ is the reference image and t is the text describing the desired modifications. The challenge for this task is to accurately identify the target image $I _ { T }$ that best matches the query among all potential candidates in the image corpus .

Fashion Generation. It consists of Fashion Image Captioning (FIC) and Fashion Image Generation (FIG). FIC (Yang et al., 2020) aims to generate a descriptive caption for a product based on the visual and/or textual information provided in the input. FIG aims to generate images based on the multimodal input, such as try-on (Choi et al., 2021; Gou et al., 2023) and fashion design (Baldrati et al., 2023b).

## 2.2 Multimodal Language Models

Recent research has witnessed a surge of interest in multimodal LLMs, including collaborative models (Wu et al., 2023; Yang et al., 2023b; Shen et al., 2023) and end-to-end methods (Alayrac et al., 2022; Zhao et al., 2024; Li et al., 2022; Bao et al., 2021; Wang et al., 2022b,a,a). More recently, some works also explore training LLMs with parameterefficient tuning (Li et al., 2023b; Zhang et al., 2023b) and instruction tuning (Dai et al., 2023; Liu et al., 2023a; Ye et al., 2023; Zhu et al., 2023a; Li et al., 2023a). They only focus on generation tasks, while our model UniFashion is designed as a unified framework that enables both retrieval and generation tasks.

## 2.3 Diffusion Models

Diffusion generative models (Rombach et al., 2022; Ramesh et al., 2021; Nichol et al., 2022; Ruiz et al., 2023) have achieved strong results in text conditioned image generation works. Among contemporary works that aim to condition pretrained latent diffusion models, ControlNet (Zhang et al., 2023a) proposes to extend the Stable Diffusion model with an additional trainable copy part for conditioning input. In this work, we focus on the fashion domain and propose a unified framework that can leverage latent diffusion models that directly exploit the conditioning of textual sentences and other modalities such as human body poses and garment sketches.

## 2.4 Problem Formulation

Existing fashion image retrieval and generation methods are typically designed for specific tasks, which inherently restricts their applicability to the various task forms and input/output forms in the fashion domain. To train a unified model that can handle multiple fashion tasks, our approach introduces a versatile framework capable of handling multiple fashion tasks by aligning the multimodal representation into the LLM and the diffusion model. This innovative strategy enhances the model’s adaptability, and it can be represented as:

$$
I _ { \mathrm { o u t } } , T _ { \mathrm { o u t } } = \mathcal { F } _ { \mathcal { T } _ { \mathrm { R e t } } , \mathcal { T } _ { \mathrm { G e n } } } ( I _ { \mathrm { i n } } , T _ { \mathrm { i n } } ; \Theta ) ,\tag{1}
$$

where $\mathcal { F } _ { \mathcal { T } }$ represents the unified model parameterized by Θ, it consists of retrieval module $\mathcal { T } _ { R e t }$ and generative module $\tau _ { G e n }$

## 3 Proposed Model: UniFashion

In this section, we introduce the UniFashion to unify the fashion retrieval and generation tasks into a single model. By combining retrieval and generative modules, the proposed UniFashion employs a two-stage training strategy to capture relatedness between image and language information. Consequently, it can seamlessly switch between two operational modes for cross-modal tasks and composed modal tasks.

## 3.1 Phase 1: Cross-modal Pre-training

In the first stage, we conduct pre-training on the retrieval and generative modules to equip the Large Language Model (LLM) and diffusion model with strong cross-modal fashion representation capabilities for the next phase.

## 3.1.1 Cross-modal Retrieval

For cross-modal retrieval tasks, given a batch of image caption pairs $p = \{ I , C \}$ , we first calculate their unimodal representations using an independent method. In particular, we adopt a lightweight Querying Transformer, i.e., Q-Former in BLIP-2 (Li et al., 2023b), to encode the multimodal inputs, as it is effective in bridging the modality gap. To avoid information leaks, we employ a unimodal self-attention mask (Li et al., 2023b), where the queries and text are not allowed to see each other:

$$
\begin{array} { r l } & { Z _ { I } = \mathrm { Q – F o r m e r } ( I , q ) , } \\ & { Z _ { C } = \mathrm { Q – F o r m e r } ( C ) . } \end{array}\tag{2}
$$

where the output sequence $Z _ { I }$ is the encoding result of an initialized learnable query q with the input image and $Z _ { C }$ is the encoded caption, which contains the embedding of the output of the [CLS] token $e _ { c l s } ,$ , which is a representation of the input caption text. Since $Z _ { I }$ contains multiple output embeddings (one from each query), we first compute the pairwise similarity between each query output and $e _ { c l s } .$ , and then select the highest one as the imagetext similarity. In our experiments, we employ 32 queries in $q ,$ with each query having a dimension of 768, which is the same as the hidden dimension of the Q-Former. For cross-modal learning objective, we leverage the Image-Text Contrastive Learning (ITC) and Image-Text Matching (ITM) method.

The first loss term is image-text contrastive loss, which has been widely adopted in existing text-toimage retrieval models. Specifically, the image-text contrastive loss is defined as:

$$
\mathcal { L } _ { \mathrm { I T C } } ( X , Y ) = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp [ \lambda ( X _ { i } ^ { T } \cdot Y ^ { i } ) ] } { \sum _ { j = 1 } ^ { B } \exp [ \lambda ( X _ { i } ^ { T } \cdot Y ^ { j } ) ] } ,\tag{3}
$$

where λ is a learnable temperature parameter. ITM aims to learn fine-grained alignment between image and text representation. It is a binary classification task where the model is asked to predict whether an image-text pair is positive (matched) or negative (unmatched), it is defined as,

$$
\mathcal { L } _ { \mathrm { I T M } } ( X , Y ) = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp f _ { \theta } ( X _ { i } , Y _ { i } ) } { \sum _ { j = 1 } ^ { B } \exp f _ { \theta } ( X _ { j } , Y _ { i } ) } ,\tag{4}
$$

Then, we maximize their similarities via symmetrical contrastive loss:

$$
\mathcal { L } _ { \mathrm { c r o s s } } = \mathcal { L } _ { \mathrm { I T C } } ( t _ { c } , Z _ { I } ) + \mathcal { L } _ { \mathrm { I T M } } ( Z _ { C } , Z _ { I } ) ,\tag{5}
$$

## 3.1.2 Cross-modal Generation

As depicted in Fig. 2, after the learnable queries q pass through the multimodal encoder, they are capable of integrating the visual information with textual guidance. However, in Section 3.1.1, we did not specify a learning target for q. Empirically, the $q$ that has been merged with the reference image and edited text information should be equivalent to the encoding of the target image. This implies that we should be able to reconstruct the target image and its caption based on $q .$ In this section, we will employ generative objectives to improve the representation of augmented q.

In the first stage, we connect the Q-Former (equipped with a frozen image encoder) to a Large Language Model (LLM) to harness the LLM’s prowess in language generation, and to a diffusion model to exploit its image generation capabilities. Notably, we exclusively train the model using image-text pairs throughout this process. As depicted in Figure 2, we employ a Task Specific Adapter (TSA) layer to linearly project the output query embeddings q to match the dimensionality of the embeddings used by the LLM and diffusion model. In this stage, we freeze the parameters of the Q-Former and fine-tune only the adapter layers, connecting LLM and diffusion models. This approach allows us to develop a discriminative model that can evaluate whether queries q can generate the target image and its corresponding caption.

![](images/42d846cb3b6f1bbb4cd0067c73def9840fec8d18cc894dd9e7e6c616b9af943a.jpg)  
Figure 2: Overview of the training framework of our UniFashion model. Phase 1 - Cross-modal Pre-training: UniFashion acquires robust cross-modal fashion representation capabilities through pre-training, leveraging both the language model and the diffusion model. Phase 2 - Composed Multimodal Fine-tuning: The model undergoes fine-tuning to process both image and text inputs, refining its ability to learn composed modal representations. This is achieved by aligning the multimodal encoder with the LLM and the diffusion model for enhanced performance.

Target Caption Generation. The adapter layer is placed before the LLM to map the output of Q-Former to the text embedding space of the LLM. To synchronize the space of Q-Former with that of the LLM, we propose to use the image-grounded text generation (ITG) objective to drive the model to generate texts based on the input image by computing the auto-regressive loss:

$$
\mathcal { L } _ { \mathrm { I T G } } = - \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \log p _ { \phi } ( w _ { l } ^ { g } | w _ { < l } ^ { g } , f _ { \theta } ( q ) ) ,\tag{6}
$$

where $w ^ { g } = ( w _ { 1 } ^ { g } , . . . , w _ { L } ^ { g } )$ represents the groundtruth caption of image I with length L, $q \quad =$ Q-Former $( I , q )$ , ϕ denotes the LLM’s parameters, and θ denotes the text adapter layers’ parameters.

Target Image Generation. In the first stage, our task also aims to reconstruct the image $\hat { I _ { T } }$ from $q .$ As in standard latent diffusion models, given an encoded input x, the proposed denoising network is trained to predict the noise stochastically added to x. The corresponding objective function can be specified as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { q 2 I } } = \mathbb { E } _ { \epsilon ^ { y } , \mathbf { x } _ { 0 } } [ \| \epsilon ^ { x } - \epsilon _ { \eta } ^ { x } ( \mathbf { x } _ { t ^ { x } } , f _ { \zeta } ( q ) , t ^ { x } ) \| ^ { 2 } ] , } \end{array}\tag{7}
$$

where η denotes the u-net models’ parameters and ζ denotes the image adapter layers’ parameters. The overall loss in the first stage can be expressed:

$$
\mathcal { L } _ { \mathrm { p h } 1 } = \mathcal { L } _ { \mathrm { c r o s s } } + \mathcal { L } _ { \mathrm { I T G } } + \mathcal { L } _ { \mathrm { q 2 T } } .\tag{8}
$$

After the first training stage, we can leverage the LLM and diffusion model as discriminators to guide the generation of composed queries.

## 3.2 Phase 2: Composed Multimodal Fine-tuning

In this phase, the inputs are reference image and guidance text, and we fine-tune the model for composed multimodal retrieval and generation tasks.

## 3.2.1 Composed Image Retrieval

For CIR task, the target image $I _ { T }$ generally encompasses the removal of objects and the modification of attributes in the reference image. To solve this problem, as depicted in Fig. 2, the multimodal encoder is utilized to extract features from the reference image and the guide text. It joint embeds the given pair $p = \{ I _ { R } , t \}$ in a sequential output. Specifically, a set of learnable queries $q$ concatenated with text guidance t is introduced to interact with the features of the reference image. Finally, the output of Q-Former is the multimodal synthetic prompt $Z _ { R }$ . We use a bi-directional self-attention mask, similar to the one used in BLIP2 (Li et al., 2023b), where all queries and texts can attend to each other. The output query embeddings $Z _ { R }$ thus capture multimodal information:

$$
\begin{array} { r l } & { Z _ { R } = \mathrm { Q - F o r m e r } ( I _ { R } , t , q _ { R } ) , } \\ & { Z _ { T } = \mathrm { Q - F o r m e r } ( I _ { T } , q _ { T } ) . } \end{array}\tag{9}
$$

Noting that the output sequence $Z _ { R }$ consists of learnable queries q and encoded text guidance t, which includes $e _ { c l s }$ , the embedding of the output of the [CLS] token. On the other hand, the target image’s output sequence $Z _ { T }$ consists only of learnable queries. Therefore, we can use $Z _ { R }$ as a representation that incorporates information from the reference image and the guidance text and align it with the features of the target image $Z _ { T }$ . Moreover, as UniFashion acquires the ability to generate captions for images from Sec. 3.1.2, we can generate captions for the candidate images and use $e _ { c l s }$ to retrieve the caption $Z _ { C }$ of the target image. Then, the final contrastive loss for the CIR task is:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c i r } } = \mathcal { L } _ { \mathrm { I T C } } ( e _ { c l s } , Z _ { T } ) + \mathcal { L } _ { \mathrm { I T C } } ( e _ { c l s } , Z _ { C } ) } \\ { + \mathcal { L } _ { \mathrm { I T M } } ( \pmb { t } , Z _ { T } ) , \qquad } \end{array}\tag{10}
$$

## 3.2.2 Composed Multimodal Generation

For these generation tasks, we freeze the LLM parameters and tune the parameters of the taskspecific adapters, the diffusion model, and the $\mathrm { Q } \mathrm { - }$ Former. The loss function for the target image’s caption generation is formulated in a way that is similar to Eq. 6:

$$
\mathcal { L } _ { \Pi \mathrm { G } } = - \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \log p _ { \phi } ( w _ { l } ^ { g } | w _ { < l } ^ { g } , f _ { \theta } ( q _ { R } ) ) ,\tag{11}
$$

The loss function for the target image generation is formulated in a way that is similar to Eq. 7:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { q 2 I } } = \mathbb { E } _ { \epsilon ^ { y } , { \bf x } _ { 0 } } [ \| \epsilon ^ { x } - \epsilon _ { \eta } ^ { x } ( { \bf x } _ { t ^ { x } } , f _ { \zeta } ( q _ { R } ) , t ^ { x } ) \| ^ { 2 } ] , } \end{array}\tag{12}
$$

The overall loss in the second stage can be expressed as:

$$
\mathcal { L } _ { \mathrm { s t a g e 2 } } = \mathcal { L } _ { \mathrm { c i r } } + \mathcal { L } _ { \mathrm { I T G } } + \mathcal { L } _ { \mathrm { q 2 I } } .\tag{13}
$$

## 3.3 Instruction-Tuning LLMs for Different Caption Style

Liu et al.’s work shows that LLMs have the potential to handle multimodal tasks based on text description of images. Due to the different styles of captions in different fashion datasets, we adopt different instructions to tune the LLM so that it can generate captions of different styles.

We designed different instructions for different datasets and tasks, as shown in Table 7. General instruction template is denoted as follows:

USER: <Img><queries></Img> + Instruction. Assistant: <answer>.

For the <image> placeholder, we substitute it with the output of Multimodal Encoder. To avoid overfitting to the specific task and counteract the model’s inclination to generate excessively short outputs, we have devised specific instructions, which enable the LLM to produce concise responses when necessary.

## 4 Experiments

## 4.1 Experimental Setup

We initialize the multimodal encoder using BLIP2’s Q-Former. Following the approach of LLaVA-1.5 (Liu et al., 2023a), we initialize the LLM from Vicuna-1.5 (Zheng et al., 2023). For the diffusion module, we adopt the autoencoder and denoising U-Net from Stable Diffusion v1.4, as utilized in StableVITON. The weights of the U-Net are initialized from Paint-by-Example. To achieve more refined person textures, we employ a VAE that has been fine-tuned on the VITONHD dataset, as done in StableVITON. The statistics of the two-stage datasets can be found in Table 6. For cross-modal retrieval, we evaluated UniFashion on FashionGen validation set. For the image captioning task, UniFashion is evaluated in the Fashion-Gen dataset. For the composed image retrieval task, we evaluated the Fashion-IQ validation set. To maintain consistency with previous work, for the composed image generation task, we fine-tuned UniFashion and evaluated it on the VITON-HD and MGD datasets. More details can be found in Appendix B.

Phase 1: For multimodal representation learning, we follow BLIP2 and pretrain the Q-Former on fashion image-text pairs. To adapt the model for multimodal generation, we freeze the parameters of Q-Former and fine-tune the MLLM and diffusion model with their task specific adapters separately. Due to the different styles of captions in different fashion datasets, we adopt the approach of instruction tuning to train the LLM so that it can generate captions of different styles. More details can be found in Appendix 3.3.

Phase 2: In order to make UniFashion have the composed retrieval and generation abilities, we freeze the parameters of LLM and diffusion model, only fine-tune the multimodal encoder.

<table><tr><td rowspan="2">Model</td><td colspan="3">Image to Text</td><td colspan="3">Text to Image</td><td rowspan="2">Mean</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>FashionBERT (Li et al., 2022)</td><td>23.96</td><td>46.31</td><td>52.12</td><td>26.75</td><td>46.48</td><td>55.74</td><td>41.89</td></tr><tr><td>OSCAR (Alayrac et al., 2022)</td><td>23.39</td><td>44.67</td><td>52.55</td><td>25.10</td><td>49.14</td><td>56.68</td><td>41.92</td></tr><tr><td>KaledioBERT (Li et al., 2023b)</td><td>27.99</td><td>60.09</td><td>68.37</td><td>33.88</td><td>60.60</td><td>68.59</td><td>53.25</td></tr><tr><td>EI-CLIP (Li et al., 2023b)</td><td>38.70</td><td>72.20</td><td>84.25</td><td>40.06</td><td>71.99</td><td>82.90</td><td>65.02</td></tr><tr><td>MVLT (Dai et al., 2023)</td><td>33.10</td><td>77.20</td><td>91.10</td><td>34.60</td><td>78.00</td><td>89.50</td><td>67.25</td></tr><tr><td>FashionViL (Zhu et al., 2023a)</td><td>65.54</td><td>91.34</td><td>96.30</td><td>61.88</td><td>87.32</td><td>93.22</td><td>82.60</td></tr><tr><td>FAME-ViL (Liu et al., 2023a)</td><td>65.94</td><td>91.92</td><td>97.22</td><td>62.86</td><td>87.38</td><td>93.52</td><td>83.14</td></tr><tr><td>UniFashion (Ours)</td><td>71.44</td><td>93.79</td><td>97.51</td><td>71.41</td><td>93.69</td><td>97.47</td><td>87.55</td></tr></table>

Table 1: Performance comparison of UniFashion and baseline models on the FashionGen dataset for cross-modal retrieval tasks.

<table><tr><td rowspan="2">Model</td><td colspan="4">Image Captioning</td></tr><tr><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr</td></tr><tr><td>FashionBERT</td><td>3.30</td><td>9.80</td><td>29.70</td><td>30.10</td></tr><tr><td>OSCAR</td><td>4.50</td><td>10.90</td><td>30.10</td><td>30.70</td></tr><tr><td>KaleidoBERT</td><td>5.70</td><td>12.80</td><td>32.90</td><td>32.60</td></tr><tr><td>FashionViL</td><td>16.18</td><td>25.60</td><td>37.23</td><td>39.30</td></tr><tr><td>FAME-ViL</td><td>30.73</td><td>25.04</td><td>55.83</td><td>150.4</td></tr><tr><td>UniFashion</td><td>35.53</td><td>29.32</td><td>54.59</td><td>169.5</td></tr></table>

Table 2: The Performance of UniFashion in the image captioning task on the FashionGen dataset.

## 4.2 Datasets

We test the effectiveness of UniFashion by experimenting on different tasks including fashion image captioning, cross-modal retrieval, composed image retrieval and composed image generation.

We use the FashionGen and FshaionIQ (Lin et al., 2014) datasets for retrieval tasks. Fashion-Gen contains 68k fashion products accompanied by text descriptions. Each product includes 1 - 6 images from different angles, resulting in 260.5k image-text pairs for training and 35.5k for testing. Fashion-IQ contains 18k training triplets (that is, reference image, modifying text, target image) and 6k validation triplets over three categories: Dress, Shirt, and Toptee. Each pair (reference image, target image) is manually annotated with two modifying texts, which are concatenated.

For fashion image captioning tasks, we utilize the FashionGen (Zang et al., 2021) dataset. Additionally, to enhance our model’s capability in the CIR task, which involves the ability to retrieve captions for target images, we have annotated images from the training set of Fashion-IQ. Recognizing that manually annotating all the images would be time-consuming and resource-intensive, we draw inspiration from the success of recent MLLM models such as LLaVA in text-annotation tasks, and propose leveraging LLaVA 1.5 (13B) to semi-automatically annotate the dataset. More details can be found in Appendix C.

## 4.3 Evaluation Methods

We compare our models with previous state-of-theart methods on each task. For extensive and fair comparisons, all prior competitors are based on large-scale pre-trained models.

Cross-modal Retrieval Evaluation. We consider both image-to-text retrieval and text-to-image retrieval with random 100 protocols used by previous methods. 100 candidates are randomly sampled from the same category to construct a retrieval database. The goal is to locate the positive match depicting the same garment instance from these 100 same-category negative matches. We utilize Recall@K as the evaluation metric, which reflects the percentage of queries whose true target ranked within the top K candidates.

Fashion Image Captioning Evaluation. For evaluating the performance of caption generation, we utilize BLEU-4, METEOR, ROUGE-L, and CIDEr as metrics.

Composed Fashion Image Retrieval Evaluation. We compare our UniFashion with CIR methods and the FAME-ViL model of V + L that is oriented towards fashion in the original protocol used by Fashion-IQ. For this task, we also utilize Recall@K as the evaluation metric.

Composed Fashion Image Generation Evaluation. We compare our UniFashion with try-on methods on VITON-HD dataset and fashion design works on MGD dataset. To evaluate the quality of image generation, we use the Frechet Inception Distance (FID) score to measure the divergence between two multivariate normal distributions and employ the CLIP Score (CLIP-S) provided in the TorchMetrics library to assess the adherence of the image to the textual conditioning input (for fashion design task).

<table><tr><td rowspan="3">Model</td><td colspan="4">Modalities</td><td colspan="3">Metrics</td></tr><tr><td>Text</td><td>Sketch</td><td>Pose</td><td>Cloth</td><td>FID↓</td><td>KID↓</td><td>CLIP-S</td></tr><tr><td>try-on task</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VITON-HD (Choi et al., 2021)</td><td>x</td><td>x</td><td>√</td><td>√</td><td>12.12</td><td>3.23</td><td></td></tr><tr><td>Paint-by-Example (Yang et al., 2023a)</td><td>x</td><td>x</td><td>√</td><td>√</td><td>11.94</td><td>3.85</td><td></td></tr><tr><td>GP-VTON (Xie et al., 2023)</td><td>x</td><td>x</td><td>√</td><td>√</td><td>13.07</td><td>4.66</td><td></td></tr><tr><td>StableVITON (Kim et al., 2024)</td><td>x</td><td>x</td><td>√</td><td>√</td><td>8.23</td><td>0.49</td><td></td></tr><tr><td>UniFashion (Ours)</td><td>x</td><td>x</td><td>√</td><td>√</td><td>8.42</td><td>0.67</td><td></td></tr><tr><td>fashion design task</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SDEdit (Meng et al., 2021)</td><td>√</td><td>√</td><td>√</td><td>x</td><td>15.12</td><td>5.67</td><td>28.61</td></tr><tr><td>MGD (Baldrati et al., 2023b)</td><td>√</td><td>√</td><td>√</td><td>x</td><td>12.81</td><td>3.86</td><td>30.75</td></tr><tr><td>UniFashion (Ours)</td><td>√</td><td>√</td><td>√</td><td>x</td><td>12.43</td><td>3.74</td><td>31.29</td></tr></table>

Table 3: Performance analysis of unpaired settings on the VITON-HD and MGD datasets across different input modalities.
<table><tr><td rowspan="2">Model</td><td colspan="2">Dress</td><td colspan="2">Shirt</td><td colspan="2">Toptee</td><td colspan="3">Average</td></tr><tr><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>Avg.</td></tr><tr><td>FashionVLP (Goenka et al., 2022)</td><td>32.42</td><td>60.29</td><td>31.89</td><td>58.44</td><td>38.51</td><td>68.79</td><td>34.27</td><td>62.51</td><td>48.39</td></tr><tr><td>CASE (Levy et al., 2023)</td><td>47.44</td><td>69.36</td><td>48.48</td><td>70.23</td><td>50.18</td><td>72.24</td><td>48.79</td><td>70.68</td><td>59.74</td></tr><tr><td>AMC (Żhu et al., 2023b)</td><td>31.73</td><td>59.25</td><td>30.67</td><td>59.08</td><td>36.21</td><td>66.06</td><td>32.87</td><td>61.64</td><td>47.25</td></tr><tr><td>CoVR-BLIP (Ventura et al., 2024)</td><td>44.55</td><td>69.03</td><td>48.43</td><td>67.42</td><td>52.60</td><td>74.31</td><td>48.53</td><td>70.25</td><td>59.39</td></tr><tr><td>MGUR (Chen et al., 2022)</td><td>32.61</td><td>61.34</td><td>33.23</td><td>62.55</td><td>41.40</td><td>72.51</td><td>35.75</td><td>65.47</td><td>50.61</td></tr><tr><td>LinCIR (Gu et al., 2024)</td><td>38.08</td><td>60.88</td><td>46.76</td><td>65.11</td><td>50.48</td><td>71.09</td><td>45.11</td><td>65.69</td><td>55.4</td></tr><tr><td>CMAP (♀i et al., 2024)</td><td>36.44</td><td>64.25</td><td>34.83</td><td>60.06</td><td>41.79</td><td>69.12</td><td>37.64</td><td>64.42</td><td>51.03</td></tr><tr><td>CLIP4CIR (Baldrati et al., 2023a)</td><td>33.81</td><td>59.40</td><td>39.99</td><td>60.45</td><td>41.41</td><td>65.37</td><td>38.32</td><td>61.74</td><td>50.03</td></tr><tr><td>FAME-ViL (Han et al., 2023)</td><td>42.19</td><td>67.38</td><td>47.64</td><td>68.79</td><td>50.69</td><td>73.07</td><td>46.84</td><td>69.75</td><td>58.29</td></tr><tr><td>TG-CIR (Wen et al., 2Ó23)</td><td>45.22</td><td>69.66</td><td>52.60</td><td>72.52</td><td>56.14</td><td>77.10</td><td>51.32</td><td>73.09</td><td>58.05</td></tr><tr><td>Re-ranking (Liu et al., 2023b)</td><td>48.14</td><td>71.43</td><td>50.15</td><td>71.25</td><td>55.23</td><td>76.80</td><td>51.17</td><td>73.13</td><td>62.15</td></tr><tr><td>SPRC (Bai et al., 2023)</td><td>49.18</td><td>72.43</td><td>55.64</td><td>73.89</td><td>59.35</td><td>78.58</td><td>54.92</td><td>74.97</td><td>64.85</td></tr><tr><td>UniFashion w/o cap</td><td>49.65</td><td>72.17</td><td>56.88</td><td>74.12</td><td>59.29</td><td>78.11</td><td>55.27</td><td>74.80</td><td>65.04</td></tr><tr><td>UniFashion w/o img</td><td>32.49</td><td>49.11</td><td>44.70</td><td>59.63</td><td>43.16</td><td>60.26</td><td>40.12</td><td>56.33</td><td>48.22</td></tr><tr><td>UniFashion</td><td>53.72</td><td>73.66</td><td>61.25</td><td>76.67</td><td>61.84</td><td>80.46</td><td>58.93</td><td>76.93</td><td>67.93</td></tr></table>

Table 4: Comparative evaluation of UniFashion and variants and baseline models on the Fashion-IQ dataset for composed image retrieval task. Best and second-best results are highlighted in bold and underlined, respectively.

<table><tr><td>Model</td><td>CMR</td><td>CIR</td><td>FIC</td><td>FIG</td></tr><tr><td>Base</td><td>87.38</td><td>64.76</td><td></td><td></td></tr><tr><td>Base+LLM</td><td>87.49</td><td>65.04</td><td>36.21</td><td></td></tr><tr><td>Base+LLM w/ cap</td><td>87.49</td><td>66.83</td><td>36.21</td><td></td></tr><tr><td>Base+LLM+diff.</td><td>87.55</td><td>67.93</td><td>35.53</td><td>12.43</td></tr></table>

Table 5: Ablation study and analysis of UniFashion across FashionGen, Fashion-IQ, and VITON-HD Datasets. Metrics reported include average image-totext and text-to-image recall for cross-modal retrieval (CMR), average recall for composed image retrieval (CIR), BLEU-4 for Fashion Image Captioning, and FID for Fashion image generation (FIG).

## 4.4 Comparative Analysis of Baselines and Our Method

UniFashion exhibits superior performance across all datasets compared to baselines. Tab. 1 presents the evaluation results for each baseline and our models in FashionGen data sets for crossmodal retrieval. UniFashion outperforms most of the baseline models on both the text-to-image and image-to-text tasks. Following FAME-ViL, we also adopt a more challenging and practical protocol that conducts retrieval on the entire product set, which is in line with actual product retrieval scenarios. In Tab. 2, we performed a comparison between our UniFashion and other baselines on the FashionGen dataset for the image captioning task. By integrating the powerful generative ability of the LLM, our model performed significantly better than the traditional multimodal models in this task. In Tab. 4, we conducted a comparison between our UniFashion and CIR-specialist methods. Our findings are in line with those of Tab. 1.

After fine-tuning UniFashion on image generation/editing tasks with multimodal inputs, it exhibits outstanding performance. Tab. 3 evaluates the quality of the generated image of UniFashion in the VITON-HD unpaired setting. In order to verify that our model can achieve good results in a variety of modal inputs, we have conducted tests, respectively, on the traditional try-on task and the fashion design task proposed in MGD. For a fair evaluation with baselines, all the models are trained at a 512 × 384 resolution. To confirm the efficacy of our approach, we assess the realism using FID and KID score on all the tasks and using CLIP-S score for fashion design task. As can be seen, the proposed UniFashion model consistently outperforms competitors in terms of realism (i.e., FID and KID) and coherence with input modalities (i.e., CLIP-S), indicating that our method can better encode multimodal information. Meanwhile, although our model is slightly lower than Stable-VITON on the try-on task, this is because we froze the parameters of the diffusion model on the try-on task and only fine-tuned the Q-former part, but it can still achieve top2 results. The visual results can be found in Appendix E.

## 4.5 Ablation Study

UniFashion allows for more flexible execution of multimodal composed tasks. In Tab. 4, we also carry out ablation studies on different retrieval methods. Since UniFashion is capable of generating captions, for the CIR task, we initially utilize UniFashion to generate the captions of candidate images and then conduct the image retrieval task (denoted as UniFashion w/o cap) and the caption retrieval task (denoted as UniFashion w/o img). We find that our single-task variant has already achieved superior performance in the relevant field. Furthermore, due to the generative ability of our model, the pregenerated candidate library optimizes the model’s performance in this task. For specific implementation details, please refer to Appendix C.

We investigate the impact of different modules in UniFashion on various fashion tasks. In Tab. 5, we perform an ablation study on the proposed model architecture, with a focus on LLM and diffusion models. For comparison on the crossmodal retrieval task (CMR), we design the base model as directly fine-tuning BLIP2 without any new modules. The results indicate that the base model performs relatively well on this task and that the introduction of other modules does not lead to significant improvements. However, in the CIR task, the introduction of LLM and diffusion models as supervision can lead to significant improvements, especially when utilizing pregenerated captions by UniFashion to assist in retrieval, resulting in greater benefits. At the same time, we note that, after introducing the diffusion model, it may have some negative impact on the model’s image captioning ability, possibly due to the inherent alignment differences between LLM and the diffusion model.

## 5 Conclusion

We have introduced UniFashion, a unified framework designed to tackle challenges in multimodal generation and retrieval within the fashion domain. By integrating embedding and generative tasks using a diffusion model and LLM, UniFashion enables controllable, high-fidelity generation, significantly outperforming previous single-task state-ofthe-art models across various fashion tasks. Our model’s adaptability in handling complex visionlanguage tasks demonstrates its potential to enhance e-commerce scenarios and fashion-related applications. This study highlights the importance of exploring the learning synergy between multimodal generation and retrieval, offering a promising direction for future research in the fashion domain.

## Limitations

In this section, we discuss limitations of our work and offer further insights into research within the fashion domain.

Computational Requirements. UniFashion integrates multiple complex modules, including Q-Former, LLM, and diffusion models, which result in higher computational complexity during training. However, during the inference stage, the computational complexity of UniFashion is comparable to that of current state-of-the-art models. For retrieval tasks, only the Q-Former module is needed to calculate the similarity between the input image or text and the pre-stored candidate features in the database, eliminating the need to utilize the LLM and diffusion model components for inference. For composed image generation tasks, such as fashion design, our model relies on diffusion processes, which may take longer. In our experiments, we tested the performance of our model on an A100 (80G) GPU. During inference, using 1000 examples from the VITON-HD dataset, UniFashion took approximately 3.15 seconds per image generation. We believe exploring more efficient sampling methods, such as DPM-Solver++ (Lu et al., 2022), could improve the overall efficiency of UniFashion.

## Acknowledgements

We thank the anonymous reviewers for their valuable feedback. This research was partially supported by the grant of HK ITF ITS/359/21FP.

## References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736.

Yang Bai, Xinxing Xu, Yong Liu, Salman Khan, Fahad Khan, Wangmeng Zuo, Rick Siow Mong Goh, and Chun-Mei Feng. 2023. Sentence-level prompts benefit composed image retrieval. arXiv preprint arXiv:2310.05473.

Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. 2022. Effective conditioned and composed image retrieval combining clip-based features. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21466–21474.

Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. 2023a. Composed image retrieval using contrastive learning and task-oriented clip-based features. ACM Transactions on Multimedia Computing, Communications and Applications, 20(3):1–24.

Alberto Baldrati, Davide Morelli, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. 2023b. Multimodal garment designer: Humancentric latent diffusion models for fashion image editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23393– 23402.

Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. 2021. Beit: Bert pre-training of image transformers. In International Conference on Learning Representations.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Yiyang Chen, Zhedong Zheng, Wei Ji, Leigang Qu, and Tat-Seng Chua. 2022. Composed image retrieval with text feedback via multi-grained uncertainty regularization. arXiv preprint arXiv:2211.07394.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An opensource chatbot impressing gpt-4 with 90%\* chatgpt quality.

Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. 2021. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14131– 14140.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. 2023. Instructblip: Towards general-purpose vision-language models with instruction tuning.

Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. 2023. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499.

Yujie Feng, Zexin Lu, Bo Liu, Liming Zhan, and Xiao-Ming Wu. 2023. Towards llm-driven dialogue state tracking. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 739–755.

Dehong Gao, Linbo Jin, Ben Chen, Minghui Qiu, Peng Li, Yi Wei, Yi Hu, and Hao Wang. 2020. Fashionbert: Text and image matching with adaptive loss for cross-modal retrieval. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2251–2260.

Sonam Goenka, Zhaoheng Zheng, Ayush Jaiswal, Rakesh Chada, Yue Wu, Varsha Hedau, and Pradeep Natarajan. 2022. Fashionvlp: Vision language transformer for fashion retrieval with feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14105–14115.

Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. 2023. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In Proceedings of the 31st ACM International Conference on Multimedia, pages 7599–7607.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 6904–6913.

Geonmo Gu, Sanghyuk Chun, Wonjae Kim, Yoohoon Kang, and Sangdoo Yun. 2024. Language-only training of zero-shot composed image retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13225–13234.

Xiao Han, Xiatian Zhu, Licheng Yu, Li Zhang, Yi-Zhe Song, and Tao Xiang. 2023. Fame-vil: Multi-tasking vision-language model for heterogeneous fashion tasks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2669–2680.

Xintong Han, Zuxuan Wu, Phoenix X Huang, Xiao Zhang, Menglong Zhu, Yuan Li, Yang Zhao, and Larry S Davis. 2017. Automatic spatially-aware fashion concept discovery. In Proceedings of the IEEE international conference on computer vision, pages 1463–1471.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840– 6851.

Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. 2024. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8176–8185.

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. 2017. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal ofcomputer vision, 123:32–73.

Matan Levy, Rami Ben-Ari, Nir Darshan, and Dani Lischinski. 2023. Data roaming and early fusion for composed image retrieval. arXiv preprint arXiv:2303.09429.

Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. 2023a. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023b. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR.

Shenshen Li, Xing Xu, Xun Jiang, Fumin Shen, Zhe Sun, and Andrzej Cichocki. 2024. Cross-modal attention preservation with self-contrastive learning for composed query-based image retrieval. ACM Transactions on Multimedia Computing, Communications and Applications, 20(6):1–22.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023a. Visual instruction tuning. arXiv preprint arXiv:2304.08485.

Qijiong Liu, Xiaoyu Dong, Jiaren Xiao, Nuo Chen, Hengchang Hu, Jieming Zhu, Chenxu Zhu, Tetsuya Sakai, and Xiao-Ming Wu. 2024a. Vector quantization for recommender systems: A review and outlook. arXiv preprint arXiv:2405.03110.

Qijiong Liu, Jieming Zhu, Quanyu Dai, and Xiao-Ming Wu. 2022. Boosting deep ctr prediction with a plugand-play pre-trainer for news recommendation. In Proceedings ofthe 29th International Conference on Computational Linguistics, pages 2823–2833.

Qijiong Liu, Jieming Zhu, Yanting Yang, Quanyu Dai, Zhaocheng Du, Xiao-Ming Wu, Zhou Zhao, Rui Zhang, and Zhenhua Dong. 2024b. Multimodal pretraining, adaptation, and generation for recommendation: A survey. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 6566–6576.

Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, and Stephen Gould. 2021. Image retrieval on real-life images with pre-trained vision-and-language models. in 2021 ieee. In CVF International Conference on Computer Vision (ICCV)(2021), pages 2105–2114.

Zheyuan Liu, Weixuan Sun, Damien Teney, and Stephen Gould. 2023b. Candidate set re-ranking for composed image retrieval with dual multi-modal encoder. arXiv preprint arXiv:2305.16304.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2022. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095.

Haoyu Ma, Handong Zhao, Zhe Lin, Ajinkya Kale, Zhangyang Wang, Tong Yu, Jiuxiang Gu, Sunav Choudhary, and Xiaohui Xie. 2022. Ei-clip: Entityaware interventional contrastive learning for ecommerce cross-modal retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18051–18061.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2021. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073.

Niklas Muennighoff, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Douwe Kiela. 2024. Generative representational instruction tuning. arXiv preprint arXiv:2402.09906.

Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. 2022. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, pages 16784–16804. PMLR.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. 2021. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference

on Computer Vision and Pattern Recognition, pages 10684–10695.

Negar Rostamzadeh, Seyedarian Hosseini, Thomas Boquet, Wojciech Stokowiec, Ying Zhang, Christian Jauvin, and Chris Pal. 2018. Fashion-gen: The generative fashion dataset and challenge. arXiv preprint arXiv:1806.08317.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494.

Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. 2022. A-okvqa: A benchmark for visual question answering using world knowledge. In European Conference on Computer Vision, pages 146–162. Springer.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. arXiv preprint arXiv:2303.17580.

Guangyuan Shi, Qimai Li, Wenlong Zhang, Jiaxin Chen, and Xiao-Ming Wu. 2023. Recon: Reducing conflicting gradients from the root for multi-task learning. arXiv preprint arXiv:2302.11289.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR.

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Lucas Ventura, Antoine Yang, Cordelia Schmid, and Gül Varol. 2024. Covr: Learning composed video retrieval from web video captions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 5270–5279.

Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren

Zhou, and Hongxia Yang. 2022a. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International Conference on Machine Learning, pages 23318–23340. PMLR.

Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. 2022b. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. arXiv preprint arXiv:2208.10442.

Haokun Wen, Xian Zhang, Xuemeng Song, Yinwei Wei, and Liqiang Nie. 2023. Target-guided composed image retrieval. In Proceedings of the 31st ACM International Conference on Multimedia, pages 915– 923.

Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. 2023. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671.

Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, and Rogerio Feris. 2021. Fashion iq: A new dataset towards retrieving images by natural language feedback. In Proceedings ofthe IEEE/CVF Conference on computer vision and pattern recognition, pages 11307–11317.

Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, and Xiaodan Liang. 2023. Gp-vton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23550–23559.

Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. 2023a. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18381–18391.

Xuewen Yang, Heming Zhang, Di Jin, Yingru Liu, Chi-Hao Wu, Jianchao Tan, Dongliang Xie, Jue Wang, and Xin Wang. 2020. Fashion captioning: Towards generating accurate descriptions with semantic rewards. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIII 16, pages 1–17. Springer.

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. 2023b. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381.

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. 2023. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178.

Xiaoxue Zang, Lijuan Liu, Maria Wang, Yang Song, Hao Zhang, and Jindong Chen. 2021. Photochat: A human-human dialogue dataset with photo sharing behavior for joint image-text modeling. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6142–6152.

Haode Zhang, Haowen Liang, Yuwei Zhang, Li-Ming Zhan, Xiao-Ming Wu, Xiaolei Lu, and Albert Lam. 2022. Fine-tuning pre-trained language models for few-shot intent detection: Supervised pre-training and isotropization. In Proceedings of the 2022 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 532–542.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023a. Adding conditional control to text-to-image diffusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 3836–3847.

Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. 2023b. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199.

Xiangyu Zhao, Bo Liu, Qijiong Liu, Guangyuan Shi, and Xiao-Ming Wu. 2024. EasyGen: Easing multimodal generation with BiDiffuser and LLMs. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1351–1370, Bangkok, Thailand. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023a. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592.

Hongguang Zhu, Yunchao Wei, Yao Zhao, Chunjie Zhang, and Shujuan Huang. 2023b. Amc: Adaptive multi-expert collaborative network for text-guided image retrieval. ACM Transactions on Multimedia Computing, Communications and Applications, 19(6):1–22.

Mingchen Zhuge, Dehong Gao, Deng-Ping Fan, Linbo Jin, Ben Chen, Haoming Zhou, Minghui Qiu, and Ling Shao. 2021. Kaleido-bert: Vision-language pre-training on fashion domain. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12647–12657.

## A Basics of Diffusion Models

After the initial proposal of diffusion models by (Sohl-Dickstein et al., 2015), they have demonstrated remarkable capacity for generating highquality and diverse data. DDPM (Ho et al., 2020) connects diffusion and score matching models through a noise prediction formulation, while DDIM (Song et al., 2020) proposes an implicit generative model that generates deterministic samples from latent variables.

Given a data point sampled from a real data distribution $x _ { 0 } \in q ( x )$ , during forward diffusion, x<sub>0</sub> is gradually “corrupted” at each step t by adding Gaussian noise to the output of step t-1. It produces a sequence of noisy samples $\mathbf { x } _ { 1 } , \cdots , \mathbf { x } _ { T }$ . Then, diffusion models learn to reverse the process:

$$
\begin{array} { r } { p ( \mathbf x _ { 0 : T } ) = p ( \mathbf x _ { T } ) \prod _ { t = 1 } ^ { T } p _ { \theta } ( \mathbf x _ { t - 1 } | \mathbf x _ { t } ) , } \\ { p _ { \theta } ( \mathbf x _ { t - 1 } | \mathbf x _ { t } ) = \mathcal N ( \mathbf x _ { t - 1 } ; \mu _ { t } ( \mathbf x _ { t } , t ) , \sigma _ { t } ^ { 2 } \mathbf I ) , } \end{array}\tag{14}
$$

where $p ( \mathbf { x } _ { T } ) \ = \ \mathcal { N } ( \mathbf { x } _ { T } ; 0 , \mathbf { I } )$ is the standard Gaussian distribution and $\mu _ { t } ( \cdot )$ is the parameterization of the predicted mean. Diffusion models are trained to maximize the marginal likelihood of the data $\mathbb { E } [ \log p _ { \theta } ( \mathbf { x } _ { 0 } ) ]$ , and the canonical objective is the variational lower bound of $\log p _ { \theta } ( \mathbf { x } _ { 0 } )$

Stable Diffusion Model. Latent diffusion models (LDMs) operate in the latent space of a pre-trained autoencoder achieving higher computational efficiency while preserving the generation quality. Stable diffusion model is composed of an autoencoder with an encoder E and a decoder D, a conditional U-Net denoising model $\epsilon _ { \theta } .$ and a CLIP-based text encoder. With the fixed encoder $\mathbb { E } ,$ an input image x is first transformed to a lower-dimensional latent space $z _ { 0 } = \mathbb { E } ( x )$ . The decoder D performs the opposite operation, decoding $z _ { \mathrm { 0 } }$ into the pixel space. When considering a latent variable z and its noisy counterpart $z _ { t } ,$ , which is obtained by incrementally adding noises to z over t steps, the latent diffusion models are designed to train the $\epsilon _ { \theta } ( \cdot )$ to predict the added noise ϵ using a standard mean squared error loss:

$$
\mathcal { L } : = \mathbb { E } _ { z , \epsilon , t } [ \| \epsilon - \epsilon _ { \theta } ( \mathbf { z } _ { t } , t ) \| ^ { 2 } ] .\tag{15}
$$

Multimodal Conditional Generation. In the context of our current work, we have a particular focus on the pre-trained multimodal latent diffusion models. For a multimodal conditional generation, given a target image $\mathbf { x } _ { 0 } .$ , the input condition $\mathbf { y } _ { 0 }$ could contain different constraints. The aim is to model the conditional data distribution $q ( \mathbf { x } _ { 0 } | \mathbf { y } _ { 0 } )$ , where $\mathbf { y } _ { 0 }$ contains different modalities prompts. The conditioning mechanism is implemented by first encoding conditional information, then the denoising network $\epsilon _ { \theta }$ conditions on $y _ { 0 }$ via cross-attention. The label $y _ { 0 }$ in a class-conditional diffusion model $\epsilon _ { \theta } ( x _ { t } | y _ { 0 } )$ is replaced with a null label with a fixed probability during training.

<table><tr><td>Data types</td><td>Dataset</td><td>Size</td><td>Stage 1</td><td>Stage 2 |</td><td>Metrics</td></tr><tr><td>CMR</td><td>FashionGen (Lin et al., 2014) Fashion200K (Krishna et al., 2017)</td><td>260.5K 172K</td><td>V V</td><td>V X</td><td>R@K 一</td></tr><tr><td>CIR</td><td>Fashion-IQ (Liu et al., 2023a)</td><td>18K</td><td>X</td><td>V</td><td>R@K</td></tr><tr><td>FIC</td><td>FashionGen (Liu et al., 2023a) Fashion-IQ-Cap</td><td>260.5K 60K</td><td>V V</td><td>V X</td><td>BLEU,CIDEr,METEOR,ROUGE-L</td></tr><tr><td>FIG</td><td>VITON-HD (Goyal et al., 2017) MGD (Schwenk et al., 2022)</td><td>83K 66K</td><td>X X</td><td>V V</td><td>FID, KID FID,KID,CLIP-S</td></tr></table>

Table 6: Description of datasets used in two stages.

![](images/436c4b58db77a77d2deb7d01a529430f42a05c28f62478a21e352e199ce76fae.jpg)  
Figure 3: The architecture of UniFashion for fine-tuning on the image editing task. Firstly, we supply the cloth sketch and text guidance to the multimodal encoder. Then, the diffusion model receives the output of the multimodal encoder, along with the cloth sketches and human features (i.e., agnostic-mask), to subsequently generate the desired images.

## B Implementation Details

LLM During the first phase, due to the flexibility brought by the modular architectural design of BLIP-2, we are able to adapt the model to a broad spectrum of LLMs. In order to effectively utilize the capabilities of the existing MLLM models, we adopted LLaVA-1.5 as the LLM module of the model. Technically, we leverage LoRA to enable a small subset of parameters within UniFashion to be updated concurrently with two layers of adapter during this phase. Specifically, the lora rank is 128 and lora alpha is 256. We utilize the AdamW optimizer with $\beta _ { 0 } = 0 . 9 , \beta _ { 1 } = 0 . 9 9$ , and weight decay of 0. The LLMs are trained with a cosine learning rate of 2e-5 and a warmup rate of 0.03. We use a batch size of 32 for the tuned LLMs.

Diffusion Module We inherit the autoencoder and the denoising U-Net of the Stable Diffusion v1.4. The weights of the U-Net from Paint-by-Example are used to initialize our denoising U-Net. To achieve more refined person texture, a VAE fine-tuned on the VITONHD dataset from StableVITON is utilized. We train the model using an AdamW optimizer with a fixed learning rate of 1e-4 for 360k iterations, employing a batch size of 32. For inference, we employ the pseudo linear multi-step sampler, with the number of sampling steps set to 50.

## C Datasets

For fashion image captioning tasks, we utilize the FashionGen (Zang et al., 2021) dataset. Additionally, to enhance our model’s capability in the CIR task, which involves the ability to retrieve captions for target images, we have annotated images from the training set of Fashion-IQ. Recognizing that manually annotating all the images would be timeconsuming and resource-intensive, we draw inspiration from the success of recent MLLM models such as LLaVA in text-annotation tasks, and propose leveraging LLaVA 1.5 (13B) to semi-automatically annotate the dataset. We perform word lemmatization to reduce each word to its root form. Such pre-processing stage is crucial for the Fashion-IQ dataset, as the captions do not describe a single garment but instead express the properties to modify in a given image to match its target. As shown in Fig. 4, by analysis of the captions in Fashion-IQ, we extracted key words that describe clothing information such as color, sleeve, pattern, lace, etc., as prompts for MLLM (LLaVA 1.5). We then instructed the model to generate the corresponding captions referencing words that match the image features, as shown in Fig. 5. After this process, we got the captions for Fashion-IQ dataset. The trained UniFashion from this dataset (Fashion-IQ-cap) can generate captions for images in the evaluation set of Fashion-IQ to assist in the CIR task. More results can be seen in Fig. 6.

![](images/ab9b9d164ec7f0c21ff405b63a24fba843b48e90433e01fb45f7ae0179e3cd89.jpg)  
Figure 4: Vocabulary of the frequent words scaled by frequency for dresses.

## D Instruction Formats

Due to the disparity in caption styles across different fashion datasets, we employ diverse instructions to fine-tune the LLM, enabling it to generate captions of varying styles. Specifically, the Fashion200K dataset inclines towards providing brief descriptions, the FashionGen dataset is prone to offering professional captions, and in Fashion-IQ-cap, the captions are detailed. Consequently, we have designed distinct instructions for different datasets and tasks, as illustrated in Table 7.

![](images/d36e7e60465144f1908b1ac7a01b9fae590bce73528df87ed05d554a99041efc.jpg)  
Figure 5: Illustration of Instruction-Following Data. The top section displays an image alongside its original captions from Fashion-IQ dataset. The bottom section presents detailed captions generated by LLaVA-1.5. The original captions are not prompts for generation but are provided for comparison with the newly generated caption.

## E Visual Results

Figure 3 illustrates the architecture of UniFashion for fine-tuning on the image editing task. Initially, we input the cloth sketch and text guidance into the multimodal encoder. The diffusion model then receives the output from the multimodal encoder, along with the cloth sketches and human features (such as the agnostic mask), to generate the desired images. We compare UniFashion with the MGD (Baldrati et al., 2023b) model for this task. In Fig. 7, we compare the images generated by our approach with the competitor in the VITON-HD (Choi et al., 2021) paired setting. In Fig. 8, we show the generation effects of UniFashion in the VITON-HD unpaired setting. Our method, unlike the MGD method that employs a warping module to generate input sketches, directly uses in-shop garment sketches and is capable of generating images that align more accurately with the provided captions and cloth sketches.

![](images/de7f546319cf25916507bde478f5cc0f3646113fb76a3efcd726e3187170812c.jpg)

<table><tr><td>Dataset</td><td>Instruction</td></tr><tr><td>Fashion200K</td><td>USER:&lt;image&gt;+Short description. Assistant:</td></tr><tr><td>FashionGen</td><td>USER:&lt;image&gt;+Write a detail and professional description for the cloth. Assistant:</td></tr><tr><td>Fashion-IQ-cap</td><td>USER:&lt;image&gt;+Describe the cloth&#x27;s style, color, design... and other key points. Assistant:</td></tr></table>

Table 7: Examples of task instruction templates.

![](images/aa32c1211380f0cef8f521c9389b19901798fbea9bc9c25be8ce2cfc3dc59307.jpg)

The shirt is black and has a pocket and tailored button tab. It is a short sleeve shirt with a collar. The shirt is made of a fabric that is darker than the pocket and button tab. The shirt is designed to be conservative and modest, with a simple pattern.

![](images/f5bbc3514a11ad598acb9681d74e829353c3d1004575272ef819ea9a5d73926c.jpg)

The dress is a dark green, long, and fitted. It has a collar and is sleeveless. The dress is made of a shiny fabric and has a flowery pattern. The design is elegant and the dress is more revealing than conservative.

The dress is a short, leopard print dress. It is a fitted dress, which is designed to be revealing. The dress is made of a shiny material, which adds to its overall style. The animal print pattern on the dress is the main focus of the design, making it a unique and eye-catching piece.

![](images/af3f7314682838d8eab12b7105276015b0c43de0f3c076baff39395684d84d6e.jpg)

The dress is pink and has a floral pattern. It is a short dress, with a fitted top and a flared bottom. The dress is not revealing, but it is not conservative either. It is a sexy dress, with a collar and a belt. The dress is not crocheted, and it is not loose or tight. It is a colorful dress, and it is elegant.

![](images/e4ca1afe25575ed014dea69a657c42ca638ba485f2f6f70fc658f2018f8e52c2.jpg)

The dress is black and white, with a checkered pattern. It is a short dress, revealing and sexy, with a fitted style. The dress has a collar and is not crocheted. The pattern is smaller and has a more floral style. The dress is elegant and casual, with a flowery design.

![](images/87b2f12b509dae39252bc7a7bf15bb6840ddf3b8445ad11dbbcf16df452516af.jpg)

The dress is black and has a gold top half. It is a short dress with a fitted top and a flared bottom. The dress is sleeveless and has a low neckline, giving it a revealing and sexy appearance. The overall style of the dress is elegant and sophisticated, making it suitable for formal or semi-formal events.

![](images/260cd9d9fef2aa8e05547acc4415c383e00c9748d7dea238ddfb70d5a43f7230.jpg)

A blue and white striped dress. The dress is long and has a fitted style. It is a one-piece dress, and the woman is wearing black heels with it. The dress is not particularly revealing or conservative, but it is not overly sexy or modest either. It is a simple, patterned dress that is neither floral nor plain. The dress is elegant and casual, and it is made of a shiny material.

![](images/fc29b7a0fd94311705bb270b10fbbaff8c4ee8d467181bc48fb5ff587723279b.jpg)

The dress is black with red designs. It is a strapless dress, which is a type of revealing dress. The dress is short, which is a characteristic of a more revealing dress. The dress is patterned, which adds to its visual appeal. The dress is not crocheted, and it does not have a collar. The dress is flowery, which is a characteristic of a more elegant dress. The dress is shiny, which adds to its visual appeal. The dress is darker, which is a characteristic of a more revealing dress. The dress is fitted, which is a characteristic of a more elegant dress

Figure 6: Caption generation results using our method with images from the Fashion-IQ dataset.
<table><tr><td>Model Types</td><td>Task Domain</td><td>Model</td><td>Main Structure</td><td>XMR</td><td>CIR</td><td>Text Generation</td><td>Image Generation</td></tr><tr><td rowspan="2">Cross-modal Retrieval</td><td>General</td><td>CLIP (2021)</td><td>Dual-stream Transfomer</td><td>V</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Fashion</td><td>FashionBERT (2020)</td><td>Single-stream Transfomer</td><td>V</td><td>x</td><td>X</td><td>X</td></tr><tr><td>Multimodal LLM</td><td>General</td><td>LLaVA (2023)</td><td>CLIP, LLM</td><td>X</td><td>X</td><td>V</td><td>X</td></tr><tr><td>Composed Image Retrieval</td><td>General</td><td>SPRC (2024)</td><td>CLIP, Qformer</td><td>X</td><td>V</td><td>X</td><td>X</td></tr><tr><td rowspan="2">Conditional Diffusion</td><td>General</td><td>ControlNet (2023)</td><td>Stable diffusion</td><td>X</td><td>X</td><td>X</td><td>V</td></tr><tr><td>Fashion</td><td>StableVITON (2023)</td><td>Stable diffusion</td><td>X</td><td>x</td><td>X</td><td>V</td></tr><tr><td rowspan="3">Unified Model</td><td>General</td><td>NExT-GPT (2023)</td><td>ImageBind, LLM, Diffusion</td><td>X</td><td>X</td><td>V</td><td>V</td></tr><tr><td>Fashion</td><td>FAME-ViL (2023)</td><td>Dual-stream Transfomer</td><td>V</td><td>V</td><td>V</td><td>X</td></tr><tr><td>General</td><td>BLIP2 (2023)</td><td>CLIP, Qformer, LLM</td><td>V</td><td>X</td><td>√</td><td>X</td></tr><tr><td>Unified Model (Ours)</td><td>Fashion</td><td>UniFashion</td><td>CLIP, Qformer, LLM, Diffusion</td><td>V</td><td>√</td><td>V</td><td>V</td></tr></table>

Table 8: Comparison of different multimodal models. XMR: Cross-modal retrieval tasks; CIR: Compoesd image retrieval task.

Agnostic-mask

Captions

Cloth Sketch

MGD-Generated

UniFashion-Generated

Ground Truth

![](images/947bb3205448c0172d0f9507018ca596cbf73fd94945f3ddff6e9db817838212.jpg)

Figure 7: Qualitative comparison on VITON-HD paired test set. From left to right: agnostic-mask image, caption, cloth sketch, MGD-generated image, UniFashion (ours)-generated image and ground truth. Our method is capable of generating images that align more accurately with the given captions and cloth sketch. For optimal viewing, please zoom in.

Reference Image

Agnostic-mask

Captions

![](images/3db27a2011321dd466187204171c04d98899f442470fe4b08eab677d4ff3778f.jpg)  
white petite t-shirt only macy, white perforated leather front tee, white detail tee

![](images/965d73e8fedab8bd71764973e26665433177ccf1dfc05f9f6dda469753b5d6bb.jpg)  
short-sleeve top only macy, sheer t-shirt, orange slub tee  
high-neck top, longsleeve top, silver high neck jersey top  
black long sleeve eyelash lace top, black long-sleeved lace top, long sleeve lace  
white long-sleeve plisse, front long sleeve bardot, only white and long sleeves  
black petite printed mock-neck top only macy, blue floralprint top, green ray floral-printed blouse

Cloth Sketch

![](images/b6a947683b0a1d9818a94ce10511d4940dcab54ad814f95966bfd8d0a81d14b4.jpg)

MGD-Generated

UniFashion-Generated

![](images/fa59e838d720203bc448457732876a0221d0e9c0362068950933d882c6547874.jpg)

![](images/72f0f320cf2de543f40d66e20a64c9d15f9ce05606389d15d9e2f722689a4efb.jpg)

Figure 8: Qualitative comparison on VITON-HD unpaired test set. From left to right: original image, agnostic-mask image, captions, MGD input sketch, MGD-generated image, UniFashion input sketch and UniFashion (ours)- generated image. Our model is capable of generating images that align more accurately with the provided captions and cloth sketch. For optimal viewing, please zoom in.