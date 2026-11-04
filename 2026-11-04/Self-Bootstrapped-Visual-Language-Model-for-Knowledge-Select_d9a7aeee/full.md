# Self-Bootstrapped Visual-Language Model for Knowledge Selection and Question Answering

Dongze Hao<sup>1,2</sup>, Qunbo Wang<sup>1</sup>, Longteng Guo<sup>1</sup>, Jie Jiang<sup>1</sup>, Jing Liu<sup>1,2</sup>\*

<sup>1</sup>Institute of Automation, Chinese Academy of Sciences, <sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences {haodongze2021,qunbo.wang}@ia.ac.cn, {longteng.guo,jie.jiang,jliu}@nlpr.ia.ac.cn

## Abstract

While large visual-language models (LVLM) have shown promising results on traditional visual question answering benchmarks, it is still challenging for them to answer complex VQA problems which requires diverse world knowl edge. Motivated by the research of retrieval augmented generation in the field of natural language processing, we use Dense Passage Retrieval (DPR) to retrieve related knowledge to help the model answer questions. However, DPR conduct retrieving in natural language space, which may not ensure comprehensive acquisition of image information. Thus, the retrieved knowledge is not truly conducive to helping answer the question, affecting the performance of the overall system. To address this issue, we propose a novel framework that leverages the visual-language model to select the key knowledge retrieved by DPR and answer questions. The framework consists of two modules: Selector and Answerer, where both are initialized by the LVLM and parameterefficiently finetuned by self-bootstrapping: find key knowledge in the retrieved knowledge documents using the Selector, and then use them to finetune the Answerer to predict answers; obtain the pseudo-labels of key knowledge documents based on the predictions of the An swerer and weak supervision labels, and then finetune the Selector to select key knowledge; repeat. Our framework significantly enhances the performance of the baseline on the chal lenging open-domain Knowledge-based VQA benchmark, OK-VQA, achieving a state-ofthe-art accuracy of 62.83%. Our code is publicly available at https://github.com/ haodongze/Self-KSel-QAns.

## 1 Introduction

Recently, there has been an impressive advancement in large visual-language models (LVLM) (Li et al., 2023; Alayrac et al., 2022; Liu et al., 2023;

Dai et al., 2023). They usually use a mapping network to inject visual features into the semantic space of the large language model (Brown et al., 2020; Zhang et al., 2022; Touvron et al., 2023; vic, 2023; Touvron et al., 2023) and demonstrate strong capabilities on multimodal perception and reasoning. Thus, they achieve significant progress in conventional visual question answering benchmarks (Antol et al., 2015; Goyal et al., 2017; Hudson and Manning, 2019) which primarily focus on addressing straightforward questions that only necessitate visual perception and recognition. However, it is still challenging for the LVLMs to answer visual questions which require broader world knowledge and common sense (Wang et al., 2017; Marino et al., 2019; Schwenk et al., 2022).

Motivated by the research of retrievalaugmented generation (Karpukhin et al., 2020a) in the field of natural language processing, we use Dense Passage Retrieval (DPR) to retrieve related world knowledge to help the model answer questions. However, when using DPR, we need to transform the image into texts to retrieve the related knowledge, which leads to the underutilization of visual information. Thus, the retrieved knowledge may be unfaithful and affects the model performance. To address the issue, we consider the LVLM as the knowledge selector to find helpful knowledge from candidate retrieved knowledge by DPR. Then the selected knowledge is fed into the LVLM to predict the answer.

In this paper, we introduce a novel framework where we adopt the large visual-language model to perform knowledge selection and question answering. Our framework comprises two modules: a Selector and an Answerer. We train two modules by repeating the following process: the Selector first identifies important knowledge from the candidate knowledge documents retrieved by the pre-trained retriever; then, the Answerer takes the key knowledge documents as the input knowledge and is finetuned to generate the answer; next, we generate pseudo-labels of key knowledge documents according to the Answerer’s predictions and weak supervision labels; finally, we refine the Selector to assess the relevance of retrieved knowledge documents in answering the question. This strategy of self-bootstrapping enhances the ability of knowledge selection and answer generation consistently, enabling the model to accurately respond to knowledge-intensive questions.

We conduct extensive experiments on the opendomain knowledge-based VQA benchmark (OK-VQA (Marino et al., 2019)) to validate the effectiveness of the proposed framework, where our method largely outperforms the baseline and achieves the state-of-the-art performance of 62.83%, only finetuning 0.16% parameters with LoRA (Hu et al., 2022a). We also conduct comprehensive ablations to validate the impact of different components of the proposed framework, including the Effect of Selector and Answerer, cycle training of the framework, varying the number of key knowledge documents, and so on.

Our contributions are summarized as follows:

• We introduce a novel framework that leverages the large visual-language model to select key knowledge and use them to answer questions, respectively.

• We propose a new self-bootstrap learning method to train the Selector and Answerer, where the Selector chooses key knowledge documents for the Answerer and the Answerer provides pseudo-labels for the Selector.

• We achieve a state-of-the-art performance of 62.83% on the OK-VQA dataset, surpassing the previous state-of-the-art method. Notably, this improvement is achieved by fine-tuning only 0.16% of parameters using LoRA.

## 2 Related work

Large Visual-Language Models. Recently, large visual-language models (Li et al., 2023; Alayrac et al., 2022; Liu et al., 2023; Dai et al., 2023) have demonstrated remarkable visuallanguage understanding and reasoning capabilities, owing to the advancement of larger language models (Brown et al., 2020; Zhang et al., 2022; Touvron et al., 2023; vic, 2023; Touvron et al., 2023). These methods typically consist of a frozen visual encoder (Radford et al., 2021), a visual-language connector (Li et al., 2023), and a large language model (Chung et al., 2022; Zhang et al., 2022; vic, 2023). The models are firstly pre-trained on large-scale visual-text datasets to align visual features to the language embedding space. After pretraining, the large language model can understand the visual details. Then, the model is finetuned to adapt to various visual-language tasks. In this study, we adopt BLIP2, one of the widely used models, as our backbone for bootstrapping knowledge selection and question answering with it.

Knowledge-based VQA. Conventional VQA benchmarks (Goyal et al., 2017; Hudson and Manning, 2019) primarily focus on basic visual perception and reasoning tasks and numerous studies have achieved promising results on these benchmarks (Anderson et al., 2017; Zhang et al., 2021; Tan and Bansal, 2019; Lu et al., 2019; Li et al., 2022; Wang et al., 2022). Different from them, the knowledgebased VQA task (Wang et al., 2017; Marino et al., 2019; Schwenk et al., 2022) requires models to incorporate diverse world knowledge to respond to questions about visual content, which is more challenging. Recent studies (Gardères et al., 2020; Wu et al., 2022; Lin and Byrne, 2022; Gui et al., 2021) have explored various open-domain world knowledge sources, such as ConceptNet (Speer et al., 2017), Wikipedia (Vrandeciˇ c and Krötzsch´ , 2014), Google Search Corpus (Luo et al., 2021). They retrieve the relevant knowledge documents from the knowledge bases and integrate them into the answering model to generate predictions. Except for using explicit knowledge, some methods also take GPT-3 (Brown et al., 2020) as an implicit knowledge producer. They either prompt GPT-3 with in-context examples to predict answers directly (Yang et al., 2022; Hu et al., 2022b; Shao et al., 2023), or use GPT-3 to generate answer candidates with evidence serving as textual implicit knowledge bases (Gui et al., 2021; Lin et al., 2022), leading to significant performance improvements. Different from these approaches, we employ a large visuallanguage model to select key retrieved knowledge and reason on the knowledge to answer questions.

## 3 Method

In this section, we first introduce the preliminaries of Knowledge Retrieval and LVLM, which are the foundation of our framework. Then, we present the design of the Selector and Answerer for knowledge selection and question answering on knowledge respectively. Finally, we illustrate the self-bootstrap training method of two designed modules.

![](images/68e6d218acc6b95d55763dbc3ba81afd8718509cceb5f9a9832bfa4e8fa47be3.jpg)  
Knowledge 1: …with two of the most famous voices in cartoons, both supplied by mel blanc, sylvester's sloppy "sufferin succotash" and tweety's baby-voiced "i tawt i taw a puddy tat… Knowledge 2: …maybe one of the most widely known cat cartoon, garfield is one cat with attitude. he isn't interested in much, except lasagna, napping, lasagna, teasing the dog…  
Knowledge k: …why some of our favorite cartoon characters throughout the years have been feline in nature. maybe one of the most widely known cat cartoon, garfield is one cat with attitude… Sel Prompt: Does the retrieved knowledge document provide the key information to help answer the question? Ans Prompt: Short Answer  
Figure 1: Our framework consists of two modules: a Selector and an Answerer. Selector (left) selects the top-T knowledge documents for the Answerer (right), and the Answerer focuses on important knowledge information to predict answers. Both modules utilize the same frozen visual module to extract image features. We train the fully connected (FC) layer and fine-tune the language model using LoRA, which amounts to only 0.16% of the total parameters. For detailed training procedures of the two modules, refer to Alg. 1. The original knowledge is retrieved using DPR, and for brevity, we omit the retrieval process here (details can be found in Section 3.1).

## 3.1 Preliminaries

Knowledge Retrieval. We adopt the Dense Passage Retrieval (DPR) (Karpukhin et al., 2020b) to retrieve the knowledge documents. We transform the image into raw texts composed of captions, objects, attributes, and OCR (Optical Character Recognition). Then we compute the similarity scores between the query and knowledge documents sim $( q _ { i } , D _ { j } ) = \mathbf { q } _ { i } ^ { T } \cdot \mathbf { d } _ { j }$ and exploit FAISS (Johnson et al., 2019) to index Top-k related knowledge documents $\mathcal { P } _ { i } = \{ P _ { i , 1 } , P _ { i , 2 } , . . . , P _ { i , k } \}$ for i-th query.

Large Visual-Language Model. In our work, both knowledge selection and question-answering modules adopt BLIP-2 (Li et al., 2023) as the backbone. The architecture of BLIP-2 comprises a frozen image encoder (Dosovitskiy et al., 2020; Fang et al., 2023), a Q-Former (Li et al., 2023), and a pre-trained language model (Chung et al., 2022). Given an image $I _ { i } ,$ the frozen image encoder outputs a set of visual features $\{ \mathbf { h } _ { i , 1 } , \mathbf { h } _ { i , 2 } , . . . , \mathbf { h } _ { i , m } \}$ Q-Former takes extracted visual features as input, and outputs language-aligned visual features $\{ \mathbf { v } _ { i , 1 } , \mathbf { v } _ { i , 2 } , . . . , \mathbf { v } _ { i , l } \}$ . These visual features are concatenated with the textual word embeddings, which are fed into the language model for generation. Through pre-training on large-scale image-caption datasets, Q-Former can effectively project visual features into the feature space of the Language Large Model (LLM). We freeze the visual encoder and Q-former during training. We train the fully connected layer and use LoRA (Hu et al., 2022a) to finetune the LLM (only finetune 0.16% of total parameters).

## 3.2 Selector and Answerer

Selector. After obtaining the Top-k knowledge documents using DPR for the i-th sample, we aim to choose t most important knowledge documents from the retrieved documents, where t is smaller than k. As shown in Fig. 1, we first use the frozen image encoder and Q-former to extract the image features $\mathbf { V } _ { i } ,$ where these features are extracted once and then used by the Selector and the Answerer. Then image features $\mathbf { V } _ { i }$ are fed into the independent fully-connected layer to obtain the visual embeddings $\mathbf { E } _ { i } ^ { v }$ . We concatenate the question, a retrieved knowledge document, and the Selection prompt "Does the retrieved knowledge document provide the key information to help answer the question?" into one sentence S. Next, visual embeddings E<sup>v</sup> and the text are concatenated and fed into the LLM (Flan-T5 (Chung et al., 2022) is adopted in our work). Last, we use the probability of generating the word ‘yes’ as the score of each retrieved knowledge document $P _ { i , j }$ , denoted as $s _ { i , j } = L L M ( c o n c a t ( \mathbf { E } _ { i } ^ { v } , S _ { i } ) )$ , and we select top-t documents $\mathcal { \hat { P } } _ { i } = \{ \hat { P } _ { i , 1 } , \hat { P } _ { i , 2 } , . . . , \hat { P } _ { i , t } \}$ based on the scores. The Selector can be conceptualized as follows:

$$
\hat { \mathcal { P } } _ { i } = = S e l e c t o r ( I _ { i } , Q _ { i } , \mathcal { P } _ { i } ) , | \hat { \mathcal { P } } _ { i } | = t\tag{1}
$$

Answerer. After obtaining the selected knowledge documents, we aim to reason on the knowledge to answer questions. As shown in Fig. 1, we process the same image features to obtain the different visual embeddings $\mathbf { E } _ { i } ^ { v }$ via the fully-connected layer of the Answerer. Next, we concatenate the question and the knowledge into one sentence $S ^ { \prime }$ using the template "Question: {} Knowledge: {} Answer: ". We concatenate the visual embeddings and the text, which are fed into the LLM with different LoRA parameters to get the answer. The model outputs corresponding answers based on different documents. The Answerer can be conceptualized as follows:

$$
a _ { i } = A n s w e r e r ( I _ { i } , Q _ { i } , \hat { \mathcal { P } } _ { i } )\tag{2}
$$

Then the final answer is based on the majority vote. We also tried different knowledge reasoning methods, such as concatenating (the results can be seen in the ablation study).

## 3.3 Self-Bootstrap Learning

To enable the Selector and Answerer to select key knowledge and answer questions, we bootstrap them with each other in a style of cycle training. We repeat the following process for the given i-th sample $\{ I _ { i } , Q _ { i } , \mathcal { P } _ { i } , \mathcal { A } _ { i } \}$ of the training dataset:

Answerer Training. We use Eq. 1 to get the selected knowledge documents $\hat { \mathcal { P } } _ { i }$ . The image $I _ { i }$ is fed into the frozen ViT and Q-former to obtain the image features $\mathbf { V } _ { i }$ . We use the trainable $F C _ { a n s }$ layer to output the visual embeddings ${ \bf E } _ { a n s , i } ^ { v }$ We concatenate the visual embedding, the question $Q _ { i }$ and each selected knowledge document $\hat { P } _ { i , j }$ to construct t triplets for the sample, where $j = 1 , 2 , \dots , t .$ . Then we finetune the Answerer with LoRA under the supervision of the ground truth answer $A _ { i } { \mathrm { : } }$

$$
\begin{array} { r l } & { { \bf E } _ { a n s , i } ^ { v } = F C _ { a n s } ( { \bf V } _ { i } ) , } \\ & { { \cal L } _ { a n s } = - \displaystyle \sum _ { j = 1 } ^ { t } \log L L M _ { a n s } ( a _ { i } ^ { * } | { \bf E } _ { a n s , i } ^ { v } , Q _ { i } , \hat { P } _ { i } ^ { j } ) , } \end{array}\tag{3}
$$

where $a _ { i } ^ { * }$ is the most frequent answer in the humanannotated answer set $A _ { i }$

Algorithm 1 Pipeline of cycle training   
Input:   
KB-VQA dataset $\begin{array} { r l r } { { \mathcal { D } } } & { { } = } & { \{ I _ { i } , Q _ { i } , \mathcal { A } _ { i } | i \} \quad = } \end{array}$   
$1 , 2 , \ldots , N \} ;$   
Retrieved knowledge documents $\begin{array} { r l } { \mathcal { P } _ { i } } & { { } = } \end{array}$   
$\{ P _ { i } ^ { 1 } , P _ { i } ^ { 2 } , \ldots , P _ { i } ^ { k } \} ; \bar { I _ { i } } , Q _ { i } , \mathcal { P } _ { i }$ , and $\mathbf { \mathcal { A } } _ { i }$ denote   
image, question, document set, and answer set   
of i-th sample   
Output: Knowledge selection model Selector;   
Question answering model Answerer   
for sample in do   
Stage 1:   
1: Using Selector to select top-t documents   
$\hat { \mathcal { P } } _ { i }$ from the retrieved knowledge documents   
$\mathcal { P } _ { i }$ as Eq. 1   
2: Finetuning Answerer on $\{ I _ { i } , Q _ { i } , \hat { \mathcal { P } } _ { i } , \mathcal { A } _ { i } \}$   
supervised by the ground-truth answer as   
Eq. 3.   
Stage 2:   
1: Using Answerer to predict answers for   
retrieved knowledge documents $\mathcal { P } _ { i }$ as Eq. 2   
2: Generating to pseudo labels $\{ y _ { i , j } \}$ for re  
trieved knowledge documents $\mathcal { P } _ { i }$ as Eq. 4   
3: Finetuning Selector on   
$\{ I _ { i } , Q _ { i } , \mathcal { P } _ { i } , \{ y _ { i , j } \} \}$ supervised by the   
pseudo label as Eq. 5.   
end for

Selector Training. We first use Eq. 2 to predict answers based on each retrieved knowledge document $P _ { i , j }$ . Then we assign pseudo labels to the retrieved documents according to model predictions and weak supervision labels (Luo et al., 2021; Lin and Byrne, 2022; Lin et al., 2023). We use $" y e s "$ and $" \mathrm { n o " }$ as pseudo labels, where label a document as positive knowledge if Answerer can output the correct answer using that document and the document contains any of the answers in $\mathbf { \mathcal { A } } _ { i }$

$$
y _ { i , j } = { \left\{ \begin{array} { l l } { \operatorname { y e s } , \quad i f { \mathrm { ~ } } a _ { i } = a _ { i } ^ { * } \land } \\ { \qquad P _ { i , j } { \mathrm { ~ } } \operatorname { c o n t a i n s ~ a n ~ a n s w e r ~ i n } { \mathrm { ~ } } \mathcal { A } _ { i } } \\ { \operatorname { n o } , \quad e l s e } \end{array} \right. }\tag{4}
$$

After obtaining the pseudo label of each retrieved knowledge document, we use the trainable $F C _ { s e l }$ layer to output the visual embeddings ${ \bf E } _ { s e l , i } ^ { v }$ we concatenate the visual embedding, the question $Q _ { i }$ and each retrieved knowledge document $P _ { i , j }$ to construct k triplets for the sample, where $j = 1 , 2 , \dots , k$ . Then we finetune the Selector with LoRA under the supervision of pseudo labels:

Table 1: Performance comparison with state-of-the-art (SOTA) methods on the OK-VQA dataset. Knowledge Sources: ConceptNet (C); Wikipedia (W); Google Search (GS); Google Images (GI). The best result in the table is bolded. The results show that our method achieves the state-of-the-art performance.
<table><tr><td>Models</td><td>Large Models</td><td> $K _ { t r a i n }$ </td><td> $K _ { t e s t }$ </td><td>Knowledge Resource</td><td>Acc (%)</td></tr><tr><td>BAN+AN (Marino et al., 2019)</td><td></td><td></td><td></td><td>W</td><td>25.6</td></tr><tr><td>ConceptBERT (Gardères et al., 2020)</td><td></td><td></td><td></td><td>C</td><td>33.7</td></tr><tr><td>KRISP (Marino et al., 2021)</td><td></td><td></td><td></td><td>C+W</td><td>38.4</td></tr><tr><td>Visual Retriever-Reader (Luo et al., 2021)</td><td></td><td>100</td><td>100</td><td>GS</td><td>39.2</td></tr><tr><td>MAVEx (Wu et al., 2022)</td><td></td><td></td><td></td><td>W+C + GI</td><td>39.4</td></tr><tr><td>PICa (Yang et al., 2022)</td><td>GPT-3 (175B)</td><td></td><td></td><td>GPT-3</td><td>48.0</td></tr><tr><td>TRiG(Ensemble) (Gao et al., 2022)</td><td>T5-1arge (770M)</td><td>100</td><td>100</td><td>W</td><td>50.5</td></tr><tr><td>KAT(Single) (Gui et al., 2021)</td><td>T5-1arge (770M)</td><td>40</td><td>40</td><td>W + GPT-3</td><td>53.1</td></tr><tr><td>KAT(Ensemble) (Gui et al., 2021)</td><td>T5-1arge (770M)</td><td>40</td><td>40</td><td>W + GPT-3</td><td>54.4</td></tr><tr><td>RA-VQA (Lin and Byrne, 2022)</td><td>T5-1arge (770M)</td><td>5</td><td>50</td><td>GS</td><td>54.5</td></tr><tr><td>REVIVE(Single) (Lin et al., 2022)</td><td>T5-large (770M)</td><td>40</td><td>40</td><td>W+GPT-3</td><td>56.6</td></tr><tr><td>REVIVE(Ensemble) (Lin et al., 2022)</td><td>T5-1arge (770M)</td><td>40</td><td>40</td><td>W+GPT-3</td><td>58.0</td></tr><tr><td>PromptCap (Hu et al., 2022b)</td><td>GPT-3 (175B)</td><td></td><td></td><td>GPT-3</td><td>60.4</td></tr><tr><td>Prophet (Shao et al., 2023)</td><td>GPT-3 (175B)</td><td></td><td></td><td>GPT-3+MCAN</td><td>61.1</td></tr><tr><td>FillingGap (Wang et al., 2023)</td><td>GPT-3 (175B)</td><td></td><td></td><td>GPT-3</td><td>61.3</td></tr><tr><td>SimpleBaseline (Xenos et al., 2023)</td><td>LLaMA 2 (13B)</td><td></td><td></td><td>LLaMA 2</td><td>61.2</td></tr><tr><td>Cola-FT (Chen et al., 2024)</td><td>FLAN-T5(11B)</td><td></td><td></td><td>BLIP+OFA</td><td>62.4</td></tr><tr><td>Flamingo (Alayrac et al., 2022)</td><td>Flamingo (80B)</td><td></td><td></td><td>Pretrain</td><td></td></tr><tr><td>InstructBLIP (Dai et al., 2023)</td><td>InstructBLIP Vicuna (7B)</td><td></td><td></td><td>Pretrain</td><td>57.8</td></tr><tr><td>Qwen-VL (Bai et al., 2023)</td><td>Qwen-VL(Qwen-7B)</td><td></td><td></td><td>Pretrain</td><td>62.1</td></tr><tr><td>MM-Reasoner (Khademi et al., 2023)</td><td></td><td></td><td></td><td>GPT-4</td><td>58.6</td></tr><tr><td>BLIP2 (fine-tuned) (Li et al., 2023)</td><td>Flamingo (80B)</td><td>一</td><td></td><td></td><td>60.8</td></tr><tr><td>RA-VQA-v2 (Lin et al., 2023)</td><td>BLIP2 T5-XL (3B) BLIP2 T5-XL (3B)</td><td>5</td><td>- 5</td><td>Pretrain</td><td>55.4</td></tr><tr><td></td><td></td><td>5</td><td>5</td><td>GS</td><td>62.1</td></tr><tr><td>PreFLMR (Lin et al., 2024)</td><td>BLIP2 T5-XL (3B)</td><td></td><td></td><td>GS</td><td>61.8</td></tr><tr><td>Ours</td><td>BLIP2 T5-XL (3B)</td><td>5</td><td>5</td><td>GS</td><td>62.8</td></tr></table>

$$
\begin{array} { r l } & { \mathbf { E } _ { s e l , i } ^ { v } = F C _ { s e l } ( \mathbf { V } _ { i } ) , } \\ & { L _ { s e l } = - \displaystyle \sum _ { j = 1 } ^ { k } \log L L M _ { s e l } ( y _ { i , j } \vert \mathbf { E } _ { s e l , i } ^ { v } , Q _ { i } , P _ { i } ^ { j } ) } \end{array}\tag{5}
$$

We provide the overall training pipeline in Alg. 1. Through continuous iteration, the Selector will provide more crucial knowledge for the Answerer to accurately respond to questions. Meanwhile, the improvement in the Answerer’s reasoning ability will also result in more precise pseudo-labeling, further enhancing the Selector’s discriminative power. During the inference stage, we utilize the Selector to choose key knowledge, and then instruct the Answerer to respond to questions based on this knowledge.

## 4 Experiments

## 4.1 Experimental Setup

Dataset. We conduct extensive experiments on OK-VQA (Marino et al., 2019) to evaluate the effectiveness of our method. OK-VQA is a challenging open-domain knowledge-based VQA dataset that requires models to leverage various external knowledge sources to answer questions. The dataset contains 14,055 questions and 14,031 images, whereas the training set and testing set have 9k and 5k image-question pairs, respectively. Due to no knowledge base being provided for OK-VQA, we need to choose the proper knowledge base for the dataset. In this paper, we adopt Google Search Corpus (Luo et al., 2021) as the knowledge base which is collected in the websites using the Google Search API.

Evaluation Metric. We use the standard VQA metric (Antol et al., 2015) to evaluate the performance of the model. Given the prediction of the question a and the groudtruth answer set ${ \mathcal { A } } ,$ the VQA accuracy is calculated as:

$$
A c c u r a c y ( a , \mathcal { A } ) = \operatorname* { m i n } ( \frac { \# A ( a ) } { 3 } , 1 ) ,\tag{6}
$$

where the groudtruth answer set $\mathcal { A }$ is annotated by different humans, $\# A ( a )$ denotes the occurrence of a in .

Implementation Details. In our experiment, we adopt BLIP2 T5-XL (3B) (Li et al., 2023) to initialize the Selector and Answerer. We freeze the image encoder and Q-former, with both the Selector and Answerer sharing the same visual module. We finetune the fully connected layer and use LoRA (Hu et al., 2022a) to train the LLM. We use the default huggingface-PEFT setting: r=8, lora\_alpha=32, lora\_dropout=0.1. We use Adam as the optimizer and set the batch size to 8. We use the warm-up strategy which trains the model with an initial learning rate of 1e-4 and warm-up factor of 0.05 for 1000 steps and then utilizes a cosine annealing learning strategy with an initial learning rate of 1e-4 and a final learning rate of 0 after 10 epochs. We use top-30 knowledge documents retrieved by a pre-trained DPR (Lin and Byrne, 2022) as candidates for Selector and use the selected top-5 documents from the 30 documents for the Answerer to train and infer, denoted as $K _ { c a n d i d a t e } = 3 0 , K _ { t r a i n } = 5 , K _ { t e s t } = 5$ . We use 2 Nvidia A800 GPUs (80G) for all experiments.

## 4.2 Comparison with State-of-the-art Methods

As shown in Tab. 1, we can see that early models (BAN+AN (Marino et al., 2019), ConceptBERT (Gardères et al., 2020), KRISP (Marino et al., 2021), Visual Retriever-Reader (Luo et al., 2021), and MAVEx (Wu et al., 2022)) have a weak performance, achieving a VQA accuracy from 25.6% to 39.4%. Recently, by introducing larger models (T5- large, GPT-3, LLaMA, Vicuna) and diverse knowledge resources (ConceptNet, Wikipedia, Google Web Search and Google Images), the performance has a significant performance improvement, achieving a VQA accuracy of 62.4%. Our method aims to augment the reasoning ability to answer knowledgeintensive questions of the large visual-language model. When directly finetuning BLIP2 T5-XL on OKVQA, the model has a low performance of 55.44%. By introducing external knowledge, the performance has a significant performance improvement. Different from RA-VQA-v2 (Lin et al., 2023) and PreFLMR (Lin et al., 2024), we do not train a multimodal retriever from scratch which requires expensive annotations and high computational costs. We directly leverage the large visuallanguage model to select key knowledge from the retrieved knowledge by DPR like the process of re-ranking. With the same knowledge resources (i.e., Google Search), our method achieves 62.8% accuracy, outperforming other state-of-the-art models. It is worth noting that we do not use GPT-3 and we only train the 0.16% parameters of the model.

Table 2: Comparison of our selector with different knowledge selection strategies. We select 5 knowledge documents from top-30 knowledge candidates retrieved by DPR. DPR Score refers to selecting top-5 knowledge based on similarity scores. Random Selection means randomly selecting 5 knowledge documents from 30 candidate knowledge documents. Selector denotes choosing 5 key knowledge documents by the Selector.
<table><tr><td> $K _ { t r a i n }$ </td><td> $K _ { t e s t }$ </td><td>Knowledge Selection</td><td>Acc (%)</td></tr><tr><td>5</td><td>1</td><td>Random Selection</td><td>50.45</td></tr><tr><td>5</td><td>1</td><td>DPR Score</td><td>58.80</td></tr><tr><td>5</td><td>1</td><td>Selector</td><td>61.62</td></tr><tr><td>5</td><td>5</td><td>Random Selection</td><td>55.05</td></tr><tr><td>5</td><td>5</td><td>DPR Score</td><td>60.69</td></tr><tr><td>5</td><td>5</td><td>Selector</td><td>62.83</td></tr></table>

These results demonstrate the effectiveness of the proposed approach.

## 4.3 Ablation Study

We conduct the ablation studies to evaluate different components of our framework on OK-VQA.

Effect of Selector. We conduct the ablation study to evaluate the effectiveness of Selector in our method. We show the results in Tab. 2. From the results, we can observe: our framework, leveraging key knowledge documents selected by the Selector, consistently outperforms the Answerer when using the same number of documents retrieved by DPR. We improve the performance by 2.14% and 1.88% with 1 and 5 test knowledge documents, compared to DPR-based retrieval. When using the randomly selected documents, the model performs worst. These results demonstrate that top-ranked knowledge documents based on DPR scores are not optimal for question answering and our key knowledge selection module can identify relevant documents for accurate question answering, ensuring the coherence of knowledge retrieval and question-answering processes.

Effect of different knowledge reasoning methods of Answerer. In Tab. 3, we present a comparison of Answerer using different knowledge reasoning methods. The results show that the performance using the strategy of voting surpasses that of concatenating under different knowledge selection settings. We argue that directly combining all the knowledge documents into a lengthened document makes it difficult for Answerer to reason on them, which is easily influenced by noisy information. In contrast, it is easier for Answerer to reason on each document to predict the answer. Simple voting can choose the best answer.

Table 3: Effect of different knowledge reasoning methods of Answerer. Concatenating denotes that we combine the key knowledge documents into one sentence and feed it into Answerer to predict the final answer. Voting means that we feed different key knowledge documents into Answerer to predict different answers and choose the best answer based on majority voting.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>VQA Model</td><td rowspan=1 colspan=1>Acc (%)</td></tr><tr><td rowspan=1 colspan=1>ConcatenatingVoting</td><td rowspan=1 colspan=1>BLIP2 (fine-tuned)</td><td rowspan=1 colspan=1>59.1160.69</td></tr><tr><td rowspan=1 colspan=1>ConcatenatingVoting</td><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>62.0662.83</td></tr></table>

Table 4: Effect of the self-bootstrap learning method.
<table><tr><td>Method</td><td>Acc (%)</td></tr><tr><td>Baseline</td><td>60.69</td></tr><tr><td>Independent training</td><td>59.02</td></tr><tr><td>Cycle training</td><td>62.83</td></tr></table>

Effect of the self-bootstrap learning method. To evaluate the effectiveness of our self-bootstrap learning method, we compare the method with the strategy of independent training of two modules. We finetune the Answerer with the knowledge documents retrieved by DPR as the baseline. Independent training means that we take two passes in answerer training and one pass for selector training on the entire dataset. Cycle training means that we train the answerer and selector on each batch data of the dataset simultaneously. The results in Tab. 4 show that the model with cycle training outperforms the model with independent training by 3.81%. The VQA score of using independent training is even lower than the baseline. These results demonstrate that our cycle training method can effectively boost the Selector and Answerer each other, which makes the model find key knowledge documents and leverage the knowledge to answer questions.

Effect of different methods of pseudo-labeling. In Tab. 5, we compare the model performance with different methods of pseudo-labeling. When using the model predictions as guidance, the model has a VQA score of 62.31%. When adding the weak supervision as the guidance, the model’s VQA score increases from 62.31% to 62.83%. The results demonstrate that using weak supervision labels preserves potentially useful documents, aiding the Answerer in accurately answering questions.

Table 5: Ablation study on different methods of pseudolabeling.
<table><tr><td>Model predictions</td><td>Weak supervision labels</td><td>Acc (%)</td></tr><tr><td>√ √</td><td>√</td><td>62.31 62.83</td></tr></table>

Table 6: Ablation study on different numbers of candidate documents and selected documents.
<table><tr><td> $K _ { c a n d i d a t e }$ </td><td> $K _ { t r a i n }$ </td><td> $K _ { t e s t }$ </td><td>Acc (%)</td></tr><tr><td>5</td><td>1</td><td>1</td><td>57.90</td></tr><tr><td>5</td><td>1</td><td>5</td><td>58.32</td></tr><tr><td>10</td><td>1</td><td>1</td><td>58.61</td></tr><tr><td>10</td><td>1</td><td>5</td><td>59.40</td></tr><tr><td>10</td><td>5</td><td>5</td><td>61.86</td></tr><tr><td>15</td><td>5</td><td>5</td><td>62.31</td></tr><tr><td>30</td><td>5</td><td>5</td><td>62.83</td></tr><tr><td>30</td><td>5</td><td>1</td><td>61.62</td></tr></table>

Effect of key knowledge documents selection ranges and quantities. In Tab. 6, we evaluate key knowledge document selection using various numbers of candidate documents and selected documents. From the results, we have the following findings: (1) As the number of selected documents increases, the model’s performance improves. This indicates that using more documents to train and test contributes to answering questions. (2) Using more documents for training can improve the performance a lot (the 2nd line v.s. the last line). However, using more documents for testing has almost no improvement (the 3rd line v.s. 4th line). (3) When the number of candidate documents increases, the model’s performance improves. The result demonstrates that low-ranked documents based on DPR scores may contain useful information for question answering. It is necessary for the model to select key knowledge documents.

Table 7: Ablation study on different documents selection in Answerer fine-tuning.
<table><tr><td colspan="2">Knowledge Selection</td><td rowspan="2">Acc (%)</td></tr><tr><td>Training</td><td>Inference</td></tr><tr><td>DPR</td><td>Selector</td><td>62.31</td></tr><tr><td>Selector</td><td>DPR</td><td>60.75</td></tr><tr><td>Selector</td><td>Selector</td><td>62.83</td></tr></table>

![](images/92344f0486cdfc2b076e90f91c7b0246986d604f436d5b8388b914b84a69f3a9.jpg)  
BLIP2 (fine-tuned) w knowledge from DPR Figure 2: Qualitative results on the test split of OK-VQA. We compared our method with a model that fine-tunes <sup>…unlike</sup> <sup>new</sup> <sup>yorker</sup> <sup>cartoons,</sup> <sup>in</sup> <sup>which,</sup> <sup>you</sup> <sup>are</sup> <sup>actually</sup> <sup>missing</sup> <sup>the</sup> <sup>joke,</sup> <sup>garfield</sup> <sup>is</sup> <sup>in</sup> <sup>fact</sup> <sub>garfield</sub>BLIP2 with knowledge ranked by DPR. The middle segment of the graph represents knowledge from various methods used to answer questions. On the right side of the graph, different answers are depicted when using distinct <sub>sloppy</sub> <sub>"sufferin succotash"</sub> <sub>and</sub> <sub>tweety's baby-voiced</sub> <sub>"i tawt i taw</sub> <sub>a</sub> <sub>puddy</sub> <sub>tat…</sub>knowledge. Green and red colors indicate whether the selected final answer is correct.

Effect of different knowledge documents selection in Answerer fine-tuning. Tab. 7 compares <sup>Q:</sup> <sup>What</sup> <sup>is</sup> <sup>a</sup> <sup>famous</sup> <sup>cartoon</sup> <sup>not</sup> <sup>even</sup> <sup>designed</sup> <sup>to</sup> <sup>be</sup> <sup>funny</sup> <sup>(jerry</sup> <sup>kni</sup>Answerer fine-tuning with different document se-<sup>…</sup> <sup>maybe</sup> <sup>one</sup> <sup>of</sup> <sup>the</sup> <sup>most</sup> <sup>widely</sup> <sup>known</sup>lection strategies. The results show that our framework performs optimally when utilizing Selector in maybe one of the most widely known cboth Answerer training and inference. This is likely because the Selector provides more informative key knowledge documents and using both Selector ensures the consistency between the training domain and testing domain.

Performance of the knowledge retrieval. In tab. 8, we evaluate our Selector in the knowledge retrieval task. Following previous methods (Luo et al., 2021; Lin and Byrne, 2022), we adopt pseudo relevance to measure if the retrieved document is relevant to the query due to the absence of groundtruth document. We use Recall to measure the performance of the the knowledge retrieval. From the results, we can see that our Selector improves the performance of DPR a lot. This means our Selector can retrieve more relevant knowledge documents which help answer questions. Compared to other retrievers, our Selector achieves the second best performance. Although FLMR outperforms our Selector in the knowledge retrieval, our framework achieves better accuracy in VQA (shown in Tab. 1) with the same backbone. This indicates that the knowledge documents selected by Selector have better consistency with Answerer.

Table 8: Retrieval performance on Google Search (GS).
<table><tr><td rowspan=1 colspan=1>Retriever</td><td rowspan=1 colspan=1>R@5</td><td rowspan=1 colspan=1>R@10</td></tr><tr><td rowspan=1 colspan=1>VRR (Luo et al., 2021)RA-VQA-FrDPR (Lin and Byrne, 2022)RA-VQA (Lin and Byrne, 2022)FLMR (Lin et al., 2023)</td><td rowspan=1 colspan=1>80.481.2582.8489.32</td><td rowspan=1 colspan=1>88.5588.5189.0094.02</td></tr><tr><td rowspan=1 colspan=1>DPR (Lin and Byrne, 2022)Our Selector</td><td rowspan=1 colspan=1>82.9388.66</td><td rowspan=1 colspan=1>89.9593.56</td></tr></table>

## 4.4 Qualitative Analysis

In Fig. 2, We present a case study comparing our method with a model that fine-tunes BLIP2 using knowledge ranked by DPR. In the first case, topranked knowledge documents from DPR misguide the model, resulting in incorrect predictions. However, our method’s Selector chooses key knowledge documents that aid in predicting correct answers. In the second case, each knowledge document from DPR contains irrelevant information, leading to an incorrect final answer. Despite the top-1 document from the Selector resulting in a wrong answer, our method identifies other key knowledge documents for generating correct answers. Through majority voting, the final selected answer is correct. These cases demonstrate our method’s ability to extract informative knowledge from retrieved documents to support accurate question answering.

## 5 Conclusion

In this paper, we propose a novel framework that leverages the large visual-language model to construct two modules: (1) Selector for finding key retrieved knowledge and (2) Answerer for reasoning on the knowledge to predict answers. We design a self-bootstrap learning method to improve their abilities, where the Selector chooses key knowledge documents for the Answerer and the Answerer provides pseudo-labels for the Selector. Compared with state-of-the-art methods, our method achieves better performance on a challenging open-domain knowledge-based VQA benchmark (OK-VQA) and we conduct a comprehensive analysis to evaluate the effectiveness of our method.

## 6 Acknowledgements

This work was supported by the Artificial Intelligence-National Science and Technology Major Project (2023ZD0121200), the National Natural Science Foundation of China (U21B2043, 62102416), and the Key Research and Development Program of Jiangsu Province under Grant BE2023016-3.

## 7 Limitations

Although our framework can effectively select key knowledge documents for answering question, it is inevitable that the knowledge still contains noise. In some cases, the model itself can answer the question without external knowledge, introducing extra knowledge may affect the performance. In the future, we can explore to dynamically select required knowledge to help itself answer questions.

In addition, there is a concern on the generalizability of the proposed method on other domains, especially when the initial DPR model can not retrieve gold standard context. In the future, we consider adopting a stronger multimodal retriever model to obtain more useful candidate knowledge documents, which enhances the generalizability of our framework.

## References

2023. Vicuna. https://github.com/lm-sys/ FastChat.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736.

Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. 2017. Bottom-up and top-down attention for image captioning and visual question answering. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6077–6086.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. 2015. Vqa: Visual question answering. In Proceedings ofthe IEEE international conference on computer vision, pages 2425–2433.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Liangyu Chen, Bo Li, Sheng Shen, Jingkang Yang, Chunyuan Li, Kurt Keutzer, Trevor Darrell, and Ziwei Liu. 2024. Large language models are visual reasoning coordinators. Advances in Neural Information Processing Systems, 36.

Zhuo Chen, Jiaoyan Chen, Yuxia Geng, Jeff Z Pan, Zonggang Yuan, and Huajun Chen. 2021. Zero-shot visual question answering using knowledge graph. In The Semantic Web–ISWC 2021: 20th International Semantic Web Conference, ISWC 2021, Virtual Event, October 24–28, 2021, Proceedings 20, pages 146– 162. Springer.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Albert Li, Pascale Fung, and Steven C. H. Hoi. 2023. Instructblip: Towards general-purpose vision-language models with instruction tuning. ArXiv, abs/2305.06500.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. 2023. Eva: Exploring the limits of masked visual representation learning at scale. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19358– 19369.

Feng Gao, Qing Ping, Govind Thattai, Aishwarya Reganti, Ying Nian Wu, and Prem Natarajan. 2022. Transform-retrieve-generate: Natural languagecentric outside-knowledge visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5067–5077.

François Gardères, Maryam Ziaeefard, Baptiste Abeloos, and Freddy Lecue. 2020. Conceptbert: Concept-aware representation for visual question answering. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 489–498.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913.

Liangke Gui, Borui Wang, Qiuyuan Huang, Alex Hauptmann, Yonatan Bisk, and Jianfeng Gao. 2021. Kat: A knowledge augmented transformer for vision-andlanguage. arXiv preprint arXiv:2112.08614.

Yangyang Guo, Liqiang Nie, Yongkang Wong, Yibing Liu, Zhiyong Cheng, and Mohan Kankanhalli. 2022. A unified end-to-end retriever-reader framework for knowledge-based vqa. In Proceedings of the 30th ACM International Conference on Multimedia, pages 2061–2069.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022a. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Yushi Hu, Hang Hua, Zhengyuan Yang, Weijia Shi, Noah A Smith, and Jiebo Luo. 2022b. Promptcap: Prompt-guided task-aware image captioning. arXiv preprint arXiv:2211.09699.

Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709.

Yu Jiang, Vivek Natarajan, Xinlei Chen, Marcus Rohrbach, Dhruv Batra, and Devi Parikh. 2018. Pythia v0. 1: the winning entry to the vqa challenge 2018. arXiv preprint arXiv:1807.09956.

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7(3):535–547.

Amita Kamath, Christopher Clark, Tanmay Gupta, Eric Kolve, Derek Hoiem, and Aniruddha Kembhavi. 2022. Webly supervised concept expansion for general purpose vision models. In European Conference on Computer Vision, pages 662–681. Springer.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020a. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769– 6781, Online. Association for Computational Linguistics.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick˘ Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020b. Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.

Mahmoud Khademi, Ziyi Yang, Felipe Frujeri, and Chenguang Zhu. 2023. MM-reasoner: A multimodal knowledge-aware framework for knowledgebased visual question answering. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 6571–6581, Singapore. Association for Computational Linguistics.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR.

Weizhe Lin and Bill Byrne. 2022. Retrieval augmented visual question answering with outside knowledge. arXiv preprint arXiv:2210.03809.

Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, and Bill Byrne. 2023. Fine-grained late-interaction multi-modal retrieval for retrieval augmented visual question answering. ArXiv, abs/2309.17133.

Weizhe Lin, Jingbiao Mei, Jinghong Chen, and Bill Byrne. 2024. Preflmr: Scaling up finegrained late-interaction multi-modal retrievers. (arXiv:2402.08327).

Yuanze Lin, Yujia Xie, Dongdong Chen, Yichong Xu, Chenguang Zhu, and Lu Yuan. 2022. Revive: Regional visual representation matters in knowledgebased visual question answering. Advances in Neural Information Processing Systems, 35:10560–10571.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. arXiv preprint arXiv:2304.08485.

Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. 2019. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 32.

Man Luo, Yankai Zeng, Pratyay Banerjee, and Chitta Baral. 2021. Weakly-supervised visual-retrieverreader for knowledge-based question answering. arXiv preprint arXiv:2109.04014.

Kenneth Marino, Xinlei Chen, Devi Parikh, Abhinav Gupta, and Marcus Rohrbach. 2021. Krisp: Integrating implicit and symbolic knowledge for opendomain knowledge-based vqa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14111–14121.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. 2019. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR.

Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. 2022. A-okvqa: A benchmark for visual question answering using world knowledge. In European Conference on Computer Vision.

Zhenwei Shao, Zhou Yu, Meng Wang, and Jun Yu. 2023. Prompting large language models with answer heuristics for knowledge-based visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14974– 14983.

Robyn Speer, Joshua Chin, and Catherine Havasi. 2017. Conceptnet 5.5: An open multilingual graph of general knowledge. In Proceedings ofthe AAAI conference on artificial intelligence, volume 31.

Hao Tan and Mohit Bansal. 2019. Lxmert: Learning cross-modality encoder representations from transformers. arXiv preprint arXiv:1908.07490.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Denny Vrandeciˇ c and Markus Krötzsch. 2014. Wiki-´ data: a free collaborative knowledgebase. Communications ofthe ACM, 57(10):78–85.

Peng Wang, Qi Wu, Chunhua Shen, Anthony Dick, and Anton Van Den Hengel. 2017. Fvqa: Fact-based visual question answering. IEEE transactions on pattern analysis and machine intelligence, 40(10):2413– 2427.

Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. 2022. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International Conference on Machine Learning, pages 23318–23340. PMLR.

Ziyue Wang, Chi Chen, Peng Li, and Yang Liu. 2023. Filling the image information gap for vqa: Prompting large language models to proactively ask questions. arXiv preprint arXiv:2311.11598.

Jialin Wu, Jiasen Lu, Ashish Sabharwal, and Roozbeh Mottaghi. 2022. Multi-modal answer validation for knowledge-based vqa. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pages 2712–2721.

Alexandros Xenos, Themos Stafylakis, Ioannis Patras, and Georgios Tzimiropoulos. 2023. A simple baseline for knowledge-based visual question answering. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 14871–14877.

Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lijuan Wang. 2022. An empirical study of gpt-3 for few-shot knowledgebased vqa. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 36, pages 3081– 3089.

Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. 2021. Vinvl: Revisiting visual representations in vision-language models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 5579–5588.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

Table 9: Performance comparison with state-of-the-art (SOTA) methods on the FVQA dataset.
<table><tr><td>Method</td><td>Acc-1</td></tr><tr><td>Human</td><td>77.99</td></tr><tr><td>UnifER (Guo et al., 2022)</td><td>55.04</td></tr><tr><td>FVQA (Wang et al., 2017)</td><td>56.91</td></tr><tr><td>ZS-VQA (Chen et al., 2021)</td><td>58.27</td></tr><tr><td>FVQA(Ensemble) (Wang et al., 2017) MM-Reasoner(Ensemble) (Khademi et al., 2023)</td><td>58.76</td></tr><tr><td>Ours</td><td>61.10 63.3</td></tr></table>

Table 10: Performance comparison with state-of-the-art (SOTA) methods on the A-OKVQA dataset.
<table><tr><td>Method</td><td colspan="2">Direct Answer</td></tr><tr><td>ClipCap (Schwenk et al., 2022)</td><td>val 18.1</td><td>test 15.8</td></tr><tr><td>Pythia (Jiang et al., 2018)</td><td>25.2</td><td>21.9</td></tr><tr><td>ViLBERT (Lu et al., 2019)</td><td>30.6</td><td>25.9</td></tr><tr><td>LXMERT (Tan and Bansal, 2019)</td><td>30.7</td><td>25.9</td></tr><tr><td>KRISP (Marino et al., 2021)</td><td>33.7</td><td>27.1</td></tr><tr><td>GPV-2 (Kamath et al., 2022)</td><td>48.6</td><td>40.7</td></tr><tr><td>BLIP-2 T5-XL (Li et al., 2023)</td><td>53.2</td><td>49.7</td></tr><tr><td>PromptCap + GPT-3 (Hu et al., 2022b)</td><td></td><td>59.6</td></tr><tr><td>Ours</td><td>56.3 57.2</td><td>56.4</td></tr></table>

Table 11: Computational cost of our framework.
<table><tr><td>Kcandidate</td><td>Memory (GB)</td><td>Running Time (sec./sample)</td></tr><tr><td>10</td><td>21.3</td><td>1.0</td></tr><tr><td>15</td><td>21.4</td><td>1.1</td></tr><tr><td>30</td><td>22.7</td><td>1.3</td></tr></table>

tational cost of our framework has a small increase.

## A Appendix

## A.1 Experiments on Other Datasets.

We also evaluate our method on FVQA (Fang et al., 2023) and A-OKVQA (Schwenk et al., 2022) to demonstrate the effectiveness of our method. FVQA is a VQA dataset that mostly contains questions requiring external knowledge to answer, and provides supporting fact triplets alongside the image-question-answer triplets. A-OKVQA is an augmented successor of OK-VQA, containing 25K image-question pairs that require broader commonsense and world knowledge to answer. Due to A-OKVQA does not provide the knowledge source, we use Wikipedia (Vrandeciˇ c and Krötzsch ´ , 2014) as the knowledge base.

In Tab. 11, we show the computational cost of our framework using different knowledge candidates. As the number of candidate knowledge, the compu-

As shown in Tab. 9, our method surpasses previous state-of-the-art methods, which demonstrates the effectiveness and generalization of our method. Tab. 10 shows the comparative results on the challenging A-OKVQA dataset. Our method achieved competitive results, which demonstrates the effectiveness of our method.

## A.2 Evaluation of Computational Cost