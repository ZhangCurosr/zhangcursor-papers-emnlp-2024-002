# Uncertainty in Language Models: Assessment through Rank-Calibration

Xinmeng Huang\*† Shuo Li\*†

Mengxin Yu† Matteo Sesia Hamed Hassani†

Insup Lee†

Osbert Bastani§† Edgar Dobriban§†

## Abstract

Language Models (LMs) have shown promising performance in natural language generation. However, as LMs often generate incorrect or hallucinated responses, it is crucial to correctly quantify their uncertainty in responding to given inputs. In addition to verbalized confidence elicited via prompting, many uncertainty measures $( e . g .$ , semantic entropy and affinitygraph-based measures) have been proposed. However, these measures can differ greatly, and it is unclear how to compare them, partly because they take values over different ranges (e.g., [0, ∞) or [0, 1]). In this work, we address this issue by developing a novel and practical framework, termed Rank-Calibration, to assess uncertainty and confidence measures for LMs. Our key tenet is that higher uncertainty (or lower confidence) should imply lower generation quality, on average. Rank-calibration quantifies deviations from this ideal relationship in a principled manner, without requiring ad hoc binary thresholding of the correctness score (e.g., ROUGE or METEOR). The broad applicability and the granular interpretability of our methods are demonstrated empirically. The code to replicate our experiments is here.

## 1 Introduction

Language Models (LMs), especially Large Language Models (LLMs), have shown promising performance in Natural Language Generation (NLG). These models, fitted on huge text corpora, can produce responses resembling those of humans (Touvron et al., 2023b; OpenAI, 2023). However, since LMs often generate wrong or hallucinated responses (Weidinger et al., 2021; Xiao and Wang, 2021; Huang et al., 2024), it is crucial to correctly quantify their level of uncertainty in responding to particular inputs.

![](images/3cdd9987818a371ed1a1180f46685f069c953a98dd567c5fa8289129827104ec.jpg)

![](images/3e43300b2bcba9559e8359836eb7767c7e90b49648ed2842855f1882b0bcf8ea.jpg)  
Figure 1: Indication diagrams comparing two uncertainty measures, $U _ { \mathrm { N L L } }$ (negative log-likelihood) and $U _ { \mathrm { E c c } }$ (eccentricity), for the GPT-3.5-turbo model on the TriviaQA benchmark. The red bars indicate the average correctness of different outputs, as a function of the corresponding relative uncertainty levels. The blue and shallow red areas—deviating from the anti-diagonal line—indicate where the uncertainty measures are overoptimistic and pessimistic, respectively. Their sum is our rank-miscalibration metric (i.e., RCE), which here is lower for $U _ { \mathrm { N L L } }$ than $U _ { \mathrm { E c c } } .$ See Sec. 4.3 for details.

Uncertainty quantification is well-explored in supervised learning, specifically in classification (e.g., Lichtenstein et al., 1977; Gal and Ghahramani, 2016; Lakshminarayanan et al., 2017, etc). In classification, a confidence measure is an estimate of the probability that the predicted class $\widehat { Y }$ matches the true class label Y (Lichtenstein et al., 1977; Lee et al., 2023). A confidence measure C is considered calibrated if it reflects the probability of correct prediction, i.e., $\mathbb { P } ( \widehat { Y } = Y \mid { C } ) = C ,$ , for all values in $C ^ { \bullet } \mathrm { s }$ range. The Expected Calibration Error (ECE) measures the miscalibration of a confidence measure (Harrell, 2015; Naeini et al., 2015):

$$
\mathbb { E } _ { C } \Big [ \Big | \mathbb { P } ( \widehat { Y } = Y | C ) - C \Big | \Big ] .\tag{ECE}
$$

In classification, confidence measures are predominantly built on model logits (Guo et al., 2017; Kull et al., 2019). However, these methods are less suitable for NLG tasks. First, the label space is often too large to assess correctness via ${ \widehat { Y } } = Y$ , since LMs produce potentially long textual responses $\widehat { Y }$ for any given input. Second, for LMs, logits encode the likelihood of selecting the next token and do not necessarily capture linguistic sense (Mielke et al., 2022). Third, even hand-crafted prompts intended to make LMs express confidence explicitly may not lead to reliable confidence values because elicitation is heavily tied to prompt formats (Zhao et al., 2021; Xiong et al., 2024).

Recent works have studied uncertainty measures as an alternative to confidence measures. These capture the “dispersion" of an LMs’ potential outputs for a fixed input. Kuhn et al. (2023) introduce semantic entropy, which incorporates linguistic invariances arising from the shared meaning of generated responses. Lin et al. (2023) extend semantic entropy by leveraging the affinity matrices induced by entailment scores of generated outputs. Further, Chen et al. (2024) characterize differential entropy in the embedding space with EigenScore, via the covariance of embeddings of potential responses.

Uncertainty measures are more general and arguably more principled than confidence measures for LMs, but they lack a universal assessment metric such as ECE. A key issue is that uncertainty measures are not necessarily commensurate. For instance, the semantic entropy (Kuhn et al., 2023) can take arbitrarily large positive values, whereas the EigV measure of Lin et al. (2023) depends on the number of responses generated. This makes it difficult to understand, evaluate, and compare uncertainty measures via a unified lens.

This paper develops a principled framework to assess the quality of uncertainty and confidence measures for LMs. We provide a novel and practical framework, termed Rank-Calibration. Specifically, our contributions are as follows.

• We mathematically formalize the assessment of uncertainty/confidence measures for LMs in NLG tasks, going beyond binary correctness.

• We demonstrate empirically that existing assessment metrics (e.g., AUROC, ECE, etc) have several limitations, including a heavy dependence on the LM's performance, instability caused by ad hoc binarization of correctness scores, and incompatibility with diverse uncertainty ranges.

• We address these limitations by starting from a basic principle: lower uncertainty/higher confidence should indicate higher-quality generation. We thus propose assessing uncertainty measures in terms of rank-calibration and introduce a suitable metric, the Rank-Calibration Error (RCE).

• To make rank-calibration practical, we introduce the Empirical RCE—an estimate of RCE based on a finite dataset. Moreover, we introduce novel indication diagrams, previewed in Fig. 1, that intuitively visualize the deviation of any uncertainty/confidence measure from the monotonicity required for rank-calibration.

• We experimentally demonstrate the broader applicability and granular interpretability of our proposed methods. Comprehensive ablation studies are conducted to examine its robustness.

## 2 Correctness and Uncertainty for LMs

Let V be the token vocabulary of an LM and $\mathcal { V } ^ { \star } : = \cup _ { \ell \geq 1 } \mathcal { V } ^ { \ell }$ the space of sequences of arbitrary length. Given a query $\ b { x } \in \ b { \nu } ^ { \star }$ , an LM M can generate output $\widehat { \pmb { y } } \triangleq ( \widehat { \pmb { y } } _ { \ell } ) _ { \ell \geq 1 } \in \mathcal { V } ^ { \star }$ by sequentially sampling from the distribution $\begin{array} { r } { \mathbb { P } ( \pmb { \widehat { \pmb { y } } } | \pmb { x } ) : = \prod _ { \ell \geq 1 } \mathbb { P } ( \widehat { \pmb { y } } _ { \ell } | } \end{array}$ $x , { \widehat { y } } _ { < \ell } )$ . Here, $\widehat { y } _ { \ell } \in \mathcal { V }$ is the l-th generated token and $\mathbb { P } \triangleq \mathbb { P } ^ { M }$ is the generative distribution of M.

We work with a deterministic correctness function $A : \mathcal { V } ^ { \star } \times \mathcal { V } ^ { \star } $ R mapping each pair $( \pmb { x } ; \widehat { \pmb { y } } )$ to a correctness value $A ( \pmb { x } ; \widehat { \pmb { y } } )$ . In practice, correctness is often not a binary variable in NLG tasks and can be assessed in at least two different ways. For the reader's convenience, the concepts and notations used in the paper are summarized in Table 1.

• Reference matching. Given certain reference answers $\{ \pmb { y } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ associated with x, a similarity score between the output $\widehat { \pmb { y } }$ and $\{ \pmb { y } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ can be interpreted as a correctness value. Similarity scores commonly utilized for this purpose include the Rouge score, BLEU score, and outputs of other discriminative LMs.

• Human evaluation. Correctness or quality may be evaluated by human experts, possibly integrating multiple opinions (e.g., averaging). This approach does not require reference answers and is as “trustworthy"as the humans involved.

An uncertainty measure is a (possibly random) function $U ^ { \mathcal { M } } : \mathcal { V } ^ { \star } \times \mathcal { V } ^ { \star } \to \mathbb { R } , ( \pmb { x } ; \pmb { \widehat { y } } ) \mapsto U ^ { \mathcal { M } } ( \pmb { x } ; \pmb { \widehat { y } } )$ associated with the LM that maps any pair $( { \pmb x } ; \widehat { \pmb y } )$ to an uncertainty value.¹ We will omit M and write $U ( \pmb { x } ; \widehat { \pmb { y } } ) , \mathbb { P } ( \cdot \mid \pmb { x } )$ when the choice of the LM is clear. Some examples are reviewed below, while additional examples and details are in Appendix B.

<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $\nu$ </td><td>Token vocabulary</td></tr><tr><td> $\mathcal { V } ^ { \star }$ </td><td>Space of token sequences</td></tr><tr><td>x</td><td>Input context,  $\ b x \in \ b \nu ^ { \star }$ </td></tr><tr><td> $\widehat { \pmb { y } }$ </td><td>Gen. output  $\widehat { \pmb { y } } = ( \widehat { y } _ { \ell } ) _ { \ell \geq 1 } \in \mathcal { V } ^ { \star }$ </td></tr><tr><td> $\mathbb { P } \triangleq \mathbb { P } ^ { M }$ </td><td>Generative dist. of LM M</td></tr><tr><td> $A ( \cdot ; \cdot )$ </td><td>A deterministic correctness function</td></tr><tr><td> $\{ \pmb { y } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ </td><td>Reference answers for input x</td></tr><tr><td> $U ^ { \mathcal { M } } ( { \pmb x } ; \widehat { \pmb y } )$ </td><td>Uncertainty measure for LM M</td></tr><tr><td> $C ^ { M } ( { \pmb x } ; \widehat { \pmb y } )$ </td><td>Confidence measure for LM M</td></tr><tr><td> $\mathrm { r e g } ( u )$ </td><td>Regression fn.  $\mathbb { E } _ { { \pmb { x } } , { \widehat { \pmb { y } } } } [ A \ | \ U = u ]$ </td></tr></table>

Table 1: Summary of notations.

• NLL. In classification, the softmax of the last-layer logits determines a model's prediction (Guo et al., 2017). In NLG tasks, one can view the Negative Log-Likelihood (NLL),

$$
U _ { \mathrm { N L L } } ( \pmb { x } , \pmb { \widehat { y } } ) { : = } - \ln ( \mathbb { P } ( \pmb { \widehat { y } } \mid \pmb { x } ) ) ,
$$

as an indicator of uncertainty where ${ \widehat { \pmb { y } } } =$ $( \widehat { y } _ { \ell } ) _ { \ell \geq 1 }$ is a generated response. A natural extension accounting for the length of responses applies length normalization; this is also known as the Perplexity measure (Jelinek et al., 1977).

• Entropy. The predictive entropy of the distribution $\mathbb { P } ( \cdot \mid x )$ is large when the same input may lead to diverse outputs, and it is defined as

$$
U _ { \mathrm { E } } ( { \pmb x } ) { : = } { - } \mathbb { E } _ { { \pmb { \hat { y } } } \sim \mathbb { P } ( \cdot | { \pmb x } ) } [ \mathrm { l n } ( \mathbb { P } ( \pmb { \hat { y } } \mid { \pmb x } ) ) ] .
$$

Malinin and Gales (2021) propose a variant of this, $U _ { \mathrm { E - L N } } ( \pmb { x } )$ , utilizing the length-normalized log-likelihood ln $( \mathbb { P } ( \pmb { \widehat { y } } | \pmb { x } ) ) / \mathrm { l e n } ( \pmb { \widehat { y } } )$ . Kuhn et al. (2023) argue that different responses with the same meaning should be viewed as equals in this context, regardless of token-level differences. They propose the semantic entropy,

$$
U _ { \mathrm { S E } } ( \pmb { x } ) { : = - \mathbb { E } _ { \pmb { \hat { y } } \sim \mathbb { P } ( \cdot | \pmb { x } ) } [ \ln ( \mathbb { P } ( c ( \pmb { \hat { y } } ) \mid \pmb { x } ) ) ] , }
$$

where $c ( \widehat { \mathbf { y } } )$ is the semantic concept of ${ \widehat { \mathbf { y } } } ,$ provided by another language modeling method.

• Affinity graph. Lin et al. (2023) calculate uncertainty using a weighted adjacency graph built upon semantic affinities. Consider an $a f f i n i t y$ model e, mapping pairs of responses $\widehat { \mathbf { y } } , \widehat { \mathbf { y } } ^ { \prime }$ to values in $[ 0 , 1 ]$ . Given K independent samples $\{ \widehat { \pmb { y } } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ from $\mathbb { P } ( \cdot | { \pmb x } )$ , the model e induces a symmetric adjacency matrix $W = [ w _ { i , j } ] _ { i , j = 1 } ^ { K } ,$ with $w _ { i , j } = ( e ( \widehat { \pmb y } ^ { ( i ) } ; \widehat { \pmb y } ^ { ( j ) } ) + e ( \widehat { \pmb y } ^ { ( j ) } ; \widehat { \pmb y } ^ { ( i ) } ) ) / 2$ for all $i , j$ . Let $\begin{array} { r } { D = [ \mathbb { 1 } [ j = i ] \sum _ { k = 1 } ^ { K } w _ { k , j } ] _ { i , j = 1 } ^ { K } } \end{array}$ be the corresponding degree matrix and $\{ \lambda _ { k } \} _ { k = 1 } ^ { K }$ be the eigenvalues of the Laplacian $L = I -$ $D ^ { - 1 / 2 } W \bar { D } ^ { - 1 / 2 }$ . Then, the uncertainty measures proposed in Lin et al. (2023) include

$$
\begin{array} { l } { { \displaystyle U _ { \mathrm { E i g V } } ( \pmb { x } ) : = \sum _ { k = 1 } ^ { K } \mathrm { m a x } \{ 0 , 1 - \lambda _ { k } \} , } } \\ { { \displaystyle U _ { \mathrm { D e g } } ( \pmb { x } ) : = 1 - \mathrm { t r a c e } ( D ) / K ^ { 2 } , } } \\ { { \displaystyle U _ { \mathrm { E c c } } ( \pmb { x } ) : = \| [ \pmb { v } _ { 1 } , \pmb { v } _ { 2 } , . . . , \pmb { v } _ { K } ] \| _ { 2 } , } } \end{array}
$$

where $\{ \pmb { v } _ { k } \} _ { k = 1 } ^ { K }$ are suitable vectors associated with $L ,$ see Lin et al. (2023). Intuitively, $U _ { \mathrm { E i g V } } ( { \pmb x } )$ approximately counts the connected components in the graph represented by W, while $U _ { \mathrm { D e g } } ( \pmb { x } )$ and $U _ { \mathrm { E c c } } ( \pmb { x } )$ reflect the diversity of outputs.

The diverse uncertainty measures reviewed above produce outputs with different ranges. For instance, $U _ { \mathrm { N L L } } , U _ { \mathrm { S E } }$ , and $U _ { \mathrm { E i g V } }$ can yield any number in $[ 0 , \infty )$ , whereas $U _ { \mathrm { D e g } }$ and $U _ { \mathrm { E c c } }$ are bounded in [0, 1]; see Fig. 3 [bottom] for a visual illustration. This mismatch in output ranges motivates the need for a novel unified assessment framework.

As we shall see, our assessment framework can handle not only any uncertainty measure but also the closely related concept of confidence measures (Zhao et al., 2021; Mielke et al., 2022; Xiong et al., 2024). A confidence measure can be cast as a (possibly random) function $C ^ { \mathcal { M } } : \mathcal { V } ^ { \star } \times \mathcal { V } ^ { \star }  [ 0 , 1 ]$ $( \pmb { x } ; \widehat { \pmb { y } } ) \mapsto C ^ { \mathcal { M } } ( \pmb { x } ; \widehat { \pmb { y } } )$ with output taking values in $[ 0 , 1 ]$ . Intuitively, confidence and uncertainty measures serve similar purposes, although in a complementary way—high confidence should correlate with low uncertainty, and vice versa.

With this notation in place, we are now ready to state our goals and give a more detailed preview of our proposed framework. Given a benchmark dataset $\{ ( \pmb { x } _ { i } , \{ \pmb { y } _ { i } ^ { ( m ) } \} _ { m = 1 } ^ { M _ { i } } ) \} _ { i = 1 } ^ { n } ,$ where each $M _ { i } \geq 0$ denotes the number of reference answers for ${ \mathbf { } } x _ { i } .$ we aim to quantify the performance of an uncertainty measure $U$ (or a confidence measure C) as follows. First, we obtain the paired values of uncertainty and correctness $\{ ( U ( \pmb { x } _ { i } , \pmb { \widehat { y } } _ { i } ) , A ( \pmb { x } _ { i } ; \pmb { \widehat { y } } _ { i } ) ) \} _ { i = 1 } ^ { n }$ by independently sampling $\widehat { \pmb { y } } _ { i } \sim \mathbb { P } ( \cdot | \mathbf { x } _ { i } )$ for each $1 \leq i \leq n$ Then, we evaluate $\mathcal { E } ( \{ ( U ( \pmb { x } _ { i } , \widehat { \pmb { y } } _ { i } ) , A ( \pmb { x } _ { i } ; \widehat { \pmb { y } } _ { i } ) ) \} _ { i = 1 } ^ { n } )$ for each $1 \leq i \leq n ,$ using a suitable metric $\mathcal { E } . ^ { 2 }$ To account for the randomness in sampling ${ \widehat { \mathbf { y } } } _ { i } ,$ we may draw multiple independent responses $\{ \widehat { \pmb { y } } _ { i } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ iid $\mathbb { P } ( \cdot \mid x _ { i } )$ and take the average as the final result $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \mathcal { E } ( \{ ( U ( \pmb { x } _ { i } , \widehat { \pmb { y } } _ { i } ^ { ( k ) } ) , A ( \pmb { x } _ { i } , \widehat { \pmb { y } } _ { i } ^ { ( k ) } ) ) \} _ { i = 1 } ^ { n } ) / K . } \end{array}$

![](images/efdfb52ad906338fca9978d26fefbe32c63c7ba708300cbe0af0856958a87304.jpg)  
Figure 2: Common workflow for assessing the quality of an LM uncertainty/confidence measure. The key ingredients are: a base LM M (e.g., Llama-2-7b-chat), a correctness function A (e.g., the Rouge-L score), a benchmark dataset $\{ { \pmb x } _ { i } , \{ { \pmb y } _ { i } ^ { ( m ) } \} _ { m = 1 } ^ { M _ { i } } \} _ { i = 1 } ^ { n }$ (e.g., TriviaQA), an assessment metric ε (e.g., AUROC), and the uncertainty measure U $( e . g . , U _ { \mathrm { D e g } } )$ The workflow proceeds in five stages: generation, correctness calculation, correctness discretization, uncertainty quantification, and evaluation. Notably, the threshold τ in correctness discretization is usually chosen heuristically (Kuhn et al., 2023; Xiong et al., 2024; Lin et al., 2023, etc), which can be problematic, as demonstrated in Sec. 3. Our proposed RCE-based assessment removes this stage by using the correctness values directly.

The closest works have been discussed in Sec. 1 and 2, and more related works are reviewed in Appendix A.

## 3 Limitations of Existing Assessments

This section illustrates some limitations of existing assessments for LM uncertainty measures via a case study applying the GPT-3.5-turbo (Ouyang et al., 2022) model on the TriviaQA benchmark (Joshi et al., 2017). We use the validation set of TriviaQA, which contains 11, 322 questionanswer pairs (after deduplication). We use the same prompt template as that in Lin et al. (2023). The template is shown in Appendix E.2.

The uncertainty measures examined here include the negative log-likelihood $U _ { \mathrm { N L L } }$ , the semantic entropy $U _ { \mathrm { S E } }$ (Kuhn et al., 2023), the affinity-graphbased measures $U _ { \mathrm { E i g V } } , U _ { \mathrm { E c c } } .$ and $U _ { \mathrm { D e g } }$ (Lin et al., 2023), with the affinity determined by the NLI model (He et al., 2021), and the verbalized confidence $C _ { \mathrm { V e r b } }$ (Xiong et al., 2024); see the definitions in Appendix B. These include both white box and grey box measures,³ as well as a diversity of prompt strategies. We use the Rouge-L score as the correctness function A. We follow a common assessment pipeline (Kuhn et al., 2023; Lin et al., 2023; Xiong et al., 2024), as depicted in Fig. 2. The assessment metrics are detailed in Appendix C.

![](images/5a6abbf90110c1fd47800d70b57b2582260097914b6e67236d85d938a1421110.jpg)  
Figure 3: Top: AUROCs of uncertainty/confidence measures with various thresholds. Bottom: Output ranges of uncertainty/confidence measures. Both results are for GPT-3.5-turbo on the TriviaQA benchmark.

Ad hoc correctness thresholding. Most existing assessment metrics $( e . g .$ , AUROC, AUPRC, ECE, etc) are rooted in classification and require binary labels (i.e., A ∈ {True or False}). Consequently, an ad hoc threshold $\tau \in \mathbb { R }$ is often introduced to map continuous correctness values to binary labels, i.e., $\bar { A } _ { \tau } ( \pmb { x } ; \pmb { \widehat { y } } ) : = 1 [ A ( \pmb { x } ; \pmb { \widehat { y } } ) \geq \tau ]$ (Lin et al., 2023; Kuhn et al., 2023). Thus, the response is viewed as correct if the correctness value $A ( \pmb { x } ; \widehat { \pmb { y } } )$ is at least τ, and incorrect otherwise.

However, thresholding can lead to inconsistencies. Taking AUROC as an example, we plot the assessed results of uncertainty/confidence measures under varying thresholds in Fig. 3 [top]. The relative AUROC results of distinct measures vary drastically with the choice of τ. For example, $U _ { \mathrm { N L L } }$ appears inferior to other methods if $\tau < 0 . 2$ , but it becomes the best measure if $\tau > 0 . 8$ This is especially concerning given that there seems to be no principled way to set this threshold. The same limitation also affects other metrics $( e . g . , \mathrm { A U P R C }$ AUARC) and configurations; see Appendix E.4.

Diverse output ranges. The second limitation of existing assessments is rooted in the diverse output ranges of the uncertainty or confidence measures. As shown in Fig. 3 [bottom], the output ranges of different uncertainty measures vary significantly. For example, the values of $U _ { \mathrm { S E } }$ can be higher than 100 while the values of $U _ { \mathrm { E c c } }$ and $U _ { \mathrm { D e g } }$ are small by definition. This diversity of output ranges prevents the direct use of calibration-based metrics such as ECE, which takes variables with inputs in [0, 1].

Strong dependence on LM performance. While the quality of uncertainty/confidence measures should be disentangled from the generation performance of the LM, there is often a strong relation between the two concepts. We argue that many existing metrics (e.g., AUROC, AUPRC, AUARC) can be misleading due to this entanglement. Taking AUARC as an example, if the base LM is powerful and all correctness values of its responses are high (e.g., within [0.9, 1.0]), then the evaluated AUARC will be high for any uncertainty/confidence measure, regardless of its quality. This is undesirable because our goal is to provide an overall assessment of the uncertainty measure, which may in the future need to be applied to different LMs. While the ECE metric provides a limited “disentangling" effect, in the sense that it can reflect that highly accurate models may be poorly calibrated (i.e., with high ECE values) (Guo et al., 2017), it is not applicable to uncertainty measures in general.

Desiderata of evaluation. The aforementioned challenges suggest that the evaluation of LM uncertainty measures should take into account the following key desiderata: (1) avoidance of ad hoc correctness thresholding, (2) applicability to diverse output ranges of uncertainty measures, and (3) decoupling from the generative performance of the LM. Moreover, the evaluation framework should be practical. We view these criteria as important, but not necessarily exhaustive for an ideal assessment. Future research may identify other requisites and further improve our framework accordingly.

## 4 Rank-Calibration

In this section, we introduce a novel assessment framework satisfying the criteria outlined in Sec. 3.

## 4.1 Rank-Calibration & RCE

Define the regression function $\mathrm { r e g } ( \cdot ) : \mathbb { R } \to \mathbb { R } , u \mapsto$ $\mathbb { E } _ { \pmb { x } , \pmb { \hat { y } } } [ A ( \pmb { x } ; \pmb { \hat { y } } ) \ | \ U ( \pmb { x } ; \pmb { \hat { y } } ) = \ u ]$ , representing the expected correctness level A conditional on an uncertainty level $U = u$ . Here, x is a random query sampled from the distribution associated with a specific benchmark dataset, while $\widehat { \pmb { y } } | \pmb { x } \sim \mathbb { P } ( \cdot | \pmb { x } )$ is a random output sampled from the generative distribution of the LM. We start from the observation that, ideally, a lower uncertainty level should correspond to higher generation accuracy. This is equivalent to saying that the regression function should ideally be monotone decreasing.

Since U is a random variable depending on $( { \pmb x } ; \widehat { \pmb y } )$ , reg(U) is also random. If reg(·) is monotonically decreasing, then $U \ \leq \ u$ implies $\operatorname { r e g } ( U ) \geq \operatorname { r e g } ( u )$ . Thus, for any value u in the range of $U _ { : }$

$$
\mathbb { P } ( U \leq u ) = \mathbb { P } ( \mathrm { r e g } ( U ) \geq \mathrm { r e g } ( u ) ) .\tag{1}
$$

Equation (1) suggests a direct relation between an uncertainty level u and its corresponding expected correctness level $\mathrm { r e g } ( u )$ . For example, for a value of u in the in bottom 10% of the distribution of U, the expected correctness level $\mathrm { r e g } ( u ) = \mathbb { E } [ A \ |$ $U = u ]$ is in the top 10% in the distribution of reg $( U ) = \mathbb { E } [ A \mid U ]$ . We call this desired property of uncertainty measures Rank-Calibration.

Definition 1 (RANK-CALIBRATION). We say that an uncertainty measure U is rank-calibrated if (1) holds for any u in U's range: on average, a lower uncertainty implies a higher generative quality.

Rank-calibration is related to, yet distinct from, the usual notion of calibration in the classification context (Lichtenstein et al., 1977; Guo et al., 2017). We defer the detailed discussion to Sec. 4.2. We remark that the principle of rank calibration is also discussed in a concurrent work (Zhang et al., 2024). Unlike our work and (Zhang et al., 2024), (Penha and Hauff, 2021) use the terminology “rank" to denote the relevance comparison of candidate responses in the binning of ECE calculation.

To quantify the distance of a given uncertainty measure from the ideal rank-calibration, we propose the following Rank-Calibration Error (RCE), inspired by ECE for calibration.

Definition 2 (RANK-CALIBRATION ERROR). The RCE of an uncertainty measure U is defined as

$$
\begin{array} { r } { \mathbb { E } _ { U } \left[ \left| \mathbb { P } _ { U ^ { \prime } } ( \mathrm { r e g } ( U ^ { \prime } ) \geq \mathrm { r e g } ( U ) ) - \mathbb { P } _ { U ^ { \prime } } ( U ^ { \prime } \leq U ) \right| \right] , } \end{array}\tag{RCE}
$$

where $U ^ { \prime }$ is an independent copy of U.

Extension to confidence measures. While primarily motivated by uncertainty measures with incommensurate ranges, rank-calibration also applies to confidence measures. Ideally, higher values $o f \epsilon$ a confidence measure should imply higher generation accuracy. Thus, defining ${ \overline { { \operatorname { r e g } } } } ( c ) : = \mathbb { E } [ A \mid C = c ]$ for all c in the range of $C ,$ we can adapt RCE to

$$
\begin{array} { r } { \mathbb { E } _ { C } \left[ \left| \mathbb { P } _ { C ^ { \prime } } ( \overline { { \mathrm { r e g } } } ( C ^ { \prime } ) \geq \overline { { \mathrm { r e g } } } ( C ) ) - \mathbb { P } _ { C ^ { \prime } } ( C ^ { \prime } \geq C ) \right| \right] , } \end{array}\tag{2}
$$

where $C ^ { \prime }$ is an independent copy of $C .$ This gauges deviations from the equivalence between $C \geq c$ and ${ \overline { { \operatorname { r e g } } } } ( C ) \geq { \overline { { \operatorname { r e g } } } } ( c )$ . Since rank-calibration provides a different perspective from calibration—see Sec. 4.2—(2) serves as a supplement to ECE in assessing confidence measures.

## 4.2 Comparison with Classical Calibration

For a binary correctness value function A taking values in $\{ 0 , 1 \}$ , rank-calibration relaxes classical calibration by absorbing all strictly decreasing transformations.

Theorem 1. Suppose the correctness function A takes values in {0, 1}. If an uncertainty measure U is rank-calibrated, i.e., its RCE is zero, then there exists a unique strictly decreasing transformation $g ^ { \star } \colon \mathbb { R } \to [ 0 , 1 ]$ such that $C _ { g ^ { \star } } : = g ^ { \star } ( U )$ is calibrated, i.e., its ECE is zero. If a confidence measure $C$ is calibrated, then for any strictly decreasing transformation $h \colon \mathbb { R } \to \mathbb { R }$ , the induced uncertainty measure $U _ { h } : = h ( C )$ is rank-calibrated.

Proof. If U is rank-calibrated, the regression function $u \ \mapsto \ \mathrm { r e g } ( u ) = \mathbb { E } [ A \ | \ U = u ] \ \in \ [ 0 , 1 ]$ is strictly decreasing over all values in $U ^ { \star } { } _ { \mathbf { s } }$ range with positive density (or mass). Moreover, $\mathbb { P } ( A = 1 \ |$ $\operatorname { r e g } ( U ) = \operatorname { r e g } ( u ) ) = \operatorname { \mathbb { E } } [ A | U = u ] = \operatorname { r e g } ( u )$ . Therefore, $\operatorname { r e g } ( U )$ is a calibrated confidence measure, and reg is strictly decreasing. The uniqueness follows as $\mathbb { P } ( A = 1 \mid g ( U ) ) = \mathbb { E } [ A \mid U ] = \mathrm { r e g } ( U )$ for any strictly monotone function.

On the other hand, if C is calibrated, then $C =$ $\mathbb { P } ( A = 1 \mid C ) = \mathbb { E } [ A \mid C ]$ almost surely. For any strictly decreasing h, we have $\mathbb { E } [ A \mid U _ { h } ] = \mathbb { E } [ A \mid$ $C ] = C$ almost surely because $h$ is a one-to-one map. Therefore, for any given c and uncertainty value $u _ { h } = h ( c )$ , it holds almost surely that

$$
U _ { h } = h ( C ) \leq u _ { h } = h ( c ) \iff C \geq c
$$

$$
\iff \mathbb { E } [ A \mid C ] \geq \mathbb { E } [ A \mid C = c ]
$$

$$
\iff \mathbb { E } [ A \mid U _ { h } ] \geq \mathbb { E } [ A \mid U _ { h } = u _ { h } ] ,
$$

which implies $U _ { h }$ is rank-calibrated.

Theorem 1 implies that, for a binary correctness function, one can construct a calibrated confidence measure from an uncertainty measure with monotone transformations if and only if the uncertainty measure is rank-calibrated. However, RCE and ECE gauge different quantities: ECE captures the absolute difference between the predicted and true probabilities, while RCE reflects the deviation from a monotonic correspondence between uncertainty and the expected correctness. These two notions are generally not directly comparable.

For example, consider the special case where a continuous-valued confidence measure $C$ is completely uninformative and the regressed correctness ${ \overline { { \operatorname { r e g } } } } : c \mapsto \mathbb { E } [ A | C = c ]$ is a constant for all confidence levels $c .$ Then, the RCE defined in (2) reports a large value of $1 / 2$ , reflecting its poor indicativeness. However, the ECE can be large or small depending on the averaged distance between $C ^ { \bullet } \mathrm { { s } }$ output and reg. More generally, we find no relation in the results of ECE and RCE through the following result, proved in Appendix D.

Proposition 1. Let the correctness function $A \in$ $\{ 0 , 1 \}$ be binary. For any $\alpha , \beta \in ( 0 , 1 / 2 ]$ , there is a confidence measure $C$ such that its RCE is α while the ECE is $\beta .$

## 4.3 Empirical RCE & Indication Diagram

Now, as in Sec. 2, consider a dataset $\{ ( u _ { i } , a _ { i } ) \} _ { i = 1 } ^ { n }$ of uncertainty and correctness values computed over a benchmark dataset where each $\begin{array} { r l } { u _ { i } } & { { } = } \end{array}$ $U ( \pmb { x } ; \widehat { \pmb { y } } _ { i } ) , a _ { i } = A ( \pmb { x } _ { i } ; \widehat { \pmb { y } } _ { i } )$ , and $\widehat { \mathbf { y } } _ { i }$ is a response generated by the LM. The true value of RCE is unknown, as it refers to an average over the distribution from which the data are drawn.

Empirical RCE. The RCE involves the unknown probabilities $\mathbb { P } ( U \leq u )$ and $\mathbb { P } ( \mathrm { r e g } ( U ) \geq \mathrm { r e g } ( u ) )$ which generally need to be estimated. Estimating the latter is challenging as the regression function is also unknown and needs to be estimated.

To address this, we adopt a piecewise constant regression or binning strategy, as in non-parametric statistics (Tsybakov, 2009). First, we group the uncertainty values $\{ u _ { i } \} _ { i = 1 } ^ { n }$ into B equal-mass intervals, each containing $\lceil n / B \rceil { - } 0 \Gamma$ , when needed, $\lfloor n / B \rfloor$ —elements. The boundaries of the b-th $( 1 \leq b \leq B )$ bin are the $( b - 1 ) / B \mathrm { { \cdot } t h }$ and $b / B { \cdot } \mathrm { t h }$ quantiles of $( u _ { i } ) _ { i = 1 } ^ { n }$ . Let ${ \mathcal { T } } _ { b } \subseteq \{ 1 , \dots , n \}$ be the set of indices of the datapoints whose uncertainty values fall into the b-th bin. The expected correctness level over the b-th bin can be estimated as

$$
\mathrm { c r c } _ { b } : = \frac { 1 } { | \mathcal { T } _ { b } | } \sum _ { i \in \mathcal { T } _ { b } } a _ { i } ,
$$

when $\left. \mathcal { T } _ { b } \right. > 0$ From now on, we will interpret $0 / 0 : = \ 0 ;$ and we extend to $\left. \mathcal { T } _ { b } \right. = 0$ in this way. Clearly, crcb is an unbiased estimator of E $[ A \mid U \in$ the i-th bin], which approximates reg(U) accurately given a narrow bin and abundant data. We similarly estimate the average uncertainty within the b-th bin as

$$
\mathrm { u c t } _ { b } = \frac { 1 } { | \mathcal { T } _ { b } | } \sum _ { i \in \mathcal { T } _ { b } } u _ { i } .
$$

As crc and uct estimate the per-bin averages of reg(U) and U, for each b, we estimate I $\mathbb { P } ( U \leq u _ { i } )$ and $\mathbb { P } ( \mathrm { r e g } ( U ) \geq \mathrm { r e g } ( u _ { i } ) )$ for $i \in \mathcal { Z } _ { b }$ as follows:

$$
{ \widehat { \mathbb { P } } } ( \mathrm { r e g } ( U ) \geq \mathrm { r e g } ( u _ { i } ) ) { : = } \frac { 1 } { B - 1 } \sum _ { b ^ { \prime } \neq b } \mathbb { 1 } [ \mathrm { c r c } _ { b ^ { \prime } } \geq \mathrm { c r c } _ { b } ] ,
$$

$$
{ \widehat { \mathbb { P } } } ( U \leq u _ { i } ) { : = } \frac { 1 } { B - 1 } \sum _ { b ^ { \prime } \neq b } \mathbb { 1 } [ \mathrm { { u c t } } _ { b ^ { \prime } } \leq \mathrm { { u c t } } _ { b } ] .
$$

A rank-calibrated measure has ${ \widehat { \mathbb { P } } } ( U \leq u _ { i } )$ ≈ $\widehat { \mathbb { P } } ( \mathrm { r e g } ( U ) \geq \mathrm { r e g } ( u _ { i } ) )$ for all $1 \leq i \leq n$ . We thus compute the empirical Rank-Calibration Error estimator (Empirical RCE) by taking an average of the per-bin rank differences of correctness and uncertainty values. More precisely,

$$
{ \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \left| { \widehat { \mathbb { P } } } ( \operatorname { r e g } ( U ) \geq \operatorname { r e g } ( u _ { i } ) ) - { \widehat { \mathbb { P } } } ( U \leq u _ { i } ) \right| .
$$

(Empirical RCE)

The difference between the estimated probabilities for a given bin represent the ranking gap (i.e., blue and shallow red areas in Fig. 1). We use the Empirical RCE as the main metric to assess uncertainty and confidence measures in the paper.

Indication diagram. Similar to reliability diagrams representing miscalibration (Lichtenstein et al., 1977; Niculescu-Mizil and Caruana, 2005), we can also visualize rank-miscalibration in diagrams $( e . g . , \operatorname { F i g . } 1 )$ . In particular, we plot the relative percentile (between 0% and 100%) of the expected correctness level $( i . e . , \mathrm { r e g } ( U ) )$ as a function of the relative percentile of uncertainty (i.e., U). We term these plots indication diagrams. If a measure is rank-calibrated—i.e., if (1) holds—then the indication diagram should lie on the anti-diagonal line percent(reg(u)) = 1-percent(u). Deviations from this line represent rank-miscalibration.

Advantages of rank-calibration. We summarize the advantages of the rank-calibration framework by revising the desiderata from Sec. 3. First the empirical RCE does not require any thresholding of the correctness values. Second, rankcalibration assesses the monotonicity of uncertainty values by leveraging relative ranks, which makes it independent of the output range. Third, similar to ECE, the RCE is not directly tied to the generation performance of the LM. Finally, our assessment is practical for any uncertainty/confidence measures.

## 5 Experiments

We provide more comprehensive experiments and justify the advantages of our assessment.

## 5.1 Experiment Setup

We consider both open-source and commercial LMs, including Llama-2-7b, Llama-2-7b-chat (Touvron et al., 2023b) (an instruction fine-tuned version of Llama-2-7b), and GPT-3.5-turbo (Ouyang et al., 2022). See Appendix E.1 for more details.

We conduct assessments on the validation sets of four datasets: TriviaQA (Joshi et al., 2017), Natural Questions (Kwiatkowski et al., 2019), SQuAD-1 (Rajpurkar et al., 2016), and Meadow (Wang et al., 2020). For assessment over the open-ended and challenging Meadow, we only use the more advanced model GPT-3.5-turbo. To account for randomness in the evaluation, we repeat experiments bootstrapping each dataset 20 times. See more details of datasets in Appendix E.2.

We use multiple correctness functions, including the Rouge-L score, BERT similarity, and ChatGPT evaluation, all widely applied before (Kuhn et al., 2023; Xiong et al., 2024). ChatGPT correctness is only used for GPT-3.5-turbo with temperature 1.0. See Appendix E.3 for more details.

The uncertainty/confidence measures to be assessed are the same as in Sec. $3 , ( i . e . , U _ { \mathrm { N L L } } , U _ { \mathrm { S E } } ,$ $U _ { \mathrm { E c c } } , U _ { \mathrm { D e g } } , U _ { \mathrm { E i g V } }$ , and $C _ { \mathrm { V e r b } } )$ . We first illustrate that our proposed assessment has broad applicability and granular interpretability. Furthermore, we qualitatively show that uncertainty measures with lower RCE values reliably indicate correctness. Finally, we study robustness by empirically checking the impact of temperature and correctness functions on RCE (Demšar, 2006). More results for different configurations are given in Table 4.

<table><tr><td>Dataset</td><td>Correctness</td><td>Temperature</td><td> $U _ { \mathrm { E c c } }$ </td><td> $U _ { \mathrm { D e g } }$ </td><td> $U _ { \mathrm { E i g V } }$ </td><td> $U _ { \mathrm { N L L } }$ </td><td> $U _ { \mathrm { S E } }$ </td><td> $C _ { \mathrm { { V e r b } } }$ </td></tr><tr><td rowspan="8">nq-open</td><td rowspan="2">bert</td><td>0.6</td><td> $\underline { { 0 . 1 9 9 _ { \pm 0 . 0 4 0 } } }$ </td><td> $\underline { { \mathbf { 0 . 0 4 6 _ { \pm 0 . 0 0 8 } } } }$ </td><td> $\underline { { 0 . 0 5 2 _ { \pm 0 . 0 1 0 } } }$ </td><td> $\underline { { 0 . 1 0 1 _ { \pm 0 . 0 1 5 } } }$ </td><td> $0 . 0 6 2 _ { \pm 0 . 0 1 0 }$ </td><td>nan</td></tr><tr><td>1.0</td><td> $0 . 2 3 6 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $\mathbf { \overline { { 0 . 0 3 5 _ { \pm 0 . 0 0 8 } } } }$ </td><td> $\overline { { 0 . 0 3 8 _ { \pm 0 . 0 0 7 } } }$ </td><td> $0 . 0 9 7 _ { \pm 0 . 0 1 7 }$ </td><td> $\underline { { 0 . 0 5 5 _ { \pm 0 . 0 1 2 } } }$ </td><td>nan</td></tr><tr><td rowspan="2">meteor</td><td>0.6</td><td> $\overline { { 0 . 1 9 0 { \scriptstyle \pm 0 . 0 3 9 } } }$ </td><td> $\overline { { { \bf 0 . 0 6 2 _ { \pm 0 . 0 0 8 } } } }$ </td><td> $0 . 0 6 7 _ { \pm 0 . 0 1 0 }$ </td><td> $0 . 1 7 6 _ { \pm 0 . 0 1 8 }$ </td><td> $0 . 0 7 2 _ { \pm 0 . 0 0 9 }$ </td><td>nan</td></tr><tr><td>1.0</td><td> $\overline { { 0 . 2 2 4 _ { \pm 0 . 0 3 4 } } }$ </td><td> $\mathbf { 0 . 0 4 4 _ { \pm 0 . 0 0 6 } }$ </td><td> $0 . 0 4 6 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $\underline { { 0 . 2 0 9 _ { \pm 0 . 0 2 3 } } }$ </td><td> $0 . 0 7 4 _ { \pm 0 . 0 1 5 }$ </td><td>nan</td></tr><tr><td rowspan="2">rougeL</td><td>0.6</td><td> $\overline { { 0 . 1 9 8 _ { \pm 0 . 0 3 9 } } }$ </td><td> $\overline { { \mathbf { 0 . 0 5 3 _ { \pm 0 . 0 1 1 } } } }$ </td><td> $\overline { { 0 . 0 5 7 _ { \pm 0 . 0 1 0 } } }$ </td><td> $\overline { { 0 . 1 6 7 _ { \pm 0 . 0 1 3 } } }$ </td><td> $\underline { { 0 . 0 6 0 _ { \pm 0 . 0 1 2 } } }$ </td><td>nan</td></tr><tr><td>1.0</td><td> $\overline { { 0 . 2 2 7 _ { \pm 0 . 0 3 5 } } }$ </td><td> $0 . 0 3 5 _ { \pm 0 . 0 0 7 }$ </td><td> $\overline { { \mathbf { 0 . 0 3 3 _ { \pm 0 . 0 0 6 } } } }$ </td><td> $\underline { { 0 . 2 1 1 \pm 0 . 0 2 1 } }$ </td><td> $\underline { { 0 . 0 6 9 _ { \pm 0 . 0 1 6 } } }$ </td><td>nan</td></tr><tr><td rowspan="2">rougel</td><td>0.6</td><td> $\overline { { 0 . 1 9 9 _ { \pm 0 . 0 3 9 } } }$ </td><td> $\overline { { { \bf 0 . 0 5 4 _ { \pm 0 . 0 1 0 } } } }$ </td><td> $\overline { { 0 . 0 5 7 _ { \pm 0 . 0 1 0 } } }$ </td><td> $0 . 1 6 7 _ { \pm 0 . 0 1 4 }$ </td><td> $\overline { { 0 . 0 6 1 _ { \pm 0 . 0 1 3 } } }$ </td><td>nan</td></tr><tr><td>1.0</td><td> $\overline { { 0 . 2 2 7 _ { \pm 0 . 0 3 5 } } }$ </td><td> $0 . 0 3 4 _ { \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 0 3 3 _ { \pm 0 . 0 0 6 } }$ </td><td> $0 . 2 1 2 _ { \pm 0 . 0 2 1 }$ </td><td> $0 . 0 6 9 _ { \pm 0 . 0 1 5 }$ </td><td>nan</td></tr><tr><td rowspan="8">squad</td><td rowspan="2">bert</td><td>0.6</td><td> $\overline { { 0 . 2 0 8 _ { \pm 0 . 0 3 3 } } }$ </td><td> $\underline { { 0 . 0 6 5 _ { \pm 0 . 0 1 4 } } }$ </td><td> $0 . 0 7 5 _ { \pm 0 . 0 1 7 }$ </td><td> $\underline { { \mathbf { 0 . 0 4 8 _ { \pm 0 . 0 0 7 } } } }$ </td><td> $\underline { { 0 . 0 6 3 _ { \pm 0 . 0 1 2 } } }$ </td><td>nan</td></tr><tr><td>1.0</td><td> $\overline { { 0 . 2 7 6 { \scriptstyle \pm 0 . 0 3 9 } } }$ </td><td> $0 . 0 6 7 _ { \pm 0 . 0 1 1 }$ </td><td> $0 . 0 6 3 _ { \pm 0 . 0 1 0 }$ </td><td> $\mathbf { 0 . 0 3 8 _ { \pm 0 . 0 0 6 } }$ </td><td> $0 . 0 9 8 _ { \pm 0 . 0 1 2 }$ </td><td>nan</td></tr><tr><td rowspan="2">meteor</td><td>0.6</td><td> $\overline { { 0 . 2 1 6 { \scriptstyle \pm 0 . 0 3 8 } } }$ </td><td> $\overline { { 0 . 3 0 3 _ { \pm 0 . 0 2 6 } } }$ </td><td> $\overline { { 0 . 2 6 5 _ { \pm 0 . 0 2 2 } } }$ </td><td> $\mathbf { 0 . 0 6 3 _ { \pm 0 . 0 1 3 } }$ </td><td> $\overline { { 0 . 1 8 2 _ { \pm 0 . 0 2 9 } } }$ </td><td>nan</td></tr><tr><td>1.0</td><td> $0 . 3 0 0 { \scriptstyle \pm 0 . 0 4 6 }$ </td><td> $\overline { { 0 . 2 9 2 _ { \pm 0 . 0 3 5 } } }$ </td><td> $0 . 2 5 0 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $\mathbf { 0 . 0 6 4 _ { \pm 0 . 0 1 1 } }$ </td><td> $0 . 2 7 4 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td>nan</td></tr><tr><td rowspan="2">rougeL</td><td>0.6</td><td> $\overline { { 0 . 2 3 9 _ { \pm 0 . 0 3 6 } } }$ </td><td> $\overline { { 0 . 1 7 7 _ { \pm 0 . 0 2 6 } } }$ </td><td> $0 . 1 4 3 _ { \pm 0 . 0 2 0 }$ </td><td> $\mathbf { 0 . 0 5 2 _ { \pm 0 . 0 1 1 } }$ </td><td> $\overline { { 0 . 1 2 7 _ { \pm 0 . 0 2 0 } } }$ </td><td>nan</td></tr><tr><td>1.0</td><td> $0 . 3 0 4 { \scriptstyle \pm 0 . 0 3 6 }$ </td><td> $0 . 1 7 9 _ { \pm 0 . 0 3 3 }$ </td><td> $0 . 1 3 7 _ { \pm 0 . 0 2 4 }$ </td><td> $\mathbf { \overline { { 0 . 0 5 3 _ { \pm 0 . 0 1 2 } } } }$ </td><td> $\overline { { 0 . 2 1 0 { \scriptstyle \pm 0 . 0 2 7 } } }$ </td><td>nan</td></tr><tr><td rowspan="2">rougel</td><td>0.6</td><td> $\overline { { 0 . 2 3 8 _ { \pm 0 . 0 3 7 } } }$ </td><td> $\overline { { 0 . 1 8 3 _ { \pm 0 . 0 2 7 } } }$ </td><td> $\overline { { 0 . 1 4 8 _ { \pm 0 . 0 2 2 } } }$ </td><td> $\mathbf { 0 . 0 5 3 _ { \pm 0 . 0 1 0 } }$ </td><td> $\overline { { 0 . 1 2 9 _ { \pm 0 . 0 2 1 } } }$ </td><td>nan</td></tr><tr><td>1.0</td><td> $\overline { { 0 . 3 0 3 _ { \pm 0 . 0 3 5 } } }$ </td><td> $\overline { { 0 . 1 8 5 _ { \pm 0 . 0 3 3 } } }$ </td><td> $\overline { { 0 . 1 4 3 _ { \pm 0 . 0 2 5 } } }$ </td><td> $\overline { { 0 . 0 5 3 _ { \pm 0 . 0 1 2 } } }$ </td><td></td><td></td></tr><tr><td rowspan="8">triviaqa</td><td rowspan="2">bert</td><td>0.6</td><td> $\overline { { 0 . 1 4 0 _ { \pm 0 . 0 2 4 } } }$ </td><td></td><td></td><td></td><td> $\underline { { 0 . 2 1 3 _ { \pm 0 . 0 2 6 } } }$ </td><td>nan</td></tr><tr><td>1.0</td><td> $0 . 2 1 3 _ { \pm 0 . 0 3 0 }$ </td><td> $\underline { { 0 . 0 6 2 _ { \pm 0 . 0 1 6 } } }$ </td><td> $\overline { { 0 . 0 6 1 _ { \pm 0 . 0 1 5 } } }$ </td><td> $\overline { { { \bf 0 . 0 2 0 _ { \pm 0 . 0 0 4 } } } }$ </td><td> $\underline { { 0 . 0 2 7 _ { \pm 0 . 0 0 7 } } }$ </td><td>nan</td></tr><tr><td rowspan="2">meteor</td><td>0.6</td><td> $\overline { { 0 . 1 4 5 _ { \pm 0 . 0 2 7 } } }$ </td><td> $\underline { { 0 . 0 2 5 _ { \pm 0 . 0 0 6 } } }$   $\overline { { 0 . 0 6 7 _ { \pm 0 . 0 1 7 } } }$ </td><td> $0 . 0 3 4 _ { \pm 0 . 0 0 6 }$   $\overline { { 0 . 0 6 4 _ { \pm 0 . 0 1 5 } } }$ </td><td> $\overline { { { \bf 0 . 0 1 4 _ { \pm 0 . 0 0 2 } } } }$   $\overline { { { \bf 0 . 0 3 4 _ { \pm 0 . 0 0 9 } } } }$ </td><td> $0 . 0 3 6 _ { \pm 0 . 0 0 6 }$   $\overline { { 0 . 0 7 5 _ { \pm 0 . 0 1 6 } } }$ </td><td>nan</td></tr><tr><td>1.0</td><td> $0 . 2 0 6 _ { \pm 0 . 0 3 2 }$ </td><td></td><td></td><td></td><td></td><td>nan</td></tr><tr><td rowspan="2">rougeL</td><td>0.6</td><td> $\overline { { 0 . 1 4 1 _ { \pm 0 . 0 2 1 } } }$ </td><td> $\mathbf { 0 . 0 3 5 _ { \pm 0 . 0 0 7 } }$ </td><td> $0 . 0 4 6 _ { \pm 0 . 0 0 5 }$ </td><td> $0 . 0 4 9 _ { \pm 0 . 0 0 8 }$ </td><td> $\underline { { 0 . 0 8 4 _ { \pm 0 . 0 0 7 } } }$ </td><td>nan</td></tr><tr><td>1.0</td><td> $\overline { { 0 . 2 0 4 _ { \pm 0 . 0 3 5 } } }$ </td><td> $\overline { { 0 . 0 6 2 _ { \pm 0 . 0 1 4 } } }$ </td><td> $\underline { { 0 . 0 6 1 _ { \pm 0 . 0 1 4 } } }$ </td><td> $\mathbf { 0 . 0 2 4 _ { \pm 0 . 0 0 5 } }$ </td><td> $\underline { { 0 . 0 3 4 _ { \pm 0 . 0 0 5 } } }$ </td><td>nan</td></tr><tr><td rowspan="2">rouge1</td><td>0.6</td><td> $\overline { { 0 . 1 4 1 _ { \pm 0 . 0 2 1 } } }$ </td><td> $\overline { { 0 . 0 2 7 _ { \pm 0 . 0 0 6 } } }$   $\overline { { 0 . 0 6 2 _ { \pm 0 . 0 1 4 } } }$ </td><td> $\underline { { 0 . 0 4 0 _ { \pm 0 . 0 0 4 } } }$   $\overline { { 0 . 0 6 2 _ { \pm 0 . 0 1 3 } } }$ </td><td> $\mathbf { \overline { { 0 . 0 2 2 _ { \pm 0 . 0 0 2 } } } }$   $\mathbf { 0 . 0 2 4 _ { \pm 0 . 0 0 5 } }$ </td><td> $0 . 0 5 1 _ { \pm 0 . 0 0 7 }$ </td><td>nan</td></tr><tr><td>1.0</td><td> $\overline { { 0 . 2 0 3 _ { \pm 0 . 0 3 5 } } }$ </td><td> $0 . 0 2 7 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 0 4 0 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 0 2 2 _ { \pm 0 . 0 0 2 } }$ </td><td> $0 . 0 3 4 _ { \pm 0 . 0 0 6 }$   $0 . 0 5 1 _ { \pm 0 . 0 0 7 }$ </td><td>nan nan</td></tr></table>

Table 2: RCE results for Llama-2-chat with various experimental configurations.

## 5.2 Broader Applicability

Previous assessments have some limitations in open-ended tasks. First, as shown in Fig. 4 [top], the correctness distribution in open-ended tasks $( e . g .$ , the Meadow dataset) is less concentrated around zero and one compared to the TriviaQA correctness distribution. Consequently, if correctness were binarized with thresholding, the assessed results would be highly impacted by the threshold choice, as illustrated in Fig. 4 [bottom]. As such, using continuous-valued correctness scores is common in open-ended tasks (Cohan et al., 2018; Uppalapati et al., 2023). Since RCE does not require thresholding, our rank-calibration assessment does not suffer from the above issue.

## 5.3 Granular Interpretability

Beyond the rank-calibration error, the indication diagrams can be instrumental in understanding the performance of uncertainty measures. We show the indication diagrams of $U _ { \mathrm { { N L I } } }$ and $U _ { \mathrm { S E } }$ for GPT-3.5-turbo on TriviaQA in Fig. 1. More indication diagrams can be found in the Appendix.

![](images/c8fb3019f0dfa0bb46e61d5c3831ea470f1e0bbefac1b757879cba5280306ec9.jpg)

![](images/54a8938e3b27b2fdc55511be15242be107e13b5734095aa08b40175ede834f9d.jpg)

First, indication diagrams consistently reflect the effect of rank-miscalibration. The indication diagram of $U _ { \mathrm { N L L } }$ (Fig. 1 [left]) has more overlap between the red and blue bars, compared to that of $U _ { \mathrm { E c c } }$ (Fig. 1 [right]), reflecting a lower RCE level (0.038 with $U _ { \mathrm { N L L } }$ v.s. 0.151 with $U _ { \mathrm { E c c } } )$ . The high overlap suggests that the relative ranks of uncertainty values are more aligned with those of correctness levels, leading to better rank-calibration.

![](images/39532031b9905526ea732ebe8f86551be24f052e9da127ad61f49e3930d5bc20.jpg)  
Figure 4: Top: Rouge-L correctness distributions of GPT-3.5-turbo on the TriviaQA (left) and Meadow (right) benchmarks. Bottom: AUROCs of assessed measures for GPT-3.5-turbo on Meadow, with Rouge-L correctness and various thresholds.

Second, indication diagrams can shed light onto which uncertainty levels may be problematic. For example, in Fig. 1 [right], we observe that for an uncertainty in the top 75th percentile, $U _ { \mathrm { E c c } }$ tends to be overpessimistic: $U _ { \mathrm { E c c } }$ assigns high uncertainty values to high-quality generations.

## 5.4 Qualitative Illustration

To illustrate the effectiveness of the RCE as an evaluation metric for uncertainty measures, we present two TriviaQA instances and contrast $U _ { \mathrm { N L L } }$ (having RCE 0.037) with $U _ { \mathrm { S E } }$ (having RCE 0.051) for GPT-3.5. Here, x is the question input, y is the answer in the dataset, $\widehat { \pmb { y } }$ is the LM response, and $\mathbb { P } ( U \leq u )$ signifies the relative magnitudes of LM's uncertainty level according to $U _ { \mathrm { { N L I } } }$ and $U _ { \mathrm { S E } }$

In the first instance, the generation is factually incorrect and $U _ { \mathrm { { N L L } } }$ assigns a high uncertainty value to the response, i.e. $\mathbb { P } ( U _ { \mathrm { N L L } } \leq u ) \approx 1$ . In the second scenario, where the generation is correct, $U _ { \mathrm { { N L L } } }$ succeeds in providing a lower uncertainty level, $i . e .$ $\mathbb { P } ( U _ { \mathrm { N L L } } \leq u ) \approx 0$ . Yet, $U _ { \mathrm { S E } }$ assigns a lower uncertainty to a poorer generation and a higher uncertainty to a better generation! These instances showcase that $U _ { \mathrm { { N L I } } }$ , is more reliable than $U _ { \mathrm { S E } }$ here, which is consistent with the RCE-assessed results. Additional qualitative results are given in Table 5.

x: On September 28th,   
NASA announced that what   
had been detected on Mars?   
y: flowing water   
y: Possible signs of life   
P $( U _ { \mathrm { S E } } \leq u )$ 0.813   
$\mathbb { P } ( U _ { \mathrm { N L L } } \leq u )$ 0.930   
x: "Feel Like Making   
Love″ and "The First Time   
Ever I Saw Your Face″ were   
hit singles for which fe  
male artist?   
y: roberta flack   
y: Roberta Flack   
P(USE ≤u): 0.864   
P(UNLL ≤u)： 0.046

## 5.5 Post-hoc Recalibration

Recalibrating uncertainty/confidence measures with poor rank-calibration can be of interest; for ECE, this is sometimes known as Mincer-Zamowitz regression (Mincer and Zarnowitz, 1969). As discussed in Sec. 4.2, an ECE-calibrated measure is also RCE-calibrated. However, RCE is invariant to monotone transformations, which means that approaches like Platt scaling (Platt, 1999) and isotonic regression (Zadrozny and Elkan, 2002) will not improve rank-calibration. Therefore, we suggest using histogram binning (or, piecewise constant regression), which includes non-monotone transforms (Zadrozny and Elkan, 2001). Table 3 and Fig. 10 and 11 list the RCE results of $U _ { \mathrm { S E } }$ for GPT-3.5-turbo before and after calibration. We observe the calibrated measure is significantly better rank-calibrated, showing the effectiveness of this strategy. See the more experimental details and results in Appendix F.2.

<table><tr><td>Dataset</td><td>Correctness</td><td>Temperature</td><td> $U _ { \mathrm { S E } }$ </td><td> $U _ { \mathrm { S E , c a l } }$ </td></tr><tr><td rowspan="4">meadow</td><td>bert</td><td>1.0</td><td>0.177±0.027</td><td>0.083±0.016</td></tr><tr><td>meteor</td><td>1.0</td><td>0.132±0.018</td><td>0.066±0.015</td></tr><tr><td>rougeL</td><td>1.0</td><td>0.113±0.022</td><td>0.063±0.014</td></tr><tr><td>rouge1</td><td>1.0</td><td>0.113±0.018</td><td>0.061±0.012</td></tr><tr><td rowspan="4">nq-open</td><td>bert</td><td>1.0</td><td>0.050±0.007</td><td>0.026±0.007</td></tr><tr><td>meteor</td><td>1.0</td><td>0.060±0.009</td><td>0.033±0.011</td></tr><tr><td>rougeL</td><td>1.0</td><td>0.052±0.008</td><td>0.030±0.010</td></tr><tr><td>rouge1</td><td>1.0</td><td>0.051±0.008</td><td>0.029±0.010</td></tr><tr><td rowspan="4">squad</td><td>bert</td><td>1.0</td><td>0.113±0.013</td><td>0.050±0.013</td></tr><tr><td>meteor</td><td>1.0</td><td>0.086±0.014</td><td>0.046±0.010</td></tr><tr><td>rougeL</td><td>1.0</td><td>0.100±0.011</td><td>0.037±0.008</td></tr><tr><td>rouge1</td><td>1.0</td><td>0.103±0.011</td><td>0.039±0.007</td></tr><tr><td rowspan="13">triviaqa</td><td>bert</td><td>0.5</td><td>0.052±0.009</td><td>0.030±0.010</td></tr><tr><td rowspan="3"></td><td>1.0</td><td>0.052±0.012</td><td>0.027±0.008</td></tr><tr><td>1.5</td><td>0.081 ±0.009</td><td>0.029±0.007</td></tr><tr><td>0.5</td><td>0.234±0.019</td><td>0.058±0.015</td></tr><tr><td rowspan="3">meteor</td><td>1.0</td><td>0.209±0.012</td><td>0.047±0.014</td></tr><tr><td>1.5</td><td>0.176±0.015</td><td>0.036±0.012</td></tr><tr><td>0.5</td><td>0.050±0.008</td><td>0.028±0.007</td></tr><tr><td rowspan="2">rougeL</td><td>1.0</td><td>0.059±0.009</td><td>0.026±0.007</td></tr><tr><td>1.5</td><td>0.104±0.007</td><td>0.028±0.006</td></tr><tr><td rowspan="3">rougel</td><td>0.5</td><td>0.050±0.008</td><td>0.028±0.006</td></tr><tr><td>1.0</td><td>0.060±0.009</td><td>0.027±0.006</td></tr><tr><td>1.5</td><td>0.105±0.008</td><td>0.028±0.008</td></tr></table>

Table 3: RCE results of $U _ { \mathrm { S E } }$ and $U _ { \mathrm { S E , c a l } }$ after rankcalibration for GPT-3.5-turbo with various experimental configurations.

## 5.6 Robustness Analysis

We conduct ablation studies to analyze the robustness of our assessment to key hyperparameters, including temperatures, correctness scores, and sample sizes. We further propose a method to make robust comparisons between uncertainty measures via the Critical Difference (CD) Diagram (Demšar, 2006). Detailed information and results are in Appendix F.4.

## 6 Conclusion

This paper investigates the limitations of common assessments for LM uncertainty/confidence measures. We develop an alternate framework, termed rank-calibration, to assess their quality. Our approach does not require binarizing correctness at ad hoc thresholds and is compatible with uncertainty measures taking values in any output range. We experimentally show the broad applicability and the granular interpretability of our method, and provide a comprehensive robustness analysis. Future directions include developing uncertainty measures with guaranteed rank-calibration and enhancing generative pipelines of LMs $( e . g .$ , the retrieval-augmented generation) with rank-calibrated measures.

## Ackowledgement

This research was partially supported by the NSF, ONR, AFOSR, ARO, and Sloan Foundation, ARO Award W911NF20-1-0080, EnCORE, TILOS, and a GenAI Research Grant from the University of Southern California.

## Limitation & Broader Impact

The empirical RCE estimate has not been subjected to a thorough statistical analysis. The performance of assessed uncertainty and confidence measures (e.g., the vanilla verbalized confidence $C _ { \mathrm { V e r b } } )$ have not been optimized, since the paper focuses on a new assessment approach rather than benchmarking. Human correctness evaluation is not performed, due to our limited budget.

This work is designed to unveil the issues in the existing approaches for evaluating LM uncertainty/confidence measures, and to introduce an alternate, principled assessment to the LM community. We believe there are no ethical concerns associated with our research.

## References

Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, pages 65–72, Ann Arbor, Michigan. Association for Computational Linguistics.

Steven Bird, Ewan Klein, and Edward Loper. 2009. Natural language processing with Python: analyzing text with the natural language toolkit. "O'Reilly Media, Inc.".

Glenn W Brier. 1950. Verification of forecasts expressed in terms of probability. Monthly weather review, 78(1):1–3.

Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. 2024. INSIDE: LLMs' internal states retain the power of hallucination detection. In The Twelfth International Conference on Learning Representations.

Jiuhai Chen and Jonas Mueller. 2023. Quantifying uncertainty in answers from any language model via intrinsic and extrinsic confidence assessment. arXiv preprint arXiv:2308.16175.

Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, and Nazli Goharian. 2018. A discourse-aware attention model for abstractive summarization of long documents.

Morris H DeGroot and Stephen E Fienberg. 1983. The comparison and evaluation of forecasters. Journal of the Royal Statistical Society: Series D (The Statistician), 32(1-2):12–22.

Janez Demšar. 2006. Statistical comparisons of classifiers over multiple data sets. The Journal of Machine learning research, 7:1–30.

Yarin Gal and Zoubin Ghahramani. 2016. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pages 1050–1059. PMLR.

Tilmann Gneiting and Adrian E Raftery. 2007. Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association, 102(477):359–378.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. 2017. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR.

Kartik Gupta, Amir Rahimi, Thalaiyasingam Ajanthan, Thomas Mensink, Cristian Sminchisescu, and Richard Hartley. 2021. Calibration of neural networks using splines. In International Conference on Learning Representations.

Frank E Harrell. 2015. Regression modeling strategies with applications to linear models, logistic and ordinal regression, and survival analysis.

Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2021. Deberta: Decoding-enhanced bert with disentangled attention.

Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.

Xinmeng Huang, Shuo Li, Edgar Dobriban, Osbert Bastani, Hamed Hassani, and Dongsheng Ding. 2024. One-shot safety alignment for large language models via optimal dualization. arXiv preprint arXiv:2405.19544.

Siddhartha Jain, Ge Liu, Jonas Mueller, and David Gifford. 2020. Maximizing overall diversity for improved uncertainty estimates in deep ensembles. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 4264–4271.

Fred Jelinek, Robert L Mercer, Lalit R Bahl, and James K Baker. 1977. Perplexity—a measure of the difficulty of speech recognition tasks. The Journal of the Acoustical Society of America, 62(S1):S63–S63.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. 2022. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In The Eleventh International Conference on Learning Representations.

Meelis Kull, Miquel Perello Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, and Peter Flach. 2019. Beyond temperature scaling: Obtaining wellcalibrated multi-class probabilities with dirichlet calibration. Advances in neural information processing systems, 32.

Ananya Kumar, Percy S Liang, and Tengyu Ma. 2019. Verified uncertainty calibration. Advances in Neural Information Processing Systems, 32.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. 2017. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30.

Donghwan Lee, Xinmeng Huang, Hamed Hassani, and Edgar Dobriban. 2023. T-cal: An optimal test for the calibration of predictive models. Journal of Machine Learning Research, 24(335):1–72.

Shiyu Liang, Yixuan Li, and R. Srikant. 2018. Enhancing the reliability of out-of-distribution image detection in neural networks. In International Conference on Learning Representations.

Sarah Lichtenstein, Baruch Fischhoff, and Lawrence D Phillips. 1977. Calibration of probabilities: The state of the art. In Decision Making and Change in Human Affairs: Proceedings of the Fifth Research Conference on Subjective Probability, Utility, and Decision Making, Darmstadt, 1–4 September, 1975, pages 275– 324. Springer.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Teaching models to express their uncertainty in words. Transactions on Machine Learning Research.

Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. 2023. Generating with confidence: Uncertainty quantification for black-box large language models.

Andrey Malinin and Mark Gales. 2021. Uncertainty estimation in autoregressive structured prediction. In International Conference on Learning Representations.

Potsawee Manakul, Adian Liusie, and Mark JF Gales. 2023. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. arXiv preprint arXiv:2303.08896.

Meta. 2023. Llama access request form meta ai. https://ai.meta.com/resources/ models-and-libraries/1llama-downloads/. (Accessed on 12/13/2023).

Sabrina J Mielke, Arthur Szlam, Emily Dinan, and Y-Lan Boureau. 2022. Reducing conversational agents' overconfidence through linguistic calibration. Transactions of the Association for Computational Linguistics, 10:857–872.

Jacob A Mincer and Victor Zarnowitz. 1969. The evaluation of economic forecasts. In Economic forecasts and expectations: Analysis of forecasting behavior and performance, pages 3–46. NBER.

Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. 2015. Obtaining well calibrated probabilities using bayesian binning. In Proceedings of the AAAI conference on artificial intelligence, volume 29.

Alexandru Niculescu-Mizil and Rich Caruana. 2005. Predicting good probabilities with supervised learning. In International Conference on Machine Learning, pages 625–632.

Jeremy Nixon, Michael W Dusenberry, Linchuan Zhang, Ghassen Jerfel, and Dustin Tran. 2019. Measuring calibration in deep learning. In CVPR workshops, volume 2.

OpenAI. 2023. Gpt-4 technical report.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback.

Georgios Papadopoulos, Peter J Edwards, and Alan F Murray. 2001. Confidence estimation methods for neural networks: A practical comparison. IEEE transactions on neural networks, 12(6):1278–1287.

Nicolas Papernot and Patrick McDaniel. 2018. Deep k-nearest neighbors: Towards confident, interpretable and robust deep learning. arXiv preprint arXiv:1803.04765.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward

Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32, pages 8024–8035. Curran Associates, Inc.

Gustavo Penha and Claudia Hauff. 2021. On the calibration and uncertainty of neural learning to rank models for conversational search. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 160–170.

John Platt. 1999. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3):61–74.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks.

Carlos Riquelme, George Tucker, and Jasper Snoek. 2018. Deep bayesian bandits showdown: An empirical comparison of bayesian deep networks for thompson sampling. In International Conference on Learning Representations.

Leonard J Savage. 1971. Elicitation of personal probabilities and expectations. Journal of the American Statistical Association, 66(336):783–801.

Chenglei Si, Chen Zhao, Sewon Min, and Jordan Boyd-Graber. 2022. Re-examining calibration: The case of question answering. arXiv preprint arXiv:2205.12507.

Sree Harsha Tanneru, Chirag Agarwal, and Himabindu Lakkaraju. 2023. Quantifying uncertainty in natural language explanations of large language models. arXiv preprint arXiv:2311.03533.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher Manning. 2023. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5433–5442.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti

Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models.

Alexandre B Tsybakov. 2009. Introduction to Nonparametric Estimation. Springer.

Padma Jyothi Uppalapati, Madhavi Dabbiru, . K. Venkata Rao, Omer F. Rana, Rajiv Misra, Alexander Pfeiffer, Luigi Troiano, Nishtha Kesswani, and K. Venkata Rao. 2023. A comprehensive survey on summarization techniques. SN Computer Science, 4:1–9.

Lucy Lu Wang, Kyle Lo, Yoganand Chandrasekhar, Russell Reas, Jiangjiang Yang, Doug Burdick, Darrin Eide, Kathryn Funk, Yannis Katsis, Rodney Michael Kinney, Yunyao Li, Ziyang Liu, William Merrill, Paul Mooney, Dewey A. Murdick, Devvret Rishi, Jerry Sheehan, Zhihong Shen, Brandon Stilson, Alex D. Wade, Kuansan Wang, Nancy Xin Ru Wang, Christopher Wilhelm, Boya Xie, Douglas M. Raymond, Daniel S. Weld, Oren Etzioni, and Sebastian Kohlmeier. 2020. CORD-19: The COVID-19 open research dataset. In Proceedings of the 1st Workshop on NLP for COVID-19 at ACL 2020, Online. Association for Computational Linguistics.

Laura Weidinger, John Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, et al. 2021. Ethical and social risks of harm from language models. arXiv preprint arXiv:2112.04359.

Robert L Winkler, Javier Munoz, José L Cervera, José M Bernardo, Gail Blattenberger, Joseph B Kadane, Dennis V Lindley, Allan H Murphy, Robert M Oliver, and David Ríos-Insua. 1996. Scoring rules and the evaluation of probabilities. Test, 5:1–60.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame,

Quentin Lhoest, and Alexander M. Rush. 2020. Huggingface's transformers: State-of-the-art natural language processing.

Yijun Xiao and William Yang Wang. 2021. On hallucination and predictive uncertainty in conditional language generation. arXiv preprint arXiv:2103.15025.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, YIFEI LI, Jie Fu, Junxian He, and Bryan Hooi. 2024. Can LLMs express their uncertainty? an empirical evaluation of confidence elicitation in LLMs. In The Twelfth International Conference on Learning Representations.

Bianca Zadrozny and Charles Elkan. 2001. Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers. In Intertional Conference on Machine Learning, volume 1, pages 609–616.

Bianca Zadrozny and Charles Elkan. 2002. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 694–699.

Caiqi Zhang, Fangyu Liu, Marco Basaldella, and Nigel Collier. 2024. Luq: Long-text uncertainty quantification for llms. arXiv preprint arXiv:2403.20279.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In International Conference on Machine Learning, pages 12697–12706. PMLR.

## A Additional Related Work

Uncertainty measures in supervised learning. The quantification of uncertainties in model outputs in supervised learning has a long history (e.g., Lichtenstein et al., 1977, etc). Overparametrized models such as neural networks pose unique challenges to estimate uncertainty and improve model calibration(Guo et al., 2017; Papadopoulos et al., 2001; Riquelme et al., 2018). Various approaches have been introduced to mimic Bayesian inference (Gal and Ghahramani, 2016), to utilize simple deep ensembles (Lakshminarayanan et al., 2017; Jain et al., 2020), and to identify training samples that are out-of-distribution (Liang et al., 2018; Papernot and McDaniel, 2018). Nonetheless, it is not clear how to adapt these strategies to language modeling, where the output can be text with complex structure.

Uncertainty measures in language modeling. To gauge the uncertainty level associated with the outputs of LMs, Kuhn et al. (2023) introduces the concept of semantic entropy, which integrates linguistic consistencies stemming from shared meanings. In a similar vein, Kadavath et al. (2022); Lin et al. (2022): Xiong et al. (2024) encourage LMs to analyze their own responses and come up with a “probability" that a response is correct. In related work, Manakul et al. (2023) uses sampling to identify instances of fabricated information. Recently, Tian et al. (2023) explore methods for deriving confidence measures for reinforcement-learning-trained LMs. Lin et al. (2023) draw a distinction between estimating uncertainty and confidence for LMs. Similarly, Chen and Mueller (2023) introduce a method for detecting bad and speculative responses from a pre-trained LM with a confidence score. Tanneru et al. (2023) propose two novel measures to quantify the uncertainty of LM-generated explanations. Although considerable research focuses on developing uncertainty and confidence measures for LMs, the evaluation of their effectiveness is less studied.

Assessments of uncertainty measures. Early assessment of confidence measures in classification scenarios leveraged proper scoring rules (Savage, 1971; DeGroot and Fienberg, 1983; Gneiting and Raftery, 2007), such as the Brier score (Brier, 1950) and the KL divergence (Winkler et al., 1996). Other assessments include plotting calibration curves, also known as reliability diagrams (estimated probabilities against predicted ones) (Harrell, 2015). More recently, the ECE metric—or mean absolute calibration error—has gained popularity in machine learning (Harrell, 2015; Naeini et al., 2015), along with many variants (Kumar et al., 2019; Nixon et al., 2019; Gupta et al., 2021; Lee et al., 2023; Si et al., 2022).

In the realm of uncertainty quantification for LMs, the assessment based on ECE remains viable. However, it necessitates the introduction of ad hoc threshold to derive binary labels. Moreover, the applicability of ECE is limited, as it does not directly apply to LM uncertainty measures that fall outside the interval [0, 1]. Our work introduces an assessment centered around rank-calibration, a critical property that ideal uncertainty measures should satisfy. This assessment is applicable to both confidence and uncertainty measures and eliminates the need for thresholding the correctness values.

## B Common Uncertainty/Confidence Measures for LMs

In this section, we introduce common measures of uncertainty and confidence in detail.

• NLL & Perplexity. Let $\widehat { \pmb y } = ( \widehat { y } _ { \ell } ) _ { \ell \ge 1 }$ be the generated response. Then the Negative Log-Likelihood (NLL) is

$$
U _ { \mathrm { N L L } } ( \pmb { x } , \pmb { \widehat { y } } ) : = - \ln ( \mathbb { P } ( \pmb { \widehat { y } } \mid \pmb { x } ) ) = - \sum _ { \ell \geq 1 } \ln ( \mathbb { P } ( \widehat { y } _ { \ell } \mid \pmb { x } , \widehat { y } _ { < \ell } ) ) .
$$

A natural extension accounts for the variable length of responses by applying length normalization. Suppose that the number of tokens of the response  is len(y), the length-normalized NLL is defined as

$$
U _ { \mathrm { N L L - L N } } ( \pmb { x } , \widehat { \pmb { y } } ) : = - \frac { 1 } { \mathrm { l e n } ( \widehat { \pmb { y } } ) } \sum _ { \ell = 1 } ^ { \mathrm { l e n } ( \widehat { \pmb { y } } ) } \mathrm { l n } ( \mathbb { P } ( \widehat { \pmb { y } } _ { \ell } | \pmb { x } , \widehat { \pmb { y } } _ { < \ell } ) ) .
$$

Roughly speaking, this can be viewed as the average nats per token in the generated text; if using log2 instead of ln, it would be the average bits per token. The exponential of the length-normalized NLL is

known as the Perplexity: $U _ { \mathrm { P e r p } } ( \pmb { x } ; \pmb { \widehat { y } } ) : = \exp ( U _ { \mathrm { N L L - L N } } ( \pmb { x } , \pmb { \widehat { y } } ) )$ (Jelinek et al., 1977). The perplexity can also be viewed as the inverse of the geometric mean of the token-wise probabilities.

• Entropy. Entropy is a well-known type of uncertainty measure. The predictive entropy of the distribution $\mathbb { P } ( \cdot \mid x )$ is defined as

$$
U _ { \mathrm { E } } ( { \pmb x } ) : = - \mathbb { E } _ { \pmb { \hat { y } } \sim \mathbb { P } ( \cdot | { \pmb x } ) } [ \mathrm { l n } ( \mathbb { P } ( \pmb { \hat { y } } | { \pmb x } ) ) ] .
$$

Entropy gauges the information one has about the potential output given the input, and has high values when outputs are diverse. Malinin and Gales (2021) propose a variant ${ \cal U } _ { \mathrm { E - L N } } ( { \pmb x } )$ using the length-normalized log-likelihood $\ln ( \mathbb { P } ( \pmb { \mathscr { y } } | \pmb { x } ) ) / \mathrm { L e n g t h } ( \pmb { \widehat { y } } )$ . Kuhn et al. (2023) argues that responses with an identical meaning should be viewed as equal; even if they differ at the token level. They thus propose the semantic entropy

$$
U _ { \mathrm { S E } } ( \pmb { x } ) : = - \mathbb { E } _ { \pmb { \widehat { y } } \sim \mathbb { P } ( \cdot | \pmb { x } ) } [ \mathrm { l n } ( \mathbb { P } ( c ( \pmb { \widehat { y } } ) \mid \pmb { x } ) ) ] ,
$$

where $c ( \widehat { \mathbf { y } } )$ is a semantic concept of output ${ \widehat { \mathbf { y } } } ,$ as determined by another machine learning method. We can similarly define the length-normalized semantic entropy as

$$
\begin{array} { r } { U _ { \mathrm { S E - L N } } ( \pmb { x } ) : = \mathbb { E } _ { \pmb { \widehat { y } } \sim \mathbb { P } ( \cdot | \pmb { x } ) } [ \ln ( \mathbb { P } ( c ( \pmb { \widehat { y } } ) \mid \pmb { x } ) ) / \mathrm { l e n } ( \pmb { \widehat { y } } ) ] . } \end{array}
$$

• Affinity graph. Recently, Lin et al. (2023) use a weighted adjacency graph built upon semantic affinities between outputs to reflect uncertainty. Given an entailment-contradiction affinity model e that maps pairs $\widehat { \mathbf { y } } , \widehat { \mathbf { y } } ^ { \prime }$ of responses to values in [0, 1], e induces a symmetric adjacency matrix $W =$ $[ w _ { i , j } ] _ { i , j = 1 } ^ { K }$ with responses $\{ \widehat { \pmb { y } } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ sampled from $\mathbb { P } ( \cdot | { \pmb x } )$ , where for all $i , j , w _ { i , j } = ( e ( \widehat { \pmb y } ^ { ( i ) } ; \widehat { \pmb y } ^ { ( j ) } ) +$ $e ( \widehat { \pmb y } ^ { ( j ) } ; \widehat { \pmb y } ^ { ( i ) } ) ) / 2$ . Let $\begin{array} { r } { D = [ \mathbb { 1 } [ j = i ] \sum _ { k = 1 } ^ { K } w _ { k , j } ] _ { i , j = 1 } ^ { K } } \end{array}$ be the matrix of degrees and $\{ \lambda _ { k } \} _ { k = 1 } ^ { K }$ be the eigenvalues of the Laplacian $L { = } I { - } D ^ { - 1 / 2 } W D ^ { - 1 / 2 }$ . Measures proposed in Lin et al. (2023) include

$$
\begin{array} { l l } { { \displaystyle U _ { \mathrm { E i g V } } ( { \pmb x } ) : = \sum _ { k = 1 } ^ { K } \mathrm { m a x } \{ 0 , 1 - \lambda _ { k } \} , } } \\ { { \displaystyle U _ { \mathrm { D e g } } ( { \pmb x } ) : = 1 - \mathrm { t r a c e } ( D ) / K ^ { 2 } , } } \\ { { \displaystyle U _ { \mathrm { E c c } } ( { \pmb x } ) : = \| [ { \pmb v } _ { 1 } , { \pmb v } _ { 2 } , . . . , { \pmb v } _ { K } ] \| _ { 2 } . } } \end{array} \qquad  &  { \cal C } _ { \mathrm { D e g } } ( { \pmb x } ; { \hat { \pmb y } ^ { ( i ) } } ) : = D _ { i , i } / K ,
$$

where $\{ \pmb { v } _ { k } \} _ { k = 1 } ^ { K }$ are certain centralized vectors associated with the spectral decomposition of L. Here, $U _ { \mathrm { E i g V } } ( { \pmb x } )$ is approximates the number of connected components in the graph represented by W, while $U _ { \mathrm { D e g } } ( \pmb { x } )$ and $U _ { \mathrm { E c c } } ( \pmb { x } )$ reflect the diversity of outputs.

• Verbalized confidence. Verbalized confidence generally refers to the textual confidence output by an LM. For example, if an LM is highly uncertain about its answer, it may inform the user by saying, e.g. “I am only 20% confident in this answer." This is often implemented by feeding handcrafted prompts to advanced LMs such as GPT-4 (OpenAI, 2023). Many prompting strategies have been used in the literature to enhance this procedure (Zhao et al., 2021; Kadavath et al., 2022; Lin et al., 2022; Xiong et al., 2024). Since optimizing the prompting strategy is not our focus and we do not want confidence elicitation to interfere with the generation of responses, we adopt a simple post-hoc strategy here by feeding a query-response pair to an LM and asking it how confident it believes the response correctly addresses the query. This post-hoc strategy is similar to the one used by Kadavath et al. (2022). We use the following specific prompt format:

Read the question and answer below.   
{question} {generation}   
Provide a numeric confidence that indicates your certainty about   
this answer.   
For instance, if your confidence level is 80%, it means you are 80%

certain that this answer is correct and there is a 20% chance that   
it is incorrect.   
Use the following format to provide your confidence:Confidence:   
[Your confidence, a numerical number in the range of 0-100]%."

## C Common Evaluation Metrics

In this section, we review evaluation metrics that have been commonly used to assess LM uncertainty/confidence measures. These metrics usually require binary correctness values.

• AUROC. AUROC refers to the area under the Receiver-Operating Curve (ROC). The ROC plots the true positive rate (a.k.a. recall) against the false positive rate (a.k.a. 1— specificity) at various thresholds of uncertainty levels. The true positive rate is on the y-axis, and the false positive rate is on the x-axis. An AUROC value of 1 may represent a perfect uncertainty measure; a value of 0.5 suggests no discriminative ability (equivalent to random uncertainty levels). The AUROC can be more useful for evaluation in imbalanced scenarios where correct responses are much more (or less) frequent than incorrect responses.

• AUPRC. AUPRC refers to the area under the Precision-Recall Curve (PRC), which plots the positive predictive value (a.k.a. precision) against the true positive rate (a.k.a. recall) at various threshold settings. Precision is on the y-axis, and recall is on the x-axis. Similar to AUROC, it is valuable in imbalanced dataset scenarios but focuses more on the performance of the positive (minority) class (i.e., correct responses). Variants of AURPC include AUPRC-Positive and AUPRC-Negative, which focus on gauging the ability of uncertainty measures to identify correct responses and incorrect responses, respectively.

• AUARC. AUARC refers to the area under the Accuracy-Rejection Curve (ARC) that plots the accuracy of generation against a rejection rate (the proportion of generated responses for which the model abstains from making a prediction). The curve shows how the accuracy of generation improves as it is allowed to reject uncertain responses. A higher AUARC value means that an LM can generate more correct responses as it increasingly avoids uncertain (based on the level of specific uncertainty measures) cases. This metric is useful for evaluating uncertainty measures in scenarios where LMs can defer responses for which they are not confident.

• ECE. ECE stands for the expected calibration error, a metric used to evaluate the calibration of confidence measures, particularly in classification tasks. Calibration refers to how well the confidence levels align with the actual proportion of correct generation. For an ideally calibrated confidence measure, if the confidence level is 70%, then approximately 70% of generated responses should be correct. ECE quantifies the difference between the confidence levels and the realized correct proportion. A lower ECE indicates better calibration, meaning the confidence measure is more reflective of the actual correct proportion. A confidence measure with an ECE close to zero is considered well-calibrated.

## D Proof of Proposition 1

Case 1. $\alpha = 1 / 2 .$ Consider the continuous case $C \sim \mathrm { U n i f } [ 1 / 2 - \beta , 1 / 2 + \beta ]$ and $\mathrm { r e g } ( C ) \equiv 1 / 2 + \beta$ almost surely $( i . e . , A \sim \mathrm { B e r n o u l l i } ( 1 / 2 + \beta ) )$ . Then $\mathbb { P } _ { C ^ { \prime } } ( \overline { { \mathrm { r e g } } } ( C ^ { \prime } ) \geq \overline { { \mathrm { r e g } } } ( C ) ) \equiv 1$ for almost surely. Since C is continuous-valued, $\mathbb { P } _ { C ^ { \prime } }$ follows the uniform distribution over [0, 1]. We thus have

$$
\mathrm { R C E } = \int _ { 0 } ^ { 1 } | 1 - p | \mathrm { d } p = \frac { 1 } { 2 } .
$$

On the other hand,

$$
\mathrm { E C E } = \int _ { 1 / 2 - \beta } ^ { 1 / 2 + \beta } \frac { | 1 / 2 + \beta - c | } { 2 \beta } \mathrm { d } c = \beta .
$$

Case 2. $\alpha \in ( 0 , 1 / 2 )$ . Consider the case $\mathrm { r e g } ( C ) \equiv 1 / 2 + \beta$ almost surely. We construct the marginal distribution of C as follows. Let $\mathbb { P } ( C = c _ { k } ) = p _ { k }$ for $1 \leq k \leq K$ with $K \ge ( 1 - 2 \alpha ) ^ { - 1 }$ . Here $p _ { 1 } = \cdot \cdot \cdot = p _ { K - 1 } = p$ while $p _ { K } = 1 - ( K - 1 ) p$ where p is the non-negative root of $( K - 1 ) p ^ { 2 } + ( 1$ $( K - 1 ) p ) ^ { 2 } = 1 - 2 \alpha$ .Since $K \geq ( 1 - 2 \alpha ) ^ { - 1 }$ , such $p \in ( 0 , ( K - 1 ) ^ { - 1 } ]$ exists. Moreover, we let $\{ c _ { k } \} _ { k = 1 } ^ { K }$ satisfy $0 \leq c _ { 1 } < \cdot \cdot \cdot < c _ { K - 1 } \leq 1 / 2 + \beta , c _ { k } + c _ { K - k } \equiv 1$ with $c _ { k } \neq 1 / 2$ for all $1 \le k < K , c _ { K } = 1 / 2$ Then, by definition, we can calculate

$$
{ \begin{array} { r l } & { { \mathrm { R C E } } = \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } \left( 1 - \sum _ { \ell \geq k } p _ { \ell } \right) = \sum _ { 1 \leq \ell < k \leq K } p _ { k } p _ { \ell } } \\ & { \qquad \displaystyle = { \frac { \left( \sum _ { k = 1 } ^ { K } p _ { k } \right) ^ { 2 } - \sum _ { k = 1 } ^ { K } p _ { k } ^ { 2 } } { 2 } } = \frac { 1 - \sum _ { k = 1 } ^ { K } p _ { k } ^ { 2 } } { 2 } = \alpha . } \end{array} }
$$

On the other hand, we have

$$
\mathrm { E C E } = \sum _ { k = 1 } ^ { K } \left| \frac { 1 } { 2 } + \beta - c _ { k } \right| p _ { k } = \beta + \frac { 1 } { 2 } - \sum _ { k = 1 } ^ { K } c _ { k } p _ { k } = \beta .
$$

This finishes the proof.

## E Additional Experiment Details

## E.1 Model Setup

Following Lin et al. (2023), we set the temperature to 0.6 for the two Llama-2 models and 1.0 for the GPT model. We quantize the two Llama-2 models to 16 bits. To ablate the influence of temperature, we also use generated responses of Llama-2-7b-chat with temperature 1.0.

## E.2 Datasets

Dataset Descriptions. TriviaQA is a challenging reading comprehension dataset, containing questionanswer pairs whose answers can be found on Wikipedia and the web. Similar to previous works, we use TriviaQA as an open-domain QA benchmark. Natural Question is a question-answering dataset containing questions issued to the Google search engine. We use Natural Questions as an open-domain QA benchmark. SQuAD-1 is a reading comprehension dataset containing questions posed by crowdworkers based on Wikipedia articles. We include SQuAD-1 as a reading comprehension benchmark, where the annotated contexts are provided in the prompt. Meadow is created by research groups working on COVID-19 problems. We use this dataset for open-ended generation, where the LM is expected to provide a title for a paper given the abstract of the paper. The correctness is justified by comparing the generated title to the true title.

Dataset Setup. TriviaQA contains 11,322 data points, Natural Questions contains 3,600 data points, SQuAD-1 contains 10,570 data points, and Meadow contains 1,000 data points. The prompt templates used are similar to those in Kuhn et al. (2023); Lin et al. (2023), and are as follows:

TriviaQA: following from Lin et al. (2023), we use the exact same prompt used in Touvron et al. (2023a): Answer these questions:

In Scotland, a bothy/bothie is a?

A: House

{question}

A:

Natural Question: Similar to Lin et al. (2023), we use an in-context learning prompt with five demonstrations:

where are the fa cup semi finals played. [SEP] A: the new Wembley Stadium.[SEP]

who was alf married to in home and away [SEP] A: Ailsa Stewart.[SEP] what is the name of the first book in the twilight series [SEP] A: Twilight.[SEP]

when is tornado season in the united states [SEP] A: March through June.[SEP]

where did the idea of a messiah come from [SEP] A: Judaism.[SEP] question [SEP] A:

SQuAD-1: Each data point in SQuAD-1 is a (question, context, reference) triplet, where the context is annotated to provide useful information to answer the question. We prompt SQuAD-1 using zero-shot prompting:

Answer the following question based on the context.

{question}

Context: {context}

## A:

Meadow: Each data point in Meadow is a (abstract, title) pair. We prompt Meadow using one-shot prompting:

Abstract: Coronavirus disease 2019 (COVID-19) threatens vulnerable patient populations, resulting in immense pressures at the local, regional, national, and international levels to contain the virus. Laboratory-based studies demonstrate that masks may offer benefits in reducing the spread of droplet-based illnesses, but few data are available to assess mask effects via executive order on a population basis. We assess the effects of a county-wide mask order on per-population mortality, intensive care unit (ICU) utilization, and ventilator utilization in Bexar County, Texas. METHODS: We used publicly reported county-level data to perform a mixed-methods beforeand-after analysis along with other sources of public data for analyses of covariance. We used a least-squares regression analysis to adjust for confounders. A Texas state-level mask order was issued on July 3, 2020, followed by a Bexar County-level order on July 15, 2020. We defined the control period as June 2 to July 2 and the postmask order period as July 8, 2020-August 12, 2020, with a 5-day gap to account for the median incubation period for cases; longer periods of 7 and 10 days were used for hospitalization and ICU admission/death, respectively. Data are reported on a per-100,000 population basis using respective US Census Bureau-reported populations. RESULTS: From June 2, 2020 through August 12, 2020, there were 40,771 reported cases of COVID-19 within Bexar County, with 470 total deaths. The average number of new cases per day within the county was 565.4 (95% confidence interval [CI] 394.6-736.2). The average number of positive hospitalized patients was 754.1 (95% CI 657.2-851.0), in the ICU was 273.1 (95% CI 238.2-308.0), and on a ventilator was 170.5 (95% CI 146.4-194.6). The average deaths per day was 6.5 (95% CI 4.4-8.6). All of the measured outcomes were higher on average in the postmask period as were covariables included in the adjusted model.When adjusting for traffic activity, total statewide caseload, public health complaints, and mean temperature, the daily caseload, hospital bed occupancy, ICU bed occupancy, ventilator occupancy, and daily mortality remained higher in the postmask period. CONCLUSIONS: There was no reduction in per-population daily mortality, hospital bed, ICU bed, or ventilator occupancy of COVID-19-positive patients attributable to the implementation of a mask-wearing mandate. [SEP] Title: Analysis of the Effects of COVID-19 Mask Mandates on Hospital

Resource Consumption and Mortality at the County Level [SEP]   
Abstract:{abstract} [SEP]   
Title:

## E.3 Correctness Functions

Rouge score. Recall-Oriented Understudy for Gist Evaluation (Rouge) score has originally been designed to evaluate machine translation or text summarization tasks. The Rouge score counts the overlapping n-grams between generated reference texts. Widely used n-grams include unigrams (Rouge-1), bigrams (Rouge-2), and the longest common subsequence (Rouge-L). Specifically, it is computed through

$$
\mathrm { R O U G E } = \frac { \left| \mathrm { ( n \mathrm { - g r a m } \in G e n e r a t i o n ) \cap ( n \mathrm { - g r a m } ) \in R e f e r e n c e } \right| } { \left| \mathrm { R e f e r e n c e } \right| } .
$$

METEOR score. The Metric for Evaluation of Translation with Explicit Ordering (METEOR) score has also been originally designed to evaluate machine translation and text summarization. Different from the Rouge score, the METEOR score considers the accuracy and fluency of the generation, as well as word order. The calculation of the METEOR score can be found in Banerjee and Lavie (2005).

BERT-similarity. The BERT-similarity is based on sentence-bert (Reimers and Gurevych, 2019). Specifically, in the first step, reference and generation texts are encoded as 768-dimensional feature vectors, respectively. Then, the correctness values are computed by calculating the cosine similarity between reference and generation vectors. In our implementation, we use sentence-Bert with bert-nlimean-tokens pre-trained weights as the encoding model.

ChatGPT evaluation. ChatGPT evaluation is calculated by prompting GPT-3.5-turbo with the question, reference, and generation; and asking it to evaluate the correctness of the generation. The template used in calculating ChatGPT correctness follows that in Lin et al. (2023):

Rate the level of consistency between the answer to the question and   
the reference answer, from 0 to 100.   
Question: In Scotland a bothy/bothie is a?   
Reference: House   
Answer: House   
Rating: 100.   
Question: Where in England was Dame Judi Dench born?   
Reference: York   
Answer:London   
Rating:0.   
Question: {question}   
Reference:{reference}   
Answer: {generated}   
Rating:

## E.4 Inconsistency due to Correctness Thresholding

We provide more evidence to show the inconsistency of AUARC and AUPRC metrics caused by the ad hoc correctness thresholding. The plots are in Fig 5, 6, 7, 8, and 9.

![](images/21ac7da1dcb0a7388c7e0a9baae6b895f69ba48d035a787d85facf2b4572b55d.jpg)

![](images/94c18e835cdef8fc16085765fb1d935e655a64bda146a811bff54047fa80dc1d.jpg)  
Figure 5: The assessed results for AUARC (left) and AUPRC (right) of uncertainty/confidence measures for GPT-3.5-turbo on the TriviaQA benchmark using the METEOR correctness score with varying thresholds.

![](images/13de3f3cb791a8475a5174888d20d196ae80f94fcfc04ea74132155ceff7fa8d.jpg)  
(a) AUROC

![](images/0876f4d7d9571e36ce57d86472d94e57c9fb06540c63bb7027581da2ca8b521c.jpg)  
(b) AUARC

![](images/0fa06b8cc5a205829617389a40c2d84013329d70b196586202ae173572323524.jpg)  
(c) AUPRC

![](images/b781899531ae2831d62470db759704c1f0b87c253532734b3c3b653cb9e5d72e.jpg)  
(d) Output ranges

Figure 6: Results for Meadow using GPT-3.5-turbo and the Rouge score.  
![](images/8a94ad3a11e67959cad780b1dc3f088b48b23c62bc084061c923ef4e2e03db15.jpg)  
(a) AUROC

![](images/f3fc9e01e2df839f2d7e8922848128e1b4333253239f05c23b9d08a92a03e796.jpg)  
(b) AUARC

![](images/07ee4534b9926923a6bfdd89f8b72ef3e2b5e2fc92babc0a3d800eacb9b0e0bb.jpg)  
(c) AUPRC

![](images/49ec0ae85c7ada90e34c5dbea09ce930a9bde4c1ccb81b521c18e76a28371de6.jpg)  
(d) Output ranges

Figure 7: Results for TriviaQA using GPT-3.5-turbo with temperature 1.5 and the bert-similarity metric.  
![](images/9745f2dbda59749c13c54f5758674f139f224ae7d406e463edc75a0a4517e38f.jpg)  
(a) AUROC

![](images/55fe9e7a9d794e97ce679a9a52290251327974e9948a629884d8b4d93a13ca36.jpg)  
(b) AUARC

![](images/aa02b7ffced771628ba2051c97a8866a1a817f287f68418045d930b53d65aa5b.jpg)  
(c) AUPRC

![](images/84d782a6cf3426d9e8c5f2aa381e970d5b03a4e018969fc7a0fdaaca3be5e14e.jpg)  
(d) Output ranges  
Figure 8: Results for TriviaQA using Llama-2-7b-chat and the Rouge score.

<table><tr><td>Model</td><td>Dataset</td><td>Correctness</td><td>Temperature</td><td> $U _ { \mathrm { E c c } }$ </td><td> $U _ { \mathrm { D e g } }$ </td><td> $U _ { \mathrm { E i g V } }$ </td><td> $U _ { \mathrm { N L L } }$ </td><td> $U _ { \mathrm { S E } }$ </td><td> $C _ { \mathrm { V e r b } }$ </td></tr><tr><td rowspan="11">Llama-2</td><td rowspan="5">nq-open</td><td>bert meteor</td><td>0.6</td><td>0.302±0.044 0.293±0.027</td><td>0.044±0.011 0.072±0.010</td><td> $0 . 0 4 6 { \scriptstyle \pm 0 . 0 0 7 }$  0.077±0.015</td><td>0.121±0.016 0.167±0.021</td><td>0.122±0.025 0.137±0.024</td><td>nan</td></tr><tr><td></td><td>0.6</td><td>0.297±0.039</td><td>0.058±0.010</td><td>0.051±0.010</td><td>0.147±0.021</td><td>0.124±0.019</td><td>nan</td></tr><tr><td>rougeL</td><td>0.6</td><td>0.297±0.038</td><td>0.057±0.011</td><td>0.051±0.010</td><td>0.148±0.021</td><td>0.124±0.020</td><td>nan</td></tr><tr><td>rouge1</td><td>0.6</td><td>0.308±0.041</td><td>0.071±0.013</td><td>0.064±0.013</td><td>0.072±0.008</td><td>0.181±0.027</td><td>nan</td></tr><tr><td>bert meteor</td><td>0.6 0.6</td><td>0.299±0.049</td><td>0.252±0.027</td><td>0.247±0.029</td><td>0.419±0.018</td><td>0.407±0.024</td><td>nan nan</td></tr><tr><td rowspan="5">squad</td><td>rougeL 0.6</td><td></td><td>0.359±0.045</td><td>0.139±0.033</td><td>0.150±0.027</td><td>0.187±0.028</td><td>0.332±0.036</td><td>nan</td></tr><tr><td>rougel</td><td>0.6</td><td>0.360±0.044</td><td>0.141±0.034</td><td>0.150±0.027</td><td>0.195±0.032</td><td>0.337±0.035</td><td>nan</td></tr><tr><td>bert</td><td>0.6</td><td>0.312±0.052</td><td>0.020±0.005</td><td>0.028±0.007</td><td>0.244±0.012</td><td>0.061±0.008</td><td></td></tr><tr><td>meteor</td><td>0.6</td><td>0.305±0.048</td><td>0.041±0.007</td><td>0.049±0.010</td><td>0.271±0.020</td><td>0.052±0.007</td><td>nan</td></tr><tr><td>triviaqa rougeL</td><td>0.6</td><td>0.305±0.050</td><td>0.026±0.005</td><td>0.033±0.006</td><td>0.206±0.020</td><td>0.051 ±0.007</td><td>nan nan</td></tr><tr><td rowspan="8"></td><td rowspan="3"></td><td>rouge1 0.6</td><td></td><td>0.307±0.049</td><td>0.026±0.005</td><td>0.034±0.006</td><td>0.209±0.019</td><td>0.052±0.007</td><td>nan</td></tr><tr><td>0.6</td><td></td><td>0.199±0.040</td><td>0.046±0.008</td><td>0.052±0.010</td><td>0.101±0.015</td><td>0.062±0.010</td><td>nan</td></tr><tr><td>bert 1.0</td><td></td><td>0.236±0.033</td><td>0.035±0.008</td><td>0.038±0.007</td><td>0.097±0.017</td><td>0.055±0.012</td><td>nan</td></tr><tr><td rowspan="12">nq-open</td><td rowspan="2">meteor</td><td>0.6</td><td>0.190±0.039</td><td>0.062±0.008</td><td>0.067±0.010</td><td>0.176±0.018</td><td>0.072±0.009</td><td>nan</td></tr><tr><td>1.0</td><td>0.224±0.034</td><td>0.044±0.006</td><td>0.046±0.007</td><td>0.209±0.023</td><td>0.074±0.015</td><td>nan</td></tr><tr><td rowspan="3">rougeL</td><td>0.6</td><td>0.198±0.039</td><td>0.053±0.011</td><td>0.057±0.010</td><td>0.167±0.013</td><td>0.060±0.012</td><td>nan</td></tr><tr><td>1.0</td><td>0.227±0.035</td><td>0.035±0.007</td><td>0.033±0.006</td><td>0.211±0.021</td><td>0.069±0.016</td><td>nan</td></tr><tr><td>0.6</td><td>0.199±0.039</td><td>0.054±0.010</td><td>0.057±0.010</td><td>0.167±0.014</td><td>0.061±0.013</td><td>nan</td></tr><tr><td rowspan="3">rougel</td><td>1.0</td><td>0.227±0.035</td><td>0.034±0.007</td><td>0.033±0.006</td><td>0.212±0.021</td><td>0.069±0.015</td><td>nan</td></tr><tr><td rowspan="2">bert</td><td>0.6</td><td>0.208±0.033</td><td>0.065±0.014</td><td>0.075±0.017</td><td>0.048±0.007</td><td>0.063±0.012</td><td>nan</td></tr><tr><td>1.0</td><td>0.276±0.039</td><td>0.067±0.011</td><td>0.063±0.010</td><td>0.038±0.006</td><td>0.098±0.012</td><td>nan</td></tr><tr><td rowspan="8">Llama-2-chat squad</td><td rowspan="3">meteor</td><td>0.6</td><td>0.216±0.038</td><td>0.303±0.026</td><td>0.265±0.022</td><td>0.063±0.013</td><td>0.182±0.029</td><td>nan</td></tr><tr><td>1.0</td><td>0.300±0.046</td><td>0.292±0.035</td><td>0.250±0.027</td><td>0.064±0.011</td><td>0.274±0.021</td><td>nan</td></tr><tr><td>0.6</td><td>0.239±0.036</td><td>0.177±0.026</td><td>0.143±0.020</td><td>0.052±0.011</td><td>0.127±0.020</td><td>nan</td></tr><tr><td rowspan="2">rougeL</td><td>1.0</td><td>0.304±0.036</td><td>0.179±0.033</td><td>0.137±0.024</td><td>0.053±0.012</td><td>0.210±0.027</td><td>nan</td></tr><tr><td>0.6</td><td>0.238±0.037</td><td>0.183±0.027</td><td>0.148±0.022</td><td>0.053±0.010</td><td>0.129±0.021</td><td>nan</td></tr><tr><td rowspan="6">triviaqa</td><td rowspan="3">rougel bert</td><td>1.0</td><td>0.303±0.035</td><td>0.185±0.033</td><td>0.143±0.025</td><td>0.053±0.012</td><td>0.213±0.026</td><td>nan</td></tr><tr><td>0.6</td><td>0.140±0.024</td><td>0.062±0.016</td><td>0.061 ±0.015</td><td>0.020±0.004</td><td>0.027±0.007</td><td>nan</td></tr><tr><td>1.0</td><td>0.213±0.030</td><td>0.025±0.006</td><td>0.034±0.006</td><td>0.014±0.002</td><td>0.036±0.006</td><td>nan</td></tr><tr><td rowspan="3">meteor</td><td>0.6</td><td>0.145±0.027</td><td>0.067±0.017</td><td>0.064±0.015</td><td>0.034±0.009</td><td>0.075±0.016</td><td>nan</td></tr><tr><td>1.0</td><td>0.206±0.032</td><td>0.035±0.007</td><td>0.046±0.005</td><td>0.049±0.008</td><td>0.084±0.007</td><td>nan</td></tr><tr><td>0.6 rougeL</td><td>0.141±0.021 0.204±0.035</td><td>0.062±0.014 0.027±0.006</td><td>0.061±0.014</td><td>0.024±0.005</td><td>0.034±0.005</td><td>nan</td></tr><tr><td rowspan="6">bert</td><td rowspan="3">rougel</td><td>1.0</td><td>0.141±0.021</td><td>0.062±0.014</td><td>0.040±0.004</td><td>0.022±0.002</td><td>0.05 1 ±0.007</td><td>nan</td></tr><tr><td>0.6</td><td>0.203±0.035</td><td>0.027±0.006</td><td>0.062±0.013</td><td>0.024±0.005</td><td>0.034±0.006</td><td>nan</td></tr><tr><td>1.0 1.0</td><td>0.284±0.035</td><td>0.178±0.030</td><td>0.040±0.004</td><td>0.022±0.002</td><td>0.051 ±0.007</td><td>nan</td></tr><tr><td rowspan="3">meteor meadow</td><td>1.0</td><td>0.292±0.045</td><td>0.134±0.027</td><td>0.174±0.025 0.137±0.026</td><td>0.112±0.022 0.074±0.012</td><td>0.177±0.027</td><td>0.288±0.033</td></tr><tr><td></td><td></td><td>0.130±0.022</td><td></td><td></td><td>0.132±0.018</td><td>0.263±0.050</td></tr><tr><td>rougeL 1.0</td><td>0.278±0.045</td><td>0.126±0.018</td><td>0.131±0.025 0.135±0.020</td><td>0.056±0.010</td><td>0.113±0.022</td><td>0.289±0.046</td></tr><tr><td rowspan="12">nq-open squad GPT-3.5</td><td rowspan="4"></td><td>rouge1</td><td>1.0</td><td>0.290±0.047</td><td></td><td></td><td>0.059±0.013</td><td>0.113±0.018</td><td>0.299±0.047</td></tr><tr><td>bert</td><td>1.0</td><td>0.151±0.025</td><td>0.050±0.012</td><td>0.065±0.014</td><td>0.039±0.008</td><td>0.050±0.007</td><td>0.487±0.005</td></tr><tr><td>meteor rougeL</td><td>1.0</td><td>0.154±0.027</td><td>0.050±0.011</td><td>0.063±0.011</td><td>0.046±0.011</td><td>0.060±0.009</td><td>0.452±0.018</td></tr><tr><td></td><td>1.0</td><td>0.151±0.022</td><td>0.048±0.011</td><td>0.062±0.012 0.063±0.012</td><td>0.034±0.009 0.034±0.009</td><td>0.052±0.008 0.051±0.008</td><td>0.487±0.006 0.487±0.006</td></tr><tr><td rowspan="7">bert meteor</td><td>rougel</td><td>1.0 1.0</td><td>0.153±0.023 0.204±0.025</td><td>0.048±0.011 0.237±0.024 0.240±0.019</td></table>

Table 4: RCE results for various experimental configurations.

![](images/fa165565ca9f1db53b614ccf0af9ae2b33dface20a3deadbd0ba02f96dc4cc17.jpg)  
(a) AUROC

![](images/0513fab05bc9fbda7511e72b1ce7ab2a2d163d44ca1b03154954a662f9f859a4.jpg)  
(b) AUARC

![](images/4e4c0dda7d6419fae38351b86bf8d4d66e0a483f4ba3839c82580f1d7fc7d960.jpg)  
(c) AUPRC

![](images/44c72e932705eedfb1f6f4f720319f606e7cd019ba53a5359e091689b942a6f4.jpg)  
(d) Output ranges  
Figure 9: Results for TriviaQA using Llama-2-7b-chat using temperature 1.0 and the Rouge score.

## F Additional Experimental Results

<table><tr><td>Prompt</td><td>Reference</td><td>Generation</td><td>P(UEcc ≤ u)</td><td>P(UDeg ≤ u)</td><td>P(UEigV ≤ u)</td><td>P(USE ≤ u)</td><td>P(UNLL ≤ u)</td></tr><tr><td>Q: Who did Dr. Crippen murder?</td><td>his wife</td><td>His wife</td><td>0.999</td><td>0.881</td><td>0.822</td><td>0.649</td><td>0.247</td></tr><tr><td>Q: What are the only two musical notes which have no flats?</td><td>c and f</td><td>B and F</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.898</td><td>0.691</td></tr><tr><td>Q: Which Eastenders actor has played the policeman Nick Rowan on TV?</td><td>nick berry</td><td>Mark Jordon</td><td>0.999</td><td>0.972</td><td>0.978</td><td>0.954</td><td>0.918</td></tr><tr><td>Q: Which &#x27;B&#x27; was the name of the mechanical shark used in the original &#x27;Jaws</td><td>bruce</td><td>Bruce</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.337</td><td>0.183</td></tr><tr><td>film? Q: Which actor does the interviewing in &#x27;Interview with a Vampire&#x27;?</td><td>christian slater</td><td>Brad Pitt</td><td>0.999</td><td>0.858</td><td>0.856</td><td>0.861</td><td>0.893</td></tr><tr><td>Q: What did my true love bring to me on the Sixth Day of Christmas?</td><td>six geese-a-laying</td><td>Six geese a-laying</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.736</td><td>0.688</td></tr><tr><td>Q: In January 1957, Russell Endean became the first batsman to be dismissed</td><td>handling the ball</td><td>Handling the ball</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.901</td><td>0.368</td></tr><tr><td>from a test cricket match for doing what? Q: What are the first names of the two dancing instructors in the UK television</td><td>barry and yvonne</td><td>Barry and Yvonne</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.846</td><td>0.627</td></tr><tr><td>series &#x27;Hi De Hi&#x27;? Q: Who became the host of the UK television game show Blankety Blank in</td><td>les dawson</td><td>Les Dawson</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.180</td><td>0.040</td></tr><tr><td>1984? Q: How much, in pounds sterling, does the Best in Show Winner receive at the</td><td>100 pounds</td><td>£100</td><td>0.999</td><td>0.920</td><td>0.908</td><td>0.830</td><td>0.787</td></tr><tr><td>annual Crufts Dog Show?</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Q: In the Billy Bunter stories, what is the surname of Bunter&#x27;s form teacher? Q: Which play is featured in the film The Producers?</td><td>quelch springtime for hitler</td><td>Quelch</td><td>0.999 0.999</td><td>0.761 0.761</td><td>0.769</td><td>0.999</td><td>0.558</td></tr><tr><td>Q: What provoked the war between Honduras and El Salvador in 1969?</td><td>a football match</td><td>Springtime for Hitler</td><td></td><td>0.761</td><td>0.769 0.769</td><td>0.967</td><td>0.341</td></tr><tr><td>Q: Which character was played by Linda Thorson in The Avengers?</td><td></td><td>A soccer match</td><td>0.999</td><td></td><td></td><td>0.535</td><td>0.711</td></tr><tr><td></td><td>tara king</td><td>Tara King</td><td>0.999</td><td>0.824</td><td>0.885</td><td>0.919</td><td>0.399</td></tr><tr><td>Q: According to a traditional English proverb, what is better than none?</td><td>half a loaf</td><td>A bad excuse</td><td>0.999</td><td>0.972</td><td>0.978</td><td>0.931</td><td>0.908</td></tr><tr><td>Q: In which Welsh village is there only one gay, apparently?!</td><td>llandewi breffi</td><td>Llanddewi Brefi</td><td>0.999</td><td>0.926</td><td>0.963</td><td>0.950</td><td>0.906</td></tr><tr><td>Q: On September 28th, NASA announced that what had been detected on Mars?</td><td>flowing water</td><td>Possible signs of life</td><td>0.999</td><td>0.965</td><td>0.963</td><td>0.813</td><td>0.930</td></tr><tr><td>Q: What are the first four words of the Bible, as recorded in Genesis?</td><td>in the beginning god</td><td>In the beginning, God</td><td>0.653</td><td>0.650 0.858</td><td>0.651 0.837</td><td>0.574</td><td>0.557</td></tr><tr><td>Q: Which national anthem was originally called the &#x27;War Song for the Rhine Army&#x27;?</td><td>marsellaise</td><td>German national anthem</td><td>0.694</td><td></td><td></td><td>0.785</td><td>0.888</td></tr><tr><td>Q: Name the UK budget holiday company specialising in Turkey and Greece which went bust in July 2010?</td><td>goldtrail</td><td>Goldtrail</td><td>0.999</td><td>0.920</td><td>0.902</td><td>0.894</td><td>0.655</td></tr><tr><td>Q: Who has been President of France twice, but never been elected to the</td><td>alain poher</td><td>François Mitterrand</td><td>0.999</td><td>0.920</td><td>0.902</td><td>0.854</td><td>0.864</td></tr><tr><td>position? Q: What is the name of Madonna&#x27;s proposed chain of fitness clubs?</td><td>hard candy fitness</td><td>Hard Candy Fitness</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.996</td><td></td></tr><tr><td>Q: Elvis Presley sang a few lines in German on which US hit song?</td><td>wooden heart</td><td>Wooden Heart</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.998</td><td>0.183 0.270</td></tr><tr><td>Q: What was the name of the book that was a collection of Aubrey Beardsley&#x27;s work, published by Leonard Smithers in 1897?</td><td>a book of fifty drawings</td><td>The Yellow Book</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.950</td><td>0.775</td></tr><tr><td>Q: Dishes prepared with spinach can be referred to as what?</td><td>la florentine</td><td>Spinach dishes</td><td>0.999</td><td>0.920</td><td>0.902</td><td></td><td></td></tr><tr><td>Q: Which English civil engineer&#x27;s most famous project was the construction of</td><td>sir john wolfe-barry</td><td>Sir John Wolfe Barry</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.943 0.830</td><td>0.899 0.633</td></tr><tr><td>Tower Bridge over the River Thames in London?</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Q: Where did the space probe New Horizons launched by NASA in 2006 aim to investigate?</td><td>pluto and the kuiper belt</td><td>Pluto and the Kuiper Belt</td><td>0.999</td><td>0.905</td><td>0.904</td><td>0.905</td><td>0.576</td></tr><tr><td>Q: Where woud you find a nave or an apse?</td><td>in a church</td><td>In a church</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.236</td><td>0.185</td></tr><tr><td>Q: What is the name of Jay-Z and Beyonce&#x27;s daughter? Q: &#x27;Feel Like Making Love&#x27; and &#x27;The First Time Ever I Saw Your Face&#x27; were</td><td>blue ivy roberta flack</td><td>Blue Ivy Roberta Flack</td><td>0.999 0.999</td><td>0.976 0.761</td><td>0.965 0.769</td><td>0.975</td><td>0.354</td></tr><tr><td>hit singles for which female artist?</td><td></td><td></td><td></td><td></td><td></td><td>0.864</td><td>0.046</td></tr><tr><td>Q: In the nursery rhyme, who pulled pussy out of the well?</td><td>little tommy stout</td><td>Tommy</td><td>0.999</td><td>0.976</td><td>0.987</td><td>0.962</td><td>0.882</td></tr><tr><td>Q: &quot;In the film of the same name, what was the name of &quot;&quot;The Hustler&quot;&quot;?&quot; Q: In Camberwick Green on Children&#x27;s TV who was the commander of Pippin</td><td>&quot;&quot;&quot;fast eddie&quot;&quot; felson&quot;</td><td>Fast Eddie Felson</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.708</td><td>0.692</td></tr><tr><td>Fort?</td><td>captain snort</td><td>Captain Snort</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.961</td><td>0.156</td></tr><tr><td>Q: In Chigley on Children&#x27;s TV who owned the steam railway and drove the</td><td>lord belborough</td><td>Lord Belborough</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.951</td><td>0.401</td></tr><tr><td>steam engine &#x27;Bessie&#x27;? Q: Who won the gold medal in the women&#x27;s Skeleton Bob at the 2010 Vancouver</td><td>amy williams</td><td>Amy Williams</td><td>0.999</td><td>0.881</td><td>0.822</td><td>0.676</td><td>0.265</td></tr><tr><td>Winter Olympics? Q: What decoration, a Cross, was first awarded in 1995 to Corporal Wayne</td><td>conspicuous gallantry</td><td>George Cross</td><td>0.999</td><td>0.844</td><td>0.783</td><td>0.801</td><td></td></tr><tr><td>Mills for his actions in Bosnia? Q: What was the French sounding winner of the 2011 Epsom Derby?</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.899</td></tr><tr><td>Q: Who originally provided the voice for TV&#x27;s &#x27;Basil Brush&#x27;?</td><td>pour moi ivan owen</td><td>Pour Moi Ivan Owen</td><td>0.999 0.999</td><td>0.761 0.761</td><td>0.769</td><td>0.321</td><td>0.101</td></tr><tr><td>Q: &quot;Which actress played &#x27;Valeria&quot;&quot; in the film Carry On Screaming?&quot;</td><td>fenella fielding</td><td>Fenella Fielding</td><td>0.999</td><td>0.761</td><td>0.769 0.769</td><td>0.987 0.862</td><td>0.454</td></tr><tr><td>Q: Which of the &#x27;Spice Girls&#x27; advertised &#x27;Milky Way&#x27; ob t.v.?</td><td>emma bunton (baby spice)</td><td>Victoria Beckham (Posh Spice)</td><td>0.999</td><td>0.949</td><td>0.963</td><td>0.985</td><td>0.206</td></tr><tr><td>Q: Give any year in the life of the Portuguese prince known as Henry the</td><td>1394-1460</td><td>1394-1460</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.680</td><td>0.847</td></tr><tr><td>Navigator.</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.671</td></tr><tr><td>Q: On which horse did Sir Gordon Richards ride his only Epsom Derby winner?</td><td>pinza</td><td>Pinza</td><td>0.999</td><td>0.824</td><td>0.885</td><td>0.987</td><td>0.229</td></tr><tr><td>Q: What was the name of the aeroplane in which Wiley Post became the first</td><td>&#x27;winnie mae</td><td>Winnie Mae</td><td>0.999</td><td>0.761</td><td>0.769</td><td>0.849</td><td>0.654</td></tr><tr><td>pilot to fly solo around the world?</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Q: Who was the husband of Rebekah Brooks from 2002 to 2009? Q: Whole Again and Eternal Flame were Number Ones for which girl group in</td><td>ross kemp atomic kitten</td><td>Ross Kemp Atomic Kitten</td><td>0.999 0.999</td><td>0.761 0.761</td><td>0.769 0.769</td><td>0.826 0.180</td><td>0.746 0.026</td></tr><tr><td>2001? Q: During a penalty shoot out in soccer where should the non participating</td><td>in the centre circle</td><td>Outside of the penalty area</td><td>0.999</td><td>0.985</td><td>0.987</td><td>0.987</td><td>0.960</td></tr><tr></table>

Table 5: Examples of correctness and the according uncertainty levels.

## F.1 Qualitative Illustration

x: In 1840 the world's first postage stamps printed were the   
Penny Black and which other?   
y: twopenny blue   
y: The Penny Red   
P(UsE ≤ u): 0.825   
P(UNLL ≤ u): 0.864   
x: Championship dragon boat racing calls for a specialised long   
boat, a team of paddlers (typically 20), a sweeper to steer and   
which other of these?   
y: a drummer and drum   
y: A drummer   
P(UsE ≤u): 0.946   
P(UNLL ≤u)： 0.704   
x: Who has the highest suicide rate in the UK?   
y: men - by a ratio of roughly 4 to 1   
Î: Middle-aged men   
P(UsE ≤ u): 0.745   
P(UNLL ≤ u): 0.894   
x: Which East Midlands club holds the Football League record   
for most games played?   
y: nots county   
y: Notts County   
P(UsE ≤ u)： 0.842   
P(UNLL ≤u)： 0.793  
We provide more instances to show the qualitative effect of our RCE-based assessment in Table 5.

## F.2 Recalibration with Histogram Binning

We use equal-mass histogram binning to recalibrate, in a post-hoc manner, the performance of an uncertainty (or confidence) measure on a specific benchmark. Specifically, given a dataset $\{ ( u _ { i } , a _ { i } ) \} _ { i = 1 } ^ { n }$ of uncertainty and correctness values computed over a benchmark, where each $u _ { i } = U ( \pmb { x } ; \pmb { \hat { y } } _ { i } ) , a _ { i } = A ( \pmb { x } _ { i } ; \pmb { \hat { y } } _ { i } )$ and $\widehat { \mathbf { y } } _ { i }$ is a response generated by the LM. Then, we first randomly split it into the calibration set $\{ ( u _ { i } , a _ { i } ) \} _ { i = 1 } ^ { n _ { \mathrm { c a l } } }$ and the test set $\{ ( u _ { i } , a _ { i } ) \} _ { i = n _ { \mathrm { c a l } } + 1 } ^ { n }$ . Similar to the operations in Sec. 4.3, we partition the range of U into B bins $\{ \mathrm { b i n } _ { b } \} _ { b = 1 } ^ { B }$ whose boundaries are quantiles of $\{ ( u _ { i } , a _ { i } ) \} _ { i = n _ { \mathrm { c a l } } + 1 } ^ { n }$ . Then, we estimate the expected correctness level over the bin as

$$
\mathrm { c r c } _ { b , \mathrm { c a l } } : = \frac { 1 } { | \mathcal { T } _ { b , \mathrm { c a l } } | } \sum _ { i \in \mathcal { T } _ { b , \mathrm { c a l } } } a _ { i }
$$

where ${ \mathcal { T } } _ { b , { \mathrm { c a l } } } \ \triangleq \ \{ i \ : \ 1 \leq \ i \ \leq \ n _ { \mathrm { c a l } } , u _ { i } \ \in \ \mathrm { b i n } _ { b } \}$ . We re-calibrate the measure U, defining $U _ { \mathrm { c a l } }$ via $U _ { \mathrm { c a l } } ( \pmb { x } ; \pmb { \widehat { y } } ) = \mathrm { c r c } _ { b , \mathrm { c a l } }$ for any $U ( { \pmb x } ; \widehat { \pmb y } ) \in$ bin. We split the benchmark data equally into calibration and test sets and evaluate the performance of the calibrated measure on the test set. Table 3 and Fig. 10 and 11 list the RCE results of $U _ { \mathrm { S E } }$ for GPT-3.5-turbo before and after calibration. We observe the calibrated measure is significantly better rank-calibrated, showing the effectiveness of this strategy.

While effective, one should note that such a post-hoc recalibration strategy concerns a specific benchmark and is not a focus of our work. We leave devising benchmark-agnostic calibrated uncertainty/confidence measures for future work.

![](images/78bc8521b506503068fe57d6f6734ad116ecc5f7056a6b158429578ddfd77958.jpg)

![](images/c781af52ef431bf358d027500d4bbee71029a4843f66d76f700009942b761cfe.jpg)  
(a) Meadow

![](images/4ed91fbb94847252255b2f8954f3adfd8ad2ccc776c84844a80bb8e0cd742c98.jpg)

![](images/ce41038cb9afabbbb007344a69bbc39469387bd611eb28bfdfc1720d2b158e18.jpg)  
(b) NQ-Open

![](images/d053805cbf6cd8f4d1c57d7aeab427379459be62a1ddc5c6ee9d2dcd865fae18.jpg)

![](images/2ed7f09e502df39c030ba1a1c4ffcca109b5a1b0b9953b4b5b4c16f260c1d946.jpg)  
(c) Squad

![](images/13f1f6e96264bb66828dfd8bfaa870bf87f0285cff984cee4a9af943e4ce00f9.jpg)

![](images/b49c41edd29e152b78724722af5daa13fca8246bd9efe0a29d189e5820f255a0.jpg)  
(d) TrviaQA  
Figure 10: Indication diagrams of $U _ { \mathrm { S E } }$ and $U _ { \mathrm { S E , c a l } }$ (post-calibrated) for GPT-3.5-turbo (temperature 1.0) on various benchmarks with the Meteor correctness.

## F.3 Critical Difference Diagrams

Here, we propose to combine the RCE metric with the critical difference (CD) diagram (Demšar, 2006). Critical Difference diagrams are built on the Wilcoxon signed rank test and the Friedman test, giving a non-parametric comparison of multiple approaches aggregated over several trials.

![](images/a2e297888f54e12500228f610fb60b18ea57c000d2b83b2d01565ba1905612f3.jpg)  
Figure 12: CD diagram of Llama-2-chat on TriviaQA.

As a demonstration, the CD diagram of assessed measures for Llama-2-chat on TriviaQA is shown in Fig. 12. The positions of various methods represent their averaged ranks over various experimental configurations (e.g., temperature, LM, bootstrap, etc), where a lower averaged rank indicates that the corresponding measure (e.g., 1.04 for $U _ { \mathrm { N L L } } )$ performs better than others in an averaged sense. Here, a thick horizontal segment connects measures $( e . g . , U _ { \mathrm { D e g } }$ and $U _ { \mathrm { E i g V } } )$ if the difference between their averaged ranks is within the critical length determined by related hypothesis testing procedures. Measures that are disconnected $( e . g . , U _ { \mathrm { E c c } } , U _ { \mathrm { D e g } }$ , and $U _ { \mathrm { { N L L } } } )$ have statistically significant differences in performance.

## F.4 Robustness Analysis

The RCE of uncertainty measures in practice may be affected by several factors. Therefore, we conduct ablation studies to analyze whether RCE is robust to two crucial key factors: correctness scores and model temperatures.

Correctness functions. We show RCEs for various models and correctness scores on TriviaQA and SQuAD in Fig 13. Each result is obtained using bootstrapping with 20 fixed seeds. We observe that the ranking of uncertainty measures is robust to correctness scores. For instance, we show the critical diagrams using GPT-3.5 on TriviaQA with varying correctness scores in Fig 14. In this setting, $U _ { \mathrm { N L L } }$

![](images/90b082b9c07c13944b33018a039b5975348c5d7e5fc61f58a9e86ae1f65fb41b.jpg)

![](images/f815d9e2395c83ad41d166f6fd005366a79a902ed682b48059368a63d42960e8.jpg)  
(a) Bert Similarity

![](images/4e46bb7a237ac4cc160c62942a1ff70f93221e7b16611ace16c714c37c1f26d3.jpg)

![](images/e2fbe3042b7a5b9674d70446a157fc670d9f98fdfc12ab2239ec8ba91f00cfab.jpg)  
(b) Meteor Score

![](images/f65b0a902846f43a409269820a92bc8f63ecd3a3d3aeadfb5cd261075bd5c7d4.jpg)

(c) Rouge Score  
![](images/e00ddd388546972e2a72c31840e867995617dae47b7663b682b102f42d21ad76.jpg)

![](images/cbbc3d91590d5b58f40a4836bcb5d2047510406eb8efd892555efc6813630f8f.jpg)

![](images/61ed1a5828822c94baa9c2cbab74176c5aa87de4f410777ae685cf95a8a471b6.jpg)  
(d) Rouge1 Score  
Figure 11: Indication diagrams of $U _ { \mathrm { S E } }$ and $U _ { \mathrm { S E , c a l } }$ (post-calibrated) for GPT-3.5-turbo (temperature 1.5) on TriviaQA with various correctness scores.

$U _ { \mathrm { S E } }$ and $C _ { \mathrm { V e r b } }$ rank consistently higher across different correctness scores. Second, as shown in Table 4, RCE values using different correctness scores are relatively stable. For instance, when using GPT-3.5 on TriviaQA, the RCE values of NLL are 0.065, 0.054, 0.037, and 0.039 with bert\_similarity, meteor, rouge-L, and rouge-1 scores, which are close.

Temperature setting. We show the RCEs for various models and temperatures on TriviaQA and SQuAD in Fig. 15. As above, each result is obtained using bootstrapping with 20 fixed seeds. The findings are similar to those regarding correctness scores. First, as shown in Fig. 16, while RCE values are not constant, $U _ { \mathrm { N L L } }$ ranks consistently highest across different temperatures. When only the best uncertainty measure is considered, the RCE rankings at different temperatures give consistent results. Second, the RCE values are stable across different temperatures. For instance, when using GPT-3.5 with the Rouge-L score, the RCE values are 0.041, 0.038, 0.034 with temperatures 0.5, 1.0, and 1.5.

Influence of sample size. We show that the empirical RCE is robust regarding the influence of sample size, which is crucial in scenarios where labeled data is hard to acquire. To this end, we conducted a new experiment using less data in the RCE computation, simulating scenarios where only a small amount of labeled data can is available. Specifically, we utilize 20%, 40%, 80%, 100% of the TriviaQA dataset in computing the empirical RCE values of uncertainty/confidence measures for the GPT-3.5 model with temperature 1.0. The RCE results under the Bert-similarity and RougeL correctness are in Table 6. The binning scheme is the same as the one used in the paper (i.e., 20 equal-mass bins). From the new experimental results, we observe that the RCE results are fairly stable, up to reasonable standard deviations (denoted by the subscript numbers), for moderately large datasets.

## F.5 Conclusive Comparison

While the RCE values and rankings are often stable when correctness score and temperature vary, there are exceptional situations where uncertainty measures rankings might fluctuate. This poses a challenge when aiming for conclusive comparisons for uncertainty measures across varying hyperparameter situations. To make conclusive comparisons aiming to identify a best method, we can use CD diagrams by taking multiple hyperparameter choices into account. For example, to draw conclusions agnostic to model temperature, we plot CD diagrams that show RCE rankings averaged from data collected at different temperatures, as shown in Fig. 17. Based on these results, comparisons agnostic to the temperature can be made: $U _ { \mathrm { N L I } }$ overall outperforms other methods with GPT-3.5 and Llama-2-chat on TriviaQA; $U _ { \mathrm { E i g V } }$ and $U _ { \mathrm { D e g } }$ overall show statistically similar performance with Llama-2-chat on TriviaQA.

![](images/cb45cd5e6c1cf7ae74b2b056ad2396a8bd92d01f82fae0948e5af4dd6bdd4d98.jpg)

![](images/35f57be684b2890ccd9815234019aa5bfbec9df226c68419f3ca320fe06062e8.jpg)

![](images/a1350ab547c7f72c0a5bee89f0f95524b43e48dc52b3aa76292d5a11569768c9.jpg)

![](images/7ead636394d84a0a1213b39246ee13b6a1e6aa0cd7f69ec386a8e7a9ddec49aa.jpg)

Figure 13: Box plots with various correctness functions under various configurations. The first row is for GPT-3.5- turbo on TriviaQA; the second row is for GPT-3.5-turbo on SQuAD; the third is for Llama-2-7b-chat on TriviaQA; and the fourth row is for Llama-2-7b-chat on SQuAD.
<table><tr><td>Proportion</td><td>Correctness</td><td> $U _ { \mathrm { E c c } }$ </td><td> $U _ { \mathrm { D e g } }$ </td><td> $U _ { \mathrm { E i g V } }$ </td><td> $U _ { \mathrm { N L L } }$ </td><td> $U _ { \mathrm { S E } }$ </td><td> $C _ { \mathrm { { V e r b } } }$ </td></tr><tr><td rowspan="4">bert</td><td>20%</td><td> $0 . 1 7 6 _ { \pm 0 . 0 2 2 }$ </td><td> $0 . 1 5 3 _ { \pm 0 . 0 2 3 }$ </td><td> $0 . 1 5 2 _ { \pm 0 . 0 2 4 }$ </td><td> $0 . 0 5 8 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 0 8 0 _ { \pm 0 . 0 1 5 }$ </td><td> $0 . 2 5 4 _ { \pm 0 . 0 4 2 }$ </td></tr><tr><td>40%</td><td> $\overline { { 0 . 1 7 1 _ { \pm 0 . 0 2 0 } } }$ </td><td> $\overline { { 0 . 1 5 1 _ { \pm 0 . 0 2 1 } } }$ </td><td> $\overline { { 0 . 1 5 4 _ { \pm 0 . 0 2 0 } } }$ </td><td> $0 . 0 4 8 _ { \pm 0 . 0 1 0 }$ </td><td> $\overline { { 0 . 0 8 3 _ { \pm 0 . 0 1 3 } } }$ </td><td> $\overline { { 0 . 2 1 1 _ { \pm 0 . 0 4 5 } } }$ </td></tr><tr><td>80%</td><td> $\overline { { 0 . 1 6 2 _ { \pm 0 . 0 2 2 } } }$ </td><td> $\overline { { 0 . 1 5 3 _ { \pm 0 . 0 1 6 } } }$ </td><td> $\overline { { 0 . 1 5 1 _ { \pm 0 . 0 1 7 } } }$ </td><td> $\overline { { 0 . 0 4 3 _ { \pm 0 . 0 1 0 } } }$ </td><td> $\overline { { 0 . 0 6 2 _ { \pm 0 . 0 1 2 } } }$ </td><td> $\overline { { 0 . 2 0 3 _ { \pm 0 . 0 3 1 } } }$ </td></tr><tr><td>100%</td><td> $\overline { { 0 . 1 5 2 _ { \pm 0 . 0 2 5 } } }$ </td><td> $\overline { { 0 . 1 2 9 _ { \pm 0 . 0 2 0 } } }$ </td><td> $\overline { { 0 . 1 3 3 _ { \pm 0 . 0 2 0 } } }$ </td><td> $\overline { { 0 . 0 3 9 _ { \pm 0 . 0 0 7 } } }$ </td><td> $\overline { { 0 . 0 5 2 _ { \pm 0 . 0 1 2 } } }$ </td><td> $\overline { { 0 . 1 8 2 _ { \pm 0 . 0 2 5 } } }$ </td></tr><tr><td rowspan="4">rougeL</td><td>20%</td><td> $\overline { { 0 . 1 7 8 _ { \pm 0 . 0 2 0 } } }$ </td><td> $\underline { { 0 . 1 5 3 _ { \pm 0 . 0 2 4 } } }$ </td><td> $\overline { { 0 . 1 5 3 _ { \pm 0 . 0 2 3 } } }$ </td><td> $\overline { { 0 . 0 6 1 _ { \pm 0 . 0 1 0 } } }$ </td><td> $0 . 0 9 8 _ { \pm 0 . 0 1 6 }$ </td><td> $\overline { { 0 . 2 3 8 _ { \pm 0 . 0 3 5 } } }$ </td></tr><tr><td>40%</td><td> $\overline { { 0 . 1 7 2 _ { \pm 0 . 0 2 2 } } }$ </td><td> $\overline { { 0 . 1 5 3 _ { \pm 0 . 0 2 1 } } }$ </td><td> $\overline { { 0 . 1 5 6 { \scriptstyle \pm 0 . 0 1 7 } } }$ </td><td> $\overline { { 0 . 0 4 8 _ { \pm 0 . 0 0 9 } } }$ </td><td> $\overline { { 0 . 0 9 0 { \scriptstyle \pm 0 . 0 1 0 } } }$ </td><td> $\overline { { 0 . 1 9 4 _ { \pm 0 . 0 4 0 } } }$ </td></tr><tr><td>80%</td><td> $\overline { { 0 . 1 5 6 _ { \pm 0 . 0 2 0 } } }$ </td><td> $\overline { { 0 . 1 4 5 _ { \pm 0 . 0 1 7 } } }$ </td><td> $\overline { { 0 . 1 4 6 _ { \pm 0 . 0 1 7 } } }$ </td><td> $\overline { { 0 . 0 4 2 _ { \pm 0 . 0 0 9 } } }$ </td><td> $\overline { { 0 . 0 7 3 _ { \pm 0 . 0 1 3 } } }$ </td><td> $\overline { { 0 . 1 9 0 _ { \pm 0 . 0 3 0 } } }$ </td></tr><tr><td>100%</td><td> $\overline { { 0 . 1 5 1 _ { \pm 0 . 0 2 4 } } }$ </td><td> $\overline { { 0 . 1 2 6 _ { \pm 0 . 0 1 9 } } }$ </td><td> $\overline { { 0 . 1 2 9 _ { \pm 0 . 0 1 9 } } }$ </td><td> $\overline { { 0 . 0 3 8 _ { \pm 0 . 0 0 7 } } }$ </td><td> $0 . 0 5 9 _ { \pm 0 . 0 0 9 }$ </td><td> $\overline { { 0 . 1 8 1 _ { \pm 0 . 0 2 6 } } }$ </td></tr></table>

Table 6: RCE results for GPT-3.5-turbo (temperature 1.0) performing on the TriviaQA data with various dataset sizes under the Bert-similarity and RougeL correctness.

![](images/1e058bb3ce0d44c47aca92d1705ba70881f3ba0bf603b89ec1b38e8ec097c9e6.jpg)  
Figure 14: CD diagrams using GPT-3.5 on TriviaQA with different correctness scores.

![](images/26141f596d408512dc21a9c1cef3a9db8c7a79388ddccdede71adea5c5239bf5.jpg)

![](images/a3eddbe5378372f8c5e7ea772670c3fe39b1b875394da4cf65194964cd2da83c.jpg)  
Figure 15: Box plots based on the generations of GPT-3.5-turbo and Llama-2-7b-chat with varying temperatures. The first row represents GPT-3.5-turbo with temperatures 0.5, 1.0, and 1.5, while the second row represents Llama-2-7b-chat with temperatures 0.6 and 1.0. Both results are evaluated on TriviaQA dataset.

![](images/c60e0efa3571285f1b18fb639475a9315137a4c40ee29c90fc1051b727158bca.jpg)  
Figure 16: CD diagrams on using GPT-3.5 TriviaQA with temperature 0.5, 1.0, and 1.5.

![](images/2ec8a92c567d259ad612edc04b83ec96643a80c75eefd19578a4c338e1f81679.jpg)  
Figure 17: Conclusive comparison via critical difference diagrams. The first plot is with GPT-3.5-turbo on TriviaQA with temperatures 0.5, 1.0, and 1.5; the second is with Llama-2-chat on TriviaQA with temperatures 0.6 and 1.0.

## F.6 Library Information

The details of the main libraries used in our experiments are as in Table 7.
<table><tr><td>Package</td><td>Version</td><td>Package</td><td>Version</td></tr><tr><td>transformer (Wolf et al., 2020)</td><td>4.32.1</td><td>nltk (Bird et al., 2009)</td><td>3.8.1</td></tr><tr><td>spacy (Honnibal and Montani, 2017) rouge-score (Lin, 2004)</td><td>3.6.1 0.1.2</td><td>torch (Paszke et al., 2019)</td><td>2.0.1</td></tr></table>

Table 7: Information on main libraries used.

## F.7 Artifact License and Terms

We use four datasets, namely, Natural Questions, TriviaQA, SQuAD-1, and Meadow. Natural Questions is under the CC BY-SA 3.0 license, TriviaQA and Meadow are under the Apache License 2.0, and

SQuAD-1 is under the CC BY-SA 4.0 license. We used two LLMs, namely ChatGPT-3.5 and Llama-2. ChatGPT-3.5-turbo usage is subject to OpenAI's Sharing & Publication Policy and Usage Policies. Llama-2 is under the Llama-2 Community License (Meta, 2023). Our implementation and the data collected are under the MIT License.

Our use of the existing artifacts is consistent with their original intended use. Our created artifacts intend to verify our proposed method in our submission, which is consistent with the original access conditions.

## G AI Assistant Usage

We used Copilot to assist with coding.