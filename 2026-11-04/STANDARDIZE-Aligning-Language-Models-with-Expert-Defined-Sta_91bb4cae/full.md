# STANDARDIZE: Aligning Language Models with Expert-Defined Standards for Content Generation

Joseph Marvin Imperial<sup>Ω,Λ</sup> Gail Forey<sup>Λ</sup> Harish Tayyar Madabushi<sup>Λ</sup>

<sup>Λ</sup>University of Bath, UK

<sup>Ω</sup>National University, Philippines

jmri20@bath.ac.uk gf370@bath.ac.uk htm43@bath.ac.uk

## Abstract

Domain experts across engineering, healthcare, and education follow strict standards for producing quality content such as technical manuals, medication instructions, and children’s reading materials. However, current works in controllable text generation have yet to explore using these standards as references for control. Towards this end, we introduce STANDARD-IZE, a retrieval-style in-context learning-based framework to guide large language models to align with expert-defined standards. Focusing on English language standards in the education domain as a use case, we consider the Common European Framework of Reference for Languages (CEFR) and Common Core Standards (CCS) for the task of open-ended content generation. Our findings show that models can gain 45% to 100% increase in precise accuracy<sub>Given</sub> <sub>this</sub> <sub>prompt</sub> across open and commercial LLMs evaluated,ahead, a solitary fi demonstrating that the use of knowledge artifacts extracted from standards and integrating them in the generation process can effec-readable for B1 lea tively guide models to produce better standard-observes the follow aligned content.<sup>1</sup>

## 1 Introduction

One of the most realized benefits of large language<sub>3.</sub> <sub>Grammatical</sub> <sub>Com</sub> model (LLM) research is how it became widelycontain future forms, fu adopted by the public. In particular, the rise of chatstyle model interfaces, such as ChatGPT and Perplexity, has allowed non-technical users to fully utilize these tools in accomplishing day-to-day tasks and activities, such as getting help with writing, documenting code, and providing recommendations. A key technological advancement behind this is the use of reward-based methods such as Reinforcement Learning for Human Feedback (RLHF, Ouyang et al. (2022)), which embeds human preferences to generative models for better-aligned outputs with respect to the task at hand.

![](images/705909e222b6a5a1afef9728d9f66ada0d275ee505bd532be99fda795e61ee09.jpg)  
Figure 1: In contrast to the simple prompting method used by teachers, the proposed STANDARDIZE framework aims to improve the performance of generative models for content generation by using the fine-grained information found in expert-defined standards. The <sub>Exemplars</sub> framework involves a three-part process starting with the (i) extraction of target specifications from the prompt, <sup>dark</sup> <sup>old</sup> <sup>forest</sup> <sup>up</sup> <sup>Given</sup> <sup>this</sup> <sup>prompt:</sup> <sup>In</sup> <sup>the</sup> <sup>dark</sup> (ii) lookup and retrieval of information that matches the target specifications from the specified standard, and (iii) knowledge augmentation to produce artifacts that ake sure they are Continue the story and make srepresent the standard itself for integration into the generation process with generative models.

be long but not Dream by Shakespeare.Despite the growing literature of complex <sup>ostly</sup> <sup>chronological</sup>algorithms and architectures for enriching the ty: The text mayinstruction-following capabilities of LLMs, the <sup>n</sup> <sup>the</sup> <sup>past,</sup> <sup>repeated</sup>missing puzzle piece that seems to have not garnered equal attention from the community is the integration of actual standards or guidelines crafted by domain experts as a reference of control. For example, in education and language assessment, standards such as the Common European Framework of Reference for Languages (CEFR) serve as an accredited guide for administrators in charge of the creation of educational curriculum content. This standard provides fine-grained specifications of text complexity that different levels of learners can understand depending on their language proficiency (North, 2007, 2014). To be able to automatically generate text content (e.g., narratives or short stories) using an LLM that is acceptable by CEFR standards and captures a student’s topic interest at the same time can serve as a powerful tool in classroom engagement for educators in the long run. Thus, this research gap is an opportunity where the complex instruction-following capabilities of language models can provide assistance, particularly for tasks requiring the generation of text content since this is one of the areas where these models objectively perform well (Chung et al., 2022; Wei et al., 2021; Gatt and Krahmer, 2018).

Towards this end, we tackle the main research question: How can we align large language models for content generation tasks using expertdefined standards? We list our major contributions from this study as follows:

1. We introduce STANDARD-CTG, a new task formalizing the challenge of generating text using generative language models with expertdefined standards as a for controllability.

2. We propose STANDARDIZE, a new retrievalstyle in-context learning framework that extracts knowledge artifacts from standards such as aspect information, exemplars, and manually crafted linguistic variables to improve the performances of generative language models for content generation.

3. We introduce significantly improved performances for GPT-4 and Llama for the task of STANDARD-CTG using two of the most widely recognized academic standards, CEFR and CCS, across diverse evaluation procedures.

## 2 Expert-Defined Standards

## 2.1 Background

According to the International Organization for Standardization (ISO)<sup>2</sup>, standards are documented guidelines often containing rich detail in describing requirements, specifications, and criteria. These guidelines are defined and continuously improved by experts in various domains, such as education, healthcare, and accounting, to name a few. Using standards ensures an institution’s products and processes are consistent and reproducible (Sadler, 2017).

In the context of education and language assessment, standards are usually in the form of either (a)

content standards such as documentations of a common language for ease of communication, writing, and content production, and (b) performance standards such as state-administered tests for reading and mathematical problem-solving competencies. This study focuses on content-based standards used in education and language assessment to be integrated into a generative model’s text generation process. The alignment with existing standards for any generated text material is crucial to ensure quality and consistency before being used in classroom settings (La Marca et al., 2000).

## 2.2 Standards in Education and Language Assessment

We discuss the two selected English standards we consider as test cases for this study.

The Common European Framework of Reference for Languages (CEFR) is one of the well-known standard language framework<sup>3</sup> developed by The Council of Europe and used for assessing general language competencies such as reading, writing, and listening (North, 2007, 2014). The CEFR uses a six-point level scale of A1, A2, B1, B2, C1, and C2, which denotes increasing complexities in instructional content development. We use the level descriptors compiled by Natova (2021), which cover three aspects, namely (1) Meaning/Purpose, (2) Structure, and (3) Grammatical Complexity, describing the characteristics of desired content per level as shown in Table 9. We omit a fourth aspect of Reader’s Knowledge Demands from the standard as this heavily depends on the reader’s background knowledge and is entirely subjective (Forey, 2020; Forey and Cheung, 2019).

The Common Core Standards (CCS) is an academic standard<sup>4</sup> developed by the US National Governors Association and the Council of Chief State School Officers (CCSSO) which has been widely adopted by schools across the United States for its K-12 curriculum. In this study, we adapt the recommended model of CCS for assessing text complexity, which includes two main variables: (1) Qualitative Dimensions and (2) Quantitative Dimensions. However, similar to the CEFR standard, we do not include the last variable, which is Reader Considerations, as this requires professional judgment or a teacher’s intervention. The description of each aspect of CCS is detailed in Table 9.

## 3 Standard-Aligned Content Generation (STANDARD-CTG)

Given the importance of adhering to expert-defined standards in the context of language assessment, we introduce a new task we refer to as standardaligned content generation (STANDARD-CTG). The overarching goal of STANDARD-CTG is to pave the way for new approaches that aim to integrate the conventional methodologies of controllable text generation in NLP with actual constraints provided by domain experts across interdisciplinary fields such as education, engineering, and medicine through documented standards. To align with terminologies used in education and other noncomputing literature, in this work, we use the term content generation instead of text generation as usually seen in technical NLP literature.

We represent the task of STANDARD-CTG using the following formulation:

$$
\begin{array} { r } { \mathbf { S T A N D A R D - C T G ( X , D _ { S t a n d a r d } ) } } \\ { = \mathcal { L } ( \mathcal { M } _ { \theta } ( \mathbf { X } , \tilde { \mathbf { K } } _ { \mathrm { S t a n d a r d } } ) , \mathbf { E } ) } \end{array}\tag{1}
$$

where $\mathcal { L }$ is a general evaluator that tests how close a language model’s $\mathcal { M } _ { \theta }$ generated content X is with gold-standard examples E through learning transformed knowledge representations $\tilde { \mathbf { K } } _ { \mathrm { S t a n d a r d } }$ of the selected standard $ { \mathbf { D } } _ { \mathrm { S t a n d a r d } }$ . The evaluator $\mathcal { L }$ can assume many forms, including model-based, distance-based, and reference-based scoring. We pattern our major experiments in the succeeding sections based on this formulation.

## 4 The STANDARDIZE Framework

Given that expert-defined standards are naturally information-rich, lengthy, and complex, our main hypothesis in this study is that in order for a generative language model to produce content that is aligned with the specifications provided by a standard, the information found in the standard must be considered in the generation process. The challenge then is redirected towards how any information extracted can be represented as something that the generative model will find useful.

Towards addressing STANDARD-CTG, we propose STANDARDIZE, a retrieval-style in-context learning-based framework that exploits the rich information found in standards and transforms this into knowledge artifacts to improve the quality of content produced by generative models. Figure 1 encapsulates this framework in a visual manner. In the succeeding sections, we discuss the proposed STANDARDIZE framework more thoroughly.

Target Specification Extraction is performed first to obtain informative tags in the prompt and to correctly match this information within the standards. For academic standards in language assessment, these specifications should provide information about who will be content delivered to (target audience) and using what specific standard out of many (CEFR or CCS). Thus, these two information tags are the basic required input for the process. As an example shown in Figure 1, the extracted specifications provided in the prompt are A2 readers, which points to a particular group of learners requiring low-leveled reading materials, and CEFR scale, which denotes the selected standard where properties of A2-level texts are described.

Specification Lookup and Retrieval is then performed next upon extracting the target specifications. A lookup process is done to find a match with the selected standard, usually in the form of a database or an external machine-readable file. In this work, we simply transformed the level-specific descriptors from Natova (2021) into a .csv file. The information from the standard in the form of aspects (or characteristics) that match the target specifications is then retrieved. The length and complexity of a standard’s level of information regarding its specifications may vary. As shown in Figure 1 for the CEFR standard, the retrieved information that matches the desired level of complexity for the target audience (A2 readers) can be checked at Table 9.

Knowledge Augmentation is done last but is the most important process of the pipeline. We propose a further technical augmentation of information found in standards to obtain knowledge artifacts in the prompts. These knowledge artifacts can range from simple additional information already present in the standard to complex representations, such as incorporating actual linguistic features to control the granularity of the generation process. Recent works surveying the performance of open and closed models have shown that non-informative style of prompting language models, such as the teacher style shown in Figure 1, is effective only to a certain extent and may be biased towards content generation in lower levels, such as A2 or B1 in the CEFR standards (Imperial and Madabushi, 2023; Ribeiro et al., 2023).

## 5 Knowledge Artifacts for STANDARDIZE

In this section, we discuss the knowledge artifacts $\tilde { \mathbf { K } } _ { \mathrm { S t a n d a r d } }$ extracted from the two educational standards $ { \mathbf { D } } _ { \mathrm { S t a n d a r d } }$ used in the STANDARDIZE framework and how they are integrated into the generation setup via in-context learning.

Baseline (Teacher Style) We treat the Teacher Style method as seen in Figure 1, where a simple, non-enriched prompt contains the target category from each standard, as the baseline for performance. We use this term in observance of how non-technical users, especially teachers, interact with generative chat interfaces (Imperial and Tayyar Madabushi, 2023).

Aspect Information (STANDARDIZE-A) represents the specific descriptive information provided in the standard. In the context of standards for content generation, aspect information is generally attributed to linguistic criteria of content with respect to its target audience. Figure 2 shows how aspect information from a standard (e.g., CEFR) can be integrated into the actual prompt. The addition of aspect criteria information ensures that the generative model will have access to explicit characteristics of the desired generated content in different dimensions.

Linguistic Flags (STANDARDIZE-L) represent the controllable attribute-based variables of a standard that a generative model can use to steer the direction of content generation. In the STANDARD-IZE framework, this process serves as a rewrite function where a generative model is asked to produce an initial content first using another method prompting (e.g., aspect information in Figure 2), and rewrites this by comparing linguistic flag values of the initially generated content against the mean value of a gold standard dataset of the target level. An example is illustrated in Figure 3 where the mean type-token ratio of a collection of goldstandard B1-level text 12.5 is added to the prompt while being compared to the current type-token value of the story, which is 4.2. A verbalizer is used to transform the computed linguistic flags into natural language prompts. The keywords increase and decrease are used in constructing the prompts to provide a sense of direction for the generative model.

![](images/9c5815c94ba180777ce3ecc9f5e0315f89ae809ba5f83119a106ed5b259e4623.jpg)  
Figure 2: A standard contains recommended characteristics of content across one or more domain-specific aspects or criteria. This figure shows an example of the CEFR standard where the set of criteria includes depth of meaning, structure, and grammatical complexity.

![](images/214a1289a57b44a9c835a6cedab7303dc12cd164280ce002997a2fb059d5f4fa.jpg)  
Figure 3: A standard contains aspect definition which can be represented by flags such as linguistic variables. Given the mean values from gold-standard data in the target level, the generative model can then be steered to push the property of its generated content using directional instructions such as increase or decrease.

In this work, we select 2 to 4 linguistic flags for both CEFR and CCS as reported in Table 9. The selection of what linguistic flags to use can be as simple as referring to what the definitions of aspects provide and need not be exhaustively many. For example, in CEFR, the Organization aspect is defined through different levels as "text is often short and observes chronological and predictable structure" for A2 and "text is can be long but not complex" for B1. Thus, we select average sentence and word lengths as a linguistic flag to capture this aspect. The full table of average values of linguistic flags can be found in Appendix A.5.

![](images/2420410d67ad5a8f3f9780cd190aafb1620ec6774771d943c395c1b1dd9cecd5.jpg)  
Figure 4: A standard contains recommended exemplars that serve as gold-standard reference. This figure shows an example of the CEFR standard where three wellknown pieces of literature are provided as examples of content that conforms to the target level specified (B1).

Exemplars (STANDARDIZE-E) represent the recommended examples by experts or developers of standards for reference of users. The addition of exemplars or any artifact found in the standard that showcases gold-standard output allows the generative model to have a sense of implicit knowledge during the content generation process. For example, in Figure 4, the exemplars for a B1-level content include Frankenstein by Mary Shelley, a well-known piece of gothic fiction. Although indirectly, any large language model trained using internet data (e.g., Wikipedia dumps) may have already formed a sense of knowledge of how this literature looks like (Karamolegkou et al., 2023; Petroni et al., 2019). We use the actual recommended exemplars from the CCS while we collected exemplars from the Penguin Readers publishing platform<sup>5</sup> which provides expert-curated literature for CEFR. The full list of exemplars for both standards can be found in the Appendix A.4.

All (STANDARDIZE-⋆) represents the combination of all extract knowledge artifacts mentioned above in one prompt.

## 6 Experimental Setup

In this section, we detail the specifications and technical configurations for the study’s main experiments. We also cover information on the datasets used, models, and generation tasks.

## 6.1 Tasks and Datasets

For this study, we specifically center our experimentation on the general task of story or narrative generation. We consider the subfield’s rich literature and active research community in NLP (Alhussain and Azmi, 2021), as well as being one of the most common examples demonstrated across the education community regarding the use of generative text interfaces for content generation (Kasneci et al., 2023; Whalen et al., 2023). Further, we differentiate two tasks used in our work for narrative generation as listed below.

Task 1: Context Assisted Story Generation. For this setup, we provide preliminary context in the form of 50 to 70 words (or approximately 3 to 5 sentences) in the prompt to guide the generative language model in producing the story continuation. We select the CEFR as the standard of choice to evaluate this approach and use the European Language Grid (ELG) corpus<sup>67</sup> compiled by Breuker (2022) to construct the prompts. The balanced corpus contains 300 CEFR-aligned English texts produced by experts and distributed across five levels A2, B1, B2, C1, C2 with 60 instances each. A1 is omitted due to lack of resources (n < 20).

Task 2: Theme Word Story Generation. In contrast to the previous setup, this method introduces only a single theme word for the generative language to produce a narrative from scratch, which allows for increased diversity in the content (Daza et al., 2016; Peng et al., 2018). To compile a theme words list, we select 50 random English noun words in plural form (e.g., dragons, mysteries, voyages) from the Corpus of Contemporary American English (COCA) (Davies, 2009) and prompt the generative model iteratively for each level in the standard. We investigate the application of CCS as the standard of choice in this setup.

## 6.2 Models

We select a number of generative language models $\mathcal { M } _ { \theta }$ for content generation, each with its own advantage. For the open models, we use a number of well-known models in the 2B-7B range, including Llama2-Chat-7B (Touvron et al., 2023a), OpenChat-7B (Wang et al., 2023), and Longform-2.7B (Köksal et al., 2023). For the closed model, we use GPT-4-Turbo (OpenAI, 2023). More information on the models can be found in Appendix A.3.

## 6.3 Automatic Evaluation

We perform a diverse set of evaluation methods given examples from gold-standard datasets E to test the qualities of the generated content of models, as discussed further below.

Model-Based Classifiers. For the context-assisted story generation task using CEFR standards with 5 classes, we use a Random Forest classifier trained from a separate collection of Cambridge Exams dataset with CEFR labels used in the works of Xia et al. (2016) and Imperial and Tayyar Madabushi (2023). This classifier has an accuracy of 0.912 using 79 length-normalized<sup>8</sup> linguistic features. For the theme word story generation using CCS standards with 2 classes, we used an XGBoost classifier from the work of (Imperial, 2021) trained from the only CCSaligned data found online and compiled by Flor et al. (2013) with an accuracy of 0.917 using a combination of BERT embeddings and the same linguistic features stated above. Due to its limited size of 168, we grouped the dataset into binary categories, elementary (grades 4 8) and advanced (grades 9 12), with 48 and 73 documents per class, respectively. We consider both classifiers in our work for their high accuracies (> 90%).

Fluency and Diversity. We evaluate the level of fluency and content diversity of the generated content by the models as done in previous narrative generation works (DeLucia et al., 2021; See et al., 2019). The former is measured through perplexity with an external GPT-2 model, while the latter is the density of distinct n-grams.

Linguistic Similarity. We evaluate the level of linguistic similarity of the generated content against the gold-standard datasets for CEFR (ELG) and CCS (COCA) as mentioned in Section 6. For this method, we calculate the mean Euclidean distance of all the linguistic flags used for both standards and their levels listed in Table 9. This method provides a notion of how close the characteristics of a set of model-generated texts (e.g., GPT-4 generated B1 texts) is to its equivalent gold standard (e.g., actual B1-level texts written by experts).

## 6.4 Expert Annotator Evaluation

To confirm the quality of model-generated content, we also perform an evaluation using judgment from domain experts. Through our university network, we collaborated with three experts with 15 30 years of experience in linguistic and language assessment with frameworks such as CEFR, CCS, TOEFL, and IELTS. Drawing on the methods used in previous studies (DeLucia et al., 2021), we asked the experts to judge the model-generated content through the following variables below. Additional information on the human evaluation can be found in Appendix A.6.

Grammaticality and Coherence. The former variable evaluates the level of naturalness or fluency of the generated output as if it has been written by a native English speaker. The latter measures the level of cohesion between sentences where the narrative stays on-topic, and the text overall builds a consistent story and the flow of information is smooth and easy to follow.

Grade Complexity Distinction. This variable measures the obviousness of the complexity of a generated story on a target level (e.g., A1) with respect to another story of a different level (e.g., A2). This variable is relatively more challenging than the other metrics, as the difference between adjacent levels may not be as straightforward without referring to the quantitative characteristics of the texts. However, we included this assessment in the evaluation process to judge the quality of the model-generated texts.

<table><tr><td>Model</td><td>Precise Accuracy</td><td>Adjacent Accuracy</td><td>Fluency (perplexity)</td><td>Diversity (distinct-n)</td></tr><tr><td colspan="5">Llama2 7B</td></tr><tr><td>- Teacher Style</td><td>0.203</td><td>0.636</td><td> ${ \bf 1 3 . 1 8 9 \pm 4 . 8 8 }$ </td><td> $0 . 1 5 6 \pm 0 . 0 3$ </td></tr><tr><td>- STANDARDIZE-A</td><td>0.270</td><td>0.626</td><td> $1 3 . 6 9 4 \pm 7 . 7 4$ </td><td> $0 . 1 5 5 \pm 0 . 0 2$ </td></tr><tr><td> $- \mathbf { S } \mathbf { T A N D A R D I Z E } { - } E$ </td><td>0.320</td><td>0.683</td><td> $1 5 . 5 7 6 \pm 3 . 3 1$ </td><td> $0 . 1 8 8 \pm 0 . 0 1$ </td></tr><tr><td> $- \mathbf { S } \mathbf { T A N D A R D I Z E }  – L$ </td><td>0.273</td><td>0.606</td><td> $2 0 . 1 7 5 \pm 4 . 4 7$ </td><td> $0 . 1 8 6 \pm 0 . 0 1$ </td></tr><tr><td>- STANDARDIZE-★</td><td>0.354</td><td>0.670</td><td> $1 7 . 8 9 2 \pm 3 . 9 4$ </td><td> $\mathbf { 0 . 1 9 3 \pm 0 . 0 1 }$ </td></tr><tr><td colspan="5">OpenChat 7B</td></tr><tr><td>- Teacher Style</td><td>0.237</td><td>0.626</td><td> $2 2 . 0 3 9 \pm 7 . 7 0$ </td><td> $0 . 1 7 0 \pm 0 . 0 2$ </td></tr><tr><td> $- \mathbf { S } \mathbf { T A N D A R D I Z E } { - } A$ </td><td>0.243</td><td>0.630</td><td> $2 1 . 1 9 5 \pm 7 . 6 6$ </td><td> $0 . 1 7 1 \pm 0 . 0 2$ </td></tr><tr><td>- STANDARDIZE-E</td><td>0.253</td><td>0.600</td><td> ${ \bf 1 } 3 . 9 3 { \bf 1 } \pm 2 . 9 7$ </td><td> $0 . 1 7 8 \pm 0 . 0 1$ </td></tr><tr><td> $\cdot \cdot \cdot \mathrm { S T A N D A R D I Z E } { \cdot } L$ </td><td>0.270</td><td>0.546</td><td> $1 8 . 1 8 2 \pm 8 . 5 2$ </td><td> ${ \bf 0 . 1 7 9 \pm 0 . 0 2 }$ </td></tr><tr><td> $S _ { \mathrm { T A N D A R D I Z E ^ { - } \star } }$ </td><td>0.253</td><td>0.596</td><td> $1 2 . 8 0 6 \pm 2 . 7 0$ </td><td> $0 . 1 7 1 \pm 0 . 0 3$ </td></tr><tr><td colspan="5">Longform 3B</td></tr><tr><td>- Teacher Style</td><td>0.230</td><td>0.606</td><td> $1 8 . 2 0 9 \pm 6 . 0 1$ </td><td> $0 . 1 5 9 \pm 0 . 0 2$ </td></tr><tr><td> $- \mathbf { S } \mathbf { T A N D A R D I Z E } { - } A$ </td><td>0.223</td><td>0.610</td><td> $1 7 . 9 8 2 \pm 9 . 2 1$ </td><td> $0 . 1 5 7 \pm 0 . 0 2$ </td></tr><tr><td> $- \mathbf { S } \mathbf { T A N D A R D I Z E } { - } E$ </td><td>0.257</td><td>0.496</td><td> $2 5 . 0 7 5 \pm 8 . 8 0$ </td><td> ${ \bf 0 . 1 9 2 \pm 0 . 1 1 }$ </td></tr><tr><td> $- \mathbf { S } \mathbf { T A N D A R D I Z E } { - } L$ </td><td>0.283</td><td>0.586</td><td> $\mathbf { 1 6 . 9 2 6 \pm 6 . 9 1 }$ </td><td> $0 . 1 6 1 \pm 0 . 0 3$ </td></tr><tr><td> $- \mathrm { S T A N D A R D I Z E } { \mathrm { - } } \star$ </td><td>0.277</td><td>0.543</td><td> $1 6 . 8 0 6 \pm 7 . 4 0$ </td><td> $0 . 1 7 0 \pm 0 . 0 4$ </td></tr><tr><td colspan="5">GPT-4</td></tr><tr><td>- Teacher Style</td><td>0.227</td><td>0.630</td><td> $2 7 . 3 5 7 \pm 6 . 3 0$ </td><td> $0 . 1 8 7 \pm 0 . 0 8$ </td></tr><tr><td> $- \mathbf { S } \mathbf { T A N D A R D I Z E } { - } A$ </td><td>0.397</td><td>0.846</td><td> $2 9 . 7 2 9 \pm 9 . 5 8$ </td><td> $0 . 1 7 4 \pm 0 . 0 1$ </td></tr><tr><td> $- \mathbf { S } \mathbf { T A N D A R D I Z E } { - } E$ </td><td>0.307</td><td>0.703</td><td> $3 0 . 3 5 7 \pm 9 . 7 9$ </td><td> $0 . 1 8 2 \pm 0 . 0 1$ </td></tr><tr><td> $- \mathbf { S } \mathbf { T A N D A R D I Z E } { - } L$ </td><td>0.480</td><td>0.906</td><td> $2 4 . 1 1 5 \pm 7 . 0 4$ </td><td> $0 . 1 9 4 \pm 0 . 0 3$ </td></tr><tr><td></td><td>0.540</td><td>0.803</td><td> $\mathbf { 2 2 . 5 9 1 \pm 1 . 6 1 }$ </td><td></td></tr><tr><td> $- \mathbf { S } _ { \mathrm { T A N D A R D I Z E - \star } }$ </td><td></td><td></td><td></td><td> ${ \bf 0 . 2 1 8 \pm 0 . 0 5 }$ </td></tr></table>

<table><tr><td>Model</td><td>Precise Accuracy</td><td>Fluency (perplexity)</td><td>Diversity (distinct-n)</td></tr><tr><td colspan="4">Llama2 7B</td></tr><tr><td>- Teacher Style</td><td>0.470</td><td> $1 7 . 9 3 6 \pm 4 . 3 2$ </td><td> $0 . 1 8 4 \pm 0 . 0 1$ </td></tr><tr><td> $- \mathbf { S } \mathbf { T A N D A R D I Z E } { - } A$ </td><td>0.580</td><td> $2 2 . 0 7 0 \pm 1 . 7 5$ </td><td> $0 . 1 7 1 \pm 0 . 0 1$ </td></tr><tr><td> $\mathbf { S } _ { \mathrm { T A N D A R D I Z E } - } E$ </td><td>0.570</td><td> ${ \pm } 3 . 4 8 4 \pm 2 . 5 0$ </td><td> $\mathbf { 0 . 1 9 3 \ : \pm 0 . 0 1 }$ </td></tr><tr><td> $- \mathrm { S T A N D A R D I Z E } { \cdot } L$ </td><td>0.720</td><td> $1 5 . 0 6 6 \pm 2 . 4 7$   $1 4 . 7 0 7 \pm 2 . 4 0$ </td><td> $0 . 1 9 1 \pm 0 . 0 1$   $\mathbf { 0 . 1 9 3 \ : \pm 0 . 0 1 }$ </td></tr><tr><td> $\mathbf { S T A N D A R D I Z E { - } { \star } }$ </td><td>0.623</td><td></td><td></td></tr><tr><td colspan="4">OpenChat 7B</td></tr><tr><td> $- \mathrm { T e a c h e r } \mathrm { S t y l e }$ </td><td>0.470</td><td> $1 6 . 1 1 6 \pm 1 2 . 3 9$ </td><td> $0 . 1 6 6 \pm 0 . 0 5$ </td></tr><tr><td> $\cdot \cdot \cdot \operatorname { S T A N D A R D I Z E } - A$ </td><td>0.550</td><td> $1 9 . 4 4 4 \pm 2 . 5 7$   $\smash { 1 9 . 4 4 4 \pm 2 . 5 } / $ </td><td> $0 . 1 7 2 \pm 0 . 0 1$ </td></tr><tr><td>- STANDARDIZE-E</td><td>0.490</td><td> $1 2 . 4 3 8 \pm 1 . 8 5$ </td><td> $0 . 1 7 8 \pm 0 . 0 1$ </td></tr><tr><td> $\cdot \ S _ { \mathrm { T A N D A R D I Z E } - L }$ </td><td>0.580</td><td> $1 3 . 7 3 4 \pm 2 . 5 3$ </td><td> ${ \bf 0 . 1 8 0 \pm 0 . 0 1 }$ </td></tr><tr><td> $S _ { \mathrm { \Delta T A N D A R D I Z E - } } \star$ </td><td>0.560</td><td> $\mathbf { 1 0 . 7 1 7 \pm 1 . 5 3 }$ </td><td> $0 . 1 6 9 \pm 0 . 0 1$ </td></tr><tr><td colspan="4">Longform 3B</td></tr><tr><td>- Teacher Style  $- \mathbf { S } \mathbf { T A N D A R D I Z E } { - } A$ </td><td>0.500 0.450</td><td> $1 3 . 6 5 7 \pm 5 . 3 9$   $1 7 . 9 1 8 \pm 4 . 7 4$ </td><td> $0 . 1 5 4 \pm 0 . 0 4$   $0 . 1 4 8 \pm 0 . 0 1$ </td></tr><tr><td> $- \mathrm { S T A N D A R D I Z E } – E$ </td><td>0.510</td><td> $1 4 . 2 7 7 \pm 2 . 7 9$ </td><td> $0 . 1 5 1 \pm 0 . 0 2$ </td></tr><tr><td> $- \mathrm { S T A N D A R D I Z E } { \cdot } L$ </td><td>0.610</td><td> $1 3 . 3 9 8 \pm 3 . 9 3$ </td><td> $0 . 1 4 8 \pm 0 . 0 4$ </td></tr><tr><td> $- \mathrm { S T A N D A R D I Z E } { \mathrm { - } } \star$ </td><td>0.620</td><td> $\mathbf { 1 0 . 4 0 0 \pm 1 . 5 3 }$ </td><td> ${ \bf 0 . 1 6 9 \pm 0 . 0 1 }$ </td></tr><tr><td colspan="4">GPT-4</td></tr><tr><td>- Teacher Style</td><td>0.590</td><td> $3 2 . 4 4 7 \pm 7 . 4 6$ </td><td> $0 . 1 9 5 \pm 0 . 0 1$ </td></tr><tr><td> $- \mathbf { S } \mathbf { T A N D A R D I Z E } { - } A$ </td><td>0.550</td><td> $3 1 . 7 6 5 \pm 1 1 . 3 0$ </td><td> $0 . 1 6 9 \pm 0 . 0 1$ </td></tr><tr><td> $- \mathrm { S T A N D A R D I Z E } – E$ </td><td>0.520</td><td> $2 9 . 9 1 2 \pm 6 . 8 1$ </td><td> $0 . 1 8 4 \pm 0 . 0 1$ </td></tr><tr><td> $- \mathbf { S } \mathbf { T A N D A R D I Z E } { - } L$ </td><td>0.610</td><td> $2 6 . 9 1 2 \pm 6 . 1 1$ </td><td> $0 . 1 5 5 \pm 0 . 0 1$ </td></tr><tr><td> $S _ { \mathrm { \Delta T A N D A R D I Z E - } } \star$ </td><td>0.790</td><td> $\pm \mathbf { 1 . 2 7 7 \pm 4 . 5 0 }$ </td><td> ${ \bf 0 . 1 9 8 \pm 0 . 0 1 }$ </td></tr></table>

Table 1: Experiment results comparing the conventional teacher style prompting with the STANDARDIZE framework for the Common European Framework of Reference for Languages (CEFR) standards.  
Table 2: Experiment results comparing the conventional teacher style prompting with the STANDARD-IZE framework for the Common Core Standards (CCS).

## 7 Results and Discussion

We discuss the results of our experiments procedures with the methods from the STANDARDIZE framework.

## 7.1 Standard Alignment via Classification Performance

The overall performance of models for CEFR and CCS are reported in Tables 1 and 2. For CEFR, the top-performing setup across the four models all belong to the STANDARDIZE framework. We report over a 100% increase in performance using the best setup with GPT-4 with STANDARDIZE-⋆ in precise accuracy from 0.227 to 0.540 and a 43% increase for adjacent accuracy from 0.630 to 0.906 compared to the teacher style method. Through STANDARDIZE, open models also gained substantial boosts in performance, such as Longform up by 23%, OpenChat up by 14%, and Llama2 by 74%. In terms of adjacent accuracies, GPT-4 remained the best model for preserving the ordinality of the labels with 0.906, up by 44%. With CCS, the general scores obtained in this setup are higher compared to CEFR with five classes due to binary labeling. We see a similar pattern where all open and closed models obtained the best performance, with boosts ranging from 3% to 45% using linguistic flags STANDARDIZE-L and a combination of all knowledge artifacts STANDARD-IZE-⋆ to refine the generated content toward the target level. From these findings, we provide concrete evidence that using the actual content of the standards through knowledge artifact representations from STANDARDIZE may be crucial when prompting LLMs via in-context learning to produce standard-aligned content for classroom use.

## 7.2 Standard Alignment via Linguistic Similarity

We visualize the distributions of the best performing STANDARDIZE methods in Figures 6 to 8 with comparison to the teacher style method. From the results, we observe that the general trend of using STANDARDIZE produces a more stable distribution across the variables it is explicitly controlling for (e.g., average sentence length or type token diversity as listed in Table 9), particularly with the CCS standards. We also notice that the distributions using STANDARDIZE-L also produce distributions closer to the mean (represented as a yellow star) from their corresponding gold-standard data. Moreover, in terms of linguistic similarity, as reported in Table 3, STANDARDIZE makes the quality of model generations more similar to the linguistic characteristics of the gold standard datasets in

CEFR and CCS. Overall, these findings further strengthen the evidence of using STANDARDIZE in producing linguistically similar content with gold-standard data compared to the conventional teacher style method.

<table><tr><td>Setup</td><td>A2</td><td>B1</td><td>B2</td><td>C1</td><td>C2</td></tr><tr><td>Teacher Style</td><td>136.7</td><td>96.7</td><td>169.9</td><td>307.3</td><td>291.6</td></tr><tr><td>STANDARDIZE-*</td><td>61.4</td><td>106.2</td><td>97.64</td><td>219.6</td><td>234.7</td></tr></table>

<table><tr><td>Setup</td><td>Elementary</td><td>Advanced</td></tr><tr><td>Teacher Style</td><td>76.1</td><td>157.9</td></tr><tr><td>STANDARDIZE-*</td><td>63.8</td><td>125.7</td></tr></table>

Table 3: Mean Euclidean distances of generated content using simple teacher style prompting vs. STANDARD-IZE-⋆ for CEFR (top) and CCS (bottom).

## 7.3 Assessment of Generation Qualities via Expert Judgment and Automatic Metrics

For both computed fluency and content diversity, we see similar results from the previous evaluation techniques where the best performing models are all models improved through the STANDARDIZE framework particularly OpenChat, Longform, and GPT-4. Looking at expert evaluations as reported in Figure 5, we observe consistent high ratings on grammaticality and coherence of the topi performing model, GPT-4 with STANDARDIZE-⋆, for both CEFR and CCS with an average of 3.13 and 3.35, respectively. On the grade complexity distinction, all three expert evaluators were able to achieve high accuracies (> 0.70) in selecting correct simple and complex texts from the model-generated data, denoting the obviousness of complexity. Likewise, all expert evaluation tests achieved strong inter-rater reliability scores (> 0.30) through Kendall’s W (Kendall, 1948). With these findings, we affirm the effectivity of the STANDARDIZE framework through expert judgment on generating more fluent, grammatical, grade-distinct, and diverse content compared to the teacher-style approach.

## 8 Implications to Generative Models for Education

We discuss important points highlighting the real-world implications of our study within and beyond language model experimentations.

Validity on Global Education Context. Our main contribution, the STANDARDIZE framework, leverages the idea of a more holistic method for capturing the intricacies and complexities of educational standards for content generation. Our experiments with the CEFR and CCS standards showcase an opportunity for the generated texts of language model interfaces such as GPT-4, which are commonly used by educators and teachers, to be aligned with international language proficiency levels. Moreover, showing the effectiveness of STANDARDIZE on the aforementioned internationally recognized academic standards used in European and Northern American schools signifies the framework’s strong potential for cross-curricula application. Thus, we invite future researchers to explore, validate, and propose derivations of our base framework for their own languages and language-specific standards for content generation.

Towards More Personalized Content Generation. Investigating the potential of generative models for personalized learning, such as providing adaptive feedback aligned with students’ needs, is an active area in AI for education (Kasneci et al., 2023; Meyer et al., 2023; Sailer et al., 2023; Tack and Piech, 2022). This work contributes toward the goal of helping educators craft more personalized content for learners using the capabilities of large language models based on an assigned language proficiency level described by a standard. While we present a novel task specifically targeted for the NLP community to encourage research in this direction (STANDARD-CTG as covered in Section 3), our results may be useful for educators by providing context on better methods for generating level or target audience-specific texts by prompting language models using information found in educational standards.

## 9 Related Work

Research in complexity-controlled generation has explored diverse variables in terms of text format, granularity, and task variation. The work of Agrawal and Carpuat (2019) introduced controlling for specific complexity in the machine translation task. The following works of Agrawal and Carpuat (2023) and Ribeiro et al. (2023) explored gradespecific text simplification and summarization using control tokens and reinforcement learning, respectively. Currently, only two works have investigated incorporating CEFR for language learning content generation. Stowe et al. (2022) and Imperial and Tayyar Madabushi (2023) both made use of CEFR-aligned text for NLG. However, none of them made use of the actual guideline information found in CEFR during the generation process.

![](images/e06344839b020a8fe48450a45014feef502d81e209ff716c964b95ebe866e9de.jpg)  
(a) Expert evaluation on the generation quality of the GPT-4 model with STANDARD-IZE-⋆ for CEFR. Inter-rater reliability using Kendall’s W is 0.34 which denotes moderate agreement.

![](images/8352ea2bea8315aea4230e7ffbef0337bc7d1f00094760f05cd177b96952306b.jpg)  
(b) Expert evaluation on the generation quality of the GPT-4 model with STANDARDIZE-⋆ for CCS. Inter-rater reliability using Kendall’s W is 0.40 which denotes strong agreement

![](images/9f933c36d44f849514dce97c51cf1082838d62c8e2ea16380594a381ba6eecdd.jpg)  
(c) Performance of expert evaluators on estimating the complexity of generated content for CEFR and CCS. Inter-rater reliability using Kendall’s W is 0.45 which denotes strong agreement.  
Figure 5: Overview of mean ratings of grammaticality or fluency, coherence, and grade complexity distinction from the human expert evaluations using the top-performing models for CEFR and CCS. All evaluation procedures obtain generally favorable results as well as acceptable inter-rater reliability scores (equal and above the threshold of 0.30)

Our study’s main novelty is the holistic capture of expert-defined standards by exploring possible representations we call artifacts that can improve how a language model refines its content generation process with respect to a target language proficiency level. We emphasize the importance of the use of in-context learning without additional finetuning in this work to preserve the capabilities of models across other language-related tasks. Our STANDARDIZE framework derives motivation from Zhou et al. (2023) and Ram et al. (2023), where a verbalizer is used to transform quantitative constraints into natural language for prompting, as well as the use of a lookup and retrieval phase where aspect information is added in the prompt to influence model controllability.

## 10 Conclusion

In this work, we proposed the STANDARDIZE framework using knowledge artifacts that allowed large language models such as Llama2 and GPT-4 to gain significant performance boosts (45% - 100%) on generating content aligned with educational standards as well as preserving important narrative qualities such as fluency, grammaticality, coherence, and grade distinctness. From this, we see a very promising potential for cross-domain and cross-standard generalization of our proposed method with the range of educational contexts around the world and invite future work to build on our baseline models.

## Ethical Considerations

All datasets and corpora used in this study, such as the ELG (Breuker, 2022), Cambridge Exams (Xia et al., 2016), and CCS (Flor et al., 2013), are already established and accessible for research purposes. We observe a specific tone in the discussion of our experiments, emphasizing that the main motivation of the work is that language models such as GPT-4 can provide assistance in producing content that is more aligned or faithful with the constraints of standards such as CEFR or CCS without implying that they can replace experts in the field or produce better quality than the gold-standard data. Further, we also do not imply that any model enriched by any computational method to produce more standard-aligned content can replace the standard itself. Overall, we do not foresee any serious ethical issues in this study.

## Limitations

Language Coverage of Standards. This work is mainly centered on the use of datasets and standards for the English language. While standards for language assessment, such as CEFR, have expanded through the years with versions to cover other languages, such as German, Czech, and Italian (Vajjala and Rama, 2018), we do not claim that our results will be able to generalize and have the same advantages with these languages. However, investigating this direction may be a good research opportunity for future work.

Dependence on Evaluation Methods. As observed in Section 7, we made sure to cover a variety of evaluation procedures for testing standard alignment instead of only using modelbased methods such as a classifier. The limitation here is that trained classifiers are dependent on factors such as their accuracy, the quantity of data, the complexity of the training algorithm, and the quality of features. Thus, other means of evaluating alignment that is more direct, such as computed feature distances against a gold-standard dataset, is always recommended. Moreover, our model-based CEFR and CCS evaluators make use of artifacts such as datasets and tools for feature extraction from peer-reviewed papers (Xia et al., 2016; Flor et al., 2013). We are aware of paid third-party services online that promise more accurate classification of labels in CEFR, but they generally do not provide details on linguistic predictors used for prediction. Thus, this may not be a practical option for research.

Attribute-Based Standards. The standards used in this study, CEFR and CCS, are attribute-based standards that specify recommended characteristics of texts that are countable (e.g., sentence length or average number of words). These specifications contribute towards the overall complexity of texts which are within the scope of CEFR and CCS. Standards in other domains may come in different forms of constraints, such as dependence on an external specialized vocabulary or following specific sequential processes to arrive at a result. Moreover, our exploration of CEFR and CCS standards is centered on the downstream task of narrative generation, as this fits the most generic form of reading material in classrooms. We leave the exploration of extending the STANDARDIZE framework to other domains that also observe attribute-based specifications as well as other adjacent text generation tasks (e.g., summary generation) in future work.

## Acknowledgements

We are grateful to the anonymous reviewers and Action Editors in ARR for their feedback on the improvement of this paper and to Dr. Brian North for the insightful discussions on capturing language standards, including CEFR, as part of the theoretical component of this work. We also thank Dr. Samantha Curle and Dr. Reka Jablonkai from the Department of Education at the University of Bath for helping with the evaluation of model-generated texts. This work made use of the Hex GPU cloud of the Department of Computer Science at the University of Bath. JMI is supported by the National University Philippines and the UKRI Centre for Doctoral Training in Accountable, Responsible, and Transparent AI [EP/S023437/1] of the University of Bath. We attribute the black icons used in Figure 1 to the collections of Design Circle and Victor Zukeran from the Noun Project and the colored teacher icon from Flaticon.

## References

Sweta Agrawal and Marine Carpuat. 2019. Controlling Text Complexity in Neural Machine Translation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1549– 1564, Hong Kong, China. Association for Computational Linguistics.

Sweta Agrawal and Marine Carpuat. 2023. Controlling Pre-trained Language Models for Grade-Specific Text Simplification. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 12807–12819, Singapore. Association for Computational Linguistics.

Arwa I Alhussain and Aqil M Azmi. 2021. Automatic Story Generation: A Survey of Approaches. ACM Computing Surveys (CSUR), 54(5):1–38.

Mark Breuker. 2022. CEFR Labelling and Assessment Services. In European Language Grid: A Language Technology Platform for Multilingual Europe, pages 277–282. Springer International Publishing Cham.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling Instruction-Finetuned Language Models. arXiv preprint arXiv:2210.11416.

Mark Davies. 2009. The 385+ million word Corpus of Contemporary American English (1990–2008+): Design, architecture, and linguistic insights. International Journal ofCorpus Linguistics, 14(2):159–190.

Angel Daza, Hiram Calvo, and Jesús Figueroa-Nazuno. 2016. Automatic Text Generation by Learning from Literary Structures. In Proceedings of the Fifth Workshop on Computational Linguistics for Literature, pages 9–19, San Diego, California, USA. Association for Computational Linguistics.

Alexandra DeLucia, Aaron Mueller, Xiang Lisa Li, and João Sedoc. 2021. Decoding Methods for Neural Narrative Generation. In Proceedings ofthe 1st Workshop on Natural Language Generation, Evaluation, and Metrics (GEM 2021), pages 166–185, Online. Association for Computational Linguistics.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical Neural Story Generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889–898, Melbourne, Australia. Association for Computational Linguistics.

Michael Flor, Beata Beigman Klebanov, and Kathleen M. Sheehan. 2013. Lexical Tightness and Text Complexity. In Proceedings of the Workshop on Natural Language Processingfor Improving Textual Accessibility, pages 29–38, Atlanta, Georgia. Association for Computational Linguistics.

Gail Forey. 2020. A whole school approach to SFL metalanguage and the explicit teaching of language for curriculum learning. Journal ofEnglishfor Academic Purposes, 44:100822.

Gail Forey and Lok Ming Eric Cheung. 2019. The benefits of explicit teaching of language for curriculum learning in the physical education classroom. English for Specific Purposes, 54:91–109.

Albert Gatt and Emiel Krahmer. 2018. Survey of the State of the Art in Natural Language Generation: Core tasks, applications and evaluation. Journal of Artificial Intelligence Research, 61:65–170.

Joseph Marvin Imperial. 2021. BERT embeddings for automatic readability assessment. In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pages 611–618, Held Online. INCOMA Ltd.

Joseph Marvin Imperial and Harish Tayyar Madabushi. 2023. Uniform Complexity for Text Generation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 12025–12046, Singapore. Association for Computational Linguistics.

Joseph Marvin Imperial and Harish Tayyar Madabushi. 2023. Flesch or fumble? evaluating readability standard alignment of instruction-tuned language models. In Proceedings ofthe Third Workshop on Natural Language Generation, Evaluation, and Metrics (GEM), pages 205–223, Singapore. Association for Computational Linguistics.

Antonia Karamolegkou, Jiaang Li, Li Zhou, and Anders Søgaard. 2023. Copyright Violations and Large Language Models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 7403–7412, Singapore. Association for Computational Linguistics.

Enkelejda Kasneci, Kathrin Seßler, Stefan Küchemann, Maria Bannert, Daryna Dementieva, Frank Fischer, Urs Gasser, Georg Groh, Stephan Günnemann, Eyke

Hüllermeier, et al. 2023. ChatGPT for Good? On Opportunities and Challenges of Large Language Models for Education. Learning and Individual Differences, 103:102274.

Maurice George Kendall. 1948. Rank correlation methods. American Psychological Association.

Abdullatif Köksal, Timo Schick, Anna Korhonen, and Hinrich Schütze. 2023. LongForm: Optimizing Instruction Tuning for Long Text Generation with Corpus Extraction. arXiv preprint arXiv:2304.08460.

Paul M La Marca, Doris Redfield, and Phoebe C Winter. 2000. State Standards and State Assessment Systems: A Guide to Alignment. Series on Standards and Assessments.

Bruce W. Lee and Jason Lee. 2023. LFTK: Handcrafted Features in Computational Linguistics. In Proceedings ofthe 18th Workshop on Innovative Use ofNLP for Building Educational Applications (BEA 2023), pages 1–19, Toronto, Canada. Association for Computational Linguistics.

Jesse G Meyer, Ryan J Urbanowicz, Patrick CN Martin, Karen O’Connor, Ruowang Li, Pei-Chen Peng, Tiffani J Bright, Nicholas Tatonetti, Kyoung Jae Won, Graciela Gonzalez-Hernandez, et al. 2023. Chatgpt and large language models in academia: opportunities and challenges. BioData Mining, 16(1):20.

Ivanka Natova. 2021. Estimating CEFR Reading Comprehension Text Complexity. The Language Learning Journal, 49(6):699–710.

Brian North. 2007. The CEFR Illustrative Descriptor Scales. The Modern Language Journal, 91(4):656– 659.

Brian North. 2014. The CEFR in practice, volume 4. Cambridge University Press.

OpenAI. 2023. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Nanyun Peng, Marjan Ghazvininejad, Jonathan May, and Kevin Knight. 2018. Towards Controllable Story Generation. In Proceedings ofthe First Workshop on Storytelling, pages 43–49, New Orleans, Louisiana. Association for Computational Linguistics.

Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. 2019. Language Models as Knowledge Bases? In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP),

pages 2463–2473, Hong Kong, China. Association for Computational Linguistics.

Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2023. In-Context Retrieval-Augmented Language Models. Transactions ofthe Association for Computational Linguistics, 11:1316–1331.

Leonardo F. R. Ribeiro, Mohit Bansal, and Markus Dreyer. 2023. Generating Summaries with Controllable Readability Levels. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 11669–11687, Singapore. Association for Computational Linguistics.

D Royce Sadler. 2017. Academic achievement standards and quality assurance. Quality in Higher Education, 23(2):81–99.

Michael Sailer, Elisabeth Bauer, Riikka Hofmann, Jan Kiesewetter, Julia Glas, Iryna Gurevych, and Frank Fischer. 2023. Adaptive feedback from artificial neural networks facilitates pre-service teachers’ diagnostic reasoning in simulation-based learning. Learning and Instruction, 83:101620.

Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, et al. 2021. Multitask Prompted Training Enables Zero-Shot Task Generalization. In International Conference on Learning Representations.

Abigail See, Aneesh Pappu, Rohun Saxena, Akhila Yerukola, and Christopher D. Manning. 2019. Do Massively Pretrained Language Models Make Better Storytellers? In Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL), pages 843–861, Hong Kong, China. Association for Computational Linguistics.

Kevin Stowe, Debanjan Ghosh, and Mengxuan Zhao. 2022. Controlled Language Generation for Language Learning Items. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 294–305, Abu Dhabi, UAE. Association for Computational Linguistics.

Anaïs Tack and Chris Piech. 2022. The AI Teacher Test: Measuring the Pedagogical Ability of Blender and GPT-3 in Educational Dialogues. In Proceedings of the 15th International Conference on Educational Data Mining, page 522.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. 2023. Alpaca: A Strong, Replicable Instruction-Following Model. Stanford Centerfor Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html, 3(6):7.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal

Azhar, et al. 2023a. LLaMA: Open and Efficient Foundation Language Models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv preprint arXiv:2307.09288.

Sowmya Vajjala and Taraka Rama. 2018. Experiments with Universal CEFR Classification. In Proceedings of the Thirteenth Workshop on Innovative Use of NLP for Building Educational Applications, pages 147–153, New Orleans, Louisiana. Association for Computational Linguistics.

Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, and Yang Liu. 2023. OpenChat: Advancing Open-source Language Models with Mixed-Quality Dataa. arXiv preprint arXiv:2309.11235.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. 2022. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5085–5109, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned Language Models are Zero-Shot Learners. In International Conference on Learning Representations.

Jeromie Whalen, Chrystalla Mouza, et al. 2023. Chat-GPT: Challenges, Opportunities, and Implications for Teacher Education. Contemporary Issues in Technology and Teacher Education, 23(1):1–23.

Menglin Xia, Ekaterina Kochmar, and Ted Briscoe. 2016. Text Readability Assessment for Second Language Learners. In Proceedings of the 11th Workshop on Innovative Use ofNLPfor Building Educational Applications, pages 12–22, San Diego, CA. Association for Computational Linguistics.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. OPT: Open Pre-trained Transformer Language Models. arXiv preprint arXiv:2205.01068.

Wangchunshu Zhou, Yuchen Eleanor Jiang, Ethan Wilcox, Ryan Cotterell, and Mrinmaya Sachan. 2023. Controlled text generation with natural language instructions. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 42602–42613. PMLR.

## A Appendix

## A.1 Libraries and Dependencies

We have used the following dependencies and Python libraries for the study: Linguistic Feature Tool Kit (LFTK) (Lee and Lee, 2023), Spacy (https://spacy.io/), Scikit-Learn (https: //scikit-learn.org/stable/), OpenAI API (https://openai.com/blog/open ai-api).

## A.2 Corpus Statistics

We provide basic statistical information about the various corpora used in the study.

<table><tr><td>Level</td><td>Size</td><td>Average Word Count</td><td>Average Sentence Count</td></tr><tr><td>A2</td><td>60</td><td>186.55</td><td>18.91</td></tr><tr><td>B1</td><td>60</td><td>264.25</td><td>15.90</td></tr><tr><td>B2</td><td>60</td><td>517.71</td><td>31.71</td></tr><tr><td>C1</td><td>60</td><td>728.93</td><td>40.70</td></tr><tr><td>C2</td><td>60</td><td>749.73</td><td>37.55</td></tr></table>

Table 4: Statistics of the ELG corpus (Breuker, 2022) used for the CEFR context assisted story generation task.

<table><tr><td>Grade</td><td>Size</td><td>Average Word Count</td><td>Average Sentence Count</td></tr><tr><td>Elementary</td><td>48</td><td>204.91</td><td>28.55</td></tr><tr><td>Advanced</td><td>73</td><td>255.17</td><td>31.08</td></tr></table>

Table 5: Statistics of the official CCS-aligned corpus (Flor et al., 2013) used as gold-standard dataset for the STANDARDIZE-L artifact and for training the CCS classifier used in Section 7.

<table><tr><td>Level</td><td>Size</td><td>Average Word Count</td><td>Average Sentence Count</td></tr><tr><td>A2</td><td>64</td><td>60.87</td><td>11.53</td></tr><tr><td>B1</td><td>60</td><td>122.38</td><td>16.25</td></tr><tr><td>B2</td><td>71</td><td>265.35</td><td>37.03</td></tr><tr><td>C1</td><td>67</td><td>355.71</td><td>43.37</td></tr><tr><td>C2</td><td>69</td><td>333.86</td><td>38.41</td></tr></table>

Table 6: Statistics of the Cambridge Exams corpus (Xia et al., 2016) used as gold-standard dataset for the STAN-DARDIZE-L artifact and for training the CEFR classifier used in Section 7.

## A.3 Additional Information on Models and Inference

We set the minimum generated new tokens to 30 and the maximum to 300, as well as set the nucleus sampling decoding (top-p) to 0.95 as done with previous works on story generation (Imperial and Madabushi, 2023; DeLucia et al., 2021; See et al., 2019). The actual sizes of the open models range from 5GB to 15 GB max. We used a hosted GPU cloud with 4 NVIDIA Ti 3090 with 24GB memory size for model inference.

Llama2-Chat (Touvron et al., 2023b) is one of the community-recognized open instruction-tuned models released by Meta and an improved version of Llama 1 (Touvron et al., 2023a). For this task, we use the 7B version<sup>9</sup> finetuned from over a million human preference data and optimized for chat and dialogue use cases. We prioritized the addition of this model in our study for its accessibility to the general NLP community.

Longform-OPT (Köksal et al., 2023) is a recent instruction-tuned model optimized for long text generation using the LongForm dataset. For this study, we use the OPT model variant<sup>10</sup> (Zhang et al., 2022) with 2.7B parameters as this version obtained the best performance for the short story generation task using the WRITINGPROMPTS dataset (Fan et al., 2018) against other instructiontuned models such as Alpaca-LLaMA (Taori et al., 2023), FlanT5 (Chung et al., 2022), Tk-Instruct (Wang et al., 2022), and T0++ (Sanh et al., 2021).

OpenChat (Wang et al., 2023) is the most recent open model in our experiment setup, which currently is reported to be the best 7B model as of this writing and outperforms closed models such as ChatGPT (March) across a number of benchmark tasks such as GSM8K and TruthfulQA. In contrast to Llama and GPT models, which used RLHF (Ouyang et al., 2022), OpenChat is trained with mixed-quality data which is composed of high-quality expert data and sub-optimal web data with no preference labels. We use the 7B version<sup>11</sup> of this model variant released in January 2024.

GPT-4 (OpenAI, 2023) is the only closed model included in this study. We decide to add this model to our experiment for its global recognition through its easy-to-use interface among interdisciplinary fields, particularly in education (Kasneci et al., 2023). We use the version<sup>12</sup> finetuned with proprietary training data up to April 2023 with a 128K context window.

## A.4 Exemplars List

We list the actual list of literary exemplars used for the STANDARDIZE framework. We manually selected at most three classical exemplars as reference for the language models.

<table><tr><td>Level</td><td>Exemplars</td></tr><tr><td>A2</td><td>A Christmas Carol by Charles Dickens The Adventures Of Huckleberry Finn by Mark Twain The Little Prince by Antoine de Saint-Exupery</td></tr><tr><td>B1</td><td>Frankenstein by Mary Shelley Wuthering Heights by Emily Bronte Midsummer Night&#x27;s Dream by Shakespeare</td></tr><tr><td>B2</td><td>Moby Dick by Herman Melville Jane Eyre by Charlotte Bronte Sense and Sensibility by Jane Austen</td></tr><tr><td>C1</td><td>Animal Farm by George Orwell Anna Karenina by Leo Tolstoy Great Expectations by Charles Dickens</td></tr><tr><td>C2</td><td>Oliver Twist by Charles Dickens Crime and Punishment by Fyodor Dostoevsky Les Miserables by Victor Hugo</td></tr></table>

Table 7: The full exemplar list used for CEFR standards obtained from the Penguin Reader website (https: //www.penguinreaders.co.uk/).

<table><tr><td>Grade</td><td>Exemplars</td></tr><tr><td>Elementary</td><td>Little Women by Louisa May Alcott The Adventures of Tom Sawyer by Mark Twain The Road Not Taken by Robert Frost</td></tr><tr><td>Advanced</td><td>Jane Eyre by Charlotte Brontë The Great Gatsby by F. Scott Fitzgerald Fahrenheit 451 by Ray Bradbury</td></tr></table>

Table 8: The full exemplar list used for CCS standards obtained from the official website (https://www. thecorestandards.org/ELA-Literacy/).

## A.5 Mean Values of Linguistic Flags

We provide the computed averages of the linguistic flags from the aspects of the two standards, CEFR and CCS, used in this work reported in Tables 10 and 11.

## A.6 Additional Information on Human Expert Evaluation

We created and distributed the evaluation instrument through QuestionPro (https://www.qu estionpro.com/). In contrast to non-expert validation techniques where all instances are distributed automatically to available annotator platforms such as Amazon Turk, we use a representative random sample of our data for evaluation in consideration with the experts’ time constraints. For all tests, we randomly sampled 10% of the total generated narrative content using the bestperforming model, which is both the GPT-4 model with STANDARDIZE-⋆, for each corresponding task associated with CEFR and CCS as described in Section 6.

For grammaticality and coherence evaluation, we adapted the same four-point Likert scale from the work of DeLucia et al. (2021) for evaluating select model-generated content found through this link: https://github.com/JHU-CLSP/ gpt2-narrative-decoding/. Snapshots of the instruction and test instances presented to experts for evaluation can be viewed in Figures 10 and 11.

For the grade complexity distinction, we adapted a simpler select-one response type where for each test instance being evaluated, we select a random test instance from the adjacent next level of the target test instance and ask the experts to select which two examples of model-generated content are more simpler or complex. The idea here is that the expert should be able to tell the obviousness of the complexity of the test instance by indicating which is simpler or more complex. Snapshots of the instruction and test instances presented to experts for evaluation can be viewed in Figures 12 and 13.

Overall, our human evaluation design has been validated by the experts in language assessment we collaborated with through preliminary discussions on the scope, instrument, target outcomes, and presentation of the results from the task. As a form of compensation, we offered £30 upon completion of the entire task, which the experts took about approximately 30 45 minutes. The experts will also be acknowledged in this paper upon publication.

![](images/e06d5cdddd6c1353ae931ec70838e579766b69865b30e1fa9f44f166e202e55b.jpg)

![](images/9f8fefe9fac34f8379524c1cc954dc5311f507653876a19e3604d1bc6d989290.jpg)

Figure 6: Distribution of average sentence length between CEFR using (left) and CCS (right) using their best performing models, GPT-4 and Llama2, with STANDARDIZE-L.  
![](images/9afa896d671f2baf80ea8a0d6cc6688b74f85f033274a42a5b821d43cf2832cb.jpg)

![](images/c55e6cf5ed9c216d20b88f642f24a86822d2011405b789b1c85a796643c1f0ff.jpg)  
Figure 7: Distribution of average entity density between CEFR using (left) and CCS (right) using their best performing models, GPT-4 and Llama2, with STANDARDIZE-L.

![](images/d524aadeed38e4eb73beb4304def996201c81f7400a628f5b8b637248a8d17ea.jpg)

![](images/7a05e6734ee1a3a6e9a03d5cc339a4538cfaa092f3c6a856b2c277117aa1e5b2.jpg)  
Figure 8: Distribution of type token ratio between CEFR using (left) and CCS (right) using their best performing models, GPT-4 and Llama2, with STANDARDIZE-L.

<table><tr><td>Level</td><td>Meaning and Purpose</td><td>Organisation and Stucture</td><td>Grammatical Complexity</td></tr><tr><td>A2</td><td>The text is clear and concrete, aiming to describe appearance, places, routines, preferences, or tell a simple story.</td><td>The text is often short and observes chronological and predictable structure.</td><td>The text contains comparison of adjectives, rel- ative clauses, quantifiers, past simple of to be and full verbs, passive voice of present and past simple.</td></tr><tr><td>B1</td><td>The text is clear and concrete, aiming to describe appearance, places, routines, preferences, or tell a simple story. The text may also provide opinions and instructions or explanations, easy to understand and visualise, excluding ambiguity and diverse in-</td><td>The text can be long but not complex, and observes mostly chronological with unex- pected changes of direction, digressions or flashbacks.</td><td>The text contains future forms, future in the past, &#x27;used to&#x27; about repeated actions, present perfect simple, clauses for purpose and con- trast, reporting statements, tag questions.</td></tr><tr><td>B2</td><td>The text provides opinions and instruc- tions/explanations, easy to understand and visualise, excluding ambiguity and diverse in- terpretations. The text also gives description, classification, argumentation or a combination of these, allowing greater ambiguity and various interpretations.</td><td>The text can be long but not complex, and observes chronological or spatial with possible statement of various aspects of a phenomenon.</td><td>The text contains past continuous, past per- fect, passive voice of perfect and continuous, &#x27;would&#x27; about habits, reporting questions, in- finitives and -ing forms.</td></tr><tr><td>C1</td><td>The text may serve different purposes and may be combined with multiple levels of meaning. The descriptions and instructions in the text are detailed and may be hard to visualise.</td><td>The text is often lengthy, complex, and observes logical organisation, starting with a claim followed by reasons, proving it, or changing view-points.</td><td>The text contains compound adjectives, condi- tional sentences, inversion, future perfect, cleft and non-finite clauses, modals about the past.</td></tr><tr><td>C2</td><td>The text may serve different purposes and may be combined with multiple levels of meaning. The text may also show exploration of hypotheses, causes and effects, etc. The details of the text are complex to follow and visualise.</td><td>The text is often lengthy, complex, and observes presentation which may start with the ending/final result and go back to the possible causes.</td><td>The text contains combination of multiple ad- jectives, inversion with hardly and only when, comment clauses, non-finite perfect clauses, ellipsis, passive impersonal constructions.</td></tr><tr><td>Linguistic Flags</td><td>Automatic Readability Formula, Type Token Ratio (2)</td><td>Total and average sentence and word lengths, Subordinating and coordinating coniunctions (4)</td><td>Age-of-Acquisition and USubtlex densities, entity density per sentence (3)</td></tr></table>

(a) The specifications provided by the Common European Framework of Reference for Languages (CEFR) cover aspects of meaning, organization, and grammatical complexity for all levels.
<table><tr><td>Aspects</td><td>Qualitative (Meaning)</td><td>Qualitative (Syntax)</td><td>Quantitative (Length)</td></tr><tr><td>Description</td><td>The text can range from containing a sin- gle level of meaning to multiple levels of meaning based on complexity.</td><td>A text with low complexity tends to have simple, well-marked, and conventional structures, whereas a text of high complexity tends to have complex, im- plicit, and unconventional structures. Simple texts tend to relate events in chronological order, while complex texts make more frequent use of flashbacks, flash-forwards, and other manipulations of time and</td><td>That text that has longer words and longer sentences are more difficult to read than shorter ones. A text with many long words and/or sentences is thus rated by these formulas as harder to read than a text with many short words and/or sen- tences would be.</td></tr><tr><td>Linguistic Flags</td><td>Entity densities per sentence, Total proper noun density (2)</td><td>Type Token Ratio, Subordinating and coordinating conjunctions (3)</td><td>Total and average sentence and word lengths (3)</td></tr></table>

(b) The specifications of the Common Core Standards (CCS) cover qualitative and quantitative aspects. Unlike the CEFR, the CCS’s model does not require categorization per level.

Table 9: The full content of the CEFR and CCS standards with corresponding manually selected representative linguistic flags for each aspect.
<table><tr><td>Aspect</td><td>Linguistic Flag</td><td>A2</td><td>B1</td><td>B2</td><td>C1</td><td>C2</td></tr><tr><td rowspan="3">Meaning and Purpose</td><td>average_entities_per_sentence</td><td>0.92</td><td>0.93</td><td>0.68</td><td>0.7</td><td>0.5</td></tr><tr><td>average_AoA_per_sentence</td><td>51.4</td><td>76.7</td><td>82.6</td><td>94.4</td><td>109.9</td></tr><tr><td>average_USubtlex_per_sentence</td><td>69.7</td><td>93.1</td><td>95.5</td><td>101.2</td><td>115.8</td></tr><tr><td rowspan="4">Organization and Structure</td><td>total_word_count</td><td>60.8</td><td>122.3</td><td>265.3</td><td>355.7</td><td>333.8</td></tr><tr><td>total_sentence_count</td><td>11.5</td><td>16.2</td><td>37.0</td><td>43.3</td><td>38.4</td></tr><tr><td>average_sentence_length</td><td>5.3</td><td>7.5</td><td>7.4</td><td>8.7</td><td>9.3</td></tr><tr><td>total_conjunctions_count</td><td>3.6</td><td>5.3</td><td>11.2</td><td>11.9</td><td>13.0</td></tr><tr><td rowspan="2">Grammaticality Complexity</td><td>ARI_formula_readability</td><td>7.1</td><td>10.6</td><td>11.2</td><td>13.4</td><td>14.4</td></tr><tr><td>correlated_type_token_ratio</td><td>7.8</td><td>9.5</td><td>12.1</td><td>13.2</td><td>13.5</td></tr></table>

Table 10: The average values of linguistic flags for each level in the CEFR standard.

<table><tr><td>Aspect</td><td>Linguistic Flag</td><td>Elementary</td><td>Advanced</td></tr><tr><td>Qualitative (Meaning)</td><td>average_entities_per_sentence</td><td>0.6</td><td>0.8</td></tr><tr><td></td><td>average_proper_nouns</td><td>7.3</td><td>15.5</td></tr><tr><td>Qualitative (Syntax)</td><td>average_coordinating_conjunction</td><td>2.5</td><td>3.1</td></tr><tr><td></td><td>average_subordinating_conjunction</td><td>6.5</td><td>14.9</td></tr><tr><td></td><td>correlated_type_token_ratio</td><td>9.1</td><td>11.4</td></tr><tr><td></td><td>total_word_count</td><td>141.2</td><td>255.2</td></tr><tr><td>Quantitative (Length)</td><td>total_sentence_count</td><td>24.9</td><td>31.0</td></tr><tr><td></td><td>average_sentence_length</td><td>6.2</td><td>9.6</td></tr></table>

Table 11: The average values of linguistic flags for each level in the CCS standard.

![](images/8ce5c292a03a23050277ef10c8aacbef8b92b9ab8b5dd8061109160c06575f63.jpg)  
Table 12: Sample generations with the teacher style method and variations of the STANDARDIZE framework using the best model (GPT-4) for the context-assisted story generation observing CEFR standards. Some examples are truncated for brevity.

![](images/cea8857af6a49281434a20458b5e88db23813961132f8463b11349dd5062992d.jpg)  
Table 13: Sample generations with the teacher style method and variations of the STANDARDIZE framework using the best model (Llama2) for the theme word story generation observing CCS standards. Some examples are truncated for brevity.

![](images/f8aeba1da36f1f35e815c577ca89a60fa70f594d6e594bada67341e885eb4b0b.jpg)  
Figure 9: Landing page of the QuestionPro platform used for collecting expert evaluations.

![](images/e805273bf3d6e1c5c8c5b4324b657a020e3301c992bbec0fed6aaba6c39a578b.jpg)  
Figure 10: Instructions presented to expert evaluators for assessing the grammaticality or fluency and coherence of model-generated content for CEFR and CCS through QuestionPro. The setup is derived from DeLucia et al. (2021).

![](images/978a9dede9dbdeec4997f54c351b09a143c569b9ea2f011e14008db244f8ac4a.jpg)  
Figure 11: An example of randomly selected generated content presented to expert evaluators to assess grammaticality or fluency and coherence. The example is truncated for brevity.

![](images/036f1f9253bd22aec6d78d1dcd9202c965604ea7c93465109f59d70c596a65cc.jpg)  
Figure 12: Instructions presented to expert evaluators for assessing the grade complexity distinction of modelgenerated content for CEFR and CCS through QuestionPro.

![](images/a94282c346d0f7b8fced772f96ddc5d575e7d32d7376a3b37e7e3653205d2877.jpg)  
Figure 13: An example of two instances of generated content presented to expert evaluators to assess which one is more simpler or more complex denoting obviousness in their grade complexity. The example is truncated for brevity.