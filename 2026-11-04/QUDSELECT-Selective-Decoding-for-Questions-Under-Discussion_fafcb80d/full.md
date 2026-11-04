# QUDSELECT: Selective Decoding for Questions Under Discussion Parsing

a Suvarna♡∗ Xiao Liu♢∗ Tanmay Parekh♡ Kai-Wei Chang♡ Nanyun Peng♡ ♡ Computer Science Department, University of California, Los Angeles ♢ Wangxuan Institute of Computer Technology, Peking University   
{asuvarna31,tparekh,kwchang,violetpeng}@cs.ucla.edu lxlisa@pku.edu.cn

## Abstract

Question Under Discussion (QUD) is a discourse framework that uses implicit questions to reveal discourse relationships between sentences. In QUD parsing, each sentence is viewed as an answer to a question triggered by an anchor sentence in prior context. The resulting QUD structure is required to conform to several theoretical criteria like answer compatibility (how well the question is answered), making QUD parsing a challenging task. Previous works construct QUD parsers in a pipelined manner (i.e. detect the trigger sentence in context and then generate the question). However, these parsers lack a holistic view of the task and can hardly satisfy all the criteria. In this work, we introduce QUDSELECT, a joint-training framework that selectively decodes the QUD dependency structures considering the QUD criteria. Using instruction-tuning, we train models to simultaneously predict the anchor sentence and generate the associated question. To explicitly incorporate the criteria, we adopt a selective decoding strategy of sampling multiple QUD candidates during inference, followed by selecting the best one with criteria scorers. Our method outperforms the state-of-the-art baseline models by 9% in human evaluation and 4% in automatic evaluation, demonstrating the effectiveness of our framework. Code and data are in https://github.com/ asuvarna31/qudselect.

## 1 Introduction

Discourse structure describes the relationships between different sentences of an article or conversation. The ability to understand discourse structure is crucial for natural language processing tasks such as text summarization (Durrett et al., 2016), conditional generation (Narayan et al., 2023), and narrative understanding (Xu et al., 2024). Recent works have adapted the Question Under Discussion (QUD) framework to analyze discourse structures (Benz and Jasinskaja, 2017; Riester et al., 2021). In the QUD framework (Van Kuppevelt, 1995; Roberts, 2012), the relationships between sentences in an article are characterized by (implicit) free-form questions. Each question is evoked by an anchor sentence in prior context, and answered by an answer sentence in the subsequent content. For instance, in Figure 1, the relationship between sentence 3 (referred to as s<sub>3</sub>) and the previous context is that s<sub>3</sub> answers the question “Which movie has the most Oscar nominations?” evoked by the anchor sentence s<sub>1</sub>.

![](images/f22e565993ef51d7c31322f37eeb2a27a09ff66cc08e7d7489f5a099cc414110.jpg)  
Figure 1: An article snippet along with the associated QUD dependency structure. Each edge from $s _ { i }$ to s<sub>j</sub> with attribute q indicates sentence $s _ { j }$ anchors the question q, and sentence s<sub>i</sub> answers the question q.

The QUD structures involve contextuallygrounded questions that adhere to three theoretical criteria (De Kuthy et al., 2018; Wu et al., 2023; Riester et al., 2018): a) answer compatibility: the question must be answerable by the answer sentence in the discourse, like s<sub>2</sub> directly answers the question “Who starred in Forrest Gump?” in Figure 1; b) givenness: the question should only contain concepts that are accessible to the reader from prior context or common knowledge, like “Forrest Gump” in the question; and c) anchor relevance: the question should be relevant to the anchor sentence, e.g., the aforementioned question can be triggered in s<sub>1</sub>.

![](images/334a4aa74a42548326a9489e1023e3f7438d4c368a11ee2c97f3c8ddf9eb8b51.jpg)  
Figure 2: Overview of our QUDSELECT framework.

Previous works on QUD parsing break down the task into two steps: anchor selection and question generation. De Kuthy et al. (2020) develop a rulebased method for the question generation step, Ko et al. (2023) train task-specific models for each step, while Wu et al. (2023) prompt large language models (LLMs) in a stepwise manner. However, these approaches lack a holistic view of the task, causing the predicted QUDs to often fail to satisfy all the criteria. For instance, GPT-4 fails to generate questions that are fully grounded on the anchor sentence in 50% of the cases.<sup>1</sup>

To address these challenges, we propose QUD-SELECT, a joint-training framework that selectively decodes QUD structures by incorporating the criteria, as shown in Figure 2. Specifically, we instruction-tune models to jointly predict the anchor sentence and the corresponding question given an answer sentence (e.g., s<sub>13</sub>) and prior context (e.g., s<sub>1</sub>, . . . , s<sub>12</sub> of the article). We propose selective decoding where we sample multiple anchor and question pairs, score them using criteria scorers, and finally, select the best scored pair.

Experiments conducted on the DCQA (Ko et al., 2022) dataset show that QUDSELECT outperforms baselines by \~9% on average in human evaluation. To reduce resource and cost-intensive expert evaluation, we develop automatic evaluators trained on human annotations, and conduct a larger-scale automatic evaluation. The automatic evaluation results show that QUDSELECT achieves around a \~4% improvement over the selected baselines. Further analyses reveal that the performance could be further improved with more selected candidates.

## 2 Related Work

QUD is a linguistic framework that analyzes discourse and pragmatics by viewing each sentence as an answer to an implicit question triggered in prior context (Van Kuppevelt, 1995; Roberts, 2012; Benz and Jasinskaja, 2017). While theoretical discussions around QUDs relied on constructed examples, Riester (2019) introduced an annotation framework for reconstructing QUDs from data. Westera et al. (2020), Ko et al. (2022) and Hesse et al. (2020) annotated Ted-talk transcripts and news articles respectively in an expectation-driven manner, where questions are triggered while reading (i.e., unseen discourse progression) while De Kuthy et al. (2018) annotated two interview transcripts with full, hierarchical questions.

Recent works have begun adapting QUD for automatic discourse parsing (Ko et al., 2022, 2023; Wu et al., 2023), narrative graph construction (Xu et al., 2024) and decontextualization of scientific documents (Newman et al., 2023). QUD fits well for understanding the structure and coherence of texts that are intended to provide argumentation (Liu et al., 2024) and complex reasoning (Hu et al., 2022), and has potential applications to enhance document understanding in information extraction (Parekh et al., 2023, 2024a; Huang et al., 2024) with applications in wider domains like epidemiology (Parekh et al., 2024b) and biomedical science (Ma et al., 2023). Ko et al. (2023) introduced a QUD parser trained on DCQA (Ko et al., 2022) that consists of an anchor selection and a question generation pipeline. Wu et al. (2023) evaluated QUDs generated by LLMs by few-shot prompting in a two-step manner: question generation followed by anchor generation. Xu et al. (2024) followed a QUD style annotation for generating narrative graphs by incorporating retrospective questions triggered from succeeding context.

<table><tr><td rowspan="2">Model</td><td colspan="3">Answer Compatibility</td><td colspan="3">Givenness</td><td colspan="3">Anchor Relevance</td><td rowspan="2">Avg. (↑)</td></tr><tr><td>Dir. (↑) Unfocus. No Ans.(↓)</td><td></td><td></td><td>No New (↑)</td><td>Ans. leak.  $( \downarrow )$ </td><td>Hall. (↓)</td><td>Fully G. (↑) Partial. G.</td><td></td><td> $\mathbf { N o G } . \left( \downarrow \right)$ </td></tr><tr><td colspan="10">AUTOMATIC EVALUATION</td></tr><tr><td>Pipeline</td><td>68.2</td><td>4.5</td><td>27.3</td><td>83.7</td><td>10.0</td><td>6.3</td><td>63.6</td><td>0.0</td><td>36.4</td><td>71.8</td></tr><tr><td>LLaMA2-7B</td><td>67.4</td><td>12.9</td><td>19.7</td><td>88.3</td><td>6.7</td><td>5.0</td><td>52.7</td><td>17.7</td><td>29.6</td><td>69.5</td></tr><tr><td>+ QUDSELECT</td><td>70.4</td><td>8.2</td><td>21.4</td><td>91.8</td><td>6.0</td><td>2.2</td><td>61.0</td><td>12.4</td><td>26.6</td><td>74.4</td></tr><tr><td>Mistral-7B</td><td>71.4</td><td>8.7</td><td>19.9</td><td>89.3</td><td>6.0</td><td>4.7</td><td>58.0</td><td>15.9</td><td>26.1</td><td>72.9</td></tr><tr><td>+ QUDSELECT</td><td>74.1</td><td>9.0</td><td>16.9</td><td>86.5</td><td>7.2</td><td>6.2</td><td>68.3</td><td>11.0</td><td>20.7</td><td>76.3</td></tr><tr><td>GPT-4 + QUDSELECT</td><td>92.7 90.0</td><td>3.3 4.1</td><td>4.0 5.9</td><td>78.7 80.0</td><td>18.9 15.0</td><td>2.4 5.0</td><td>51.9 62.5</td><td>32.0 21.4</td><td>16.1 16.0</td><td>74.4 77.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">HUMAN EVALUATION</td></tr><tr><td>Pipeline</td><td>52.5</td><td>15.0</td><td>32.5</td><td>53.8</td><td>28.7</td><td>17.5</td><td>50.0</td><td>32.5</td><td>17.5</td><td>52.1</td></tr><tr><td>Mistral-7B</td><td>67.0 67.1</td><td>15.4</td><td>17.6 12.9</td><td>60.3</td><td>23.6</td><td>16.1 2.4</td><td>58.6</td><td>29.0</td><td>12.4</td><td>62.0</td></tr><tr><td>+ QUDSELECT</td><td></td><td>20.0</td><td></td><td>77.6</td><td>20.0</td><td></td><td>68.2</td><td>24.7</td><td>7.1</td><td>71.0</td></tr></table>

Table 1: Automatic and human evaluation results. Numbers are in percentages (%). Best results are in bold, and the best results of open-source models (if not the best overall) are underlined. Avg. indicates the average ratio of ideal QUDs (the first option of each criterion). We abbreviate Direct Answer as Dir. Ans., Indirect Answer as Indir. Ans., Answer Leakeage as Ans. Leak., Hallucination as Hall., and Grounded as G.

## 3 The QUDSELECT Framework

Task Formulation Given a document with n sentences $\textit { D } = \ \{ s _ { 1 } , s _ { 2 } , . . . , s _ { n } \}$ , QUD parsing aims to build a QUD dependency tree. We formulate the QUD parsing task as edge-level prediction following previous works (De Kuthy et al., 2018; Ko et al., 2023): given an answer sentence $s _ { i } \in \{ s _ { 2 } , \ldots , s _ { n } \} ^ { 2 }$ , models are asked to predict the anchor sentence $a _ { i } \in \{ s _ { 1 } , . . . , s _ { i - 1 } \}$ and generate the question $q _ { i }$

Overview Figure 2 illustrates the structure of our QUDSELECT framework. We first instruction tune a joint QUD parser §3.1. Then, we propose selective decoding §3.2 to select the best candidate from sampled anchor sentence, question pairs.

## 3.1 QUD Parser Training

Unlike previous works that use separate models for anchor prediction and question generation, we exploit the instruction following ability of LLMs (Wang et al., 2022) to perform these two steps jointly, as demonstrated in Figure 2 (left). This joint inference provides the model with a holistic view of the task. Given the answer sentence $s _ { i }$ and context of sentences prior to $s _ { i }$ , models are instructed to output the anchor $a _ { i }$ and the question $q _ { i }$ . We provide the instruction-response template in Appendix A.

## 3.2 Selective Decoding

To incorporate specific criteria during inference, we sample multiple anchor sentence, question candidates and select the best one by using simple criteria scorers.

To generate multiple QUD candidates for a context $\{ s _ { 1 } , \dotsc , s _ { i - 1 } \}$ and an answer sentence $s _ { i } .$ we sample multiple anchor sentences and question candidates by selectively utilizing beam-search with a wide beam while decoding. Following prior work (De Kuthy et al., 2018; Benz and Jasinskaja, 2017; Wu et al., 2023), we assume that every answer sentence has a corresponding question. First, for anchor prediction, we prompt the model with sentence $s _ { i }$ is anchored by sentence using a beam size k to generate k possible anchors. Post deduplication of anchor candidates, we again utilize beam-search with size k to generate k question candidates for each anchor sentence. This encourages diversity in both the prediction of anchor sentences and questions.

We apply m criteria $\mathcal { C } = \{ c _ { 1 } , \ldots , c _ { m } \}$ to assess the quality of generated candidates from different aspects. Each criterion assigns a score $c _ { j } ( a , q ) \in$ [0, 1] to a candidate $\langle a , q \rangle$ , and the overall score is the summation of all criteria $\Sigma _ { j = 1 } ^ { m } ( c _ { j } ( a , q ) )$ . The candidate with the highest overall score is selected as the final prediction.

Criteria Scorers. We consider the three key principles of QUD as our criteria: answer-compatibility, givenness, and anchor relevance. We implement reference-free and training-free scorers for each of them.

Answer Compatibility: This criterion indicates that the question q should be answerable by the answer sentence $s _ { i } .$ . We regard this as a natural language inference (NLI) task, and use the probability that $s _ { i }$ entails q measured by an off-the-shelf NLI model (bart-large-mnli) as the compatibility score.

Givenness: This criterion evaluates if the question only consists of information from the context. An ideal question should be naturally invoked from the context, without concepts that appear out of thin air. We measure the givenness with content word overlap between $q$ and the context s<sub>1...i 1</sub>. We extract lemmas $L _ { q }$ and $L _ { c }$ of all content words (nouns, verbs, adjectives, and adverbs) in the question and the context, and compute the givenness score as $| L _ { q } \cap L _ { c } | / | L _ { q } |$

Anchor Relevance: This criterion measures if the question q is relevant to the anchor sentence a. Similar to the givenness score, we approximate it with content word overlap between a and the focus of $q .$ We regard the maximum noun phrase of $q$ as its focus $f _ { q } ,$ , and extract lemmas $L _ { f q }$ and $L _ { a }$ of all content words in $f _ { q }$ and $a .$ . The relevance score is computed as $| L _ { f q } \cap L _ { a } | / | L _ { f q } |$

## 4 Experimental Setup

Models and Datasets We utilize the DCQA dataset (Ko et al., 2022) for training and evaluating QUD parsers. The DCQA dataset consists of 22k English questions across 606 news articles. We use two instruction-tuned models LLaMA2-7B (Touvron et al., 2023) and Mistral-7B (Jiang et al., 2023) as base models of our framework. To explore the effectiveness of selective decoding on closedsource models, we also apply it to GPT-4 (Achiam et al., 2023). We sample k = 10 candidates for each answer sentence. Implementation details can be found in Appendix A.

Baselines We compare against two existing QUD parsers: the Pipeline approach (Ko et al., 2023) and GPT-4 prompting (Wu et al., 2023). We also provide ablation of not using selective decoding during inference, i.e., QUDSELECT with k = 1.

Human Evaluation We follow the annotation guidelines outlined in QUDEVAL (Wu et al., 2023) and evaluate the generated QUDs for answer compatibility, givenness, and anchor relevance. Detailed classification of the criteria is in Appendix B. We evaluate 100 questions across 8 articles from the DCQA test set. We recruit three annotators from Amazon’s Mechanical Turk after extensive training and screening. We report the majority vote results and achieve an average inter-annotator agreement of 68.3% averaged across all evaluated dimensions. More details are in Appendix C.

Automatic Evaluation While human evaluation is more accurate for evaluating the efficacy of QUD parsing models, it is time-consuming and expensive to collect at scale. To this end, we apply supervised classifiers to judge the generated QUDs. Specifically, we train RoBERTa classifiers (Liu et al., 2019) on the expert annotated data in QUDEVAL for answer compatibility and anchor relevance, and Longformer (Beltagy et al., 2020) for givenness due to the longer context length. We achieve a macro F1 score of 0.48 for answer compatibility, 0.42 for givenness, and 0.53 for anchor relevance, outperforming or matching the best existing automatic evaluators. Detailed comparisons with other evaluators are in Appendix D. We conduct the automatic evaluation on on 400 questions per model across 22 articles from the entire DCQA test set.

## 5 Results and Analysis

## 5.1 Main Results

Automatic Evaluation Results. Table 1 (top) reports the automatic evaluation results. QUD-SELECT (Mistral-7B) outperforms the previously established pipeline baseline on all the three criteria. And QUDSELECT improves the performance of instruction tuned Mistral-7B, LLaMA2-7B and GPT-4, leading to  4% improvement over models without QUDSELECT.

Human Evaluation Results Table 1 (bottom) reports the human evaluation results. We compare the best open-source model from Table 1, QUDS-ELECT (Mistral-7B), with Pipeline and Mistral-7B. QUDSELECT (Mistral-7B) generates 67% directly answered questions, 78% questions with no unseen concepts, and 68% fully grounded questions. This highlights the effectiveness of our framework in generating QUDs that satisfy the desired criteria.

Error Analysis Our detailed classifications of the evaluation metrics (Appendix §B) allow us to categorize the various errors made by the models. We find from Table 1 that GPT-4 generates higher percentage of directly answered QUDs but these QUDs are more likely to have answer leakage errors. This indicates that GPT-4 tends to include aspects from the answer sentence in the question that increases the answer compatibility but reduces the givenness. We find that QUDSELECT improves GPT-4 performance by reducing the answer leakage error and improving the relevance of the anchor. Overall, we find that QUDSELECT improves the validity of the answers and increases the grounding of the questions in the anchor which leads to performance improvements for all models.

<table><tr><td colspan="2">QUDSELECT (Mistral)</td></tr><tr><td>Answer:  $s _ { 3 }$  Anchor:  $s _ { 1 }$  of nuclear material cannot be adequately traced from country to  $\mathrm { c o u n t r y ? } ^ { \prime }$ </td><td>QUD: &quot;Why is it important that U.S. exports√Direct answer √No new concepts √Fully grounded</td></tr><tr><td>Answer:  $s _ { 4 }$  Anchor:  $s _ { 2 }$  QUD: “Who commissioned the report?&quot;</td><td>√Direct answer √No new concepts √Fully grounded</td></tr><tr><td>Pipeline (Ko et al. (2023))</td><td></td></tr><tr><td>Answer:  $s _ { 3 }$  Anchor:  $s _ { 2 } \mathrm { Q U D } \colon$  outlook on nuclear materials?&quot;</td><td>&quot;What does Glenn think is the future XNon answer XAnswer leakage √Partially grounded</td></tr><tr><td>Answer:  $s _ { 4 }$  Anchor:  $s _ { 2 }$  QUD: &quot;Who is the Sen. Glenn from?&quot;</td><td>XNonsensical question</td></tr></table>

Table 2: Example QUDs generated by QUDSELECT (Mistral) and the pipeline method for a test article. The full article text can be found in Appendix Figure 5. $s _ { i }$ indicates the i-th sentence in the article.

![](images/0249dbac93b473727c4a38d47d4d8014775e7839620fcb1deb7dae6f1631d6aa.jpg)  
Figure 3: Hyperparameter analysis on the number of candidates. QUDSELECT shows improved performance with an increased number of candidates.

## 5.2 Hyperparameter Study

To study the performance sensitivity of QUDSE-LECT to the number of candidates k, we vary k from 1 to 20 for QUDSELECT (LLaMA2-7B) and QUDSELECT (Mistral-7B) and show the performance in Figure 3. The performance reveals an upward trend as k grows for Answer Compatibility and Anchor Relevance while Givenness is sacrificed by a small margin for better overall performance. With k = 10, QUDSELECT significantly outperforms the selected baselines without significant runtime overhead.

## 5.3 Case Study

In Table 2, we show the QUDs generated by QUD-SELECT (Mistral-7B) and the Pipeline model for a news article (Appendix Figure 5) along with the human annotations for each question. Most QUDs generated by QUDSELECT (Mistral-7B) are explicitly answerable, include no unseen concepts, and are fully grounded in the anchor. In contrast, the Pipeline method generates incomplete questions or incompatible question-answer pairs for the given article. This demonstrates the overall effectiveness of QUDSELECT in generating high-quality QUDs.

## 6 Conclusion

In this work, we propose QUDSELECT, a joint framework for generating QUD structures by integrating key theoretical criteria. To achieve this, we reformulate the QUD parsing as an instruction tuning task and selectively decode the candidate questions and anchors. Furthermore, we develop automated evaluation methods trained on expert annotations to reduce the reliance on labor-intensive expert evaluations and facilitate model development for QUD parsing. Experiments demonstrate that QUDSELECT significantly outperforms baselines in both automatic and human evaluations.

## Acknowledgements

We thank Hritik Bansal and Sidi Lu for their constructive comments. We thank the anonymous reviewers for their helpful discussions and suggestions. Our work was supported by Optum Labs, Amazon Alexa AI Research Award, an Amazon Research Award via UCLA Science Hub and the Amazon Fellowship (Tanmay Parekh) and we express our gratitude for their support.

## Limitation

QUDSELECT generates the QUD structure as a dependency tree where each sentence is connected to a prior context via a question. This does not guarantee the generation of full, hierarchical QUDs where the answer of a QUD entails the answer of its descendants (Roberts, 2012). Furthermore, QUDSE-LECT generates each QUD edge independently and does not model the relationships between questions. Thus, we leave the exploration of such discourse level constraints to future work.

Sampling Cost. Although the time cost increases when sampling more candidates for QUD-SELECT, the number of sampled unique anchors does not increase, due to the limited number of reasonable anchors in an article. The average number of unique anchors is less than 3 when k = 20. Therefore, the growth of sampling cost is approximately linear to k. We find that increasing the number of candidates leads to an increase in the model performance §5.2.

## Ethical Consideration

Our framework relies on open-source and closedsource LLMs that may generate harmful and biased outputs. Therefore, it should be used with human supervision. For human evaluation, we recruit annotators from Amazon’s Mechanical Turk, and all annotators are fairly paid more than \$15 USD per hour (which varies depending on the time spent per HIT), which is higher than the national minimum wage where the annotators are recruited.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Ron Artstein and Massimo Poesio. 2008. Inter-coder agreement for computational linguistics. Computa tional linguistics, 34(4):555–596.

Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.

Anton Benz and Katja Jasinskaja. 2017. Questions under discussion: From sentence to discourse.

Kordula De Kuthy, Madeeswaran Kannan, Haemanth Santhi Ponnusamy, and Detmar Meurers. 2020. Towards automatically generating questions under discussion to link information and discourse structure. In Proceedings of the 28th International Conference on Computational Linguistics, pages 5786–5798.

Kordula De Kuthy, Nils Reiter, and Arndt Riester. 2018. Qud-based annotation of discourse structure and information structure: Tool and evaluation. In Pro-

ceedings ofthe Eleventh International Conference on Language Resources and Evaluation (LREC 2018).

Greg Durrett, Taylor Berg-Kirkpatrick, and Dan Klein. 2016. Learning-based single-document summarization with compression and anaphoricity constraints. arXiv preprint arXiv:1603.08887.

Christoph Hesse, Anton Benz, Maurice Langner, Felix Theodor, and Ralf Klabunde. 2020. Annotating quds for generating pragmatically rich texts. In Proceedings of the Workshop on Discourse Theories for Text Planning, pages 10–16.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Nan Hu, Zirui Wu, Yuxuan Lai, Xiao Liu, and Yansong Feng. 2022. Dual-channel evidence fusion for fact verification over texts and tables. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5232–5242.

Kuan-Hao Huang, I-Hung Hsu, Tanmay Parekh, Zhiyu Xie, Zixuan Zhang, Prem Natarajan, Kai-Wei Chang, Nanyun Peng, and Heng Ji. 2024. TextEE: Benchmark, reevaluation, reflections, and future challenges in event extraction. In Findings ofthe Associationfor Computational Linguistics ACL 2024, pages 12804– 12825, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.

AQ Jiang, A Sablayrolles, A Mensch, C Bamford, DS Chaplot, D de las Casas, F Bressand, G Lengyel, G Lample, L Saulnier, et al. 2023. Mistral 7b (2023). arXiv preprint arXiv:2310.06825.

Wei-Jen Ko, Cutter Dalton, Mark Simmons, Eliza Fisher, Greg Durrett, and Junyi Jessy Li. 2022. Discourse comprehension: A question answering framework to represent sentence connections. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 11752–11764, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Wei-Jen Ko, Yating Wu, Cutter Dalton, Dananjay Srinivas, Greg Durrett, and Junyi Jessy Li. 2023. Discourse analysis via questions and answers: Parsing dependency structures of questions under discussion. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 11181–11195, Toronto, Canada. Association for Computational Linguistics.

Xiao Liu, Yansong Feng, and Kai-Wei Chang. 2024. Casa: Causality-driven argument sufficiency assessment. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5282–5302.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Mingyu Derek Ma, Alexander Taylor, Wei Wang, and Nanyun Peng. 2023. DICE: Data-efficient clinical event extraction with generative models. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15898–15917, Toronto, Canada. Association for Computational Linguistics.

Aman Madaan, Amrith Setlur, Tanmay Parekh, Barnabas Poczos, Graham Neubig, Yiming Yang, Ruslan Salakhutdinov, Alan W Black, and Shrimai Prabhumoye. 2020. Politeness transfer: A tag and generate approach. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1869–1881, Online. Association for Computational Linguistics.

Shashi Narayan, Joshua Maynez, Reinald Kim Amplayo, Kuzman Ganchev, Annie Louis, Fantine Huot, Anders Sandholm, Dipanjan Das, and Mirella Lapata. 2023. Conditional generation with a questionanswering blueprint. Transactions ofthe Association for Computational Linguistics, 11:974–996.

Benjamin Newman, Luca Soldaini, Raymond Fok, Arman Cohan, and Kyle Lo. 2023. A question answering framework for decontextualizing user-facing snippets from scientific documents. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3194–3212.

Tanmay Parekh, I-Hung Hsu, Kuan-Hao Huang, Kai-Wei Chang, and Nanyun Peng. 2023. GENEVA: Benchmarking generalizability for event argument extraction with hundreds of event types and argument roles. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3664–3686, Toronto, Canada. Association for Computational Linguistics.

Tanmay Parekh, I-Hung Hsu, Kuan-Hao Huang, Kai-Wei Chang, and Nanyun Peng. 2024a. Contextual label projection for cross-lingual structured prediction. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5738–5757, Mexico City, Mexico. Association for Computational Linguistics.

Tanmay Parekh, Anh Mac, Jiarui Yu, Yuxuan Dong, Syed Shahriar, Bonnie Liu, Eric Yang, Kuan-Hao Huang, Wei Wang, Nanyun Peng, and Kai-Wei Chang. 2024b. Event detection from social media for epidemic prediction. In Proceedings of the 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5758–5783, Mexico City, Mexico. Association for Computational Linguistics.

Arndt Riester. 2019. Constructing qud trees. In Questions in discourse, pages 164–193. Brill.

Arndt Riester, Lisa Brunetti, and Kordula De Kuthy. 2018. Annotation guidelines for questions under discussion and information structure. Information structure in lesser-described languages: Studies in prosody and syntax, pages 403–443.

Arndt Riester, Amalia Canes Nápoles, and Jet Hoek. 2021. Combined discourse representations: Coherence relations and questions under discussion. In Proceedings of the First Workshop on Integrating Perspectives on Discourse Annotation, pages 26–30.

Craige Roberts. 2012. Information structure: Towards an integrated formal theory of pragmatics. Semantics and pragmatics, 5:6–1.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models.

Jan Van Kuppevelt. 1995. Discourse structure, topicality and questioning. Journal oflinguistics, 31(1):109– 147.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. 2022. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. arXiv preprint arXiv:2204.07705.

Matthijs Westera, Laia Mayol, and Hannah Rohde. 2020. TED-Q: TED talks and the questions they evoke. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 1118–1127, Marseille, France. European Language Resources Association.

Yating Wu, Ritika Mangla, Greg Durrett, and Junyi Jessy Li. 2023. QUDeval: The evaluation of questions under discussion discourse parsing. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5344– 5363, Singapore. Association for Computational Linguistics.

Liyan Xu, Jiangnan Li, Mo Yu, and Jie Zhou. 2024. Graph representation of narrative context: Coherence dependency via retrospective questions. arXiv preprint arXiv:2402.13551.

Da Yin, Xiao Liu, Fan Yin, Ming Zhong, Hritik Bansal, Jiawei Han, and Kai-Wei Chang. 2023. Dynosaur: A dynamic growth paradigm for instruction-tuning data curation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4031–4047.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations.

## A QUDSELECT Implementation Details

We instruction-tune QUD parsers in the format of Figure 4. Similar to Yin et al. (2023), we apply LORA (low-rank adaptation, Hu et al. (2021)) with learning rate 2e 5, $l o r a _ { r a n k } = 2 5 6 .$ , and $l o r a _ { a l p h a } = 2 5 6$ . Models are trained for 2 epochs with batch size 128. During inference, we sample QUD candidates with k beams and temperature 1. All the experiments are performed with 48GB NVIDIA A6000 GPUs.

### Instruction: Given the answer sentence, reason through the   
context to find the most likely sentence where a question can be   
generated.   
### Input:   
Context: {context}   
Answer sentence: {Answer}   
### Response: Sentence {Answer ID} is anchored by sentence   
{Anchor ID}, answering the question of “{Question}".  
Figure 4: Prompt format for instruction tuning QUD parsers.

## B Evaluation Criteria Details

We follow the evaluation protocol outlined in (Wu et al., 2023) for our human and automatic evaluation.

• Answer Compatibility: This criterion indicates that the question q should be answerable by the answer sentence $s _ { i }$ . For evaluation, we classify each $q - s _ { i }$ pair as a) Direct and Explicit Answer (Dir.): $s _ { i }$ answers the q explicitly, b) Unfocused (Unfocus.): some parts of $s _ { i }$ answer q indirectly, or c) Not Answered: $s _ { i }$ does not answer q.

• Givenness: This criterion evaluates if the question only consists of information from the context. An ideal question should be naturally evoked from the context, without concepts that are not accessible to the reader from common knowledge. This criterion has the following categories a) No new concepts (No New): $q$ does not contain any concepts beyond the context or common knowledge, b) Answer leakage (Ans. leak.): q contains concepts that are not in the context but in $s _ { i } , \mathbf { c } )$ Hallucination (hall.): q contains new concepts that are not answer-leakage.

• Anchor Relevance: This criterion measures if the question q is relevant to and naturally evoked from the anchor sentence a. This criterion has the following categories a) Fully Grounded (Fully G.): q contains concepts from anchor a, b) Partially Grounded (Partial G.): q contains some concepts from anchor a and is not directly addressing the focus of a, c) Not grounded (No G.): q is completely irrelevant to a.

## C Human Evaluation Details

We provide the annotation template and training materials in Figure 6 and 7. All annotators were recurited from Amazon’s Mechanical Turk and fairly paid more than \$15 USD per hour which varied depending on the time spend per HIT (more than the national minimum wage where the annotators are recruited). To ensure high quality annotations, the annotators were provided with extensive guidelines and training (Figure 7).

We measure inter-annotator agreement with Krippendorff’s α. As shown in Table 3, annotators achieve “moderate" agreement across Answer Compatibility and Givenness (Artstein and Poesio, 2008). Since, relevance between two concepts (question and achor) is highly dependent on the annotators’ comprehension of the article, we find that agreement score for Anchor Relevance is “fair" (Artstein and Poesio, 2008). We also note the pairwise agreement in Table 3. The agreements are comparable with those in QUDEVAL, and indicate a certain degree of subjectivity in QUD analysis.

<table><tr><td></td><td>Comp.</td><td>Givn.</td><td>Relv.</td></tr><tr><td>Pair-Wise Agreement</td><td>70.0%</td><td>75.0%</td><td>60.0%</td></tr><tr><td>Krippendorff&#x27;s α</td><td>0.68</td><td>0.64</td><td>0.43</td></tr></table>

Table 3: Inter-annotator agreement for human judges.

## D Automatic Evaluator Details

We train automatic evaluators with the human annotations from QUDEVAL. Experienced human annotators assess the answer compatability, giveness, and anchor relevance of 2,040 machine-generated QUDs from 51 articles. We randomly split the articles into training/validation/test sets with the ratio of 60%/15%/25%.

We fine-tune classifiers for each criterion individually. Similar to Madaan et al. (2020), we use RoBERTa-large (Liu et al., 2019) as the backbone model of answer compatibility and anchor relevance, and Longformer-base (Beltagy et al., 2020) as the backbone model of givenness due to the longer context length. For answer compatibility, the input to the model is the question and the answer sentence, and the output is one of the three labels Dir-Ans., Unfocus., and Not-Ans. For givenness, the input is the context (sentences before the anchor sentence in the article) and the question, and the answer is one of the three labels No-New., Ans-leak., and Hallu. For anchor relevance, the input is the question and the anchor sentence, and the output is one of the three labels Full., Some., and No-G. Models are fine-tuned for 10 epochs with the learning rate 1e 5 and batch size 32.

We report the F1 scores of our automatic evaluators in Table 4. For reference, we also provide the F1 scores of the random baseline, and the best reference-free and reference-based metrics from QUDEVAL (Wu et al., 2023). GPT-Scr (w/o ref) and GPT-Scr (w/ ref) indicate prompting GPT-4 to score without and with the human-annotated reference QUD. BERTScore means calculating the similarity between the candidate and reference QUD with BERTScore (Zhang et al., 2019). The rulebased method checks if all content words in the candidate question are presented in the context. Please refer to the QUDEVAL paper for more details. Note that the results of random and ours are conducted on our held-out test set, while the results of baseline evaluators are conducted on two held-out articles. Our evaluators are better than or comparable with the baselines, highlighting the credibility of using them in automatic evaluation.

<table><tr><td>Compatibility</td><td>Dir-Ans.</td><td>Unfocus.</td><td>Not-Ans.</td><td>Macro F1</td></tr><tr><td>Random</td><td>0.68</td><td>0.03</td><td>0.15</td><td>0.29</td></tr><tr><td>GPT-Scr (w/o ref)</td><td>0.70</td><td>0.05</td><td>0.36</td><td>0.37</td></tr><tr><td>BERTScore</td><td>0.51</td><td>0.14</td><td>0.43</td><td>0.36</td></tr><tr><td>Ours</td><td>0.84</td><td>0.28</td><td>0.32</td><td>0.48</td></tr><tr><td>Givenness</td><td>No-New.</td><td>Ans-leak.</td><td>Hallu.</td><td>Macro F1</td></tr><tr><td>Random</td><td>0.65</td><td>0.29</td><td>0.10</td><td>0.35</td></tr><tr><td>Rule-based</td><td>0.52</td><td>0.40</td><td>0.19</td><td>0.37</td></tr><tr><td>GPT-Scr (w/ ref)</td><td>0.65</td><td>0.35</td><td>0.1</td><td>0.37</td></tr><tr><td>Ours</td><td>0.74</td><td>0.23</td><td>0.30</td><td>0.42</td></tr><tr><td>Relevance</td><td>Full.</td><td>Some.</td><td>No-G.</td><td>Macro F1</td></tr><tr><td>Random</td><td>0.52</td><td>0.22</td><td>0.21</td><td>0.32</td></tr><tr><td>GPT-Scr (w/o ref)</td><td>0.73</td><td>0.41</td><td>0.57</td><td>0.57</td></tr><tr><td>GPT-Scr (w/ ref)</td><td>0.63</td><td>0.26</td><td>0.22</td><td>0.37</td></tr><tr><td>Ours</td><td>0.79</td><td>0.32</td><td>0.48</td><td>0.53</td></tr></table>

Table 4: Automatic evaluator assessment in F1.

## E Evaluating the Correctness of the Selected Anchor

In §4 we focus on three criteria: answer compatibility, givenness and anchor relevance. We highlight that anchor relevance refers to the measure of relevance between the question and anchor (§B. Therefore, in our evaluation framework we evaluate the correctness of the selected anchor as how relevant it is to the question. An anchor that is incorrect or not relevant would be considered “not-grounded”. From Table 1, we see that QUDSELECT reduces the percentage of not grounded questions generated by the model and therefore improves the overall quality of the QUDs generated. To further analyse the correctness of the anchor selection we report the agreement accuracy (Table 5) of the the selected anchor sentences with the human annotated anchors from the DCQA dataset. Note that this is a partial notion of accuracy and does not accurately represent the quality of a model, since it is natural for different questions to be triggered from different sentences (Ko et al., 2023).

<table><tr><td>Model</td><td>Anchor Agreement</td></tr><tr><td>Pipeline</td><td>47.9%</td></tr><tr><td>LLaMA2-7B</td><td>48.7%</td></tr><tr><td>+ QUDSELECT</td><td>45.7%</td></tr></table>

Table 5: Anchor agreement score between the selected anchor and the human-annotated anchors from the DCQA dataset.

## F Article of Case Study

We provide the article snippet used in the case study in Figure 5. The article is from the DCQA dataset. We also provide questions generated by other models in Table 6.

![](images/8b5f8f9002a827201fafda01f0f7d4f5e64663963c873af23272169cb3a6677c.jpg)  
Figure 5: Article snippet used in case study.

<table><tr><td colspan="2">LLaMA2</td></tr><tr><td>Answer:  $s _ { 4 }$  Anchor:  $s _ { 3 }$  band?&quot;</td><td>QUD: &quot;What is deadly contra- XNon answer √No new concepts XPartially grounded</td></tr><tr><td>Answer:  $s _ { 3 }$  Anchor:  $s _ { 1 }$  nuclear material?&quot;&quot;</td><td>QUD: &quot;Why is it difficult to trace XNon answer √No new concepts √Fully grounded</td></tr><tr><td>QUDSELECT (LLaMA2)</td><td></td></tr><tr><td>Answer:  $s _ { 4 }$  Anchor:  $s _ { 2 }$  port?&quot;</td><td>QUD: “Who requested the re- √Direct answer √No new concepts √Fully grounded</td></tr><tr><td>Answer: s3 Anchor:  $s _ { 1 }$  inability to trace nuclear material?&quot;</td><td>QUD: &quot;What is the reason for the √Indirect Answer √No new concepts XPartially grounded</td></tr><tr><td>GPT4</td><td></td></tr><tr><td>Answer:  $s _ { 6 }$  Anchor:  $s _ { 6 }$  sional report reveal about the quantity of nuclear material that has accumulated globally?&quot;</td><td>QUD: “What does the congres- XGenerated the answer as the anchor and led to answer leakage</td></tr><tr><td>Answer:  $s _ { 4 }$  Anchor:  $s _ { 2 }$  for commissioning the report on the traceability of U.S. nuclear material exports?&quot;</td><td>QUD: “Who was responsible √No new concepts √Fully grounded</td></tr></table>

Table 6: Example QUDs generated by different models. The full article text can be found in Appendix Figure 5. $s _ { i }$ indicates the i-th sentence in the article.

![](images/548a1083145f2ef56cb77f18d504b6c61809168e7228b9e69b0bf8b93a544118.jpg)  
Figure 6: The annotation template for human evaluation. We ask annotators to classify the given QUD, anchor and answer for Givenness, Answer Compatibility, and Anchor Relevance.

![](images/10cccf67531212b434b4e652477e57c2f4f9899ce4a560b2b899c9bb022b1a4d.jpg)  
Figure 7: Additional training materials and instructions for human evaluation.