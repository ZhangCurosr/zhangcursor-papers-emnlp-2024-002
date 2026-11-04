# TOPVIEWRS: Vision-Language Models as Top-View Spatial Reasoners

Chengzu Li\*, Caiqi Zhang\*, Han Zhou, Nigel Collier, Anna Korhonen, Ivan Vulic´ Language Technology Lab, University of Cambridge {cl917, cz391, hz416, nhc30, alk23, iv250}@cam.ac.uk https://topviewrs.github.io

## Abstract

Top-view perspective denotes a typical way in which humans read and reason over different types of maps, and it is vital for localization and navigation of humans as well as of ‘non-human agents, such as the ones backed by large Vision-Language Models (VLMs). Nonetheless, spatial reasoning capabilities of modern VLMs in this setup remain unattested and underexplored. In this work, we study their capability to understand and reason over spatial relations from the top view. The focus on top view also enables controlled evaluations at different granularity of spatial reasoning; we clearly disentangle different abilities (e.g., recognizing particular objects versus understanding their relative positions). We introduce the TOPVIEWRS (Top-View Reasoning in Space) dataset, consisting of 11,384 multiple-choice questions with either realistic or semantic top-view map as visual input. We then use it to study and evaluate VLMs across 4 perception and reasoning tasks with different levels of complexity. Evaluation of 10 representative open- and closedsource VLMs reveals the gap of more than 50% compared to average human performance, and it is even lower than the random baseline in some cases. Although additional experiments show that Chain-of-Thought reasoning can boost model capabilities by 5.82% on average, the overall performance of VLMs remains limited. Our findings underscore the critical need for enhanced model capability in top-view spatial reasoning and set a foundation for further research towards human-level proficiency of VLMs in real-world multimodal tasks.

## 1 Introduction

Large Language Models (LLMs) such as Llama 2 and 3 (Touvron et al., 2023), Mistral (Jiang et al., 2023), and GPT models (OpenAI, 2022) have delivered impressive performance across a range of text-based tasks and applications such as question answering, language generation, and arithmetic reasoning (Qin et al., 2023a; Zhao et al., 2023). Building on these text-only LLMs, the so-called Vision Language Models (VLMs), equipped with the capability to process and reason over multimodal vision-language information, have enabled multi-modal processing (Yin et al., 2023; Wu et al., 2023). They ground language reasoning ability of LLMs into the information of different modalities (Chandu et al., 2021). Prominent examples of VLMs such as LLaVA (Liu et al., 2023b), GPT-4V (OpenAI, 2023), and Gemini (Google, 2024), have demonstrated strong performance across applications such as visual question answering (Li et al., 2023d), image captioning (Diesendruck et al., 2024), and object grounding (Zheng et al., 2024).

Spatial reasoning, one of the fundamental desirable properties of and requirements for VLMs, has also gained increased attention recently (Rajabi and Kosecka, 2023; Liu et al., 2023a; Chen et al., 2024). It requires grounding the model’s reasoning ability with natural language into its visual perception of the surrounding environment (Freksa, 1991). In particular, it involves two critical steps: (i) interpreting the environment visually, and (ii) reasoning over spatial relations. As a fundamental ability for the model to recognize, understand, and navigate through the physical world, it plays a crucial role in various downstream tasks such as vision-language generation (Li et al., 2024a) and embodied AI (Cho et al., 2024). However, previous research has focused on exploring spatial reasoning abilities of VLMs only from a conventional first-person perspective view (Liu et al., 2023a). In this work, we aim to study and evaluate spatial understanding and reasoning capability of VLMs from the top-view perspective, also referred to as the bird’s-eye view (Li et al., 2024b).

When compared to the conventional perspective view, top view offers better natural alignment: it is the typical perspective used for reading maps or presenting floor plans. Moreover, it is inherently more complex: top-view maps encapsulate a wealth of information about different scenes, locations, objects and their relationships in the environment based on a single image. In addition to the photorealistic top-view maps, semantic top-view maps (Nanwani et al., 2023; Li et al., 2024a) use different colors to represent different types of objects; we run experiments with both map types, see Figure 1.

![](images/872c398432bcd01136b5f49dd3413be43a2af124663db2136052526eb21b7308.jpg)  
Figure 1: Illustration of the four evaluation tasks with an incremental level of complexity on the two types of top-view maps (photo-realistic versus semantic maps), covering top-view perception and spatial reasoning abilities, with 9 sub-tasks in total (red font), focusing on different, well-defined VLM abilities. The radar graphs (top right) compare the representative models’ performance on all sub-tasks, indicating a large gap with human performance.

One advantage of top-view maps is that they define a controlled and interpretable experimental framework. Indoor scenes, which are the focus of this work, typically feature a relatively stable set of objects and layouts, making them ideal for controlled studies. This allows us to disentangle and investigate various aspects of spatial reasoning and VLMs’ capabilities in a controlled manner.<sup>1</sup>

In this work, we thus investigate the basic topview spatial understanding and reasoning abilities of current state-of-the-art VLMs across four tasks of gradually increasing complexity, and their finergrained sub-tasks. The tasks are as follows. 1) Top-View Recognition assesses whether the model can recognize concrete objects and scenes in topview maps. 2) Top-View Localization evaluates the ability to localize objects or regions on a map based on textual descriptions. (3) Static Spatial Reasoning investigates whether the model can reason about spatial relationships among localized objects and regions within the map. (4) Dynamic Spatial Reasoning evaluates reasoning about spatial relations along the points of a dynamic navigation path. Figure 1 illustrates all the tasks with concrete examples. As one key finding of this study, conducted evaluations reveal that current VLMs lack sufficient capability to effectively tackle top-view spatial reasoning challenges, indicating substantial room for improvement in future research.

Contributions. 1) We define the top-view spatial reasoning challenge for VLMs via 4 carefully designed tasks of increasing complexity, also encompassing 9 distinct fine-grained sub-tasks with a structured design of the questions focusing on different model abilities. 2) We collect the TOPVIEWRS dataset, comprising 11,384 multiplechoice questions with either photo-realistic or semantic top-view maps of real-world scenarios through a pipeline of automatic collection followed by human alignment. 3) We use TOPVIEWRS to evaluate and study 10 VLMs from different model families and sizes, highlighting the substantial performance gap compared to humans.<sup>2</sup>

## 2 Related Work

Top-View Map Understanding. There are only limited studies in NLP focused on the use of topview maps, though considerable research has been conducted within the broader AI community on the so-called bird’s-eye view, which is an instance of top view. This body of work has explored applications in autonomous driving (Unger et al., 2023; Li et al., 2024c), with contributions on fusing different types of views (Qin et al., 2023b) and working with arbitrary camera setups (Peng et al., 2023). In other application scenarios, Yan et al. (2021) introduce a bird’s-eye view person re-identification task.

Efforts to bridge top-view images with natural language in applications beyond the above are less diverse. The WAY dataset, proposed by Hahn et al. (2020), contains 6,154 dialogs aimed at localizing an observer’s position on a top-view map through conversations between an observer and a locator. This dataset has inspired follow-up research focusing on merging vision with dialog information (Zhang et al., 2024a) and leveraging pretraining strategies to enhance performance (Hahn and Rehg, 2022). In general, prior research does not assess VLMs’ basic spatial reasoning abilities with topview images and lacks fine-grained and controllable analyses of these fundamental abilities.

Spatial Reasoning on Multi-Modal Vision-Text. There has been a body of work on text-only spatial reasoning with the advancement of LLMs (Yamada et al., 2024), within the context of relative spatial relation recognition (Mirzaee et al., 2021; Shi et al., 2022), natural language navigation (Yamada et al., 2024), and planning (Momennejad et al., 2023) (see Appendix A for a more complete overview).

Cross-modal spatial reasoning puts forward higher requirements for the models in terms of language grounding (Rozanova et al., 2021; Rajabi and Kosecka, 2023). Liu et al. (2023a) investigate spatial reasoning with 2D natural realistic frontview images and Chen et al. (2024) extend the analysis to 3D point clouds. The environmental contexts become more diverse compared to synthetic symbols in text-only spatial reasoning, ranging from indoor environments (Koch et al., 2024) to outdoor street views (Chen et al., 2019). Regarding typical tasks, visual QA (VQA) is the mainstream task for benchmarking spatial reasoning abilities (Dong et al., 2021; Banerjee et al., 2021; Liu et al., 2023a; Li et al., 2023a,b; Kamath et al., 2023), while other tasks include vision-language navigation (Chen et al., 2019; Li et al., 2024a) and user interface grounding (Rozanova et al., 2021).<sup>3</sup>

We stress that none of the prior research efforts allows for disentangled evaluation of models’ spatial reasoning abilities. Prior work typically conflates object recognition with spatial reasoning. We thus design a dataset and conduct a study that not only offers insight into fundamental abilities but also allows for easier interpretation of results (§4).

## 3 Task Definition

Following prior work (Li et al., 2023a), we frame all tasks as multiple-choice QA tasks. Given a top-view (realistic or semantic) map of a room M, the model must choose the correct option o<sub>i</sub> from the four options provided $O = \{ o _ { 0 } , o _ { 1 } , o _ { 2 } , o _ { 3 } \}$ that answers the question.<sup>4</sup> This format simplifies the evaluation and interpretation of the results.

Top-View Maps. We provide two different types of top-view maps to the models: realistic maps $M _ { \mathrm { R e a l } }$ and semantic maps $M _ { \mathrm { S e m } }$ . Realistic maps are constructed by placing a simulated orthographic camera above the scene to capture a photo-realistic top-view image. Semantic maps represent objects in the scene with colored bounding boxes. Each object is assigned a specific color and labeled at the same relative coordinates on the map to preserve the object’s semantic information and spatial allocation. In comparison to realistic maps, semantic maps simplify the initial step of spatial reasoning (i.e., environment interpretation) by labeling the object types with corresponding colors and excluding irrelevant additional details such as shape and texture found in realistic top-view maps. Given the customizable and flexible nature of colorobject mapping, the semantic map can also serve as an ideal testbed for evaluating models’ out-ofdistribution (OOD) performance, thereby encouraging further exploration beyond the scope of this work. Example maps are in Figure 1.

Tasks and Sub-Tasks. We define 4 different tasks which cover a total of 9 finer-grained sub-tasks, with concrete examples shown in Figure 1. The tasks are designed to have an increasing level of complexity, where each subsequent task depends on the abilities measured in the preceding one(s).

(1) Top-View Recognition evaluates the fundamental ability to interpret the input map, and covers two sub-tasks: Object Recognition and Scene Recognition. It does not require the model to identify specific locations of objects and rooms.

(2) Top-View Localization investigates whether the model can localize objects or rooms in the topview map based on textual descriptions, including Object Localization and Scene Localization as two sub-tasks. Beyond understanding the top-view map as a whole, it requires the model to ground entities in the map, representing the model’s ability to align spatial descriptions with corresponding locations.

(3) Static Spatial Reasoning aims to evaluate the model’s spatial reasoning ability with more complex questions. It includes two sub-tasks: reasoning over Scene Counting and Relative Spatial Relations between different objects and rooms. These questions require the model to perform multi-step reasoning based on the recognition and localization of entities in the top-view map.

(4) Dynamic Spatial Reasoning. Finally, we introduce a novel task that involves dynamic spatial reasoning over top-view maps in the context of agent navigation. It requires the model to understand the sequential relations along the points of the navigation path (sub-task Dynamic Action Counting) and answer spatial questions with regard to the dynamic navigation path (sub-task Dynamic Relative Spatial Reasoning) and the circumstantial environments (Dynamic Spatial Localization).

## 4 TOPVIEWRS Dataset

In order to study and evaluate the abilities of stateof-the-art VLMs on the 4 tasks spanning 9 subtasks from §3, we now introduce a novel evaluation dataset, TOPVIEWRS, which focuses on top-view maps of indoor scenes (i.e., houses and rooms), discussed in what follows.

Dataset Features. It introduces several advancements and innovative features that distinguish it from all prior visual spatial reasoning datasets.

1) Multi-Scale Top-View Maps: The selected topview maps of indoor scenes (see Figure 1) provide a more natural representation of spatial environments that aligns with human cognitive map (Epstein et al., 2017). This makes benchmarking spatial awareness more straightforward and meanwhile mitigates spurious correlations in the positions between objects commonly found in realistic front-view images. Compared to the front view, the multi-scale top-view maps of single rooms and full houses add more divergence in the granularity of the entities (objects or rooms) in spatial reasoning. Meanwhile, we provide both realistic maps and semantic maps for more comprehensive evaluation. 2) Realistic Environmental Scenarios with Rich Object Sets: We provide real-world environments from indoor scenes, with 80 objects per scene on average, ensuring a natural distribution and complexity of object locations. This also sets it apart from existing front-view spatial reasoning datasets, which typically contain only a handful of objects. 3) Structured Question Framework: Unlike previous datasets (Li et al., 2023a; Liu et al., 2023a; Kamath et al., 2023), which conflate spatial reasoning with object recognition, our dataset clearly defines 4 tasks including 9 sub-tasks in total using diverse question templates. This structured approach allows for a fine-grained evaluation and analysis of models’ capabilities from various perspectives and levels of granularity.

Dataset Collection. We employ a two-stage data collection strategy that includes automatic collectionfrom a simulator and alignment through human judgment. First, to approximate real-life scenarios, we use the Matterport3D dataset (Chang et al., 2017), which includes 90 building-scale scenes with instance-level semantic and room-level region annotations in 3D meshes. We filter these to exclude multi-floor and low-quality scenes, selecting 7 scenes with an average of 80 objects and 12 rooms each. Realistic top-view maps are extracted using orthographic cameras, and semantic top-view maps are constructed using the Habitat (Manolis Savva\* et al., 2019; Szot et al., 2021) simulation environment. We then design a structured question framework with 15 templates to minimize human labor and standardize the data collection process. To ensure quality, a second stage of manual humanjudgment aligns and verifies the data, ensuring questions are natural and correct. Participants are encouraged to discard or modify data points to improve quality, maintaining alignment with human judgments. We refer readers to Appendix B for further details regarding the data collection process.

![](images/86db73e01df70ca53d91b2550ae40c3d34e5fd60d59b673aa982a7fd5bc99f98.jpg)  
Figure 2: TOPVIEWRS data statistics, showing distribution of task sizes, objects, regions, spatial and relative spatial descriptions in realistic and semantic map settings, where the tasks are described with their initials for visualization.

Dataset Statistics. TOPVIEWRS comprises a total of 11,384 multiple-choice questions after human verification, with 5,539 questions associated with realistic top-view maps, and 5,845 with semantic top-view maps. Human verification keeps 587/784 questions from the automatic collection phase for Top-View Recognition, 1,077/1,384 for Top-View Localization, 2,340/3,080 for Static Spatial Reasoning. The choices are uniformly distributed over choices A (25.5%), B (24.6%), C (24.5%) and D (25.4%). Figure 2 shows the distribution of different tasks, objects, regions and spatial descriptions. The size of each task aligns with its corresponding difficulty level, where the easier task comprises fewer examples. We provide further insights and technical details in Appendix B.4.

## 5 Experiments and Results

Models and Implementation. We test a representative selection of both open-sourced and closesourced models which achieve state-of-the-art performance on a range of multimodal benchmarks (Liu et al., 2023c; Li et al., 2023a) in a zero-shot inference setup. Regarding open-sourced models, we study and evaluate Idefics (9B & 80B) (Laurençon et al., 2023), LLaVA-Next (7B, 13B & 34B) (Liu et al., 2024), InternLM-XComposer2 (7B) (Dong et al., 2024), Qwen-VL (7B) (Bai et al., 2023). The chosen close-sourced models are GPT-4V (OpenAI, 2023) and Gemini (Google, 2024).<sup>5</sup> All the models are implemented within the VLMEvalKit framework (OpenCompass Contributors, 2023).

Prompts. For realistic maps, we provide the VLMs with the task description along with the multiplechoice question. For semantic maps, in addition to the information above, we also introduce the concept of a semantic map to the model and provide the color-object mapping in the prompt in order to facilitate its understanding of the abstract map. We only provide the color-object mappings of the colors that are presented in the semantic map as a preprocessing strategy in order to exclude irrelevant information. For the specific prompting templates used in this paper, we refer to Appendix C.2.

Evaluation Measures. We measure multiplechoice QA accuracy via Exact Match (EM) and Partial Match (PM). EM measures whether the predicted option indices are exactly the same as the label indices. However, there may be cases where the correct answer to the question can be considered partially correct, e.g., the answer is top right while the prediction is top left. PM then calculates the proportion of overlapping words between the predicted answer and the gold answer. It is calculated based on the correctness of the text spans (or words) of predicted options, as given by:

$$
P M = \frac { \left| \left\{ \mathrm { l a b e l s } \right\} \cap \left\{ \mathrm { p r e d i c t i o n s } \right\} \right| } { \operatorname* { m a x } \left( \left| \left\{ \mathrm { l a b e l s } \right\} \right| , \left| \left\{ \mathrm { p r e d i c t i o n s } \right\} \right| \right) }
$$

## 5.1 Results and Discussion

We first discuss the models’ performance across our four tasks, with results summarized in Table 1, and fine-grained sub-task performance illustrated in Figure 3. We find that the performance of current state-of-the-art VLMs is unsatisfactory on the proposed TOPVIEWRS benchmark with model-wise average EM and PM over all tasks below 50%. Gemini is the best-performing model for realistic maps, while GPT-4V excels in semantic maps. For some models, such as Qwen-VL, the results are sometimes much worse than the random baseline. This issue primarily arises from the models’ difficulty in following the instructions to choose from the four provided options.

Models perform better on recognition and localization tasks compared to reasoning tasks. Top-View Recognition consistently demonstrates the highest performance across all models. Gemini shows human-comparable performance with the EM score over 90%. Top-View Localization exhibits lower performance compared to Top-View Recognition, followed by Static Spatial Reasoning. This highlights that the decline in model performance is primarily due to the lack of spatial reasoning, rather than the domain shift introduced by the top-view map. The performance difference of various tasks with different levels of complexity also underscores the advantage of our benchmark to capture well-defined and disentangled phenomena, which allows for controlled studies in controlled environments.

Regarding Dynamic Spatial Reasoning, models perform better on this task than on the previous tasks. Fine-grained performance in Figure 3 indicates that the improved performance primarily stems from high accuracy in dynamic action counting and spatial localization, which constitute 18% and 66% of the data respectively for this task. We attribute the high accuracy in these areas to the equivalence between navigation path symbols and visual prompting (Shtedritski et al., 2023). Despite these advancements, the overall EM accuracy remains below 40%, and models still struggle with reasoning over dynamic relative spatial relations.

Larger models do not always show better spatial awareness. Surprisingly, our results reveal that larger model sizes do not consistently translate to better performance. In Top-View Recognition, closed-source models outperform open-source models by 31.10% EM with realistic maps and 29.33% EM with semantic maps. However, the performance gap narrows as the task complexity increases. Using realistic maps as the visual input, Gemini stands out by achieving a minimum of 5.53% higher EM accuracy in Static Spatial Reasoning compared to other models, while GPT-4V performs worse than Idefics-9B on both Static and Dynamic Spatial Reasoning tasks. This indicates a lack of significant difference in spatial awareness between closed-source and open-source models for tasks with higher complexity, despite the disparity in their model sizes. This trend holds true within open-sourced models as well. Both Idefics and LLaVANext model families in some cases show comparable or worse performance with larger model variants than with smaller ones. Similar observations have been made by previous studies (Zhong et al., 2021; Shi et al., 2024). We conjecture that this might be caused by inadequate evidence of the scaling law (Kaplan et al., 2020) in the computer vision community (Tian et al., 2024). The results on TOPVIEWRS thus advocate for further investigation and analysis in this area.

Models perform better in easier tasks with semantic maps. In simple tasks such as Top-View Recognition, models generally perform better with semantic maps than with realistic maps, except for Qwen-VL, showing an improvement of 20.35%. However, this advantage decreases in more complex tasks. For Top-View Localization and Static Spatial Reasoning, models struggle to utilize semantic top-view maps, yielding performances akin to random baselines in both EM and PM accuracy. One possible explanation is that the semantic topview image and the input prompt with color-object mapping deviate too much from the models’ training data distribution. This is further evidenced by the predictions from open-sourced models such as Qwen-VL, which fail to respond to instructions and answer with numbers or RGB values 91.25% of the time for Top-View Localization and 47.65% of the time for Static Spatial Reasoning.

<table><tr><td colspan="2">Model Model Size</td><td colspan="2">Idefics</td><td colspan="4">LLaVANext</td><td colspan="2">XComposer2</td><td colspan="2">Qwen-VL GPT-4V</td><td rowspan="2">Gemini</td></tr><tr><td colspan="2">Realistic Map</td><td colspan="2">9B 80B</td><td colspan="2">vicuna 7B mistral 7B</td><td>vicuna 13B</td><td>34B</td><td>7B</td><td>7B</td><td>API</td><td>API</td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td><td>61.64</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Top-View Recognition</td><td>EM PM</td><td>41.10 41.10</td><td>26.71</td><td>67.47</td><td>61.30</td><td></td><td></td><td>67.81</td><td>37.67</td><td>27.05</td><td>69.52</td><td>90.41</td></tr><tr><td></td><td></td><td>26.88</td><td>67.64</td><td>61.47</td><td></td><td>61.99</td><td>67.81</td><td>37.67</td><td>27.26</td><td>69.86</td><td>90.58</td></tr><tr><td rowspan="2">Top-View Localization</td><td>EM</td><td>30.39</td><td>30.00</td><td>42.16</td><td>33.92</td><td>41.18</td><td>50.98</td><td>27.84</td><td></td><td>16.27</td><td>46.27</td><td>48.24</td></tr><tr><td>PM</td><td>46.42</td><td>46.08</td><td>56.67</td><td>48.63</td><td>54.31</td><td></td><td>61.76</td><td>41.86</td><td>26.31</td><td>60.39</td><td>60.98</td></tr><tr><td rowspan="2">Static Spatial Reasoning</td><td>EM</td><td>24.07</td><td>26.07</td><td>19.87</td><td>24.36</td><td></td><td>20.25</td><td>22.73</td><td>25.79</td><td>14.71</td><td>22.16</td><td>31.61</td></tr><tr><td>PM</td><td>33.68</td><td>38.52</td><td>34.40</td><td>37.34</td><td></td><td>36.26</td><td>35.56</td><td>38.73</td><td>21.15</td><td>35.59</td><td>45.22</td></tr><tr><td>Dynamic Spatial Reasoning</td><td>EM</td><td>38.10</td><td>27.94</td><td>38.81</td><td>24.31</td><td>29.08</td><td>23.79</td><td>24.07</td><td></td><td>22.11</td><td>30.29</td><td>32.60</td></tr><tr><td>Semantic Map</td><td>PM</td><td>40.88</td><td>30.68</td><td>42.15</td><td>26.69</td><td>32.89</td><td></td><td>27.28</td><td>26.86</td><td>24.65</td><td>33.86</td><td>35.80</td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td><td>88.14</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Top-View Recognition</td><td>EM</td><td>60.68</td><td>59.32</td><td>88.81</td><td>80.00</td><td></td><td></td><td>94.58</td><td>43.05</td><td>19.66</td><td>97.29</td><td>94.92</td></tr><tr><td>PM</td><td>60.68</td><td>59.32</td><td>88.81</td><td>80.00</td><td></td><td>88.49</td><td>94.58</td><td>43.05</td><td>20.05</td><td>97.29</td><td>94.92</td></tr><tr><td rowspan="2">Top-View Localization</td><td>EM</td><td>31.21</td><td>27.34</td><td>25.40</td><td>32.10</td><td>17.28</td><td></td><td>38.45</td><td>24.87</td><td>9.70</td><td>44.44</td><td>35.27</td></tr><tr><td>PM</td><td>47.62</td><td>45.41</td><td>44.27</td><td>47.80</td><td></td><td>23.66</td><td>53.79</td><td>41.09</td><td>13.99</td><td>58.55</td><td>49.91</td></tr><tr><td rowspan="2">Static Spatial Reasoning</td><td>EM</td><td>23.82</td><td>28.07</td><td>18.72</td><td>24.28</td><td></td><td>16.63</td><td>18.41</td><td>23.05</td><td>14.85</td><td>21.73</td><td>26.22</td></tr><tr><td>PM</td><td>34.13</td><td>38.17</td><td>30.57</td><td>37.26</td><td></td><td>29.94 19.92</td><td>31.22</td><td>35.50</td><td>21.99</td><td>33.09</td><td>39.12</td></tr><tr><td rowspan="2">Dynamic Spatial Reasoning</td><td>EM</td><td>36.67</td><td>34.55</td><td>37.45 40.69</td><td>26.23</td><td></td><td></td><td>33.12</td><td>21.60</td><td>23.55</td><td>39.30</td><td>31.41</td></tr><tr><td>PM</td><td>39.92</td><td>37.75</td><td></td><td>28.89</td><td></td><td>23.63</td><td>36.86</td><td>24.32</td><td>26.09</td><td>43.20</td><td>34.86</td></tr></table>

Table 1: Comparison of 10 models on both realistic and semantic top-view maps. Performance is analysed according to four tasks with EM and PM. The best performance for each task is illustrated in bold.  
![](images/852fdce8ad2423e25b58d2cf69241411c7fc6de009508746ea19c847f9aa5296.jpg)  
(a) Performance with realistic top-view maps  
(b) Performance with semantic top-view maps  
Figure 3: Visualization of fine-grained comparison with 10 models and humans on 9 sub-tasks using realistic and semantic top-view maps, demonstrating that most current models perform on par with random baseline in spatial reasoning and has a large gap with human performance. Exact numbers are reported in Table 15 in the Appendix.

Fine-Grained Insights with Sub-Tasks. Models using realistic maps excel more in the sub-task of Scene Recognition, which involves larger entities, compared to Object Recognition. This gap is also evident in a 12.66% and 19.73% performance difference between object-level and scene-level localization with both map types. Conversely, with semantic maps, the model struggles more with scenelevel recognition than with realistic maps, showing an 11.09% lower performance than object-level recognition among closed-source models. Most models perform similarly to a random baseline in reasoning over spatial relations but show higher accuracy in scene counting. This likely occurs because 95% of the correct room counts are within a narrow range (1 or 2), reflecting real-life distributions. Thus, models leverage commonsense knowledge as the shortcut for counting, as seen in the 54.73% performance gap (with GPT-4V) between counting scenes and actions. However, the spatial localization and reasoning abilities of both open-source and closed-source models still remain unsatisfactory, even at the level of sub-tasks.

## 5.2 Further Discussion

Gap to Human Performance. We now study how humans perform on this dataset and the gap between current models and human performance. To this end, we recruited 4 human participants who were not involved in dataset creation for human evaluation. A total of 60 data points with realistic top-view maps are randomly selected from the subtasks, covering all fine-grained question types.<sup>6</sup> We use Fleiss Kappa as the measure of inter-annotator agreement. The kappa score is 0.747, indicating substantial agreement shared by the human participants according to Landis and Koch (1977). The average performance of the human participants is shown in Table 2. The experimental results show that there is still a large gap with human performance by over 50% across all the sub-tasks that involve spatial awareness. We also observe that with GPT-4V, human performs 47.8% higher than the model on average. The gap between human and model performance is larger on complex reasoning tasks compared to the recognition tasks, indicating plenty of room for improvement.

<table><tr><td>Task</td><td>Ability</td><td>Size</td><td>Human</td><td>GPT-4V</td></tr><tr><td rowspan="2">TVR</td><td>Object Recognition</td><td>5</td><td>95</td><td>100</td></tr><tr><td>Scene Recognition</td><td>5</td><td>100</td><td>80</td></tr><tr><td rowspan="2">TVL</td><td>Object Localization</td><td>5</td><td>95</td><td>20</td></tr><tr><td>Scene Localization</td><td>10</td><td>85</td><td>60</td></tr><tr><td rowspan="2">SSR</td><td>Scene Counting</td><td>5</td><td>100</td><td>80</td></tr><tr><td>Relative Spatial Relation</td><td>10</td><td>80</td><td>0</td></tr><tr><td rowspan="3">DSR</td><td>Dynamic Action Counting</td><td>5</td><td>85</td><td>0</td></tr><tr><td>Dynamic Spatial Localization</td><td>10</td><td>85</td><td>40</td></tr><tr><td>Dynamic Relative Spatial Reasoning</td><td>5</td><td>85</td><td>0</td></tr><tr><td colspan="2">Average Score</td><td></td><td>90.0</td><td>42.2</td></tr></table>

Table 2: Performance (EM) between human and GPT-4V on all the sub-tasks, demonstrating a huge gap between GPT-4V and human.

Chain-of-Thought Helps Elicit Spatial Reasoning. Due to the compositionality of Static Spatial Reasoning based on Top-View Recognition and Localization in task design, the model is supposed to answer the question based on the locations of the entities in the top-view map. Inspired by this requirement, we explored whether Chain-of-Thought (CoT) reasoning (Wei et al., 2022) could facilitate spatial reasoning by initially prompting the model to localize entities before producing the final answer to the question. To implement this, we modified the instruction to include: “You should first localize the entity and then answer the question based on the locations”, thereby encouraging the model to process information and think step by step.

<table><tr><td rowspan="2">Model</td><td colspan="3">GPT-4V</td><td colspan="3">Gemini</td></tr><tr><td>w/o. CoT</td><td>w. CoT</td><td>Δ</td><td>w/o.CoT</td><td>w. CoT</td><td>Δ</td></tr><tr><td>RGB Overall</td><td>22.16</td><td>26.74</td><td>+4.58</td><td>31.61</td><td>40.02</td><td>+8.41</td></tr><tr><td>Scene Counting</td><td>76.74</td><td>25.58</td><td>-51.16</td><td>53.49</td><td>48.84</td><td>-4.65</td></tr><tr><td>Relative Spatial Relations</td><td>19.82</td><td>26.79</td><td>+6.97</td><td>30.68</td><td>39.64</td><td>+8.96</td></tr><tr><td>Semantic Overall</td><td>21.73</td><td>28.07</td><td>+6.34</td><td>26.22</td><td>30.16</td><td>+3.94</td></tr><tr><td>Scene Counting</td><td>37.50</td><td>47.92</td><td>+10.42</td><td>20.83</td><td>29.17</td><td>+8.34</td></tr><tr><td>Relative Spatial Relations</td><td>21.12</td><td>27.31</td><td>+6.19</td><td>26.43</td><td>30.20</td><td>+3.77</td></tr></table>

Table 3: Comparison of model performance (EM) w/ and w/o Chain of Thought (CoT) on Static Spatial Reasoning, showing that CoT helps elicit spatial reasoning.

Considering that CoT has shown effectiveness in larger models (Wei et al., 2022; Li et al., 2023c), we conducted experiments with GPT-4V and Gemini to evaluate this hypothesis. As shown in Table 3, incorporating CoT into the reasoning process notably enhances performance. Specifically, the models’ accuracy improved by 4.58% when using realistic maps and 6.34% with semantic maps for GPT-4V. This improvement underscores the potential of step-by-step reasoning in enhancing the efficacy of spatial reasoning tasks, but there is still a substantial performance gap to the human ceiling.

## 6 Conclusion

In this study, we designed four tasks to examine the capabilities of VLMs as top-view spatial reasoners, progressing from basic top-view map comprehension to dynamic spatial reasoning along navigation paths. To enable investigation into topview spatial reasoning abilities of VLMs, we collected a novel dataset, TOPVIEWRS, which includes 11,384 multiple-choice questions, featuring photo-realistic and semantic top-view maps as the visual input. Our extensive experiments involved evaluating 10 VLMs across various model families and sizes on TOPVIEWRS. The results highlight a critical observation: particularly in complex reasoning tasks, VLMs frequently perform only as well as a random baseline, with even more pronounced deficits when handling tasks with semantic maps. Moreover, there is a noticeable performance gap compared to human annotators, underscoring the significant potential for further improvements in this area. In response to these findings, we discovered that employing chain-of-thought reasoning enhances model performance in spatial reasoning by 5.82%. Despite this progress, the overall performance of VLMs on spatial reasoning remains less than satisfactory. We hope that our study can set the stage for future research in multimodal spatial reasoning and encourage further investigations into refining the reasoning techniques, moving VLMs closer to human-level proficiency in understanding and reasoning over real-world environments.

## Limitations

The TOPVIEWRS dataset primarily evaluates model performance in entity recognition, localization, and spatial reasoning over 2D top-view maps. However, it does not yet include task-oriented planning with spatial awareness, which involves more complex sequential decision-making and dynamic interactions.

Further, our dataset assumes one correct answer per question, but exploring scenarios with multiple correct answers or no correct answers could further challenge systems and provide valuable insights.

We also advocate for further research to explore how spatial awareness in models impacts downstream tasks such as navigation instruction generation (Li et al., 2024a) and task completion by language agents in real-world environments (Parashar et al., 2023).

Moreover, our study is currently limited to 2D top-view maps, whereas spatial reasoning can encompass a variety of modalities and perspectives, such as 3D point clouds.

From the perspective of the models, the rapid progress in VLMs makes it hard to include all new releases such as Idefics 2 (Laurençon et al., 2024). Additionally, multimodal in-context learning (MICL) remains underexplored and is only supported by VLMs trained with interleaved imagetext data (Baldassini et al., 2024). Although not universal across all VLMs, MICL has been effective in handling out-of-distribution tasks (Zhang et al., 2024b), which could also be interesting in TOPVIEWRS, especially with semantic maps as visual inputs. In future work, we aim to extend our analysis to include more modalities, evaluate a broader range of models and their capabilities, and investigate additional downstream tasks involving spatial awareness.

## Ethics Statement

Our research strictly follows ethical guidelines, focusing on data privacy, bias mitigation, and societal impact. During the dataset construction, we carefully check the licenses of the software we use and follow it strictly. The human participants in our study are recruited from our university with bachelor’s degree and are guaranteed compensation above the local minimum average. They have consented to the use of their annotations in our research. We do not see any potential risk of our project.

## Acknowledgments

The work was partly supported by a Royal Society University Research Fellowship Inclusive and Sustainable Language Technology for a Truly Multilingual World’ (no 221137) awarded to Ivan Vulic.´ We thank Fangyu Liu and Ye Mao for constructive feedback on the draft of this paper.

## References

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A versatile visionlanguage model for understanding, localization, text reading, and beyond. Preprint, arXiv:2308.12966.

Folco Bertini Baldassini, Mustafa Shukor, Matthieu Cord, Laure Soulier, and Benjamin Piwowarski. 2024. What makes multimodal in-context learning work? Preprint, arXiv:2404.15736.

Pratyay Banerjee, Tejas Gokhale, Yezhou Yang, and Chitta Baral. 2021. Weakly supervised relative spatial reasoning for visual question answering. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 1908–1918.

Cesar Cadena, Luca Carlone, Henry Carrillo, Yasir Latif, Davide Scaramuzza, Jose Neira, Ian Reid, and John J. Leonard. 2016. Past, present, and future of simultaneous localization and mapping: Toward the robust-perception age. IEEE Transactions on Robotics, 32(6):1309–1332.

Khyathi Raghavi Chandu, Yonatan Bisk, and Alan W Black. 2021. Grounding ‘grounding’ in NLP. In Findings ofthe Associationfor Computational Linguistics: ACL-IJCNLP 2021, pages 4283–4305, Online. Association for Computational Linguistics.

Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. 2017. Matterport3d: Learning from rgb-d data in indoor environments. International Conference on 3D Vision (3DV).

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess, Pete Florence, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. 2024. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. Preprint, arXiv:2401.12168.

Howard Chen, Alane Suhr, Dipendra Misra, Noah Snavely, and Yoav Artzi. 2019. Touchdown: Natural language navigation and spatial reasoning in visual

street environments. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE.

Junmo Cho, Jaesik Yoon, and Sungjin Ahn. 2024. Spatially-aware transformers for embodied agents. In The Twelfth International Conference on Learning Representations.

Maurice Diesendruck, Jianzhe Lin, Shima Imani, Gayathri Mahalingam, Mingyang Xu, and Jie Zhao. 2024. Learning how to ask: Cycle-consistency refines prompts in multimodal foundation models. Preprint, arXiv:2402.08756.

Tianai Dong, Alberto Testoni, Luciana Benotti, and Raffaella Bernardi. 2021. Visually grounded follow-up questions: a dataset of spatial questions which require dialogue history. In Proceedings of Second International Combined Workshop on Spatial Language Understanding and Grounded Communication for Robotics, pages 22–31, Online. Association for Computational Linguistics.

Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. 2024. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. Preprint, arXiv:2401.16420.

Russell Epstein, E Z Patai, Joshua Julian, and Hugo Spiers. 2017. The cognitive map in humans: Spatial navigation and beyond. Nature Neuroscience, 20:1504–1513.

Christian Freksa. 1991. Qualitative Spatial Reasoning, pages 361–372. Springer Netherlands, Dordrecht.

Gemini Team Google. 2024. Gemini: A family of highly capable multimodal models. Preprint, arXiv:2312.11805.

Meera Hahn, Jacob Krantz, Dhruv Batra, Devi Parikh, James Rehg, Stefan Lee, and Peter Anderson. 2020. Where are you? localization from embodied dialog. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 806–822, Online. Association for Computational Linguistics.

Meera Hahn and James M. Rehg. 2022. Transformerbased localization from embodied dialog with largescale pre-training. In Proceedings ofthe 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 295–301, Online only. Association for Computational Linguistics.

Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 2023. 3d-llm: Injecting the 3d world into large language models. Preprint, arXiv:2307.12981.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7B. Preprint, arXiv:2310.06825.

Amita Kamath, Jack Hessel, and Kai-Wei Chang. 2023. What’s “up” with vision-language models? investigating their struggle with spatial reasoning. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 9161– 9175, Singapore. Association for Computational Linguistics.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. Preprint, arXiv:2001.08361.

Sebastian Koch, Narunas Vaskevicius, Mirco Colosi, Pedro Hermosilla, and Timo Ropinski. 2024. Open3dsg: Open-vocabulary 3d scene graphs from point clouds with queryable objects and open-set relationships. Preprint, arXiv:2402.12259.

J. Richard Landis and Gary G. Koch. 1977. The measurement of observer agreement for categorical data. Biometrics, 33(1):159–174.

Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. 2023. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Preprint, arXiv:2306.16527.

Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. 2024. What matters when building vision-language models? Preprint, arXiv:2405.02246.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023a. Seed-bench: Benchmarking multimodal llms with generative comprehension. Preprint, arXiv:2307.16125.

Chengzu Li, Chao Zhang, Simone Teufel, Rama Sanand Doddipatla, and Svetlana Stoyanchev. 2024a. Semantic map-based generation of navigation instructions. Preprint, arXiv:2403.19603.

Hao Li, Jinfa Huang, Peng Jin, Guoli Song, Qi Wu, and Jie Chen. 2023b. Weakly-supervised 3d spatial reasoning for text-based visual question answering. IEEE Transactions on Image Processing, 32:3367–3382.

Hongyang Li, Chonghao Sima, Jifeng Dai, Wenhai Wang, Lewei Lu, Huijie Wang, Jia Zeng, Zhiqi Li, Jiazhi Yang, Hanming Deng, Hao Tian, Enze Xie, Jiangwei Xie, Li Chen, Tianyu Li, Yang Li, Yulu Gao, Xiaosong Jia, Si Liu, Jianping Shi, Dahua Lin, and Yu Qiao. 2024b. Delving into the devils of bird’seye-view perception: A review, evaluation and recipe. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(4):2151–2170.

Hongyang Li, Chonghao Sima, Jifeng Dai, Wenhai Wang, Lewei Lu, Huijie Wang, Jia Zeng, Zhiqi Li, Jiazhi Yang, Hanming Deng, Hao Tian, Enze Xie, Jiangwei Xie, Li Chen, Tianyu Li, Yang Li, Yulu Gao, Xiaosong Jia, Si Liu, Jianping Shi, Dahua Lin, and Yu Qiao. 2024c. Delving into the devils of bird’seye-view perception: A review, evaluation and recipe. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(4):2151–2170.

Liunian Harold Li, Jack Hessel, Youngjae Yu, Xiang Ren, Kai-Wei Chang, and Yejin Choi. 2023c. Symbolic chain-of-thought distillation: Small models can also “think” step-by-step. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2665– 2679, Toronto, Canada. Association for Computational Linguistics.

Yunxin Li, Longyue Wang, Baotian Hu, Xinyu Chen, Wanqi Zhong, Chenyang Lyu, Wei Wang, and Min Zhang. 2023d. A comprehensive evaluation of gpt-4v on knowledge-intensive visual question answering. Preprint, arXiv:2311.07536.

Fangyu Liu, Guy Emerson, and Nigel Collier. 2023a. Visual spatial reasoning. Transactions of the Associationfor Computational Linguistics, 11:635–651.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024. Llava-next: Improved reasoning, OCR, and world knowledge.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. 2023c. Mmbench: Is your multi-modal model an all-around player? Preprint, arXiv:2307.06281.

Manolis Savva\*, Abhishek Kadian\*, Oleksandr Maksymets\*, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, and Dhruv Batra. 2019. Habitat: A Platform for Embodied AI Research. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV).

Roshanak Mirzaee and Parisa Kordjamshidi. 2022. Transfer learning with synthetic corpora for spatial role labeling and reasoning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6148–6165, Abu

Dhabi, United Arab Emirates. Association for Computational Linguistics.

Roshanak Mirzaee, Hossein Rajaby Faghihi, Qiang Ning, and Parisa Kordjamshidi. 2021. SPARTQA: A textual question answering benchmark for spatial reasoning. In Proceedings ofthe 2021 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 4582–4598, Online. Association for Computational Linguistics.

Ida Momennejad, Hosein Hasanbeig, Felipe Vieira Frujeri, Hiteshi Sharma, Nebojsa Jojic, Hamid Palangi, Robert Ness, and Jonathan Larson. 2023. Evaluating cognitive maps and planning in large language models with cogeval. In Advances in Neural Information Processing Systems, volume 36, pages 69736–69751. Curran Associates, Inc.

Laksh Nanwani, Anmol Agarwal, Kanishk Jain, Raghav Prabhakar, Aaron Monis, Aditya Mathur, Krishna Murthy Jatavallabhula, A. H. Abdul Hafez, Vineet Gandhi, and K. Madhava Krishna. 2023. Instance-level semantic maps for vision language navigation. In 2023 32nd IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). IEEE.

OpenAI. 2022. Chatgpt blog post.

OpenAI. 2023. GPT-4V(ision) Technical Work and Authors.

OpenCompass Contributors. 2023. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/ opencompass.

Priyam Parashar, Vidhi Jain, Xiaohan Zhang, Jay Vakil, Sam Powers, Yonatan Bisk, and Chris Paxton. 2023. Slap: Spatial-language attention policies. In Proceedings of The 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pages 3571–3596. PMLR.

Lang Peng, Zhirong Chen, Zhangjie Fu, Pengpeng Liang, and Erkang Cheng. 2023. Bevsegformer: Bird’s eye view semantic segmentation from arbitrary camera rigs. In 2023 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV). IEEE.

Chengwei Qin, Aston Zhang, Zhuosheng Zhang, Jiaao Chen, Michihiro Yasunaga, and Diyi Yang. 2023a. Is ChatGPT a general-purpose natural language processing task solver? In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 1339–1384, Singapore. Association for Computational Linguistics.

Zequn Qin, Jingyu Chen, Chao Chen, Xiaozhi Chen, and Xi Li. 2023b. Unifusion: Unified multi-view fusion transformer for spatial-temporal representation in bird’s-eye-view. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 8690–8699.

Navid Rajabi and Jana Kosecka. 2023. Towards grounded visual spatial reasoning in multi-modal vision language models. Preprint, arXiv:2308.09778.

Julia Rozanova, Deborah Ferreira, Krishna Dubba, Weiwei Cheng, Dell Zhang, and Andre Freitas. 2021. Grounding natural language instructions: Can large language models capture spatial information? Preprint, arXiv:2109.08634.

Baifeng Shi, Ziyang Wu, Maolin Mao, Xin Wang, and Trevor Darrell. 2024. When do we not need larger vision models? Preprint, arXiv:2403.13043.

Zhengxiang Shi, Qiang Zhang, and Aldo Lipani. 2022. Stepgame: A new benchmark for robust multi-hop spatial reasoning in texts. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 11321–11329.

Aleksandar Shtedritski, Christian Rupprecht, and Andrea Vedaldi. 2023. What does clip know about a red circle? visual prompt engineering for vlms. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE.

Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondrus, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, and Dhruv Batra. 2021. Habitat 2.0: Training home assistants to rearrange their habitat. In Advances in Neural Information Processing Systems (NeurIPS).

Damien Teney, Lingqiao Liu, and Anton Van Den Hengel. 2017. Graph-structured representations for visual question answering. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3233–3241.

Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. 2024. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Preprint, arXiv:2404.02905.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu,

Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models. Preprint, arXiv:2307.09288.

David Unger, Nikhil Gosala, Varun Ravi Kumar, Shubhankar Borse, Abhinav Valada, and Senthil Yogamani. 2023. Multi-camera bird’s eye view perception for autonomous driving. Preprint, arXiv:2309.09080.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837. Curran Associates, Inc.

Jason Weston, Antoine Bordes, Sumit Chopra, and Tomás Mikolov. 2016. Towards ai-complete question answering: A set of prerequisite toy tasks. In 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings.

Jiayang Wu, Wensheng Gan, Zefeng Chen, Shicheng Wan, and Philip S. Yu. 2023. Multimodal large language models: A survey. Preprint, arXiv:2311.13165.

Wenshan Wu, Shaoguang Mao, Yadong Zhang, Yan Xia, Li Dong, Lei Cui, and Furu Wei. 2024. Visualizationof-thought elicits spatial reasoning in large language models. Preprint, arXiv:2404.03622.

Yutaro Yamada, Yihan Bao, Andrew Kyle Lampinen, Jungo Kasai, and Ilker Yildirim. 2024. Evaluating spatial understanding of large language models. Transactions on Machine Learning Research.

Cheng Yan, Guansong Pang, Lei Wang, Jile Jiao, Xuetao Feng, Chunhua Shen, and Jingjing Li. 2021. Bvperson: A large-scale dataset for bird-view person re-identification. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 10943–10952.

Zhun Yang, Adam Ishay, and Joohyung Lee. 2023. Coupling large language models with logic programming for robust and general reasoning from text. In Findings of the Association for Computational Linguistics: ACL 2023, pages 5186–5219, Toronto, Canada. Association for Computational Linguistics.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. 2023. A survey on multimodal large language models. Preprint, arXiv:2306.13549.

Chao Zhang, Mohan Li, Ignas Budvytis, and Stephan Liwicki. 2024a. Dialoc: An iterative approach to embodied dialog localization. Preprint, arXiv:2403.06846.

Xingxuan Zhang, Jiansheng Li, Wenjing Chu, Junjia Hai, Renzhe Xu, Yuqing Yang, Shikai Guan, Jiazheng Xu, and Peng Cui. 2024b. On the out-ofdistribution generalization of multimodal large language models. Preprint, arXiv:2402.06599.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A survey of large language models. Preprint, arXiv:2303.18223.

Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024. Gpt-4v(ision) is a generalist web agent, if grounded. Preprint, arXiv:2401.01614.

Ruiqi Zhong, Dhruba Ghosh, Dan Klein, and Jacob Steinhardt. 2021. Are larger pretrained language models uniformly better? comparing performance at the instance level. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 3813–3827, Online. Association for Computational Linguistics.

## A Additional Related Work

In addition to Section 2 which provides a brief overview of previous work most relevant to our work, for completeness we also provide additional related work focused on unimodal spatial reasoning from text only.

Spatial Reasoning on Text. Spatial reasoning has been investigated with the advancement of LLMs (Yamada et al., 2024). Various benchmarks have been proposed to evaluate models’ spatial reasoning abilities, including relative spatial relation recognition (Weston et al., 2016; Mirzaee et al., 2021; Shi et al., 2022), natural language navigation (Yamada et al., 2024), and planning (Momennejad et al., 2023). Mirzaee and Kordjamshidi (2022) suggest that introducing synthetic data of spatial reasoning when pre-training helps to improve the spatial awareness of the model. Yang et al. (2023) justify the feasibility of using a logical form as an intermediate representation to improve the spatial reasoning ability in easy scenarios. Instead of describing the spatial relations with natural language, Wu et al. (2024) feed the model with a 2D square grid similar to ASCII-art format and prove that visualising the reasoning procedure explicitly helps to improve the model’s ability in multi-hop spatial reasoning. Constrained by language de scriptions, most datasets focus on reasoning over symbols within simple scenarios (e.g. grid-based navigation) and are synthetically generated. However, real-life scenarios are often more complex and rich in physical semantics. This raises concerns about the models’ actual spatial reasoning abilities compared to their proficiency in under standing linguistic patterns.

## B Further Details on Dataset Construction

The TOPVIEWRS is derived from Matterport3D (Chang et al., 2017) and is supposed to be used for non-commercial academic use only, under the Term of Use (Matterport End User Licence Agreement For Academic Use of Model Data).

In addition to the main content in Section 4, we provide further details with regard to TOPVIEWRS dataset construction in what follows.

## B.1 Top-View Map Construction

To ensure high-quality top-view map representations, we exclude the 3D environments with low coverage of mesh grids. We also prefer environments that are single-floor, in order to avoid the obstruction of objects from different floors. After manually going through 90 building-scale 3D environments from Matterport3D (Chang et al., 2017), we select a total of 7 scenes: 17DRP5sb8fy, 2azQ1b91cZZ, 2t7WUuJeko7, 5LpN3gDmAk7, EU6Fwq7SyZv, 8WUmhLawc2A, i5noydFURQK.

Photo-Realistic Top-View Map. We extract realistic top-view maps using MeshLab by placing an orthographic camera on the top of the 3D scenes and taking a camera shot.

Semantic Top-View Map. We construct them using the Habitat simulation environment (Manolis Savva\* et al., 2019; Szot et al., 2021). For each building floor, Matterport3D contains the 2D and 3D semantic segmentation human annotations, which can be retrieved to identify the type of objects as well as the rooms. The 3D coordinates of the entity’s (object and room) center $( x _ { i } , y _ { i } , h _ { i } )$ and the size of the entity’s bounding box $( w _ { x } , w _ { y } , w _ { h } )$ can also be retrieved as part of the circumstantial information. This information is then used for the construction of the semantic top-view map.

When we obtain the object information for the purpose of constructing a top-view semantic map, we design certain rules to exclude specific types of objects from all 40 object annotation categories of Matterport3D. We believe these objects could either 1) be less meaningful in terms of semantics or 2) take up a large area in the semantic map, which obstructs other objects beneath. The filtered objects include:‘misc’, ‘ceiling’, ‘objects’, ‘floor’, ‘wall’, ‘void’, ‘curtain’, ‘column’, ‘beam’, ‘board panel’.

We also filter out the objects based on their heights $h _ { o b j }$ and sizes $w _ { o b j }$ compared to the rooms’ heights $h _ { r o o m }$ and sizes $w _ { r o o m }$ . We only keep the objects if they satisfy the following relations:

$$
\begin{array} { l l l } { 0 . 9 \times ( h _ { r o o m } - \displaystyle \frac { 1 } { 2 } w _ { r o o m } ) \leq h _ { o b j } - \displaystyle \frac { 1 } { 2 } w _ { o b j } } \\ { 1 . 1 \times ( h _ { r o o m } + \displaystyle \frac { 1 } { 2 } w _ { r o o m } ) \geq h _ { o b j } + \displaystyle \frac { 1 } { 2 } w _ { o b j } } \end{array}
$$

After having all the object annotations, we use the get\_topdown\_map API of the Habitat simulator to get the top-down map of the scene, which describes the navigable area and the overall shape of the environment, but without any object annotations. Based on this map, we then draw the bounding boxes with different colors to represent the objects in the environments. Considering that the objects on the top may obstruct the bottom objects in the top-view map, to mimic this characteristic, we create the semantic top-view map based on the heights of the objects, where lower objects are drawn first. Table 4 shows the mapping between the RGB values and object types used for the creation of a semantic top-view map in our work.

After having the top-view maps of the whole floor, we crop them into smaller rooms according to the region boundaries obtained from the Habitat simulator.

## B.2 Structured Question Framework Design

In order to minimize human labor and standardize the collection pipeline, we adopt the templatebased question generation method following the practice of Liu et al. (2023a); we design 15 different templates in total to construct the sub-tasks for each task. In particular, we consider benchmarking different perspectives of the model’s ability within each task in a fine-grained manner when designing the templates. The question templates are also multi-scale in terms of objects or rooms with full or partial top-view maps for Top-View Recognition, Top-View Localization and Static Spatial Reasoning. For Dynamic Spatial Reasoning, the designed questions evaluate the recognition and reasoning from the scale of single navigation points (Dynamic Action Counting and Spatial Localization) to the whole path (Dynamic Relative Spatial Reasoning).

Below we provide the designed templates for all 9 sub-tasks, with some examples shown in Figure 1. In what follows, we also introduce the logic for selecting the correct answer and other wrong choices when constructing the multiple-choice questions.

## B.2.1 Top-View Recognition

Table 5 shows the templates we use for the Top-View Recognition task. Considering that some objects and rooms may be hard to recognize from the top view, in addition to the set of filtered objects, we also remove some objects (‘picture’, ‘mirror’, ‘window’, ‘blinds’, ‘towel’, ‘furniture’, ‘door’, ‘tv\_monitor’, ‘cabinet’) and rooms (‘hallway’, ‘entryway/foyer/lobby’, ‘tv’) when we use the templates to generate questions.

## B.2.2 Top-View Localization

Table 6 shows the templates for the Top-View Localization task. For the objects, we adopt the same set as for Top-View Recognition. Concerning rooms, we define a set of rooms that are easy and natural to recognize for humans, spanning: ‘office’, ‘workout/gym/exercise’, ‘kitchen’, ‘bedroom’, ‘dining room’, ‘bar’, ‘balcony’, ‘toilet’, ‘bathroom’, ‘living room’, ‘stairs’.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>RGB Values          Label</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[31, 119,180]</td><td rowspan=1 colspan=1>void</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>[174, 199, 232]</td><td rowspan=1 colspan=1>wall</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>[255, 127, 14]</td><td rowspan=1 colspan=1>floor</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>[255, 187, 120]</td><td rowspan=1 colspan=1>chair</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[44, 160, 44]</td><td rowspan=1 colspan=1>door</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[152,223, 138]</td><td rowspan=1 colspan=1>table</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[214, 39, 40]</td><td rowspan=1 colspan=1>picture</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[255,152, 150]</td><td rowspan=1 colspan=1>cabinet</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[148, 103, 189]</td><td rowspan=1 colspan=1>cushion</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[197,176,213]</td><td rowspan=1 colspan=1>window</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[140,86,75]</td><td rowspan=1 colspan=1>sofa</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[196,156, 148]</td><td rowspan=1 colspan=1>bed</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[227, 119, 194]</td><td rowspan=1 colspan=1>curtain</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[247,182,210]</td><td rowspan=1 colspan=1>chest_ofdrawers</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[51, 105, 30]</td><td rowspan=1 colspan=1>plant</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[199, 199, 199]</td><td rowspan=1 colspan=1>sink</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[188,189, 34]</td><td rowspan=1 colspan=1>stairs</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>[219, 219, 141]</td><td rowspan=1 colspan=1>ceiling</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>[23, 190, 207]</td><td rowspan=1 colspan=1>toilet</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>[158,218,229]</td><td rowspan=1 colspan=1>stool</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>[57, 59, 121]</td><td rowspan=1 colspan=1>towel</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>[82, 84, 163]</td><td rowspan=1 colspan=1>mirror</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>[107,110,207]</td><td rowspan=1 colspan=1>tv_monitor</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>[156, 158, 222]</td><td rowspan=1 colspan=1>shower</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>[99, 121, 57]</td><td rowspan=1 colspan=1>column</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>[140, 162, 82]</td><td rowspan=1 colspan=1>bathtub</td></tr><tr><td rowspan=1 colspan=2>[</td><td rowspan=1 colspan=1>181,207, 107</td><td rowspan=1 colspan=1>]        counter</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>[206,219, 156]</td><td rowspan=1 colspan=1>fireplace</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>[140, 109, 49]</td><td rowspan=1 colspan=1>lighting</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>[189,158,57]</td><td rowspan=1 colspan=1>beam</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>[231, 186, 82]</td><td rowspan=1 colspan=1>railing</td></tr><tr><td rowspan=1 colspan=2>[</td><td rowspan=1 colspan=1>231,203, 148]</td><td rowspan=1 colspan=1>shelving</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>[132, 60, 57]</td><td rowspan=1 colspan=1>blinds</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>[173, 73, 74]</td><td rowspan=1 colspan=1>gym_equipment</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>[214,97, 107]</td><td rowspan=1 colspan=1>seating</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>[231, 150, 156]</td><td rowspan=1 colspan=1>board_panel</td></tr><tr><td rowspan=1 colspan=2>[</td><td rowspan=1 colspan=1>123, 65, 115]</td><td rowspan=1 colspan=1>furniture</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>[165,81,148]</td><td rowspan=1 colspan=1>appliances</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>[206, 109, 189]</td><td rowspan=1 colspan=1>clothes</td></tr><tr><td rowspan=1 colspan=2>[</td><td rowspan=1 colspan=1>222,158,214]</td><td rowspan=1 colspan=1>objects</td></tr></table>

Table 4: RGB values and corresponding labels.

<table><tr><td colspan="2">Object Recognition</td></tr><tr><td>Template 1</td><td>Which of the following objects are in the room?</td></tr><tr><td>Template 2</td><td>Which of the following objects are not in the room?</td></tr><tr><td>Scene Recognition</td><td></td></tr><tr><td>Template 1</td><td>What room is this?</td></tr><tr><td>Template 2</td><td>What types of rooms are included in the top-view map below?</td></tr></table>

Table 5: Templates for Object and Scene Recognition sub-tasks.

<table><tr><td rowspan=1 colspan=1>Object Localization</td></tr><tr><td rowspan=1 colspan=1>Template 1 Where is the &lt;object&gt; in the top-down map?</td></tr><tr><td rowspan=1 colspan=1>Scene Localization</td></tr><tr><td rowspan=1 colspan=1>Template 1 Where is the &lt;room&gt; in the top-down map?</td></tr><tr><td rowspan=1 colspan=1>Template 2 What objects does &lt;room&gt; have?</td></tr></table>

Table 6: Templates for Object and Scene Localization sub-tasks.

## B.2.3 Static Spatial Reasoning

Table 7 lists the templates for the Static Spatial Reasoning task. For rooms, we restrict the regions within the same range as in Top-View Localization. Concerning objects, we focus on the objects that are common and large enough to recognize in daily life, which includes: ‘chair’, ‘table’, ‘cushion’, ‘sofa’, ‘bed’, ‘chest\_of\_drawers’, ‘sink’, ‘toilet’, ‘bathtub’, ‘stool’, ‘plant’, ‘stairs’, ‘shower’, ‘fireplace’, ‘gym\_equipment’, ‘seating’.

## B.2.4 Dynamic Spatial Reasoning

For Dynamic Action Counting, we define that a valid turn should involve more than a 30-degree rotation. For Dynamic Relative Spatial Reasoning, the direction is also defined by the relative spatial relation between the starting point and ending point, where the spatial description is determined by 30- degree intervals.

Multiple-Choice Question-Answer Pairs. For the answer to the questions, because we have all the spatial information and semantic annotation of the objects in the scene, we write a set of rules with code for each type of question in order to automatically obtain the golden answer according to the simulation environments. For all the wrong choices in the multiple-choice settings, they are randomly chosen from other possible candidates of the same kind (e.g. objects, rooms, numbers, etc.). After having all the options for multiple-choice questions, we randomize the order of the options to make the correct choices evenly distributed among possible options A, B, C, and D.

<table><tr><td colspan="2">Scene Counting</td></tr><tr><td>Template 1</td><td>How many &lt;room&gt; are there in the map?</td></tr><tr><td>Relative Spatial Relation</td><td></td></tr><tr><td>Template 1</td><td>What&#x27;s &lt;object1&gt;&#x27;s relative spa- tial relation to &lt;object2&gt;?</td></tr><tr><td>Template 2</td><td>What&#x27;s &lt;room1&gt;&#x27;s relative spa- tial relation to &lt;room2&gt;?</td></tr></table>

Table 7: Templates for Scene Counting and Relative Spatial Relation sub-tasks.

<table><tr><td rowspan=1 colspan=1>Dynamic Action Counting</td></tr><tr><td rowspan=1 colspan=1>Template 1    How many turning &lt;action&gt; arethere along the path?</td></tr><tr><td rowspan=1 colspan=1>Dynamic Relative Spatial Reasoning</td></tr><tr><td rowspan=1 colspan=1>Template 1    Which direction does the naviga-tion path head for?</td></tr><tr><td rowspan=1 colspan=1>Dynamic Spatial Localization</td></tr><tr><td rowspan=1 colspan=1>Template 1    What rooms does the navigationpath pass by?</td></tr><tr><td rowspan=1 colspan=1>Template 2    At which room does the naviga-tion path &lt;action&gt;?</td></tr><tr><td rowspan=1 colspan=1>Template 3    At which object does the naviga-tion path &lt;action&gt;?</td></tr></table>

Table 8: Templates for Dynamic Action Counting, Dynamic Relative Spatial Reasoning, and Dynamic Spatial Localization sub-tasks

## B.3 Alignment with Human Judgments

In our preliminary quality control, we realized that semantic annotations of environments may sometimes be inaccurate. Moreover, even though we exclude some unreasonable objects, the top view of certain objects can sometimes be challenging to recognize, even for humans. To address these issues, we have implemented a second stage in our dataset creation process: alignment and verification based on human judgments.

When validating the automatically collected data, the human participants are supposed to check the correctness of the question-answer pair and choose one of the following four actions according to their own judgments: 1) skip the instance if it cannot be repaired and/or looks strange, 2) modify the pair by replacing the options or the entities in the question in order to make it answerable by humans, 3) correct the answer if it is wrong, 4) keep the data if it is answerable by humans and correct. In order to ensure the quality of the dataset, we communicated to the human participants that they are supposed to be cautious when ‘accepting’ a data point/instance. On a practical level, the participants may either discard this data point or modify the options of this data to make the correct choice more distinguishable by humans. This helps to exclude the data points where different human judges may diverge and thus ensure the alignment between the dataset and general human judgments. We assure that the alignment process does not include any information with regard to personal identification or offensive content.

In our experiments, we also provide the corresponding rules of how we obtain the answer for the model with textual description in the prompt (see Appendix C.2).

## B.4 Dataset Statistics

We provide further insight into different portions of the TOPVIEWRS dataset with regard to the object and room distribution in Figure 4, whereas statistics over different sub-tasks are provided in Table 9.

The visualization demonstrates that the objects or regions that are hard to recognize (e.g. gym equipment, utility room, etc.) have fewer occurrences in the dataset compared to those which are easier to identify and typically more common (e.g. bed, table, bedroom, etc.). Bed, chair and table are the top-3 most frequently mentioned objects and bedroom, dining room and living room are the most common regions in the dataset. Among all the spatial descriptions, the diagonal spatial relations (e.g. top right, up left) are more frequently referred to as the correct choice as relative spatial descriptions in Static Spatial Reasoning while being less frequently used as absolute spatial descriptions in Top-View Localization.

Regarding the dataset size per each sub-task, object-level recognition and localization take a large portion of data in the Top-View Recognition and Localization tasks. For Static Spatial Reasoning, reasoning over relative spatial relations takes the main part of the data. Dynamic Spatial Localization has the largest number of data instances overall. The numbers are different with realistic maps and semantic maps for each task. The disparity stems from the second stage of dataset creation, where the human annotators have excluded more data points associated with more complex, photorealistic maps due to various possible reasons.

## C Experiments: Additional Information

## C.1 Inference Parameters

We adopt most of the inference parameters for each model from the implementations of VLMEvalKit (OpenCompass Contributors, 2023). Table 10 shows the configuration of the inference process for different models. If not specified in Table 10, we use the default configuration in Huggingface.

## C.2 Prompts

Table 11 and 12 show the prompt templates of each task used in the main experiments (Table 1) with realistic and semantic top-view maps as visual input, respectively. Table 13 and 14 show the prompt templates used for Chain-of-Thought reasoning using realistic and semantic top-view maps (Table 3).

Within the prompt templates, <QUESTION> and <OPTIONS> are replaced with the question and option list $\textit { O } = \{ o _ { 0 } , o _ { 1 } , o _ { 2 } , o _ { 3 } \}$ (e.g. “A. bed; B. chair; C. table; D. cushion”). For semantic topview maps, <MAPPING> is replaced with the RGBobject mapping, as shown below.

(196, 156, 148) -> bed

(44, 160, 44) -> door

In the task of Dynamic Spatial Reasoning, <TASK-SPECIFIC INSTRUCTION> contains the rules of how we obtain the answer from the simulator for the sub-task Dynamic Action Counting, which is described as follows.

Suppose you are a navigation agent tracing the path. Your job is to assess whether there's a turn at each intermediate point and sum up the total turns for the final outcome.

![](images/d5c591b367043f19bd3a3920b5fb2d0894cac2a9be88d224fa1522e3104d9245.jpg)

Object Distribution of Correct Answer (Semantic map)  
![](images/f3124c9c058a12bb2f4c710c09503711f3cc2d0c70ef5298a1a3be3aa46c936d.jpg)

Room Distribution of Correct Answer (Realistic Map)  
![](images/acb76909ed4faa1eb14f45f63c282ff44a8a7bfeb908c6a7fb8a9d7d1ac8cbef.jpg)

Room Distribution of Correct Answer (Semantic Map)  
![](images/906493206dc929e0bce548f2ca17bcc0324a640bd0a6c91f9f53f9bd7f3fad72.jpg)  
Figure 4: Additional statistics of the TOPVIEWRS dataset.

<table><tr><td>Task</td><td>Sub-Task</td><td>Realistic</td><td>Semantic</td></tr><tr><td rowspan="2">TVR</td><td>Object Recognition</td><td>195</td><td>198</td></tr><tr><td>Scene Recognition</td><td>97</td><td>97</td></tr><tr><td rowspan="2">TVL</td><td>Object Localization</td><td>410</td><td>470</td></tr><tr><td>Scene Localization</td><td>100</td><td>97</td></tr><tr><td rowspan="2">SSR</td><td>Scene Counting</td><td>43</td><td>48</td></tr><tr><td>Relative Spatial Relation</td><td>1,004</td><td>1,245</td></tr><tr><td rowspan="3">DSR</td><td>Dynamic Action Counting</td><td>668</td><td>668</td></tr><tr><td>Dynamic Spatial Localization</td><td>2,436</td><td>2,436</td></tr><tr><td>Dynamic Relative Spatial Reasoning</td><td>586</td><td>586</td></tr><tr><td>Total</td><td></td><td>5,539</td><td>5,845</td></tr></table>

Table 9: Distribution of sub-tasks with realistic and semantic top-view maps.

<table><tr><td rowspan=1 colspan=1>Idefics 9B&amp;80B</td></tr><tr><td rowspan=1 colspan=1>max_new_tokens    20</td></tr><tr><td rowspan=1 colspan=1>LLaVANext 7B&amp;13B&amp;34 B</td></tr><tr><td rowspan=1 colspan=1>temperature       0</td></tr><tr><td rowspan=1 colspan=1>num_beams       1</td></tr><tr><td rowspan=1 colspan=1>max_new_tokens    20</td></tr><tr><td rowspan=1 colspan=1>do_sample      False</td></tr><tr><td rowspan=1 colspan=1>top_p        None</td></tr><tr><td rowspan=1 colspan=1>XComposer2</td></tr><tr><td rowspan=1 colspan=1>temperature        1</td></tr><tr><td rowspan=1 colspan=1>beams         5</td></tr><tr><td rowspan=1 colspan=1>max_token       20</td></tr><tr><td rowspan=1 colspan=1>repetition_penalty    1</td></tr><tr><td rowspan=1 colspan=1>do_sample      False</td></tr><tr><td rowspan=1 colspan=1>Qwen-VL</td></tr><tr><td rowspan=1 colspan=1>max_new_tokens    20</td></tr><tr><td rowspan=1 colspan=1>GPT4V</td></tr><tr><td rowspan=1 colspan=1>temperature       0</td></tr><tr><td rowspan=1 colspan=1>max_tokens      1024</td></tr><tr><td rowspan=1 colspan=1>img_size       512</td></tr><tr><td rowspan=1 colspan=1>img_detail      low</td></tr><tr><td rowspan=1 colspan=1>Gemini</td></tr><tr><td rowspan=1 colspan=1>temperature       0</td></tr><tr><td rowspan=1 colspan=1>max_tokens      1024</td></tr></table>

Table 10: Configurations of inference parameters.

For other sub-tasks in Dynamic Spatial Reasoning, <TASK-SPECIFIC INSTRUCTION> is replaced with an empty string.

## C.3 Additional Experimental Results

Table 15 shows the fine-grained sub-task performance of all the models, which corresponds to Figure 3 in the main paper.

![](images/dc736c80783de5f30210cd4344e9d0623f61503d05a63db4f3cc4829cbbdc53c.jpg)  
Table 11: Prompt templates for main experiments with realistic top-view maps.

![](images/bca7b33e7639ee02e5b3d51301a7f60b88e4566c9a650739b671d2c501a58db2.jpg)  
Table 12: Prompt templates for main experiments with semantic top-view maps.

![](images/e67c28076d871c9032fdc14a4fd5fbdf181694522f2068067865c813e5023def.jpg)  
Table 13: Prompt templates for Chain-of-Thought experiments with realistic top-view maps.

![](images/7ccbd9bc09aa64cbed570ca8477d0995cec933179cb38e46ca46d74a2cd66bd2.jpg)  
Table 14: Prompt templates for Chain-of-Thought experiments with semantic top-view maps.

<table><tr><td colspan="2"></td><td colspan="2">Idefics</td><td colspan="4">LLaVANext</td><td rowspan="2">XComposer2 7B</td><td rowspan="2">Qwen-VL 7B</td><td rowspan="2">GPT-4V API</td><td rowspan="2">Gemini API</td></tr><tr><td colspan="2">Model Size</td><td colspan="2">9B 80B</td><td colspan="2">vicuna 7B mistral 7B vicuna 13B</td></tr><tr><td colspan="8"></td></tr><tr><td>Realistic Map</td><td></td><td>25.64</td><td>66.15</td><td>61.03</td><td>58.97</td><td>65.64</td><td>38.97</td><td>17.95</td><td>68.21</td><td></td><td>89.23</td></tr><tr><td>TVR</td><td>Object Recognition Scene Recognition</td><td>32.31 58.76</td><td>28.87</td><td>70.10</td><td>61.86</td><td>67.01</td><td>72.16</td><td>35.05</td><td>45.36</td><td>72.16</td><td>92.78</td></tr><tr><td rowspan="2">TVL</td><td>Object Localization</td><td>26.83</td><td>26.10</td><td>40.24</td><td>30.24</td><td>40.00</td><td>50.49</td><td>26.34</td><td>16.34</td><td>40.73</td><td>45.21</td></tr><tr><td>Scene Localization</td><td>45.00</td><td>46.00</td><td>50.00</td><td>49.00</td><td>46.00</td><td>53.00</td><td>34.00</td><td>16.00</td><td>69.00</td><td>61.00</td></tr><tr><td rowspan="2">SSR</td><td>Scene Counting</td><td>25.58</td><td>32.56</td><td>16.28</td><td>18.60</td><td>2.33</td><td>16.28</td><td>25.58</td><td>48.84</td><td>76.74</td><td>53.49</td></tr><tr><td>Relative Spatial Relations</td><td>24.00</td><td>25.80</td><td>20.02</td><td>24.60</td><td>21.02</td><td>23.01</td><td>25.80</td><td>13.25</td><td>19.82</td><td>30.68</td></tr><tr><td rowspan="2">DSR</td><td>Dynamic Action Counting</td><td>31.89</td><td>26.80</td><td>27.54</td><td>26.95</td><td>25.30</td><td>27.40</td><td>27.54</td><td>32.34</td><td>22.01</td><td>26.95</td></tr><tr><td>Dynamic Spatial Localization</td><td>42.57</td><td>29.27</td><td>45.03</td><td>23.89</td><td>32.88</td><td>24.63</td><td>22.62</td><td>20.11</td><td>34.15</td><td>35.30</td></tr><tr><td>Dynamic Relative Spatial Reasoning</td><td></td><td>26.62</td><td>23.72</td><td>25.77</td><td>23.04</td><td>17.58</td><td>16.21</td><td>26.11</td><td>18.77</td><td>23.72</td><td>27.82</td></tr><tr><td colspan="2">Semantic Map</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">TVR</td><td>Object Recognition</td><td>54.55</td><td>51.01</td><td>92.93</td><td>87.37</td><td>92.42</td><td>98.48</td><td>50.00</td><td>12.63</td><td>100.00</td><td>99.49</td></tr><tr><td>Scene Recognition</td><td>73.20</td><td>76.29</td><td>80.41</td><td>64.95</td><td>79.38</td><td>86.60</td><td>28.87</td><td>34.02</td><td>91.75</td><td>85.57</td></tr><tr><td rowspan="2">TVL</td><td>Object Localization</td><td>28.51</td><td>23.19</td><td>22.98</td><td>28.94</td><td>10.85</td><td>34.47</td><td>24.47</td><td>6.17</td><td>41.49</td><td>31.28</td></tr><tr><td>Scene Localization</td><td>44.33</td><td>47.42</td><td>37.11</td><td>47.42</td><td>48.45</td><td>57.73</td><td>26.80</td><td>26.80</td><td>58.76</td><td>54.64</td></tr><tr><td rowspan="2">SSR</td><td>Scene Counting</td><td>37.50</td><td>50.00</td><td>12.50</td><td>10.42</td><td>6.25</td><td>4.17</td><td>14.58</td><td>22.92</td><td>37.50</td><td>20.83</td></tr><tr><td>Relative Spatial Relations</td><td>23.29</td><td>27.23</td><td>18.96</td><td>24.82</td><td>17.03</td><td>18.96</td><td>23.37</td><td>14.54</td><td>21.12</td><td>26.43</td></tr><tr><td rowspan="3">DSR</td><td>Dynamic Action Counting</td><td>36.68</td><td>27.69</td><td>33.38</td><td>26.80</td><td>28.89</td><td>25.30</td><td>27.25</td><td>30.39</td><td>22.90</td><td>26.95</td></tr><tr><td>Dynamic Spatial Localization</td><td>39.20</td><td>38.79</td><td>41.87</td><td>25.82</td><td>17.98</td><td>39.00</td><td>19.46</td><td>22.95</td><td>47.17</td><td>33.42</td></tr><tr><td>Dynamic Relative Spatial Reasoning</td><td>26.11</td><td>24.74</td><td>23.72</td><td>27.30</td><td>17.75</td><td>17.58</td><td>24.06</td><td>18.26</td><td>25.26</td><td>28.16</td></tr></table>

Table 15: Fine-grained results of 10 VLMs on different sub-tasks, corresponding to the visualization in Figure 3.