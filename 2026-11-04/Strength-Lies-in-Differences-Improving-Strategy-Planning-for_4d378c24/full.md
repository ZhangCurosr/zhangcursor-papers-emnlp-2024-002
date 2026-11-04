# Strength Lies in Differences! Improving Strategy Planning for Non-collaborative Dialogues via Diversified User Simulation

Tong Zhang♠♡, Chen Huang♠♡, Yang Deng♢, Hongru Liang♠♡, Jia Liu♣, Zujie Wen♣, Wenqiang Lei♠♡\*, Tat-Seng Chua<sup>⋆</sup>

Sichuan University  Singapore Management University

Ant Group, China ⋆ National University of Singapore

Engineering Research Center of Machine Learning and Industry Intelligence, Ministry of Education, China

{scu.zhangtong, huangc.scu}@gmail.com {lianghongru, wenqianglei}@scu.edu.cn {jianiu.lj, zujie.wzj}@antgroup.com ydeng@smu.edu.sg chuats@comp.nus.edu.sg

## Abstract

We investigate non-collaborative dialogue agents, which are expected to engage in strategic conversations with diverse users, for securing a mutual agreement that leans favorably towards the system’s objectives. This poses two main challenges for existing dialogue agents: 1) The inability to integrate user-specific characteristics into the strategic planning, and 2) The difficulty of training strategic planners that can be generalized to diverse users. To address these challenges, we propose TRIP to enhance the capability in tailored strategic planning, incorporating a user-aware strategic planning module and a population-based training paradigm. Through experiments on benchmark non-collaborative dialogue tasks, we demonstrate the effectiveness of TRIP in catering to diverse users.

## 1 Introduction

Non-collaborative dialogues, such as negotiation (He et al., 2018) and persuasion (Wang et al., 2019), occur when the agent and user hold conflicting interests (Deng et al., 2023a,b; Lei et al., 2022). Typically, both parties need to employ various strategies to achieve an agreement favorable to themselves (Keizer et al., 2017; Zhan et al., 2024). As user resistance varies depending on the agent’s strategies (Shi et al., 2019; Dutt et al., 2021), it is imperative for the agent to perform strategic planning tailored to diverse users. Relying on a one-sizefits-all strategy can leave the agent vulnerable to others taking advantage due to its lack of adaptability and flexibility (Yang et al., 2021; Deng et al., 2024; Xu et al., 2023).

Recent efforts have resorted to large language models (LLMs) as dialogue agents to perform noncollaborative tasks (Deng et al., 2023d; Fu et al.,

2023; Zhang et al., 2023a). They aim to guide the response of LLMs through mixed-initiative prompts (Chen et al., 2023; Deng et al., 2023d; Zhang et al., 2023a) or incorporating an external strategy planner (Yu et al., 2023; Deng et al., 2023e). However, these initiatives has been criticized regarding its performance in real-world scenarios (Deng et al., 2023e; Kwon et al., 2024), where users have various non-collaborative strategies. We attribute this outcome to the neglect of two crucial aspects: 1) Existing methods fail to incorporate explicit user-specific characteristics into their strategic planning, instead relying solely on the conversational history. Importantly, by creating informative representations of individual users, agents can adapt their behaviors and devise tailored strategies (Jang et al., 2020; Yang et al., 2021). 2) Their training paradigm fails to generate strategic planners that generalize well to diverse users. Their paradigms are oversimplified, relying on a single user simulator for interactive training. This simulator is restricted in generating varied noncollaborative behaviors, often exhibiting a focus on prioritizing user contentment (Zhang et al., 2023c; Durmus et al., 2023; Bianchi et al., 2024). Essentially, agents trained in this manner are accustomed to engage with a single user exclusively, leading to rigidity and obstinacy when encountering new users with different interaction behaviors (Wang et al., 2023; Safdari et al., 2023).

To provide more evidence for the above analysis, we establish an evaluation protocol, which situates diverse user simulators with varying noncollaborative behaviors. We investigate the limitations of current LLM-based dialogue agents on strategic planning (cf. Section 3 for details). The evaluation results clearly demonstrate that existing agents struggle to tailor their strategies for diverse users, leading to sub-optimal performances.

This limitation compromises the practical utility of these agents, both in functioning as a successful agent in conversational AI and in providing social skills training in pedagogy. The key challenges lie in making dialogue agents aware of diverse non-collaborative user behaviors and devising tailored strategies for individual users.

To tackle these challenges, we design a simple yet effective method, called TRIP, to improve LLMs’ capability in Tailored stRategIc Planning. TRIP includes a user-aware strategic planning module and a population-based training paradigm. Specifically, the strategic planning module incorporates user-specific characteristics into strategic planning using the Theory-of-Mind (ToM) (Premack and Woodruff, 1978; Wimmer and Perner, 1983). This involves analyzing users mental states and future possible actions during interactions to understand their interests (Yang et al., 2021; Chawla et al., 2023a). Moreover, instead of relying on a solitary user simulator, our populationbased training paradigm promotes the adaptation of the strategic planning module to various users, achieved by training it with more diverse user simulators. Each simulator is equipped with extensive sets of non-collaborative strategies and role-playing personas (Chen et al., 2024). As such, TRIP essentially manipulates the experience of the dialogue agent, enabling it to recognize the importance of tailoring strategies for individual users. Our key contributions are concluded below:

• We emphasize the significance of tailoring strategies for diverse users in non-collaborative dialogues. We verify the inadequacies of current LLM-based dialogue agents in this aspect.

• We propose TRIP to achieve tailored strategic planning, which includes a user-aware strategic planning module and a population-based training paradigm.

• We conduct experiments on benchmark noncollaborative dialogue tasks (i.e., negotiation and persuasion). Our findings suggest that TRIP is proficient in catering to diverse users using tailored strategies, consistently outperforming baselines across different tasks.

## 2 Related Work

Our research is closely tied to the strategic planning and training paradigms to address the noncollaborative tasks in the era of LLMs. We provide a literature review and highlight our differences.

Strategic planning for non-collaborative dialogues. Recent researches have introduced various methods based on LLMs to enhance their effectiveness in strategic planning. These methods can be categorized into two types: 1) Developing stimulus prompts to unleash the potential ofLLMs. (Chen et al., 2023) validate the effectiveness of using mixed-initiative prompts to tackle proactive dialogue challenges. (Deng et al., 2023d) and (Zhang et al., 2023a) encourage LLMs to engage in self reflection to plan their next actions. (Fu et al., 2023) employ self-play simulations to iteratively refine strategic planning by soliciting feedback from other LLMs. Nonetheless, as highlighted by (Deng et al. 2023e), the effectiveness of these approaches is impeded by non-trainable parameters. 2) Equip ping LLMs with an external strategy planner. The planner is capable of generating prompts at each turn, providing nuanced, instance-specific guidance and control over LLMs. This could be integrated using methods like Monte Carlo Tree Search (Yu et al., 2023) or a plug-in model (Deng et al., 2023e), which can be fine-tuned for improving the strategic planning capability without affecting the functionalities of LLM-powered dialogue agents. However, these methods still struggle to achieve promising re sults due to their inability to integrate user-specific characteristics into their strategic planning. Com plementary to (Deng et al., 2023e), our work inves tigates the importance of tailored strategic planning by modeling user-related characteristics explicitly. Training paradigms for non-collaborative dialogues. Current training paradigms involve the dialogue agent interacting with a single user simulator to enhance its strategic planning capabilities. In specific, (Chawla et al., 2023b) build a user simulator that mimics human-human dialogue data in a supervised manner, while (Yu et al., 2023; Deng et al., 2023e) resort to a role playing LLM-based user simulator. However, a single user simulator can only represent the behaviors of one or a type of users, potentially leading to the under-representation of other users’ behaviors, as evidenced by (Liu et al., 2023; Shi et al., 2019). Therefore, existing training paradigms fail to produce strategic planners that cater to diverse users with varying behaviors. In this paper, our work investigates the importance of tailored strate gic planning by diversifying the user’s behaviors using population-based training.

![](images/5b2136c631a0c8050e8103f49ec1150ce5a36e17e3f72152ca88f21f7af40441.jpg)  
Figure 1: The overall evaluation process.

## 3 Strategic Planning Evaluation

We introduce a novel evaluation protocol to analyze the limitations of existing LLM-based dialogue agents and highlight their inability to handle users exhibiting various non-collaborative behaviors. The overall evaluation process is illustrated in Figure 1. See more details of our evaluation protocol in Appendix A.

## 3.1 Evaluation Setup

Evaluation Overview. The environment encompasses various synthetic user simulators showcasing diverse non-collaborative behaviors. In the evaluation process, each dialogue agent must interact with these simulators (Deng et al., 2023e). During their interactions, the dialogue agent and user simulator alternate in employing strategies in their responses with the ultimate aim of maximizing their own self-interest. The interactions continues until the conversational goal is achieved or the maximum number of turns is reached. We gather these interactions and assess the agents performances.

Baselines. We consider two representative baselines: Standard agent (i.e., vanilla LLM without any modification) and PPDPP agent (Deng et al., 2023e), which is current SOTA agent with a trainable external strategy planner<sup>1</sup>.

Diverse User Simulators. Our simulators are synthesized with non-collaborative behaviors, guided by their task-relevant personas. As evidenced by previous study (Deng et al., 2023c; Bianchi et al., 2024; Huang et al., 2024), LLMs are limited to demonstrate non-collaborative behaviors. To this end, we prompt non-collaborative behaviors explicitly into LLMs using the resisting strategies that are designed to foil persuasion attempts (Fransen et al., 2015; Tian et al., 2020; Dutt et al., 2021). Initially, we equip LLMs with different personas (Jiang et al., 2023; Zhou et al., 2023b; Zhang et al., 2023b), which are used to select non-collaborative behaviors from the set of resisting strategies. Following (Wang et al., 2019; Jiang et al., 2024), we consider two types of personas, including Big-Five Personality<sup>2</sup> (Goldberg, 1992) and Decision-Making Styles<sup>3</sup> (Scott and Bruce, 1995), together with LLM-generated cohesive description for each fine-grained persona. Additionally, we employ resisting strategies outlined by (Dutt et al., 2021) to direct the behavior of simulators. Finally, our mixed-initiative role-play prompt for each agent includes the assigned persona, a set of resisting strategies, and conversation context. These elements aid in guiding user simulators to exhibit diverse noncollaborative behaviors. In total, we develop 300 diverse user simulators for each evaluation task, representing 20 persona categories (i.e., Big-Five Personality Decision-Making Styles).

Evaluation Tasks. In line with (Deng et al., 2023d; Wang et al., 2019), we conduct experiments on two benchmark non-collaborative tasks: the price negotiation task, utilizing the $\mathrm { t e s t } ^ { 4 }$ dataset of Craigslist-Bargain (CB) (He et al., 2018) and the charity persuasion task, employing the test dataset of PersuasionForGood (P4G) (Wang et al., 2019). Notably, the dialogue agents play the role of buyer and persuader, respectively, to accomplish their goals.

Evaluation Metrics. Following (Deng et al., 2023e), we consider three commonly used metrics: Success Rate (SR), Average Turn (AT) and Sale-to-List Ratio (SL%). The SR measures effectiveness by the percentage of goal achievement within a maximum number of turns, while AT measures efficiency by the average number of turns required to achieve the goal. As for the CB task, we additionally adopt the SL% (Zhou et al., 2019) to determine the effectiveness of goal completion. Formally, the SL% is expressed as $( P _ { d e a l } - P _ { t a r g e t } ^ { s e l l e r } ) / ( P _ { t a r g e t } ^ { b u y e r } - P _ { t a r g e t } ^ { s e l l e r } )$ , where $P _ { d e a l }$ is the final deal price, $P _ { t a r g e t } ^ { b u y e r }$ and $P _ { t a r g e t } ^ { s e l l e r }$ are the target prices of both parties. A higher SL% represents the buyer gets more benefits from the deal. If failing to reach a deal at the end, we set SL% as 0.

<table><tr><td colspan="2">Personas</td><td colspan="3">Price Negotiation</td><td colspan="2">Persuasion for Good</td></tr><tr><td colspan="2"></td><td> $\mathrm { S R \uparrow }$ </td><td> $\operatorname { A T } \downarrow$ </td><td> ${ \mathrm { S L } } \% \uparrow$ </td><td>SR↑</td><td>AT↓</td></tr><tr><td rowspan="5">Big Five</td><td>Openness</td><td> $\overline { { 0 . 7 6 _ { \uparrow 0 . 2 3 } } }$ </td><td> $\overline { { 6 . 6 6 _ { \uparrow 0 . 6 3 } } }$ </td><td> $\overline { { 0 . 3 4 _ { \uparrow 0 . 1 2 } } }$ </td><td> $\overline { { 0 . 4 7 _ { \uparrow 0 . 3 4 } } }$ </td><td> $\overline { { 8 . 9 2 _ { \uparrow 1 . 0 0 } } }$ </td></tr><tr><td>Conscientiousness</td><td> $0 . 6 9 _ { \uparrow 0 . 2 5 }$ </td><td> $7 . 2 0 _ { \uparrow 1 . 0 4 }$ </td><td> $0 . 2 7 _ { \uparrow 0 . 0 6 }$ </td><td> $0 . 3 9 _ { \uparrow 0 . 3 3 }$ </td><td> $8 . 9 0 _ { \uparrow 1 . 1 0 }$ </td></tr><tr><td>Extraversion</td><td> $0 . 7 4 _ { \uparrow 0 . 1 6 }$ </td><td> $6 . 1 7 _ { \uparrow 1 . 4 7 }$ </td><td> $0 . 3 9 _ { \uparrow 0 . 1 5 }$ </td><td> $0 . 4 5 _ { \uparrow 0 . 3 5 }$ </td><td> $8 . 7 3 _ { \uparrow 1 . 2 5 }$ </td></tr><tr><td>Agreeableness</td><td> $0 . 4 0 _ { \uparrow 0 . 0 1 } \star$ </td><td> $6 . 8 2 _ { \uparrow 0 . 7 1 }$ </td><td> $0 . 2 8 _ { \uparrow 0 . 0 6 }$ </td><td> $0 . 1 8 _ { \uparrow 0 . 1 2 }$ </td><td> $9 . 8 5 _ { \uparrow 0 . 1 3 ^ { \star } }$ </td></tr><tr><td>Neuroticism</td><td> $0 . 3 1 _ { \downarrow 0 . 0 2 } \star$ </td><td> $\underline { { 6 . 8 1 _ { \uparrow 1 . 1 2 } } }$ </td><td> $0 . 2 0 _ { \downarrow 0 . 0 2 } \star$ </td><td> $0 . 1 2 _ { \uparrow 0 . 0 2 } \star$ </td><td> $9 . 7 8 _ { \uparrow 0 . 1 4 } \star$ </td></tr><tr><td rowspan="4">Decision</td><td>Analytical</td><td> $\overline { { 0 . 3 7 _ { \uparrow 0 . 0 4 } \star } }$ </td><td> $\overline { { 7 . 0 7 _ { \uparrow 0 . 6 1 } } }$ </td><td> $0 . 2 6 _ { \uparrow 0 . 0 6 } \star$ </td><td> $\overline { { 0 . 1 6 _ { \uparrow 0 . 0 9 } } }$ </td><td> $\overline { { 9 . 4 3 _ { \uparrow 0 . 5 6 } \star } }$ </td></tr><tr><td>Directive</td><td> $0 . 4 1 _ { \uparrow 0 . 0 5 } \star$ </td><td> $6 . 7 1 \substack { \uparrow 1 . 4 8 }$ </td><td> $0 . 1 8 _ { \downarrow 0 . 0 3 } \star$ </td><td> $0 . 1 2 _ { \downarrow 0 . 0 2 } \star$ </td><td> $9 . 3 1 _ { \uparrow 0 . 6 2 }$ </td></tr><tr><td>Behavioral</td><td> $0 . 7 8 _ { \uparrow 0 . 2 5 }$ </td><td> $6 . 4 5 _ { \uparrow 1 . 2 0 }$ </td><td> $0 . 3 9 _ { \uparrow 0 . 1 6 }$ </td><td> $0 . 5 3 _ { \uparrow 0 . 3 7 }$ </td><td> $8 . 9 4 _ { \uparrow 1 . 0 4 }$ </td></tr><tr><td>Conceptual</td><td> $0 . 7 7 _ { \uparrow 0 . 2 3 }$ </td><td> $6 . 6 2 _ { \uparrow 0 . 7 8 }$ </td><td> $0 . 4 2 _ { \uparrow 0 . 1 7 }$ </td><td> $0 . 4 9 _ { \uparrow 0 . 3 6 }$ </td><td> $9 . 0 2 _ { \uparrow 0 . 9 4 }$ </td></tr><tr><td colspan="2">Overall Performance</td><td> $\overline { { 0 . 5 8 _ { \uparrow 0 . 1 4 } } }$ </td><td> $\overline { { 6 . 7 2 _ { \uparrow 1 . 0 1 } } }$ </td><td> $\overline { { 0 . 3 1 _ { \uparrow 0 . 0 9 } } }$ </td><td> $\overline { { 0 . 3 2 _ { \uparrow 0 . 2 3 } } }$ </td><td> $9 . 2 0 _ { \uparrow 0 . 7 6 }$ </td></tr></table>

Table 1: The performance of the PPDPP dialogue agent testing across various personas of user simulators. Red (Blue) indicates the increased (decreased) performance compared to Standard dialogue agent. The symbol ⋆ indicates that this performance exhibits minimal variation, specifically within a 5% range of the maximum value. The effectiveness of PPDPP varies significantly across different user personas.

## 3.2 Experimental Findings

We analyze the performances of existing dialogue agents across user simulators with various noncollaborative behaviors. Specifically, we assess the advancements of PPDPP compared to the Standard agent. As illustrated in Table 1, while PPDPP shows a notable improvement in overall performance, it does not adapt well to users employing different non-collaborative strategies. Its effectiveness varies significantly among users with different personas, with its advantage over the Standard not being significant in 17.77% of cases (e.g., it increases SR by 0.02 for Analytical in price negotiation.), and even performing worse than the Standard in 8.88% of cases (e.g., it decreases SR by 0.02 for Neuroticism in price negotiation). This motivates the need for a dialogue agent to perform strategic planning tailored to diverse users<sup>5</sup>.

## 4 TRIP: Tailored Strategic Planning

To enhance LLMs’ tailored strategic planning, we propose an effective method TRIP, which develops an external planner by modeling user characteristics and training with diverse user simulators. As illustrated in Figure 2, our TRIP includes a useraware strategic planning module and a populationbased training paradigm. The former aims to explicitly model user characteristics (e.g., mental states and future actions), while the latter incorporates diverse user simulators for training simultaneously.

## 4.1 User-Aware Strategic Planning

TRIP aims to explicitly infer user characteristics and then incorporate them into the strategic planning module, parameterized by a trainable BERT. In particular, building upon the advanced Theoryof-Mind capability of LLMs (Sap et al., 2022; Moghaddam and Honey, 2023), TRIP captures users’ mental states and future possible actions during interactions to understand their interests and predicts how $\mathrm { T R I P } ^ { \prime } \mathrm { s }$ responses may influence them. In this case, mental states pertains to what they aim to accomplish, such as the target price or whether they will donate, while future actions relates to what the user is likely to discuss next (Hu et al., 2023; Zhou et al., 2023a). Formally, given the dialogue history $D = \left( u _ { 1 } ^ { s y s } , u _ { 1 } ^ { u s r } , . . . , u _ { t } ^ { s y s } , u _ { t } ^ { u s r } \right)$ where $u _ { i } ^ { s y s }$ and $u _ { i } ^ { u s r }$ denote the i-th utterances of both parties and t is the number of utterances, we feed the dialogue history D into the LLM and prompt it to infer mental states and future actions in an open-ended manner, $. . . , P _ { L L M } ( \mathcal { M } , \mathcal { F } | D )$ . Subsequently, we feed the $\{ \mathcal { M } , \mathcal { F } , D \}$ into the strategy planner π to predict the next strategy. The output space of π<sub>θ</sub> is a set of strategies<sup>6</sup> pre-defined by (Deng et al., 2023e; Wang et al., 2019), each of them is attached with a pre-defined natural language instructions.

## 4.2 Population-based Training Paradigm

Given that a single user simulator tends to favor limited behaviors while under-represents others (Shi et al., 2019; Liu et al., 2023), we explore training a dialogue agent using a set of user simulators employing different non-collaborative strategies to accommodate diverse users. To achieve this, we propose a population-based reinforcement learning (RL) training paradigm, which aims to enhance the adaptability of a dialogue agent to new user groups by training with larger and more diverse populations (Charakorn et al., 2020). We offer a comprehensive explanation of this approach below. Population Setup. Similar to Section 3.1, we build 40 diverse user simulators, each embodying a specific persona description. We ensure an balanced representation of each persona category within our user simulators for population-based RL training. We donate these simulators as $K = k _ { 1 } , k _ { 2 } , . . . k _ { 4 0 }$ During each iteration, we sample among K using a distribution $p ,$ allowing the dialogue agent S to interact with it. The distribution $p$ is initialized based on the frequency of various personas.

![](images/0d44e8d5b7ab1d7c65b1a1c2618012d4e1e5a112719fb7bc28fe8d4031bec82a.jpg)  
Figure 2: TRIP Overview. This method includes a user-aware strategic planning module (UASP) and a populationbased training paradigm (PBTP). The UASP incorporates user-specific characteristics into strategic planning using the Theory-of-Mind (ToM). The PBTP diversifies training user simulators to promote agents’ adaptation. We use numbers to indicate the overall process of TRIP.

Reward Design. Following (Deng et al., 2023e), we prompt LLMs to judge the conversation progress at each turn and transform it into scalar rewards. Specifically, in the negotiation task, we employ a separate GPT3.5 (OpenAI, 2022) to assess whether both parties have reached a deal. In the persuasion task, we ask the GPT3.5-based user simulator to express its willingness to donation. Our rewards are determined based on three situations: 1) Successful goal achievement by the dialogue agent results in a significant positive reward, defined as 1.0 in the charity persuasion task and the value of SL% in the price negotiation task. 2) Failure to achieve goals leads to a substantial negative reward of -1.0 for the dialogue agent. 3) Furthermore, we assign a small negative reward (-0.1) per turn to penalize the lengthy conversation, which promotes the efficient goal achievement.

Optimization. During RL training, we maximize the expected reward of the strategy planner $\pi _ { \theta }$ by utilizing the REINFORCE algorithm (Williams, 1992): θ  θ  α log $\pi _ { \theta } R _ { t }$ , where θ denotes the trainable parameter of the strategy planner, α denotes the learning rate, and $R _ { t }$ is the total reward accumulating from turn t to the final turn T: $R _ { t } =$ $\scriptstyle \sum _ { t ^ { \prime } = t } ^ { T } \gamma ^ { T - t ^ { \prime } } r _ { t ^ { \prime } }$ , where $\gamma$ is a discount factor.

## 5 Experiments

This sections aims to evaluate the effectiveness of our TRIP, following the evaluation protocol proposed in Section 3.1. We initially report the overall performances of dialogue agents in Section 5.1. Next, we conduct an in-depth analysis to reveal the tailored strategies of TRIP in Section 5.2. Finally, we perform ablation studies in Section 5.3 to sort out the performance variation of different user awareness and training population, and find a dominant predictor for the tailored strategic planning. LLM-based baselines. We consider LLM-based dialogue agents with two types of strategic planning modules, as discussed in Section 2. 1) Prompt-based planning, including Standard, Pro-CoT (Deng et al., 2023d) and ICL-AIF (Fu et al., 2023), which use mixed-initiative prompts, CoT, and AI feedback to select next strategies, respectively. 2) External strategy planners, including GDP-MCTS (Yu et al., 2023) and PPDPP (Deng et al., 2023e), which utilize Monte Carlo Tree Search and a trainable plug-in for determining nextstep strategies, respectively. Note that all baselines fail to model user-specific characteristics explicitly and are trained using one user simulator. Implementation details are presented in Appendix B. Evaluation Metrics. We use the same automatic metrics mentioned in section 3.1. Furthermore, we conduct human evaluation to assess the practical effectiveness of these dialogue agents. See more details of human evaluation in Appendix C.

![](images/93fc5e105461bc7171cd5461dfec96255a602f4e7ce7c074d26b2efc623f9e7e.jpg)

![](images/bc38405e4845ba0a82f14c5a5bd82cfa7007c94e0fe9eb0854e7c1550ffe12a8.jpg)  
Figure 3: The agents performance across various personas. We report their success rate on two tasks, namely price negotiation (Left) and charity persuasion (Right). TRIP achieves balanced improvements on all personas, significantly outperforming other agents by a considerable margin. Due to limited space, we report other results using different metrics in Appendix D.

## 5.1 Overall Performance

We evaluate the overall and fine-grained performance of all agents using automatic metrics in Table 2 and Figure 3. Additionally, we report human evaluation in Figure 4 to gauge their performance during interactions with real users.

TRIP is a promising method for achieving effective non-collaborative strategies tailored for diverse users. As illustrated in Table 2, TRIP significantly outperforms all the baselines with a noticeable margin across two tasks. It not only efficiently achieves the conversational goal (less AT) but also effectively accomplishes tasks (higher SR and higher SL%). Moreover, as depicted in Figure 3, TRIP shows balanced improvements across different user personas, significantly outperforming other agents by a substantial margin, in contrast to the biased improvements of PPDPP in Section 3.2. This suggests that TRIP is capable of generating strategies that generalize well to diverse users. This also implies that the behavior pattern pf a single LLM-based user simulator is limited in scope. Moreover, our human evaluation results in Figure 4 show our TRIP largely outperform the Standard and PPDPP when interacting with real users. Notably, we observed that PPDPP does not consistently surpass the Standard approach across the two tasks. For instance, while it achieves a higher success rate in the negotiation task, it necessitates more interaction rounds. This evidences the effectiveness and practical utility of our proposed TRIP.

<table><tr><td rowspan="2">Agents</td><td colspan="3">Price Negotiation</td><td colspan="2">Persuasion for Good</td></tr><tr><td>SR↑</td><td>AT↓</td><td>SL%↑</td><td>SR↑</td><td>AT↓</td></tr><tr><td>Standard</td><td>0.4444</td><td>7.73</td><td>0.2222</td><td>0.0930</td><td>9.96</td></tr><tr><td>ProCoT</td><td>0.6040</td><td>7.62</td><td>0.2307</td><td>0.1833</td><td>9.90</td></tr><tr><td>ICL-AIF</td><td>0.3411</td><td>8.42</td><td>0.2503</td><td>0.1667</td><td>9.91</td></tr><tr><td>GDP-MCTS</td><td>0.4444</td><td>7.63</td><td>0.2401</td><td>0.2466</td><td>9.74</td></tr><tr><td>PPDPP</td><td>0.5855</td><td>6.72</td><td>0.3144</td><td>0.3233</td><td>9.20</td></tr><tr><td>TRIP (Ours)</td><td>0.6888</td><td>6.34</td><td>0.4096</td><td>0.5533</td><td>8.51</td></tr></table>

Table 2: Overall evaluation. TRIP is promising for achieving effective non-collaborative strategies.

![](images/5f633fd3a4200951378b31a79a0433ac6b5ea662507dee5668b697b6ad8e0584.jpg)

![](images/7b778bafa839c28cf9d9b73e9f64e7ea33fa8a035f88072e52204979541f9b28.jpg)  
(a) Success Rate (Left) and Average Turn (Right) on Price Negotiation

![](images/df0ac4a00b434f56564748b77f2a6efedcaf318abc97f25dae250f03c5ecd0f8.jpg)

![](images/0547a4515146e0c76e22e7a1d8b8019f0d2baa9590e3f1310ad46f0a4b9b88df.jpg)  
(b) Success Rate (Left) and Average Turn (Right) on Charity Persuasion  
Figure 4: Human Evaluation Results. TRIP shows a high practical utility to deal with real users.

## 5.2 Strategy Analysis

In this section, we analyze the effectiveness of our TRIP in tailored strategic planning. Specifically, in each user interaction, we gather the strategies employed by each agent at every turn and combine them in a sequential order to form a strategy sequence. Then, we compare the strategy sequences employed by different agents. We utilize BERT (Devlin et al., 2018) and the t-SNE method (Van der Maaten and Hinton, 2008) to encode each strategy sequence into an embedding vector. Subsequently, we use the Euclidean distance measure to calculate the average distance between any two strategy sequences used by agents with the same persona, as well as the average distance between any two strategy sequences used by agents with different personas. This is akin to the metrics (i.e., the Intra-Class and Inter-Class analysis) used in the metric learning community (Roth et al., 2019) and we term them as the Intra-Persona and Inter-Persona. The results are shown in Table 3.

![](images/d171d8f98e2fc36fd494d2691498b5afa4b921a98a4dd7d2b3656f681b7e7520.jpg)  
Figure 5: Case study on the charity persuasion task (Top-3 conversation rounds). The user resisting strategies and agent strategies are marked in bleu and red respectively. While PPDPP repeats its strategy usage pattern to different user types, TRIP effectively tailor its strategies for different users. When dealing with theOpenness persona (Left), TRIP introduces the charitable organization and evoke specific emotions to sway users’ decision. Conversely, in addressing the Neuroticism persona (Right), TRIP tends to discuss personal experiences related to charity and employs reasoning persuade the user.

<table><tr><td>Models</td><td>Intra-Persona↓|</td><td>Inter-Persona↑</td></tr><tr><td>Standard</td><td>24.93</td><td>13.51</td></tr><tr><td>ProCoT</td><td>21.37</td><td>15.65</td></tr><tr><td>ICL-AIF</td><td>22.84</td><td>15.33</td></tr><tr><td>GDP-MCTS</td><td>20.72</td><td>16.09</td></tr><tr><td>PPDPP</td><td>19.37</td><td>17.28</td></tr><tr><td>TRIP (Ours)</td><td>16.14</td><td>20.26</td></tr></table>

Table 3: The strategy distribution of different agents. The Intra-Persona metric donates the average distance for a particular persona. The Inter-Persona metric donate the average distance for different personas. TRIP achieves the best performance, showcasing its effectiveness in devising tailored strategies for diverse users.

TRIP demonstrates a greater awareness of population dynamics, resulting in reduced variance across specific user simulators. As shown in Table 3, TRIP achieves the lowest Intra-Persona and the highest Inter-Persona. This indicates that the strategy sequences of TRIP exhibit similarity when interacting with users sharing the same personas and non-collaborative behaviors. Also, these sequences are distinct when compared to users with different personas. This further reveals that TRIP holds advantages in devising tailored strategies for diverse users.

For better understanding, we present a case study in Figure 5 and examine the strategy sequence employed by PPDPP and TRIP in an charity persuasion task. Specifically, PPDPP repeats its strategy usage pattern to different user types, briefly using of credentials and citing organizational impacts to establish credibility and earn the persuadee’s trust. In contrast, TRIP demonstrates a deeper understanding of the users and provides more tailored strategies. When dealing with the Neuroticism persona, TRIP tends to discuss personal experiences related to charity and employs reasoning persuade the user. Conversely, in addressing the Openness persona, TRIP introduces the charitable organization and evoke specific emotions to sway users’ decision. The strategy sequence used by TRIP is believed to be more persuasive, as demonstrated by (Barford and Smillie, 2016; Wang et al., 2019), stating that the Openness users are inclined to embrace novelty and be easily influenced by emotions, while the Neuroticism users are more likely to be influenced by others’ personal experiences. In this regard, we believe that these strategic differences may provide valuable insights for the future research on the non-collaborative dialogues.

<table><tr><td rowspan="2">Models</td><td colspan="3">Price Negotiation</td><td colspan="2">Persuasion for Good</td></tr><tr><td>SR↑</td><td>AT↓</td><td>SL%↑</td><td>SR↑</td><td>AT↓</td></tr><tr><td>TRIP</td><td>0.6888</td><td>6.34</td><td>0.4096</td><td>0.5533</td><td>8.51</td></tr><tr><td> $\mathbf { T R I P _ { w / o } U A }$ </td><td>0.6988</td><td>6.38</td><td>0.3881</td><td>0.5133</td><td>8.69</td></tr><tr><td>TRIPW/o POP</td><td>0.5766</td><td>7.00</td><td>0.3505</td><td>0.4400</td><td>8.95</td></tr><tr><td>TRIPW/ 10 POP &amp; W/o UA</td><td>0.6377</td><td>6.73</td><td>0.3543</td><td>0.4700</td><td>8.79</td></tr><tr><td>TRIPW/ 10 POP</td><td>0.6700</td><td>6.12</td><td>0.3537</td><td>0.4733</td><td>8.72</td></tr><tr><td>PPDPP</td><td>0.5855</td><td>6.72</td><td>0.3144</td><td>0.3233</td><td>9.20</td></tr></table>

Table 4: The evaluation results of ablation study. The user-aware strategic planning module and populationbased training are effective to improve agents and complement each other.

## 5.3 Ablation Study

This section aims to sort out the performance variation of different user awareness and training population. To analyze the effectiveness of each design, we consider the following variants of TRIP.

$\mathbf { T R I P _ { w / o } P O P } \mathrm { : }$ We eliminate the population-based training approach from TRIP and instead have TRIP engage with a single fixed LLM-based user simulator for training, without any specific roleplaying persona.

$\mathbf { T R I P _ { w / o } U A } \colon$ We remove the user-aware strategic planning module, and only takes the conversation history as inputs to plan next strategies.

$\bf { T R I P } _ { w / }$ <sub>10 POP</sub>: It utilizes 10 personas for population training, each simulator is randomly selected from a pool of 20 persona categories.

• TRIP<sub>w/ 10 POP & w/o UA</sub>: In this variant, we remove the user-aware strategic planning module from TRIP w/ 10 POP.

We summarize the overall performance of each model variation Table 4. Based on these results, we draw the following observations:

User-aware strategic planning and populationbased training paradigm are both effective to produce tailored strategic planning. Specifically, compared to $\mathrm { T R I P _ { w / o } U A } .$ , we note TRIP improves the persuasion success rate $( 0 . 3 2 3 3  0 . 4 4 0 0 )$ and the deal benefit SL% $( 0 . 3 1 4 4  0 . 3 5 0 5 )$ . This suggest that incorporating user mental states and future actions can assist the agent in developing more effective strategies. Notably, this variant slightly decreases the deal success rate $( 0 . 6 9 8 8 \to 0 . 6 8 8 8 )$ This can be attributed to the fact that deeply modeling user characteristics may inadvertently decrease the seller’s willingness to engage in the deal, as the focus is on maximizing one’s own benefits. Moreover, compared to $\mathrm { T R I P } _ { \mathrm { w / o } } \mathrm { P O P } .$ , we observe that TRIP yield positive improvements across all metrics, such as significant increase in SL% (0.3505 $ 0 . 4 0 9 6 )$ . This demonstrates that diversifying the behaviors of training user simulators effectively improves the agent’s performance.

![](images/b20ef067e3c2a066d67a9b4c262afc2b79a8e1d5746e5e3e1de616733df98c5e.jpg)

![](images/45540751510289c464f3b4e0e881dceb9156080bf24bc5e00018df5a7d3f7dcc.jpg)  
Figure 6: The test performance of different number of training user simulators. PPDPP converges easily but has a limited upper bound in terms of performance.

Diverse training populations is more beneficial to improve the adaptability of dialogue agents, but it may also present additional training challenges. As shown in Table 4, compared to $\mathrm { T R I P _ { w / o } U A }$ and $\mathrm { T R I P _ { w / o } }$ <sub>POP</sub>, we find that diverse training populations is more important for $\mathrm { T R I P } ^ { * } \mathrm { s }$ superiority. Moreover, we find that $\mathrm { T R I P _ { w / o } U A }$ demonstrates higher performances than $\mathrm { T R I P } _ { \mathrm { w } } /$ <sub>10 POP & w/o UA</sub> and PPDPP (i.e., A single fixed user simulator). To provide a detailed understanding of the impact of the number of training user simulators, we present their test performance of in 1000 training interactions, as depicted in Figure 6. Particularly, during the initial 400 interactions, we observe that $\mathrm { T R I P _ { w / o } U A }$ and $\mathrm { T R I P _ { w } } /$ <sub>10 POP & w/o UA</sub> exhibit slower convergence compared to PPDPP. This suggests that not keeping the training user simulator fixed can introduce instability in the initial training phase, as also noted in (Lewis et al., 2017). However, beyond 500 interactions, the training process of $\mathrm { T R I P _ { w / o } U A }$ stabilizes, leading to a significant performance enhancement, surpassing the other two agents. Additionally, it is observed that $P P D P P \mathrm { { s } }$ performance declines after specific interactions (e.g., 600 in price negotiation), suggesting that extensive interactions with a single user simulator cannot consistently enhance agents’ performance.

## 6 Conclusion

In this study, we investigate the inadequacies of current LLM-based dialogue agents in catering in diverse non-cooperative users. To address this, we propose TRIP, a method designed to tailor strategic planning for non-collaborative dialogues. The idea behind our TRIP is simple, involving a user-aware strategic planning module and a population-based training paradigm. Experimental results across diverse users demonstrate the superior effectiveness and efficiency of TRIP. We consider our work as laying the groundwork for enhancing the adaptability and flexibility of non-cooperative dialogue agents in the era of LLMs. Moving forward, we plan to further explore the potential of populationaware agents in reducing the capital expenditure associated with training and coaching novice agents.

## Limitations

In this section, we discuss the limitations of this work from the following perspectives:

Sensitivity of Prompts. Similar to other studies on prompting LLMs (Deng et al., 2023d), the evaluation results are expected to be influenced by the prompts. Following (Deng et al., 2023e), we employ the mixed-initiative format to formulate our prompts, as it offers stability and control. The impact of prompts and their optimality present important areas of investigation within LLMs, calling for exploration in future studies.

Limited Non-collaborative Tasks. We only conduct our experiments on the two non-collaborative dialogue tasks (i.e., price negotiation and charity persuasion) due to their status as classic and widely-recognized benchmarks (Deng et al., 2023d; Chawla et al., 2023a). In the future, we plan to apply our proposed TRIP in a broader range of non-collaborative dialogue scenarios (Zhang et al., 2024; Zhou et al., 2023b; Zhang et al., 2023b).

## Acknowledgements

This work was supported in part by the National Natural Science Foundation of China (No. 62272330 and No. 62206191); in part by the Natural Science Foundation of Sichuan (No. 2023NSFSC0473); in part by the Fundamental Research Funds for the Central Universities (No. 2023SCU12089 and No. YJ202219); in part by the Singapore Ministry of Education (MOE) Academic Research Fund (AcRF) Tier 1 grant (No. MSS24C004).

## References

Kate A Barford and Luke D Smillie. 2016. Openness and other big five traits in relation to dispositional

mixed emotions. Personality and individual differences, 102:118–122.

Federico Bianchi, Patrick John Chia, Mert Yuksekgonul, Jacopo Tagliabue, Dan Jurafsky, and James Zou. 2024. How well can llms negotiate? negotiationarena platform and analysis. arXiv preprint arXiv:2402.05863.

Rujikorn Charakorn, Poramate Manoonpong, and Nat Dilokthanakul. 2020. Investigating partner diversification methods in cooperative multi-agent deep reinforcement learning. In Neural Information Processing: 27th International Conference, ICONIP 2020, Bangkok, Thailand, November 18–22, 2020, Proceedings, Part V 27, pages 395–402. Springer.

Kushal Chawla, Weiyan Shi, Jingwen Zhang, Gale Lucas, Zhou Yu, and Jonathan Gratch. 2023a. Social influence dialogue systems: A survey of datasets and models for social influence tasks. In Proceedings ofthe 17th Conference ofthe European Chapter of the Association for Computational Linguistics, pages 750–766.

Kushal Chawla, Ian Wu, Yu Rong, Gale Lucas, and Jonathan Gratch. 2023b. Be selfish, but wisely: Investigating the impact of agent personality in mixedmotive human-agent interactions. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13078–13092, Singapore. Association for Computational Linguistics.

Maximillian Chen, Xiao Yu, Weiyan Shi, Urvi Awasthi, and Zhou Yu. 2023. Controllable mixed-initiative dialogue generation through prompting. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 951–966, Toronto, Canada. Association for Computational Linguistics.

Nuo Chen, Yan Wang, Yang Deng, and Jia Li. 2024. The oscars of ai theater: A survey on role-playing with language models.

Yang Deng, Wenqiang Lei, Minlie Huang, and Tat-Seng Chua. 2023a. Goal awareness for conversational AI: Proactivity, non-collaborativity, and beyond. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 6: Tutorial Abstracts), pages 1–10, Toronto, Canada. Association for Computational Linguistics.

Yang Deng, Wenqiang Lei, Minlie Huang, and Tat-Seng Chua. 2023b. Rethinking conversational agents in the era of llms: Proactivity, non-collaborativity, and beyond. In Proceedings ofthe Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, SIGIR-AP ’23, page 298–301, New York, NY, USA. Association for Computing Machinery.

Yang Deng, Wenqiang Lei, Wai Lam, and Tat-Seng Chua. 2023c. A survey on proactive dialogue systems: Problems, methods, and prospects. arXiv preprint arXiv:2305.02750.

Yang Deng, Wenqiang Lei, Lizi Liao, and Tat-Seng Chua. 2023d. Prompting and evaluating large language models for proactive dialogues: Clarification, target-guided, and non-collaboration.

Yang Deng, Lizi Liao, Zhonghua Zheng, Grace Hui Yang, and Tat-Seng Chua. 2024. Towards humancentered proactive conversational agents. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, page 807–818, New York, NY, USA. Association for Computing Machinery.

Yang Deng, Wenxuan Zhang, Wai Lam, See-Kiong Ng, and Tat-Seng Chua. 2023e. Plug-and-play policy planner for large language model powered dialogue agents. arXiv preprint arXiv:2311.00262.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Esin Durmus, Karina Nyugen, Thomas I Liao, Nicholas Schiefer, Amanda Askell, Anton Bakhtin, Carol Chen, Zac Hatfield-Dodds, Danny Hernandez, Nicholas Joseph, et al. 2023. Towards measuring the representation of subjective global opinions in language models. arXiv preprint arXiv:2306.16388.

Ritam Dutt, Sayan Sinha, Rishabh Joshi, Surya Shekhar Chakraborty, Meredith Riggs, Xinru Yan, Haogang Bao, and Carolyn Penstein Rosé. 2021. Resper: Computationally modelling resisting strategies in persuasive conversations. arXiv preprint arXiv:2101.10545.

Marieke L Fransen, Edith G Smit, and Peeter WJ Verlegh. 2015. Strategies and motives for resistance to persuasion: An integrative framework. Frontiers in psychology, 6:1201.

Yao Fu, Hao Peng, Tushar Khot, and Mirella Lapata. 2023. Improving language model negotiation with self-play and in-context learning from ai feedback.

Lewis R Goldberg. 1992. The development of markers for the big-five factor structure. Psychological assessment, 4(1):26.

He He, Derek Chen, Anusha Balakrishnan, and Percy Liang. 2018. Decoupling strategy and generation in negotiation dialogues. arXiv preprint arXiv:1808.09637.

Zhiyuan Hu, Yue Feng, Yang Deng, Zekun Li, See-Kiong Ng, Anh Tuan Luu, and Bryan Hooi. 2023. Enhancing large language model induced task-oriented dialogue systems through look-forward motivated goals.

Chen Huang, Peixin Qin, Yang Deng, Wenqiang Lei, Jiancheng Lv, and Tat-Seng Chua. 2024. Concept – an evaluation protocol on conversational recommender systems with system-centric and user-centric factors.

Chen Huang, Peixin Qin, Wenqiang Lei, and Jiancheng Lv. 2023. Reduce human labor on evaluating conversational information retrieval system: A humanmachine collaboration approach. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 10876–10891, Singapore. Association for Computational Linguistics.

Youngsoo Jang, Jongmin Lee, and Kee-Eung Kim. 2020. Bayes-adaptive monte-carlo planning and learning for goal-oriented dialogues. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 7994–8001.

Guangyuan Jiang, Manjie Xu, Song-Chun Zhu, Wenjuan Han, Chi Zhang, and Yixin Zhu. 2024. Evaluating and inducing personality in pre-trained language models. Advances in Neural Information Processing Systems, 36.

Hang Jiang, Xiajie Zhang, Xubo Cao, Jad Kabbara, and Deb Roy. 2023. Personallm: Investigating the ability of gpt-3.5 to express personality traits and gender differences. arXiv preprint arXiv:2305.02547.

Simon Keizer, Markus Guhe, Heriberto Cuayáhuitl, Ioannis Efstathiou, Klaus-Peter Engelbrecht, Mihai Dobre, Alex Lascarides, and Oliver Lemon. 2017. Evaluating persuasion strategies and deep reinforcement learning methods for negotiation dialogue agents. In Proceedings ofthe 15th Conference ofthe European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers, pages 480–484, Valencia, Spain. Association for Computational Linguistics.

Deuksin Kwon, Emily Weiss, Tara Kulshrestha, Kushal Chawla, Gale M Lucas, and Jonathan Gratch. 2024. Are llms effective negotiators? systematic evaluation of the multifaceted capabilities of llms in negotiation dialogues. arXiv preprint arXiv:2402.13550.

Wenqiang Lei, Yao Zhang, Feifan Song, Hongru Liang, Jiaxin Mao, Jiancheng Lv, Zhenglu Yang, and Tat-Seng Chua. 2022. Interacting with non-cooperative user: A new paradigm for proactive dialogue policy.

Mike Lewis, Denis Yarats, Yann Dauphin, Devi Parikh, and Dhruv Batra. 2017. Deal or no deal? end-to-end learning of negotiation dialogues. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 2443–2453.

Yu Li, Josh Arnold, Feifan Yan, Weiyan Shi, and Zhou Yu. 2021. Legoeval: An open-source toolkit for dialogue system evaluation via crowdsourcing. arXiv preprint arXiv:2105.01992.

Yajiao Liu, Xin Jiang, Yichun Yin, Yasheng Wang, Fei Mi, Qun Liu, Xiang Wan, and Benyou Wang. 2023. One cannot stand for everyone! leveraging multiple user simulators to train task-oriented dialogue systems. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1–21.

Shima Rahimi Moghaddam and Christopher J Honey. 2023. Boosting theory-of-mind performance in large language models via prompting. arXiv preprint arXiv:2304.11490.

OpenAI. 2022. Introducing chatgpt. https://openai. com/blog/chatgpt.

OpenAI. 2023. Gpt-4 technical report. ArXiv, abs/2303.08774.

David Premack and Guy Woodruff. 1978. Does the chimpanzee have a theory of mind? Behavioral and brain sciences, 1(4):515–526.

Karsten Roth, Biagio Brattoli, and Bjorn Ommer. 2019. Mic: Mining interclass characteristics for improved metric learning. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV).

Mustafa Safdari, Greg Serapio-García, Clément Crepy, Stephen Fitz, Peter Romero, Luning Sun, Marwa Abdulhai, Aleksandra Faust, and Maja Mataric. 2023.´ Personality traits in large language models. arXiv preprint arXiv:2307.00184.

Maarten Sap, Ronan LeBras, Daniel Fried, and Yejin Choi. 2022. Neural theory-of-mind? on the limits of social intelligence in large lms. arXiv preprint arXiv:2210.13312.

Susanne G Scott and Reginald A Bruce. 1995. Decisionmaking style: The development and assessment of a new measure. Educational and psychological measurement, 55(5):818–831.

Weiyan Shi, Kun Qian, Xuewei Wang, and Zhou Yu. 2019. How to build user simulators to train rl-based dialog systems. arXiv preprint arXiv:1909.01388.

Youzhi Tian, Weiyan Shi, Chen Li, and Zhou Yu. 2020. Understanding user resistance strategies in persuasive conversations. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4794–4798, Online. Association for Computational Linguistics.

Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-sne. Journal of machine learning research, 9(11).

Xintao Wang, Yaying Fei, Ziang Leng, and Cheng Li. 2023. Does role-playing chatbots capture the character personalities? assessing personality traits for roleplaying chatbots. arXiv preprint arXiv:2310.17976.

Xuewei Wang, Weiyan Shi, Richard Kim, Yoojung Oh, Sijia Yang, Jingwen Zhang, and Zhou Yu. 2019. Persuasion for good: Towards a personalized persuasive dialogue system for social good. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 5635–5649, Florence, Italy. Association for Computational Linguistics.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.

Ronald J Williams. 1992. Simple statistical gradientfollowing algorithms for connectionist reinforcement learning. Machine learning, 8:229–256.

Heinz Wimmer and Josef Perner. 1983. Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children’s understanding of deception. Cognition, 13(1):103–128.

Zelai Xu, Chao Yu, Fei Fang, Yu Wang, and Yi Wu. 2023. Language agents with reinforcement learning for strategic play in the werewolf game. arXiv preprint arXiv:2310.18940.

Runzhe Yang, Jingxiao Chen, and Karthik Narasimhan. 2021. Improving dialog systems for negotiation with personality modeling.

Xiao Yu, Maximillian Chen, and Zhou Yu. 2023. Prompt-based Monte-Carlo tree search for goaloriented dialogue policy planning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7101–7125, Singapore. Association for Computational Linguistics.

Haolan Zhan, Yufei Wang, Tao Feng, Yuncheng Hua, Suraj Sharma, Zhuang Li, Lizhen Qu, Zhaleh Semnani Azad, Ingrid Zukerman, and Gholamreza Haffari. 2024. Let’s negotiate! a survey of negotiation dialogue systems. arXiv preprint arXiv:2402.01097.

Qiang Zhang, Jason Naradowsky, and Yusuke Miyao. 2023a. Ask an expert: Leveraging language models to improve strategic reasoning in goal-oriented dialogue models. arXiv preprint arXiv:2305.17878.

Tong Zhang, Junhong Liu, Chen Huang, Jia Liu, Hongru Liang, Zujie Wen, and Wenqiang Lei. 2023b. Towards effective automatic debt collection with persona awareness. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 32–45, Singapore. Association for Computational Linguistics.

Tong Zhang, Peixin Qin, Yang Deng, Chen Huang, Wenqiang Lei, Junhong Liu, Dingnan Jin, Hongru Liang, and Tat-Seng Chua. 2024. CLAMBER: A benchmark of identifying and clarifying ambiguous information needs in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10746–10766, Bangkok, Thailand. Association for Computational Linguistics.

Xijia Zhang, Yue Guo, Simon Stepputtis, Katia Sycara, and Joseph Campbell. 2023c. Explaining agent behavior with large language models. arXiv preprint arXiv:2309.10346.

Pei Zhou, Aman Madaan, Srividya Pranavi Potharaju, Aditya Gupta, Kevin R McKee, Ari Holtzman, Jay Pujara, Xiang Ren, Swaroop Mishra, Aida Nematzadeh, et al. 2023a. How far are large language models from agents with theory-of-mind? arXiv preprint arXiv:2310.03051.

Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, et al. 2023b. Sotopia: Interactive evaluation for social intelligence in language agents. arXiv preprint arXiv:2310.11667.

Yiheng Zhou, Yulia Tsvetkov, Alan W Black, and Zhou Yu. 2019. Augmenting non-collaborative dialog systems with explicit semantic and strategic dialog history.

## A Details about Evaluation Protocol

## A.1 Building User Simulators

Due to the significant human labor required for real-user evaluations (Huang et al., 2023), our experiments utilize user simulators instead.

## A.1.1 Persona Generation

We prompt GPT4 (OpenAI, 2023) to generate diverse user personas by selecting attributes from two persona types, namely Big-Five Personality and Decision-Making Styles. Specifically, We allow GPT-4 to choose an attribute for each persona type, resulting in attribute-based user personas comprised of two fields, each containing a distinct attribute value. The prompt we use is provided in Table 11. In total, we create 20 attribute-based user personas and ensure that the number of each attribute is balanced. We then prompt GPT4 to rephrase these attribute-based personas into 300 cohesive persona descriptions. The prompt we use is provided in Table 12.

## A.1.2 Non-collaborative Behavior Prompting

We leverage the resisting strategies outlined in (Dutt et al., 2021) as users’ non-collaborative behaviors. We provide the detailed explanations of these resisting strategies in Table 7. We design detailed instructions and incorporate these resisting strategies with their explanations into our user simulator prompting.

## A.1.3 Comprehensive Prompting

By incorporating the persona description and resisting strategies, we construct comprehensive prompts for our user simulators. Specifically, our prompt includes two parts: task background and conversation history. In the task background, we guide LLMs to role-play their assigned personas with a set of role-play instructions and resisting strategies. We provide the comprehensive user simulator prompts across two tasks in Table 13 and 14.

## A.2 Evaluation Tasks

Following (Bianchi et al., 2024; Deng et al., 2023e), we consider two classic tasks as our evaluation scenarios, including price negotiation (He et al., 2018) and charity persuasion (Wang et al., 2019). The price negotiation task involves open-ended price negotiations where a buyer influences the seller towards a reasonable price, while the seller aims to maximize their own profit. The charity persuasion task involves asymmetric interactions guided by a persuader who endeavors to persuade the other party to make a charitable donation. Our evaluation is based on these two tasks, requiring the evaluated dialogue agents to take on the roles of buyer and persuader, respectively, in order to achieve their goals. To support our evaluations, we adopt the test dataset of CraigslistBargain (He et al., 2018) and PersuasionForGood (Wang et al., 2019), making use of their pre-annotated background information to streamline our assessment process. For the negotiation task, the background information includes item details and the desired price of each party. For the persuasion task, it involves determining if the individual being persuaded initially intends to make a donation. These background information serve as specific scenarios for our evaluation.

<table><tr><td>CB</td><td>Seller (User)</td><td>Buyer (Agent)</td></tr><tr><td>Target prices</td><td>285$</td><td>142$</td></tr><tr><td>Item</td><td colspan="2">A skillfully lugged and elegantly pantographed road bike</td></tr><tr><td>Goals</td><td>Maximize the price|</td><td>Minimize the price</td></tr><tr><td>Ending condition</td><td colspan="2">When either party accepts</td></tr><tr><td>Max. # of turns</td><td colspan="2">10 rounds of interaction</td></tr></table>

Table 5: The evaluation scenario of price negotiation. This case is selected from the validate set of Craigslist-Bargain Dataset (He et al., 2018).

<table><tr><td>P4G</td><td>Persuader (Agent)</td><td>Persuadee (User)</td></tr><tr><td>Charity info</td><td>It works to help fight poverty around the world</td><td></td></tr><tr><td>Goals</td><td>Convince the persuadee to donate | Foil the persuasion</td><td></td></tr><tr><td>Ending condition</td><td colspan="2">When the persuadee agree to donate.</td></tr><tr><td>Max. # of turns</td><td colspan="2">10 rounds of interaction</td></tr></table>

Table 6: The evaluation scenario of charity persuasion.

<table><tr><td>Resisting Strategy</td><td>Persuasion (P4G)</td><td>Negotiation (CB)</td></tr><tr><td>Source Derogation</td><td>Attacks/doubts the organisation&#x27;s credibility.</td><td>Attacks the other party or questions the item.</td></tr><tr><td>Counter Argument</td><td>Argues that the responsibility of donation is not on them or refutes a previous statement.</td><td>Provides a non-personal argument/factual re- sponse to refute a previous claim or to justify a new claim.</td></tr><tr><td>Personal Choice</td><td>Attempts to saves face by asserting their per- sonal preference such as their choice of charity and their choice of donation.</td><td>Provides a personal reason for disagreeing with the current situation or chooses to agree with the situation provided some specific condition is met.</td></tr><tr><td>Information Inquiry</td><td>Ask for factual information about the organisa- tion for clarification or as an attempt to stall.</td><td>Requests for clarification or asks additional infor- mation about the item or situation.</td></tr><tr><td>Self Pity</td><td>Provides a self-centred reason for not being able/willing to donate at the moment.</td><td>Provides a reason (meant to elicit sympathy) for disagreeing with the current terms.</td></tr><tr><td>Hesitance</td><td>Attempts to stall the conversation by either stat- ing they would donate later or is currently un- sure about donating.</td><td>Stalls for time and is hesitant to commit; specif- ically, they seek to further the conversation and provide a chance for the other party to make a better offer.</td></tr><tr><td>Self-assertion</td><td>Explicitly refuses to donate without even pro- viding a factual/personal reason.</td><td>Asserts a new claim or refutes a previous claim with an air of finality/ confidence.</td></tr><tr><td>Others</td><td>Do not explicitly foil the persuasion attempts.</td><td>Do not explicitly foil the negotiation attempts.</td></tr></table>

Table 7: The resisting strategies for P4G and CB tasks.

<table><tr><td colspan="3">Single-turn</td><td colspan="2">Multi-turn</td></tr><tr><td>Setting</td><td>Natural</td><td>Useful</td><td>Natural</td><td>Useful</td></tr><tr><td>Human</td><td>18%</td><td>20%</td><td>15%</td><td>22%</td></tr><tr><td>TRIP</td><td>45%</td><td>42%</td><td>34%</td><td>31%</td></tr><tr><td>Tie</td><td>37%</td><td>38%</td><td>51%</td><td>48%</td></tr></table>

Table 8: Comparison on user simulators and real users. The Cohen’s Kappa between annotators is 0.67.

## A.3 Reliability Analysis

Prior to conducting the interactive evaluation, we validate the reliability of using LLMs as user simulators that demonstrate non-collaborative behaviors. Following the approach described in (Deng et al., 2023e), we engage 5 human experts in conversations with two groups, including our diverse user simulators and 10 real users across two evaluation tasks. We collect 50 dialogues from each group and evaluate the user responses in both singleturn and multi-turn open-ended conversations. The evaluation focuses on the naturalness and utility of the generated responses in these conversation settings. Naturalness refers to the fluency and human-like nature of the responses, while utility indicates their consistency with the role instructions and non-collaborative behaviors. We employ two annotators to conduct pairwise evaluations by rating "Win/Tie/Lose" between the two samples. As shown in Table 8, the user simulators exhibit a notably superior performance compared to real users, particularly when it comes to the naturalness of responses in multi-turn conversations, which showcases the impressive language generation capabilities inherent in LLMs. Furthermore, even compared with human-annotated dialogues, the GPT3.5-based simulator shows competitive performance. These results validate the reliability of adopting GPT3.5 as the user simulator.

## A.4 Interactive Evaluation Protocol

During the evaluation, each dialogue agent must engage with these simulators (Deng et al., 2023e). During interactions, the dialogue agent and user simulator alternate in employing strategies in their responses with the ultimate aim of maximizing their own self-interest. The interactions continues until the conversational goal is achieved or the maximum number of turns T (i.e., T is set to 10 for both tasks) is reached. To determine goal achievement, we utilize AI feedback to assess whether the task goal has been reached. Specifically, in price negotiation task, we employ a separate GPT3.5 (i.e., $L L M _ { r w d } )$ to assess whether both parties have reached a deal. We prompt $L L M _ { r w d }$ to generate feedback for the binary question “Have they reached a deal?”. If the output of $L L M _ { r w d }$ indicates that both parties have reached an agreement, we consider this as goal achievement. In charity persuasion task, we additionally prompt the user simulator to express his willingness to make a donation at the end of each turn. In particular, we query the user simulator "Would you be interested in donating to Save the Children?". If the feedback is positive, we regard this as goal achievement. Conversely, if the goal is not achieved, the interaction continues.

Due to the subjectivity of the planning outcome as well as the variance of the LLM-generated output, we follow a common practice (Wang et al., 2022; Deng et al., 2023e) to alleviate these issues by sampling the decoded sequences l (i.e., l is set to 10 for both tasks) times.

## B Implementation Details

## B.1 TRIP Implementation Details

## B.1.1 Theory-of-Mind

We leverage the strong Theory-of-Mind capability of GPT3.5 to infer the mental states and user future actions during interaction. The prompt we use is provided in Table 15 and 16.

## B.1.2 Strategy Prompting

Here, we present the dialogue agent strategies utilized in our experiments. Initially, we outline the strategies along with their explanations for two tasks in Table 9 and 10. We then offer a comprehensive overview of our TRIP prompting in Table 19 and 20.

## B.1.3 Supervised Fine-Tuning

We initialize our strategy planner by imitating human-human dialogue datasets in CraigslistBargain and PersuasionForGood through supervised fine-tuning (SFT). In specific, we adopt the strategy annotations in the train dataset to support our SFT. we optimize the strategy planner by minimizing the cross-entropy loss between the predicted strategy y<sub>i</sub> and the human annotated strategy yˆ<sub>i</sub>:

$$
\mathcal { L } _ { C E } = - \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left[ y \log \hat { y } _ { i } + ( 1 - y _ { i } ) \log ( 1 - \hat { y } _ { i } ) \right]
$$

Regarding the training hyper-parameters, we set the batch size 16 and the learning rate 6e-6, and utilize the AdamW optimizer with a weight decay of 0.01. We save the checkpoint based on the best performance at the validation set.

## B.1.4 Online RL Training

After SFT, we optimize our strategy planner through REINFORCE algorithm. In specific, our training involves 1000 episodes, with a learning rate of 1e-6, a discount factor 0.999, and the maximum conversation turn of each episode 10. All the training experiments are run on a server equipped with 4 Tesla V100 GPUs.

## B.2 Baselines Implementation Details

We implement the existing LLM-based dialogue agents by following previous works.

Standard: simply prompts LLMs to chat with users using task instructions without considering any dialogue strategy.

ProCoT: we follow (Deng et al., 2023d) and prompt LLM to analyze the dialogue status and plan next strategy, and then generate a response based on the planned strategy. We provide its prompt design in Table 17.

ICL-AIF: we follow (Fu et al., 2023) and prompt another GPT3.5 for verbal feedback, offering suggestions to the dialogue agent upon completion of an interaction. Our implementation involves presenting three suggestions at the conclusion of each interaction, while ensuring that only the most recent 20 suggestions are retained to prevent indefinite expansion. The prompt we use is provided in Table 18.

GDP-MCTS: we follow (Yu et al., 2023) and implement open-MCTS to help LLM for strategic planning. This method is originally proposed for charity persuasion dialogues. In order to further accommodate the price negotiation applications, we just need to modify the task instruction and the role-playing description.

PPDPP: we follow (Deng et al., 2023e) and adopt the BERT<sup>7</sup> model (Devlin et al., 2018) as our external planner. We implement PPDPP based on the training details provided in the original paper. We have made adjustments to the task instructions and role-playing descriptions, adapting them for use in the context of charity persuasion.

## C Human Evaluation

Inspired by (Yu et al., 2023), we conduct interactive human evaluation using the LegoEval platform (Li et al., 2021) with crowdworkers on Amazon Mechanical Turk. We primarily sought to evaluate TRIP against two competitive baselines (i.e., Standard and PPDPP). In specific, we hire 20 crowdworkers with varying personas to converse with our three agents based on the price negotiation and charity persuasion tasks. After conversations, we collect 50 dialogues for each agent and calculate their performances using the same metrics mentioned in Section 3.1.

## D More Experimental Results

In addition to the Success Rate, we report the agents performance across various personas using the metrics of Average Turn and Sale-to-List Ratio, as depicted in Figure 8 and Figure 7. We discover that the overall performance and analysis conclusions remain largely consistent with Section 5.1.

![](images/67ffd98fc44b46d0b70ee40247e07fb2d8e59de30d0cc257d7d46737a9de9143.jpg)  
Figure 7: The agents performance across various personas. We report their SL % on the price negotiation task. TRIP achieves balanced improvements on all personas, significantly outperforming other agents by a considerable margin.

![](images/d24a3b9b3732b16d2fd38626bc6da3e9db4e13ce28b373d29b9598d216fb1ac6.jpg)

![](images/d6b206916ba49a4761c95105dcf25942ea41eec4fe491a4584f73cb995d11f9d.jpg)  
Figure 8: The agents performance across various personas. We report their average turn on two tasks, namely price negotiation (Left) and charity persuasion (Right). TRIP achieves balanced improvements on all personas, significantly outperforming other agents by a considerable margin.

<table><tr><td colspan="1" rowspan="1">Dialogue Strategy</td><td colspan="1" rowspan="1">Explanation</td></tr><tr><td colspan="1" rowspan="1">Greetings</td><td colspan="1" rowspan="1">Please say hello or chat randomly.</td></tr><tr><td colspan="1" rowspan="1">Ask a question</td><td colspan="1" rowspan="1">Please ask any question about product, year, price, usage, etc.</td></tr><tr><td colspan="1" rowspan="1">Answer a question</td><td colspan="1" rowspan="1">Please provide information about the product, year, usage, etc.</td></tr><tr><td colspan="1" rowspan="1">Propose the first price</td><td colspan="1" rowspan="1">Please initiate a price or a price range for the product.</td></tr><tr><td colspan="1" rowspan="1">Propose a counter price</td><td colspan="1" rowspan="1">Please propose a new price or a new price range.</td></tr><tr><td colspan="1" rowspan="1">Use comparatives</td><td colspan="1" rowspan="1">Please propose a vague price by using comparatives with exist-ing price.</td></tr><tr><td colspan="1" rowspan="1">Confirm information</td><td colspan="1" rowspan="1">Please ask a question about the information to be confirmed.</td></tr><tr><td colspan="2" rowspan="1">Affirm confirmation        Please give an affirmative response to a confirm.</td></tr><tr><td colspan="2" rowspan="1">Deny confirmation          Please give a negative response to a confirm.</td></tr><tr><td colspan="2" rowspan="1">Agree with the proposal  Please agree with the proposed price.</td></tr><tr><td colspan="2" rowspan="1">Disagree with a proposal  Please disagree with the proposed price.</td></tr><tr><td colspan="1" rowspan="1">Logical Appeal</td><td colspan="1" rowspan="1">Please use of reasoning and evidence to convince the persuadee.</td></tr><tr><td colspan="1" rowspan="1">Emotion Appeal</td><td colspan="1" rowspan="1">Please elicit the specific emotions to influence the persuadee.</td></tr><tr><td colspan="1" rowspan="1">Credibility Appeal</td><td colspan="1" rowspan="1">Please use credentials and cite organizational impacts to es-tablish credibility and earn the user's trust. The informationusually comes from an objective source (e.g., the organization'swebsite or other well-established websites).</td></tr><tr><td colspan="1" rowspan="1">Foot in the Door</td><td colspan="1" rowspan="1">Please use the strategy of starting with small donation requeststo facilitate compliance followed by larger requests.</td></tr><tr><td colspan="1" rowspan="1">Self-Modeling</td><td colspan="1" rowspan="1">Please use the self-modeling strategy where you first indicatesthe persuadee own intention to donate and chooses to act as arole model for the persuadee to follow.</td></tr><tr><td colspan="1" rowspan="1">Personal Story</td><td colspan="1" rowspan="1">Please use narrative exemplars to illustrate someone donationexperiences or the beneficiaries positive outcomes, which canmotivate others to follow the actions.</td></tr><tr><td colspan="1" rowspan="1">Donation Information</td><td colspan="1" rowspan="1">Please provide specific information about the donation task,such as the donation procedure, donation range, etc. By pro-viding detailed action guidance, this strategy can enhance thepersuadee's self-efficacy and facilitates behavior compliance.</td></tr><tr><td colspan="1" rowspan="1">Source-related Inquiry</td><td colspan="1" rowspan="1">Please ask if the persuadee is aware of the organization (i.e.,the source in our specific donation task).</td></tr><tr><td colspan="1" rowspan="1">Task-related Inquiry</td><td colspan="1" rowspan="1">Please ask about the persuadee opinion and expectation relatedto the task, such as their interests in knowing more about theorganization.</td></tr><tr><td colspan="1" rowspan="1">Personal-related Inquiry</td><td colspan="1" rowspan="1">Please asks about the persuadee previous personal experiencesrelevant to charity donation.</td></tr></table>

Table 9: The negotiation strategies used in our TRIP agent.

Table 10: The persuasion strategies used in our TRIP agent.

![](images/7d14492858f7a313b7da943514bcb7ec699c269c8e3b44fba599f7592ff7b7b4.jpg)  
Table 11: The prompt of user persona generation.

![](images/9b35c29b681efd103b863d38b04d5f6ed773f42773342cd59e4e154ad2eef0c4.jpg)  
Table 12: The prompt of user persona rephrase.

![](images/2dc51447b0d6415c34efe8b868e96e0787bd240c03ec3a53d4b732b034d74d95.jpg)  
Table 13: The comprehensive prompt of user simulators in the price negotiation task.

![](images/0ae7c13633bed343b4bf2e49dd6286f108e0004b9a11bc7823f0365db55ff4db.jpg)  
Table 14: The comprehensive user simulator prompt for the charity persuasion task.

<table><tr><td>The Theory-of-Mind prompt for the price negotiation task</td></tr><tr><td>You are an expert in price bargain.</td></tr><tr><td>Now give you a conversation history between a buyer and a seller, you need to infer the mental states</td></tr><tr><td>and future actions of the seller.</td></tr><tr><td>********</td></tr><tr><td>Conversation History</td></tr><tr><td>********</td></tr><tr><td>The Theory-of-Mind prompt for the charity persuasion task</td></tr><tr><td>You are an expert in charity persuasion.</td></tr><tr><td>Now give you a conversation history between a persuader and a persuadee, you need to infer the mental</td></tr><tr><td>states and future actions of the persuadee.</td></tr><tr><td>********</td></tr><tr><td>Conversation History</td></tr><tr><td>********</td></tr></table>

Table 15: The ToM prompt for the price negotiation task.

Table 16: The ToM prompt for the charity persuasion task.

<table><tr><td>The prompt of the ProCoT agent</td></tr><tr><td>The Price Negotiation Task Assume you are the buyer. Given the conversation history, in order to reach a better deal with the seller, please select the most appropriate dialogue strategy. You can only reply by selecting one of the following dialogue strategy to reach the goal: Greetings. Ask a question. Answer a question. Propose the first price. Propose a counter price. Use comparatives. Confirm information. Affirm confirmation. Deny confirmation. Agree with the proposal. Disagree with a proposal. The following is the conversation history: [conversation]</td></tr><tr><td>The Charity Persuasion Task Assume you are the Persuader. Given the conversation history, in order to convince the persuadee to donate for charity, please select the most appropriate dialogue strategy.</td></tr><tr><td></td></tr><tr><td>You can only reply by selecting one of the following dialogue strategy to reach the goal: Logical appeal, Emotion appeal, Credibility appeal, Foot in the door, Self-modeling, Personal story, Donation information, Source-related inquiry, Task-related inquiry, Personal-related inquiry.</td></tr></table>

Table 17: The prompt design of the ProCoT agent.

<table><tr><td>The prompt of the ICL-AIF agent</td></tr><tr><td>The Price Negotiation Task Now enter the role-playing mode. In the following conversation, you will play as a coach in a bargain game. There will be a buyer and a seller bargaining about a product price. Your task is to read the conversation between the buyer and the seller, then provide suggestions to the buyer about how to buy the product with a lower price. Each suggestion should be only one short and succinct sentence. The following is the conversation: [conversation] Question: What are your suggestions? Answer:</td></tr><tr><td>The Charity Persuasion Task Now enter the role-playing mode. In the following conversation, you will play as a coach in a persuasion game. There will be a persuader who is trying to persuade a persuadee for charity donation.</td></tr><tr><td>Your task is to read the conversation between the persuader and the persuadee, then provide suggestions to the persuader about how to convince the persuadee to make a donation. Each suggestion should be</td></tr><tr><td>only one short and succinct sentence. The following is the conversation: [conversation] Question: What are your suggestions? Answer:</td></tr><tr><td>The prompt of our TRIP agent in price negotiation</td></tr><tr><td>Now enter the role-playing mode. In the following conversation, you will play as a buyer in a price bargaining game.</td></tr><tr><td>You are the buyer who is trying to buy the %s with the price of %s. Product description: %s</td></tr><tr><td>Please reply with only one short and succinct sentence. [action] Now start the game</td></tr></table>

Table 18: The prompt design of the ICL-AIF agent.

Table 19: The prompt design of the TRIP agent for price negotiation.

<table><tr><td>The prompt of our TRIP agent in charity persuasion</td></tr><tr><td>Now enter the role-playing mode. In the following conversation, you will play as a Persuader who is trying to persuade the Persuadee to donate to the charity called Save the Children. Save the Children is head-quartered in London, and they work to help fight poverty around the world.</td></tr><tr><td>Children need help in developing countries and war zones. Small donations like $1 or $2 go a long</td></tr><tr><td>way to help.</td></tr><tr><td>You are the Persuader who is trying to convince the Persuadee to donate to a charity called Save the Children. [action]</td></tr></table>

Table 20: The prompt design of the TRIP agent for charity persuasion.