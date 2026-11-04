# What’s Mine becomes Yours: Defining, Annotating and Detecting Context-Dependent Paraphrases in News Interview Dialogs

Anna Wegmann<sup>1</sup>, Tijs van den Broek<sup>2</sup> and Dong Nguyen<sup>1</sup>

<sup>1</sup>Utrecht University, Utrecht, The Netherlands

<sup>2</sup>Vrije Universiteit Amsterdam, Amsterdam, The Netherlands {a.m.wegmann, d.p.nguyen}@uu.nl, t.a.vanden.broek@vu.nl

## Abstract

Best practices for high conflict conversations like counseling or customer support almost always include recommendations to paraphrase the previous speaker. Although paraphrase classification has received widespread attention in NLP, paraphrases are usually considered independent from context, and common models and datasets are not applicable to dialog settings. In this work, we investigate paraphrases across turns in dialog (e.g., Speaker 1: “That book is mine.” becomes Speaker 2: “That book is yours.”). We provide an operationalization of context-dependent paraphrases, and develop a training for crowd-workers to classify paraphrases in dialog. We introduce ContextDeP, a dataset with utterance pairs from NPR and CNN news interviews annotated for contextdependent paraphrases. To enable analysis on label variation, the dataset contains 5,581 annotations on 600 utterance pairs. We present promising results with in-context learning and with token classification models for automatic paraphrase detection in dialog.

## 1 Introduction

Repeating or paraphrasing what the previous speaker said has time and time again been found to be important in human-to-human or human-tocomputer dialogs: It encourages elaboration and introspection in counseling (Rogers, 1951; Miller and Rollnick, 2012; Hill, 1992; Shah et al., 2022), can help deescalate conflicts in crisis negotiations (Vecchi et al., 2005; Voss and Raz, 2016; Vecchi et al., 2019), can have a positive impact on relationships (Weger Jr et al., 2010; Roos, 2022), can increase the perceived response quality of dialog systems (Weizenbaum, 1966; Dieter et al., 2019) and generally provides tangible understanding-checks to ground what both speakers agree on (Clark, 1996; Jurafsky and Martin, 2019).

Fortunately, in NLP, paraphrases have received wide-spread attention: Researchers have created

Guest: And people always prefer, of course, to see the pope as the principal celebrant of the mass. So that’s good. That’ll be tonight. And it will be his 26th mass and it will be the 40th or, rather, the 30th time that this is offered in round the world transmission. And it will be my 20th time in doing it as a television commentator from Rome so.

Host: Yes, you’ve been doing this for a while now.

Figure 1: Context-Dependent Paraphrase in a News Interview. The interview host paraphrases part of the guest’s utterance. It is only a paraphrase in the current context (e.g., doing something 20 times and doing somethingfor a while are not generally synonymous). Our annotators provide word-level highlighting. The color’s intensity shows the share of annotators that selected the word. Here, most annotators selected the same text spans, some included “from Rome” as part of what is paraphrased by the host. We underline the paraphrase identified by our fine-tuned DeBERTa token classifier.

numerous paraphrase datasets (Dolan and Brockett, 2005; Zhang et al., 2019; Dong et al., 2021; Kanerva et al., 2023), developed methods to automatically identify paraphrases (Zhang et al., 2019; Wei et al., 2022a; Zhou et al., 2022), and used paraphrase datasets to train semantic sentence representations (Reimers and Gurevych, 2019; Gao et al., 2021) and benchmark LLMs (Wang et al., 2018; bench authors, 2023). However, most previous work (1) has focused on context-independent paraphrases, i.e., texts that are semantically equivalent independent from the given context, and has not investigated the automatic detection of paraphrases across turns in dialog, (2) has classified paraphrases at the level of full texts even though paraphrases often only occur in portions of larger texts (see also Figure 1), (3) uses a small number of 1–3 annotations per paraphrase pair (Dolan and Brockett, 2005; Kanerva et al., 2023), (4) only annotate text pairs that are “likely” to include paraphrases using heuristics such as lexical similarity (Dolan and Brockett, 2005), although, especially for the dialog setting, we can not expect lexical similarity to be high for all or even most paraphrase pairs (e.g., the pair in Figure 1 only overlaps in two words) and (5) either use short annotation instructions (Dolan and Brockett, 2005) that rely on annotator intuitions or long and complex instructions (Kanerva et al., 2023) that limit the total number of annotators.

<table><tr><td rowspan="2"></td><td colspan="2">Agreement</td><td rowspan="2">Single Example with High Variation Shortened Example</td><td rowspan="2">Vote</td></tr><tr><td>Dataset</td><td>Acc. α</td></tr><tr><td>BALANCED</td><td>0.71</td><td>0.32</td><td>Guest: [...]Maybe the money will help. Host: can’t hurt, let&#x27;s put it that way.</td><td>9/20</td></tr><tr><td>RANDOM</td><td>0.72</td><td>0.23</td><td>G: So both parties agree thatwe need tostop horrific acts ofviolence against animals. But everyone is standing behind this. It is time tostop horrific acts of brutality on animals. H: Britain&#x27;s Queen Elizabeth&#x27;s senior dresser writes &quot;If her majesty is due to attend an engagement in particularly cold weather from 2019 onwards fake fur will be used to make sure she stays warm.&quot; it&#x27;s a very stark example ofa monarch following publicopinion in the U.K. which is moving away from fur and it very much</td><td>7/15</td></tr><tr><td>PARA</td><td></td><td>0.65 0.19</td><td>embraces prevention of cruelty to the animals. G: [...]it could be programmed in.But again, you&#x27;d have to set that up as part of your flight plan. H: So you&#x27;d have to say I&#x27;m going to drop to 5,000 feet, then go back up to 35,000 feet,and you would have had to have done that at the beginning.</td><td>8/15</td></tr></table>

Table 1: Agreement Scores as an Indicator of Plausible Variation. For each dataset, we display the “accuracy” with the majority vote (Acc.) which is the mean overlap of a rater’s classification with the majority vote classification excluding the current rater and Krippendorff (1980)’s alpha (α) for the binary classifications by all raters over all pairs. The relatively low K’s α scores can be explained by pairs where either label is plausible. We display such an example for each dataset with the share of annotators classifying it it as a paraphrase (Vote).

We address all five limitations with this work. First, we are, to the best of our knowledge, the first to focus on operationalizing, annotating and automatically detecting context-dependent paraphrases across turns in dialog. Dialog is a setting that is uniquely sensitive to context (Grice, 1957, 1975; Davis, 2002), e.g., “doing this for a while now” and “20th time [...] as a television commentator” in Figure 1 are not generally semantically equivalent. Second, instead of classifying whether two complete texts A and B are paraphrases of each other, we focus on classifying whether there exists a selection of a text B that paraphrases a selection of a text A, and identifying the text spans that constitute the paraphrase pair (e.g., Figure 1). Third, we collect a larger number of annotations of up to 21 per item in line with typical efforts to address plausible human label variation (Nie et al., 2020; Sap et al., 2022). Even though contextdependent paraphrase identification in dialog might at first seem straight forward with a clear ground truth, similar to other “objective” tasks in NLP (Uma et al., 2021), human annotators (plausibly) disagree on labels (Dolan and Brockett, 2005; Kanerva et al., 2023). For example, consider the first text pair in Table 1. “[The money] can’t hurt” can be interpreted in at least two different ways: as a statement with approximately the same meaning as “the money will help” or as an opposing statement meaning the money actually won’t help but at least “It can’t hurt” either. Fourth, instead of using heuristics to select text pairs for annotations, we choose a dialog setting where paraphrases are relatively likely to occur: transcripts of NPR and CNN news interviews (Zhu et al., 2021) since in (news) interviews paraphrasing or more generally active listening is encouraged (Clayman and Heritage, 2002; Hight and Smyth, 2002; Sedorkin et al., 2023). While the interview domain shows some unique characteristics limiting generalizability (e.g., hosts using paraphrases to simplify the guest’s statements for the audience), the interview domain is is suitable to demonstrate our new task and includes a diverse set of topics and guests. Fifth, we develop an annotation procedure that goes beyond relying on intuitions and is scalable to a large number of annotators: an accessible example-centric, handson, 15-minute training before annotation.

In short, we operationalize context-dependent paraphrases in dialog with a definition and an iteratively developed hands-on training for annotators. Then, annotators classify paraphrases and identify the spans of text that constitute the paraphrase. We release ContextDeP (Context-Dependent Paraphrases in news interviews), a dataset with 5,581 annotations on 600 utterance pairs from NPR and CNN news interviews. We use in-context learning (ICL) with generative models like Llama 2 or GPT-4 and fine-tune a DeBERTa token classifier to detect paraphrases in dialog. We reach promising results of F1 scores from 0.73 to 0.81. Generative models perform better at classification, while the token classifier provides text spans without parsing errors. We hope to advance dialog based evaluations of LLMs and the reliable detection of paraphrases in dialog. Code<sup>1</sup>, annotated data<sup>2,3</sup> and the trained model<sup>4</sup> are publicly available for research purposes.

<table><tr><td rowspan=1 colspan=3>What?       Shortened Examples</td></tr><tr><td rowspan=1 colspan=3>Guest: I know they are cruel.ClearHost: You know they are cruel.Contextual</td></tr><tr><td rowspan=1 colspan=2>Equiva-lence ⊆ CP</td><td rowspan=2 colspan=1>the president.H: The president has been using Chicagoas a punching bag.</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=2 colspan=2>Approxi-</td><td rowspan=3 colspan=1>G: I&#x27;m like, &quot;Fortnite&quot;, what is that? Idon&#x27;t even know what it is –H: So, you weren&#x27;t even familiar?</td></tr><tr><td rowspan=1 colspan=1>Appro</td></tr><tr><td rowspan=2 colspan=2>mateContextualEquiva-lence ⊆ CP</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>G: My wife is going through the samething herself.H: She&#x27;s also looking for work.</td></tr></table>

Table 2: Contextual Paraphrases (CP). We include text spans ( CP) that range from clear to approximate equivalence for the given context. Few examples are very clear. Deciding between approximate equivalence and non-equivalence turns out to be a difficult task. In our dataset, annotator agreement scores can be used as a proxy for the ambiguity of an item.

## 2 Related Work

Paraphrases have most successfully been classified by encoder architectures with fine-tuned classification heads (Zhang et al., 2019; Wahle et al., 2023) and more recently using in-context learning with generative models like GPT-3.5 and Llama 2 (Wei et al., 2022a; Wang et al., 2022c; Wahle et al., 2023). To the best of our knowledge, only Wang et al. (2022a) go beyond classifying paraphrases at the complete sentence level. They use a DeBERTa token classifier to highlight text spans that are not part of a paraphrase, i.e., the reverse of our task.

<table><tr><td>What?</td><td>Shortened Example</td></tr><tr><td>Addition- al Conclu-</td><td>Guest: If you&#x27;re not in our country, there are no constitutional protections for you. Host: So, you don&#x27;t have a problem with</td></tr><tr><td>sions or Facts CP Isolated</td><td>Facebook giving the government access to the private accounts of people applying to enter the U.S.? G: There are militant groups out there fir-</td></tr><tr><td>Equiva- lence CP</td><td>ing against the military. H: Why did the army decide today to move in and clear out the camp?</td></tr></table>

Table 3: Non-Paraphrases in Dialog. We do not include text pairs (⊈ CP) that are semantically related but where the second speaker does not actually rephrase a point the first speaker makes. Frequent cases are text spans that might only be considered approximately equivalent when taken out of context (underlined) and pairs that have too distant meanings, for example, when the interviewer continues with the same or a related topic but adds further-reaching conclusions or new facts.

Paraphrase taxonomies commonly go beyond binary classifications to make more fine-grained distinctions between paraphrase types, often including considerations w.r.t. the context of the text pairs. Bhagat and Hovy (2013) and Kovatchev et al. (2018) describe substitutions and other lexical operations that result in paraphrases in a given sentential context. Shwartz and Dagan (2016) show that context information can reverse semantic relations between phrases. Vila et al. (2014) discuss text pairs that are equivalent when one presupposes encyclopedic or situational knowledge (e.g., referents or intentions<sup>5</sup>), but exclude them as non-paraphrases. Further, to the best of our knowledge, most previous work annotate sentence pairs without considering the document context, with Kanerva et al. (2023) being the only exception, and no previous work looking at detecting paraphrases in dialog.

Dialog act taxonomies aim to classify the communicative function of an utterance in dialog and commonly include acts such as Summarize/Reformulate (Stolcke et al., 2000; Core and Allen, 1997). However, generally, communicative function can be orthogonal to meaning equivalence. For example, the paraphrase from Table 2 “So you weren’t even familiar?” would probably be a Declarative Yes-No-Question dialog act (Stolcke et al., 2000), while the non-paraphrase “So you don’t have a problem with ... ?” in Table 3 would also be a Declarative Yes-No-Question. We see paraphrase detection in dialog as more elementary and complementary to investigating communicative function of utterances.

## 3 Context-Dependent Paraphrases in Dialog

In NLP, paraphrases typically are pairs of text that are approximately equivalent in meaning (Bhagat and Hovy, 2013), since full equivalence usually only applies for practically identical strings (Bhagat and Hovy, 2013; Dolan and Brockett, 2005) – with some scholars even claiming that different sentences can never be fully equivalent in meaning (Hirst, 2003; Clark, 1992; Bolinger, 1974). The field of NLP has mostly focused on paraphrases that are context-independent, i.e., approximately equivalent without considering a given context (Dolan and Brockett, 2005; Wang et al., 2018; Zhang et al., 2019). Some studies have operationalized paraphrases using more fine-grained taxonomies, where context is sometimes considered (Bhagat and Hovy, 2013; Vila et al., 2014; Kovatchev et al., 2018). However, only a few datasets include such paraphrases (Kovatchev et al., 2018; Kanerva et al., 2023) and to the best of our knowledge none that focus on context-dependent paraphrases or dialog data.

We define a context-dependent paraphrase as two text excerpts that are at least approximately equivalent in meaning in a given situation but not necessarily in all non-absurd situations.<sup>6</sup> For example, consider the first exchange in Table 2. In this situation, “I” uttered by the first speaker and “You” uttered by the second speaker are clearly signifying the same person. However, if uttered by the same speaker “I” and “you” probably do not signify the same person. The text pair in Table 2 is thus equivalent in at least one but not in all non-absurd situations. The text excerpts forming context-dependent paraphrases do not have to be complete utterances. In many cases they are portions of utterances, see highlights in Figure 1. Note that in dialog, the second speaker should rephrase part of the first speaker’s point in the given situation (context condition) and not just talk about something semantically related (equivalence condition).

Context-dependent paraphrases range from clear (first example in Table 2) to approximate contextual equivalence (last example in Table 2). When the guest says “My wife is going through the same thing”, it seems reasonable to assume that the host is using contextual knowledge to infer that “the same thing” and “looking for a job” are equivalent for the given exchange. Even though in this last example the meaning of the two utterances could also be subject to different interpretations, we still consider such cases to be context-dependent paraphrases for two reasons: (1) similar to findings in context-independent paraphrase detection, limiting ourselves to very clear cases would mostly result in uninteresting, practically identical strings and (2) we ultimately want to identify paraphrases in human dialog, which is full of implicit contextual meaning (Grice, 1957, 1975; Davis, 2002).

We specifically exclude common cases of disagreements between annotators<sup>7</sup> that we consider not to be context-dependent paraphrases in dialog, see Table 3. First, we exclude text spans that might be considered approximately equivalent when they are looked at in isolation but do not represent a paraphrase of the guest’s point in the given situation (e.g., “the military” and “the army” in Table 3). Second, we exclude text pairs that diverge too much from the original meaning when the second speaker adds conclusions, inferences or new facts. In an interview setting, journalists make use of different question types and communication strategies relating to their agenda (Clayman and Heritage, 2002) that can sometimes seem like paraphrases. For example in Table 3, the host’s question “So, you ...?” could be read as a paraphrase with the goal of checking understanding with the guest. However, it is more likely to be a declarative conclusion that goes beyond what the guest said.

## 4 Dataset

Generally, people do not paraphrase each other in every conversation. We focus on the news interview setting, because paraphrasing, or more generally active listening, is a common practice for journalists (Clayman and Heritage, 2002; Hight and Smyth, 2002; Sedorkin et al., 2023). We therefore also only consider whether the journalist (the interview host) paraphrases the interview guest and not the other way around. We use Zhu et al. (2021)’s MediaSum corpus which consists of over 450K news interview transcripts and their summaries from 1999–2019 NPR and 2000–2020 CNN interviews.<sup>8</sup>

<table><tr><td>Dataset</td><td>size</td><td># paraphrases</td><td># anns/item</td></tr><tr><td>BALANCED</td><td>100</td><td>54</td><td>20.1</td></tr><tr><td>RANDOM</td><td>100</td><td>13</td><td>5.7</td></tr><tr><td>PARA</td><td>400</td><td>254</td><td>7.5</td></tr><tr><td>Total</td><td>600</td><td>321</td><td>9.3</td></tr></table>

Table 4: Dataset Statistics. For each dataset, we display the size, the number of paraphrases according to the majority vote and the average annotations per text pair.

## 4.1 Preprocessing

We only include two-person interviews, i.e., a conversation between an interview host and a guest. We remove interviews with fewer than four turns, utterances that only consist of two words or of more than 200 words, and the first and last turns of interviews (often welcoming addresses and goodbyes). Overall, this leaves 34,419 interviews with 148,522 (guest, host)-pairs. See App. B.1 for details.

## 4.2 Data Samples for Annotation

Even though paraphrases are relatively likely in the news interview setting, most randomly sampled text pairs still do not include paraphrases. To distribute annotation resources to text pairs that are likely to be paraphrase, previous work usually selects pairs based on heuristics like textual similarity features, e.g., word overlap, edit distance, or semantic similarity (Dolan and Brockett, 2005; Su and Yan, 2017; Dong et al., 2021). However, these approaches are systematically biased towards selecting more obvious, often lexically similar text pairs, possibly excluding many context-dependent paraphrases. For example, the guest and host utterance in Figure 1 have varying lengths, only overlap in three words and have a semantic similarity score of only 0.13<sup>9</sup>. Similar to Kanerva et al. (2023), we instead use a manual selection of promising text pairs for annotation: We (1) randomly sample a set of text pairs and (2) manually classify at each of them to (3) select three sets of text pairs that vary in their paraphrase distribution for the more resource-intensive crowd-sourced annotations: the RANDOM, BALANCED and PARA set.

Lead Author Annotation. We shuffle and uniformly sample 1,304 interviews. For each interview, we sample a maximum of 5 consecutive (guest, host)-pairs. To select promising paraphrase candidates, the lead author then manually classifies all 4,450 text pairs as paraphrases vs. nonparaphrases (see App. B.2 for details).<sup>10</sup> In total, about 14.9% of the sampled text pairs are classified as paraphrases by the lead author. On a random set of 100 (guest, host)-pairs (RANDOM), we later compare the lead author’s classifications with the crowd-sourced paraphrase classifications (see App. B.2). 89% of the lead author’s classifications are the same as the crowd majority. Note that the lead author’s classifications do not affect the quality of the annotations released with the dataset but only the text pairs that are selected for annotation. However, using lead author annotations instead of lexical level heuristics should increase paraphrase diversity in the released dataset beyond high lexical similarity pairs.

<table><tr><td>Split</td><td># (guest, host)-pairs</td><td># annotations</td></tr><tr><td>Train</td><td>420</td><td>3896</td></tr><tr><td>Dev</td><td>88</td><td>842</td></tr><tr><td>Test</td><td>92</td><td>843</td></tr><tr><td>Total</td><td>600</td><td>5,581</td></tr></table>

Table 5: Split of Dataset. For each set, we show the number of text pairs and the total number of annotations.

Paraphrase Candidate Selection. We sample three datasets for annotation that differ in their estimated paraphrase distributions (based on the lead author annotations): BALANCED is a set 100 text pairs sampled for equal representation of paraphrases and non-paraphrases. We annotate this dataset first with a high number of annotators per (guest, host)-pair, to decide on a crowd-worker allocation strategy that performs well for paraphrases as well as non-paraphrases. RANDOM is a uniform random sample of 100 text pairs. One main use of the dataset is to evaluate the quality of crowdworker annotations on a random sample. PARA is a set of 400 text pairs with an estimated 84% of paraphrases designed to increase the variety of paraphrases in our dataset. Details on the sampling of the three datasets can be found in App. B.3.

## 5 Annotation

We first describe the annotation task (§5.1). Then, we discuss why the annotation task is difficult and a clear ground truth classification might not exist in many cases (§5.2). Therefore, we dynamically collect many judgments for text pairs with high disagreements (§5.4). The annotation of utterance pairs takes place in two rounds with Prolific crowd-workers: (1) training crowd-workers (§5.3) and (2) annotating paraphrases with trained crowdworkers (§5.4 and §5.5).

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Guest</td><td colspan="2">Host</td></tr><tr><td>α</td><td>A∩B A∪B</td><td>α</td><td>A∩B A∪B</td></tr><tr><td>BALANCED</td><td>0.42</td><td>0.51</td><td>0.48</td><td>0.63</td></tr><tr><td>RANDOM</td><td>0.53</td><td>0.63</td><td>0.53</td><td>0.64</td></tr><tr><td>PARA</td><td>0.43</td><td>0.50</td><td>0.50</td><td>0.64</td></tr></table>

Table 6: Agreement on highlights. For pairs that at least two annotators classified a paraphrase, we display the average lexical overlap between the highlights (Jaccard Index displayed as $\frac { \ r _ { A \cap B } } { \ r _ { A \cup B } }$ ) and Krippendorff’s unitizing α over all words for guest and host highlights, see Krippendorff (1995).

## 5.1 Annotation Task

Given a (guest, host) utterance pair, annotators (1) classify whether the host is paraphrasing any part of the guest’s utterance and, if so, (2) highlight the paraphrase in the guest and host utterance. This results in data points like the one in Figure 1. Note that our setup differs from prior work, which usually involves classifying whether an entire text B is a paraphrase of an entire text A (e.g., Dolan and Brockett, 2005). Instead, given texts A and B, our task is to determine whether there exists a selection of words from text B and text A, where the selection of text B is a paraphrase of the selection of text A. Our annotators are not only performing binary classification, but they also highlight the position of the paraphrase. To the best of our knowledge, we are the first to approach paraphrase detection in this way. Moreover, in contrast to previous work, the considered text pairs are usually longer than just one sentence and are contextualized dialog turns.

## 5.2 Plausible Label Variation

The task of annotating context-independent paraphrases is already difficult. Disagreements between human annotators are common (Dolan and Brockett, 2005; Krishna et al., 2020; Kanerva et al., 2023) — even with extensive manuals for annotators (Kanerva et al., 2023). In related semantic tasks like textual entailment,<sup>11</sup> disagreements have been linked to plausible label variations inherent to the task

<table><tr><td>Shortened Examples</td></tr><tr><td>G: we don&#x27;t really know what went into their algorithm</td></tr><tr><td>to make it turn out that way. H: We&#x27;re talking about algorithms, buts should we be</td></tr><tr><td>talking about the humans who design the algorithms?</td></tr><tr><td>G: In Harrison County.</td></tr><tr><td>H: In Harrison County. Are you [...]</td></tr><tr><td></td></tr></table>

Table 7: Low Quality Annotations. We show human highlights that can be considered wrong or noisy. When absent, we underline the correct highlights.

(Pavlick and Kwiatkowski, 2019; Nie et al., 2020;   
Jiang and de Marneffe, 2022).

Our task setup adds further challenges: First, instead of classifying full sentence pairs, annotators have to read relatively long texts and decide whether any portion of the text pair is a paraphrase. Second, while in previous work annotators usually had to decide if two texts are generally approximately equivalent, they now need to identify paraphrases in a highly contextual setting with often incomplete information.

As a result, similar to the task of textual entailment, we expect classifying context-dependent paraphrases in dialog to not always have a clear ground truth. We display examples of plausible label variation in Table 1. To handle label variation, common strategies are performing quality checks with annotators (Jiang and de Marneffe, 2022) and recruiting a larger number of annotators for a single item (Nie et al., 2020; Sap et al., 2022). We do both, see our approach in §5.3 and §5.4.

## 5.3 Annotator Training

When annotating paraphrases, the instructions for annotators are often short, do not explain challenges and rely on annotator intuitions (Dolan and Brockett, 2005; Lan et al., 2017).<sup>12</sup> In contrast, Kanerva et al. (2023) recently used an elaborate 17-page manual. However, they relied on only 6 expert annotators that might not be able to represent the full complexity of the task (§5.2). We aim for a trade-off between short intuition-based and long complex instructions that facilitates recruitment of a larger number of annotators: an accessible example-centric, hands-on 15-minute training of annotators that teaches our operationalization of context-dependent paraphrases (§3). We provide (1) a short paraphrase definition, (2) examples of context-dependent paraphrases showing clear and approximate equivalence (c.f. Table 2), (3) examples of common difficulties with paraphrase classification in dialog (c.f. Table 3 and §3), and use (4) a hands-on approach where annotators have to already classify and highlight paraphrases after receiving instructions. Only once they make the right choice on what is (Table 2) and is not a paraphrase (Table 3) and highlight the correct spans they are shown the next set of instructions. Only annotators that undergo the full training and pass two comprehension and two attention checks are part of our released dataset. Overall, 49% of the annotators who finished the training passed it. See App. C for the instructions and further details.

<table><tr><td></td><td colspan="4">Classification</td><td colspan="3">Highlighting</td></tr><tr><td>Model</td><td>Extract ↓</td><td>F1↑</td><td>Prec ↑</td><td>Rec ↑</td><td>Extract↓</td><td>Jacc Guest ↑</td><td>Jacc Host ↑</td></tr><tr><td>llama 2 7B</td><td>1%</td><td>0.66</td><td>0.49</td><td>0.98</td><td>59%</td><td>0.34</td><td>0.44</td></tr><tr><td>vicuna 7B</td><td>1%</td><td>0.29</td><td>0.67</td><td>0.19</td><td>32%</td><td>0.30</td><td>0.46</td></tr><tr><td>Mistral 7B Instruct v0.2</td><td>3%</td><td>0.62</td><td>0.66</td><td>0.58</td><td>66%</td><td>0.40</td><td>0.51</td></tr><tr><td>openchat 3.5</td><td>0%</td><td>0.66</td><td>0.76</td><td>0.58</td><td>64%</td><td>0.46</td><td>0.50</td></tr><tr><td>gemma 7B</td><td>1%</td><td>0.64</td><td>0.66</td><td>0.63</td><td>48%</td><td>0.24</td><td>0.51</td></tr><tr><td>Mixtral 8x7B Instruct v0.1</td><td>0%</td><td>0.74</td><td>0.73</td><td>0.74</td><td>65%</td><td>0.35</td><td>0.52</td></tr><tr><td>Llama 2 70B</td><td>0%</td><td>0.66</td><td>0.72</td><td>0.61</td><td>71%</td><td>0.29</td><td>0.56</td></tr><tr><td>GPT-4</td><td>0%</td><td>0.81</td><td>0.78</td><td>0.84</td><td>17%</td><td>0.67</td><td>0.71</td></tr><tr><td>DeBERTa v3 large AGGREGATED</td><td>■</td><td>0.73</td><td>0.67</td><td>0.81</td><td>■</td><td>0.52</td><td>0.66</td></tr><tr><td>DeBERTa v3 large ALL</td><td>■</td><td>0.66</td><td>0.82</td><td>0.56</td><td>■</td><td>0.45</td><td>0.64</td></tr></table>

Table 8: Modeling Results. We boldface the best and underline the second best performance. We display the extraction error of predictions from generative models and, for classification, the F1, precision and recall score as well as, for highlights, the Jaccard Index for the guest and host utterances. Higher values are better ( ) except for extraction errors ( ). GPT-4 is the best classification model, while, overall, DeBERTa is the best highlight model as it does not lead to any extraction errors.

## 5.4 Annotator Allocation

To the best of our knowledge, text pairs in paraphrase datasets receive a fixed number of 1, up to a maximum of 5 annotations (Kanerva et al., 2023; Zhang et al., 2019; Lan et al., 2017; Dolan and Brockett, 2005). However, this might not be enough to represent the inherent plausible variation to the task (§5.2). We have each pair in BAL-ANCED annotated by 20–21 trained annotators to simulate different annotator allocation strategies (App. C.4). Then, for RANDOM and PARA, we use a dynamic allocation strategy: Each pair receives at least 3 annotations. We dynamically collect more annotations, up to 15, on pairs with high disagreement (i.e., entropy > 0.8). Overall, this results in an average of 9 annotations per text pair across our released dataset.

## 5.5 Results

We discuss annotations results (tables 1, 4, 6) on our datasets BALANCED, RANDOM and PARA.

Classification agreement as an indicator of variation. Agreement for classification is relatively low (Table 1). We inspect a sample of 100 annotations on the RANDOM set and manually assess annotation quality. 90% of the annotations can be said to be at least plausible (see Table 7 for low quality and Table 1 for plausible variation examples), which is in line with the fact that we only use high quality annotators (§5.3). Further, we manually analyze the 42 annotations of ten randomly sampled annotators: Nine annotators consistently provide high quality annotations, while the other annotator chooses “not a paraphrase” a few times too often (see Appendix C.7 for details). As a result, we assume that most disagreements are due to the inherent plausible label variation of the task (§5.2).

Higher agreement on paraphrase position. Krippendorff’s unitizing α on the highlights is higher than in other areas<sup>13</sup> (see Table 6). We also calculate the “Intersection-over-union” between the highlighted words (i.e., Jaccard Index), a common and interpretable evaluation measure for annotator highlights (Herrewijnen et al., 2024; Mendez Guzman et al., 2022; Mathew et al., 2021; Malik et al., 2021). It seems that while annotations vary on whether there is a paraphrase or not, they agree frequently on the position of the possible paraphrase. On average, at least 50% of the highlighted words are the same between annotations.<sup>14</sup> Agreement is higher on the host utterance, because on average the host utterance is shorter than the guest utterance (33 < 85 words).

<table><tr><td>T</td><td>Preds G D</td><td>Shortened Examples</td><td></td></tr><tr><td></td><td>X x</td><td>√</td><td>G: He was the most famous guy in the world of sports... H: The most famous Italian...</td></tr><tr><td></td><td>√ X √</td><td></td><td>G: A lot of them were the Bay Area influx that came up and bought homes to flip. You know what flipping is, right? H: Mm-hmm. Buying a house, improving</td></tr></table>

Table 9: Model Errors. We show examples of prediction errors made by DeBERTa (D) and GPT-4 (G). We display model predictions (D/G) for paraphrases (✓) and non-paraphrases (✗) and compare it to the crowdmajority (T). If one model predicted a paraphrase the corresponding text spans are underlined. For comparison, we also display the crowd majority highlights.

Label variation is highest for paraphrases. Between the datasets, classification agreement is lowest for PARA. This is what we expected since it has the largest portion of “hard” non-repetition paraphrases (see App. B.3). Krippendorff’s α is lower for the RANDOM than the BALANCED set, even though we expected the RANDOM set to include easier decisions for annotators (RAN-DOM includes more unrelated non-paraphrases, see App. B.3). As the other agreement heuristic is relatively high on RANDOM, the lower α values could be a result of Krippendorff’s measure being sensitive to imbalanced label distributions (Riezler and Hagmann, 2022), see also Table 4 displaying the imbalanced distribution for RANDOM.

## 6 Modeling

In Table 5, we do a random 70, 15, 15 split of our 5,581 annotations, along the 600 unique pairs.

Token Classifier. Similar to Wang et al. (2022a), we fine-tune a large DeBERTa model<sup>15</sup> (He et al., 2020) on token classification to highlight the paraphrase positions (for hyperparameters, see App. D.2). We train two models: using all 3,896 training annotations (“ALL” in Table 8) and using the majority aggregated training annotations over the 420 unique (guest, host) training pairs (“AG-GREGATED” in Table 8). We consider a model to have predicted a paraphrase for a pair if at least one token is highlighted with softmax probability $\geq 0 . 5$ in both texts. For each model, we average performances over three seeds.

<table><tr><td>Shortened Example</td></tr><tr><td>G: then he goes on andreferencesand</td></tr><tr><td>makes mention of Rudy Giuliani three times in this</td></tr><tr><td>conversation</td></tr><tr><td>H: And Rudy Giulianiwas a private lawyer not a gov-</td></tr><tr><td>ernment official, sowhy is he coming upso much in</td></tr><tr><td>this conversation between two world leaders?</td></tr></table>

Table 10: Highlighting Differences. We show examples of highlights made by DeBERTa, GPT-4 and human highlights. Lower intensity means less human annotators selected the word. While GPT-4 struggles with providing highlights at all (c.f. extraction error in Table 8), DeBERTa highlights tend to be too sparse (just “Rudy Giuliani”, “coming” and “conversation” in the host utterance). Here, we highlight words, when the softmax probability $\mathrm { i s > 0 . 4 \bar { 4 } ^ { 1 7 } }$ instead of $\ge 0 . 5 .$ . On the complete test set, this also increases the mean Jaccard Index (by 0.06/0.01 for guest/host compared to Table 8).

In-Context Learning. We further prompt the following generative models (see URLs in App. D.1) to both classify and highlight the position of paraphrases: Llama 2 7B and 70B (Touvron et al., 2023), Vicuna 7B (Zheng et al., 2023), Mistral 7B Instruct v0.2 (Jiang et al., 2023), Openchat 3.5 (Wang et al., 2023), Gemma 7B (Team et al., 2024), Mixtral 8x7B Instruct v0.1 (Jiang et al., 2024) and $\mathsf { G P T } \mathsf { - } 4 ^ { 1 6 }$ (Achiam et al., 2023). We design the prompt to be as close as possible to the annotator training using a few-shot setup (Brown et al., 2020; Zhao et al., 2021) with all 8 examples shown during annotator training. We also provide explanations in the prompt (Wei et al., 2022b; Ye and Durrett, 2022) and use selfconsistency by prompting the models 10 (GPT-4 and Llama 70B: 3) times (Wang et al., 2022b). For the prompt and further hyperparameter settings see App. D.1.

Results. For evaluation, we consider a pair to contain a paraphrase if it has been classified by a majority of crowd-workers and a word to be part of the paraphrase if it has been highlighted by a majority of crowd-workers. We leave softevaluation approaches to future work (Uma et al., 2021), among others because of challenges in extracting label distributions for in-context learning in a straight-forward way (Hu and Levy, 2023; Lee et al., 2023). See Table 8 for test set performances. Performances for the token classifier are the mean over three seeds. Performances for the generative models is the majority vote for the 3–10 self-consistency calls. We display the F1 score for classification and, as before (§5.5), Intersection-Over-Union of the highlighted words for guest and host utterance highlights (Jaccard Indices), see, for example, DeYoung et al. (2020). For in-context learning, we also display how often we could not extract the highlights or classifications from model responses. Note that the test set contains 93 elements, so differences between models might appear bigger than they are.

Overall, GPT-4 and Mixtral 8x7B achieve the best results in paraphrase classification. In highlighting, our DeBERTa token classifiers and GPT-4 achieve the best overlap with human annotations. However, due to problems with extracting highlights from model responses (e.g., hallucinations, see App. D.3), our fine-tuned DeBERTa token classifiers are probably the best choice to extract the position of paraphrases. While the DeBERTa AGGREGATED model achieves higher F1 scores, the DeBERTa ALL model has the highest precision out of all models. We provide our best-performing DeBERTa AGGREGATED model (model with seed 202 and F1 score of 0.76) on the Hugging Face Hub<sup>18</sup> and use it in the following error analysis.

Error Analysis. We consider the bestperforming classification and highlighting models for error analysis, i.e., GPT-4 and DeBERTa AGGREGATED. We manually analyze a sample of misclassifications, for examples see Table 9. Overall, the classification quality is better for GPT-4. The DeBERTa classifier finds more paraphrases (note that DeBERTa AGGREGATED for seed 202 has a recall of 0.86) but also predicts more false positives than GPT-4. For both models, the items with incorrect predictions also show higher human disagreement. The average entropy for human classifications is lower for the correct (0.45 for DeBERTa, 0.45 for GPT-4) than for the incorrect model predictions (0.59 for DeBERTa, 0.67 for GPT-4). DeBERTa highlights shorter spans of text (on average 6.6/6.2, compared to 16.7/10.9 for GPT-4 for guest/host respectively), while GPT-4 usually highlights complete (sub-)sentences. GPT-4 highlights are largely of good quality, however they often can not be extracted (see App. D.3). The DeBERTa highlights can seem “chopped up” and missing key information (e.g., the original host highlights in Table 10 are just “Rudy Giuliani”, “coming” and “conversation”). We recommend performing a classification of an utterance pairs as a paraphrase when there exist softmax probabilities  0.5 for both guest and host utterance, but then selecting the highlights also based on softmax probabilities lower than 0.5. Alternatively, the best DeBERTa ALL model<sup>19</sup> provides fewer but seemingly more consistent highlights (see Appendix D.3). One possible reason for this could be that DeBERTa ALL was trained on individual highlights provided by single annotators, rather than on aggregated highlights.

## 7 Conclusion

A majority of work on paraphrases in NLP has looked at the semantic equivalence of sentence pairs in context-independent settings. However, the human dialog setting is highly contextual and typical methods fall short. We provide an operationalization of context-dependent paraphrases and an up-scalable hands-on training for annotators. We demonstrate the annotation approach by providing 5,581 annotations on a set of 600 turn pairs from news interviews. Next to paraphrase classifications, we also provide annotations for paraphrase positions in utterances. In-context learning and token classification both show promising results on our dataset. With this work, we contribute to the automatic detection of paraphrases in dialog. We hope that this will benefit both NLP researchers in the creation of LLMs and social science researchers in analyzing paraphrasing in human-to-human or human-to-computer dialogues on a larger scale.

## Limitations

Even though the number of our unique text pairs is relatively small, we release a high number of high quality annotations per text pair (5,581 annotations on 600 text pairs). Releasing more annotations on fewer “items” (here: text pairs), has increasingly been more common in NLP (Nie et al., 2020; Sap et al., 2022). Further, big datasets become less necessary with better generative models: Using only eight paraphrases pairs in our prompt already led to promising results. We further use the full 3,896 annotations from the training set to train a token classifier showing competitive results with the open generative models. However, the token classifier and other potential fine-tuning approaches would probably profit from a bigger dataset.

Even though our dataset of news interviews showed frequent, different and diverse occurrences of paraphrasing, it might not be representative of paraphrasing behavior in conversations across different contexts and social groups. In the future, we aim to expand our dataset with further out-ofdomain items.

Our data creation process was not aimed at scalability. While our developed annotator training procedure can easily be scaled to a larger group of crowd-workers, we manually selected text pairs for annotation. Future work could scale this by skipping manual selection and accepting a more imbalanced dataset or using our trained classifiers as a heuristic to identify likely paraphrases.

Even though we carefully prepared the annotator training and took several steps to ensure highquality annotations, there remain several choices that were out of our scope to experiment with, but might have improved quality even more. For example, experimenting with different visualizations of paraphrase highlighting, text fonts, giving annotators an option to add confidence scores for classifications and so on.

We only use one prompt that is as close as possible to the instructions the human annotators receive. We use the same prompt with the exact same formatting for all different generative LLMs. However, experimenting with different prompts might improve performance (Weng, 2023) and some models might benefit from certain formatting or phrasing. We leave in-depth testing of prompts to future work. Further, it might be possible to improve the performance of our DeBERTa model, through providing contextual information (like speaker names and interview summary). Currently, these are only provided to the generative models.

In this work we collect a high number of human annotations per item and highlight the plausible label variation in our dataset. However, we use hard instead of soft-evaluation approaches (Uma et al., 2021) for the computational models. We do this because, among others, extracting label distributions for in-context learning is challenging (Hu and Levy, 2023; Lee et al., 2023). We leave the development of a soft evaluation approach to future work but want to highlight the potential of our dataset here: The high number of annotations per item enables the modeling of classifications and text highlights as distributions, similar to Zhang and de Marneffe (2021). Further, our dataset provides anonymized unique ids for all annotators and enables modeling of different perspectives, e.g., with similar methods to Sachdeva et al. (2022) and Deng et al. (2023).

We do not differentiate between different communicative functions, intentions or strategies that affect the presence of paraphrases in a dialog. This is relevant as paraphrases might, for example, be a more conscious choice by interviewers (Clayman and Heritage, 2002) or a more unconscious occurrence similar to the linguistic alignment of the references for discussed objects (Xu and Reitter, 2015; Garrod and Anderson, 1987). With this work, we hope to provide an outline of the general class of context-dependent paraphrases in dialog that lays the groundwork for further, fine-grained distinctions.

## Ethical Considerations

We hope that the ethical concerns of reusing a public dataset (Zhu et al., 2021) are minimal. Especially, since the CNN and NPR interviews are between public figures and were broadcast publicly, with consent, on national radio and TV.

Our dataset might not be representative of English paraphrasing behavior in dialogs across different social groups and contexts as it is taken from U.S. news interviews with public figures from two broadcasters. We caution against using our models without validation on out-of-domain data.

We performed several studies with U.S.-based crowd-workers as part of this work. We payed participants a median of 11.41\$/h which is above federal minimum wage. Crowd-workers consented to the release of their annotations. We do not release identifying ids of crowd-workers.

We confirm to have read and that we abide by the ACL Code of Ethics. Beside the mentioned ethical considerations, we do not foresee immediate risks of our work.

## Acknowledgements

We thank the anonymous ARR reviewers for their constructive comments. Further, we thank the NLP Group at Utrecht University and, specifically, Elize Herrewijnen, Massimo Poesio, Kees van Deemter, Yupei Du, Qixiang Fang, Melody Sepahpour-Fard, Shane Kaszefski Yaschuk, Pablo Mosteiro, and Albert Gatt, for, among others, feedback on writing and presentation, discussions on annotator disagreement and testing multiple iterations of our annotation scheme. We thank Charlotte Vaaßen, Martin Wegmann and Hella Winkler for feedback on our annotation scheme. We thank Barbara Bziuk for feedback on presentation. This research was supported by the “Digital Society - The Informed Citizen” research programme, which is (partly) financed by the Dutch Research Council (NWO), project 410.19.007.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. GPT-4 technical report. Computing Research Repository, arXiv:2303.08774.

Ion Androutsopoulos and Prodromos Malakasiotis. 2010. A survey of paraphrasing and textual entailment methods. Journal ofArtificial Intelligence Research, 38:135–187.

BIG bench authors. 2023. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research.

Rahul Bhagat and Eduard Hovy. 2013. Squibs: What is a paraphrase? Computational Linguistics, 39(3):463– 472.

Dwight Bolinger. 1974. Meaning and form. Transactions of the New York Academy of Sciences, 36(2 Series II):218–233.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc.

Samuel Carton, Qiaozhu Mei, and Paul Resnick. 2018. Extractive adversarial networks: High-recall explanations for identifying personal attacks in social media posts. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3497–3507, Brussels, Belgium. Association for Computational Linguistics.

Santiago Castro. 2017. Fast Krippendorff: Fast computation of Krippendorff’s alpha agreement measure. https://github.com/pln-fing-udelar/ fast-krippendorff.

Eve V Clark. 1992. Conventionality and contrast: Pragmatic principles with lexical consequences. In Frames, Fields, and Contrasts: New Essays in Semantic and Lexical Organization, pages 171–188. Lawrence Erlbaum Associates.

Herbert H Clark. 1996. Using language. Cambridge University Press.

Steven Clayman and John Heritage. 2002. The news interview: Journalists and public figures on the air. Cambridge University Press.

Mark G Core and James Allen. 1997. Coding dialogs with the DAMSL annotation scheme. In AAAI Fall Symposium on Communicative Aaction in Humans and Machines, volume 56, pages 28–35. Boston, MA.

Wayne A Davis. 2002. Meaning, expression and thought. Cambridge University Press.

Naihao Deng, Xinliang Zhang, Siyang Liu, Winston Wu, Lu Wang, and Rada Mihalcea. 2023. You are what you annotate: Towards better models through annotator representations. In Findings ofthe Association for Computational Linguistics: EMNLP 2023, pages 12475–12498, Singapore. Association for Computational Linguistics.

Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong, Richard Socher, and Byron C. Wallace. 2020. ERASER: A benchmark to evaluate rationalized NLP models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4443–4458, Online. Association for Computational Linguistics.

Justin Dieter, Tian Wang, Arun Tejasvi Chaganty, Gabor Angeli, and Angel X. Chang. 2019. Mimic and rephrase: Reflective listening in open-ended dialogue. In Proceedings ofthe 23rd Conference on Computational Natural Language Learning (CoNLL), pages 393–403, Hong Kong, China. Association for Computational Linguistics.

William B. Dolan and Chris Brockett. 2005. Automatically constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005).

Qingxiu Dong, Xiaojun Wan, and Yue Cao. 2021. ParaSCI: A large scientific paraphrase dataset for longer paraphrase generation. In Proceedings ofthe

16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 424–434, Online. Association for Computational Linguistics.

Sean P. Engelson and Ido Dagan. 1996. Minimizing manual annotation cost in supervised training from corpora. In 34th Annual Meeting of the Association for Computational Linguistics, pages 319–326, Santa Cruz, California, USA. Association for Computational Linguistics.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Simon Garrod and Anthony Anderson. 1987. Saying what you mean in dialogue: A study in conceptual and semantic co-ordination. Cognition, 27(2):181– 218.

H Paul Grice. 1957. Meaning. The philosophical review, 66(3):377–388.

H Paul Grice. 1975. Logic and conversation. In Speech acts, pages 41–58. Brill.

Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2020. DeBERTa: Decoding-enhanced bert with disentangled attention. Computing Research Repository, arXiv:2006.03654.

Elize Herrewijnen, Dong Nguyen, Floris Bex, and Kees van Deemter. 2024. Human-annotated rationales and explainable text classification: a survey. Frontiers in Artificial Intelligence, 7:1260952.

Joe Hight and Frank Smyth. 2002. Tragedies & journalists: A guide for more effective coverage. Dart Center for Journalism and Trauma.

Clara E Hill. 1992. An overview of four measures developed to test the Hill process model: Therapist intentions, therapist response modes, client reactions, and client behaviors. Journal of Counseling & Development, 70(6):728–739.

Graeme Hirst. 2003. Paraphrasing paraphrased. In Keynote address for The Second International Workshop on Paraphrasing: Paraphrase acquisition and Applications.

Jennifer Hu and Roger Levy. 2023. Prompting is not a substitute for probability measurements in large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5040–5060, Singapore. Association for Computational Linguistics.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel,

Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7B. Computing Research Repository, arXiv:2310.06825.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. Computing Research Repository, arXiv:2401.04088.

Nan-Jiang Jiang and Marie-Catherine de Marneffe. 2022. Investigating reasons for disagreement in natural language inference. Transactions ofthe Associationfor Computational Linguistics, 10:1357–1374.

Dan Jurafsky and James H Martin. 2019. Speech and language processing (3rd ed. draft).

Jenna Kanerva, Filip Ginter, Li-Hsin Chang, Iiro Rastas, Valtteri Skantsi, Jemina Kilpeläinen, Hanna-Mari Kupari, Aurora Piirto, Jenna Saarni, Maija Sevón, et al. 2021. Annotation guidelines for the Turku paraphrase corpus. Computing Research Repository, arXiv:2108.07499.

Jenna Kanerva, Filip Ginter, Li-Hsin Chang, Iiro Rastas, Valtteri Skantsi, Jemina Kilpeläinen, Hanna-Mari Kupari, Aurora Piirto, Jenna Saarni, Maija Sevón, and et al. 2023. Towards diverse and contextually anchored paraphrase modeling: A dataset and baselines for finnish. Natural Language Engineering, page 1–35.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199– 22213.

Venelin Kovatchev, M. Antònia Martí, and Maria Salamó. 2018. ETPC - a paraphrase identification corpus annotated with extended paraphrase typology and negation. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA).

Klaus Krippendorff. 1980. Content analysis: An introduction to its methodology. Sage publications.

Klaus Krippendorff. 1995. On the reliability of unitizing continuous data. Sociological Methodology, pages 47–76.

Kalpesh Krishna, John Wieting, and Mohit Iyyer. 2020. Reformulating unsupervised style transfer as paraphrase generation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 737–762, Online. Association for Computational Linguistics.

Wuwei Lan, Siyu Qiu, Hua He, and Wei Xu. 2017. A continuously growing dataset of sentential paraphrases. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing,

pages 1224–1234, Copenhagen, Denmark. Association for Computational Linguistics.

Noah Lee, Na Min An, and James Thorne. 2023. Can large language models capture dissenting human voices? In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4569–4585, Singapore. Association for Computational Linguistics.

Vijit Malik, Rishabh Sanjay, Shubham Kumar Nigam, Kripabandhu Ghosh, Shouvik Kumar Guha, Arnab Bhattacharya, and Ashutosh Modi. 2021. ILDC for CJPE: Indian legal documents corpus for court judgment prediction and explanation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4046–4062, Online. Association for Computational Linguistics.

Binny Mathew, Punyajoy Saha, Seid Muhie Yimam, Chris Biemann, Pawan Goyal, and Animesh Mukherjee. 2021. HateXplain: A benchmark dataset for explainable hate speech detection. Proceedings of the AAAI Conference on Artificial Intelligence, 35(17):14867–14875.

Erick Mendez Guzman, Viktor Schlegel, and Riza Batista-Navarro. 2022. RaFoLa: A rationaleannotated corpus for detecting indicators of forced labour. In Proceedings ofthe Thirteenth Language Resources and Evaluation Conference, pages 3610– 3625, Marseille, France. European Language Resources Association.

William R Miller and Stephen Rollnick. 2012. Motivational interviewing: Helping people change. Guilford press.

Yixin Nie, Xiang Zhou, and Mohit Bansal. 2020. What can we learn from collective human opinions on natural language inference data? In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9131–9143, Online. Association for Computational Linguistics.

Ellie Pavlick and Tom Kwiatkowski. 2019. Inherent disagreements in human textual inferences. Transactions ofthe Associationfor Computational Linguistics, 7:677–694.

Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Édouard Duchesnay. 2011. Scikit-learn: Machine learning in python. Journal ofMachine Learning Research, 12:2825–2830.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing

and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Stefan Riezler and Michael Hagmann. 2022. Validity, reliability, and significance: Empirical methodsfor NLP and data science. Springer Nature.

Carl Ransom Rogers. 1951. Client-centered therapy: Its current practice, implications, and theory. Houghton Mifflin, Boston.

Carla Roos. 2022. Everyday Diplomacy: dealing with controversy online and face-to-face. Ph.D. thesis, University of Groningen.

Pratik Sachdeva, Renata Barreto, Geoff Bacon, Alexander Sahn, Claudia von Vacano, and Chris Kennedy. 2022. The measuring hate speech corpus: Leveraging rasch measurement theory for data perspectivism. In Proceedings ofthe 1st Workshop on Perspectivist Approaches to NLP @LREC2022, pages 83–94, Marseille, France. European Language Resources Association.

Maarten Sap, Swabha Swayamdipta, Laura Vianna, Xuhui Zhou, Yejin Choi, and Noah A. Smith. 2022. Annotators with attitudes: How annotator beliefs and identities bias toxic language detection. In Proceedings ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 5884–5906, Seattle, United States. Association for Computational Linguistics.

Skipper Seabold and Josef Perktold. 2010. Statsmodels: econometric and statistical modeling with python. SciPy, 7:1.

Gail Sedorkin, Amy Forbes, Ralph Begleiter, Travis Parry, and Lisa Svanetti. 2023. Interviewing: A guide for journalists and writers. Routledge.

Raj Sanjay Shah, Faye Holt, Shirley Anugrah Hayati, Aastha Agarwal, Yi-Chia Wang, Robert E Kraut, and Diyi Yang. 2022. Modeling motivational interviewing strategies on an online peer-to-peer counseling platform. Proceedings of the ACM on Human-Computer Interaction, 6(CSCW2):1–24.

Vered Shwartz and Ido Dagan. 2016. Adding context to semantic data-driven paraphrasing. In Proceedings of the Fifth Joint Conference on Lexical and Computational Semantics, pages 108–113, Berlin, Germany. Association for Computational Linguistics.

Andreas Stolcke, Klaus Ries, Noah Coccaro, Elizabeth Shriberg, Rebecca Bates, Daniel Jurafsky, Paul Taylor, Rachel Martin, Carol Van Ess-Dykema, and Marie Meteer. 2000. Dialogue act modeling for automatic tagging and recognition of conversational speech. Computational Linguistics, 26(3):339–374.

Yu Su and Xifeng Yan. 2017. Cross-domain semantic parsing via paraphrasing. In Proceedings ofthe 2017 Conference on Empirical Methods in Natural Language Processing, pages 1235–1246, Copenhagen, Denmark. Association for Computational Linguistics.

Jamar Sullivan Jr., Will Brackenbury, Andrew McNutt, Kevin Bryson, Kwam Byll, Yuxin Chen, Michael Littman, Chenhao Tan, and Blase Ur. 2022. Explaining why: How instructions and user interfaces impact annotator rationales when labeling text data. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 521–531, Seattle, United States. Association for Computational Linguistics.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. Computing Research Repository, arXiv:2403.08295.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. Computing Research Repository, arXiv:2307.09288.

Alexandra N Uma, Tommaso Fornaciari, Dirk Hovy, Silviu Paun, Barbara Plank, and Massimo Poesio. 2021. Learning from disagreement: A survey. Journal of Artificial Intelligence Research, 72:1385–1470.

Gregory M Vecchi, Vincent B Van Hasselt, and Stephen J Romano. 2005. Crisis (hostage) negotiation: current strategies and issues in high-risk conflict resolution. Aggression and Violent Behavior, 10(5):533–551.

Gregory M Vecchi, Gilbert KH Wong, Paul WC Wong, and Mary Ann Markey. 2019. Negotiating in the skies of hong kong: The efficacy of the behavioral influence stairway model (BISM) in suicidal crisis situations. Aggression and violent behavior, 48:230– 239.

Marta Vila, M Antònia Martí, and Horacio Rodríguez. 2014. Is this a paraphrase? What kind? Paraphrase boundaries and typology. Open Journal ofModern Linguistics, 4(01):205.

Chris Voss and Tahl Raz. 2016. Never split the difference: Negotiating as if your life depended on it. Random House.

Jan Philip Wahle, Bela Gipp, and Terry Ruas. 2023. Paraphrase types for generation and detection. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12148– 12164, Singapore. Association for Computational Linguistics.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353–355, Brussels, Belgium. Association for Computational Linguistics.

Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, and Yang Liu. 2023. OpenChat: Advancing open-source language models with mixedquality data. Computing Research Repository, arXiv:2309.11235.

Shuohang Wang, Ruochen Xu, Yang Liu, Chenguang Zhu, and Michael Zeng. 2022a. ParaTag: A dataset of paraphrase tagging for fine-grained labels, NLG evaluation, and data augmentation. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 7111–7122, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022b. Self-consistency improves chain of thought reasoning in language models. Computing Research Repository, arXiv:2203.11171.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. 2022c. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5085–5109, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Harry Weger Jr, Gina R Castle, and Melissa C Emmett. 2010. Active listening in peer interviews: The influence of message paraphrasing on perceptions of listening skill. International Journal of Listening, 24(1):34–49.

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2022a. Finetuned language models are zero-shot learners. International Conference on Learning Representations.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022b. Chain-of-thought prompting elicits reasoning in large language models. In

Advances in Neural Information Processing Systems, volume 35, pages 24824–24837. Curran Associates, Inc.

Joseph Weizenbaum. 1966. Eliza—a computer program for the study of natural language communication between man and machine. Communications of the ACM, 9(1):36–45.

Lilian Weng. 2023. Prompt engineering. lilianweng.github.io.

Ka Wong and Praveen Paritosh. 2022. k-Rater Reliability: The correct unit of reliability for aggregated human annotations. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 378– 384, Dublin, Ireland. Association for Computational Linguistics.

Yang Xu and David Reitter. 2015. An evaluation and comparison of linguistic alignment measures. In Proceedings ofthe 6th Workshop on Cognitive Modeling and Computational Linguistics, pages 58–67, Denver, Colorado. Association for Computational Linguistics.

Xi Ye and Greg Durrett. 2022. The unreliability of explanations in few-shot prompting for textual reasoning. In Advances in Neural Information Processing Systems, volume 35, pages 30378–30392. Curran Associates, Inc.

Xinliang Frederick Zhang and Marie-Catherine de Marneffe. 2021. Identifying inherent disagreement in natural language inference. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4908–4915, Online. Association for Computational Linguistics.

Yuan Zhang, Jason Baldridge, and Luheng He. 2019. PAWS: Paraphrase adversaries from word scrambling. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1298–1308, Minneapolis, Minnesota. Association for Computational Linguistics.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 12697–12706. PMLR.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623. Curran Associates, Inc.

Chao Zhou, Cheng Qiu, and Daniel E Acuna. 2022. Paraphrase identification with deep learning: A review of datasets and methods. Computing Research Repository, arXiv:1503.06733.

Chenguang Zhu, Yang Liu, Jie Mei, and Michael Zeng. 2021. MediaSum: A large-scale media interview dataset for dialogue summarization. In Proceedings ofthe 2021 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 5927–5934, Online. Association for Computational Linguistics.

<table><tr><td></td><td colspan="2">Preprocessed</td><td colspan="2">Sampled</td><td colspan="2">Released</td></tr><tr><td></td><td># i</td><td>#gh</td><td>#i</td><td>#gh</td><td>#i</td><td>#gh</td></tr><tr><td>all</td><td>34419</td><td>148522</td><td>1304</td><td>4450</td><td>480</td><td>600</td></tr><tr><td>NPR</td><td>11506</td><td>49065</td><td>423</td><td>1550</td><td>167</td><td>218</td></tr><tr><td>CNN</td><td>22913</td><td>99457</td><td>881</td><td>2900</td><td>313</td><td>382</td></tr></table>

Table 11: Dataset Statistics. Number of interviews (#i) and (guest, host)-pairs (# gh) respectively after preprocessing (§4.1), random sampling (§4.2) and the selection of paraphrase candidates for annotation (§4.2).

## A Context-Dependent Paraphrases in Dialog

Should one include repetitions? Repetitions have been typically included in paraphrase taxonomies (Bhagat and Hovy, 2013; Zhou et al., 2022) even though, e.g., Kanerva et al. (2023) asked annotators to exclude such pairs as they considered them uninteresting paraphrases. However, distinguishing repetitions from paraphrases turns out to be especially hard in dialog: speakers tend to leave words out when they repeat and adapt the pronouns to match their perspective (e.g., I -> you). We therefore include repetitions in our definition of context-dependent paraphrases. In fact, those mainly make up the “Clear Contextual Equivalence” Paraphrases (see Table 2).

## B Dataset

Topic of the Dataset. The topics of the CNN and NPR news interviews (Zhu et al., 2021) are mostly centered around U.S. politics (e.g., presidential or local elections, 9/11, foreign policy in the middle east), sports (e.g., baseball, football), domestic natural disasters or crimes and popular culture (e.g., interviews with book authors).

Utterance Pair IDs. We use unique IDs for utterance pairs. For example, for NPR-4-2, “NPR-4” is the ID used for interviews<sup>20</sup> as done in Zhu et al. (2021), “2” is the position of the start of the guest utterance in the utterance list as separated into turns by Zhu et al. (2021), in this case “Thank you.”.

## B.1 Preprocessing

We give details on the three preprocessing steps (see §4.1).

1. Filtering for 2-person interviews. We filter 49,420 NPR and 414,176 CNN interviews from Zhu et al. (2021) for 2-person interviews only.

This can be challenging: In the speaker list, authors sometimes have non-unique identifiers (e.g., ‘STEVE PROFFITT’, ‘PROFFITT’ or ‘S. PROF-FITT’ refer to the same speaker). If one author identifier string is contained in the other we assume them to be the same speaker.<sup>21</sup> We generally assume the first speaker to be the host. We remove 538 NPR and 1,917 CNN interviews because the identifier of the second speaker includes the keywords “host” or “anchor” — thus contradicting our assumption. This leaves 14,000 NPR and 50,301 CNN 2-person interviews.

2. Removing first and last turns of an interview. The first turns in our 2-person interviews are usually (reactions to) welcoming addresses and acknowledgments by host and guest<sup>22</sup>, while the last often contain goodbyes or acknowledgments<sup>23</sup>. We remove the first two and the last two (guest, host)-pairs. This step removes 2,409 NPR and 26,419 CNN interviews because they are fewer than 5-turns long. For the remaining interviews, this removes 34,773 NPR and 71,646 CNN (guest, host)-pairs.

3. Removing short and long utterances. We further remove short guest utterances of 1–2 words as they leave not much to paraphrase.<sup>24</sup> 3,540 NPR and 12,675 CNN pairs are removed like this. We also remove pairs where the host utterance consists of only 1–2 words.<sup>25</sup>. 2,940 NPR and 11,389 CNN pairs are removed like this. We also remove pairs where guest or host utterance consist of more than 200 words.<sup>26</sup> Overall, this leaves 148,522 (guest, host)-pairs in 34,419 interviews for potential annotation, see Table 11.

![](images/f83ae1653529e981b5ff38ff4c9c57814cbb3a6226d492d6911481a8ce171583.jpg)  
Figure 2: Label distribution after first author annotations performed in two batches. First author label classification was performed in two batches. The first batch consists of 750 text pairs, the second of 3,700.

## B.2 First Author Annotations

We provide more details on the first author annotations for selecting paraphrase candidates (§4.2).

Deciding on first author annotations. Since the share of paraphrases in randomly sampled (guest, host)-pairs was only at around 5-15% in initial pilots with lab members, similar to previous work, we opted to do a pre-selection of text pairs before proceeding with the more resourceintensive paraphrase annotation (c.f. §5.5 and App. C). However, commonly used automatic heuristics were not suitable for the highly contextual discourse setting (c.f. §4.2). Instead, we experimented with discarding obvious “non-paraphrases” through crowd-sourced annotations and compared it to manual annotations by the lead author, ultimately deciding on using lead author annotations. One of the reasons was that discarding obvious “non-paraphrases” was more resource intensive and difficult for crowd-workers than expected, making the resources needed for discarding nonparaphrases too close to annotating paraphrases themselves – which defeats the purpose of doing a pre-selection in the first place.

Changing lead author annotations from discarding obvious non-paraphrases to keeping interesting paraphrases. On an initial set of 750 random (guest, host)-pairs, we remained with the initial idea of discarding obvious non-paraphrase pairs. However, due to a resulting high share of uninteresting or improbable paraphrase pairs, we

<table><tr><td>Paraphrase</td><td>88</td></tr><tr><td>High Lexical Similarity</td><td>59</td></tr><tr><td>Repetition</td><td>45</td></tr><tr><td>Perspective-Shift</td><td>10</td></tr><tr><td>Directional</td><td>17</td></tr><tr><td>Difficult Decision</td><td>16</td></tr><tr><td>Non-Paraphrase</td><td>519</td></tr><tr><td>High Lexical Similarity Partial</td><td>&gt; 18 &gt; 24</td></tr><tr><td>Unrelated</td><td>&gt; 103</td></tr><tr><td>Topically Related</td><td>&gt; 83</td></tr><tr><td>Conclusion</td><td>46</td></tr><tr><td></td><td></td></tr><tr><td>Ambiguous</td><td>18</td></tr><tr><td>Missing Context</td><td>125</td></tr></table>

Table 12: Statistics Labels First Batch. For 750 manually reviewed pairs, we also labeled several other categories. We found 88 paraphrases, 519 non-paraphrases, 18 ambiguous cases and 125 where the missing context impeded a definite decision. Note that we tried to not assign ambiguous if we were leaning to one category over another. Other categorizations include: “perspectiveshift” (the perspective shifts between guest and host, e.g., “you” -> “I”), “directional” (guest or host utterance is entailed from or subsumed in the other), “partial” (a subsection could be understood as a paraphrase, but the overall larger section is clearly not a paraphrase), “related” (two utterances are closely related but no paraphrases), “conclusion” (host draws a conclusion or adds an interpretation that goes beyond a paraphrase). Some labels were only added in the last 200 annotations and therefore include the “>” indication.

<table><tr><td>Dataset</td><td>Overlap Lead and Crowd</td></tr><tr><td>BALANCED</td><td>0.72</td></tr><tr><td>RANDOM</td><td>0.89</td></tr><tr><td>PARA</td><td>0.72</td></tr></table>

Table 13: Lead vs. Crowd Classifications. We display the average overlap between the lead author’s classifications and the majority vote of the crowd. The overlap is the highest on the RANDOM set. Probably because we keep all obvious non-paraphrases for classification and the annotators face less ambiguous (guest, host)-pairs to classify.

opted to classify paraphrases vs. non-paraphrases instead of possible paraphrases vs. obvious nonparaphrases. The lead author re-annotated the initial set of 750 paraphrase candidates and annotated 4450 additional (guest, host)-pairs for paraphrase vs. non-paraphrase. In the first batch, the lead author additionally labeled a variety of different paraphrase types/difficulties (e.g., high lexical similarity, missing context, unrelated), see also Table 12, in the second batch this was restricted to repetition paraphrase, paraphrase and non-paraphrase. The distribution of these three categories is displayed in Figure 2.

<table><tr><td>Type (guest, host)-pair</td><td>#</td><td>acc.</td></tr><tr><td>Paraphrase</td><td>46</td><td>0.80</td></tr><tr><td>High Lexical Similarity</td><td>24</td><td>0.92</td></tr><tr><td>Repetition Context-Dependent</td><td>16</td><td>0.88</td></tr><tr><td>Perspective-Shift</td><td>10</td><td>0.90</td></tr><tr><td>Directional</td><td>12</td><td>0.67</td></tr><tr><td>Other Difficult Cases</td><td>12</td><td>0.58</td></tr><tr><td>Non-Paraphrase</td><td>54</td><td>0.81</td></tr><tr><td>Unrelated utterances</td><td>13</td><td>1.00</td></tr><tr><td>More Difficult</td><td>41</td><td>0.76</td></tr><tr><td>Topically related</td><td>24</td><td>0.67</td></tr><tr><td>High Lexical Similarity</td><td>11</td><td>0.64</td></tr><tr><td>Partial</td><td>10</td><td>0.80</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Conclusion</td><td>11</td><td>0.55</td></tr></table>

Table 14: Selection of 100 Paraphrase Candidates for detailed Annotation. The sample was selected based on assigned categories during paraphrase candidate annotation. Categories within Paraphrase and Non-Paraphrase can overlap. We display “accuracy” w.r.t. first author annotations.

Relation to with Crowd Majority Annotations. We display the overlap between the lead author’s paraphrase classifications and the released classifications of the crowd majority in Table B.2.

## B.3 Paraphrase Candidate Selection

Based on the lead author classifications into paraphrase, non-paraphrase and repetition, we build three datasets for annotation (main paper §4.2). We display the first author classification distribution for the three datasets in Figure 3.

BALANCED. The BALANCED set is a sample of 100 (guest, host)-pairs that were randomly sampled based on the first batch of lead author annotations (§B.2). We had additional lead author labels available for this set, see Table 14 for the distribution of these on the BALANCED set. Constraints were 50 paraphrases and 50 non-paraphrases. In order to include more complex cases, we sampled more difficult than unrelated non paraphrase pairs and we limited the number of repetition paraphrases (51% of paraphrases are repetitions in the full batch, but only 33% of paraphrases in BAL-ANCED are repetitions). Due to a sampling error, we ended up with a 46/56 split. Later, we calculate the majority vote of the 20–21 annotations per (guest, host)-pair on this set, and then evaluate it by comparing it against the lead author classification, see “acc.” column.

![](images/248dad51f9d0d2597f0e37cc8d82a7c9227ac1edd68d78d83c19b71311757ed3.jpg)  
Figure 3: Distribution of Labels by Lead Author. We display the estimated number of (non-)paraphrases from the lead author annotations for the random subsample (RANDOM), the BALANCED sample and the wider paraphrase variety sample (PARA). Note, RANDOM consists of 100 elements, however only 98 are included in this statistic here (leading to numbers like 6.1). 2 pairs were not classified by the lead author because they were too ambiguous or were missing context information to reach a decision. We exclude such pairs in all other samples.

RANDOM. The random set is a sample of 100 (guest, host)-pairs that was uniformly sampled from the second batch of lead author annotations (§B.2).

PARA. After selecting the RANDOM set, the PARA set of 400 (guest, host)-pairs was sampled to reach a specified total 350 paraphrases and 150 non-paraphrases together with the RANDOM set.<sup>27</sup> The PARA set was selected to make the total number of non-repetition paraphrases together with RANDOM reach 300, while limiting the amount of repetition paraphrases to 50. Conversely, nonparaphrases were sampled to add up to 150. This led to 66 non-paraphrases and 334 paraphrases being sampled for the PARA set.

## C Annotations

## C.1 Development of Annotator Training.

The eventual study design used in this work (see §5) is the product of iterative improvement with lab members, other volunteers and Prolific annotators. They iterative steps can roughly be separated into:

(1) The lead author repeatedly annotated the same set of (guest, host)-pairs with a time difference of one week. See an example of early selfdisagreement in Table 15.

(2) With insights from (1) and our definition of context-dependent paraphrases, we created annotator instructions. We iteratively improved instructions while testing them with volunteers, lab members and Prolific crowd-workers. See examples of disagreements that led to changes in Table 15.

(3) Based on insights from (2), we introduced an intermediate annotator training that explains paraphrase annotation in a “hands-on” way: Annotators have to correctly annotate a teaching example to get to the next page instead of just reading an instruction. As soon as the correct selection is made, an explanation is show (e.g., Figures 6 and 10). After some testing rounds, we also require annotators to pass 2 attention (see Figure 12) as well as 2 comprehension checks (see Figures 5 and 11).

(4) We test the developed training on a selection of 20 (guest, host)-pairs out of which 10 were classified as clearly containing a paraphrase, and 10 as containing no paraphrase by the lead author, half of all examples we considered to be more difficult to classify (e.g., paraphrase with a low lexical overlap, non-paraphrase with a high lexical overlap). Two lab members reached pairwise Cohen of 0.51 after receiving training. Two newly recruited Prolific annotators reached average pairwise Cohen of 0.42 after going through training. Due to the inherent difficulty of the task and the good annotation quality when manually inspecting the 20 examples for each annotator, we carry on with this training setup.

## C.2 Annotator Training.

We train participants to recognize paraphrases (see Figure 4–13 for the instructions they received). We presented (guest, host)-pairs with their MediaSum summaries, the date of the interview and the interviewer names for context.<sup>28</sup> Participants were only admitted to the paraphrase annotation if they passed two attention checks (see Figure 12) and two comprehension checks (see Figure 5 and 11).

Comprehension Checks. Similar to examples in Table 2, they are presented with a clear paraphrase pair (App. Figure 5) and a less obvious context-dependent paraphrase pair (App. Figure 11) that they have to classify as a paraphrase. Additionally, they are only allowed to highlight the text spans that are a part of the paraphrase.

Training Stats. Of the initial 347 Prolific annotators who started the training, 95 aborted the study without giving a reason<sup>29</sup> and 126 were excluded from further studies because they failed at least one comprehension (29%) or attention check (24%) during training. Since annotators can perform annotations after training over a span of several days, we further exclude single annotation sessions, where the annotator fails any of two attention checks.

<table><tr><td rowspan=1 colspan=1>Who?</td><td rowspan=1 colspan=1>Example</td><td rowspan=1 colspan=1>see Instructions</td></tr><tr><td rowspan=4 colspan=1>Self-Disa-gree-ment</td><td rowspan=1 colspan=1>Guest: [..] So there was a consensus organization last year that people from genetics and</td><td rowspan=4 colspan=1>(C)distinguishparaphrases frominferences, con-clusions or “just&quot;”highly   relatedutterances</td></tr><tr><td rowspan=1 colspan=1>ethics law got together and said, in theory, it should be acceptable to try this in human</td></tr><tr><td rowspan=1 colspan=1>beings. The question will be, how much safety and evidence do we have to have from</td></tr><tr><td rowspan=1 colspan=1>animal models before we say it&#x27;s acceptable.Host: When it comes to this issue, let&#x27;s face it, while there are the concerns here in theUnited States, it&#x27;s happening in other countries.</td></tr><tr><td rowspan=9 colspan=1>LabMem-bersProlificAnno-tators</td><td rowspan=1 colspan=1>Guest: Hey, it&#x27;s going to be a long and a long week, and we&#x27;re going to use every singleminute of it to make sure that Americans know that Al Gore and Joe Lieberman are fightingfor working families, right here in Los Angeles and across America.Host: And are you guys ready to go?</td><td rowspan=3 colspan=1>(P) shortsubselections oftokens might be“paraphrases&quot; thatdo not adequatelyrepresent thecontent of theguest&#x27;s utterance</td></tr><tr><td rowspan=1 colspan=1>Guest: [...] There are militant groups out there firing against the military. And we just - wereally don&#x27;t know who is whom.Host: Why did the army decide today to move in and clear out the camp?</td></tr><tr><td rowspan=1 colspan=1>Guest: Police have indicated that they have been getting cooperation from the peopleinvolved, of course, they are looking at all of her personal relationships to see if there wereany problems there. [...]Host: Well what have family members told you? I know you&#x27;ve talked to various membersof her family. I understand she never missed her shifts at the restaurant where she worked.[...]</td></tr><tr><td rowspan=1 colspan=1>Guest: Yes, it is, all $640,000Host: That&#x27;s a lot of dough.</td><td rowspan=1 colspan=1>(CD) emphasizesituational aspectto annotators, (H)ask for token-levelaccuracy of high-lights</td></tr><tr><td rowspan=1 colspan=1>Guest: [...] He was an employee that worked downtown Cleveland and saw it fall out of thearmored car carrier, and pick it up, and took it, and placed it in his car.Host: And he&#x27;s been holding it ever since?</td><td rowspan=1 colspan=1>similar to (C)</td></tr><tr><td rowspan=1 colspan=1>Guest: [...] Would I ever thought that this would be happening, no, it is, it&#x27;s crazy? Just</td><td rowspan=3 colspan=1>(AT) use annota-tor screening tothrow out annota-tors more likelyto produce non-sensical pairs</td></tr><tr><td rowspan=1 colspan=1>enjoy the moment.Host: [...] , Magic Johnson was saying that when he first started taking meetings withinvestors or with business people, they didn&#x27;t take him seriously, but he thought maybe they</td></tr><tr><td rowspan=1 colspan=1>just wanted his autograph. [...]</td></tr><tr><td rowspan=1 colspan=1>Guest: [..] they say, you, you must sue “Fortnite&quot;, and I&#x27;m like, “Fortnite&quot;, what is that? Idon&#x27;t even know what it is —Host: So you weren&#x27;t even familiar?</td><td rowspan=1 colspan=1>(AT) throw out an-notators that donot select obviouspairs</td></tr></table>

Table 15: Examples of Disagreements in Paraphrase Annotation Pilots. All of the presented examples were highlighted by at least one annotator and selected as not showing any paraphrases at all by at least one other annotator. We show examples from three different conditions: Self-disagreement for the lead author, disagreements between volunteers/lab members and disagreements between Prolific annotators. These disagreements informed later training instructions: For (C), see Figure 6; for (P), see Figure 9; for (CD), see Figure 10; for (H), see Figure 8; for (AT), we chose the separate training setup with attention and comprehension checks, see Figures 5, 11 and 12. Early on, we chose to include repetitions in our paraphrase definition since it turned out to be conceptually difficult to separate the two – especially in a context-dependent setting (e.g., is “You don’t know.” a repetition of “I do not know it.” or not?), see Figure 4.

![](images/5ae73e1ec21e6767f60f4fd73fee2c7891c1e1095424c91a403a7059eeaf0d42.jpg)  
Figure 5: Annotator Training (2). Comprehension Check Paraphrase. Variations of the the shown highlighting are accepted.  
Figure 7: Annotator Training (4). Multiple Sentences.

![](images/306430695c87896e2be06264bf3ca5dcb77b861628a74a2a94b6c6bd1b913161.jpg)  
Figure 10: Annotator Training (7). Using context information  
Figure 9: Annotator Training (6). Partial vs actual paraphrase

![](images/501604b515bc311e5f719cda742877893f4b16beb6e705550170145c286b7039.jpg)  
Figure 12: Annotator Training (10). Two attention checks shown at different times during training.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Short Description</td><td rowspan=1 colspan=1>Example</td></tr><tr><td rowspan=1 colspan=1>In the GuestStatement</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Highlight</td><td rowspan=1 colspan=1>Referred-To is the target of the paraphrase</td><td rowspan=1 colspan=1>see below</td></tr><tr><td rowspan=1 colspan=1>In the HostReply</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=9 colspan=1>Highlight</td><td rowspan=9 colspan=1>Paraphrases reword or repeat what the guestsaidSubtleties- several ways of highlighting are acceptable- can include perspective shifts (here -&gt;New Jersey, I -&gt; you)</td><td rowspan=1 colspan=1>Guest: they say, you, you must sue &quot;Fortnite&quot;, andI&#x27;m like,&quot;Fortnite&quot;, what is that? I don&#x27;t even know what it isor: Guest: they say, you, you must sue &quot;Fortnite&quot;, and I&#x27;m like,&quot;Fortnite&quot;,what is that?I don&#x27;t even know what it isHost: So you weren&#x27;t even familiar?</td></tr><tr><td rowspan=1 colspan=1>Guest: That&#x27;s what we have been asking the senators to actuallycome and negotiate with usHost: I knowyou want to negotiate</td></tr><tr><td rowspan=1 colspan=1>Guest: She was very pleasant.Talked about family life. Theychatted about errands they need to run and things like that.</td></tr><tr><td rowspan=1 colspan=1>Host: Well,she talkeda lotabout her family and her kids. And you</td></tr><tr><td rowspan=1 colspan=1>get a personal sense for how they&#x27;re living day by day.</td></tr><tr><td rowspan=1 colspan=1>Guest (De Lacy Davis): I think the entire group would be amenable</td></tr><tr><td rowspan=1 colspan=1>toshipping him here to us, which is what we felt would be a betterenvironment to give him a new start.</td></tr><tr><td rowspan=1 colspan=1>Host: So that would be, actually, coming to New Jersey and being</td></tr><tr><td rowspan=1 colspan=1>under the auspices, frankly, of De Lacy Davis.</td></tr><tr><td rowspan=4 colspan=1>Do NotHighlight</td><td rowspan=1 colspan=1>additional content like consequences, inferencesGueor facts</td><td rowspan=1 colspan=1>st: if you&#x27;re not in our country, there are no constitutionalprotections for youHost: So, you don&#x27;t have a problem with Facebook giving thegovernment access to the private accounts of people applying toenter the U.S.?Guest: ... And so that&#x27;s the main question I&#x27;ve been asking peoplehere, is, was the price worth it?Host: You&#x27;re telling me people on the ground don&#x27;t see it that way.</td></tr><tr><td rowspan=1 colspan=1>vague references or (short) terms that are only</td><td rowspan=3 colspan=1>Guest: ... they are looking at all of her personal relationships to seeif there were any problems there.Host: Well what have family members told you? I know you&#x27;vetalked to various members of her family.</td></tr><tr><td rowspan=1 colspan=1>related in content but are not actually rephrasing</td></tr><tr><td rowspan=1 colspan=1>the guest&#x27;s point</td></tr></table>

Figure 13: Annotator Training (9). Overview Table shown to annotators

## C.3 Annotation After Training.

Next, the trained annotators were asked to highlight paraphrases. See Figure 14 for an example of the annotation interface. Annotators had access to a summary of their training at all times, see Figure 13. We again included two attention checks. Answers failing either attention check are removed from the dataset.

## C.4 Annotator Allocation Strategy

To the best of our knowledge, what constitutes a “good” number of annotators per item has not been investigated for paraphrase classification.

Summary. Based on the 20–21 annotations per item for the BALANCED set, we simulate fixed and dynamic strategies to recruit up to 20 annotations per item. We evaluate the different strategies w.r.t. closeness to the annotations of all 20–21 annotators. When considering resource cost and performance trade-offs, dynamic recruitment strategies performed better than allocating a fixed number of annotators for each item.

Details. We consider three different strategies for allocating annotators to an item: (1) using a fixed number for all items, (2) for each item, dynamically allocate annotators until n of them agree and (3) similar to Engelson and Dagan (1996), for each item, dynamically allocate annotators until the entropy is below a given threshold t or a maximum number of annotators has been allocated. We simulate each of these strategies using the annotations on BALANCED. We evaluate the strategies on (a) cost, i.e., the average number of annotators per item and (b) performance via (i) the overlap between the full 20 annotator majority vote (i.e., we assume this is the best possible result) and the predicted majority vote for the considered strategy and (ii) k-rater-reliability (Wong and Paritosh, 2022) — a measure to compare the agreement between aggregated votes. Note, for the dynamic setup we change the original calculation of kRR (Wong and Paritosh, 2022) by dynamically recruiting more or less annotators per item and thus aggregating the votes of a varying instead of a fixed number of annotators.

Results. See Figure 15 for the results. We selected a practical resource limit of an average 8 annotators per items and the requirement of at least 90% accuracy with the majority vote and 0.7 kRR (dotted lines). We decide on strategy (3) dynamically recruiting annotators (minimally 3, maximally 15) until entropy is below 0.8. Also with other min/max parameters this was a good trade-off between accuracy, kRR and average # of annotators. The average number of annotators needed per item is then about 6.8. In this way, most items receive annotations from 3 annotators, while difficult ones receive up to 15.

![](images/2adb815a53fefed45d4ae9c8bae38135bfdb57226bad859c7582e002edbd60c2.jpg)  
Figure 14: Interface for highlighting categories. Annotators are asked to highlight the categories on word level.

## C.5 Annotator Payment.

Via Prolific’s internal screening system, we recruited native speakers located in the US. Payment for a survey was only withheld if annotators failed two attention checks within the same survey or when a comprehension check at the very beginning of the study was failed<sup>30</sup> in line with Prolific guidelines.<sup>31</sup> Across all Prolific studies performed for this work (including pilots), we payed participants a median of $8 . 9 8 \mathcal { L } / h \approx 1 1 . 4 1 \mathfrak { P } / h ^ { 3 2 }$ which is above federal minimum wage in the US.<sup>33</sup>

![](images/3750ff029c200f41a614c3c3b39ca5f667d0f806a467ff236f41cc102bab3185.jpg)  
(a) Accuracy w.r.t. 20 annotators

![](images/b770f774eda526ca4584f427660355e982a76b96a083ef0602180b7ea7e4c7fb.jpg)  
(b) kRR

Figure 15: Annotator Recruitment Strategies. To decide the number of annotators for a specific item, we test three different strategies: (1) using a fixed number of annotators across all items (ALL), (2) increasing the number of annotators until at least n annotators agree for each item (absolute) and (3) increasing the number of annotators from 3 until the entropy is smaller than a given threshold (entropy) or a maximum of 10, 15 or 20 annotators is reached. We display the accuracy of the methods compared to using all 20 annotations in (15a) and the reliability measure kRR depending on the average number of annotators used (Wong and Paritosh, 2022) in (15b). We set a maximum average cost of 8 annotators per item and require a minimum accuracy of 90% as well as a minimum kRR of 0.70. When a strategy fulfills these requirements (i.e., falls in the upper left quadrants for (a) and (b)), we display the entropy thresholds for (3) and absolute number of annotators for (2).  
![](images/d11a1b1edaac22ead1216233135d93dbdcaf8d31ada4c95547be44fca9db1bb8.jpg)  
(a) Duration

![](images/7ca803ae908e9d8f4f6477bff8a84e9fd39b11752427a4fd9961764e63efd0af.jpg)  
(b) Quality Checks Passed  
Figure 16: On BALANCED, later training sessions take longer and pass fewer quality checks. In 16a, we display the seconds the nth annotator needs to go through the training session. The annotators are ordered according to the dates they completed training. Annotations were distributed across 6 different days in June 2023. The green line represents the median duration time of the first n participants. The red line displays the initially estimated completion time of 900 seconds according to pilot studies. The blue line is a linear regression estimate of the duration and it’s 95% confidence interval. On average, participants participating on a later date need more time to finish. In 16b, we display the summed number of the first n participants that passed the quality checks during training. The grey line represents the angle bisector, i.e., if every participant would pass all quality checks. Later participants are less likely to pass the quality checks.

## C.6 Varying Annotator Behavior over Time.

For the BALANCED set, we performed separate training and annotation rounds. See Figure 16 for the completion times and share of passed quality checks of Prolific annotators in the training session. Participants that were recruited later performed worse: they pass less quality checks and need more time. This effect was noticeable but it is not quite clear to us why this happens. We recruit all participants at once for later studies and not iteratively as for the BALANCED set, to avoid effects that have to do with study age. The effect on the quality of the released annotations should be minimal as we discard annotators that do not pass our quality checks. It does have an effect on the pay per hour for our participants, which we had initially estimated to be much higher.

## C.7 Intra-Annotator Annotations Quality

We manually randomly sample ten annotators (with anonymized PROLIFIC ids 60, 6, 86, 84, 47, 31, 68, 88, 41, 92) and analyze 42 of their annotatations. Nine annotators consistently provide plausible annotations, while the other annotator chooses “not a paraphrase” a few times too often. We also noticed some other annotator-specific tendencies, for example, one annotator might tend to highlight fewer words, more words or prefer exact lexical matches.

## C.8 Anonymization

We replace all Prolific annotator IDs with nonidentifiable IDs. We only make the non-identifiable IDs public.

## D Modeling

## D.1 In-Context Learning

Models. We provide the Huggingface URLs to our used models. Vicuna 7B: https:// huggingface.co/lmsys/vicuna-7b-v1.5, Mistral 7B Instruct v0.2: https://huggingface.co/ mistralai/Mistral-7B-Instruct-v0.2, Openchat: https://huggingface.co/openchat/ openchat-3.5-0106, Gemma 7B: https: //huggingface.co/google/gemma-7b-it, Mixtral 8x7B Instruct v0.1: https://huggingface. co/mistralai/Mixtral-8x7B-Instruct-v0.1, Llama 7B: https://huggingface.co/ meta-llama/Llama-2-7b-hf and Llama 70B: https://huggingface.co/meta-llama/ Llama-2-70b-hf.

Prompt. We use a few-shot prompt that is close to the original annotator training and instructions, see Figure 17. We use chain-of-thought like explanations, i.e., always starting with “Let’s think step by step.” and ending with “Therefore, the answer is”, (Kojima et al., 2022) and a few-shot setup showing all 8 examples showed to annotators during training (Figures 4–12). For GPT-4, we use a temperature of 1, self-consistency through prompting the model 3 times (Wang et al., 2022b) and the default top\_p nucleus sampling value of 1, a maximum of new tokens to 512. For all the huggingface models, we use a temperature of 1, self-consistency through prompting the model 10 times (only 3 times for Lllama 70B due to resource limits) and a top\_k sampling of the top 10 tokens, a maximum of new tokens of 400 for all other models. Note, there are many more prompts and choices we could have tried that are out-of-the scope of this work. Further steps could have included separating the classification and highlighting task, experimenting with further phrasings and so on. We leave this to future work.

## D.2 Token Classification

We use settings very close to Wang et al. (2022a) and test different learning rates and number of epochs with 3 different seeds each. We use the "save best model" option to save the model after the epoch which yielded the best result on the dev set. For the results, see Figure 16. We use a learning rate of 3e-3 and 12 epochs for further modeling.

<table><tr><td>Learning Rate</td><td>Epoch|</td><td>F1</td></tr><tr><td>1e-3</td><td>8</td><td> $0 . 6 1 \pm 0 . 0 4$ </td></tr><tr><td>3e-3</td><td>8</td><td> $0 . 6 4 \pm 0 . 0 6$ </td></tr><tr><td>5e-3</td><td>8</td><td> $0 . 5 2 \pm 0 . 1 5$ </td></tr><tr><td>3e-3</td><td>4</td><td> $0 . 6 5 \pm 0 . 0 7$ </td></tr><tr><td>3e-3</td><td>12</td><td> ${ \bf 0 . 6 5 \pm 0 . 0 0 }$ </td></tr><tr><td>3e-3</td><td>16</td><td> $0 . 6 0 \pm 0 . 1 0$ </td></tr></table>

Table 16: Hyperparameter tuning on the DEV set. We train a token classifier for learning rates 1e-3, 3e-3, 5e-3 and epochs 4, 8, 12 and 16 for 3 seeds. We keep learning rate fixed at 3e-3 when varying the number of epochs and epoch fixed at 8 when varyig the learning rates. Best options of learning rate and epoch are underlined. Best F1 score is boldfaced.

A P a r a p h r a s e i s a r e w o r d i n g o r r e p e t i t i o n o f c o n t e n t i n t h e g u e s t ' s s t a t e m e n t . I t r e p h r a s e s what t h e g u e s t s a i d .   
Given an i n t e r v i e w on − w i t h t h e summary : F r e s h P r i n c e S t a r A l fo n s o R i b e i r o Sues Over Dance Moves ; Rapper 2 M i l l y   
A l l e g e s His Dance Moves were Copied .   
G ue s t and Host s a y t h e fo l l o w i n g :   
Guest (TERRENCE FERGUSON, RAPPER) : I g u e s s i t was s e a s o n 5 when t h e y p r e m i e r e d i t i n t h e game . A bunch of DMs, a bunch   
o f T w i t t e r r e q u e s t s , e − m a i l s , e v e r y t h i n g was l i k e , you , y o u r game i s i n t h e dance , you need t o sue , " F o r t n i t e "   
s t o l e i t . Even l i k e b i g a r t i s t s , m ajor a r t i s t s l i k e J o e B u t t o n s and s t u f f , t h e y have t h e i r own l i k e show , d a i l y   
s t r u g g l e , t h e y say , you , you must s u e " F o r t n i t e " , and I ' m l i k e , " F o r t n i t e " , what i s t h a t ? I don ' t even know what i t   
i s −−   
Host (QUEST) : So you weren ' t even f a m i l i a r ?   
I n t h e r e p l y , d o e s t h e h o s t p a r a p h r a s e s o m e t h i n g s p e c i f i c t h e g u e s t s a y s ?   
E x p l a n a t i o n : Let ' s t h i n k s t e p by s t e p .   
T e r r e n c e F e r g u s o n s a y s a t t h e end o f h i s t u r n t h a t he didn ' t know F o r t n i t e .   
Quest , t h e h o s t o f t h e i n t e r v i e w , r e p e a t s t h a t t h e g u e s t doesn ' t know F o r t n i t e .   
So t h e y b o t h s a y t h a t t h e g u e s t d i d n ' t know F o r t n i t e . T h e r e f o r e , t h e a n s w e r i s y e s , t h e h o s t i s p a r a p h r a s i n g t h e g u e s t .   
V e r b a t i m Quote Guest : " I ' m l i k e , " F o r t n i t e " , what i s t h a t ? I don ' t even know what i t i s "   
V e r b a t i m Quote Host : " you weren ' t even f a m i l i a r ? "   
C l a s s i f i c a t i o n : Yes .   
Given an i n t e r v i e w on 2013 −10 −1 wi th t h e summary : . . .   
G u e s t and Host s a y t h e fo l l o w i n g :   
Guest (REP . RAUL LABRADOR (R) , IDAHO) :   
Host ( BLITZER ) :   
I n t h e r e p l y , d o e s t h e h o s t p a r a p h r a s e s o m e t h i n g s p e c i f i c t h e g u e s t s a y s ?   
E x p l a n a t i o n : Let ' s t h i n k s t e p by s t e p . EXPLANATION T h e r e fo r e , t h e answer i s yes , h o s t i s p a r a p h r a s i n g t h e g u e s t .   
V e r b a t i m Quote Guest : "We would l i k e t h e s e n a t o r s t o a c t u a l l y come and n e g o t i a t e w i t h us . "   
V e r b a t i m Quote Host : " you want t o n e g o t i a t e   
C l a s s i f i c a t i o n : Yes .   
ITEM   
E x p l a n a t i o n :   
V e r b a t i m Quote Guest : None   
V e r b a t i m Quote Host : None .   
C l a s s i f i c a t i o n : No .   
ITEM   
E x p l a n a t i o n :   
V e r b a t i m Q u o t e G u e s t : " S h e " " T a l k e d a b o u t f a m i l y l i f e . " " e r r a n d s t h e y n e e d t o r u n a n d t h i n g s l i k e t h a t . "   
V e r b a t i m Quote Host : " s h e t a l k e d " " a b o u t h e r fa m i l y and h e r k i d s . " " how t h e y ' r e l i v i n g day by day . "   
C l a s s i f i c a t i o n : Yes .   
ITEM   
E x p l a n a t i o n :   
V e r b a t i m Quote Guest : None .   
V e r b a t i m Quote Host : None .   
C l a s s i f i c a t i o n : No .   
ITEM   
E x p l a n a t i o n :   
V e r b a t i m Quote Guest : None .   
V e r b a t i m Quote Host : None .   
C l a s s i f i c a t i o n : No .   
ITEM   
E x p l a n a t i o n :   
V e r b a t i m Quote Guest : " s h i p p i n g him h e r e t o me"   
V e r b a t i m Quote H o s t : " coming t o New J e r s e y and b e i n g u n d e r t h e a u s p i c e s " " o f De Lacy D a v i s . "   
C l a s s i f i c a t i o n : Yes .   
ITEM   
E x p l a n a t i o n :   
V e r b a t i m Quote Guest : " I ' m t o s e e him . "   
V e r b a t i m Quote H o s t : " him " " h a v e a v i s i t from you "   
C l a s s i f i c a t i o n : Yes .   
Given an i n t e r v i e w on DATE wi th t h e summary : SUMMARY   
G u e s t a n d H o s t s a y t h e f o l l o w i n g :   
Guest (NAME) : UTTERANCE   
Host (NAME) : UTTERANCE   
E x p l a n a t i o n : Let ' s t h i n k s t e p by s t e p .  
Figure 17: Prompt Template close to Annotator Instructions The used prompt template is based closely on our annotator training and instructions. Phrasings were adapted to match the prompt-setting but kept the same where possible. See the full prompt in our Github Repository.

## D.3 Highlighting Analysis

We compare the highlights provided by DeBERTa AGGREGATED<sup>34</sup> and DeBERTa ALL<sup>35</sup> on 10 text pairs from the test set that were classified as paraphrases by both models. We provide examples in Table 17. DeBERTa ALL highlights are shorter, often more on point and arguably more consistent than DeBERTa AGGREGATED highlights. We also manually analyzed 10 text pairs from the test set that GPT-4 classified as paraphrases. We provide examples of GPT-4 highlights in Table 18. Generally, they seem of good quality, but have the tendency to span complete sub-sentences, even if not all is relevant.

Hallucinations. One of the biggest problems for in-context learning are the extractions of the highlighting from the model responses which has errors in up to 71% of the cases in Table 8. Most of these errors can be split into two categories: (1) inconsistent highlighting, where the model classifies a paraphrase but does not highlight text spans in both, the guest and host utterance and (2) hallucinations, where the model highlights spans that do not exist in that form in the guest or host utterance. Hallucination is more prevalent than inconsistent highlighting for GPT-4, where in most cases it leaves out words (e.g., “coming back to a normal winter” vs. “coming back daryn to a normal winter”), in some other cases it adds or replaces words (e.g.,“he’s a counterpuncher” vs. “he’s counterpuncher”), uses morphological variation (e.g., “you’ve” vs. “you have”) or quotes from the wrong source (e.g., from the host when considering the guest utterance). Most of these extraction errors seem to be resolvable by humans when looking at them manually, so it might be possible to address them in future work with a more advanced matching algorithm or by querying GPT-4 until one gets a parsable response. When looking at the classifications by GPT-4 they often seem plausible, even when counted as incorrect with the F1 score.

## D.4 Computing Infrastructure

The fine-tuning of 18 DeBERTa token classifier, and the inference of 7 generative models took about approximately 260 GPU hours with one A100 card with 80GB RAM on a Linux computing cluster.

We use scikit-learn 1.2.2 (Pedregosa et al., 2011), statsmodels 0.14.1 (Seabold and Perktold, 2010) and krippendorff 0.6.1 (Castro, 2017) for evaluation.

## E Use of AI Assistants

We used ChatGPT and GitHub Copilot for coding, to look up commands and sporadically to generate functions. Generated functions are marked in our code. Generated functions were tested w.r.t. expected behavior. We did not use AI assistants for writing.

<table><tr><td>AGG</td><td></td><td></td><td>Shortened Examples</td></tr><tr><td></td><td>√</td><td>x togo and help them.</td><td>G: There are people that are in that age range where we know they&#x27;re high risk,whyare they going to thesupermarket tobuy their own groceries?Get the community, the neighborhood H: if you&#x27;re going tohelp somebody byhelping them maybe get their groceries, how long</td></tr><tr><td></td><td></td><td></td><td>does the coronavirus live on surfaces? G: And people always prefer, of course, to see the pope as the principal celebrant of the mass. So that&#x27;s good. That&#x27;ll be tonight. And it will be his 26th mass and it will be the 40th or, rather, the 30th time that this is offered in round the world transmission. Andit will be my 20th time in doing it as a television commentatorfrom Romeso. H: Yes,</td></tr><tr><td></td><td></td><td></td><td>you&#x27;ve been doing this for a while now. G: Well, what happened was we finally waved down a Coast Guard helicopter. And what they were looking for were people with disabilities and medical conditions, which none of us really had. They didn&#x27;t lift any of us into the helicopter or anything. What they told us was to basicallywalkout of our house, up the street,trying tofight against the currentthat was</td></tr><tr><td></td><td>x</td><td>G: x</td><td>going theopposite wayof where we needed to go. H: Soyouwalked through that current! to get to the higher ground or get to a drier spot? They&#x27;ve now spent $6 million on thisBenghaziinvestigation. They keep coming up with more and more interviews. H: On Benghazi, Trey Gowdy now says your committee has interviewed 75 witnesses.</td></tr></table>

Table 17: DeBERTa ALL vs DeBERTa AGGREGATED highlights. Paraphrase highlights predicted by the best DeBERTa ALL (i.e., seed 201 with F1 score of 0.72) and the best DeBERTa AGG model (i.e., seed 202 F1 score of 0.76, same as in the main paper). Even though DeBERTa AGG gets better F1 scores on classification, the DeBERTa ALL highlights are arguably more on point. For comparison, we also display the human highlights if they exist. Note, highlights can exist even if the crowd majority vote did not predict a paraphrase.

<table><tr><td>GPT-4</td><td></td><td>Shortened Examples</td></tr><tr><td>V</td><td>Library. x</td><td>H: Congressman, short of, though, having a thank-you note attached to a check that went to the Clinton Library, what is it exactly that is going to prove that there was a quid pro quo, that these pardons were</td></tr><tr><td rowspan="2">x</td><td rowspan="2"></td><td></td></tr><tr><td>They&#x27;ve now spent $6 million on this Benghazi investigation. more and more interviews.</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td>H:On Benghazi, Trey Gowdy now says your committee has interviewed 75 witnesses.</td></tr><tr><td>G: [Trump]is appointing very young judges. H: [...] if you&#x27;re 50-plus, you&#x27;re probably too old for the Trump Administration to be seriously</td></tr></table>

Table 18: GPT-4 highlights. Paraphrase highlights predicted by GPT-4. For comparison, we also display the human highlights if they exist. Note, highlights can exist even if the crowd majority vote did not predict a paraphrase.