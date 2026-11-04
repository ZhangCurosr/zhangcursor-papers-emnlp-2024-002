# To Word Senses and Beyond: Inducing Concepts with Contextualized Language Models

Bastien Liétard and Pascal Denis and Mikaela Keller University of Lille, Inria, CNRS, Centrale Lille, UMR 9189 - CRIStAL, F-59000 Lille, France first\_name.last\_name@inria.fr

## Abstract

Polysemy and synonymy are two crucial interrelated facets of lexical ambiguity. While both phenomena are widely documented in lexical resources and have been studied extensively in NLP, leading to dedicated systems, they are often being considered independently in practictal problems. While many tasks dealing with polysemy (e.g. Word Sense Disambiguiation or Induction) highlight the role of word’s senses, the study of synonymy is rooted in the study of concepts, i.e. meanings shared across the lexicon. In this paper, we introduce Concept Induction, the unsupervised task of learning a soft clustering among words that defines a set of concepts directly from data. This task generalizes Word Sense Induction. We propose a bi-level approach to Concept Induction that leverages both a local lemma-centric view and a global cross-lexicon view to induce concepts. We evaluate the obtained clustering on SemCor’s annotated data and obtain good performance (BCubed F<sub>1</sub> above 0.60). We find that the local and the global levels are mutually beneficial to induce concepts and also senses in our setting. Finally, we create static embeddings representing our induced concepts and use them on the Word-in-Context task, obtaining competitive performance with the State-ofthe-Art.

## 1 Introduction

A crucial challenge in understanding natural language comes from the fact that the mapping between word forms and lexical meanings is manyto-many, due to polysemy (i.e., the multiplicity of meanings for a given form)<sup>1</sup> and synonymy (i.e., the multiplicity of forms for expressing a given meaning). Both polysemy and synonymy have been thoroughly studied in NLP, but mostly as independent problems, giving rise to dedicated systems. Thus, Word Sense Disambiguiation (WSD)

aims at correctly mapping word occurrences to one of its senses (Raganato et al., 2017), while Word Sense Induction (WSI), its unsupervised counterpart, aims at clustering word occurrences into latent senses directly from data (Manandhar et al., 2010; Jurgens and Klapaftis, 2013). More recently, researchers have proposed the task of Word-in-Context (WiC), which consists in classifying pairs of word occurrences depending on whether they realize the same sense or not (Pilehvar and Camacho-Collados, 2019). All these works take a word centric view, which aims at identifying or characterizing the different senses of a given word, where these senses are bound to a word. Another line of work, which takes a broader lexicon-wide perspective, is concerned with identifying synonyms, which are equivalence classes over different words that point to the same concept (Zhang et al., 2021; Ghanem et al., 2023), where concepts are semantic entities that are not bound to a word. In WordNet (Miller, 1995; Fellbaum, 1998), concepts are called synsets, defined as sets of synonyms. However, outside of lexical resources, synonymy and polysemy are usually considered as independent problems in the NLP literature. Yet, these two views are complementary. In lexicology, they correspond to two perspectives on the word-meaning mapping: semasiology and onomasiology. The former is the word-to-meanings view, where one can observe polysemy by looking at the different meanings a given word has. The latter is the meaning-to-words view, in which one can study synonymy by looking at the inventory of words that speakers use to express the same meaning.

In this paper, we propose a new task, called Concept Induction, that directly aims at learning concepts in an unsupervised manner from raw text. More precisely, this task aims at learning a soft clustering over a target lexicon (i.e., a set of words), in such a way that each cluster corresponds to a (latent) concept. Thus, this task both addresses polysemy (since polysemous words should appear in multiple clusters) and synonymy (since synonymous words should appear in the same cluster(s)). Inducing concepts can be interesting for many external applications, like building lexical resources for low-resources languages (Velasco et al., 2023), and can bring a different perspective in computational studies of meaning, moving the usual wordcentric focus to a more meaning-centric state.

Our approach to Concept Induction relies on word occurrences for a target lexicon, represented as word embeddings derived from a Contextualized Language Model (in this case, BERT Large (Devlin et al., 2019)), which are then grouped, using hard clustering algorithms, into concept denoting clusters. While these concept clusters could in principle be obtained directly from word occurrences, we propose a bi-level methodology that leverages both a local, lemma-centric clustering (i.e., operating on only specific word occurrences), and a global, cross-lexicon clustering (i.e., operating on all words occurrences). From this perspective, our approach generalizes, and in fact builds upon classical Word Sense Induction, in that word senses are learned jointly alongside with concepts. We hypothesize that an approach taking both complementary resolutions in account will lead to improved Concept Induction and Word Sense Induction, i.e. that the two objectives can be mutually beneficial.

To validate our approach, we carried out experiments on the SemCor dataset, which provides a set of concepts (taking the form of WordNet synsets) related to word occurrences. We found that our bi-level clustering approach accurately learn concepts, achieving $F _ { 1 }$ scores above 0.60 on the task of Concept Induction compared to WordNet’s synsets, outperforming competing approaches that use only local and global views. This demonstrates the benefits of our bi-level approach, and its ability to leverage both local and global views when inducing concepts. Interestingly, we show that the benefits go both ways: our proposed approach outperforms lemma-centric approaches when evaluated for WSI. Finally, we show that concept-aware static embeddings derived from our approach are also competitive with state-of-the-art approaches efficient on the Word-in-Context task, while using less training data. Through the new task of concept induction, we also contribute in a new way to the ongoing debate regarding the ability to align vector representations extracted from Contextualized Language Models to the semantic representations posited by (psycho-)linguists. In this vein, we conduct a qualitative evaluation of obtained clusters to ensure they indeed reflect concepts and gather synonyms. The source code we used for experiments is available at https: $/ / { \tt g i }$ thub.com/blietard/concept-induction.

## 2 Related Work

## 2.1 Lexical resources for concepts

Princeton’s WordNet (PWN) (Miller, 1995; Fellbaum, 1998) is a lexical database that has been been the most widely used as a reference for most wordsense-related tasks for many years. In Word-Net, the entry corresponding to a lemma has different wordsenses, each of them mapping to a synset. Synsets are WordNet’s equivalents of our concepts. Lemmas whose wordsenses belong to the same synset are synonymous. WordNet 3.0 contains 117,659 synsets and is built from the work of psycholinguists and lexicographers, that not only describes synonymy but also other lexical relations such as hypernymy/hyponymy, antonymy, meronymy/holonymy, etc. But the amount of resources needed to create such lexical databases with human experts is considerable, making them a very rare and precious resource. They are not available for a large number of active languages, and even more rare for dead languages (Bizzoni et al., 2014; Khan et al., 2022).

## 2.2 Word senses with Language Models

With the recent development of neural Contextualized Language Models (CLM), several work use their hidden-layers to extract vector representations of word usages and retrieve word senses. These representations are fed to a classification (for WSD) or a clustering (in the case of WSI) algorithm to distinguish the word’s senses (Scarlini et al., 2020; Nair et al., 2020; Saidi and Jarray, 2023). These embeddings-based approaches have applications in other fields: Kutuzov and Giulianelli (2020) and Martinc et al. (2020) use sense clusters found using CLM embeddings to study the change in meaning of words, and Chronis and Erk (2020) propose a many-Kmeans method to investigate semantic similarity and relatedness. Another line of work uses list of substitute tokens sampled from the CLM head to infer senses (Amrami and Goldberg, 2019; Eyal et al., 2022) and are sucessful on WSI benchmarks like Manandhar et al. (2010) and Jurgens and Klapaftis (2013).

## 2.3 Structures of Meaning in CLM

Recent research probes neural CLMs for alignements between representations from their latent spaces and semantic patterns and relations. Section 7.2 of Haber and Poesio (2024) summarizes findings about polysemy in contextualized CLMs, showing that these models were able to detect polysemy and in some cases distinguish actual polysemy from homonymy. They report that representations from different senses may however overlap. Hanna and Marecekˇ (2021) shows that pretrained BERT embed knowledge of hypernymy but is limited to the more common hyponyms.

Velasco et al. (2023) build on top of WSI techniques in an attempt to automatically construct a WordNet for Filipino, thus proposing a modeling of synonymy in this language. However, the evaluation of the synsets they obtained is limited by the lack of sense-annotated data for Filipino, and they could not evaluate the impact of their methodology on the two levels (senses and concepts).

Works like Ethayarajh (2019) and Chronis and Erk (2020) study the kind of information that was distributed across layers. The former concludes that syntactic and word-order information are distributed in the first layers while in deeper layers, representations are heavily influenced by contexts. The latter demonstrates, with a multi-prototypes embedding approach, that semantic similarity is best found in moderately late layers, while relatedness is best found in last layers.

## 3 Concept Induction

Our main motivation behind Concept Induction is to present a view of the mapping between words and their meaning(s).<sup>2</sup> This view is systemic, meaning that it should not be defined for individual words neither for individual concepts, but rather acknowledging these as a whole with interactions and relations. This extends beyond the primary objective of WSI, which defines word senses as pertaining to individual words only and does not explore relations between lemmas or concepts.

## 3.1 Basic notions

Consider a set of target words (or lemmas) and for each lemma, we have a set of occurrences of this word in a context (e.g. a sentence or a phrase). The set of target lemmas is referred to as the lexicon, while the corpus is the set of all occurrences. Our goal is to study the meaning of target words as they are used in the corpus.

In this study we call sense of a word its usage to refer to a concept. A polysemous word has multiple senses, each of them referring to a distinct concept. Two words are said to be synonyms for a given concept when each of them has one of their senses referring to this shared concept. Senses are defined “locally”, i.e. bound to an individual word of the lexicon, as opposed to concepts which are defined “globally”, i.e. across the whole lexicon. An occurrence of a word w realizes one of w’s senses. Consider the words “test” and “trial” and the following corpus: (A) the jury found them guilty in a fair trial. (B) candidates competed in a trial of skill. (C) the hero underwent a test of strength. The corpus is composed of two occurrences of “trial” and one occurrence of “test.” In the corpus, “trial” is polysemous. Its first sense, illustrated in A, refers to a process oflaw. Its second sense, in B, refers to the concept of the act of undergoing testing. The sense of “test” in sentence C also corresponds to this concept: it’s a case where “test” and “trials” are synonymous. Shifting the focus from senses to concepts, we will say that B and C instantiate the same concept, while A is an instance of a different concept.

## 3.2 Task definition

The goal of Concept Induction (CI) is to automatically learn a set of concepts directly from the data, i.e. learning a soft clustering $C ^ { \dot { W } }$ in the set of target words W that should correspond to the multiple concepts instantiated by occurrences of the corpus. $C ^ { \hat { W } }$ is a soft clustering because a word can be assigned to several clusters (when it is polysemous). Using a different perspective than WSI, the framework of Concept Induction provides a more complete view on meaning across the lexicon. Both WSI and CI capture polysemy, but CI also reveals synonymy across the lexicon. Like WSI, Concept Induction does not require a pre-defined set of concepts.

## 3.3 Formal framework

Let W be the lexicon. For all word w in W, we denote $o _ { i } ^ { w }$ the i-th occurrence of w in the corpus. We define $O ^ { w } = \{ o _ { i } ^ { w } \} _ { 1 \leq i \leq m _ { w } }$ the set of $m _ { w }$ occurrences of w. The corpus, denoted O, is the union of all $O ^ { w }$

![](images/8e47ffe3dec78460433d8ba7348ab657faec098ece71114e48a07f2748fb3388.jpg)  
Figure 1: Illustration of our framework. The words “trial” is polysemous and has two senses corresponding to two different concepts, and is synonym with “test” for this second meaning.

For a given word $w \in W$ , the set $O ^ { w }$ can be partitionned according to its different senses. We denote $s _ { j } ^ { w }$ the part of occurrences of w in the corpus corresponding to the j-th sense of w. We refer to these groups of occurrences as the sense clusters of w. The set $S ^ { w } = \{ s _ { j } ^ { w } \} _ { 1 \leq j \leq n _ { w } }$ forms a partition of $O ^ { w }$ , and we call S the set of all sense clusters of all words, i.e. $\textstyle S = \bigcup _ { w \in W } S ^ { w }$ . S is a “local” (lemmacentric) partition of the whole O. The task of Word Sense Induction aims at learning the partition S given a corpus O.

In this work, we aim at dividing the corpus into concepts instead of senses. We denote $c _ { k }$ the group of occurrences of words corresponding to the concept indexed by $k ,$ and $C ~ = ~ \{ c _ { k } \} _ { 1 \leq k \leq p }$ the partition of O in $p$ concept clusters. Unlike sense clusters of $S ,$ , a concept cluster $c _ { k } \in C$ can gather occurrences of different words: C is a “global” partition. Each occurrence $o _ { i } ^ { w }$ of a word $w \in W$ is associated to a sense cluster $s _ { j } ^ { w }$ and a concept cluster $c _ { k } \in C$ . We can say that a concept corresponding to $c _ { k }$ is instantiated by occurrence $o _ { i } ^ { w }$ through the sense corresponding to $s _ { j } ^ { w }$ , or conversely that $o _ { i } ^ { w }$ uses the sense reflected in $s _ { j } ^ { w }$ to mean the concept described by concept cluster $c _ { k }$ . All occurrences of sense cluster $s _ { j } ^ { w } \in S$ appear in the same concept cluster $c _ { k } \in C .$

In summary, S and $C$ are partitions of O and are naturally constrained as follows:

1. By definition, a sense in S is associated to one and only one word $w \in W$

2. An occurrence $o _ { i } ^ { w }$ realizes exactly one sense $s _ { j } ^ { w } \in S$

3. An occurrence $o _ { i } ^ { w }$ instantiates exactly one concept $c _ { k } \in C$

4. In a given sense $s _ { j } ^ { w } \in S$ , all occurrences are assigned to the same concept $c _ { k } \in C$

5. All $s _ { j } ^ { w } \in S ^ { w }$ (i.e. same word) refer to distinct concepts.

From the partition $C$ on occurrences, one can derive $C ^ { W }$ , a clustering of the set of words W into concepts. To each concept cluster $c _ { k } \in C$ we associate a cluster in $C ^ { W }$ that contains all lemmas of W whose occurrences were assigned to $c _ { k }$ . In $C ^ { W }$ , a polysemous word with n senses appears in n distinct clusters (one per sense), and synonyms appear in at least one common cluster (one per shared concept).

We denote $\hat { C } ^ { W }$ the word-level soft-clustering and $\hat { C }$ the partition of occurrences that are learned on the data.

In Figure 1 we illustrate this framework, using a corpus of occurrences of the words “test” and “trial”. In this scenario, $W = \{ { } ^ { \ast } \mathrm { t e s t } ^ { \ast } , { } ^ { \ast } \mathrm { t r i a l } ^ { \ast } \}$ and two concepts are instantiated: a process oflaw to determine someone’s guilt and a challenge to evaluate a skill. The lemma “trial” exhibits two senses as it has occurrences corresponding to both concepts: “trial” is polysemous. The second concept is also instantiated by occurrences of “test”, therefore “trial” and “test” show synonymy in this case. This toy example also follows all constraints formulated above.

## 4 Methodology

In this section we describe the methods we propose and evaluate for Concept Induction. We learn a clustering $\hat { C } ^ { W }$ drawing inspiration from the relations between O, S, $C$ and $C ^ { W }$ . In particular, the overall objective of our methodology consist in finding $C$ (i.e. partition occurrences into concept clusters) to derive $C ^ { W }$ . Section 3.3 highlighted that there are two levels of partitions: a local level (senses) and a global one (concepts). The proposed approaches rely on both levels and the use of a Contextualized Language Model (CLM) to gather representations of occurrences influenced by the context.

## 4.1 Proposed Bi-level Method

Local (lemma-centric) clustering Firstly, we propose to learn a word-sense partition for each target words individually. Using the CLM hidden layers, we extract a vector representation (the occurrence embedding) of every occurrence $o _ { i } ^ { w }$ We then learn a partition $\hat { S ^ { w } }$ of each $O ^ { w }$ using a clustering algorithm on the embeddings. Each $\hat { S } ^ { w }$ describes the locally estimated sense clusters of word w. Jointly considering these partitions for all $w \in W$ , we obtain a partition $\hat { S }$ of the whole set of occurrences $O .$ This partition is local in the sense that each word has its occurrences clustered independently from other words.

Global (cross-lexicon) clustering Once we have a local clustering ${ \hat { S } } ,$ we turn from considering words independently to consider all words together. In this step, we learn a global clustering by merging local clusters of occurrences. To do so, we average embeddings of all occurrences in the same local cluster to get a single embedding representing each local cluster. Then we run a second clustering algorithm, this time using the averaged embeddings of local clusters. This global clustering defines a new partition $\hat { C }$ of the the corpus O: when two local clusters $\hat { s } _ { j } ^ { w _ { 1 } }$ and $\hat { s } _ { j ^ { \prime } } ^ { w _ { 2 } }$ are merged into the same global cluster $\hat { c } _ { k }$ (because their embeddings were clustered together), all their occurrences are assigned to global cluster $\hat { c } _ { k }$ . From this global occurrence partition $\hat { C }$ we can easily extract $\hat { C } ^ { W }$ , a word-level soft-clustering of lemmas whose occurrences appear in the same $\hat { c } _ { k }$

This Bi-level method directly implements the system of contraints described in Section 3.3. Only constraint 5 is not enforced by design. Indeed, our local clusters being learned and not informed by an expert, the local clustering step may make errors, especially if the data for a given word are sparse. Allowing the global clustering to merge local clusters enables the correction of local clustering’s recall errors using information from the global level.

We also want to highlight that the proposed methodology is generic, in the sense that it is not tied to a specific choice of clustering algorithm.

## 4.2 Local-only and Global-only

Sense-inducing systems (WSI approaches) that create only local clusters of occurrences for each word are said to be Local-only systems. We use them as baseline models that only produce word-level clusters of size 1 and do not reflect synonymy, but still learn polysemy.

On the other hand, consider a system in which each occurrence is mapped to its own local cluster (i.e. no actual local clustering step), and the global step divides occurrences directly into global clusters. We refer to this kind of system as Global-only approaches. They allow to evaluate how useful the local clustering step is in the process: we hypothesize that the local step in Bi-level will reduce potential variance in occurrences by aggregating them, increasing Precision compared to Global-only.

## 5 Experiments

In this section, we evaluate the abilities of the proposed methods to induce concepts and compare the proposed bi-level approach to other methods. We investigate the advantages of the bi-level approach not only for the global viewpoint but also in the local setting.

## 5.1 Settings

Data. We choose to use the annotated part of the SemCor $3 . { \cal O } ^ { 3 }$ corpus. This dataset contains occurrences for a wide number of words, and morphosyntactic annotations provide their lemma and their Part-of-Speech tag. Among all lemmas having at least 10 annotated occurrences, we keep only nouns (excluding proper nouns)<sup>4</sup> composed only of alphabetical characters with a minimum length or 3 letters. The resulting lexicon W contains 1,560 different lemmas, for which we gather a corpus O containing a total of 52,997 occurrences<sup>5</sup>. SemCor is also semantically annotated, with each occurrence of a target lemma assigned to a synset in WordNet, that we consider to be the concept it refers to. We derive a reference partition of the occurrences C and a reference soft-clustering of the words $C ^ { W }$ from annotations, for a total of 3,855 different concepts (WordNet’s synsets) covered in O. This set of concepts is the subset of WordNet corresponding to the textual data.

Evaluation of Concept Induction We compare the learned word clustering $\hat { C } ^ { W }$ to the reference $C ^ { W }$ . We choose to use the BCubed metrics (Bagga and Baldwin, 1998), obtaining Precision and Recall for the evaluated clustering compared to the reference, as well as an $F _ { 1 }$ score. To account for overlapping clusters, we use the Extended BCubed metrics proposed by Amigó et al. (2009), which has already been used as evaluation in SemEval2013 WSI task (Jurgens and Klapaftis, 2013).

Using BCubed metrics, for a given evaluated clustering, low precision would mean that grouped lemmas should not have been clustered together because none of their occurrence annotations map to a shared concept according to annotations . Low recall means that the evaluated system fails to capture clusters of lemmas whose occurrences share a concept according to annotations. The number of common clusters between two words also impacts BCubed metrics: if two lemmas appear together in too many clusters compared to the reference clustering, precision is decreased; if the number of common clusters is too low, recall is decreased.

Development. To learn the clustering, candidate systems have access to the full set of occurrencesin-context but not their annotations. To choose the appropriate set of hyperparameters, we create a Dev split of the annotations by randomly sampling 10% of concepts and revealing semantic annotations of the corresponding occurrences. We use them to evaluate Concept Induction for this small set of concepts, and choose the set of hyperparameters that scores best in BCubed $F _ { 1 }$

Evaluation splits In the final evaluation phase, we compute scores on all concepts / all occurrences, including the Dev split, as concepts in it are part of the whole subset of WordNet described by Sem-Cor’s annotations. In the full data, we found that 88% of the concepts were instantiated using only a single lemma. To better evaluate cases of synonymy, we also evaluate systems on a subset of the corpus, denoted “Synon”, that contains only occurrences of concepts showing synonymy (the remaining 12% of concepts, instantiated through at least 2 distinct lemmas). Statistics are provided in Table 5 in Appendix B. Note that it only changes the set of concepts/lemmas for which the system is being evaluated, not the clustering’s training data.

## 5.2 Systems and baselines

Clustering Algorithms. We try two different clustering algorithms relying on different paradigms: Kmeans (used in Chronis and Erk (2020)), a centroid-based algorithm with a fixed number of clusters, and Agglomerative clustering (used in Saidi and Jarray (2023); Velasco et al. (2023); dubbed “Agglo” for short), a deterministic hierarchical approach using a distance threshold to create a dynamic number of clusters instead of using a fixed one. Another difference between Kmeans and Agglo is that the former assumes that expected clusters are of nearly-spherical shape and balanced in number of points, while the latter does not make assumptions on the shape of data. Details of tested hyperaprameter values are provided in Appendix C.

Representations. Following Chronis and Erk (2020) and Eyal et al. (2022), we use BERT Large (Devlin et al., 2019), a masked language model with 24 layers and 345M parameters. This allows for direct comparisons with these approaches. Also, BERT Large was found by Haber and Poesio (2021) to allow for better grouping of sense interpretations than other LLMs.<sup>6</sup> We average subwords’ embeddings if needed. It is a common practice in previous work on semantic-related tasks to use the average of the last 4 layers to get embeddings; we decided to adopt the same "4 layers average pooling" strategy, but trying with different possible sets of layers (see Appendix C). Therefore, for a set of four layers, we average hidden states across the selected layers to get a single 1024-dimensional vector. We found that layers 14 to 17 obtained the best results on Dev for all methods (global/local-only and bilevel).

<table><tr><td colspan="4">Full data</td><td colspan="3">Synon.</td></tr><tr><td>Concept Induction</td><td>P</td><td>R</td><td> $\mathrm { F } _ { 1 }$ </td><td>P</td><td>R</td><td> $\mathrm { F } _ { 1 }$ </td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Lemmas</td><td>1.0</td><td>.43</td><td>.61</td><td>1.0</td><td>.61</td><td>.50</td></tr><tr><td>Oracle WSI Local-only Systems</td><td>1.0</td><td>.75</td><td>.86</td><td>1.0</td><td>.39</td><td>.56</td></tr><tr><td>Kmeans Local</td><td>.73</td><td>.70</td><td>.71</td><td>.67</td><td>.38</td><td>.49</td></tr><tr><td>Agglo Local</td><td>.92</td><td>.53</td><td>.67</td><td>.92</td><td>.35</td><td>.50</td></tr><tr><td>Eyal et al. (2022)</td><td>.31</td><td>.75</td><td>.44</td><td>.37</td><td>.39</td><td>.38</td></tr><tr><td>CI Systems</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kmeans Global</td><td>.48</td><td>.65</td><td>.56</td><td>.68</td><td>.54</td><td>.60</td></tr><tr><td>Kmeans Bi-level</td><td>.70</td><td>.59</td><td>.64</td><td>.82</td><td>.47</td><td>.59</td></tr><tr><td></td><td>.61</td><td>.60</td><td>.60</td><td>.82</td><td></td><td></td></tr><tr><td>Agglo Global</td><td></td><td></td><td></td><td></td><td>.50</td><td>.62</td></tr><tr><td>Agglo Bi-level</td><td>.75</td><td>.60</td><td>.66</td><td>.86</td><td>.49</td><td>.62</td></tr></table>

Table 1: Concept Induction BCubed Precision (P), Recall (R) and F<sub>1</sub>on the SemCor data averaged over 5 runs.

Sense-inducing systems. Comparison to Localonly systems will give a (strong) baseline just by inducing senses without aiming at concepts. We used the same clustering algorithms. We also implement the WSI method proposed by Eyal et al. (2022). It relies on a different paradigm, using the Language Model for substitution instead of word embeddings. From lists of substitutes, they build a graph of substitutes in which they find communities and then assign each occurrence to a community of substitutes to find the wordsenses. Because Localonly methods only induce senses, their hyperparameters are chosen to maximize a WSI objective on polysemous words of the dev split.

Baselines We construct a candidate clustering $\hat { C } ^ { W }$ where each lemma has its own cluster. This baseline model is referred to as the “Lemmas” baseline. This is to evaluate the extent to which the information contained by the lemma alone can be used to induce concepts without any knowledge on word senses neither on context. As a second baseline, we create for each lemma as many singletons as the number of different concepts its occurrences are annotated with. All created clusters are of size 1: we account perfectly for polysemy but not at all for synonymy. This second baseline is dubbed “Oracle WSI”.

## 5.3 Concept Induction in SemCor

In Table 1 we display the Concept Induction scores $( \mathrm { F } _ { 1 } )$ of proposed baselines and systems on the full SemCor data and on the Synon. split. On the full data, both the Lemmas and Oracle WSI baselines achieve very good performance because they have, by design, a perfect precision (they do not cluster lemmas at all and do not overestimate the number of clusters) and because 88% of concepts are instantiated with only a single lemma (thus their recall is still good). However, they are very limited on the Synon. split of the data, where concepts are instantiated with multiple lemmas.

The proposed Concept Induction systems reach scores ranging from .56 to .66 on the full data, half of them outperforming the Lemmas baseline, and from .59 to .62 on the Synon. split, outperforming all other systems. While still challenging, it exhibits that it is indeed possible to induce WordNetbased concepts in a corpus using LMs hidden layers vectors.

We also see that Kmeans-based approaches are consistently outperformed by Agglomerative methods. This indicates that the representational spaces in LM hidden layers are not organized in a nearlyspherical fashion as Kmeans algorithm assumes, but rather are populated less uniformly. This is reflected in precision and recall: Agglomerative systems reach a higher precision than Kmeans with similar recall.

Overall, results are in favor of Bi-level approaches over Global-only systems, with substantial improvements in $\mathrm { F } _ { 1 }$ on the full data while obtaining (nearly) identical performance on concepts of multiple lemmas, and large increases in precision while the loss in recall is minimal. This demonstrates that considering the local (lemmacentric) perspective is beneficial to a global (crosslexicon) view when inducing concepts. The local clustering, with the subsequent representation averaging, helps reducing variance in occurrences and therefore allow to reach higher levels of precision in the global clustering compared to Global-only. We would also like to emphasize that, while Globalonly systems are more simple in design, their computational cost is usually higher than Bi-level ones, especially when the clustering algorithm’s time complexity is quadratic with respect to the number of occurrences.

## 5.4 Qualitative Analysis of Concepts Clusters

We manually annotate word clusters (obtained from our best-performing approach, the Agglo Bi-level system) containing at least 2 lemmas according to the semantic similarity between lemmas. Distribution of cluster sizes (in number of lemmas) can be found in Appendix D. We distinguish four categories: synonyms when lemmas are cognitive synonyms (e.g. “necessity” and “need”), nearsynonyms for lemmas close to be synonyms but showing slight difference in meaning (e.g. “duty” and “task”, the former being stronger than the latter),<sup>7</sup> related when lemmas show a topical (e.g. “dirt”, “sand” and “mud”) or lexical relations (e.g. antonyms like “man” and “woman”) and invalid clusters when lemmas show no semantic relation (e.g. “child” and “idea”).

<table><tr><td></td><td colspan="3">Cluster size</td></tr><tr><td></td><td>2</td><td>3</td><td>4+</td></tr><tr><td>Nb. of annotated clusters</td><td>50</td><td>50</td><td>23</td></tr><tr><td>Category (% of annotated clusters)</td><td></td><td></td><td></td></tr><tr><td>Synonyms</td><td>42</td><td>38</td><td>17</td></tr><tr><td>Near-synonyms</td><td>24</td><td>24</td><td>35</td></tr><tr><td>Related</td><td>26</td><td>36</td><td>48</td></tr><tr><td>Invalid</td><td>08</td><td>02</td><td>0</td></tr></table>

Table 2: Qualitative manual evaluation of obtained word clusters of size $\geq 2$

Proportions of these annotations are displayed in Table 2 with respect to the cluster size, the number of lemmas in the cluster. For a given cluster size, if the number of clusters exceeds 50, we randomly sample 50 clusters to be annotated. Overall, the proportion of synonyms and near-synonyms is generally above 50% and less than 10% of clusters are invalid, indicating that most learned concepts are reliable and meaningful. We argue that the remaining related term clusters, while not synonyms, may still be interesting in less fine-grained studies. The portion of related clusters is in line with findings from previous work showing that BERT was also reflective of other lexical relations, such as hypernymy (Hanna and Marecekˇ , 2021).

## 5.5 Benefits at the Local Level

We now turn back to the local level and assess whether the information brought at the global level helps distinguishing senses of individual words. Here we do not evaluate the word-level soft clustering, but the occurrence-level division of Sem-Cor’s data, considering each word independently. In other words, we evaluate WSI in SemCor using annotations as the reference sense clustering.

Evaluation of induced senses For each word $w \in W$ , we compare how its set of occurrences $O ^ { w }$ is divided in $\hat { \hat { C } }$ to how it is divided in the reference C provided by annotations using BCubed metrics, and we average scores obtained across W. We display the WSI BCubed $\mathrm { F } _ { 1 } .$ , as in previous WSI tasks like Jurgens and Klapaftis (2013). Following Amrami and Goldberg (2019), we report $\rho$ the Spearman correlation coefficient between the number of clusters a lemma is assigned to and its number of senses according to annotations, to ensure that the number of created senses actually scales with the actual degree of polysemy.

<table><tr><td colspan="2">WSI  $\mathbf { F _ { 1 } }$ </td><td rowspan="2"> $\rho$ </td></tr><tr><td>Local-only Systems</td><td></td></tr><tr><td>Kmeans Local</td><td>.61</td><td>NA</td></tr><tr><td>Agglo Local</td><td>.77</td><td>.04</td></tr><tr><td>Eyal et al. (2022)</td><td>.46</td><td>.51</td></tr><tr><td>CI Systems</td><td></td><td></td></tr><tr><td>Kmeans Global</td><td>.76</td><td>.51</td></tr><tr><td>Kmeans Bi-level</td><td>.78</td><td>.30</td></tr><tr><td>Agglo Global</td><td></td><td></td></tr><tr><td></td><td>.80</td><td>.53</td></tr><tr><td>Agglo Bi-level</td><td>.80</td><td>.46</td></tr></table>

Table 3: WSI BCubed $\mathrm { F } _ { 1 }$ and sense number correlation coefficient $\rho$ on SemCor full data. Not computed for Kmeans because the number of cluster is constant.

Note that, for CI systems, we evaluate the division of occurrences provided by the final clustering $\hat { C }$ (i.e. how occurrences are clustered after the global step and its potential merge operations). The quality of sense clusters induced by the local-step only is actually evaluated with Local-only systems.

Local results. Results of this local evaluation are displayed in Table 3. Let us recall that Localonly systems’ hyperparameters are chosen to maximize the WSI F<sub>1</sub>on the dev split, while those of CI systems maximize the Concept Induction $\mathrm { F } _ { 1 }$ Nonetheless, one can observe that all CI systems outperform their Local-only counterparts, achieving higher WSI $\mathrm { F } _ { 1 }$ and $\rho$ even though their hyperparameters are not chosen to match the WSI itself. This indicates that the information brought at the global level by considering cross-lexicon relations may indeed help improving WSI, and benefits between local and global levels go both ways.

We explain the relatively poor performance of State-of-the-Art WSI system by the fact that we are in a particular setting, where the number of occurrences per lemma is relatively low in SemCor (30 per lemma on average) and so is the average number of occurrences per concept. Data sparsity is a favorable ground for word senses to be misrepresented. As such, methods meant to be applied on larger datasets like the one of Eyal et al. (2022) may not work as well as expected. Our results show the limitations of these systems when the amount of training data is low and the interest of aiming at concepts to get senses. This scenario is motivated in areas where data are not available in large quantities and still require to induce senses. In the case of the study of Lexical Semantic Change (the evolution of word meanings over time), recent works perform WSI in diachronic corpora that are often unbalanced and small (Tahmasebi et al., 2021).

<table><tr><td>Model</td><td>Acc.</td></tr><tr><td>Eyal et al. (2022) (CBOW) Eyal et al. (2022) (Skip-Grams)</td><td>59.3 61.9</td></tr><tr><td>Ours (Agglo global)</td><td>58.8</td></tr><tr><td>Ours (Agglo bi-level)</td><td>59.7</td></tr></table>

Table 4: Accuracy scores on the nouns of the WiC test dataset (Pilehvar and Camacho-Collados, 2019).

## 6 Extrinsic Evaluation with Concept-aware Embeddings

In their work, Eyal et al. (2022) derive sense-aware static embeddings from their WSI method, training them on the Wikipedia dataset and used them for the Word-in-Context (WiC) task. They achieve nearly-SotA results on the dataset proposed by Pilehvar and Camacho-Collados (2019), and report to be outperformed only by methods using external lexical knowledge and resources. We proceed to the same extrinsic evaluation of our work, constructing concept-aware embeddings using concept clusters of Concept Induction systems (Global-only and Bi-level Agglo). To obtain such embeddings, we average all vectors representating occurrences in SemCor contained each global cluster to get one vector per concept cluster.

The WiC task consists of determining whether two occurrences of a target lemma w correspond to the same sense. The WiC dataset’s target words are nouns and verbs, but like in the rest of this paper, we restrict our scope to nouns.

To solve the task, we use BERT Large to create representations of the two target occurrences. Each of them is assigned to a concept by finding the closest concept-aware using cosine distance. The decision depends on whether the two occurrences are mapped to the same concept (true) or to distinct ones (false). Results are displayed in Table 4. Our concept-aware embeddings obtain very similar results to those of their sense-aware embeddings, with ours derived from our bi-level approach even outperforming their CBOW method. Interestingly, our embeddings were trained with far fewer resources than theirs, as we used 52 997 occurrences from the SemCor dataset while they used a dump of Wikipedia, gathering millions of occurrences. This emphasizes the value of concept-aware embeddings: the use of cross-lexicon information allows competitive results with fewer resources.

## 7 Conclusion

In this paper, we argued that, while word senses allow to investigate polysemy, concepts are a larger perspective that allows the study of polysemy as well as synonymy. We defined Concept Induction, the unsupervised task to learn a soft-clustering of words in a large lexicon, directly from their incontext occurrences in a corpus. Then, we proposed a formulation of this problem in terms of local (lemma-centric) and global (cross-lexicon) complementary views, and tested an approach that uses information from both levels using contextualized Language Models. On concept-annotated SemCor corpus, we found that this bi-level view was beneficial for Concept Induction, and even for Word Sense Induction with a low amount of training data. We validated the quality of obtained clusters with manual annotations, ensuring that clusters mostly correspond to actual synonyms and concepts. Finally, we showcased an external application of our methodology to create concept-aware embeddings that can be competitive to other methods on semantic tasks, such as Word-in-Context.

Concept Induction opens the way for a different perspective on lexical semantics in NLP, and can be a basis for many studies of lexical meanings as it is expressive enough to reflect relations on both sides of the word-meaning mapping.

## 8 Limitations

The formal framework we defined uses terminology and notions from rather structuralist/relational assumptions of the language’s lexical system (e.g. senses, discrete concepts, etc.). We made this choice based on how lexical databases like Word-Net (and its derivatives), or other like the Historical Thesaurus of English for instance, are designed using the "word/sense/concept" structure. From a purely practical point of view, this choice makes sense as these resources would be the primary source for task data’s annotations. Conceptually, senses are also a notion widely used in computational linguistics and we wanted to propose Concept Induction as a step "beyond" this conventional aspect and its related tasks. Future research may explore definitions/extensions of Concept Induction outside of this structuralist/relational framework, towards cognitive semantics for instance (Geeraerts, 2010).

Evaluating Concept Induction is mainly limited by the low amount of suitable annotated corpora. Not only the data need to be annotated in concepts, but these annotations must cover a wide variety of lemmas for synonymy to be sufficiently represented in the corpus. Future work may find or create datasets meeting these requirements to evaluate Concept Induction outside of SemCor.

For now, the study is limited to nouns. Performances of benchmarked algorithms and systems may change with other Part-of-Speech tags.

Our Bi-level method allows the global clustering to merge local clusters, leveraging lexicon-level information to be used to correct Word Sense Induction errors at the lemma-level. By its sequential nature, our method does not allow to split local clusters using global-level information, which could lead to better results. Further research directions include creating an iterative version of our methodology (alternating local and global clustering), or attempting to tackle both clustering objectives simultaneously with bi-level constrained clustering.

Our results about sense-induction at the local level showed that usual WSI methods may not be robust in our setting where there are few occurrences for some lemmas. We demonstrated that, in this setting, concept-inducing methods provided a better division in word senses. In many fields of linguistics, corpora are not very large and do not contain hundreds of occurrences for each word. Nonetheless, it is still uncertain if this observed advantage of CI systems would still hold on bigger datasets with many occurrences per lemma, a setting better-suited for usual WSI methods.

In this paper, we limited our study to Nouns, the morpho-syntactic class exhibiting the most prominent semantic features. We leave to further research the study of Concept Induction for Verbs, Adjectives, or the heterogeneous family of Adverbs.

## 9 Ethical Considerations

Our methodology uses pretrained Contextualized Language Models, which are know to encode and replicate social biases contained in their training data and sometimes amplify them. While we do not observe surface-level biases arising when manually annotating concept clusters, it is still an open question of how these social biases may influence or even change results when inducing concepts in SemCor.

## Acknowledgements

We gratefully thank the anynymous reviewers for their insightful comments. This research was funded by Inria Exploratory Action COMANCHE.

## References

Enrique Amigó, Julio Gonzalo, Javier Artiles, and Felisa Verdejo. 2009. A comparison of extrinsic clustering evaluation metrics based on formal constraints. Information retrieval, 12:461–486.

Asaf Amrami and Yoav Goldberg. 2019. Towards better substitution-based word sense induction.

Amit Bagga and Breck Baldwin. 1998. Entity-based cross-document coreferencing using the vector space model. In 36th Annual Meeting of the Association for Computational Linguistics and 17th International Conference on Computational Linguistics, Volume 1, pages 79–85, Montreal, Quebec, Canada. Association for Computational Linguistics.

Yuri Bizzoni, Federico Boschetti, Harry Diakoff, Riccardo Del Gratta, Monica Monachini, and Gregory Crane. 2014. The making of Ancient Greek WordNet. In Proceedings ofthe Ninth International Conference on Language Resources and Evaluation (LREC’14), pages 1140–1147, Reykjavik, Iceland. European Language Resources Association (ELRA).

Gabriella Chronis and Katrin Erk. 2020. When is a bishop not like a rook? when it’s like a rabbi! multiprototype BERT embeddings for estimating semantic relationships. In Proceedings of the 24th Conference on Computational Natural Language Learning, pages 227–244, Online. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Kawin Ethayarajh. 2019. How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 55–65, Hong Kong, China. Association for Computational Linguistics.

Matan Eyal, Shoval Sadde, Hillel Taub-Tabib, and Yoav Goldberg. 2022. Large scale substitution-based word sense induction. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4738–4752, Dublin, Ireland. Association for Computational Linguistics.

Christiane Fellbaum. 1998. WordNet: An electronic lexical database. MIT press.

Alexandre François. 2022. Lexical tectonics: Mapping structural change in patterns of lexification. Zeitschriftfür Sprachwissenschaft, 41(1):89–123.

Dirk Geeraerts. 2010. Theories of Lexical Semantics. Oxford University Press.

Sana Ghanem, Mustafa Jarrar, Radi Jarrar, and Ibrahim Bounhas. 2023. A benchmark and scoring algorithm for enriching Arabic synonyms. In Proceedings of the 12th Global Wordnet Conference, pages 274–283, University of the Basque Country, Donostia - San Sebastian, Basque Country. Global Wordnet Association.

Janosch Haber and Massimo Poesio. 2021. Patterns of polysemy and homonymy in contextualised language models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2021, pages 2663–2676, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Janosch Haber and Massimo Poesio. 2024. Polysemy— Evidence from linguistics, behavioral science, and contextualized language models. Computational Linguistics, 50(1):351–417.

Michael Hanna and David Marecek. 2021. ˇ Analyzing BERT’s knowledge of hypernymy via prompting. In Proceedings of the Fourth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networksfor NLP, pages 275–282, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Martin Haspelmath. 2023. Coexpression and synexpression patterns across languages: comparative concepts and possible explanations. Frontiers in Psychology, 14.

David Jurgens and Ioannis Klapaftis. 2013. SemEval-2013 task 13: Word sense induction for graded and non-graded senses. In Second Joint Conference on Lexical and Computational Semantics (\*SEM), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval 2013), pages 290–299, Atlanta, Georgia, USA. Association for Computational Linguistics.

Fahad Khan, Francisco J. Minaya Gómez, Rafael Cruz González, Harry Diakoff, Javier E. Diaz Vera, John P. McCrae, Ciara O’Loughlin, William Michael Short, and Sander Stolk. 2022. Towards the construction of a WordNet for Old English. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 3934–3941, Marseille, France. European Language Resources Association.

Andrey Kutuzov and Mario Giulianelli. 2020. UiO-UvA at SemEval-2020 task 1: Contextualised embeddings for lexical semantic change detection. In Proceedings of the Fourteenth Workshop on Semantic Evaluation, pages 126–134, Barcelona (online). International Committee for Computational Linguistics.

Suresh Manandhar, Ioannis Klapaftis, Dmitriy Dligach, and Sameer Pradhan. 2010. SemEval-2010 task 14: Word sense induction &disambiguation. In Proceedings ofthe 5th International Workshop on Semantic Evaluation, pages 63–68, Uppsala, Sweden. Association for Computational Linguistics.

Matej Martinc, Syrielle Montariol, Elaine Zosa, and Lidia Pivovarova. 2020. Capturing evolution in word usage: Just add more clusters? In Companion Proceedings of the Web Conference 2020, WWW ’20, page 343–349, New York, NY, USA. Association for Computing Machinery.

Aaron F McDaid, Derek Greene, and Neil Hurley. 2011. Normalized mutual information to evaluate overlapping community finding algorithms. arXiv preprint arXiv:1110.2515.

George A Miller. 1995. Wordnet: a lexical database for english. Communications ofthe ACM, 38(11):39–41.

Sathvik Nair, Mahesh Srinivasan, and Stephan Meylan. 2020. Contextualized word embeddings encode aspects of human-like word sense knowledge.

Mohammad Taher Pilehvar and Jose Camacho-Collados. 2019. WiC: the word-in-context dataset for evaluating context-sensitive meaning representations. In Proceedings of the 2019 Conference of the North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1267–1273, Minneapolis, Minnesota. Association for Computational Linguistics.

Alessandro Raganato, Jose Camacho-Collados, and Roberto Navigli. 2017. Word sense disambiguation: A unified evaluation framework and empirical comparison. In Proceedings of the 15th Conference of the European Chapter ofthe Associationfor Computational Linguistics: Volume 1, Long Papers, pages 99–110, Valencia, Spain. Association for Computational Linguistics.

Rakia Saidi and Fethi Jarray. 2023. Sentence transformers and distilbert for arabic word sense induction.

Bianca Scarlini, Tommaso Pasini, and Roberto Navigli. 2020. Sensembert: Context-enhanced sense embeddings for multilingual word sense disambiguation. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):8758–8765.

Marija Stanojevic. 2009. Cognitive synonymy: A general overview. Facta Universitatis Series: Linguistics and Literature, 07:193–200.

Nina Tahmasebi, Lars Borin, and Adam Jatowt. 2021. Survey of computational approaches to lexical semantic change detection. Computational approaches to semantic change, 6(1).

Dan John Velasco, Axel Alba, Trisha Gail Pelagio, Bryce Anthony Ramirez, Jan Christian Blaise Cruz, Unisse Chua, Briane Paul Samson, and Charibeth Cheng. 2023. Towards automatic construction of Filipino WordNet: Word sense induction and synset induction using sentence embeddings. In Proceedings of the First Workshop in South East Asian Language Processing, pages 1–12, Nusa Dua, Bali, Indonesia. Association for Computational Linguistics.

Jingqing Zhang, Luis Bolanos Trujillo, Tong Li, Ashwani Tanwar, Guilherme Freire, Xian Yang, Julia Ive, Vibhor Gupta, and Yike Guo. 2021. Self-supervised detection of contextual synonyms in a multi-class setting: Phenotype annotation use case. In Proceedings $o f$ the 2021 Conference on Empirical Methods in Natural Language Processing, pages 8754–8769, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

## A Extended BCubed to Evaluate CI and WSI

The extension of BCubed for overlapping clusters rely on two quantities, Multiplicity Precision (MP) and Multiplicity Recall (MR). In the case of Concept Induction, MP and MR between two lemmas are defined as follows:

$$
\begin{array} { r l } & { \mathrm { M P } ( w _ { 1 } , w _ { 2 } ) = } \\ & { \frac { M i n \left( | f ( w _ { 1 } ) \cap f ( w _ { 2 } ) | , | g ( w _ { 1 } ) \cap g ( w _ { 2 } ) | \right) } { | f ( w _ { 1 } ) \cap f ( w _ { 2 } ) | } } \end{array}
$$

$$
\begin{array} { r l } & { \mathrm { M R } ( w _ { 1 } , w _ { 2 } ) = } \\ & { \quad \frac { M i n \left( | f ( w _ { 1 } ) \cap f ( w _ { 2 } ) | , | g ( w _ { 1 } ) \cap g ( w _ { 2 } ) | \right) } { | g ( w _ { 1 } ) \cap g ( w _ { 2 } ) | } } \end{array}
$$

with $w _ { 1 }$ and $w _ { 2 }$ two lemmas, and g a reference clustering function and $f$ the clustering function we want to evaluate. MP (resp. MR) can be computed for every lemma $w _ { 1 }$ with every other lemma $w _ { 2 }$ sharing at least one cluster with $w _ { 1 }$ in $f$ (resp. in $g )$ . We denote $\mathrm { M P } ( w _ { 1 } , \cdot )$ and $\mathrm { M P } ( w _ { 1 } , \cdot )$ the obtained averages. In the case of non-overlapping clusters, this formulation gives the same result as the original (non-extended) BCubed. To evaluate WSI, the formulation is the same but we do not evaluate at the word-level but at the occurrencelevel.

Precision, Recall and F-score are obtained as follows:

$$
\begin{array} { c } { { \mathrm { P r e c i s i o n } = \displaystyle \frac { 1 } { | W | } \sum _ { w \in W } \mathrm { M P } ( w , \cdot ) } } \\ { { \mathrm { R e c a l l } = \displaystyle \frac { 1 } { | W | } \sum _ { w \in W } \mathrm { M R } ( w , \cdot ) } } \\ { { \mathrm { F } _ { \beta } = ( 1 + \beta ^ { 2 } ) \displaystyle \frac { \mathrm { R e c a l l } \times \mathrm { P r e c i s i o n } } { \beta ^ { 2 } \times \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } } . } } \end{array}
$$

By default we fix $\beta = 1$ , as we compare the learned clustering and the reference clustering as equals and therefore do not find that Precision and Recall should be weighted differently.

Amigó et al. (2009) showed that the benefits of BCubed over other clustering scores. For instance, Rand Index does not handle well the case of many small clusters, which is likely to be the case for Concept Induction. We also prefer Extended BCubed over Overlapping Normalized Mutual Information (McDaid et al., 2011) as the latter is matching-based. That is, the repetition (or nonrepetition) of identical clusters will have no impact on the measure. However, we can easily imagine identical clusters of words to be repeated as they may correspond to distinct concepts. In Extended BCubed, repeated clusters are taken in account as we measure the number of times two lemmas are clustered together. The denominator of MP ensures that over-estimating the number of common clusters is also penalized, and those of MR ensures that under-estimating is penalized. M in operators are there to prevent both quantities to grow over 1.

## B Splits and dataset statistics

In Table 5 we display statistics over the different splits we used. Dev is a subset containing a sample of 10% of concepts and their occurrences. Synon. is a subset containing only concepts instantiated with 2 lemmas or more, and their occurrences.

## C Used hyperparameters and layers

## C.1 CLM layers

Prior work like Ethayarajh (2019) showed that later layers usually correlates with deeper levels of contextualization and more semantic information, Chronis and Erk (2020) showed that moderatelylate were preferred for lexical similarity while very last layers were preferred for semantic relatedness. To get embeddings, we try 4 sets of layers corresponding to different depths: first layers (1 to 4), moderately early layers (8 to 11), moderately late (14 to 17), and last layers (21 to 24). To get the representation of a word’s occurrence, we simply average its embeddings from the four chosen layers into one single 1024-dimensional embedding. For Concept Induction, we find that best results were obtained using layers 14 to 17, that are the reported results.

<table><tr><td></td><td>#Occs</td><td>#Lemmas</td><td>#Concepts</td><td>#Occs/Concept</td><td>#Occs/Lemmas</td><td> $d _ { \mathrm { L e x } }$ </td><td> $d _ { \mathrm { P o l y s e m y } }$ </td></tr><tr><td>Full data</td><td>52&#x27;997</td><td> $1 ^ { \cdot } 5 6 0$ </td><td> $3 ^ { \circ } 8 5 5$ </td><td>13.75</td><td>33.97</td><td>1.14</td><td>2.83</td></tr><tr><td>Dev</td><td>4795</td><td>389</td><td>386</td><td>12.42</td><td>12.33</td><td>1.14</td><td>1.13</td></tr><tr><td>Synon</td><td>13&#x27;158</td><td>630</td><td>447</td><td>29.44</td><td>20.89</td><td>2.24</td><td>1.59</td></tr></table>

Table 5: Statistics on the different data splits in annotated SemCor. The split “Synon” only contains occurrences of concepts instantiated with multiple lemmas (cases of synonymy). $d _ { \mathrm { L e x } }$ is the average number of unique lemmas per concept, $d _ { \mathrm { P o l y s e m y } }$ is the average number of distinct concepts per lemma.
<table><tr><td>Systems</td><td>Best hyperparameters</td></tr><tr><td>Local-only Kmeans</td><td> $k = 3$ </td></tr><tr><td>Local-only Agglo</td><td> $\mathrm { l i n k a g e } = \mathrm { a v e r a g e } , \nu = 1 . 0$ </td></tr><tr><td>Global-only Kmeans</td><td> $\pi = 1 2 0 \%$ </td></tr><tr><td>Global-only Agglo</td><td> $\mathrm { l i n k a g e = a v e r a g e } , \nu = 3 . 5$ </td></tr><tr><td>Bi-level Kmeans</td><td> $k = \bar { 8 } , \pi = 1 2 0 \%$ </td></tr><tr><td>Bi-level Agglo</td><td>linkage = average (both),  $\nu _ { l o c a l } = 0 . 0 , \nu _ { g l o b a l } = 4 . 5$ </td></tr><tr><td>Bi-level Kmeans (local Agglo)</td><td> $\mathrm { l i n k a g e } = \mathrm { a v e r a g e } \ \nu _ { l o c a l } = 0 . 0 , \pi = 1 2 0 \%$ </td></tr><tr><td>Bi-level Agglo (local Kmeans)</td><td> $k = 1 0 , \mathrm { l i n k a g e } = \mathrm { a v e r a g e } , \nu _ { g l o b a l } = 4 . 5$ </td></tr></table>

Table 6: Best hyperparameters on the Dev split.

## C.2 Hyperparameters

For Eyal et al. (2022), we tried different resolution, varying it from 1e-3 to 10, for the Louvain clustering but found very little to no effect.

For Kmeans at the local level, we varied the number of clusters k between 2 and 10. For Agglomerative clustering at both levels, we tried single, average and complete linkage.

The distance threshold in Agglo τ was indexed on the distribution of distances. We fixed an hyperparameter ν and derived $\tau = \arg ( d ) - \nu . { \operatorname { s t d } } ( d )$ with d the distribution of distances between clustered instances. We made ν vary between -4 and +8. For global Kmeans, the number of clusters was indexed using a proportion π on the number of lemmas (e.g. 120%  W), π varying from 40% to 400%. This may help transfering hyperaparameters to other dataset in future research.

Best hyperaparameters choices are in Table 6

## D Concept Clusters Size Distribution

The distribution of the concept cluster size (in number of lemmas) obtained with Bi-level Agglo system can be found in Figure 2

![](images/ee1899dd4390a8004dc6823becce7f4ac15879a6ee38acee5c56219a29f1185e.jpg)  
Figure 2: Distribution of cluster size (in number of lemmas) obtained by the Bi-level Agglo system.

## E Scientific Artifacts

We used WordNet and SemCor, both properties of Princeton University. Licence can be found at https://wordnet.princeton.edu/ license-and-commercial-use.