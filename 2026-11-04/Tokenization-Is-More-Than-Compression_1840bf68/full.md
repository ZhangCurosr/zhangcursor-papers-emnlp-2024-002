# Tokenization Is More Than Compression

Craig W. Schmidt<sup>†</sup> Varshini Reddy<sup>†</sup> Haoran Zhang<sup>†,‡</sup> Alec Alameddine<sup>†</sup> Omri Uzan<sup>§</sup> Yuval Pinter<sup>§</sup> Chris Tanner<sup>†,¶</sup>

<sup>†</sup>Kensho Technologies <sup>‡</sup>Harvard Univ <sup>§</sup>Dept of Computer Science <sup>¶</sup>MIT Cambridge, MA Cambridge, MA Ben-Gurion Univ of the Negev Cambridge, MA Beer Sheva, Israel

{craig.schmidt,varshini.reddy,alec.alameddine,chris.tanner}@kensho.com haoran\_zhang@g.harvard.edu {omriuz@post,uvp@cs}.bgu.ac.il

## Abstract

Tokenization is a foundational step in natural language processing (NLP) tasks, bridging raw text and language models. Existing tokenization approaches like Byte-Pair Encoding (BPE) originate from the field of data compression, and it has been suggested that the effectiveness of BPE stems from its ability to condense text into a relatively small number of tokens. We test the hypothesis that fewer tokens lead to better downstream performance by introducing PathPiece, a new tokenizer that segments a document’s text into the minimum number of tokens for a given vocabulary. Through extensive experimentation we find this hypothesis not to be the case, casting doubt on the understanding of the reasons for effective tokenization. To examine which other factors play a role, we evaluate design decisions across all three phases of tokenization: pre-tokenization, vocabulary construction, and segmentation, offering new insights into the design of effective tokenizers. Specifically, we illustrate the importance of pretokenization and the benefits of using BPE to initialize vocabulary construction. We train 64 language models with varying tokenization, ranging in size from 350M to 2.4B parameters, all of which are made publicly available.

## 1 Introduction

Tokenization is an essential step in NLP that translates human-readable text into a sequence of distinct tokens that can be subsequently used by statistical models (Grefenstette, 1999). Recently, a growing number of studies have researched the effects of tokenization, both in an intrinsic manner and as it affects downstream model performance (Singh et al., 2019; Bostrom and Durrett, 2020; Hofmann et al., 2021, 2022; Limisiewicz et al., 2023; Zouhar et al., 2023b). To rigorously inspect the impact of tokenization, we consider tokenization as three distinct, sequential stages:

1. Pre-tokenization: an optional set of initial rules that restricts or enforces the creation of certain tokens (e.g., splitting a corpus on whitespace, thus preventing any tokens from containing whitespace).

2. Vocabulary Construction: the core algorithm that, given a text corpus $\mathcal { C }$ and desired vocabulary size m, constructs a vocabulary of tokens $t _ { k } \in \mathcal V$ , such that $| \nu | = m$ , while adhering to the pre-tokenization rules.

3. Segmentation: given a vocabulary and a document $d ,$ segmentation determines how to split d into a series of $K _ { d }$ tokens $t _ { 1 } , \dots , t _ { k } , \dots , t _ { K _ { d } } ,$ with all $t _ { k } \in \mathcal { V } ,$ such that the concatenation of the tokens strictly equals d. Given a corpus of documents , we will define the corpus token count (CTC) as the total number of tokens used in each segmentation, $\begin{array} { r } { \mathrm { C T C } ( \mathcal { C } ) = \sum _ { d \in \mathcal { C } } K _ { d } . } \end{array}$

As an example, segmentation might decide to split the text intractable into “int ract able”, “in trac table”, “in tractable”, or “int r act able”.

We will refer to this step as segmentation, although in other works it is also called “inference” or even “tokenization”.

The widely used Byte-Pair Encoding (BPE) tokenizer (Sennrich et al., 2016) originated in the field of data compression (Gage, 1994). Gallé (2019) argues that it is effective because it compresses text to a short sequence of tokens. Goldman et al. (2024) varied the number of documents in the tokenizer training data for BPE, and found a correlation between CTC and downstream performance. To investigate the hypothesis that having fewer tokens necessarily leads to better downstream performance, we design a novel tokenizer, PATHPIECE, that, for a given document d and vocabulary , finds a segmentation with the minimum possible

$K _ { d } .$ The PATHPIECE vocabulary construction routine is a top-down procedure that heuristically minimizes CTC on a training corpus. PATHPIECE is ideal for studying the effect of CTC on downstream performance, as we can vary decisions at each tokenization stage.

We extend these experiments to the most commonly used tokenizers, focusing on how downstream task performance is impacted by the major stages of tokenization and vocabulary sizes. Toward this aim, we conducted experiments by training 64 language models (LMs): 54 LMs with 350M parameters; 6 LMs with 1.3B parameters; and 4 LMs with 2.4B parameters. We provide open-source, public access to PATHPIECE<sup>1</sup>, and our trained vocabularies and LMs<sup>2</sup>.

## 2 Preliminaries

Ali et al. (2024) and Goldman et al. (2024) examined the effect of tokenization on downstream performance of LLM tasks, reaching opposite conclusions on the importance of CTC. Zouhar et al. (2023a) also find that low token count alone does not necessarily improve performance. Mielke et al. (2021) give a survey of subword tokenization.

## 2.1 Pre-tokenization Methods

Pre-tokenization is a process of breaking text into chunks, which are then tokenized independently. A token is not allowed to cross these pre-tokenization boundaries. BPE, WordPiece, and Unigram all require new chunks to begin whenever a space is encountered. If a space appears in a chunk, it must be the first character; hence, we will call this “FirstSpace”. Thus “ New” is allowed but “New York” is not. Gow-Smith et al. (2022) examine treating spaces as individual tokens, which we will call “Space” pre-tokenization, while Jacobs and Pinter (2022) suggest marking spaces at the end of tokens, and Gow-Smith et al. (2024) propose dispensing them altogether in some settings. Llama (Touvron et al., 2023) popularized the idea of having each digit always be an individual token, which we call “Digit” pre-tokenization.

## 2.2 Vocabulary Construction

We focus on byte-level, lossless subword tokenization. Subword tokenization algorithms split text into word and subword units based on their frequency and co-occurrence patterns from their “training” data, effectively capturing morphological and semantic nuances in the tokenization process (Mikolov et al., 2011).

We analyze BPE, WordPiece, and Unigram as baseline subword tokenizers, using the implementations from HuggingFace<sup>3</sup> with ByteLevel pretokenization enabled. We additionally study SaGe, a context-sensitive subword tokenizer, using version 2.0.<sup>4</sup>

Byte-Pair Encoding Sennrich et al. (2016) introduced Byte-Pair Encoding (BPE), a bottom-up method where the vocabulary construction starts with single bytes as tokens. It then merges the most commonly occurring pair of adjacent tokens in a training corpus into a single new token in the vocabulary. This process repeats until the desired vocabulary size is reached. Issues with BPE and analyses of its properties are discussed in Bostrom and Durrett (2020); Klein and Tsarfaty (2020); Gutierrez-Vasques et al. (2021); Yehezkel and Pinter (2023); Saleva and Lignos (2023); Liang et al. (2023); Lian et al. (2024); Chizhov et al. (2024); Bauwens and Delobelle (2024). Zouhar et al. (2023b) build an “exact” algorithm which optimizes compression for BPE-constructed vocabularies.

WordPiece WordPiece is similar to BPE, except that it uses Pointwise Mutual Information (PMI) (Bouma, 2009) as the criteria to identify candidates to merge, rather than a count (Wu et al., 2016; Schuster and Nakajima, 2012). PMI prioritizes merging pairs that occur together more frequently than expected, relative to the individual token frequencies.

Unigram Language Model Unigram works in a top-down manner, starting from a large initial vocabulary and progressively pruning groups of tokens that induce the minimum likelihood decrease of the corpus (Kudo, 2018). This selects tokens to maximize the likelihood of the corpus, according to a simple unigram language model.

SaGe Yehezkel and Pinter (2023) proposed SaGe, a subword tokenization algorithm incorporating contextual information into an ablation loss via a skipgram objective. SaGe also operates top-down, pruning from an initial vocabulary to a desired size.

## 2.3 Segmentation Methods

Given a tokenizer and a vocabulary of tokens, segmentation converts text into a series of tokens. We included all 256 single-byte tokens in the vocabulary of all our experiments, ensuring any text can be segmented without out-of-vocabulary issues.

Certain segmentation methods are tightly coupled to the vocabulary construction step, such as merge rules for BPE or the maximum likelihood approach for Unigram. Others, such as the WordPiece approach of greedily taking the longest prefix token in the vocabulary at each point, can be applied to any vocabulary; indeed, there is no guarantee that a vocabulary will perform best downstream with the segmentation method used to train it (Uzan et al., 2024). Additional segmentation schemes include Dynamic Programming BPE (He et al., 2020), BPE-Dropout (Provilkov et al., 2020), and FLOTA (Hofmann et al., 2022).

## 3 PATHPIECE

Several efforts over the last few years (Gallé, 2019; Zouhar et al., 2023a, inter alia) have suggested that the empirical advantage of BPE as a tokenizer in many NLP applications, despite its unawareness of language structure, can be traced to its superior compression abilities, providing models with overall shorter sequences during learning and inference. Inspired by this claim we introduce PATHPIECE, a lossless subword tokenizer that, given a vocabulary and document d, produces a segmentation minimizing the total number of tokens needed to split d. We additionally provide a vocabulary construction procedure that, using this segmentation, attempts to find a  minimizing the corpus token count (CTC).<sup>5</sup> PATHPIECE provides an ideal testing laboratory for the compression hypothesis by virtue of its maximally efficient segmentation.

## 3.1 Segmentation

PATHPIECE requires that all single-byte tokens are included in vocabulary to run correctly. PATH-PIECE works by finding a shortest path through a directed acyclic graph (DAG), where each byte i of training data forms a node in the graph, and two nodes j and i contain a directed edge if the byte segment $[ j , i ]$ is a token in . We describe PATHPIECE segmentation in Algorithm 1, where L is a limit on the maximum width of a token in bytes, which we set to 16. It has a complexity of

$O ( n L )$ , which follows directly from the two nested for-loops. For each byte i in $d ,$ it computes the shortest path length $p l [ i ]$ in tokens up to and including byte $i ,$ and the width wid[i] of a token with that shortest path length. In choosing $w i d [ i ]$ , ties between multiple tokens with the same shortest path length $p l [ i ]$ can be broken randomly, or the one with the longest wid[i] can be chosen, as shown here.<sup>6</sup> Then, a backward pass constructs the shortest possible segmentation from the wid[i] values computed in the forward pass.

Algorithm 1 PATHPIECE segmentation.   
1: procedure PATHPIECE(d, , L)   
2: $n  \mathrm { l e n } ( d )$ ▷ document length   
3: $p l [ 1 \mathrel { : } n ] \stackrel { \cdot } {  } \infty$ ▷ shortest path length   
4: wid $[ 1 : n ] \gets 0$ ▷ shortest path tok width   
5: $\mathbf { f o r } e \gets \bar { 1 } , n$ do ▷ token end   
6: for $w  1 , L$ do ▷ token width   
7: $s \gets e - w + 1$ ▷ token start   
8: i $\mathbf { f } \ s \geq 1$ then ▷ s in range   
9: $\mathbf { i f } d [ s : e ] \in \mathcal { V }$ then   
10: $\mathbf { \dot { i } } \mathbf { f } \ s = 1$ then ▷ 1 tok path   
11: $p l [ e ] \gets 1$   
12: $w i d [ e ]  w$   
13: else   
14: $n l \gets p l [ s - 1 ] + 1$   
15: $\mathbf { i f } n l \le \dot { p l } [ e ]$ then   
16: $p l [ e ] \gets n l$   
17: $w i d [ e ]  w$   
18: $T \gets [ ]$ ▷ output token list   
19: $e \gets n$ ▷ start at end of d   
20: while $\gtrsim \geq 1$ do   
21: $s \gets e - w i d [ e ] + 1$ ▷ token start   
22: $T . \mathrm { a p p e n d } ( d [ s : e ] )$ ▷ append token   
23: $e \gets e - w i d [ e ]$ ▷ back up a token   
24: return reversed(T) ▷ reverse order

## 3.2 Vocabulary Construction

PATHPIECE’s vocabulary is built in a top-down manner, attempting to minimize the corpus token count (CTC), by starting from a large initial vocabulary $\mathcal { V } _ { 0 }$ and iteratively omitting batches of tokens. The $\mathcal { V } _ { 0 }$ may be initialized from the most frequently occurring byte n-grams in the corpus, or from a large vocabulary trained by BPE or Unigram. We enforce that all single-byte tokens remain in the vocabulary and that all tokens are L bytes or shorter.

For a PATHPIECE segmentation $t _ { 1 } , \ldots , t _ { K _ { d } }$ of a document d in the training corpus ${ \mathcal { C } } ,$ we would like to know the increase in the overall length of the segmentation $K _ { d }$ after omitting each token t from our vocabulary and then recomputing the segmentation. Tokens with a low overall increase are good candidates to remove from the vocabulary.

To avoid the very expensive $O ( n L | \mathcal { V } | )$ computation of each segmentation from scratch, we make a simplifying assumption that allows us to compute these increases more efficiently: we omit a specific token $t _ { k }$ , for $k \in [ 1 , \ldots , K _ { d } ]$ in the segmentation of a particular document d, and compute the minimum increase $M I _ { k d } \ge 0$ in the total tokens $K _ { d }$ from not having that token $t _ { k }$ in the segmentation of d. We then aggregate these token count increases $M I _ { k d }$ for each token $t \in \nu$ . We can compute the $M I _ { k d }$ without actually re-segmenting any documents, by reusing the shortest path information computed by Algorithm 1 during segmentation.

Any segmentation not containing $t _ { k }$ must either contain a token boundary somewhere inside of $t _ { k }$ breaking it in two, or it must contain a token that entirely contains $t _ { k }$ as a superset. We enumerate all occurrences for these two cases, and we find the minimum increase M $I _ { k d }$ among them. Let $t _ { k }$ start at index s and end at index e, inclusive. Path length $p l [ j ]$ represents the number of tokens required for the shortest path up to and including byte $j .$ . We also run Algorithm 1 backwards on $d ,$ computing a similar vector of backwards path lengths $b p l [ j ]$ representing the number of tokens on a path from the end of the data up to and including byte $j .$ . The minimum length of a segmentation with a token boundary after byte j is thus:

$$
K _ { j } ^ { b } = p l [ j ] + b p l [ j + 1 ] .\tag{1}
$$

We have added an extra constraint on the shortest path, that there is a break at $j ,$ , so clearly $K _ { j } ^ { b } \geq K _ { d }$ The minimum increase for the case of having a token boundary within $t _ { k }$ is thus:

$$
M I _ { k d } ^ { b } = \operatorname* { m i n } _ { j = s , . . . , e - 1 } K _ { j } ^ { b } - K _ { d } .\tag{2}
$$

The minimum increase from omitting $t _ { k }$ could also be from a segmentation containing a strict superset of $t _ { k }$ . Let this superset token be $t _ { k } ^ { \prime }$ , with start $s ^ { \prime }$ and end $e ^ { \prime }$ inclusive. To be a strict superset entirely containing $t _ { k }$ , then either $s ^ { \prime } < s$ and $e ^ { \prime } \geq$ $e ,$ or $s ^ { \prime } \leq s$ and $e ^ { \prime } > e ,$ , subject to the constraint that the width $w ^ { \prime } = e ^ { \prime } - s ^ { \prime } + 1 \leq L$ . In this case, the minimum length when using the superset token $t _ { k } ^ { \prime }$ would be:

$$
K _ { t _ { k } ^ { \prime } } ^ { s } = p l [ s ^ { \prime } - 1 ] + b p l [ e ^ { \prime } + 1 ] + 1 ,\tag{3}
$$

which is the path length to get to the byte before $t _ { k } ^ { \prime } { : }$ , plus the path length from the end of the data backwards to the byte after $t _ { k } ^ { \prime } .$ , plus 1 for the token $t _ { k } ^ { \prime }$ itself.

We retain a list of the widths of the tokens ending at each byte.<sup>7</sup> The set of superset tokens $S$ can be found by examining the potential $e ^ { \prime } { \mathrm { , } }$ , and then seeing if the tokens ending at $e ^ { \prime }$ form a strict superset. Similar to the previous case, we can compute the minimum increase from replacing $t _ { k }$ with a superset token by taking the minimum increase over the superset tokens $S { \mathrm { : } }$

$$
M I _ { k d } ^ { s } = \operatorname* { m i n } _ { t _ { k } ^ { \prime } \in S } K _ { t _ { k } ^ { \prime } } ^ { s } - K _ { d } .\tag{4}
$$

We then aggregate over the documents to get the overall increase for each $t \in \nu$

$$
M I _ { t } = \sum _ { d \in \mathcal { C } } \sum _ { k = 1 | t _ { k } = t } ^ { K _ { d } } \operatorname* { m i n } ( M I _ { k d } ^ { b } , M I _ { k d } ^ { s } ) .\tag{5}
$$

One iteration of this vocabulary construction procedure will have complexity $O ( n L ^ { 2 } )$ 7

## 3.3 Connecting PATHPIECE and Unigram

We note a connection between PATHPIECE and Unigram. In Unigram, the probability of a segmentation $t _ { 1 } , \ldots , t _ { K _ { d } }$ is the product of the unigram token probabilities $p ( t _ { k } )$

$$
P ( t _ { 1 } , \dots , t _ { K _ { d } } ) = \prod _ { k = 1 } ^ { K _ { d } } p ( t _ { k } ) .\tag{6}
$$

Taking the negative log of this product converts the objective from maximizing the likelihood to minimizing the sum of $- \log ( p ( t _ { k } ) )$ terms. While Unigram is solved by the Viterbi (1967) algorithm, it can also be solved by a weighted version of PATH-PIECE with weights $\mathbf { o f } - \log ( p ( t _ { k } ) )$ . Conversely, a solution minimizing the number of tokens can be found in Unigram by taking all $p ( t _ { k } ) : = 1 / | \nu |$

## 4 Experiments

We used the Pile corpus (Gao et al., 2020; Biderman et al., 2022) for language model pre-training, which contains 825GB of English text data from 22 high-quality datasets. We constructed the tokenizer vocabularies over the MiniPile dataset (Kaddour, 2023), a 6GB subset of the Pile. We use the MosaicML Pretrained Transformers (MPT) decoderonly language model architecture.<sup>8</sup> Appendix B gives the full set of model parameters, and Appendix D discusses model convergence.

## 4.1 Downstream Evaluation Tasks

To evaluate and analyze the performance of our tokenization process, we select 10 benchmarks from lm-evaluation-harness (Gao et al., 2023).<sup>9</sup> These are all multiple-choice tasks with 2, 4, or 5 options, and were run with 5-shot prompting. We use arc\_easy (Clark et al., 2018), copa (Brassard et al., 2022), hendrycksTests-marketing (Hendrycks et al., 2021), hendrycksTests-sociology (Hendrycks et al., 2021), mathqa (Amini et al., 2019), piqa (Bisk et al., 2019), qa4mre\_2013 (Peñas et al., 2013), race (Lai et al., 2017), sciq (Welbl et al., 2017), and wsc273 (Levesque et al., 2012). Appendix C gives a full description of these tasks.

## 4.2 Tokenization Stage Variants

We conduct the 18 experimental variants listed in Table 1, each repeated at the vocabulary sizes of 32,768, 40,960, and 49,152.<sup>10</sup> For baseline vocabulary creation methods, we used BPE, Unigram, WordPiece, and SaGe. We also consider two variants of PATHPIECE where ties in the shortest path are broken either by the longest token (PATHPIECEL), or randomly (PATHPIECER). For the vocabulary initialization required by PATH-PIECE and SaGe, we experimented with the most common n-grams, as well as with a large initial vocabulary trained with BPE or Unigram. We also varied the pre-tokenization schemes for PATH-PIECE and SaGe, using either no pre-tokenization or combinations of “FirstSpace”, “Space”, and “Digit” described in §2.1. Tokenizers usually use the same segmentation approach used in vocabulary construction. PATHPIECEL’s shortest path segmentation can be used with any vocabulary, so we apply it to vocabularies trained by BPE and Unigram. We also apply a Greedy left-to-right longest-token segmentation approach to these vocabularies.

## 5 Results

Table 1 reports the downstream performance across all our experimental settings.<sup>11</sup> A random baseline for these 10 tasks yields 32%. The OVERALL AVG column indicates the average results over the three vocabulary sizes. The RANK column refers to the rank of each variant with respect to the OVERALL AVG column (Rank 1 is best), which we will sometimes use as a succinct way to refer to a variant.

## 5.1 Vocabulary Size

![](images/dc20de568761803b51faae63617c8939031d5c00f20f7c89ae4f83d21f11076a.jpg)  
Figure 1: Effect of vocabulary size on downstream performance. For each tokenizer variant, we show the overall average, along with the three averages by vocabulary size, labeled according to the ranks in Table 1.

Figure 1 gives the overall average, along with the individual averages, for each of the three vocabulary sizes for each variant, labeled according to the rank from Table 1. We observe that there is a high correlation between downstream performance at different vocabulary sizes. The pairwise $R ^ { 2 }$ values for the accuracy of the 32,768 and 40,960 runs was 0.750; between 40,960 and 49,152 it was 0.801; and between 32,768 and 49,152 it was 0.834. This corroborates the effect shown graphically in Figure 1 that vocabulary size is not a crucial decision over this range of sizes. Given this high degree of correlation, we focus our analysis on the overall average accuracy. This averaging removes some of the variance amongst individual language model runs. Thus, unless specified otherwise, our analyses present performance averaged over vocabulary sizes.

<table><tr><td>Rank</td><td>Vocab Constr</td><td>Init Voc</td><td>Pre-tok</td><td>Segment</td><td>Overall</td><td>32,768</td><td>40,960</td><td>49,152</td></tr><tr><td>1</td><td rowspan="4">PathPieceL</td><td>BPE</td><td>FirstSpace</td><td rowspan="4">PathPieceL</td><td>49.4</td><td>49.3</td><td>49.4</td><td>49.4</td></tr><tr><td>9</td><td>Unigram</td><td>FirstSpace</td><td>48.0</td><td>47.0</td><td>48.5</td><td>48.4</td></tr><tr><td>15</td><td>n-gram</td><td>FirstSpDigit FirstSpace</td><td>44.8</td><td>44.6</td><td>44.9</td><td>45.0</td></tr><tr><td>16</td><td>n-gram</td><td></td><td>44.7</td><td>44.8 45.5</td><td>43.9</td></tr><tr><td>2</td><td rowspan="2">Unigram</td><td rowspan="2"></td><td rowspan="2">FirstSpace</td><td>Likelihood</td><td>49.0</td><td>49.2</td><td>49.1</td><td>48.8</td></tr><tr><td>7</td><td>Greedy</td><td>48.3</td><td>47.9</td><td>48.5</td><td>48.6</td></tr><tr><td>17</td><td rowspan="2"></td><td></td><td rowspan="2"></td><td>PathPieceL</td><td>43.6</td><td>43.6</td><td>43.1</td><td>44.0</td></tr><tr><td>3</td><td></td><td>Merge</td><td>49.0</td><td>49.0</td><td>50.0</td><td>48.1</td></tr><tr><td>4</td><td rowspan="2">BPE</td><td></td><td rowspan="2">FirstSpace</td><td>Greedy</td><td>49.0</td><td>48.3</td><td>49.1</td><td>49.5</td></tr><tr><td>13</td><td></td><td>PathPieceL</td><td>46.5</td><td>45.6</td><td>46.7</td><td>47.2</td></tr><tr><td>5</td><td colspan="2">WordPiece</td><td>FirstSpace</td><td>Greedy</td><td>48.8</td><td>48.5</td><td>49.1</td><td>48.8</td></tr><tr><td>6</td><td rowspan="4">SaGe</td><td>BPE</td><td>FirstSpace</td><td rowspan="4">Greedy</td><td>48.6</td><td>48.0</td><td>49.2</td><td>48.8</td></tr><tr><td>8</td><td>n-gram</td><td>FirstSpace</td><td>48.0</td><td>47.5</td><td>48.5</td><td>48.0</td></tr><tr><td>10</td><td>Unigram</td><td>FirstSpace</td><td>47.7</td><td>48.4</td><td>46.9</td><td>47.8</td></tr><tr><td>11</td><td>n-gram</td><td>FirstSpDigit</td><td>47.5</td><td>48.4</td><td></td><td>47.2</td></tr><tr><td></td><td rowspan="3">PathPieceR</td><td rowspan="3">n-gram</td><td>SpaceDigit</td><td rowspan="3">PathPieceR</td><td></td><td></td><td>46.9</td><td></td></tr><tr><td>12 14</td><td>FirstSpDigit None</td><td>46.7 45.5</td><td>47.5 45.3</td><td>45.4 45.8</td><td>47.3</td></tr><tr><td></td><td></td><td>43.2</td><td>43.5</td><td>44.0</td><td>45.5 42.2</td></tr><tr><td colspan="6">Random</td><td>32.0</td><td>32.0</td><td>32.0</td><td>32.0</td></tr></table>

Table 1: Summary of 350M parameter model downstream accuracy (%) across 10 tasks. The “Overall” column averages across the three vocabulary sizes. The “Rank” column refers to the Overall column, best to worst.

## 5.2 Overall performance

To determine which of the differences in the overall average accuracy in Table 1 are statistically significant, we conduct a one-sided Wilcoxon signed-rank test (Wilcoxon, 1945) on the paired differences of the 30 accuracy scores (three vocabulary sizes over ten tasks), for each pair of variants. All p-values reported in this paper use this test.

![](images/cbf208fd04f3ec71725e8ff398c181a745b518cbcbee9f95bb0b9a495f04b138.jpg)  
Figure 2: Pairwise p-values for 350M model results. Boxes outlined in black represent $p > 0 . 0 5$ . The top 6 tokenizers are all competitive, and there is no statistically significantly best approach.

Figure 2 displays all pairwise p-values in a color map. Each column designates a tokenization variant by its rank in Table 1, compared to all the ranks below it. A box is outlined in black if $p > 0 . 0 5$ where we cannot reject the null. While PATHPIE-CEL-BPE had the highest overall average on these tasks, the top five tokenizers, PATHPIECEL-BPE, Unigram, BPE, BPE-Greedy, and WordPiece do not have any other row in Figure 2 significantly different from them. Additionally, SaGe-BPE (rank 6) is only barely worse than PATHPIECEL-BPE $( p = 0 . 0 4 7 )$ , and should probably be included in the list of competitive tokenizers. Thus, our first key result is that there is no tokenizer algorithm better than all others to a statistically significant degree.

All the results reported thus far are for language models with identical architectures and 350M parameters. To examine the dependency on model size, we trained larger models of 1.3B parameters for six of our experiments, and 2.4B parameters for four of them. In the interest of computational time, these larger models were only trained with a single vocabulary size of 40,960. In Figure 6 in subsection 6.4, we report models’ average performance across 10 tasks. See Figure 7 in Appendix D for an example checkpoint graph at each model size. The main result from these models is that the relative performance of the tokenizers does vary by model size, and that there is a group of high performing tokenizers that yield comparable results. This aligns with our finding that the top six tokenizers are not statistically better than one another at the 350M model size.

![](images/b078818d726e5d84c379e639f70eacfe007eeb1d4e506e4fa28ade59483d0785.jpg)  
Figure 3: Effect of corpus token count (CTC) vs average accuracy of individual vocabulary sizes.

## 5.3 Corpus Token Count vs Accuracy

Figure 3 shows the corpus token count (CTC) versus the accuracy of each vocabulary size, given in Table 11. We do not find a straightforward relationship between the two. Ali et al. (2024) recently examined the relationship between CTC and downstream performance for three different tokenizers, and also found it was not correlated on English language tasks.

The two models with the highest CTC are PATH-PIECE with Space pre-tokenization (12), which is to be expected given each space is its own token, and SaGe with an initial Unigram vocabulary (10). The Huggingface Unigram models in Figure 3 had significantly higher CTC than the corresponding BPE models, unlike Bostrom and Durrett (2020) and Gow-Smith et al. (2022), which report a difference of only a few percent with SentencePiece Unigram. Ali et al. (2024) point out that due to differences in pre-processing, the Huggingface Unigram tokenizer behaves quite differently than the SentencePiece Unigram tokenizer, which may explain this discrepancy.

In terms of accuracy, PATHPIECE with no pretokenization (18) and Unigram with PATHPIECE segmentation (17) both did quite poorly. Notably, the range of CTC is quite narrow within each vocabulary construction method, even while changes in pre-tokenization and segmentation lead to significant accuracy differences. While there are confounding factors present in this chart (e.g., pretokenization, vocabulary initialization, and that more tokens allow for additional computations by the downstream model) it is difficult to discern any trend that lower CTC leads to improved performance. If anything, there seems to be an inverted U-shaped curve with respect to the CTC and downstream performance. The Pearson correlation coefficient between CTC and average accuracy was found to be 0.241. Given that a lower CTC value signifies greater compression, this result suggests a weak negative relationship between the amount of compression and average accuracy.

<table><tr><td colspan="2">Comparison Pearson Correlation</td></tr><tr><td>CTC and Ave Acc</td><td>0.241</td></tr><tr><td>Rényi Eff and Ave Acc (α=1.5) Rényi Eff and Ave Acc (α=2.0) Rényi Eff and Ave Acc (α=2.5) Rényi Eff and Ave Acc (α=3.0)</td><td>-0.221 -0.169 -0.151 -0.144</td></tr><tr><td>Rényi Eff and Ave Acc (α=3.5)</td><td>-0.141</td></tr><tr><td>CTC and Rényi Eff (α=2.5)</td><td>-0.891</td></tr></table>

Table 2: Pearson Correlation of CTC and Average Accuracy, or Rényi efficiency for various orders α with Average Accuracy, or CTC and Rényi efficiency at α = 2.5.

Zouhar et al. (2023a) introduced an informationtheoretic measure based on Rényi efficiency that correlates with downstream performance for their application.<sup>12</sup> It has an order parameter α, with a recommended value of 2.5. We present the Rényi efficiencies and CTC for all models in Table 11 in Appendix G, and summarize their Pearson correlation with average accuracy in Table 2. For the data of Figure 3, all the correlations for various α also have a weak negative association. They are slightly less negative than the association for CTC, although it is not nearly as large as the benefit they saw over sequence length in their application. We note the strong relationship between compression and Rényi efficiency, as the Pearson correlation of CTC and Rényi efficiency with α=2.5 is 0.891.

By varying aspects of BPE, Gallé (2019) and Goldman et al. (2024) suggests we should expect downstream performance to decrease with CTC, while in contrast Ali et al. (2024) did not find a strong relation when varying the tokenizer. Our extensive results varying a number of stages of tokenization suggest it is not inherently beneficial to use fewer tokens. Rather, the particular way that the CTC is varied can lead to different conclusions.

## 6 Analysis

We now analyze the results across the various experiments in a more controlled manner. Our experiments allow us to examine changes in each stage of tokenization, holding the rest constant, revealing design decisions making a significant difference.<sup>13</sup>

## 6.1 Pre-tokenization

For PATHPIECER with an n-gram initial vocabulary, we can isolate pre-tokenization. PATHPIECE is efficient enough to process entire documents with no pre-tokenization, giving it full freedom to minimize the corpus token count (CTC).

Adding pre-tokenization constrains PATH-PIECE’s ability to minimize tokens, giving a natural way to vary the number of tokens. Figure 4 shows that PATHPIECE minimizes the number of tokens used over a corpus when trained with no pre-tokenization (18). The other variants restrict spaces to either be the first character of a token (14), or their own token (12).<sup>14</sup> Consider the example PATHPIECE tokenization in Table 3 for the three pre-tokenization methods. The NONE mode uses the word-boundary-spanning tokens “ation is”, $\mathbf { \bar { \Sigma } } ^ {  } \mathbf { \Sigma } _ { \cup } \mathbf { \bar { \Sigma } } \cup \mathbf { \bar { \Sigma } }$ , and $^ { 6 6 } \mathrm { e } _ { \perp } \hat { \varsigma } ^ { \prime \prime }$ . The lack of morphological alignment demonstrated in this example is likely more important to downstream model performance than a simple token count.

In Figure 4 we observe a statistically significant increase in overall accuracy for our downstream tasks, as a function of CTC. Gow-Smith et al. (2022) found that Space pre-tokenization lead to worse performance, while removing the spaces entirely helps<sup>15</sup>. Thus, this particular result may be specific to PATHPIECER.

## 6.2 Vocabulary Construction

One way to examine the effects of vocabulary construction is to compare the resulting vocabularies of top-down methods trained using an initial vocabulary to the method itself. Figure 5 presents an area-proportional Venn diagram of the overlap in 40,960-sized vocabularies between BPE (6) and variants of PATHPIECEL (1) and SaGe (6) that were trained using an initial BPE vocabulary of size $2 ^ { 1 8 } = 2 6 2$ , 144.<sup>16</sup> While BPE and PATHPIE-CEL overlap considerably, SaGe produces a more distinct set of tokens.

![](images/59aff11ca8ba402976df63655ec8575ebe85e5299a2f24ed7f6e7f0e276befbd.jpg)  
Figure 4: The impact of pre-tokenization on Corpus Token Count (CTC) and Overall Accuracy. Ranks in parentheses refer to performance in Table 1.

![](images/b81642006f23e9b2d21c3f8e3a7b346c67cbb463f755344a8816305bb71d72f7.jpg)  
Figure 5: Venn diagram comparing 40,960 token vocabularies of BPE, PathPieceL and SaGe – the latter two were both initialized from a BPE vocabulary of 262,144.

## 6.3 Initial Vocabulary

PATHPIECE, SaGe, and Unigram all require an initial vocabulary.<sup>17</sup> For PATHPIECE and SaGe, we experimented with initial vocabularies of size 262,144 constructed from either the most frequent n-grams, or trained using either BPE or Unigram. For PATHPIECEL, using a BPE initial vocabulary (1) is statistically better than both Unigram (9) and n-grams (16), with $p \leq 0 . 0 1$ . Using an n-gram initial vocabulary leads to the lowest performance, with statistical significance. Comparing ranks 6, 8, and 10 reveals the same pattern for SaGe, although the difference between 8 and 10 is not significant.

<table><tr><td>Rank</td><td>Pre-tokenization</td><td>Example</td></tr><tr><td>12</td><td>SpaceDigit</td><td>The valuation is estimated</td></tr><tr><td>14</td><td>FirstSpDigit</td><td>Thevaluationisestimated</td></tr><tr><td>18 None</td><td></td><td> $\mathsf { \Gamma } _ { \mathsf { U } } \mathsf { a } \mathrm { I } \mathsf { u } \ a t \mathrm { i } \mathsf { o } \mathsf { n } _ { \mathsf { U } } \mathsf { i } \mathsf { s }$  estimated</td></tr></table>

Table 3: Example PATHPIECE tokenizations of “The valuation is estimated to be \$213M”; vocabulary size of 32,768.

## 6.4 Effect of Model Size

To examine the dependency on model size, we build larger models of 1.3B parameters for 6 of our experiments, and 2.4B parameters for 4 of them. These models were trained over the same 200 billion tokens. In the interest of computational time, these larger models were only run at a single vocabulary size of 40,960. The average results over the 10 task accuracies for these models is given in Figure 6. See Table 14 in Appendix G for the numerical values.

![](images/e4910cd41a4edcc46a3659ffb13dbe4c54601b9a6db394fc0f53bab82142e7bd.jpg)  
Figure 6: 40,960 vocab average accuracy at various models sizes

It is noteworthy from the prevalence of crossing lines in Figure 6 that the relative performance of the tokenizers do vary by model size, and that there is a group of tokenizers that are trading places being at the top for various model sizes. This aligns with our observation that the top 6 tokenizers were within the noise, and not significantly better than each other in the 350M models.

## 7 Conclusion

We investigate the hypothesis that reducing the corpus token count (CTC) would improve downstream performance, as suggested by Gallé (2019) and Goldman et al. (2024) when they varied aspects of BPE. When comparing CTC and downstream accuracy across all our experimental settings in Figure 3, we do not find a clear relationship between the two. We expand on the findings of Ali et al. (2024) who did not find a strong relation when comparing 3 tokenizers, as we run 18 experiments varying the tokenizer, initial vocabulary, pre-tokenizer, and inference method. Our results suggest compression is not a straightforward explanation of what makes a tokenizer effective.

Finally, this work makes several practical contributions: (1) vocabulary size has little impact on downstream performance over the range of sizes we examined (§5.1); (2) five different tokenizers all perform comparably, with none outperforming at statistical significance (§5.2); (3) BPE initial vocabularies work best for top-down vocabulary construction (§6.3). To further encourage research in this direction, we make all of our trained vocabularies publicly available, along with the model weights from our 64 language models.

## Limitations

The objective of this work is to offer a comprehensive analysis of the tokenization process. However, our findings were constrained to particular tasks and models. Given the degrees of freedom, such as choice of downstream tasks, model, vocabulary size, etc., there is a potential risk of inadvertently considering our results as universally applicable to all NLP tasks; results may not generalize to other domains of tasks.

Additionally, our experiments were exclusively with English language text, and it is not clear how these results will extend to other languages. In particular, our finding that pre-tokenization is crucial for effective downstream accuracy is not applicable to languages without space-delimited words.

We conducted experiments for three district vocabulary sizes, and we reported averaged results across these experiments. With additional compute resources and time, it could be beneficial to conduct further experiments to gain a better estimate of any potential noise. For example, in Figure 7 of Appendix D, the 100k checkpoint at the 1.3B model size is worse than expected, indicating that noise could be an issue.

Finally, the selection of downstream tasks can have a strong impact on results. To allow for meaningful results, we attempted to select tasks that were neither too difficult nor too easy for the 350M parameter models, but other choices could lead to different outcomes. There does not seem to be a good, objective criteria for selecting a finite set of task to well-represent global performance.

## Ethics Statement

We have used the commonly used public dataset The Pile, which has not undergone a formal ethics review (Biderman et al., 2022). Our models may include biases from the training data.

Our experimentation has used considerable energy. Each 350M parameter run took approximately 48 hours on (4) p4de nodes, each containing 8 NVIDIA A100 GPUs. We trained 62 models, including the 8 RandTrain runs in Appendix F. The (6) 1.3B parameters models took approximately 69 hours to train on (4) p4de nodes, while the (4) 2.4B models took approximately 117 hours to train on (8) p4de nodes. In total, training required 17,304 hours of p4de usage (138,432 GPU hours).

## Acknowledgments

Thanks to Charles Lovering at Kensho for his insightful suggestions, and to Michael Krumdick, Mike Arov, and Brian Chen at Kensho for their help with the language model development process. This research was supported in part by the Israel Science Foundation (grant No. 1166/23). Thanks to an anonymous reviewer who pointed out the large change in CTC when comparing Huggingface BPE and Unigram, in contrast to the previous literature using the SentencePiece implementations (Kudo and Richardson, 2018).

## References

Mehdi Ali, Michael Fromm, Klaudia Thellmann, Richard Rutmann, Max Lübbering, Johannes Leveling, Katrin Klug, Jan Ebert, Niclas Doll, Jasper Schulze Buschhoff, Charvi Jain, Alexander Arno Weber, Lena Jurkschat, Hammam Abdelwahab, Chelsea John, Pedro Ortiz Suarez, Malte Ostendorff, Samuel Weinbach, Rafet Sifa, Stefan

Kesselheim, and Nicolas Flores-Herr. 2024. Tokenizer choice for llm training: Negligible or crucial?

Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. 2019. Mathqa: Towards interpretable math word problem solving with operation-based formalisms.

Thomas Bauwens and Pieter Delobelle. 2024. BPEknockout: Pruning pre-existing BPE tokenisers with backwards-compatible morphological semisupervision. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5810–5832, Mexico City, Mexico. Association for Computational Linguistics.

Stella Biderman, Kieran Bicheno, and Leo Gao. 2022. Datasheet for the pile. CoRR, abs/2201.07311.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2019. Piqa: Reasoning about physical commonsense in natural language.

Kaj Bostrom and Greg Durrett. 2020. Byte pair encoding is suboptimal for language model pretraining. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 4617–4624, Online. Association for Computational Linguistics.

Gerlof Bouma. 2009. Normalized (pointwise) mutual information in collocation extraction. Proceedings ofGSCL, 30:31–40.

Ana Brassard, Benjamin Heinzerling, Pride Kavumba, and Kentaro Inui. 2022. Copa-sse: Semi-structured explanations for commonsense reasoning.

Pavel Chizhov, Catherine Arnett, Elizaveta Korotkova, and Ivan P. Yamshchikov. 2024. Bpe gets picky: Efficient vocabulary refinement during tokenizer training.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv, abs/1803.05457.

Marco Cognetta, Vilém Zouhar, Sangwhan Moon, and Naoaki Okazaki. 2024. Two counterexamples to tokenization and the noiseless channel. In Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 16897–16906, Torino, Italia. ELRA and ICCL.

Pavlos S. Efraimidis. 2010. Weighted random sampling over data streams. CoRR, abs/1012.0256.

Philip Gage. 1994. A new algorithm for data compression. C Users J., 12(2):23–38.

Matthias Gallé. 2019. Investigating the effectiveness of BPE: The power of shorter sequences. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1375–1381, Hong Kong, China. Association for Computational Linguistics.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. 2020. The pile: An 800gb dataset of diverse text for language modeling.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2023. A framework for few-shot language model evaluation.

Omer Goldman, Avi Caciularu, Matan Eyal, Kris Cao, Idan Szpektor, and Reut Tsarfaty. 2024. Unpacking tokenization: Evaluating text compression and its correlation with model performance.

Edward Gow-Smith, Dylan Phelps, Harish Tayyar Madabushi, Carolina Scarton, and Aline Villavicencio. 2024. Word boundary information isn’t useful for encoder language models. In Proceedings of the 9th Workshop on Representation Learningfor NLP (RepL4NLP-2024), pages 118–135, Bangkok, Thailand. Association for Computational Linguistics.

Edward Gow-Smith, Harish Tayyar Madabushi, Carolina Scarton, and Aline Villavicencio. 2022. Improving tokenisation by alternative treatment of spaces. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11430–11443, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Gregory Grefenstette. 1999. Tokenization, pages 117– 133. Springer Netherlands, Dordrecht.

Ximena Gutierrez-Vasques, Christian Bentz, Olga Sozinova, and Tanja Samardzic. 2021. From characters to words: the turning point of BPE merges. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 3454–3468, Online. Association for Computational Linguistics.

Xuanli He, Gholamreza Haffari, and Mohammad Norouzi. 2020. Dynamic programming encoding for subword segmentation in neural machine translation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3042–3051, Online. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding.

Valentin Hofmann, Janet Pierrehumbert, and Hinrich Schütze. 2021. Superbizarre is not superb: Derivational morphology improves BERT’s interpretation of complex words. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3594–3608, Online. Association for Computational Linguistics.

Valentin Hofmann, Hinrich Schuetze, and Janet Pierrehumbert. 2022. An embarrassingly simple method to mitigate undesirable properties of pretrained language model tokenizers. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 385–393, Dublin, Ireland. Association for Computational Linguistics.

Cassandra L Jacobs and Yuval Pinter. 2022. Lost in space marking. arXiv preprint arXiv:2208.01561.

Jean Kaddour. 2023. The minipile challenge for dataefficient language models.

Stav Klein and Reut Tsarfaty. 2020. Getting the ##life out of living: How adequate are word-pieces for modelling complex morphology? In Proceedings of the 17th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology, pages 204–209, Online. Association for Computational Linguistics.

Taku Kudo. 2018. Subword regularization: Improving neural network translation models with multiple subword candidates. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 66–75, Melbourne, Australia. Association for Computational Linguistics.

Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66–71, Brussels, Belgium. Association for Computational Linguistics.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. Race: Large-scale reading comprehension dataset from examinations.

Hector J. Levesque, Ernest Davis, and Leora Morgenstern. 2012. The winograd schema challenge. In 13th International Conference on the Principles ofKnowledge Representation and Reasoning, KR 2012, Proceedings of the International Conference on Knowledge Representation and Reasoning, pages 552–561. Institute of Electrical and Electronics Engineers Inc. 13th International Conference on the Principles of

Knowledge Representation and Reasoning, KR 2012 ; Conference date: 10-06-2012 Through 14-06-2012.

Haoran Lian, Yizhe Xiong, Jianwei Niu, Shasha Mo, Zhenpeng Su, Zijia Lin, Peng Liu, Hui Chen, and Guiguang Ding. 2024. Scaffold-bpe: Enhancing byte pair encoding with simple and effective scaffold token removal.

Davis Liang, Hila Gonen, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer, and Madian Khabsa. 2023. XLM-V: Overcoming the vocabulary bottleneck in multilingual masked language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 13142–13152, Singapore. Association for Computational Linguistics.

Tomasz Limisiewicz, Jiˇrí Balhar, and David Marecek.ˇ 2023. Tokenization impacts multilingual language modeling: Assessing vocabulary allocation and overlap across languages. In Findings ofthe Association for Computational Linguistics: ACL 2023, pages 5661–5681, Toronto, Canada. Association for Computational Linguistics.

Sabrina J. Mielke, Zaid Alyafeai, Elizabeth Salesky, Colin Raffel, Manan Dey, Matthias Gallé, Arun Raja, Chenglei Si, Wilson Y. Lee, Benoît Sagot, and Samson Tan. 2021. Between words and characters: A brief history of open-vocabulary modeling and tokenization in nlp.

Tomas Mikolov, Ilya Sutskever, Anoop Deoras, Hai Son Le, Stefan Kombrink, and Jan Honza Cernocký. 2011. Subword language model-<sup>ˇ</sup> ing with neural networks. Preprint available at: https://api.semanticscholar.org/ CorpusID:46542477.

Anselmo Peñas, Eduard Hovy, Pamela Forner, Álvaro Rodrigo, Richard Sutcliffe, and Roser Morante. 2013. Qa4mre 2011-2013: Overview of question answering for machine reading evaluation. In CLEF 2013, LNCS 8138, pages 303–320.

Ivan Provilkov, Dmitrii Emelianenko, and Elena Voita. 2020. BPE-dropout: Simple and effective subword regularization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1882–1892, Online. Association for Computational Linguistics.

Jonne Saleva and Constantine Lignos. 2023. What changes when you randomly choose BPE merge operations? not much. In Proceedings of the Fourth Workshop on Insightsfrom Negative Results in NLP, pages 59–66, Dubrovnik, Croatia. Association for Computational Linguistics.

Mike Schuster and Kaisuke Nakajima. 2012. Japanese and korean voice search. In 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5149–5152.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, Berlin, Germany. Association for Computational Linguistics.

Jasdeep Singh, Bryan McCann, Richard Socher, and Caiming Xiong. 2019. BERT is not an interlingua and the bias of tokenization. In Proceedings of the 2nd Workshop on Deep Learning Approaches for Low-Resource NLP (DeepLo 2019), pages 47–55, Hong Kong, China. Association for Computational Linguistics.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models.

Omri Uzan, Craig W. Schmidt, Chris Tanner, and Yuval Pinter. 2024. Greed is all you need: An evaluation of tokenizer inference methods. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 813–822, Bangkok, Thailand. Association for Computational Linguistics.

A. Viterbi. 1967. Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. IEEE Transactions on Information Theory, 13(2):260–269.

Jeffrey S. Vitter. 1985. Random sampling with a reservoir. ACM Transactions on Mathematical Software, 11(1):37–57.

Johannes Welbl, Nelson F. Liu, and Matt Gardner. 2017. Crowdsourcing multiple choice science questions. ArXiv, abs/1707.06209.

F Wilcoxon. 1945. Individual comparisons by ranking methods. biom. bull., 1, 80–83.

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation.

Shaked Yehezkel and Yuval Pinter. 2023. Incorporating context into subword vocabularies. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 623–635, Dubrovnik, Croatia. Association for Computational Linguistics.

Vilém Zouhar, Clara Meister, Juan Gastaldi, Li Du, Mrinmaya Sachan, and Ryan Cotterell. 2023a. Tokenization and the noiseless channel. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5184–5207, Toronto, Canada. Association for Computational Linguistics.

Vilém Zouhar, Clara Meister, Juan Gastaldi, Li Du, Tim Vieira, Mrinmaya Sachan, and Ryan Cotterell. 2023b. A formal perspective on byte-pair encoding. In Findings of the Association for Computational Linguistics: ACL 2023, pages 598–614, Toronto, Canada. Association for Computational Linguistics.

## A Expanded description of PATHPIECE

This section provides a self-contained explanation of PATHPIECE, expanding on the one in §3, with additional details on the vocabulary construction and complexity.

In order to design an optimal vocabulary , it is first necessary to know how the vocabulary will be used to tokenize. There can be no best vocabulary in the abstract. Thus, we first present a new lossless subword tokenizer PATHPIECE. This tokenization over our training corpus will provide the context to design a coherent vocabulary.

## A.1 Tokenization for a given vocabulary

We work at the byte level, and require that all 256 single byte tokens are included in any given vocabulary . This avoids any out-of-vocabulary tokens by falling back to single bytes in the worst case.

Tokenization can be viewed as a compression problem, where we would like to tokenize text in a few tokens as possible. This has direct benefits, as it allows more text to fit in a given context window. A Minimum Description Length (MDL) argument can also be made that the tokenization using the fewest tokens best describes the data, although we saw in Subsection 6.1 this may not always hold in practice.

Tokenizers such as BPE and WordPiece make greedy decisions, such as choosing which pair of current tokens to merge to create a new one, which results in tokenizations that may use more tokens than necessary. In contrast, PATHPIECE will find an optimal tokenization by finding a shortest path through a Directed Acyclic Graph (DAG). Informally, each byte i of training data forms a node in the graph, and there is an edge if the w byte sequence ending at i is a token in .

An implementation of PATHPIECE is given in Algorithm 2, where input d is a text document of n bytes,  is a given vocabulary, and $L$ is a limit on the maximum width of a token in bytes. It has complexity $O ( n L )$ , following directly from the two nested for-loops. It iterates over the bytes i in $d ,$ computing 4 values for each. It computes the shortest path length $p l [ i ]$ in tokens up to and including byte i, the width wid[i] of a token with that shortest path length, and the solution count sc[i] of optimal solutions found thus far with that shortest length. We also remember the valid tokens of width 2 or more ending at each location i in vt[i], which will be used in the next section.

There will be multiple tokenizations with the same optimal length, so some sort of tiebreaker is needed. The longest token or a randomly selected token are obvious choices. We have presented the random tiebreaker method here, where a random solution is selected in a single pass in lines 29-32 of the listing using an idea from reservoir sampling (Vitter, 1985).

A backward pass through d constructs the optimal tokenization from the wid[e] values from the forward pass.

## A.2 Optimal Vocabulary Construction

## A.2.1 Vocabulary Initialization

We will build an optimal vocabulary by starting from a large initial one, and sequentially omitting batches of tokens. We start with the most frequently occurring byte n-grams in a training corpus, of width 1 to $L ,$ or a large vocabulary trained by BPE or Unigram. We then add any single byte tokens that were not already included, making room by dropping the tokens with the lowest counts. In our experiments we used an initial vocabulary size of $| \mathcal { V } | = 2 ^ { 1 8 } = 2 6 2 , 1 4 4$

## A.2.2 Increase from omitting a token

Given a PATHPIECE tokenization $t _ { 1 } , \dots , t _ { K _ { d } }$ $\forall d \in { \mathcal { C } }$ for training corpus , we would like to know the increase in the overall length of a tokenization $K = \textstyle \sum _ { d } K _ { d }$ from omitting a given token t from our vocabulary, $\mathcal { V } \backslash \{ t \}$ and recomputing the tokenization. Tokens with a low increase are good candidates to remove from the vocabulary (Kudo, 2018). However, doing this from scratch for each t would be a very expensive $O ( n L | \mathcal { V } | )$ operation.

We make a simplifying assumption that allows us to compute these increases more efficiently. We omit a specific token $t _ { k }$ in the tokenization of document $d ,$ and compute the minimum increase $M I _ { k d }$ in $K _ { d }$ from not having that token $t _ { k }$ in the tokenization of d. We then aggregate over the documents to get the overall increase for t:

Algorithm 2 PATHPIECE segmentation.   
1: procedure PATHPIECE(d, , L)   
2: n len(d) ▷ document length   
3: for $i \gets 1 ,$ n do   
4: $w i d [ i ]  0$ ▷ shortest path token   
5: $p l [ i ] ^ { \ast }  \infty$ ▷ shortest path len   
6: $s c [ i ] \gets 0$ ▷ solution count   
7: $v t [ i ]  [ ]$ ▷ valid token list   
8: for $e \gets 1 , n$ do ▷ token end   
9: for $v  1 , L \mathbf { d o }$ ▷ token width   
10: $s \gets e - w + 1$ ▷ token start   
11: ${ \mathrm { i f ~ } } s \geq 1 { \mathrm { ~ t h e n } }$ ▷ s in range   
12: $t  d [ s : e ]$ ▷ token   
13: $\mathbf { i f } t \in \dot { \mathcal { V } }$ then   
14: $\mathbf { i f } \ s = 1$ then ▷ 1 tok path   
15: $w i d [ e ]  w$   
16: $p l [ e ] \dot { } \gets 1$   
17: $s c [ e ] \gets 1$   
18: else   
19: if w $\geq 2$ then   
20: $v t [ e ] .$ . appen $\mathsf { I } ( w )$   
21: nl $ p l [ s - 1 ] + 1$   
22: if nl $< \dot { p l } [ e ]$ then   
23: $p l [ e ] \gets n l$   
24: $w i d [ e ]  w$   
25: $s c [ e ] \dot {  } 1$   
26: else if $\dot { \boldsymbol { n } l } = \boldsymbol { p l } [ \boldsymbol { e } ]$ then   
27: sc[e] $ s c [ e ] + 1$   
28: $r \gets \mathrm { r a n d ( ) }$   
29: $\mathbf { i f } r \le 1 / s c [ e ]$ then   
30: $w i d [ e ] \doteq w$   
31: $T \gets [ ]$ ▷ output token list   
32: $e \gets n$ ▷ start at end of d   
33: while e $; \geq 1$ do   
34: $w \gets w i d [ e ]$ ▷ width of short path tok   
35: $s \gets e - \dot { w } + 1$ ▷ token start   
36: $t  d [ s : e ]$ ▷ token   
37: $T . { \mathrm { a p p e n d } } ( t )$   
38: $e  e - w$ ▷ back up a token   
39: return reversed(T) ▷ reverse order

$$
M I _ { t } = \sum _ { d \in \mathcal { C } } \sum _ { k = 1 | t _ { k } = t } ^ { K _ { d } } M I _ { k d } .\tag{7}
$$

This is similar to computing the increase from $\nu \backslash$ t , but ignores interaction effects from having several occurrences of the same token t close to each other in a given document.

With PATHPIECE, it turns out we can compute the minimum increase in tokenization length without actually recomputing the tokenization. Any tokenization not containing $t _ { k }$ must either contain a token boundary somewhere inside of $t _ { k }$ breaking it in two, or it must contain a token that entirely contains $t _ { k }$ as a superset. Our approach will be to enumerate all the occurrences for these two cases, and to find the minimum increase $M I _ { k d }$ overall.

Before considering these two cases, there is a shortcut that often tells us that there would be no increase due to omitting $t _ { k }$ ending at index e. We computed the solution count vector $s c [ e ]$ when running Algorithm 2. If $s c [ e ] > 1$ for a token ending at $e ,$ then the backward pass could simply select one of the alternate optimal tokens, and find an overall tokenization of the same length.

Let $t _ { k }$ start at index s and end at index $e ,$ inclusive. Remember that path length $p l [ i ]$ represents the number of tokens required for shortest path up to and including byte i. We can also run Algorithm 2 backwards on $d ,$ computing a similar vector of backwards path lengths $b p l [ i ]$ , representing the number of tokens on a path from the end of the data up to and including byte i. The overall minimum length of a tokenization with a token boundary after byte j is thus:

$$
K _ { j } ^ { b } = p l [ j ] + b p l [ j + 1 ] .\tag{8}
$$

We have added an extra constraint on the shortest path, that there is a break at $j ,$ so clearly $K _ { j } ^ { b r } \geq$ $p l [ n ]$ . The minimum increase for the case of having a token boundary within $t _ { k }$ is thus:

$$
M I _ { k d } ^ { b } = \operatorname* { m i n } _ { j = s , \ldots , e - 1 } K _ { j } ^ { b } - p l [ n ] .\tag{9}
$$

Each token $t _ { k }$ will have no more than $L - 1$ potential internal breaks, so the complexity of computing $M I _ { k d } ^ { b }$ is $O ( L )$

The minimum increase from omitting $t _ { k }$ could also be on a tokenization containing a strict superset of $t _ { k }$ . Let this superset token be $t _ { k } ^ { \prime } .$ with start $s ^ { \prime }$ and end $e ^ { \prime }$ inclusive. To be a strict superset jumping over $t _ { k } .$ , we must have $s ^ { \prime } < s$ and $e ^ { \prime } \geq e$ , or $s ^ { \prime } \leq s$ and $e ^ { \prime } > e ,$ subject to the constraint that the width $w ^ { \prime } = e ^ { \prime } - s ^ { \prime } + 1 \leq L$ . In this case, the minimum length of using the superset token $t _ { k } ^ { \prime }$ would be:

$$
K _ { t _ { k } ^ { \prime } } ^ { s } = p l [ s ^ { \prime } - 1 ] + b p l [ e ^ { \prime } + 1 ] + 1 ,\tag{10}
$$

which is the path length to get to the byte before $t _ { k } ^ { \prime }$ plus the path length go backwards to the byte after $t _ { k } ^ { \prime }$ , plus 1 for the token $t _ { k } ^ { \prime }$ itself.

We remembered a list of the widths of the tokens ending at each byte, $v t [ e ]$ in Algorithm 2. The set of superset tokens S can be found by examining the $O ( L )$ potential $e ^ { \prime } { \mathrm { , } }$ , and then seeing if the $w ^ { \prime } \in v t [ e ^ { \prime } ]$ give tokens forming a strict superset. There are $O ( L )$ potential tokens ending at $e ^ { \prime }$ in $\boldsymbol { v } t [ \boldsymbol { e } ^ { \prime } ]$ , so the overall complexity of finding the superset tokens is $O ( L ^ { 2 } )$

Similar to the previous case, we can compute the minimum increase from replacing $t _ { k }$ with a superset token by taking the minimum increase over the superset tokens:

$$
M I _ { k d } ^ { s } = \operatorname* { m i n } _ { t _ { k } ^ { \prime } \in S } K _ { t _ { k } ^ { \prime } } ^ { s } - p l [ n ] .\tag{11}
$$

Finally, the overall minimum increase $M I _ { k d }$ from omitting $t _ { k }$ is simply

$$
M I _ { k d } = \operatorname* { m i n } ( M I _ { k d } ^ { b } , M I _ { k d } ^ { s } ) .\tag{12}
$$

When aggregating over all $t _ { k }$ according to eq (7), one iteration of the vocabulary construction procedure will have complexity $O ( n L ^ { 2 } )$

## B Language Model Parameters

The 350M parameter models were trained using the MPT architecture<sup>18</sup> with the following parameters:

```yaml
# Model
model:
name: mpt_causal_lm
init_deice: meta
d_model: 1024
n_heads: 16
n_layers: 24
expansion_ratio: 4
max_seq_len: 2048
attn_config:
alibi: true
attn_impl: triton
clip_qkv: 6
# Optimization
device_eval_batch_size: 5
device_train_microbatch_size: 32
global_train_batch_size: 1024 # ~2M token
max_duration: 100000ba # ~200B tokens
optimizer:
name: decoupled_adamw
lr: 3.0e-4
betas:
- 0.9
- 0.95
eps: 1.0e-08
weight_decay: 0.0001
scheduler:
name: cosine_with_warmup
t_warmup: 0.05dur
alpha_f: 0.1
# System
precision: amp_bf16
# Algos and Callbacks
algorithms:
gradient_clipping:
clipping_threshold: 1
clipping_type: norm
```

The 1.3B parameter models simply changes: d\_model: 1024

The 2.4B parameter models updates:

d\_model: 2560   
n\_heads: 20   
n\_layers: 32

## C Description of Downstream Tasks

To evaluate the performance of our various tokenization experiments, we select ten competitive benchmarks from lm-evaluation-harness (Gao et al., 2023)<sup>19</sup>, that we broadly categorize into three types of Question Answering (QA) tasks: Knowledge-based, Common-sense Reasoning and Context-based.

Knowledge Based Tasks Knowledge based tasks in this study expect LLMs to answer questions based on domain-specific internal retrieval. Our Knowledge-based baselines in this work include:

SciQ: The SciQ task, proposed by Welbl et al. (2017) contains a total of 13,679 science exam questions. The questions are in multiple-choice format with 4 answer options each. An additional text is provided as supporting evidence for a majority of the answers.

ARC (AI2 Reasoning Challenge): Clark et al. (2018) compiles grade-school level, multiplechoice science question dataset consists of 7,787 science exam questions that are split into “easy” and “hard” sets. For this study, we employ the easy set of 5,197 questions, each having 4 answer choices.

MathQA: Amini et al. (2019) introduce a dataset of math word problems that require LLMs to use their internal understanding of mathematical equations and arithmetic comprehension. Similar to SciQ, this dataset consists of 37k multiple-choice questions with the equations for each used annotated.

HendrycksTest: Hendrycks et al. (2021) provide a comprehensive suite of of multiple-choice tests for assessing text models in multi-task contexts. It comprises of 57 tasks such as elementary mathematics, US history, law of which we use the sociology and marketing tests.

Commonsense Reasoning Tasks These tasks assess the model’s capability to infer and reason about everyday scenarios based on implicit knowledge.

COPA (Choice ofPlausible Alternatives): COPA proposed by Brassard et al. (2022) is a benchmark for assessing progress in open-domain commonsense causal reasoning. It consists of 1000 questions where each question is composed of a premise and two alternatives. The task is to select the alternative that more plausibly has a causal relation with the premise.

PiQA (Physical Interaction Question Answering): Bisk et al. (2019) introduce a task that assess the understanding of physical commonsense by language models. Comprised of everyday situation with a preference for atypical solutions, this dataset is formulated as multiple choice question with two possible solutions choices for each question.

Winograd Schema Challenge: Levesque et al. (2012) define a task with a pair of sentences that differ only in one or two words and that contain a referential ambiguity that is resolved in opposite directions in the two sentences. This dataset of 273 tasks test language model understanding of the content of the text and disambiguation ability.

Context Based Tasks These tasks are reliant on understanding context and drawing conclusions from it.

RACE (Reading Comprehensionfrom Examinations): RACE proposed by Lai et al. (2017) is a collection of English questions set aside to Chinese school students. Each item is divided into two parts, a passage that the student must read and a set of 4 potential answers, requiring extraction and reasoning capabilities.

QA4MRE (Question Answering for Machine Reading Evaluation): QA4MRE by Peñas et al. (2013) is a benchmark designed to resolve reading comprehension challenges. This task focuses on reading of single documents and identifying the answers to a set of questions. Questions are in the form of multiple choice with one correct option.

Our goal was to select tasks where a 350M parameter model could do significantly better than random chance, avoiding evaluation right at the noisier random threshold. We started with the tasks that had a non-zero random score (indicating multiple choice), and then chose tasks where BPE at a vocabulary size 40,960 could do well above random. In the end, the average accuracy across models was more than 15% above random on all tasks.

Note that in results tables we have shortened the name hendrycksTest-marketing to marketing, hendrycksTest-sociology to sociology, and qa4mre\_2013 to qa4mre.

## D Effect of model convergence

Each model was trained on around 200 billion tokens. Figure 7 gives a plot of the average accuracy for PathPieceL with a BPE initial vocabulary and a vocabulary size of 40,960 at various checkpoints in the language model training process. It also shows checkpoints for the larger 1.3B and 2.4B models discussed in the Limitations section. With the exception of the 100k checkpoint at 1.3B, the model appears to be continually improving. It is unclear why the 100k checkpoint did so poorly.

![](images/9753cbdd7f3386981f0647257919ec4966d3c7e7e50b9c939cc0cb1c75ff7202.jpg)  
Figure 7: Checkpoint accuracy values for PathPieceL with an initial vocabulary from BPE and a vocabulary size of 40,960, evaluated at 5 checkpoints.

## E Additional Analysis

Here we additional details for results from §6 that are just summarized in the text in the interest of space.

## E.1 Segmentation

Tokenizers often use the segmentation strategy that is used in vocabulary construction. However, any vocabulary can also be used with PATHPIECE and with the greedy left-to-right segmentation methods.

We find that BPE works quite well with greedy segmentation (overall rank 4, insignificantly different from the top rank), but not with the shortestpath segmentation of PATHPIECEL (13).

Unigram, on the other hand, seems to be more tightly tied to its default maximum likelihood segmentation (2), which was significantly better than both Greedy (7) and PATHPIECEL (17).

## E.2 Digit Pre-tokenization

We have two examples isolating Digit pre-tokenization, when a digit must always be its own token.

![](images/7db02ac7e2e0bae8c1eb53f14250bf1a3f049b4b2abe9d68571c2f10de93183e.jpg)  
Figure 8: Segmentation of BPE.

Pairwise p-values between the pairs of runs are p(3,4)=0.52, p(3,13)=4.4e-5, p(4,13)=8.8e-6.  
![](images/4ed89c53aca1e85fbea1b873120cc6cd9644b56737aeaa705c1ecc35e7beb752.jpg)  
Figure 9: Segmentation of Unigram. Pairwise p-values between the pairs of runs are p(2,7)=0.041, p(2,17)=2.9e-06, p(7,17)=2.9e-06

Figure 10 shows Digit hurts for Sage with an ngram initial vocabulary, while Figure 11 shows no significant differences for PathPieceL, also with an n-gram initial vocabulary.

![](images/21e9e807d64fdbadd77edd22cf92555e748f162e0cc48f364a7fbdca52ebca90.jpg)  
Figure 10: Pre-tokenization of Sage, n-gram initial, p=0.025.

With the exception of mathqa, none of our downstream tasks were particularly mathematical in nature. It is likely this makes it hard to make a definitive judgement on Digit with our experiments.

## E.3 Vocabulary Construction

Figure 12 gives a Venn diagram of the overlap in vocabularies between Unigram, PathPieceL, and SaGe, when both PathPieceL and SaGe were constructed from a large initial vocabulary of size 262,144 from Unigram. As with Figure 5, we see that PathPiece is more similar to Unigram, while SaGe chose more distinct tokens.

![](images/fef0a393488a2e7a474cbd04ba4266284cdcfd9e1c1b67e7ac0521216a24c235.jpg)  
Figure 11: Pre-tokenization of PathPieceL n-gram, p=0.54.

![](images/78e6be00a4336ab92a91ae3c447485639f2edbb4a3b17f32a17c8073c040ceec.jpg)  
Figure 12: Venn diagrams comparing 40,960 token vocabularies of Unigram, PathPieceL and SaGe, where the latter two were both trained from a initial Unigram vocabulary of size 262,144

## E.4 PathPiece tie breaking

The difference in tie breaking between choosing the longest token with PathPieceL versus choosing randomly with PathPieceR turns out not to be significant, as seen in in Figure 13.

![](images/4434ee120dd5f49aaeee280932a474e00177449c83fa57fce9d140623f384664.jpg)  
Figure 13: Tiebreaking PathPieceL vs PathPieceR with n-gram, p=0.067.

## F RandTrain

None of our experiments completely isolate the effect of the vocabulary construction step. We created a new baseline random vocabulary construction approach, RandTrain, in an attempt to do so. It is meant to work with a top-down method like SaGe or PathPieceL, and uses the same initial vocabulary, pre-tokenization, and segmentation as either of those, with a simple vocabulary construction algorithm.

We compute a count for each token in the vocabulary. For the top n-gram initial vocabulary it is simply the n-gram count from the training corpus. For a BPE initial vocabulary we tokenized the training corpus with BPE and the large initial vocabulary, and then use the occurrence counts of each token. We normalize these counts into target selection probabilities $p _ { k }$ for token $t _ { k }$

The RandTrain vocabulary construction process is simply to randomly sample our desired vocabulary size m of tokens from the initial vocabulary, proportionally to $p _ { k }$ , without replacement. Sampling without replacement is necessary to avoid have duplicate words in the vocabulary. Interestingly, this is not possible if there are any $p _ { k } \ >$ $1 / m$ , which are termed infeasible or overweight items (Efraimidis, 2010). The intuition behind this is when selecting m items without replacement, it is not possible to select a given item more than once. So even if an item is always selected in a sample, the selection probability will be $p _ { k } = 1 / m$

We sampled without replacement using the A-ES Algorithm described in Efraimidis (2010). A significant number the most common tokens in the vocabulary were infeasible and hence were unable to reach their target $p _ { k }$ . A token with a higher $p _ { k }$ is more likely to be sampled than a token with a lower one, but they may significantly differ from their target $p _ { k }$

We build 6 RandTrain models with 3 different types of pre-tokenization, and with Greedy segmentation to compare to $\mathrm { S a G e , }$ , and PathPieceL segmentation to compare to PathPieceL. We only used a single vocabulary size of 40,960, so p-values are only computed on the 10 task accuracies, rather than the 30 used elsewhere. Task level accuracies are given in Table 6 and Table 7 in Appendix G.

Before comparing RandTrain to SaGe and Path-PieceL, we will compare our RandTrain runs to each other, with different segmentation approaches. In Figure 14 and Figure 16 we have pairs of Rand-Train runs that only vary by the segmentation method.

In line with Subsection E.1, Greedy performs significantly better than PathPieceL segmentation in all 3 cases. However, for the two cases with an n-gram initial vocabulary the PathPieceL segmentation did extremely poorly. The RandTrain vocabulary construction, n-gram initial vocabulary, and PathPieceL segmentation interact somehow to give accuracies well below any others.

![](images/9435fc5a6d0594e7c63b4856420d51dccb640440da5c37ea101c681bcf13a36b.jpg)  
Figure 14: Comparison of Greedy and PathPieceL segmentation, with RandTrain vocabulary construction, BPE initial vocab, and FirstSpace pre-tokenization, p=0.0273

![](images/86cad0bc1445f354c89cdf11e087d1e786ab9d5bd26e5bc3619217cf106ce451.jpg)  
Figure 15: Comparison of Greedy and PathPieceL segmentation, with RandTrain vocabulary construction, n-gram initial vocab, and FirstSpace pre-tokenization, p=0.00195

This makes the comparison of RandTrain to Path-PieceL less informative. We can see in Figure 17 that PathPieceL is significantly better than Rand-Train with a BPE initial vocabulary.

However, the other two comparisons in Figure 18 are Figure 19 are not that meaningful. They are significantly better, but that is more about the weak baseline of RandTrain with PathPieceL segmentation than anything positive about PathPieceL.

The remaining comparison between SaGe and RandTrain is more interesting. In Figure 20 and Figure 21 SaGe was not significantly better than RandTrain, with a p-value of 0.0645.

The cases is even worse for the two n-gram initial vocabulary cases. In Figure 21 the p-value was a 0.688, and in Figure 22 RandTrain was actually better, although not significantly.

We saw in Table 1 that both PathPieceL-BPE and SaGe-BPE are effective tokenizers. In attempting to isolate the benefit from the vocabulary construction step, we see that PathPieceL-BPE outperforms our simple baseline. However, SaGe was unable to outperform the baseline, perhaps implying that RandTrain may actually be a simple but fairly effective vocabulary construction method.

![](images/18da80e9170deb82d9a846194c5adfcb18f882ea848d4cd323de5e77b8e8a39c.jpg)  
Figure 16: Comparison of Greedy and PathPieceL segmentation, with RandTrain vocabulary construction, n-gram initial vocab, and FirstSpaceDigit pre-tokenization, p=0.00293

![](images/1fb07117d0ca3d215aa643d963e3c58eb10c432f7dd33396cdb6e4bd391407fc.jpg)  
Figure 17: Comparison of PathPieceL and RandTrain, with BPE initial vocab, and FirstSpace pre-tokenization, p=0.0137

## G Detailed Experimental Results

This section gives the detailed accuracy results for the 10 downstream evaluation tasks on each model that was trained. The tables are divided by the vocabulary size used, with Table 4 and Table 5 for 32,768; Table 6 and Table 7 for 40,960; and Table 8 and Table 9 for 49,152. The highest value or values (in the case of ties) are shown in bold. Table 10 show the same results as Table 1, but are sorted from best to worst by rank. The corpus token count (CTC), Rényi efficiencies, and average accuracies for the 54 runs in Figure 3 are given in Table 11.

The detailed accuracy results for our 1.3B parameter models, which were all performed at a single vocabulary size of 40,960, are given in Table 12 and Table 13. Average accuracy results for larger models of 1.3B and 2.4B parameters are given in Table 14. See §7 for more discussion of this table.

![](images/2d83b5893c29607c9f3d24e7bf24c19ee58debc3440a906f2d99d09d057c8dd0.jpg)  
Figure 18: Comparison of PathPieceL and RandTrain, with n-gram initial vocab, and FirstSpace pre-tokenization, p=9.77e-4

![](images/896e9c9fae04217834d17cd40fe53da01f1b6dbd76c66daf55ad6fbb7493a014.jpg)  
Figure 19: Comparison of PathPieceL and RandTrain, with n-gram initial vocab, and FirstSpaceDigits pre-tokenization, p=0.00977

![](images/1cfbe5acde98255cb1dd27fe9fec96a1ccbe4cde4d9d05f1278bbe4b4701227d.jpg)  
Figure 20: Comparison of SaGe and RandTrain, with BPE initial vocab, and FirstSpace pre-tokenization, p=0.0645

![](images/902f5211f866bdb63002a189ffa830f6c63e43fd240d801a460583fc9c44dd84.jpg)  
Figure 21: Comparison of SaGe and RandTrain, with n-gram initial vocab, and FirstSpace pre-tokenization, p=0.688

![](images/0fe0069d06d355011bb23cac5d8146a549a74db0e9dce3fdb576a25448f4bf18.jpg)  
Figure 22: Comparison of RandTrain and SaGe, with n-gram initial vocab, and FirstSpaceDigit pre-tokenization, p=0.15

<table><tr><td>Vocab Constr</td><td>Init Voc</td><td>Pre-tok</td><td>Segment</td><td>Avg</td><td>arc_easy</td><td>copa</td><td>mktg</td><td>mathqa</td><td>piqa</td></tr><tr><td rowspan="3">BPE</td><td></td><td>FirstSpace</td><td>Merge</td><td>48.8</td><td>51.2</td><td>69.0</td><td>32.9</td><td>23.9</td><td>66.3</td></tr><tr><td></td><td>FirstSpace</td><td>Greedy</td><td>48.3</td><td>51.9</td><td>66.0</td><td>32.9</td><td>23.7</td><td>65.6</td></tr><tr><td></td><td>FirstSpace</td><td>PathPieceL</td><td>45.6</td><td>45.6</td><td>61.0</td><td>29.9</td><td>23.0</td><td>60.5</td></tr><tr><td rowspan="3">Unigram</td><td></td><td>FirstSpace</td><td>Likelihood</td><td>49.2</td><td>50.7</td><td>73.0</td><td>30.8</td><td>23.1</td><td>66.3</td></tr><tr><td></td><td>FirstSpace</td><td>Greedy</td><td>47.9</td><td>50.3</td><td>68.0</td><td>31.2</td><td>23.1</td><td>65.2</td></tr><tr><td></td><td>FirstSpace</td><td>PathPieceL</td><td>43.6</td><td>41.2</td><td>57.0</td><td>31.6</td><td>22.0</td><td>60.6</td></tr><tr><td>WordPiece</td><td></td><td>FirstSpace</td><td>Greedy</td><td>48.5</td><td>52.5</td><td>64.0</td><td>32.5</td><td>23.9</td><td>65.6</td></tr><tr><td rowspan="4">SaGe</td><td>BPE</td><td>FirstSpace</td><td>Greedy</td><td>47.9</td><td>49.7</td><td>67.0</td><td>26.5</td><td>23.2</td><td>65.9</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>Greedy</td><td>48.4</td><td>50.3</td><td>71.0</td><td>29.5</td><td>22.0</td><td>65.1</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>Greedy</td><td>47.5</td><td>48.8</td><td>64.0</td><td>29.5</td><td>23.0</td><td>66.6</td></tr><tr><td>Unigram</td><td>FirstSpace</td><td>Greedy</td><td>48.4</td><td>52.0</td><td>74.0</td><td>27.8</td><td>22.7</td><td>65.7</td></tr><tr><td rowspan="4">PathPieceL</td><td>BPE</td><td>FirstSpace</td><td>PathPieceL</td><td>49.3</td><td>50.8</td><td>68.0</td><td>34.2</td><td>23.0</td><td>66.4</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>PathPieceL</td><td>44.8</td><td>42.3</td><td>61.0</td><td>27.4</td><td>23.0</td><td>61.2</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceL</td><td>44.6</td><td>42.3</td><td>62.0</td><td>31.2</td><td>22.8</td><td>61.2</td></tr><tr><td>Unigram</td><td>FirstSpace</td><td>PathPieceL</td><td>46.9</td><td>50.4</td><td>64.0</td><td>24.8</td><td>23.5</td><td>66.2</td></tr><tr><td rowspan="3">PathPieceR</td><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceR</td><td>45.3</td><td>46.9</td><td>67.0</td><td>26.9</td><td>22.4</td><td>59.9</td></tr><tr><td>n-gram</td><td>None</td><td>PathPieceR</td><td>43.5</td><td>42.5</td><td>65.0</td><td>26.1</td><td>22.8</td><td>61.7</td></tr><tr><td>n-gram</td><td>SpaceDigit</td><td>PathPieceR</td><td>47.5</td><td>48.6</td><td>68.0</td><td>32.9</td><td>23.3</td><td>65.0</td></tr><tr><td>Random</td><td></td><td></td><td></td><td>32.0</td><td>25.0</td><td>50.0</td><td>25.0</td><td>20.0</td><td>50.00</td></tr></table>

Table 4: 350M parameter model, 32,768 token vocabulary, accuracy (%) on average and initial 5 tasks

<table><tr><td>Vocab Constr</td><td>Init Voc</td><td>Pre-tok</td><td>Segment</td><td>qa4mre</td><td>race</td><td>sciq</td><td>sociology</td><td>wsc273</td></tr><tr><td rowspan="3">BPE</td><td></td><td>FirstSpace</td><td>Merge</td><td>29.6</td><td>29.2</td><td>87.3</td><td>30.9</td><td>67.8</td></tr><tr><td></td><td>FirstSpace</td><td>Greedy</td><td>27.5</td><td>30.7</td><td>88.0</td><td>30.9</td><td>66.3</td></tr><tr><td></td><td>FirstSpace</td><td>PathPieceL</td><td>28.2</td><td>29.0</td><td>83.8</td><td>28.4</td><td>66.3</td></tr><tr><td rowspan="3">Unigram</td><td></td><td>FirstSpace</td><td>Likelihood</td><td>31.0</td><td>30.2</td><td>86.4</td><td>31.8</td><td>68.5</td></tr><tr><td></td><td>FirstSpace</td><td>Greedy</td><td>28.9</td><td>30.6</td><td>86.9</td><td>31.8</td><td>62.6</td></tr><tr><td></td><td>FirstSpace</td><td>PathPieceL</td><td>29.9</td><td>27.5</td><td>74.6</td><td>26.4</td><td>65.6</td></tr><tr><td>WordPiece</td><td></td><td>FirstSpace</td><td>Greedy</td><td>32.0</td><td>30.7</td><td>88.5</td><td>27.9</td><td>67.4</td></tr><tr><td rowspan="4">SaGe</td><td>BPE</td><td>FirstSpace</td><td>Greedy</td><td>31.7</td><td>30.2</td><td>89.0</td><td>28.4</td><td>67.8</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>Greedy</td><td>31.0</td><td>30.3</td><td>86.6</td><td>32.3</td><td>66.0</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>Greedy</td><td>30.0</td><td>31.0</td><td>87.8</td><td>25.9</td><td>68.5</td></tr><tr><td>Unigram</td><td>FirstSpace</td><td>Greedy</td><td>29.6</td><td>28.9</td><td>88.2</td><td>32.3</td><td>63.0</td></tr><tr><td rowspan="4">PathPieceL</td><td>BPE</td><td>FirstSpace</td><td>PathPieceL</td><td>28.5</td><td>31.1</td><td>88.8</td><td>35.3</td><td>67.0</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>PathPieceL</td><td>30.3</td><td>27.3</td><td>80.0</td><td>32.8</td><td>62.6</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceL</td><td>27.8</td><td>25.5</td><td>79.2</td><td>31.3</td><td>62.6</td></tr><tr><td>Unigram</td><td>FirstSpace</td><td>PathPieceL</td><td>29.6</td><td>30.6</td><td>87.6</td><td>24.4</td><td>68.1</td></tr><tr><td rowspan="3">PathPieceR</td><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceR</td><td>28.5</td><td>29.4</td><td>78.6</td><td>28.9</td><td>64.5</td></tr><tr><td>n-gram</td><td>None</td><td>PathPieceR</td><td>27.1</td><td>27.0</td><td>77.7</td><td>28.9</td><td>56.0</td></tr><tr><td>n-gram</td><td>SpaceDigit</td><td>PathPieceR</td><td>25.0</td><td>29.4</td><td>85.7</td><td>32.3</td><td>64.8</td></tr><tr><td>Random</td><td></td><td></td><td></td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>50.0</td></tr></table>

Table 5: 350M parameter model, 32,768 token vocabulary, accuracy (%) on remaining 5 tasks

<table><tr><td>Vocab Constr</td><td>Init Voc</td><td>Pre-tok</td><td>Segment</td><td>Avg</td><td>arc_easy</td><td>copa</td><td>mktg</td><td>mathqa</td><td>piqa</td></tr><tr><td rowspan="3">BPE</td><td></td><td>FirstSpace</td><td>Merge</td><td>50.0</td><td>52.7</td><td>70.0</td><td>31.6</td><td>24.3</td><td>66.9</td></tr><tr><td></td><td>FirstSpace</td><td>Greedy</td><td>49.1</td><td>52.3</td><td>66.0</td><td>27.4</td><td>22.9</td><td>66.9</td></tr><tr><td></td><td>FirstSpace</td><td>PathPieceL</td><td>46.7</td><td>48.0</td><td>58.0</td><td>27.4</td><td>23.4</td><td>62.1</td></tr><tr><td colspan="2"></td><td>FirstSpace</td><td>Likelihood</td><td>49.1</td><td>51.4</td><td>71.0</td><td>32.1</td><td>23.4</td><td>66.1</td></tr><tr><td colspan="2">Unigram Unigram</td><td>FirstSpace</td><td>Greedy</td><td>48.5</td><td>49.9</td><td>64.0</td><td>30.3</td><td>23.3</td><td>65.7</td></tr><tr><td colspan="2"></td><td>FirstSpace</td><td>PathPieceL</td><td>43.1</td><td>40.5</td><td>56.0</td><td>28.6</td><td>23.0</td><td>60.3</td></tr><tr><td colspan="2">WordPiece</td><td>FirstSpace</td><td>Greedy</td><td>49.1</td><td>52.3</td><td>70.0</td><td>28.6</td><td>23.7</td><td>66.5</td></tr><tr><td rowspan="4">SaGe</td><td>BPE</td><td>FirstSpace</td><td>Greedy</td><td>49.2</td><td>50.8</td><td>70.0</td><td>29.9</td><td>23.2</td><td>66.4</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>Greedy</td><td>46.9</td><td>48.4</td><td>67.0</td><td>30.3</td><td>22.6</td><td>64.0</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>Greedy</td><td>48.5</td><td>49.8</td><td>68.0</td><td>32.9</td><td>22.8</td><td>65.4</td></tr><tr><td>Unigram</td><td>FirstSpace</td><td>Greedy</td><td>46.9</td><td>51.7</td><td>65.0</td><td>28.6</td><td>23.9</td><td>65.2</td></tr><tr><td rowspan="4">PathPieceL</td><td>BPE</td><td>FirstSpace</td><td>PathPieceL</td><td>49.4</td><td>52.1</td><td>71.0</td><td>29.9</td><td>23.9</td><td>66.9</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>PathPieceL</td><td>45.5</td><td>42.6</td><td>63.0</td><td>30.3</td><td>22.7</td><td>60.9</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceL</td><td>44.9</td><td>44.0</td><td>60.0</td><td>29.9</td><td>22.6</td><td>60.8</td></tr><tr><td>Unigram</td><td>FirstSpace</td><td>PathPieceL</td><td>48.5</td><td>51.7</td><td>71.0</td><td>31.2</td><td>24.2</td><td>66.2</td></tr><tr><td rowspan="3">PathPieceR</td><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceR</td><td>45.8</td><td>47.5</td><td>63.0</td><td>28.2</td><td>22.4</td><td>60.7</td></tr><tr><td>n-gram</td><td>None</td><td>PathPieceR</td><td>44.0</td><td>41.2</td><td>66.0</td><td>26.5</td><td>21.6</td><td>62.4</td></tr><tr><td>n-gram</td><td>SpaceDigit</td><td>PathPieceR</td><td>45.4</td><td>46.3</td><td>64.0</td><td>32.1</td><td>22.7</td><td>60.0</td></tr><tr><td rowspan="8">RandTrain</td><td>BPE</td><td>FirstSpace</td><td>Greedy</td><td>48.6</td><td>50.5</td><td>70.0</td><td>29.5</td><td>23.4</td><td>65.8</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>Greedy</td><td>47.9</td><td>50.0</td><td>63.0</td><td>29.5</td><td>23.3</td><td>65.3</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>Greedy</td><td>48.3</td><td>50.3</td><td>70.0</td><td>28.2</td><td>24.3</td><td>65.8</td></tr><tr><td>n-gram</td><td>None</td><td>Greedy</td><td>42.2</td><td>41.3</td><td>55.0</td><td>27.4</td><td>21.7</td><td>63.2</td></tr><tr><td>BPE</td><td>FirstSpace</td><td>PathPieceL</td><td>46.5</td><td>45.8</td><td>65.0</td><td>30.8</td><td>23.3</td><td>62.8</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceL</td><td>38.8</td><td>31.2</td><td>48.0</td><td>27.8</td><td>22.6</td><td>54.7</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>PathPieceL</td><td>40.0</td><td>30.7</td><td>55.0</td><td>26.5</td><td>20.8</td><td>55.4</td></tr><tr><td>n-gram</td><td>None</td><td>PathPieceL</td><td>36.8</td><td>27.7</td><td>56.0</td><td>28.6</td><td>22.8</td><td>54.5</td></tr><tr><td>random</td><td></td><td></td><td></td><td>32.0</td><td>25.0</td><td>50.0</td><td>25.0</td><td>20.0</td><td>50.0</td></tr></table>

Table 6: 350M parameter model, 40,960 token vocabulary, accuracy (%) on average and initial 5 tasks

<table><tr><td>Vocab Constr</td><td>Init Voc</td><td>Pre-tok</td><td>Segment</td><td>qa4mre</td><td>race</td><td>sciq</td><td>sociology</td><td>wsc273</td></tr><tr><td rowspan="3">BPE</td><td></td><td>FirstSpace</td><td>Merge</td><td>32.4</td><td>30.1</td><td>87.7</td><td>35.3</td><td>69.2</td></tr><tr><td></td><td>FirstSpace</td><td>Greedy</td><td>31.7</td><td>30.9</td><td>88.3</td><td>35.8</td><td>68.9</td></tr><tr><td></td><td>FirstSpace</td><td>PathPieceL</td><td>30.3</td><td>30.2</td><td>83.8</td><td>35.3</td><td>68.1</td></tr><tr><td rowspan="3">Unigram</td><td></td><td>FirstSpace</td><td>Likelihood</td><td>29.6</td><td>30.8</td><td>86.4</td><td>32.8</td><td>67.8</td></tr><tr><td></td><td>FirstSpace</td><td>Greedy</td><td>32.4</td><td>29.6</td><td>86.7</td><td>32.8</td><td>70.3</td></tr><tr><td></td><td>FirstSpace</td><td>PathPieceL</td><td>30.3</td><td>27.4</td><td>75.0</td><td>27.4</td><td>62.3</td></tr><tr><td>WordPiece</td><td></td><td>FirstSpace</td><td>Greedy</td><td>31.0</td><td>30.3</td><td>87.7</td><td>32.8</td><td>68.1</td></tr><tr><td rowspan="4">SaGe</td><td>BPE</td><td>FirstSpace</td><td>Greedy</td><td>28.9</td><td>30.2</td><td>89.5</td><td>34.8</td><td>67.8</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>Greedy</td><td>30.6</td><td>28.1</td><td>85.8</td><td>32.3</td><td>59.7</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>Greedy</td><td>29.2</td><td>30.0</td><td>88.4</td><td>33.3</td><td>65.2</td></tr><tr><td>Unigram</td><td>FirstSpace</td><td>Greedy</td><td>26.8</td><td>29.1</td><td>86.9</td><td>31.3</td><td>60.1</td></tr><tr><td rowspan="4">PathPieceL</td><td>BPE</td><td>FirstSpace</td><td>PathPieceL</td><td>31.0</td><td>29.6</td><td>87.3</td><td>34.3</td><td>67.8</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>PathPieceL</td><td>29.9</td><td>27.9</td><td>81.0</td><td>34.8</td><td>61.9</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceL</td><td>27.5</td><td>28.2</td><td>80.7</td><td>30.9</td><td>64.1</td></tr><tr><td>Unigram</td><td>FirstSpace</td><td>PathPieceL</td><td>31.3</td><td>29.7</td><td>86.3</td><td>29.9</td><td>63.7</td></tr><tr><td rowspan="3">PathPieceR</td><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceR</td><td>29.9</td><td>30.8</td><td>82.1</td><td>27.4</td><td>66.3</td></tr><tr><td>n-gram</td><td>None</td><td>PathPieceR</td><td>23.6</td><td>28.3</td><td>73.8</td><td>35.8</td><td>60.4</td></tr><tr><td>n-gram</td><td>SpaceDigit</td><td>PathPieceR</td><td>27.5</td><td>28.7</td><td>78.2</td><td>31.3</td><td>63.0</td></tr><tr><td rowspan="8">RandTrain</td><td>BPE</td><td>FirstSpace</td><td>Greedy</td><td>32.0</td><td>29.6</td><td>86.9</td><td>30.9</td><td>67.4</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>Greedy</td><td>30.6</td><td>30.0</td><td>87.5</td><td>31.3</td><td>68.1</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>Greedy</td><td>29.9</td><td>29.7</td><td>85.3</td><td>32.8</td><td>67.0</td></tr><tr><td>n-gram BPE</td><td>None</td><td>Greedy</td><td>28.2</td><td>27.8</td><td>75.9</td><td>26.4</td><td>55.0</td></tr><tr><td></td><td>FirstSpace</td><td>PathPieceL</td><td>32.8</td><td>28.5</td><td>80.3</td><td>30.9</td><td>64.5</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceL</td><td>31.3</td><td>24.2</td><td>62.1</td><td>30.4</td><td>55.3</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>PathPieceL</td><td>28.9</td><td>23.6</td><td>66.8</td><td>33.8</td><td>59.0</td></tr><tr><td>n-gram</td><td>None</td><td>PathPieceL</td><td>21.5</td><td>24.9</td><td>51.8</td><td>28.9</td><td>51.7</td></tr><tr><td>random</td><td></td><td></td><td></td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>50.0</td></tr></table>

Table 7: 350M parameter model, 40,960 token vocabulary, accuracy (%) on remaining 5 tasks

<table><tr><td>Vocab Constr</td><td>Init Voc</td><td>Pre-tok</td><td>Segment</td><td>Avg</td><td>arc_easy</td><td>copa</td><td>mktg</td><td>mathqa</td><td>piqa</td></tr><tr><td rowspan="3">BPE</td><td></td><td>FirstSpace</td><td>Merge</td><td>48.1</td><td>52.3</td><td>65.0</td><td>31.6</td><td>23.7</td><td>65.7</td></tr><tr><td></td><td>FirstSpace</td><td>Greedy</td><td>49.5</td><td>53.9</td><td>72.0</td><td>31.6</td><td>24.2</td><td>68.4</td></tr><tr><td></td><td>FirstSpace</td><td>PathPieceL</td><td>47.2</td><td>48.6</td><td>69.0</td><td>26.9</td><td>22.8</td><td>63.1</td></tr><tr><td rowspan="3">Unigram</td><td></td><td>FirstSpace</td><td>Likelihood</td><td>48.8</td><td>52.3</td><td>69.0</td><td>35.0</td><td>23.9</td><td>66.1</td></tr><tr><td></td><td>FirstSpace</td><td>Greedy</td><td>48.6</td><td>51.6</td><td>68.0</td><td>32.1</td><td>24.4</td><td>65.7</td></tr><tr><td></td><td>FirstSpace</td><td>PathPieceL</td><td>44.0</td><td>39.4</td><td>57.0</td><td>30.3</td><td>23.3</td><td>61.2</td></tr><tr><td colspan="2">WordPiece</td><td>FirstSpace</td><td>Greedy</td><td>48.8</td><td>52.6</td><td>68.0</td><td>28.2</td><td>23.5</td><td>66.2</td></tr><tr><td rowspan="4">SaGe</td><td>BPE</td><td>FirstSpace</td><td>Greedy</td><td>48.8</td><td>51.9</td><td>71.0</td><td>29.9</td><td>22.6</td><td>65.5</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>Greedy</td><td>47.2</td><td>46.6</td><td>67.0</td><td>31.2</td><td>22.7</td><td>63.4</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>Greedy</td><td>48.0</td><td>49.7</td><td>66.0</td><td>31.6</td><td>21.6</td><td>65.7</td></tr><tr><td>Unigram</td><td>FirstSpace</td><td>Greedy</td><td>47.8</td><td>49.7</td><td>68.0</td><td>29.9</td><td>23.5</td><td>64.6</td></tr><tr><td rowspan="4">PathPieceL</td><td>BPE</td><td>FirstSpace</td><td>PathPieceL</td><td>49.4</td><td>51.9</td><td>69.0</td><td>29.9</td><td>24.5</td><td>66.6</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>PathPieceL</td><td>43.9</td><td>42.4</td><td>56.0</td><td>28.6</td><td>23.8</td><td>60.3</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceL</td><td>45.0</td><td>44.5</td><td>59.0</td><td>28.2</td><td>22.3</td><td>59.5</td></tr><tr><td>Unigram</td><td>FirstSpace</td><td>PathPieceL</td><td>48.4</td><td>51.4</td><td>67.0</td><td>29.5</td><td>24.7</td><td>65.2</td></tr><tr><td rowspan="3">PathPieceR</td><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceR</td><td>45.5</td><td>46.0</td><td>62.0</td><td>25.6</td><td>22.1</td><td>61.6</td></tr><tr><td>n-gram</td><td>None</td><td>PathPieceR</td><td>42.2</td><td>42.6</td><td>64.0</td><td>22.2</td><td>22.4</td><td>60.9</td></tr><tr><td>n-gram</td><td>SpaceDigit</td><td>PathPieceR</td><td>47.3</td><td>48.7</td><td>68.0</td><td>34.2</td><td>21.9</td><td>65.1</td></tr><tr><td>random</td><td></td><td></td><td></td><td>32.0</td><td>25.0</td><td>50.0</td><td>25.0</td><td>20.0</td><td>50.0</td></tr></table>

Table 8: 350M parameter model, 49,152 token vocabulary, accuracy (%) on average and initial 5 tasks

<table><tr><td>Vocab Constr</td><td>Init Voc</td><td>Pre-tok</td><td>Segment</td><td>qa4mre</td><td>race</td><td>sciq</td><td>sociology</td><td>wsc273</td></tr><tr><td rowspan="3">BPE</td><td></td><td>FirstSpace</td><td>Merge</td><td>28.9</td><td>31.0</td><td>87.3</td><td>28.9</td><td>67.0</td></tr><tr><td></td><td>FirstSpace</td><td>Greedy</td><td>29.6</td><td>31.2</td><td>88.4</td><td>29.4</td><td>66.3</td></tr><tr><td></td><td>FirstSpace</td><td>PathPieceL</td><td>31.0</td><td>30.7</td><td>85.4</td><td>31.8</td><td>63.0</td></tr><tr><td rowspan="3">Unigram</td><td></td><td>FirstSpace</td><td>Likelihood</td><td>27.5</td><td>30.3</td><td>89.1</td><td>28.9</td><td>65.9</td></tr><tr><td></td><td>FirstSpace</td><td>Greedy</td><td>32.4</td><td>29.5</td><td>86.7</td><td>32.3</td><td>63.7</td></tr><tr><td></td><td>FirstSpace</td><td>PathPieceL</td><td>33.1</td><td>26.0</td><td>74.5</td><td>27.9</td><td>67.0</td></tr><tr><td>WordPiece</td><td></td><td>FirstSpace</td><td>Greedy</td><td>29.2</td><td>31.1</td><td>88.0</td><td>34.3</td><td>66.7</td></tr><tr><td rowspan="4">SaGe</td><td>BPE</td><td>FirstSpace</td><td>Greedy</td><td>29.6</td><td>31.2</td><td>87.5</td><td>32.3</td><td>65.9</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>Greedy</td><td>29.2</td><td>28.8</td><td>86.4</td><td>34.3</td><td>61.9</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>Greedy</td><td>28.8</td><td>30.2</td><td>87.5</td><td>33.8</td><td>64.5</td></tr><tr><td>Unigram</td><td>FirstSpace</td><td>Greedy</td><td>28.9</td><td>31.4</td><td>87.0</td><td>29.9</td><td>65.6</td></tr><tr><td rowspan="4">PathPieceL</td><td>BPE</td><td>FirstSpace</td><td>PathPieceL</td><td>31.0</td><td>31.4</td><td>87.5</td><td>31.3</td><td>70.7</td></tr><tr><td>n-gram</td><td>FirstSpace</td><td>PathPieceL</td><td>27.5</td><td>26.7</td><td>80.8</td><td>32.3</td><td>60.8</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceL</td><td>28.9</td><td>30.0</td><td>80.6</td><td>35.8</td><td>61.2</td></tr><tr><td>Unigram</td><td>FirstSpace</td><td>PathPieceL</td><td>29.2</td><td>30.5</td><td>88.5</td><td>32.8</td><td>65.6</td></tr><tr><td rowspan="3">PathPieceR</td><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceR</td><td>29.6</td><td>29.5</td><td>82.8</td><td>30.9</td><td>64.5</td></tr><tr><td>n-gram</td><td>None</td><td>PathPieceR</td><td>25.7</td><td>27.5</td><td>72.5</td><td>27.4</td><td>57.1</td></tr><tr><td>n-gram</td><td>SpaceDigit</td><td>PathPieceR</td><td>27.5</td><td>28.7</td><td>84.0</td><td>28.9</td><td>66.3</td></tr><tr><td>Random</td><td></td><td></td><td></td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>50.0</td></tr></table>

Table 9: 350M parameter model, 49,152 token vocabulary, accuracy (%) on remaining 5 tasks

<table><tr><td>Rank</td><td>Vocab Constr</td><td>Init Voc</td><td>Pre-tok</td><td>Segment</td><td>Overall avg</td><td>32,768 avg</td><td>40,960 avg</td><td>49,152 avg</td></tr><tr><td>1</td><td>PathPieceL</td><td>BPE</td><td>FirstSpace</td><td>PathPieceL</td><td>49.4</td><td>49.3</td><td>49.4</td><td>49.4</td></tr><tr><td>2</td><td>Unigram</td><td></td><td>FirstSpace</td><td>Likelihood</td><td>49.0</td><td>49.2</td><td>49.1</td><td>48.8</td></tr><tr><td>3</td><td>BPE</td><td></td><td>FirstSpace</td><td>Merge</td><td>49.0</td><td>48.8</td><td>50.0</td><td>48.1</td></tr><tr><td>4</td><td>BPE</td><td></td><td>FirstSpace</td><td>Greedy</td><td>49.0</td><td>48.3</td><td>49.1</td><td>49.5</td></tr><tr><td>5</td><td>WordPiece</td><td></td><td>FirstSpace</td><td>Greedy</td><td>48.8</td><td>48.5</td><td>49.1</td><td>48.8</td></tr><tr><td>6</td><td>SaGe</td><td>BPE</td><td>FirstSpace</td><td>Greedy</td><td>48.6</td><td>47.9</td><td>49.2</td><td>48.8</td></tr><tr><td>7</td><td>Unigram</td><td></td><td>FirstSpace</td><td>Greedy</td><td>48.3</td><td>47.9</td><td>48.5</td><td>48.6</td></tr><tr><td>8</td><td>SaGe</td><td>n-gram</td><td>FirstSpace</td><td>Greedy</td><td>48.0</td><td>47.5</td><td>48.5</td><td>48.0</td></tr><tr><td>9</td><td>PathPieceL</td><td>Unigram</td><td>FirstSpace</td><td>PathPieceL</td><td>48.0</td><td>46.9</td><td>48.5</td><td>48.4</td></tr><tr><td>10</td><td>SaGe</td><td>Unigram</td><td>FirstSpace</td><td>Greedy</td><td>47.7</td><td>48.4</td><td>46.9</td><td>47.8</td></tr><tr><td>11</td><td>SaGe</td><td>n-gram</td><td>FirstSpDigit</td><td>Greedy</td><td>47.5</td><td>48.4</td><td>46.9</td><td>47.2</td></tr><tr><td>12</td><td>PathPieceR</td><td>n-gram</td><td>SpaceDigit</td><td>PathPieceR</td><td>46.7</td><td>47.5</td><td>45.4</td><td>47.3</td></tr><tr><td>13</td><td>BPE</td><td></td><td>FirstSpace</td><td>PathPieceL</td><td>46.5</td><td>45.6</td><td>46.7</td><td>47.2</td></tr><tr><td>14</td><td>PathPieceR</td><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceR</td><td>45.5</td><td>45.3</td><td>45.8</td><td>45.5</td></tr><tr><td>15</td><td>PathPieceL</td><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceL</td><td>44.8</td><td>44.6</td><td>44.9</td><td>45.0</td></tr><tr><td>16</td><td>PathPieceL</td><td>n-gram</td><td>FirstSpace</td><td>PathPieceL</td><td>44.7</td><td>44.8</td><td>45.5</td><td>43.9</td></tr><tr><td>17</td><td>Unigram</td><td></td><td>FirstSpace</td><td>PathPieceL</td><td>43.6</td><td>43.6</td><td>43.1</td><td>44.0</td></tr><tr><td>18</td><td>PathPieceR</td><td>n-gram</td><td>None</td><td>PathPieceR</td><td>43.2</td><td>43.5</td><td>44.0</td><td>42.2</td></tr><tr><td colspan="2">Random</td><td></td><td></td><td></td><td>32.0</td><td>32.0</td><td>32.0</td><td>32.0</td></tr></table>

Table 10: Summary of 350M parameter model downstream accuracy (%), sorted by rank

<table><tr><td>Rank</td><td>Vocab Size</td><td>Avg Acc</td><td>CTC</td><td>Eff α=1.5</td><td>Eff α=2</td><td>Eff α=2.5</td><td>Eff α=3</td><td>Eff α=3.5</td></tr><tr><td>1</td><td>32,768</td><td>49.3</td><td>1.48</td><td>0.604</td><td>0.516</td><td>0.469</td><td>0.441</td><td>0.422</td></tr><tr><td>1</td><td>40,960</td><td>49.4</td><td>1.46</td><td>0.589</td><td>0.503</td><td>0.457</td><td>0.429</td><td>0.411</td></tr><tr><td>1</td><td>49,152</td><td>49.4</td><td>1.44</td><td>0.578</td><td>0.492</td><td>0.448</td><td>0.420</td><td>0.402</td></tr><tr><td>22</td><td>32,768</td><td>49.2</td><td>1.79</td><td>0.461</td><td>0.371</td><td>0.324</td><td>0.295</td><td>0.277</td></tr><tr><td></td><td>40,960</td><td>49.1</td><td>1.77</td><td>0.451</td><td>0.362</td><td>0.316</td><td>0.289</td><td>0.271</td></tr><tr><td>2</td><td>49,152</td><td>48.8</td><td>1.76</td><td>0.444</td><td>0.356</td><td>0.311</td><td>0.284</td><td>0.266</td></tr><tr><td>333</td><td>32,768</td><td>48.8</td><td>1.52</td><td>0.594</td><td>0.505</td><td>0.459</td><td>0.431</td><td>0.414</td></tr><tr><td></td><td>40,960</td><td>50.0</td><td>1.49</td><td>0.579</td><td>0.491</td><td>0.446</td><td>0.420</td><td>0.403</td></tr><tr><td></td><td>49,152</td><td>48.1</td><td>1.47</td><td>0.567</td><td>0.481</td><td>0.437</td><td>0.411</td><td>0.394</td></tr><tr><td>4</td><td>32,768</td><td>48.3</td><td>1.50</td><td>0.605</td><td>0.517</td><td>0.471</td><td>0.442</td><td>0.423</td></tr><tr><td>4</td><td>40,960</td><td>49.1</td><td>1.48</td><td>0.590</td><td>0.504</td><td>0.458</td><td>0.430</td><td>0.412</td></tr><tr><td>4</td><td>49,152</td><td>49.5</td><td>1.46</td><td>0.579</td><td>0.494</td><td>0.449</td><td>0.421</td><td>0.403</td></tr><tr><td>5</td><td>32,768</td><td>48.5</td><td>1.54</td><td>0.598</td><td>0.507</td><td>0.461</td><td>0.433</td><td>0.415</td></tr><tr><td>5</td><td>40,960</td><td>49.1</td><td>1.51</td><td>0.583</td><td>0.494</td><td>0.448</td><td>0.421</td><td>0.404</td></tr><tr><td>5</td><td>49,152</td><td>48.8</td><td>1.49</td><td>0.571</td><td>0.483</td><td>0.439</td><td>0.412</td><td>0.396</td></tr><tr><td>6</td><td>32,768</td><td>47.9</td><td>1.78</td><td>0.545</td><td>0.466</td><td>0.422</td><td>0.396</td><td>0.378</td></tr><tr><td>6</td><td>40,960</td><td>49.2</td><td>1.76</td><td>0.533</td><td>0.455</td><td>0.413</td><td>0.387</td><td>0.369</td></tr><tr><td>6</td><td>49,152</td><td>48.7</td><td>1.75</td><td>0.523</td><td>0.447</td><td>0.405</td><td>0.379</td><td>0.362</td></tr><tr><td>7</td><td>32,768</td><td>47.9</td><td>1.81</td><td>0.510</td><td>0.431</td><td>0.387</td><td>0.359</td><td>0.340</td></tr><tr><td>7</td><td>40,960</td><td>48.5</td><td>1.79</td><td>0.500</td><td>0.423</td><td>0.381</td><td>0.354</td><td>0.335</td></tr><tr><td>7</td><td>49,152</td><td>48.6</td><td>1.77</td><td>0.493</td><td>0.416</td><td>0.375</td><td>0.348</td><td>0.330</td></tr><tr><td>8</td><td>32,768</td><td>47.5</td><td>1.63</td><td>0.629</td><td>0.536</td><td>0.482</td><td>0.447</td><td>0.424</td></tr><tr><td>88</td><td>40,960</td><td>48.5</td><td>1.62</td><td>0.615</td><td>0.524</td><td>0.470</td><td>0.437</td><td>0.415</td></tr><tr><td></td><td>49,152</td><td>48.0</td><td>1.62</td><td>0.605</td><td>0.515</td><td>0.462</td><td>0.429</td><td>0.407</td></tr><tr><td>9</td><td>32,768</td><td>46.9</td><td>1.74</td><td>0.508</td><td>0.419</td><td>0.372</td><td>0.343</td><td>0.323</td></tr><tr><td>9</td><td>40,960</td><td>48.5</td><td>1.72</td><td>0.491</td><td>0.403</td><td>0.356</td><td>0.328</td><td>0.309</td></tr><tr><td>9</td><td>49,152</td><td>48.4</td><td>1.72</td><td>0.477</td><td>0.389</td><td>0.343</td><td>0.315</td><td>0.296</td></tr><tr><td>10</td><td>32,768</td><td>48.4</td><td>2.02</td><td>0.485</td><td>0.409</td><td>0.366</td><td>0.339</td><td>0.320</td></tr><tr><td>10</td><td>40,960</td><td>46.9</td><td>2.01</td><td>0.474</td><td>0.401</td><td>0.358</td><td>0.331</td><td>0.313</td></tr><tr><td>10</td><td>49,152</td><td>47.8</td><td>2.01</td><td>0.466</td><td>0.393</td><td>0.352</td><td>0.325</td><td>0.307</td></tr><tr><td>11</td><td>32,768</td><td>48.4</td><td>1.77</td><td>0.587</td><td>0.512</td><td>0.470</td><td>0.443</td><td>0.425</td></tr><tr><td>11</td><td>40,960</td><td>46.9</td><td>1.76</td><td>0.575</td><td>0.501</td><td>0.460</td><td>0.433</td><td>0.415</td></tr><tr><td>11</td><td>49,152</td><td>47.2</td><td>1.76</td><td>0.565</td><td>0.492</td><td>0.452</td><td>0.426</td><td>0.408</td></tr><tr><td>12</td><td>32,768</td><td>47.5</td><td>2.33</td><td>0.236</td><td>0.164</td><td>0.138</td><td>0.124</td><td>0.116</td></tr><tr><td>12</td><td>40,960</td><td>45.4</td><td>2.30</td><td>0.228</td><td>0.159</td><td>0.133</td><td>0.120</td><td>0.112</td></tr><tr><td>12</td><td>49,152</td><td>47.3</td><td>2.29</td><td>0.223</td><td>0.155</td><td>0.130</td><td>0.117</td><td>0.109</td></tr><tr><td>13</td><td>32,768</td><td>45.6</td><td>1.50</td><td>0.606</td><td>0.518</td><td>0.470</td><td>0.442</td><td>0.423</td></tr><tr><td>13</td><td>40,960</td><td>46.7</td><td>1.47</td><td>0.591</td><td>0.504</td><td>0.458</td><td>0.430</td><td>0.412</td></tr><tr><td>13</td><td>49,152</td><td>47.2</td><td>1.45</td><td>0.579</td><td>0.494</td><td>0.449</td><td>0.421</td><td>0.403</td></tr><tr><td>14</td><td>32,768</td><td>45.3</td><td>1.46</td><td>0.616</td><td>0.532</td><td>0.490</td><td>0.465</td><td>0.448</td></tr><tr><td>14</td><td>40,960</td><td>45.8</td><td>1.43</td><td>0.602</td><td>0.519</td><td>0.478</td><td>0.453</td><td>0.437</td></tr><tr><td>14</td><td>49,152</td><td>45.5</td><td>1.42</td><td>0.591</td><td>0.508</td><td>0.468</td><td>0.444</td><td>0.428</td></tr><tr><td>15</td><td>32,768</td><td>44.6</td><td>1.47</td><td>0.620</td><td>0.533</td><td>0.490</td><td>0.464</td><td>0.447</td></tr><tr><td>15</td><td>40,960</td><td>44.9</td><td>1.44</td><td>0.605 0.594</td><td>0.520 0.509</td><td>0.478 0.468</td><td>0.453 0.443</td><td>0.436 0.427</td></tr><tr><td>15 16</td><td>49,152 32,768</td><td>45.0 44.8</td><td>1.42 1.36</td></table>

Table 11: Average Accuracy (%) vs. Corpus Token Count (CTC, in billions) by vocabulary size, for Figure 3. Also includes the corresponding Rényi efficiency (Zouhar et al., 2023a) for various orders α.

<table><tr><td>Vocab Constr</td><td>Init Voc</td><td>Pre-tok</td><td>Segment</td><td>Avg</td><td>arc_easy</td><td>copa</td><td>mktg</td><td>mathqa</td><td>piqa</td></tr><tr><td>BPE</td><td></td><td>FirstSpace</td><td>Merge</td><td>53.1</td><td>62.0</td><td>77.0</td><td>32.1</td><td>25.0</td><td>71.1</td></tr><tr><td>Unigram</td><td></td><td>FirstSpace</td><td>Likelihood</td><td>52.4</td><td>60.6</td><td>71.0</td><td>30.3</td><td>25.2</td><td>71.0</td></tr><tr><td rowspan="2">SaGe</td><td>BPE</td><td>FirstSpace</td><td>Greedy</td><td>52.2</td><td>62.0</td><td>72.0</td><td>27.4</td><td>24.5</td><td>71.6</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>Greedy</td><td>50.7</td><td>60.3</td><td>71.0</td><td>28.6</td><td>22.8</td><td>69.4</td></tr><tr><td rowspan="3">PathPieceL</td><td>BPE</td><td>FirstSpace</td><td>PathPieceL</td><td>49.2</td><td>57.4</td><td>66.0</td><td>27.8</td><td>24.3</td><td>65.9</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceL</td><td>47.6</td><td>49.7</td><td>67.0</td><td>24.8</td><td>23.4</td><td>63.2</td></tr><tr><td>n-gram</td><td>SpaceDigit</td><td>PathPieceL</td><td>46.3</td><td>51.1</td><td>59.0</td><td>28.6</td><td>23.3</td><td>63.8</td></tr><tr><td>Random</td><td></td><td></td><td></td><td>32.0</td><td>25.0</td><td>50.0</td><td>25.0</td><td>20.0</td><td>50.0</td></tr></table>

Table 12: 1.3B parameter model, 40,960 token vocabulary, accuracy (%) on average and initial 5 tasks

<table><tr><td>Vocab Constr</td><td>Init Voc</td><td>Pre-tok</td><td>Segment</td><td>qa4mre</td><td>race</td><td>sciq</td><td>sociology</td><td>wsc273</td></tr><tr><td>BPE</td><td></td><td>FirstSpace</td><td>Merge</td><td>32.4</td><td>34.9</td><td>93.0</td><td>26.4</td><td>76.9</td></tr><tr><td>Unigram</td><td></td><td>FirstSpace</td><td>Likelihood</td><td>37.7</td><td>33.0</td><td>91.8</td><td>28.9</td><td>74.4</td></tr><tr><td rowspan="2">SaGe</td><td>BPE</td><td>FirstSpace</td><td>Greedy</td><td>34.9</td><td>34.8</td><td>92.5</td><td>25.9</td><td>76.2</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>Greedy</td><td>29.9</td><td>32.9</td><td>91.5</td><td>29.4</td><td>71.1</td></tr><tr><td rowspan="3">PathPieceL</td><td>BPE</td><td>FirstSpace</td><td>PathPieceL</td><td>31.0</td><td>33.3</td><td>89.4</td><td>26.4</td><td>70.7</td></tr><tr><td>n-gram</td><td>FirstSpDigit</td><td>PathPieceL</td><td>31.0</td><td>31.6</td><td>86.1</td><td>29.4</td><td>70.0</td></tr><tr><td>n-gram</td><td>SpaceDigit</td><td>PathPieceL</td><td>28.9</td><td>31.3</td><td>87.1</td><td>22.4</td><td>67.0</td></tr><tr><td>Random</td><td></td><td></td><td></td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>50.0</td></tr></table>

Table 13: 1.3B parameter model, 40,960 token vocabulary, accuracy (%) on remaining 5 tasks

<table><tr><td>Voc Con</td><td>Init V</td><td>Pre-tok</td><td>Seg</td><td>350M avg</td><td>350M rnk</td><td>1.3B avg</td><td>1.3B rnk</td><td>2.4B avg</td><td>2.4B rnk</td></tr><tr><td>BPE</td><td></td><td>FirSp</td><td>Merge</td><td>50.0</td><td>1</td><td>53.1</td><td>1</td><td>54.2</td><td>3</td></tr><tr><td>PathPL</td><td>BPE</td><td>FirSp</td><td>PathPL</td><td>49.4</td><td>3</td><td>49.2</td><td>5</td><td>52.7</td><td>4</td></tr><tr><td>PathPL</td><td>n-gram</td><td>FirSpD</td><td>PathPL</td><td>44.9</td><td>6</td><td>47.6</td><td>6</td><td></td><td></td></tr><tr><td>SaGe</td><td>BPE</td><td>FirSp</td><td>Greedy</td><td>49.2</td><td>2</td><td>52.2</td><td>3</td><td>55.0</td><td>1</td></tr><tr><td>SaGe</td><td>n-gram</td><td>FirSpD</td><td>Greedy</td><td>46.9</td><td>5</td><td>50.7</td><td>4</td><td></td><td></td></tr><tr><td>Unigram</td><td></td><td>FirSp</td><td>Likeli</td><td>49.1</td><td>4</td><td>52.4</td><td>2</td><td>54.7</td><td>2</td></tr></table>

Table 14: Downstream accuracy (%) of 10 tasks with vocab size 40,960, for various model sizes