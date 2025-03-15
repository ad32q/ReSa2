This is an early project and its corresponding code, which is for anonymization testing.  Given its initial stage, the code is far from its full - fledged state.  Therefore, it will be gradually supplemented and improved to achieve better performance and functionality. The code in this repository can run in some environments, but environment errors may occur in other environments.

This is the source code for the paper "ReSa2: A Two - Stage Retrieval - Sampling Algorithm for Negative Sampling in Dense Retrieval".  Tanonymization testing process and gain a deeper understanding of how it handles negative sampling to enhance the retrieval performance.


# Appendix

## Instruction
In this experiment, we replicated a series of methods and made certain adjustments to their parameters and sequences according to our evaluation criteria to achieve their best performance.

Due to the limited length of the main text, the settings and descriptions of each method are as follows.

## TopK
For the direct sampling of TopK hard negative samples, we randomly sampled from the hard negative samples ranked among the top 200 in the ANN. This is also the parameter setting for ANCE. We also compared sampling with the optimal parameter of our method, which is 75. However, when the value of K was 200, the best results were obtained, and this value was adopted in the reporting of our paper. The results of sampling with Top75 and then conducting training are as follows: mrr@10 is 33.55, recall@1000 is 94.9. The nCDG@10 value in TREC-2019 (62.70) and the nCDG@10 value in TREC-2020 (59.42) are both inferior to the results of Top200.

|  | **Top75** | **Top200** |
| --- | --- | --- |
| **MRR@10** | 33.6 | 34.2 |
| **recall@50** | 78.6 | 81.2 |
| **recall@100** | 84.2 | 86.9 |
| **recall@1000** | 94.9 | 96.2 |
| **nDCG@10 (TREC-19)** | 59.4 | 67.3 |
| **nDCG@10 (TREC-20)** | 62.7 | 66.9 |

## TriSampler
Regarding TriSampler, there are two different interpretations of the quasi-triangle principle mentioned in the original method. We replicated the method according to these two different interpretations and adopted the approach that can effectively screen out samples. In this approach, we denote it as T1. The setting of $\theta$ in this method is slightly different from the other one.

We denote $\mathbf{V}_{d^+} - \mathbf{V}_q$ as $\mathbf{V}_1$ and $\mathbf{V}_{d^-} - \mathbf{V}_q$ as $\mathbf{V}_2$. This can be expressed by the following formulas:

\[
\mathbf{V}_1 = \mathbf{V}_{d^+} - \mathbf{V}_q
\]

\[
\mathbf{V}_2 = \mathbf{V}_{d^-} - \mathbf{V}_q
\]

Next, we calculate the angle $\alpha$ between $\mathbf{V}_1$ and $\mathbf{V}_2$. According to the dot - product formula of vectors $\mathbf{a} \cdot \mathbf{b}=\|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$ (where $\theta$ is the angle between $\mathbf{a}$ and $\mathbf{b}$), we can obtain:

\[
\alpha = \arccos\left(\frac{\mathbf{V}_1 \cdot \mathbf{V}_2}{\|\mathbf{V}_1\| \|\mathbf{V}_2\|}\right)
\]

The other approach is denoted as T2. The calculation of $\theta$ in this approach is as follows:

\[
\theta=\left|\arccos\left(\frac{s(\mathbf{v}_q,\mathbf{v}_{d^+})}{\|\mathbf{v}_q\| \cdot \|\mathbf{v}_{d^+}\|}\right)-\arccos\left(\frac{s(\mathbf{v}_q,\mathbf{v}_{d^-})}{\|\mathbf{v}_q\| \cdot \|\mathbf{v}_{d^-}\|}\right)\right|
\]

We also solely employed these two principles to screen negative samples to examine the effectiveness. The specific results are presented in Table 2. $\theta$ is set to 60 degrees for all cases. 
## SimANS
For SimANS, we adopted the same settings as the original method, directly extracting the final negative samples from the hard negative samples ranked among the top 100 in the ANN according to the probability distribution of the similarity score distance, and reported this in the paper. Meanwhile, through the ablation of TriSampler and our method, we found that compared with directly extracting the final samples, the probability distribution of the similarity score distance proposed by SimANS can achieve better results when further sampling after extracting the transition samples.

| Model | Evaluation Metrics |  |  |
| --- | --- | --- | --- |
|  | MRR@10 | Recall@50 | Recall@1000 |
| T2 | 34.9 | 80.9 | 95.9 |
| T1 | 33.8 | 78.4 | 94.7 |
| Remove the second-stage retrieval | 35.1 | -- | 96.4 |
| SimANS | 33.3 | 77.5 | 94.7 |

## TREC
Since the TREC 2019 (2020) Deep Learning Track has been closed, we were unable to obtain the entire dataset for testing. The dataset for testing that we obtained from the official website is only a part of it, because a considerable number of queries could not yield results in the qrel file. Therefore, the nCDG@10 data we measured is only used as an additional supplement, and we did not use it as a criterion for evaluating the quality of sampling in the main text. 



This Markdown-formatted content is presented in English. Note that for the formula parts, they are just shown in text form. If you need better formula rendering effects in actual use, you may need to combine with some tools or platforms that support formula rendering (such as MathJax, etc.). 
