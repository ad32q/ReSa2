This is an early project and its corresponding code, which is for anonymization testing.  Given its initial stage, the code is far from its full - fledged state.  Therefore, it will be gradually supplemented and improved to achieve better performance and functionality. The code in this repository can run in some environments, but environment errors may occur in other environments.

This is the source code for the paper "ReSa2: A Two - Stage Retrieval - Sampling Algorithm for Negative Sampling in Dense Retrieval".  Tanonymization testing process and gain a deeper understanding of how it handles negative sampling to enhance the retrieval performance.


\documentclass{article}
\usepackage{graphicx} % Required for inserting images
\usepackage{booktabs}
\usepackage{tabularx}
% \title{appendix}
\usepackage{multirow}

\begin{document}

\newpage
\appendix


\section{Instruction}
In this experiment, we replicated a series of methods and made certain adjustments to their parameters and sequences according to our evaluation criteria to achieve their best performance.

Due to the limited length of the main text, the settings and descriptions of each method are as follows.
\subsection{TopK}
For the direct sampling of TopK hard negative samples, we randomly sampled from the hard negative samples ranked among the top 200 in the ANN. This is also the parameter setting for ANCE.  We also compared sampling with the optimal parameter of our method, which is 75. However, when the value of K was 200, the best results were obtained, and this value was adopted in the reporting of our paper. The results of sampling with Top75 and then conducting training are as follows: mrr@10 is 33.55, recall@1000 is 94.9. The nCDG@10 value in TREC-2019 (62.70) and the nCDG@10 value in TREC-2020 (59.42) are both inferior to the results of Top200.
\begin{table}[h]
    \centering
    \caption{Performance metrics for DPR with different top-K settings across MS-MARCO, TREC-19, and TREC-20.}
    \begin{tabularx}{\textwidth}{l *{2}{X}}
        \toprule
        & \textbf{Top75} & \textbf{Top200} \\
        \midrule
        \textbf{MRR@10} & 33.6 & 34.2 \\
        \textbf{recall@50} & 78.6 & 81.2 \\
        \textbf{recall@100} & 84.2 & 86.9 \\
        \textbf{recall@1000} & 94.9 & 96.2 \\
        \textbf{nDCG@10 (TREC-19)} & 59.4 & 67.3 \\
        \textbf{nDCG@10 (TREC-20)} & 62.7 & 66.9 \\
        \bottomrule
    \end{tabularx}
    \label{tab:performance_metrics_swapped}
\end{table}
\subsection{TriSampler}
Regarding TriSampler, there are two different interpretations of the quasi - triangle principle mentioned in the original method. We replicated the method according to these two different interpretations and adopted the approach that can effectively screen out samples. In this approach, we denote it as T1. The setting of $\theta$ in this method is slightly different from the other one.

We denote $\mathbf{V}_{d^+} - \mathbf{V}_q$ as $\mathbf{V}_1$ and $\mathbf{V}_{d^-} - \mathbf{V}_q$ as $\mathbf{V}_2$. This can be expressed by the following formulas:
\begin{equation}
\mathbf{V}_1 = \mathbf{V}_{d^+} - \mathbf{V}_q
\end{equation}
\begin{equation}
\mathbf{V}_2 = \mathbf{V}_{d^-} - \mathbf{V}_q
\end{equation}

Next, we calculate the angle $\alpha$ between $\mathbf{V}_1$ and $\mathbf{V}_2$. According to the dot - product formula of vectors $\mathbf{a} \cdot \mathbf{b}=\|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$ (where $\theta$ is the angle between $\mathbf{a}$ and $\mathbf{b}$), we can obtain:
\begin{equation}
\alpha = \arccos\left(\frac{\mathbf{V}_1 \cdot \mathbf{V}_2}{\|\mathbf{V}_1\| \|\mathbf{V}_2\|}\right)
\end{equation}

The other approach is denoted as T2. The calculation of $\theta$ in this approach is as follows:
\begin{equation}
\theta=\left|\arccos\left(\frac{s(\mathbf{v}_q,\mathbf{v}_{d^+})}{\|\mathbf{v}_q\|\cdot\|\mathbf{v}_{d^+}\|}\right)-\arccos\left(\frac{s(\mathbf{v}_q,\mathbf{v}_{d^-})}{\|\mathbf{v}_q\|\cdot\|\mathbf{v}_{d^-}\|}\right)\right|
\end{equation}

We also solely employed these two principles to screen negative samples to examine the effectiveness. The specific results are presented in the Table 2. $\theta$ is set to 60 degrees for all cases.


\subsection{SimANS}
For SimANS, we adopted the same settings as the original method, directly extracting the final negative samples from the hard negative samples ranked among the top 100 in the ANN according to the probability distribution of the similarity score distance, and reported this in the paper. Meanwhile, through the ablation of TriSampler and our method, we found that compared with directly extracting the final samples, the probability distribution of the similarity score distance proposed by SimANS can achieve better results when further sampling after extracting the transition samples. 



\begin{table}[htbp]
\centering
\caption{Retrieval Performance Comparison on MS-MARCO Dataset}
\label{tab:marco_results}
\begin{tabular}{lccc}
\toprule
\multirow{2}{*}{Model}& \multicolumn{3}{c}{Evaluation Metrics} \\
\cmidrule(lr){2-4}
 & MRR@10 & Recall@50 & Recall@1000 \\
\midrule
T2& 34.9& 80.9& 95.9\\
T1 & 33.8 & 78.4 & 94.7 \\
Remove the second-stage retrieval& 35.1& --& 96.4\\
SimANS& 33.3 & 77.5 & 94.7 \\
\bottomrule
\end{tabular}

\vspace{5pt}
\footnotesize
\end{table}





\subsection{TREC}
Since the TREC 2019 (2020) Deep Learning Track has been closed, we were unable to obtain the entire dataset for testing. The dataset for testing that we obtained from the official website is only a part of it, because a considerable number of queries could not yield results in the qrel file. Therefore, the nCDG@10 data we measured is only used as an additional supplement, and we did not use it as a criterion for evaluating the quality of sampling in the main text. 
\end{document}
