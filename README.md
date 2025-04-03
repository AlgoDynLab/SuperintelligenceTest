# Code for paper "An Agnostic Test for Narrow, General, and Super Intelligence Based On the Principles of Recursive Compression and Algorithmic Probability"

Code for paper [link](https://www.researchgate.net/publication/381114144_A_Test_of_Intelligence_for_Automated_Programming_Languages)

# The score

In general, when prompting an Large Language Model (LLM) to create a formula to reproduce a given sequence, when the result is correct, the following formula types are possible:

- Prints (the formular simply reproduced the target sequence without any attempt to encode or express it logically. This response type reflects a failure to abstract or deduce any underlying pattern, simply outputting the sequence as is).
- Ordinal (the formula provided a mapping based on the indices where '1's occur in the sequence. This response reflects an attempt by the model to analyse and map some logical structure to the sequence, making it more valuable than simply reproducing it verbatim)
- Non-Both (these responses avoided both simple reproduction and ordinal mapping, reflecting a more sophisticated approach to understanding and encoding the pattern. Such responses are the most valuable as they imply a deeper analysis and potentially creative logic to represent the sequence)

On the other hand, a fourth possible outcome is the case where the LLM was unable to reproduce the sequence correctly. Thus, the SuperARC-Seq score $\varphi$ can be calculated as:

```math
\varphi = \delta_1\rho_1 +\frac{\delta_2\rho_2}{2}+\frac{\delta_3\rho_3}{4} - \rho_4.
```

where:

- $\rho_1$ is the percentage of Correct \& Non-Prints \& Non-Ordinal results;
- $\rho_2$ is the percentage of Correct \& Ordinal results;
- $\rho_3$ is the percentage of Correct \& Prints results;
- $\rho_4$ is the percentage of Incorrect results;
- $\delta_i$, for $i = 1,2,3$ is a weighting factor determining how well the LLM compressed the original sequence and ranges from 0 (no compression) to 1 (perfect compression in an algorithmic sense). This weighting factor is calculated by using the principles of Algorithmic Information Theory.

It can be seen that $\sum \rho_i = 1$ and that $\varphi \in [-1,1]$ encompasses different behaviours. For example, $\varphi \in [0,0.25]$ if only print-type models are outputted. Also, $\varphi \in [0,0.5]$ if only ordinal-like formulas are created. Finally, $\varphi \in [0,1]$ in cases where the LLMs create formulas that are always correct, do not copy nor create ordinal mappings. The ranges will be populated with varying compression levels corresponding to the algorithms obtained. Overall, if the score is -1, all the formulas were wrong. If it is 0, it can represent the case where half the outputs were correct and half wrong, with the formulas produced with highest compression levels. So, in a regular half and half case, since compression will not be optimal, the test score is negative.

# Leaderboard for SuperARC-Seq

![Ranking](rankingSuperARC.png)


| Model              | $\rho_1$ | $\rho_2$ | $\rho_3$ | $\rho_4$ | $\delta_1$ | $\delta_2$ | $\delta_3$ | $\varphi$ |
|--------------------|------------|------------|------------|------------|--------------|--------------|--------------|-------------|
| ASI (AIXI/BDM/CTM) | 1.000      | 0.000      | 0.000      | 0.000      | 1.000        | 0.000        | 1.000        | 1.000       |
| mistral            | 0.670      | 0.000      | 1.000      | 0.180      | 1.553        | 0.000        | 0.116        | 0.100       |
| gpt\_4o\_mini      | 0.390      | 0.000      | 1.460      | 0.000      | 1.551        | 0.000        | 0.160        | 0.090       |
| claude\_3.7        | 0.400      | 0.810      | 0.320      | 0.320      | 1.060        | 0.032        | 0.313        | 0.067       |
| chatgpt\_4.5       | 0.440      | 1.000      | 0.000      | 0.410      | 1.008        | 0.030        | 0.000        | 0.063       |
| cursor\_small      | 0.490      | 0.000      | 1.000      | 0.360      | 0.691        | 0.000        | 0.141        | 0.061       |
| deepseek           | 0.590      | 0.000      | 1.000      | 0.260      | 0.345        | 0.000        | 0.116        | 0.059       |
| qwen               | 0.590      | 0.000      | 1.000      | 0.260      | 0.204        | 0.000        | 0.116        | 0.056       |
| o1\_mini           | 0.520      | 0.640      | 0.140      | 0.550      | 0.595        | 0.058        | 0.224        | 0.051       |
| gemini             | 0.410      | 0.000      | 1.000      | 0.440      | 0.255        | 0.000        | 0.141        | 0.047       |
| gpt\_4o            | 0.410      | 0.000      | 0.360      | 1.080      | 1.384        | 0.000        | 0.313        | 0.039       |
| grok\_3            | 0.480      | 0.020      | 0.000      | 1.350      | 1.027        | 0.424        | 0.000        | 0.023       |
| meta               | 0.670      | 0.000      | 0.000      | 1.180      | 0.285        | 0.000        | 0.000        | 0.017       |
| claude\_3.5        | 0.140      | 0.460      | 0.000      | 1.250      | 0.129        | 0.116        | 0.000        | 0.007       |
| o1\_preview        | 0.010      | 0.290      | 0.140      | 1.410      | 2.493        | 0.048        | 0.224        | 0.000       |

Both the plot and the Leaderboard table show how most frontier models are close to each other in their performance under this test and far from Artificial General Intelligence (AGI) or Artificial Super Intelligence (ASI) goals according to this test. ASI would be able to distinguish simpler from complex sequences and generate predictive models for each accordingly, as AIXI or Coding Theorem Method/Block Decomposition Method (CTM/BDM) would do as instantiations of universal AI hence ASI. Today, LLMs only produce or retrieve models for sequences that were seen and found in their original training sets, given that increasing the sequences' lengths impacts the LLM performance in identifying the sequence, hence indicating sequences are not recognised from first principles but from simplistic pattern matching.


### Description of the files

- `00_asking_OpenAI.ipynb`: Make questions to OpenAI API.
- `01_auto_runing_codeAnswers.ipynb`: Execute code given by LLMs according to programming language.
- `02_correctness_test.ipynb`: Measure and plot the correctness of the answers.
- `03_Timeseries_LLM_experiments.ipynb`: Runs experiments with TimeGPT and Chronos.
- `04_Lag_Llama_experimentation.ipynb`: Runs experiments with Lag-Llama. Code executed in Google Colab.
- `05_PLOTS_paper.ipynb`: Plots for the paper.
- `06_BDM_metrics.ipynb`: Compute BDM metrics.
- `07_formulas_experimentation.ipynb`: Free form experiments with formulae.
- `08_random_binary_experiments.ipynb`: Process of random binary sequences.
- `09_random_bin_seq_process.ipynb`: Process results in 08.
- `10_formulae_evaluation.py`: Formulae experimentation.
