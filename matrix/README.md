
# Introduction
This directory contains code to run the Matrix Game experiments. 

# Setup
All code is developed with Python 3.9.13. We recommend creating a conda environment based on the `env.yml` file:
```conda env create -f env.yml```

# Prompts Used in the Paper
The exact prompts used in the paper can be generated by `zs_matrix_prompts.py`. You can print out prompts by running the following:
```python ultimatum_prompts.py --game [battle, chicken, stag, prisoners] --objective [welfare, equality, rawlsian, pareto]```
Add the `--scrambled` flag to generate prompts that have scrambled joint outcomes. 
```python ultimatum_prompts.py --game [battle, chicken, stag, prisoners] --objective [welfare, equality, rawlsian, pareto] --scrambled```
To generate the blank prompts used as our baseline, run the following:
```python ultimatum_prompts.py --game [battle, chicken, stag, prisoners] --blank```

# Training Models
We generate our prompts in a batched manner, ask the LLM to annotate our prompts, and save the LLM's answers. We then use those answers as reward signals when training an RL agent. The LLM's responses can be found in the `zs_lm_responses` and `zs_blank_lm_responses` for our regular zero-shot prompts and the baseline zero-shot prompts respectively. 

In order to train an RL agent, and evaluate the RL agent's trained behavior, run the following:
```python matrix.py --reward_model [gpt3, rl, blank]```
We manually parse the answers from the LLM and save them in `matrix.py` (e.g., the rewards according to GPT3 are on lines 33-50).