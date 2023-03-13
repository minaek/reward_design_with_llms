# Reward Design with Language Models
The code in this repository is based on the paper [Reward Design with Language Models](https://arxiv.org/pdf/2303.00001.pdf).
This repository contains the prompts that we used for each domain as well as code to train an RL agent with an LLM in the loop using those prompts.
Each domain (Ultimatum Game, Matrix Games, DealOrNoDeal) has a separate directory and will need a seperate conda/virtual environment.
Please check out the READMEs in each directory for more information on how to run things.

# Using GPT3
We use GPT3 for our experiments. You will need to have an API key from them saved in your `~/.bashrc` or `~/.zshrc` under the variable `OPENAI_API_KEY`.
