#!/usr/bin/env python3

import random
import json
from tracemalloc import stop
from urllib.request import urlopen

import openai


class OpenAIModel:
    def __init__(self, api_key, temperature=0.0, top_p=0.7, best_of=1):
        openai.api_key = api_key
        self.model = "text-davinci-002"
        self.temperature = temperature
        self.top_p = top_p
        self.best_of = best_of
        # for gpt series of models
        url = "https://huggingface.co/gpt2/resolve/main/vocab.json"
        response = urlopen(url)
        self.token_idx = json.loads(response.read())
        self.token_idx = {
            s.replace("\u0120", " "): i for s, i in self.token_idx.items()
        }

    def vocabulary(self):
        # sort keys by value, then return the keys
        vocab = sorted(self.token_idx.keys(), key=lambda k: self.token_idx[k])
        return vocab

    def predict_token(self, prompt, max_tokens=256, stop=None):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            logprobs=5,
            temperature=self.temperature,
            top_p=self.top_p,
            best_of=self.best_of,
            max_tokens=max_tokens,
            stop=stop,
        )
        response_dict = response.choices[0].logprobs.top_logprobs[0]
        text = response.choices[0].text
        return text
