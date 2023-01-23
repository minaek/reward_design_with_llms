# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Various helpers.
"""

import pdb
import random
from typing import Iterable, List
import yaml
import os

import numpy as np
import torch
from models import CdaRnnModel, DialogModel
from utils.agent import RlAgent
from coarse_dialogue_acts.agent import CdaAgent
from types import SimpleNamespace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def backward_hook(grad):
    """Hook for backward pass."""
    print(grad)
    pdb.set_trace()
    return grad


def save_model(model, path_name):
    """Serializes model to a file."""
    if path_name != "":
        i = path_name.rfind("/")
        if not os.path.exists(path_name[:i]):
            os.makedirs(path_name[:i])
            print(f"Created directory {path_name[:i]}")
        with open(path_name, "wb") as f:
            torch.save(model, f)
    else:
        print("Path does not exist.")
        raise ValueError


def load_model(file_name: str, cuda: bool = True):
    # If on CPU, explicitly map weights to CPU.
    # Otherwise, use default mapping (hence, None)
    map_location = "cpu" if not cuda else None
    device = "cuda" if cuda else "cpu"
    """Reads model from a file."""
    with open(file_name, "rb") as f:
        return torch.load(f, map_location=map_location).to(device)


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def use_cuda(enabled, device_id=0):
    """Verifies if CUDA is available and sets default device to be device_id."""
    if not enabled:
        return None
    # assert torch.cuda.is_available(), 'CUDA is not available'
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    torch.cuda.set_device(device_id)
    return device_id


def gen_choices(cnts, idx=0, choice=[]):
    """Generate all the valid choices.
    It generates both yours and your opponent choices.
    """
    if idx >= len(cnts):
        return [
            (choice[:], [n - c for n, c in zip(cnts, choice)]),
        ]
    choices = []
    for c in range(cnts[idx] + 1):
        choice.append(c)
        choices += gen_choices(cnts, idx + 1, choice)
        choice.pop()
    return choices


def compute_score(vals, picks):
    """Compute the score of the selection."""
    assert len(vals) == len(picks)
    return np.sum([v * p for v, p in zip(vals, picks)])


def parse_ctx(line):
    """
    returns context for alice and bob in train, val, test.txt
    """
    to_int = lambda l: [int(x) for x in l]
    start = line.find("<input>") + 7
    end = line.find("</input>")
    alice_ctx = to_int(line[start:end].split())
    start = line.find("<partner_input>") + 15
    end = line.find("</partner_input>")
    bob_ctx = to_int(line[start:end].split())
    return alice_ctx, bob_ctx


class ContextGenerator(object):
    """Dialogue context generator. Generates contexes from the file."""

    def __init__(self, context_file):
        self.ctxs = []
        with open(context_file, "r") as f:
            ctx_pair = []
            for line in f:
                ctx = line.strip().split()
                ctx_pair.append(ctx)
                if len(ctx_pair) == 2:
                    self.ctxs.append(ctx_pair)
                    ctx_pair = []

    def sample(self):
        return random.choice(self.ctxs)

    def iter(
        self, nepoch: int = None, neps: int = None, is_random=False
    ) -> Iterable[List[List[str]]]:
        """
        Iterate through all of the contexts specified in the context_file

        Args:
            nepoch: The number of times to iterate through every context in the file
            n_eps: The number of contexts to generate.

        Note: Specify either nepoch or n_eps, but not both

        Returns: A generator where each element contains a list of length 2,
            each specifying the utilities and counts for each agent in the game
        """
        if nepoch is not None and neps is not None:
            raise ValueError("Specify either number of epochs or episodes")

        if nepoch is not None:
            for e in range(nepoch):
                if is_random:
                    random.shuffle(self.ctxs)
                for ctx in self.ctxs:
                    yield ctx
        elif neps is not None:
            n = 0
            while n < neps:
                if is_random:
                    random.shuffle(self.ctxs)
                for ctx in self.ctxs:
                    yield ctx
                    n += 1
                    if n == neps:
                        break
        else:
            raise NotImplementedError


def get_agent_type(model):
    if isinstance(model, CdaRnnModel):
        return CdaAgent
    elif isinstance(model, DialogModel):
        return RlAgent
    else:
        raise ValueError("unknown model type: %s" % model)


def load_agent(model, args, name, train):
    """
    Loads agent from model file
    :param model:
    :param cuda: boolean
    :param name: string
    :param train: boolean
    :return: agent
    """
    agent_model = load_model(model, args.cuda)
    agent_ty = get_agent_type(agent_model)
    agent = agent_ty(agent_model, args, name=name, train=train)
    if not args.cuda:
        agent.model.device_id = None
        agent.model.ctx_encoder.device_id = None
    return agent


def format_choice(choices):
    """
    Formats a list of choices into training data format:
    item0=x item1=y item2=z item0=a item1=b item2=c
    :param choices: nested list
    :return: string
    """
    if "<no_agreement>" in choices[0]:
        return choices[0]
    elif "<no_agreement>" in choices[1]:
        return choices[1]
    return choices[0]


def load_args(args):
    """
    Loads arguments from yaml file, giving priority to any argparse arguments specified by the user
    """
    args.config = "configs/configs.yaml"
    config = SimpleNamespace(**yaml.load(open(args.config), Loader=yaml.FullLoader))
    config.__dict__.update(
        (k, v) for k, v in vars(args).items() if v
    )  # prioritize argparse args
    return config


def check_params(old_model, model):
    old_dict = old_model.state_dict()
    curr_dict = model.state_dict()
    for key in old_dict:
        eq = (old_dict[key] == curr_dict[key]).all()
        if not eq:
            return False
    return True


def copy_params(old_model, model):
    old_dict = old_model.state_dict()
    model_dict = model.state_dict()
    model_dict.update(old_dict)
    model.load_state_dict(model_dict)


def freeze_params(old_model, model):
    old_dict = old_model.state_dict()
    for n, param in model.named_parameters():
        if n in old_dict:
            param.requires_grad = False


def unfreeze_params(old_model, model):
    old_dict = old_model.state_dict()
    for n, param in model.named_parameters():
        if n in old_dict:
            param.requires_grad = True
