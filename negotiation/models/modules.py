# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Helper functions for module initialization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_rnn(rnn, init_range, weights=None, biases=None):
    """Initializes RNN uniformly."""
    weights = weights or ["weight_ih_l0", "weight_hh_l0"]
    biases = biases or ["bias_ih_l0", "bias_hh_l0"]
    # Init weights
    for w in weights:
        rnn._parameters[w].data.uniform_(-init_range, init_range)
    # Init biases
    for b in biases:
        rnn._parameters[b].data.fill_(0)


def init_rnn_cell(rnn, init_range):
    """Initializes RNNCell uniformly."""
    init_rnn(rnn, init_range, ["weight_ih", "weight_hh"], ["bias_ih", "bias_hh"])


def init_cont(cont, init_range):
    """Initializes a container uniformly."""
    for m in cont:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-init_range, init_range)
        if hasattr(m, "bias"):
            m.bias.data.fill_(0)


class CudaModule(nn.Module):
    """A helper to run a module on a particular device using CUDA."""

    def __init__(self, device_id):
        super(CudaModule, self).__init__()
        self.device_id = device_id

    def to_device(self, m):
        if self.device_id is not None:
            return m.cuda(self.device_id)
        return m


class RnnContextEncoder(CudaModule):
    """A module that encodes dialogues context using an RNN."""

    def __init__(self, n, k, nembed, nhid, init_range, device_id):
        super(RnnContextEncoder, self).__init__(device_id)
        self.nhid = nhid

        # use the same embedding for counts and values
        self.embeder = nn.Embedding(n, nembed)
        # an RNN to encode a sequence of counts and values
        self.encoder = nn.GRU(input_size=nembed, hidden_size=nhid, bias=True)

        self.embeder.weight.data.uniform_(-init_range, init_range)
        init_rnn(self.encoder, init_range)

    def forward(self, ctx):
        ctx_h = self.to_device(torch.zeros(1, ctx.size(1), self.nhid))
        # create embedding
        ctx_emb = self.embeder(ctx)
        # run it through the RNN to get a hidden representation of the context
        _, ctx_h = self.encoder(ctx_emb, Variable(ctx_h))
        return ctx_h


class MlpContextEncoder(CudaModule):
    """Simple encoder for the dialogue context. Encoder counts and values via MLP."""

    def __init__(
        self, n: int, k: int, nembed: int, nhid: int, init_range: int, device_id: int
    ):
        """
        Args:
            n: The number of possible token values for the context.
            k: The number of tokens that make up a full context
            nembed: The size of the embedding layer
            nhid: The size of the hidden layer
            init_range: The range of values to initialize the parameters with
        """
        super(MlpContextEncoder, self).__init__(device_id)

        # create separate embedding for counts and values
        self.cnt_enc = nn.Embedding(n, nembed).to(device)
        self.val_enc = nn.Embedding(n, nembed).to(device)

        self.encoder = nn.Sequential(nn.Tanh(), nn.Linear(k * nembed, nhid)).to(device)

        self.cnt_enc.weight.data.uniform_(-init_range, init_range)
        self.val_enc.weight.data.uniform_(-init_range, init_range)
        init_cont(self.encoder, init_range)

    def forward(self, ctx):
        idx = np.arange(ctx.size(0) // 2)
        # extract counts and values
        cnt_idx = Variable(self.to_device(torch.from_numpy(2 * idx + 0)))
        val_idx = Variable(self.to_device(torch.from_numpy(2 * idx + 1)))
        cnt_idx = cnt_idx.to(device)
        val_idx = val_idx.to(device)

        cnt = ctx.index_select(0, cnt_idx)
        val = ctx.index_select(0, val_idx)

        # embed counts and values
        cnt_emb = self.cnt_enc(cnt.to(device))
        val_emb = self.val_enc(val.to(device))

        # element wise multiplication to get a hidden state
        h = torch.mul(cnt_emb, val_emb)
        # run the hidden state through the MLP
        h = h.transpose(0, 1).contiguous().view(ctx.size(1), -1)
        ctx_h = self.encoder(h.to(device)).unsqueeze(0).to(device)
        return ctx_h  # (1,1,64)


class MlpOutcomeEncoder(CudaModule):
    """Simple encoder for the dialogue context. Encoder counts and values via MLP."""

    def __init__(
        self, n: int, k: int, nembed: int, nhid: int, init_range: int, device_id: int
    ):
        """
        Args:
            n: The number of possible token values for the context.
            k: The number of tokens that make up a full context
            nembed: The size of the embedding layer
            nhid: The size of the hidden layer
            init_range: The range of values to initialize the parameters with
        """
        super(MlpOutcomeEncoder, self).__init__(device_id)

        # create separate embedding for counts and values
        self.alice_enc = nn.Embedding(n, nembed)
        self.bob_enc = nn.Embedding(n, nembed)

        self.encoder = nn.Sequential(nn.Tanh(), nn.Linear(2 * nembed + 1, 2))

        self.alice_enc.weight.data.uniform_(-init_range, init_range)
        self.bob_enc.weight.data.uniform_(-init_range, init_range)
        init_cont(self.encoder, init_range)

    def forward(self, ctx):
        is_agreement, alice_outcome, bob_outcome = 0, 0, 0
        # embed counts and values
        alice_emb = self.alice_enc(alice_outcome)
        bob_emb = self.bob_enc(bob_outcome)

        # element wise multiplication to get a hidden state
        h = [is_agreement, alice_emb, bob_emb]  # sth to do with torch here
        # run the hidden state through the MLP
        h = h.transpose(0, 1).contiguous().view(ctx.size(1), -1)
        ctx_h = self.encoder(h.to(device)).unsqueeze(0).to(device)
        return ctx_h
