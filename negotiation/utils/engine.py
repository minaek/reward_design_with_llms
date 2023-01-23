# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Training utilities.
"""

import copy
import random
import time
from typing import Iterable, Callable

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from utils.utils import device


class Criterion(object):
    """Weighted CrossEntropyLoss."""

    def __init__(self, dictionary, device_id=None, bad_toks=[], reduction="mean"):
        w = torch.Tensor(len(dictionary)).fill_(1)
        for tok in bad_toks:
            w[dictionary.get_idx(tok)] = 0.0
        if device_id is not None:
            w = w.cuda(device_id)
        self.crit = nn.CrossEntropyLoss(w, reduction=reduction).to(device)

    def __call__(self, out, tgt):
        out = out.to(device)
        tgt = tgt.to(device)
        return self.crit(out, tgt)


class CriterionUnweighted(object):
    """CrossEntropyLoss."""

    def __init__(self, pad_token, reduction="mean"):
        self.crit = nn.CrossEntropyLoss(reduction=reduction, ignore_index=pad_token)

    def __call__(self, out, tgt):
        out = out.to(device)
        tgt = tgt.to(device)
        return self.crit(out, tgt)


class Engine(object):
    """The training engine.

    Performs training and evaluation.
    """

    def __init__(self, model, args, device_id=None, verbose=False, corpus_type="human"):
        self.model = model
        self.args = args
        self.device_id = device_id
        self.verbose = verbose
        self.opt = optim.SGD(
            self.model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            nesterov=(self.args.nesterov and self.args.momentum > 0),
        )
        self.crit = Criterion(self.model.word_dict, device_id=device_id)

        self.sel_crit = Criterion(
            self.model.item_dict, device_id=device_id, bad_toks=["<disconnect>"]
        )
        self.corpus_type = corpus_type

    @staticmethod
    def forward(model, batch):
        """A helper function to perform a forward pass on a batch."""
        # extract the batch into context, input, target and selection target
        ctx, inpt, tgt, sel_tgt = map(Variable, batch)

        # get context hidden state
        ctx_h = model.forward_context(ctx)
        # create initial hidden state for the language rnn
        lang_h = model.zero_hid(ctx_h.size(1), model.args.nhid_lang)

        # perform forward for the language model
        out, lang_h = model.forward_lm(inpt, lang_h, ctx_h)
        target = tgt

        # perform forward for the selection
        sel_out = model.forward_selection(inpt, lang_h, ctx_h)

        return out, lang_h, target, sel_out, sel_tgt

    def get_model(self):
        """Extracts the model."""
        return self.model

    def train_pass(self, N, trainset):
        """Training pass."""
        # make the model trainable
        self.model.train()

        total_loss = 0
        total_correct = []
        start_time = time.time()

        # training loop
        for batch in trainset:
            self.t += 1
            # forward pass
            out, hid, tgt, sel_out, sel_tgt = Engine.forward(self.model, batch)

            # compute accuracy
            out = out.view(-1, N)
            pred = torch.max(out, dim=1).indices.to(device)
            tgt = tgt.to(device)
            total_correct.append((pred == tgt).sum().float() / len(tgt))

            # compute LM loss and selection loss
            loss = self.crit(out, tgt)
            loss += self.sel_crit(sel_out, sel_tgt) * self.model.args.sel_weight

            self.opt.zero_grad()
            # backward step with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.opt.step()

            total_loss += loss.item()

        total_loss /= len(trainset)
        total_correct = np.mean(total_correct)
        time_elapsed = time.time() - start_time
        return total_loss, time_elapsed, total_correct.item()

    def train_single(self, N, trainset):
        """A helper function to train on a random batch."""
        batch = random.choice(trainset)
        out, hid, tgt, sel_out, sel_tgt = Engine.forward(self.model, batch)

        loss = (
            self.crit(out.view(-1, N), tgt)
            + self.sel_crit(sel_out, sel_tgt) * self.model.args.sel_weight
        )

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.opt.step()
        return loss

    def valid_pass(self, N, validset, validset_stats):
        """Validation pass."""
        was_training = False
        if self.model.training:
            was_training = True
        # put the model into the evaluation mode
        self.model.eval()

        valid_loss, select_loss, total_correct = 0, 0, []
        for batch in validset:
            # compute forward pass
            with torch.no_grad():
                out, hid, tgt, sel_out, sel_tgt = Engine.forward(self.model, batch)

            # compute accuracy
            out = out.view(-1, N)
            pred = torch.max(out, dim=1).indices
            pred = pred.to(device)
            tgt = tgt.to(device)
            total_correct.append((pred == tgt).sum().float() / len(tgt))

            # evaluate LM and selection losses
            valid_loss += tgt.size(0) * self.crit(out, tgt).item()
            select_loss += self.sel_crit(sel_out, sel_tgt).item()

        if was_training:
            self.model.train()
        # dividing by the number of words in the input, not the tokens modeled,
        # because the latter includes padding
        total_correct = np.mean(total_correct)
        return (
            valid_loss / validset_stats["nonpadn"],
            select_loss / len(validset),
            total_correct.item(),
        )

    def iter(self, N, epoch, lr, traindata, validdata):
        """Performs on iteration of the training.
        Runs one epoch on the training and validation datasets.
        """
        trainset, _ = traindata
        validset, validset_stats = validdata

        train_loss, train_time, train_acc = self.train_pass(N, trainset)
        valid_loss, valid_select_loss, valid_acc = self.valid_pass(
            N, validset, validset_stats
        )

        if self.verbose:
            print(
                "| epoch %03d | trainloss %.3f | trainppl %.3f | trainacc %.3f | s/epoch %.2f | lr %0.8f"
                % (epoch, train_loss, np.exp(train_loss), train_acc, train_time, lr)
            )
            print(
                "| epoch %03d | validloss %.3f | validppl %.3f | validacc %.3f"
                % (epoch, valid_loss, np.exp(valid_loss), valid_acc)
            )

        return train_loss, valid_loss, valid_select_loss

    def train(self, corpus, N=None, callbacks: Iterable[Callable] = []):
        """Entry point."""
        N = len(corpus.word_dict) if N is None else N
        best_model, best_valid_select_loss = None, 1e100
        lr = self.args.lr
        last_decay_epoch = 0
        self.t = 0

        validdata = corpus.valid_dataset(self.args.bsz, device_id=self.device_id)
        for epoch in range(1, self.args.max_epoch + 1):
            traindata = corpus.train_dataset(self.args.bsz, device_id=self.device_id)
            train_loss, valid_loss, valid_select_loss = self.iter(
                N, epoch, lr, traindata, validdata
            )

            if valid_select_loss < best_valid_select_loss:
                best_valid_select_loss = valid_select_loss
                best_model = copy.deepcopy(self.model)
                best_model.flatten_parameters()

            for cb_fn in callbacks:
                cb_fn(epoch)

        if self.verbose:
            print(
                "| start annealing | best validselectloss %.3f | best validselectppl %.3f"
                % (best_valid_select_loss, np.exp(best_valid_select_loss))
            )

        self.model = best_model
        if self.args.decay_every == 0:
            return train_loss, valid_loss, valid_select_loss

        for epoch in range(self.args.max_epoch + 1, 100):
            if epoch - last_decay_epoch >= self.args.decay_every:
                last_decay_epoch = epoch
                lr /= self.args.decay_rate
                if lr < self.args.min_lr:
                    break
                self.opt = optim.SGD(self.model.parameters(), lr=lr)

            traindata = corpus.train_dataset(self.args.bsz, device_id=self.device_id)
            train_loss, valid_loss, valid_select_loss = self.iter(
                N, epoch, lr, traindata, validdata
            )

            for cb_fn in callbacks:
                cb_fn(epoch)

        return train_loss, valid_loss, valid_select_loss
