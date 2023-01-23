# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A library that is responsible for data reading.
"""

import copy
import os
import random
from collections import OrderedDict
from typing import Tuple, List, Dict, Iterable

import numpy as np
import torch

# special tokens
SPECIAL = [
    '<eos>',
    '<unk>',
    '<selection>',
    '<pad>',
]

# tokens that stops either a sentence or a conversation
STOP_TOKENS = [
    '<eos>',
    '<selection>',
]

Example = Tuple[List[int], List[int], List[int]]


def get_tag(tokens: List[str], tag: str) -> List[str]:
    """Extracts the value inside the given tag."""
    return tokens[tokens.index('<' + tag + '>') + 1:tokens.index('</' + tag + '>')]


def read_lines(file_name: str) -> List[str]:
    """Reads all the lines from the file."""
    assert os.path.exists(file_name), 'file does not exists %s' % file_name
    lines = []
    with open(file_name, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines


class Dictionary(object):
    """Maps words into indices.

    It has forward and backward indexing.
    """

    def __init__(self, init=True):
        self.word2idx = OrderedDict()
        self.idx2word = []
        if init:
            # add special tokens if asked
            for i, k in enumerate(SPECIAL):
                self.word2idx[k] = i
                self.idx2word.append(k)

    def add_word(self, word: str) -> int:
        """Adds a new word, if the word is in the dictionary, just returns its index."""
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def i2w(self, idx: Iterable[int]) -> List[str]:
        """Converts a list of indices into words."""
        return [self.idx2word[i] for i in idx]

    def w2i(self, words: Iterable[str]) -> List[int]:
        """Converts a list of words into indices. Uses <unk> for the unknown words."""
        unk = self.word2idx.get('<unk>', None)
        return [self.word2idx.get(w, unk) for w in words]

    def get_idx(self, word: str) -> int:
        unk = self.word2idx.get('<unk>', None)
        return self.word2idx.get(word, unk)

    def get_word(self, idx: int) -> str:
        """Gets word by its index."""
        return self.idx2word[idx]

    def __len__(self):
        return len(self.idx2word)

    @classmethod
    def read_tag(cls, file_name: str, tag: str, freq_cutoff: int = -1, init_dict: bool = True) -> 'Dictionary':
        """
        Convert the tokens between a tag in a dataset into a dictionary

        Args:
            file_name: Location of the txt file with dialogues

            tag: The XML tag which contains the tokens that you want to parse

            freq_cutoff: A minimum number of times the token must appear for it to be added to the dictionary.
                By default, all tokens that are seen at least once are added

            init_dict: If True, will initialize the dictionary with the
                default special tokens <eos>, <unk>, <selection>, and <pad>

        Returns: A Dictionary that contains all of the tokens in between the specified tag
            for all examples in the training file

        """
        token_freqs = OrderedDict()
        with open(file_name, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                tokens = get_tag(tokens, tag)
                for token in tokens:
                    token_freqs[token] = token_freqs.get(token, 0) + 1
        dictionary = cls(init=init_dict)
        token_freqs = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)
        for token, freq in token_freqs:
            if freq > freq_cutoff:
                dictionary.add_word(token)
        return dictionary

    @classmethod
    def from_file(cls, file_name: str, freq_cutoff: int) -> Tuple['Dictionary', 'Dictionary', 'Dictionary']:
        """Constructs a dictionary from the given file."""
        assert os.path.exists(file_name)
        word_dict = cls.read_tag(file_name, 'dialogue', freq_cutoff=freq_cutoff)
        item_dict = cls.read_tag(file_name, 'output', init_dict=False)
        context_dict = cls.read_tag(file_name, 'input', init_dict=False)
        return word_dict, item_dict, context_dict


class WordCorpus(object):
    """An utility that stores the entire dataset.

    It has the train, valid and test datasets and corresponding dictionaries.
    """

    def __init__(self, path, freq_cutoff=2, train='train.txt',
                 valid='val.txt', test='test.txt', verbose=False):
        self.verbose = verbose
        # only add words from the train dataset
        self.word_dict, self.item_dict, self.context_dict = Dictionary.from_file(
            os.path.join(path, train),
            freq_cutoff=freq_cutoff)

        # construct all 3 datasets
        self.train = self.tokenize(os.path.join(path, train)) if train else []
        self.valid = self.tokenize(os.path.join(path, valid)) if valid else []
        self.test = self.tokenize(os.path.join(path, test)) if test else []

        # find out the output length from the train dataset
        self.output_length = max([len(x[2]) for x in self.train])

    def tokenize(self, file_name: str) -> List[Example]:
        """
        Tokenize and numericalize the dataset found at filename.

        Args:
            file_name: The location of the dataset

        Returns: A list of examples. Each example contains:
            input_idxs: A numerical representation of the context, which includes the number of items
                in the game as well as the individual utilities for each item.

            word_idxs: A list of token indexes for each of the words spoken in the dialogue. This includes divider tokens
                like "YOU: ", "THEM: ", "<selection>", etc.

            item_idxs: An index representing the allocation given to the user at the end of the game
                Example index: "item0=0 item1=1 item2=2" -> 55
        """
        lines = read_lines(file_name)
        random.shuffle(lines)

        unk = self.word_dict.get_idx('<unk>')
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split()
            input_tokens = get_tag(tokens, 'input')
            dialogue_tokens = get_tag(tokens, 'dialogue')

            input_idxs = self.context_dict.w2i(input_tokens)
            word_idxs = self.get_word_indices(dialogue_tokens, input_tokens)
            item_idxs = self.item_dict.w2i(get_tag(tokens, 'output'))
            dataset.append((input_idxs, word_idxs, item_idxs))
            # compute statistics
            total += len(input_idxs) + len(word_idxs) + len(item_idxs)
            unks += np.count_nonzero([idx == unk for idx in word_idxs])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%, datapoints %d' % (
                file_name, total, unks, 100. * unks / total, len(lines)))
        return dataset

    def get_word_indices(self, dialogue_tokens: List[str], input_tokens: List[str]) -> List[int]:
        return self.word_dict.w2i(dialogue_tokens)

    def train_dataset(self, bsz: int, shuffle=True, device_id=None):
        return self._split_into_batches(copy.copy(self.train), bsz,
                                        shuffle=shuffle, device_id=device_id)

    def valid_dataset(self, bsz: int, shuffle=True, device_id=None):
        return self._split_into_batches(copy.copy(self.valid), bsz,
                                        shuffle=shuffle, device_id=device_id)

    def test_dataset(self, bsz: int, shuffle=True, device_id=None):
        return self._split_into_batches(copy.copy(self.test), bsz, shuffle=shuffle,
                                        device_id=device_id)

    def _split_into_batches(self, dataset, bsz, shuffle=True, device_id=None):
        """Splits given dataset into batches."""
        if shuffle:
            random.shuffle(dataset)

        # sort and pad
        dataset.sort(key=lambda x: len(x[1]))
        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0
        }

        for i in range(0, len(dataset), bsz):
            # groups contexes, words and items
            inputs, words, items = [], [], []
            for j in range(i, min(i + bsz, len(dataset))):
                inputs.append(dataset[j][0])
                words.append(dataset[j][1])
                items.append(dataset[j][2])

            # the longest dialogue in the batch
            max_len = len(words[-1])

            # pad all the dialogues to match the longest dialogue
            for j in range(len(words)):
                stats['n'] += max_len
                stats['nonpadn'] += len(words[j])
                words[j] += [pad] * (max_len - len(words[j]))

            # construct tensor for context
            ctx = torch.LongTensor(inputs).transpose(0, 1).contiguous()
            data = torch.LongTensor(words).transpose(0, 1).contiguous()
            # construct tensor for selection target
            sel_tgt = torch.LongTensor(items).transpose(0, 1).contiguous().view(-1)
            #if 16 in sel_tgt:
            #    print("FOUND THE FRIGGING THING")
            if device_id is not None:
                ctx = ctx.cuda(device_id)
                data = data.cuda(device_id)
                sel_tgt = sel_tgt.cuda(device_id)

            # construct tensor for input and target
            inpt = data.narrow(0, 0, data.size(0) - 1)
            tgt = data.narrow(0, 1, data.size(0) - 1).view(-1)

            batches.append((ctx, inpt, tgt, sel_tgt))

        if shuffle:
            random.shuffle(batches)

        return batches, stats

    def _split_into_batches_partner(self, dataset, bsz, shuffle=True, device_id=None, num_partners=2):
        """
        Splits given dataset into batches.
        dataset = (input_idxs, word_idxs, choice_idxs, agent_labels)
        """
        if shuffle:
            random.shuffle(dataset)

        # sort and pad
        dataset.sort(key=lambda x: len(x[1]))
        pad = self.word_dict.get_idx('<pad>')
        partner_pad = num_partners

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0
        }

        for i in range(0, len(dataset), bsz):
            # groups contexts, words and items
            inputs, words, items, partner_labels = [], [], [], []
            for j in range(i, min(i + bsz, len(dataset))):
                inputs.append(dataset[j][0])
                words.append(dataset[j][1])
                items.append(dataset[j][2])
                partner_labels.append(dataset[j][3])

            # the longest dialogue in the batch
            max_len = len(words[-1])

            # pad all the dialogues to match the longest dialogue
            for j in range(len(words)):
                stats['n'] += max_len
                stats['nonpadn'] += len(words[j])
                words[j] += [pad] * (max_len - len(words[j]))
                partner_labels[j] += [partner_pad] * (max_len - len(partner_labels[j]))

            # construct tensor for context
            ctx = torch.LongTensor(inputs).transpose(0, 1).contiguous()
            data = torch.LongTensor(words).transpose(0, 1).contiguous()
            labels = torch.LongTensor(partner_labels).transpose(0, 1).contiguous()
            # construct tensor for selection target
            sel_tgt = torch.LongTensor(items).transpose(0, 1).contiguous().view(-1)
            if device_id is not None:
                ctx = ctx.cuda(device_id)
                data = data.cuda(device_id)
                sel_tgt = sel_tgt.cuda(device_id)

            # construct tensor for input and target
            inpt = data.narrow(0, 0, data.size(0) - 1)
            tgt = data.narrow(0, 1, data.size(0) - 1).view(-1)
            #partner_inpt = data.narrow(0, 0, data.size(0))
            partner_tgt = labels.narrow(0, 1, labels.size(0) - 1).view(-1)

            #batches.append((ctx, inpt, tgt, sel_tgt, partner_inpt, partner_tgt))
            batches.append((ctx, inpt, tgt, sel_tgt, partner_tgt))

        if shuffle:
            random.shuffle(batches)

        return batches, stats
