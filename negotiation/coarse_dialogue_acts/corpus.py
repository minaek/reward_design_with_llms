from typing import List

from utils.data import WordCorpus, get_tag, Dictionary, Example, read_lines
from coarse_dialogue_acts.parser import Parser
from coarse_dialogue_acts.lexicon import Lexicon
from coarse_dialogue_acts.kb import KB
from coarse_dialogue_acts.dialogue_state import DialogueState
from coarse_dialogue_acts.dictionary import ActDictionary
from coarse_dialogue_acts.lf import LogicalForm
import os
import random
import numpy as np


class ActCorpus(WordCorpus):
    """
    Implementation of Percy's Coarse Dialogue Acts parser on top of the Word Corpus.
    Each proposal/insist vector is treated as a 'word' in the dictionary in addition to non-proposal words (e.g., "Agree")
    Size of act_dict is around 256 (not too big)
    """

    def __init__(self, path, freq_cutoff=2, train='train.txt',
                 valid='val.txt', test='test.txt', verbose=False):
        self.act_dict = ActDictionary()
        super().__init__(path, freq_cutoff=freq_cutoff, train=train, valid=valid, test=test, verbose=verbose)
        # Mapping to word_dict so that upstream classes can easily use this corpus
        self.word_dict = ActDictionary()

    def get_word_indices(self, dialogue_tokens: List[str], input_tokens: List[str]) -> List[int]:
        context = [int(x) for x in input_tokens]
        kb = KB(context)
        sentences = split_into_sentences(dialogue_tokens)
        assert sentences[-1][-1] == '<selection>'
        lexicon = Lexicon(['ball', 'hat', 'book'])
        parsers = [(Parser(agent, kb, lexicon), DialogueState(agent, kb)) for agent in (0, 1)]

        lfs = []
        for i, s in enumerate(sentences[:-1]):  # Ignore selection sentence
            # Alternate between parsers so that each parser only keeps track of the opponent state
            listener_id = (i + 1) % 2
            curr_parser, curr_state = parsers[listener_id]
            lf, _ = curr_parser.parse(s, curr_state)

            # Prepend the "YOU:" or "THEM:" tokens
            lfs.append(LogicalForm(s[0]))
            lfs.append(lf)

            # print(s, lf.intent, lf.my_proposal)

            for p, ds in parsers:
                ds.update(listener_id, lf)

        # Add the final selection sentence
        lfs.extend([LogicalForm(tok) for tok in sentences[-1]])

        # print('=' * 20)

        return self.act_dict.act2idx(lfs)


def split_into_sentences(dialogue_tokens: List[str]) -> List[List[str]]:
    """
    Split a conversation into a list of sentences
    Args:
        dialogue_tokens: A conversation split up into tokens

    Returns: List of of token sequences, each representing a different sentence

    Examples:
        >>> split_into_sentences(['YOU:', 'I', 'want', 'hats', '<eos>', 'THEM:', 'ok', '<eos>', 'YOU:', '<selection>'])
        [['YOU:', 'I', 'want', 'hats', '<eos>'], ['THEM:', 'ok', '<eos>'], ['YOU:', '<selection>']]
    """
    sentences = []
    curr_sentence = []
    for token in dialogue_tokens:
        curr_sentence.append(token)
        if token == '<eos>':
            if len(curr_sentence) > 0:
                sentences.append(curr_sentence)
                curr_sentence = []

    if len(curr_sentence) > 0:
        sentences.append(curr_sentence)

    return sentences


class DatasetStatsCorpus(ActCorpus):
    def __init__(self, path, freq_cutoff=2, train='train.txt', valid='val.txt', test='test.txt', verbose=False):
        self.verbose = verbose
        # only add words from the train dataset
        self.word_dict, self.item_dict, self.context_dict = Dictionary.from_file(
            os.path.join(path, train),
            freq_cutoff=freq_cutoff)
        # Mapping to word_dict so that upstream classes can easily use this corpus
        self.act_dict = ActDictionary()
        self.word_dict = ActDictionary()

        # construct all 3 datasets
        self.train = self.tokenize(os.path.join(path, train)) if train else []
        # self.valid = self.tokenize(os.path.join(path, valid)) if valid else []
        # self.test = self.tokenize(os.path.join(path, test)) if test else []

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

        # unk = self.word_dict.get_idx('<unk>')
        unk = '<unk>'
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split()
            input_tokens = get_tag(tokens, 'input')
            partner_input_tokens = get_tag(tokens, 'partner_input')
            dialogue_tokens = get_tag(tokens, 'dialogue')
            output_tokens = np.array(get_tag(tokens, 'output'))
            output_tokens = output_tokens.reshape(2, -1)

            input_tokens = [int(x) for x in input_tokens]
            partner_input_tokens = [int(x) for x in partner_input_tokens]
            word_tokens = self.word_dict.i2w(self.get_word_indices(dialogue_tokens, input_tokens))
            dataset.append(([input_tokens, partner_input_tokens], word_tokens, output_tokens))
            # compute statistics
            total += len(input_tokens) + len(word_tokens) + len(output_tokens)
            unks += np.count_nonzero([idx == unk for idx in word_tokens])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%, datapoints %d' % (
                file_name, total, unks, 100. * unks / total, len(lines)))
        return dataset


if __name__ == '__main__':
    loc = '../data/negotiate'

    # Test 1
    dialogue = ['YOU:', 'I', 'want', 'hats', '<eos>', 'THEM:', 'ok', '<eos>', 'YOU:', '<selection>']
    result = [['YOU:', 'I', 'want', 'hats', '<eos>'], ['THEM:', 'ok', '<eos>'], ['YOU:', '<selection>']]
    assert split_into_sentences(dialogue) == result

    corpus = ActCorpus(path=loc)
    print(corpus)
