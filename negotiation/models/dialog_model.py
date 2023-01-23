# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
An RNN based dialogue model. Performance both language and choice generation.
"""
from typing import List, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable

from utils.data import STOP_TOKENS, WordCorpus
from utils.domain import get_domain
from models import modules

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DialogModel(modules.CudaModule):
    corpus_ty = WordCorpus

    def __init__(self, word_dict, item_dict, context_dict, output_length, args, device_id):
        super(DialogModel, self).__init__(device_id)

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.item_dict = item_dict
        self.context_dict = context_dict
        self.args = args

        # embedding for words
        self.word_encoder = nn.Embedding(len(self.word_dict), args.nembed_word).to(device)

        # context encoder
        ctx_encoder_ty = modules.RnnContextEncoder if args.rnn_ctx_encoder \
            else modules.MlpContextEncoder
        self.ctx_encoder = ctx_encoder_ty(len(self.context_dict), domain.input_length(),
                                          args.nembed_ctx, args.nhid_ctx, args.init_range, device_id).to(device)

        # a reader RNN, to encode words
        self.reader = nn.GRU(
            input_size=args.nhid_ctx + args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True).to(device)
        self.decoder = nn.Linear(args.nhid_lang, args.nembed_word).to(device)
        # a writer, a RNNCell that will be used to generate utterances
        self.writer = nn.GRUCell(
            input_size=args.nhid_ctx + args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True).to(device)

        # tie the weights of reader and writer
        self.writer.weight_ih = self.reader.weight_ih_l0
        self.writer.weight_hh = self.reader.weight_hh_l0
        self.writer.bias_ih = self.reader.bias_ih_l0
        self.writer.bias_hh = self.reader.bias_hh_l0

        self.dropout = nn.Dropout(args.dropout)

        # a bidirectional selection RNN
        # it will go through input words and generate by the reader hidden states
        # to produce a hidden representation
        self.sel_rnn = nn.GRU(
            input_size=args.nhid_lang + args.nembed_word,
            hidden_size=args.nhid_attn,
            bias=True,
            bidirectional=True).to(device)

        # mask for disabling special tokens when generating sentences
        self.special_token_mask = torch.FloatTensor(len(self.word_dict))
        # mask for disabling non-context proposals
        # self.context_mask = torch.FloatTensor(len(self.word_dict))

        # attention to combine selection hidden states
        self.attn = nn.Sequential(
            torch.nn.Linear(2 * args.nhid_attn, args.nhid_attn),
            nn.Tanh(),
            torch.nn.Linear(args.nhid_attn, 1)
        ).to(device)

        # selection encoder, takes attention output and context hidden and combines them
        self.sel_encoder = nn.Sequential(
            torch.nn.Linear(2 * args.nhid_attn + args.nhid_ctx, args.nhid_sel),
            nn.Tanh()
        ).to(device)
        # selection decoders, one per each item
        self.sel_decoders = nn.ModuleList()
        for i in range(output_length):
            self.sel_decoders.append(nn.Linear(args.nhid_sel, len(self.item_dict)).to(device))

        self.init_weights()

        # fill in the mask
        for i in range(len(self.word_dict)):
            w = self.word_dict.get_word(i)
            special = domain.item_pattern.match(w) or w in ('<unk>', 'YOU:', 'THEM:', '<pad>')
            self.special_token_mask[i] = -999 if special else 0.0

        self.special_token_mask = self.to_device(self.special_token_mask)

    def set_device_id(self, device_id):
        self.device_id = device_id
        self.special_token_mask = self.to_device(self.special_token_mask)

    def flatten_parameters(self):
        self.reader.flatten_parameters()
        self.sel_rnn.flatten_parameters()

    def zero_hid(self, bsz, nhid=None, copies=None):
        """A helper function to create an zero hidden state."""
        nhid = self.args.nhid_lang if nhid is None else nhid
        copies = 1 if copies is None else copies
        hid = torch.zeros(copies, bsz, nhid)
        hid = self.to_device(hid)
        return Variable(hid).to(device)

    def init_weights(self):
        """Initializes params uniformly."""
        self.decoder.weight.data.uniform_(-self.args.init_range, self.args.init_range)
        self.decoder.bias.data.fill_(0)

        modules.init_rnn(self.reader, self.args.init_range)

        self.word_encoder.weight.data.uniform_(-self.args.init_range, self.args.init_range)

        modules.init_cont(self.attn, self.args.init_range)
        modules.init_cont(self.sel_encoder, self.args.init_range)
        modules.init_cont(self.sel_decoders, self.args.init_range)

    def read(self, inpt: Tensor, lang_h: Tensor, ctx_h: Tensor, prefix_token: str = 'THEM:') -> Tuple[Tensor, Tensor]:
        """Reads a given utterance."""
        # inpt contains the pronounced utterance
        # add a "THEM:" token to the start of the message
        if prefix_token is not None:
            prefix = self.word2var(prefix_token).unsqueeze(0).to(device)
            inpt = inpt.to(device)
            inpt = torch.cat([prefix, inpt])

        # embed words
        inpt_emb = self.word_encoder(inpt)

        # append the context embedding to every input word embedding
        ctx_h_rep = ctx_h.expand(inpt_emb.size(0), ctx_h.size(1), ctx_h.size(2))
        inpt_emb = torch.cat([inpt_emb, ctx_h_rep], 2)

        # finally read in the words
        out, lang_h = self.reader(inpt_emb, lang_h)

        return out, lang_h

    def write(self, lang_h: torch.Tensor, ctx_h: torch.Tensor, max_words: int, temperature: float,
              context_mask: List[float],
              stop_tokens: List[str] = STOP_TOKENS, resume: bool = False) -> Tuple[
        List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a sentence word by word and feed the output of the
        previous time step as input to the next.
        Args:
            lang_h: The current language model hidden state. Size = (1, 1, nhid_lang)
            ctx_h: The current context embedding. Size = (1, 1, nhid_ctx)
            max_words: The maximum length of the sentence that can be generated
            temperature: Softmax Temperature
            stop_tokens: List of tokens that represent the end of the current agent's turn
            resume: If False, the writer will be fed an initial "YOU:" token

        Returns: (
            logprobs: A list of 1x1 tensors representing the log prob of the words generated
            outs: A tensor of shape (seq_len, 1) representing the indices of the words generated
            lang_h: The new hidden state of the agent. Size = (1, nhid_lang)
            lang_hs: All hidden states generated. Size = (seq_len, nhid_lang)
        )
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        outs, logprobs, lang_hs = [], [], []
        # remove batch dimension from the language and context hidden states
        lang_h = lang_h.squeeze(1)
        ctx_h = ctx_h.squeeze(1)
        inpt = None if resume else self.word2var('YOU:')
        # generate words until max_words have been generated or <selection>
        for _ in range(max_words): # max words is just 1 in our case
            if inpt:
                # add the context to the word embedding
                inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
                # update RNN state with last word
                lang_h = self.writer(inpt_emb, lang_h)
                lang_hs.append(lang_h)

            # decode words using the inverse of the word embedding matrix
            out = self.decoder(lang_h)
            scores = F.linear(out, self.word_encoder.weight).div(temperature)
            # subtract constant to avoid overflows in exponentiation
            scores = scores.add(-scores.max().item()).squeeze(0)

            # disable special tokens from being generated in a normal turns
            if not resume:
                mask = Variable(self.special_token_mask).to(device)
                scores = scores.add(mask)
                context_mask = Variable(self.to_device(torch.FloatTensor(context_mask))).to(device)
                scores = scores.add(context_mask)

            prob = F.softmax(scores, dim=0)
            logprob = F.log_softmax(scores, dim=0)

            # word = prob.multinomial(1).detach()
            word = torch.argmax(prob).unsqueeze(0).detach()
            logprob = logprob.gather(0, word)

            logprobs.append(logprob)
            outs.append(word.view(word.size()[0], 1))

            inpt = word

            # check if we generated an <eos> token
            if self.word_dict.get_word(word.item()) in stop_tokens:
                break

        # update the hidden state with the <eos> token
        inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
        lang_h = self.writer(inpt_emb, lang_h)
        lang_hs.append(lang_h)

        # add batch dimension back
        lang_h = lang_h.unsqueeze(1)

        return logprobs, torch.cat(outs), lang_h, torch.cat(lang_hs), prob, out

    def score(self, sen, lang_h: torch.Tensor, ctx_h: torch.Tensor, temperature: float, context_mask,
              resume: bool = False, prefix_token='YOU:', metric='likelihood', verbose=False):
        """
        Scores the [metric] of the given sentence
        For each of the first n-1 words, records the [metric] of the next word
        """

        with torch.no_grad():
            # append "YOU: " to start of sentence
            prefix = self.word2var(prefix_token).unsqueeze(0).to(device)
            sen = sen.to(device)
            sen = torch.cat([prefix, sen])
            lang_h = lang_h.squeeze(1)
            ctx_h = ctx_h.squeeze(1)
            tgt_scores = []
            tgt_probs = []

            for i in range(len(sen) - 1):  # iterate over first n-1 words of sentence
                inpt = sen[i]
                tgt = sen[i + 1]

                # add context to word embedding
                inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)

                # update hidden state with last word
                lang_h = self.writer(inpt_emb, lang_h)

                # decode hidden state
                out = self.decoder(lang_h)
                scores = F.linear(out, self.word_encoder.weight).div(temperature)
                scores = scores.add(-scores.max().item()).squeeze(0)

                # disable special tokens from being generated in normal turns
                if not resume:
                    mask = Variable(self.special_token_mask).to(device)
                    scores = scores.add(mask)
                    context_mask = Variable(torch.FloatTensor(context_mask)).to(device)
                    scores = scores.add(context_mask)

                prob = F.softmax(scores, dim=0)
                word = torch.argmax(prob).unsqueeze(0).detach()
                # print("predicted word: ", word.item(), prob[word].item(), tgt.item(), prob[tgt].item())
                # logprob = F.log_softmax(scores, dim=0)

                if metric == 'likelihood':  # [0,1]
                    tgt_scores.append(prob[tgt])  # changing log likelihood to probability
                    # tgt_probs.append(prob[tgt])
                elif metric == 'entropy':  # [-log(|vocab|), 0]
                    # remove negative because we're taking the _minimum_ in calling code (corresponds to max entropy!)
                    entropies = [p * torch.log(p) if p != 0 else p for p in prob]
                    tgt_scores.append(torch.stack(entropies).sum())
                elif metric == 'margin':  # [0,1]
                    tgt_prob = prob[tgt]
                    max_prob = torch.max(prob).unsqueeze(0).detach()
                    margin_confidence = torch.abs(tgt_prob - max_prob)
                    tgt_scores.append(
                        1. - margin_confidence)  # subtracting from 1 to ensure larger margins have smaller scores
            return tgt_scores

    def word2var(self, word):
        """Creates a variable from a given word."""
        x = torch.Tensor(1).fill_(self.word_dict.get_idx(word)).long()
        return Variable(x).to(device)

    def forward_selection(self, inpt, lang_h, ctx_h):
        """Forwards selection pass."""
        # run a birnn over the concatenation of the input embeddings and
        # language model hidden states
        inpt = inpt.to(device)
        inpt_emb = self.word_encoder(inpt)
        h = torch.cat([lang_h, inpt_emb], 2)
        h = self.dropout(h)

        # runs selection rnn over the hidden state h
        attn_h = self.zero_hid(h.size(1), self.args.nhid_attn, copies=2)
        h, _ = self.sel_rnn(h, attn_h)

        # perform attention
        h = h.transpose(0, 1).contiguous()
        logit = self.attn(h.view(-1, 2 * self.args.nhid_attn)).view(h.size(0), h.size(1))
        prob = F.softmax(logit, dim=1).unsqueeze(2).expand_as(h)
        attn = torch.sum(torch.mul(h, prob), 1, keepdim=True).transpose(0, 1).contiguous()

        # concatenate attention and context hidden and pass it to the selection encoder
        h = torch.cat([attn, ctx_h], 2).squeeze(0)
        h = self.dropout(h)
        h = self.sel_encoder.forward(h).to(device)

        # generate logits for each item separately
        outs = [decoder.forward(h) for decoder in self.sel_decoders]
        return torch.cat(outs)

    def generate_choice_logits(self, inpt, lang_h, ctx_h):
        """Similar to forward_selection, but is used while selfplaying.
        Thus it is dealing with batches of size 1.
        """
        # run a birnn over the concatenation of the input embeddings and
        # language model hidden states
        inpt_emb = self.word_encoder(inpt)
        h = torch.cat([lang_h.unsqueeze(1), inpt_emb], 2)
        h = self.dropout(h)

        # runs selection rnn over the hidden state h
        attn_h = self.zero_hid(h.size(1), self.args.nhid_attn, copies=2)
        h, _ = self.sel_rnn(h, attn_h)
        h = h.squeeze(1)

        # perform attention
        logit = self.attn(h).squeeze(1)
        prob = F.softmax(logit, dim=0).unsqueeze(1).expand_as(h)
        attn = torch.sum(torch.mul(h, prob), 0, keepdim=True)

        # concatenate attention and context hidden and pass it to the selection encoder
        ctx_h = ctx_h.squeeze(1)
        h = torch.cat([attn, ctx_h], 1)
        h = self.sel_encoder.forward(h)

        # generate logits for each item separately
        logits = [decoder.forward(h).squeeze(0) for decoder in self.sel_decoders]
        return logits

    def write_batch(self, bsz, lang_h, ctx_h, temperature, max_words=100):
        """Generate sentences for a batch simultaneously."""
        eod = self.word_dict.get_idx('<selection>')

        # resize the language hidden and context hidden states
        lang_h = lang_h.squeeze(0).expand(bsz, lang_h.size(2))
        ctx_h = ctx_h.squeeze(0).expand(bsz, ctx_h.size(2))

        # start the conversation with 'YOU:'
        inpt = torch.LongTensor(bsz).fill_(self.word_dict.get_idx('YOU:'))
        inpt = Variable(self.to_device(inpt)).to(device)

        outs, lang_hs = [], [lang_h.unsqueeze(0)]
        done = set()
        # generate until max_words are generated, or all the dialogues are done
        for _ in range(max_words):
            # embed the input
            inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
            # pass it through the writer and get new hidden state
            lang_h = self.writer(inpt_emb, lang_h)
            out = self.decoder(lang_h)
            # tie weights with encoder
            scores = F.linear(out, self.word_encoder.weight).div(temperature)
            # subtract max to make softmax more stable
            scores.sub_(scores.max(1, keepdim=True)[0].expand(scores.size(0), scores.size(1)))
            out = torch.multinomial(scores.exp(), 1).squeeze(1)
            # save outputs and hidden states
            outs.append(out.unsqueeze(0))
            lang_hs.append(lang_h.unsqueeze(0))
            inpt = out

            data = out.data.cpu()
            # check if all the dialogues in the batch are done
            for i in range(bsz):
                if data[i] == eod:
                    done.add(i)
            if len(done) == bsz:
                break

        # run it for the last word to get correct hidden states
        inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
        lang_h = self.writer(inpt_emb, lang_h)
        lang_hs.append(lang_h.unsqueeze(0))

        # concatenate outputs and hidden states into single tensors
        return torch.cat(outs, 0), torch.cat(lang_hs, 0)

    def score_sent(self, sent, lang_h, ctx_h, temperature):
        """Computes likelihood of a given sentence."""
        score = 0
        # remove batch dimension from the language and context hidden states
        lang_h = lang_h.squeeze(1)
        ctx_h = ctx_h.squeeze(1)
        inpt = Variable(torch.LongTensor(1)).to(device)
        inpt.data.fill_(self.word_dict.get_idx('YOU:'))
        inpt = self.to_device(inpt)
        lang_hs = []

        for word in sent:
            # add the context to the word embedding
            inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
            # update RNN state with last word
            lang_h = self.writer(inpt_emb, lang_h)
            lang_hs.append(lang_h)

            # decode words using the inverse of the word embedding matrix
            out = self.decoder(lang_h)
            scores = F.linear(out, self.word_encoder.weight).div(temperature)
            # subtract constant to avoid overflows in exponentiation
            scores = scores.add(-scores.max().item()).squeeze(0)

            mask = Variable(self.special_token_mask).to(device)
            scores = scores.add(mask)

            logprob = F.log_softmax(scores)
            score += logprob[word[0]].item()
            inpt = Variable(word).to(device)

        # update the hidden state with the <eos> token
        inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
        lang_h = self.writer(inpt_emb, lang_h)
        lang_hs.append(lang_h)

        # add batch dimension back
        lang_h = lang_h.unsqueeze(1)

        return score, lang_h, torch.cat(lang_hs)

    def forward_context(self, ctx):
        """Run context encoder."""
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #ctx = ctx.to(device)
        return self.ctx_encoder(ctx.to(device))

    def forward_lm(self, inpt, lang_h, ctx_h):
        """Run forward pass for language modeling."""
        # embed words
        inpt_emb = self.word_encoder(inpt.to(device)).to(device)

        # append the context embedding to every input word embedding
        ctx_h_rep = ctx_h.narrow(0, ctx_h.size(0) - 1, 1).expand(
            inpt.size(0), ctx_h.size(1), ctx_h.size(2)).to(device)
        inpt_emb = torch.cat([inpt_emb, ctx_h_rep], 2)

        inpt_emb = self.dropout(inpt_emb)

        out, _ = self.reader(inpt_emb, lang_h)
        decoded = self.decoder(out.view(-1, out.size(2)))

        # tie weights between word embedding/decoding
        decoded = F.linear(decoded, self.word_encoder.weight.to(device))

        return decoded.view(out.size(0), out.size(1), decoded.size(1)), out
