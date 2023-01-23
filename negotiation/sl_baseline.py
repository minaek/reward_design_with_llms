import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
import torch.optim as optim

from coarse_dialogue_acts.corpus import ActCorpus
from models import modules
from utils.domain import ObjectDivisionDomain
import base_prompts
import utils.utils as utils


class RNN(nn.Module):
    def __init__(self, word_dict, context_dict):
        super(RNN, self).__init__()

        self.nhid_ctx = 64
        self.nhid_lang = 128
        self.nembed_word = 128
        self.word_dict = word_dict
        self.context_dict = context_dict
        domain = ObjectDivisionDomain()
        self.ctx_encoder = modules.MlpContextEncoder(
            len(self.context_dict),
            domain.input_length() * 2,
            64,
            self.nhid_ctx,
            0.1,
            None,
        )  # multiplying 2 because feeding in Alice and Bob's context
        # embedding for words
        self.word_encoder = nn.Embedding(len(word_dict), self.nembed_word)
        self.reader = nn.GRU(
            input_size=self.nhid_ctx + self.nembed_word,
            hidden_size=self.nhid_lang,
            bias=True,
        )
        # self.outcome_encoder = modules.MlpOutcomeEncoder(
        #    n=10, k=None, nembed=64, nhid=None, init_range=0.1, device_id=None
        # )
        self.outcome_encoder = nn.Sequential(
            nn.Tanh(), nn.Linear(self.nhid_lang + 3, 2)
        )

    def word2var(self, word):
        """Creates a variable from a given word."""
        x = torch.Tensor(1).fill_(self.word_dict.get_idx(word)).long()
        return Variable(x)

    def zero_hid(self, bsz, nhid=None, copies=None):
        """A helper function to create an zero hidden state."""
        nhid = self.nhid_lang if nhid is None else nhid
        copies = 1 if copies is None else copies
        hid = torch.zeros(copies, bsz, nhid)
        return Variable(hid)

    def _encode(self, inpt, dictionary):
        """A helper function that encodes the passed in words using the dictionary.

        inpt: is a list of strings.
        dictionary: prebuild mapping, see Dictionary in data.py
        """
        encoded = torch.LongTensor(dictionary.w2i(inpt)).unsqueeze(1)
        return encoded

    def forward_ctx(self, ctx):
        # encoded context
        ctx = self._encode(ctx, self.context_dict)
        # hidded state of context
        ctx_h = self.ctx_encoder(Variable(ctx))
        return ctx_h

    def forward_lm(self, inpt, lang_h, ctx_h):
        """Run forward pass for language modeling."""
        # embed words
        inpt_emb = self.word_encoder(inpt)  # (2,1,128)

        # append the context embedding to every input word embedding
        # ctx_h (1,1,64)
        ctx_h_rep = ctx_h.expand(
            inpt_emb.size(0), ctx_h.size(1), ctx_h.size(2)
        )  # (2,1,64)
        inpt_emb = torch.cat(
            [inpt_emb, ctx_h_rep], 2
        )  # [(2,1,128), (2,1,64)] -> (2,1,192)

        _, out = self.reader(inpt_emb, lang_h)  # (1, 1, 128)
        return out

    def forward_outcome(self, outcome, lang_h):
        # encode output
        outcome = self._encode(outcome, self.context_dict)  # (3,1)
        lang_h = lang_h.squeeze(0).T  # (128,1)
        outcome = torch.cat([lang_h, outcome], 0).squeeze()  # (131,)
        out = self.outcome_encoder(Variable(outcome))
        return out

    def forward(self, ctx, inputs, outcome):
        """Main forward function."""
        # forward context
        ctx_h = self.forward_ctx(ctx)
        # forward inputs
        lang_h = self.zero_hid(1)  # do I initialize this here, probably not, (1,1,128)
        for dict in inputs:
            inpt = Variable(self._encode([dict["cda"]], self.word_dict))
            prefix = self.word2var(dict["prefix_token"]).unsqueeze(0)
            inpt = torch.cat([prefix, inpt])
            lang_h = self.forward_lm(inpt, lang_h, ctx_h)
        # forward ouptputs
        out = self.forward_outcome(outcome, lang_h)
        return out


def get_data_i(chunk, extract_label=True):
    data_i = {"ctx": None, "inputs": [], "outcome": None, "label": None}
    # extract context
    chunk[0] = chunk[0].strip()
    alice_counts = [x[-1] for x in chunk[0].split(" ") if "count" in x]
    alice_vals = [x[-2] for x in chunk[0].split(" ") if "value" in x]
    alice_ctx = [val for pair in zip(alice_counts, alice_vals) for val in pair]
    chunk[1] = chunk[1].strip()
    bob_counts = [x[-1] for x in chunk[1].split(" ") if "count" in x]
    bob_vals = [x[-2] for x in chunk[1].split(" ") if "value" in x]
    bob_ctx = [val for pair in zip(bob_counts, bob_vals) for val in pair]
    for c in alice_ctx + bob_ctx:
        if not c.isdigit():
            raise ValueError
    data_i["ctx"] = alice_ctx + bob_ctx

    # extract inputs
    dialogue_idxs = [j for j in range(len(chunk)) if "----" in chunk[j]]
    assert len(dialogue_idxs) == 2
    dialogue = chunk[dialogue_idxs[0] + 1 : dialogue_idxs[1]]
    for line in dialogue:
        speaker = line[: line.find(":")].strip()
        prefix_token = "YOU:" if speaker == "Alice" else "THEM:"
        cda = line[line.find(":") + 1 :].strip()
        cda = cda.replace("book", "item0")
        cda = cda.replace("hat", "item1")
        cda = cda.replace("ball", "item2")
        data_i["inputs"].append({"cda": cda, "prefix_token": prefix_token})

    # extract outcome
    if "Agreement!" in chunk[dialogue_idxs[-1] + 1]:
        is_agreement = "1"
    elif "Disagreement?!" in chunk[dialogue_idxs[-1] + 1]:
        is_agreement = "0"
    else:
        raise ValueError
    alice_points = None
    for word in chunk[dialogue_idxs[-1] + 2].split(" "):
        if word.isdigit():
            alice_points = word
    assert alice_points is not None
    bob_points = None
    for word in chunk[dialogue_idxs[-1] + 3].split(" "):
        if word.isdigit():
            bob_points = word
    assert bob_points is not None
    outcome = [is_agreement, alice_points, bob_points]
    data_i["outcome"] = outcome

    # extract label
    if extract_label:
        label = None
        label_idx = None
        for l, line in enumerate(chunk):
            if "Is Alice a " in line or "Is Alice an " in line:
                label_idx = l + 1
                break
        if "Yes" in chunk[label_idx].split(" ")[0]:
            label = 1
        elif "No" in chunk[label_idx].split(" ")[0]:
            label = 0
        else:
            raise ValueError
        data_i["label"] = label
    return data_i


def get_data(style):
    if style == "stubborn":
        prompts = base_prompts.stubborn()[0].strip().split("\n")
    elif style == "competitive":
        prompts = base_prompts.competitive()[0].strip().split("\n")
    elif style == "pushover":
        prompts = base_prompts.pushover()[0].strip().split("\n")
    elif style == "versatile":
        prompts = base_prompts.versatile()[0].strip().split("\n")
    else:
        raise ValueError

    idxs = [i for i in range(len(prompts)) if "=========" in prompts[i]]
    chunks = []
    for i, idx in enumerate(idxs):
        if i < len(idxs) - 1:
            chunks.append(prompts[idx + 1 : idxs[i + 1]])
        else:
            chunks.append(prompts[idx + 1 :])

    data = []
    for i in range(len(chunks)):
        data_i = get_data_i(chunks[i])
        data.append(data_i)
    return data


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(keepdim=True)[0]

    return y == top_pred


def train(data, corpus, style):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    model = RNN(corpus.word_dict, corpus.context_dict)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    epoch = 0
    while True:
        model.train()
        epoch_loss, epoch_acc = 0, 0
        for i in range(len(data)):
            optimizer.zero_grad()
            out = model.forward(
                ctx=data[i]["ctx"],
                inputs=data[i]["inputs"],
                outcome=data[i]["outcome"],
            )
            label = torch.tensor(data[i]["label"])
            # print(out.argmax(keepdim=True)[0], label)
            loss = criterion(out, label.long())
            acc = calculate_accuracy(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        epoch_loss /= len(data)
        epoch_acc /= len(data)
        print(f"Epoch: {epoch+1:02}")
        print(f"\tTrain Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc*100:.2f}%")
        epoch += 1
        if epoch_acc == 1.0 or epoch > 100:
            break

    # assert epoch_acc == 1.0
    output_model_file = f"trained_models/sl_baseline/{style}.th"
    utils.save_model(model, output_model_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="sl baseline")
    parser.add_argument("--style", type=str, default="stubborn", help="For ICLR 2022")
    args = parser.parse_args()
    data = get_data(args.style)
    corpus = ActCorpus(
        path="data",
        freq_cutoff=20,
        train="train.txt",
        valid="val.txt",
        test="test.txt",
        verbose=True,
    )
    train(data, corpus, args.style)
