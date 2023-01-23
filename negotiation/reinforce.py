# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Reinforcement learning via Policy Gradient (REINFORCE).
"""

import argparse
import os
import pickle as pkl

import utils.utils as utils
from utils.utils import get_agent_type
from utils.dialog import Dialog, DialogLogger
from utils.engine import Engine
from utils.utils import ContextGenerator
from utils.rewards import get_reward_type

from sl_baseline import RNN


class Reinforce(object):
    """Facilitates a dialogue between two agents and constantly updates them."""

    def __init__(self, dialog, ctx_gen, args, engines, corpus, logger=None, name="rl"):
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.args = args
        self.engines = engines
        self.corpus = corpus
        self.logger = logger if logger else DialogLogger()
        self.name = name

    def run(self):
        """Entry point of the training."""
        # Assumes that both models are running on the same device_id
        assert self.engines[0].device_id == self.engines[1].device_id
        n = 0
        for e in range(self.args.nepoch):
            for ctxs in self.ctx_gen.iter(nepoch=1):
                n += 1
                self.logger.dump("=" * 80)
                self.dialog.test_prompt = "=" * 80 + "\n"
                # run dialogue, it is responsible for reinforcing the agents
                skip = self.dialog.run(
                    ctxs, self.logger, update=(True, False), forced=False, training=True
                )
                if skip:
                    continue
                self.logger.dump("=" * 80)
                self.dialog.test_prompt += "=" * 80
                self.logger.dump("")
                self.dialog.test_prompt += ""

                # if self.args.model == "gpt3" and n % 50 == 0:
                #    time.sleep(120)
                if n % 100 == 0:
                    self.logger.dump(
                        "%d: %s" % (n, self.dialog.show_metrics()), forced=True
                    )

            self.logger.dump("final: %s" % self.dialog.show_metrics(), forced=True)
            output_model_file = (
                f"{self.args.trained_model_path}/alice_epoch{e}_seed{self.args.seed}.th"
            )
            if not os.path.exists(self.args.trained_model_path):
                os.mkdir(self.args.trained_model_path)
            utils.save_model(self.dialog.agents[0].model, output_model_file)
        with open(f"logs/{self.name}_answers_{self.args.seed}.pkl", "wb") as f:
            pkl.dump(self.dialog.gpt3_answers, f)
        with open(f"logs/{self.name}_answers_{self.args.seed}.txt", "w") as f:
            for (neg, pred_str, pred) in self.dialog.gpt3_answers:
                f.write(neg + "\n")
                f.write(pred_str + "\n\n")


def main():
    parser = argparse.ArgumentParser(description="Reinforce")
    parser.add_argument(
        "--data", type=str, default="./data", help="location of the data corpus"
    )
    parser.add_argument(
        "--model_type", type=str, default="cda_rnn_model", help="model type"
    )
    parser.add_argument(
        "--unk_threshold",
        type=int,
        default=20,
        help="minimum word frequency to be in dictionary",
    )
    parser.add_argument(
        "--alice_model",
        type=str,
        default="trained_models/sl/sl1.th",
        help="Alice model file",
    )
    parser.add_argument(
        "--bob_model",
        type=str,
        default="trained_models/sl/sl1.th",
        help="Bob model file",
    )
    parser.add_argument("--output_model_file", type=str, help="output model file")
    parser.add_argument("--output_model_path", type=str, help="output model file")
    parser.add_argument("--trained_model_path", type=str, help="output model file")
    parser.add_argument(
        "--context_file",
        type=str,
        default="data/selfplay_lite.txt",
        help="context file",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="temperature")
    parser.add_argument("--cuda", action="store_true", default=False, help="use CUDA")
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="print out converations"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--score_threshold",
        type=int,
        default=6,
        help="successful dialog should have more than score_threshold in score",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="log successful dialogs to file for training",
    )
    parser.add_argument(
        "--smart_bob", action="store_true", default=False, help="make Bob smart again"
    )
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--eps", type=float, default=0.5, help="eps greedy")
    parser.add_argument(
        "--nesterov",
        action="store_true",
        default=False,
        help="enable nesterov momentum",
    )
    parser.add_argument("--momentum", type=float, default=0.1, help="momentum for sgd")
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate")
    parser.add_argument("--clip", type=float, default=0.5, help="gradient clip")
    parser.add_argument("--rl_lr", type=float, default=0.1, help="RL learning rate")
    parser.add_argument("--rl_clip", type=float, default=1, help="RL gradient clip")
    parser.add_argument(
        "--ref_text",
        type=str,
        default="data/train.txt",
        help="file with the reference text",
    )
    parser.add_argument("--bsz", type=int, default=8, help="batch size")
    parser.add_argument(
        "--sv_train_freq", type=int, default=-1, help="supervision train frequency"
    )
    parser.add_argument("--nepoch", type=int, default=1, help="number of epochs")
    parser.add_argument(
        "--visual", action="store_true", default=False, help="plot graphs"
    )
    parser.add_argument(
        "--domain", type=str, default="object_division", help="domain for the dialogue"
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default="utility",
        help="Type of Reward function to run Reinforce with",
    )
    parser.add_argument(
        "--novelty_model",
        type=str,
        default="trained_models/sl/sl1.th",
        help="Model used to score novelty of dialogue",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="train.txt",
        help="file that contains training data",
    )
    parser.add_argument(
        "--val_file", type=str, default="val.txt", help="file that contains val data"
    )
    parser.add_argument(
        "--test_file", type=str, default="test.txt", help="file that contains test data"
    )
    # ICLR Arguments
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="sched",
        help="The training schedule for coordinating SL and RL updates",
    )
    parser.add_argument(
        "--p_sl_update",
        type=float,
        default=0.3,
        help="The probability of performing an sl update when using a RandTrainer",
    )
    parser.add_argument(
        "--self_play_updates",
        type=int,
        default=1,
        help="The number of self play updates to perform per round of training",
    )
    parser.add_argument(
        "--supervised_updates",
        type=int,
        default=0,
        help="The number of supervised updates to perform per round of training",
    )
    parser.add_argument(
        "--p_freeze",
        type=float,
        default=1,
        help="The probability of freezing Bob during training. Freezing impacts both SL and RL updates",
    )
    # Bob SL Training Args
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=35,
        help="The number of epochs to train Bob after human annotation",
    )
    # parser.add_argument('--bsz', type=int, default=25,
    #                    help='batch size')
    parser.add_argument(
        "--decay_rate",
        type=float,
        default=9.0,
        help="decrease learning rate by this factor",
    )
    parser.add_argument(
        "--decay_every",
        type=int,
        default=1,
        help="decrease learning rate after decay_every epochs",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="min threshold for learning rate annealing",
    )
    parser.add_argument(
        "--nembed_word", type=int, default=256, help="size of word embeddings"
    )
    parser.add_argument(
        "--nembed_ctx", type=int, default=64, help="size of context embeddings"
    )
    parser.add_argument(
        "--nhid_lang",
        type=int,
        default=128,
        help="size of the hidden state for the language module",
    )
    parser.add_argument(
        "--nhid_ctx",
        type=int,
        default=64,
        help="size of the hidden state for the context module",
    )
    parser.add_argument(
        "--nhid_strat",
        type=int,
        default=128,
        help="size of the hidden state for the strategy module",
    )
    parser.add_argument(
        "--nhid_attn",
        type=int,
        default=256,
        help="size of the hidden state for the attention module",
    )
    parser.add_argument(
        "--nhid_sel",
        type=int,
        default=256,
        help="size of the hidden state for the selection module",
    )
    parser.add_argument(
        "--sel_weight", type=float, default=0.5, help="selection weight"
    )
    parser.add_argument(
        "--init_range", type=float, default=0.1, help="initialization range"
    )
    parser.add_argument(
        "--rnn_ctx_encoder",
        action="store_true",
        default=False,
        help="weather to use RNN for encoding the context",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate in embedding layer"
    )
    parser.add_argument("--style", type=str, default="stubborn", help="For ICLR 2022")
    parser.add_argument(
        "--model",
        type=str,
        default="rl",
        help="For ICLR 2022: {gpt3, sl_baseline, rl, gpt2}",
    )

    seeds = [0, 1, 2]
    for seed in seeds:
        args = parser.parse_args()
        args.nepoch = 1
        name = f"{args.model}_{args.style}"
        print("NAME: ", name)
        args.trained_model_path = f"trained_models/{name}"
        args.seed = seed
        args.log_file = f"logs/{name}{args.seed}.txt"

        device_id = utils.use_cuda(args.cuda)
        utils.set_seed(args.seed)

        reward_ty = get_reward_type(args.reward_type)
        rewarder = reward_ty()

        alice_model = utils.load_model(args.alice_model, cuda=args.cuda)
        alice_ty = get_agent_type(alice_model)
        # Alice is a RL based agent, meaning that she will be learning while selfplaying
        alice = alice_ty(alice_model, args, name="Alice", train=True, rewarder=rewarder)

        # we keep Bob frozen, i.e. we don't update his parameters
        bob_model = utils.load_model(args.bob_model, cuda=args.cuda)
        bob_ty = get_agent_type(bob_model)
        bob = bob_ty(bob_model, args, name="Bob", train=False, rewarder=rewarder)

        args.novelty_model = utils.load_model(args.novelty_model, cuda=args.cuda)

        if not args.cuda:
            bob.model.device_id = None
            bob.model.ctx_encoder.device_id = None
            alice.model.device_id = None
            alice.model.ctx_encoder.device_id = None

        dialog = Dialog([alice, bob], args)
        logger = DialogLogger(log_file=args.log_file)
        ctx_gen = ContextGenerator(args.context_file)

        assert alice_model.corpus_ty == bob_model.corpus_ty
        corpus = alice_model.corpus_ty(
            args.data,
            freq_cutoff=args.unk_threshold,
            train=args.train_file,
            valid=args.val_file,
            test=args.test_file,
            verbose=True,
        )
        engines = [
            Engine(alice_model, args, device_id, verbose=False),
            Engine(bob_model, args, device_id, verbose=False),
        ]

        reinforce = Reinforce(dialog, ctx_gen, args, engines, corpus, logger, name)
        reinforce.run()


if __name__ == "__main__":
    main()
