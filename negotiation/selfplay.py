# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Selfplaying util.
"""

import argparse

from utils.dialog import Dialog, DialogLogger
from utils.utils import ContextGenerator, get_agent_type, set_seed, load_model


class SelfPlay(object):
    """Selfplay runner."""

    def __init__(self, dialog, ctx_gen, args, logger=None):
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.args = args
        self.logger = logger if logger else DialogLogger()

    def run(self):
        n = 0
        forced = False
        # goes through the list of contexes and kicks off a dialogue
        for ctxs in self.ctx_gen.iter(nepoch=1):
            n += 1
            self.logger.dump("=" * 80, forced=forced)
            self.dialog.run(
                ctxs, self.logger, update=(False, False), forced=forced, training=False
            )
            self.logger.dump("=" * 80, forced=forced)
            self.logger.dump("%d: %s" % (n, self.dialog.show_metrics()), forced=True)
            self.logger.dump("", forced=forced)


def main():
    parser = argparse.ArgumentParser(description="selfplaying script")
    parser.add_argument("--alice_model", type=str, help="Alice model file")
    parser.add_argument("--bob_model", type=str, help="Bob model file")
    parser.add_argument(
        "--context_file",
        type=str,
        default="data/selfplay_lite.txt",
        help="context file",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="temperature")
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
        "--max_turns", type=int, default=20, help="maximum number of turns in a dialog"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="log successful dialogs to file for training",
    )
    parser.add_argument(
        "--smart_alice",
        action="store_true",
        default=False,
        help="make Alice smart again",
    )
    parser.add_argument(
        "--fast_rollout",
        action="store_true",
        default=False,
        help="to use faster rollouts",
    )
    parser.add_argument(
        "--rollout_bsz", type=int, default=100, help="rollout batch size"
    )
    parser.add_argument(
        "--rollout_count_threshold", type=int, default=3, help="rollout count threshold"
    )
    parser.add_argument(
        "--smart_bob", action="store_true", default=False, help="make Bob smart again"
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default="data/train.txt",
        help="file with the reference text",
    )
    parser.add_argument(
        "--domain", type=str, default="object_division", help="domain for the dialogue"
    )
    parser.add_argument("--cuda", action="store_true", default=False, help="use CUDA")
    parser.add_argument("--eps", type=float, default=0.0, help="eps greedy")
    parser.add_argument(
        "--novelty_model",
        type=str,
        default="trained_models/sl/sl_seed1.th",
        help="Model used to score novelty of dialogue",
    )
    parser.add_argument("--style", type=str, default="stubborn", help="For ICLR 2023")
    parser.add_argument("--model", type=str, default="rl", help="For ICLR 2023")
    seeds = [0, 1, 2]
    for seed in seeds:
        args = parser.parse_args()
        args.verbose = True
        args.seed = seed
        name = f"{args.model}_{args.style}"
        args.alice_model = f"trained_models/{name}/alice_epoch0_seed{args.seed}.th"
        args.bob_model = "trained_models/sl/sl1.th"
        args.novelty_model = args.bob_model
        args.log_file = f"eval_logs/{name}{args.seed}.txt"

        set_seed(args.seed)

        alice_model = load_model(args.alice_model, cuda=args.cuda)
        alice_ty = get_agent_type(alice_model)
        alice = alice_ty(alice_model, args, name="Alice", train=False)

        bob_model = load_model(args.bob_model, cuda=args.cuda)
        bob_ty = get_agent_type(bob_model)
        bob = bob_ty(bob_model, args, name="Bob", train=False)

        args.novelty_model = load_model(args.novelty_model, cuda=args.cuda)

        if not args.cuda:
            bob.model.device_id = None
            bob.model.ctx_encoder.device_id = None
            alice.model.device_id = None
            alice.model.ctx_encoder.device_id = None

        dialog = Dialog([alice, bob], args)
        logger = DialogLogger(log_file=args.log_file)
        ctx_gen = ContextGenerator(args.context_file)

        selfplay = SelfPlay(dialog, ctx_gen, args, logger)
        selfplay.run()

    # means = []
    # stds = []
    # for seed in seeds:
    #    mean, std = eval_selfplay(
    #        args.model, args.style, seed, args.include_selfish_reward, args.plus_n
    #    )
    #    means.append(mean)
    #    stds.append(std)
    # print(means)
    # print("mean: ", np.mean(means), "std: ", np.mean(stds))


if __name__ == "__main__":
    main()
