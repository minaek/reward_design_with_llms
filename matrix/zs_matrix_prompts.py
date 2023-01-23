import pickle as pkl
from tkinter import W


def battle():
    s = (
        "We have a two-player game where P1 and P2 can choose one of these options.\n"
        + "Options:\n"
        + "A. if action1(P1) and action1(P2) => P1 gets reward of 2, P2 gets reward of 1.\n"
        + "B. if action1(P1) and action2(P2) => P1 gets reward of 0, P2 gets reward of 0.\n"
        + "C. if action2(P1) and action1(P2) => P1 gets reward of 0, P2 gets reward of 0.\n"
        + "D. if action2(P1) and action2(P2) => P1 gets reward of 1, P2 gets reward of 2.\n"
    )
    return s


def battle_scrambled():
    s = (
        "We have a two-player game where P1 and P2 can choose one of these options.\n"
        + "Options:\n"
        + "A. if action1(P1) and action1(P2) => P1 gets reward of 0, P2 gets reward of 0.\n"
        + "B. if action1(P1) and action2(P2) => P1 gets reward of 0, P2 gets reward of 0.\n"
        + "C. if action2(P1) and action1(P2) => P1 gets reward of 2, P2 gets reward of 1.\n"
        + "D. if action2(P1) and action2(P2) => P1 gets reward of 1, P2 gets reward of 2.\n"
    )
    return s


def prisoners():
    s = (
        "We have a two-player game where P1 and P2 can choose one of these options.\n"
        + "Options:\n"
        + "A. if action1(P1) and action1(P2) => P1 gets reward of 3, P2 gets reward of 3.\n"
        + "B. if action1(P1) and action2(P2) => P1 gets reward of 0, P2 gets reward of 5.\n"
        + "C. if action2(P1) and action1(P2) => P1 gets reward of 5, P2 gets reward of 0.\n"
        + "D. if action2(P1) and action2(P2) => P1 gets reward of 1, P2 gets reward of 1.\n"
    )
    return s


def prisoners_scrambled():
    s = (
        "We have a two-player game where P1 and P2 can choose one of these options.\n"
        + "Options:\n"
        + "A. if action1(P1) and action1(P2) => P1 gets reward of 5, P2 gets reward of 0.\n"
        + "B. if action1(P1) and action2(P2) => P1 gets reward of 0, P2 gets reward of 5.\n"
        + "C. if action2(P1) and action1(P2) => P1 gets reward of 1, P2 gets reward of 1.\n"
        + "D. if action2(P1) and action2(P2) => P1 gets reward of 3, P2 gets reward of 3.\n"
    )
    return s


def stag():
    s = (
        "We have a two-player game where P1 and P2 can choose one of these options.\n"
        + "Options:\n"
        + "A. if action1(P1) and action1(P2) => P1 gets reward of 2, P2 gets reward of 2.\n"
        + "B. if action1(P1) and action2(P2) => P1 gets reward of -3, P2 gets reward of 1.\n"
        + "C. if action2(P1) and action1(P2) => P1 gets reward of 1, P2 gets reward of -3.\n"
        + "D. if action2(P1) and action2(P2) => P1 gets reward of 1, P2 gets reward of 1.\n"
    )
    return s


def stag_scrambled():
    s = (
        "We have a two-player game where P1 and P2 can choose one of these options.\n"
        + "Options:\n"
        + "A. if action1(P1) and action1(P2) => P1 gets reward of 1, P2 gets reward of 1.\n"
        + "B. if action1(P1) and action2(P2) => P1 gets reward of 2, P2 gets reward of 2.\n"
        + "C. if action2(P1) and action1(P2) => P1 gets reward of 1, P2 gets reward of -3.\n"
        + "D. if action2(P1) and action2(P2) => P1 gets reward of -3, P2 gets reward of 1.\n"
    )
    return s


def chicken():
    s = (
        "We have a two-player game where P1 and P2 can choose one of these options.\n"
        + "Options:\n"
        + "A. if action1(P1) and action1(P2) => P1 gets reward of 2, P2 gets reward of 2.\n"
        + "B. if action1(P1) and action2(P2) => P1 gets reward of 1, P2 gets reward of 3.\n"
        + "C. if action2(P1) and action1(P2) => P1 gets reward of 3, P2 gets reward of 1.\n"
        + "D. if action2(P1) and action2(P2) => P1 gets reward of 0, P2 gets reward of 0.\n"
    )
    return s


def chicken_scrambled():
    s = (
        "We have a two-player game where P1 and P2 can choose one of these options.\n"
        + "Options:\n"
        + "A. if action1(P1) and action1(P2) => P1 gets reward of 3, P2 gets reward of 1.\n"
        + "B. if action1(P1) and action2(P2) => P1 gets reward of 2, P2 gets reward of 2.\n"
        + "C. if action2(P1) and action1(P2) => P1 gets reward of 1, P2 gets reward of 3.\n"
        + "D. if action2(P1) and action2(P2) => P1 gets reward of 0, P2 gets reward of 0.\n"
    )
    return s


str2game = {"battle": battle, "stag": stag, "prisoners": prisoners, "chicken": chicken}
str2game_scrambled = {
    "battle": battle_scrambled,
    "stag": stag_scrambled,
    "prisoners": prisoners_scrambled,
    "chicken": chicken_scrambled,
}
blank_str = "Which option(s) should players choose? Let's think step by step:"


def rawlsian_fairness(game, scrambled=False, blank=False):
    if scrambled:
        game_prompt = str2game_scrambled[game]() + "\n"
    else:
        game_prompt = str2game[game]() + "\n"
    s = (
        "Which option(s) result in Rawlsian fair rewards? Let's think step by step:\n"
        + "Rawlsian fairness is"
    )
    if blank:
        return game_prompt + blank_str
    else:
        return game_prompt + s


def pareto(game, scrambled=False, blank=False):
    if scrambled:
        game_prompt = str2game_scrambled[game]() + "\n"
    else:
        game_prompt = str2game[game]() + "\n"
    s = (
        "Which option(s) are Pareto-optimal? Let's think step by step:\n"
        + "An outcome is Pareto-optimal if"
    )
    if blank:
        return game_prompt + blank_str
    else:
        return game_prompt + s


def equality(game, scrambled=False, blank=False):
    if scrambled:
        game_prompt = str2game_scrambled[game]() + "\n"
    else:
        game_prompt = str2game[game]() + "\n"
    s = (
        "Which option(s) result in equality of rewards? Let's think step by step:\n"
        + "Equality of rewards is"
    )
    if blank:
        return game_prompt + blank_str
    else:
        return game_prompt + s


def welfare(game, scrambled=False, blank=False):
    if scrambled:
        game_prompt = str2game_scrambled[game]() + "\n"
    else:
        game_prompt = str2game[game]() + "\n"
    s = (
        "Which option(s) result in the greatest total welfare? Let's think step by step:\n"
        + "Total welfare is"
    )
    if blank:
        return game_prompt + blank_str
    else:
        return game_prompt + s


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="matrix games")
    parser.add_argument(
        "--game",
        type=str,
        default="battle",
        help="[battle, chicken, stag, prisoners]",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="welfare",
        help="[welfare, equality, rawlsian, pareto]",
    )
    parser.add_argument(
        "--blank",
        action="store_true",
        default=False,
        help="Including this flag will omit the user's objective in the prompt (used as our baseline)",
    )
    parser.add_argument(
        "--scrambled",
        action="store_true",
        default=False,
        help="Including this flag will scramble the order of the joint outcomes.",
    )
    args = parser.parse_args()
    if args.objective == "welfare":
        s = welfare(args.game, scrambled=args.scrambled, blank=args.blank)
    elif args.objective == "equality":
        s = equality(args.game, scrambled=args.scrambled, blank=args.blank)
    elif args.objective == "rawlsian":
        s = rawlsian_fairness(args.game, scrambled=args.scrambled, blank=args.blank)
    elif args.objective == "pareto":
        s = pareto(args.game, scrambled=args.scrambled, blank=args.blank)
    print(s)
