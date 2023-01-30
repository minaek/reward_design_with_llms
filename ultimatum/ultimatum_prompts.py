import pickle as pkl
import numpy as np

is_desirable = {0: "Yes", 1: "No"}


def get_low_high_percentage_prompt(threshold):
    base_prompt = "P1 and P2 are playing the Ultimatum Game. P1 proposes how they should split $10  and P2 can either accept or reject. If P2 accepts, then the deal is done. If P2 rejects, then both parties get nothing. "
    with open(f"low_high_percentage_{threshold}.pkl", "rb") as f:
        data = pkl.load(f)
        train, test = data["train"], data["test"]
    prompts = []
    for test_datum in test:
        prompt = base_prompt
        for datum in train:
            example = f"P1 proposes a split of ${datum[0]} for P1 and ${datum[1]} for P2. P2 rejected this offer. A desirable outcome is defined as one where P2 punishes P1's selfish behavior. Is the outcome desirable?\n{is_desirable[datum[-1]]}"
            prompt += "\n\n" + example

        test = f"P1 proposes a split of ${test_datum[0]} for P1 and ${test_datum[1]} for P2. P2 rejected this offer. A desirable outcome is defined as one where P2 punishes P1's selfish behavior. Is the outcome desirable?"
        prompt += "\n\n" + test
        prompts.append(prompt)
    return prompts


def get_low_high_payoff_prompt(threshold):
    base_prompt = "P1 and P2 are playing the Ultimatum Game. P1 proposes how they should split $10  and P2 can either accept or reject. If P2 accepts, then the deal is done. If P2 rejects, then both parties get nothing. "
    with open(f"low_high_payoff_{threshold}.pkl", "rb") as f:
        data = pkl.load(f)
        train, test = data["train"], data["test"]
    prompts = []
    for test_datum in test:
        prompt = base_prompt
        for datum in train:
            example = f"P1 proposes a split of ${datum[0]} for P1 and ${datum[1]} for P2. P2 rejected this offer. A desirable outcome is defined as one where P2 punishes P1's selfish behavior unless P2 gets a sizeable amount of money. Is the outcome desirable?\n{is_desirable[datum[-1]]}"
            prompt += "\n\n" + example

        test = f"P1 proposes a split of ${test_datum[0]} for P1 and ${test_datum[1]} for P2. P2 rejected this offer. A desirable outcome is defined as one where P2 punishes P1's selfish behavior unless P2 gets a sizeable amount of money. Is the outcome desirable?"
        prompt += "\n\n" + test
        prompts.append(prompt)
    return prompts


def get_inequity_aversion_prompt():
    base_prompt = "P1 and P2 are playing the Ultimatum Game. P1 proposes how they should split $10  and P2 can either accept or reject. If P2 accepts, then the deal is done. If P2 rejects, then both parties get nothing. "
    with open(f"inequity_aversion_None.pkl", "rb") as f:
        data = pkl.load(f)
        train, test = data["train"], data["test"]
    prompts = []
    for test_datum in test:
        prompt = base_prompt
        for datum in train:
            example = f"P1 proposes a split of ${np.around(datum[0],1)} for P1 and ${np.around(datum[1],1)} for P2. P2 rejected this offer. A desirable outcome is defined as one where P2 punishes P1's selfish behavior. Is the outcome desirable?\n{is_desirable[datum[-1]]}"
            prompt += "\n\n" + example

        test = f"P1 proposes a split of ${test_datum[0]} for P1 and ${test_datum[1]} for P2. P2 rejected this offer. A desirable outcome is defined as one where P2 punishes P1's selfish behavior. Is the outcome desirable?"
        prompt += "\n\n" + test
        prompts.append(prompt)
    return prompts


def get_low_high_percentage_prompt_shorter(threshold):
    base_prompt = "P1 and P2 are playing the Ultimatum Game. P1 proposes how they should split $10  and P2 can either accept or reject. If P2 accepts, then the deal is done. If P2 rejects, then both parties get nothing. "
    train = []
    if threshold == 0.3:
        train = [[9.78, 0.22]]
        cot = f"P2 receives ${train[0][1]} < $3 so P2 should reject this offer. Therefore, the outcome is desirable"
    elif threshold == 0.6:
        train = [[4.71, 5.29]]
        cot = f"P2 receives ${train[0][1]} < $6 so P2 should reject this offer. Therefore, the outcome is desirable"
    else:
        raise ValueError
    with open(f"low_high_percentage_{threshold}.pkl", "rb") as f:
        data = pkl.load(f)
        test = data["test"]
    prompts = []
    for test_datum in test:
        prompt = base_prompt
        for datum in train:
            example = f"P1 proposes a split of ${datum[0]} for P1 and ${datum[1]} for P2. P2 rejected this offer. A desirable outcome is defined as one where P2 punishes P1's selfish behavior. Is the outcome desirable? Let's think step by step:\n"
            prompt += "\n\n" + example + cot

        test = f"P1 proposes a split of ${test_datum[0]} for P1 and ${test_datum[1]} for P2. P2 rejected this offer. A desirable outcome is defined as one where P2 punishes P1's selfish behavior. Is the outcome desirable? Let's think step by step:"
        prompt += "\n\n" + test
        prompts.append(prompt)
    return prompts


def get_low_high_payoff_prompt_shorter(threshold):
    base_prompt = "P1 and P2 are playing the Ultimatum Game. P1 proposes how they should split a sum of money and P2 can either accept or reject. If P2 accepts, then the deal is done. If P2 rejects, then both parties get nothing. "
    train = []
    if threshold == 10:
        train = [[5.4, 4.6]]
        cot = f"P2 gets ${train[0][1]} but P2 thinks anything below $10 is not sizeable. {train[0][1]} is less than 10 so P2 should reject the offer. Therefore, the outcome is desirable."
    elif threshold == 100:
        train = [[700, 300]]
        cot = f"P2 gets ${train[0][1]} but P2 thinks anything below $100 is not sizeable. {train[0][1]} is more than 100 so P2 should accept the offer. Therefore, the outcome is not desirable."
    else:
        raise ValueError
    with open(f"low_high_payoff_{threshold}.pkl", "rb") as f:
        data = pkl.load(f)
        test = data["test"]
    prompts = []
    for test_datum in test:
        prompt = base_prompt
        for datum in train:
            example = f"P1 proposes a split of ${datum[0]} for P1 and ${datum[1]} for P2. P2 rejected this offer. A desirable outcome is defined as one where P2 punishes P1's selfish behavior unless P2 gets a sizeable amount of money. Is the outcome desirable? Let's think step by step:\n"
            prompt += "\n\n" + example + cot
        test = f"P1 proposes a split of ${test_datum[0]} for P1 and ${test_datum[1]} for P2. P2 rejected this offer. A desirable outcome is defined as one where P2 punishes P1's selfish behavior unless P2 gets a sizeable amount of money. Is the outcome desirable? Let's think step by step:"
        prompt += "\n\n" + test
        prompts.append(prompt)
    return prompts


def get_inequity_aversion_prompt_shorter():
    base_prompt = "P1 and P2 are playing the Ultimatum Game. P1 proposes how they should split $10  and P2 can either accept or reject. If P2 accepts, then the deal is done. If P2 rejects, then both parties get nothing. "
    with open(f"inequity_aversion_None.pkl", "rb") as f:
        data = pkl.load(f)
        test = data["test"]
    train = [[2.3, 7.7]]
    cot = f"P2 received $7.7 but P2 thinks anything that is not $5 is selfish behavior. $7.7 is not equal to $5 so P2 should reject this offer. Therefore, the outcome is desirable."
    prompts = []
    for test_datum in test:
        prompt = base_prompt
        for datum in train:
            example = f"P1 proposes a split of ${datum[0]} for P1 and ${datum[1]} for P2. P2 rejected this offer. A desirable outcome is defined as one where P2 punishes P1's selfish behavior. Is the outcome desirable? Let's think step by step:\n"
            prompt += "\n\n" + example + cot

        test = f"P1 proposes a split of ${test_datum[0]} for P1 and ${test_datum[1]} for P2. P2 rejected this offer. A desirable outcome is defined as one where P2 punishes P1's selfish behavior. Is the outcome desirable? Let's think step by step:"
        prompt += "\n\n" + test
        prompts.append(prompt)
    return prompts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ultimatum")
    parser.add_argument(
        "--condition",
        type=str,
        default="low_high_percentage",
        help="[low_high_percentage, low_high_payoff, inequity_aversion]",
    )
    parser.add_argument(
        "--shorter",
        action="store_true",
        default=False,
        help="Including this flag will evaluate the case where we use 1 example with an explanation.",
    )
    args = parser.parse_args()
    thresholds_by_condition = {
        "low_high_percentage": [0.3, 0.6],
        "low_high_payoff": [10, 100],
        "inequity_aversion": [None],
    }

    for threshold in thresholds_by_condition[args.condition]:
        if args.shorter:
            if args.condition == "low_high_percentage":
                prompt = get_low_high_percentage_prompt_shorter(threshold)
            elif args.condition == "low_high_payoff":
                prompt = get_low_high_payoff_prompt_shorter(threshold)
            elif args.condition == "inequity_aversion":
                prompt = get_inequity_aversion_prompt_shorter()

        else:
            if args.condition == "low_high_percentage":
                prompt = get_low_high_percentage_prompt(threshold)
            elif args.condition == "low_high_payoff":
                prompt = get_low_high_payoff_prompt(threshold)
            elif args.condition == "inequity_aversion":
                prompt = get_inequity_aversion_prompt()
        print(prompt)
