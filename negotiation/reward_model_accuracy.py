import numpy as np
import pickle as pkl

from eval import (
    stubborn,
    competitive,
    versatile,
    pushover,
)


def txt_to_pkl():
    with open("logs/gpt3_answers_stubborn.txt", "r") as f:
        lines = f.readlines()
    idxs = [i for i in range(len(lines)) if lines[i] == "\n"]
    to_save = []
    j = 0
    i = 0
    for idx in idxs[::2]:
        pred_idx = idxs[i + 1]
        neg = lines[j:idx]
        neg_str = " ".join(neg)
        pred_str = lines[idx + 1 : pred_idx][-1]
        pred = None
        if "no" in pred_str:
            pred = 0
        elif "yes" in pred_str:
            pred = 1
        else:
            raise ValueError
        to_save.append((neg_str, pred_str, pred))
        j = pred_idx + 1
        i += 2

    with open("logs/gpt3_answers/stubborn.pkl", "wb") as f:
        pkl.dump(to_save, f)


def parse_logs_into_chunks(model_type, condition, seed):
    with open(f"logs/{model_type}_lite_{condition}{seed}.txt", "r") as f:
        lines = f.readlines()
        break_idxs = [i for i in range(len(lines)) if lines[i] == "\n"]
        j = 0
        chunks = []
        for break_idx in break_idxs:
            chunks.append(lines[j:break_idx])
            j = break_idx + 1
    return chunks


str2func = {
    "stubborn": stubborn,
    "versatile": versatile,
    "competitive": competitive,
    "pushover": pushover,
}


def main(model, style, seed):
    metric_count = []
    # with open(f"logs/gpt3_answers_{style}{seed}.pkl", "rb") as f:
    label_freqs = {0: 0, 1: 0}
    pred_freqs = {0: 0, 1: 0}
    correct_labels = {0: 0, 1: 0}

    fname = f"logs/{model}_{style}_answers_{seed}.pkl"

    print(fname)
    with open(fname, "rb") as f:
        gpt3_answers = pkl.load(f)

    for i, (neg_str, pred_str, pred) in enumerate(gpt3_answers):
        neg_lst = neg_str.split("\n")
        ground_truth = str2func[style](neg_lst)
        label_freqs[ground_truth] += 1
        pred_freqs[pred] += 1
        if ground_truth == pred:
            metric_count.append(1)
            correct_labels[pred] += 1
        else:
            metric_count.append(0)

    print("label freqs: ", label_freqs)
    print("pred freqs: ", pred_freqs)
    # print("correct labels: ", correct_labels)
    print("LEN: ", len(gpt3_answers))
    return np.mean(metric_count), np.std(metric_count)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="gpt3_accuracy")
    parser.add_argument("--style", type=str, default="stubborn", help="For ICLR 2023")
    parser.add_argument("--model", type=str, default="rl", help="For ICLR 2023")
    args = parser.parse_args()
    means = []
    stds = []
    for seed in range(3):
        mean, std = main(
            args.model,
            args.style,
            seed,
        )
        means.append(mean)
        stds.append(std)
        print(means, std)
    print(means)
    print("mean: ", np.mean(means), "std: ", np.mean(stds), len(means))
