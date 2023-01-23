import numpy as np
import pickle as pkl


def punish_selfish(threshold: float, num_train=10, num_test=50):
    data = []
    total_amount = 10
    for _ in range(num_train + num_test):
        p1_percent = np.random.uniform()
        p2_percent = 1.0 - p1_percent
        label = 0 if p2_percent < threshold else 1
        datum = [
            np.around(total_amount * p1_percent, 2),
            np.around(total_amount * p2_percent, 2),
            label,
        ]
        while datum in data:
            p1_percent = np.random.uniform()
            p2_percent = 1.0 - p1_percent
            label = 0 if p2_percent < threshold else 1
            datum = [
                np.around(total_amount * p1_percent, 2),
                np.around(total_amount * p2_percent, 2),
                label,
            ]
        data.append(datum)
    return data[:10], data[10:]


def low_high(threshold: float, num_train=10, num_test=50):
    data = []

    def get_datum():
        percent1 = np.around(np.random.uniform(), 2)
        percent2 = 1.0 - percent1
        percents = [percent1, percent2]
        percents.sort(reverse=True)  # sorted in descending order
        magnitudes = [0.1, 1, 10, 100, 1000]
        magnitude = np.random.choice(magnitudes)
        p1_money = np.around(percents[0] * magnitude, 2)
        p2_money = np.around(percents[1] * magnitude, 2)
        label = 0 if p2_money < threshold else 1
        assert p2_money <= p1_money
        datum = [p1_money, p2_money, label]
        return datum

    for _ in range(num_train + num_test):
        datum = get_datum()
        while datum in data:
            datum = get_datum()
        data.append(datum)
    return data[:10], data[10:]


def inequity_aversion(num_train=10, num_test=50):
    data = []

    def get_datum():
        percent1 = np.around(np.random.uniform(), 2)
        percent2 = 1.0 - percent1
        label = 1 if percent2 == 0.5 else 0
        p1_money = np.around(percent1 * 10, 2)
        p2_money = np.around(percent2 * 10, 2)
        datum = [p1_money, p2_money, label]
        return datum

    for _ in range(num_train + num_test):
        datum = get_datum()
        while datum in data:
            datum = get_datum()
        data.append(datum)
    return data[:10], data[10:]


if __name__ == "__main__":
    # PUNISH SELFISH BEHAVIOR
    # threshold = 0.6
    # train, test = punish_selfish(threshold)
    # with open(f"ultimatum/punish_selfish_{threshold}.pkl", "wb") as f:
    #    pkl.dump({"train": train, "test": test}, f)

    # LOW VS HIGH
    # threshold = 100
    # train, test = low_high(threshold)
    # with open(f"ultimatum/low_high_{threshold}.pkl", "wb") as f:
    #    pkl.dump({"train": train, "test": test}, f)

    # INEQUITY AVERSION
    train, test = inequity_aversion()
    with open(f"ultimatum/inequity_aversion.pkl", "wb") as f:
        pkl.dump({"train": train, "test": test}, f)
