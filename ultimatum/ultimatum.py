import numpy as np
import pickle as pkl
import gym
from gym import spaces
from stable_baselines3 import DQN


class Ultimatum(gym.Env):
    """Environment for Ultimatum Game."""

    def __init__(self, path):
        super(Ultimatum, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(2,))
        with open(path, "rb") as f:
            data = pkl.load(f)
            self.test_set = data["test"]

    def reset(self):
        # randomly sample from test set
        idx = np.random.choice(range(len(self.test_set)))
        self.obs = self.test_set[idx][:-1]
        self.label = self.test_set[idx][-1]
        return self.obs

    def step(self, action):
        done = True
        reward = int(self.label == action)
        next_obs = np.array([0, 0])
        info = {}
        return next_obs, reward, done, info


def train(train_path, seed):
    env = Ultimatum(train_path)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        train_freq=2,
        learning_starts=20,
        exploration_fraction=0.5,
        learning_rate=0.0001,
    ).learn(10000)
    return model


def train_RL(train_path, test_path, seed=0):
    model = train(train_path, seed)
    test_env = Ultimatum(test_path)
    obs = test_env.reset()
    model.set_env(test_env)
    rewards = []
    for i, datum in enumerate(test_env.test_set):
        obs = np.array(datum[:-1])
        test_env.obs = datum[:-1]
        test_env.label = datum[-1]
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        # print("Step {}".format(i), "Action: ", action, "Obs: ", datum, "reward: ", reward)
        rewards.append(reward)
        if reward == 0:
            print(datum, action)
    # print(f"mean correct: {np.mean(rewards)}, std: {np.around(np.std(rewards),2)}")
    return np.mean(rewards), np.std(rewards)


def evaluate_lm_responses(path):
    with open(f"{path}.pkl", "rb") as f:
        ground_truth = pkl.load(f)["test"]
    with open(f"lm_responses/{path}.pkl", "rb") as f:
        lm_responses = pkl.load(f)

    is_desirable_text2int = {"Yes": 0, "No": 1}
    lm_correct = []
    to_save = []
    for i, datum in enumerate(lm_responses):
        test_prompt = datum[0].split("\n")[-1].split(" ")
        moneys = []
        for token in test_prompt:
            if "$" in token:
                moneys.append(float(token[1:]))

        assert moneys[0] == ground_truth[i][0] and moneys[1] == ground_truth[i][1]
        lm_response = datum[1].strip()
        lm_response = is_desirable_text2int[lm_response]
        lm_correct.append(int(lm_response == ground_truth[i][-1]))
        if lm_correct[-1] == 0:
            print(moneys)
        to_save.append([moneys[0], moneys[1], lm_response])
    # with open(f"ultimatum/lm_responses/{path}_train.pkl", "wb") as f:
    #    to_save = {"test": to_save}
    #    pkl.dump(to_save, f)
    print(
        "mean correct ", np.mean(lm_correct), "std: ", np.around(np.std(lm_correct), 2)
    )


def evaluate_lm_responses_shorter(path):
    with open(f"{path}.pkl", "rb") as f:
        ground_truth = pkl.load(f)["test"]
    with open(f"lm_responses_shorter/{path}.pkl", "rb") as f:
        lm_responses = pkl.load(f)

    not_des_str = "Therefore, the outcome is not desirable"
    des_str = "Therefore, the outcome is desirable"
    lm_correct = []
    to_save = []
    for i, datum in enumerate(lm_responses):
        test_prompt = datum[0].split("\n")[-1].split(" ")
        moneys = []
        for token in test_prompt:
            if "$" in token:
                moneys.append(float(token[1:]))

        assert moneys[0] == ground_truth[i][0] and moneys[1] == ground_truth[i][1]
        lm_response = datum[1].strip()
        lm_answer = None
        if not_des_str in lm_response:
            lm_answer = 1
        elif des_str in lm_response:
            lm_answer = 0
        else:
            raise ValueError
        lm_correct.append(int(lm_answer == ground_truth[i][-1]))
        if lm_correct[-1] == 0:
            print(moneys)
        to_save.append([moneys[0], moneys[1], lm_answer])
    # with open(f"ultimatum/lm_responses_shorter/{path}_train.pkl", "wb") as f:
    #    to_save = {"test": to_save}
    #    pkl.dump(to_save, f)
    print(
        "mean correct ", np.mean(lm_correct), "std: ", np.around(np.std(lm_correct), 2)
    )


def main(condition, shorter, num_seeds, model, thresholds):
    for threshold in thresholds:
        partial_path = f"{condition}_{threshold}"
        if model == "gpt3":
            if shorter:
                evaluate_lm_responses_shorter(partial_path)
                model_path = f"lm_responses_shorter/{partial_path}_train.pkl"
            else:
                evaluate_lm_responses(partial_path)
                model_path = f"lm_responses/{partial_path}_train.pkl"
        elif model == "sl":
            if shorter:
                model_path = f"sl_responses_shorter/{partial_path}_train.pkl"
            else:
                model_path = f"sl_responses/{partial_path}_train.pkl"
        path = f"{partial_path}.pkl"
        means = []
        stds = []
        for seed in range(num_seeds):
            mean, std = train_RL(train_path=model_path, test_path=path, seed=seed)
            means.append(mean)
            stds.append(std)
        print("mean: ", np.mean(means), "std: ", np.mean(stds))


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
        "--model",
        type=str,
        default="gpt3",
        help="[gpt3, sl]",
    )
    parser.add_argument(
        "--shorter",
        action="store_true",
        default=False,
        help="Including this flag will evaluate the case where we use 1 example with an explanation.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    thresholds_by_condition = {
        "low_high_percentage": [0.3, 0.6],
        "low_high_payoff": [10, 100],
        "inequity_aversion": [None],
    }
    main(
        args.condition,
        args.shorter,
        args.num_seeds,
        args.model,
        thresholds=thresholds_by_condition[args.condition],
    )
