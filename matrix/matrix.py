import numpy as np
import pickle as pkl
import gym
from gym import spaces
from stable_baselines3 import DQN


class Matrix(gym.Env):
    """Environment for Matrix Games."""

    def __init__(self, model, game, sol_concept):
        super(Matrix, self).__init__()
        self.action_space = spaces.Discrete(4)  # 0->CC, 1-> CD, 2->DC, 3->DD
        self.observation_space = spaces.Discrete(1)
        rl_rewards = {
            ("battle", "welfare"): [0, 3],
            ("battle", "equality"): [1, 2],
            ("battle", "rawlsian"): [0, 3],
            ("battle", "pareto"): [0, 3],
            ("prisoners", "welfare"): [0],
            ("prisoners", "equality"): [0, 3],
            ("prisoners", "rawlsian"): [0],
            ("prisoners", "pareto"): [0, 1, 2],
            ("stag", "welfare"): [0],
            ("stag", "equality"): [0, 3],
            ("stag", "rawlsian"): [0],
            ("stag", "pareto"): [0],
            ("chicken", "welfare"): [0, 1, 2],
            ("chicken", "equality"): [0, 3],
            ("chicken", "rawlsian"): [0],
            ("chicken", "pareto"): [0, 1, 2],
        }
        gpt3_rewards = {
            ("battle", "welfare"): [0, 3],
            ("battle", "equality"): [1, 2],
            ("battle", "rawlsian"): [0],
            ("battle", "pareto"): [0],
            ("prisoners", "welfare"): [0],
            ("prisoners", "equality"): [0],
            ("prisoners", "rawlsian"): [0],
            ("prisoners", "pareto"): [1, 2],
            ("stag", "welfare"): [0],
            ("stag", "equality"): [0],
            ("stag", "rawlsian"): [0, 2, 3],
            ("stag", "pareto"): [0],
            ("chicken", "welfare"): [0, 1, 2],
            ("chicken", "equality"): [0],
            ("chicken", "rawlsian"): [0],
            ("chicken", "pareto"): [3],
        }
        random_rewards = {
            ("battle", "welfare"): [-1],
            ("battle", "equality"): [-1],
            ("battle", "rawlsian"): [1, 2],
            ("battle", "pareto"): [0],
            ("prisoners", "welfare"): [0],
            ("prisoners", "equality"): [1, 2],
            ("prisoners", "rawlsian"): [0],
            ("prisoners", "pareto"): [0],
            ("stag", "welfare"): [0],
            ("stag", "equality"): [0, 3],
            ("stag", "rawlsian"): [0],
            ("stag", "pareto"): [3],
            ("chicken", "welfare"): [0],
            ("chicken", "equality"): [0, 1, 2],
            ("chicken", "rawlsian"): [3],
            ("chicken", "pareto"): [0],
        }
        blank_rewards = {
            ("battle", "welfare"): [0],
            ("battle", "equality"): [0],
            ("battle", "rawlsian"): [0],
            ("battle", "pareto"): [0],
            ("prisoners", "welfare"): [0, 3],
            ("prisoners", "equality"): [0, 3],
            ("prisoners", "rawlsian"): [0, 3],
            ("prisoners", "pareto"): [0, 3],
            ("stag", "welfare"): [-1],
            ("stag", "equality"): [-1],
            ("stag", "rawlsian"): [-1],
            ("stag", "pareto"): [-1],
            ("chicken", "welfare"): [0, 3],
            ("chicken", "equality"): [0, 3],
            ("chicken", "rawlsian"): [0, 3],
            ("chicken", "pareto"): [0, 3],
        }
        self.correct_answers = rl_rewards[(game, sol_concept)]
        if model == "rl":
            self.answers = rl_rewards[(game, sol_concept)]
        elif model == "gpt3":
            self.answers = gpt3_rewards[(game, sol_concept)]
        elif model == "random":
            self.answers = random_rewards[(game, sol_concept)]
        elif model == "blank":
            self.answers = blank_rewards[(game, sol_concept)]
        else:
            raise ValueError
        # self.correct_answers = self.correct_answers[(game, sol_concept)]

    def reset(self):
        return np.array([0])

    def step(self, action):
        done = True
        if action in self.answers:
            reward = 1
        else:
            reward = 0
        next_obs = np.array([0])
        info = {}
        return next_obs, reward, done, info


def train(model, game, sol_concept, seed):
    env = Matrix(model, game, sol_concept)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        train_freq=2,
        learning_starts=20,
        exploration_fraction=0.5,
        learning_rate=0.0001,
    ).learn(500)
    return model


def main(model_type):
    data2 = {}
    for sol_concept in ["welfare", "equality", "rawlsian", "pareto"]:
        data = {}
        for game in ["battle", "prisoners", "stag", "chicken"]:
            rewards = []
            for seed in range(3):
                model = train(model_type, game, sol_concept, seed)
                test_env = Matrix(model_type, game, sol_concept)
                obs = test_env.reset()
                model.set_env(test_env)
                action, _ = model.predict(obs, deterministic=True)
                if action in test_env.correct_answers:
                    reward = 1
                else:
                    reward = 0
                rewards.append(reward)
            mean, std = np.mean(rewards), np.std(rewards)
            data[game] = (mean, std)
        means, stds = [], []
        for game in data:
            means.append(data[game][0])
            stds.append(data[game][1])
        print(
            f"{sol_concept}!! mean over games: ",
            np.mean(means),
            "stds over games: ",
            np.mean(stds),
        )
        data2[sol_concept] = (np.mean(means), np.mean(stds))
    print(data2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="matrix games")
    parser.add_argument(
        "--reward_model",
        type=str,
        default="gpt3",
        help="[gpt3, rl, blank]",
    )
    args = parser.parse_args()
    assert args.reward_model in ["gpt3", "rl", "blank"]
    main(args.reward_model)
