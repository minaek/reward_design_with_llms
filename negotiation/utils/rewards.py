from typing import Type
import numpy as np


class Rewarder:
    def __init__(self):
        self.all_rewards = []

    def calc_reward(self, agree: bool, reward: float, partner_reward: float) -> float:
        raise NotImplementedError()

    def normalize_reward(self, reward: float) -> float:
        mu = np.mean(self.all_rewards)
        sig = max(1e-4, np.std(self.all_rewards))
        return (reward - mu) / sig


class UtilityRewarder(Rewarder):
    """Standard Reward from FB code"""

    def calc_reward(self, agree: bool, reward: float, partner_reward: float) -> float:
        # reward = reward if agree else 0
        if not agree:
            x = 5
        self.all_rewards.append(reward)

        return self.normalize_reward(reward)


class FairRewarder(Rewarder):
    """Fair Reward from Percy's code"""

    def calc_reward(self, agree: bool, reward: float, partner_reward: float) -> float:
        reward = abs(reward - partner_reward) * -0.1 if agree else -1
        self.all_rewards.append(reward)

        return self.normalize_reward(reward)


class ProSocialRewarder(Rewarder):
    """See Cao et all"""

    def __init__(self, reward_weight):
        super(ProSocialRewarder, self).__init__()
        self.reward_weight = reward_weight

    def calc_reward(self, agree: bool, reward: float, partner_reward: float) -> float:
        reward = self.reward_weight * reward + (1 - self.reward_weight) * partner_reward
        self.all_rewards.append(reward)

        return self.normalize_reward(reward)


class UtilityFairRewarder(Rewarder):
    """
    Optimizes for reward, pareto optimality
    """

    def calc_reward(self, agree, reward, partner_reward, pareto):
        final_reward = 0
        if agree:
            final_reward = reward + pareto
        self.all_rewards.append(final_reward)
        return self.normalize_reward(final_reward)


class UtilityFairRewarderNormalized(Rewarder):
    """
    Optimizes for reward, pareto optimality where we normalize Alice's score so it is between 0 and 1
    """

    def calc_reward(self, agree, reward, partner_reward, pareto):
        final_reward = 0
        if agree:
            final_reward = (reward / 10.0) + pareto
        self.all_rewards.append(final_reward)
        return self.normalize_reward(final_reward)


def get_reward_type(reward_type: str) -> Type[Rewarder]:
    if reward_type == "utility":
        return UtilityRewarder
    elif reward_type == "fair":
        return FairRewarder
    elif reward_type == "prosocial":
        return ProSocialRewarder
    elif reward_type == "all":
        return UtilityFairRewarder
    elif reward_type == "all_norm":
        return UtilityFairRewarderNormalized
    raise ValueError("Invalid Reward Type")
