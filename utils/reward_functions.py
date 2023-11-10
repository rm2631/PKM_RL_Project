import numpy as np
import matplotlib.pyplot as plt


def _skewed_rewards(
    reward_lower_bound: int,
    reward_upper_bound: int,
    nb_reward_bins: int,
    reward_negative_values: bool = False,
):
    skew_factor = 2.0
    reward_bins = []
    lower_thresholds = np.linspace(0, 1, nb_reward_bins)

    multiplier = 1
    if reward_negative_values:
        lower_thresholds = np.flip(lower_thresholds)
        multiplier = -1

    for lower_threshold in lower_thresholds:
        reward = reward_lower_bound + (reward_upper_bound - reward_lower_bound) * (
            lower_threshold**skew_factor
        )
        ## Round to 2 decimal places
        reward = round(reward, 1)
        reward_bins.append((multiplier * lower_threshold, reward))

    return reward_bins


def _linear_rewards(
    reward_lower_bound: int,
    reward_upper_bound: int,
    nb_reward_bins: int,
    reward_negative_values: bool = False,
):
    bin_size = 1 / nb_reward_bins
    reward_size = (reward_upper_bound - reward_lower_bound) / nb_reward_bins
    reward_bins = []

    if not reward_negative_values:
        for i in range(nb_reward_bins + 1):
            lower_threshold = i * bin_size
            reward = reward_lower_bound + i * reward_size
            reward_bins.append((lower_threshold, reward))
    else:
        for i in reversed(range(nb_reward_bins + 1)):
            lower_threshold = i * -bin_size
            reward = reward_lower_bound + i * reward_size
            reward_bins.append((lower_threshold, reward))

    return reward_bins


def _calculate_reward_allocation(
    reward_function,
    current_values: [int],
    previous_values: [int],
    reward_lower_bound: int,
    reward_upper_bound: int,
    nb_reward_bins: int,
    reward_negative_values: bool = False,
):
    reward_bins = reward_function(
        reward_lower_bound, reward_upper_bound, nb_reward_bins, reward_negative_values
    )
    reward_list = []
    for current_value, previous_value in zip(current_values, previous_values):
        if previous_value == 0:
            bin_index = 0 if not reward_negative_values else -1
        else:
            difference_perc = (current_value - previous_value) / previous_value
            for index in range(nb_reward_bins - 1):
                if reward_bins[index][0] <= difference_perc < reward_bins[index + 1][0]:
                    bin_index = index
                    break
                else:
                    bin_index = 0 if not reward_negative_values else -1
        reward = reward_bins[bin_index][1]
        reward_list.append(reward)
    assert len(reward_list) == len(current_values)
    return reward_list


def handle_hp_change_reward(current_value: [int], previous_value: [int]):
    """
    Reward function for handling hp change
    """
    reward_list = _calculate_reward_allocation(
        _linear_rewards, current_value, previous_value, 0, 2, 10
    )
    reward = sum(reward_list)
    return reward


def handle_xp_change_reward(current_value: [int], previous_value: [int]):
    """
    Reward function for handling xp change
    """
    reward_list = _calculate_reward_allocation(
        _skewed_rewards, current_value, previous_value, 0, 5, 10
    )
    reward = sum(reward_list)
    return reward


def handle_opponent_hp_change_reward(current_value: [int], previous_value: [int]):
    """
    Reward function for handling opponent hp change
    """
    reward_list = _calculate_reward_allocation(
        _skewed_rewards,
        current_value,
        previous_value,
        0,
        1,
        10,
        reward_negative_values=True,
    )
    reward = sum(reward_list)
    return reward


def handle_downed_pokemon(current_value: [int], previous_value: [int]):
    """
    Reward function for handling downed pokemon
    """
    downed_pokemon_reward = -1
    reward_list = []
    for current_value, previous_value in zip(current_value, previous_value):
        if current_value == 0 and previous_value > 0:
            reward_list.append(downed_pokemon_reward)
    reward = sum(reward_list)

    if reward != 0:
        ## If the reward is not 0, we want to make sure that the reward is not too big to avoid traumas
        reward = max(reward, -1)
    return reward


if __name__ == "__main__":
    lower_bound = 0
    upper_bound = 5
    nb_bins = 20

    for function in [_skewed_rewards]:
        for reward_negative_values in [False, True]:
            lower_thresholds, rewards = zip(
                *function(
                    lower_bound,
                    upper_bound,
                    nb_bins,
                    reward_negative_values=reward_negative_values,
                )
            )

            plt.plot(lower_thresholds, rewards, marker="o", linestyle="-")
            plt.title("Skewed Reward Allocations")
            plt.xlabel("Lower Threshold")
            plt.ylabel("Reward")
            plt.grid(True)
            plt.show()
