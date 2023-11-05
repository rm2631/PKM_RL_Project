def _calculate_reward_allocation(
    current_values: [int],
    previous_values: [int],
    reward_lower_bound: int,
    reward_upper_bound: int,
    nb_reward_bins: int,
    reward_negative_values: bool = False,
):
    step_size = 1 / nb_reward_bins
    reward_step_size = (reward_upper_bound - reward_lower_bound) / nb_reward_bins
    if not reward_negative_values:
        reward_bins = [
            (i * step_size, (reward_lower_bound + i * reward_step_size))
            for i in range(nb_reward_bins + 1)
        ]
    else:
        reward_bins = [
            (i * -step_size, (reward_lower_bound + i * reward_step_size))
            for i in reversed(range(nb_reward_bins + 1))
        ]
    reward_list = []
    for current_value, previous_value in zip(current_values, previous_values):
        if previous_value == 0:
            bin_index = 0 if not reward_negative_values else -1
        else:
            difference_perc = (current_value - previous_value) / previous_value
            for index in range(nb_reward_bins):
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
    reward_list = _calculate_reward_allocation(current_value, previous_value, 0, 3, 20)
    reward = sum(reward_list)
    return reward


def handle_xp_change_reward(current_value: [int], previous_value: [int]):
    """
    Reward function for handling xp change
    """
    reward_list = _calculate_reward_allocation(current_value, previous_value, 0, 5, 10)
    reward = sum(reward_list)
    return reward


def handle_opponent_hp_change_reward(current_value: [int], previous_value: [int]):
    """
    Reward function for handling opponent hp change
    """
    reward_list = _calculate_reward_allocation(
        current_value, previous_value, 0, 1, 5, reward_negative_values=True
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
