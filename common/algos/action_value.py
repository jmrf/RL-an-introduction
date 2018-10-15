import numpy as np

from common import incremental_average, incremental_weighted_average


def action_value_sampling(bandit, k, steps, epsilon, alpha, initial_value):

    # action values and rewards over time
    action_values = np.full(k, initial_value)
    num_selections = np.ones(k)
    rewards = np.zeros(steps)
    is_optimal = np.zeros(steps)

    for s in range(steps):
        # select an action (greedy with prob=epsilon)
        explore = epsilon > np.random.rand()
        action_idx = np.random.randint(k) if explore else np.argmax(action_values)
        num_selections[action_idx] += 1
        # execute action
        rewards[s] = bandit.execute_action(action_idx)
        is_optimal[s] = int(bandit.get_optimal_action() == action_idx)
        # update the average
        if alpha < 1:
            action_values[action_idx] = incremental_weighted_average(
                rewards[s],
                action_values[action_idx],
                num_selections[action_idx],
                alpha
            )
        else:
            action_values[action_idx] = incremental_average(
                rewards[s],
                action_values[action_idx],
                num_selections[action_idx]
            )

    return rewards, is_optimal
