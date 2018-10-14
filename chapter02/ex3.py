import argparse
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt

from common import logger
from common import incremental_average
from common.problems.k_armed_bandit import kArmedBandit


def action_value_alg(bandit, k, steps, epsilon):

    # action values and rewards over time
    action_values = np.zeros(k)
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
        action_values[action_idx] = incremental_average(
            rewards[s],
            action_values[action_idx],
            num_selections[action_idx]
        )

    return rewards, is_optimal


def ex_2_3(runs, k, steps, epsilons, stationary):

    # init the bandit problem (stationary)
    bandit = kArmedBandit(k, stationary=stationary)

    # init the plots
    fig, [ax1, ax2] = plt.subplots(2, sharey=True)
    fig.suptitle(
        f"Value action sampling for {'stationary' if stationary else 'nonstationary'} {k}-arm bandit")

    epsilon_iter = tqdm(epsilons)
    for eps in epsilon_iter:
        epsilon_iter.set_description(f"Ex 2.3 - epsilon={eps:.3f}")
        avg_rewards = np.zeros(steps)
        avg_optimals = np.zeros(steps)
        avg_explorations = 0
        # run the experiment 'runs' times
        for r in tqdm(range(runs)):
            rewards, is_optimal = action_value_alg(bandit, k, steps, eps)
            # accumulate rewards across steps
            avg_rewards += rewards
            avg_optimals += is_optimal

        # compute the average of all runs
        avg_rewards /= runs
        avg_optimals /= runs
        # plot all configurations:
        # Average rewards over time
        ax1.plot(avg_rewards, label=f"$\epsilon$={eps:.2f}")
        # % of times optimal action was chosen
        ax2.plot(avg_optimals, label=f"$\epsilon$={eps:.2f}")

    ax1.legend()
    ax2.legend()
    ax1.set_ylabel('Average rewards')
    ax2.set_ylabel('% Optimal action')
    ax2.set_xlabel('Steps')
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=10, help="Number of bandit arms")
    parser.add_argument('--stationary', action='store_true', help="Whether to simulate a stationary arm-bandit or not")
    parser.add_argument('-s', '--steps', type=int, default=1000, help="Steps to run the action value average estimates")
    parser.add_argument('-r', '--runs', type=int, default=2000, help="Runs to average over")
    parser.add_argument(
        '-e', '--epsilons', type=float, nargs='+', default=1e-2, help="list of greedy probabilities to compare")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    ex_2_3(
        runs=args.runs,
        k=args.k,
        steps=args.steps,
        epsilons=args.epsilons,
        stationary=args.stationary
    )
