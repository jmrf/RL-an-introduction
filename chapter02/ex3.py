import argparse
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt

from common import logger
from common.algos.action_value import action_value_sampling
from common.problems.k_armed_bandit import kArmedBandit


def ex_2_3(runs, k, steps, epsilons, initial_value=0, stationary=True, alpha=1):

    # init the bandit problem (stationary)
    bandit = kArmedBandit(k, stationary=stationary)

    # init the plots
    fig, [ax1, ax2] = plt.subplots(2, sharey=True)
    fig.suptitle(
        f"Value action sampling for {'stationary' if stationary else 'nonstationary'} {k}-arm bandit")

    epsilon_iter = tqdm(epsilons)
    for eps in epsilon_iter:
        epsilon_iter.set_description(f"Ex 2.3 - epsilon={eps:.3f}")
        # we use realistic initial values instead of optimistic ones
        avg_rewards = np.zeros(steps)
        avg_optimals = np.zeros(steps)
        avg_explorations = 0
        # run the experiment 'runs' times
        for r in tqdm(range(runs)):
            rewards, is_optimal = action_value_sampling(bandit, k, steps, eps, alpha, initial_value)
            # accumulate rewards across steps
            avg_rewards += rewards
            avg_optimals += is_optimal

        # compute the average of all runs
        avg_rewards /= runs
        avg_optimals /= runs
        # plot all configurations:
        # Average rewards over time
        ax1.plot(avg_rewards, label=f"$\epsilon$={eps:.2f} | $Q_1(a)={initial_value}$")
        # % of times optimal action was chosen
        ax2.plot(avg_optimals, label=f"$\epsilon$={eps:.2f} | $Q_1(a)={initial_value}$")

    print()

    ax1.legend()
    ax2.legend()
    ax1.set_ylabel('Average rewards')
    ax2.set_ylabel('% Optimal action')
    ax2.set_xlabel('Steps')
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    # k-armed bandit configuration
    parser.add_argument('-k', type=int, default=10, help="Number of bandit arms")
    parser.add_argument('--stationary', action='store_true', help="Whether to simulate a stationary arm-bandit or not")
    # action value smapling algorithm configurations
    parser.add_argument('--initial_value', type=float, default=0, help="initial action values")
    parser.add_argument('--alpha', type=float, default=1, help="discount factor")
    parser.add_argument('-s', '--steps', type=int, default=1000, help="Steps to run the action value average estimates")
    parser.add_argument('-r', '--runs', type=int, default=1000, help="Runs to average over")
    parser.add_argument(
        '-e', '--epsilons', type=float, nargs='+', default=1e-2, help="list of greedy probabilities to compare")
    # meta configurations
    parser.add_argument('--rseed', type=int, default=123, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logger.info(f"Freezing random seed to {args.rseed}")
    np.random.seed(args.rseed)

    ex_2_3(
        runs=args.runs,
        k=args.k,
        steps=args.steps,
        epsilons=args.epsilons,
        initial_value=args.initial_value,
        stationary=args.stationary,
        alpha=args.alpha
    )
