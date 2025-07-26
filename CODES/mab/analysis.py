import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from copy import deepcopy

# Local imports (assumes a project structure with separate modules)
from .environment import AdEnvironment
from .algorithms import BaseMAB, EpsilonGreedy, UCB
from .experiment import run_experiment

# ------------------------------------------------------------
# Utility to run experiments in an isolated copy of environment
# ------------------------------------------------------------
def run_fresh_experiment(env: AdEnvironment, algorithm: BaseMAB, n_steps: int) -> Dict[str, Any]:
    """
    Runs an experiment using a fresh copy of the environment
    to prevent shared state between experiments.
    """
    fresh_env = deepcopy(env)
    return run_experiment(fresh_env, algorithm, n_steps)


# ------------------------------------------------------------
# Print a summary of experiment results in a human-readable way
# ------------------------------------------------------------
def print_summary(results: Dict[str, Dict[str, Any]]):
    print("\nAnalysis Summary:")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name}:")
        print(f"  ➤ Total Reward:    {result['total_reward']}")
        print(f"  ➤ Average Reward:  {result['average_reward']:.4f}")
        print(f"  ➤ Total Regret:    {result['total_regret']:.4f}\n")


# ------------------------------------------------------------
# Analyze how different epsilon values affect Epsilon-Greedy
# ------------------------------------------------------------
def analyze_epsilon_effect(env: AdEnvironment, n_steps: int, epsilons: List[float] = None):
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]

    results = {}
    # Run experiment for each epsilon value
    for eps in epsilons:
        alg = EpsilonGreedy(env.n_ads, epsilon=eps)
        results[f'ε={eps}'] = run_fresh_experiment(env, alg, n_steps)

    # Plot cumulative rewards and regret for each epsilon
    plt.figure(figsize=(15, 5))

    # Plot: Cumulative Rewards
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['cumulative_rewards'], label=name)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Effect of ε on Cumulative Rewards')
    plt.legend()
    plt.grid(True)

    # Plot: Cumulative Regret
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        plt.plot(result['cumulative_regret'], label=name)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('Effect of ε on Cumulative Regret')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print_summary(results)


# ------------------------------------------------------------
# Analyze how different 'c' values affect UCB algorithm
# ------------------------------------------------------------
def analyze_ucb_parameter(env: AdEnvironment, n_steps: int, c_values: List[float] = None):
    if c_values is None:
        c_values = [0.5, 1.0, 2.0, 4.0, 8.0]

    results = {}
    for c in c_values:
        alg = UCB(env.n_ads, c=c)
        results[f'c={c}'] = run_fresh_experiment(env, alg, n_steps)

    # Plot results
    plt.figure(figsize=(15, 5))

    # Plot: Cumulative Rewards
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['cumulative_rewards'], label=name)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Effect of c on Cumulative Rewards (UCB)')
    plt.legend()
    plt.grid(True)

    # Plot: Cumulative Regret
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        plt.plot(result['cumulative_regret'], label=name)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('Effect of c on Cumulative Regret (UCB)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print_summary(results)


# ------------------------------------------------------------
# Simulate a scenario where CTRs change gradually over time
# ------------------------------------------------------------
def run_continuous_change_experiment(env: AdEnvironment, algorithm: BaseMAB, 
                                     n_steps: int, change_rate: float = 0.0001) -> Dict[str, Any]:
    algorithm.reset()
    fresh_env = deepcopy(env)

    cumulative_rewards = []
    cumulative_regret = []
    ctr_history = []
    total_reward = 0
    regret = 0

    for step in range(n_steps):
        # Gradually change each ad's CTR with Gaussian noise
        for ad in fresh_env.ads:
            ad.true_ctr = np.clip(ad.true_ctr + np.random.normal(0, change_rate), 0.01, 0.99)

        best_ctr = max(ad.true_ctr for ad in fresh_env.ads)

        ad_index = algorithm.select_ad()
        reward = fresh_env.get_reward(ad_index, step)
        algorithm.update(ad_index, reward)

        total_reward += reward
        regret += best_ctr - reward
        cumulative_rewards.append(total_reward)
        cumulative_regret.append(regret)
        ctr_history.append([ad.true_ctr for ad in fresh_env.ads])

    ctr_history = np.array(ctr_history)

    # Plot CTR evolution and cumulative regret
    plt.figure(figsize=(15, 5))

    # Plot: CTR evolution
    plt.subplot(1, 2, 1)
    for i in range(fresh_env.n_ads):
        plt.plot(ctr_history[:, i], label=f'Ad {i+1}')
    plt.xlabel('Steps')
    plt.ylabel('CTR')
    plt.title('CTR Evolution Over Time')
    plt.legend()
    plt.grid(True)

    # Plot: Cumulative Regret
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_regret, color='red')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret with Dynamic CTRs')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print result summary
    print(f"\n➡️ Total Reward: {total_reward}")
    print(f"➡️ Average Reward: {total_reward / n_steps:.4f}")
    print(f"➡️ Total Regret: {regret:.4f}")

    return {
        'cumulative_rewards': cumulative_rewards,
        'cumulative_regret': cumulative_regret,
        'ctr_history': ctr_history,
        'total_reward': total_reward,
        'average_reward': total_reward / n_steps,
        'total_regret': regret
    }


# ------------------------------------------------------------
# Simulate a sudden change in CTRs at a specific time point
# ------------------------------------------------------------
def run_sudden_change_experiment(env: AdEnvironment, algorithm: BaseMAB,
                                 n_steps: int, change_point: int, new_ctrs: List[float]) -> Dict[str, Any]:
    algorithm.reset()
    fresh_env = deepcopy(env)

    cumulative_rewards = []
    cumulative_regret = []
    ctr_history = []
    total_reward = 0
    regret = 0

    for step in range(n_steps):
        # Apply sudden CTR change at the change point
        if step == change_point:
            for i, ad in enumerate(fresh_env.ads):
                ad.true_ctr = new_ctrs[i]

        current_ctrs = [ad.true_ctr for ad in fresh_env.ads]
        best_ctr = max(current_ctrs)

        ad_index = algorithm.select_ad()
        reward = fresh_env.get_reward(ad_index, step)
        algorithm.update(ad_index, reward)

        total_reward += reward
        regret += best_ctr - reward
        cumulative_rewards.append(total_reward)
        cumulative_regret.append(regret)
        ctr_history.append(current_ctrs)

    ctr_history = np.array(ctr_history)

    # Plot CTR change and regret
    plt.figure(figsize=(15, 5))

    # Plot: CTR change with vertical line at change point
    plt.subplot(1, 2, 1)
    for i in range(fresh_env.n_ads):
        plt.plot(ctr_history[:, i], label=f'Ad {i+1}')
    plt.axvline(change_point, color='black', linestyle='--', label='Change Point')
    plt.xlabel('Steps')
    plt.ylabel('CTR')
    plt.title('Sudden Change in CTRs Over Time')
    plt.legend()
    plt.grid(True)

    # Plot: Regret with change point indicator
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_regret, color='red')
    plt.axvline(change_point, color='black', linestyle='--')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret with Sudden CTR Change')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"\n➡️ Total Reward: {total_reward}")
    print(f"➡️ Average Reward: {total_reward / n_steps:.4f}")
    print(f"➡️ Total Regret: {regret:.4f}")

    return {
        'cumulative_rewards': cumulative_rewards,
        'cumulative_regret': cumulative_regret,
        'ctr_history': ctr_history,
        'total_reward': total_reward,
        'average_reward': total_reward / n_steps,
        'total_regret': regret
    }
