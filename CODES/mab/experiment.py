import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from .environment import AdBandit, AdEnvironment
from .algorithms import BaseMAB, EpsilonGreedy, UCB, ThompsonSampling
import pandas as pd
import seaborn as sns
from .visuals import plot_decision_heatmap, plot_radar_comparison


def run_experiment(env: AdEnvironment, algorithm: BaseMAB, n_steps: int) -> Dict[str, Any]:
    """Run one complete experiment with a given algorithm and environment."""
    algorithm.reset()

    # Initialize metric trackers
    total_reward = 0
    total_regret = 0
    cumulative_rewards = []
    cumulative_regret = []
    selected_arms = []

    # Identify the optimal arm at the beginning
    optimal_arm = np.argmax([ad.true_ctr for ad in env.ads])
    optimal_ctr = env.ads[optimal_arm].true_ctr

    # Execute each step of the experiment
    for step in range(n_steps):
        selected_arm = algorithm.select_ad()
        selected_arms.append(selected_arm)

        reward = env.get_reward(selected_arm, step)
        algorithm.update(selected_arm, reward)

        # Update reward and regret
        total_reward += reward
        step_regret = optimal_ctr - env.ads[selected_arm].true_ctr
        total_regret += step_regret

        cumulative_rewards.append(total_reward)
        cumulative_regret.append(total_regret)

    # Calculate how often the optimal arm was selected
    optimal_selections = sum(1 for arm in selected_arms if arm == optimal_arm)
    optimal_percentage = (optimal_selections / n_steps) * 100

    # Print basic debug info
    print(f"\nDebug info for algorithm:")
    print(f"Optimal arm: {optimal_arm}")
    print(f"Optimal CTR: {optimal_ctr:.4f}")
    print(f"Total reward: {total_reward}")
    print(f"Average reward: {total_reward/n_steps:.4f}")
    print(f"Total optimal selections: {optimal_selections}")
    print(f"Optimal percentage: {optimal_percentage:.2f}%")

    return {
        'total_reward': total_reward,
        'average_reward': total_reward / n_steps,
        'total_regret': total_regret,
        'cumulative_rewards': cumulative_rewards,
        'cumulative_regret': cumulative_regret,
        'optimal_selections': optimal_selections,
        'optimal_percentage': optimal_percentage,
        'selected_arms': selected_arms
    }


def compare_algorithms(env: AdEnvironment, n_steps: int, random_seed: int = 42):
    """Compare multiple algorithms on the same environment."""
    np.random.seed(random_seed)

    def clone_env(original_env: AdEnvironment) -> AdEnvironment:
        ads_copy = [AdBandit(ad.true_ctr, ad.name) for ad in original_env.ads]
        return AdEnvironment(ads_copy)

    algorithms = {
        'Epsilon-Greedy (ε=0.1)': EpsilonGreedy(env.n_ads, epsilon=0.1),
        'UCB (c=2.0)': UCB(env.n_ads, c=2.0),
        'Thompson Sampling': ThompsonSampling(env.n_ads)
    }

    results = {}
    selected_arms_dict = {}

    for name, alg in algorithms.items():
        print(f"\nRunning {name}...")
        local_env = clone_env(env)
        result = run_experiment(local_env, alg, n_steps)

        # Determine optimal arm
        best_arm = np.argmax([ad.true_ctr for ad in local_env.ads])
        arm_chosen_count = sum(1 for (chosen_ad, _) in alg.rewards_history if chosen_ad == best_arm)
        optimal_percentage = (arm_chosen_count / n_steps) * 100

        # Store metrics
        results[name] = {
            'total_reward': result['total_reward'],
            'average_reward': result['average_reward'],
            'total_regret': result['total_regret'],
            'cumulative_rewards': result['cumulative_rewards'],
            'cumulative_regret': result['cumulative_regret'],
            'optimal_percentage': optimal_percentage
        }

        selected_arms_dict[name] = [chosen_ad for chosen_ad, _ in alg.rewards_history]

        print(f"\n{name}:")
        print(f"Total Reward: {results[name]['total_reward']}")
        print(f"Average Reward: {results[name]['average_reward']:.4f}")
        print(f"Total Regret: {results[name]['total_regret']:.4f}")
        print(f"Optimal Arm Chosen %: {optimal_percentage:.2f}%")

    # Plot cumulative reward and regret over time
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['cumulative_rewards'], label=name)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards Over Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for name, result in results.items():
        plt.plot(result['cumulative_regret'], label=name)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret Over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot heatmap of arm selection decisions
    plot_decision_heatmap(selected_arms_dict, env.n_ads)


def run_dynamic_experiment(env: AdEnvironment, algorithm: BaseMAB,
                           n_steps: int, change_point: int,
                           new_ctrs: List[float]) -> Dict[str, Any]:
    """Run experiment with a sudden change in the environment (non-stationary case)."""
    algorithm.reset()
    cumulative_rewards = []
    cumulative_regret = []
    total_reward_so_far = 0

    for step in range(n_steps):
        # Change CTRs at the change point
        if step == change_point:
            for ad, new_ctr in zip(env.ads, new_ctrs):
                ad.true_ctr = new_ctr
            env.optimal_ctr = max(new_ctrs)

        ad_index = algorithm.select_ad()
        reward = env.get_reward(ad_index, step)
        algorithm.update(ad_index, reward)

        total_reward_so_far += reward
        cumulative_rewards.append(total_reward_so_far)
        cumulative_regret.append(env.total_regret)

    return {
        'cumulative_rewards': cumulative_rewards,
        'cumulative_regret': cumulative_regret,
        'total_reward': total_reward_so_far,
        'average_reward': total_reward_so_far / n_steps,
        'total_regret': env.total_regret
    }


def repeated_experiment(env_builder, algorithm_class, n_ads, n_steps=1000, n_trials=10, **kwargs):
    """Run the same experiment multiple times to gather statistical data."""
    all_cumulative_rewards = []
    all_total_rewards = []
    all_total_regrets = []

    for i in range(n_trials):
        np.random.seed(i)
        env = env_builder(seed=i)
        alg = algorithm_class(n_ads, **kwargs)
        result = run_experiment(env, alg, n_steps)

        all_cumulative_rewards.append(result['cumulative_rewards'])
        all_total_rewards.append(result['total_reward'])
        all_total_regrets.append(result['total_regret'])

    print(f"\n✅ {algorithm_class.__name__} over {n_trials} trials:")
    print(f"  ➤ Avg Reward:  {np.mean(all_total_rewards):.2f} ± {np.std(all_total_rewards):.2f}")
    print(f"  ➤ Avg Regret:  {np.mean(all_total_regrets):.2f} ± {np.std(all_total_regrets):.2f}")

    return {
        'cumulative_rewards': all_cumulative_rewards,
        'rewards': all_total_rewards,
        'regrets': all_total_regrets
    }
