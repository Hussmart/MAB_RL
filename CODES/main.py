import numpy as np
from mab.environment import AdBandit, AdEnvironment
from mab.algorithms import EpsilonGreedy, UCB, ThompsonSampling
from mab.experiment import (
    compare_algorithms,
    run_dynamic_experiment,
    repeated_experiment,
    run_experiment
)
from mab.analysis import (
    analyze_epsilon_effect,
    analyze_ucb_parameter,
    run_continuous_change_experiment,
    run_sudden_change_experiment
)
from mab.visuals import (
    plot_confidence_intervals,
    plot_statistical_heatmap,
    plot_decision_heatmap,
    plot_radar_comparison
)


def build_environment_random(seed=42):
    # Creates a randomized ad environment with 5 arms using Beta distribution for CTRs
    np.random.seed(seed)
    ads = [AdBandit(true_ctr=np.random.beta(2, 5), name=f"Ad {i+1}") for i in range(5)]
    return AdEnvironment(ads)


def main():
    n_steps = 5000

    # 1. Compare the performance of all algorithms in a basic setting
    print("1. Running basic comparison of algorithms...")
    env1 = build_environment_random()
    compare_algorithms(env1, n_steps)

    # 2. Analyze how different epsilon values affect ε-Greedy algorithm behavior
    print("\n2. Analyzing effect of epsilon parameter...")
    env2 = build_environment_random()
    analyze_epsilon_effect(env2, n_steps, epsilons=[0.01, 0.05, 0.1, 0.2, 0.3])

    # 3. Analyze how varying the 'c' parameter influences UCB performance
    print("\n3. Analyzing effect of UCB parameter...")
    env3 = build_environment_random()
    analyze_ucb_parameter(env3, n_steps, c_values=[0.5, 1.0, 2.0, 4.0, 8.0])

    # 4. Test Thompson Sampling in an environment where CTRs slowly evolve
    print("\n4. Running experiment with continuously changing CTRs...")
    env4 = build_environment_random()
    algorithm4 = ThompsonSampling(env4.n_ads)
    result = run_continuous_change_experiment(
        env=env4,
        algorithm=algorithm4,
        n_steps=10000,
        change_rate=0.0001
    )
    print(f"\n➡️ Total Reward: {result['total_reward']}")
    print(f"➡️ Average Reward: {result['average_reward']:.4f}")
    print(f"➡️ Total Regret: {result['total_regret']:.4f}")

    # 5. Evaluate algorithm performance in response to sudden CTR changes
    print("\n5. Running experiment with sudden CTR change at step 500...")
    env5 = build_environment_random()
    algorithm5 = ThompsonSampling(env5.n_ads)
    result = run_sudden_change_experiment(
        env=env5,
        algorithm=algorithm5,
        n_steps=1000,
        change_point=500,
        new_ctrs=[0.03, 0.04, 0.02, 0.12, 0.06]  # Ad 4 becomes the best after change
    )

    # 6. Perform statistical analysis and visualization of experimental results
    print("\n6. Running statistical analysis and visualizations...")

    # Define algorithm instances with fixed parameters
    algo_classes = {
        "ε-Greedy": lambda n_ads: EpsilonGreedy(n_ads=n_ads, epsilon=0.1),
        "UCB": lambda n_ads: UCB(n_ads=n_ads, c=2.0),
        "Thompson": lambda n_ads: ThompsonSampling(n_ads=n_ads)
    }

    all_ci_results = {}
    all_stat_rewards = {}

    # Run multiple trials and store rewards for confidence interval and statistical testing
    for name, algo_fn in algo_classes.items():
        result = repeated_experiment(
            env_builder=build_environment_random,
            algorithm_class=algo_fn,
            n_ads=5,
            n_steps=1000,
            n_trials=10
        )
        all_ci_results[name] = result['cumulative_rewards']  # For confidence interval plots
        all_stat_rewards[name] = result['rewards']           # For t-test heatmap

    # Plot confidence intervals and statistical significance
    plot_confidence_intervals(all_ci_results)
    plot_statistical_heatmap(all_stat_rewards)

    # 7. Run experiments to gather data for radar chart comparison
    print("\n7. Creating radar chart comparison...")
    env_radar = build_environment_random()
    n_steps_radar = 10000

    # Initialize each algorithm with selected parameters
    epsilon_greedy = EpsilonGreedy(n_ads=env_radar.n_ads, epsilon=0.1)
    ucb = UCB(n_ads=env_radar.n_ads, c=2.0)
    thompson = ThompsonSampling(n_ads=env_radar.n_ads)

    algorithms = {
        "Epsilon-Greedy (ε=0.1)": epsilon_greedy,
        "UCB (c=2.0)": ucb,
        "Thompson Sampling": thompson
    }

    algorithms_stats = {}

    # Run experiment for each algorithm and calculate performance metrics
    for name, algo in algorithms.items():
        env_copy = env_radar.clone()  # Use a separate environment for each run
        
        result = run_experiment(env_copy, algo, n_steps_radar)
        
        # Calculate percentage of optimal arm selection
        best_arm = np.argmax([ad.true_ctr for ad in env_copy.ads])
        arm_chosen_count = sum(1 for (chosen_ad, _) in algo.rewards_history if chosen_ad == best_arm)
        optimal_percentage = (arm_chosen_count / n_steps_radar) * 100
        
        algorithms_stats[name] = {
            'average_reward': result['average_reward'],
            'total_regret': result['total_regret'],
            'optimal_percentage': optimal_percentage
        }

        # Output performance details
        print(f"\n{name} results:")
        print(f"Total Regret: {result['total_regret']:.4f}")
        print(f"Optimal Arm %: {optimal_percentage:.2f}%")
        print(f"Average Reward: {result['average_reward']:.4f}")

    # Plot radar chart for comparative performance visualization
    plot_radar_comparison(algorithms_stats)


if __name__ == "__main__":
    main()
