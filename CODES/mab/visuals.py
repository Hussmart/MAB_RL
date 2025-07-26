import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Plot line charts with confidence intervals (standard deviation) over time
def plot_confidence_intervals(all_results: dict):
    records = []
    for algo, trials in all_results.items():
        for t_id, rewards in enumerate(trials):
            for step, reward in enumerate(rewards):
                records.append({
                    "step": step,
                    "reward": reward,
                    "algorithm": algo,
                    "trial": t_id
                })
    df = pd.DataFrame(records)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="step", y="reward", hue="algorithm", ci="sd")
    plt.title("✨ Confidence Interval Over Time")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Perform pairwise t-tests and plot p-values as a heatmap to show statistical significance
def plot_statistical_heatmap(reward_data: dict):
    algos = list(reward_data.keys())
    p_values = np.zeros((len(algos), len(algos)))

    for i in range(len(algos)):
        for j in range(len(algos)):
            if i != j:
                _, p = ttest_ind(reward_data[algos[i]], reward_data[algos[j]])
                p_values[i, j] = p
            else:
                p_values[i, j] = np.nan  # Diagonal (self comparison) is not applicable

    plt.figure(figsize=(8, 6))
    sns.heatmap(p_values, annot=True, xticklabels=algos, yticklabels=algos, cmap="coolwarm", cbar_kws={"label": "p-value"})
    plt.title("🎓 Algorithm Statistical Comparison (t-test)")
    plt.tight_layout()
    plt.show()

# Show which arm was selected at each time step using a binary heatmap
def plot_decision_heatmap(selected_arms: dict, n_arms: int):
    for algo_name, arm_list in selected_arms.items():
        mat = np.zeros((n_arms, len(arm_list)))
        for step, arm in enumerate(arm_list):
            mat[arm, step] = 1  # Mark the selected arm at each step
        plt.figure(figsize=(12, 4))
        sns.heatmap(mat, cmap="YlGnBu", cbar=True)
        plt.title(f"🧠 Decision Heatmap - {algo_name}")
        plt.xlabel("Step")
        plt.ylabel("Arm")
        plt.tight_layout()
        plt.show()

# Radar chart to visually compare average reward, normalized regret, and optimal arm selection
def plot_radar_comparison(results: dict):
    """Create a radar chart comparing algorithm performance."""
    
    # Define the metrics to plot
    labels = ['Avg Reward', 'Total Regret (1-Scaled)', 'Optimal Arm %']
    n_labels = len(labels)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Generate angles for each axis
    angles = np.linspace(0, 2*np.pi, n_labels, endpoint=False).tolist()
    angles += angles[:1]  # Close the radar chart loop
    
    # Print raw data for each algorithm
    print("\nRaw values:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"Average Reward: {metrics['average_reward']:.4f}")
        print(f"Total Regret: {metrics['total_regret']:.4f}")
        print(f"Optimal Arm %: {metrics['optimal_percentage']:.2f}%")
    
    # Determine the maximum regret for normalization
    max_regret = max(s['total_regret'] for s in results.values())
    
    for name, stats in results.items():
        # Normalize metrics to [0, 1] range
        avg_reward = stats['average_reward']
        regret = 1 - (stats['total_regret'] / max_regret)  # Lower regret is better
        optimal_pct = stats['optimal_percentage'] / 100.0

        values = [avg_reward, regret, optimal_pct]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.25)

    # Formatting and styling the radar chart
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)

    # Add legend and grid
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Algorithm Comparison (Radar Chart)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
