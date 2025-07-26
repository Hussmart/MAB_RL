# MAB_RL


#  Multi-Armed Bandit Simulation for Ad Optimization

This project simulates a **Multi-Armed Bandit (MAB)** environment to help product managers and data scientists understand how to optimize online ad performance using exploration-exploitation strategies. The goal is to find the best-performing ad version (i.e., the one with the highest Click-Through Rate - CTR) with minimal display cost.

##  Project Description

Imagine you're managing an online advertising platform and want to identify the best ad version to maximize CTR. Each ad version is represented as a "bandit arm," and your goal is to learn which arm yields the best results — without knowing their true CTRs beforehand.

This project builds a simulated MAB environment from scratch and compares the effectiveness of different algorithms in learning the optimal ad strategy.

##  Environment Simulation

- Simulates **k arms** (e.g., 5 ad versions), each with a hidden real CTR.
- Pulling an arm returns a binary reward: `1` (click) or `0` (no click), based on its true CTR.
- CTRs can be constant or sampled from a distribution (e.g., Beta).

##  Implemented Algorithms

###  Epsilon-Greedy
- Randomly explores with probability ε.
- Exploits the best-known arm the rest of the time.
- Includes analysis on different ε values.

###  Upper Confidence Bound (UCB)
- Balances exploration and exploitation using confidence intervals.
- Prioritizes less-tested arms with high uncertainty.

###  Thompson Sampling
- Bayesian approach using Beta distributions.
- Often converges faster than others in practice.

##  Evaluation & Analysis

- **Cumulative Reward** and **Regret** plotted over time.
- Performance comparisons: speed of convergence, reward optimization, robustness.
- Detailed experiments to analyze:
  - Parameter sensitivity (e.g., ε in Epsilon-Greedy, c in UCB)
  - **Dynamic CTR scenarios** where arm performance changes mid-simulation.

## 📈 Sample Outputs

- Reward vs. time plots
- Regret comparisons across algorithms
- Sensitivity analysis and discussion
