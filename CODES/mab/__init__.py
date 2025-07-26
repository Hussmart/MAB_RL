# Importing required classes and functions from internal modules
from .environment import AdBandit, AdEnvironment                     # Environment and bandit definitions
from .algorithms import EpsilonGreedy, UCB, ThompsonSampling         # Multi-armed bandit algorithms
from .experiment import compare_algorithms, run_experiment, run_dynamic_experiment  # Experiment utilities

# __all__ defines the public API of the module when using 'from module import *'
__all__ = [
    'AdBandit',                  # Class representing a single ad (bandit)
    'AdEnvironment',            # Class representing the simulation environment
    'EpsilonGreedy',            # Epsilon-Greedy algorithm class
    'UCB',                      # Upper Confidence Bound algorithm class
    'ThompsonSampling',         # Thompson Sampling algorithm class
    'compare_algorithms',       # Function to compare multiple algorithms
    'run_experiment',           # Function to run a standard experiment
    'run_dynamic_experiment'    # Function to run an experiment in a dynamic (non-stationary) environment
]
