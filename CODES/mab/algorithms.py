import numpy as np
from scipy.stats import beta
from abc import ABC, abstractmethod
from typing import List

# Abstract base class for all Multi-Armed Bandit (MAB) algorithms
class BaseMAB(ABC):
    """Base class for Multi-Armed Bandit algorithms"""
    
    def __init__(self, n_ads: int):
        self.n_ads = n_ads  # Number of available ads (arms)
        self.reset()
    
    @abstractmethod
    def reset(self):
        """Reset the algorithm state"""
        self.rewards_history = []  # Stores (chosen_ad, reward) tuples for analysis
    
    @abstractmethod
    def select_ad(self) -> int:
        """Select an ad to show"""
        raise NotImplementedError  # Must be implemented by subclasses
    
    @abstractmethod
    def update(self, chosen_ad: int, reward: int):
        """Update algorithm's state with the reward received"""
        self.rewards_history.append((chosen_ad, reward))  # Log chosen ad and its reward



# Epsilon-Greedy strategy implementation
class EpsilonGreedy(BaseMAB):
    def __init__(self, n_ads: int, epsilon: float = 0.1):
        self.epsilon = epsilon  # Probability of choosing a random ad (exploration)
        super().__init__(n_ads)
    
    def reset(self):
        super().reset()
        self.counts = np.zeros(self.n_ads)  # Number of times each ad was selected
        self.values = np.zeros(self.n_ads)  # Estimated value (average reward) of each ad
    
    def select_ad(self) -> int:
        # With probability epsilon, choose a random ad
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_ads)
        # Otherwise, choose the ad with the highest estimated value
        return int(np.argmax(self.values))
    
    def update(self, chosen_ad: int, reward: int):
        super().update(chosen_ad, reward)
        self.counts[chosen_ad] += 1
        n = self.counts[chosen_ad]
        value = self.values[chosen_ad]
        # Update estimated value using incremental mean formula
        self.values[chosen_ad] = ((n - 1) / n) * value + (1 / n) * reward



# Upper Confidence Bound (UCB) algorithm implementation
class UCB(BaseMAB):
    def __init__(self, n_ads: int, c: float = 2.0):
        self.c = c  # Confidence level parameter
        super().__init__(n_ads)
    
    def reset(self):
        super().reset()
        self.counts = np.zeros(self.n_ads)  # Number of times each ad has been selected
        self.values = np.zeros(self.n_ads)  # Estimated value of each ad
        self.t = 0  # Time step counter
    
    def select_ad(self) -> int:
        self.t += 1
        # In the first n_ads steps, select each ad once
        if self.t <= self.n_ads:
            return self.t - 1
        
        # Compute UCB values for each ad
        ucb_values = self.values + self.c * np.sqrt(np.log(self.t) / (self.counts + 1e-10))
        return int(np.argmax(ucb_values))  # Select ad with highest UCB
    
    def update(self, chosen_ad: int, reward: int):
        super().update(chosen_ad, reward)
        self.counts[chosen_ad] += 1
        n = self.counts[chosen_ad]
        value = self.values[chosen_ad]
        # Update estimated value
        self.values[chosen_ad] = ((n - 1) / n) * value + (1 / n) * reward



# Thompson Sampling algorithm implementation
class ThompsonSampling(BaseMAB):
    def __init__(self, n_ads: int):
        super().__init__(n_ads)
    
    def reset(self):
        super().reset()
        self.successes = np.zeros(self.n_ads)  # Number of successes (reward=1) for each ad
        self.failures = np.zeros(self.n_ads)  # Number of failures (reward=0) for each ad
    
    def select_ad(self) -> int:
        # Sample from the Beta distribution for each ad to simulate posterior distribution
        samples = [np.random.beta(s + 1, f + 1) for s, f in zip(self.successes, self.failures)]
        return int(np.argmax(samples))  # Select ad with the highest sampled value
    
    def update(self, chosen_ad: int, reward: int):
        super().update(chosen_ad, reward)
        if reward == 1:
            self.successes[chosen_ad] += 1  # Increment success count
        else:
            self.failures[chosen_ad] += 1  # Increment failure count
