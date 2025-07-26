import numpy as np
from typing import List
import copy

# Represents a single advertisement arm (bandit)
class AdBandit:
    def __init__(self, true_ctr: float, name: str = None):
        self.true_ctr = true_ctr  # True Click-Through Rate (CTR)
        self.name = name or f"Ad_{true_ctr:.3f}"  # Default name based on CTR if none provided

    def show_ad(self) -> int:
        """
        Simulates showing the ad to a user.
        Returns 1 if clicked, 0 otherwise, based on the true CTR.
        """
        return int(np.random.random() < self.true_ctr)


# Represents the environment containing multiple ad bandits
class AdEnvironment:
    def __init__(self, ads: List[AdBandit]):
        self.ads = ads  # List of ad arms (bandits)
        self.n_ads = len(ads)  # Total number of ads
        self.reset()

    def reset(self):
        """
        Resets environment state including total regret and optimal CTR.
        """
        self.optimal_ctr = max(ad.true_ctr for ad in self.ads)  # Best possible CTR
        self.total_regret = 0
        self.history = []  # Stores history of (selected_ad_index, reward)

    def update_ctrs(self, step: int):
        """
        Dynamically updates CTRs based on the current step.
        This allows simulation of a non-stationary environment.
        """
        new_ctrs = get_dynamic_ctrs(step)
        for ad, new_ctr in zip(self.ads, new_ctrs):
            ad.true_ctr = new_ctr
        self.optimal_ctr = max(new_ctrs)  # Update best possible CTR after change

    def get_reward(self, ad_index: int, step: int) -> int:
        """
        Show the selected ad and return the reward (0 or 1).
        Also updates CTRs and calculates instantaneous regret.

        Args:
            ad_index (int): The index of the ad being shown.
            step (int): The current time step (used to update CTRs).

        Returns:
            int: 1 if clicked, 0 otherwise.
        """
        self.update_ctrs(step)  # ← Key line: dynamic environment behavior
        reward = self.ads[ad_index].show_ad()
        regret = self.optimal_ctr - self.ads[ad_index].true_ctr  # Difference from optimal
        self.total_regret += regret
        self.history.append((ad_index, reward))  # Log for analysis
        return reward

    def clone(self) -> "AdEnvironment":
        """
        Creates a deep copy of the current environment (including CTRs).
        Useful for ensuring independent runs in experiments.
        """
        cloned_ads = [AdBandit(ad.true_ctr, ad.name) for ad in self.ads]
        return AdEnvironment(cloned_ads)


def get_dynamic_ctrs(step: int) -> List[float]:
    """
    Returns a list of CTRs that change at a predefined point in time.
    Simulates a non-stationary reward distribution.

    Args:
        step (int): Current time step

    Returns:
        List[float]: Updated list of true CTRs for all ads
    """
    if step < 500:
        return [0.50, 0.30, 0.20, 0.10, 0.05]  # Initially, Ad 0 is best
    else:
        return [0.20, 0.25, 0.55, 0.15, 0.10]  # After step 500, Ad 2 becomes best
