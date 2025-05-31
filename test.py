import random
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy

from network import tactis
import data_generators as dg
import utils


if __name__ == '__main__':
    # Example: sample data
    samples = np.array([1, 1, 1, 2, 1, 1, 1])

    # Step 1: Count frequencies (histogram)
    values, counts = np.unique(samples, return_counts=True)

    # Step 2: Convert counts to probabilities
    probs = counts / counts.sum()

    # Step 3: Compute entropy (e.g., base 2 for bits)
    H = entropy(probs, base=2)
    print(f"Estimated entropy: {H}")
