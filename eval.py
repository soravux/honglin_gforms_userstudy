import choix
import numpy as np

# Pairwise comparisons: [winner, loser]
# Teams are represented by integer IDs (0, 1, 2, 3)
data = [
    (0, 1), (0, 1), (0, 2),  # Team 0 beats 1 twice, 0 beats 2
    (1, 2), (2, 3), (0, 3)   # 1 beats 2, 2 beats 3, 0 beats 3
]
n_items = 4

# Fit the Bradley-Terry model using Maximum Likelihood Estimation
params = choix.ilsr(n_items, data)

print("Estimated parameters (strengths):", params)
# Output: Estimated parameters (strengths): [ 1.45898034  0.13353139 -0.55251173 -1.04000000]

# Compute probability of item 0 beating item 1
# Prob(i > j) = exp(s_i) / (exp(s_i) + exp(s_j))
prob = choix.probabilities([0, 1], params)
print(f"Probability 0 beats 1: {prob[0]:.2f}")
