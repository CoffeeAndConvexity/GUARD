import numpy as np

def compute_least_exploitable_pure_strategies(payoffs):
    nb_actions = payoffs.shape

    row_max = np.max(payoffs, axis=0)
    row_exploit = row_max - payoffs

    col_min = np.min(payoffs, axis=1)
    col_exploit = payoffs - col_min[:, np.newaxis]

    final_exploitability = row_exploit + col_exploit

    flat_index = np.argmin(final_exploitability)
    coordinates = np.unravel_index(flat_index, final_exploitability.shape)
    coordinates = [int(x) for x in coordinates]

    return (float(np.min(final_exploitability)), coordinates)