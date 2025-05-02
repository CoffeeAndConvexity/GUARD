import random
import numpy as np

def generate_random_utility_matrix_like(utility_matrix):
    """
    Takes a 2D numpy array utility_matrix and returns a matrix of the same shape
    with float values uniformly randomly sampled between the min and max value
    of the input matrix.
    """
    min_val = np.min(utility_matrix)
    max_val = np.max(utility_matrix)
    
    random_matrix = np.random.uniform(low=min_val, high=max_val, size=utility_matrix.shape)
    return random_matrix

def generate_random_target_utility_matrix_like(target_utility_matrix, general_sum, respect_sign_roles=False):
    """
    Randomizes a 4×T target utility matrix for schedule-form games.

    Parameters:
    - target_utility_matrix: np.ndarray of shape (4, T)
    - general_sum: bool, whether the game is general-sum or zero-sum
    - respect_sign_roles: bool, if True in general-sum mode, defenders are only assigned values from the original defender range,
      and attackers from the original attacker range. Prevents role sign reversal.

    Returns:
    - randomized_matrix: np.ndarray of shape (4, T)
    """
    if general_sum:
        if respect_sign_roles:
            T = target_utility_matrix.shape[1]

            # Split original values
            defender_rows = target_utility_matrix[0:2, :]
            attacker_rows = target_utility_matrix[2:4, :]

            # Get independent min/max ranges
            def_min = np.min(defender_rows)
            def_max = np.max(defender_rows)
            att_min = np.min(attacker_rows)
            att_max = np.max(attacker_rows)

            defender_random = np.random.uniform(low=def_min, high=def_max, size=(2, T))
            attacker_random = np.random.uniform(low=att_min, high=att_max, size=(2, T))

            randomized_matrix = np.vstack([defender_random, attacker_random])
        else:
            min_val = np.min(target_utility_matrix)
            max_val = np.max(target_utility_matrix)
            randomized_matrix = np.random.uniform(low=min_val, high=max_val, size=target_utility_matrix.shape)

        return randomized_matrix

    else:
        # Zero-sum: only randomize attacker rows, mirror for defenders
        abs_vals = np.abs(target_utility_matrix)
        min_val = np.min(abs_vals)
        max_val = np.max(abs_vals)

        T = target_utility_matrix.shape[1]
        attacker_covered = np.random.uniform(low=min_val, high=max_val, size=T)
        attacker_uncovered = np.random.uniform(low=min_val, high=max_val, size=T)

        defender_covered = -attacker_covered
        defender_uncovered = -attacker_uncovered

        randomized_matrix = np.vstack([
            defender_uncovered,   # row 0
            defender_covered,     # row 1
            attacker_covered,     # row 2
            attacker_uncovered    # row 3
        ])

        return randomized_matrix


def generate_random_schedule_mapping_like(original_schedule_mapping, num_samples=50):
    """
    Randomizes a schedule mapping by:
    - Sampling 3 random targets for each new schedule (duplicates allowed, but deduped via set)
    - Assigning each schedule a random cost between min and max original costs
    - Generating identical mappings for both defenders (0 and 1)

    Args:
        original_schedule_mapping (dict): Original mapping {defender_id: [(set of targets, cost), ...]}
        num_samples (int): Number of randomized schedules to generate

    Returns:
        dict: New randomized schedule mapping with same structure
    """
    # Get all unique targets from original mapping
    all_targets = set()
    all_costs = []
    for schedule_list in original_schedule_mapping.values():
        for target_set, cost in schedule_list:
            all_targets.update(target_set)
            all_costs.append(cost)

    all_targets = list(all_targets)
    min_cost = min(all_costs)
    max_cost = max(all_costs)

    new_schedules = []
    for _ in range(num_samples):
        # Randomly sample 3 (possibly repeating) targets and use set() to deduplicate
        sampled_targets = set(random.choices(all_targets, k=3))
        random_cost = float(np.random.uniform(min_cost, max_cost))
        new_schedules.append((sampled_targets, random_cost))

    # Use the same randomized schedule for both defenders 0 and 1
    return {
        0: new_schedules.copy(),
        1: new_schedules.copy()
    }