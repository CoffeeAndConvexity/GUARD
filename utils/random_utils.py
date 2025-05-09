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

    If general_sum:
        - Randomize covered/uncovered values separately for attacker and defender.
        - Ensure |covered| <= |uncovered| by swapping values when needed.
        - If respect_sign_roles is True, defender values remain negative, attacker values remain positive.

    If zero-sum:
        - Randomize attacker values, and set defender values to be their negative symmetric.

    Returns:
        np.ndarray of shape (4, T)
    """
    T = target_utility_matrix.shape[1]

    if general_sum:
        # Row indices
        DEF_U, DEF_C, ATT_C, ATT_U = 0, 1, 2, 3

        # Get min/max ranges for each type
        def_uncovered_vals = target_utility_matrix[DEF_U]
        def_covered_vals   = target_utility_matrix[DEF_C]
        att_covered_vals   = target_utility_matrix[ATT_C]
        att_uncovered_vals = target_utility_matrix[ATT_U]

        def_u_min, def_u_max = def_uncovered_vals.min(), def_uncovered_vals.max()
        def_c_min, def_c_max = def_covered_vals.min(), def_covered_vals.max()
        att_c_min, att_c_max = att_covered_vals.min(), att_covered_vals.max()
        att_u_min, att_u_max = att_uncovered_vals.min(), att_uncovered_vals.max()

        # Sample random values for each row independently
        def_uncovered_rand = np.random.uniform(def_u_min, def_u_max, size=T)
        def_covered_rand   = np.random.uniform(def_c_min, def_c_max, size=T)
        att_covered_rand   = np.random.uniform(att_c_min, att_c_max, size=T)
        att_uncovered_rand = np.random.uniform(att_u_min, att_u_max, size=T)

        # Enforce |covered| ≤ |uncovered| by swapping if needed
        for t in range(T):
            # Defender
            if abs(def_covered_rand[t]) > abs(def_uncovered_rand[t]):
                def_covered_rand[t], def_uncovered_rand[t] = def_uncovered_rand[t], def_covered_rand[t]

            # Attacker
            if abs(att_covered_rand[t]) > abs(att_uncovered_rand[t]):
                att_covered_rand[t], att_uncovered_rand[t] = att_uncovered_rand[t], att_covered_rand[t]

        if respect_sign_roles:
            # Ensure defenders are negative, attackers positive
            def_covered_rand = -np.abs(def_covered_rand)
            def_uncovered_rand = -np.abs(def_uncovered_rand)
            att_covered_rand = np.abs(att_covered_rand)
            att_uncovered_rand = np.abs(att_uncovered_rand)

        randomized_matrix = np.vstack([
            def_uncovered_rand,
            def_covered_rand,
            att_covered_rand,
            att_uncovered_rand
        ])
        return randomized_matrix

    else:
        # Zero-sum: randomize attacker utilities, defender is symmetric negative
        abs_vals = np.abs(target_utility_matrix)
        min_val = np.min(abs_vals)
        max_val = np.max(abs_vals)

        attacker_covered = np.random.uniform(low=min_val, high=max_val, size=T)
        attacker_uncovered = np.random.uniform(low=min_val, high=max_val, size=T)

        # Enforce |covered| ≤ |uncovered|
        for t in range(T):
            if attacker_covered[t] > attacker_uncovered[t]:
                attacker_covered[t], attacker_uncovered[t] = attacker_uncovered[t], attacker_covered[t]

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


def generate_random_target_utility_matrix_like_v2(target_utility_matrix, general_sum, respect_sign_roles=False):
    """
    Randomizes a 4×T target utility matrix for schedule-form games.

    If general_sum:
        - Randomize all values (def/att, covered/uncovered) from a single shared range.
        - Range is defined as: min(abs(covered values)), max(abs(uncovered values)).
        - Enforce |covered| ≤ |uncovered| per player via flipping if needed.
        - If respect_sign_roles is True, defender values remain negative, attacker values remain positive.

    If zero-sum:
        - Randomize attacker values, and set defender values to be their negative symmetric.

    Returns:
        np.ndarray of shape (4, T)
    """
    T = target_utility_matrix.shape[1]

    if general_sum:
        # Row indices
        DEF_U, DEF_C, ATT_C, ATT_U = 0, 1, 2, 3

        # Get global range from all values
        all_covered = np.abs(np.concatenate([
            target_utility_matrix[DEF_C],
            target_utility_matrix[ATT_C]
        ]))
        all_uncovered = np.abs(np.concatenate([
            target_utility_matrix[DEF_U],
            target_utility_matrix[ATT_U]
        ]))

        range_min = np.min(all_covered)
        range_max = np.max(all_uncovered)

        # Sample values from this unified range
        def_covered_rand   = np.random.uniform(range_min, range_max, size=T)
        def_uncovered_rand = np.random.uniform(range_min, range_max, size=T)
        att_covered_rand   = np.random.uniform(range_min, range_max, size=T)
        att_uncovered_rand = np.random.uniform(range_min, range_max, size=T)

        # Flip values if |covered| > |uncovered|
        for t in range(T):
            if abs(def_covered_rand[t]) > abs(def_uncovered_rand[t]):
                def_covered_rand[t], def_uncovered_rand[t] = def_uncovered_rand[t], def_covered_rand[t]
            if abs(att_covered_rand[t]) > abs(att_uncovered_rand[t]):
                att_covered_rand[t], att_uncovered_rand[t] = att_uncovered_rand[t], att_covered_rand[t]

        if respect_sign_roles:
            def_covered_rand = -np.abs(def_covered_rand)
            def_uncovered_rand = -np.abs(def_uncovered_rand)
            att_covered_rand = np.abs(att_covered_rand)
            att_uncovered_rand = np.abs(att_uncovered_rand)

        randomized_matrix = np.vstack([
            def_uncovered_rand,
            def_covered_rand,
            att_covered_rand,
            att_uncovered_rand
        ])
        return randomized_matrix

    else:
        # Zero-sum version (same as before)
        abs_vals = np.abs(target_utility_matrix)
        min_val = np.min(abs_vals)
        max_val = np.max(abs_vals)

        attacker_covered = np.random.uniform(low=min_val, high=max_val, size=T)
        attacker_uncovered = np.random.uniform(low=min_val, high=max_val, size=T)

        for t in range(T):
            if attacker_covered[t] > attacker_uncovered[t]:
                attacker_covered[t], attacker_uncovered[t] = attacker_uncovered[t], attacker_covered[t]

        defender_covered = -attacker_covered
        defender_uncovered = -attacker_uncovered

        randomized_matrix = np.vstack([
            defender_uncovered,
            defender_covered,
            attacker_covered,
            attacker_uncovered
        ])
        return randomized_matrix