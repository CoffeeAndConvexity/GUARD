import gurobipy as gp
from gurobipy import GRB
import numpy as np
from collections import defaultdict

def defender_best_response_schedule_form(resources, schedules_by_resource, expected_target_values):
    """
    Defender best response in schedule-form double oracle using attacker-strategy-dependent expected target values.

    Parameters:
        resources: list of resource IDs (e.g., [0, 1, 2])
        schedules_by_resource: dict {r: list of sets}, where each set is a schedule of target nodes
        expected_target_values: dict {t: float}, representing the attack-weighted expected utility of defending t

    Returns:
        selected_schedules: dict {r: selected schedule}
        covered_targets: list of targets covered
        defender_utility: float, total expected defender utility
    """

    model = gp.Model("DefenderBR_ScheduleForm")
    model.setParam("OutputFlag", 0)

    all_targets = set()
    for scheds in schedules_by_resource.values():
        for s in scheds:
            all_targets.update(s)

    # Decision variables
    x = {}  # Binary selection: x[r, i] = 1 if resource r chooses schedule i
    for r in resources:
        for i, sched in enumerate(schedules_by_resource[r]):
            x[r, i] = model.addVar(vtype=GRB.BINARY, name=f"x_{r}_{i}")

    g = {}  # g[t] = 1 if target t is covered
    for t in all_targets:
        g[t] = model.addVar(vtype=GRB.BINARY, name=f"g_{t}")

    # Constraint 1: Each resource chooses exactly one schedule
    for r in resources:
        model.addConstr(gp.quicksum(x[r, i] for i in range(len(schedules_by_resource[r]))) == 1)

    # Constraint 2: g[t] ≤ sum of x[r, i] where schedule i for r includes t
    for t in all_targets:
        model.addConstr(
            g[t] <= gp.quicksum(
                x[r, i] for r in resources
                for i, sched in enumerate(schedules_by_resource[r])
                if t in sched
            )
        )

    # Objective: maximize total expected utility from covered targets
    model.setObjective(
        gp.quicksum(expected_target_values[t] * g[t] for t in all_targets),
        GRB.MINIMIZE
    )

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        print("Warning: Model did not solve to optimality.")
        return None, None, None

    # Retrieve defender strategies
    selected_schedules = {
        r: schedules_by_resource[r][i]
        for r in resources
        for i in range(len(schedules_by_resource[r]))
        if x[r, i].X > 0.5
    }

    covered_targets = [t for t in all_targets if g[t].X > 0.5]
    utility = model.ObjVal

    return list(selected_schedules.values()), covered_targets, utility

def attacker_best_response(all_targets, defender_distribution, defender_actions, target_values_matrix, negative=True):
    """
    Computes the best target for the attacker to attack against a defender mixed strategy.

    Parameters:
        all_targets (list[int]): All possible targets in the game (not limited to subgame).
        defender_distribution (list[float]): Probabilities over defender schedule assignments.
        defender_actions (list[list[set]]): List of defender strategies, each is a list of sets (schedules for each defender).
        target_values_matrix (np.ndarray): 4 x num_targets matrix.
        negative (bool): If True, returns negative utility for attacker (for zero-sum compatibility).

    Returns:
        best_target (int): Target with highest expected attacker utility.
        best_value (float): Expected value of attacking best_target against defender strategy.
    """
    # Step 1: Compute marginal coverage probability for each target
    target_coverage = {t: 0.0 for t in all_targets}
    for i, defender_action in enumerate(defender_actions):
        prob = defender_distribution[i]
        covered_targets = set().union(*defender_action)
        for t in covered_targets:
            if t in target_coverage:
                target_coverage[t] += prob

    # Step 2: Compute expected value for attacker for each target
    expected_utilities = {}
    for t in all_targets:
        p = target_coverage.get(t, 0.0)
        u_att_covered = target_values_matrix[2, t]
        u_att_uncovered = target_values_matrix[3, t]
        expected_utilities[t] = p * u_att_covered + (1 - p) * u_att_uncovered

    # Step 3: Return target with highest expected utility
    best_target = max(expected_utilities, key=expected_utilities.get)
    best_value = expected_utilities[best_target]

    if negative:
        best_value = -best_value

    return best_target, best_value

def compute_expected_defender_utilities(attacker_distribution, attacker_actions, target_values_matrix, all_targets):
    """
    Computes expected defender utility for each target given attacker distribution over attack targets.

    Parameters:
        attacker_distribution (list[float]): Probabilities for attacking each action (index corresponds to attacker_actions).
        attacker_actions (list[int]): List of target nodes corresponding to attacker strategies.
        target_values_matrix (np.ndarray): 4 x num_targets matrix.
            Row 0: defender utility if target is uncovered.
            Row 1: defender utility if target is covered.
        all_targets (list[int]): Full list of target node IDs.

    Returns:
        expected_utilities (dict[int, float]): Mapping from target index to expected defender utility.
    """
    expected_utilities = {t: 0.0 for t in all_targets}

    for i, target in enumerate(attacker_actions):
        prob = attacker_distribution[i]
        u_covered = target_values_matrix[1, target]
        u_uncovered = target_values_matrix[0, target]
        expected_utilities[target] = prob * u_covered + (1 - prob) * u_uncovered

    return expected_utilities

def get_score(target, schedule_assignment, target_utilities):
    """
    Returns the defender utility for a given target and defender schedule assignment.
    
    Args:
        target (int): Target node to evaluate.
        schedule_assignment (list[set]): One schedule per defender (list of sets of targets).
        target_utilities (np.ndarray): 4 x num_targets utility matrix.
                                       Row 0: Defender uncovered
                                       Row 1: Defender covered
                                       Row 2: Attacker covered
                                       Row 3: Attacker uncovered

    Returns:
        float: Defender utility value for the target, depending on whether it is covered.
    """
    is_covered = any(target in schedule for schedule in schedule_assignment)
    if is_covered:
        return target_utilities[1][target]  # Defender covered utility
    else:
        return target_utilities[0][target]  # Defender uncovered utility

def expand_subgame(U, A_a, A_d, BR_a_in_U, BR_d_in_U, target_utilities):
    """
    Expands the utility matrix U when A and/or B grow.
    
    Parameters:
    - U (np.array): Existing utility matrix of shape (n, m).
    - A (list): Updated list of attacker strategies.
    - B (list): Updated list of defender strategies.
    - A_expanded (bool): Flag indicating if A was expanded.
    - B_expanded (bool): Flag indicating if B was expanded.
    - get_score (function): Function to compute score for new pairs.
    
    Returns:
    - np.array: Updated utility matrix with new entries filled.
    """
    n, m = U.shape  # Get current matrix size
    
    if BR_a_in_U and BR_d_in_U:
        return U  # No expansion needed
    
    # Create new expanded matrix with placeholder values (assuming scores are non-positive)
    if not BR_a_in_U:
        new_m = m + 1
    else:
        new_m = m

    if not BR_d_in_U:
        new_n = n + 1
    else:
        new_n = n
        
    new_U = np.full((new_n, new_m), fill_value=1, dtype=U.dtype)
    
    # Copy over the existing values
    new_U[:n, :m] = U 

    # Compute new **column** (if A_a expanded)
    if not BR_a_in_U:
        for i in range(new_n):  # Iterate over all rows (old + new)
            new_U[i, new_m-1] = get_score(A_a[-1], A_d[i], target_utilities)
    
    # Compute new **row** (if A_d expanded)
    if not BR_d_in_U:
        for i in range(new_m):  # Iterate over all columns (old + new)
            new_U[new_n-1, i] = get_score(A_a[i], A_d[-1], target_utilities)

    return new_U


def double_oracle(schedule_form_di, initial_subgame_size=2, eps=1e-6, verbose=True):
    target_utilities = schedule_form_di["target_utilities"]
    schedules = schedule_form_di["schedules"]
    targets = list(range(target_utilities.shape[1]))
    num_defenders = len(list(schedules.keys()))
    resources = list(range(num_defenders))

    schedule_assignments = generate_defender_actions(schedules)
    A_d = schedule_assignments[:initial_subgame_size]
    A_a = targets[:initial_subgame_size]

    U_subgame = generate_schedule_game_matrix(A_a, A_d, target_utilities)[0]
    gap = np.inf
    c = 0
    
    if verbose:
        print("running...")
        
    while gap > eps:
        BR_a_in_U = False
        BR_d_in_U = False
        
        D_a, D_d, u_s = nash(U_subgame)
        BR_a, u_BRa_Dd = attacker_best_response(targets, D_d, A_d, target_utilities)
        expected_target_values = compute_expected_defender_utilities(D_a, A_a, target_utilities, targets)
        # print(expected_target_values)
        BR_d, _, _ = defender_best_response_schedule_form(resources, schedules_by_resource, expected_target_values)

        u_BRd_Da = sum(
            D_a[i] * (target_utilities[1, target] if any(target in s for s in BR_d) else target_utilities[0, target])
            for i, target in enumerate(A_a)
        )

        gap = abs(u_BRa_Dd - u_BRd_Da)

        if BR_a not in A_a:
            A_a.append(BR_a)
        else:
            BR_a_in_U = True

        for existing in A_d:
            if all(sched in existing for sched in BR_d) and all(sched in BR_d for sched in existing):
                BR_d_in_U = True
                break
        
        if not BR_d_in_U:
            A_d.append(BR_d)

        U_subgame = expand_subgame(U_subgame, A_a, A_d, BR_a_in_U, BR_d_in_U, target_utilities)
        
        c+=1

        if verbose:
            print(f" U(D_d, BR A): {u_BRa_Dd}, U(D_a, BR D): {u_BRd_Da}")
            print(f"Current Gap: {gap}")
            
    return D_a, D_d, u_s, c