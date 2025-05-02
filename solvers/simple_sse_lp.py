import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_sse_lp(schedule_di):
    """
    Solves the SSE LP for a simple schedule form game, processing input from a schedule_di dictionary.

    Parameters:
        schedule_di: dict containing
            - 'targets': list of target node IDs
            - 'schedules': dict {resource_id: list of (schedule_set, cost)}
            - 'target_utilities': np.ndarray of shape (4, num_targets) 
                                  (defender uncovered, defender covered, attacker covered, attacker uncovered)

    Returns:
        best_t (int): Optimal attacker best response t*
        best_coverage (dict): Marginal coverage {t: c_t} under defender strategy
        best_defender_utility (float): Defender utility from committing to this strategy
    """
    import gurobipy as gp
    from gurobipy import GRB

    # Unpack schedule_di
    targets = schedule_di['targets']
    schedules = schedule_di['schedules']
    target_utilities = schedule_di['target_utilities']
    max_abs_val = np.abs(target_utilities).max()
    if max_abs_val > 0:
        target_utilities = target_utilities / max_abs_val

    resources = sorted(schedules.keys())  # list of resource IDs
    
    # Build A_r: resource -> list of individual targets it can cover
    A_r = {}
    for r, sched_list in schedules.items():
        A_r[r] = []
        for sched, _ in sched_list:  # ignore cost
            for t in sched:
                if t not in A_r[r]:
                    A_r[r].append(t)

    # Build utility dictionaries carefully (targets can be non-consecutive integers)
    u_d_uncovered = {}
    u_d_covered = {}
    u_a_covered = {}
    u_a_uncovered = {}

    for idx, t in enumerate(targets):
        u_d_uncovered[t] = target_utilities[0, idx]
        u_d_covered[t] = target_utilities[1, idx]
        u_a_covered[t] = target_utilities[2, idx]
        u_a_uncovered[t] = target_utilities[3, idx]

    # --- Solve LP for each possible t_star ---

    best_t = None
    best_coverage = None
    best_defender_utility = -float("inf")

    for t_star in targets:
        model = gp.Model("SSE_LP")
        model.setParam("OutputFlag", 0)

        # Variables
        c_t = model.addVars(targets, lb=0.0, ub=1.0, name="c_t")  # marginal coverage
        c_rt = model.addVars(resources, targets, lb=0.0, ub=1.0, name="c_rt")

        # Objective: maximize defender utility at t_star
        model.setObjective(
            c_t[t_star] * u_d_covered[t_star] + (1 - c_t[t_star]) * u_d_uncovered[t_star],
            GRB.MAXIMIZE
        )

        # Coverage definition
        for t in targets:
            model.addConstr(
                c_t[t] == gp.quicksum(c_rt[r, t] for r in resources if t in A_r[r]),
                name=f"cover_def_{t}"
            )

        # Resource constraints
        for r in resources:
            model.addConstr(
                gp.quicksum(c_rt[r, t] for t in A_r[r]) <= 1,
                name=f"resource_bound_{r}"
            )

        # Attacker best response constraints
        for t in targets:
            if t == t_star:
                continue

            u_att_t = c_t[t] * u_a_covered[t] + (1 - c_t[t]) * u_a_uncovered[t]
            u_att_star = c_t[t_star] * u_a_covered[t_star] + (1 - c_t[t_star]) * u_a_uncovered[t_star]
            model.addConstr(u_att_t <= u_att_star, name=f"attacker_br_{t}")

        model.optimize()

        if model.status == GRB.OPTIMAL:
            defender_utility = model.objVal
            if defender_utility > best_defender_utility:
                best_t = t_star
                best_defender_utility = defender_utility
                best_coverage = {t: c_t[t].X for t in targets}

    return best_t, best_coverage, best_defender_utility