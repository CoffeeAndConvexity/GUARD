import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_sse_lp(targets, resources, A_r, u_d_covered, u_d_uncovered, u_a_covered, u_a_uncovered):
    """
    Solves the SSE LP for each possible attacker best response t*, returning the defender strategy
    (marginal coverage) that maximizes defender utility.

    Parameters:
        targets: list of target node IDs.
        resources: list of resource IDs.
        A_r: dict {r: list of t} that each resource r can cover.
        u_d_covered: dict {t: float}, defender utility if t is covered.
        u_d_uncovered: dict {t: float}, defender utility if t is uncovered.
        u_a_covered: dict {t: float}, attacker utility if t is covered.
        u_a_uncovered: dict {t: float}, attacker utility if t is uncovered.

    Returns:
        best_t (int): Optimal attacker best response t*.
        best_coverage (dict): Marginal coverage {t: c_t} under defender strategy.
        best_defender_utility (float): Defender utility from committing to this strategy.
    """
    best_t = None
    best_coverage = None
    best_defender_utility = -float("inf")

    for t_star in targets:
        model = gp.Model("SSE_LP")
        model.setParam("OutputFlag", 0)

        # Variables
        c_t = model.addVars(targets, lb=0.0, ub=1.0, name="c_t")  # marginal coverage
        c_rt = model.addVars(resources, targets, lb=0.0, ub=1.0, name="c_rt")

        # Objective: Maximize defender utility at t_star
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