import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_general_sum_normal_form(defender_matrix, attacker_matrix):
    """
    Solves a general-sum 2-player normal-form game using LP formulation for commitment to mixed strategies.

    Parameters:
        defender_matrix: numpy array (m x n), defender utility for each (s, t)
        attacker_matrix: numpy array (m x n), attacker utility for each (s, t)

    Returns:
        best_response_t: int, attacker pure strategy (t) against which the best defender mixed strategy commits
        best_defender_strategy: dict, {s: p_s}, optimal mixed strategy over defender pure strategies
        best_defender_utility: float, expected utility for the defender
    """
    num_defender_strategies, num_attacker_strategies = defender_matrix.shape
    best_defender_utility = -float('inf')
    best_response_t = None
    best_defender_strategy = None

    for t in range(num_attacker_strategies):
        model = gp.Model()
        model.setParam("OutputFlag", 0)

        # Variables: p_s for each defender strategy s
        p = model.addVars(num_defender_strategies, lb=0.0, ub=1.0, name="p")

        # Objective: maximize sum_s p_s * u_l(s, t)
        model.setObjective(
            gp.quicksum(p[s] * defender_matrix[s, t] for s in range(num_defender_strategies)),
            GRB.MAXIMIZE
        )

        # Constraint: sum_s p_s = 1
        model.addConstr(gp.quicksum(p[s] for s in range(num_defender_strategies)) == 1, name="prob_sum")

        # Follower best response condition: utility of t >= utility of any t'
        for t_prime in range(num_attacker_strategies):
            lhs = gp.quicksum(p[s] * attacker_matrix[s, t] for s in range(num_defender_strategies))
            rhs = gp.quicksum(p[s] * attacker_matrix[s, t_prime] for s in range(num_defender_strategies))
            model.addConstr(lhs >= rhs, name=f"br_t_{t_prime}")

        model.optimize()

        if model.status == GRB.OPTIMAL:
            defender_utility = model.objVal
            if defender_utility > best_defender_utility:
                best_defender_utility = defender_utility
                best_response_t = t
                best_defender_strategy = {s: p[s].X for s in range(num_defender_strategies)}

    return best_response_t, best_defender_strategy, best_defender_utility