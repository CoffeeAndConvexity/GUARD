import gurobipy as gp
from gurobipy import GRB
import numpy as np

def mip(utility_matrix, support_bound):
    """
    Defender-bounded support MIP from defender's POV.
    Defender (row player) tries to maximize value of the game.

    Args:
        utility_matrix (2D list): Defender utility matrix (negative values)
        support_bound (int): Max number of actions defender can place weight on

    Returns:
        obj_val: value of the game (defender's perspective)
        defender_strategy: optimal defender mixed strategy (row player)
    """
    num_rows = len(utility_matrix)
    num_cols = len(utility_matrix[0])

    # Create model
    m = gp.Model("DefenderMIP")
    m.Params.outputFlag = 0

    # Variables
    x = [m.addVar(lb=0.0, ub=1.0, name=f"x{i}") for i in range(num_rows)]  # strategy
    indicator_x = [m.addVar(vtype=GRB.BINARY, name=f"ind_x{i}") for i in range(num_rows)]  # support indicator
    g = m.addVar(lb=-float("inf"), name="g")  # defender value to maximize

    m.setObjective(g, GRB.MAXIMIZE)

    # Column player (attacker) best response constraints
    for j in range(num_cols):
        m.addConstr(
            g <= sum(utility_matrix[i][j] * x[i] for i in range(num_rows)),
            name=f"attacker_br_{j}"
        )

    # Support constraints
    for i in range(num_rows):
        m.addConstr(x[i] <= indicator_x[i], name=f"support_link_{i}")
    m.addConstr(sum(x) == 1, "prob_dist")
    m.addConstr(sum(indicator_x) <= support_bound, "support_bound")

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return None, None

    defender_strategy = np.array([var.X for var in x])
    return m.ObjVal, defender_strategy