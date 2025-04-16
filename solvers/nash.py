import gurobipy as gp
from gurobipy import GRB
import numpy as np
def nash(
    utility_matrix
    ):

    num_rows = len(utility_matrix)
    num_cols = len(utility_matrix[0])

    # Create model
    m = gp.Model("NashLP_ColPlayer")
    m.Params.outputFlag = 0
    # Create variables
    y = [m.addVar(lb=0.0, name=f"y{j}") for j in range(num_cols)]
    z = m.addVar(lb=-float("inf"), name="z")
    m.setObjective(z, GRB.MINIMIZE) 
    # Add constraints 
    for i in range(num_rows):
        m.addConstr(z >= sum(utility_matrix[i][j] * y[j] for j in range(num_cols)), f"c_row_{i}")
    m.addConstr(sum(y) == 1, "sum_y")

    m.optimize()

    return (np.array([v.X for v in y]), [c.Pi for c in m.getConstrs()][:-1], m.ObjVal)