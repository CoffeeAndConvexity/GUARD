import gurobipy as gp
from gurobipy import GRB
import numpy as np

def mip(
    utility_matrix,
    support_bound
    ):
    '''_summary_
    Bounded support player is the row player

    Args:
        utility_matrix: payoff matrix for zero-sum game 
        support_bound (int): support bound for the row player
        
    Returns:
        x (array): probability distribution over the actions of the sparse player
        obj_val: value of the game
    '''

    num_rows = len(utility_matrix)
    num_cols = len(utility_matrix[0])

    # Create model
    m = gp.Model("NashLP_RowPlayer")
    m.Params.outputFlag = 0
    
    # Create variables
    x = [m.addVar(lb=0.0, ub=1.0, name=f"x{i}") for i in range(num_rows)]
    indicator_x = [m.addVar(vtype=GRB.BINARY, name=f"ind_x{i}") for i in range(num_rows)]
    g = m.addVar(lb=-float("inf"), name="g")

    m.setObjective(g, GRB.MINIMIZE) 

    # Add constraints 
    for j in range(num_cols):
        m.addConstr(g >= sum(utility_matrix[i][j] * x[i] for i in range(num_rows)), f"c_col_{j}")
    for i in range(num_rows):
        m.addConstr(x[i] <= indicator_x[i])
    m.addConstr(sum(x) == 1, "sum_x")
    m.addConstr(sum(indicator_x) <= support_bound, "bounded_support")

    m.optimize()

    return m.ObjVal