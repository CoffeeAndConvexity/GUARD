import itertools
import numpy as np
import time
from solvers.nash import nash

def dbr(all_defender_actions, current_attacker_actions, D_a, udc, uduc):
    evs = []
    for da in all_defender_actions:
        coverage = list(itertools.chain.from_iterable(da))
        ev = 0
        for i,t in enumerate(current_attacker_actions):
            if t in coverage:
                ev += udc[t]*D_a[i] #add weighted defender covered value
            else:
                ev += uduc[t]*D_a[i] #add weighted defender uncovered value
        evs.append(ev)
    return all_defender_actions[np.argmax(np.array(evs))], max(evs)

# Linear Program Version of SF DBR
# def defender_best_response_schedule_form(resources, schedules_by_resource, expected_target_values):
#     """
#     Defender best response in schedule-form double oracle using attacker-strategy-dependent expected target values.

#     Parameters:
#         resources: list of resource IDs (e.g., [0, 1, 2])
#         schedules_by_resource: dict {r: list of (set, cost)} where each set is a schedule of target nodes
#         expected_target_values: dict {t: float}, representing the attack-weighted expected utility of defending t

#     Returns:
#         selected_schedules: list of sets, one per resource (empty set if no schedule available)
#         covered_targets: list of targets covered
#         defender_utility: float, total expected defender utility
#     """
#     model = gp.Model("DefenderBR_ScheduleForm")
#     model.setParam("OutputFlag", 0)

#     # Collect all unique targets from all schedules
#     all_targets = set()
#     for scheds in schedules_by_resource.values():
#         for s, _ in scheds:
#             all_targets.update(s)

#     # Identify usable resources (non-empty schedule lists)
#     usable_resources = [r for r in resources if len(schedules_by_resource[r]) > 0]
#     skipped_resources = [r for r in resources if r not in usable_resources]

#     # Decision variables
#     x = {}
#     for r in usable_resources:
#         for i in range(len(schedules_by_resource[r])):
#             x[r, i] = model.addVar(vtype=GRB.BINARY, name=f"x_{r}_{i}")

#     g = {}
#     for t in all_targets:
#         g[t] = model.addVar(vtype=GRB.BINARY, name=f"g_{t}")

#     # Constraint 1: Each usable resource picks exactly one schedule
#     for r in usable_resources:
#         model.addConstr(gp.quicksum(x[r, i] for i in range(len(schedules_by_resource[r]))) == 1)

#     # Constraint 2: Coverage logic
#     for t in all_targets:
#         model.addConstr(
#             g[t] <= gp.quicksum(
#                 x[r, i]
#                 for r in usable_resources
#                 for i, (sched, _) in enumerate(schedules_by_resource[r])
#                 if t in sched
#             )
#         )

#     # Objective: minimize expected utility from uncovered targets (zero-sum game setting)
#     model.setObjective(
#         gp.quicksum(expected_target_values[t] * g[t] for t in all_targets),
#         GRB.MINIMIZE
#     )

#     model.optimize()

#     if model.Status != GRB.OPTIMAL:
#         print("Warning: Model did not solve to optimality.")
#         return None, None, None

#     # Retrieve selected schedules
#     selected_schedules = {}
#     for r in usable_resources:
#         for i in range(len(schedules_by_resource[r])):
#             if x[r, i].X > 0.5:
#                 selected_schedules[r] = schedules_by_resource[r][i][0]
#                 break

#     # Add empty sets for skipped resources
#     for r in skipped_resources:
#         selected_schedules[r] = set()

#     # Return results in original resource order
#     selected_schedule_list = [selected_schedules[r] for r in resources]
#     covered_targets = [t for t in all_targets if g[t].X > 0.5]
#     utility = model.ObjVal

#     return selected_schedule_list, covered_targets, utility


def abr(all_attacker_actions, current_defender_actions, D_d, uac, uauc):
    evs = []
    for t in all_attacker_actions:
        ev = 0
        for i,da in enumerate(current_defender_actions):
            coverage = list(itertools.chain.from_iterable(da))
            if t in coverage:
                ev += uac[t]*D_d[i] #add weighted attacker covered value
            else:
                ev += uauc[t]*D_d[i] #add weighted attacker uncovered value
        evs.append(ev)
    return all_attacker_actions[np.argmax(np.array(evs))], -max(evs)

def generate_zero_sum_schedule_game_matrix(attacker_actions, defender_actions, udc, uduc):
    n = len(attacker_actions)
    m = len(defender_actions)
    U = np.zeros((n,m))

    for i, da in enumerate(defender_actions):
        coverage = list(itertools.chain.from_iterable(da))
        for j, t in enumerate(attacker_actions):
            if t in coverage:
                U[i,j] = udc[t]
            else:
                U[i,j] = uduc[t]

    return U

def get_score(target, schedule_assignment, udc, uduc):
    if target in itertools.chain.from_iterable(schedule_assignment):
        return udc[target]
    return uduc[target]

def expand_subgame(U, A_a, A_d, BR_a_in_U, BR_d_in_U, udc,uduc):
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
            new_U[i, new_m-1] = get_score(A_a[-1], A_d[i], udc, uduc)
    
    # Compute new **row** (if A_d expanded)
    if not BR_d_in_U:
        for i in range(new_m):  # Iterate over all columns (old + new)
            new_U[new_n-1, i] = get_score(A_a[i], A_d[-1], udc, uduc)

    return new_U

def double_oracle_sf(schedule_game_dict, eps=1e-12, verbose=True):
    all_defender_actions = schedule_game_dict["defender_actions"]
    all_attacker_actions = schedule_game_dict["targets"]
    udc = {t:schedule_game_dict["target_utilities"][1][i] for i,t in enumerate(all_attacker_actions)}
    uduc = {t:schedule_game_dict["target_utilities"][0][i] for i,t in enumerate(all_attacker_actions)}
    uac = {t:schedule_game_dict["target_utilities"][2][i] for i,t in enumerate(all_attacker_actions)}
    uauc = {t:schedule_game_dict["target_utilities"][3][i] for i,t in enumerate(all_attacker_actions)}

    A_d = all_defender_actions[:1]
    A_a = all_attacker_actions[:1]
    # print(A_d,A_a)
    U_subgame = generate_zero_sum_schedule_game_matrix(A_a, A_d, udc, uduc)
    # print(U_subgame)
    gap = np.inf
    gaps = []
    iteration_times = []
    c=0

    while gap > eps:
        start = time.time()
        # print(U_subgame)
        BR_a_in_U = False
        BR_d_in_U = False
        # print("A_a, A_d")
        # print(A_a,A_d)
        # print("Da, Dd, u")
        D_a, D_d, u_s = nash(U_subgame)
        # print(D_a, D_d, u_s)
        BR_a, u_BRa_Dd = abr(all_attacker_actions, A_d, D_d, uac, uauc)
        BR_d, u_BRd_Da = dbr(all_defender_actions, A_a, D_a, udc, uduc)
        # print("BR a, BR d")
        # print(BR_a,BR_d)
        # print("U BRa Dd, U BRd, Da")
        # print(u_BRa_Dd,u_BRd_Da)
        gap = abs(u_BRa_Dd - u_BRd_Da)
        gaps.append(gap)

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
            
        # print("BR_a_in_U,BR_d_in_U")
        # print(BR_a_in_U,BR_d_in_U)
        U_subgame = expand_subgame(U_subgame, A_a, A_d, BR_a_in_U, BR_d_in_U, udc,uduc)
        end = time.time()
        iteration_times.append(end-start)
        c+=1

        if verbose:
            print(f" U(D_d, BR A): {u_BRa_Dd}, U(D_a, BR D): {u_BRd_Da}")
            print(f"Current Gap: {gap}")

    return D_a, D_d, u_s, A_a, A_d, c, iteration_times, gaps