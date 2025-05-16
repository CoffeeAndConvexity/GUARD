import gurobipy as gp
from gurobipy import GRB
import numpy as np
from collections import defaultdict, Counter
import time

from solvers.nash import nash

def best_response_d(graph, D, T, S, tau, home_base_map, P, V, force_return=False):
    """
    Solves the Defender Best Response (DBR) problem using Gurobi.

    Parameters:
    - graph: NetworkX graph representing the game space.
    - D: Number of defenders.
    - T: Number of timesteps (including t=0. Fix in future).
    - S: Set of attacker-selected targets (nodes).
    - tau: Minimum timesteps required for interdiction.
    - home_base_map: Each defender D: their home base node(s).
    - P: Dictionary of attacker target selection probabilities {node: probability}.
    - V: Dictionary of target values {node: value}.

    Returns:
    - defender_paths: A list of optimal paths for each defender.
    """
    for node in graph.nodes:
        graph.add_edge(node, node)  # Add a self-loop to allow waiting

    # Initialize Gurobi model
    m = gp.Model("Defender_Best_Response")
    m.setParam("OutputFlag", 0)  # Suppress output

    # Decision Variables
    v = m.addVars(graph.nodes, range(T + 1), range(D), vtype=GRB.BINARY, name="v")  # Defender presence
    g = m.addVars(S, vtype=GRB.BINARY, name="g")  # Interdiction status

    # 1. Defender Starts and Ends at a Home Base (optionally same home base)
    for d in range(D):
        home_bases = home_base_map[d]

        if force_return:
            # Start and end must be at the same home base node
            for hb in home_bases:
                m.addConstr(v[hb, 0, d] == v[hb, T, d], name=f"force_return_same_{hb}_{d}")
            m.addConstr(gp.quicksum(v[i, 0, d] for i in home_bases) == 1, name=f"start_home_{d}")
        else:
            # Start and end can be at different home bases
            m.addConstr(gp.quicksum(v[i, 0, d] for i in home_bases) == 1, name=f"start_home_{d}")
            m.addConstr(gp.quicksum(v[i, T, d] for i in home_bases) == 1, name=f"end_home_{d}")

        for i in graph.nodes:
            if i not in home_bases:
                m.addConstr(v[i, 0, d] == 0, name=f"no_start_{i}_{d}")
                m.addConstr(v[i, T, d] == 0, name=f"no_end_{i}_{d}")

    # 2. Flow Conservation for Defender Movement
    for t in range(1, T + 1):
        for d in range(D):
            for i in graph.nodes:
                neighbors = list(graph.neighbors(i))
                if neighbors:
                    m.addConstr(v[i, t, d] <= gp.quicksum(v[u, t - 1, d] for u in neighbors),
                                name=f"flow_{i}_{t}_{d}")

    # 3. Ensure each defender is in exactly one node at any timestep
    for t in range(T + 1):
        for d in range(D):
            m.addConstr(gp.quicksum(v[i, t, d] for i in graph.nodes) == 1, name=f"one_location_{t}_{d}")

    # 4. Interdiction Constraints
    for t in S:  # Loop over all targets
        m.addConstr(
        tau*g[t] <= sum(v[t, t_step, d] for t_step in range(T) for d in range(D)),
        name=f"interdiction_{t}"
        )


    # Objective: Maximize Interdiction Utility Weighted by Attacker Probabilities
    m.setObjective(gp.quicksum(P[s] * g[s] * V[s] for s in S), GRB.MAXIMIZE)

    # Solve the model
    m.optimize()

    paths=[]
    if m.Status == GRB.OPTIMAL:
        for d in range(D):
            path = []
            for t in range(T + 1):
                for i in graph.nodes:
                    if v[i, t, d].X > 0.5:  # Defender is present at node i at timestep t
                        path.append(i)
                        break
            paths.append(path)
    else:
        print(f"Optimization did not converge. Status: {m.Status}")
    return paths

def get_interdiction_probabilities(D_d, T, defender_actions,tau):
    num_targets = len(T)
    target_probabilities = {t.node: 0 for t in T}
    target_probabilities[None] = 0

    for i, paths in enumerate(defender_actions):
        prob = D_d[i]
        if prob == 0:
            continue  # Skip if this path option has zero probability

        # Count visits per target in this path option
        target_visit_counts = {t.node: 0 for t in T}
        for path in paths:
            for node in path:
                if node in target_visit_counts:
                    target_visit_counts[node] += 1
        
        # Update target probabilities for being visited >= tau times
        for target in T:
            if target_visit_counts[target.node] >= tau:
                target_probabilities[target.node] += prob

    return target_probabilities

def best_response_a(P_t, T, k):
    nodes = [t.node for t in T]
    scores = [t.attacker_value*(1-P_t[t.node]) for t in T]
    return tuple([x for _, x in sorted(zip(scores, nodes),reverse=True)][:min(len(T),k)])

def get_interdictions(Tdi, defender_action, tau):
    visited = [n for path in defender_action for n in path]
    c = Counter(visited)
    return [t for t in Tdi.keys() if c[t] >= tau]

def get_score(attacker_action, defender_action, Tdi, tau):
    interdictions = get_interdictions(Tdi, defender_action, tau)
    return -sum(list([Tdi[t] if t not in interdictions and t is not None else 0 for t in attacker_action]))

def get_attack_probabilities(D_a, attacker_actions, Tdi):
    target_probabilities = defaultdict(float)
    for t in Tdi:
        target_probabilities[t] = 0

    num_actions = len(attacker_actions)

    for action_idx in range(num_actions):
        action_prob = D_a[action_idx]  # Probability of selecting this strategy
        unique_targets = set(attacker_actions[action_idx])  # Get all targets selected in this strategy
        for target in unique_targets:
            target_probabilities[target] += action_prob  # Aggregate probabilities

    return dict(target_probabilities)

def expand_subgame(U, A_a, A_d, BR_a_in_U, BR_d_in_U, Tdi, tau):
    """
    Expands the utility matrix U when A and/or B grow.
    
    Parameters:
    - U (np.array): Existing utility matrix of shape (n, m).
    - A (list): Updated list of attacker actions.
    - B (list): Updated list of defender actions.
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

            test = get_score(A_a[-1], A_d[i], Tdi, tau)

            new_U[i, new_m-1] = get_score(A_a[-1], A_d[i], Tdi, tau)

    
    # Compute new **row** (if A_d expanded)
    if not BR_d_in_U:
        for i in range(new_m):  # Iterate over all columns (old + new)
            new_U[new_n-1, i] = get_score(A_a[i], A_d[-1], Tdi, tau)
    return new_U

def double_oracle(game, eps=1e-6, verbose=True):
    Tdi = {t.node: t.attacker_value for t in game.targets}
    Tdi[None] = 0

    home_base_mapping = {i:game.home_bases[i] for i in range(game.num_defenders)}

    #Initialize Subgame
    A_a = [best_response_a({t.node:0 for t in game.targets}, game.targets, game.num_attackers)]
    P_a = get_attack_probabilities([1], A_a, Tdi) # D_a initial, initial actions, targets

    BR_d0 = best_response_d(game.graph, game.num_defenders, game.num_timesteps-1, [t.node for t in game.targets], 
                            game.defense_time_threshold, home_base_mapping, P_a, {t.node:t.attacker_value for t in game.targets}, game.force_return)

    A_d = np.array(BR_d0).T[np.newaxis, :, :]

    subgame_rows = [[get_score(A_a[0], A_d[0], Tdi, game.defense_time_threshold)]]
    U_subgame = np.vstack(subgame_rows).astype(float)

    gap = np.inf
    c = 0
    iteration_times = []
    gaps = []
        
    while gap > eps:
        start = time.time()
        BR_a_in_U = False
        BR_d_in_U = False
        
        #Solve subgame
        D_a, D_d, u_s = nash(U_subgame)

        #Get useful distributions for best responses
        P_a = get_attack_probabilities(D_a, A_a, Tdi)
        P_d = get_interdiction_probabilities(D_d, game.targets, A_d, game.defense_time_threshold)

        #Get best responses
        BR_a = best_response_a(P_d, game.targets, game.num_attackers)
        BR_d = best_response_d(game.graph, game.num_defenders, game.num_timesteps-1, [t.node for t in game.targets], game.defense_time_threshold, home_base_mapping, P_a, {t.node:t.attacker_value for t in game.targets}, game.force_return)

        #Get best response utilities and equilibrium gap
        u_BRa_Dd = -sum([(1-P_d[t])*Tdi[t] for t in list(set(BR_a)) if t is not None])
        u_BRd_Da = -sum([Tdi[t]*P_a[t] if t not in get_interdictions(Tdi, BR_d, game.defense_time_threshold) else 0 for t in P_a])

        gap = abs(u_BRa_Dd - u_BRd_Da)
        gaps.append(gap)

        #Expand subgame action sets and subgame U matrix
        if BR_a not in A_a:

            A_a.append(BR_a)
        else:
            BR_a_in_U = True

        BR_d_arr = np.array(BR_d).T  # shape (T, D)
        BR_d_in_U = any(np.array_equal(BR_d_arr, a) for a in A_d)

        if not BR_d_in_U:
            append_BR_d = np.expand_dims(BR_d_arr, axis=0)
            A_d = np.concatenate((A_d, append_BR_d), axis=0)
        else:
            BR_d_in_U = True

        U_subgame = expand_subgame(U_subgame, A_a, A_d, BR_a_in_U, BR_d_in_U, Tdi, game.defense_time_threshold)

        end = time.time()
        iteration_times.append(end-start)
        c+=1
        
        if verbose:
            print(f" U(D_d, BR A): {u_BRa_Dd}, U(D_a, BR D): {u_BRd_Da}")
            print(f"Current Gap: {gap}")

    return D_a, D_d, u_s, A_a, A_d, c, iteration_times, gaps