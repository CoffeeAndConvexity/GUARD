import gurobipy as gp
from gurobipy import GRB
import numpy as np
from collections import defaultdict, Counter

from nash import nash

def best_response_d(graph, D, T, S, tau, home_base_map, P, V):
    """
    Solves the Defender Best Response (DBR) problem using Gurobi.

    Parameters:
    - graph: NetworkX graph representing the game space.
    - D: Number of defenders.
    - T: Number of timesteps (including t=0. Fix in future).
    - S: Set of attacker-selected targets (nodes).
    - tau: Minimum timesteps required for interdiction.
    - home_base_map: Each defender D: their home base node.
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

    # 1. Defender Starts and Ends at Home Base
    for d in range(D):
        home_base = home_base_map[d]
        m.addConstr(v[home_base, 0, d] == 1, name=f"start_home_{d}")
        m.addConstr(v[home_base, T, d] == 1, name=f"end_home_{d}")
        for i in graph.nodes:
            if i != home_base:
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

from collections import Counter

def get_interdiction_probabilities(D_d, T, defender_actions):
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
        
        # Update target probabilities for being visited >= 2 times
        for target in T:
            if target_visit_counts[target.node] >= 2:
                target_probabilities[target.node] += prob

    return target_probabilities

def best_response_a(P_t, T, k):
    nodes = [t.node for t in T]
    scores = [t.value*(1-P_t[t.node]) for t in T]
    return tuple([x for _, x in sorted(zip(scores, nodes),reverse=True)][:min(len(T),k)])
    # return tuple(sorted([t.node for t in T], key=lambda node: next(t.value for t in T if t.node == node) * (1 - P_t[node]), reverse=True)[:k])

def get_interdictions(Tdi, defender_action):
    visited = [n for path in defender_action for n in path]
    c = Counter(visited)
    return [t for t in Tdi.keys() if c[t] >= 2]

def get_score(attacker_action, defender_action, Tdi):
    interdictions = get_interdictions(Tdi, defender_action)
    return -sum(list(set([Tdi[t] if t not in interdictions and t is not None else 0 for t in attacker_action])))

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

def expand_subgame(U, A_a, A_d, BR_a_in_U, BR_d_in_U, Tdi):
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
            new_U[i, new_m-1] = get_score(A_a[-1], A_d[i], Tdi)
    
    # Compute new **row** (if A_d expanded)
    if not BR_d_in_U:
        for i in range(new_m):  # Iterate over all columns (old + new)
            new_U[new_n-1, i] = get_score(A_a[i], A_d[-1], Tdi)

    return new_U

def double_oracle(game, tau, eps, initial_subgame_size=2, verbose=True):
    Tdi = {t.node: t.value for t in game.targets}
    Tdi[None] = 0

    num_timesteps = len(game.defender_strategies[0])
    num_attackers = game.attacker_strategies.shape[2]
    num_defenders = game.defender_strategies.shape[2]
    home_base_mapping = {i:game.home_bases[i] for i in range(num_defenders)}

    #Preprocess attacker action matrix into list of selection tuples
    a_selections = [tuple(game.attacker_strategies[i][0]) for i in range(len(game.attacker_strategies))]

    #Initialize Subgame
    A_a = a_selections[:initial_subgame_size]
    A_d = game.defender_strategies[:initial_subgame_size]
    
    subgame_rows = []
    for i in range(initial_subgame_size):
        subgame_rows.append([get_score(A_a[j], A_d[i], Tdi) for j in range(initial_subgame_size)])
    U_subgame = np.vstack(subgame_rows)
    
    gap = np.inf
    c = 0
        
    while gap > eps:
        BR_a_in_U = False
        BR_d_in_U = False
        
        #Solve subgame
        D_a, D_d, u_s = nash(U_subgame)
        
        #Get useful distributions for best responses
        P_a = get_attack_probabilities(D_a, A_a, Tdi)
        P_d = get_interdiction_probabilities(D_d, game.targets, A_d)
        
        #Get best responses
        BR_a = best_response_a(P_d, game.targets, num_attackers)
        BR_d = best_response_d(game.graph, num_defenders, num_timesteps-1, [t.node for t in game.targets], tau, home_base_mapping, P_a, [t.value for t in game.targets])

        #Get best response utilities and equilibrium gap
        u_BRa_Dd = -sum([(1-P_d[t])*Tdi[t] for t in list(set(BR_a)) if t is not None])
        u_BRd_Da = -sum([Tdi[t]*P_a[t] if t not in get_interdictions(Tdi, BR_d) else 0 for t in P_a])
        
        gap = abs(u_BRa_Dd - u_BRd_Da)
        
        #Expand subgame action sets and subgame U matrix
        if BR_a not in A_a:
            A_a.append(BR_a)
        else:
            BR_a_in_U = True

        if not np.any(np.all(np.array(BR_d).T == A_d, axis=(1, 2))):
            append_BR_d = np.expand_dims(np.array(BR_d).T, axis=0)
            A_d = np.concatenate((A_d, append_BR_d), axis=0)
        else:
            BR_d_in_U = True

        U_subgame = expand_subgame(U_subgame, A_a, A_d, BR_a_in_U, BR_d_in_U, Tdi)

        c+=1
        
        if verbose:
            print(f" U(D_d, BR A): {u_BRa_Dd}, U(D_a, BR D): {u_BRd_Da}")
            print(f"Current Gap: {gap}")

    return D_a, D_d, u_s, c