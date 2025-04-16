import numpy as np
import itertools
import networkx as nx

def format_strategies(strategy_matrix):
    """
    Post-processes a strategy matrix to a 2D array where each row is a strategy
    and each column corresponds to a timestep. Supports multiple players, and squeezes
    any extra dimensions.
    
    Input:
    - strategy_matrix: 3D numpy array of shape (num_strategies, num_timesteps, num_players)
    
    Returns:
    - 2D numpy array where each row is a strategy and each column corresponds to a timestep.
      If there's only one player, it removes the extra dimension.
    """
    num_strategies, num_timesteps, num_players = strategy_matrix.shape
    
    # Squeeze out the extra dimension if there's only one attacker
    if num_players == 1:
        formatted_strategies = np.squeeze(strategy_matrix, axis=2)
    else:
        formatted_strategies = []
        for strategy in strategy_matrix:
            formatted_strategy = []
            for timestep in range(num_timesteps):
                attacker_positions = tuple(strategy[timestep][a] for a in range(num_players))
                formatted_strategy.append(attacker_positions)
            formatted_strategies.append(formatted_strategy)
        
        formatted_strategies = np.array(formatted_strategies)
    
    return formatted_strategies

def bfs_shortest_paths(graph, start):
    return nx.single_source_shortest_path_length(graph, start)

def greedy_movement_cost(start_node, targets, dist):
    """Greedy path cost visiting all targets from start_node (no return)."""
    unvisited = set(targets)
    current = start_node
    total_cost = 0

    while unvisited:
        next_node = min(unvisited, key=lambda x: dist[current].get(x, float('inf')))
        step_cost = dist[current].get(next_node, float('inf'))
        if step_cost == float('inf'):
            return float('inf')
        total_cost += step_cost
        current = next_node
        unvisited.remove(next_node)

    return total_cost

def true_movement_cost(start_node, targets, dist):
    """Exact minimal path cost visiting all targets from start_node (no return)."""
    min_cost = float('inf')

    for perm in itertools.permutations(targets):
        cost = 0
        current = start_node
        for node in perm:
            step_cost = dist[current].get(node, float('inf'))
            if step_cost == float('inf'):
                cost = float('inf')
                break
            cost += step_cost
            current = node
        min_cost = min(min_cost, cost)

    return min_cost

def get_full_path_with_dwell_and_return(graph, start_node, targets, dwell_time):
    min_total_path = None
    min_total_cost = float("inf")
    min_total_steps = float("inf")

    for perm in itertools.permutations(targets):
        path = [start_node]
        current = start_node
        total_cost = 0
        total_steps = 0

        try:
            for node in perm:
                segment = nx.shortest_path(graph, source=current, target=node)
                if segment[0] == current:
                    segment = segment[1:]
                path.extend(segment)
                total_cost += len(segment)
                total_steps += len(segment)

                path.extend([node] * (dwell_time - 1))
                total_cost += (dwell_time - 1)

                current = node

            return_segment = nx.shortest_path(graph, source=current, target=start_node)
            if return_segment[0] == current:
                return_segment = return_segment[1:]
            path.extend(return_segment)
            total_cost += len(return_segment)
            total_steps += len(return_segment)

            if total_cost < min_total_cost:
                min_total_cost = total_cost
                min_total_steps = total_steps
                min_total_path = list(path)

        except nx.NetworkXNoPath:
            continue

    if min_total_path is None:
        return None, float("inf"), float("inf")

    return min_total_path, min_total_cost, min_total_steps

def get_simple_defendable_targets(graph, start_node, targets, num_timesteps, dwell_time):
    """
    Helper function to get all individually defendable targets from a given start node.
    
    Returns:
        Set of defendable target nodes.
    """
    defendable_targets = set()
    for target in targets:
        try:
            path_to_target = nx.shortest_path_length(graph, source=start_node, target=target)
            round_trip_time = 2 * path_to_target
            if round_trip_time + dwell_time <= num_timesteps:
                defendable_targets.add(target)
        except nx.NetworkXNoPath:
            continue
    return defendable_targets

def get_shortest_path_permutation(graph, start_node, targets):
    """Returns the shortest path (node list) visiting all targets from start_node (no return)."""
    min_path = None
    min_cost = float('inf')

    for perm in itertools.permutations(targets):
        path = [start_node]
        current = start_node
        total_path = []
        total_cost = 0
        valid = True

        for node in perm:
            try:
                segment = nx.shortest_path(graph, source=current, target=node)
            except nx.NetworkXNoPath:
                valid = False
                break

            # Avoid duplicating current node
            if segment[0] == current:
                segment = segment[1:]

            total_path += segment
            total_cost += len(segment)
            current = node

        if valid and total_cost < min_cost:
            min_cost = total_cost
            min_path = [start_node] + total_path

    return min_path, min_cost