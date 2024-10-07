import numpy as np
import networkx as nx
from .player import Attacker, Defender
from .target import Target

class SecurityGame:
    def __init__(self, graph, num_timesteps, attacker_start_nodes, moving_defender_start_nodes, stationary_defender_start_nodes, targets, num_attackers=1, num_moving_defenders=1, num_stationary_defenders=0, allow_wait=True):
        self.graph = graph
        self.num_timesteps = num_timesteps
        self.attacker_start_nodes = attacker_start_nodes
        self.moving_defender_start_nodes = moving_defender_start_nodes
        self.stationary_defender_start_nodes = stationary_defender_start_nodes
        self.num_attackers = num_attackers
        self.num_moving_defenders = num_moving_defenders
        self.num_stationary_defenders = num_stationary_defenders
        self.allow_wait = allow_wait
        self.targets = targets
         
        # Initialize attacker and defender units
        self.attacker_units = [Attacker(f"Attacker {i+1}", start_node) for i, start_node in enumerate(attacker_start_nodes[:num_attackers])]
        self.moving_defender_units = [Defender(f"Moving Defender {i+1}", start_node, capture_radius=1) for i, start_node in enumerate(moving_defender_start_nodes[:num_moving_defenders])]
        self.stationary_defender_units = [Defender(f"Stationary Defender {i+1}", start_node, capture_radius=1) for i, start_node in enumerate(stationary_defender_start_nodes[:num_stationary_defenders])]

    def generate_strategy_matrix(self, player_type):
        """
        Generates a strategy matrix for the given player (Attacker or Defender) with support for multiple units
        and optionally waiting at nodes during the strategy.
        
        Parameters:
        - player_type: "attacker" or "defender"
        
        Returns:
        - strategy_matrix: 2D numpy array where each row represents a possible path or strategy.
        """
        num_timesteps = self.num_timesteps
        all_paths = []
        allow_wait = self.allow_wait  # Check if waiting is allowed
    
        if player_type == "attacker":
            moving_units = self.attacker_units
            start_nodes = self.attacker_start_nodes
        elif player_type == "defender":
            moving_units = self.moving_defender_units
            stationary_units = self.stationary_defender_units
            start_nodes = self.moving_defender_start_nodes
            stationary_start_nodes = self.stationary_defender_start_nodes
        else:
            raise ValueError("Invalid player_type. Choose 'attacker' or 'defender'.")
    
        # Handle stationary units for defenders (stationary units don't move)
        if player_type == "defender":
            stationary_positions = list(itertools.product(stationary_start_nodes, repeat=len(stationary_units)))
    
        # DFS logic to generate all possible strategies for moving units
        def dfs(current_positions, path):
            if len(path) == num_timesteps:
                all_paths.append(path)
                return
    
            # For each unit, either move to a neighboring node or wait
            for i, current_position in enumerate(current_positions):
                neighbors = list(self.graph.neighbors(current_position))
    
                # If allowed, add the current node as a waiting option
                if allow_wait or not neighbors:
                    dfs(tuple(current_positions), path + [tuple(current_positions)])
    
                for neighbor in neighbors:
                    new_positions = list(current_positions)
                    new_positions[i] = neighbor
                    dfs(tuple(new_positions), path + [tuple(new_positions)])
    
        # Initialize DFS for each moving unit from its allowed starting nodes
        for start_node_combination in itertools.product(start_nodes, repeat=len(moving_units)):
            moving_paths = []
            dfs(start_node_combination, [start_node_combination])
    
            # Combine moving defender strategies with stationary defender positions
            if player_type == "defender":
                for stationary_combination in stationary_positions:
                    for path in all_paths:
                        combined_path = [(move_pos, stationary_pos) for move_pos, stationary_pos in zip(path, [stationary_combination]*num_timesteps)]
                        moving_paths.append(combined_path)
    
            else:
                moving_paths = all_paths  # Attackers don't have stationary units
    
        # Convert list of paths to a numpy array for easy manipulation
        strategy_matrix = np.array(moving_paths)
    
        return strategy_matrix
    
    def play_game_with_strategies(self, defender_strategy, attacker_strategy):
        """
        Play the game using specific strategies for both the defenders and attackers.
        The strategies are lists of nodes for each unit (both attackers and defenders).
        """
        # Initialize scores for each attacker
        reached_targets = set()  # This will store target nodes that have been reached
        attacker_scores = [0] * self.num_attackers
        interdicted = [False] * self.num_attackers  # Track which attackers have been interdicted
        
        # Loop over each timestep
        for t in range(self.num_timesteps):
            current_defender_positions = defender_strategy[t]  # Tuple of defender positions (moving, stationary)
            current_attacker_positions = attacker_strategy[t]  # Tuple of attacker positions
            
            # Ensure the node labels are correctly formatted as integers
            current_defender_positions = [
                int(d[0]) if isinstance(d, (list, np.ndarray)) else int(d) 
                for d in current_defender_positions
            ]
            current_attacker_positions = [
                int(a[0]) if isinstance(a, (list, np.ndarray)) else int(a) 
                for a in current_attacker_positions
            ]
            
            # Check for interdictions
            for attacker_idx, attacker_position in enumerate(current_attacker_positions):
                if interdicted[attacker_idx]:
                    continue  # Skip interdicted attackers
                
                # Compare each attacker position with all moving defender positions
                for defender_idx, defender_position in enumerate(current_defender_positions[:self.num_moving_defenders]):
                    if nx.shortest_path_length(self.graph, source=attacker_position, target=defender_position) <= self.moving_defender_units[defender_idx].capture_radius:
                        interdicted[attacker_idx] = True
                        break  # Interdiction occurred for this attacker
                
                # Compare attacker position with all stationary defender positions
                for defender_idx, defender_position in enumerate(current_defender_positions[self.num_moving_defenders:], start=0):
                    if nx.shortest_path_length(self.graph, source=attacker_position, target=defender_position) <= self.stationary_defender_units[defender_idx].capture_radius:
                        interdicted[attacker_idx] = True
                        break  # Interdiction occurred for this attacker
            
                # Update score for this attacker (if not interdicted yet)
                if not interdicted[attacker_idx]:
                    for target in self.targets:
                        if attacker_position == target.node and target.node not in reached_targets:
                            attacker_scores[attacker_idx] += -target.value
                            reached_targets.add(target.node)
        
            # If all attackers are interdicted, end the game early
            if all(interdicted):
                break
        
        # Return the total score (sum of all attackers' scores)
        return sum(attacker_scores)

    def generate_utility_matrix(self):
        """
        Generate the utility matrix by simulating all combinations of strategies
        for both attackers and defenders with multiple units.
        """
    
        defender_matrix = self.generate_strategy_matrix("defender")
        attacker_matrix = self.generate_strategy_matrix("attacker")
    
        utility_matrix = np.zeros((len(defender_matrix), len(attacker_matrix)))
    
        for i, defender_strategy in enumerate(defender_matrix):
            for j, attacker_strategy in enumerate(attacker_matrix):
                utility_matrix[i, j] = self.play_game_with_strategies(defender_strategy, attacker_strategy)
    
        return utility_matrix