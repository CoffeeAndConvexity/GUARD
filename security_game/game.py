import numpy as np
import networkx as nx
import itertools
from .player import Attacker, Defender
from .target import Target

class Game:
    def __init__(self, graph, num_timesteps,moving_attacker_start_nodes, moving_defender_start_nodes, stationary_defender_start_nodes, targets, interdiction_protocol, num_moving_attackers=1, num_stationary_attackers=0, num_moving_defenders=1, num_stationary_defenders=0, moving_defender_capture_radius=1, stationary_defender_capture_radius=1, moving_attacker_end_nodes=[], moving_defender_end_nodes=[], allow_wait=True):
        self.graph = graph
        self.interdiction_protocol = interdiction_protocol
        self.num_timesteps = num_timesteps
        self.moving_attacker_start_nodes = moving_attacker_start_nodes
        self.moving_defender_start_nodes = moving_defender_start_nodes
        self.moving_attacker_end_nodes = moving_attacker_end_nodes
        self.moving_defender_end_nodes = moving_defender_end_nodes
        self.stationary_defender_start_nodes = stationary_defender_start_nodes
        self.num_moving_attackers = num_moving_attackers
        self.num_stationary_attackers = num_stationary_attackers
        self.num_moving_defenders = num_moving_defenders
        self.num_stationary_defenders = num_stationary_defenders
        self.moving_defender_capture_radius = moving_defender_capture_radius
        self.stationary_defender_capture_radius = stationary_defender_capture_radius
        self.allow_wait = allow_wait
        self.targets = targets
         
        # Initialize attacker and defender units
        self.moving_attacker_units = [Attacker(f"Moving Attacker {i+1}", start_node) for i, start_node in enumerate(moving_attacker_start_nodes[:num_moving_attackers])]
        self.stationary_attacker_units = [Attacker(f"Stationary Attacker {i+1}", None) for i in range(num_stationary_attackers)]
        self.moving_defender_units = [Defender(f"Moving Defender {i+1}", start_node, capture_radius=moving_defender_capture_radius) for i, start_node in enumerate(moving_defender_start_nodes[:num_moving_defenders])]
        self.stationary_defender_units = [Defender(f"Stationary Defender {i+1}", start_node, capture_radius=stationary_defender_capture_radius) for i, start_node in enumerate(stationary_defender_start_nodes[:num_stationary_defenders])]

    def generate_moving_player_strategies(self, graph, num_moving_units, start_nodes, num_timesteps, allow_wait, end_nodes=[]):
        """
        Generates all possible movement strategies for moving players on the graph.
    
        Parameters:
        - graph: NetworkX graph representing the game environment.
        - start_nodes: List of potential start nodes for the players.
        - num_timesteps: Number of timesteps in the game.
        - allow_wait: Whether waiting at a node is allowed.
    
        Returns:
        - List of all possible paths (strategies) for the moving players.
        """
        if num_moving_units == 0:
            # No moving units, return an empty list
            return []
    
        all_paths = []
    
        # DFS logic to generate all possible strategies
        def dfs(current_positions, path):
            if len(path) == num_timesteps:
                all_paths.append(path)
                return
    
            for i, current_position in enumerate(current_positions):
                neighbors = list(graph.neighbors(current_position))
    
                # Add waiting option if allowed or no neighbors
                if allow_wait or not neighbors:
                    dfs(tuple(current_positions), path + [tuple(current_positions)])
    
                # Move to neighboring nodes
                for neighbor in neighbors:
                    new_positions = list(current_positions)
                    new_positions[i] = neighbor
                    dfs(tuple(new_positions), path + [tuple(new_positions)])
    
        # Initialize DFS for all combinations of start nodes
        for start_node_combination in itertools.product(start_nodes, repeat=num_moving_units):
            dfs(start_node_combination, [start_node_combination])

        if end_nodes:
            all_paths = [list(s) for s in set(tuple(i) for i in [p for p in all_paths if all(n in end_nodes for n in list(set(p[-1])))])]
        return all_paths

    def generate_strategy_matrix(self, player_type):
        """
        Generates a strategy matrix for the given player (Attacker or Defender) with support for
        both moving and stationary units.
    
        Parameters:
        - player_type: "attacker" or "defender"
    
        Returns:
        - strategy_matrix: 2D numpy array representing all possible strategies.
        """
        num_timesteps = self.num_timesteps
        allow_wait = self.allow_wait
    
        if player_type == "attacker":
            num_stationary_units = self.num_stationary_attackers
            num_moving_units = self.num_moving_attackers
            moving_start_nodes = self.moving_attacker_start_nodes
            moving_end_nodes = self.moving_attacker_end_nodes
        elif player_type == "defender":
            num_stationary_units = self.num_stationary_defenders
            num_moving_units = self.num_moving_defenders
            moving_start_nodes = self.moving_defender_start_nodes
            moving_end_nodes = self.moving_defender_end_nodes
        else:
            raise ValueError("Invalid player_type. Choose 'attacker' or 'defender'.")
    
        # Generate strategies for moving units
        moving_paths = self.generate_moving_player_strategies(
            self.graph, num_moving_units, moving_start_nodes, num_timesteps, allow_wait, moving_end_nodes
        )
    
        # Generate strategies for stationary units
        if player_type == "attacker":
            stationary_targets = [target.node for target in self.targets] + [None]
            stationary_strategies = list(
                itertools.product(stationary_targets, repeat=num_stationary_units)
            )
        else:
            stationary_strategies = list(
                itertools.product(self.stationary_defender_start_nodes, repeat=num_stationary_units)
            )
    
        # Handle the case where there are no moving units
        if num_moving_units == 0:
            # Only stationary strategies
            combined_strategies = [
                [stationary_strategy] * num_timesteps
                for stationary_strategy in stationary_strategies
            ]
        elif num_stationary_units == 0:
            # Only moving strategies
            combined_strategies = moving_paths
        else:
            # Combine moving and stationary strategies
            combined_strategies = []
            for moving_path in moving_paths:  # Each moving_path is a list of positions for moving units at each timestep
                for stationary_strategy in stationary_strategies:  # Each stationary_strategy is a tuple of stationary unit positions
                    # Combine moving and stationary positions for each timestep
                    combined_path = [
                        tuple(moving_positions) + stationary_strategy
                        for moving_positions in moving_path
                    ]
                    combined_strategies.append(combined_path)
    
        # Convert to numpy array for consistency
        return np.array(combined_strategies)
    
    def play_game_with_strategies(self, defender_strategy, attacker_strategy):
        """
        Simulate the game with given defender and attacker strategies.
        """
        reached_targets = set()  # Tracks which targets have been reached
        moving_attacker_score = 0
        stationary_attacker_score = 0
    
        # Step 1: Handle moving attackers
        interdicted = [False] * self.num_moving_attackers
    
        for t in range(self.num_timesteps):
            # Get current positions for moving attackers and defenders
            current_attacker_positions = attacker_strategy[t][:self.num_moving_attackers]
            current_defender_positions = defender_strategy[t]
    
            # Use ip.moving_interdiction to determine which attackers are interdicted
            newly_interdicted = self.interdiction_protocol.moving_interdiction(
                current_attacker_positions, 
                current_defender_positions, 
                [unit.capture_radius for unit in self.moving_defender_units + self.stationary_defender_units]
            )
            
            # Update interdicted status
            for idx, is_interdicted in enumerate(newly_interdicted):
                interdicted[idx] = interdicted[idx] or is_interdicted
    
            # Skip processing for interdicted attackers
            for idx, position in enumerate(current_attacker_positions):
                if interdicted[idx]:
                    continue
    
                # Check if the attacker reached an unreached target
                for target in self.targets:
                    if position == target.node and target.node not in reached_targets:
                        moving_attacker_score -= target.value
                        reached_targets.add(target.node)
    
            # Exit early if all moving attackers are interdicted
            if all(interdicted):
                break
    
        # Step 2: Handle stationary attackers
        if self.num_stationary_attackers > 0:
            stationary_attacker_positions = [
                attacker_strategy[0][self.num_moving_attackers + idx]
                for idx in range(self.num_stationary_attackers)
            ]
    
            # Use ip.stationary_interdiction to determine stationary attacker interdictions
            stationary_interdictions = self.interdiction_protocol.stationary_interdiction(
                stationary_attacker_positions,
                defender_strategy,
                self.interdiction_protocol.defense_time_threshold
            )
    
            # Update scores for stationary attackers
            for idx, position in enumerate(stationary_attacker_positions):
                if not stationary_interdictions[idx] and position not in reached_targets:
                    for target in self.targets:
                        if position == target.node:
                            stationary_attacker_score -= target.value
                            reached_targets.add(target.node)
    
        # Step 3: Return the total score
        return moving_attacker_score + stationary_attacker_score

    def generate_utility_matrix(self):
        """
        Generate the utility matrix by simulating all combinations of strategies
        for both attackers and defenders, accounting for stationary and moving units.
        """
        # Generate strategies for defenders and attackers
        defender_matrix = self.generate_strategy_matrix("defender")
        attacker_matrix = self.generate_strategy_matrix("attacker")
        
        # Initialize the utility matrix
        utility_matrix = np.zeros((len(defender_matrix), len(attacker_matrix)))
        
        # Loop through all combinations of strategies
        for i, defender_strategy in enumerate(defender_matrix):
            for j, attacker_strategy in enumerate(attacker_matrix):
                # Simulate the game with the current strategies and store the outcome
                utility_matrix[i, j] = self.play_game_with_strategies(defender_strategy, attacker_strategy)
        
        return utility_matrix