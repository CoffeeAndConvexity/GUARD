import numpy as np
import copy
import networkx as nx
import itertools
from .player import Attacker, Defender
from .target import Target

class Game:
    def __init__(self, graph, num_timesteps,moving_attacker_start_nodes, moving_defender_start_nodes, stationary_defender_start_nodes, targets, interdiction_protocol, num_moving_attackers=1, num_stationary_attackers=0, num_moving_defenders=1, num_stationary_defenders=0, moving_defender_capture_radius=1, stationary_defender_capture_radius=1, moving_attacker_end_nodes=[], moving_defender_end_nodes=[], allow_wait=True, force_return=False):
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
        self.force_return = force_return
         
        # Initialize attacker and defender units
        self.moving_attacker_units = [Attacker(f"Moving Attacker {i+1}", start_node) for i, start_node in enumerate(moving_attacker_start_nodes[:num_moving_attackers])]
        self.stationary_attacker_units = [Attacker(f"Stationary Attacker {i+1}", None) for i in range(num_stationary_attackers)]
        self.moving_defender_units = [Defender(f"Moving Defender {i+1}", start_node, capture_radius=moving_defender_capture_radius) for i, start_node in enumerate(moving_defender_start_nodes[:num_moving_defenders])]
        self.stationary_defender_units = [Defender(f"Stationary Defender {i+1}", start_node, capture_radius=stationary_defender_capture_radius) for i, start_node in enumerate(stationary_defender_start_nodes[:num_stationary_defenders])]

    def generate_moving_player_actions(
        self,
        graph,
        num_moving_units,
        start_nodes,
        num_timesteps,
        allow_wait,
        end_nodes=[],
        force_return_to_start=False
    ):
        """
        Generates all possible movement actions for moving players on the graph with specific start and end node constraints.

        Parameters:
        - graph: NetworkX graph representing the game environment.
        - num_moving_units: Number of moving players.
        - start_nodes: List of tuples where each element contains allowed start nodes for a unit.
        - num_timesteps: Number of timesteps in the game.
        - allow_wait: Whether waiting at a node is allowed.
        - end_nodes: List of tuples where each element contains allowed end nodes for a unit.
        - force_return_to_start: If True, unit must end at its starting node (overrides end_nodes per path).

        Returns:
        - List of all valid movement actions, formatted as timestep-major.
        """
            
        if num_moving_units == 0:
            return []

        if (start_nodes and len(start_nodes) != num_moving_units) or (end_nodes and len(end_nodes) != num_moving_units):
            raise ValueError(f"start_nodes and end_nodes must be either empty or have exactly {num_moving_units} elements.")
        
        if not start_nodes:
            start_nodes = [list(graph.nodes)] * num_moving_units
        if not end_nodes:
            end_nodes = [list(graph.nodes)] * num_moving_units

        def generate_paths(valid_start_nodes, valid_end_nodes, force_return=False):
            """Generates all valid paths for a single unit respecting start and end constraints."""
            all_paths = []

            def dfs(current_path, original_start):
                if len(current_path) == num_timesteps:
                    if (not valid_end_nodes or current_path[-1] in valid_end_nodes) and \
                    (not force_return or current_path[-1] == original_start):
                        all_paths.append(list(current_path))
                    return

                current_node = current_path[-1]
                neighbors = list(graph.neighbors(current_node))

                if allow_wait or not neighbors:
                    dfs(current_path + [current_node], original_start)

                for neighbor in neighbors:
                    dfs(current_path + [neighbor], original_start)

            for start_node in valid_start_nodes:
                dfs([start_node], start_node)

            return all_paths

        unit_paths = []
        for i in range(num_moving_units):
            unit_paths.append(
                generate_paths(
                    start_nodes[i],
                    end_nodes[i] if end_nodes else [],
                    force_return=force_return_to_start
                )
            )

        valid_combinations = list(itertools.product(*unit_paths))

        formatted_paths = []
        for path_tuple in valid_combinations:
            timestep_paths = list(zip(*path_tuple))
            formatted_paths.append(timestep_paths)

        return formatted_paths

    def generate_actions(self, player_type):
        """
        Generates a action matrix for the given player (Attacker or Defender) with support for
        both moving and stationary units.
    
        Parameters:
        - player_type: "attacker" or "defender"
    
        Returns:
        - action_matrix: 2D numpy array representing all possible actions.
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
    
        # Generate actions for moving units
        moving_paths = self.generate_moving_player_actions(
            self.graph, num_moving_units, moving_start_nodes, num_timesteps, allow_wait, moving_end_nodes, self.force_return
        )
    
        # Generate actions for stationary units
        if player_type == "attacker":
            stationary_targets = [target.node for target in self.targets] + [None]
            stationary_actions = list(itertools.combinations(stationary_targets, num_stationary_units)
            )
        else:
            stationary_actions = list(
                itertools.product(self.stationary_defender_start_nodes, repeat=num_stationary_units)
            )
    
        # Handle the case where there are no moving units
        if num_moving_units == 0:
            # Only stationary actions
            combined_actions = [
                [stationary_action] * num_timesteps
                for stationary_action in stationary_actions
            ]
        elif num_stationary_units == 0:
            # Only moving actions
            combined_actions = moving_paths
        else:
            # Combine moving and stationary actions
            combined_actions = []
            for moving_path in moving_paths:  # Each moving_path is a list of positions for moving units at each timestep
                for stationary_action in stationary_actions:  # Each stationary_action is a tuple of stationary unit positions
                    # Combine moving and stationary positions for each timestep
                    combined_path = [
                        tuple(moving_positions) + stationary_action
                        for moving_positions in moving_path
                    ]
                    combined_actions.append(combined_path)
        return np.array(combined_actions)

    
    def evaluate_actions(self, defender_action, attacker_action):
        """
        Simulate the game with given defender and attacker actions.
        """
        reached_targets = set()  # Tracks which targets have been reached
        moving_attacker_score = 0
        stationary_attacker_score = 0
    
        # Step 1: Handle moving attackers
        interdicted = [False] * self.num_moving_attackers
    
        for t in range(self.num_timesteps):
            # Get current positions for moving attackers and defenders
            current_attacker_positions = attacker_action[t][:self.num_moving_attackers]
            current_defender_positions = defender_action[t]
    
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
                        moving_attacker_score -= target.attacker_value #whole game is from defender POV, subtract positive
                        reached_targets.add(target.node)
    
            # Exit early if all moving attackers are interdicted
            if all(interdicted):
                break
    
        # Step 2: Handle stationary attackers
        if self.num_stationary_attackers > 0:
            stationary_attacker_positions = [
                attacker_action[0][self.num_moving_attackers + idx]
                for idx in range(self.num_stationary_attackers)
            ]
    
            # Use ip.stationary_interdiction to determine stationary attacker interdictions
            stationary_interdictions = self.interdiction_protocol.stationary_interdiction(
                stationary_attacker_positions,
                defender_action,
                self.interdiction_protocol.defense_time_threshold
            )
    
            # Update scores for stationary attackers
            for idx, position in enumerate(stationary_attacker_positions):
                if not stationary_interdictions[idx] and position not in reached_targets:
                    for target in self.targets:
                        if position == target.node:
                            stationary_attacker_score -= target.attacker_value
                            reached_targets.add(target.node)
    
        # Step 3: Return the total score
        return moving_attacker_score + stationary_attacker_score


    def generate_utility_matrix(self, general_sum, defender_step_cost):
        """
        Generate the utility matrix by simulating all combinations of actions
        for both attackers and defenders, accounting for stationary and moving units.

        If general_sum is True, two utility matrices are returned (attacker and defender utilities),
        and they are normalized together based on the maximum absolute value.
        """
        # Generate actions for defenders and attackers
        defender_matrix = self.generate_actions("defender")
        attacker_matrix = self.generate_actions("attacker")

        if not general_sum:
            # Initialize the utility matrix
            utility_matrix = np.zeros((len(defender_matrix), len(attacker_matrix)))
            
            # Loop through all combinations of actions
            for i, defender_action in enumerate(defender_matrix):
                for j, attacker_action in enumerate(attacker_matrix):
                    # Simulate the game with the current actions and store the outcome
                    utility_matrix[i, j] = self.evaluate_actions(defender_action, attacker_action)

            max_abs_value = np.abs(utility_matrix).max()
            if max_abs_value > 0:
                utility_matrix = utility_matrix / max_abs_value
            
            return utility_matrix, None, None

        else:
            num_defender_actions = len(defender_matrix)
            num_attacker_actions = len(attacker_matrix)

            # Initialize the utility matrices
            defender_utility_matrix = np.zeros((num_defender_actions, num_attacker_actions))
            attacker_utility_matrix = np.zeros((num_defender_actions, num_attacker_actions))

            # Loop through all combinations of actions
            for i, defender_action in enumerate(defender_matrix):
                for j, attacker_action in enumerate(attacker_matrix):
                    # Play the general-sum game with both actions
                    attacker_score, defender_score = self.evaluate_actions_general(defender_action, attacker_action, defender_step_cost)

                    # Store the respective scores
                    defender_utility_matrix[i, j] = defender_score
                    attacker_utility_matrix[i, j] = attacker_score

            return None, attacker_utility_matrix, defender_utility_matrix

    def evaluate_actions_general(self, defender_action, attacker_action, defender_step_cost):
        """
        Simulate the game with given defender and attacker actions, incorporating defender movement cost.

        Parameters:
            defender_action (list[tuple[int]]): Defender positions for each timestep.
            attacker_action (list[tuple[int]]): Attacker positions for each timestep.
            defender_step_cost (float): Cost incurred per movement step taken by any defender.

        Returns:
            (attacker_score, defender_score): Tuple of utility values for attacker and defender.
        """
        reached_targets = set()
        defender_score = 0
        moving_attacker_score = 0
        stationary_attacker_score = 0

        # Step 1: Handle moving attackers
        interdicted = [False] * self.num_moving_attackers

        for t in range(self.num_timesteps):
            current_attacker_positions = attacker_action[t][:self.num_moving_attackers]
            current_defender_positions = defender_action[t]

            newly_interdicted = self.interdiction_protocol.moving_interdiction(
                current_attacker_positions,
                current_defender_positions,
                [unit.capture_radius for unit in self.moving_defender_units + self.stationary_defender_units]
            )

            for idx, is_interdicted in enumerate(newly_interdicted):
                interdicted[idx] = interdicted[idx] or is_interdicted

            for idx, position in enumerate(current_attacker_positions):
                if interdicted[idx]:
                    continue

                for target in self.targets:
                    if position == target.node and target.node not in reached_targets:
                        moving_attacker_score += target.attacker_value
                        defender_score += target.defender_value
                        reached_targets.add(target.node)

            if all(interdicted):
                break

        # Step 2: Handle stationary attackers
        if self.num_stationary_attackers > 0:
            stationary_attacker_positions = [
                attacker_action[0][self.num_moving_attackers + idx]
                for idx in range(self.num_stationary_attackers)
            ]

            stationary_interdictions = self.interdiction_protocol.stationary_interdiction(
                stationary_attacker_positions,
                defender_action,
                self.interdiction_protocol.defense_time_threshold
            )

            for idx, position in enumerate(stationary_attacker_positions):
                if not stationary_interdictions[idx] and position not in reached_targets:
                    for target in self.targets:
                        if position == target.node:
                            stationary_attacker_score += target.attacker_value
                            defender_score += target.defender_value
                            reached_targets.add(target.node)

        # Step 3: Penalize defender for steps taken
        num_steps = 0
        for d in range(len(defender_action[0])):  # iterate over each defender
            path = [defender_action[t][d] for t in range(self.num_timesteps)]
            steps = sum(1 for i in range(1, len(path)) if path[i] != path[i - 1])
            num_steps += steps

        defender_score -= defender_step_cost * num_steps


        return moving_attacker_score + stationary_attacker_score, defender_score