from security_game.game import Game
from security_game.target import Target
from security_game.interdiction_protocol import InterdictionProtocol
from utils.strategy_utils import true_movement_cost, bfs_shortest_paths, get_full_path_with_dwell_and_return, get_simple_defendable_targets

import networkx as nx
import numpy as np
import itertools

class SecurityGame(Game):
    """
    A middle-layer class representing a classic security game.
    This class is specialized to have:
    - All defenders as moving defenders.
    - All attackers as stationary attackers.
    """

    def __init__(
        self,
        graph,
        targets,
        num_timesteps,
        num_attackers=1,
        num_defenders=1,
        defender_start_nodes = [],
        defender_end_nodes = [],
        interdiction_protocol=None,  # Allow passing a custom protocol
        defense_time_threshold=2,  # Only used if no IP provided
        generate_utility_matrix=False,
        schedule_form=False
    ):
        """
        Initialize a security game with fixed attacker/defender roles.
        
        num_attackers: Number of stationary attackers.
        num_defenders: Number of moving defenders.
        graph: The graph on which the game is played.
        targets: List of targets, each a Target object with 'node' and 'value'.
        num_timesteps: Number of timesteps in the game.
        defender_start_nodes: List of possible start nodes for moving defenders.
        defender_end_nodes: List of possible end nodes for moving defenders.
        interdiction_protocol: Custom InterdictionProtocol (optional).
        defense_time_threshold: Timesteps required to successfully defend a target.
        """

        # If no custom interdiction protocol is given, create a default one
        if interdiction_protocol is None:
            interdiction_protocol = InterdictionProtocol(graph, defense_time_threshold)

        # Call the base class constructor with security game parameterization
        super().__init__(
            num_stationary_attackers=num_attackers,
            num_moving_attackers=0, # No moving attackers
            num_stationary_defenders=0, # No stationary defenders
            num_moving_defenders=num_defenders,
            graph=graph,
            targets=targets,
            num_timesteps=num_timesteps,
            moving_defender_capture_radius=0, # No moving attackers no no need for capture radius
            stationary_defender_capture_radius=0, # No stationary defenders
            stationary_defender_start_nodes=[], # No stationary defenders
            moving_attacker_start_nodes=[], # No moving attackers
            moving_defender_start_nodes=defender_start_nodes,
            moving_attacker_end_nodes=[], # No moving attackers
            moving_defender_end_nodes=defender_end_nodes,
            interdiction_protocol=interdiction_protocol
        )

    def find_valid_schedules(self, start_node, defender_step_cost=1.0, simple=False):
        target_nodes = [t.node for t in self.targets]
        defendable_targets = get_simple_defendable_targets(
                self.graph, start_node, target_nodes,
                self.num_timesteps, self.interdiction_protocol.defense_time_threshold
            )
        if simple:
            return [({t}, 2 * nx.shortest_path_length(self.graph, source=start_node, target=t) * defender_step_cost) for t in defendable_targets]

        valid_schedules = []

        def backtrack(current_set, remaining_nodes):
            if current_set:
                path, total_cost, total_steps = get_full_path_with_dwell_and_return(
                    self.graph, start_node, current_set, self.interdiction_protocol.defense_time_threshold
                )
                if path is not None and total_cost <= self.num_timesteps:
                    real_cost = total_steps * defender_step_cost
                    valid_schedules.append((set(current_set), real_cost))

            for i, node in enumerate(remaining_nodes):
                backtrack(current_set | {node}, remaining_nodes[i+1:])

        backtrack(set(), list(defendable_targets))
        return valid_schedules

    def generate_defender_actions_with_costs(self, schedule_dict):
        """
        Generates all possible joint defender strategies from a dictionary mapping
        each defender to their list of (schedule, cost) tuples.

        Returns:
            defender_actions: list of lists of sets (each inner list is one full defender action)
            defender_costs: list of floats representing the total cost for each joint strategy
        """
        sorted_defenders = sorted(schedule_dict.keys())
        schedule_lists = [schedule_dict[d] for d in sorted_defenders]

        all_combinations = list(itertools.product(*schedule_lists))

        defender_actions = []
        defender_costs = []

        for combo in all_combinations:
            schedules = [item[0] for item in combo]
            total_cost = sum(item[1] for item in combo)
            defender_actions.append(schedules)
            defender_costs.append(total_cost)

        return defender_actions, defender_costs

    def generate_schedule_game_matrix(self, attacker_actions, defender_actions, defender_costs, target_utility_matrix, extra_coverage_weight):
        """
        Builds utility matrices for all combinations of defender and attacker actions.

        Parameters:
            attacker_actions (list[int]): List of target nodes (attacker strategies).
            defender_actions (list[list[set]]): Each outer list is one defender action;
                inner list is per-defender selected schedule (a set of targets).
            defender_costs (list[float]): Cost associated with each defender action.
            target_utility_matrix (np.ndarray): 4 x num_targets array:
                [0]: defender utility if uncovered
                [1]: defender utility if covered
                [2]: attacker utility if covered
                [3]: attacker utility if uncovered
            extra_coverage_weight (float): Multiplier for extra coverage instances.
        Returns:
            tuple:
                - attacker_util_matrix (np.ndarray): shape (len(defender_actions), len(attacker_actions))
                - defender_util_matrix (np.ndarray): same shape
        """
        num_defender_actions = len(defender_actions)
        num_attacker_actions = len(attacker_actions)

        defender_util_matrix = np.zeros((num_defender_actions, num_attacker_actions))
        attacker_util_matrix = np.zeros((num_defender_actions, num_attacker_actions))

        for i, d_action in enumerate(defender_actions):
            target_coverage_count = {}
            for schedule in d_action:
                for t in schedule:
                    target_coverage_count[t] = target_coverage_count.get(t, 0) + 1

            for j, atk_target in enumerate(attacker_actions):
                num_covers = target_coverage_count.get(j, 0)

                if num_covers == 0:
                    defender_util = target_utility_matrix[0][j]  # uncovered
                    attacker_util = target_utility_matrix[3][j]
                else:
                    weight = extra_coverage_weight ** (num_covers - 1)
                    defender_util = target_utility_matrix[1][j] * weight
                    attacker_util = target_utility_matrix[2][j] * weight

                defender_util_matrix[i, j] = defender_util - defender_costs[i]
                attacker_util_matrix[i, j] = attacker_util

        return attacker_util_matrix, defender_util_matrix

    def get_target_utility_matrix(self, attacker_penalty_factor=1, defender_penalty_factor=1):
        """
        Returns a 4 x num_targets utility matrix:
            Row 0: Defender utility (target uncovered)
            Row 1: Defender utility (target covered)
            Row 2: Attacker utility (target covered)
            Row 3: Attacker utility (target uncovered)

        Parameters:

            penalty_factor: Float for scaling the defender utility when target is covered (default=3)

        Returns:
            np.ndarray of shape (4, num_targets)
        """
        attacker_uncovered = np.array([t.attacker_value for t in self.targets])
        defender_uncovered = np.array([t.defender_value for t in self.targets])
        
        defender_covered = defender_uncovered / defender_penalty_factor
        attacker_covered = attacker_uncovered / attacker_penalty_factor

        return np.vstack([
            defender_uncovered,
            defender_covered,
            attacker_covered,
            attacker_uncovered
        ])

    def schedule_form(self, generate_utility_matrix, simple, attacker_penalty_factor, defender_penalty_factor, extra_coverage_weight, defender_step_cost):
        schedule_form_di = {}
        schedule_di = {}
        # for now one home base per defender for simplicity
        for i, home_base in enumerate([tup[0] for tup in self.moving_defender_start_nodes]):
            schedule_di[i] = self.find_valid_schedules(home_base, defender_step_cost, simple)
        schedule_form_di["schedules"] = schedule_di
        schedule_form_di["target_utilities"] = self.get_target_utility_matrix(attacker_penalty_factor, defender_penalty_factor)

        if generate_utility_matrix:
            attacker_actions = [t.node for t in self.targets]
            defender_actions, defender_costs = self.generate_defender_actions_with_costs(schedule_di)
            schedule_form_di["attacker_utility_matrix"], schedule_form_di["defender_utility_matrix"] = self.generate_schedule_game_matrix(attacker_actions, defender_actions, defender_costs, schedule_form_di["target_utilities"], extra_coverage_weight)
        else:
            schedule_form_di["attacker_utility_matrix"], schedule_form_di["defender_utility_matrix"] = None, None
        return schedule_form_di
        