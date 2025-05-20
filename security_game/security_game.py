from security_game.game import Game
from security_game.target import Target
from security_game.interdiction_protocol import InterdictionProtocol
from utils.strategy_utils import true_movement_cost, bfs_shortest_paths, get_full_path_with_dwell_and_return, get_simple_defendable_targets, deduplicate_general_schedules
from utils.random_utils import generate_random_target_utility_matrix_like
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
        schedule_form=False,
        force_return=False
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
            interdiction_protocol=interdiction_protocol,
            force_return=force_return
        )

    def find_valid_schedules(self, start_node, defender_step_cost=0, simple=False):
        target_nodes = [t.node for t in self.targets]
        defendable_targets = []

        for t in target_nodes:
            try:
                to_t = nx.shortest_path_length(self.graph, source=start_node, target=t)
                from_t = nx.shortest_path_length(self.graph, source=t, target=start_node)
                total_steps = to_t + from_t + self.interdiction_protocol.defense_time_threshold
                if total_steps <= self.num_timesteps:
                    defendable_targets.append(t)
            except nx.NetworkXNoPath:
                continue  # Skip if no round-trip path exists

        if simple:
            return [
                ({t}, (nx.shortest_path_length(self.graph, source=start_node, target=t) +
                    nx.shortest_path_length(self.graph, source=t, target=start_node)) * defender_step_cost)
                for t in defendable_targets
            ]

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
        Generates all possible joint defender actions from a dictionary mapping
        each defender to their list of (schedule, cost) tuples.

        Defenders with no schedules are assigned an empty set with 0 cost as a placeholder.

        Returns:
            defender_actions: list of lists of sets (each inner list is one full defender action)
            defender_costs: list of floats representing the total cost for each joint action
        """
        sorted_defenders = sorted(schedule_dict.keys())
        # Replace empty schedule lists with a dummy no-op schedule
        schedule_lists = [
            schedule_dict[d] if schedule_dict[d] else [({}, 0)]
            for d in sorted_defenders
        ]
        all_combinations = list(itertools.product(*schedule_lists))

        defender_actions = []
        defender_costs = []

        for combo in all_combinations:
            schedules = [item[0] for item in combo]
            total_cost = sum(item[1] for item in combo)
            defender_actions.append(schedules)
            defender_costs.append(total_cost)

        return defender_actions, defender_costs

    def generate_schedule_game_matrix(self, attacker_actions, defender_actions, defender_costs, target_utility_matrix, general_sum):
        """
        Builds utility matrices for all combinations of defender and attacker actions,
        then normalizes both matrices together based on the largest absolute value.

        Parameters:
            attacker_actions (list[int]): List of target nodes (attacker action).
            defender_actions (list[list[set]]): Each outer list is one defender action;
                inner list is per-defender selected schedule (a set of targets).
            defender_costs (list[float]): Cost associated with each defender action.
            target_utility_matrix (np.ndarray): 4 x num_targets array:
                [0]: defender utility if uncovered
                [1]: defender utility if covered
                [2]: attacker utility if covered
                [3]: attacker utility if uncovered

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
                num_covers = target_coverage_count.get(atk_target, 0)

                if num_covers == 0:
                    defender_util = target_utility_matrix[0][j]  # uncovered
                    attacker_util = target_utility_matrix[3][j]  # uncovered
                else:
                    defender_util = target_utility_matrix[1][j]  # covered
                    attacker_util = target_utility_matrix[2][j]  # covered

                defender_util_matrix[i, j] = defender_util - defender_costs[i]
                attacker_util_matrix[i, j] = attacker_util


        max_abs_value = max(
            np.abs(defender_util_matrix).max(),
            np.abs(attacker_util_matrix).max()
        )

        if max_abs_value > 0:
            defender_util_matrix = defender_util_matrix / max_abs_value
            attacker_util_matrix = attacker_util_matrix / max_abs_value

        return attacker_util_matrix, defender_util_matrix

    def get_target_utility_matrix(self, attacker_penalty_factor=1, defender_penalty_factor=1):
        """
        Returns a 4 x num_targets utility matrix:
            Row 0: Defender utility (target uncovered)
            Row 1: Defender utility (target covered)
            Row 2: Attacker utility (target covered)
            Row 3: Attacker utility (target uncovered)

        Parameters:
            attacker_penalty_factor: Float for scaling the attacker utility when target is covered (default=1)
            defender_penalty_factor: Float for scaling the defender utility when target is covered (default=1)
            normalize: Boolean. If True, normalize entire target utility matrix.

        Returns:
            np.ndarray of shape (4, num_targets)
        """
        attacker_uncovered = np.array([t.attacker_value for t in self.targets])
        defender_uncovered = np.array([t.defender_value for t in self.targets])

        defender_covered = defender_uncovered / defender_penalty_factor
        attacker_covered = attacker_uncovered / attacker_penalty_factor

        matrix = np.vstack([
            defender_uncovered,
            defender_covered,
            attacker_covered,
            attacker_uncovered
        ])
    

        max_abs_val = np.abs(matrix).max()
        if max_abs_val != 0:
            matrix = matrix / max_abs_val  # normalize by largest absolute value, keep signs

        return matrix


    def schedule_form(self, generate_utility_matrix, generate_actions, general_sum, simple, attacker_penalty_factor, defender_penalty_factor, randomize_target_utility_matrix, defender_step_cost):
        schedule_form_di = {}
        schedule_di = {}

        if self.num_moving_defenders != len(self.moving_defender_start_nodes):
            raise ValueError(f"home_base_assignments must have exactly {self.num_moving_defenders} elements.")

        for i, home_bases in enumerate(self.moving_defender_start_nodes):
            for h in home_bases:
                if i not in schedule_di:
                    schedule_di[i] = self.find_valid_schedules(h, defender_step_cost, simple)
                else:
                    schedule_di[i].extend(self.find_valid_schedules(h, defender_step_cost, simple))
        schedule_form_di["schedules"] = deduplicate_general_schedules(schedule_di)
        schedule_form_di["target_utilities"] = self.get_target_utility_matrix(attacker_penalty_factor, defender_penalty_factor)
        if randomize_target_utility_matrix:
            schedule_form_di["target_utilities"] = generate_random_target_utility_matrix_like(schedule_form_di["target_utilities"],general_sum=general_sum, respect_sign_roles=True)
        schedule_form_di["targets"] = [t.node for t in self.targets]

        if generate_utility_matrix:
            attacker_actions = [t.node for t in self.targets]
            defender_actions, defender_costs = self.generate_defender_actions_with_costs(schedule_di)
            schedule_form_di["defender_actions"]=defender_actions
            schedule_form_di["attacker_utility_matrix"], schedule_form_di["defender_utility_matrix"] = self.generate_schedule_game_matrix(attacker_actions, defender_actions, defender_costs, schedule_form_di["target_utilities"], general_sum)
        else:
            schedule_form_di["attacker_utility_matrix"], schedule_form_di["defender_utility_matrix"] = None, None
            if generate_actions:
                defender_actions, defender_costs = self.generate_defender_actions_with_costs(schedule_di)
                schedule_form_di["defender_actions"] = defender_actions
            else:
                schedule_form_di["defender_actions"] = None
        return schedule_form_di
        