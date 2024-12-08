class InterdictionProtocol:
    def __init__(self, graph, defense_time_threshold):
        """
        Initialize the interdiction protocol.

        graph: The graph on which the game is played.
        defense_time_threshold: Number of timesteps a defender must spend at a target to successfully defend it.
        """
        self.graph = graph
        self.defense_time_threshold = defense_time_threshold

    def moving_interdiction(self, attacker_positions, defender_positions, capture_radii):
        """
        Determine which moving attackers are interdicted by moving defenders.

        attacker_positions: List of current positions of all moving attackers.
        defender_positions: List of current positions of all moving defenders.
        capture_radii: List of capture radii for each moving defender.
        return: A list of booleans indicating whether each moving attacker is interdicted.
        """
        interdicted = [False] * len(attacker_positions)

        for attacker_idx, attacker_position in enumerate(attacker_positions):
            for defender_idx, defender_position in enumerate(defender_positions):
                # Calculate shortest path distance
                distance = nx.shortest_path_length(
                    self.graph, source=attacker_position, target=defender_position
                )
                # Check if within capture radius
                if distance <= capture_radii[defender_idx]:
                    interdicted[attacker_idx] = True
                    break  # Stop checking other defenders for this attacker

        return interdicted

    def stationary_interdiction(self, stationary_attacker_positions, defender_strategy, defense_time):
        """
        Determine which stationary attackers are interdicted by defenders.

        stationary_attacker_positions: List of target nodes chosen by stationary attackers.
        defender_strategy: List of defender positions over all timesteps (list of lists).
        defense_time: Number of timesteps a defender must spend at a target to interdict it.
        return: A list of booleans indicating whether each stationary attacker is interdicted.
        """
        # Count the number of timesteps defenders spend at each target
        defended_targets = {}
        for defender_positions in defender_strategy:
            for defender_position in defender_positions:
                if defender_position not in defended_targets:
                    defended_targets[defender_position] = 0
                defended_targets[defender_position] += 1

        # Determine interdiction for each stationary attacker
        interdicted = []
        for attacker_target in stationary_attacker_positions:
            if defended_targets.get(attacker_target, 0) >= defense_time:
                interdicted.append(True)
            else:
                interdicted.append(False)

        return interdicted