from security_game.game import Game
from security_game.target import Target
from security_game.interdiction_protocol import InterdictionProtocol

class SecurityGame(Game):
    """
    A middle-layer class representing a classic security game.
    This class is specialized to have:
    - All defenders as moving defenders.
    - All attackers as stationary attackers.
    """

    def __init__(
        self,
        num_attackers,
        num_defenders,
        graph,
        targets,
        num_timesteps,
        defender_start_nodes,
        defender_end_nodes = [],
        interdiction_protocol=None,  # Allow passing a custom protocol
        defense_time_threshold=2  # Only used if no IP provided
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