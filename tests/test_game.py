import pytest
import networkx as nx
from security_game.game import Game
from security_game.target import Target
from security_game.player import Attacker, Defender
from security_game.interdiction_protocol import InterdictionProtocol

def test_game_initialization():
    # Create a simple graph and initialize a SecurityGame instance
    G = nx.grid_2d_graph(3, 3)
    G = nx.convert_node_labels_to_integers(G)
    targets = [Target(4,5), Target(7,3)]
    ip = InterdictionProtocol(graph=G, defense_time_threshold=2)
    game = Game(G, interdiction_protocol=ip, num_timesteps=4,
                    moving_attacker_start_nodes=[0, 3],
                    moving_defender_start_nodes=[8, 5],
                    stationary_defender_start_nodes=[2, 6],
                    num_moving_attackers=0 , 
                    num_stationary_attackers=1,
                    num_moving_defenders=1, 
                    num_stationary_defenders=0,
                    moving_defender_capture_radius=1,
                    stationary_defender_capture_radius=1,
                    allow_wait=True,
                    targets = targets
                    )

    # Assert that game attributes are correctly set
    assert game.num_timesteps == 4
    assert len(game.targets) == 2
    assert len(game.moving_attacker_start_nodes) == 2
    assert len(game.moving_defender_units) == 1
    assert len(game.stationary_defender_units) == 0

def test_attacker_behavior():
    # Test Attacker class methods and attributes
    G = nx.grid_2d_graph(3, 3)
    G = nx.convert_node_labels_to_integers(G)
    targets = [Target(4,5), Target(7,3)]
    ip = InterdictionProtocol(graph=G, defense_time_threshold=2)
    game = Game(G, interdiction_protocol=ip, num_timesteps=4,
                    moving_attacker_start_nodes=[0],
                    moving_defender_start_nodes=[8, 5],
                    stationary_defender_start_nodes=[2, 6],
                    num_moving_attackers=1, 
                    num_stationary_attackers=1,
                    num_moving_defenders=1, 
                    num_stationary_defenders=0,
                    moving_defender_capture_radius=1,
                    stationary_defender_capture_radius=1,
                    allow_wait=True,
                    targets = targets
                    )

    attacker = game.moving_attacker_units[0]  # Initialize the attacker from the game

    # Check initial position
    assert attacker.current_node == 0, "Attacker should start at node 0"

    # Simulate attacker movement
    attacker.move(1)
    assert attacker.current_node == 1, "Attacker should have moved to node 1"
    assert attacker.path == [0, 1], "Attacker's path should be updated"

    # Test target selection (chooses the highest value target by default)
    chosen_target = attacker.choose_target(game.targets)
    assert chosen_target.node == 4, "Attacker should choose the highest value target at node 4"

    # Move attacker towards the target and verify they reach it
    attacker.move(4)
    assert attacker.current_node == 4, "Attacker should have moved to node 4 (target)"
    assert attacker.path == [0, 1, 4], "Attacker's path should be updated to include the target"

    # Verify that the target has been captured and its value added to the attacker
    assert chosen_target.value == 5, "Target value should be 5"


def test_defender_behavior():
    # Test Defender class methods and attributes
    G = nx.grid_2d_graph(3, 3)
    G = nx.convert_node_labels_to_integers(G)
    targets = [Target(4,5), Target(7,3)]
    ip = InterdictionProtocol(graph=G, defense_time_threshold=2)
    game = Game(G, interdiction_protocol=ip, num_timesteps=4,
                    moving_attacker_start_nodes=[0],
                    moving_defender_start_nodes=[1],
                    stationary_defender_start_nodes=[2, 6],
                    num_moving_attackers=1, 
                    num_stationary_attackers=1,
                    num_moving_defenders=1, 
                    num_stationary_defenders=0,
                    moving_defender_capture_radius=1,
                    stationary_defender_capture_radius=1,
                    allow_wait=True,
                    targets = targets
                    )

    defender = game.moving_defender_units[0]  # Initialize the defender from game
    attacker = game.moving_attacker_units[0]              # Initialize the attacker from game

    # Check initial positions
    assert defender.current_node == 1, "Defender should start at node 1"
    assert attacker.current_node == 0, "Attacker should start at node 0"

    # Simulate defender movement
    defender.move(2)
    assert defender.current_node == 2, "Defender should have moved to node 2"
    assert defender.path == [1, 2], "Defender's path should be updated"

    # Test attacker capture
    caught = defender.is_attacker_caught(attacker, game)
    assert caught == False, "Attacker should not be caught"

    # Move attacker closer and check capture again
    attacker.move(2)
    caught = defender.is_attacker_caught(attacker, game)
    assert caught == True, "Attacker should be caught within the capture radius"

def test_security_game_output():
    # Generate a simple grid graph for the security game
    G = nx.grid_2d_graph(3, 3)
    G = nx.convert_node_labels_to_integers(G)
    targets = [Target(4,5), Target(7,3)]
    ip = InterdictionProtocol(graph=G, defense_time_threshold=2)
    game = Game(G, interdiction_protocol=ip, num_timesteps=4,
                    moving_attacker_start_nodes=[0, 3],
                    moving_defender_start_nodes=[8, 5],
                    stationary_defender_start_nodes=[2, 6],
                    num_moving_attackers=1, 
                    num_stationary_attackers=1,
                    num_moving_defenders=1, 
                    num_stationary_defenders=1,
                    moving_defender_capture_radius=1,
                    stationary_defender_capture_radius=1,
                    allow_wait=True,
                    targets = targets
                    )

    # Generate the attacker and defender strategy matrices
    attacker_strategies = game.generate_strategy_matrix("attacker")
    defender_strategies = game.generate_strategy_matrix("defender")

    # Assert strategy matrices are correct length
    assert len(attacker_strategies) == 297
    assert len(defender_strategies) == 198
    
    # Generate the utility matrix from the game
    utility_matrix = game.generate_utility_matrix()
    
    # Assert utility matrix has correct dimensions
    assert utility_matrix.shape[0] == len(defender_strategies)
    assert utility_matrix.shape[1] == len(attacker_strategies)
