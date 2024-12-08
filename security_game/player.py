import random
import networkx as nx

class Player:
    def __init__(self, name, start_node):
        self.name = name
        self.current_node = start_node
        self.path = [start_node]  # Track the path taken
     
    def move(self, new_node):
        self.current_node = new_node
        self.path.append(new_node)

class Attacker(Player):
    def __init__(self, name, start_node):
        super().__init__(name, start_node)

    def choose_target(self, targets):
        # Simple heuristic: pick the target with the highest value
        target = max(targets, key=lambda t: t.value)
        return target

class Defender(Player):
    def __init__(self, name, start_node, capture_radius):
        super().__init__(name, start_node)
        self.capture_radius = capture_radius
     
    def is_attacker_caught(self, attacker, game):
        # Check if attacker is within capture radius
        path_length = nx.shortest_path_length(game.graph, source=self.current_node, target=attacker.current_node)
        return path_length <= self.capture_radius