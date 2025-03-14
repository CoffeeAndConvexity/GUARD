import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from security_game.security_game import SecurityGame
from security_game.target import Target
sys.path.append('..')
from utils.target_utils import get_density_scores, get_centroid_scores

class GreenSecurityGame:
    def __init__(self, data, coordinate_rectangle, scoring_method, num_clusters=None, num_rows=5, num_columns=5):
        self.data = data
        self.coordinate_rectangle = coordinate_rectangle
        self.scoring_method = scoring_method
        self.num_clusters = num_clusters
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.graph = None
        self.defender_strategies = None
        self.attacker_strategies = None
        self.utility_matrix = None

    def create_grid(self):
        min_lat, max_lat, min_long, max_long = self.coordinate_rectangle
        lats = np.linspace(min_lat, max_lat, self.num_rows + 1)
        longs = np.linspace(min_long, max_long, self.num_columns + 1)

        cell_side_length_km = (
            geodesic((min_lat, min_long), (max_lat, min_long)).km / self.num_rows,
            geodesic((min_lat, min_long), (min_lat, max_long)).km / self.num_columns,
        )

        grid = [
            (lats[i], lats[i + 1], longs[j], longs[j + 1])
            for i in range(self.num_rows)
            for j in range(self.num_columns)
        ]

        return grid, cell_side_length_km

    def get_scores(self):
        if self.scoring_method == "density":
            return get_density_scores(self.data, self.coordinate_rectangle, self.num_columns, self.num_rows)
        elif self.scoring_method == "centroid":
            if not self.num_clusters:
                raise ValueError("num_clusters must be specified for centroid scoring method.")
            return get_centroid_scores(self.data, self.coordinate_rectangle, self.num_columns, self.num_rows, self.num_clusters)
        else:
            raise ValueError("Invalid scoring method. Choose 'density' or 'centroid'.")

    def fill_missing_cells(self, grid_dict):
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                if (row, col) not in grid_dict:
                    grid_dict[(row, col)] = 0
        return grid_dict

    def convert_to_graph(self, grid_dict):
        G = nx.Graph()

        max_value = max(grid_dict.values())
        min_value = min(grid_dict.values())
        range_value = max_value - min_value

        normalized_values = {
            key: (value - min_value) / range_value if range_value > 0 else 1.0
            for key, value in grid_dict.items()
        }

        for (row, col), norm_value in normalized_values.items():
            inverted_row = self.num_rows - 1 - row
            G.add_node((inverted_row, col), score=norm_value, position=(row, col))

        for inverted_row in range(self.num_rows):
            for col in range(self.num_columns):
                neighbors = [(inverted_row, col + 1), (inverted_row + 1, col)]
                for neighbor in neighbors:
                    if neighbor in G.nodes:
                        G.add_edge((inverted_row, col), neighbor)

        return nx.convert_node_labels_to_integers(G, label_attribute="position")

    def get_home_base_label(self, home_base):
        """
        Get the integer label of the home base node in the graph.
    
        returns: Integer label of the home base node.
        raises: ValueError: If the home base node is not found in the graph.
        """
        if not hasattr(self, 'graph') or self.graph is None:
            raise ValueError("Graph has not been created. Run generate() first.")
    
        for node, attributes in self.graph.nodes(data=True):
            if attributes.get("position") == home_base:
                return node
    
        raise ValueError(f"Home base node with position {self.home_base} not found in the graph.")

    def draw_graph(self, figsize=(12, 10), base_node_size=300, font_size=10, cmap='Blues'):
        """
        Draws the graph using the positions stored in the node "position" attribute.
        Node sizes and colors are scaled with the node "score" attribute.
    
        graph: The NetworkX graph to draw.
        figsize: Tuple specifying the figure size.
        base_node_size: Base size for the nodes; scaling is applied relative to this.
        font_size: Font size for node labels.
        cmap: Colormap for scaling node colors.
        """

        if not hasattr(self, 'graph') or self.graph is None:
            raise ValueError("Graph has not been created. Run generate() first.")
    
        # Determine grid dimensions for y-axis inversion
        max_row = max(node[1]["position"][0] for node in self.graph.nodes(data=True))
    
        # Corrected node positions to invert rows
        positions = {node[0]: (node[1]["position"][1], max_row - node[1]["position"][0]) for node in self.graph.nodes(data=True)}  # (col, inverted row)
    
        # Extract node scores for scaling
        scores = nx.get_node_attributes(self.graph, "score")
        if not scores:
            raise ValueError("The graph does not have 'score' attributes for its nodes.")
    
        # Normalize scores for node size and color scaling
        score_values = np.array(list(scores.values()))
        min_score, max_score = score_values.min(), score_values.max()
        normalized_scores = (score_values - min_score) / (max_score - min_score + 1e-6)
    
        # Scale node sizes and colors
        node_sizes = base_node_size * (1 + normalized_scores)  # Scale size with scores
        node_colors = normalized_scores  # Use normalized scores for colors
    
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
    
        # Draw the graph
        nodes = nx.draw_networkx_nodes(
            self.graph,
            pos=positions,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="black",
            cmap=cmap,
            ax=ax
        )
        edges = nx.draw_networkx_edges(self.graph, pos=positions, ax=ax)
        labels = nx.draw_networkx_labels(self.graph, pos=positions, font_size=font_size, ax=ax)
    
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_score, vmax=max_score))
        sm.set_array([])  # Dummy array for the colorbar
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Node Score')
    
        # Show plot
        plt.title("Graph Visualization with Node Scores")
        plt.axis("off")
        plt.show()

    def generate(self, num_attackers, num_defenders, home_base, num_timesteps, interdiction_protocol=None, defense_time_threshold=2):
        grid, _ = self.create_grid()
        scores = self.get_scores()
        scores = self.fill_missing_cells(scores)
        self.graph = self.convert_to_graph(scores)

        targets = [
            Target(node=i, value=data["score"])
            for i, data in self.graph.nodes(data=True)
            if data["score"] > 0
        ]

        home_base_label = self.get_home_base_label(home_base)

        game = SecurityGame(
            num_attackers=num_attackers,
            num_defenders=num_defenders,
            graph=self.graph,
            targets=targets,
            num_timesteps=num_timesteps,
            defender_start_nodes=[home_base_label],
            defender_end_nodes=[home_base_label],
            interdiction_protocol=interdiction_protocol,
            defense_time_threshold=defense_time_threshold,
        )

        self.defender_strategies = game.generate_strategy_matrix("defender")
        self.attacker_strategies = game.generate_strategy_matrix("attacker")

        self.utility_matrix = game.generate_utility_matrix()