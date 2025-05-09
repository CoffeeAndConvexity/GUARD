import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from security_game.domain_specific_sg import DomainSpecificSG
from security_game.security_game import SecurityGame
from security_game.target import Target
sys.path.append('..')
from utils.target_utils import get_density_scores, get_centroid_scores
from utils.misc_utils import get_nearest_nodes_from_coords, get_nearest_node_tuples, point_line_distance

class GreenSecurityGame(DomainSpecificSG):
    def __init__(self, data, coordinate_rectangle, scoring_method, num_clusters=None, num_rows=5, num_columns=5, escape_line_points = None):
        self.data = data
        self.coordinate_rectangle = coordinate_rectangle
        self.scoring_method = scoring_method
        self.num_clusters = num_clusters
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.escape_line_points = escape_line_points
        self.graph = None
        self.num_timesteps = None
        self.num_attackers = None
        self.num_defenders = None
        self.defender_actions = None
        self.attacker_actions = None
        self.utility_matrix = None

    def create_grid(self):
        min_lat, max_lat, min_long, max_long = self.coordinate_rectangle
        lats = np.linspace(min_lat, max_lat, self.num_rows + 1)
        longs = np.linspace(min_long, max_long, self.num_columns + 1)

        cell_side_length_km = (
            geodesic((min_lat, min_long), (max_lat, min_long)).km / self.num_rows,
            geodesic((min_lat, min_long), (min_lat, max_long)).km / self.num_columns,
        )

        grid = []
        lat_center_grid = []
        for i in range(self.num_rows):
            row_lat = []
            for j in range(self.num_columns):
                lat_min, lat_max = lats[i], lats[i + 1]
                long_min, long_max = longs[j], longs[j + 1]
                grid.append((lat_min, lat_max, long_min, long_max))
                row_lat.append(((lat_min + lat_max) / 2, (long_min + long_max) / 2))
            lat_center_grid.append(row_lat)

        return grid, cell_side_length_km, lat_center_grid

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

    # def convert_to_graph(self, grid_dict, lat_center_grid, general_sum):
    #     G = nx.Graph()

    #     if general_sum:
    #         for (row, col), raw_value in grid_dict.items():
    #             inverted_row = self.num_rows - 1 - row
    #             lat, lon = lat_center_grid[row][col]
    #             G.add_node((inverted_row, col), score=raw_value, position=(row, col), x=lon, y=lat)
    #     else:
    #         max_value = max(grid_dict.values())
    #         min_value = min(grid_dict.values())
    #         range_value = max_value - min_value

    #         normalized_values = {
    #             key: (value - min_value) / range_value if range_value > 0 else 1.0
    #             for key, value in grid_dict.items()
    #         }

    #         for (row, col), norm_value in normalized_values.items():
    #             inverted_row = self.num_rows - 1 - row
    #             lat, lon = lat_center_grid[row][col]
    #             G.add_node((inverted_row, col), score=norm_value, position=(row, col), x=lon, y=lat)

    #     for inverted_row in range(self.num_rows):
    #         for col in range(self.num_columns):
    #             neighbors = [(inverted_row, col + 1), (inverted_row + 1, col)]
    #             for neighbor in neighbors:
    #                 if neighbor in G.nodes:
    #                     G.add_edge((inverted_row, col), neighbor)

    #     return nx.convert_node_labels_to_integers(G, label_attribute="position")

    def convert_to_graph(self, grid_dict, lat_center_grid, general_sum, random_target_values=False):
        G = nx.Graph()
        raw_values = []

        for (row, col), raw_value in grid_dict.items():
            inverted_row = self.num_rows - 1 - row
            lat, lon = lat_center_grid[row][col]

            raw_values.append(raw_value)

            node_attrs = {
                "score": raw_value,
                "position": (row, col),
                "x": lon,
                "y": lat
            }

            if general_sum:
                if hasattr(self, "escape_line_points") and self.escape_line_points is not None:
                    escape_distance = point_line_distance(
                        (lon, lat),
                        self.escape_line_points[0],
                        self.escape_line_points[1]
                    )
                    node_attrs["escape_proximity"] = escape_distance
                else:
                    node_attrs["escape_proximity"] = None

            G.add_node((inverted_row, col), **node_attrs)

        # Overwrite node scores with random values if requested for zero sum games where no later randomization will happen
        if random_target_values and raw_values and not general_sum:
            min_score = min(raw_values)
            max_score = max(raw_values)

            # Step 1: Assign randomized scores
            randomized_scores = {}
            for node in G.nodes:
                if G.nodes[node]["score"] > 0:
                    rand_val = float(np.random.uniform(min_score, max_score))
                    randomized_scores[node] = rand_val

            # Step 2: Normalize to [0, 1] by dividing by max value
            max_rand = max(randomized_scores.values())
            if max_rand > 0:
                for node, val in randomized_scores.items():
                    G.nodes[node]["score"] = val / max_rand
            else:
                for node in randomized_scores:
                    G.nodes[node]["score"] = 0.0
            # Remove escape proximity if present — we don't want real-world logic to influence
            # for node in G.nodes:
            #     G.nodes[node].pop("escape_proximity", None)

        # Add edges
        for inverted_row in range(self.num_rows):
            for col in range(self.num_columns):
                neighbors = [(inverted_row, col + 1), (inverted_row + 1, col)]
                for neighbor in neighbors:
                    if neighbor in G.nodes:
                        G.add_edge((inverted_row, col), neighbor)

        return nx.convert_node_labels_to_integers(G, label_attribute="position")

    def get_node_label(self, node):
        """
        Get the integer label of the any node in the graph.
    
        returns: Integer label of the node.
        raises: ValueError: If the node is not found in the graph.
        """
        if not hasattr(self, 'graph') or self.graph is None:
            raise ValueError("Graph has not been created. Run generate() first.")
    
        for n, attributes in self.graph.nodes(data=True):
            if attributes.get("position") == node:
                return n
    
        raise ValueError(f"Home base node with position {node} not found in the graph.")

    def draw_graph(self, figsize=(12, 10), base_node_size=300, font_size=10, cmap='Reds'):
        """
        Draws the graph using the positions stored in the node "position" attribute.
        Nodes are red if they are targets, blue if they are home bases, and both if overlapping.
        """
        if not hasattr(self, 'graph') or self.graph is None:
            raise ValueError("Graph has not been created. Run generate() first.")

        max_row = max(node[1]["position"][0] for node in self.graph.nodes(data=True))
        positions = {
            node[0]: (node[1]["position"][1], max_row - node[1]["position"][0])
            for node in self.graph.nodes(data=True)
        }

        # Get scores (only for target nodes)
        scores = nx.get_node_attributes(self.graph, "score")
        all_nodes = list(self.graph.nodes())
        node_colors = []
        for node in all_nodes:
            node_colors.append(scores.get(node, 0.0))  # default to 0 for non-targets

        score_values = np.array(node_colors)
        max_score = score_values.max()

        # Use a discrete fixed size for target nodes
        node_sizes = np.where(score_values > 0, base_node_size * 1.6, base_node_size)

        fig, ax = plt.subplots(figsize=figsize)

        # Draw graph nodes (targets in red, others white)
        nodes = nx.draw_networkx_nodes(
            self.graph,
            pos=positions,
            node_color=score_values,
            node_size=node_sizes,
            edgecolors="black",
            cmap=cmap,
            ax=ax
        )
        nx.draw_networkx_edges(self.graph, pos=positions, ax=ax)
        nx.draw_networkx_labels(self.graph, pos=positions, font_size=font_size, ax=ax)

        # Draw home bases in blue
        label_added = False
        for tup in self.home_bases:
            for h in tup:
                if h in self.graph.nodes:
                    x, y = positions[h]
                    ax.scatter(
                        x, y,
                        s=280,
                        c="blue",
                        alpha=0.7,
                        edgecolors="white",
                        linewidth=1.5,
                        label="Home Base" if not label_added else None,
                        zorder=5
                    )
                    label_added = True

        # Colorbar for target scores
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=max_score))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Target Score')

        # Final legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        plt.title("Green Security Game Graph")
        plt.axis("off")
        plt.show()

    def generate(self, num_attackers, num_defenders, home_base_assignments, num_timesteps, interdiction_protocol=None, defense_time_threshold=2, generate_utility_matrix=False, generate_actions=True, force_return=False, schedule_form=False, general_sum=False, random_target_values=False, randomize_target_utility_matrix=False, attacker_animal_value=1, defender_animal_value=1, defender_step_cost=0, simple=True, attacker_penalty_factor=3, defender_penalty_factor=3, alpha=1):
        self.num_timesteps = num_timesteps
        self.num_attackers = num_attackers
        self.num_defenders = num_defenders
        self.defense_time_threshold = defense_time_threshold
        self.force_return = force_return

        grid, cell_side_length_km, lat_center_grid = self.create_grid()
        # print(cell_side_length_km)
        # print(defender_step_cost)
        defender_step_cost = defender_step_cost * (cell_side_length_km[0] + cell_side_length_km[1]) / 2
        # print(defender_step_cost)
        scores = self.get_scores()
        scores = self.fill_missing_cells(scores)
        self.graph = self.convert_to_graph(scores, lat_center_grid, general_sum, random_target_values)

        # Step 1: collect escape proximities for targets only
        proximities = [
            data["escape_proximity"]
            for i, data in self.graph.nodes(data=True)
            if data["score"] > 0 and data.get("escape_proximity") is not None
        ]
        if proximities:
            min_prox = min(proximities)
            max_prox = max(proximities)
            range_prox = max_prox - min_prox if max_prox > min_prox else 1.0  # prevent divide by zero

        # Step 2: define the targets
        if general_sum:
            targets = []
            for i, data in self.graph.nodes(data=True):
                if data["score"] > 0:
                    prox = data.get("escape_proximity", None)
                    if prox is not None:
                        normalized_prox = 1 - (prox - min_prox) / range_prox  # inverted
                        proximity_multiplier = 1 + alpha * normalized_prox
                    else:
                        proximity_multiplier = 1  # no adjustment if missing

                    attacker_val = data["score"] * attacker_animal_value * proximity_multiplier
                    defender_val = -data["score"] * defender_animal_value

                    targets.append(Target(node=i, attacker_value=attacker_val, defender_value=defender_val))
        # if general_sum:
        #     targets = [
        #         Target(node=i, attacker_value=data["score"]*attacker_animal_value*(1+alpha*data["escape_proximity"]), defender_value=-data["score"]*defender_animal_value)
        #         for i, data in self.graph.nodes(data=True)
        #         if data["score"] > 0
        #     ]
        else:
            targets = [
                Target(node=i, attacker_value=data["score"], defender_value=-data["score"])
                for i, data in self.graph.nodes(data=True)
                if data["score"] > 0
            ]


        self.targets = targets

        # home_base_labels = [(self.get_node_label(node),) for node in home_base_assignments]
        
        # Home bases in GSGs are 1 per defender
        # self.home_bases = [tup[0] for tup in home_base_labels]
        self.home_bases = get_nearest_node_tuples(self.graph, home_base_assignments)
        home_base_labels = self.home_bases

        game = SecurityGame(
            num_attackers=num_attackers,
            num_defenders=num_defenders,
            graph=self.graph,
            targets=targets,
            num_timesteps=num_timesteps,
            defender_start_nodes=home_base_labels,
            defender_end_nodes=home_base_labels,
            interdiction_protocol=interdiction_protocol,
            defense_time_threshold=defense_time_threshold,
            force_return=force_return
        )

        if schedule_form:
            self.defender_actions = None
            self.attacker_actions = None
            self.utility_matrix, self.attacker_utility_matrix, self.defender_utility_matrix = None, None, None
            sf_defender_step_cost = defender_step_cost if general_sum else 0
            self.schedule_form_dict = game.schedule_form(generate_utility_matrix, generate_actions, general_sum, simple, attacker_penalty_factor, defender_penalty_factor, randomize_target_utility_matrix, defender_step_cost=sf_defender_step_cost)
        else:
            self.schedule_form_dict = None
            if generate_utility_matrix:
                self.defender_actions = game.generate_actions("defender")
                self.attacker_actions = game.generate_actions("attacker")
                self.utility_matrix, self.attacker_utility_matrix, self.defender_utility_matrix = game.generate_utility_matrix(general_sum, defender_step_cost)
            else:
                self.utility_matrix, self.attacker_utility_matrix, self.defender_utility_matrix = None, None, None
                if generate_actions:
                    self.defender_actions = game.generate_actions("defender")
                    self.attacker_actions = game.generate_actions("attacker")
                else:
                    self.defender_actions = None
                    self.attacker_actions = None
                

