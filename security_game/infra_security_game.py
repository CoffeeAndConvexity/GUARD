import geopandas as gpd
import sys
import pandas as pd
import numpy as np
import requests
import osmnx as ox
import networkx as nx
import random
from security_game.domain_specific_sg import DomainSpecificSG
from security_game.security_game import SecurityGame
from security_game.target import Target
from shapely.geometry import Point
from matplotlib import pyplot as plt
from utils.misc_utils import get_nearest_nodes_from_coords, get_nearest_node_tuples


class InfraSecurityGame(DomainSpecificSG):
    def __init__(self, infra_df, block_gdf, infra_weights, bbox, escape_point=None, mode="block", population_scaler=1):
        self.infra_df = infra_df  # Infra feature locations
        self.block_gdf = block_gdf  # Census block population data
        self.infra_weights = infra_weights  # Dictionary of infra type multipliers
        self.bbox = bbox
        self.mode = mode
        self.population_scaler = population_scaler
        self.escape_point = escape_point
        self.infra_pop_df = None # Population-assigned infra df
        self.graph = None  # Security game graph
        self.num_timesteps = None
        self.num_attackers = None
        self.num_defenders = None
        self.defender_actions = None
        self.attacker_actions = None
        self.utility_matrix = None
    
    def assign_population(self, mode, radius=None):
        infra_gdf = gpd.GeoDataFrame(
            self.infra_df,
            geometry=gpd.points_from_xy(self.infra_df.x, self.infra_df.y),
            crs=self.block_gdf.crs
        )
        
        if mode == "block":
            joined = gpd.sjoin(infra_gdf, self.block_gdf[['POP20', 'geometry']], how="left", predicate="within")
            self.infra_df['population'] = joined['POP20']
        
        elif mode == "radius":
            if radius is None:
                raise ValueError("Radius must be specified for 'radius' mode.")
            
            infra_gdf = infra_gdf.to_crs(epsg=3857)  # Convert to meters-based CRS
            block_gdf = self.block_gdf.to_crs(epsg=3857)
            
            infra_gdf['buffer'] = infra_gdf.geometry.buffer(radius)
            self.infra_df['population'] = infra_gdf['buffer'].apply(
                lambda buf: block_gdf[block_gdf.geometry.intersects(buf)]['POP20'].sum()
                if not block_gdf[block_gdf.geometry.intersects(buf)].empty else float('nan')
            )
        else:
            raise ValueError("Invalid mode. Choose 'block' or 'radius'.")
        
        self.infra_pop_df = self.infra_df[['id', 'x', 'y', 'type', 'population']]

    def create_security_game_graph(self, general_sum, random_target_values=False):
        north, south, east, west = self.bbox
        self.graph = ox.graph_from_bbox(north, south, east, west, network_type="drive")
        self.graph = nx.convert_node_labels_to_integers(self.graph)
        self.graph = self.graph.to_undirected()

        for node in self.graph.nodes:
            self.graph.nodes[node]['target'] = False
            self.graph.nodes[node]['score'] = 0

        min_x, min_y, max_x, max_y = (
            min(nx.get_node_attributes(self.graph, "x").values()),
            min(nx.get_node_attributes(self.graph, "y").values()),
            max(nx.get_node_attributes(self.graph, "x").values()),
            max(nx.get_node_attributes(self.graph, "y").values()),
        )

        self.assign_population(mode="block", radius=None)
        self.infra_pop_df = self.infra_pop_df[
            (self.infra_df["x"] >= min_x) & (self.infra_df["x"] <= max_x) &
            (self.infra_df["y"] >= min_y) & (self.infra_df["y"] <= max_y)
        ]

        self.infra_pop_df["geometry"] = self.infra_pop_df.apply(lambda row: Point(row["x"], row["y"]), axis=1)
        infra_pop_gdf = gpd.GeoDataFrame(self.infra_pop_df, geometry="geometry", crs="EPSG:4326")

        targets_di = {}
        for _, row in infra_pop_gdf.iterrows():
            closest_node = ox.distance.nearest_nodes(self.graph, row["x"], row["y"])
            base_weight = self.infra_weights.get(row["type"], 1.0)
            score = base_weight * (1 + np.log(row["population"] + 1)) ** self.population_scaler

            if closest_node in targets_di:
                targets_di[closest_node][1].append(score)
            else:
                targets_di[closest_node] = [closest_node, [score], None]

        # Go back and set score for each target as max of the repeats + an avg of the rest
        for t in targets_di:
            if len(targets_di[t][1]) > 1:
                targets_di[t][2] = max(targets_di[t][1]) + np.mean(targets_di[t][1])
            else:
                targets_di[t][2] = targets_di[t][1][0]

        for node, (_, _, val) in targets_di.items():
            self.graph.nodes[node]["target"] = True
            self.graph.nodes[node]["score"] = val
            
            # If random, overwrite all real-world scores before any escape proximity adjustments
        if random_target_values and not general_sum:
            scores = [v[2] for v in targets_di.values()]
            min_score = min(scores)
            max_score = max(scores)

            # Assign randomized scores
            randomized_scores = {}
            for node in self.graph.nodes:
                if self.graph.nodes[node]["target"]:
                    rand_val = float(np.random.uniform(min_score, max_score))
                    randomized_scores[node] = rand_val

            # Normalize scores so the max is 1.0
            max_rand = max(randomized_scores.values())
            if max_rand > 0:
                for node, rand_val in randomized_scores.items():
                    self.graph.nodes[node]["score"] = rand_val / max_rand
            else:
                for node in randomized_scores:
                    self.graph.nodes[node]["score"] = 0.0

            return  # Exit early to skip escape logic when using random scores

        # Only apply escape proximity if not random
        if hasattr(self, "escape_point") and self.escape_point is not None:
            escape_x, escape_y = self.escape_point
            for node, data in self.graph.nodes(data=True):
                if data["target"]:
                    node_x, node_y = data["x"], data["y"]
                    distance = np.sqrt((escape_x - node_x) ** 2 + (escape_y - node_y) ** 2)
                    self.graph.nodes[node]["escape_proximity"] = distance

    def draw_graph(self):
        if self.graph is None:
            raise ValueError("Graph has not been generated. Run generate() first.")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor("black")
        ox.plot_graph(self.graph, ax=ax, node_size=50, edge_color="gray", show=False, close=False)
        
        target_xs, target_ys, sizes = [], [], []
        for target in self.targets:
            x, y = self.graph.nodes[target.node]["x"], self.graph.nodes[target.node]["y"]
            # Annotate target value in white
            ax.annotate(f"{target.attacker_value:.3f}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, color="white")
            target_xs.append(x)
            target_ys.append(y)
            sizes.append(target.attacker_value)
        
        # Normalize sizes so the largest target is a reasonable max size
        max_size = max(sizes) if sizes else 1
        scaled_sizes = [100 + 200 * (s / max_size) for s in sizes]  # min 100, max 300
        # Plot target nodes in red
        ax.scatter(target_xs, target_ys, s=scaled_sizes, c="red", label="Target Nodes", alpha=0.7)
        
        # Plot home base nodes in blue
        for tup in self.home_bases:
            for h in tup:
                home_x, home_y = self.graph.nodes[h]["x"], self.graph.nodes[h]["y"]
                ax.scatter(home_x, home_y, s=200, c="blue", edgecolors="white", label="Home Base", linewidth=1.5)
        
        # Annotate all nodes with their integer labels in yellow
        for node in self.graph.nodes:
            x, y = self.graph.nodes[node]["x"], self.graph.nodes[node]["y"]
            ax.annotate(str(node), (x, y), textcoords="offset points", xytext=(-8, -8), fontsize=8, color="yellow")
    
        # Manually add legend for node labels
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label="Target Nodes"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, markeredgecolor="white", label="Home Base"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label="Node Labels")
        ]
        ax.legend(handles=legend_elements, loc="lower right")
        
        plt.title("Infra Security Game")
        plt.show()
    
    def generate(self, num_attackers, num_defenders, home_base_assignments, num_timesteps, interdiction_protocol=None, defense_time_threshold=2, generate_utility_matrix=False, generate_actions=True, force_return=False, schedule_form=False, general_sum=False, random_target_values=False, randomize_target_utility_matrix=False, attacker_feature_value=1, defender_feature_value=1, defender_step_cost=0, simple=True, attacker_penalty_factor=3, defender_penalty_factor=3, alpha=1):
        self.num_timesteps = num_timesteps
        self.num_attackers = num_attackers
        self.num_defenders = num_defenders
        self.defense_time_threshold = defense_time_threshold
        self.force_return = force_return
        self.create_security_game_graph(general_sum, random_target_values)

        if general_sum:
            # normalize escape proximities among targets
            escape_distances = [data.get("escape_proximity", 0) for i, data in self.graph.nodes(data=True) if data["target"]]
            if escape_distances:
                max_escape = max(escape_distances)
                min_escape = min(escape_distances)
                range_escape = max_escape - min_escape if max_escape != min_escape else 1.0

            targets = []
            for i, data in self.graph.nodes(data=True):
                if data["target"]:
                    escape_proximity = data.get("escape_proximity", 0)
                    norm_escape = (max_escape - escape_proximity) / range_escape if escape_distances else 1.0  # closer is higher
                    adjusted_attacker_value = data["score"] * attacker_feature_value * (1 + alpha*norm_escape)

                    targets.append(
                        Target(
                            node=i,
                            attacker_value=adjusted_attacker_value,
                            defender_value=-data["score"] * defender_feature_value
                        )
                    )
        else:
            targets = [
                Target(node=i, attacker_value=data["score"], defender_value=-data["score"])
                for i, data in self.graph.nodes(data=True)
                if data["target"]
            ]
            print([t.attacker_value for t in targets])
        
        self.targets = targets
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
            self.attacker_sactions = None
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
                