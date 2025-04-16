import geopandas as gpd
import sys
import pandas as pd
import requests
import osmnx as ox
import networkx as nx
import random
from security_game.security_game import SecurityGame
from security_game.target import Target
from shapely.geometry import Point

class InfraSecurityGame:
    def __init__(self, power_df, block_gdf, power_weights, bbox, mode="block"):
        self.power_df = power_df  # Power feature locations
        self.block_gdf = block_gdf  # Census block population data
        self.power_weights = power_weights  # Dictionary of power type multipliers
        self.bbox = bbox
        self.mode = mode
        self.graph = None  # Security game graph
        self.home_base = None
        self.defender_strategies = None
        self.attacker_strategies = None
        self.utility_matrix = None
    
    def assign_population(self, mode, radius=None):
        power_gdf = gpd.GeoDataFrame(
            self.power_df,
            geometry=gpd.points_from_xy(self.power_df.x, self.power_df.y),
            crs=self.block_gdf.crs
        )
        
        if mode == "block":
            joined = gpd.sjoin(power_gdf, self.block_gdf[['POP20', 'geometry']], how="left", predicate="within")
            self.power_df['population'] = joined['POP20']
        
        elif mode == "radius":
            if radius is None:
                raise ValueError("Radius must be specified for 'radius' mode.")
            
            power_gdf = power_gdf.to_crs(epsg=3857)  # Convert to meters-based CRS
            block_gdf = self.block_gdf.to_crs(epsg=3857)
            
            power_gdf['buffer'] = power_gdf.geometry.buffer(radius)
            self.power_df['population'] = power_gdf['buffer'].apply(
                lambda buf: block_gdf[block_gdf.geometry.intersects(buf)]['POP20'].sum()
                if not block_gdf[block_gdf.geometry.intersects(buf)].empty else float('nan')
            )
        else:
            raise ValueError("Invalid mode. Choose 'block' or 'radius'.")
        
        return self.power_df[['id', 'x', 'y', 'power', 'population']]
    
    def create_security_game_graph(self):
        north, south, east, west = self.bbox
        self.graph = ox.graph_from_bbox(north, south, east, west, network_type="drive")
        self.graph = nx.convert_node_labels_to_integers(self.graph)
        self.graph = self.graph.to_undirected() # Turn this off when we want to implement directed graphs

        for node in self.graph.nodes:
            self.graph.nodes[node]['target'] = False # Replace 'new_attribute' and 'some_value'
            self.graph.nodes[node]['score'] = 0 # Replace 'new_attribute' and 'some_value'
        
        min_x, min_y, max_x, max_y = (
            min(nx.get_node_attributes(self.graph, "x").values()),
            min(nx.get_node_attributes(self.graph, "y").values()),
            max(nx.get_node_attributes(self.graph, "x").values()),
            max(nx.get_node_attributes(self.graph, "y").values()),
        )
        
        self.power_df = self.power_df[
            (self.power_df["x"] >= min_x) & (self.power_df["x"] <= max_x) &
            (self.power_df["y"] >= min_y) & (self.power_df["y"] <= max_y)
        ]
        
        self.power_df["geometry"] = self.power_df.apply(lambda row: Point(row["x"], row["y"]), axis=1)
        power_gdf = gpd.GeoDataFrame(self.power_df, geometry="geometry", crs="EPSG:4326")
        
        targets_di = {}
        for _, row in power_gdf.iterrows():
            closest_node = ox.distance.nearest_nodes(self.graph, row["x"], row["y"])
            target_value = row["population"] * self.power_weights.get(row["power"], 1.0)
            
            if closest_node in targets_di:
                targets_di[closest_node][1] += target_value
            else:
                targets_di[closest_node] = [closest_node, target_value]
            
            self.graph.nodes[closest_node]["target"] = True  
            self.graph.nodes[closest_node]["score"] = targets_di[closest_node][1]

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
            ax.annotate(f"{target.value:.1f}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, color="white")
            target_xs.append(x)
            target_ys.append(y)
            sizes.append(target.value)
        
        # Plot target nodes in red
        ax.scatter(target_xs, target_ys, s=sizes, c="red", label="Target Nodes", alpha=0.7)
        
        # Plot home base node in blue
        home_x, home_y = self.graph.nodes[self.home_base]["x"], self.graph.nodes[self.home_base]["y"]
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
        ax.legend(handles=legend_elements, loc="upper right")
        
        plt.title("Power Grid Infra Security Game")
        plt.show()
    
    def generate(self, num_attackers, num_defenders, home_base_assignments, num_timesteps, interdiction_protocol=None, defense_time_threshold=2, generate_utility_matrix=False, schedule_form=False, general_sum=False, attacker_feature_value=1, defender_feature_value=1, defender_step_cost=0, simple=True, attacker_penalty_factor=3, defender_penalty_factor=3, extra_coverage_weight=1.0):
        self.create_security_game_graph()
    
        if general_sum:
            targets = [
                Target(node=i, attacker_value=data["score"]*attacker_feature_value, defender_value=-data["score"]*defender_feature_value)
                for i, data in self.graph.nodes(data=True)
                if data["target"] == True
            ]
        else:
            targets = [
                Target(node=i, attacker_value=data["score"], defender_value=-data["score"])
                for i, data in self.graph.nodes(data=True)
                if data["target"] == True
            ]

        
        self.targets = targets
        self.home_bases = home_base_assignments
        home_base_labels = [(node,) for node in self.home_bases]
    
        game = SecurityGame(
            num_attackers=num_attackers,
            num_defenders=num_defenders,
            graph=self.graph,
            targets=targets,
            num_timesteps=num_timesteps,
            defender_start_nodes=home_base_labels,
            defender_end_nodes=home_base_labels,
            interdiction_protocol=interdiction_protocol
        )

        if schedule_form:
            self.defender_strategies = None
            self.attacker_strategies = None
            self.utility_matrix, self.attacker_utility_matrix, self.defender_utility_matrix = None, None, None
            sf_defender_step_cost = defender_step_cost if general_sum else 0
            self.schedule_form_dict = game.schedule_form(generate_utility_matrix, simple, attacker_penalty_factor, defender_penalty_factor, extra_coverage_weight, defender_step_cost=sf_defender_step_cost)
        else:
            self.defender_strategies = game.generate_strategy_matrix("defender")
            self.attacker_strategies = game.generate_strategy_matrix("attacker")
            self.schedule_form_dict = None
            if generate_utility_matrix:
                self.utility_matrix, self.attacker_utility_matrix, self.defender_utility_matrix = game.generate_utility_matrix(general_sum, defender_step_cost)
            else:
                self.utility_matrix, self.attacker_utility_matrix, self.defender_utility_matrix = None, None, None