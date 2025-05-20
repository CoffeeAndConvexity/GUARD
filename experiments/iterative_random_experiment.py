import pandas as pd
import geopandas as gpd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import itertools
import sys
sys.path.append('..')
from security_game.target import Target
from security_game.green_security_game import GreenSecurityGame
from security_game.infra_security_game import InfraSecurityGame

from solvers.mip import mip
from solvers.nash import nash
from solvers.double_oracle import double_oracle
from solvers.double_oracle_sf import double_oracle_sf
from solvers.no_regret import regret_matching
from solvers.simple_sse_lp import solve_sse_lp
from solvers.nfg_sse_lp import solve_general_sum_normal_form

from utils.random_utils import generate_random_utility_matrix_like, generate_random_target_utility_matrix_like, generate_random_target_utility_matrix_like_v2

import time
from matplotlib import pyplot as plt
from collections import defaultdict, Counter
import copy

# Preprocess GSG Data
df = pd.read_csv("lobeke.csv")
df.dropna(inplace=True)

lat_min, lon_min = 2.0530, 15.8790
lat_max, lon_max = 2.2837, 16.2038

coordinate_rectangle = [lat_min, lat_max, lon_min, lon_max]

boulou_camp = (2.2,15.9)
# lobeke_camp = (2.25,15.75)
kabo_djembe = (2.0532352380408088, 16.085709866529694)
bomassa = (2.2037280296158355, 16.187056364164913)
inner_post = (2.2,15.98)
sangha_river = [(2.2837, 16.16283352464626),(2.053, 16.066212728001727)]


# Preprocess ISG Data
gdf = gpd.read_file("chinatown_infra.geojson")

# Step 1: Handle relevant columns
infra_columns = [
    "id", "name", "power", "man_made", "amenity",
    "generator:method", "generator:source", "geometry"
]
available_columns = [col for col in infra_columns if col in gdf.columns]
gdf = gdf[available_columns].copy()

# Step 2: Extract generator type if present
gdf["generator_type"] = gdf.get("generator:method")
if "generator_type" not in gdf.columns or gdf["generator_type"].isnull().all():
    gdf["generator_type"] = gdf.get("generator:source")

# Step 3: Construct unified 'type' column
gdf["type"] = gdf.get("power")
if "amenity" in gdf.columns:
    gdf["type"] = gdf["type"].combine_first(gdf["amenity"])
if "man_made" in gdf.columns:
    gdf["type"] = gdf["type"].combine_first(gdf["man_made"])

# Step 4: Refine generator classification (solar vs. other)
gdf.loc[(gdf["type"] == "generator") & (gdf["generator_type"] == "photovoltaic"), "type"] = "solar_generator"
gdf.loc[(gdf["type"] == "generator") & (gdf["generator_type"] == "solar"), "type"] = "solar_generator"

# Step 5: Drop raw columns now that 'type' is finalized
df_simple = gdf[["id", "name", "type", "geometry"]].copy()

# Step 6: Separate nodes and ways
df_nodes = df_simple[df_simple["id"].str.contains("node")].copy()
df_nodes["x"] = df_nodes.geometry.x
df_nodes["y"] = df_nodes.geometry.y
df_nodes = df_nodes.drop(columns=["geometry"])

df_ways = df_simple[df_simple["id"].str.contains("way")].copy()
df_ways = df_ways.set_geometry("geometry").to_crs("EPSG:32618")
df_ways["centroid"] = df_ways.geometry.centroid
df_ways = df_ways.set_geometry("centroid").to_crs("EPSG:4326")
df_ways["x"] = df_ways.geometry.x
df_ways["y"] = df_ways.geometry.y
df_ways = df_ways.drop(columns=["geometry", "centroid"])

# Step 7: Combine nodes and ways
df_combined = pd.concat([df_nodes, df_ways], ignore_index=True)
df_combined = pd.concat([df_nodes, df_ways], ignore_index=True)
ny_blocks_gdf =  gpd.read_file("tl_2020_36_tabblock20.shp")
INFRA_WEIGHTS = {
    # Power Infrastructure
    "plant": 1.5,
    "generator": 1.35,
    "solar_generator": 0.95,
    "substation": 1.45,
    "transformer": 1.25,
    "tower": 1.1,
    "pole": 0.85,
    "line": 1.0,
    "minor_line": 0.9,
    "cable": 0.95,
    "switchgear": 1.2,
    "busbar": 0.8,
    "bay": 0.85,
    "converter": 1.05,
    "insulator": 0.75,
    "portal": 0.75,
    "connection": 0.7,
    "compensator": 1.0,
    "rectifier": 0.95,
    "inverter": 0.95,
    "storage": 0.9,

    # Healthcare
    "hospital": 1.5,
    "clinic": 1.35,

    # Education
    "school": 1.25,
    "university": 1.4,

    # Water & Sanitation
    "water_works": 1.45,
    "wastewater_plant": 1.4,

    # Government & Emergency Services
    "fire_station": 1.3,
    "police": 1.4,
    "courthouse": 1.2,

    # Critical Infrastructure
    "bunker_silo": 1.0,

    # Communications
    "communications_tower": 1.25,
}

# Bounding box for Hoboken, NJ
# bbox_hoboken_small = (40.752635, 40.745600, -74.030386,-74.043903)
bbox_hoboken_low = (40.745411, 40.735486, -74.025857,-74.041479)
bbox_hoboken_east = (40.748337, 40.734641,-74.022961,-74.031286)
bbox_downtown = (40.718721, 40.714078, -73.996074, -74.002651)
bbox_downtown_large = (40.7215, 40.710, -73.9935, -74.010)
college_police = (40.743293077312465, -74.02670221027175)
police_station = (40.73768931976651, -74.02990519431108)
traffic_police = (40.7366602084371, -74.03449866349136)
downtown_station = (40.71232433042349, -74.00187755238431)
fifth_ave_station = (40.71637413934789, -73.9973285259067)
fifth_precinct = (40.71625547686622, -73.99736909131171)
booking_station = (40.716191530904815, -74.00102237385177)
police_plaza = (40.71236124409745, -74.00173715463521)
troop_nyc = (40.71657885026091, -74.00641139014367)
first_precinct = (40.720411300417446, -74.0070247584372)
brooklyn_bridge = (40.712438951145266, -74.004937098962)


def run_random_gsg_nfg_do():
    schedule_form_kwargs = {
        "schedule_form": False,
        "simple": False,
        "attacker_penalty_factor": 5,
        "defender_penalty_factor": 5,
    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  1, 
        "defender_feature_value": 1, 
        "defender_step_cost": 0, 
    }

    supports=[]
    defender_actions_sizes = []
    attacker_actions_sizes = []
    iterations_to_converge = []
    iteration_times_li = []
    gaps_li = []
    num_defenders_li = []

    for num_defenders in [1,2]:
        for seed in seeds:
            np.random.seed(seed)
            print(f"starting seed: {seed}")
            gsg = GreenSecurityGame(df, coordinate_rectangle, "centroid", num_clusters=10, num_rows=7, num_columns=7)
            gsg.generate(num_attackers=1, 
                    num_defenders=num_defenders, 
                    home_base_assignments=[(kabo_djembe, bomassa, inner_post) for i in range(num_defenders)],
                    num_timesteps=8, 
                    generate_utility_matrix=False,
                    random_target_values=True,
                    defense_time_threshold=1, 
                    generate_actions=False, 
                    force_return=False, 
                    general_sum=False, 
                    **schedule_form_kwargs)

            D_a, D_d, u, A_a, A_d, c, iteration_times, gaps = double_oracle(gsg,eps=1e-12, verbose=False) #How to randomize for NFG DO?
            support = sum([1 for p in D_d if p!=0])
            supports.append(support)
            iterations_to_converge.append(c)
            defender_actions_sizes.append(len(A_d))
            attacker_actions_sizes.append(len(A_a))
            iteration_times_li.append(iteration_times)
            gaps_li.append(gaps)
            num_defenders_li.append(num_defenders)

    df = pd.DataFrame()
    seedli = seeds + seeds
    df["seed"] = seedli
    df["num_timesteps"] = [8 for i in range(len(seedli))]
    df["num_defenders"] = num_defenders_li
    df["num_attackers"] = [1 for i in range(len(seedli))]
    df["num_clusters"] = [10 for i in range(len(seedli))]
    df["dims"] = [7 for i in range(len(seedli))]
    df["defense_time_threshold"] = [1 for i in range(len(seedli))]
    df["force_return"] = [False for i in range(len(seedli))]
    df["iterations_to_converge"] = iterations_to_converge
    df["defender_actions_size"] = defender_actions_sizes
    df["attacker_actions_size"] = attacker_actions_sizes
    df["iteration_time"] = iteration_times_li
    df["gap"] = gaps_li
    df["def_support"] = supports


    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_time", "gap"], ignore_index=True)

    # Add a column for iteration number within each grouped trial
    exploded_df["iteration_number"] = exploded_df.groupby(
        ["seed","num_defenders"]
    ).cumcount()

    # Reorder
    cols = [
        "seed", "num_timesteps","num_attackers", "num_defenders", "num_clusters", "dims", "force_return", "defense_time_threshold",
        "iteration_number", "iteration_time", "gap", 
        "iterations_to_converge", "def_support", "defender_actions_size", "attacker_actions_size"
    ]
    exploded_df = exploded_df[cols]

    # Make iteration number 1-indexed
    exploded_df["iteration_number"] += 1
    exploded_df.to_csv("GSG_NFG_DO_RANDOM_TARGET_VALUES_FR_FALSE.csv")


def run_rm_gsg_nfg_rm():
    schedule_form_kwargs = {
        "schedule_form": False,
        "simple": False,
        "attacker_penalty_factor": 5,
        "defender_penalty_factor": 5,
    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  1, 
        "defender_feature_value": 1, 
        "defender_step_cost": 0, 
    }

    gsg = GreenSecurityGame(df, coordinate_rectangle, "centroid", num_clusters=10, num_rows=7, num_columns=7)
    gsg.generate(num_attackers=1, 
                num_defenders=1, 
                home_base_assignments=[(kabo_djembe, bomassa, inner_post)], 
                num_timesteps=8, 
                generate_utility_matrix=True, 
                random_target_values=False,
                defense_time_threshold=1, 
                generate_actions=False, 
                force_return=False, 
                general_sum=False, 
                **schedule_form_kwargs)

    gaps_li = []
    interval_times_li = []
    for seed in seeds:
        print(f"starting seed {seed}")
        np.random.seed(seed)
        random_matrix = np.random.uniform(low=np.min(gsg.utility_matrix), high=np.max(gsg.utility_matrix), size=gsg.utility_matrix.shape)
        print("game done generating, running")
        D_d, U, gaps, interval_times = regret_matching(random_matrix, runtime=120, interval=5, iterations=10000, averaging=0, alternations=False, plus=False, predictive=False, verbose=True)
        gaps_li.append(gaps)
        interval_times_li.append(interval_times)

    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps": [8 for i in range(len(seeds))],
        "num_attackers": [1 for i in range(len(seeds))],
        "num_defenders": [1 for i in range(len(seeds))],
        "defense_time_threshold": [1 for i in range(len(seeds))],
        "force_return": [False for i in range(len(seeds))],
        "iteration_times": interval_times_li,
        "gaps": gaps_li
    })

    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_times", "gaps"], ignore_index=True)

    # Add a column for iteration number within each trial
    exploded_df["interval_number"] = exploded_df.groupby(["seed"]).cumcount()

    # Reorder
    cols = ["seed","num_timesteps", "num_attackers", "num_defenders", "defense_time_threshold", "force_return", "interval_number", "iteration_times", "gaps"]
    exploded_df = exploded_df[cols]

    exploded_df["interval_number"] = exploded_df["interval_number"]+1
    exploded_df["iteration_number"] = exploded_df["interval_number"]*5
    exploded_df.to_csv("GSG_NFG_RM_RANDOM_MATRIX_FR_FALSE.csv")

def run_rm_gsg_nfg_rmp():
    schedule_form_kwargs = {
        "schedule_form": False,
        "simple": False,
        "attacker_penalty_factor": 5,
        "defender_penalty_factor": 5,
    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  1, 
        "defender_feature_value": 1, 
        "defender_step_cost": 0, 
    }

    gsg = GreenSecurityGame(df, coordinate_rectangle, "centroid", num_clusters=10, num_rows=7, num_columns=7)
    gsg.generate(num_attackers=1, 
                num_defenders=1, 
                home_base_assignments=[(kabo_djembe, bomassa, inner_post)], 
                num_timesteps=8, 
                generate_utility_matrix=True, 
                random_target_values=False,
                defense_time_threshold=1, 
                generate_actions=False, 
                force_return=False, 
                general_sum=False, 
                **schedule_form_kwargs)

    gaps_li = []
    interval_times_li = []
    for seed in seeds:
        print(f"starting seed {seed}")
        np.random.seed(seed)
        random_matrix = np.random.uniform(low=np.min(gsg.utility_matrix), high=np.max(gsg.utility_matrix), size=gsg.utility_matrix.shape)
        print("game done generating, running")
        D_d, U, gaps, interval_times = regret_matching(random_matrix, runtime=120, interval=5, iterations=10000, averaging=1, alternations=True, plus=True, predictive=False, verbose=True)
        gaps_li.append(gaps)
        interval_times_li.append(interval_times)

    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps": [8 for i in range(len(seeds))],
        "num_attackers": [1 for i in range(len(seeds))],
        "num_defenders": [1 for i in range(len(seeds))],
        "defense_time_threshold": [1 for i in range(len(seeds))],
        "force_return": [True for i in range(len(seeds))],
        "iteration_times": interval_times_li,
        "gaps": gaps_li
    })

    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_times", "gaps"], ignore_index=True)

    # Add a column for iteration number within each trial
    exploded_df["interval_number"] = exploded_df.groupby(["seed"]).cumcount()

    # Reorder
    cols = ["seed","num_timesteps", "num_attackers", "num_defenders", "defense_time_threshold", "force_return", "interval_number", "iteration_times", "gaps"]
    exploded_df = exploded_df[cols]

    exploded_df["interval_number"] = exploded_df["interval_number"]+1
    exploded_df["iteration_number"] = exploded_df["interval_number"]*5
    exploded_df.to_csv("GSG_NFG_RMP_RANDOM_TARGET_VALUES_TO_MATRIX.csv")

def run_rm_gsg_nfg_prmp():
    schedule_form_kwargs = {
        "schedule_form": False,
        "simple": False,
        "attacker_penalty_factor": 5,
        "defender_penalty_factor": 5,
    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  42, 
        "defender_feature_value": 69, 
        "defender_step_cost": 32.5, 
    }

    gsg = GreenSecurityGame(df, coordinate_rectangle, "centroid", num_clusters=10, num_rows=7, num_columns=7)
    gsg.generate(num_attackers=1, 
                num_defenders=1, 
                home_base_assignments=[(kabo_djembe, bomassa, inner_post)], 
                num_timesteps=8, 
                generate_utility_matrix=True, 
                random_target_values=False,
                defense_time_threshold=1, 
                generate_actions=False, 
                force_return=False, 
                general_sum=False, 
                **schedule_form_kwargs)

    gaps_li = []
    interval_times_li = []
    for seed in seeds:
        print(f"starting seed {seed}")
        np.random.seed(seed)
        random_matrix = np.random.uniform(low=np.min(gsg.utility_matrix), high=np.max(gsg.utility_matrix), size=gsg.utility_matrix.shape)
        print("game done generating, running")
        D_d, U, gaps, interval_times = regret_matching(random_matrix, runtime=120, interval=5, iterations=10000, averaging=2, alternations=True, plus=True, predictive=True, verbose=True)
        gaps_li.append(gaps)
        interval_times_li.append(interval_times)

    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps": [8 for i in range(len(seeds))],
        "num_attackers": [1 for i in range(len(seeds))],
        "num_defenders": [1 for i in range(len(seeds))],
        "defense_time_threshold": [1 for i in range(len(seeds))],
        "force_return": [False for i in range(len(seeds))],
        "iteration_times": interval_times_li,
        "gaps": gaps_li
    })

    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_times", "gaps"], ignore_index=True)

    # Add a column for iteration number within each trial
    exploded_df["interval_number"] = exploded_df.groupby(["seed"]).cumcount()

    # Reorder
    cols = ["seed","num_timesteps", "num_attackers", "num_defenders", "defense_time_threshold", "force_return", "interval_number", "iteration_times", "gaps"]
    exploded_df = exploded_df[cols]

    exploded_df["interval_number"] = exploded_df["interval_number"]+1
    exploded_df["iteration_number"] = exploded_df["interval_number"]*5
    exploded_df.to_csv("GSG_NFG_PRMP_RANDOM_MATRIX.csv")


def run_random_gsg_sf_do():
    schedule_form_kwargs = {
        "schedule_form": True,
        "simple": False,
        "attacker_penalty_factor": 5,
        "defender_penalty_factor": 5,
    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_animal_value":  2350, 
        "defender_animal_value": 22966, 
        "defender_step_cost": 0, 
    }

    supports=[]
    iterations_to_converge = []
    defender_actions_sizes = []
    attacker_actions_sizes = []
    iteration_times_li = []
    gaps_li = []

    gsg = GreenSecurityGame(df, coordinate_rectangle, "centroid", num_clusters=10, num_rows=7, num_columns=7, escape_line_points=None)
    gsg.generate(num_attackers=1, 
                num_defenders=2, 
                home_base_assignments=[(kabo_djembe, bomassa, inner_post),(kabo_djembe, bomassa, inner_post)], 
                num_timesteps=8, 
                generate_utility_matrix=False, 
                defense_time_threshold=1, 
                force_return=True, 
                **schedule_form_kwargs,
                **general_sum_kwargs)
    for seed in seeds:
        print(f"starting seed: {seed}")
        np.random.seed(seed)
        sdict_copy = copy.deepcopy(gsg.schedule_form_dict)
        sdict_copy["target_utilities"] = generate_random_target_utility_matrix_like_v2(gsg.schedule_form_dict["target_utilities"], general_sum=False, respect_sign_roles=True)
        D_a, D_d, u, A_a, A_d, c, iteration_times, gaps = double_oracle_sf(sdict_copy,eps=1e-12, verbose=False)
        support = sum([1 for p in D_d if p!=0])
        supports.append(support)
        iterations_to_converge.append(c)
        defender_actions_sizes.append(len(A_d))
        attacker_actions_sizes.append(len(A_a))
        iteration_times_li.append(iteration_times)
        gaps_li.append(gaps)
    
    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps": [8 for i in range(len(seeds))],
        "num_defenders": [2 for i in range(len(seeds))],
        "num_clusters": [10 for i in range(len(seeds))],
        "dims": [7 for i in range(len(seeds))],
        "defense_time_threshold": [1 for i in range(len(seeds))],
        "def_support": supports,
        "iterations_to_converge": iterations_to_converge,
        "iteration_time": iteration_times_li,
        "gap": gaps_li,
        "defender_actions_size": defender_actions_sizes,
        "attacker_actions_size": attacker_actions_sizes
    })

    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_time", "gap"], ignore_index=True)

    # Add a column for iteration number within each grouped trial
    exploded_df["iteration_number"] = exploded_df.groupby(
        ["seed"]
    ).cumcount()

    # Reorder
    cols = ["seed",
        "num_timesteps", "num_defenders", "num_clusters", "dims", "defense_time_threshold",
        "iteration_number", "iteration_time", "gap", 
        "iterations_to_converge", "def_support", "defender_actions_size", "attacker_actions_size"
    ]
    exploded_df = exploded_df[cols]

    # Make iteration number 1-indexed
    exploded_df["iteration_number"] += 1
    exploded_df.to_csv("GSG_SF_DO_RANDOM_TARGET_VALUES.csv")

def run_rm_gsg_sf_rm():
    schedule_form_kwargs = {
        "schedule_form": True,
        "simple": False,
        "attacker_penalty_factor": 5,
        "defender_penalty_factor": 5,
    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  1, 
        "defender_feature_value": 1, 
        "defender_step_cost": 0, 
    }

    gaps_li = []
    interval_times_li = []

    gsg = GreenSecurityGame(df, coordinate_rectangle, "centroid", num_clusters=10, num_rows=7, num_columns=7)
    gsg.generate(num_attackers=1, 
                num_defenders=2, 
                home_base_assignments=[(kabo_djembe, bomassa, inner_post),(kabo_djembe, bomassa, inner_post)], 
                num_timesteps=8, 
                generate_utility_matrix=True, 
                random_target_values=False,
                defense_time_threshold=1, 
                generate_actions=False, 
                force_return=True, 
                general_sum=False, 
                **schedule_form_kwargs)

    for seed in seeds:
        print(f"starting seed {seed}")
        np.random.seed(seed)
        random_matrix = np.random.uniform(low=np.min(gsg.schedule_form_dict["defender_utility_matrix"]), high=np.max(gsg.schedule_form_dict["defender_utility_matrix"]), size=gsg.schedule_form_dict["defender_utility_matrix"].shape)
        print("game done generating, running")
        D_d, U, gaps, interval_times = regret_matching(random_matrix, runtime=120, interval=5, iterations=10000, averaging=0, alternations=False, plus=False, predictive=False, verbose=True)
        gaps_li.append(gaps)
        interval_times_li.append(interval_times)

    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps": [8 for i in range(len(seeds))],
        "num_attackers": [1 for i in range(len(seeds))],
        "num_defenders": [2 for i in range(len(seeds))],
        "defense_time_threshold": [1 for i in range(len(seeds))],
        "force_return": [True for i in range(len(seeds))],
        "iteration_times": interval_times_li,
        "gaps": gaps_li
    })

    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_times", "gaps"], ignore_index=True)

    # Add a column for iteration number within each trial
    exploded_df["interval_number"] = exploded_df.groupby(["seed"]).cumcount()

    # Reorder
    cols = ["seed","num_timesteps", "num_attackers", "num_defenders", "defense_time_threshold", "force_return", "interval_number", "iteration_times", "gaps"]
    exploded_df = exploded_df[cols]

    exploded_df["interval_number"] = exploded_df["interval_number"]+1
    exploded_df["iteration_number"] = exploded_df["interval_number"]*5
    exploded_df.to_csv("GSG_SF_RM_RANDOM_MATRIX.csv")

def run_rm_gsg_sf_rmp():
    schedule_form_kwargs = {
        "schedule_form": True,
        "simple": False,
        "attacker_penalty_factor": 5,
        "defender_penalty_factor": 5,
    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  1, 
        "defender_feature_value": 1, 
        "defender_step_cost": 0, 
    }

    gaps_li = []
    interval_times_li = []

    gsg = GreenSecurityGame(df, coordinate_rectangle, "centroid", num_clusters=10, num_rows=7, num_columns=7)
    gsg.generate(num_attackers=1, 
                num_defenders=2, 
                home_base_assignments=[(kabo_djembe, bomassa, inner_post),(kabo_djembe, bomassa, inner_post)], 
                num_timesteps=8, 
                generate_utility_matrix=True, 
                random_target_values=False,
                defense_time_threshold=1, 
                generate_actions=False, 
                force_return=True, 
                general_sum=False, 
                **schedule_form_kwargs)

    for seed in seeds:
        print(f"starting seed {seed}")
        np.random.seed(seed)
        random_matrix = np.random.uniform(low=np.min(gsg.schedule_form_dict["defender_utility_matrix"]), high=np.max(gsg.schedule_form_dict["defender_utility_matrix"]), size=gsg.schedule_form_dict["defender_utility_matrix"].shape)
        print("game done generating, running")
        D_d, U, gaps, interval_times = regret_matching(random_matrix, runtime=120, interval=5, iterations=10000, averaging=1, alternations=True, plus=True, predictive=False, verbose=True)
        gaps_li.append(gaps)
        interval_times_li.append(interval_times)

    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps": [8 for i in range(len(seeds))],
        "num_attackers": [1 for i in range(len(seeds))],
        "num_defenders": [2 for i in range(len(seeds))],
        "defense_time_threshold": [1 for i in range(len(seeds))],
        "force_return": [True for i in range(len(seeds))],
        "iteration_times": interval_times_li,
        "gaps": gaps_li
    })

    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_times", "gaps"], ignore_index=True)

    # Add a column for iteration number within each trial
    exploded_df["interval_number"] = exploded_df.groupby(["seed"]).cumcount()

    # Reorder
    cols = ["seed","num_timesteps", "num_attackers", "num_defenders", "defense_time_threshold", "force_return", "interval_number", "iteration_times", "gaps"]
    exploded_df = exploded_df[cols]

    exploded_df["interval_number"] = exploded_df["interval_number"]+1
    exploded_df["iteration_number"] = exploded_df["interval_number"]*5
    exploded_df.to_csv("GSG_SF_RMP_NEW_RANDOM_MATRIX.csv")

def run_rm_gsg_sf_prmp():
    schedule_form_kwargs = {
        "schedule_form": True,
        "simple": False,
        "attacker_penalty_factor": 5,
        "defender_penalty_factor": 5,
    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  1, 
        "defender_feature_value": 1, 
        "defender_step_cost": 0, 
    }

    gaps_li = []
    interval_times_li = []

    gsg = GreenSecurityGame(df, coordinate_rectangle, "centroid", num_clusters=10, num_rows=7, num_columns=7)
    gsg.generate(num_attackers=1, 
                num_defenders=2, 
                home_base_assignments=[(kabo_djembe, bomassa, inner_post),(kabo_djembe, bomassa, inner_post)], 
                num_timesteps=8, 
                generate_utility_matrix=True, 
                random_target_values=False,
                defense_time_threshold=1, 
                generate_actions=False, 
                force_return=True, 
                general_sum=False, 
                **schedule_form_kwargs)

    for seed in seeds:
        print(f"starting seed {seed}")
        np.random.seed(seed)
        random_matrix = np.random.uniform(low=np.min(gsg.schedule_form_dict["defender_utility_matrix"]), high=np.max(gsg.schedule_form_dict["defender_utility_matrix"]), size=gsg.schedule_form_dict["defender_utility_matrix"].shape)
        print("game done generating, running")
        D_d, U, gaps, interval_times = regret_matching(random_matrix, runtime=120, interval=5, iterations=10000, averaging=2, alternations=True, plus=True, predictive=True, verbose=True)
        gaps_li.append(gaps)
        interval_times_li.append(interval_times)

    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps": [8 for i in range(len(seeds))],
        "num_attackers": [1 for i in range(len(seeds))],
        "num_defenders": [2 for i in range(len(seeds))],
        "defense_time_threshold": [1 for i in range(len(seeds))],
        "force_return": [True for i in range(len(seeds))],
        "iteration_times": interval_times_li,
        "gaps": gaps_li
    })

    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_times", "gaps"], ignore_index=True)

    # Add a column for iteration number within each trial
    exploded_df["interval_number"] = exploded_df.groupby(["seed"]).cumcount()

    # Reorder
    cols = ["seed","num_timesteps", "num_attackers", "num_defenders", "defense_time_threshold", "force_return", "interval_number", "iteration_times", "gaps"]
    exploded_df = exploded_df[cols]

    exploded_df["interval_number"] = exploded_df["interval_number"]+1
    exploded_df["iteration_number"] = exploded_df["interval_number"]*5
    exploded_df.to_csv("GSG_SF_PRMP_RANDOM_MATRIX.csv")


def run_random_isg_nfg_do():
    schedule_form_kwargs = {
        "schedule_form": False,
        "simple": False,
        "attacker_penalty_factor": 3,
        "defender_penalty_factor": 3,
    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  1, 
        "defender_feature_value": 1, 
        "defender_step_cost": 0, 
    }

    num_timesteps_li = []
    num_attackers_li = []
    num_defenders_li = []
    supports=[]
    dims_li = []
    dts = []
    frs = []
    defender_actions_sizes = []
    attacker_actions_sizes = []
    iterations_to_converge = []
    iteration_times_li = []
    gaps_li = []

    for num_defenders in [1,2,3]:
        for seed in seeds:
            print(f"starting seed {seed}")
            isg = InfraSecurityGame(df_combined, ny_blocks_gdf, INFRA_WEIGHTS, bbox=bbox_downtown_large)
            isg.generate(num_attackers=1,
                        num_defenders=num_defenders,
                        home_base_assignments=[(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza) for i in range(num_defenders)],
                        num_timesteps=8,
                        generate_utility_matrix=False,
                        random_target_values=True,
                        generate_actions=False,
                        force_return=True,
                        defense_time_threshold=1,
                        **general_sum_kwargs,
                        **schedule_form_kwargs)
            D_a, D_d, u, A_a, A_d, c, iteration_times, gaps = double_oracle(isg,eps=1e-12, verbose=False)
            support = sum([1 for p in D_d if p!=0])
            supports.append(support)
            num_defenders_li.append(num_defenders)
            iterations_to_converge.append(c)
            defender_actions_sizes.append(len(A_d))
            attacker_actions_sizes.append(len(A_a))
            iteration_times_li.append(iteration_times)
            gaps_li.append(gaps)

    df = pd.DataFrame()
    seedli = seeds + seeds + seeds
    df["seed"] = seedli
    df["num_timesteps"] = [8 for i in range(len(seedli))]
    df["num_defenders"] = num_defenders_li
    df["num_attackers"] = [1 for i in range(len(seedli))]
    df["defense_time_threshold"] = [1 for i in range(len(seedli))]
    df["force_return"] = [True for i in range(len(seedli))]
    df["iterations_to_converge"] = iterations_to_converge
    df["defender_actions_size"] = defender_actions_sizes
    df["attacker_actions_size"] = attacker_actions_sizes
    df["iteration_time"] = iteration_times_li
    df["gap"] = gaps_li
    df["def_support"] = supports


    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_time", "gap"], ignore_index=True)

    # Add a column for iteration number within each grouped trial
    exploded_df["iteration_number"] = exploded_df.groupby(
        ["seed","num_defenders"]
    ).cumcount()

    # Reorder
    cols = [
        "seed", "num_timesteps","num_attackers", "num_defenders", "force_return", "defense_time_threshold",
        "iteration_number", "iteration_time", "gap", 
        "iterations_to_converge", "def_support", "defender_actions_size", "attacker_actions_size"
    ]
    exploded_df = exploded_df[cols]

    # Make iteration number 1-indexed
    exploded_df["iteration_number"] += 1
    exploded_df.to_csv("ISG_NFG_DO_RANDOM_TARGET_VALUES.csv")


def run_rm_isg_nfg_rm():
    schedule_form_kwargs = {
        "schedule_form": False,
        "simple": False,
        "attacker_penalty_factor": 3,
        "defender_penalty_factor": 3,
    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  1, 
        "defender_feature_value": 1, 
        "defender_step_cost": 0, 
    }

    isg = InfraSecurityGame(df_combined, ny_blocks_gdf, INFRA_WEIGHTS, bbox=bbox_downtown_large)
    isg.generate(num_attackers=1,
                num_defenders=1,
                home_base_assignments=[(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza)],
                num_timesteps=8,
                generate_utility_matrix=True,
                generate_actions=False,
                force_return=False,
                defense_time_threshold=1,
                **general_sum_kwargs,
                **schedule_form_kwargs)

    gaps_li = []
    interval_times_li = []
    for seed in seeds:
        print(f"starting seed {seed}")
        np.random.seed(seed)
        random_matrix = np.random.uniform(low=np.min(isg.utility_matrix), high=np.max(isg.utility_matrix), size=isg.utility_matrix.shape)
        print("game done generating, running")
        D_d, U, gaps, interval_times = regret_matching(random_matrix, runtime=120, interval=5, iterations=10000, averaging=0, alternations=False, plus=False, predictive=False, verbose=True)
        gaps_li.append(gaps)
        interval_times_li.append(interval_times)

    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps": [8 for i in range(len(seeds))],
        "num_attackers": [1 for i in range(len(seeds))],
        "num_defenders": [1 for i in range(len(seeds))],
        "defense_time_threshold": [1 for i in range(len(seeds))],
        "force_return": [False for i in range(len(seeds))],
        "iteration_times": interval_times_li,
        "gaps": gaps_li
    })

    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_times", "gaps"], ignore_index=True)

    # Add a column for iteration number within each trial
    exploded_df["interval_number"] = exploded_df.groupby(["seed"]).cumcount()

    # Reorder
    cols = ["seed","num_timesteps", "num_attackers", "num_defenders", "defense_time_threshold", "force_return", "interval_number", "iteration_times", "gaps"]
    exploded_df = exploded_df[cols]

    exploded_df["interval_number"] = exploded_df["interval_number"]+1
    exploded_df["iteration_number"] = exploded_df["interval_number"]*5
    exploded_df.to_csv("ISG_NFG_RM_RANDOM_MATRIX.csv")


def run_rm_isg_nfg_rmp():
    schedule_form_kwargs = {
        "schedule_form": False,
        "simple": False,
        "attacker_penalty_factor": 3,
        "defender_penalty_factor": 3,

    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  1, 
        "defender_feature_value": 1, 
        "defender_step_cost": 0, 
    }

    isg = InfraSecurityGame(df_combined, ny_blocks_gdf, INFRA_WEIGHTS, bbox=bbox_downtown_large)
    isg.generate(num_attackers=1,
                num_defenders=1,
                home_base_assignments=[(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza)],
                num_timesteps=8,
                generate_utility_matrix=True,
                generate_actions=False,
                force_return=False,
                defense_time_threshold=1,
                **general_sum_kwargs,
                **schedule_form_kwargs)

    gaps_li = []
    interval_times_li = []
    for seed in seeds:
        print(f"starting seed {seed}")
        np.random.seed(seed)
        random_matrix = np.random.uniform(low=np.min(isg.utility_matrix), high=np.max(isg.utility_matrix), size=isg.utility_matrix.shape)
        print("game done generating, running")
        D_d, U, gaps, interval_times = regret_matching(random_matrix, runtime=120, interval=5, iterations=10000, averaging=1, alternations=True, plus=True, predictive=False, verbose=True)
        gaps_li.append(gaps)
        interval_times_li.append(interval_times)
        df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps": [8 for i in range(len(seeds))],
        "num_attackers": [1 for i in range(len(seeds))],
        "num_defenders": [1 for i in range(len(seeds))],
        "defense_time_threshold": [1 for i in range(len(seeds))],
        "force_return": [False for i in range(len(seeds))],
        "iteration_times": interval_times_li,
        "gaps": gaps_li
    })

    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_times", "gaps"], ignore_index=True)

    # Add a column for iteration number within each trial
    exploded_df["interval_number"] = exploded_df.groupby(["seed"]).cumcount()

    # Reorder
    cols = ["seed","num_timesteps", "num_attackers", "num_defenders", "defense_time_threshold", "force_return", "interval_number", "iteration_times", "gaps"]
    exploded_df = exploded_df[cols]

    exploded_df["interval_number"] = exploded_df["interval_number"]+1
    exploded_df["iteration_number"] = exploded_df["interval_number"]*5
    exploded_df.to_csv("ISG_NFG_RMP_NEW_RANDOM_MATRIX.csv")

def run_rm_isg_nfg_prmp():
    schedule_form_kwargs = {
        "schedule_form": False,
        "simple": False,
        "attacker_penalty_factor": 3,
        "defender_penalty_factor": 3,

    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  1, 
        "defender_feature_value": 1, 
        "defender_step_cost": 0, 
    }

    isg = InfraSecurityGame(df_combined, ny_blocks_gdf, INFRA_WEIGHTS, bbox=bbox_downtown_large)
    isg.generate(num_attackers=1,
                num_defenders=1,
                home_base_assignments=[(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza)],
                num_timesteps=8,
                generate_utility_matrix=True,
                generate_actions=False,
                force_return=False,
                defense_time_threshold=1,
                **general_sum_kwargs,
                **schedule_form_kwargs)

    gaps_li = []
    interval_times_li = []
    for seed in seeds:
        print(f"starting seed {seed}")
        np.random.seed(seed)
        random_matrix = np.random.uniform(low=np.min(isg.utility_matrix), high=np.max(isg.utility_matrix), size=isg.utility_matrix.shape)
        print("game done generating, running")
        D_d, U, gaps, interval_times = regret_matching(random_matrix, runtime=120, interval=5, iterations=10000, averaging=2, alternations=True, plus=True, predictive=True, verbose=True)
        gaps_li.append(gaps)
        interval_times_li.append(interval_times)
    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps": [8 for i in range(len(seeds))],
        "num_attackers": [1 for i in range(len(seeds))],
        "num_defenders": [1 for i in range(len(seeds))],
        "defense_time_threshold": [1 for i in range(len(seeds))],
        "force_return": [False for i in range(len(seeds))],
        "iteration_times": interval_times_li,
        "gaps": gaps_li
    })

    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_times", "gaps"], ignore_index=True)

    # Add a column for iteration number within each trial
    exploded_df["interval_number"] = exploded_df.groupby(["seed"]).cumcount()

    # Reorder
    cols = ["seed","num_timesteps", "num_attackers", "num_defenders", "defense_time_threshold", "force_return", "interval_number", "iteration_times", "gaps"]
    exploded_df = exploded_df[cols]

    exploded_df["interval_number"] = exploded_df["interval_number"]+1
    exploded_df["iteration_number"] = exploded_df["interval_number"]*5
    exploded_df.to_csv("ISG_NFG_PRMP_RANDOM_MATRIX.csv")

def run_random_isg_sf_do():
    schedule_form_kwargs = {
        "schedule_form": True,
        "simple": False,
        "attacker_penalty_factor": 3,
        "defender_penalty_factor": 3,
    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  1, 
        "defender_feature_value": 1, 
        "defender_step_cost": 0, 
    }

    num_timesteps_li = []
    num_attackers_li = []
    num_defenders_li = []
    supports=[]
    dims_li = []
    dts = []
    frs = []
    defender_actions_sizes = []
    attacker_actions_sizes = []
    iterations_to_converge = []
    iteration_times_li = []
    gaps_li = []

    isg = InfraSecurityGame(df_combined, ny_blocks_gdf, INFRA_WEIGHTS, bbox=bbox_downtown_large)
    isg.generate(num_attackers=1,
                num_defenders=2,
                home_base_assignments=[(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza),(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza)],
                num_timesteps=8,
                generate_utility_matrix=False,
                random_target_values=False,
                force_return=True,
                defense_time_threshold=1,
                **general_sum_kwargs,
                **schedule_form_kwargs)

    for seed in seeds:
        print(f"starting seed {seed}")
        
        sdict_copy = copy.deepcopy(isg.schedule_form_dict)
        sdict_copy["target_utilities"] = generate_random_target_utility_matrix_like(isg.schedule_form_dict["target_utilities"], general_sum=False, respect_sign_roles=True)
        
        D_a, D_d, u, A_a, A_d, c, iteration_times, gaps = double_oracle_sf(sdict_copy,eps=1e-12, verbose=False)
        support = sum([1 for p in D_d if p!=0])
        supports.append(support)
        iterations_to_converge.append(c)
        defender_actions_sizes.append(len(A_d))
        attacker_actions_sizes.append(len(A_a))
        iteration_times_li.append(iteration_times)
        gaps_li.append(gaps)

    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps": [8 for i in range(len(seeds))],
        "num_defenders": [2 for i in range(len(seeds))],
        "defense_time_threshold": [1 for i in range(len(seeds))],
        "def_support": supports,
        "iterations_to_converge": iterations_to_converge,
        "iteration_time": iteration_times_li,
        "gap": gaps_li,
        "defender_actions_size": defender_actions_sizes,
        "attacker_actions_size": attacker_actions_sizes
    })

    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_time", "gap"], ignore_index=True)

    # Add a column for iteration number within each grouped trial
    exploded_df["iteration_number"] = exploded_df.groupby(
        ["seed"]
    ).cumcount()

    # Reorder
    cols = ["seed",
        "num_timesteps", "num_defenders", "defense_time_threshold",
        "iteration_number", "iteration_time", "gap", 
        "iterations_to_converge", "def_support", "defender_actions_size", "attacker_actions_size"
    ]
    exploded_df = exploded_df[cols]

    # Make iteration number 1-indexed
    exploded_df["iteration_number"] += 1
    exploded_df.to_csv("ISG_SF_DO_2DEF_RANDOM_TARGET_VALUES.csv")


def run_rm_isg_sf_rm():
    schedule_form_kwargs = {
        "schedule_form": True,
        "simple": False,
        "attacker_penalty_factor": 3,
        "defender_penalty_factor": 3,
    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  1, 
        "defender_feature_value": 1, 
        "defender_step_cost": 0, 
    }

    isg = InfraSecurityGame(df_combined, ny_blocks_gdf, INFRA_WEIGHTS, bbox=bbox_downtown_large)
    isg.generate(num_attackers=1,
                num_defenders=2,
                home_base_assignments=[(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza),(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza)],
                num_timesteps=8,
                generate_utility_matrix=True,
                generate_actions=False,
                force_return=True,
                defense_time_threshold=1,
                **general_sum_kwargs,
                **schedule_form_kwargs)

    gaps_li = []
    interval_times_li = []

    for seed in seeds:
        print(f"starting seed {seed}")
        np.random.seed(seed)
        random_matrix = np.random.uniform(low=np.min(isg.schedule_form_dict["defender_utility_matrix"]), high=np.max(isg.schedule_form_dict["defender_utility_matrix"]), size=isg.schedule_form_dict["defender_utility_matrix"].shape)
        print("game done generating, running")
        D_d, U, gaps, interval_times = regret_matching(random_matrix, runtime=120, interval=5, iterations=10000, averaging=0, alternations=False, plus=False, predictive=False, verbose=True)
        gaps_li.append(gaps)
        interval_times_li.append(interval_times)
    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps": [8 for i in range(len(seeds))],
        "num_attackers": [1 for i in range(len(seeds))],
        "num_defenders": [2 for i in range(len(seeds))],
        "defense_time_threshold": [1 for i in range(len(seeds))],
        "force_return": [True for i in range(len(seeds))],
        "iteration_times": interval_times_li,
        "gaps": gaps_li
    })

    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_times", "gaps"], ignore_index=True)

    # Add a column for iteration number within each trial
    exploded_df["interval_number"] = exploded_df.groupby(["seed"]).cumcount()

    # Reorder
    cols = ["seed","num_timesteps", "num_attackers", "num_defenders", "defense_time_threshold", "force_return", "interval_number", "iteration_times", "gaps"]
    exploded_df = exploded_df[cols]

    exploded_df["interval_number"] = exploded_df["interval_number"]+1
    exploded_df["iteration_number"] = exploded_df["interval_number"]*5
    exploded_df.to_csv("ISG_SF_RM_RANDOM_MATRIX.csv")

def run_rm_isg_sf_rmp():
    schedule_form_kwargs = {
        "schedule_form": True,
        "simple": False,
        "attacker_penalty_factor": 3,
        "defender_penalty_factor": 3,
    }

    general_sum_kwargs = {
        "general_sum": False,
        "attacker_feature_value":  42, 
        "defender_feature_value": 69, 
        "defender_step_cost": 32.5, 
    }

    isg = InfraSecurityGame(df_combined, ny_blocks_gdf, INFRA_WEIGHTS, bbox=bbox_downtown_large)
    isg.generate(num_attackers=1,
                num_defenders=2,
                home_base_assignments=[(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza),(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza)],
                num_timesteps=8,
                generate_utility_matrix=True,
                generate_actions=False,
                force_return=True,
                defense_time_threshold=1,
                **general_sum_kwargs,
                **schedule_form_kwargs)

    gaps_li = []
    interval_times_li = []

    for seed in seeds:
        print(f"starting seed {seed}")
        np.random.seed(seed)
        random_matrix = np.random.uniform(low=np.min(isg.schedule_form_dict["defender_utility_matrix"]), high=np.max(isg.schedule_form_dict["defender_utility_matrix"]), size=isg.schedule_form_dict["defender_utility_matrix"].shape)
        print(random_matrix.shape)
        print("game done generating, running")
        D_d, U, gaps, interval_times = regret_matching(random_matrix, runtime=120, interval=5, iterations=10000, averaging=1, alternations=True, plus=True, predictive=False, verbose=True)
        gaps_li.append(gaps)
        interval_times_li.append(interval_times)
    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps": [8 for i in range(len(seeds))],
        "num_attackers": [1 for i in range(len(seeds))],
        "num_defenders": [2 for i in range(len(seeds))],
        "defense_time_threshold": [1 for i in range(len(seeds))],
        "force_return": [True for i in range(len(seeds))],
        "iteration_times": interval_times_li,
        "gaps": gaps_li
    })

    # Step 2: Explode list columns
    exploded_df = df.explode(["iteration_times", "gaps"], ignore_index=True)

    # Add a column for iteration number within each trial
    exploded_df["interval_number"] = exploded_df.groupby(["seed"]).cumcount()

    # Reorder
    cols = ["seed","num_timesteps", "num_attackers", "num_defenders", "defense_time_threshold", "force_return", "interval_number", "iteration_times", "gaps"]
    exploded_df = exploded_df[cols]

    exploded_df["interval_number"] = exploded_df["interval_number"]+1
    exploded_df["iteration_number"] = exploded_df["interval_number"]*5
    exploded_df.to_csv("ISG_SF_NEW_RMP_RANDOM_MATRIX.csv")

def run_rm_isg_sf_prmp():
schedule_form_kwargs = {
    "schedule_form": True,
    "simple": False,
    "attacker_penalty_factor": 3,
    "defender_penalty_factor": 3,
}

general_sum_kwargs = {
    "general_sum": False,
    "attacker_feature_value":  42, 
    "defender_feature_value": 69, 
    "defender_step_cost": 32.5, 
}

isg = InfraSecurityGame(df_combined, ny_blocks_gdf, INFRA_WEIGHTS, bbox=bbox_downtown_large)
isg.generate(num_attackers=1,
             num_defenders=2,
             home_base_assignments=[(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza),(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza)],
             num_timesteps=8,
             generate_utility_matrix=True,
             generate_actions=False,
             force_return=True,
             defense_time_threshold=1,
             **general_sum_kwargs,
             **schedule_form_kwargs)

gaps_li = []
interval_times_li = []

for seed in seeds:
    print(f"starting seed {seed}")
    np.random.seed(seed)
    random_matrix = np.random.uniform(low=np.min(isg.schedule_form_dict["defender_utility_matrix"]), high=np.max(isg.schedule_form_dict["defender_utility_matrix"]), size=isg.schedule_form_dict["defender_utility_matrix"].shape)
    print("game done generating, running")
    D_d, U, gaps, interval_times = regret_matching(random_matrix, runtime=120, interval=5, iterations=10000, averaging=2, alternations=True, plus=True, predictive=True, verbose=True)
    gaps_li.append(gaps)
    interval_times_li.append(interval_times)
df = pd.DataFrame({
    "seed":seeds,
    "num_timesteps": [8 for i in range(len(seeds))],
    "num_attackers": [1 for i in range(len(seeds))],
    "num_defenders": [2 for i in range(len(seeds))],
    "defense_time_threshold": [1 for i in range(len(seeds))],
    "force_return": [True for i in range(len(seeds))],
    "iteration_times": interval_times_li,
    "gaps": gaps_li
})

# Step 2: Explode list columns
exploded_df = df.explode(["iteration_times", "gaps"], ignore_index=True)

# Add a column for iteration number within each trial
exploded_df["interval_number"] = exploded_df.groupby(["seed"]).cumcount()

# Reorder
cols = ["seed","num_timesteps", "num_attackers", "num_defenders", "defense_time_threshold", "force_return", "interval_number", "iteration_times", "gaps"]
exploded_df = exploded_df[cols]

exploded_df["interval_number"] = exploded_df["interval_number"]+1
exploded_df["iteration_number"] = exploded_df["interval_number"]*5
exploded_df.to_csv("ISG_SF_PRMP_RANDOM_MATRIX.csv")