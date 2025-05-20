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

from utils.random_utils import generate_random_utility_matrix_like, generate_random_target_utility_matrix_like, generate_random_schedule_mapping_like, generate_random_target_utility_matrix_like_v2

import time
from matplotlib import pyplot as plt
from collections import defaultdict, Counter
import copy
from scipy import stats

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

seeds = [1,2,3,4,5,6,7,8,9,10]

# Preprocess GSG Data
df = pd.read_csv("lobeke.csv")
df.dropna(inplace=True)

# Lobeke National Park Bounding Box
# lat_min, lon_min = 2.05522, 15.8790
# lat_max, lon_max = 2.2837, 16.2038

lat_min, lon_min = 2.0530, 15.8790
lat_max, lon_max = 2.2837, 16.2038

coordinate_rectangle = [lat_min, lat_max, lon_min, lon_max]


boulou_camp = (2.2,15.9)
# lobeke_camp = (2.25,15.75)
kabo_djembe = (2.0532352380408088, 16.085709866529694)
bomassa = (2.2037280296158355, 16.187056364164913)
inner_post = (2.2,15.98)
sangha_river = [(2.2837, 16.16283352464626),(2.053, 16.066212728001727)]


gdf = gpd.read_file("chinatown_infra.geojson")

# Preprocess ISG Data
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

def run_gsg_simple_sse_lp():
    schedule_form_kwargs = {
        "schedule_form": True,
        "simple": True,
        "attacker_penalty_factor": 5,
        "defender_penalty_factor": 5,
    }

    general_sum_kwargs = {
        "general_sum": True,
        "attacker_animal_value":  2350, 
        "defender_animal_value": 22966, 
        "defender_step_cost": 0, 
    }


    num_timesteps_li = []
    num_defenders_li = []
    num_clusters_li = []
    dims_li = []
    dts = []
    supports=[]

    dus = []
    runtimes = []
    i=0

    for num_timesteps in [7,8,9,10,11]:
        for num_defenders in [1,2,3]:
            for num_clusters in [7,8,9,10,11,12]:
                for dims in [7,8,9,10]:
                    for dt in [1,2]:
                        gsg = GreenSecurityGame(df, coordinate_rectangle, "centroid", num_clusters=num_clusters, num_rows=dims, num_columns=dims, escape_line_points=sangha_river)
                        gsg.generate(num_attackers=1, 
                                    num_defenders=num_defenders, 
                                    home_base_assignments=[(kabo_djembe, bomassa, inner_post) for i in range(num_defenders)], 
                                    num_timesteps=num_timesteps, 
                                    generate_utility_matrix=False, 
                                    defense_time_threshold=dt, 
                                    force_return=True, 
                                    **schedule_form_kwargs,
                                    **general_sum_kwargs)
                        start = time.time()
                        _, coverage, du = solve_sse_lp(gsg.schedule_form_dict)
                        end=time.time()
                        num_timesteps_li.append(num_timesteps)
                        num_defenders_li.append(num_defenders)
                        num_clusters_li.append(num_clusters)
                        dims_li.append(dims)
                        dts.append(dt)
                        
                        support = sum([1 for t,p in coverage.items() if p!=0])
                        dus.append(du)
                        runtimes.append(end-start)
                        supports.append(support)
                        print(f"{i}/719: timesteps:{num_timesteps}, num_defenders:{num_defenders}, num_clusters:{num_clusters}, dims:{dims}, dt:{dt}, support:{support}, du:{du}")
                        i+=1
    df = pd.DataFrame()
    df["num_timesteps"] = num_timesteps_li
    df["num_defenders"] = num_defenders_li
    df["num_clusters"] = num_clusters_li
    df["dims"] = dims_li
    df["defense_time_threshold"] = dts
    df["defender_utility"] = dus
    df["support"] = supports
    df["runtime"] = runtimes
    df.to_csv("GSG_SIMPLE_SSE_v2.csv")


def run_gsg_general_sse_lp():
    schedule_form_kwargs = {
        "schedule_form": True,
        "simple": False,
        "attacker_penalty_factor": 5,
        "defender_penalty_factor": 5
    }

    general_sum_kwargs = {
        "general_sum": True,
        "attacker_animal_value":  2350, 
        "defender_animal_value": 22966, 
        "defender_step_cost": 1.17, 
    }


    num_timesteps_li = []
    num_defenders_li = []
    num_clusters_li = []
    dims_li = []
    dts = []
    supports=[]

    dus = []
    runtimes = []
    i=0

    for num_timesteps in [7,8,9,10]:
        for num_defenders in [1,2,3]:
            for num_clusters in [7,8,9,10]:
                for dims in [7,8,9,10]:
                    for dt in [1,2]:
                        gsg = GreenSecurityGame(df, coordinate_rectangle, "centroid", num_clusters=num_clusters, num_rows=dims, num_columns=dims, escape_line_points=sangha_river)
                        gsg.generate(num_attackers=1, 
                                    num_defenders=num_defenders, 
                                    home_base_assignments=[(kabo_djembe, bomassa, inner_post) for i in range(num_defenders)], 
                                    num_timesteps=num_timesteps, 
                                    generate_utility_matrix=True, 
                                    defense_time_threshold=dt, 
                                    force_return=True,
                                    **schedule_form_kwargs,
                                    **general_sum_kwargs)
                        
                        start = time.time()
                        _, coverage, du = solve_general_sum_normal_form(gsg.schedule_form_dict["defender_utility_matrix"], gsg.schedule_form_dict["attacker_utility_matrix"])
                        end=time.time()
                        num_timesteps_li.append(num_timesteps)
                        num_defenders_li.append(num_defenders)
                        num_clusters_li.append(num_clusters)
                        dims_li.append(dims)
                        dts.append(dt)
                        
                        support = sum([1 for t,p in coverage.items() if p!=0])
                        dus.append(du)
                        runtimes.append(end-start)
                        supports.append(support)
                        print(f"{i}/383: timesteps:{num_timesteps}, num_defenders:{num_defenders}, num_clusters:{num_clusters}, dims:{dims}, dt:{dt}, support:{support}, du:{du}")
                        i+=1
    df = pd.DataFrame()
    df["num_timesteps"] = num_timesteps_li
    df["num_defenders"] = num_defenders_li
    df["num_clusters"] = num_clusters_li
    df["dims"] = dims_li
    df["defense_time_threshold"] = dts
    df["defender_utility"] = dus
    df["support"] = supports
    df["runtime"] = runtimes
    df.to_csv("GSG_NFG_SSE_v2.csv")


def run_isg_simple_sse_lp():
    schedule_form_kwargs = {
        "schedule_form": True,
        "simple": True,
        "attacker_penalty_factor": 3,
        "defender_penalty_factor": 3
    }

    general_sum_kwargs = {
        "general_sum": True,
        "attacker_feature_value":  1, 
        "defender_feature_value": 100, 
        "defender_step_cost": 0,
        "alpha":.5
    }
    num_timesteps_li = []
    num_defenders_li = []
    supports=[]
    dts = []

    dus = []
    runtimes = []
    i=0

    for num_timesteps in [7,8,9,10]:
        for num_defenders in [1,2,3]:
            for dt in [1,2]:
                isg = InfraSecurityGame(df_combined, ny_blocks_gdf, INFRA_WEIGHTS, bbox=bbox_downtown_large)
                isg.generate(num_attackers=1, 
                            num_defenders=num_defenders, 
                            home_base_assignments=[(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza) for i in range(num_defenders)], 
                            num_timesteps=num_timesteps, 
                            generate_utility_matrix=False, 
                            generate_actions=False, 
                            force_return=True, 
                            defense_time_threshold=dt, 
                            **schedule_form_kwargs,
                            **general_sum_kwargs, 
                            )
                start = time.time()
                _, coverage, du = solve_sse_lp(isg.schedule_form_dict)
                end=time.time()
                support = sum([1 for t,p in coverage.items() if p!=0])
                supports.append(support)
                num_timesteps_li.append(num_timesteps)
                num_defenders_li.append(num_defenders)
                dts.append(dt)
                dus.append(du)
                runtimes.append(end-start)
                print(f"{i}/23: timesteps:{num_timesteps}, num_defenders:{num_defenders}, dt:{dt}, support:{support}, du:{du}")
                i+=1
    df = pd.DataFrame()
    df["num_timesteps"] = num_timesteps_li
    df["num_defenders"] = num_defenders_li
    df["defense_time_threshold"] = dts
    df["defender_utility"] = dus
    df["support"] = supports
    df["runtime"] = runtimes
    df.to_csv("ISG_SIMPLE_SSE.csv")

def run_isg_general_sse_lp():
    schedule_form_kwargs = {
        "schedule_form": True,
        "simple": False,
        "attacker_penalty_factor": 3,
        "defender_penalty_factor": 3
    }

    general_sum_kwargs = {
        "general_sum": True,
        "attacker_feature_value":  1, 
        "defender_feature_value": 100, 
        "defender_step_cost": 1, 
        "alpha":.5
    }

    num_timesteps_li = []
    num_defenders_li = []
    supports=[]
    dts = []

    dus = []
    runtimes = []
    i=0

    for num_timesteps in [7,8,9,10]:
        for num_defenders in [1,2,3]:
            for dt in [1,2]:
                isg = InfraSecurityGame(df_combined, ny_blocks_gdf, INFRA_WEIGHTS, bbox=bbox_downtown_large, escape_point=None)
                isg.generate(num_attackers=1, 
                            num_defenders=num_defenders, 
                            home_base_assignments=[(fifth_precinct,booking_station, troop_nyc, first_precinct, police_plaza) for i in range(num_defenders)], 
                            num_timesteps=num_timesteps, 
                            generate_utility_matrix=True, 
                            generate_actions=False, 
                            force_return=True, 
                            defense_time_threshold=dt, 
                            **general_sum_kwargs, 
                            **schedule_form_kwargs)
                start = time.time()
                _, coverage, du = solve_general_sum_normal_form(isg.schedule_form_dict["defender_utility_matrix"], isg.schedule_form_dict["attacker_utility_matrix"])
                end=time.time()
                support = sum([1 for t,p in coverage.items() if p!=0])
                supports.append(support)
                num_timesteps_li.append(num_timesteps)
                num_defenders_li.append(num_defenders)
                dts.append(dt)
                dus.append(du)
                runtimes.append(end-start)
                print(f"{i}/23: timesteps:{num_timesteps}, num_defenders:{num_defenders}, dt:{dt}, support:{support}, du:{du}")
                i+=1
    df = pd.DataFrame()
    df["num_timesteps"] = num_timesteps_li
    df["num_defenders"] = num_defenders_li
    df["defense_time_threshold"] = dts
    df["defender_utility"] = dus
    df["support"] = supports
    df["runtime"] = runtimes

    df.to_csv("ISG_NFG_SSE.csv")