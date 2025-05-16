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
import time
from matplotlib import pyplot as plt
from collections import defaultdict, Counter

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

seeds = [1,2,3,4,5,6,7,8,9,10]

def run_nfg_sparsity_exp(t):
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
                num_timesteps=10, 
                generate_utility_matrix=True, 
                defense_time_threshold=2, 
                generate_actions=False, 
                force_return=False, 
                general_sum=False, 
                **schedule_form_kwargs)

    start = time.time()
    nD_a, nD_d, nu = nash(gsg.utility_matrix)
    end = time.time()
    nruntime = end-start
    nsupport = sum([1 for p in nD_d if p!=0])

    mip_us = []
    mip_supports = []
    mip_runtimes = []

    for i in range(1,10):
        start = time.time()
        print(f"starting i={i} at time {start}")
        mu, mD_d = mip(gsg.utility_matrix,i)
        end = time.time()
        print(f"finished i={i} in {end-start} seconds with u={mu}")
        msupport = sum([1 for p in mD_d if p!=0])
        mip_us.append(mu)
        mip_supports.append(i)
        mip_runtimes.append(end-start)
        if abs(mu-nu) <= 1e-8:
            break

    df = pd.DataFrame({
        "num_timesteps":[t for i in range(len(mip_supports))],
        "num_attackers":[1 for i in range(len(mip_supports))],
        "num_defenders":[1 for i in range(len(mip_supports))],
        "num_clusters":[10 for i in range(len(mip_supports))],
        "dims":[7 for i in range(len(mip_supports))],
        "defense_time_threshold":[1 for i in range(len(mip_supports))],
        "force_return":[False for i in range(len(mip_supports))],
        "num_defender_actions": [len(gsg.defender_actions) for i in range(len(mip_supports))],
        "nash_value":[nu for i in range(len(mip_supports))],
        "nash_support":[nsupport for i in range(len(mip_supports))],
        "nash_runtime":[nruntime for i in range(len(mip_supports))],
        "mip_value":mip_us,
        "mip_support":mip_supports,
        "mip_runtime":mip_runtimes,
    })
    df.to_csv(f"GSG_NFG_T{t}_1A_SPARSITY.csv")

def run_random_nfg_sparsity_exp(t):
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

    boulou_camp = (2.2,15.9)
    # lobeke_camp = (2.25,15.75)
    kabo_djembe = (2.0532352380408088, 16.085709866529694)
    bomassa = (2.2037280296158355, 16.187056364164913)
    inner_post = (2.2,15.98)

    gsg = GreenSecurityGame(df, coordinate_rectangle, "centroid", num_clusters=10, num_rows=7, num_columns=7)
    gsg.generate(num_attackers=1, 
                num_defenders=1, 
                home_base_assignments=[(kabo_djembe, bomassa, inner_post)], 
                num_timesteps=9, 
                generate_utility_matrix=True, 
                defense_time_threshold=1, 
                generate_actions=False, 
                force_return=False, 
                general_sum=False, 
                **schedule_form_kwargs)

    nruntimes = []
    nsupports = []
    nus = []
    mip_us_per_seed = []
    mip_supports_per_seed = []
    mip_runtimes_per_seed = []

    for seed in seeds:
        print(f"starting seed {seed}")
        np.random.seed(seed)
        random_matrix = generate_random_utility_matrix_like(gsg.utility_matrix)
        start = time.time()
        nD_a, nD_d, nu = nash(random_matrix)
        end = time.time()
        nruntime = end-start
        nsupport = sum([1 for p in nD_d if p!=0])
        nus.append(nu)
        print(nsupport,nu)
        nruntimes.append(nruntime)
        nsupports.append(nsupport)
        
        mip_us = []
        mip_supports = []
        mip_runtimes = []
        
        for i in range(1,60):
            start = time.time()
            print(f"starting i={i} at time {start}")
            mu, mD_d = mip(random_matrix,i)
            end = time.time()
            print(f"finished i={i} in {end-start} seconds with u={mu}")
            msupport = sum([1 for p in mD_d if p!=0])
            mip_us.append(mu)
            mip_supports.append(i)
            mip_runtimes.append(end-start)
            if abs(mu-nu) <= 1e-12:
                mip_us_per_seed.append(mip_us)
                mip_supports_per_seed.append(mip_supports)
                mip_runtimes_per_seed.append(mip_runtimes)
                break
    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps":[t for i in range(len(seeds))],
        "num_attackers":[1 for i in range(len(seeds))],
        "num_defenders":[1 for i in range(len(seeds))],
        "num_clusters":[10 for i in range(len(seeds))],
        "dims":[7 for i in range(len(seeds))],
        "defense_time_threshold":[1 for i in range(len(seeds))],
        "force_return":[False for i in range(len(seeds))],
        "num_defender_actions": [len(gsg.defender_actions) for i in range(len(seeds))],
        "nash_value":nus,
        "nash_support":nsupports,
        "nash_runtime":nruntimes,
        "mip_value":mip_us_per_seed,
        "mip_support":mip_supports_per_seed,
        "mip_runtime":mip_runtimes_per_seed,
    })
    df.to_csv(f"GSG_NFG_T{t}_1D1A_SPARSITY_RANDOM_MATRIX.csv")

def run_sfg_sparsity_exp(t):
    schedule_form_kwargs = {
        "schedule_form": True,
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
                num_defenders=2, 
                home_base_assignments=[(kabo_djembe, bomassa, inner_post),(kabo_djembe, bomassa, inner_post)], 
                num_timesteps=8, 
                generate_utility_matrix=True, 
                defense_time_threshold=2, 
                generate_actions=False, 
                force_return=True, 
                general_sum=False, 
                **schedule_form_kwargs)

    start = time.time()
    nD_a, nD_d, nu = nash(gsg.schedule_form_dict["defender_utility_matrix"])
    end = time.time()
    nruntime = end-start
    nsupport = sum([1 for p in nD_d if p!=0])

    mip_us = []
    mip_supports = []
    mip_runtimes = []

    for i in range(1,10):
        start = time.time()
        print(f"starting i={i} at time {start}")
        mu, mD_d = mip(gsg.schedule_form_dict["defender_utility_matrix"],i)
        end = time.time()
        print(f"finished i={i} in {end-start} seconds with u={mu}")
        msupport = sum([1 for p in mD_d if p!=0])
        mip_us.append(mu)
        mip_supports.append(i)
        mip_runtimes.append(end-start)
        if abs(mu-nu) <= 1e-12:
            break
    df = pd.DataFrame({
        "num_timesteps":[t for i in range(len(mip_supports))],
        "num_attackers":[1 for i in range(len(mip_supports))],
        "num_defenders":[2 for i in range(len(mip_supports))],
        "num_clusters":[10 for i in range(len(mip_supports))],
        "dims":[7 for i in range(len(mip_supports))],
        "defense_time_threshold":[2 for i in range(len(mip_supports))],
        "force_return":[True for i in range(len(mip_supports))],
        "num_defender_actions": [gsg.schedule_form_dict["defender_utility_matrix"].shape[0] for i in range(len(mip_supports))],
        "nash_value":[nu for i in range(len(mip_supports))],
        "nash_support":[nsupport for i in range(len(mip_supports))],
        "nash_runtime":[nruntime for i in range(len(mip_supports))],
        "mip_value":mip_us,
        "mip_support":mip_supports,
        "mip_runtime":mip_runtimes,
    })

    df.to_csv(f"GSG_SF_T{t}_SPARSITY_FINAL.csv")


def run_random_sfg_sparsity_exp(t):
    schedule_form_kwargs = {
        "schedule_form": True,
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
                num_defenders=2, 
                home_base_assignments=[(kabo_djembe, bomassa, inner_post),(kabo_djembe, bomassa, inner_post)], 
                num_timesteps=10, 
                generate_utility_matrix=True, 
                defense_time_threshold=2, 
                generate_actions=False, 
                force_return=False, 
                general_sum=False, 
                **schedule_form_kwargs)

    start = time.time()
    nD_a, nD_d, nu = nash(gsg.schedule_form_dict["defender_utility_matrix"])
    end = time.time()
    nruntime = end-start
    nsupport = sum([1 for p in nD_d if p!=0])


    nruntimes = []
    nsupports = []
    nus = []
    mip_us_per_seed = []
    mip_supports_per_seed = []
    mip_runtimes_per_seed = []

    for seed in seeds:
        print(f"starting seed {seed}")
        np.random.seed(seed)
        random_matrix = generate_random_utility_matrix_like(gsg.schedule_form_dict["defender_utility_matrix"])
        print(random_matrix.shape)
        start = time.time()
        nD_a, nD_d, nu = nash(random_matrix)
        end = time.time()
        nruntime = end-start
        nsupport = sum([1 for p in nD_d if p!=0])
        nus.append(nu)
        print(nsupport,nu)
        nruntimes.append(nruntime)
        nsupports.append(nsupport)
        
        mip_us = []
        mip_supports = []
        mip_runtimes = []
        
        for i in range(1,60):
            start = time.time()
            print(f"starting i={i} at time {start}")
            mu, mD_d = mip(random_matrix,i)
            end = time.time()
            print(f"finished i={i} in {end-start} seconds with u={mu}")
            msupport = sum([1 for p in mD_d if p!=0])
            mip_us.append(mu)
            mip_supports.append(i)
            mip_runtimes.append(end-start)
            if abs(mu-nu) <= 1e-12:
                mip_us_per_seed.append(mip_us)
                mip_supports_per_seed.append(mip_supports)
                mip_runtimes_per_seed.append(mip_runtimes)
                break

    df = pd.DataFrame({
        "seed":seeds,
        "num_timesteps":[t for i in range(len(seeds))],
        "num_attackers":[1 for i in range(len(seeds))],
        "num_defenders":[2 for i in range(len(seeds))],
        "num_clusters":[10 for i in range(len(seeds))],
        "dims":[7 for i in range(len(seeds))],
        "defense_time_threshold":[2 for i in range(len(seeds))],
        "force_return":[False for i in range(len(seeds))],
        "num_defender_actions": [len(gsg.schedule_form_dict["defender_actions"]) for i in range(len(seeds))],
        "nash_value":nus,
        "nash_support":nsupports,
        "nash_runtime":nruntimes,
        "mip_value":mip_us_per_seed,
        "mip_support":mip_supports_per_seed,
        "mip_runtime":mip_runtimes_per_seed,
    })

    df.to_csv(f"GSG_SF_T{t}_SPARSITY_RANDOM_MATRIX.csv")