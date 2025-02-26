# Security Games

## Example 1: Green Security Game

### Introduction
The Security Game framework provides a structured way to model adversarial interactions over a graph. The `SecurityGame` class serves as the parent for all security games, fixing certain constraints such as all attackers being stationary and defenders moving, a designated home base, and structured strategy generation. This example walks through a `GreenSecurityGame` implementation using real-world animal movement data to model patrol strategies for protecting wildlife.

---

## Load and Inspect Animal Location Data
We begin by loading a dataset of elephant movements within Lobeke National Park.

```python
import pandas as pd
# Data collected from publio domain study: Elephant Research - Kalamaloue National Park (Cameroon)
df = pd.read_csv("lobeke.csv")
df.dropna(inplace=True)
df
```

### Data Preview
| animal_id | lat   | long  | timestamp               |
|-----------|-------|-------|-------------------------|
| 14118     | 2.476 | 15.769 | 2002-03-30 00:00:00.000 |
| 14118     | 2.502 | 15.740 | 2002-03-31 00:00:00.000 |
| 14118     | 2.506 | 15.744 | 2002-03-31 00:00:00.000 |
| 14118     | 2.507 | 15.747 | 2002-03-31 00:00:00.000 |
| 14118     | 2.479 | 15.771 | 2002-04-01 00:00:00.000 |

This dataset contains recorded locations of elephants over time. Each row represents a unique observation with an `animal_id`, latitude (`lat`), longitude (`long`), and timestamp.

---

## Define the Geographic Boundaries
We define a bounding box for Lobeke National Park to filter the dataset to a more manageable sub-area.

```python
# Lobeke National Park Bounding Box
lat_min, lon_min = 2.05522, 15.8790
lat_max, lon_max = 2.2837, 16.2038

coordinate_rectangle = [lat_min, lat_max, lon_min, lon_max]
```

This bounding box will be used for visualization and game setup.

---

## Visualizing Animal Locations
We can now visualize the spatial distribution of the recorded animal locations within the park.

```python
from utils.visualization import plot_animal_locations

plot_animal_locations(df, coordinate_rectangle)
```

![image](https://github.com/user-attachments/assets/1fde2a46-7220-495c-9911-58fe19c14410)


This animal density distribution will inform target creation. As a result, the generated game more accurately reflects the real world data.

---

## Initialize and Generate the Green Security Game
Now, we create an instance of `GreenSecurityGame`, setting up the security game over a grid representation of the park.

```python
from security_game.green_security_game import GreenSecurityGame

gsg = GreenSecurityGame(df, coordinate_rectangle, "centroid", num_clusters=5, num_rows=5, num_columns=5)
gsg.generate(num_attackers=2, num_defenders=1, home_base=(3,3), num_timesteps=8)
```

### Breakdown:
- `df`: Animal location data
- `coordinate_rectangle`: Bounding box for spatial reference
- `centroid`: Method for determining target locations
- `num_clusters=5`: Clusters data to 5 centroids with scores assigned to corresponding grid cells
- `num_rows=5, num_columns=5`: Defines a 5x5 grid representation of the park
- `num_attackers=2`: Two simulated poachers (attackers)
- `num_defenders=1`: One ranger unit (defender)
- `home_base=(3,3)`: Defenders must start and end at a home base node
- `num_timesteps=8`: Simulation runs for 8 time steps

---

## Visualizing the Game Graph
```python
gsg.draw_graph()
```

![image](https://github.com/user-attachments/assets/73770d37-599d-4718-a2e3-39d1d3efbeb2)

This graph represents the game structure, with nodes for different cells of the grid and connections representing movement lateral cell to cell movement options.


## Exploring Strategy Space
We can now inspect the strategies available to both players.

```python
len(gsg.defender_strategies)
```
**Output:**
```python
3571
```
This indicates that the defender has 3,571 possible patrol strategies.

```python
len(gsg.attacker_strategies)
```
**Output:**
```python
36
```
Attackers (poachers) have 36 potential movement strategies.


## Inspecting the Utility Matrix
The utility matrix captures game outcomes based on all possible attacker and defender strategy combinations.

```python
gsg.utility_matrix
```
**Output:**
```python
array([[-1.        , -1.27158774, -1.20612813, ..., -0.29387187,
        -0.44428969,  0.        ],
       [-1.        , -1.27158774, -1.20612813, ..., -0.29387187,
        -0.44428969,  0.        ],
       [-1.        , -1.27158774, -1.20612813, ..., -0.29387187,
        -0.44428969,  0.        ],
       ...,
       [-1.        , -1.27158774, -1.20612813, ..., -0.29387187,
        -0.44428969,  0.        ],
       [-1.        , -1.27158774, -1.20612813, ..., -0.29387187,
        -0.44428969,  0.        ],
       [-1.        , -1.27158774, -1.20612813, ..., -0.29387187,
        -0.44428969,  0.        ]])
```
Each row corresponds to a defender strategy, and each column represents an attacker strategy. The values indicate the utility of the game where a given attacker strategy and defender strategy are executed.


# Example 2: Infrastructure Security Game

## Overview
The infrastructure security game models a scenario where attackers target power features (elements of the power grid) to disrupt electricity supply, impacting residents based on population density and type of power feature. The game uses real-world data to determine realistic target values.

## Data Collection

### Querying OpenStreetMap for Power Features
We use an Overpass Turbo query to extract power-related infrastructure data for the NYC metro area:

```sql
[out:json][timeout:25];
(
  // Nodes, ways, and relations with the "power" key in Manhattan
  node["power"](40.4774,-74.2591,40.9176,-73.7004);
  way["power"](40.4774,-74.2591,40.9176,-73.7004);
  relation["power"](40.4774,-74.2591,40.9176,-73.7004);
);
out body;
>;
out skel qt;
```

The query returns a dataset containing power-related nodes, ways, and relations. Sample output:

```plaintext
id                @id                 power      type           geometry
0   relation/12529566  relation/12529566  None       multipolygon  MULTIPOLYGON ((-73.70169 40.75254, ...
1   relation/13567034  relation/13567034  None       multipolygon  POLYGON ((-74.16847 40.55964, -74.16...
...
18891 rows × 132 columns
```

In the Overpass Turbo UI, this query looks like:
![image](https://github.com/user-attachments/assets/9573c1af-85f9-4f26-951e-0521f9884ce0)


## Data Preprocessing

### Removing Unnecessary Data
We remove solar panel entries since they are abundant but not critical to this security analysis:

```python
df = gdf[~gdf["generator:method"].str.contains("photovoltaic", na=False)]
df_simple = df[["id", "power", "geometry"]]
```

### Extracting Nodes and Ways
We separate nodes and ways, processing their coordinates accordingly:

#### Nodes
```python
df_nodes = df_simple[df_simple["id"].str.contains("node")].reset_index()
df_nodes["x"] = df_nodes["geometry"].apply(lambda g: g.x)
df_nodes["y"] = df_nodes["geometry"].apply(lambda g: g.y)
df_nodes = df_nodes.drop("geometry", axis=1)
```

#### Ways (Reprojecting and Finding Centroids)
```python
projected_crs = "EPSG:32618"  # UTM Zone 18N (for NYC)
geographic_crs = "EPSG:4326"  # Lat/Lon

df_ways = df_simple[df_simple["id"].str.contains("way")].reset_index(drop=True)
df_ways = df_ways.set_geometry("geometry").to_crs(projected_crs)
df_ways["centroid"] = df_ways["geometry"].centroid
df_ways = df_ways.set_geometry("centroid").to_crs(geographic_crs)
df_ways["x"] = df_ways["centroid"].x
df_ways["y"] = df_ways["centroid"].y
df_ways = df_ways.drop(columns=["geometry", "centroid"])
```

### Merging Nodes and Ways
```python
df_combined = pd.concat([df_nodes, df_ways], ignore_index=True)
df_combined
```
**Output:**
```python
index     id             power      x           y
0    3455.0  node/302151657   tower  -74.110075  40.783746
1    3456.0  node/302151663   tower  -74.110214  40.781399
...
8104 rows × 5 columns
```

## Population Data Integration

### Loading Census Block Data
For this example, I will focus on a section of Hoboken, NJ, just across the river from Manhattan. We will use 2020 Census TIGER/Line block data for New Jersey to assign values to targets:

```python
nj_blocks_gdf = gpd.read_file("tl_2020_34_tabblock20.shp")
nj_blocks_gdf
```
**Output:**
```python
STATEFP20  COUNTYFP20  TRACTCE20  BLOCKCE20  GEOID20         POP20  geometry
34         023        001903    2001       340230019032001  100     POLYGON ((-74.34374 40.53623, ...
```

This setup allows us to use real-world infrastructure and population data to model a security game where attackers and defenders strategize over protecting or disrupting critical power infrastructure.

## Inspect Power Feature and Population Dataa
Let's use some utils from under-the-hood to inspect the power feature and population data in the subsection of NJ near NYC, across the Hudson River. 

```python
from utils.target_utils import assign_population
from utils.visualization import plot_power_features

power_df = assign_population(df_combined, nj_blocks_gdf, mode="block", radius=None)

# Define min/max bounds of power features
minx, miny, maxx, maxy = power_df["x"].min(), power_df["y"].min(), power_df["x"].max(), power_df["y"].max()

# Filter blocks within this bounding box
nj_blocks = nj_blocks_gdf.cx[minx:maxx, miny:maxy]

plot_power_features(power_df, nj_blocks)
```
![image](https://github.com/user-attachments/assets/acb19867-8519-4789-956f-7793c9d6972b)

This image shows nodes (blue) and ways (green) in the NYC-metro NJ area, with the points scaled by block population.

## Building Infrastructure Security Game
Let's use the `InfraSecurityGame` class to initialize and generate an ISG.

```python
from security_game.infra_security_game import InfraSecuritygame

# Power feature weights
POWER_WEIGHTS = {
    "plant": 2.0,
    "generator": 1.0,
    "line": 0.5,
    "cable": 0.25,
    "minor_line": 0.1,
    "tower": 0.75,
    "pole": 0.25,
    "substation": 1.0,
    "transformer": 0.5,
    "switchgear": 0.3,
    "busbar": 0.1,
    "bay": 0.1,
    "converter": 0.1,
    "insulator": 0.1,
    "portal": 0.1,
    "connection": 0.1,
    "compensator": 0.3,
    "rectifier": 0.1,
    "inverter": 0.1,
    "storage": 0.05,
}

# Bounding box for Hoboken, NJ
bbox_hoboken_small = (40.752635, 40.745600, -74.030386,-74.043903)

isg = InfraSecurityGame(power_df, nj_blocks, POWER_WEIGHTS, bbox=bbox_hoboken_small)
isg.generate(num_attackers=2, num_defenders=1, num_timesteps=15, home_base=8)
```

Now let's visualize the game graph:

```python
isg.draw_graph()
```
![image](https://github.com/user-attachments/assets/aa2ff6f5-1d58-45d2-b699-7912894053a5)

And inspect the output strategies and utility matrix:

```python
print(len(isg.attacker_strategies))
isg.attacker_strategies
```
**Output:**
```python
144
array([[[16, 16],
        [16, 16],
        [16, 16],
        ...,
        [16, 16],
        [16, 16],
        [16, 16]],

       [[16, 25],
        [16, 25],
        [16, 25],
        ...,
        [16, 25],
        [16, 25],
        [16, 25]],

       [[16, 12],
        [16, 12],
        [16, 12],
        ...,
        [16, 12],
        [16, 12],
        [16, 12]],

       ...,

       [[None, 14],
        [None, 14],
        [None, 14],
        ...,
        [None, 14],
        [None, 14],
        [None, 14]],

       [[None, 13],
        [None, 13],
        [None, 13],
        ...,
        [None, 13],
        [None, 13],
        [None, 13]],

       [[None, None],
        [None, None],
        [None, None],
        ...,
        [None, None],
        [None, None],
        [None, None]]], dtype=object)
```

```python
print(len(isg.defender_strategies))
isg.defender_strategies
```
**Output:**
```python
29249
array([[[ 8],
        [23],
        [23],
        ...,
        [ 9],
        [ 9],
        [ 8]],

       [[ 8],
        [ 8],
        [23],
        ...,
        [41],
        [ 9],
        [ 8]],

       [[ 8],
        [ 8],
        [ 8],
        ...,
        [ 8],
        [ 8],
        [ 8]],

       ...,

       [[ 8],
        [23],
        [14],
        ...,
        [ 8],
        [ 8],
        [ 8]],

       [[ 8],
        [ 8],
        [ 8],
        ...,
        [41],
        [ 9],
        [ 8]],

       [[ 8],
        [ 8],
        [23],
        ...,
        [41],
        [ 9],
        [ 8]]])
```

```python
isg.utility_matrix
```
**Output:**
```python
array([[ -44.25, -121.25, -121.25, ..., -188.  ,  -53.25,    0.  ],
       [ -44.25, -121.25, -121.25, ...,    0.  ,  -53.25,    0.  ],
       [ -44.25, -121.25, -121.25, ..., -188.  ,  -53.25,    0.  ],
       ...,
       [ -44.25, -121.25, -121.25, ..., -188.  ,  -53.25,    0.  ],
       [ -44.25, -121.25, -121.25, ...,    0.  ,  -53.25,    0.  ],
       [ -44.25,  -44.25, -121.25, ...,    0.  ,  -53.25,    0.  ]])
```
