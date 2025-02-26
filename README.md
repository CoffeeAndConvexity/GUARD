# Security Games: Example 1 - Green Security Game

## Introduction
The Security Game framework provides a structured way to model adversarial interactions over a graph. The `SecurityGame` class serves as the parent for all security games, fixing certain constraints such as all attackers being stationary and defenders moving, a designated home base, and structured strategy generation. This example walks through a `GreenSecurityGame` implementation using real-world animal movement data to model patrol strategies for protecting wildlife.

---

## Step 1: Load and Inspect Animal Location Data
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

## Step 2: Define the Geographic Boundaries
We define a bounding box for Lobeke National Park to filter the dataset to a more manageable sub-area.

```python
# Lobeke National Park Bounding Box
lat_min, lon_min = 2.05522, 15.8790
lat_max, lon_max = 2.2837, 16.2038

coordinate_rectangle = [lat_min, lat_max, lon_min, lon_max]
```

This bounding box will be used for visualization and game setup.

---

## Step 3: Visualizing Animal Locations
We can now visualize the spatial distribution of the recorded animal locations within the park.

```python
from utils.visualization import plot_animal_locations

plot_animal_locations(df, coordinate_rectangle)
```

![image](https://github.com/user-attachments/assets/1fde2a46-7220-495c-9911-58fe19c14410)


This animal density distribution will inform target creation. As a result, the generated game more accurately reflects the real world data.

---

## Step 4: Initialize and Generate the Green Security Game
Now, we create an instance of `GreenSecurityGame`, setting up the security game over a grid representation of the park.

```python
from security_game import GreenSecurityGame

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

## Step 5: Visualizing the Game Graph
```python
gsg.draw_graph()
```

![image](https://github.com/user-attachments/assets/73770d37-599d-4718-a2e3-39d1d3efbeb2)

This graph represents the game structure, with nodes for different cells of the grid and connections representing movement lateral cell to cell movement options.

---

## Step 6: Exploring Strategy Space
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

---

## Step 7: Inspecting the Utility Matrix
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

---
