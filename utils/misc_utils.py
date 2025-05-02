from scipy.spatial import KDTree
import numpy as np
import networkx as nx


def get_nearest_nodes_from_coords(graph, coords):
    """
    Given a list of (lat, lon) coordinates, return nearest node IDs from the graph.

    Parameters:
        graph: networkx graph with 'x' (longitude) and 'y' (latitude) node attributes.
        latlon_list: list of (lat, lon) coordinate tuples.

    Returns:
        list of node IDs nearest to each input coordinate.
    """
    node_ids = list(graph.nodes)
    node_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in node_ids]  # (lon, lat)
    tree = KDTree(node_coords)
    query_coords = [(lon, lat) for lat, lon in coords]  # flip input
    _, idxs = tree.query(query_coords)
    return [node_ids[i] for i in idxs]

def get_nearest_node_tuples(graph, coord_groups):
    """
    Given a list of coordinate tuples for each defender unit, return a list of tuples,
    where each tuple contains the nearest graph node to each coordinate in the input tuple.

    Parameters:
        graph: networkx graph with 'x' (longitude) and 'y' (latitude) node attributes.
        coord_groups: list of tuples/lists, where each sub-tuple contains (lat, lon) coordinate pairs.

    Returns:
        List of tuples of nearest node IDs for each group of coordinates.
    """
    node_ids = list(graph.nodes)
    node_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in node_ids]  # (lon, lat)
    tree = KDTree(node_coords)

    result = []
    for coord_group in coord_groups:
        node_tuple = tuple(
            node_ids[tree.query((lon, lat))[1]] for lat, lon in coord_group  # flip to (lon, lat)
        )
        result.append(node_tuple)

    return result


def point_line_distance(point, line_start, line_end):
    """Helper to compute distance from a point to a line segment."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Line segment length squared
    line_mag_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if line_mag_sq == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)  # Line start == line end

    # Projection onto the line (parametric t)
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_mag_sq))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)