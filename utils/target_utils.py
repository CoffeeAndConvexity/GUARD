from sklearn.cluster import KMeans

# def get_density_scores(df, coordinate_rectangle, num_columns, num_rows):
#     """
#     Assign scores to grid cells based on the direct animal location data.
    
#     df: DataFrame with columns ['animal_id', 'lat', 'long', 'timestamp'].
#     coordinate_rectangle: Tuple (min_lat, max_lat, min_lon, max_lon).
#     num_columns: Number of columns in the grid.
#     num_rows: Number of rows in the grid.
    
#     returns: A dictionary mapping grid cell indices (row, col) to animal count scores.
#     """
    
#     min_lat, max_lat, min_lon, max_lon = coordinate_rectangle
#     cell_height = (max_lat - min_lat) / num_rows
#     cell_width = (max_lon - min_lon) / num_columns

#     # Filter points within the bounding box
#     cell_df = df.copy(deep=True)[
#         (df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
#         (df['long'] >= min_lon) & (df['long'] <= max_lon)
#     ]
    
#     # Calculate grid cell indices for each point
#     cell_df['row'] = np.floor((cell_df['lat'] - min_lat) / cell_height).astype(int)
#     cell_df['col'] = np.floor((cell_df['long'] - min_lon) / cell_width).astype(int)

#     # Count occurrences in each grid cell
#     density_scores = (
#         cell_df.groupby(['row', 'col'])
#         .size()
#         .to_dict()
#     )

#     return density_scores

def get_density_scores(df, coordinate_rectangle, num_columns, num_rows):
    """
    Assign scores to grid cells based on estimated number of animals (not raw observation count).
    
    df: DataFrame with columns ['animal_id', 'lat', 'long', 'timestamp'].
    coordinate_rectangle: Tuple (min_lat, max_lat, min_lon, max_lon).
    num_columns: Number of columns in the grid.
    num_rows: Number of rows in the grid.
    
    Returns:
        Dictionary mapping (row, col) -> estimated number of animals.
    """
    min_lat, max_lat, min_lon, max_lon = coordinate_rectangle
    cell_height = (max_lat - min_lat) / num_rows
    cell_width = (max_lon - min_lon) / num_columns

    # Filter points within bounding box
    cell_df = df.copy(deep=True)[
        (df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
        (df['long'] >= min_lon) & (df['long'] <= max_lon)
    ]

    if cell_df.empty:
        return {}

    # Compute animal/observation weight
    num_observations = len(cell_df)
    num_animals = cell_df['animal_id'].nunique()
    weight = num_animals / num_observations if num_observations > 0 else 0

    # Map observations to grid cells
    cell_df['row'] = np.floor((cell_df['lat'] - min_lat) / cell_height).astype(int)
    cell_df['col'] = np.floor((cell_df['long'] - min_lon) / cell_width).astype(int)

    # Count occurrences in each grid cell and apply animal weight
    raw_counts = cell_df.groupby(['row', 'col']).size().to_dict()
    density_scores = {cell: count * weight for cell, count in raw_counts.items()}

    return density_scores

def get_centroid_scores(df, coordinate_rectangle, num_columns, num_rows, num_clusters):
    """
    Assign scores to grid cells based on centroids from clustering animal data,
    adjusted by estimated animal counts (not observation counts).

    df: DataFrame with columns ['animal_id', 'lat', 'long', 'timestamp'].
    coordinate_rectangle: Tuple (min_lat, max_lat, min_lon, max_lon).
    num_columns: Number of columns in the grid.
    num_rows: Number of rows in the grid.
    num_clusters: Number of clusters for k-means.
    
    Returns:
        - Dictionary mapping grid cell (row, col) -> adjusted score.
        - Tuple: (cell_width, cell_height).
    """
    min_lat, max_lat, min_lon, max_lon = coordinate_rectangle
    cell_height = (max_lat - min_lat) / num_rows
    cell_width = (max_lon - min_lon) / num_columns

    # Filter points within bounding box
    cell_df = df.copy(deep=True)[
        (df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
        (df['long'] >= min_lon) & (df['long'] <= max_lon)
    ]
    
    if cell_df.empty:
        return {}, (cell_width, cell_height)

    # Calculate normalization factor: animals / observations
    num_observations = len(cell_df)
    num_animals = cell_df['animal_id'].nunique()
    weight = num_animals / num_observations if num_observations > 0 else 0

    # K-means clustering
    coordinates = cell_df[['lat', 'long']].to_numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    cell_df['cluster'] = kmeans.fit_predict(coordinates)

    cluster_centers = kmeans.cluster_centers_
    cluster_sizes = cell_df.groupby('cluster').size().to_dict()

    # Assign weighted scores to grid cells
    centroid_scores = {}
    for cluster_idx, (lat, lon) in enumerate(cluster_centers):
        row = int((lat - min_lat) / cell_height)
        col = int((lon - min_lon) / cell_width)
        cell_key = (row, col)
        
        adjusted_score = cluster_sizes[cluster_idx] * weight
        centroid_scores[cell_key] = centroid_scores.get(cell_key, 0) + adjusted_score

    return centroid_scores

    def assign_population(power_df, block_gdf, mode="block", radius=None):
        """
        Assigns population to power features based on Census block data.

        Parameters:
        - power_df: DataFrame with columns ['id', 'x', 'y'] representing power feature locations.
        - block_gdf: GeoDataFrame with block geometries and population data.
        - mode: "block" assigns population of the block containing the feature; "radius" sums population of blocks within radius.
        - radius: Radius in meters for "radius" mode (ignored for "block" mode).

        Returns:
        - A new DataFrame with an added 'population' column.
        """

        # Convert power features into a GeoDataFrame
        power_gdf = gpd.GeoDataFrame(
            power_df,
            geometry=gpd.points_from_xy(power_df.x, power_df.y),
            crs=block_gdf.crs  # Ensure matching coordinate reference system
        )

        if mode == "block":
            # Spatial join to find the containing block
            joined = gpd.sjoin(power_gdf, block_gdf[['POP20', 'geometry']], how="left", predicate="within")
            power_df['population'] = joined['POP20']

        elif mode == "radius":
            if radius is None:
                raise ValueError("Radius must be specified for 'radius' mode.")

            # Convert meters to CRS units (assumes projected CRS, otherwise use geopy or pyproj for accurate conversion)
            power_gdf = power_gdf.to_crs(epsg=3857)  # Convert to meters-based CRS
            block_gdf = block_gdf.to_crs(epsg=3857)

            # Create buffer zones and find intersecting blocks
            power_gdf['buffer'] = power_gdf.geometry.buffer(radius)
            power_df['population'] = power_gdf['buffer'].apply(
                lambda buf: block_gdf[block_gdf.geometry.intersects(buf)]['POP20'].sum()
                if not block_gdf[block_gdf.geometry.intersects(buf)].empty else float('nan')
            )

        else:
            raise ValueError("Invalid mode. Choose 'block' or 'radius'.")

        return power_df[['id', 'x', 'y', 'power', 'population']]