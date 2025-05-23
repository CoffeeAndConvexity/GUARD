import osmnx as ox
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class GraphGenerator:
    def __init__(self):
        pass
    
    def create_graph(self, place_name, network_type='drive'):
        """
        Create a graph from a specified place.
        
        Parameters:
        - place_name: Name of the place (e.g., "Hudson Yards, New York, USA").
        - network_type: Type of network to create (e.g., 'drive', 'walk', 'bike').
        
        Returns:
        - G: The generated graph.
        """
        G = ox.graph_from_place(place_name, network_type=network_type)
        return G
    
    def get_strongly_connected_component(self, G):
        """
        Get the largest strongly connected component of the graph.
        
        Parameters:
        - G: The graph to analyze.
        
        Returns:
        - G_s: The largest strongly connected component as a new graph.
        """
        strong = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
        G_s = G.subgraph(strong[0])
        G_s = nx.convert_node_labels_to_integers(G_s)
        return G_s
    
    def plot_graph(self, G):
        """
        Plot the graph using OSMnx's plotting functions.
        
        Parameters:
        - G: The graph to plot.
        """
        fig, ax = ox.plot_graph(G)
        plt.show()
    
    def scale_node_coordinates(self, G):
        """
        Scale the node coordinates of the graph.
        
        Parameters:
        - G: The graph whose node coordinates will be scaled.
        
        Returns:
        - G: The graph with scaled coordinates added as node attributes.
        """
        coords = [(G.nodes[x]['x'], G.nodes[x]['y']) for x in range(len(G.nodes))]
        min_max_scaler = MinMaxScaler()
        coords_scaled = min_max_scaler.fit_transform(coords)
        
        for i in range(len(G.nodes)):
            G.nodes[i]['scaled_x'], G.nodes[i]['scaled_y'] = coords_scaled[i, 0], coords_scaled[i, 1]
        
        return G

    def generate_simplified_graph(self, place_name, network_type='drive', show=True):
        """
        Generate a simplified graph from a place, extract its largest strongly connected component,
        scale the node coordinates, and plot the graph.
        
        Parameters:
        - place_name: Name of the place (e.g., "Hudson Yards, New York, USA").
        - network_type: Type of network to create (e.g., 'drive', 'walk', 'bike').
        
        Returns:
        - G_s: The simplified and scaled graph.
        """
        # Step 1: Create the graph from the place
        G = self.create_graph(place_name, network_type)
        
        # Step 2: Get the largest strongly connected component
        G_s = self.get_strongly_connected_component(G)
        
        # Step 3: Scale the node coordinates
        G_s = self.scale_node_coordinates(G_s)
        
        # Step 4: Plot the graph if show == True
        if show:
            self.plot_graph(G_s)
        
        return G_s

# Example usage
if __name__ == "__main__":
    graph_gen = GraphGenerator()
    G_hy_simplified = graph_gen.generate_simplified_graph("Hudson Yards, New York, USA")
    print(G_hy_simplified.nodes(data=True))