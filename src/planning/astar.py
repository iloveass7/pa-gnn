import networkx as nx
from src.planning.heuristics import physics_aware_heuristic

def run_astar(G, start, goal, gamma_r=0.4, gamma_s=0.1):
    """
    Run A* path planning on the NetworkX graph.
    Nodes with 'active'=False are obstacles and should not be traversed.
    """
    # Create a subgraph with only active nodes
    active_nodes = [n for n, d in G.nodes(data=True) if d.get('active', True)]
    
    # Check if start/goal are valid
    if start not in active_nodes or goal not in active_nodes:
        return None  # No path if start/goal is obstacle
        
    G_active = G.subgraph(active_nodes)
    
    def heuristic(n, g):
        return physics_aware_heuristic(n, g, G_active, gamma_r, gamma_s)
        
    try:
        path_nodes = nx.astar_path(G_active, start, goal, heuristic=heuristic, weight='weight')
        
        # Compile path details
        path_details = []
        for n in path_nodes:
            node_data = G.nodes[n]
            
            # Determine dominant risk source
            h_p = node_data.get('h_physics', 0.0)
            h_l = node_data.get('h_learned', 0.0)
            dom_source = "Physics" if h_p > h_l else "CNN"
            if h_p == 0 and h_l == 0: dom_source = "None"
            
            detail = {
                'node_id': n,
                'pos': node_data['pos'],
                'risk': node_data.get('risk', node_data.get('h_final', 0.0)),
                'dominant_source': dom_source
            }
            path_details.append(detail)
            
        return path_details
        
    except nx.NetworkXNoPath:
        return None
