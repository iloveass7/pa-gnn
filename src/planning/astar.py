import numpy as np
import networkx as nx
from src.planning.heuristics import physics_aware_heuristic


def run_astar(G, start, goal, gamma_r=0.0, gamma_s=0.0, admissible=True):
    """
    Run A* path planning on the active-node subgraph.

    Args:
        G:          NetworkX graph — nodes have 'active', 'pos', 'risk', 'slope', etc.
        start:      Start node ID
        goal:       Goal node ID
        gamma_r:    Risk inflation for weighted-A* heuristic (ignored when admissible=True)
        gamma_s:    Slope inflation for weighted-A* heuristic (ignored when admissible=True)
        admissible: If True (default), uses pure Euclidean h(n) — guarantees optimal path.
                    If False, uses weighted-A* h(n) = d_euc*(1+γ_r*risk+γ_s*slope).

    Returns:
        list of dicts (one per path node) or None if no path exists.
        Each dict contains: node_id, pos, risk, dominant_source,
                            euclidean_to_goal (for PLR computation).
    """
    active_nodes = [n for n, d in G.nodes(data=True) if d.get('active', True)]

    if start not in active_nodes or goal not in active_nodes:
        return None

    G_active = G.subgraph(active_nodes)

    goal_pos = np.array(G.nodes[goal]['pos'])

    def heuristic(n, g):
        return physics_aware_heuristic(
            n, g, G_active,
            gamma_r=gamma_r, gamma_s=gamma_s,
            admissible=admissible,
        )

    try:
        path_nodes = nx.astar_path(G_active, start, goal,
                                   heuristic=heuristic, weight='weight')

        # Euclidean straight-line distance start→goal (denominator for PLR)
        start_pos = np.array(G.nodes[start]['pos'])
        straight_line = float(np.linalg.norm(goal_pos - start_pos)) + 1e-8

        path_details = []
        for n in path_nodes:
            nd  = G.nodes[n]
            h_p = nd.get('h_physics', 0.0)
            h_l = nd.get('h_learned', 0.0)

            if h_p == 0 and h_l == 0:
                dom = "None"
            elif h_p > h_l:
                dom = "Physics"
            else:
                dom = "CNN"

            n_pos = np.array(nd['pos'])
            path_details.append({
                'node_id':           n,
                'pos':               nd['pos'],
                'risk':              nd.get('risk', nd.get('h_final', 0.0)),
                'dominant_source':   dom,
                'euclidean_to_goal': float(np.linalg.norm(goal_pos - n_pos)),
                'straight_line':     straight_line,   # constant — PLR denominator
            })

        return path_details

    except nx.NetworkXNoPath:
        return None

