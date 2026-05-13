import numpy as np
from scipy.spatial import distance


def physics_aware_heuristic(n, goal, G, gamma_r=0.0, gamma_s=0.0, admissible=True):
    """
    Heuristic for A* path planning on the PA-GNN graph.

    Args:
        n:          Current node ID
        goal:       Goal node ID
        G:          NetworkX graph with node attribute 'pos' = (y, x)
        gamma_r:    Risk inflation factor (only used when admissible=False)
        gamma_s:    Slope inflation factor (only used when admissible=False)
        admissible: If True (default), returns pure Euclidean distance h(n) = d_euc.
                    This is ADMISSIBLE — never overestimates true cost because every
                    edge already carries a weight ≥ its Euclidean component.
                    If False, returns weighted-A* h(n) = d_euc*(1+γ_r*risk+γ_s*slope),
                    which may overestimate and yields ε-suboptimal paths (useful for
                    speed trade-off; must be documented as weighted A* in the thesis).

    Returns:
        float: heuristic estimate h(n)
    """
    pos_n = np.array(G.nodes[n]['pos'])
    pos_g = np.array(G.nodes[goal]['pos'])
    d_euc = distance.euclidean(pos_n, pos_g)

    if admissible or (gamma_r == 0.0 and gamma_s == 0.0):
        # Admissible: pure Euclidean — satisfies h(n) ≤ true_cost for all n
        return d_euc
    else:
        # Weighted A* (ε-suboptimal) — faster but not optimal.
        # Thesis must state: "We use weighted A* with ε ≤ (1 + γ_r + γ_s)"
        risk_n = G.nodes[n].get('risk', G.nodes[n].get('h_final', 0.0))
        s_n    = G.nodes[n].get('slope', 0.0)
        return d_euc * (1.0 + gamma_r * risk_n + gamma_s * s_n)

