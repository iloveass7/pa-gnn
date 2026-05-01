import numpy as np
from scipy.spatial import distance

def physics_aware_heuristic(n, goal, G, gamma_r=0.4, gamma_s=0.1):
    """
    Physics-aware heuristic for A* path planning.
    h(n) = d_Euclidean(n,g) * (1 + gamma_r*(1 - p_hat_n) + gamma_s*S_n)
    """
    pos_n = np.array(G.nodes[n]['pos'])
    pos_g = np.array(G.nodes[goal]['pos'])
    
    # Euclidean distance
    d_euc = distance.euclidean(pos_n, pos_g)
    
    # Risk score (in the blueprint, p_hat was traversability, so 1 - p_hat was risk)
    risk_n = G.nodes[n].get('risk', G.nodes[n].get('h_final', 0.0))
    
    # Slope
    s_n = G.nodes[n].get('slope', 0.0)
    
    # Heuristic: directly use risk_n since higher risk means higher cost
    h_n = d_euc * (1.0 + gamma_r * risk_n + gamma_s * s_n)
    return h_n
