import numpy as np
from skimage.future import graph
from scipy.spatial import distance

def build_adjacency_and_edges(label_map, features, centroids, is_hazardous, 
                              alpha_w=0.6, beta_w=0.25, gamma_w=0.15):
    """
    Build Region Adjacency Graph and compute edge weights.
    Returns:
        edges: list of (u, v) tuples
        edge_weights: list of weights
        features_updated: features array with hazardous_neighbour_count populated
    """
    rag = graph.RAG(label_map)
    
    N = features.shape[0]
    features_updated = np.copy(features)
    
    # 1. Calculate hazardous neighbors (Feature 14)
    for n in rag.nodes():
        if n >= N: continue
        neighbors = list(rag.neighbors(n))
        haz_count = sum(1 for nb in neighbors if nb < N and is_hazardous[nb])
        features_updated[n, 13] = haz_count
        
    # 2. Build edges
    edges = []
    edge_weights = []
    
    H, W = label_map.shape
    max_dist = np.sqrt(H**2 + W**2)
    
    for u, v in rag.edges():
        if u >= N or v >= N: continue
        
        # Risk term: average of mean_H_final (feature index 7)
        risk_u = features_updated[u, 7]
        risk_v = features_updated[v, 7]
        avg_risk = (risk_u + risk_v) / 2.0
        
        # Distance term: normalized euclidean distance
        dist = distance.euclidean(centroids[u], centroids[v])
        norm_dist = dist / max_dist
        
        # Slope discontinuity term: abs diff of mean_S (feature index 2)
        slope_u = features_updated[u, 2]
        slope_v = features_updated[v, 2]
        slope_diff = abs(slope_u - slope_v)
        
        # Final edge weight
        weight = alpha_w * avg_risk + beta_w * norm_dist + gamma_w * slope_diff
        
        # Add undirected edges
        edges.append((u, v))
        edge_weights.append(weight)
        edges.append((v, u))
        edge_weights.append(weight)
        
    return edges, edge_weights, features_updated
