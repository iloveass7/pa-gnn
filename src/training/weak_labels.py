import torch
from torch_geometric.utils import k_hop_subgraph

def compute_weak_labels(edge_index, node_labels, active_mask, hops=2, weak_value=0.7):
    """
    Propagate weak labels to 2-hop neighborhoods of confirmed hazards.
    Args:
        edge_index: (2, E)
        node_labels: (N,) tensor of hard labels (1.0 for hazard, 0.0 for safe)
        active_mask: (N,) boolean tensor of active nodes
        hops: int, number of hops
        weak_value: float, value to assign to weak labels
    Returns:
        updated_labels: (N,) tensor with weak labels
    """
    N = node_labels.size(0)
    updated_labels = node_labels.clone()
    
    # We only propagate if node_labels >= 0.9 (confirmed hazard).
    # Ignore invalid nodes (-1)
    hazard_nodes = torch.where((node_labels >= 0.9) & active_mask)[0]
    
    if len(hazard_nodes) > 0:
        # Find k-hop neighborhood
        # subset, edge_index, mapping, edge_mask
        subset, _, _, _ = k_hop_subgraph(hazard_nodes.tolist(), hops, edge_index, num_nodes=N)
        
        # We assign weak_value to subset nodes that are currently < 0.1 and active
        weak_mask = (node_labels < 0.1) & (node_labels >= 0.0) & active_mask
        
        in_subset = torch.zeros(N, dtype=torch.bool, device=node_labels.device)
        in_subset[subset] = True
        
        to_update = weak_mask & in_subset
        updated_labels[to_update] = weak_value
        
    return updated_labels
