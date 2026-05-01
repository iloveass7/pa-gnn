import torch
from torch_geometric.data import Data
from src.graph.superpixels import compute_superpixels
from src.graph.node_features import compute_node_features
from src.graph.adjacency import build_adjacency_and_edges

class GraphBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def build(self, image, fusion_dict, target=None):
        """
        Orchestrator to build PyG Data object.
        Args:
            image: (1 or 3, H, W) original image tensor
            fusion_dict: output of EndToEndFusionModel containing S, R, D, h_physics, h_learned, h_final, alpha
            target: (1, H, W) optional ground truth risk map
        """
        device = image.device
        
        # 1. Superpixels
        slic_cfg = self.cfg.graph.slic
        label_map = compute_superpixels(
            image, 
            n_segments=slic_cfg.n_segments,
            compactness=slic_cfg.compactness,
            sigma=slic_cfg.sigma
        )
        
        # 2. Node features
        features, centroids, is_hazardous, node_targets = compute_node_features(
            label_map,
            image,
            fusion_dict['h_physics'],
            fusion_dict['h_learned'],
            fusion_dict['h_final'],
            fusion_dict['alpha'],
            fusion_dict['S'],
            fusion_dict['R'],
            fusion_dict['D'],
            hazard_threshold=self.cfg.graph.hazard_threshold,
            target=target
        )
        
        # 3. Adjacency & Edges
        ew_cfg = self.cfg.graph.edge_weights
        edges, edge_weights, full_features = build_adjacency_and_edges(
            label_map, features, centroids, is_hazardous,
            alpha_w=ew_cfg.alpha_w,
            beta_w=ew_cfg.beta_w,
            gamma_w=ew_cfg.gamma_w
        )
        
        # If no edges were formed (unlikely but possible), handle it gracefully
        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0,), dtype=torch.float32, device=device)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32, device=device)
            
        x = torch.tensor(full_features, dtype=torch.float32, device=device)
        pos = torch.tensor(centroids, dtype=torch.float32, device=device)
        
        # 4. Create active mask (True = active, False = deactivated obstacle)
        active_mask = ~torch.tensor(is_hazardous, dtype=torch.bool, device=device)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        data.active_mask = active_mask
        data.label_map = torch.tensor(label_map, dtype=torch.long, device=device)
        if node_targets is not None:
            data.y = torch.tensor(node_targets, dtype=torch.float32, device=device)
        
        return data
