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
            image:       Tensor (C, H, W) or (1, C, H, W) — original image, values [0,1]
            fusion_dict: dict with keys S, R, D, h_physics, h_learned, h_final, alpha
            target:      Tensor (1, H, W) or None — pixel-level risk map (-1 = ignore)

        Returns:
            data: torch_geometric.data.Data with:
                data.x            (N, 14)  node features
                data.edge_index   (2, E)   edge connectivity
                data.edge_attr    (E,)     edge weights
                data.pos          (N, 2)   centroid positions
                data.y            (N,)     node targets (-1 = ignore; all-ignore when no target)
                data.active_mask  (N,)     bool — True = active during path planning
        """
        # Handle batched image (1, C, H, W) → (C, H, W) for graph ops
        if image.dim() == 4:
            image = image[0]
        device = image.device

        # ── 1. Superpixels ─────────────────────────────────────────────────
        slic_cfg  = self.cfg.graph.slic
        label_map = compute_superpixels(
            image,
            n_segments=slic_cfg.n_segments,
            compactness=slic_cfg.compactness,
            sigma=slic_cfg.sigma,
        )

        # ── 2. Node features ────────────────────────────────────────────────
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
            target=target,
        )

        # ── 3. Adjacency & Edges ────────────────────────────────────────────
        ew_cfg = self.cfg.graph.edge_weights
        edges, edge_weights, full_features = build_adjacency_and_edges(
            label_map, features, centroids, is_hazardous,
            alpha_w=ew_cfg.alpha_w,
            beta_w=ew_cfg.beta_w,
            gamma_w=ew_cfg.gamma_w,
        )

        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long,    device=device)
            edge_attr  = torch.empty((0,),   dtype=torch.float32, device=device)
        else:
            edge_index = torch.tensor(edges,        dtype=torch.long,    device=device).t().contiguous()
            edge_attr  = torch.tensor(edge_weights, dtype=torch.float32, device=device)

        x   = torch.tensor(full_features, dtype=torch.float32, device=device)
        pos = torch.tensor(centroids,     dtype=torch.float32, device=device)

        # ── 4. Active mask ──────────────────────────────────────────────────
        is_hazardous_tensor = torch.tensor(is_hazardous, dtype=torch.bool, device=device)

        if target is not None:
            # Training: all nodes active so GNN learns from hazard examples too
            active_mask = torch.ones(len(is_hazardous), dtype=torch.bool, device=device)
        else:
            # Inference: deactivate confirmed hazard nodes for A* planning
            active_mask = ~is_hazardous_tensor

        # ── 5. Build Data object ────────────────────────────────────────────
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        data.active_mask = active_mask

        # Always set data.y as a float tensor so downstream code never gets None
        # or AttributeError. When there is no target, all nodes are marked ignore (-1).
        if node_targets is not None:
            data.y = torch.tensor(node_targets, dtype=torch.float32, device=device)
        else:
            data.y = torch.full((x.size(0),), -1.0, dtype=torch.float32, device=device)

        # label_map is stored for inference/visualization scripts that need to
        # back-project per-node values onto pixel space.
        # NOTE: PyG DataLoader with batch_size>1 incorrectly stacks (H,W) tensors
        # along dim-0. This is safe for single-image inference. The PrecomputedGraphDataset
        # strips label_map before returning batched data during training.
        data.label_map = torch.tensor(label_map, dtype=torch.long, device=device)

        return data

