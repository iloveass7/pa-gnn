# pyrefly: ignore [missing-import]
import torch
import numpy as np
import networkx as nx
import time
from pathlib import Path
from src.models.fusion.fusion_model import EndToEndFusionModel
from src.models.gnn.graph_builder import GraphBuilder
from src.models.gnn.gatv2 import PAGATv2
from src.planning.astar import run_astar


class PA_GNN_Pipeline:
    def __init__(self, base_cfg, cnn_cfg, phys_cfg, fusion_cfg, gat_cfg, device,
                 fusion_ckpt=None, gat_ckpt=None):
        self.device   = device
        self.gat_cfg  = gat_cfg

        self.fusion_model = EndToEndFusionModel(cnn_cfg, phys_cfg, fusion_cfg, freeze_cnn=True).to(device)
        self.fusion_model.eval()

        self.graph_builder = GraphBuilder(gat_cfg)

        self.gat_model = PAGATv2(gat_cfg).to(device)
        self.gat_model.eval()

        if fusion_ckpt and Path(fusion_ckpt).exists():
            ckpt = torch.load(fusion_ckpt, map_location=device, weights_only=False)
            self.fusion_model.load_state_dict(ckpt['model_state_dict'])

        if gat_ckpt and Path(gat_ckpt).exists():
            ckpt = torch.load(gat_ckpt, map_location=device, weights_only=False)
            self.gat_model.load_state_dict(ckpt['model_state_dict'])

    @torch.no_grad()
    def run(self, image, start_coords=None, goal_coords=None,
            run_baseline='proposed', benchmark=False,
            ground_truth=None):
        """
        Full pipeline: image → path.

        Args:
            image:         Tensor (C, H, W) — image in [0,1]
            start_coords:  (y, x) tuple or None
            goal_coords:   (y, x) tuple or None
            run_baseline:  'proposed' | 'b1_euclidean' | 'b2_physics' |
                           'b3_learned' | 'b4_static' | 'b5_no_gnn' | 'oracle'
            benchmark:     if True, return per-stage timing dict
            ground_truth:  Tensor (1, H, W) risk map — required for 'oracle' baseline

        Returns:
            (path_details, data, fusion_dict, timings)
        """
        timings = {} if benchmark else None

        img_b = image.unsqueeze(0).to(self.device)

        # ── Stages 1-4: Physics + CNN + Fusion ────────────────────────────
        t0 = time.time() if benchmark else None
        fusion_dict = self.fusion_model(img_b)
        if benchmark: timings['stages_1_to_4_fusion'] = time.time() - t0

        # ── Stage 5: Superpixel Graph Construction ─────────────────────────
        if benchmark: t0 = time.time()
        data = self.graph_builder.build(image, fusion_dict)
        if benchmark: timings['stage_5_graph'] = time.time() - t0

        if data.x.size(0) == 0:
            return None, data, fusion_dict, timings

        # Cache CPU numpy views
        x_np   = data.x.cpu().numpy()
        pos    = data.pos.cpu().numpy()
        active = data.active_mask.cpu().numpy()

        # ── Stage 6: GATv2 Traversability Refinement ──────────────────────
        if benchmark: t0 = time.time()

        if run_baseline == 'b5_no_gnn':
            # C3 — B5: Skip GATv2, use H_final directly as risk scores
            risk_scores = x_np[:, 7]                           # feature 7 = mean_H_final
        elif run_baseline == 'oracle':
            # C3 — Oracle: use ground-truth pixel risk map aggregated per node
            if ground_truth is not None:
                gt_np = ground_truth.squeeze().cpu().numpy()   # (H, W)
                from src.graph.superpixels import compute_superpixels
                label_map = compute_superpixels(
                    image,
                    n_segments=self.gat_cfg.graph.slic.n_segments,
                    compactness=self.gat_cfg.graph.slic.compactness,
                    sigma=self.gat_cfg.graph.slic.sigma,
                )
                N = data.x.size(0)
                risk_scores = np.zeros(N, dtype=np.float32)
                from skimage.measure import regionprops
                for prop in regionprops(label_map):
                    if prop.label - 1 < N:                     # regionprops is 1-indexed
                        mask = label_map == prop.label
                        valid = gt_np[mask]
                        valid = valid[valid >= 0]
                        risk_scores[prop.label - 1] = valid.mean() if len(valid) > 0 else 0.0
            else:
                # Fallback: treat as b5 if no GT provided
                risk_scores = x_np[:, 7]
        else:
            # Proposed + all other baselines: run GATv2
            data      = data.to(self.device)
            preds     = self.gat_model(data.x, data.edge_index, data.edge_attr)
            risk_scores = preds.cpu().numpy()
            data      = data.cpu()

        if benchmark: timings['stage_6_gnn'] = time.time() - t0

        edges     = data.edge_index.cpu().numpy()
        edge_attr = data.edge_attr.cpu().numpy()

        # ── Build NetworkX graph ───────────────────────────────────────────
        G = nx.Graph()
        N = x_np.shape[0]

        for i in range(N):
            G.add_node(
                i,
                pos=(pos[i, 0], pos[i, 1]),
                slope=x_np[i, 2],
                h_physics=x_np[i, 5],
                h_learned=x_np[i, 6],
                h_final=x_np[i, 7],
                alpha=x_np[i, 8],
                risk=float(risk_scores[i]),
                active=bool(active[i]),
            )

        ew_cfg = self.gat_cfg.graph.edge_weights
        alpha_w, beta_w, gamma_w = ew_cfg.alpha_w, ew_cfg.beta_w, ew_cfg.gamma_w
        max_pos_dist = np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)) + 1e-8

        # Deactivation threshold: C2 — set by config; retry logic below handles relaxation
        deact_thresh = self.gat_cfg.graph.get('deactivation_threshold', 0.30)
        deact_risk   = 1.0 - deact_thresh                      # default: risk > 0.70 → blocked

        for i in range(edges.shape[1]):
            u, v = edges[0, i], edges[1, i]
            dist     = np.linalg.norm(pos[u] - pos[v]) / max_pos_dist
            diff_s   = abs(x_np[u, 2] - x_np[v, 2])

            if run_baseline == 'proposed' or run_baseline in ('b5_no_gnn', 'oracle'):
                # Deactivate high-risk nodes
                if risk_scores[u] > deact_risk: G.nodes[u]['active'] = False
                if risk_scores[v] > deact_risk: G.nodes[v]['active'] = False
                avg_risk = (float(risk_scores[u]) + float(risk_scores[v])) / 2.0
                w = alpha_w * avg_risk + beta_w * dist + gamma_w * diff_s

            elif run_baseline == 'b1_euclidean':
                G.nodes[u]['active'] = True
                G.nodes[v]['active'] = True
                w = np.linalg.norm(pos[u] - pos[v])

            elif run_baseline == 'b2_physics':
                G.nodes[u]['active'] = True
                G.nodes[v]['active'] = True
                avg_hp = (x_np[u, 5] + x_np[v, 5]) / 2.0
                w = alpha_w * avg_hp + beta_w * dist

            elif run_baseline == 'b3_learned':
                G.nodes[u]['active'] = True
                G.nodes[v]['active'] = True
                avg_hl = (x_np[u, 6] + x_np[v, 6]) / 2.0
                w = alpha_w * avg_hl + beta_w * dist

            elif run_baseline == 'b4_static':
                # M4: proper static 50/50 fusion (α=0.5 fixed, no adaptive weighting)
                G.nodes[u]['active'] = True
                G.nodes[v]['active'] = True
                avg_static = 0.5 * (x_np[u, 5] + x_np[v, 5]) / 2.0 + \
                             0.5 * (x_np[u, 6] + x_np[v, 6]) / 2.0
                w = alpha_w * avg_static + beta_w * dist + gamma_w * diff_s

            else:
                w = float(edge_attr[i])

            G.add_edge(u, v, weight=float(w))

        # ── Stage 7: A* Path Planning (with adaptive threshold — C2) ───────
        if start_coords is None: start_coords = (10, 10)
        if goal_coords  is None: goal_coords  = (500, 500)

        start_node = int(np.argmin(np.linalg.norm(pos - np.array(start_coords), axis=1)))
        goal_node  = int(np.argmin(np.linalg.norm(pos - np.array(goal_coords),  axis=1)))

        G.nodes[start_node]['active'] = True
        G.nodes[goal_node]['active']  = True

        # C2: Adaptive threshold relaxation — retry with looser thresholds if A* fails
        # This prevents the 60% success-rate failure when dense hazard blocks all paths
        relaxation_steps = [deact_risk, 0.80, 0.90, 1.01]   # 1.01 = unblock everything
        path_details = None

        for thresh in relaxation_steps:
            if thresh != deact_risk:
                # Reactivate nodes that were blocked by the previous (stricter) threshold
                for node in G.nodes():
                    if risk_scores[node] <= thresh:
                        G.nodes[node]['active'] = True
                G.nodes[start_node]['active'] = True
                G.nodes[goal_node]['active']  = True

            if benchmark: t0 = time.time()
            path_details = run_astar(G, start_node, goal_node,
                                     gamma_r=0.0, gamma_s=0.0)  # C5: admissible
            if benchmark: timings['stage_7_astar'] = time.time() - t0

            if path_details is not None:
                if thresh != deact_risk:
                    # Tag the path so evaluator can log the relaxed threshold used
                    for step in path_details:
                        step['relaxed_threshold'] = thresh
                break

        if benchmark:
            timings['total'] = sum(timings.values())

        return path_details, data, fusion_dict, timings

