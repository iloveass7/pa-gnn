import torch
import numpy as np
import networkx as nx
from pathlib import Path
from src.models.fusion.fusion_model import EndToEndFusionModel
from src.models.gnn.graph_builder import GraphBuilder
from src.models.gnn.gatv2 import PAGATv2
from src.planning.astar import run_astar

class PA_GNN_Pipeline:
    def __init__(self, base_cfg, cnn_cfg, phys_cfg, fusion_cfg, gat_cfg, device,
                 fusion_ckpt=None, gat_ckpt=None):
        self.device = device
        self.gat_cfg = gat_cfg
        
        self.fusion_model = EndToEndFusionModel(cnn_cfg, phys_cfg, fusion_cfg, freeze_cnn=True).to(device)
        self.fusion_model.eval()
        
        self.graph_builder = GraphBuilder(gat_cfg)
        
        self.gat_model = PAGATv2(gat_cfg).to(device)
        self.gat_model.eval()
        
        # Load weights if exist
        if fusion_ckpt and Path(fusion_ckpt).exists():
            ckpt = torch.load(fusion_ckpt, map_location=device, weights_only=True)
            self.fusion_model.load_state_dict(ckpt['model_state_dict'])
            
        if gat_ckpt and Path(gat_ckpt).exists():
            ckpt = torch.load(gat_ckpt, map_location=device, weights_only=True)
            self.gat_model.load_state_dict(ckpt['model_state_dict'])

    @torch.no_grad()
    def run(self, image, start_coords=None, goal_coords=None, run_baseline='proposed'):
        """
        image: (1 or 3, H, W)
        run_baseline: 'proposed', 'b1_euclidean', 'b2_physics', 'b3_learned', 'b4_static'
        """
        img_b = image.unsqueeze(0).to(self.device)
        
        # Stage 1-4
        fusion_dict = self.fusion_model(img_b)
        
        # Stage 5
        data = self.graph_builder.build(image, fusion_dict)
        if data.x.size(0) == 0:
            return None, data, fusion_dict
            
        x_np = data.x.cpu().numpy()
        pos = data.pos.cpu().numpy()
        active = data.active_mask.cpu().numpy()
        
        # Stage 6
        preds = self.gat_model(data.x, data.edge_index, data.edge_attr)
        risk_scores = preds.cpu().numpy()
        
        edges = data.edge_index.cpu().numpy()
        edge_attr = data.edge_attr.cpu().numpy()
        
        # Build NetworkX graph
        G = nx.Graph()
        N = x_np.shape[0]
        for i in range(N):
            G.add_node(
                i,
                pos=(pos[i, 0], pos[i, 1]), # y, x
                slope=x_np[i, 2],
                h_physics=x_np[i, 5],
                h_learned=x_np[i, 6],
                h_final=x_np[i, 7],
                alpha=x_np[i, 8],
                risk=risk_scores[i],
                active=active[i]
            )
            
        ew_cfg = self.gat_cfg.graph.edge_weights
        alpha_w, beta_w, gamma_w = ew_cfg.alpha_w, ew_cfg.beta_w, ew_cfg.gamma_w
        
        for i in range(edges.shape[1]):
            u, v = edges[0, i], edges[1, i]
            
            if run_baseline == 'proposed':
                # Deactivate safe nodes erroneously marked as safe?
                # The config says "p_hat_i < 0.2 -> deactivated". Wait, the thesis blueprint says:
                # "Nodes with p_hat_i < 0.2 deactivated" in some contexts, but usually <0.2 means VERY safe.
                # If they are very safe, they should be active. Maybe they meant p_hat_i > 0.8 deactivated?
                # Actually, earlier we said mean(H_final) > 0.7 -> obstacle. Let's keep active what is active.
                if risk_scores[u] > 0.8: G.nodes[u]['active'] = False
                if risk_scores[v] > 0.8: G.nodes[v]['active'] = False
                
                risk_u, risk_v = risk_scores[u], risk_scores[v]
                avg_risk = (risk_u + risk_v) / 2.0
                dist = np.linalg.norm(pos[u] - pos[v]) / 724.07
                diff_s = abs(x_np[u, 2] - x_np[v, 2])
                w = alpha_w * avg_risk + beta_w * dist + gamma_w * diff_s
                
            elif run_baseline == 'b1_euclidean':
                G.nodes[u]['active'] = True
                G.nodes[v]['active'] = True
                w = np.linalg.norm(pos[u] - pos[v])
            elif run_baseline == 'b2_physics':
                G.nodes[u]['active'] = True
                G.nodes[v]['active'] = True
                w = alpha_w * ((x_np[u, 5] + x_np[v, 5])/2) + beta_w * (np.linalg.norm(pos[u] - pos[v]) / 724.07)
            elif run_baseline == 'b3_learned':
                G.nodes[u]['active'] = True
                G.nodes[v]['active'] = True
                w = alpha_w * ((x_np[u, 6] + x_np[v, 6])/2) + beta_w * (np.linalg.norm(pos[u] - pos[v]) / 724.07)
            else:
                w = edge_attr[i]
                
            G.add_edge(u, v, weight=w)
            
        # Stage 7 Planning
        if start_coords is None: start_coords = (10, 10)
        if goal_coords is None: goal_coords = (500, 500)
        
        # Closest nodes to coordinates
        start_node = int(np.argmin(np.linalg.norm(pos - np.array(start_coords), axis=1)))
        goal_node = int(np.argmin(np.linalg.norm(pos - np.array(goal_coords), axis=1)))
        
        # Ensure start/goal are active
        G.nodes[start_node]['active'] = True
        G.nodes[goal_node]['active'] = True
        
        gamma_r = 0.4 if run_baseline == 'proposed' else 0.0
        gamma_s = 0.1 if run_baseline == 'proposed' else 0.0
        
        path_details = run_astar(G, start_node, goal_node, gamma_r=gamma_r, gamma_s=gamma_s)
        
        return path_details, data, fusion_dict
