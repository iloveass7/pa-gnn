import sys
import argparse
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.segmentation import mark_boundaries
import networkx as nx

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders.ai4mars_loader import AI4MarsDataset
from src.data.loaders.ctx_loader import CTXDataset
from src.models.fusion.fusion_model import EndToEndFusionModel
from src.models.gnn.graph_builder import GraphBuilder

def visualize_graph(data, image, save_path, title_prefix=""):
    """
    Visualize superpixel boundaries and the graph structure.
    """
    img_np = image.mean(dim=0).cpu().numpy()
    label_map = data.label_map.cpu().numpy()
    
    # 1. Superpixel boundaries
    bound_img = mark_boundaries(img_np, label_map, color=(1, 0, 0), mode='inner')
    
    # 2. Graph structure
    pos = data.pos.cpu().numpy()
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    x = data.x.cpu().numpy()
    active_mask = data.active_mask.cpu().numpy()
    
    # H_final is feature index 7
    h_final = x[:, 7]
    
    G = nx.Graph()
    for i in range(len(pos)):
        # Convert (y, x) to (x, y) for plotting
        G.add_node(i, pos=(pos[i, 1], pos[i, 0]), h_final=h_final[i], active=active_mask[i])
        
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        G.add_edge(u, v, weight=edge_attr[i])
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot boundaries
    ax1.imshow(bound_img)
    ax1.set_title(f"{title_prefix} Superpixel Boundaries (SLIC)")
    ax1.axis('off')
    
    # Plot graph
    ax2.imshow(img_np, cmap='gray', alpha=0.5)
    
    # Extract properties for plotting
    node_positions = nx.get_node_attributes(G, 'pos')
    node_colors = [data['h_final'] for node, data in G.nodes(data=True)]
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Only draw if there are edges
    if len(edges) > 0:
        nx.draw_networkx_edges(G, node_positions, ax=ax2, edge_color=weights, 
                               edge_cmap=plt.cm.Blues, width=1.5, alpha=0.7)
                           
    # Draw nodes
    # Active nodes
    active_nodes = [n for n, d in G.nodes(data=True) if d['active']]
    inactive_nodes = [n for n, d in G.nodes(data=True) if not d['active']]
    
    if len(active_nodes) > 0:
        nx.draw_networkx_nodes(G, node_positions, nodelist=active_nodes, 
                               node_color=[node_colors[n] for n in active_nodes], 
                               cmap=plt.cm.RdYlGn_r, vmin=0, vmax=1, node_size=30, ax=ax2)
                           
    # Draw inactive nodes (obstacles) as black squares
    if len(inactive_nodes) > 0:
        nx.draw_networkx_nodes(G, node_positions, nodelist=inactive_nodes, 
                               node_color='black', node_shape='s', node_size=40, ax=ax2)
                           
    ax2.set_title(f"{title_prefix} Graph Structure (Nodes=H_final, Black=Deactivated)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_cfg', type=str, default='configs/base.yaml')
    parser.add_argument('--cnn_cfg', type=str, default='configs/cnn/mobilenetv3.yaml')
    parser.add_argument('--phys_cfg', type=str, default='configs/physics.yaml')
    parser.add_argument('--fusion_cfg', type=str, default='configs/fusion/adaptive_fusion.yaml')
    parser.add_argument('--gat_cfg', type=str, default='configs/gnn/gatv2.yaml')
    args = parser.parse_args()

    base_cfg = load_config(args.base_cfg)
    cnn_cfg = load_config(args.cnn_cfg)
    phys_cfg = load_config(args.phys_cfg)
    fusion_cfg = load_config(args.fusion_cfg)
    gat_cfg = load_config(args.gat_cfg)
    
    set_seed(base_cfg.project.seed)
    device = get_device(base_cfg.project.device)
    
    results_dir = Path(base_cfg.paths.results) / "stage5"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Models
    fusion_model = EndToEndFusionModel(cnn_cfg, phys_cfg, fusion_cfg, freeze_cnn=True).to(device)
    graph_builder = GraphBuilder(gat_cfg)
    
    # Data Loaders
    print("Loading datasets...")
    ai4mars_cfg = load_config("configs/datasets/ai4mars.yaml")
    test_ds = AI4MarsDataset.from_config(base_cfg, ai4mars_cfg, split="test")
    
    ctx_cfg = load_config("configs/datasets/ctx.yaml")
    ctx_ds = CTXDataset.from_config(base_cfg, ctx_cfg, max_tiles=100)
    
    # AI4Mars images (5)
    print("Processing AI4Mars images...")
    for i in range(5):
        image, target, meta = test_ds[i]
        image_b = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            fusion_dict = fusion_model(image_b)
            
        data = graph_builder.build(image, fusion_dict)
        print(f"  AI4Mars[{i}]: Nodes={data.x.size(0)}, Edges={data.edge_index.size(1)}, Active={data.active_mask.sum().item()}")
        
        visualize_graph(data, image, results_dir / f"graph_ai4mars_{i}.png", title_prefix=f"AI4Mars[{i}]")
        
    # CTX images (3)
    print("Processing CTX images...")
    indices = ctx_ds.select_demo_tiles(n=3, seed=base_cfg.project.seed)
    for i, idx in enumerate(indices):
        image, meta = ctx_ds[idx]
        image_b = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            fusion_dict = fusion_model(image_b)
            
        data = graph_builder.build(image, fusion_dict)
        print(f"  CTX[{idx}]: Nodes={data.x.size(0)}, Edges={data.edge_index.size(1)}, Active={data.active_mask.sum().item()}")
        
        visualize_graph(data, image, results_dir / f"graph_ctx_{idx}.png", title_prefix=f"CTX[{idx}]")
        
    print(f"Verification completed. Visualizations saved to {results_dir}")

if __name__ == "__main__":
    main()
