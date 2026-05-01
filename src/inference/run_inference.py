import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import get_device
from src.inference.pipeline import PA_GNN_Pipeline
from src.visualization.paths import plot_path_on_image
from src.utils.io import ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help="Path to input image")
    parser.add_argument('--start', type=str, default="10,10", help="Start (y,x)")
    parser.add_argument('--goal', type=str, default="500,500", help="Goal (y,x)")
    parser.add_argument('--baseline', type=str, default='proposed', help="Which baseline to run")
    parser.add_argument('--base_cfg', type=str, default='configs/base.yaml')
    parser.add_argument('--cnn_cfg', type=str, default='configs/cnn/mobilenetv3.yaml')
    parser.add_argument('--phys_cfg', type=str, default='configs/physics.yaml')
    parser.add_argument('--fusion_cfg', type=str, default='configs/fusion/adaptive_fusion.yaml')
    parser.add_argument('--gat_cfg', type=str, default='configs/gnn/gatv2.yaml')
    args = parser.parse_args()

    base_cfg = load_config(args.base_cfg)
    device = get_device(base_cfg.project.device)
    
    # Parse coords
    sy, sx = map(int, args.start.split(','))
    gy, gx = map(int, args.goal.split(','))
    
    # Load image
    img = Image.open(args.image_path).convert('L')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0) # (1, H, W)
    
    # Pipeline
    pipeline = PA_GNN_Pipeline(
        base_cfg, load_config(args.cnn_cfg), load_config(args.phys_cfg),
        load_config(args.fusion_cfg), load_config(args.gat_cfg), device
    )
    
    print(f"Running PA-GNN Pipeline (Baseline: {args.baseline})")
    path_details, data, fusion_dict, _ = pipeline.run(
        img_tensor, 
        start_coords=(sy, sx), 
        goal_coords=(gy, gx), 
        run_baseline=args.baseline
    )
    
    out_dir = Path("results/inference")
    ensure_dir(out_dir)
    
    out_file = out_dir / f"path_{args.baseline}.png"
    plot_path_on_image(img_tensor, data.label_map, path_details, out_file, title=f"PA-GNN ({args.baseline})")
    
    print(f"Path plotted to {out_file}")
    
    if path_details:
        print(f"Path found with {len(path_details)} waypoints.")
        max_risk = max(p['risk'] for p in path_details)
        print(f"Max Risk on Path: {max_risk:.4f}")

if __name__ == "__main__":
    main()
