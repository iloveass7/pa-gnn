import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import torch

def plot_path_on_image(image, label_map, path_details, save_path, title="Path Plan"):
    """
    Overlay path on original image with risk coloring.
    """
    if isinstance(image, torch.Tensor):
        img_np = image.mean(dim=0).cpu().numpy()
    else:
        img_np = image
        
    if isinstance(label_map, torch.Tensor):
        label_map = label_map.cpu().numpy()
        
    # Mark boundaries
    bound_img = mark_boundaries(img_np, label_map, color=(1, 0, 0), mode='inner', alpha=0.3)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bound_img)
    ax.set_title(title)
    ax.axis('off')
    
    if path_details is not None and len(path_details) > 0:
        # Extract coordinates (y, x) -> (x, y) for plotting
        xs = [p['pos'][1] for p in path_details]
        ys = [p['pos'][0] for p in path_details]
        risks = [p['risk'] for p in path_details]
        
        # Plot path lines
        ax.plot(xs, ys, color='white', linewidth=2, linestyle='--', alpha=0.8)
        
        # Plot waypoints colored by risk
        sc = ax.scatter(xs, ys, c=risks, cmap=plt.cm.RdYlGn_r, vmin=0, vmax=1, s=50, edgecolors='black', zorder=5)
        
        # Mark start and goal
        ax.plot(xs[0], ys[0], 'bo', markersize=12, label='Start')
        ax.plot(xs[-1], ys[-1], 'go', markersize=12, label='Goal')
        
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Waypoint Risk')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "NO PATH FOUND", color='red', fontsize=20, ha='center', va='center', transform=ax.transAxes)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
