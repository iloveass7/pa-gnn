import numpy as np
from skimage.measure import regionprops

def compute_node_features(label_map, image, h_physics, h_learned, h_final, alpha_map, 
                          slope, roughness, discontinuity, hazard_threshold=0.7, target=None):
    """
    Compute 14-dim feature vector per superpixel node.
    """
    if image.size(0) == 3:
        img_gray = image.mean(dim=0).cpu().numpy()
    else:
        img_gray = image.squeeze(0).cpu().numpy()
        
    s_np = slope.squeeze(0).cpu().numpy()
    r_np = roughness.squeeze(0).cpu().numpy()
    d_np = discontinuity.squeeze(0).cpu().numpy()
    h_p_np = h_physics.squeeze(0).cpu().numpy()
    h_l_np = h_learned.squeeze(0).cpu().numpy()
    h_f_np = h_final.squeeze(0).cpu().numpy()
    alpha_np = alpha_map.squeeze(0).cpu().numpy()
    
    if target is not None:
        target_np = target.squeeze(0).cpu().numpy()
    else:
        target_np = None
    
    # Entropy of CNN prediction
    p = np.clip(h_l_np, 1e-6, 1.0 - 1e-6)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    props = regionprops(label_map)
    N = len(props)
    
    features = np.zeros((N, 14), dtype=np.float32)
    centroids = np.zeros((N, 2), dtype=np.float32)
    is_hazardous = np.zeros(N, dtype=bool)
    node_targets = np.zeros(N, dtype=np.float32) if target_np is not None else None
    
    for i, prop in enumerate(props):
        mask = label_map == prop.label
        
        if target_np is not None:
            # If there's an ignore region (-1), handle it. If mean is < 0, maybe set to -1
            valid_pixels = target_np[mask] >= 0
            if valid_pixels.sum() > 0:
                node_targets[i] = target_np[mask][valid_pixels].mean()
            else:
                node_targets[i] = -1.0 # Ignored node
                
        # 1-13 features
        features[i, :13] = [
            img_gray[mask].mean(),        # mean_intensity
            img_gray[mask].std(),         # intensity_std
            s_np[mask].mean(),            # mean_S
            r_np[mask].mean(),            # mean_R
            d_np[mask].mean(),            # mean_D
            h_p_np[mask].mean(),          # mean_H_physics
            h_l_np[mask].mean(),          # mean_H_learned
            h_f_np[mask].mean(),          # mean_H_final
            alpha_np[mask].mean(),        # mean_alpha
            entropy[mask].mean(),         # segmentation_entropy
            prop.centroid[1],             # centroid_x (col)
            prop.centroid[0],             # centroid_y (row)
            prop.area                     # area
        ]
        
        centroids[i] = [prop.centroid[0], prop.centroid[1]] # (y, x)
        
        if features[i, 7] > hazard_threshold: # mean_H_final > threshold
            is_hazardous[i] = True
            
    return features, centroids, is_hazardous, node_targets
