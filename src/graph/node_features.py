import numpy as np
from skimage.measure import regionprops


def compute_node_features(label_map, image, h_physics, h_learned, h_final, alpha_map,
                          slope, roughness, discontinuity, hazard_threshold=0.7, target=None):
    """
    Compute 14-dim feature vector per superpixel node.

    Feature index:
        0  mean_intensity    [0, 1]
        1  intensity_std     [0, 1]
        2  mean_S  (slope)   [0, 1]
        3  mean_R            [0, 1]
        4  mean_D            [0, 1]
        5  mean_H_physics    [0, 1]
        6  mean_H_learned    [0, 1]
        7  mean_H_final      [0, 1]
        8  mean_alpha        [0, 1]
        9  entropy           [0, 1]  (binary entropy of H_learned, max=1 at p=0.5)
        10 centroid_x        [0, 1]  normalized by W
        11 centroid_y        [0, 1]  normalized by H
        12 area              [0, 1]  normalized by H*W
        13 haz_neighbour_cnt [0, 1]  populated + normalized by adjacency.py

    All features are [0, 1] after normalization so GATv2 attention is not
    dominated by scale differences (M5 fix).
    """
    # Handle batched input (1, C, H, W) or (1, H, W)
    while image.dim() > 3:
        image = image.squeeze(0)

    if image.size(0) == 3:
        img_gray = image.mean(dim=0).cpu().numpy()
    else:
        img_gray = image.squeeze(0).cpu().numpy()

    H, W = img_gray.shape
    total_pixels = float(H * W)

    def _to_2d(t):
        """Squeeze any leading batch/channel dims down to (H, W)."""
        arr = t.cpu().numpy()
        while arr.ndim > 2:
            arr = arr.squeeze(0)
        return arr

    s_np     = _to_2d(slope)
    r_np     = _to_2d(roughness)
    d_np     = _to_2d(discontinuity)
    h_p_np   = _to_2d(h_physics)
    h_l_np   = _to_2d(h_learned)
    h_f_np   = _to_2d(h_final)
    alpha_np = _to_2d(alpha_map)

    if target is not None:
        target_np = _to_2d(target) if target.dim() >= 2 else target.cpu().numpy()
    else:
        target_np = None

    # Binary entropy of H_learned — peaks at 1.0 when p=0.5 (maximum uncertainty)
    p       = np.clip(h_l_np, 1e-6, 1.0 - 1e-6)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)   # already in [0, 1]

    props       = regionprops(label_map)
    N           = len(props)

    features     = np.zeros((N, 14), dtype=np.float32)
    centroids    = np.zeros((N,  2), dtype=np.float32)
    is_hazardous = np.zeros(N, dtype=bool)
    node_targets = np.zeros(N, dtype=np.float32) if target_np is not None else None

    for i, prop in enumerate(props):
        mask = label_map == prop.label

        if target_np is not None:
            valid_pixels = target_np[mask] >= 0
            if valid_pixels.sum() > 0:
                node_targets[i] = float(target_np[mask][valid_pixels].mean())
            else:
                node_targets[i] = -1.0

        cy, cx = prop.centroid       # (row=y, col=x)
        area   = prop.area

        # ── Raw features ──────────────────────────────────────────────────
        features[i, 0]  = img_gray[mask].mean()         # mean_intensity   [0,1]
        features[i, 1]  = img_gray[mask].std()          # intensity_std    [0,1]
        features[i, 2]  = s_np[mask].mean()             # mean_S           [0,1]
        features[i, 3]  = r_np[mask].mean()             # mean_R           [0,1]
        features[i, 4]  = d_np[mask].mean()             # mean_D           [0,1]
        features[i, 5]  = h_p_np[mask].mean()           # mean_H_physics   [0,1]
        features[i, 6]  = h_l_np[mask].mean()           # mean_H_learned   [0,1]
        features[i, 7]  = h_f_np[mask].mean()           # mean_H_final     [0,1]
        features[i, 8]  = alpha_np[mask].mean()         # mean_alpha       [0,1]
        features[i, 9]  = entropy[mask].mean()          # entropy          [0,1]
        features[i, 10] = cx / W                        # centroid_x → [0,1] (M5)
        features[i, 11] = cy / H                        # centroid_y → [0,1] (M5)
        features[i, 12] = area / total_pixels           # area       → [0,1] (M5)
        # features[i, 13] = haz_neighbour_cnt populated + normalized by adjacency.py

        centroids[i] = [cy, cx]

        if features[i, 7] > hazard_threshold:
            is_hazardous[i] = True

    # Safety clamp: all features must be in [0, 1] (guards against numerical noise)
    features = np.clip(features, 0.0, 1.0)

    return features, centroids, is_hazardous, node_targets

