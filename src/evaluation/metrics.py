import torch

def compute_metrics(pred, target, hazard_threshold=0.5):
    """
    Compute metrics for continuous risk prediction.
    Ignore regions where target < 0.
    """
    if isinstance(pred, dict):
        pred = pred['h_final']
        
    valid_mask = target >= 0
    if valid_mask.sum() == 0:
        return {'hazard_recall': 0.0, 'iou': 0.0, 'ece': 0.0}
        
    p_flat = pred[valid_mask]
    t_flat = target[valid_mask]
    
    # Hazard Recall
    t_hazard = t_flat > 0.7
    p_hazard = p_flat > hazard_threshold
    
    tp = (p_hazard & t_hazard).sum().float()
    fn = (~p_hazard & t_hazard).sum().float()
    hazard_recall = tp / (tp + fn + 1e-8)
    
    # IoU
    fp = (p_hazard & ~t_hazard).sum().float()
    iou = tp / (tp + fp + fn + 1e-8)
    
    # ECE (Expected Calibration Error) - simplistic 10 bins
    ece = 0.0
    num_bins = 10
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=pred.device)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        # Include upper bound in last bin
        if i == num_bins - 1:
            in_bin = (p_flat >= bin_lower) & (p_flat <= bin_upper)
        else:
            in_bin = (p_flat >= bin_lower) & (p_flat < bin_upper)
            
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            avg_conf_in_bin = p_flat[in_bin].mean()
            avg_acc_in_bin = t_flat[in_bin].mean()
            ece += torch.abs(avg_conf_in_bin - avg_acc_in_bin) * prop_in_bin
            
    return {
        'hazard_recall': hazard_recall.item(),
        'iou': iou.item(),
        'ece': ece.item()
    }


def aggregate_patch_risk(h_final, method='mean'):
    """
    Aggregate pixel-level H_final to a single patch-level risk score.
    Used for HiRISE cross-domain evaluation where ground truth is image-level.
    
    Args:
        h_final: torch.Tensor (B, 1, H, W) or (1, H, W) — fused risk map
        method: 'mean' (default) or 'max' (conservative)
        
    Returns:
        float: scalar risk score for the patch
    """
    if h_final.dim() == 4:
        h_final = h_final.squeeze(0)  # (1, H, W)
    if h_final.dim() == 3:
        h_final = h_final.squeeze(0)  # (H, W)
    
    if method == 'mean':
        return h_final.mean().item()
    elif method == 'max':
        return h_final.max().item()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def compute_hirise_metrics(predicted_risks, true_risks, threshold=0.5):
    """
    Compute patch-level classification metrics for HiRISE evaluation.
    
    Args:
        predicted_risks: list of float — aggregated predicted risk per crop
        true_risks: list of float — ground truth risk score per crop
        threshold: float — boundary between safe and hazardous
        
    Returns:
        dict with accuracy, hazard_recall, hazard_precision, safe_recall
    """
    import numpy as np
    
    pred = np.array(predicted_risks)
    true = np.array(true_risks)
    
    pred_haz = pred > threshold
    true_haz = true > threshold
    
    # Overall accuracy
    accuracy = (pred_haz == true_haz).mean()
    
    # Hazard recall (most important — how many real hazards did we catch?)
    tp = (pred_haz & true_haz).sum()
    fn = (~pred_haz & true_haz).sum()
    hazard_recall = tp / (tp + fn + 1e-8)
    
    # Hazard precision 
    fp = (pred_haz & ~true_haz).sum()
    hazard_precision = tp / (tp + fp + 1e-8)
    
    # Safe recall
    tn = (~pred_haz & ~true_haz).sum()
    safe_recall = tn / (tn + fp + 1e-8)
    
    return {
        'accuracy': float(accuracy),
        'hazard_recall': float(hazard_recall),
        'hazard_precision': float(hazard_precision),
        'safe_recall': float(safe_recall),
    }

