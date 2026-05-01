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
