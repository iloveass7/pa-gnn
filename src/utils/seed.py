"""
Reproducibility utilities.
Sets random seeds for Python, NumPy, and PyTorch to ensure deterministic results.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(preference="auto"):
    """
    Get the appropriate compute device.
    
    Args:
        preference: "auto", "cpu", or "cuda".
        
    Returns:
        torch.device
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("[Device] Using CPU")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    return device
