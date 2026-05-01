"""
Label remapping transforms.
Converts discrete terrain class labels to continuous risk scores [0, 1].

AI4Mars NAV labels (pixel values 0-3, 255=null):
    0 (soil)     → 0.1  (safe)
    1 (bedrock)  → 0.5  (uncertain)
    2 (sand)     → 0.4  (slip risk)
    3 (big_rock) → 0.9  (hazardous)
    255 (null)   → -1   (ignore in loss)

HiRISE v3 landmark labels (image-level class 0-7):
    0 (other)         → 0.15  (safe)
    1 (crater)        → 0.90  (hazardous)
    2 (dark dune)     → 0.85  (hazardous)
    3 (slope streak)  → 0.80  (hazardous)
    4 (bright dune)   → 0.50  (uncertain)
    5 (impact ejecta) → 0.55  (uncertain)
    6 (swiss cheese)  → 0.85  (hazardous)
    7 (spider)        → 0.45  (uncertain)
"""

import numpy as np


class AI4MarsLabelRemapper:
    """
    Remap AI4Mars NAV pixel-level labels to continuous risk scores.
    
    Input:  uint8 label mask (H, W) with values {0, 1, 2, 3, 255}
    Output: float32 risk map (H, W) with continuous values; -1 for ignore regions
    """
    
    # Default risk mapping (can be overridden via config)
    DEFAULT_RISK_MAP = {
        0: 0.1,    # soil → safe
        1: 0.5,    # bedrock → uncertain
        2: 0.4,    # sand → slip risk
        3: 0.9,    # big_rock → hazardous
    }
    NULL_VALUE = 255
    IGNORE_RISK = -1.0
    
    def __init__(self, risk_map=None, null_value=255):
        """
        Args:
            risk_map: Dict mapping pixel value → risk score. 
                      If None, uses defaults from config.
            null_value: Pixel value to treat as ignore (default 255).
        """
        self.risk_map = risk_map if risk_map is not None else self.DEFAULT_RISK_MAP
        self.null_value = null_value
        
        # Build lookup table for fast vectorized remapping (256 entries)
        self._lut = np.full(256, self.IGNORE_RISK, dtype=np.float32)
        for pixel_val, risk_score in self.risk_map.items():
            self._lut[pixel_val] = risk_score
    
    @classmethod
    def from_config(cls, cfg):
        """
        Create remapper from dataset config.
        
        Args:
            cfg: ConfigDict with label_remap section containing
                 soil, bedrock, sand, big_rock, null_value keys.
        """
        remap = cfg.label_remap
        risk_map = {
            0: remap.soil,
            1: remap.bedrock,
            2: remap.sand,
            3: remap.big_rock,
        }
        return cls(risk_map=risk_map, null_value=remap.null_value)
    
    def __call__(self, label_mask):
        """
        Remap label mask to risk scores.
        
        Args:
            label_mask: numpy array (H, W), dtype uint8, values in {0,1,2,3,255}.
            
        Returns:
            risk_map: numpy array (H, W), dtype float32. 
                      Valid pixels in [0, 1], ignore pixels = -1.
        """
        return self._lut[label_mask]
    
    def get_ignore_mask(self, label_mask):
        """Return boolean mask where True = ignore (null/rover/out-of-range)."""
        return label_mask == self.null_value
    
    def get_dominant_class(self, label_mask):
        """
        Get the dominant (most frequent) terrain class in a label mask,
        excluding null pixels. Used for stratified splitting.
        
        Args:
            label_mask: numpy array (H, W), dtype uint8.
            
        Returns:
            int: dominant class index (0-3), or -1 if all null.
        """
        valid = label_mask[label_mask != self.null_value]
        if len(valid) == 0:
            return -1
        values, counts = np.unique(valid, return_counts=True)
        return int(values[np.argmax(counts)])
    
    def get_risk_category(self, risk_score):
        """Categorize a risk score into safe/uncertain/hazardous."""
        if risk_score < 0:
            return "ignore"
        elif risk_score < 0.3:
            return "safe"
        elif risk_score < 0.7:
            return "uncertain"
        else:
            return "hazardous"


class HiRISELabelRemapper:
    """
    Remap HiRISE v3 image-level landmark class to continuous risk score.
    
    Input:  integer class index (0-7)
    Output: float risk score in [0, 1]
    """
    
    # Default risk mapping (8 real classes from classmap)
    DEFAULT_RISK_MAP = {
        0: 0.15,   # other → safe
        1: 0.90,   # crater → hazardous
        2: 0.85,   # dark dune → hazardous
        3: 0.80,   # slope streak → hazardous
        4: 0.50,   # bright dune → uncertain
        5: 0.55,   # impact ejecta → uncertain
        6: 0.85,   # swiss cheese → hazardous
        7: 0.45,   # spider → uncertain
    }
    
    CLASS_NAMES = [
        "other", "crater", "dark_dune", "slope_streak",
        "bright_dune", "impact_ejecta", "swiss_cheese", "spider"
    ]
    
    def __init__(self, risk_map=None):
        """
        Args:
            risk_map: Dict mapping class index → risk score.
                      If None, uses defaults.
        """
        self.risk_map = risk_map if risk_map is not None else self.DEFAULT_RISK_MAP
    
    @classmethod
    def from_config(cls, cfg):
        """
        Create remapper from dataset config.
        
        Args:
            cfg: ConfigDict with label_remap section containing
                 other, crater, dark_dune, ... keys.
        """
        remap = cfg.label_remap
        risk_map = {
            0: remap.other,
            1: remap.crater,
            2: remap.dark_dune,
            3: remap.slope_streak,
            4: remap.bright_dune,
            5: remap.impact_ejecta,
            6: remap.swiss_cheese,
            7: remap.spider,
        }
        return cls(risk_map=risk_map)
    
    def __call__(self, class_index):
        """
        Remap a class index to risk score.
        
        Args:
            class_index: int, class label 0-7.
            
        Returns:
            float: risk score in [0, 1].
        """
        if class_index not in self.risk_map:
            raise ValueError(
                f"Unknown HiRISE class index: {class_index}. "
                f"Expected 0-7."
            )
        return self.risk_map[class_index]
    
    def remap_batch(self, class_indices):
        """
        Remap a batch of class indices to risk scores.
        
        Args:
            class_indices: array-like of int class labels.
            
        Returns:
            numpy array of float risk scores.
        """
        return np.array([self(idx) for idx in class_indices], dtype=np.float32)
    
    def class_name(self, class_index):
        """Get the human-readable name for a class index."""
        return self.CLASS_NAMES[class_index]
    
    def get_risk_category(self, class_index):
        """Categorize a class into safe/uncertain/hazardous."""
        risk = self(class_index)
        if risk < 0.3:
            return "safe"
        elif risk < 0.7:
            return "uncertain"
        else:
            return "hazardous"
