"""
File I/O utilities.
Helpers for saving/loading numpy arrays, images, JSON, and model checkpoints.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def ensure_dir(path):
    """Create directory if it doesn't exist. Returns the Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_numpy(array, path):
    """Save a numpy array to .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)


def load_numpy(path):
    """Load a numpy array from .npy file."""
    return np.load(str(path))


def save_json(data, path, indent=2):
    """Save a dictionary to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path):
    """Load a dictionary from JSON file."""
    with open(Path(path), 'r', encoding='utf-8') as f:
        return json.load(f)


def load_image_grayscale(path):
    """
    Load an image as grayscale numpy array (H, W), float32, values 0-255.
    
    Args:
        path: Path to image file.
        
    Returns:
        numpy array of shape (H, W), dtype float32.
    """
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32)


def load_image_rgb(path):
    """
    Load an image as RGB numpy array (H, W, 3), float32, values 0-255.
    
    Args:
        path: Path to image file.
        
    Returns:
        numpy array of shape (H, W, 3), dtype float32.
    """
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32)


def load_label_mask(path):
    """
    Load a label mask as integer numpy array (H, W).
    
    Args:
        path: Path to label mask .png file.
        
    Returns:
        numpy array of shape (H, W), dtype uint8.
    """
    img = Image.open(path)
    return np.array(img, dtype=np.uint8)


def save_image(array, path):
    """
    Save a numpy array as an image file.
    
    Args:
        array: numpy array (H, W) or (H, W, 3), values in [0, 255] or [0, 1].
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if array.dtype in [np.float32, np.float64]:
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
    
    Image.fromarray(array).save(path)


def list_files(directory, extensions=None, recursive=False):
    """
    List files in a directory, optionally filtering by extension.
    
    Args:
        directory: Path to directory.
        extensions: List of extensions (e.g., ['.jpg', '.png']). None = all files.
        recursive: Search subdirectories.
        
    Returns:
        Sorted list of Path objects.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    if recursive:
        files = directory.rglob('*')
    else:
        files = directory.iterdir()
    
    files = [f for f in files if f.is_file()]
    
    if extensions:
        extensions = [ext.lower() for ext in extensions]
        files = [f for f in files if f.suffix.lower() in extensions]
    
    return sorted(files)
