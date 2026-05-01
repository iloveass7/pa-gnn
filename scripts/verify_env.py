"""Quick environment verification script."""
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import torchvision
print(f"TorchVision: {torchvision.__version__}")

import torch_geometric
print(f"PyG: {torch_geometric.__version__}")

import cv2
print(f"OpenCV: {cv2.__version__}")

import skimage
print(f"scikit-image: {skimage.__version__}")

import networkx
print(f"NetworkX: {networkx.__version__}")

import numpy as np
print(f"NumPy: {np.__version__}")

import scipy
print(f"SciPy: {scipy.__version__}")

import matplotlib
print(f"Matplotlib: {matplotlib.__version__}")

import pandas
print(f"Pandas: {pandas.__version__}")

import yaml
print(f"PyYAML: {yaml.__version__}")

print("\nAll packages OK!")
