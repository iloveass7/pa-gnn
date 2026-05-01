"""
Training augmentations for image + label pairs.
All spatial transforms are applied identically to both image and label.
Intensity transforms are applied to image only.

Augmentations:
    - Horizontal flip (p=0.5)
    - Vertical flip (p=0.5)
    - Random rotation (±15°)
    - Brightness jitter (±20%)
    - Contrast jitter (±20%)
    - Gaussian noise (σ ~ U(0, 0.02))
"""

import numpy as np
from PIL import Image, ImageEnhance


class JointAugmentation:
    """
    Augmentation pipeline that applies identical spatial transforms to
    both image and label, and intensity transforms to image only.
    
    Expects:
        image: numpy array (H, W), float32, normalized [0, 1]
        label: numpy array (H, W), uint8 (class indices) or float32 (risk scores)
    """
    
    def __init__(
        self,
        horizontal_flip=0.5,
        vertical_flip=0.5,
        rotation_degrees=15,
        brightness_range=0.2,
        contrast_range=0.2,
        gaussian_noise_sigma_max=0.02,
        enabled=True,
    ):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_degrees = rotation_degrees
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gaussian_noise_sigma_max = gaussian_noise_sigma_max
        self.enabled = enabled
    
    @classmethod
    def from_config(cls, cfg):
        """
        Create augmentation pipeline from dataset config.
        
        Args:
            cfg: ConfigDict with augmentation section.
        """
        aug = cfg.augmentation
        return cls(
            horizontal_flip=aug.horizontal_flip,
            vertical_flip=aug.vertical_flip,
            rotation_degrees=aug.rotation_degrees,
            brightness_range=aug.brightness_range,
            contrast_range=aug.contrast_range,
            gaussian_noise_sigma_max=aug.gaussian_noise_sigma_max,
            enabled=aug.enabled,
        )
    
    def __call__(self, image, label):
        """
        Apply augmentations to image-label pair.
        
        Args:
            image: numpy array (H, W), float32, values in [0, 1].
            label: numpy array (H, W), uint8 or float32.
            
        Returns:
            Tuple of (augmented_image, augmented_label).
        """
        if not self.enabled:
            return image, label
        
        # --- Spatial transforms (applied to both) ---
        
        # Horizontal flip
        if np.random.random() < self.horizontal_flip:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        
        # Vertical flip
        if np.random.random() < self.vertical_flip:
            image = np.flipud(image).copy()
            label = np.flipud(label).copy()
        
        # Random rotation
        if self.rotation_degrees > 0:
            angle = np.random.uniform(-self.rotation_degrees, self.rotation_degrees)
            if abs(angle) > 0.5:  # Skip trivial rotations
                image, label = self._rotate_pair(image, label, angle)
        
        # --- Intensity transforms (image only) ---
        
        # Brightness jitter
        if self.brightness_range > 0:
            factor = 1.0 + np.random.uniform(
                -self.brightness_range, self.brightness_range
            )
            image = np.clip(image * factor, 0.0, 1.0)
        
        # Contrast jitter
        if self.contrast_range > 0:
            factor = 1.0 + np.random.uniform(
                -self.contrast_range, self.contrast_range
            )
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0.0, 1.0)
        
        # Gaussian noise
        if self.gaussian_noise_sigma_max > 0:
            sigma = np.random.uniform(0, self.gaussian_noise_sigma_max)
            if sigma > 0:
                noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
                image = np.clip(image + noise, 0.0, 1.0)
        
        return image, label
    
    def _rotate_pair(self, image, label, angle):
        """
        Rotate image and label by the same angle.
        Image uses bilinear; label uses nearest-neighbour.
        """
        h, w = image.shape[:2]
        
        # Convert image to PIL for rotation
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='L')
        rotated_img = pil_img.rotate(
            angle, resample=Image.BILINEAR, fillcolor=0
        )
        result_img = np.array(rotated_img, dtype=np.float32) / 255.0
        
        # Rotate label with nearest-neighbour
        is_float_label = label.dtype in [np.float32, np.float64]
        if is_float_label:
            # For risk maps: handle ignore values (-1)
            ignore_mask = label < 0
            safe_label = label.copy()
            safe_label[ignore_mask] = 0.0
            
            lbl_uint8 = (safe_label * 255).clip(0, 255).astype(np.uint8)
            pil_lbl = Image.fromarray(lbl_uint8, mode='L')
            rotated_lbl = pil_lbl.rotate(
                angle, resample=Image.NEAREST, fillcolor=0
            )
            result_lbl = np.array(rotated_lbl, dtype=np.float32) / 255.0
            
            # Restore ignore mask (also rotated)
            mask_uint8 = (ignore_mask.astype(np.uint8) * 255)
            pil_mask = Image.fromarray(mask_uint8, mode='L')
            rotated_mask = pil_mask.rotate(
                angle, resample=Image.NEAREST, fillcolor=255
            )
            result_mask = np.array(rotated_mask) > 127
            result_lbl[result_mask] = -1.0
        else:
            # For integer class labels: fill rotated borders with 255 (null)
            pil_lbl = Image.fromarray(label.astype(np.uint8), mode='L')
            rotated_lbl = pil_lbl.rotate(
                angle, resample=Image.NEAREST, fillcolor=255
            )
            result_lbl = np.array(rotated_lbl, dtype=label.dtype)
        
        return result_img, result_lbl


class NoAugmentation:
    """Identity augmentation (pass-through) for val/test splits."""
    
    def __call__(self, image, label):
        return image, label
