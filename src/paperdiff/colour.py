from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab

__all__ = ["rgb_to_lab", "compute_colour_deltas"]


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert clipped grayscale or RGB image data to CIELAB."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return rgb2lab(np.clip(array, 0.0, 1.0))


def compute_colour_deltas(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Calculate per-pixel Euclidean CIELAB differences between matching images."""
    reference_array = np.asarray(reference, dtype=np.float32)
    candidate_array = np.asarray(candidate, dtype=np.float32)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(f"Image shapes do not match: {reference_array.shape} vs {candidate_array.shape}")

    reference_lab = rgb_to_lab(reference_array)
    candidate_lab = rgb_to_lab(candidate_array)
    delta = np.sqrt(
        (reference_lab[..., 0] - candidate_lab[..., 0]) ** 2
        + (reference_lab[..., 1] - candidate_lab[..., 1]) ** 2
        + (reference_lab[..., 2] - candidate_lab[..., 2]) ** 2
    )
    return delta.astype(np.float32)
