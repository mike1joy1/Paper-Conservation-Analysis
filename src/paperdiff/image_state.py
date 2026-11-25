from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image
from PySide6.QtGui import QColor
from scipy.ndimage import binary_erosion

from .loading import load_image

__all__ = ["analysis_options", "image_data_colour", "mask_to_colour_image"]


def mask_to_colour_image(paper_mask: np.ndarray, ink_mask: np.ndarray) -> np.ndarray:
    """Encode paper and ink membership as an 8-bit RGB image."""
    paper = np.asarray(paper_mask, dtype=bool)
    ink = np.asarray(ink_mask, dtype=bool)
    if paper.ndim == 3:
        paper = paper[..., 0]
    if ink.ndim == 3:
        ink = ink[..., 0]
    if paper.ndim != 2 or ink.shape != paper.shape:
        raise ValueError("Paper and ink masks must have matching two-dimensional shapes.")

    image = np.empty((*paper.shape, 3), dtype=np.uint8)
    image[:] = (255, 0, 0)
    image[paper & ~ink] = (255, 255, 255)
    image[~paper & ink] = (0, 0, 0)
    image[paper & ink] = (128, 128, 128)
    return image


class analysis_options:
    """Container for all analysis option settings used by the UI."""

    def __init__(self):
        """Initialise analysis settings and intersection-mask state."""
        self.show_analysis_options = False
        self.show_image_data = False
        
        self.rgb_comparison = True
        self.lab_comparison = True
        self.use_masks = True
        self.intersection_mask = True
        self.images_aligned = False
        self.mean_comparison = True
        self.per_pixel_comparison = False
        self.heatmap_grid_size_x = 50
        self.heatmap_grid_size_y = 50
        self.dec_places = 1

        self.int_paper_mask = None
        self.int_ink_mask = None

    def to_dict(self):
        """Return the reportable analysis settings as a dictionary."""
        details = {
            "show_analysis_options": self.show_analysis_options,
            "show_image_data": self.show_image_data,
            "rgb_comparison": self.rgb_comparison,
            "lab_comparison": self.lab_comparison,
            "use_masks": self.use_masks,
            "intersection_mask": self.intersection_mask,
            "images_aligned": self.images_aligned,
            "mean_comparison": self.mean_comparison,
            "per_pixel_comparison": self.per_pixel_comparison,
            "heatmap_grid_size_x": self.heatmap_grid_size_x,
            "heatmap_grid_size_y": self.heatmap_grid_size_y
        }
        return details

    def __str__(self) -> str:
        """Describe the current analysis settings for display or debugging."""
        details = {
            "show_analysis_options": self.show_analysis_options,
            "show_image_data": self.show_image_data,
            "rgb_comparison": self.rgb_comparison,
            "lab_comparison": self.lab_comparison,
            "use_masks": self.use_masks,
            "intersection_mask": self.intersection_mask,
            "images_aligned": self.images_aligned,
            "mean_comparison": self.mean_comparison,
            "per_pixel_comparison": self.per_pixel_comparison,
            "heatmap_grid_size_x": self.heatmap_grid_size_x,
            "heatmap_grid_size_y": self.heatmap_grid_size_y,
        }
        return f"analysis_options({details})"

    def sync_from_ui(self, **values):
        """Update known settings from UI values."""
        for key, value in values.items():
            if hasattr(self, key):
                setattr(self, key, value)


class image_data_colour:
    """Container for a single image and the related pixel and mask state."""

    def __init__(self, key: str, file_path: str | None = None):
        """Initialise an image's pixels, masks, and reference colours."""
        self.key = key
        self.file_path = file_path
        self.pixels: dict[str, np.ndarray] = {}
        self.masks: dict[str, np.ndarray] = {}
        self.mask_border: np.ndarray | None = None
        self.masks_exists: dict[str, bool] = {"paper": False, "ink": False}
        self.reference_colours = {
            "paper": QColor("white"),
            "ink": QColor("black"),
        }
        self._reset_masks_to_full()

    def __str__(self) -> str:
        """Summarise image array shapes, masks, and reference colours."""
        pixels_summary = {name: np.asarray(value).shape if value is not None else None for name, value in self.pixels.items()}
        masks_summary = {name: np.asarray(value).shape if value is not None else None for name, value in self.masks.items()}
        details = {
            "key": self.key,
            "file_path": self.file_path,
            "pixels": pixels_summary,
            "masks": masks_summary,
            "mask_border": None if self.mask_border is None else np.asarray(self.mask_border).shape,
            "masks_exists": self.masks_exists,
            "reference_colours": {name: colour.name() for name, colour in self.reference_colours.items()},
        }
        return f"image_data_colour({details})"

    @staticmethod
    def _mask_to_rgb(mask: np.ndarray | None) -> np.ndarray | None:
        """Convert a mask to a floating-point three-channel representation."""
        if mask is None:
            return None
        boolean_mask = np.asarray(mask, dtype=bool)
        if boolean_mask.ndim == 2:
            return np.stack([boolean_mask.astype(np.float32)] * 3, axis=-1)
        if boolean_mask.ndim == 3 and boolean_mask.shape[-1] == 1:
            return np.repeat(boolean_mask.astype(np.float32), 3, axis=-1)
        return boolean_mask.astype(np.float32)

    def _reset_masks_to_full(self, check_cancelled: Callable[[], None] | None = None):
        """Reset paper and ink masks to include every loaded pixel."""
        if "rgb" in self.pixels:
            shape = np.asarray(self.pixels["rgb"]).shape[:2]
        else:
            shape = (0, 0)

        if shape[0] == 0 or shape[1] == 0:
            self.masks = {"paper": None, "ink": None}
            self.mask_border = np.zeros((0, 0), dtype=bool)
            self.masks_exists = {"paper": False, "ink": False}
            return

        full_mask = np.ones(shape, dtype=bool)
        if check_cancelled is not None:
            check_cancelled()
        paper_mask = self._mask_to_rgb(full_mask)
        if check_cancelled is not None:
            check_cancelled()
        ink_mask = self._mask_to_rgb(full_mask)
        if check_cancelled is not None:
            check_cancelled()
        self.masks = {
            "paper": paper_mask,
            "ink": ink_mask,
        }
        self.mask_border = self._mask_to_rgb(np.zeros(shape, dtype=bool))
        self.masks_exists = {"paper": False, "ink": False}

    def clear_masks(self):
        """Remove generated masks and restore full-image masks."""
        self._reset_masks_to_full()

    def get_pixels(self, colourspace: str):
        """Return pixels in the requested colour space, generating LAB if needed."""
        if colourspace in self.pixels:
            return np.asarray(self.pixels[colourspace], dtype=np.float32)
        if colourspace == "lab":
            return self.generate_lab_pixels()
        raise KeyError(f"Unknown colourspace '{colourspace}'")

    def set_mask(self, name: str, mask: np.ndarray):
        """Store a paper or ink mask in three-channel form."""
        if name not in {"paper", "ink"}:
            raise ValueError(f"Unsupported mask '{name}'")
        array = self._mask_to_rgb(mask)
        self.masks[name] = array
        self.masks_exists[name] = True

    def export_mask(self, path: str):
        """Save the current paper and ink masks as a colour-coded PNG."""
        if self.pixels.get("rgb") is None:
            raise ValueError("Load an image before exporting its mask.")
        image = mask_to_colour_image(self.masks["paper"], self.masks["ink"])
        Image.fromarray(image).save(path, format="PNG")

    def import_mask(self, path: str):
        """Import a matching-resolution colour mask and replace both region masks."""
        rgb = self.get_pixels("rgb")
        with Image.open(path) as image:
            if image.size != (rgb.shape[1], rgb.shape[0]):
                raise ValueError(
                    f"Mask resolution {image.width} x {image.height} does not match "
                    f"image resolution {rgb.shape[1]} x {rgb.shape[0]}."
                )
            pixels = np.asarray(image.convert("RGB"))
        white = np.all(pixels == (255, 255, 255), axis=-1)
        black = np.all(pixels == (0, 0, 0), axis=-1)
        grey = np.all(pixels == (128, 128, 128), axis=-1)
        paper = white | grey
        ink = black | grey
        masks = {"paper": self._mask_to_rgb(paper), "ink": self._mask_to_rgb(ink)}
        border = self._mask_to_rgb(~paper & ~ink)
        self.masks = masks
        self.mask_border = border
        self.masks_exists = {"paper": True, "ink": True}

    def load_from_path(
        self, path: str, check_cancelled: Callable[[], None] | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        """Load RGB pixels, prepare LAB pixels, and reset masks for an image."""
        self.file_path = path
        if check_cancelled is not None:
            check_cancelled()
        image = load_image(path)
        if check_cancelled is not None:
            check_cancelled()
        if progress is not None:
            progress("Preparing RGB pixels")
        rgb = np.asarray(image, dtype=np.float32)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        if rgb.size > 0 and rgb.max() > 1.0:
            rgb = rgb / 255.0
        self.pixels["rgb"] = np.clip(rgb, 0.0, 1.0)
        if check_cancelled is not None:
            check_cancelled()
        if progress is not None:
            progress("Converting to CIELAB")
        self.generate_lab_pixels(check_cancelled=check_cancelled)
        if check_cancelled is not None:
            check_cancelled()
        if progress is not None:
            progress("Initialising masks")
        self._reset_masks_to_full(check_cancelled=check_cancelled)
        if check_cancelled is not None:
            check_cancelled()
        return self.pixels["rgb"]

    def generate_lab_pixels(self, check_cancelled: Callable[[], None] | None = None):
        """Convert stored RGB pixels to CIELAB and cache the result."""
        if "rgb" not in self.pixels:
            return None

        rgb = np.asarray(self.pixels["rgb"], dtype=np.float32)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        if rgb.size == 0:
            self.pixels["lab"] = rgb.copy()
            return self.pixels["lab"]

        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        if check_cancelled is not None:
            check_cancelled()
        r, g, b = rgb_linear[..., 0], rgb_linear[..., 1], rgb_linear[..., 2]
        x = 0.4124 * r + 0.3576 * g + 0.1805 * b
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        z = 0.0193 * r + 0.1192 * g + 0.9505 * b

        if check_cancelled is not None:
            check_cancelled()
        white_ref = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
        xyz = np.stack([x, y, z], axis=-1)
        xyz = xyz / white_ref
        f_xyz = np.where(xyz > 0.008856, xyz ** (1.0 / 3.0), (7.787 * xyz) + (16.0 / 116.0))

        if check_cancelled is not None:
            check_cancelled()
        l_channel = 116.0 * f_xyz[..., 1] - 16.0
        a_channel = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
        b_channel = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
        lab = np.stack([l_channel, a_channel, b_channel], axis=-1)
        self.pixels["lab"] = lab
        return lab

    def generate_masks(self, mode: str, threshold: int = 140, reference_weight: float = 0.5, border_pixels: int = 2):
        """Classify paper and ink pixels and exclude the requested boundary width."""
        rgb_array = self.get_pixels("rgb")
        if rgb_array is None:
            return None, None, None

        if mode == "luminance":
            brightness = np.mean(rgb_array, axis=-1)
            threshold_value = threshold / 255.0
            ink_mask = brightness <= threshold_value
        else:
            paper_colour = np.array(self.reference_colours["paper"].getRgb()[:3], dtype=np.float32) / 255.0
            ink_colour = np.array(self.reference_colours["ink"].getRgb()[:3], dtype=np.float32) / 255.0
            paper_distance = np.linalg.norm(rgb_array - paper_colour, axis=-1)
            ink_distance = np.linalg.norm(rgb_array - ink_colour, axis=-1)
            max_distance = np.sqrt(3.0)
            paper_weight = float(reference_weight)
            adjusted_paper_distance = paper_weight * paper_distance + (1.0 - paper_weight) * max_distance
            adjusted_ink_distance = (1.0 - paper_weight) * ink_distance + paper_weight * max_distance
            ink_mask = adjusted_paper_distance > adjusted_ink_distance

        original_ink_mask = np.asarray(ink_mask, dtype=bool).copy()
        original_paper_mask = (~original_ink_mask).astype(bool)

        if border_pixels > 0:
            paper_mask = binary_erosion(original_paper_mask, iterations=border_pixels)
            ink_mask = binary_erosion(original_ink_mask, iterations=border_pixels)
            border_mask = (original_ink_mask ^ binary_erosion(original_ink_mask, iterations=1)) | (
                original_paper_mask ^ binary_erosion(original_paper_mask, iterations=1)
            )
        else:
            paper_mask = original_paper_mask
            ink_mask = original_ink_mask
            border_mask = np.zeros_like(original_ink_mask, dtype=bool)

        self.masks["paper"] = self._mask_to_rgb(np.asarray(paper_mask, dtype=bool))
        self.masks["ink"] = self._mask_to_rgb(np.asarray(ink_mask, dtype=bool))
        self.mask_border = self._mask_to_rgb(np.asarray(border_mask, dtype=bool))
        self.masks_exists["paper"] = True
        self.masks_exists["ink"] = True
        return self.masks["paper"], self.masks["ink"], self.mask_border
