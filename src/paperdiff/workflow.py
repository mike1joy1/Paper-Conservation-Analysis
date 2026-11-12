"""Simple analysis workflow for already-aligned paper images.

This module is intentionally small and explicit. The research assumption is that
B1, B2 and A have already been aligned and cropped before they reach this app.
The code therefore focuses only on measuring the visual difference between the
reference image and the comparison images.
"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image

from .analysis import compute_deltas, compute_ink_mask
from .image_state import analysis_options, image_data_colour

__all__ = ["compare_images", "describe_region_change"]

def save_rgb_image(arr, folder, name, ext="png"):
    """
    Save a float RGB array (H, W, 3) with values in [0, 1] as an image.

    arr    : np.ndarray, shape (H, W, 3), floats in [0, 1]
    folder : output directory (created if it doesn't exist)
    name   : file name, with or without extension
    ext    : extension used if `name` doesn't already have one
    """
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected shape (H, W, 3), got {arr.shape}")

    # Float [0, 1] -> uint8 [0, 255], clipping stray values and rounding properly
    img8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    os.makedirs(folder, exist_ok=True)
    if not os.path.splitext(name)[1]:
        name = f"{name}.{ext.lstrip('.')}"
    path = os.path.join(folder, name)

    Image.fromarray(img8, mode="RGB").save(path)

def _as_rgb(image: np.ndarray) -> np.ndarray:
    """Convert any image-like input to a float RGB array in the 0..1 range."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array.astype(np.float32)


def _summarize_metric_map(values: np.ndarray) -> dict[str, float]:
    """Return the raw statistical summary for a region. This is kept visible so
    the output remains verifiable and audit-friendly.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "median": 0.0, "n_pixels": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "n_pixels": float(values.size),
    }


def describe_region_change(metrics: dict[str, float], region_name: str = "region") -> str:
    """Translate the numerical metrics into plain-language conservation notes.

    The app reports both the science metric (ΔE00) and the underlying Lab-direction
    components: lightness (L*) and yellow-blue shift (b*). This makes the result
    understandable to a new user without requiring them to decode the raw arrays.
    """
    mean_delta = float(metrics.get("mean", 0.0))
    dL = float(metrics.get("dL_median", metrics.get("dL_mean", 0.0)))
    db = float(metrics.get("db_median", metrics.get("db_mean", 0.0)))

    brightness = "unchanged"
    if dL > 0.5:
        brightness = "lighter"
    elif dL < -0.5:
        brightness = "darker"

    yellowing = "unchanged"
    if db > 0.5:
        yellowing = "more yellow"
    elif db < -0.5:
        yellowing = "more blue"

    return (
        f"{region_name.title()}: mean colour change {mean_delta:.2f} ΔE00; "
        f"brightness change {dL:.2f} L* ({brightness}); "
        f"yellow-blue shift {db:.2f} b* ({yellowing})."
    )


def compare_images(b1: image_data_colour, b2: image_data_colour, a: image_data_colour, op: analysis_options) -> dict:
    """Compare B1 with B2 and A using the already-aligned input images.

    The workflow:
    1. Build a paper/ink mask from B1.
    2. Compute per-pixel colour deltas between B1 and each comparison image.
    3. Summarise the change in the paper and ink regions.
    4. Return the raw numbers and a simple score for each comparison.
    """

    # Output options
    results = dict()

    results["Analysis Options"] = {
        "rgb_comparison": op.rgb_comparison,
        "lab_comparison": op.lab_comparison,
        "use_masks": op.use_masks,
        "intersection_mask": op.intersection_mask,
        "images_aligned": op.images_aligned,
        "mean_comparison": op.mean_comparison,
        "per_pixel_comparison": op.per_pixel_comparison,
        "heatmap_grid_size": op.heatmap_grid_size,
        "dec_places": op.dec_places
    }

    # Get masks
    individual_paper_mask = (not op.intersection_mask) or (op.int_paper_mask is None) or (not op.images_aligned)
    individual_ink_mask = (not op.intersection_mask) or (op.int_ink_mask is None) or (not op.images_aligned)
    
    if individual_paper_mask:
        paper_mask = {
            "b1": b1.masks["paper"],
            "b2": b2.masks["paper"],
            "a": a.masks["paper"]
        }
    else:
        paper_mask = {
                "b1": op.int_paper_mask,
                "b2": op.int_paper_mask,
                "a": op.int_paper_mask
        }

    if individual_ink_mask:
        ink_mask = {
            "b1": b1.masks["ink"],
            "b2": b2.masks["ink"],
            "a": a.masks["ink"]
        }
    else:
        ink_mask = {
                "b1": op.int_ink_mask,
                "b2": op.int_ink_mask,
                "a": op.int_ink_mask
        }
    
    
    # mask test
    total_pixels = {
        "b1": np.prod(b1.pixels.get("rgb").shape[:2]),
        "b2": np.prod(b2.pixels.get("rgb").shape[:2]),
        "a": np.prod(a.pixels.get("rgb").shape[:2])
    }
    paper_pixels = {
        "b1": np.sum(paper_mask["b1"])/3,
        "b2": np.sum(paper_mask["b2"])/3,
        "a": np.sum(paper_mask["a"])/3
    }
    ink_pixels = {
        "b1": np.sum(ink_mask["b1"])/3,
        "b2": np.sum(ink_mask["b2"])/3,
        "a": np.sum(ink_mask["a"])/3
    }
    test_union_mask = {
        "b1": np.maximum(paper_mask["b1"], ink_mask["b1"]),
        "b2": np.maximum(paper_mask["b2"], ink_mask["b2"]),
        "a": np.maximum(paper_mask["a"], ink_mask["a"])
    }
    test_intersection_mask = {
        "b1": np.minimum(paper_mask["b1"], ink_mask["b1"]),
        "b2": np.minimum(paper_mask["b2"], ink_mask["b2"]),
        "a": np.minimum(paper_mask["a"], ink_mask["a"])
    }

    results["Image Data"] = {
        "Before 1": {
            "Image Array Shape": b1.get_pixels("rgb").shape,
            "Resolution": str(b1.pixels.get("rgb").shape[0]) + "x" + str(b1.pixels.get("rgb").shape[1]),
            "Total Pixels": str(np.prod(b1.pixels.get("rgb").shape[:2])),
            "Paper Mask Array Shape": paper_mask["b1"].shape,
            "Paper Pixels": str(np.sum(paper_mask["b1"])/3),
            "Paper Percentage": str((np.sum(paper_mask["b1"])/3)/np.prod(b1.pixels.get("rgb").shape[:2])*100.0),
            "Ink Mask Array Shape": ink_mask["b1"].shape,
            "Ink Pixels": str(np.sum(ink_mask["b1"])/3),
            "Ink Percentage": str((np.sum(ink_mask["b1"])/3)/np.prod(b1.pixels.get("rgb").shape[:2])*100.0),
            "Unused Pixels": str(np.prod(b1.pixels.get("rgb").shape[:2])-np.sum(test_union_mask["b1"])/3),
            "Unused Percentage": str((np.prod(b1.pixels.get("rgb").shape[:2])-np.sum(test_union_mask["b1"])/3)/np.prod(b1.pixels.get("rgb").shape[:2])*100.0),
            "Pixels in Both": str(np.sum(test_intersection_mask["b1"])/3),
            "Pixels in Both Percentage": str((np.sum(test_intersection_mask["b1"])/3)/np.prod(b1.pixels.get("rgb").shape[:2])*100.0)
        },
        "Before 2": {
            "Image Array Shape": b2.get_pixels("rgb").shape,
            "Resolution": str(b2.pixels.get("rgb").shape[0]) + "x" + str(b2.pixels.get("rgb").shape[1]),
            "Total Pixels": str(np.prod(b2.pixels.get("rgb").shape[:2])),
            "Paper Mask Array Shape": paper_mask["b2"].shape,
            "Paper Pixels": str(np.sum(paper_mask["b2"])/3),
            "Paper Percentage": str((np.sum(paper_mask["b2"])/3)/np.prod(b2.pixels.get("rgb").shape[:2])*100.0),
            "Ink Mask Array Shape": ink_mask["b2"].shape,
            "Ink Pixels": str(np.sum(ink_mask["b2"])/3),
            "Ink Percentage": str((np.sum(ink_mask["b2"])/3)/np.prod(b2.pixels.get("rgb").shape[:2])*100.0),
            "Unused Pixels": str(np.prod(b2.pixels.get("rgb").shape[:2])-np.sum(test_union_mask["b2"])/3),
            "Unused Percentage": str((np.prod(b2.pixels.get("rgb").shape[:2])-np.sum(test_union_mask["b2"])/3)/np.prod(b2.pixels.get("rgb").shape[:2])*100.0),
            "Pixels in Both": str(np.sum(test_intersection_mask["b2"])/3),
            "Pixels in Both Percentage": str((np.sum(test_intersection_mask["b2"])/3)/np.prod(b2.pixels.get("rgb").shape[:2])*100.0)
        },
        "After": {
            "Image Array Shape": a.get_pixels("rgb").shape,
            "Resolution": str(a.pixels.get("rgb").shape[0]) + "x" + str(a.pixels.get("rgb").shape[1]),
            "Total Pixels": str(np.prod(a.pixels.get("rgb").shape[:2])),
            "Paper Mask Array Shape": paper_mask["a"].shape,
            "Paper Pixels": str(np.sum(paper_mask["a"])/3),
            "Paper Percentage": str((np.sum(paper_mask["a"])/3)/np.prod(a.pixels.get("rgb").shape[:2])*100.0),
            "Ink Mask Array Shape": ink_mask["a"].shape,
            "Ink Pixels": str(np.sum(ink_mask["a"])/3),
            "Ink Percentage": str((np.sum(ink_mask["a"])/3)/np.prod(a.pixels.get("rgb").shape[:2])*100.0),
            "Unused Pixels": str(np.prod(a.pixels.get("rgb").shape[:2])-np.sum(test_union_mask["a"])/3),
            "Unused Percentage": str((np.prod(a.pixels.get("rgb").shape[:2])-np.sum(test_union_mask["a"])/3)/np.prod(a.pixels.get("rgb").shape[:2])*100.0),
            "Pixels in Both": str(np.sum(test_intersection_mask["a"])/3),
            "Pixels in Both Percentage": str((np.sum(test_intersection_mask["a"])/3)/np.prod(a.pixels.get("rgb").shape[:2])*100.0)
        }
    }

    """
    #debug test save images
    save_rgb_image(b1.get_pixels("rgb"), "debug", "b1")
    save_rgb_image(b2.get_pixels("rgb"), "debug", "b2")
    save_rgb_image(a.get_pixels("rgb"), "debug", "a")
    save_rgb_image(paper_mask["b1"], "debug", "paper_mask_b1")
    save_rgb_image(paper_mask["b2"], "debug", "paper_mask_b2")
    save_rgb_image(paper_mask["a"], "debug", "paper_mask_a")
    save_rgb_image(ink_mask["b1"], "debug", "ink_mask_b1")
    save_rgb_image(ink_mask["b2"], "debug", "ink_mask_b2")
    save_rgb_image(ink_mask["a"], "debug", "ink_mask_a")
    """

    mask_size_paper = {
        "b1": np.sum(paper_mask["b1"])/3,
        "b2": np.sum(paper_mask["b2"])/3,
        "a": np.sum(paper_mask["a"])/3
    }

    mask_size_ink = {
        "b1": np.sum(ink_mask["b1"])/3,
        "b2": np.sum(ink_mask["b2"])/3,
        "a": np.sum(ink_mask["a"])/3
    }

    dec_place = op.dec_places
    
    # RGB comparisons
    if op.rgb_comparison:
        rgb = {
            "b1": b1.get_pixels("rgb"),
            "b2": b2.get_pixels("rgb"),
            "a": a.get_pixels("rgb")
        }
        
        # mean comparison
        if op.mean_comparison:

            rgb_paper_only = {
                "b1": rgb["b1"]*paper_mask["b1"],
                "b2": rgb["b2"]*paper_mask["b2"],
                "a": rgb["a"]*paper_mask["a"]
            }
            
            """
            #debug mask apply check
            save_rgb_image(rgb_paper_only["b1"], "debug", "paper_only_b1")
            save_rgb_image(rgb_paper_only["b2"], "debug", "paper_only_b2")
            save_rgb_image(rgb_paper_only["a"], "debug", "paper_only_a")
            """

            rgb_paper_only_flat = {
                "b1": rgb_paper_only["b1"].reshape(-1, rgb_paper_only["b1"].shape[-1]),
                "b2": rgb_paper_only["b2"].reshape(-1, rgb_paper_only["b2"].shape[-1]),
                "a": rgb_paper_only["a"].reshape(-1, rgb_paper_only["a"].shape[-1])
            }

            rgb_paper_avg = {
                "b1": rgb_paper_only_flat["b1"].sum(axis=0, dtype=np.float64)/mask_size_paper["b1"],
                "b2": rgb_paper_only_flat["b2"].sum(axis=0, dtype=np.float64)/mask_size_paper["b2"],
                "a": rgb_paper_only_flat["a"].sum(axis=0, dtype=np.float64)/mask_size_paper["a"],
            }

            rgb_paper_avg["b_avg"] = (rgb_paper_avg["b1"] + rgb_paper_avg["b2"]) / 2.0
            rgb_paper_avg["b_diff"] = np.abs(rgb_paper_avg["b1"] - rgb_paper_avg["b2"])

            rgb_ink_only = {
                "b1": rgb["b1"]*ink_mask["b1"],
                "b2": rgb["b2"]*ink_mask["b2"],
                "a": rgb["a"]*ink_mask["a"]
            }

            """
            #debug mask apply check
            save_rgb_image(rgb_ink_only["b1"], "debug", "ink_only_b1")
            save_rgb_image(rgb_ink_only["b2"], "debug", "ink_only_b2")
            save_rgb_image(rgb_ink_only["a"], "debug", "ink_only_a")
            """

            rgb_ink_only_flat = {
                "b1": rgb_ink_only["b1"].reshape(-1, rgb_ink_only["b1"].shape[-1]),
                "b2": rgb_ink_only["b2"].reshape(-1, rgb_ink_only["b2"].shape[-1]),
                "a": rgb_ink_only["a"].reshape(-1, rgb_ink_only["a"].shape[-1])
            }

            rgb_ink_avg = {
                "b1": rgb_ink_only_flat["b1"].sum(axis=0, dtype=np.float64)/mask_size_ink["b1"],
                "b2": rgb_ink_only_flat["b2"].sum(axis=0, dtype=np.float64)/mask_size_ink["b2"],
                "a": rgb_ink_only_flat["a"].sum(axis=0, dtype=np.float64)/mask_size_ink["a"],
            }

            rgb_ink_avg["b_avg"] = (rgb_ink_avg["b1"] + rgb_ink_avg["b2"]) / 2.0
            rgb_ink_avg["b_diff"] = np.abs(rgb_ink_avg["b1"] - rgb_ink_avg["b2"])

            rgb_raw_results = {
                "b1": {
                    "paper": {
                        "r": rgb_paper_avg["b1"][0],
                        "g": rgb_paper_avg["b1"][1],
                        "b": rgb_paper_avg["b1"][2]
                    },
                    "ink": {
                        "r": rgb_ink_avg["b1"][0],
                        "g": rgb_ink_avg["b1"][1],
                        "b": rgb_ink_avg["b1"][2]
                    }
                },
                "b2": {
                    "paper": {
                        "r": rgb_paper_avg["b2"][0],
                        "g": rgb_paper_avg["b2"][1],
                        "b": rgb_paper_avg["b2"][2]
                    },
                    "ink": {
                        "r": rgb_ink_avg["b2"][0],
                        "g": rgb_ink_avg["b2"][1],
                        "b": rgb_ink_avg["b2"][2]
                    }
                },
                "a": {
                    "paper": {
                        "r": rgb_paper_avg["a"][0],
                        "g": rgb_paper_avg["a"][1],
                        "b": rgb_paper_avg["a"][2]
                    },
                    "ink": {
                        "r": rgb_ink_avg["a"][0],
                        "g": rgb_ink_avg["a"][1],
                        "b": rgb_ink_avg["a"][2]
                    }
                }
            }

            rgb_scale = 100.0
            for k in rgb_paper_avg: rgb_paper_avg[k] *= rgb_scale
            for k in rgb_ink_avg: rgb_ink_avg[k] *= rgb_scale

            results["RGB Average"] = {
                "(Scale": f"{0.0:.{dec_place}f} to {rgb_scale:.{dec_place}f})",
                "Paper": {
                    "Before": {
                        "R": f"{rgb_paper_avg["b_avg"][0]:{4+dec_place}.{dec_place}f} ±{rgb_paper_avg["b_diff"][0]/2.0:{4+dec_place}.{dec_place}f}",
                        "G": f"{rgb_paper_avg["b_avg"][1]:{4+dec_place}.{dec_place}f} ±{rgb_paper_avg["b_diff"][1]/2.0:{4+dec_place}.{dec_place}f}",
                        "B": f"{rgb_paper_avg["b_avg"][2]:{4+dec_place}.{dec_place}f} ±{rgb_paper_avg["b_diff"][2]/2.0:{4+dec_place}.{dec_place}f}"
                    },
                    "After": {
                        "R": f"{rgb_paper_avg["a"][0]:{4+dec_place}.{dec_place}f}",
                        "G": f"{rgb_paper_avg["a"][1]:{4+dec_place}.{dec_place}f}",
                        "B": f"{rgb_paper_avg["a"][2]:{4+dec_place}.{dec_place}f}"
                    },
                    "Difference": {
                        "R": f"{rgb_paper_avg["a"][0]-rgb_paper_avg["b_avg"][0]:+{4+dec_place}.{dec_place}f} ±{rgb_paper_avg["b_diff"][0]/2.0:{4+dec_place}.{dec_place}f}",
                        "G": f"{rgb_paper_avg["a"][1]-rgb_paper_avg["b_avg"][1]:+{4+dec_place}.{dec_place}f} ±{rgb_paper_avg["b_diff"][1]/2.0:{4+dec_place}.{dec_place}f}",
                        "B": f"{rgb_paper_avg["a"][2]-rgb_paper_avg["b_avg"][2]:+{4+dec_place}.{dec_place}f} ±{rgb_paper_avg["b_diff"][2]/2.0:{4+dec_place}.{dec_place}f}"
                    }
                },
                "Ink": {
                    "Before": {
                        "R": f"{rgb_ink_avg["b_avg"][0]:{4+dec_place}.{dec_place}f} ±{rgb_ink_avg["b_diff"][0]/2.0:{4+dec_place}.{dec_place}f}",
                        "G": f"{rgb_ink_avg["b_avg"][1]:{4+dec_place}.{dec_place}f} ±{rgb_ink_avg["b_diff"][1]/2.0:{4+dec_place}.{dec_place}f}",
                        "B": f"{rgb_ink_avg["b_avg"][2]:{4+dec_place}.{dec_place}f} ±{rgb_ink_avg["b_diff"][2]/2.0:{4+dec_place}.{dec_place}f}"
                    },
                    "After": {
                        "R": f"{rgb_ink_avg["a"][0]:{4+dec_place}.{dec_place}f}",
                        "G": f"{rgb_ink_avg["a"][1]:{4+dec_place}.{dec_place}f}",
                        "B": f"{rgb_ink_avg["a"][2]:{4+dec_place}.{dec_place}f}"
                    },
                    "Difference": {
                        "R": f"{rgb_ink_avg["a"][0]-rgb_ink_avg["b_avg"][0]:+{4+dec_place}.{dec_place}f} ±{rgb_ink_avg["b_diff"][0]/2.0:{4+dec_place}.{dec_place}f}",
                        "G": f"{rgb_ink_avg["a"][1]-rgb_ink_avg["b_avg"][1]:+{4+dec_place}.{dec_place}f} ±{rgb_ink_avg["b_diff"][1]/2.0:{4+dec_place}.{dec_place}f}",
                        "B": f"{rgb_ink_avg["a"][2]-rgb_ink_avg["b_avg"][2]:+{4+dec_place}.{dec_place}f} ±{rgb_ink_avg["b_diff"][2]/2.0:{4+dec_place}.{dec_place}f}"
                    }
                }
            }
            

        # Per-pixel comparison (todo)

    # LAB comparisons
    if op.lab_comparison:
        lab = {
            "b1": b1.get_pixels("lab"),
            "b2": b2.get_pixels("lab"),
            "a": a.get_pixels("lab")
        }
        
        # mean comparison
        if op.mean_comparison:

            lab_paper_only = {
                "b1": lab["b1"]*paper_mask["b1"],
                "b2": lab["b2"]*paper_mask["b2"],
                "a": lab["a"]*paper_mask["a"]
            }

            lab_paper_only_flat = {
                "b1": lab_paper_only["b1"].reshape(-1, lab_paper_only["b1"].shape[-1]),
                "b2": lab_paper_only["b2"].reshape(-1, lab_paper_only["b2"].shape[-1]),
                "a": lab_paper_only["a"].reshape(-1, lab_paper_only["a"].shape[-1])
            }

            lab_paper_avg = {
                "b1": lab_paper_only_flat["b1"].sum(axis=0, dtype=np.float64)/mask_size_paper["b1"],
                "b2": lab_paper_only_flat["b2"].sum(axis=0, dtype=np.float64)/mask_size_paper["b2"],
                "a": lab_paper_only_flat["a"].sum(axis=0, dtype=np.float64)/mask_size_paper["a"],
            }

            lab_paper_avg["b_avg"] = (lab_paper_avg["b1"] + lab_paper_avg["b2"]) / 2.0
            lab_paper_avg["b_diff"] = np.abs(lab_paper_avg["b1"] - lab_paper_avg["b2"])

            lab_ink_only = {
                "b1": lab["b1"]*ink_mask["b1"],
                "b2": lab["b2"]*ink_mask["b2"],
                "a": lab["a"]*ink_mask["a"]
            }

            lab_ink_only_flat = {
                "b1": lab_ink_only["b1"].reshape(-1, lab_ink_only["b1"].shape[-1]),
                "b2": lab_ink_only["b2"].reshape(-1, lab_ink_only["b2"].shape[-1]),
                "a": lab_ink_only["a"].reshape(-1, lab_ink_only["a"].shape[-1])
            }

            lab_ink_avg = {
                "b1": lab_ink_only_flat["b1"].sum(axis=0, dtype=np.float64)/mask_size_ink["b1"],
                "b2": lab_ink_only_flat["b2"].sum(axis=0, dtype=np.float64)/mask_size_ink["b2"],
                "a": lab_ink_only_flat["a"].sum(axis=0, dtype=np.float64)/mask_size_ink["a"],
            }

            lab_ink_avg["b_avg"] = (lab_ink_avg["b1"] + lab_ink_avg["b2"]) / 2.0
            lab_ink_avg["b_diff"] = np.abs(lab_ink_avg["b1"] - lab_ink_avg["b2"])

            lab_raw_results = {
                "b1": {
                    "paper": {
                        "l": lab_paper_avg["b1"][0],
                        "a": lab_paper_avg["b1"][1],
                        "b": lab_paper_avg["b1"][2]
                    },
                    "ink": {
                        "l": lab_ink_avg["b1"][0],
                        "a": lab_ink_avg["b1"][1],
                        "b": lab_ink_avg["b1"][2]
                    }
                },
                "b2": {
                    "paper": {
                        "l": lab_paper_avg["b2"][0],
                        "a": lab_paper_avg["b2"][1],
                        "b": lab_paper_avg["b2"][2]
                    },
                    "ink": {
                        "l": lab_ink_avg["b2"][0],
                        "a": lab_ink_avg["b2"][1],
                        "b": lab_ink_avg["b2"][2]
                    }
                },
                "a": {
                    "paper": {
                        "l": lab_paper_avg["a"][0],
                        "a": lab_paper_avg["a"][1],
                        "b": lab_paper_avg["a"][2]
                    },
                    "ink": {
                        "l": lab_ink_avg["a"][0],
                        "a": lab_ink_avg["a"][1],
                        "b": lab_ink_avg["a"][2]
                    }
                }
            }

            lab_scale = 100.0
            for k in lab_paper_avg: lab_paper_avg[k] *= lab_scale/100.0
            for k in lab_ink_avg: lab_ink_avg[k] *= lab_scale/100.0

            results["LAB Average"] = {
                "(Scale": f"{-lab_scale:.{dec_place}f} to {lab_scale:.{dec_place}f})",
                "Paper": {
                    "Before": {
                        "L*": f"{lab_paper_avg["b_avg"][0]:{4+dec_place}.{dec_place}f} ±{lab_paper_avg["b_diff"][0]/2.0:{4+dec_place}.{dec_place}f}",
                        "a*": f"{lab_paper_avg["b_avg"][1]:{4+dec_place}.{dec_place}f} ±{lab_paper_avg["b_diff"][1]/2.0:{4+dec_place}.{dec_place}f}",
                        "b*": f"{lab_paper_avg["b_avg"][2]:{4+dec_place}.{dec_place}f} ±{lab_paper_avg["b_diff"][2]/2.0:{4+dec_place}.{dec_place}f}"
                    },
                    "After": {
                        "L*": f"{lab_paper_avg["a"][0]:{4+dec_place}.{dec_place}f}",
                        "a*": f"{lab_paper_avg["a"][1]:{4+dec_place}.{dec_place}f}",
                        "b*": f"{lab_paper_avg["a"][2]:{4+dec_place}.{dec_place}f}"
                    },
                    "Difference": {
                        "L*": f"{lab_paper_avg["a"][0]-lab_paper_avg["b_avg"][0]:+{4+dec_place}.{dec_place}f} ±{lab_paper_avg["b_diff"][0]/2.0:{4+dec_place}.{dec_place}f}",
                        "a*": f"{lab_paper_avg["a"][1]-lab_paper_avg["b_avg"][1]:+{4+dec_place}.{dec_place}f} ±{lab_paper_avg["b_diff"][1]/2.0:{4+dec_place}.{dec_place}f}",
                        "b*": f"{lab_paper_avg["a"][2]-lab_paper_avg["b_avg"][2]:+{4+dec_place}.{dec_place}f} ±{lab_paper_avg["b_diff"][2]/2.0:{4+dec_place}.{dec_place}f}"
                    }
                },
                "Ink": {
                    "Before": {
                        "L*": f"{lab_ink_avg["b_avg"][0]:{4+dec_place}.{dec_place}f} ±{lab_ink_avg["b_diff"][0]/2.0:{4+dec_place}.{dec_place}f}",
                        "a*": f"{lab_ink_avg["b_avg"][1]:{4+dec_place}.{dec_place}f} ±{lab_ink_avg["b_diff"][1]/2.0:{4+dec_place}.{dec_place}f}",
                        "b*": f"{lab_ink_avg["b_avg"][2]:{4+dec_place}.{dec_place}f} ±{lab_ink_avg["b_diff"][2]/2.0:{4+dec_place}.{dec_place}f}"
                    },
                    "After": {
                        "L*": f"{lab_ink_avg["a"][0]:{4+dec_place}.{dec_place}f}",
                        "a*": f"{lab_ink_avg["a"][1]:{4+dec_place}.{dec_place}f}",
                        "b*": f"{lab_ink_avg["a"][2]:{4+dec_place}.{dec_place}f}"
                    },
                    "Difference": {
                        "L*": f"{lab_ink_avg["a"][0]-lab_ink_avg["b_avg"][0]:+{4+dec_place}.{dec_place}f} ±{lab_ink_avg["b_diff"][0]/2.0:{4+dec_place}.{dec_place}f}",
                        "a*": f"{lab_ink_avg["a"][1]-lab_ink_avg["b_avg"][1]:+{4+dec_place}.{dec_place}f} ±{lab_ink_avg["b_diff"][1]/2.0:{4+dec_place}.{dec_place}f}",
                        "b*": f"{lab_ink_avg["a"][2]-lab_ink_avg["b_avg"][2]:+{4+dec_place}.{dec_place}f} ±{lab_ink_avg["b_diff"][2]/2.0:{4+dec_place}.{dec_place}f}"
                    }
                }
            }

    return results
