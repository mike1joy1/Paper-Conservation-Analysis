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
from skimage.color import deltaE_ciede94, deltaE_ciede2000, deltaE_cmc

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

def describe_rgb_change(diff,
                        names=("red", "green", "blue"),
                        thresholds=(0.005, 0.03, 0.15),
                        labels=("slightly", "moderately", "significantly")):
    """
    diff       : array-like of 3 values in [-1, 1] (new minus old, per channel)
    thresholds : upper bounds of |diff| for 'no change' / each label except the last
    labels     : one more than... no, one per band above the 'no change' band
    """
    diff = np.asarray(diff, dtype=float)
    idx = np.searchsorted(thresholds, np.abs(diff), side="right")  # 0 = no change

    out = []
    for d, name, i in zip(diff, names, idx):
        if i == 0:
            out.append(f"No change in {name}")
        else:
            direction = "more" if d > 0 else "less"
            out.append(f"{labels[i - 1].capitalize()} {direction} {name}")
    return out

def describe_lab_change(diff,
                        thresholds=(1.0, 2.0, 10.0),
                        labels=("slightly", "moderately", "significantly")):
    """
    diff       : array-like [dL, da, db] (new minus old), in standard CIELAB units
                 (L: 0 to 100, a and b: roughly -128 to +127)
    thresholds : |diff| below the first = no change; each later one starts the next label
    labels     : one label per threshold
    """
    diff = np.asarray(diff, dtype=float)
    # (name, wording if positive, wording if negative)
    channels = (("lightness", "lighter", "darker"),
                ("red/green", "more red", "more green"),
                ("blue/yellow", "more yellow", "more blue"))

    idx = np.searchsorted(thresholds, np.abs(diff), side="right")  # 0 = no change

    out = []
    for d, (name, pos, neg), i in zip(diff, channels, idx):
        if i == 0:
            out.append(f"No change in {name}")
        else:
            out.append(f"{labels[i - 1].capitalize()} {pos if d > 0 else neg}")
    return out

def delta_e_dict(ref_lab, new_lab):
    """
    ref_lab : [L, a, b] mean Lab of the reference (before) image
    new_lab : [L, a, b] mean Lab of the after image
    Returns {"ΔE76": float, "ΔE94": float, "ΔE2000": float, "ΔE CMC": float}
    """
    ref = np.asarray(ref_lab, dtype=np.float64)
    new = np.asarray(new_lab, dtype=np.float64)
    if ref.shape != (3,) or new.shape != (3,):
        raise ValueError("ref_lab and new_lab must each have 3 values")

    return {
        "ΔE76":   float(np.linalg.norm(new - ref)),
        "ΔE94":   float(deltaE_ciede94(ref, new)),
        "ΔE2000": float(deltaE_ciede2000(ref, new)),
        "ΔE CMC": float(deltaE_cmc(ref, new)),
    }

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
        "heatmap_grid_size_x": op.heatmap_grid_size_x,
        "heatmap_grid_size_y": op.heatmap_grid_size_y,
        "dec_places": op.dec_places
    }

    if op.use_masks:
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
        
    else: # No masks, so just use all pixels
        paper_mask = {
            "b1": b1.masks["paper"],
            "b2": b2.masks["paper"],
            "a": a.masks["paper"]
        }
        ink_mask = {
            "b1": b1.masks["ink"],
            "b2": b2.masks["ink"],
            "a": a.masks["ink"]
        }

        for k in paper_mask:
            paper_mask[k] = np.ones_like(paper_mask[k], dtype=bool)
        for k in ink_mask:
            ink_mask[k] = np.ones_like(ink_mask[k], dtype=bool)

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
            rgb_paper_avg["diff"] = rgb_paper_avg["a"]-rgb_paper_avg["b_avg"]



            if op.use_masks:
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
                rgb_ink_avg["diff"] = rgb_ink_avg["a"]-rgb_ink_avg["b_avg"]

            rgb_raw_results = {
                "b1": {
                    "paper": {
                        "r": rgb_paper_avg["b1"][0],
                        "g": rgb_paper_avg["b1"][1],
                        "b": rgb_paper_avg["b1"][2]
                    }
                },
                "b2": {
                    "paper": {
                        "r": rgb_paper_avg["b2"][0],
                        "g": rgb_paper_avg["b2"][1],
                        "b": rgb_paper_avg["b2"][2]
                    }
                },
                "a": {
                    "paper": {
                        "r": rgb_paper_avg["a"][0],
                        "g": rgb_paper_avg["a"][1],
                        "b": rgb_paper_avg["a"][2]
                    }
                }
            }

            if op.use_masks:
                rgb_raw_results.update({
                    "b1": {
                        "ink": {
                            "r": rgb_ink_avg["b1"][0],
                            "g": rgb_ink_avg["b1"][1],
                            "b": rgb_ink_avg["b1"][2]
                        }
                    },
                    "b2": {
                        "ink": {
                            "r": rgb_ink_avg["b2"][0],
                            "g": rgb_ink_avg["b2"][1],
                            "b": rgb_ink_avg["b2"][2]
                        }
                    },
                    "a": {
                        "ink": {
                            "r": rgb_ink_avg["a"][0],
                            "g": rgb_ink_avg["a"][1],
                            "b": rgb_ink_avg["a"][2]
                        }
                    }
                })

            rgb_scale = 100.0
            for k in rgb_paper_avg: rgb_paper_avg[k] *= rgb_scale
            if op.use_masks:
                for k in rgb_ink_avg: rgb_ink_avg[k] *= rgb_scale

            rgb_paper_avg["diff_text"] = describe_rgb_change(rgb_paper_avg["diff"]/rgb_scale)
            if op.use_masks:
                rgb_ink_avg["diff_text"] = describe_rgb_change(rgb_ink_avg["diff"]/rgb_scale)

            results["Overall change in RGB colour space"] = {
                "(Scale": f"{0.0:.{dec_place}f} to {rgb_scale:.{dec_place}f})",
                f"{"Paper" if op.use_masks else "Whole image"}": {
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
                        "R": f"{rgb_paper_avg["diff"][0]:+{4+dec_place}.{dec_place}f} ±{rgb_paper_avg["b_diff"][0]/2.0:{4+dec_place}.{dec_place}f}",
                        "G": f"{rgb_paper_avg["diff"][1]:+{4+dec_place}.{dec_place}f} ±{rgb_paper_avg["b_diff"][1]/2.0:{4+dec_place}.{dec_place}f}",
                        "B": f"{rgb_paper_avg["diff"][2]:+{4+dec_place}.{dec_place}f} ±{rgb_paper_avg["b_diff"][2]/2.0:{4+dec_place}.{dec_place}f}",
                    },
                    "Summary": {
                        "R": rgb_paper_avg["diff_text"][0],
                        "G": rgb_paper_avg["diff_text"][1],
                        "B": rgb_paper_avg["diff_text"][2]
                    }
                }
            }
            if op.use_masks:
                results["Overall change in RGB colour space"].update({
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
                            "R": f"{rgb_ink_avg["diff"][0]:+{4+dec_place}.{dec_place}f} ±{rgb_ink_avg["b_diff"][0]/2.0:{4+dec_place}.{dec_place}f}",
                            "G": f"{rgb_ink_avg["diff"][1]:+{4+dec_place}.{dec_place}f} ±{rgb_ink_avg["b_diff"][1]/2.0:{4+dec_place}.{dec_place}f}",
                            "B": f"{rgb_ink_avg["diff"][2]:+{4+dec_place}.{dec_place}f} ±{rgb_ink_avg["b_diff"][2]/2.0:{4+dec_place}.{dec_place}f}"
                        },
                        "Summary": {
                            "R": rgb_ink_avg["diff_text"][0],
                            "G": rgb_ink_avg["diff_text"][1],
                            "B": rgb_ink_avg["diff_text"][2]
                        }
                    }
                })

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
            lab_paper_avg["diff"] = lab_paper_avg["a"]-lab_paper_avg["b_avg"]

            if op.use_masks:
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
                lab_ink_avg["diff"] = lab_ink_avg["a"]-lab_ink_avg["b_avg"]

            lab_raw_results = {
                "b1": {
                    "paper": {
                        "l": lab_paper_avg["b1"][0],
                        "a": lab_paper_avg["b1"][1],
                        "b": lab_paper_avg["b1"][2]
                    }
                },
                "b2": {
                    "paper": {
                        "l": lab_paper_avg["b2"][0],
                        "a": lab_paper_avg["b2"][1],
                        "b": lab_paper_avg["b2"][2]
                    }
                },
                "a": {
                    "paper": {
                        "l": lab_paper_avg["a"][0],
                        "a": lab_paper_avg["a"][1],
                        "b": lab_paper_avg["a"][2]
                    }
                }
            }

            if op.use_masks:
                lab_raw_results.update({
                    "b1": {
                        "ink": {
                            "l": lab_ink_avg["b1"][0],
                            "a": lab_ink_avg["b1"][1],
                            "b": lab_ink_avg["b1"][2]
                        }
                    },
                    "b2": {
                        "ink": {
                            "l": lab_ink_avg["b2"][0],
                            "a": lab_ink_avg["b2"][1],
                            "b": lab_ink_avg["b2"][2]
                        }
                    },
                    "a": {
                        "ink": {
                            "l": lab_ink_avg["a"][0],
                            "a": lab_ink_avg["a"][1],
                            "b": lab_ink_avg["a"][2]
                        }
                    }
                })

            delta_e_paper_b1 = delta_e_dict(lab_paper_avg["b1"],lab_paper_avg["a"])
            delta_e_paper_b2 = delta_e_dict(lab_paper_avg["b2"],lab_paper_avg["a"])
            delta_e_paper_avg  = {k: (delta_e_paper_b1[k] + delta_e_paper_b2[k]) / 2.0 for k in delta_e_paper_b1}
            delta_e_paper_diff = {k: abs(delta_e_paper_b1[k] - delta_e_paper_b2[k]) for k in delta_e_paper_b1}

            if op.use_masks:
                delta_e_ink_b1 = delta_e_dict(lab_ink_avg["b1"],lab_ink_avg["a"])
                delta_e_ink_b2 = delta_e_dict(lab_ink_avg["b2"],lab_ink_avg["a"])
                delta_e_ink_avg  = {k: (delta_e_ink_b1[k] + delta_e_ink_b2[k]) / 2.0 for k in delta_e_ink_b1}
                delta_e_ink_diff = {k: abs(delta_e_ink_b1[k] - delta_e_ink_b2[k]) for k in delta_e_ink_b1}

            lab_scale = 100.0
            for k in lab_paper_avg: lab_paper_avg[k] *= lab_scale/100.0
            if op.use_masks:
                for k in lab_ink_avg: lab_ink_avg[k] *= lab_scale/100.0

            lab_paper_avg["diff_text"] = describe_lab_change(lab_paper_avg["diff"]*100/lab_scale)
            if op.use_masks:
                lab_ink_avg["diff_text"] = describe_lab_change(lab_ink_avg["diff"]*100/lab_scale)

            results["Overall change in CIELAB (L*a*b*) colour space"] = {
                "(L* Scale": f"{0.0:.{dec_place}f} to {lab_scale:.{dec_place}f})",
                "(a*b* Scale": f"{-lab_scale:.{dec_place}f} to {lab_scale:.{dec_place}f})",
                f"{"Paper" if op.use_masks else "Whole image"}": {
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
                    },
                    "Summary": {
                        "L*": lab_paper_avg["diff_text"][0],
                        "a*": lab_paper_avg["diff_text"][1],
                        "b*": lab_paper_avg["diff_text"][2]
                    },
                    "Perceptual Colour Difference (ΔE)": {
                        "ΔE76  ": f"{delta_e_paper_avg["ΔE76"]:{3+dec_place}.{dec_place}f} ±{delta_e_paper_diff["ΔE76"]/2.0:{4+dec_place}.{dec_place}f}",
                        "ΔE94  ": f"{delta_e_paper_avg["ΔE94"]:{3+dec_place}.{dec_place}f} ±{delta_e_paper_diff["ΔE94"]/2.0:{4+dec_place}.{dec_place}f}",
                        "ΔE2000": f"{delta_e_paper_avg["ΔE2000"]:{3+dec_place}.{dec_place}f} ±{delta_e_paper_diff["ΔE2000"]/2.0:{4+dec_place}.{dec_place}f}",
                        "ΔE CMC": f"{delta_e_paper_avg["ΔE CMC"]:{3+dec_place}.{dec_place}f} ±{delta_e_paper_diff["ΔE CMC"]/2.0:{4+dec_place}.{dec_place}f}"
                    }
                }
            }

            if op.use_masks:
                results["Overall change in CIELAB (L*a*b*) colour space"].update({
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
                        },
                        "Summary": {
                            "L*": lab_ink_avg["diff_text"][0],
                            "a*": lab_ink_avg["diff_text"][1],
                            "b*": lab_ink_avg["diff_text"][2]
                        },
                        "Perceptual Colour Difference (ΔE)": {
                            "ΔE76  ": f"{delta_e_ink_avg["ΔE76"]:{3+dec_place}.{dec_place}f} ±{delta_e_ink_diff["ΔE76"]/2.0:{4+dec_place}.{dec_place}f}",
                            "ΔE94  ": f"{delta_e_ink_avg["ΔE94"]:{3+dec_place}.{dec_place}f} ±{delta_e_ink_diff["ΔE94"]/2.0:{4+dec_place}.{dec_place}f}",
                            "ΔE2000": f"{delta_e_ink_avg["ΔE2000"]:{3+dec_place}.{dec_place}f} ±{delta_e_ink_diff["ΔE2000"]/2.0:{4+dec_place}.{dec_place}f}",
                            "ΔE CMC": f"{delta_e_ink_avg["ΔE CMC"]:{3+dec_place}.{dec_place}f} ±{delta_e_ink_diff["ΔE CMC"]/2.0:{4+dec_place}.{dec_place}f}"
                        }
                    }
                })

    return results
