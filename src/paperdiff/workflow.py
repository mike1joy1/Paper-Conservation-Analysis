"""Simple analysis workflow for already-aligned paper images.

This module is intentionally small and explicit. The research assumption is that
B1, B2 and A have already been aligned and cropped before they reach this app.
The code therefore focuses only on measuring the visual difference between the
reference image and the comparison images.
"""

from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede94, deltaE_ciede2000, deltaE_cmc

from .analysis import compute_deltas, compute_ink_mask
from .image_state import analysis_options, image_data_colour

__all__ = ["image_analysis_b1_b2_a", "describe_region_change"]

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
    Describe the direction and magnitude of each RGB channel change.

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
    Describe the direction and magnitude of each CIELAB channel change.

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
    Calculate four CIELAB colour distances between two mean colours.

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


def rgb_distance_dict(ref_rgb, new_rgb):
    """Compare sRGB means in 0..1, returning distances on the report's 0..100 scale."""
    ref = np.asarray(ref_rgb, dtype=np.float64)
    new = np.asarray(new_rgb, dtype=np.float64)
    if ref.shape != (3,) or new.shape != (3,):
        raise ValueError("ref_rgb and new_rgb must each have 3 values")

    dr, dg, db = (new - ref) * 100.0
    red_mean_8bit = (ref[0] + new[0]) * 255.0 / 2.0
    return {
        "sRGB Euclidean": float(np.sqrt(dr ** 2 + dg ** 2 + db ** 2)),
        "sRGB redmean": float(np.sqrt(
            (2.0 + red_mean_8bit / 256.0) * dr ** 2
            + 4.0 * dg ** 2
            + (2.0 + (255.0 - red_mean_8bit) / 256.0) * db ** 2
        )),
    }


def _as_rgb(image: np.ndarray) -> np.ndarray:
    """Convert grayscale or RGB image-like input to a floating-point RGB array."""
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


def image_avg(image, mask):
    """Return the masked per-channel mean, or zero when the mask is empty."""
    pixel_count = np.sum(mask)/3
    masked_image = image*mask

    if pixel_count == 0:
        return masked_image.reshape(-1, masked_image.shape[-1]).sum(axis=0, dtype=np.float64)*0
    else:
        return masked_image.reshape(-1, masked_image.shape[-1]).sum(axis=0, dtype=np.float64)/pixel_count

CIELAB_L_RANGE = 100.0
CIELAB_AB_RANGE = 128.0


def _ceilab_channel_scale(op) -> np.ndarray:
    """Per-channel multipliers mapping L* (0..100) to 0..L scale and a*/b* (-128..128) to -ab scale..ab scale."""
    ab = op.ceilab_ab_scale / CIELAB_AB_RANGE
    return np.array([op.ceilab_l_scale / CIELAB_L_RANGE, ab, ab])


def _scale_decimals(scale) -> int:
    """Decimal places giving about three significant digits for values spanning 0..scale."""
    return max(0, round(3 - math.log10(max(float(scale), 1e-9))))


def _channel_decimals(op, space):
    """Per-channel decimal places for a colour space, plus the fixed ΔE decimal places."""
    if space == "sRGB":
        return (_scale_decimals(op.srgb_scale),) * 3, op.dec_places
    ab = _scale_decimals(op.ceilab_ab_scale)
    return (_scale_decimals(op.ceilab_l_scale), ab, ab), op.dec_places


def _ceilab_scale_heading(op, dec_place) -> dict:
    l_dp, ab_dp = dec_place[0], dec_place[1]
    return {
        "(L* Scale": f"{0.0:.{l_dp}f} to {op.ceilab_l_scale:.{l_dp}f})",
        "(a*b* Scale": f"{-op.ceilab_ab_scale:.{ab_dp}f} to {op.ceilab_ab_scale:.{ab_dp}f})",
    }


def image_avg_sd(image, mask, check_cancelled: Callable[[], None] | None = None):
    """
    Per-channel mean and standard deviation of `image`
    over the unmasked pixels.
    image : (H, W, C), e.g. an RGB/Lab image or the difference between two images
    mask  : (H, W, C), 0/1, all channels equal
    Returns (avg, sd), each of shape (C,). Zeros if the mask is empty.
    
    For individual images:
        avg is the average colour
        SD is a measure of how uniform the image is (0 = completely uniform, e.g. smooth paper)
    For a differenced image (A-B):
        avg is the average colour change
        SD is a measure of how uniformly the image has changed (0 = same change everywhere)
    """
    channel_count = image.shape[-1]
    if check_cancelled is not None:
        check_cancelled()
    pixel_count = np.sum(mask) / channel_count

    if pixel_count == 0:
        return np.zeros(channel_count), np.zeros(channel_count)

    flat_image = image.reshape(-1, channel_count)
    flat_mask = np.asarray(mask).reshape(-1, channel_count)

    average = (flat_image * flat_mask).sum(axis=0, dtype=np.float64) / pixel_count
    if check_cancelled is not None:
        check_cancelled()
    squared_deviation = ((flat_image - average) ** 2) * flat_mask
    if check_cancelled is not None:
        check_cancelled()
    standard_deviation = np.sqrt(squared_deviation.sum(axis=0, dtype=np.float64) / pixel_count)
    return average, standard_deviation

def grid_averages(image_tiles, mask_tiles):
    """
    Calculate masked mean colours for each tile; return shape (y, x, 3).
    """
    y = len(image_tiles)
    x = len(image_tiles[0])
    
    averages = np.empty((y, x, 3))

    for j in range(y):
        for i in range(x):
            averages[j, i] = image_avg(image_tiles[j][i], mask_tiles[j][i])
    return averages


def _bounds(n_pixels, n_tiles):
    """Tile edges along one axis, identical to those used in split_image."""
    return np.array([i * n_pixels // n_tiles for i in range(n_tiles + 1)])


def split_image(img, x, y):
    """
    Split an image (H, W, ...) into a grid of x sub-images across and y down.
    Returns a nested list indexed [row][column].
    """
    img = np.asarray(img)
    h, w = img.shape[:2]
    if not (1 <= x <= w and 1 <= y <= h):
        raise ValueError(f"x must be 1..{w} and y must be 1..{h}")

    xs = _bounds(w, x)
    ys = _bounds(h, y)

    return [[img[ys[j]:ys[j + 1], xs[i]:xs[i + 1]] for i in range(x)]
            for j in range(y)]


def upscale_blocky(avgs, height, width, dtype=np.float32):
    """
    Expand a (y, x, 3) grid of tile colours to (height, width, 3),
    filling each tile's area with its colour.
    """
    avgs = np.asarray(avgs, dtype=dtype)
    y, x = avgs.shape[:2]
    row_counts = np.diff(_bounds(height, y))   # pixel rows in each tile row
    col_counts = np.diff(_bounds(width, x))    # pixel columns in each tile column
    return np.repeat(np.repeat(avgs, row_counts, axis=0), col_counts, axis=1)


def _interp_weights(n_pixels, n_tiles):
    """For each pixel along an axis: the two neighbouring tile indices and the blend weight."""
    edges = _bounds(n_pixels, n_tiles)
    centres = (edges[:-1] + edges[1:]) / 2.0          # tile centres, in pixel units
    px = np.arange(n_pixels) + 0.5                    # pixel centres
    # Fractional tile index of each pixel (clamped to the first/last tile at the borders)
    f = np.interp(px, centres, np.arange(n_tiles))
    i0 = np.clip(np.floor(f).astype(int), 0, max(n_tiles - 2, 0))
    i1 = np.minimum(i0 + 1, n_tiles - 1)
    t = f - i0
    return i0, i1, t


def upscale_smooth(avgs, height, width, dtype=np.float32):
    """
    Expand a (y, x, 3) grid of tile colours to (height, width, 3) with
    bilinear interpolation, treating each tile colour as the colour at
    the centre of that tile. Pixels beyond the outermost tile centres
    take the nearest tile's colour.
    """
    avgs = np.asarray(avgs, dtype=dtype)
    y, x = avgs.shape[:2]

    c0, c1, tx = _interp_weights(width, x)
    r0, r1, ty = _interp_weights(height, y)
    tx = tx.astype(dtype)[None, :, None]
    ty = ty.astype(dtype)[:, None, None]

    # Blend horizontally first: (y, width, 3)
    horiz = avgs[:, c0, :] * (1 - tx) + avgs[:, c1, :] * tx
    # Then vertically: (height, width, 3)
    return horiz[r0] * (1 - ty) + horiz[r1] * ty

def compare_b1_b2_a(
    b1, b2, a, b1_pm, b2_pm, a_pm, b1_im, b2_im, a_im, use_masks,
    scale=1.0, delta_e=False, delta_e_fn=delta_e_dict,
    check_cancelled: Callable[[], None] | None = None,
    progress: Callable[[str], None] | None = None,
    colour_space: str = "",
):
    """Compare masked before and after colour statistics and optional colour distances."""
    out_data = dict()
    
    if progress is not None:
        progress(f"Analysing {colour_space} paper: Before Image 1")
    b1p_avg, b1p_sd = image_avg_sd(b1,b1_pm,check_cancelled)
    if progress is not None:
        progress(f"Analysing {colour_space} paper: Before Image 2")
    b2p_avg, b2p_sd = image_avg_sd(b2,b2_pm,check_cancelled)
    if progress is not None:
        progress(f"Analysing {colour_space} paper: After Image")
    ap_avg,  ap_sd  = image_avg_sd(a,a_pm,check_cancelled)

    if delta_e:
        delta_e_paper_b1 = delta_e_fn(b1p_avg,ap_avg)
        delta_e_paper_b2 = delta_e_fn(b2p_avg,ap_avg)
        delta_e_paper_avg  = {k: (delta_e_paper_b1[k] + delta_e_paper_b2[k]) / 2.0 for k in delta_e_paper_b1}
        delta_e_paper_diff = {k: abs(delta_e_paper_b1[k] - delta_e_paper_b2[k]) for k in delta_e_paper_b1}

    b1p_avg *= scale
    b2p_avg *= scale
    ap_avg  *= scale
    b1p_sd  *= scale
    b2p_sd  *= scale
    ap_sd   *= scale
    
    bp_avg, bp_sd = (b1p_avg+b2p_avg)/2.0, (b1p_sd+b2p_sd)/2.0
    bp_avg_diff, bp_sd_diff = abs(b1p_avg-b2p_avg), abs(b1p_sd-b2p_sd)

    comp_avg_p_1, comp_sd_p_1 = ap_avg - b1p_avg, ap_sd - b1p_sd
    comp_avg_p_2, comp_sd_p_2 = ap_avg - b2p_avg, ap_sd - b2p_sd
    comp_avg_p,   comp_sd_p   = ap_avg - bp_avg,  ap_sd - bp_sd

    out_data["paper"] = {
        "before": {
            "avg": bp_avg,
            "sd" : bp_sd
        },
        "after": {
            "avg": ap_avg,
            "sd" : ap_sd
        },
        "change": {
            "avg": comp_avg_p,
            "sd" : comp_sd_p
        },
        "plus_minus": {
            "avg": bp_avg_diff,
            "sd" : bp_sd_diff
        }
    }

    if delta_e:
        out_data["paper"].update({
            "delta_e": {
                "avg": delta_e_paper_avg,
                "plus_minus": delta_e_paper_diff
            }
        })


    if use_masks:
        if progress is not None:
            progress(f"Analysing {colour_space} ink: Before Image 1")
        b1i_avg, b1i_sd = image_avg_sd(b1,b1_im,check_cancelled)
        if progress is not None:
            progress(f"Analysing {colour_space} ink: Before Image 2")
        b2i_avg, b2i_sd = image_avg_sd(b2,b2_im,check_cancelled)
        if progress is not None:
            progress(f"Analysing {colour_space} ink: After Image")
        ai_avg,  ai_sd  = image_avg_sd(a,a_im,check_cancelled)

        if delta_e:
            delta_e_ink_b1 = delta_e_fn(b1i_avg,ai_avg)
            delta_e_ink_b2 = delta_e_fn(b2i_avg,ai_avg)
            delta_e_ink_avg  = {k: (delta_e_ink_b1[k] + delta_e_ink_b2[k]) / 2.0 for k in delta_e_ink_b1}
            delta_e_ink_diff = {k: abs(delta_e_ink_b1[k] - delta_e_ink_b2[k]) for k in delta_e_ink_b1}

        b1i_avg *= scale
        b2i_avg *= scale
        ai_avg *= scale
        b1i_sd *= scale
        b2i_sd *= scale
        ai_sd *= scale

        
        bi_avg, bi_sd = (b1i_avg+b2i_avg)/2.0, (b1i_sd+b2i_sd)/2.0
        bi_avg_diff, bi_sd_diff = abs(b1i_avg-b2i_avg), abs(b1i_sd-b2i_sd)

        comp_avg_i_1, comp_sd_i_1 = ai_avg - b1i_avg, ai_sd - b1i_sd
        comp_avg_i_2, comp_sd_i_2 = ai_avg - b2i_avg, ai_sd - b2i_sd
        comp_avg_i,   comp_sd_i   = ai_avg - bi_avg,  ai_sd - bi_sd

        out_data["ink"] = {
            "before": {
                "avg": bi_avg,
                "sd" : bi_sd
            },
            "after": {
                "avg": ai_avg,
                "sd" : ai_sd
            },
            "change": {
                "avg": comp_avg_i,
                "sd" : comp_sd_i
            },
            "plus_minus": {
                "avg": bi_avg_diff/2.0,
                "sd" : bi_sd_diff/2.0
            }
        }

        if delta_e:
            out_data["ink"].update({
                "delta_e": {
                    "avg": delta_e_ink_avg,
                    "plus_minus": delta_e_ink_diff
                }
            })

    return out_data


def format_data_out(raw_data, paper_diff_text, ink_diff_text, c0, c1, c2, use_masks, dec_place, delta_e, delta_e_heading="Perceptual Colour Difference (ΔE)"):
    """Format region statistics and colour differences for the results report."""
    format_out = dict()

    format_out = {
        f"{"Paper" if use_masks else "Whole image"}": {
            "Before": {
                "Colour Average": {
                    c0: f"{raw_data["paper"]["before"]["avg"][0]:{4+dec_place}.{dec_place}f}  (±{raw_data["paper"]["plus_minus"]["avg"][0]/2.0:.{dec_place}f})",
                    c1: f"{raw_data["paper"]["before"]["avg"][1]:{4+dec_place}.{dec_place}f}  (±{raw_data["paper"]["plus_minus"]["avg"][1]/2.0:.{dec_place}f})",
                    c2: f"{raw_data["paper"]["before"]["avg"][2]:{4+dec_place}.{dec_place}f}  (±{raw_data["paper"]["plus_minus"]["avg"][2]/2.0:.{dec_place}f})"
                },
                "Colour Standard Deviation": {
                    c0: f"{raw_data["paper"]["before"]["sd"][0]:{4+dec_place}.{dec_place}f}  (±{raw_data["paper"]["plus_minus"]["sd"][0]/2.0:.{dec_place}f})",
                    c1: f"{raw_data["paper"]["before"]["sd"][1]:{4+dec_place}.{dec_place}f}  (±{raw_data["paper"]["plus_minus"]["sd"][1]/2.0:.{dec_place}f})",
                    c2: f"{raw_data["paper"]["before"]["sd"][2]:{4+dec_place}.{dec_place}f}  (±{raw_data["paper"]["plus_minus"]["sd"][2]/2.0:.{dec_place}f})"
                }
            },
            "After": {
                "Colour Average": {
                    c0: f"{raw_data["paper"]["after"]["avg"][0]:{4+dec_place}.{dec_place}f}",
                    c1: f"{raw_data["paper"]["after"]["avg"][1]:{4+dec_place}.{dec_place}f}",
                    c2: f"{raw_data["paper"]["after"]["avg"][2]:{4+dec_place}.{dec_place}f}"
                },
                "Colour Standard Deviation": {
                    c0: f"{raw_data["paper"]["after"]["sd"][0]:{4+dec_place}.{dec_place}f}",
                    c1: f"{raw_data["paper"]["after"]["sd"][1]:{4+dec_place}.{dec_place}f}",
                    c2: f"{raw_data["paper"]["after"]["sd"][2]:{4+dec_place}.{dec_place}f}"
                }
            },
            "Change": {
                "Colour Average": {
                    c0: f"{raw_data["paper"]["change"]["avg"][0]:+{4+dec_place}.{dec_place}f}  (±{raw_data["paper"]["plus_minus"]["avg"][0]/2.0:.{dec_place}f})",
                    c1: f"{raw_data["paper"]["change"]["avg"][1]:+{4+dec_place}.{dec_place}f}  (±{raw_data["paper"]["plus_minus"]["avg"][1]/2.0:.{dec_place}f})",
                    c2: f"{raw_data["paper"]["change"]["avg"][2]:+{4+dec_place}.{dec_place}f}  (±{raw_data["paper"]["plus_minus"]["avg"][2]/2.0:.{dec_place}f})"
                },
                "Colour Standard Deviation": {
                    c0: f"{raw_data["paper"]["change"]["sd"][0]:+{4+dec_place}.{dec_place}f}  (±{raw_data["paper"]["plus_minus"]["sd"][0]/2.0:.{dec_place}f})",
                    c1: f"{raw_data["paper"]["change"]["sd"][1]:+{4+dec_place}.{dec_place}f}  (±{raw_data["paper"]["plus_minus"]["sd"][1]/2.0:.{dec_place}f})",
                    c2: f"{raw_data["paper"]["change"]["sd"][2]:+{4+dec_place}.{dec_place}f}  (±{raw_data["paper"]["plus_minus"]["sd"][2]/2.0:.{dec_place}f})"
                },
                "Summary": {
                    c0: paper_diff_text[0],
                    c1: paper_diff_text[1],
                    c2: paper_diff_text[2]
                }
            }
        }
    }

    if delta_e:
        format_out[f"{"Paper" if use_masks else "Whole image"}"]["Change"].update(
            {
                delta_e_heading: {
                    name.ljust(6): f"{value:{3+dec_place}.{dec_place}f}  (±{raw_data['paper']['delta_e']['plus_minus'][name]/2.0:.{dec_place}f})"
                    for name, value in raw_data["paper"]["delta_e"]["avg"].items()
                }
            }
        )


    if use_masks:
        format_out.update({
            f"Ink": {
                "Before": {
                    "Colour Average": {
                        c0: f"{raw_data["ink"]["before"]["avg"][0]:{4+dec_place}.{dec_place}f}  (±{raw_data["ink"]["plus_minus"]["avg"][0]/2.0:.{dec_place}f})",
                        c1: f"{raw_data["ink"]["before"]["avg"][1]:{4+dec_place}.{dec_place}f}  (±{raw_data["ink"]["plus_minus"]["avg"][1]/2.0:.{dec_place}f})",
                        c2: f"{raw_data["ink"]["before"]["avg"][2]:{4+dec_place}.{dec_place}f}  (±{raw_data["ink"]["plus_minus"]["avg"][2]/2.0:.{dec_place}f})"
                    },
                    "Colour Standard Deviation": {
                        c0: f"{raw_data["ink"]["before"]["sd"][0]:{4+dec_place}.{dec_place}f}  (±{raw_data["ink"]["plus_minus"]["sd"][0]/2.0:.{dec_place}f})",
                        c1: f"{raw_data["ink"]["before"]["sd"][1]:{4+dec_place}.{dec_place}f}  (±{raw_data["ink"]["plus_minus"]["sd"][1]/2.0:.{dec_place}f})",
                        c2: f"{raw_data["ink"]["before"]["sd"][2]:{4+dec_place}.{dec_place}f}  (±{raw_data["ink"]["plus_minus"]["sd"][2]/2.0:.{dec_place}f})"
                    }
                },
                "After": {
                    "Colour Average": {
                        c0: f"{raw_data["ink"]["after"]["avg"][0]:{4+dec_place}.{dec_place}f}",
                        c1: f"{raw_data["ink"]["after"]["avg"][1]:{4+dec_place}.{dec_place}f}",
                        c2: f"{raw_data["ink"]["after"]["avg"][2]:{4+dec_place}.{dec_place}f}"
                    },
                    "Colour Standard Deviation": {
                        c0: f"{raw_data["ink"]["after"]["sd"][0]:{4+dec_place}.{dec_place}f}",
                        c1: f"{raw_data["ink"]["after"]["sd"][1]:{4+dec_place}.{dec_place}f}",
                        c2: f"{raw_data["ink"]["after"]["sd"][2]:{4+dec_place}.{dec_place}f}"
                    }
                },
                "Change": {
                    "Colour Average": {
                        c0: f"{raw_data["ink"]["change"]["avg"][0]:+{4+dec_place}.{dec_place}f}  (±{raw_data["ink"]["plus_minus"]["avg"][0]/2.0:.{dec_place}f})",
                        c1: f"{raw_data["ink"]["change"]["avg"][1]:+{4+dec_place}.{dec_place}f}  (±{raw_data["ink"]["plus_minus"]["avg"][1]/2.0:.{dec_place}f})",
                        c2: f"{raw_data["ink"]["change"]["avg"][2]:+{4+dec_place}.{dec_place}f}  (±{raw_data["ink"]["plus_minus"]["avg"][2]/2.0:.{dec_place}f})"
                    },
                    "Colour Standard Deviation": {
                        c0: f"{raw_data["ink"]["change"]["sd"][0]:+{4+dec_place}.{dec_place}f}  (±{raw_data["ink"]["plus_minus"]["sd"][0]/2.0:.{dec_place}f})",
                        c1: f"{raw_data["ink"]["change"]["sd"][1]:+{4+dec_place}.{dec_place}f}  (±{raw_data["ink"]["plus_minus"]["sd"][1]/2.0:.{dec_place}f})",
                        c2: f"{raw_data["ink"]["change"]["sd"][2]:+{4+dec_place}.{dec_place}f}  (±{raw_data["ink"]["plus_minus"]["sd"][2]/2.0:.{dec_place}f})"
                    },
                    "Summary": {
                        c0: ink_diff_text[0],
                        c1: ink_diff_text[1],
                        c2: ink_diff_text[2]
                    }
                }
            }
        })

        if delta_e:
            format_out["Ink"]["Change"].update(
                {
                    delta_e_heading: {
                        name.ljust(6): f"{value:{3+dec_place}.{dec_place}f}  (±{raw_data['ink']['delta_e']['plus_minus'][name]/2.0:.{dec_place}f})"
                        for name, value in raw_data["ink"]["delta_e"]["avg"].items()
                    }
                }
            )

    return format_out



def _compare_configured_images(images, paper_masks, ink_masks, use_masks, mode,
                               scale, distance_fn, check_cancelled, progress, colour_space):
    """Calculate region statistics and paired distances for the selected image configuration."""
    groups = {
        "single": (("b1",), ()),
        "two_before": (("b1", "b2"), ("a",)),
        "one_each": (("b1",), ("a",)),
        "two_after": (("b1",), ("b2", "a")),
    }
    labels = {
        "single": {"b1": "Image"},
        "two_before": {"b1": "Before Image 1", "b2": "Before Image 2", "a": "After Image"},
        "one_each": {"b1": "Before Image", "a": "After Image"},
        "two_after": {"b1": "Before Image", "b2": "After Image 1", "a": "After Image 2"},
    }
    before_keys, after_keys = groups[mode]
    output = {}
    for region, masks in (("paper", paper_masks), ("ink", ink_masks)):
        if region == "ink" and not use_masks:
            continue
        measurements = {}
        for key in before_keys + after_keys:
            if progress is not None:
                progress(f"Analysing {colour_space} {region}: {labels[mode][key]}")
            measurements[key] = image_avg_sd(images[key], masks[key], check_cancelled)

        def summarize(keys, index):
            values = [measurements[key][index] * scale for key in keys]
            mean = sum(values) / len(values)
            uncertainty = abs(values[0] - values[1]) / 2 if len(values) == 2 else None
            return mean, uncertainty

        region_data = {}
        for index, name in enumerate(("avg", "sd")):
            before, before_pm = summarize(before_keys, index)
            if not after_keys:
                region_data.setdefault("image", {})[name] = before
                region_data["image"][name + "_pm"] = None
                continue
            after, after_pm = summarize(after_keys, index)
            for section, value, uncertainty in (
                ("before", before, before_pm),
                ("after", after, after_pm),
                ("change", after - before, before_pm if before_pm is not None else after_pm),
            ):
                region_data.setdefault(section, {})[name] = value
                region_data[section][name + "_pm"] = uncertainty

        if not after_keys:
            output[region] = region_data
            continue
        distances = [
            distance_fn(measurements[before][0], measurements[after][0])
            for before in before_keys for after in after_keys
        ]
        region_data["delta_e"] = {
            name: (sum(item[name] for item in distances) / len(distances),
                   abs(distances[0][name] - distances[1][name]) / 2 if len(distances) == 2 else None)
            for name in distances[0]
        }
        output[region] = region_data
    return output


def _format_configured_data(raw_data, paper_text, ink_text, channels, use_masks,
                            channel_dps, delta_e_heading, delta_e_dp=1):
    """Format region statistics, omitting unavailable replicate uncertainties.

    channel_dps gives the decimal places for each channel; delta_e_dp is used for ΔE values.
    """
    result = {}

    def format_value(value, uncertainty, dp, signed=False):
        prefix = "+" if signed else ""
        text = f"{value:{prefix}{3 + dp}.{dp}f}"
        if uncertainty is not None:
            text += f"  (±{uncertainty:.{dp}f})"
        return text

    for region, heading, summary in (
        ("paper", "Paper" if use_masks else "Whole image", paper_text),
        ("ink", "Ink", ink_text),
    ):
        if region == "ink" and not use_masks:
            continue
        data = raw_data[region]
        output = result[heading] = {}
        sections = ("image",) if "image" in data else ("before", "after", "change")
        for section in sections:
            section_data = output[section.title()] = {}
            for field, label in (("avg", "Colour Average"), ("sd", "Colour Standard Deviation")):
                values = data[section][field]
                uncertainties = data[section][field + "_pm"]
                section_data[label] = {
                    channel: format_value(values[index], None if uncertainties is None else uncertainties[index],
                                          channel_dps[index], signed=section == "change")
                    for index, channel in enumerate(channels)
                }
            if section == "change":
                section_data["Summary"] = dict(zip(channels, summary))
                section_data[delta_e_heading] = {
                    name.ljust(6): format_value(value, uncertainty, delta_e_dp)
                    for name, (value, uncertainty) in data["delta_e"].items()
                }
    return result


def _image_report_data(image, paper_mask, ink_mask):
    """Summarise the image dimensions and paper/ink mask coverage."""
    pixels = image.get_pixels("rgb")
    total = np.prod(pixels.shape[:2])
    paper = np.sum(paper_mask) / 3
    ink = np.sum(ink_mask) / 3
    unused = total - np.sum(np.maximum(paper_mask, ink_mask)) / 3
    both = np.sum(np.minimum(paper_mask, ink_mask)) / 3
    return {
        "Image Array Shape": pixels.shape,
        "Resolution": str(pixels.shape[0]) + "x" + str(pixels.shape[1]),
        "Total Pixels": str(total),
        "Paper Mask Array Shape": paper_mask.shape,
        "Paper Pixels": str(paper),
        "Paper Percentage": str(paper / total * 100.0),
        "Ink Mask Array Shape": ink_mask.shape,
        "Ink Pixels": str(ink),
        "Ink Percentage": str(ink / total * 100.0),
        "Unused Pixels": str(unused),
        "Unused Percentage": str(unused / total * 100.0),
        "Pixels in Both": str(both),
        "Pixels in Both Percentage": str(both / total * 100.0),
    }


def _analyze_single_image(image, op, progress, check_cancelled):
    """Report one image's masked colour statistics without comparison metrics."""
    if check_cancelled is not None:
        check_cancelled()
    if progress is not None:
        progress("Preparing masks and image data")
    results = {}
    if op.show_analysis_options:
        results["Analysis Options"] = op.to_dict()
    paper = image.masks["paper"]
    ink = image.masks["ink"]
    if not op.use_masks:
        paper = np.ones_like(paper, dtype=bool)
        ink = np.ones_like(ink, dtype=bool)
    if check_cancelled is not None:
        check_cancelled()
    if progress is not None:
        progress("Summarising mask data")
    if op.show_image_data:
        if progress is not None:
            progress("Compiling image data")
        results["Image Data"] = {"Image": _image_report_data(image, paper, ink)}

    for enabled, space, heading, channels, distance_fn in (
        (op.rgb_comparison, "sRGB", "sRGB colour space", ("R", "G", "B"), rgb_distance_dict),
        (op.lab_comparison, "CEILAB", "CEILAB (L*a*b*) colour space", ("L*", "a*", "b*"), delta_e_dict),
    ):
        if not enabled:
            continue
        if progress is not None:
            progress(f"Analysing {space} colours")
        channel_dps, delta_e_dp = _channel_decimals(op, space)
        if space == "sRGB":
            scale = op.srgb_scale
            results[heading] = {"(Scale": f"{0.0:.{channel_dps[0]}f} to {scale:.{channel_dps[0]}f})"}
            pixels = image.get_pixels("rgb")
        else:
            scale = _ceilab_channel_scale(op)
            results[heading] = _ceilab_scale_heading(op, channel_dps)
            pixels = image.get_pixels("lab")
        if op.mean_comparison:
            raw = _compare_configured_images(
                {"b1": pixels}, {"b1": paper}, {"b1": ink}, op.use_masks, "single",
                scale, distance_fn, check_cancelled, progress, space,
            )
            results[heading].update(_format_configured_data(
                raw, (), (), channels, op.use_masks, channel_dps, "", delta_e_dp))
    if check_cancelled is not None:
        check_cancelled()
    return results


def image_analysis_b1_b2_a(
    b1: image_data_colour,
    b2: image_data_colour | None,
    a: image_data_colour | None,
    op: analysis_options,
    progress: Callable[[str], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
    mode: str = "two_before",
) -> dict:
    """Compare the configured before and after images and return the formatted report.

    If provided, progress is called with each task's description as it starts.
    check_cancelled may raise to stop the analysis between expensive operations.
    """

    if mode not in ("single", "two_before", "one_each", "two_after"):
        raise ValueError(f"Unknown image configuration: {mode}")
    if mode == "single":
        return _analyze_single_image(b1, op, progress, check_cancelled)
    if a is None:
        raise ValueError("The selected configuration requires an after image.")
    if mode == "one_each":
        b2 = b1
    elif b2 is None:
        raise ValueError("The selected configuration requires a middle image.")

    if check_cancelled is not None:
        check_cancelled()
    if progress is not None:
        progress("Preparing masks and image data")

    # Output options
    results = dict()

    if op.show_analysis_options:
        results["Analysis Options"] = op.to_dict()

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
            if check_cancelled is not None:
                check_cancelled()
            paper_mask[k] = np.ones_like(paper_mask[k], dtype=bool)
        for k in ink_mask:
            if check_cancelled is not None:
                check_cancelled()
            ink_mask[k] = np.ones_like(ink_mask[k], dtype=bool)

    if check_cancelled is not None:
        check_cancelled()
    if progress is not None:
        progress("Summarising mask data")
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
    if check_cancelled is not None:
        check_cancelled()
    ink_pixels = {
        "b1": np.sum(ink_mask["b1"])/3,
        "b2": np.sum(ink_mask["b2"])/3,
        "a": np.sum(ink_mask["a"])/3
    }
    if check_cancelled is not None:
        check_cancelled()
    test_union_mask = {
        "b1": np.maximum(paper_mask["b1"], ink_mask["b1"]),
        "b2": np.maximum(paper_mask["b2"], ink_mask["b2"]),
        "a": np.maximum(paper_mask["a"], ink_mask["a"])
    }
    if check_cancelled is not None:
        check_cancelled()
    test_intersection_mask = {
        "b1": np.minimum(paper_mask["b1"], ink_mask["b1"]),
        "b2": np.minimum(paper_mask["b2"], ink_mask["b2"]),
        "a": np.minimum(paper_mask["a"], ink_mask["a"])
    }

    if op.show_image_data:
        if progress is not None:
            progress("Compiling image data")
        results["Image Data"] = {
            "Before 1": _image_report_data(b1, paper_mask["b1"], ink_mask["b1"]),
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

    if op.show_image_data:
        image_data = results["Image Data"]
        if mode == "one_each":
            results["Image Data"] = {"Before": image_data["Before 1"], "After": image_data["After"]}
        elif mode == "two_after":
            results["Image Data"] = {
                "Before": image_data["Before 1"],
                "After 1": image_data["Before 2"],
                "After 2": image_data["After"],
            }

    srgb_dps, srgb_de_dp = _channel_decimals(op, "sRGB")
    ceilab_dps, ceilab_de_dp = _channel_decimals(op, "CEILAB")

    # sRGB comparisons
    if op.rgb_comparison:
        if progress is not None:
            progress("Analysing sRGB colours")
        srgb_scale = op.srgb_scale
        rgb = {
            "b1": b1.get_pixels("rgb"),
            "b2": b2.get_pixels("rgb"),
            "a": a.get_pixels("rgb")
        }

        # mean comparison
        srgb_mean_heading = "Overall change in sRGB colour space"
        results[srgb_mean_heading] = {"(Scale": f"{0.0:.{srgb_dps[0]}f} to {srgb_scale:.{srgb_dps[0]}f})"}

        if op.mean_comparison:
            raw_data_rgb = _compare_configured_images(rgb, paper_mask, ink_mask, op.use_masks, mode, srgb_scale, rgb_distance_dict, check_cancelled, progress, "sRGB")
            
            rgb_paper_diff_text = describe_rgb_change(raw_data_rgb["paper"]["change"]["avg"]/srgb_scale)
            if op.use_masks:
                rgb_ink_diff_text = describe_rgb_change(raw_data_rgb["ink"]["change"]["avg"]/srgb_scale)
            else:
                rgb_ink_diff_text = ["","",""]

            results[srgb_mean_heading].update(_format_configured_data(raw_data_rgb, rgb_paper_diff_text, rgb_ink_diff_text, ("R", "G", "B"), op.use_masks, srgb_dps, "Colour Difference (sRGB)", srgb_de_dp))

        # Per-pixel comparison (todo)
        if op.per_pixel_comparison:
            """
            rgb_tile_images = {
                "b1": split_image(rgb["b1"],op.heatmap_grid_size_x,op.heatmap_grid_size_y),
                "b2": split_image(rgb["b2"],op.heatmap_grid_size_x,op.heatmap_grid_size_y),
                "a": split_image(rgb["a"],op.heatmap_grid_size_x,op.heatmap_grid_size_y)
            }

            tile_mask_paper = {
                "b1": split_image(paper_mask["b1"],op.heatmap_grid_size_x,op.heatmap_grid_size_y),
                "b2": split_image(paper_mask["b2"],op.heatmap_grid_size_x,op.heatmap_grid_size_y),
                "a": split_image(paper_mask["a"],op.heatmap_grid_size_x,op.heatmap_grid_size_y)
            }

            rgb_reduced_tile_images = dict()
            for k in rgb_tile_images:
                rgb_reduced_tile_images[k] = grid_averages(rgb_tile_images[k],tile_mask_paper[k])

            # Compare each chunk before and after
            rgb_reduced_tile_images["b_avg"] = (rgb_reduced_tile_images["b1"] + rgb_reduced_tile_images["b2"]) / 2.0
            rgb_reduced_tile_images["diff"] = rgb_reduced_tile_images["a"]-rgb_reduced_tile_images["b_avg"]
            """

            # Process raw numbers into mean change and sd


            # Add to results


            # Make R, G and B change comparison heatmaps, recolour based on scale

            # Make all heatmaps ready to show on UI
            h = len(rgb["b1"])
            w = len(rgb["b1"][0])

            """
            #debug test for saving heatmaps
            for k in rgb_reduced_tile_images:
                save_rgb_image(rgb_reduced_tile_images[k],"debug_heatmap",k)
                
                img = upscale_blocky(rgb_reduced_tile_images[k], h, w, dtype=np.float32)
                save_rgb_image(img,"debug_heatmap",k+"_block")

                img = upscale_smooth(rgb_reduced_tile_images[k], h, w, dtype=np.float32)
                save_rgb_image(img,"debug_heatmap",k+"_smooth")
            """

            # Make UI for heatmaps and have button for save as png (same res as original) 



    # CEILAB comparisons
    if op.lab_comparison:
        if progress is not None:
            progress("Analysing CEILAB colours")
        ceilab_scale = _ceilab_channel_scale(op)
        lab = {
            "b1": b1.get_pixels("lab"),
            "b2": b2.get_pixels("lab"),
            "a": a.get_pixels("lab")
        }

        ceilab_mean_heading = "Overall change in CEILAB (L*a*b*) colour space"
        results[ceilab_mean_heading] = _ceilab_scale_heading(op, ceilab_dps)

        # mean comparison
        if op.mean_comparison:
            raw_data_lab = _compare_configured_images(lab, paper_mask, ink_mask, op.use_masks, mode, ceilab_scale, delta_e_dict, check_cancelled, progress, "CEILAB")
            
            lab_paper_diff_text = describe_lab_change(raw_data_lab["paper"]["change"]["avg"]/ceilab_scale)
            if op.use_masks:
                lab_ink_diff_text = describe_lab_change(raw_data_lab["ink"]["change"]["avg"]/ceilab_scale)
            else:
                lab_ink_diff_text = ["","",""]

            results[ceilab_mean_heading].update(_format_configured_data(raw_data_lab, lab_paper_diff_text, lab_ink_diff_text, ("L*", "a*", "b*"), op.use_masks, ceilab_dps, "Perceptual Colour Difference (ΔE)", ceilab_de_dp))


        # Per-pixel comparison (todo)
        if op.per_pixel_comparison:
            pass


    if check_cancelled is not None:
        check_cancelled()
    return results
