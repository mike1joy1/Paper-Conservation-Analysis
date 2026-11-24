from pathlib import Path

import numpy as np
import pytest

from paperdiff.image_state import analysis_options, image_data_colour
from paperdiff.loading import load_image
from paperdiff.workflow import describe_region_change, image_analysis_b1_b2_a, rgb_distance_dict


def test_evaluate_reference_pair_reports_base_and_test_metrics():
    root = Path(__file__).resolve().parent.parent
    b1 = load_image(root / "test_images" / "B1.png")
    b2 = load_image(root / "test_images" / "B2.png")
    a = load_image(root / "test_images" / "A.png")

    results = image_analysis_b1_b2_a(b1, b2, a)

    assert set(results) == {"base", "test"}
    assert results["base"]["score"] >= 0.0
    assert results["test"]["score"] >= 0.0
    assert set(results["base"]["paper"]) >= {"mean", "std", "max", "n_pixels"}
    assert set(results["base"]["ink"]) >= {"mean", "std", "max", "n_pixels"}
    assert set(results["test"]["paper"]) >= {"mean", "std", "max", "n_pixels"}
    assert set(results["test"]["ink"]) >= {"mean", "std", "max", "n_pixels"}


def test_evaluate_reference_pair_accepts_already_aligned_images():
    root = Path(__file__).resolve().parent.parent
    b1 = load_image(root / "test_images" / "B1.png")
    b2 = load_image(root / "test_images" / "B2.png")
    a = load_image(root / "test_images" / "A.png")

    results = image_analysis_b1_b2_a(b1, b2, a)

    assert results["base"]["reference"].shape == b1.shape
    assert results["test"]["reference"].shape == b1.shape
    assert np.isfinite(results["base"]["score"])
    assert np.isfinite(results["test"]["score"])


def test_describe_region_change_mentions_measured_properties():
    metrics = {"mean": 1.8, "dL_median": 1.4, "db_median": 2.3}
    text = describe_region_change(metrics, "paper")

    assert "ΔE00" in text
    assert "brightness" in text.lower()
    assert "yellow" in text.lower() or "blue" in text.lower()


def test_rgb_distances_use_report_scale_and_redmean_weights():
    black = [0.0, 0.0, 0.0]
    red = [1.0, 0.0, 0.0]
    blue = [0.0, 0.0, 1.0]

    assert rgb_distance_dict(black, black) == {"RGB Euclidean": 0.0, "RGB redmean": 0.0}
    assert rgb_distance_dict(black, red)["RGB Euclidean"] == pytest.approx(100.0)
    assert rgb_distance_dict(black, red)["RGB redmean"] == pytest.approx(100 * np.sqrt(2 + 127.5 / 256))
    assert rgb_distance_dict(black, blue)["RGB redmean"] == pytest.approx(100 * np.sqrt(2 + 255 / 256))
    assert rgb_distance_dict(red, [1.0, 0.0, 1.0])["RGB redmean"] == pytest.approx(100 * np.sqrt(2))
    assert rgb_distance_dict(red, black) == rgb_distance_dict(black, red)
    with pytest.raises(ValueError):
        rgb_distance_dict([0.0, 0.0], black)


@pytest.mark.parametrize(
    ("rgb", "lab", "expected"),
    [
        (False, False, ["Preparing masks and image data"]),
        (True, False, ["Preparing masks and image data", "Analysing RGB colours"]),
        (False, True, ["Preparing masks and image data", "Analysing CIELAB colours"]),
        (True, True, ["Preparing masks and image data", "Analysing RGB colours", "Analysing CIELAB colours"]),
    ],
)
def test_analysis_progress_follows_enabled_tasks_without_changing_results(rgb, lab, expected):
    images = []
    for key in ("b1", "b2", "a"):
        image = image_data_colour(key)
        image.pixels["rgb"] = np.full((2, 2, 3), 0.5, dtype=np.float32)
        image.clear_masks()
        images.append(image)
    options = analysis_options()
    options.rgb_comparison = rgb
    options.lab_comparison = lab
    stages = []

    baseline = image_analysis_b1_b2_a(*images, options)
    result = image_analysis_b1_b2_a(*images, options, progress=stages.append)

    assert stages == expected
    assert result == baseline


@pytest.mark.parametrize("use_masks", [False, True])
def test_rgb_distances_in_report_with_before_uncertainty(use_masks):
    images = []
    for key, pixels in (
        ("b1", [[0, 0, 0], [0, 0, 0]]),
        ("b2", [[1, 0, 0], [0, 0, 0]]),
        ("a", [[0, 0, 0], [0, 0, 1]]),
    ):
        image = image_data_colour(key)
        image.pixels["rgb"] = np.array([pixels], dtype=np.float32)
        image.set_mask("paper", np.array([[True, False]]))
        image.set_mask("ink", np.array([[False, True]]))
        images.append(image)

    op = analysis_options()
    op.use_masks = use_masks
    op.dec_places = 2
    results = image_analysis_b1_b2_a(*images, op)
    rgb = results["Overall change in RGB colour space"]
    region = rgb["Paper" if use_masks else "Whole image"]["Change"]["Colour Difference (RGB)"]
    if use_masks:
        assert region["RGB Euclidean"].strip() == "50.00  (±50.00)"
        assert region["RGB redmean"].strip() == "79.03  (±79.03)"
        ink = rgb["Ink"]["Change"]["Colour Difference (RGB)"]
        assert ink["RGB Euclidean"].strip() == "100.00  (±0.00)"
        assert ink["RGB redmean"].strip() == "173.09  (±0.00)"
    else:
        assert region["RGB Euclidean"].strip() == "60.36  (±10.36)"
    lab = results["Overall change in CIELAB (L*a*b*) colour space"]
    lab_difference = lab["Paper" if use_masks else "Whole image"]["Change"]["Perceptual Colour Difference (ΔE)"]
    assert set(lab_difference) == {"ΔE76  ", "ΔE94  ", "ΔE2000", "ΔE CMC"}
    if use_masks:
        b2_lab = images[1].get_pixels("lab")[0, 0]
        a_lab = images[2].get_pixels("lab")[0, 0]
        half_delta = np.linalg.norm(b2_lab - a_lab) / 2
        assert lab_difference["ΔE76  "].strip() == f"{half_delta:.2f}  (±{half_delta:.2f})"
