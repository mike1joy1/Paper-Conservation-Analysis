from pathlib import Path

import numpy as np
import pytest

import paperdiff.workflow as workflow
from paperdiff.image_state import analysis_options, image_data_colour
from paperdiff.loading import load_image
from paperdiff.workflow import delta_e_dict, describe_region_change, image_analysis_b1_b2_a, rgb_distance_dict


@pytest.mark.skip(reason="Tests obsolete workflow API - current API uses image_data_colour objects")
def test_evaluate_reference_pair_reports_base_and_test_metrics():
    root = Path(__file__).resolve().parent.parent
    b1 = load_image(root / "test_images" / "B1.png")
    b2 = load_image(root / "test_images" / "B2.png")
    a = load_image(root / "test_images" / "A.png")

    op = analysis_options()
    results = image_analysis_b1_b2_a(b1, b2, a, op)

    assert set(results) == {"base", "test"}
    assert results["base"]["score"] >= 0.0
    assert results["test"]["score"] >= 0.0
    assert set(results["base"]["paper"]) >= {"mean", "std", "max", "n_pixels"}
    assert set(results["base"]["ink"]) >= {"mean", "std", "max", "n_pixels"}
    assert set(results["test"]["paper"]) >= {"mean", "std", "max", "n_pixels"}
    assert set(results["test"]["ink"]) >= {"mean", "std", "max", "n_pixels"}


@pytest.mark.skip(reason="Tests obsolete workflow API - current API uses image_data_colour objects")
def test_evaluate_reference_pair_accepts_already_aligned_images():
    root = Path(__file__).resolve().parent.parent
    b1 = load_image(root / "test_images" / "B1.png")
    b2 = load_image(root / "test_images" / "B2.png")
    a = load_image(root / "test_images" / "A.png")

    op = analysis_options()
    results = image_analysis_b1_b2_a(b1, b2, a, op)

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

    assert rgb_distance_dict(black, black) == {"sRGB Euclidean": 0.0, "sRGB redmean": 0.0}
    assert rgb_distance_dict(black, red)["sRGB Euclidean"] == pytest.approx(100.0)
    assert rgb_distance_dict(black, red)["sRGB redmean"] == pytest.approx(100 * np.sqrt(2 + 127.5 / 256))
    assert rgb_distance_dict(black, blue)["sRGB redmean"] == pytest.approx(100 * np.sqrt(2 + 255 / 256))
    assert rgb_distance_dict(red, [1.0, 0.0, 1.0])["sRGB redmean"] == pytest.approx(100 * np.sqrt(2))
    assert rgb_distance_dict(red, black) == rgb_distance_dict(black, red)
    with pytest.raises(ValueError):
        rgb_distance_dict([0.0, 0.0], black)


@pytest.mark.parametrize(
    ("rgb", "lab"),
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_analysis_progress_follows_enabled_tasks_without_changing_results(rgb, lab):
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

    expected = ["Preparing masks and image data", "Summarising mask data"]
    for enabled, colour_space in ((rgb, "sRGB"), (lab, "CEILAB")):
        if enabled:
            expected.append(f"Analysing {colour_space} colours")
            for region in ("paper", "ink"):
                for image in ("Before Image 1", "Before Image 2", "After Image"):
                    expected.append(f"Analysing {colour_space} {region}: {image}")
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
    rgb = results["Overall change in sRGB colour space"]
    region = rgb["Paper" if use_masks else "Whole image"]["Change"]["Colour Difference (sRGB)"]
    if use_masks:
        assert region["sRGB Euclidean"].strip() == "50.00  (±50.00)"
        assert region["sRGB redmean"].strip() == "79.03  (±79.03)"
        ink = rgb["Ink"]["Change"]["Colour Difference (sRGB)"]
        assert ink["sRGB Euclidean"].strip() == "100.00  (±0.00)"
        assert ink["sRGB redmean"].strip() == "173.09  (±0.00)"
    else:
        assert region["sRGB Euclidean"].strip() == "60.36  (±10.36)"
    lab = results["Overall change in CEILAB (L*a*b*) colour space"]
    lab_difference = lab["Paper" if use_masks else "Whole image"]["Change"]["Perceptual Colour Difference (ΔE)"]
    assert set(lab_difference) == {"ΔE76  ", "ΔE94  ", "ΔE2000", "ΔE CMC"}
    if use_masks:
        b2_lab = images[1].get_pixels("lab")[0, 0]
        a_lab = images[2].get_pixels("lab")[0, 0]
        half_delta = np.linalg.norm(b2_lab - a_lab) / 2
        assert lab_difference["ΔE76  "].strip() == f"{half_delta:.2f}  (±{half_delta:.2f})"


@pytest.mark.parametrize(
    ("mode", "before", "after", "change", "distance", "image_names"),
    [
        ("one_each", "25.5", "127.5", "+102.0", "40.00", {"Before", "After"}),
        ("two_before", "51.0  (±25.5)", "127.5", "+76.5  (±25.5)",
         "30.00  (±10.00)", {"Before 1", "Before 2", "After"}),
        ("two_after", "25.5", "102.0  (±25.5)", "+76.5  (±25.5)",
         "30.00  (±10.00)", {"Before", "After 1", "After 2"}),
    ],
)
def test_image_configurations_report_replicate_uncertainty(mode, before, after, change, distance, image_names):
    images = []
    for key, red in (("b1", 0.1), ("b2", 0.3), ("a", 0.5)):
        image = image_data_colour(key)
        image.pixels["rgb"] = np.array([[[red, 0, 0], [red, 0, 0]]], dtype=np.float32)
        image.set_mask("paper", np.array([[True, False]]))
        image.set_mask("ink", np.array([[False, True]]))
        images.append(image)
    options = analysis_options()
    options.use_masks = True
    options.show_image_data = True
    options.dec_places = 2
    stages = []

    result = image_analysis_b1_b2_a(images[0], None if mode == "one_each" else images[1],
                                    images[2], options, mode=mode, progress=stages.append)
    assert set(result["Image Data"]) == image_names
    for name in ("Paper", "Ink"):
        region = result["Overall change in sRGB colour space"][name]
        assert region["Before"]["Colour Average"]["R"].strip() == before
        assert region["After"]["Colour Average"]["R"].strip() == after
        assert region["Change"]["Colour Average"]["R"].strip() == change
        assert region["Change"]["Colour Difference (sRGB)"]["sRGB Euclidean"].strip() == distance
        lab = result["Overall change in CEILAB (L*a*b*) colour space"][name]["Change"]
        pairs = {
            "one_each": ((0, 2),),
            "two_before": ((0, 2), (1, 2)),
            "two_after": ((0, 1), (0, 2)),
        }[mode]
        distances = [
            delta_e_dict(images[before].get_pixels("lab")[0, 0],
                         images[after].get_pixels("lab")[0, 0])["ΔE76"]
            for before, after in pairs
        ]
        expected_lab = f"{sum(distances) / len(distances):.2f}"
        if len(distances) == 2:
            expected_lab += f"  (±{abs(distances[0] - distances[1]) / 2:.2f})"
        assert lab["Perceptual Colour Difference (ΔE)"]["ΔE76  "].strip() == expected_lab
    expected_images = 2 if mode == "one_each" else 3
    assert len([stage for stage in stages if stage.startswith("Analysing sRGB paper:")]) == expected_images


@pytest.mark.parametrize("use_masks", [False, True])
def test_single_image_reports_colour_statistics_without_comparison(use_masks, monkeypatch):
    image = image_data_colour("image")
    image.pixels["rgb"] = np.array([[[0, 0, 0], [0.2, 0, 0], [0.6, 0, 0], [1, 0, 0]]], dtype=np.float32)
    image.set_mask("paper", np.array([[True, True, False, False]]))
    image.set_mask("ink", np.array([[False, False, True, True]]))
    options = analysis_options()
    options.use_masks = use_masks
    options.show_image_data = True
    options.dec_places = 1
    options.images_aligned = True
    options.intersection_mask = True
    options.int_paper_mask = np.zeros((1, 4, 3))
    options.int_ink_mask = np.zeros((1, 4, 3))
    stages = []
    def unexpected_distance(*args):
        pytest.fail("Single-image analysis must not calculate colour distances")

    monkeypatch.setattr(workflow, "rgb_distance_dict", unexpected_distance)
    monkeypatch.setattr(workflow, "delta_e_dict", unexpected_distance)

    result = image_analysis_b1_b2_a(image, None, None, options, mode="single", progress=stages.append)
    assert set(result["Image Data"]) == {"Image"}
    rgb = result["sRGB colour space"]
    lab = result["CEILAB (L*a*b*) colour space"]
    regions = ("Paper", "Ink") if use_masks else ("Whole image",)
    for region in regions:
        assert set(rgb[region]) == {"Image"}
        assert set(lab[region]) == {"Image"}
        assert set(rgb[region]["Image"]) == {"Colour Average", "Colour Standard Deviation"}
        assert "±" not in str(rgb[region])
        assert "ΔE" not in str(lab[region])
    if use_masks:
        assert rgb["Paper"]["Image"]["Colour Average"]["R"].strip() == "25.5"
        assert rgb["Paper"]["Image"]["Colour Standard Deviation"]["R"].strip() == "25.5"
        assert rgb["Ink"]["Image"]["Colour Average"]["R"].strip() == "204.0"
        assert rgb["Ink"]["Image"]["Colour Standard Deviation"]["R"].strip() == "51.0"
    else:
        assert rgb["Whole image"]["Image"]["Colour Average"]["R"].strip() == "114.8"
        assert rgb["Whole image"]["Image"]["Colour Standard Deviation"]["R"].strip() == "97.9"
    assert len([stage for stage in stages if stage.startswith("Analysing sRGB paper:")]) == 1


def test_single_image_analysis_checks_cancellation_during_statistics():
    image = image_data_colour("image")
    image.pixels["rgb"] = np.ones((2, 2, 3), dtype=np.float32)
    image.clear_masks()
    options = analysis_options()
    stages = []

    def check_cancelled():
        if "Analysing sRGB paper: Image" in stages:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        image_analysis_b1_b2_a(image, None, None, options, mode="single",
                               progress=stages.append, check_cancelled=check_cancelled)
    assert "Analysing CEILAB colours" not in stages


def test_single_image_analysis_honours_disabled_colour_and_mean_options():
    image = image_data_colour("image")
    image.pixels["rgb"] = np.ones((2, 2, 3), dtype=np.float32)
    image.clear_masks()
    options = analysis_options()
    options.rgb_comparison = False
    options.mean_comparison = False
    stages = []

    result = image_analysis_b1_b2_a(image, None, None, options, mode="single", progress=stages.append)
    assert "sRGB colour space" not in result
    assert set(result["CEILAB (L*a*b*) colour space"]) == {"(L* Scale", "(a*b* Scale"}
    assert stages == ["Preparing masks and image data", "Summarising mask data", "Analysing CEILAB colours"]


def test_ceilab_delta_e_ignores_user_scales():
    images = []
    for key, rgb in (("b1", [0.2, 0.4, 0.6]), ("b2", [0.3, 0.4, 0.5]), ("a", [0.7, 0.3, 0.1])):
        image = image_data_colour(key)
        image.pixels["rgb"] = np.array([[rgb, rgb]], dtype=np.float32)
        image.clear_masks()
        images.append(image)

    def delta_e_for(l_scale, ab_scale):
        options = analysis_options()
        options.rgb_comparison = False
        options.use_masks = False
        options.ceilab_l_scale = l_scale
        options.ceilab_ab_scale = ab_scale
        result = image_analysis_b1_b2_a(*images, options)
        return result["Overall change in CEILAB (L*a*b*) colour space"]["Whole image"]["Change"][
            "Perceptual Colour Difference (ΔE)"]

    assert delta_e_for(100.0, 128.0) == delta_e_for(1.0, 1.0) == delta_e_for(500.0, 7.0)
    expected = np.mean([
        delta_e_dict(images[before].get_pixels("lab")[0, 0], images[2].get_pixels("lab")[0, 0])["ΔE76"]
        for before in (0, 1)
    ])
    assert float(delta_e_for(1.0, 1.0)["ΔE76  "].split()[0]) == pytest.approx(expected, abs=0.05)


def test_ceilab_scales_map_l_to_zero_to_x_and_ab_to_minus_x_to_x():
    image = image_data_colour("image")
    image.pixels["rgb"] = np.array([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32)
    image.clear_masks()
    options = analysis_options()
    options.rgb_comparison = False
    options.use_masks = False
    options.dec_places = 3
    options.ceilab_l_scale = 1.0
    options.ceilab_ab_scale = 1.0

    result = image_analysis_b1_b2_a(image, None, None, options, mode="single")
    average = result["CEILAB (L*a*b*) colour space"]["Whole image"]["Image"]["Colour Average"]
    lab = image.get_pixels("lab")[0, 0]
    assert float(average["L*"]) == pytest.approx(lab[0] / 100, abs=1e-3)
    assert float(average["a*"]) == pytest.approx(lab[1] / 128, abs=1e-3)
    assert float(average["b*"]) == pytest.approx(lab[2] / 128, abs=1e-3)
    assert 0 <= float(average["L*"]) <= 1
    assert -1 <= float(average["a*"]) <= 1 and -1 <= float(average["b*"]) <= 1

