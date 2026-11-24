import threading
import time

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

import main
from main import MainWindow
from paperdiff.image_state import image_data_colour


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    yield window
    window.close()


@pytest.fixture
def image_paths(tmp_path):
    paths = {}
    for key, value in (("B1", 30), ("B2", 100), ("A", 200)):
        path = tmp_path / f"{key}.png"
        Image.new("RGB", (4, 3), (value, value, value)).save(path)
        paths[key] = str(path)
    return paths


def select_image(window, monkeypatch, key, path):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (path, ""))
    text = f"Select {window.image_key_labels[key]}"
    button = next(button for button in window.load_tab.findChildren(QPushButton) if button.text() == text)
    button.click()


def wait_until(predicate):
    deadline = time.monotonic() + 10
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate(), "Timed out waiting for analysis"


def test_select_previews_then_loads_all_images(window, monkeypatch, image_paths, tmp_path):
    original_load = image_data_colour.load_from_path
    loaded_paths = []

    def record_load(image, path):
        loaded_paths.append(path)
        return original_load(image, path)

    monkeypatch.setattr(image_data_colour, "load_from_path", record_load)
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
        assert not window.load_preview_labels[key].pixmap().isNull()
        assert window.path_labels[key].text() == path
        assert window.images[key].pixels == {}
    assert loaded_paths == []
    assert window.load_images_button.isEnabled()
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))
    assert not window.analyze_button.isEnabled()

    window.load_images_button.click()

    assert loaded_paths == list(image_paths.values())
    for key in image_paths:
        image = window.images[key]
        assert image.file_path == image_paths[key]
        assert image.pixels["rgb"].shape == (3, 4, 3)
        assert image.pixels["lab"].shape == (3, 4, 3)
        assert image.masks["paper"].shape == (3, 4, 3)
        assert not window.mask_original_labels[key]._base_pixmap.isNull()
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))
    assert window.analyze_button.isEnabled()
    assert window.alignment_checkbox.isEnabled()

    monkeypatch.setattr(QMessageBox, "critical", lambda *args: pytest.fail(args[2]))
    window.tab_widget.setCurrentWidget(window.analysis_options_tab)
    window.analyze_button.click()
    wait_until(lambda: window._analysis_thread is None)
    assert "Overall change in RGB colour space" in window.report.toPlainText()
    assert window.tab_widget.currentWidget() is window.results_tab

    window.tab_widget.setCurrentWidget(window.load_tab)
    window.images["B1"].set_mask("paper", np.zeros((3, 4), dtype=bool))
    window.mask_preview_labels["B1"].setText("old mask")
    window.report.setPlainText("old report")
    window.alignment_checkbox.setChecked(True)
    replacement = tmp_path / "replacement.png"
    Image.new("RGB", (5, 3), "red").save(replacement)
    select_image(window, monkeypatch, "A", str(replacement))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.mask_tab))
    window.load_images_button.click()
    assert window.images["A"].pixels["rgb"].shape == (3, 5, 3)
    assert not window.images["B1"].masks_exists["paper"]
    assert window.mask_preview_labels["B1"].text() == "Mask B1"
    assert window.report.toPlainText() == ""
    assert not window.alignment_checkbox.isEnabled()
    assert not window.alignment_checkbox.isChecked()


def test_failed_load_keeps_previous_set_and_blocks_downstream(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    previous_images = window.images

    select_image(window, monkeypatch, "B2", image_paths["B2"])
    original_load = image_data_colour.load_from_path
    errors = []

    def fail_second(image, path):
        if image.key == "B2":
            raise OSError("test failure")
        return original_load(image, path)

    monkeypatch.setattr(image_data_colour, "load_from_path", fail_second)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))
    window.load_images_button.click()
    assert window.images is previous_images
    assert "B2" in errors[0][2]
    assert "test failure" in errors[0][2]
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))
    assert not window.analyze_button.isEnabled()
    assert window.load_images_button.isEnabled()

    monkeypatch.setattr(image_data_colour, "load_from_path", original_load)
    window.load_images_button.click()
    assert window.images is not previous_images
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))
    assert window.analyze_button.isEnabled()


def test_invalid_or_cancelled_selection_does_not_replace_preview(window, monkeypatch, image_paths, tmp_path):
    select_image(window, monkeypatch, "B1", image_paths["B1"])
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))
    select_image(window, monkeypatch, "B1", "")
    invalid = tmp_path / "invalid.png"
    invalid.write_text("not an image")
    select_image(window, monkeypatch, "B1", str(invalid))
    assert window.selected_image_paths["B1"] == image_paths["B1"]
    assert window.path_labels["B1"].text() == image_paths["B1"]
    assert not window.load_preview_labels["B1"].pixmap().isNull()
    assert len(errors) == 1


def test_analysis_button_and_result_tabs(window, monkeypatch):
    assert window.tab_widget.tabText(window.tab_widget.indexOf(window.analysis_options_tab)) == "4) Analysis"
    assert window.tab_widget.tabText(window.tab_widget.indexOf(window.results_tab)) == "5) Results data"
    assert window.tab_widget.tabText(window.tab_widget.indexOf(window.image_results_tab)) == "6) Image results"
    assert window.analyze_button.parentWidget() is window.analysis_options_tab
    assert not window.results_tab.findChildren(QPushButton)
    layout = window.analysis_options_tab.layout()
    assert layout.indexOf(window.analysis_progress_label) > layout.indexOf(window.analysis_progress_bar)
    assert window.analysis_progress_bar.isTextVisible()

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    window.tab_widget.setCurrentWidget(window.analysis_options_tab)
    window.analyze_button.click()
    assert "Load B1, B2, and A" in warnings[0][2]
    assert window.tab_widget.currentWidget() is window.analysis_options_tab


def test_analysis_runs_off_ui_thread_and_keeps_inputs_stable(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    window.tab_widget.setCurrentWidget(window.analysis_options_tab)
    window.images["B1"].set_mask("paper", np.array([[True, True, False, True]] * 3))
    window.images["B2"].set_mask("paper", np.array([[True, False, True, True]] * 3))
    window.images["A"].set_mask("paper", np.array([[True, True, True, False]] * 3))
    window.alignment_checkbox.setChecked(True)
    window.intersection_mask_checkbox.setChecked(True)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_analysis(b1, b2, after, options, *, progress):
        calls.append((threading.get_ident(), options))
        progress("Preparing masks and image data")
        started.set()
        if not release.wait(5):
            raise RuntimeError("Timed out waiting for test")
        return {"Result": "complete"}

    monkeypatch.setattr(main, "image_analysis_b1_b2_a", slow_analysis)
    window.analyze_button.click()
    try:
        wait_until(started.is_set)
        assert calls[0][0] != threading.get_ident()
        assert calls[0][1] is not window.analysis_options
        expected_mask = np.array([[True, False, False, False]] * 3)
        assert np.array_equal(calls[0][1].int_paper_mask, np.repeat(expected_mask[..., None], 3, axis=-1))
        wait_until(lambda: window.analysis_progress_label.text().startswith("Task 2/4"))
        assert window.analysis_progress_bar.value() == 25
        assert not window.analyze_button.isEnabled()
        assert not window.analysis_settings_group.isEnabled()
        assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.load_tab))
        assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.mask_tab))
        window.analyze_images()
        assert len(calls) == 1
        responded = []
        QTimer.singleShot(0, lambda: responded.append(True))
        wait_until(lambda: bool(responded))
        assert window._analysis_thread is not None
    finally:
        release.set()
        wait_until(lambda: window._analysis_thread is None)

    assert window.report.toPlainText() == "Result: complete\n"
    assert window.tab_widget.currentWidget() is window.results_tab
    assert window.analysis_progress_bar.value() == 100
    assert window.analyze_button.isEnabled()
    assert window.analysis_settings_group.isEnabled()
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.load_tab))


def test_analysis_error_restores_controls_and_allows_retry(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    window.report.setPlainText("previous report")
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))

    def fail_analysis(*args, progress):
        progress("Preparing masks and image data")
        progress("Analysing RGB colours")
        raise ValueError("analysis broke")

    monkeypatch.setattr(main, "image_analysis_b1_b2_a", fail_analysis)
    window.analyze_button.click()
    wait_until(lambda: window._analysis_thread is None)
    assert "analysis broke" in errors[0][2]
    assert window.report.toPlainText() == "previous report"
    assert window.analysis_progress_bar.value() == 33
    assert window.analysis_progress_label.text() == "Analysis failed. Task 2/3: Analysing RGB colours"
    assert window.analyze_button.isEnabled()
    assert window.analysis_settings_group.isEnabled()
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.load_tab))

    monkeypatch.setattr(main, "image_analysis_b1_b2_a", lambda *args, **kwargs: {"retry": "ok"})
    window.analyze_button.click()
    wait_until(lambda: window._analysis_thread is None)
    assert window.report.toPlainText() == "retry: ok\n"


def test_close_waits_for_running_analysis(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    window.show()
    started = threading.Event()
    release = threading.Event()

    def slow_analysis(*args, **kwargs):
        started.set()
        if not release.wait(5):
            raise RuntimeError("Timed out waiting for test")
        return {}

    monkeypatch.setattr(main, "image_analysis_b1_b2_a", slow_analysis)
    window.analyze_button.click()
    try:
        wait_until(started.is_set)
        window.close()
        assert window.isVisible()
        assert "close" in window.analysis_progress_label.text()
    finally:
        release.set()
        wait_until(lambda: window._analysis_thread is None)
    assert not window.isVisible()


def test_analysis_reports_each_real_stage_and_completed_percentage(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    window.tab_widget.setCurrentWidget(window.analysis_options_tab)
    window.alignment_checkbox.setChecked(True)
    window.intersection_mask_checkbox.setChecked(True)

    permits = threading.Semaphore(0)
    intersection = main._intersection_masks
    original_analysis = main.image_analysis_b1_b2_a

    def paused_intersection(images):
        if not permits.acquire(timeout=5):
            raise RuntimeError("Timed out waiting for test")
        return intersection(images)

    def paused_analysis(b1, b2, after, options, *, progress):
        def pause_at_stage(description):
            progress(description)
            if not permits.acquire(timeout=5):
                raise RuntimeError("Timed out waiting for test")
        return original_analysis(b1, b2, after, options, progress=pause_at_stage)

    monkeypatch.setattr(main, "_intersection_masks", paused_intersection)
    monkeypatch.setattr(main, "image_analysis_b1_b2_a", paused_analysis)
    window.analyze_button.click()
    try:
        for task, description in enumerate((
            "Calculating intersection masks",
            "Preparing masks and image data",
            "Analysing RGB colours",
            "Analysing CIELAB colours",
        ), start=1):
            wait_until(lambda: window.analysis_progress_label.text() == f"Task {task}/4: {description}")
            assert window.analysis_progress_bar.value() == (task - 1) * 25
            permits.release()
        wait_until(lambda: window._analysis_thread is None)
    finally:
        for _ in range(4):
            permits.release()
        if window._analysis_thread is not None:
            wait_until(lambda: window._analysis_thread is None)

    assert window.analysis_progress_bar.value() == 100
    assert window.analysis_progress_label.text() == "Analysis complete (4/4 tasks)."
    assert "Overall change in RGB colour space" in window.report.toPlainText()


@pytest.mark.parametrize(("rgb", "lab", "total"), [(False, False, 1), (True, False, 2), (False, True, 2)])
def test_analysis_task_count_reflects_enabled_options(window, monkeypatch, image_paths, rgb, lab, total):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    window.rgb_comparison_checkbox.setChecked(rgb)
    window.lab_comparison_checkbox.setChecked(lab)
    window.analyze_button.click()
    wait_until(lambda: window._analysis_thread is None)
    assert window.analysis_progress_label.text() == f"Analysis complete ({total}/{total} tasks)."
    assert window.analysis_progress_bar.value() == 100
