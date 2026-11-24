import threading
import time

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

import main
import paperdiff.image_state as image_state
import paperdiff.workflow as workflow
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
    assert predicate(), "Timed out waiting for operation"


def test_select_previews_then_loads_all_images(window, monkeypatch, image_paths, tmp_path):
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.load_tab))
    for tab in (window.mask_tab, window.check_alignment_tab, window.analysis_options_tab, window.results_tab, window.image_results_tab):
        assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(tab))

    original_load = image_data_colour.load_from_path
    loaded_paths = []

    def record_load(image, path, check_cancelled=None):
        loaded_paths.append(path)
        return original_load(image, path, check_cancelled=check_cancelled)

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
    wait_until(lambda: window._loading_thread is None)

    assert loaded_paths == list(image_paths.values())
    for key in image_paths:
        image = window.images[key]
        assert image.file_path == image_paths[key]
        assert image.pixels["rgb"].shape == (3, 4, 3)
        assert image.pixels["lab"].shape == (3, 4, 3)
        assert image.masks["paper"].shape == (3, 4, 3)
        assert not window.mask_original_labels[key]._base_pixmap.isNull()
    for tab in (window.mask_tab, window.check_alignment_tab, window.analysis_options_tab):
        assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(tab))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))
    assert window.analyze_button.isEnabled()
    assert window.alignment_checkbox.isEnabled()

    monkeypatch.setattr(QMessageBox, "critical", lambda *args: pytest.fail(args[2]))
    window.tab_widget.setCurrentWidget(window.analysis_options_tab)
    window.analyze_button.click()
    wait_until(lambda: window._analysis_thread is None)
    assert "Overall change in RGB colour space" in window.report.toPlainText()
    assert window.tab_widget.currentWidget() is window.results_tab
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))

    window.tab_widget.setCurrentWidget(window.load_tab)
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))
    window.images["B1"].set_mask("paper", np.zeros((3, 4), dtype=bool))
    window.mask_preview_labels["B1"].setText("old mask")
    window.report.setPlainText("old report")
    window.alignment_checkbox.setChecked(True)
    replacement = tmp_path / "replacement.png"
    Image.new("RGB", (5, 3), "red").save(replacement)
    select_image(window, monkeypatch, "A", str(replacement))
    for tab in (window.mask_tab, window.check_alignment_tab, window.analysis_options_tab, window.results_tab, window.image_results_tab):
        assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(tab))
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    assert window.images["A"].pixels["rgb"].shape == (3, 5, 3)
    assert not window.images["B1"].masks_exists["paper"]
    assert window.mask_preview_labels["B1"].text() == "Mask B1"
    assert window.report.toPlainText() == ""
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.analysis_options_tab))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))
    assert not window.alignment_checkbox.isEnabled()
    assert not window.alignment_checkbox.isChecked()


def test_failed_load_keeps_previous_set_and_blocks_downstream(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    previous_images = window.images

    select_image(window, monkeypatch, "B2", image_paths["B2"])
    original_load = image_data_colour.load_from_path
    errors = []

    def fail_second(image, path, check_cancelled=None):
        if image.key == "B2":
            raise OSError("test failure")
        return original_load(image, path, check_cancelled=check_cancelled)

    monkeypatch.setattr(image_data_colour, "load_from_path", fail_second)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    assert window.images is previous_images
    assert "B2" in errors[0][2]
    assert "test failure" in errors[0][2]
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.analysis_options_tab))
    assert not window.analyze_button.isEnabled()
    assert window.load_images_button.isEnabled()

    monkeypatch.setattr(image_data_colour, "load_from_path", original_load)
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    assert window.images is not previous_images
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.analysis_options_tab))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))
    assert window.analyze_button.isEnabled()


def test_loading_progress_and_responsive_ui(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.show()
    permits = threading.Semaphore(0)
    calls = []
    original_load = image_data_colour.load_from_path

    def paused_load(image, path, check_cancelled=None):
        calls.append(threading.get_ident())
        if not permits.acquire(timeout=5):
            raise RuntimeError("Timed out waiting for test")
        return original_load(image, path, check_cancelled=check_cancelled)

    monkeypatch.setattr(image_data_colour, "load_from_path", paused_load)
    window.load_images_button.click()
    try:
        for task, key in enumerate(("B1", "B2", "A"), start=1):
            wait_until(lambda: window.load_status_label.text() == f"Loading {window.image_key_labels[key]} ({task}/3)")
            assert window.load_progress_bar.value() == (task - 1) * 100 // 3
            assert window.cancel_load_button.isVisible()
            assert not window.load_images_button.isEnabled()
            assert all(not button.isEnabled() for button in window.load_select_buttons.values())
            assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.analysis_options_tab))
            if task == 1:
                responded = []
                QTimer.singleShot(0, lambda: responded.append(True))
                wait_until(lambda: bool(responded))
                assert calls[0] != threading.get_ident()
            permits.release()
        wait_until(lambda: window._loading_thread is None)
    finally:
        for _ in range(3):
            permits.release()
        if window._loading_thread is not None:
            wait_until(lambda: window._loading_thread is None)
    assert window.load_progress_bar.value() == 100
    assert window.load_status_label.text() == "Images loaded (3/3)."
    assert window.cancel_load_button.isHidden()
    assert window.load_images_button.isEnabled()


@pytest.mark.parametrize("selection_changed", [False, True])
def test_cancel_loading_preserves_previous_data_and_tab_states(window, monkeypatch, image_paths, selection_changed):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    window.analyze_button.click()
    wait_until(lambda: window._analysis_thread is None)
    window.tab_widget.setCurrentWidget(window.load_tab)
    window.images["B1"].set_mask("paper", np.zeros((3, 4), dtype=bool))
    if selection_changed:
        select_image(window, monkeypatch, "B1", image_paths["B1"])
    previous_images = window.images
    previous_report = window.report.toPlainText()
    previous_tabs = [window.tab_widget.isTabEnabled(i) for i in range(1, window.tab_widget.count())]
    started = threading.Event()
    release = threading.Event()
    original_load = image_data_colour.load_from_path
    calls = []

    def paused_load(image, path, check_cancelled=None):
        calls.append(image.key)
        started.set()
        if not release.wait(5):
            raise RuntimeError("Timed out waiting for test")
        return original_load(image, path, check_cancelled=check_cancelled)

    monkeypatch.setattr(image_data_colour, "load_from_path", paused_load)
    window.load_images_button.click()
    try:
        wait_until(started.is_set)
        assert all(not window.tab_widget.isTabEnabled(i) for i in range(1, window.tab_widget.count()))
        window.cancel_load_button.click()
        assert not window.cancel_load_button.isEnabled()
        assert window.load_status_label.text() == "Cancelling image loading..."
    finally:
        release.set()
        wait_until(lambda: window._loading_thread is None)
    assert calls == ["B1"]
    assert window.images is previous_images
    assert window.images["B1"].masks_exists["paper"]
    assert window.report.toPlainText() == previous_report
    assert [window.tab_widget.isTabEnabled(i) for i in range(1, window.tab_widget.count())] == previous_tabs
    assert window.load_status_label.text() == "Image loading cancelled."

    monkeypatch.setattr(image_data_colour, "load_from_path", original_load)
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    assert window.images is not previous_images
    assert window.load_progress_bar.value() == 100
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.analysis_options_tab))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))


def test_cancel_after_loading_worker_finishes_discards_result(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    previous_images = window.images
    window.load_images_button.click()
    thread = window._loading_thread
    assert thread.wait(5000)
    window.cancel_load_button.click()
    wait_until(lambda: window._loading_thread is None)
    assert window.images is previous_images
    assert window.load_status_label.text() == "Image loading cancelled."
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.analysis_options_tab))


def test_cancel_after_decoding_skips_remaining_image_preparation(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    started = threading.Event()
    release = threading.Event()
    original_decode = image_state.load_image
    lab_calls = []

    def paused_decode(path):
        image = original_decode(path)
        started.set()
        if not release.wait(5):
            raise RuntimeError("Timed out waiting for test")
        return image

    monkeypatch.setattr(image_state, "load_image", paused_decode)
    monkeypatch.setattr(image_data_colour, "generate_lab_pixels", lambda *args, **kwargs: lab_calls.append(True))
    window.load_images_button.click()
    try:
        wait_until(started.is_set)
        window.cancel_load_button.click()
    finally:
        release.set()
        wait_until(lambda: window._loading_thread is None)
    assert not lab_calls
    assert window.load_status_label.text() == "Image loading cancelled."


def test_cancel_loading_does_not_hide_worker_error(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    started = threading.Event()
    release = threading.Event()
    errors = []

    def failing_load(image, path, check_cancelled=None):
        started.set()
        if not release.wait(5):
            raise RuntimeError("Timed out waiting for test")
        raise OSError("decoding failed")

    monkeypatch.setattr(image_data_colour, "load_from_path", failing_load)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))
    window.load_images_button.click()
    try:
        wait_until(started.is_set)
        window.cancel_load_button.click()
    finally:
        release.set()
        wait_until(lambda: window._loading_thread is None)
    assert "decoding failed" in errors[0][2]
    assert window.load_status_label.text().startswith("Loading failed.")


def test_close_waits_for_image_loading(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.show()
    started = threading.Event()
    release = threading.Event()
    original_load = image_data_colour.load_from_path

    def paused_load(image, path, check_cancelled=None):
        if image.key == "B1":
            started.set()
            if not release.wait(5):
                raise RuntimeError("Timed out waiting for test")
        return original_load(image, path, check_cancelled=check_cancelled)

    monkeypatch.setattr(image_data_colour, "load_from_path", paused_load)
    window.load_images_button.click()
    try:
        wait_until(started.is_set)
        window.close()
        assert window.isVisible()
        assert "close" in window.load_status_label.text()
    finally:
        release.set()
        wait_until(lambda: window._loading_thread is None)
    assert not window.isVisible()


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


def test_analysis_button_and_result_tabs(window):
    assert window.tab_widget.tabText(window.tab_widget.indexOf(window.analysis_options_tab)) == "4) Analysis"
    assert window.tab_widget.tabText(window.tab_widget.indexOf(window.results_tab)) == "5) Results data"
    assert window.tab_widget.tabText(window.tab_widget.indexOf(window.image_results_tab)) == "6) Image results"
    assert window.analyze_button.parentWidget() is window.analysis_options_tab
    assert not window.results_tab.findChildren(QPushButton)
    layout = window.analysis_options_tab.layout()
    assert layout.indexOf(window.analysis_progress_label) > layout.indexOf(window.analysis_progress_bar)
    assert window.analysis_progress_bar.isTextVisible()

    window.show()
    tab_bar = window.tab_widget.tabBar()
    QTest.mouseClick(tab_bar, Qt.LeftButton, pos=tab_bar.tabRect(window.tab_widget.indexOf(window.analysis_options_tab)).center())
    assert window.tab_widget.currentWidget() is window.load_tab


def test_disabled_tabs_keep_native_outline_and_show_disabled_text(window, monkeypatch, image_paths):
    window.show()
    tab_bar = window.tab_widget.tabBar()

    def tab_image(tab):
        QApplication.processEvents()
        rect = tab_bar.tabRect(window.tab_widget.indexOf(tab))
        return tab_bar.grab().toImage().copy(rect)

    disabled_mask = tab_image(window.mask_tab)
    disabled_results = tab_image(window.results_tab)
    stylesheet = window.tab_widget.styleSheet()
    window.tab_widget.setStyleSheet("")
    native_mask = tab_image(window.mask_tab)
    window.tab_widget.setStyleSheet(stylesheet)
    assert disabled_mask.size() == native_mask.size()
    assert disabled_mask.copy(0, 0, disabled_mask.width(), 3) == native_mask.copy(0, 0, native_mask.width(), 3)

    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    assert tab_image(window.mask_tab) != disabled_mask
    assert tab_image(window.results_tab) == disabled_results

    window.tab_widget.setCurrentWidget(window.analysis_options_tab)
    window.analyze_button.click()
    wait_until(lambda: window._analysis_thread is None)
    assert tab_image(window.results_tab) != disabled_results

    window.tab_widget.setCurrentWidget(window.load_tab)
    select_image(window, monkeypatch, "B1", image_paths["B1"])
    assert tab_image(window.mask_tab) == disabled_mask
    assert tab_image(window.results_tab) == disabled_results


def test_analysis_runs_off_ui_thread_and_keeps_inputs_stable(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    window.tab_widget.setCurrentWidget(window.analysis_options_tab)
    window.images["B1"].set_mask("paper", np.array([[True, True, False, True]] * 3))
    window.images["B2"].set_mask("paper", np.array([[True, False, True, True]] * 3))
    window.images["A"].set_mask("paper", np.array([[True, True, True, False]] * 3))
    window.alignment_checkbox.setChecked(True)
    window.intersection_mask_checkbox.setChecked(True)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_analysis(b1, b2, after, options, *, progress, check_cancelled):
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
        assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
        assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))
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
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))
    assert window.analysis_progress_bar.value() == 100
    assert window.analyze_button.isEnabled()
    assert window.analysis_settings_group.isEnabled()
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.load_tab))


def test_analysis_error_restores_controls_and_allows_retry(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    window.report.setPlainText("previous report")
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))

    def fail_analysis(*args, progress, check_cancelled):
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
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))

    monkeypatch.setattr(main, "image_analysis_b1_b2_a", lambda *args, **kwargs: {"retry": "ok"})
    window.analyze_button.click()
    wait_until(lambda: window._analysis_thread is None)
    assert window.report.toPlainText() == "retry: ok\n"
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))

    window.tab_widget.setCurrentWidget(window.analysis_options_tab)
    monkeypatch.setattr(main, "image_analysis_b1_b2_a", fail_analysis)
    window.analyze_button.click()
    wait_until(lambda: window._analysis_thread is None)
    assert window.report.toPlainText() == "retry: ok\n"
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.image_results_tab))


@pytest.mark.parametrize("previous_analysis", [False, True])
def test_cancel_analysis_stops_worker_and_preserves_results(window, monkeypatch, image_paths, previous_analysis):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    window.show()
    window.tab_widget.setCurrentWidget(window.analysis_options_tab)
    assert window.cancel_analysis_button.isHidden()

    if previous_analysis:
        window.analyze_button.click()
        wait_until(lambda: window._analysis_thread is None)
        window.tab_widget.setCurrentWidget(window.analysis_options_tab)
    previous_report = window.report.toPlainText()
    started = threading.Event()
    release = threading.Event()
    original_avg_sd = workflow.image_avg_sd
    calls = []
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))

    def paused_avg_sd(image, mask, check_cancelled=None):
        calls.append(True)
        started.set()
        if not release.wait(5):
            raise RuntimeError("Timed out waiting for test")
        return original_avg_sd(image, mask, check_cancelled)

    monkeypatch.setattr(workflow, "image_avg_sd", paused_avg_sd)
    window.analyze_button.click()
    try:
        wait_until(started.is_set)
        assert window.cancel_analysis_button.isVisible()
        assert window.cancel_analysis_button.geometry().left() > window.analyze_button.geometry().right()
        window.cancel_analysis_button.click()
        assert window.analysis_progress_label.text() == "Cancelling analysis..."
        assert not window.cancel_analysis_button.isEnabled()
        assert not window.analyze_button.isEnabled()
        assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.load_tab))
    finally:
        release.set()
        wait_until(lambda: window._analysis_thread is None)

    assert len(calls) == 1
    assert not errors
    assert window.report.toPlainText() == previous_report
    assert window.analysis_progress_label.text() == "Analysis cancelled."
    assert window.cancel_analysis_button.isHidden()
    assert window.analyze_button.isEnabled()
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.load_tab))
    for tab in (window.results_tab, window.image_results_tab):
        assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(tab)) == previous_analysis

    monkeypatch.setattr(workflow, "image_avg_sd", original_avg_sd)
    window.analyze_button.click()
    wait_until(lambda: window._analysis_thread is None)
    assert "Overall change in RGB colour space" in window.report.toPlainText()
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))


def test_cancel_after_worker_finishes_discards_pending_result(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    started = threading.Event()
    release = threading.Event()

    def paused_analysis(*args, **kwargs):
        started.set()
        if not release.wait(5):
            raise RuntimeError("Timed out waiting for test")
        return {"Result": "should not be shown"}

    monkeypatch.setattr(main, "image_analysis_b1_b2_a", paused_analysis)
    window.analyze_button.click()
    try:
        wait_until(started.is_set)
        thread = window._analysis_thread
        release.set()
        assert thread.wait(5000)
        window.cancel_analysis_button.click()
        wait_until(lambda: window._analysis_thread is None)
    finally:
        release.set()
        if window._analysis_thread is not None:
            wait_until(lambda: window._analysis_thread is None)

    assert window.analysis_progress_label.text() == "Analysis cancelled."
    assert window.report.toPlainText() == ""
    assert not window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))


def test_cancel_during_intersection_mask_calculation(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    window.tab_widget.setCurrentWidget(window.analysis_options_tab)
    window.alignment_checkbox.setChecked(True)
    window.intersection_mask_checkbox.setChecked(True)
    started = threading.Event()
    release = threading.Event()
    original = main._as_bool_mask
    calls = []

    def paused_mask(mask):
        calls.append(True)
        if len(calls) == 1:
            started.set()
            if not release.wait(5):
                raise RuntimeError("Timed out waiting for test")
        return original(mask)

    monkeypatch.setattr(main, "_as_bool_mask", paused_mask)
    window.analyze_button.click()
    try:
        wait_until(started.is_set)
        window.cancel_analysis_button.click()
    finally:
        release.set()
        wait_until(lambda: window._analysis_thread is None)

    assert len(calls) == 2
    assert window.analysis_progress_label.text() == "Analysis cancelled."
    assert window.report.toPlainText() == ""


def test_cancel_does_not_hide_unrelated_worker_error(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
    started = threading.Event()
    release = threading.Event()
    errors = []

    def failing_analysis(*args, **kwargs):
        started.set()
        if not release.wait(5):
            raise RuntimeError("Timed out waiting for test")
        raise ValueError("calculation failed")

    monkeypatch.setattr(main, "image_analysis_b1_b2_a", failing_analysis)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))
    window.analyze_button.click()
    try:
        wait_until(started.is_set)
        window.cancel_analysis_button.click()
    finally:
        release.set()
        wait_until(lambda: window._analysis_thread is None)

    assert "calculation failed" in errors[0][2]
    assert window.analysis_progress_label.text().startswith("Analysis failed.")


def test_close_waits_for_running_analysis(window, monkeypatch, image_paths):
    for key, path in image_paths.items():
        select_image(window, monkeypatch, key, path)
    window.load_images_button.click()
    wait_until(lambda: window._loading_thread is None)
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
    wait_until(lambda: window._loading_thread is None)
    window.tab_widget.setCurrentWidget(window.analysis_options_tab)
    window.alignment_checkbox.setChecked(True)
    window.intersection_mask_checkbox.setChecked(True)

    permits = threading.Semaphore(0)
    intersection = main._intersection_masks
    original_analysis = main.image_analysis_b1_b2_a

    def paused_intersection(images, check_cancelled=None):
        if not permits.acquire(timeout=5):
            raise RuntimeError("Timed out waiting for test")
        return intersection(images, check_cancelled=check_cancelled)

    def paused_analysis(b1, b2, after, options, *, progress, check_cancelled):
        def pause_at_stage(description):
            progress(description)
            if not permits.acquire(timeout=5):
                raise RuntimeError("Timed out waiting for test")
        return original_analysis(b1, b2, after, options, progress=pause_at_stage, check_cancelled=check_cancelled)

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
    wait_until(lambda: window._loading_thread is None)
    window.rgb_comparison_checkbox.setChecked(rgb)
    window.lab_comparison_checkbox.setChecked(lab)
    window.analyze_button.click()
    wait_until(lambda: window._analysis_thread is None)
    assert window.analysis_progress_label.text() == f"Analysis complete ({total}/{total} tasks)."
    assert window.analysis_progress_bar.value() == 100
