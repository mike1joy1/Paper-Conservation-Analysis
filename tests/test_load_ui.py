import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

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
    assert window.alignment_checkbox.isEnabled()

    monkeypatch.setattr(QMessageBox, "critical", lambda *args: pytest.fail(args[2]))
    window.analyze_images()
    assert "Overall change in RGB colour space" in window.report.toPlainText()

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
    assert window.load_images_button.isEnabled()

    monkeypatch.setattr(image_data_colour, "load_from_path", original_load)
    window.load_images_button.click()
    assert window.images is not previous_images
    assert window.tab_widget.isTabEnabled(window.tab_widget.indexOf(window.results_tab))


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
