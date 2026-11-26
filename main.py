"""Desktop app for paper conservation analysis.

The workflow is intentionally simple: the images are assumed to have been aligned
and cropped before they reach this application, and the app focuses on the
scientific comparison and report generation.
"""

import os
import sys
import traceback
from copy import copy
from pathlib import Path
from threading import Event

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

# Add the source tree to the import path so the app can import the package.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtCore import QEvent, QObject, QRectF, Qt, QSize, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QColorDialog,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QStyle,
    QStyleOptionTab,
    QTabBar,
    QTabWidget,
    QToolButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)

from paperdiff.image_state import analysis_options, image_data_colour, mask_to_colour_image
from paperdiff.loading import load_image
from paperdiff.stats import verdict_for_region
from paperdiff.workflow import describe_region_change, image_analysis_b1_b2_a


CANCEL_BUTTON_STYLE = (
    "QPushButton { background-color: #b3261e; color: white; }"
    "QPushButton:hover:enabled { background-color: #c83d34; }"
)


def _as_bool_mask(mask):
    """Convert 2D or 3-channel mask arrays to a 2D bool mask."""
    if mask is None:
        return None
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(bool)


def _intersection_masks(images, check_cancelled=None):
    """Intersect the paper and ink masks across the supplied images."""
    mask_papers = []
    mask_inks = []
    for key in images:
        if check_cancelled is not None:
            check_cancelled()
        image_data = images[key]
        if image_data.pixels.get("rgb") is None:
            return None, None

        paper_mask = _as_bool_mask(image_data.masks.get("paper"))
        ink_mask = _as_bool_mask(image_data.masks.get("ink"))
        h, w = image_data.get_pixels("rgb").shape[:2]
        if paper_mask is None:
            paper_mask = np.ones((h, w), dtype=bool)
        if ink_mask is None:
            ink_mask = np.ones((h, w), dtype=bool)
        mask_papers.append(paper_mask)
        mask_inks.append(ink_mask)

    if check_cancelled is not None:
        check_cancelled()
    paper_intersection = np.logical_and.reduce(mask_papers)
    if check_cancelled is not None:
        check_cancelled()
    ink_intersection = np.logical_and.reduce(mask_inks)
    paper_intersection = np.repeat(paper_intersection[..., np.newaxis], 3, axis=-1)
    if check_cancelled is not None:
        check_cancelled()
    ink_intersection = np.repeat(ink_intersection[..., np.newaxis], 3, axis=-1)
    return paper_intersection, ink_intersection


class AnalysisCancelled(Exception):
    """Signal a requested cancellation from an analysis checkpoint."""


class ImageLoadingCancelled(Exception):
    """Signal a requested cancellation from an image-loading checkpoint."""


class ImageLoadingThread(QThread):
    """Load a complete image set without accessing UI widgets."""

    progress = Signal(int, int, str, str)

    def __init__(self, paths, reference_colours, parent=None):
        """Snapshot paths and initialise loading results and cancellation state."""
        super().__init__(parent)
        self.paths = paths.copy()
        self.reference_colours = reference_colours
        self.cancel_requested = Event()
        self.images = {}
        self.previews = {}
        self.error = None
        self.current_key = ""
        self.current_task = 0
        self.total_tasks = len(self.paths) * 5

    def _start_task(self, description):
        """Report the next loading step for the current image."""
        self._check_cancelled()
        self.current_task += 1
        self.progress.emit(self.current_task, self.total_tasks, self.current_key, description)

    def _check_cancelled(self):
        """Stop the loading worker at a cooperative cancellation checkpoint."""
        if self.cancel_requested.is_set():
            raise ImageLoadingCancelled()

    def run(self):
        """Prepare each image and preview, recording failures for the UI thread."""
        try:
            for key, path in self.paths.items():
                self._check_cancelled()
                self.current_key = key
                self._start_task("Loading image")
                image_data = image_data_colour(key)
                image_data.reference_colours = self.reference_colours[key]
                image_data.load_from_path(path, check_cancelled=self._check_cancelled, progress=self._start_task)
                self._check_cancelled()
                self._start_task("Preparing preview")
                rgb = np.ascontiguousarray((np.clip(image_data.get_pixels("rgb"), 0.0, 1.0) * 255.0).astype(np.uint8))
                self._check_cancelled()
                h, w, channels = rgb.shape
                self.previews[key] = QImage(rgb.data, w, h, w * channels, QImage.Format.Format_RGB888).copy()
                self.images[key] = image_data
            self._check_cancelled()
        except ImageLoadingCancelled:
            self.images.clear()
            self.previews.clear()
        except Exception:
            self.error = traceback.format_exc()


class AnalysisThread(QThread):
    """Run the analysis without accessing UI widgets."""

    progress = Signal(int, int, str)

    def __init__(self, images, options, parent=None, mode="two_before"):
        """Snapshot analysis options and initialise worker progress and results."""
        super().__init__(parent)
        self.images = images
        self.mode = mode
        self.options = copy(options)
        self.result = None
        self.error = None
        self.cancel_requested = Event()
        self.use_intersection = mode != "single" and self.options.use_masks and self.options.intersection_mask
        colour_tasks = 1 + (len(images) * (1 + int(self.options.use_masks)) if self.options.mean_comparison else 0)
        self.total_tasks = (
            2 + int(self.options.show_image_data) + int(self.use_intersection)
            + colour_tasks * (int(self.options.rgb_comparison) + int(self.options.lab_comparison))
        )
        self.current_task = 0

    def _start_task(self, description):
        """Advance the task counter and report the current analysis stage."""
        self._check_cancelled()
        self.current_task += 1
        self.progress.emit(self.current_task, self.total_tasks, description)

    def _check_cancelled(self):
        """Stop the analysis worker at a cooperative cancellation checkpoint."""
        if self.cancel_requested.is_set():
            raise AnalysisCancelled()

    def run(self):
        """Compute optional combined masks and run the analysis."""
        try:
            self._check_cancelled()
            if self.use_intersection:
                self._start_task("Calculating combined masks")
                self.options.int_paper_mask, self.options.int_ink_mask = _intersection_masks(
                    self.images, check_cancelled=self._check_cancelled
                )
            self.result = image_analysis_b1_b2_a(
                self.images["B1"], self.images.get("B2"), self.images.get("A"), self.options,
                progress=self._start_task, check_cancelled=self._check_cancelled, mode=self.mode
            )
            self._check_cancelled()
        except AnalysisCancelled:
            self.result = None
        except Exception:
            self.error = traceback.format_exc()


class MaskBatchThread(QThread):
    """Generate or import masks for several images without accessing UI widgets."""

    item_done = Signal(int, int, str, str, str)

    def __init__(self, action, keys, images, paths=None, settings=None, parent=None):
        """Snapshot the images and settings used for the batch."""
        super().__init__(parent)
        self.action = action
        self.keys = list(keys)
        self.images = images
        self.paths = dict(paths or {})
        self.settings = dict(settings or {})
        self.cancel_requested = Event()

    def run(self):
        """Process each image in turn, reporting a status of ok, missing or error."""
        total = len(self.keys)
        for index, key in enumerate(self.keys, start=1):
            if self.cancel_requested.is_set():
                break
            status, detail = "ok", ""
            try:
                if self.action == "generate":
                    paper_mask, ink_mask, _border = self.images[key].generate_masks(**self.settings)
                    if paper_mask is None or ink_mask is None:
                        status = "skipped"
                else:
                    path = self.paths.get(key)
                    if not path:
                        status, detail = "error", "no export path is set"
                    else:
                        self.images[key].import_mask(path)
            except FileNotFoundError:
                status = "missing"
            except Exception as exc:
                status, detail = "error", str(exc)
            self.item_done.emit(index, total, key, status, detail)


class ZoomableLabel(QLabel):
    """A QLabel that keeps a base pixmap and allows zooming via the mouse wheel."""

    clicked = Signal()

    def __init__(self, text=""):
        """Initialise the preview image, viewport, zoom, and pan state."""
        super().__init__(text)
        self._base_pixmap = QPixmap()
        self._zoom_factor = 1.0
        self._auto_fit_zoom = True
        self._fixed_viewport = QSize(320, 240)
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.setMinimumSize(120, 80)

    def sizeHint(self):
        """Suggest the configured viewport size to the layout."""
        return self._fixed_viewport if self._fixed_viewport.isValid() else QSize(320, 240)

    def minimumSizeHint(self):
        """Return the smallest suggested preview size."""
        return QSize(120, 80)

    def _clamp_pan(self, viewport: QSize, scaled_width: int, scaled_height: int, pan_x: float | None = None, pan_y: float | None = None):
        """Keep the scaled image centred or within the viewport edges."""
        if pan_x is None:
            pan_x = self._pan_x
        if pan_y is None:
            pan_y = self._pan_y

        if scaled_width <= viewport.width():
            pan_x = (viewport.width() - scaled_width) / 2.0
        else:
            pan_x = min(0.0, max(viewport.width() - scaled_width, pan_x))

        if scaled_height <= viewport.height():
            pan_y = (viewport.height() - scaled_height) / 2.0
        else:
            pan_y = min(0.0, max(viewport.height() - scaled_height, pan_y))

        return pan_x, pan_y

    def resizeEvent(self, event):
        """Update the viewport and redraw the image after resizing."""
        super().resizeEvent(event)
        self._fixed_viewport = self.size()
        if self._base_pixmap is not None and not self._base_pixmap.isNull():
            if self._auto_fit_zoom:
                self._apply_fit_zoom()
            self._refresh_zoomed_pixmap()

    def setPixmap(self, pixmap, preserve_zoom: bool = False):
        """Set the source image, optionally retaining its current zoom and pan."""
        self._base_pixmap = pixmap if pixmap is not None else QPixmap()
        if preserve_zoom:
            self._auto_fit_zoom = False
            self._refresh_zoomed_pixmap()
            return

        self._auto_fit_zoom = True
        self._apply_fit_zoom()
        self._refresh_zoomed_pixmap()

    def setFixedViewport(self, width: int, height: int):
        """Set the viewport dimensions and fit the image to them."""
        self._fixed_viewport = QSize(width, height)
        self._apply_fit_zoom()
        self._refresh_zoomed_pixmap()

    def _fit_zoom_for_viewport(self, viewport: QSize | None = None) -> float:
        """Return the maximum zoom-out value that keeps the entire image inside the current viewport."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return 1.0

        if viewport is None:
            viewport = self._fixed_viewport if self._fixed_viewport.width() > 0 and self._fixed_viewport.height() > 0 else self.size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return 1.0

        source_size = self._base_pixmap.size()
        if source_size.width() <= 0 or source_size.height() <= 0:
            return 1.0

        fit_factor = min(viewport.width() / source_size.width(), viewport.height() / source_size.height())
        return max(0.05, min(12.0, fit_factor))

    def _apply_fit_zoom(self):
        """Default to the maximum zoom-out state so the whole image stays visible."""
        fit_zoom = self._fit_zoom_for_viewport()
        self._zoom_factor = max(0.05, min(12.0, fit_zoom))
        self._pan_x = 0.0
        self._pan_y = 0.0

    def _render_centered_pixmap(self, source_pixmap: QPixmap, viewport: QSize) -> QPixmap:
        """Paint the image inside the viewport and keep the current pan/zoom anchored."""
        canvas = QPixmap(viewport)
        canvas.fill(self.palette().color(self.backgroundRole()))

        if source_pixmap.isNull() or viewport.width() <= 0 or viewport.height() <= 0:
            return canvas

        scaled = source_pixmap.scaled(
            QSize(max(1, int(source_pixmap.width() * self._zoom_factor)), max(1, int(source_pixmap.height() * self._zoom_factor))),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        if scaled.width() <= 0 or scaled.height() <= 0:
            return canvas

        self._pan_x, self._pan_y = self._clamp_pan(viewport, scaled.width(), scaled.height())
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(int(round(self._pan_x)), int(round(self._pan_y)), scaled)
        painter.end()
        return canvas

    def _refresh_zoomed_pixmap(self):
        """Render the source pixmap at the current zoom and pan."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            super().setPixmap(QPixmap())
            return

        viewport = self._fixed_viewport if self._fixed_viewport.width() > 0 and self._fixed_viewport.height() > 0 else self.size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            viewport = self._base_pixmap.size()

        minimum_zoom = self._fit_zoom_for_viewport(viewport)
        self._zoom_factor = max(minimum_zoom, min(12.0, self._zoom_factor))
        self._pan_x, self._pan_y = self._clamp_pan(viewport, max(1, int(self._base_pixmap.width() * self._zoom_factor)), max(1, int(self._base_pixmap.height() * self._zoom_factor)))
        super().setPixmap(self._render_centered_pixmap(self._base_pixmap, viewport))

    def wheelEvent(self, event):
        """Zoom around the mouse cursor while keeping the image in bounds."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return

        self._auto_fit_zoom = False
        viewport = self.size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return

        old_zoom = self._zoom_factor
        delta = event.angleDelta().y()
        fit_zoom = self._fit_zoom_for_viewport()
        if delta > 0:
            new_zoom = min(12.0, max(fit_zoom, old_zoom * 1.15))
        elif delta < 0:
            new_zoom = max(fit_zoom, old_zoom * 0.85)
        else:
            return

        old_scaled_width = max(1, int(self._base_pixmap.width() * old_zoom))
        old_scaled_height = max(1, int(self._base_pixmap.height() * old_zoom))

        cursor_x = event.position().x()
        cursor_y = event.position().y()

        anchor_x = cursor_x - self._pan_x
        anchor_y = cursor_y - self._pan_y
        anchor_u = max(0.0, min(1.0, anchor_x / old_scaled_width))
        anchor_v = max(0.0, min(1.0, anchor_y / old_scaled_height))

        new_scaled_width = max(1, int(self._base_pixmap.width() * new_zoom))
        new_scaled_height = max(1, int(self._base_pixmap.height() * new_zoom))
        new_pan_x = cursor_x - (anchor_u * new_scaled_width)
        new_pan_y = cursor_y - (anchor_v * new_scaled_height)

        self._zoom_factor = new_zoom
        self._pan_x, self._pan_y = self._clamp_pan(viewport, new_scaled_width, new_scaled_height, new_pan_x, new_pan_y)
        self._refresh_zoomed_pixmap()
        event.accept()

    def mousePressEvent(self, event):
        """Emit a click before passing the mouse event to the label."""
        self.clicked.emit()
        super().mousePressEvent(event)

    def clear(self):
        """Clear the source image and reset zoom and pan."""
        self._base_pixmap = QPixmap()
        self._zoom_factor = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        super().clear()


class ClickableLabel(ZoomableLabel):
    """A zoomable label that emits a click signal so the alignment viewer can cycle images."""

    pass


class CardGridArea(QScrollArea):
    """Vertically scrolling area that lays fixed-size cards out in as many columns as fit."""

    SPACING = 10
    MARGIN = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards = []
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.NoFrame)
        # Transparent via autofill rather than stylesheets, which would restyle the cards' native controls.
        self.viewport().setAutoFillBackground(False)
        self._container = QWidget()
        self._container.setAutoFillBackground(False)
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(self.SPACING)
        self._grid.setContentsMargins(self.MARGIN, self.MARGIN, self.MARGIN, self.MARGIN)
        self._grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setWidget(self._container)
        self._container.setAutoFillBackground(False)

    def add_card(self, card):
        """Append a card and re-flow the grid."""
        self._cards.append(card)
        card.setParent(self._container)
        card.show()
        self._reflow()

    def remove_card(self, card):
        """Remove a card and re-flow the grid."""
        if card in self._cards:
            self._cards.remove(card)
            self._grid.removeWidget(card)
            card.deleteLater()
            self._reflow()

    def cards(self):
        """Return the cards in grid order."""
        return list(self._cards)

    def move_card(self, card, step):
        """Move a card one place earlier (-1) or later (+1); return whether it moved."""
        index = self._cards.index(card)
        target = index + step
        if not 0 <= target < len(self._cards):
            return False
        self._cards[index], self._cards[target] = self._cards[target], self._cards[index]
        self._reflow()
        return True

    def _reflow(self):
        if not self._cards:
            return
        card_width = self._cards[0].width()
        available = self.viewport().width() - 2 * self.MARGIN
        columns = max(1, (available + self.SPACING) // (card_width + self.SPACING))
        for card in self._cards:
            self._grid.removeWidget(card)
        for index, card in enumerate(self._cards):
            self._grid.addWidget(card, index // columns, index % columns)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow()


class ImageCardWidget(QFrame):
    """A reusable image card widget with before/after selection checkboxes and removal button.
    
    For load tab: displays an image with before/after classification.
    For mask tab: displays masks and generation/import controls.
    """
    
    removed = Signal(str)  # emits the image_id when remove button is clicked
    before_toggled = Signal(str, bool)  # emits (image_id, is_before)
    after_toggled = Signal(str, bool)  # emits (image_id, is_after)
    load_requested = Signal(str)  # emits the image_id when its load button is clicked
    move_requested = Signal(str, int)  # emits (image_id, -1 for left / +1 for right)
    
    PREVIEW_WIDTH = 234
    PREVIEW_HEIGHT = 176  # 4:3

    def __init__(self, image_id: str, pixmap=None, parent=None, filename: str = "", details: str = ""):
        """Initialize the card with an ID and optional preview pixmap.
        
        Args:
            image_id: Unique identifier for this image
            pixmap: QPixmap to display (optional for first setup)
            parent: Parent widget
        """
        super().__init__(parent)
        self.image_id = image_id
        self.setObjectName("imageCard")
        self._loaded = False
        # Painted manually (no stylesheet) so child checkboxes and buttons keep the native app styling.
        self.setFixedSize(250, 350)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        header = QHBoxLayout()
        self.filename_label = QLabel(filename)
        self.filename_label.setStyleSheet("font-weight: bold; font-size: 13px; border: none; background: transparent;")
        self.filename_label.setToolTip(filename)
        self.filename_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.filename_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        header.addWidget(self.filename_label, 1)
        self.remove_button = QPushButton("✕")
        self.remove_button.setFlat(True)
        self.remove_button.setFixedSize(20, 20)
        self.remove_button.setCursor(Qt.PointingHandCursor)
        self.remove_button.setStyleSheet(
            "QPushButton { border: none; background: transparent; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { color: #c0392b; }"
        )
        self.remove_button.setToolTip("Remove this image")
        self.remove_button.clicked.connect(self._on_remove_clicked)
        header.addWidget(self.remove_button)
        layout.addLayout(header)
        
        # Image preview (4:3, image fully contained)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT)
        self.preview_label.setStyleSheet("border: 1px solid #888; background: #262626;")
        self.set_pixmap(pixmap)
        layout.addWidget(self.preview_label)

        path_text, _, info_text = details.partition("\n")
        details_box = QVBoxLayout()
        details_box.setSpacing(4)
        details_box.setContentsMargins(0, 0, 0, 0)
        self.details_label = QLabel(path_text)
        self.info_label = QLabel(info_text)
        for label in (self.details_label, self.info_label):
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            label.setStyleSheet("font-size: 11px; border: none; background: transparent;")
            label.setToolTip(details)
            details_box.addWidget(label)
        details_box.addStretch(1)
        layout.addLayout(details_box, 1)
        
        # Before/After checkboxes and per-card load button
        checkbox_layout = QHBoxLayout()
        self.before_checkbox = QCheckBox("Before")
        self.before_checkbox.setToolTip("Mark this as a before image")
        self.before_checkbox.toggled.connect(lambda checked: self.before_toggled.emit(self.image_id, checked))
        checkbox_layout.addWidget(self.before_checkbox)
        
        self.after_checkbox = QCheckBox("After")
        self.after_checkbox.setToolTip("Mark this as an after image")
        self.after_checkbox.toggled.connect(lambda checked: self.after_toggled.emit(self.image_id, checked))
        checkbox_layout.addWidget(self.after_checkbox)

        checkbox_layout.addStretch(1)
        self.load_button = QPushButton("Load")
        self.load_button.setToolTip("Load this image for analysis (tick Before or After first).")
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(lambda: self.load_requested.emit(self.image_id))
        checkbox_layout.addWidget(self.load_button)
        layout.addLayout(checkbox_layout)

        arrow_layout = QHBoxLayout()
        self.left_button = QPushButton("◀")
        self.left_button.setToolTip("Move this image earlier in the order")
        self.left_button.clicked.connect(lambda: self.move_requested.emit(self.image_id, -1))
        self.right_button = QPushButton("▶")
        self.right_button.setToolTip("Move this image later in the order")
        self.right_button.clicked.connect(lambda: self.move_requested.emit(self.image_id, 1))
        for button in (self.left_button, self.right_button):
            button.setFixedWidth(40)
        arrow_layout.addWidget(self.left_button)
        arrow_layout.addStretch(1)
        arrow_layout.addWidget(self.right_button)
        layout.addLayout(arrow_layout)
        # Ticking one box unticks the other; both may be unticked.
        self.before_checkbox.toggled.connect(lambda checked: checked and self.after_checkbox.setChecked(False))
        self.after_checkbox.toggled.connect(lambda checked: checked and self.before_checkbox.setChecked(False))
        self.before_checkbox.toggled.connect(lambda _checked: self.update())
        self.after_checkbox.toggled.connect(lambda _checked: self.update())

    def set_loaded(self, loaded: bool):
        """Show a green border while the image is loaded for analysis."""
        self._loaded = loaded
        self.update()

    def paintEvent(self, event):
        """Draw the card background and border."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = self.palette()

        def blend(amount):
            """Mix the window colour towards the text colour so card colours follow light/dark themes."""
            base, text = palette.window().color(), palette.windowText().color()
            return QColor(
                *(int(b + (t - b) * amount) for b, t in zip(base.getRgb()[:3], text.getRgb()[:3]))
            )

        selected = self.before_checkbox.isChecked() or self.after_checkbox.isChecked()
        width = 2 if self._loaded else 1
        if self._loaded:
            colour = QColor("#2e9e44")
        elif selected:
            colour = blend(0.6)
        else:
            colour = blend(0.25)
        pen = QPen(colour)
        pen.setWidth(width)
        painter.setPen(pen)
        painter.setBrush(palette.window().color() if selected or self._loaded else blend(0.07))
        half = width / 2
        painter.drawRoundedRect(QRectF(half, half, self.width() - width, self.height() - width), 6, 6)
    
    def set_pixmap(self, pixmap):
        """Update the displayed image, scaled to fit entirely inside the 4:3 box."""
        if pixmap and not pixmap.isNull():
            inner = QSize(self.PREVIEW_WIDTH - 2, self.PREVIEW_HEIGHT - 2)
            self.preview_label.setPixmap(pixmap.scaled(inner, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def set_before(self, checked: bool, block_signals=False):
        """Set the before checkbox state."""
        with self._block_signals_context(self.before_checkbox) if block_signals else self._no_op_context():
            self.before_checkbox.setChecked(checked)
    
    def set_after(self, checked: bool, block_signals=False):
        """Set the after checkbox state."""
        with self._block_signals_context(self.after_checkbox) if block_signals else self._no_op_context():
            self.after_checkbox.setChecked(checked)
    
    def is_before(self) -> bool:
        """Check if marked as before."""
        return self.before_checkbox.isChecked()
    
    def is_after(self) -> bool:
        """Check if marked as after."""
        return self.after_checkbox.isChecked()
    
    def _on_remove_clicked(self):
        """Emit removal signal."""
        self.removed.emit(self.image_id)
    
    @staticmethod
    def _block_signals_context(widget):
        """Context manager to temporarily block signals."""
        class SignalBlocker:
            def __enter__(self):
                widget.blockSignals(True)
                return self
            def __exit__(self, *args):
                widget.blockSignals(False)
        return SignalBlocker()
    
    @staticmethod
    def _no_op_context():
        """No-op context manager."""
        class NoOp:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return NoOp()


class AnalysisTabBar(QTabBar):
    def initStyleOption(self, option: QStyleOptionTab, tab_index: int):
        """Suppress hover styling for disabled tabs."""
        super().initStyleOption(option, tab_index)
        if not self.isTabEnabled(tab_index):
            option.state &= ~QStyle.StateFlag.State_MouseOver


class MainWindow(QMainWindow):
    """The main window containing the analysis workflow."""

    def __init__(self):
        """Build the workflow tabs and initialise application state."""
        super().__init__()
        self.setWindowTitle("Paper Conservation Analysis")
        self.setMinimumSize(0, 0)
        self.resize(1200, 800)

        # The scientific workflow assumes the images are already aligned and cropped
        # before they are opened in this app.
        self.images = {key: image_data_colour(key) for key in ["B1", "B2", "A"]}
        self.selected_image_paths = {key: "" for key in ["B1", "B2", "A"]}
        self.image_mode = "two_before"
        self.mask_reference_colours = {key: self.images[key].reference_colours for key in ["B1", "B2", "A"]}
        self.analysis_options = analysis_options()
        self.current_alignment_view = "B1"
        self.alignment_display_mode = "original"
        self.mask_overlay_colour = QColor("blue")
        self.mask_mode = "luminance"
        self._loading_thread = None
        self._close_after_loading = False
        self._mask_batch = None
        self._mask_thread = None
        self._close_after_mask_batch = False
        self._analysis_thread = None
        self._close_after_analysis = False
        self._alignment_tab_visited = False
        
        # New dynamic image selection model
        self.loaded_images = {}  # {image_id: {"path": str, "pixmap": QPixmap, "is_before": bool, "is_after": bool}}
        self.load_image_cards = {}  # {image_id: ImageCardWidget}
        self.slot_cards = {}  # {"B1"/"B2"/"A": image_id} for images currently loaded
        self._pending_plan = ("one_each", {})
        self._pending_new = {}
        self.mask_image_cards = {}  # {image_id: MaskImageCardWidget}
        self.next_image_id = 0  # Counter for generating unique IDs
        
        # Keep image_key_labels for backward compatibility with mask tab and other UI elements
        self.image_key_labels = {
            "B1": "Before Image 1",
            "B2": "Before Image 2",
            "A": "After Image",
        }

        self.tab_widget = QTabWidget()
        original_tab_bar = self.tab_widget.tabBar()
        tab_bar = AnalysisTabBar()
        tab_bar.setObjectName(original_tab_bar.objectName())
        tab_bar.setDrawBase(original_tab_bar.drawBase())
        self.tab_widget.setTabBar(tab_bar)
        self.tab_widget.setStyleSheet(
            "QTabBar::tab:disabled { color: #808080; }"
        )
        self.setCentralWidget(self.tab_widget)

        self.load_tab = self._create_load_tab()
        self.mask_tab = self._create_mask_tab()
        self.check_alignment_tab = self._create_check_alignment_tab()
        self.analysis_options_tab = self._create_analysis_options_tab()
        self.image_results_tab = self._create_image_results_tab()

        self.tab_widget.addTab(self.load_tab, "1) Load images")
        self.tab_widget.addTab(self.mask_tab, "2) Mask")
        self.tab_widget.addTab(self.check_alignment_tab, "3) Check alignment")
        self.tab_widget.addTab(self.analysis_options_tab, "4) Analysis")
        self.tab_widget.addTab(self.image_results_tab, "5) Image results")
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        self._base_tab_tooltips = dict((
            (self.load_tab, "Select, preview, and load images."),
            (self.mask_tab, "Generate paper and ink masks."),
            (self.check_alignment_tab, "Inspect image alignment."),
            (self.analysis_options_tab, "Configure and run analysis, and view the data report."),
            (self.image_results_tab, "Image results are reserved for a future feature."),
        ))
        for tab, tooltip in self._base_tab_tooltips.items():
            self.tab_widget.setTabToolTip(self.tab_widget.indexOf(tab), tooltip)

        for index in range(1, self.tab_widget.count()):
            self.tab_widget.setTabEnabled(index, False)

        self._base_control_tooltips = {
            widget: widget.toolTip()
            for tab in self._base_tab_tooltips
            for widget in tab.findChildren(QWidget)
            if widget.toolTip()
        }
        self._refresh_disabled_tooltips()
        self._refresh_preview_box_sizes()

    def _disabled_tab_reason(self, tab):
        """Describe why a workflow tab is unavailable."""
        if self._loading_thread is not None:
            return "image loading is in progress"
        if self._analysis_thread is not None:
            return "analysis is in progress"
        if tab is self.image_results_tab and self.tab_widget.isTabEnabled(
            self.tab_widget.indexOf(self.mask_tab)
        ):
            return "analysis must be completed first"
        return "images must be loaded first"

    def _disabled_control_reason(self, widget):
        """Describe the current prerequisite for a disabled control."""
        if hasattr(self, 'normal_map_radio') and widget is self.normal_map_radio:
            return "normal map analysis is not yet available"
        if hasattr(self, 'cancel_load_button') and widget is self.cancel_load_button:
            return "cancellation is in progress"
        if hasattr(self, 'cancel_analysis_button') and widget is self.cancel_analysis_button:
            return "cancellation is in progress"
        if self._loading_thread is not None:
            return "image loading is in progress"
        if self._analysis_thread is not None:
            return "analysis is in progress"

        # Check for alignment and card widgets (old system compatibility)
        for key in ("B2", "A"):
            if hasattr(self, 'alignment_buttons') and key not in self._active_image_keys():
                if key in self.alignment_buttons and widget is self.alignment_buttons[key]:
                    return ("only one image is selected" if self.image_mode == "single"
                            else "only one before and one after image are selected")
                if key in self.load_image_cards and (widget is self.load_image_cards[key] or self.load_image_cards[key].isAncestorOf(widget)):
                    return ("only one image is selected" if self.image_mode == "single"
                            else "only one before and one after image are selected")
                if key in self.mask_image_cards and (widget is self.mask_image_cards[key] or self.mask_image_cards[key].isAncestorOf(widget)):
                    return ("only one image is selected" if self.image_mode == "single"
                            else "only one before and one after image are selected")

        for tab in self._base_tab_tooltips:
            if tab.isAncestorOf(widget) and not self.tab_widget.isTabEnabled(self.tab_widget.indexOf(tab)):
                return self._disabled_tab_reason(tab)

        if hasattr(self, 'load_images_button') and widget is self.load_images_button:
            if self.image_mode == "single":
                return "select an image first"
            return "select all three images first" if self.image_mode != "one_each" else "select both images first"
        
        if hasattr(self, 'alignment_checkbox') and widget is self.alignment_checkbox:
            if self.image_mode == "single":
                return "comparison requires at least two images"
            return "images must have the same resolution"
        
        if hasattr(self, 'intersection_mask_checkbox') and widget is self.intersection_mask_checkbox:
            if self.image_mode == "single":
                return "comparison requires at least two images"
            if self.image_mode != "single" and not self._images_have_identical_resolution():
                return "images must have the same resolution"
            return "enable Use masks in Analysis first"
        
        if hasattr(self, 'per_pixel_comparison_checkbox') and widget is self.per_pixel_comparison_checkbox:
            if self.image_mode == "single":
                return "comparison requires at least two images"
            return "enable Combined mask or turn off Use masks"
        
        if (self.mask_tab.isAncestorOf(widget) or (hasattr(self, 'use_masks_checkbox') and widget is self.use_masks_checkbox)) and hasattr(self, 'use_mask_checkbox') and not self.use_mask_checkbox.isChecked():
            return "enable Use mask on the Mask tab first"
        
        if hasattr(self, 'mask_threshold_slider') and widget is self.mask_threshold_slider:
            return "select Luminance mask generation"
        
        if hasattr(self, 'reference_weight_slider') and widget in (
            self.reference_weight_slider,
            self.change_all_paper_button if hasattr(self, 'change_all_paper_button') else None,
            self.change_all_ink_button if hasattr(self, 'change_all_ink_button') else None,
        ):
            if widget is not None:
                return "select Reference colour mask generation"
        
        if hasattr(self, 'paper_colour_buttons'):
            if widget in self.paper_colour_buttons.values() or widget in self.ink_colour_buttons.values():
                return "select Reference colour mask generation"
        
        if hasattr(self, 'heatmap_grid_slider_x') and widget in (self.heatmap_grid_slider_x if hasattr(self, 'heatmap_grid_slider_x') else None, 
                                                                    self.heatmap_grid_slider_y if hasattr(self, 'heatmap_grid_slider_y') else None):
            if widget is not None:
                return "enable Per-pixel comparison first"
        
        return "unavailable"

    def _refresh_disabled_tooltips(self):
        """Append or remove disabled-state explanations on controls and tabs."""
        if not hasattr(self, "_base_control_tooltips"):
            return
        for tab, tooltip in self._base_tab_tooltips.items():
            index = self.tab_widget.indexOf(tab)
            if not self.tab_widget.isTabEnabled(index):
                tooltip = f"{tooltip} ({self._disabled_tab_reason(tab)})"
            self.tab_widget.setTabToolTip(index, tooltip)
        for widget, tooltip in self._base_control_tooltips.items():
            if not widget.isEnabled():
                tooltip = f"{tooltip} ({self._disabled_control_reason(widget)})"
            widget.setToolTip(tooltip)

    def resizeEvent(self, event):
        """Resize preview boxes when the main window changes size."""
        super().resizeEvent(event)
        self._refresh_preview_box_sizes()

    def closeEvent(self, event):
        """Defer closing until any loading or analysis worker has finished."""
        if self._loading_thread is not None:
            self._close_after_loading = True
            self.load_status_label.setText("Loading is finishing; the window will close when it is done.")
            event.ignore()
            return
        if self._analysis_thread is not None:
            self._close_after_analysis = True
            self.analysis_progress_label.setText("Analysis is finishing; the window will close when it is done.")
            event.ignore()
            return
        if self._mask_thread is not None:
            self._close_after_mask_batch = True
            self._mask_thread.cancel_requested.set()
            self.mask_progress_label.setText("Mask processing is finishing; the window will close when it is done.")
            event.ignore()
            return
        super().closeEvent(event)

    def _refresh_preview_box_sizes(self):
        """Keep the mask preview cards and the alignment viewer scaled by the current window size, without forcing fixed internal viewport dimensions."""
        available_width = max(240, self.width() - 180)
        thumb_width = max(180, min(360, int(available_width / 3)))
        available_height = max(180, self.height() - 300)
        thumb_height = max(90, min(int(thumb_width * 0.55), available_height))

        if hasattr(self, "alignment_preview"):
            self.alignment_preview.setMinimumSize(200, 120)

        if hasattr(self, "mask_original_labels"):
            for label in self.mask_original_labels.values():
                label.setMinimumSize(80, 60)
                label.setFixedHeight(thumb_height)
                if not label._base_pixmap.isNull():
                    label._apply_fit_zoom()
                    label._refresh_zoomed_pixmap()
            for label in self.mask_preview_labels.values():
                label.setMinimumSize(80, 60)
                label.setFixedHeight(thumb_height)

    def _preview_from_array(self, label: QLabel, image):
        """Convert an image array into a Qt pixmap for display in the app."""
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)

        if array.size == 0:
            label.setText("No image")
            label.setPixmap(QPixmap())
            return

        # Most image loaders in this project provide values in the 0..1 range.
        array = np.clip(array, 0.0, 1.0)
        rgb = (array * 255.0).astype("uint8")
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())

        if isinstance(label, ZoomableLabel):
            label.setPixmap(pixmap, preserve_zoom=True)
            return

        viewport = label.size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            label.setPixmap(pixmap)
            return

        fit_pixmap = QPixmap(viewport)
        fit_pixmap.fill(label.palette().color(label.backgroundRole()))
        scaled = pixmap.scaled(viewport, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        offset_x = max(0, (viewport.width() - scaled.width()) // 2)
        offset_y = max(0, (viewport.height() - scaled.height()) // 2)
        painter = QPainter(fit_pixmap)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(offset_x, offset_y, scaled)
        painter.end()
        label.setPixmap(fit_pixmap)

    def _display_mask_preview(self, label: QLabel, mask_array=None, border_mask=None, paper_mask=None, ink_mask=None):
        """Render paper, ink, overlapping, and excluded mask pixels."""
        if paper_mask is not None and ink_mask is not None:
            rendered = mask_to_colour_image(paper_mask, ink_mask)
            h, w = rendered.shape[:2]
        else:
            if mask_array is None:
                label.setText("No mask")
                label.setPixmap(QPixmap())
                return

            mask = _as_bool_mask(mask_array)
            if border_mask is None:
                border_mask = mask ^ binary_erosion(mask, iterations=1)
            else:
                border_mask = _as_bool_mask(border_mask)

            h, w = mask.shape
            rendered = np.zeros((h, w, 3), dtype=np.uint8)
            rendered[mask] = [255, 255, 255]
            rendered[border_mask] = [255, 0, 0]

        qimage = QImage(rendered.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())

        if isinstance(label, ZoomableLabel):
            label.setPixmap(pixmap, preserve_zoom=True)
            return

        viewport = label.size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            label.setPixmap(pixmap)
            return

        fit_pixmap = QPixmap(viewport)
        fit_pixmap.fill(label.palette().color(label.backgroundRole()))
        scaled = pixmap.scaled(viewport, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        offset_x = max(0, (viewport.width() - scaled.width()) // 2)
        offset_y = max(0, (viewport.height() - scaled.height()) // 2)
        painter = QPainter(fit_pixmap)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(offset_x, offset_y, scaled)
        painter.end()
        label.setPixmap(fit_pixmap)

    def _format_file_size(self, size_bytes: int) -> str:
        """Convert byte counts into a readable value suitable for display in the UI."""
        units = ["B", "KB", "MB", "GB"]
        value = float(size_bytes)
        unit_index = 0
        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024.0
            unit_index += 1

        if unit_index == 0:
            return f"{int(value)} {units[unit_index]}"
        return f"{value:.1f} {units[unit_index]}"

    def _image_file_details(self, path: str) -> str:
        """Return a short metadata summary for an image file."""
        if not path:
            return "No image details"

        try:
            file_path = Path(path)
            file_size = file_path.stat().st_size
            with Image.open(file_path) as image:
                width, height = image.size
                bits = getattr(image, "bits", None)
                if bits is None:
                    bits = {"1": 1, "I": 32, "F": 32}.get(image.mode, 16 if image.mode.startswith("I;16") else 8)
                format_name = (image.format or file_path.suffix.lstrip(".") or "unknown").lower()
                return f"{width} × {height}px • {bits}-bit • {format_name} • {self._format_file_size(file_size)}"
        except Exception:
            suffix = Path(path).suffix.lstrip(".").lower() or "unknown"
            size = Path(path).stat().st_size if Path(path).exists() else 0
            return f"Unknown size • {suffix} • {self._format_file_size(size)}"

    def _set_preview_text(self, label: QLabel, text: str):
        """Replace a preview pixmap with text."""
        label.setText(text)
        label.setPixmap(QPixmap())

    def _contrast_text_colour(self, colour: QColor) -> str:
        """Return black or white based on the luminance of a QColor."""
        red, green, blue = colour.getRgb()[:3]
        luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0
        return "#000000" if luminance > 0.55 else "#ffffff"

    def _apply_button_colour(self, button: QPushButton, colour: QColor):
        """Set a button background colour while keeping the text readable."""
        button.setStyleSheet(
            f"background-color: {colour.name()}; color: {self._contrast_text_colour(colour)}; "
            "border: 1px solid #555;"
        )

    def _update_overlay_button_style(self):
        """Display the current overlay colour on the check-alignment button."""
        if hasattr(self, "overlay_button"):
            self._apply_button_colour(self.overlay_button, self.mask_overlay_colour)
            self.overlay_button.setText(f" Mask overlay colour: {self.mask_overlay_colour.name()} ")

    def _ensure_image_loaded_for_key(self, key: str):
        """Check that the image was explicitly loaded for analysis."""
        return self.images[key].pixels.get("rgb") is not None

    def _active_image_keys(self):
        """Return the image slots required by the selected configuration."""
        if self.image_mode == "single":
            return ("B1",)
        return ("B1", "A") if self.image_mode == "one_each" else ("B1", "B2", "A")

    def _set_image_mode(self, mode):
        """Update image mode and related internal state.
        
        Note: In the new system, image mode is determined automatically based on
        how many before/after images the user selects during loading.
        """
        if mode == self.image_mode:
            return
        self.image_mode = mode
        
        # Update the analysis settings text based on mode
        if hasattr(self, 'analysis_settings_group'):
            self.analysis_settings_group.setTitle("Analysis settings" if mode == "single" else "Comparison settings")
        
        for widget, single_text, comparison_text in (
            (self.rgb_comparison_checkbox, "sRGB analysis", "sRGB comparison"),
            (self.lab_comparison_checkbox, "CEILAB analysis", "CEILAB comparison"),
            (self.mean_comparison_checkbox, "Mean colour statistics", "Mean comparison"),
        ):
            if hasattr(self, widget.objectName() if hasattr(widget, 'objectName') else ''):
                widget.setText(single_text if mode == "single" else comparison_text)
        
        if mode == "single" and hasattr(self, 'alignment_checkbox'):
            self.alignment_checkbox.setChecked(False)
            self.alignment_checkbox.setEnabled(False)
        
        self._refresh_disabled_tooltips()
        self._refresh_analysis_options_state()
        self.current_alignment_view = "B1"
        self._update_alignment_buttons()
        self.load_images_button.setEnabled(all(self.selected_image_paths[key] for key in self._active_image_keys()))
        self.load_status_label.setText("Image configuration changed. Select the required images and click Load images.")
        self.load_progress_bar.setValue(0)
        self.analyze_button.setEnabled(False)
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_label.setText("Load the selected images before analysis.")
        self.report.clear()
        for tab in (self.mask_tab, self.check_alignment_tab, self.analysis_options_tab, self.image_results_tab):
            self.tab_widget.setTabEnabled(self.tab_widget.indexOf(tab), False)
        self._refresh_disabled_tooltips()

    def _open_image_for_key(self, key: str):
        """Select and preview an image without loading its analysis arrays."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {self.image_key_labels[key]}",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return

        try:
            with Image.open(path) as image:
                image.thumbnail((800, 600))
                rgb = image.convert("RGB")
                data = rgb.tobytes()
                qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(qimage)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Image preview failed", f"Unable to preview {self.image_key_labels[key]}:\n{exc}")
            return

        self.selected_image_paths[key] = path
        label = self.load_preview_labels[key]
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.path_labels[key].setText(path)
        self.image_details_labels[key].setText(self._image_file_details(path))
        self.load_images_button.setEnabled(all(self.selected_image_paths[k] for k in self._active_image_keys()))
        self.load_status_label.setText("Selection changed. Select the required images and click Load images.")
        self.load_progress_bar.setValue(0)
        self.analyze_button.setEnabled(False)
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_label.setText("Load the selected images before analysis.")
        for tab in (self.mask_tab, self.check_alignment_tab, self.analysis_options_tab, self.image_results_tab):
            self.tab_widget.setTabEnabled(self.tab_widget.indexOf(tab), False)
        self._refresh_disabled_tooltips()

    def _reset_loaded_data(self):
        """Clear loaded images, masks, previews, and results while retaining image selections."""
        self.images = {key: image_data_colour(key) for key in self.selected_image_paths}
        self.analysis_options.int_paper_mask = None
        self.analysis_options.int_ink_mask = None
        for key, image in self.images.items():
            image.reference_colours = self.mask_reference_colours[key]
            active = key in self._active_image_keys()
            self.mask_original_labels[key].clear()
            self.mask_original_labels[key].setText(
                f"Original {self.alignment_buttons[key].text()}" if active else "No image"
            )
            self.mask_preview_labels[key].clear()
            self.mask_preview_labels[key].setText(
                f"Mask {self.alignment_buttons[key].text()}" if active else "No image"
            )
            self.mask_export_paths[key] = ""
            self.mask_export_path_fields[key].clear()
        self.alignment_preview.clear()
        self.alignment_preview.setText(f"{self.image_key_labels[self.current_alignment_view]} not loaded")
        self.report.clear()
        self.status_label.setText("Load the selected images for analysis.")
        self.analyze_button.setEnabled(False)
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_label.setText("Load the selected images before analysis.")

    def _map_slots(self, before_ids, after_ids):
        """Map ordered before/after image ids to the fixed B1/B2/A slots; return (mode, slots) or None if unsupported."""
        nb, na = len(before_ids), len(after_ids)
        if nb > 2 or na > 2 or (nb == 2 and na == 2):
            return None
        slots = {}
        if nb == 2:
            mode = "two_before"
            slots["B1"], slots["B2"] = before_ids
            if na:
                slots["A"] = after_ids[0]
        elif na == 2:
            mode = "two_after"
            if nb:
                slots["B1"] = before_ids[0]
            slots["B2"], slots["A"] = after_ids
        else:
            mode = "one_each"
            if nb:
                slots["B1"] = before_ids[0]
            if na:
                slots["A"] = after_ids[0]
        return mode, slots

    def _clear_slot(self, key):
        """Discard the loaded data for one fixed slot."""
        self.images[key] = image_data_colour(key)
        self.images[key].reference_colours = self.mask_reference_colours[key]
        self.mask_original_labels[key].clear()
        self.mask_original_labels[key].setText("No image")
        self.mask_preview_labels[key].clear()
        self.mask_preview_labels[key].setText("No image")
        self.mask_export_paths[key] = ""
        self.mask_export_path_fields[key].clear()

    def _invalidate_analysis(self):
        """Disable everything downstream of image loading until a complete set is loaded."""
        self.analyze_button.setEnabled(False)
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_label.setText("Load the selected images before analysis.")
        self.report.clear()
        for tab in (self.mask_tab, self.check_alignment_tab, self.analysis_options_tab, self.image_results_tab):
            self.tab_widget.setTabEnabled(self.tab_widget.indexOf(tab), False)

    def _refresh_card_loaded_state(self):
        """Show the green border on cards whose images are currently loaded."""
        loaded_ids = set(self.slot_cards.values())
        for image_id, card in self.load_image_cards.items():
            card.set_loaded(image_id in loaded_ids)

    def _ordered_ids(self, image_ids):
        """Return the given image ids in the order their cards appear in the grid."""
        wanted = set(image_ids)
        return [card.image_id for card in self.image_cards_area.cards() if card.image_id in wanted]

    def _reslot(self, mode, new_slots, new_items=None):
        """Re-assign already loaded images to B1/B2/A without reloading them.

        `new_items` maps image ids to (image_data, pixmap, export_path) for images that were just loaded.
        """
        items = {}
        for slot, image_id in self.slot_cards.items():
            items[image_id] = (
                self.images[slot],
                self.mask_original_labels[slot]._base_pixmap,
                self.mask_export_paths[slot],
            )
        items.update(new_items or {})

        for slot in ("B1", "B2", "A"):
            if slot not in new_slots:
                self._clear_slot(slot)
                continue
            image_data, pixmap, export_path = items[new_slots[slot]]
            image_data.key = slot
            self.images[slot] = image_data
            self.mask_reference_colours[slot] = image_data.reference_colours
            self.selected_image_paths[slot] = image_data.file_path
            self.mask_original_labels[slot].setPixmap(pixmap, preserve_zoom=True)
            self.mask_export_paths[slot] = export_path
            self.mask_export_path_fields[slot].setText(export_path)
            preview = self.mask_preview_labels[slot]
            if image_data.masks.get("ink") is not None and image_data.masks.get("paper") is not None:
                self._display_mask_preview(
                    preview,
                    paper_mask=image_data.masks.get("paper"),
                    ink_mask=image_data.masks.get("ink"),
                    border_mask=image_data.mask_border,
                )
            else:
                preview.clear()
                preview.setText(f"Mask {slot}")
        self.image_mode = mode
        self.slot_cards = dict(new_slots)

    def _publish_loaded_state(self):
        """Enable analysis and the downstream tabs when a complete set is loaded, otherwise lock them."""
        self._refresh_card_loaded_state()
        self._update_load_tab_state()
        self.report.clear()
        self.analysis_progress_bar.setValue(0)
        for tab in (self.image_results_tab,):
            self.tab_widget.setTabEnabled(self.tab_widget.indexOf(tab), False)
        if not self._images_ready():
            self._invalidate_analysis()
            return
        self._update_reference_colour_buttons()
        self._update_mask_file_controls()
        self._update_alignment_mode_availability()
        self._display_alignment_image()
        self.status_label.setText("Images loaded. Run the analysis to create a report.")
        self.analyze_button.setEnabled(True)
        self.analysis_progress_label.setText("Ready to analyse.")
        for tab in (self.mask_tab, self.check_alignment_tab, self.analysis_options_tab):
            self.tab_widget.setTabEnabled(self.tab_widget.indexOf(tab), True)

    def _unload_card(self, image_id):
        """Drop a loaded image (for example when it is removed or reclassified); the others keep their data."""
        if image_id not in self.slot_cards.values():
            return
        remaining = self._ordered_ids(i for i in self.slot_cards.values() if i != image_id)
        before_ids = [i for i in remaining if self.loaded_images[i]["is_before"]]
        after_ids = [i for i in remaining if self.loaded_images[i]["is_after"]]
        mode, slots = self._map_slots(before_ids, after_ids)
        self._reslot(mode, slots)
        self._publish_loaded_state()

    def _on_card_move_requested(self, image_id, step):
        """Move a card left or right in the grid and re-order the loaded images to match, without reloading."""
        if self._loading_thread is not None or not self.image_cards_area.move_card(self.load_image_cards[image_id], step):
            return
        self._update_load_tab_state()
        if not self.slot_cards:
            return
        loaded = self._ordered_ids(self.slot_cards.values())
        before_ids = [i for i in loaded if self.loaded_images[i]["is_before"]]
        after_ids = [i for i in loaded if self.loaded_images[i]["is_after"]]
        mode, slots = self._map_slots(before_ids, after_ids)
        if slots != self.slot_cards:
            self._reslot(mode, slots)
            self._publish_loaded_state()

    def _on_card_load_requested(self, image_id):
        """Load one image into its slot, keeping any other images that are already loaded."""
        if self._loading_thread is not None or image_id in self.slot_cards.values():
            return
        order = self._ordered_ids([*self.slot_cards.values(), image_id])
        before_ids = [i for i in order if self.loaded_images[i]["is_before"]]
        after_ids = [i for i in order if self.loaded_images[i]["is_after"]]
        mapping = self._map_slots(before_ids, after_ids)
        if mapping is None:
            QMessageBox.warning(
                self,
                "Unsupported combination",
                "Loading this image would exceed the supported combinations:\n"
                "- 1 before, 1 after\n- 2 before, 1 after\n- 1 before, 2 after\n\n"
                "Remove a loaded image (untick it or click X) first.",
            )
            return
        self._start_loading(mapping[0], mapping[1], f"Loading {Path(self.loaded_images[image_id]['path']).name}...")

    def _load_selected_images(self):
        """Start loading all ticked before/after images, replacing anything already loaded."""
        if self._loading_thread is not None:
            return

        ordered = self._ordered_ids(self.loaded_images)
        before_ids = [i for i in ordered if self.loaded_images[i]["is_before"]]
        after_ids = [i for i in ordered if self.loaded_images[i]["is_after"]]
        if not before_ids or not after_ids:
            QMessageBox.warning(
                self, "Invalid selection", "Select at least one before image and one after image before loading."
            )
            return
        mapping = self._map_slots(before_ids, after_ids)
        if mapping is None:
            QMessageBox.warning(
                self,
                "Unsupported combination",
                f"The combination of {len(before_ids)} before and {len(after_ids)} after images is not supported.\n\n"
                "Supported combinations:\n- 1 before, 1 after\n- 2 before, 1 after\n- 1 before, 2 after",
            )
            return
        for key in ("B1", "B2", "A"):
            self._clear_slot(key)
        self.slot_cards = {}
        self._invalidate_analysis()
        self._refresh_card_loaded_state()
        self._start_loading(mapping[0], mapping[1], "Loading images...")

    def _start_loading(self, mode, slots, status_text):
        """Load every image in `slots` that is not already loaded; the rest are re-slotted when loading finishes."""
        self._pending_plan = (mode, dict(slots))
        loaded_ids = set(self.slot_cards.values())
        self._pending_new = {key: image_id for key, image_id in slots.items() if image_id not in loaded_ids}
        paths = {key: self.loaded_images[image_id]["path"] for key, image_id in self._pending_new.items()}

        for index in range(1, self.tab_widget.count()):
            self.tab_widget.setTabEnabled(index, False)
        self.upload_button.setEnabled(False)
        self.load_images_button.setEnabled(False)
        for card in self.load_image_cards.values():
            card.load_button.setEnabled(False)
            card.left_button.setEnabled(False)
            card.right_button.setEnabled(False)
        self.cancel_load_button.setEnabled(True)
        self.cancel_load_button.setText("Cancel loading")
        self.cancel_load_button.show()
        self.load_progress_bar.setValue(0)
        self.load_status_label.setText(status_text)

        default_colours = image_data_colour("B1").reference_colours
        reference_colours = {key: dict(default_colours) for key in paths}
        self._loading_thread = ImageLoadingThread(paths, reference_colours, self)
        self._loading_thread.progress.connect(self._load_progress)
        self._loading_thread.finished.connect(self._load_finished)
        self._loading_thread.start()
        self._refresh_disabled_tooltips()


    def cancel_loading(self):
        """Request cooperative cancellation of the current image load."""
        if self._loading_thread is None:
            return
        self._loading_thread.cancel_requested.set()
        self.cancel_load_button.setEnabled(False)
        self.cancel_load_button.setText("Cancelling...")
        self.load_status_label.setText("Cancelling image loading...")
        self._refresh_disabled_tooltips()
        self._refresh_disabled_tooltips()

    @Slot(int, int, str, str)
    def _load_progress(self, current, total, key, description):
        """Display the current image and estimated midpoint progress."""
        if self._loading_thread is not None and self._loading_thread.cancel_requested.is_set():
            return
        self.load_progress_bar.setValue(((2 * current - 1) * 100 + total) // (2 * total))
        suffix = " (closing when finished)" if self._close_after_loading else ""
        self.load_status_label.setText(f"Task {current}/{total}: {description} - {key}{suffix}")

    def _load_finished(self):
        """Publish a successful image set or restore controls after cancellation or failure."""
        thread = self._loading_thread
        self._loading_thread = None
        self.upload_button.setEnabled(True)
        self._update_load_tab_state()
        self.cancel_load_button.hide()
        try:
            if thread.error is not None:
                self.load_status_label.setText("Loading failed. Check the selected files and try again.")
                image_key = thread.current_key
                QMessageBox.critical(self, "Image loading failed", f"Unable to load {image_key}:\n{thread.error}")
                self._restore_if_ready()
                return
            if thread.cancel_requested.is_set():
                self.load_status_label.setText("Image loading cancelled.")
                self._restore_if_ready()
                return

            mode, slots = self._pending_plan
            new_items = {}
            for key, image_id in self._pending_new.items():
                image_data = thread.images[key]
                image_path = Path(image_data.file_path)
                new_items[image_id] = (
                    image_data,
                    QPixmap.fromImage(thread.previews[key]),
                    str(image_path.with_name(f"{image_path.stem}_mask.png")),
                )
            self._reslot(mode, slots, new_items)
            self.load_progress_bar.setValue(100)
            self._publish_loaded_state()
            if not self._images_ready():
                self.load_status_label.setText("Image loaded. Load the remaining before/after images to continue.")
                return
            count = len(self._active_image_keys())
            self.load_status_label.setText(f"Images loaded ({count}/{count}).")
        finally:
            self._refresh_disabled_tooltips()
            thread.deleteLater()
            if self._close_after_loading:
                self.close()

    def _images_ready(self):
        """Return whether every slot needed by the current mode holds a loaded image."""
        return all(self._ensure_image_loaded_for_key(key) for key in self._active_image_keys())

    def _restore_if_ready(self):
        """Re-enable downstream tabs after a failed or cancelled load if the previous set is still complete."""
        if self.slot_cards and self._images_ready():
            self.analyze_button.setEnabled(True)
            for tab in (self.mask_tab, self.check_alignment_tab, self.analysis_options_tab):
                self.tab_widget.setTabEnabled(self.tab_widget.indexOf(tab), True)
            self._refresh_disabled_tooltips()
            thread.deleteLater()
            if self._close_after_loading:
                self.close()

    def _split_tab(self, page):
        """Lay out a tab as a fixed 250 px settings column on the left and content on the right."""
        page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        columns = QHBoxLayout(page)
        settings = QWidget()
        settings.setFixedWidth(250)
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setAlignment(Qt.AlignTop)
        content_frame = QFrame()
        content_frame.setObjectName("tabViewport")
        content_frame.setStyleSheet(
            "#tabViewport { background: palette(base); border: 1px solid palette(mid); border-radius: 6px; }"
        )
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(10, 10, 10, 10)
        columns.addWidget(settings)
        columns.addWidget(content_frame, 1)
        return settings_layout, content_layout

    def _button_row(self, *buttons):
        """Lay buttons out side by side in equal halves, keeping each half's width when its partner is hidden."""
        row = QGridLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for column, button in enumerate(buttons):
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            row.addWidget(button, 0, column)
            row.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum), 1, column)
            row.setColumnStretch(column, 1)
        return row

    def _make_card_title(self, text):
        """Create the bold title shown at the top of an image card."""
        title = QLabel(text)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 13px; border: none; background: transparent;")
        return title

    def _make_image_card(self):
        """Create a bordered box holding one image and its controls."""
        card = QFrame()
        card.setObjectName("imageCard")
        card.setStyleSheet(
            "#imageCard { background: palette(window); border: 1px solid palette(mid); border-radius: 6px; }"
        )
        return card

    def _create_load_tab(self):
        """Tab 1: select, preview, and load images with before/after classification."""
        page = QWidget()
        layout, content = self._split_tab(page)

        # Left panel controls
        self.upload_button = QPushButton("Upload images")
        self.upload_button.setToolTip("Select one or more images to add them to the workspace.")
        self.upload_button.clicked.connect(self._on_upload_images_clicked)
        layout.addWidget(self.upload_button)

        self.remove_unselected_button = QPushButton("Remove unselected")
        self.remove_unselected_button.setToolTip("Remove all images that are not marked as before or after.")
        self.remove_unselected_button.setEnabled(False)
        self.remove_unselected_button.clicked.connect(self._remove_unselected_images)
        layout.addWidget(self.remove_unselected_button)
        layout.addSpacing(10)
        
        self.load_images_button = QPushButton("Load selected")
        self.load_images_button.setToolTip("Load and prepare all selected before/after images for analysis.")
        self.load_images_button.setEnabled(False)
        self.load_images_button.clicked.connect(self._load_selected_images)

        self.cancel_load_button = QPushButton("Cancel loading")
        self.cancel_load_button.setToolTip("Cancel loading between processing steps; selected images are retained.")
        self.cancel_load_button.setStyleSheet(CANCEL_BUTTON_STYLE)
        self.cancel_load_button.clicked.connect(self.cancel_loading)
        self.cancel_load_button.hide()
        layout.addLayout(self._button_row(self.load_images_button, self.cancel_load_button))

        layout.addSpacing(10)

        self.load_progress_bar = QProgressBar()
        self.load_progress_bar.setRange(0, 100)
        self.load_progress_bar.setValue(0)
        layout.addWidget(self.load_progress_bar)

        self.load_status_label = QLabel("Click 'Upload images' to select images, then mark them as before or after.")
        self.load_status_label.setWordWrap(True)
        layout.addWidget(self.load_status_label)

        layout.addStretch(1)

        # Right panel: scrollable area with wrapping image cards
        self.image_cards_area = CardGridArea()
        content.addWidget(self.image_cards_area)

        # The shared tab frame is the only border; tint its background slightly away from the window colour.
        frame = content.parentWidget()
        window_colour = self.palette().window().color()
        tint = window_colour.darker(106) if window_colour.lightness() > 128 else window_colour.lighter(120)
        frame.setStyleSheet(
            f"#tabViewport {{ background: {tint.name()}; border: 1px solid palette(mid); border-radius: 6px; }}"
        )
        content.setContentsMargins(0, 0, 0, 0)

        # Selection and loaded counts sit at the bottom of the settings column
        self.selected_count_label = QLabel("Selected - Before: 0 | After: 0")
        self.selected_count_label.setWordWrap(True)
        layout.addWidget(self.selected_count_label)
        self.loaded_count_label = QLabel("Loaded - Before: 0 | After: 0")
        self.loaded_count_label.setWordWrap(True)
        layout.addWidget(self.loaded_count_label)

        return page
    
    def _on_upload_images_clicked(self):
        """Open file dialog to select multiple images."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images to upload",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not paths:
            return
        
        for path in paths:
            self._add_image_card(path)
        
        self._update_load_tab_state()
    
    def _add_image_card(self, file_path: str):
        """Create and add an image card for the given file path."""
        try:
            with Image.open(file_path) as image:
                image.thumbnail((468, 352))
                rgb = image.convert("RGB")
                data = rgb.tobytes()
                qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(qimage)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Image preview failed", f"Unable to load image:\n{exc}")
            return

        # Generate unique image ID
        image_id = str(self.next_image_id)
        self.next_image_id += 1

        # Store image data
        self.loaded_images[image_id] = {
            "path": file_path,
            "pixmap": pixmap,
            "is_before": False,
            "is_after": False,
        }

        # Create and add card
        card = ImageCardWidget(
            image_id,
            pixmap,
            self,
            filename=Path(file_path).name,
            details=f"{file_path}\n{self._image_file_details(file_path)}",
        )
        card.removed.connect(self._on_image_card_removed)
        card.before_toggled.connect(self._on_image_before_toggled)
        card.load_requested.connect(self._on_card_load_requested)
        card.move_requested.connect(self._on_card_move_requested)
        card.after_toggled.connect(self._on_image_after_toggled)
        
        self.load_image_cards[image_id] = card
        self.image_cards_area.add_card(card)
    
    def _on_image_card_removed(self, image_id: str):
        """Remove an image card and its data."""
        self._unload_card(image_id)
        if image_id in self.load_image_cards:
            self.image_cards_area.remove_card(self.load_image_cards[image_id])
            del self.load_image_cards[image_id]
        
        if image_id in self.loaded_images:
            del self.loaded_images[image_id]
        
        self._update_load_tab_state()
    
    def _on_image_before_toggled(self, image_id: str, checked: bool):
        """Update the before flag for an image."""
        if image_id in self.loaded_images:
            self.loaded_images[image_id]["is_before"] = checked
            self._unload_card(image_id)
            self._update_load_tab_state()
    
    def _on_image_after_toggled(self, image_id: str, checked: bool):
        """Update the after flag for an image."""
        if image_id in self.loaded_images:
            self.loaded_images[image_id]["is_after"] = checked
            self._unload_card(image_id)
            self._update_load_tab_state()
    
    def _remove_unselected_images(self):
        """Remove every image card that is marked as neither before nor after."""
        if self._loading_thread is not None:
            return
        for image_id, data in list(self.loaded_images.items()):
            if not data["is_before"] and not data["is_after"]:
                self._on_image_card_removed(image_id)

    def _update_load_tab_state(self):
        """Update the load tab UI based on current image selection."""
        # Count before and after images
        before_count = sum(1 for img in self.loaded_images.values() if img["is_before"])
        after_count = sum(1 for img in self.loaded_images.values() if img["is_after"])
        total_count = len(self.loaded_images)
        idle = self._loading_thread is None
        self.remove_unselected_button.setEnabled(
            idle and any(not d["is_before"] and not d["is_after"] for d in self.loaded_images.values())
        )        
        # Enable load button if we have at least one before and one after
        can_load = before_count > 0 and after_count > 0
        self.load_images_button.setEnabled(can_load and idle)
        loaded_ids = set(self.slot_cards.values())
        for image_id, card in self.load_image_cards.items():
            data = self.loaded_images[image_id]
            card.load_button.setEnabled(idle and (data["is_before"] or data["is_after"]) and image_id not in loaded_ids)
            card.load_button.setText("Loaded" if image_id in loaded_ids else "Load")
        ordered_cards = self.image_cards_area.cards()
        for index, card in enumerate(ordered_cards):
            card.left_button.setEnabled(idle and index > 0)
            card.right_button.setEnabled(idle and index < len(ordered_cards) - 1)
        
        loaded_before = sum(1 for i in loaded_ids if self.loaded_images[i]["is_before"])
        loaded_after = sum(1 for i in loaded_ids if self.loaded_images[i]["is_after"])
        self.selected_count_label.setText(f"Selected - Before: {before_count} | After: {after_count}")
        self.loaded_count_label.setText(f"Loaded - Before: {loaded_before} | After: {loaded_after}")

    def _get_intersection_masks_for_alignment(self):
        """Return the combined paper and ink masks across all loaded images."""
        return _intersection_masks({key: self.images[key] for key in self._active_image_keys()})

    def _display_alignment_image(self):
        """Show the selected image in the alignment check area."""
        key = self.current_alignment_view
        image_data = self.images[key]
        image = image_data.get_pixels("rgb") if image_data.pixels.get("rgb") is not None else None
        if image is None:
            self.alignment_preview.setText(f"{self.image_key_labels[key]} not loaded")
            self.alignment_preview.setPixmap(QPixmap())
            return

        mode = self.alignment_display_mode
        if mode == "original" or not self.use_mask_checkbox.isChecked():
            self.alignment_preview.setPixmap(self.mask_original_labels[key]._base_pixmap, preserve_zoom=True)
            return

        if self.alignment_checkbox.isChecked():
            paper_mask, ink_mask = self._get_intersection_masks_for_alignment()
            if paper_mask is None or ink_mask is None:
                self._preview_from_array(self.alignment_preview, image)
                return
        else:
            paper_mask = image_data.masks.get("paper")
            ink_mask = image_data.masks.get("ink")
            if paper_mask is None or ink_mask is None:
                self._preview_from_array(self.alignment_preview, image)
                return

        if mode == "mask":
            self._display_mask_preview(
                self.alignment_preview,
                paper_mask=paper_mask,
                ink_mask=ink_mask,
            )
            return

        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        array = np.clip(array, 0.0, 1.0)

        paper_bool = _as_bool_mask(paper_mask)
        ink_bool = _as_bool_mask(ink_mask)
        overlay = np.array(self.mask_overlay_colour.getRgb()[:3], dtype=np.float32) / 255.0
        displayed = array.copy()

        if mode in {"paper_mask", "masked_paper"}:
            displayed[~paper_bool] = overlay
        elif mode in {"ink_mask", "masked_ink"}:
            displayed[~ink_bool] = overlay
        else:
            displayed[~paper_bool] = overlay

        rgb = (displayed * 255.0).astype("uint8")
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        self.alignment_preview.setPixmap(pixmap, preserve_zoom=True)

    def _cycle_alignment_image(self):
        """Advance the alignment preview to the next image."""
        order = self._active_image_keys()
        current_index = order.index(self.current_alignment_view)
        self.current_alignment_view = order[(current_index + 1) % len(order)]
        self._update_alignment_buttons()
        self._display_alignment_image()

    def _images_have_identical_resolution(self):
        """Check whether all loaded images have matching dimensions."""
        sizes = []
        for key in self._active_image_keys():
            image_data = self.images[key]
            image = image_data.pixels.get("rgb")
            if image is None:
                return False
            array = np.asarray(image)
            if array.ndim == 0:
                return False
            sizes.append(tuple(array.shape[:2]))
        return len(set(sizes)) == 1

    def _update_alignment_mode_availability(self):
        """Allow aligned mode only when the image resolutions match."""
        if not hasattr(self, "alignment_checkbox"):
            return

        available = self.image_mode != "single" and self._images_have_identical_resolution()
        self.alignment_checkbox.setEnabled(available)
        if not available:
            self.alignment_checkbox.setChecked(False)
        self._refresh_analysis_options_state()

    def _update_alignment_buttons(self):
        """Mark the currently displayed image in the alignment selector."""
        for key, button in self.alignment_buttons.items():
            button.setChecked(key == self.current_alignment_view)

    def _reset_alignment_zoom(self):
        """Fit and centre the alignment preview image."""
        if hasattr(self, "alignment_preview") and self.alignment_preview._base_pixmap is not None and not self.alignment_preview._base_pixmap.isNull():
            self.alignment_preview._auto_fit_zoom = True
            self.alignment_preview._apply_fit_zoom()
            self.alignment_preview._refresh_zoomed_pixmap()

    def _on_tab_changed(self, index: int):
        """Fit the alignment image on the first visit to its tab."""
        if not self._alignment_tab_visited and self.tab_widget.widget(index) is self.check_alignment_tab:
            self._alignment_tab_visited = True
            self._reset_alignment_zoom()

    def _create_check_alignment_tab(self):
        """Create the tab for visually inspecting alignment between images."""
        page = QWidget()
        layout, content = self._split_tab(page)

        self.alignment_checkbox = QCheckBox("Combined mask")
        self.alignment_checkbox.setToolTip(
            "Display the masks combined across all selected images (pixels included in every image's mask). "
            "Display only; all selected images must have the same resolution."
        )
        self.alignment_checkbox.setChecked(False)
        self.alignment_checkbox.setEnabled(False)
        self.analysis_options.images_aligned = self.alignment_checkbox.isChecked()
        self.alignment_checkbox.toggled.connect(
            lambda checked: (
                self.analysis_options.sync_from_ui(images_aligned=checked),
                self._display_alignment_image(),
            )
        )

        view_group = QGroupBox("Mask display")
        view_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        view_layout = QVBoxLayout(view_group)
        self.alignment_mode_group = QButtonGroup(page)
        self.original_radio = QRadioButton("Original")
        self.original_radio.setToolTip("Show the selected image without a mask overlay.")
        self.mask_radio = QRadioButton("Mask")
        self.mask_radio.setToolTip(
            "When masks are enabled, show paper in white, ink in black, both in grey, and excluded pixels in red."
        )
        self.masked_paper_radio = QRadioButton("Paper mask")
        self.masked_paper_radio.setToolTip("When masks are enabled, colour pixels outside the paper mask.")
        self.masked_ink_radio = QRadioButton("Ink mask")
        self.masked_ink_radio.setToolTip("When masks are enabled, colour pixels outside the ink mask.")
        self.original_radio.setChecked(True)
        self.alignment_mode_group.addButton(self.original_radio)
        self.alignment_mode_group.addButton(self.mask_radio)
        self.alignment_mode_group.addButton(self.masked_paper_radio)
        self.alignment_mode_group.addButton(self.masked_ink_radio)
        self.original_radio.toggled.connect(lambda checked: self._set_alignment_mode("original") if checked else None)
        self.mask_radio.toggled.connect(lambda checked: self._set_alignment_mode("mask") if checked else None)
        self.masked_paper_radio.toggled.connect(lambda checked: self._set_alignment_mode("paper_mask") if checked else None)
        self.masked_ink_radio.toggled.connect(lambda checked: self._set_alignment_mode("ink_mask") if checked else None)
        for radio in [self.original_radio, self.mask_radio, self.masked_paper_radio, self.masked_ink_radio]:
            view_layout.addWidget(radio)
        self.overlay_button = QPushButton(f"Mask overlay colour: {self.mask_overlay_colour.name()}")
        self.overlay_button.setToolTip("Choose the colour used for pixels outside the selected mask.")
        self.overlay_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.overlay_button.clicked.connect(self._choose_mask_overlay_colour)
        self._update_overlay_button_style()
        self._update_alignment_mode_availability()
        view_layout.addWidget(self.alignment_checkbox)
        view_layout.addWidget(self.overlay_button)
        layout.addWidget(view_group)

        button_row = QHBoxLayout()
        button_row.setAlignment(Qt.AlignLeft)
        self.alignment_buttons = {}
        for key in ["B1", "B2", "A"]:
            button = QPushButton(key)
            button.setToolTip(f"Show {self.image_key_labels[key]} in the alignment preview.")
            button.setCheckable(True)
            button.clicked.connect(lambda checked, value=key: self._set_alignment_view(value))
            button_row.addWidget(button)
            self.alignment_buttons[key] = button

        self.alignment_preview = ClickableLabel("B1")
        self.alignment_preview.setAlignment(Qt.AlignCenter)
        self.alignment_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.alignment_preview.setMinimumSize(200, 120)
        self.alignment_preview.setStyleSheet("border: 1px solid #888; background: #262626;")
        self.alignment_preview.clicked.connect(self._cycle_alignment_image)

        self.reset_alignment_zoom_button = QPushButton("Reset zoom")
        self.reset_alignment_zoom_button.setToolTip("Fit the alignment image to the preview and reset its position.")
        self.reset_alignment_zoom_button.clicked.connect(self._reset_alignment_zoom)

        selector_row = QHBoxLayout()
        selector_row.setAlignment(Qt.AlignLeft)
        selector_row.addLayout(button_row)
        selector_row.addStretch(1)
        selector_row.addWidget(self.reset_alignment_zoom_button)
        content.addLayout(selector_row)
        content.addWidget(self.alignment_preview)

        self._update_alignment_buttons()
        self._refresh_alignment_preview()
        self._reset_alignment_zoom()
        return page

    def _set_alignment_view(self, key: str):
        """Select an image for the alignment preview."""
        if key not in self._active_image_keys():
            return
        self.current_alignment_view = key
        self._update_alignment_buttons()
        self._display_alignment_image()

    def _set_alignment_mode(self, mode: str):
        """Select the alignment preview's image or mask display mode."""
        self.alignment_display_mode = mode
        self._display_alignment_image()

    def _refresh_alignment_preview(self):
        """Redraw the current alignment preview."""
        self._display_alignment_image()

    def _choose_mask_overlay_colour(self):
        """Choose the colour used to display pixels outside a mask."""
        colour = QColorDialog.getColor(self.mask_overlay_colour, self, "Select masked area colour")
        if colour.isValid():
            self.mask_overlay_colour = colour
            self._update_overlay_button_style()
            self._display_alignment_image()

    def _set_mask_mode(self, mode: str):
        """Select a mask-generation mode and update its controls."""
        self.mask_mode = mode
        use_mask_enabled = self.use_mask_checkbox.isChecked() if hasattr(self, "use_mask_checkbox") else True
        reference_mode_enabled = use_mask_enabled and mode == "reference_colour"
        luminance_mode_enabled = use_mask_enabled and mode == "luminance"

        if hasattr(self, "reference_weight_slider"):
            self.reference_weight_slider.setEnabled(reference_mode_enabled)
            self.reference_weight_label.setEnabled(reference_mode_enabled)
        if hasattr(self, "mask_threshold_slider"):
            self.mask_threshold_slider.setEnabled(luminance_mode_enabled)
            self.mask_threshold_label.setEnabled(luminance_mode_enabled)
        if hasattr(self, "change_all_paper_button"):
            self.change_all_paper_button.setEnabled(reference_mode_enabled)
        if hasattr(self, "change_all_ink_button"):
            self.change_all_ink_button.setEnabled(reference_mode_enabled)
        if hasattr(self, "paper_colour_buttons"):
            for key in ["B1", "B2", "A"]:
                self.paper_colour_buttons[key].setEnabled(reference_mode_enabled)
                self.ink_colour_buttons[key].setEnabled(reference_mode_enabled)
                if reference_mode_enabled:
                    self._apply_button_colour(self.paper_colour_buttons[key], self.mask_reference_colours[key]["paper"])
                    self._apply_button_colour(self.ink_colour_buttons[key], self.mask_reference_colours[key]["ink"])
                else:
                    self.paper_colour_buttons[key].setStyleSheet("background-color: #4d4d4d; color: #7a7a7a; border: 1px solid #666;")
                    self.ink_colour_buttons[key].setStyleSheet("background-color: #4d4d4d; color: #7a7a7a; border: 1px solid #666;")
        if hasattr(self, "generate_all_masks_button"):
            self.generate_all_masks_button.setEnabled(use_mask_enabled)
        if hasattr(self, "mask_generate_buttons"):
            for button in self.mask_generate_buttons.values():
                button.setEnabled(use_mask_enabled)
        if hasattr(self, "mask_remove_buttons"):
            for button in self.mask_remove_buttons.values():
                button.setEnabled(use_mask_enabled)
        if hasattr(self, "luminance_radio"):
            self.luminance_radio.setEnabled(use_mask_enabled)
        if hasattr(self, "colour_radio"):
            self.colour_radio.setEnabled(use_mask_enabled)
        if hasattr(self, "alignment_preview"):
            self._display_alignment_image()
        self._refresh_disabled_tooltips()

    def _set_mask_reference_colour(self, key: str, region: str):
        """Choose a paper or ink reference colour for one image."""
        image_data = self.images[key]
        colour = QColorDialog.getColor(image_data.reference_colours[region], self, f"Select {region} colour for {self.image_key_labels[key]}")
        if not colour.isValid():
            return

        image_data.reference_colours[region] = colour
        self.mask_reference_colours[key] = image_data.reference_colours
        self._update_reference_colour_buttons()

    def _change_all_reference_colours(self, region: str):
        """Choose one reference colour for the same region in all images."""
        representative = self.images["B1"].reference_colours[region]
        colour = QColorDialog.getColor(representative, self, f"Select all {region} colours")
        if not colour.isValid():
            return

        for key in self._active_image_keys():
            self.images[key].reference_colours[region] = colour
            self.mask_reference_colours[key] = self.images[key].reference_colours

        self._update_reference_colour_buttons()

    def _update_reference_colour_buttons(self):
        """Refresh reference-colour button labels and swatches."""
        for key in ["B1", "B2", "A"]:
            paper_colour = self.images[key].reference_colours["paper"]
            ink_colour = self.images[key].reference_colours["ink"]
            self._apply_button_colour(self.paper_colour_buttons[key], paper_colour)
            self._apply_button_colour(self.ink_colour_buttons[key], ink_colour)
            self.paper_colour_buttons[key].setText(f"Paper: {paper_colour.name()}")
            self.ink_colour_buttons[key].setText(f"Ink: {ink_colour.name()}")

        if hasattr(self, "change_all_paper_button"):
            self.change_all_paper_button.setStyleSheet("")
            self.change_all_paper_button.setText("Set paper colours")
        if hasattr(self, "change_all_ink_button"):
            self.change_all_ink_button.setStyleSheet("")
            self.change_all_ink_button.setText("Set ink colours")

    def _generate_mask_for_key(self, key: str):
        """Create separate paper and ink masks and keep the red border out of both."""
        if not self._ensure_image_loaded_for_key(key):
            QMessageBox.warning(self, "Missing image", f"Load {self.image_key_labels[key]} before generating a mask.")
            return

        image_data = self.images[key]
        paper_mask, ink_mask, border_mask = image_data.generate_masks(
            self.mask_mode,
            threshold=self.mask_threshold_slider.value(),
            reference_weight=self.reference_weight_slider.value() / 100.0,
            border_pixels=self.mask_border_slider.value(),
        )

        if paper_mask is None or ink_mask is None:
            return

        if image_data.masks.get("ink") is not None:
            self._display_mask_preview(
                self.mask_preview_labels[key],
                paper_mask=image_data.masks.get("paper"),
                ink_mask=image_data.masks.get("ink"),
                border_mask=image_data.mask_border,
            )

        self._update_mask_file_controls()
        self._display_alignment_image()

    def _start_mask_batch(self, action, keys, paths=None):
        """Run a mask batch on a worker thread, updating the progress bar as each image finishes."""
        if self._mask_thread is not None:
            return
        keys = list(keys)
        title = "Generating masks" if action == "generate" else "Importing masks"
        if not keys:
            return
        settings = {
            "mode": self.mask_mode,
            "threshold": self.mask_threshold_slider.value(),
            "reference_weight": self.reference_weight_slider.value() / 100.0,
            "border_pixels": self.mask_border_slider.value(),
        }
        thread = MaskBatchThread(action, keys, self.images, paths=paths, settings=settings, parent=self)
        thread.item_done.connect(self._mask_item_done)
        thread.finished.connect(self._mask_batch_finished)
        self._mask_thread = thread
        self._mask_batch = {
            "action": action, "title": title, "done": 0, "total": len(keys),
            "imported": [], "missing": [], "errors": [],
            "buttons": {
                button: button.isEnabled()
                for button in (self.generate_all_masks_button, self.auto_import_masks_button)
            },
        }
        for button in self._mask_batch["buttons"]:
            button.setEnabled(False)
        self.cancel_mask_batch_button.setEnabled(True)
        self.cancel_mask_batch_button.show()
        self.mask_progress_bar.setValue(0)
        self.mask_progress_label.setText(f"{title}: 0/{len(keys)}")
        thread.start()

    def _cancel_mask_batch(self):
        """Request cancellation of the running mask batch after the current image."""
        if self._mask_thread is None:
            return
        self._mask_thread.cancel_requested.set()
        self.cancel_mask_batch_button.setEnabled(False)
        self.mask_progress_label.setText("Cancelling...")

    def _mask_item_done(self, done, total, key, status, detail):
        """Refresh the previews and progress for one finished image."""
        batch = self._mask_batch
        if batch is None:
            return
        label = self.alignment_buttons[key].text()
        image_data = self.images[key]
        if status == "ok":
            batch["imported"].append(label)
            if batch["action"] == "generate":
                if image_data.masks.get("ink") is not None:
                    self._display_mask_preview(
                        self.mask_preview_labels[key],
                        paper_mask=image_data.masks.get("paper"),
                        ink_mask=image_data.masks.get("ink"),
                        border_mask=image_data.mask_border,
                    )
            else:
                self._display_mask_preview(
                    self.mask_preview_labels[key],
                    paper_mask=image_data.masks["paper"],
                    ink_mask=image_data.masks["ink"],
                )
            self._update_mask_file_controls()
            self._display_alignment_image()
        elif status == "missing":
            batch["missing"].append(label)
        elif status == "error":
            batch["errors"].append(f"{label}: {detail}")
        batch["done"] = done
        self.mask_progress_bar.setValue(done * 100 // total)
        self.mask_progress_label.setText(f"{batch['title']}: {done}/{total}")

    def _mask_batch_finished(self):
        """Restore the controls and report the outcome once the worker thread has stopped."""
        thread, batch = self._mask_thread, self._mask_batch
        self._mask_thread = None
        self._mask_batch = None
        if thread is not None:
            thread.deleteLater()
        if batch is None:
            return
        self.cancel_mask_batch_button.hide()
        for button, enabled in batch["buttons"].items():
            button.setEnabled(enabled)
        total = batch["total"]
        cancelled = batch["done"] < total
        if cancelled:
            self.mask_progress_label.setText(f"{batch['title']} cancelled ({batch['done']}/{total} done).")
        else:
            self.mask_progress_bar.setValue(100)
            self.mask_progress_label.setText(f"{batch['title']} complete ({total}/{total}).")
        if batch["action"] == "import":
            details = [f"Imported: {', '.join(batch['imported']) if batch['imported'] else 'none'}."]
            if batch["missing"]:
                details.append(f"No mask found: {', '.join(batch['missing'])}.")
            if batch["errors"]:
                details.append("Import failed:\n" + "\n".join(batch["errors"]))
                QMessageBox.warning(self, "Auto-import masks", "\n".join(details))
            else:
                QMessageBox.information(self, "Auto-import masks", "\n".join(details))
        if self._close_after_mask_batch:
            self.close()

    def _generate_mask_for_all(self):
        """Generate masks for all images currently loaded into the app."""
        keys = [key for key in self._active_image_keys() if self.images[key].pixels.get("rgb") is not None]
        self._start_mask_batch("generate", keys)

    def _remove_mask_for_key(self, key: str):
        """Clear the saved mask for a single image, leaving the original image visible."""
        image_data = self.images[key]
        image_data.clear_masks()

        if key in self.mask_preview_labels:
            self.mask_preview_labels[key].clear()
            self.mask_preview_labels[key].setText(f"Mask {self.alignment_buttons[key].text()}")

        self._update_mask_file_controls()
        self._display_alignment_image()

    def _import_mask_for_key(self, key: str):
        """Prompt for a mask file and import it for the selected image."""
        image_data = self.images[key]
        if not self._ensure_image_loaded_for_key(key):
            QMessageBox.warning(self, "Missing image", f"Load {self.image_key_labels[key]} before importing a mask.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, f"Import mask for {self.image_key_labels[key]}", str(Path(image_data.file_path).parent),
            "Images (*.png *.bmp *.tif *.tiff *.jpg *.jpeg)",
        )
        if not path:
            return
        try:
            self._import_mask_from_path(key, path)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            QMessageBox.critical(self, "Mask import failed", f"Unable to import the mask for {self.alignment_buttons[key].text()}:\n{exc}")

    def _import_mask_from_path(self, key: str, path: str):
        """Load a mask from a path and refresh the affected previews."""
        image_data = self.images[key]
        image_data.import_mask(path)
        self._display_mask_preview(
            self.mask_preview_labels[key],
            paper_mask=image_data.masks["paper"],
            ink_mask=image_data.masks["ink"],
        )
        self._update_mask_file_controls()
        self._display_alignment_image()

    def _auto_import_masks(self):
        """Import available masks from the configured export paths on a worker thread."""
        keys = self._active_image_keys()
        self._start_mask_batch("import", keys, paths={key: self.mask_export_paths[key] for key in keys})

    def _set_mask_export_path_for_key(self, key: str):
        """Choose and validate the PNG export destination for one image."""
        path, _ = QFileDialog.getSaveFileName(
            self, f"Set mask export path for {self.image_key_labels[key]}", self.mask_export_paths[key],
            "PNG images (*.png)", options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".png"
        if Path(path).suffix.lower() != ".png":
            QMessageBox.warning(self, "Invalid export path", "Choose a PNG file to preserve the mask colours.")
            return
        self.mask_export_paths[key] = path
        self.mask_export_path_fields[key].setText(path)

    def _export_mask_for_key(self, key: str, *, notify: bool = True):
        """Export one mask after confirming replacement of an existing file."""
        path = self.mask_export_paths[key]
        if not path:
            QMessageBox.warning(self, "Missing export path", f"Load {self.image_key_labels[key]} before exporting a mask.")
            return False
        if Path(path).exists() and QMessageBox.question(
            self, "Overwrite mask?", f"{path} already exists. Replace it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return False
        try:
            self.images[key].export_mask(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Mask export failed", f"Unable to export the mask for {self.alignment_buttons[key].text()}:\n{exc}")
            return False
        if notify:
            QMessageBox.information(self, "Mask exported", f"Mask saved to:\n{path}")
        return True

    def _export_all_masks(self):
        """Export masks to distinct configured paths and summarise the outcomes."""
        paths = [os.path.normcase(os.path.realpath(self.mask_export_paths[key]))
                 for key in self._active_image_keys() if self.mask_export_paths[key]]
        if len(set(paths)) != len(paths):
            QMessageBox.warning(self, "Mask export failed", "Choose a different export path for each image.")
            return
        exported = []
        skipped = []
        for key in self._active_image_keys():
            if self._export_mask_for_key(key, notify=False):
                exported.append(self.alignment_buttons[key].text())
            else:
                skipped.append(self.alignment_buttons[key].text())
        details = [f"Exported: {', '.join(exported) if exported else 'none'}."]
        if skipped:
            details.append(f"Skipped or failed: {', '.join(skipped)}.")
        QMessageBox.information(self, "Export all masks", "\n".join(details))

    def _reset_analysis_progress(self, *_args):
        """Clear finished progress once settings or masks change, since the report is out of date."""
        if self._analysis_thread is not None or not hasattr(self, "analysis_progress_label"):
            return
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_label.setText("Ready to analyse.")

    def _update_mask_file_controls(self):
        """Enable mask import and export controls when masking is enabled."""
        self._reset_analysis_progress()
        if not hasattr(self, "mask_export_buttons"):
            return
        enabled = self.use_mask_checkbox.isChecked()
        self.auto_import_masks_button.setEnabled(enabled)
        self.export_all_masks_button.setEnabled(enabled)
        for key in self.mask_export_buttons:
            self.mask_import_buttons[key].setEnabled(enabled)
            self.mask_export_path_buttons[key].setEnabled(enabled)
            self.mask_export_buttons[key].setEnabled(enabled)
        self._refresh_disabled_tooltips()

    def _update_mask_generation_button_text(self):
        """Reset the per-image mask generation button labels."""
        if not hasattr(self, "mask_generate_buttons"):
            return

        for key, button in self.mask_generate_buttons.items():
            button.setText("Generate mask")

    def _set_mask_controls_enabled(self, enabled: bool):
        """Enable or disable the whole mask section without overriding the active mode rules."""
        if hasattr(self, "use_mask_checkbox"):
            self.use_mask_checkbox.setChecked(enabled)

        if not enabled:
            self.analysis_options.use_masks = False
        if hasattr(self, "use_masks_checkbox"):
            self._refresh_analysis_options_state()

        if hasattr(self, "mask_mode_group"):
            for button in self.mask_mode_group.buttons():
                button.setEnabled(enabled)

        if hasattr(self, "mask_threshold_slider"):
            self.mask_threshold_slider.setEnabled(enabled and self.mask_mode == "luminance")
            self.mask_threshold_label.setEnabled(enabled and self.mask_mode == "luminance")
        if hasattr(self, "reference_weight_slider"):
            self.reference_weight_slider.setEnabled(enabled and self.mask_mode == "reference_colour")
            self.reference_weight_label.setEnabled(enabled and self.mask_mode == "reference_colour")
        if hasattr(self, "mask_border_slider"):
            self.mask_border_slider.setEnabled(enabled)
            self.mask_border_label.setEnabled(enabled)

        if hasattr(self, "paper_colour_buttons"):
            for key, button in self.paper_colour_buttons.items():
                should_enable = enabled and self.mask_mode == "reference_colour"
                button.setEnabled(should_enable)
                if should_enable:
                    self._apply_button_colour(button, self.mask_reference_colours[key]["paper"])
                else:
                    button.setStyleSheet("background-color: #4d4d4d; color: #7a7a7a; border: 1px solid #666;")

        if hasattr(self, "ink_colour_buttons"):
            for key, button in self.ink_colour_buttons.items():
                should_enable = enabled and self.mask_mode == "reference_colour"
                button.setEnabled(should_enable)
                if should_enable:
                    self._apply_button_colour(button, self.mask_reference_colours[key]["ink"])
                else:
                    button.setStyleSheet("background-color: #4d4d4d; color: #7a7a7a; border: 1px solid #666;")

        if hasattr(self, "change_all_paper_button"):
            self.change_all_paper_button.setEnabled(enabled and self.mask_mode == "reference_colour")
        if hasattr(self, "change_all_ink_button"):
            self.change_all_ink_button.setEnabled(enabled and self.mask_mode == "reference_colour")
        if hasattr(self, "generate_all_masks_button"):
            self.generate_all_masks_button.setEnabled(enabled)
        if hasattr(self, "mask_generate_buttons"):
            for button in self.mask_generate_buttons.values():
                button.setEnabled(enabled)
        if hasattr(self, "mask_remove_buttons"):
            for button in self.mask_remove_buttons.values():
                button.setEnabled(enabled)
        if hasattr(self, "mask_preview_labels"):
            for label in self.mask_preview_labels.values():
                label.setEnabled(enabled)
        if hasattr(self, "mask_original_labels"):
            for label in self.mask_original_labels.values():
                label.setEnabled(enabled)

        self._update_mask_file_controls()
        if hasattr(self, "alignment_preview"):
            self._display_alignment_image()
        self._refresh_disabled_tooltips()

    def _create_mask_tab(self):
        """Create the tab for preparing and previewing paper and ink masks."""
        page = QWidget()
        layout, content = self._split_tab(page)

        settings_group = QGroupBox("Mask settings")
        settings_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(6)
        settings_layout.setContentsMargins(8, 8, 8, 8)

        self.use_mask_checkbox = QCheckBox("Use mask")
        self.use_mask_checkbox.setToolTip("Enable mask generation and allow masks to be used in the analysis.")
        self.use_mask_checkbox.setChecked(False)
        self.use_mask_checkbox.toggled.connect(lambda checked: self._set_mask_controls_enabled(checked))
        settings_layout.addWidget(self.use_mask_checkbox)

        self._add_options_heading(settings_layout, "Threshold type")
        self.mask_mode_group = QButtonGroup(page)
        self.luminance_radio = QRadioButton("Luminance")
        self.luminance_radio.setToolTip("Separate ink and paper using an image brightness threshold.")
        self.colour_radio = QRadioButton("Reference colour")
        self.colour_radio.setToolTip("Separate ink and paper using their selected reference colours.")
        self.luminance_radio.setChecked(True)
        self.mask_mode_group.addButton(self.luminance_radio)
        self.mask_mode_group.addButton(self.colour_radio)
        self.luminance_radio.toggled.connect(lambda checked: self._set_mask_mode("luminance") if checked else None)
        self.colour_radio.toggled.connect(lambda checked: self._set_mask_mode("reference_colour") if checked else None)
        settings_layout.addWidget(self.luminance_radio)
        settings_layout.addWidget(self.colour_radio)

        self._add_options_heading(settings_layout, "Luminance")

        self.mask_threshold_slider = QSlider(Qt.Horizontal)
        self.mask_threshold_slider.setToolTip(
            "Set the brightness threshold for ink: pixels at or below it are classified as ink."
        )
        self.mask_threshold_slider.setRange(0, 255)
        self.mask_threshold_slider.setValue(80)
        self.mask_threshold_slider.setFixedHeight(18)
        self.mask_threshold_label = QLabel("Luminance threshold: 80")
        self.mask_threshold_slider.setEnabled(True)
        self.mask_threshold_slider.valueChanged.connect(
            lambda value: self.mask_threshold_label.setText(f"Luminance threshold: {value}")
        )
        settings_layout.addWidget(self.mask_threshold_label)
        settings_layout.addLayout(self._slider_with_editor(self.mask_threshold_slider, 80))

        self._add_options_heading(settings_layout, "Reference colour")
        reference_colour_row_index = settings_layout.count()
        self.reference_weight_slider = QSlider(Qt.Horizontal)
        self.reference_weight_slider.setToolTip(
            "Adjust the balance between paper and ink reference colours; higher values favour paper."
        )
        self.reference_weight_slider.setRange(0, 100)
        self.reference_weight_slider.setValue(50)
        self.reference_weight_slider.setEnabled(False)
        self.reference_weight_slider.setFixedHeight(18)
        self.reference_weight_label = QLabel("Reference weight: 0.50")
        self.reference_weight_slider.valueChanged.connect(
            lambda value: self.reference_weight_label.setText(f"Reference weight: {value / 100.0:.2f}")
        )
        settings_layout.addWidget(self.reference_weight_label)
        settings_layout.addLayout(self._slider_with_editor(self.reference_weight_slider, 50, divisor=100))

        self._add_options_heading(settings_layout, "Border")
        self.mask_border_slider = QSlider(Qt.Horizontal)
        self.mask_border_slider.setToolTip("Set how many pixels to exclude from mask boundaries.")
        self.mask_border_slider.setRange(0, 20)
        self.mask_border_slider.setValue(4)
        self.mask_border_slider.setFixedHeight(18)
        self.mask_border_label = QLabel("Mask border: 4 px")
        self.mask_border_slider.valueChanged.connect(
            lambda value: self.mask_border_label.setText(f"Mask border: {value} px")
        )
        settings_layout.addWidget(self.mask_border_label)
        settings_layout.addLayout(self._slider_with_editor(self.mask_border_slider, 4))

        layout.addWidget(settings_group)

        self.mask_preview_labels = {}
        self.mask_original_labels = {}
        self.paper_colour_buttons = {}
        self.ink_colour_buttons = {}
        self.mask_generate_buttons = {}
        self.mask_remove_buttons = {}
        self.mask_import_buttons = {}
        self.mask_export_buttons = {}
        self.mask_export_path_buttons = {}
        self.mask_export_path_fields = {}
        self.mask_export_paths = {}
        self.mask_image_cards = {}
        self.mask_card_titles = {}
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignTop)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for col, key in enumerate(["B1", "B2", "A"]):
            card = self._make_image_card()
            card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(3)

            title = self._make_card_title(self.image_key_labels[key])
            self.mask_card_titles[key] = title
            card_layout.addWidget(title)

            original_preview = ZoomableLabel(f"Original {key}")
            original_preview.setAlignment(Qt.AlignCenter)
            original_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            original_preview.setStyleSheet("border: 1px solid #888; background: #262626;")
            card_layout.addWidget(original_preview)

            mask_preview = QLabel(f"Mask {key}")
            mask_preview.setAlignment(Qt.AlignCenter)
            mask_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            mask_preview.setStyleSheet("border: 1px solid #888; background: #262626;")
            card_layout.addWidget(mask_preview)

            colour_row = QHBoxLayout()
            paper_button = QPushButton("Paper colour")
            paper_button.setToolTip(f"Choose the paper reference colour for {self.image_key_labels[key]}.")
            paper_button.setMinimumWidth(90)
            paper_button.clicked.connect(lambda checked, value=key: self._set_mask_reference_colour(value, "paper"))
            ink_button = QPushButton("Ink colour")
            ink_button.setToolTip(f"Choose the ink reference colour for {self.image_key_labels[key]}.")
            ink_button.setMinimumWidth(90)
            ink_button.clicked.connect(lambda checked, value=key: self._set_mask_reference_colour(value, "ink"))
            colour_row.addWidget(paper_button)
            colour_row.addWidget(ink_button)
            card_layout.addLayout(colour_row)

            button = QPushButton("Generate mask")
            button.setToolTip(f"Generate paper and ink masks for {self.image_key_labels[key]} using the current settings.")
            button.setMinimumWidth(120)
            button.clicked.connect(lambda checked, value=key: self._generate_mask_for_key(value))
            remove_button = QPushButton("Remove mask")
            remove_button.setToolTip(f"Remove the generated masks for {self.image_key_labels[key]}.")
            remove_button.setMinimumWidth(120)
            remove_button.clicked.connect(lambda checked, value=key: self._remove_mask_for_key(value))
            mask_row = QHBoxLayout()
            mask_row.addWidget(button)
            mask_row.addWidget(remove_button)
            card_layout.addLayout(mask_row)

            file_row = QHBoxLayout()
            import_button = QPushButton("Import mask")
            import_button.setToolTip(f"Import a colour mask for {self.image_key_labels[key]}.")
            import_button.clicked.connect(lambda checked, value=key: self._import_mask_for_key(value))
            export_button = QPushButton("Export mask")
            export_button.setToolTip(f"Export the mask for {self.image_key_labels[key]} as a PNG image.")
            export_button.clicked.connect(lambda checked, value=key: self._export_mask_for_key(value))
            file_row.addWidget(import_button)
            file_row.addWidget(export_button)
            card_layout.addLayout(file_row)

            export_path_button = QPushButton("🗀")
            export_path_button.setStyleSheet("font-weight: bold;")
            export_path_button.setFixedWidth(32)
            export_path_button.setToolTip(f"Choose the mask export folder and filename for {self.image_key_labels[key]}.")
            export_path_button.clicked.connect(lambda checked, value=key: self._set_mask_export_path_for_key(value))
            export_path_field = QLineEdit()
            export_path_field.setReadOnly(True)
            export_path_field.setPlaceholderText("Export path set when image is loaded")
            export_path_field.setToolTip(f"Current mask export path for {self.image_key_labels[key]}.")
            export_path_field.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            path_row = QHBoxLayout()
            path_row.addWidget(export_path_field)
            path_row.addWidget(export_path_button)
            card_layout.addLayout(path_row)

            self.mask_original_labels[key] = original_preview
            self.mask_preview_labels[key] = mask_preview
            self.paper_colour_buttons[key] = paper_button
            self.ink_colour_buttons[key] = ink_button
            self.mask_generate_buttons[key] = button
            self.mask_remove_buttons[key] = remove_button
            self.mask_import_buttons[key] = import_button
            self.mask_export_buttons[key] = export_button
            self.mask_export_path_buttons[key] = export_path_button
            self.mask_export_path_fields[key] = export_path_field
            self.mask_export_paths[key] = ""
            self.mask_image_cards[key] = card
            grid.addWidget(card, 0, col)

        self.change_all_paper_button = QPushButton("Set paper colours")
        self.change_all_paper_button.setToolTip("Set the same paper reference colour for all selected images.")
        self.change_all_paper_button.setMinimumWidth(180)
        self.change_all_paper_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.change_all_paper_button.clicked.connect(lambda: self._change_all_reference_colours("paper"))
        self.change_all_ink_button = QPushButton("Set ink colours")
        self.change_all_ink_button.setToolTip("Set the same ink reference colour for all selected images.")
        self.change_all_ink_button.setMinimumWidth(180)
        self.change_all_ink_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.change_all_ink_button.clicked.connect(lambda: self._change_all_reference_colours("ink"))

        self.generate_all_masks_button = QPushButton("Generate masks")
        self.generate_all_masks_button.setToolTip("Generate paper and ink masks for all loaded images.")
        self.generate_all_masks_button.setMinimumWidth(180)
        self.generate_all_masks_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.generate_all_masks_button.clicked.connect(self._generate_mask_for_all)

        self.export_all_masks_button = QPushButton("Export masks")
        self.export_all_masks_button.setToolTip("Export each mask to its configured path, confirming before overwriting.")
        self.export_all_masks_button.clicked.connect(self._export_all_masks)
        self.auto_import_masks_button = QPushButton("Import masks")
        self.auto_import_masks_button.setToolTip("Import masks found at the configured export paths for all images.")
        self.auto_import_masks_button.clicked.connect(self._auto_import_masks)

        self.cancel_mask_batch_button = QPushButton("Cancel")
        self.cancel_mask_batch_button.setToolTip("Stop after the image currently being processed.")
        self.cancel_mask_batch_button.setStyleSheet(CANCEL_BUTTON_STYLE)
        self.cancel_mask_batch_button.clicked.connect(self._cancel_mask_batch)
        self.cancel_mask_batch_button.hide()

        self.mask_progress_bar = QProgressBar()
        self.mask_progress_bar.setRange(0, 100)
        self.mask_progress_bar.setValue(0)
        self.mask_progress_label = QLabel("")
        self.mask_progress_label.setWordWrap(True)

        settings_layout.insertLayout(
            reference_colour_row_index, self._button_row(self.change_all_paper_button, self.change_all_ink_button)
        )
        layout.addLayout(self._button_row(self.export_all_masks_button, self.auto_import_masks_button))
        layout.addLayout(self._button_row(self.generate_all_masks_button, self.cancel_mask_batch_button))
        layout.addWidget(self.mask_progress_bar)
        layout.addWidget(self.mask_progress_label)

        content.addLayout(grid)
        content.addStretch(1)
        self._update_reference_colour_buttons()
        self._update_mask_generation_button_text()
        self._set_mask_controls_enabled(self.use_mask_checkbox.isChecked())
        self._set_mask_mode("luminance")
        return page

    def _add_options_heading(self, layout, text):
        """Add a bold section title to the analysis options."""
        heading = QLabel(text)
        heading.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(heading)

    def _make_reset_button(self, default_text):
        """Create the flat ↺ glyph used to reset a value to its default."""
        button = QToolButton()
        button.setText("\u21ba")
        button.setAutoRaise(True)
        button.setCursor(Qt.PointingHandCursor)
        button.setToolTip(f"Reset to default ({default_text})")
        button.setStyleSheet(
            "QToolButton { border: none; background: transparent; font-size: 16px; font-weight: bold;"
            " color: palette(highlight); }"
            "QToolButton:disabled { color: #7a7a7a; }"
        )
        return button

    def _slider_with_editor(self, slider, default, divisor=1):
        """Return a row holding the slider, a typed-value box and a reset glyph.

        The slider stays the source of truth; divisor maps its integer value to the displayed one.
        The editor and reset glyph follow the slider's value and enabled state.
        """
        decimals = len(str(divisor)) - 1
        fmt = lambda value: f"{value / divisor:.{decimals}f}"
        line_edit = QLineEdit(fmt(slider.value()))
        line_edit.setMaximumWidth(56)
        line_edit.setToolTip("Or type the value directly.")
        button = self._make_reset_button(fmt(default))

        def sync():
            line_edit.blockSignals(True)
            line_edit.setText(fmt(slider.value()))
            line_edit.blockSignals(False)
            line_edit.setEnabled(slider.isEnabled())
            button.setEnabled(slider.isEnabled() and slider.value() != default)

        def from_input():
            try:
                value = round(float(line_edit.text()) * divisor)
            except ValueError:
                sync()
                return
            slider.setValue(max(slider.minimum(), min(slider.maximum(), value)))
            sync()

        class EnabledWatcher(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.EnabledChange:
                    sync()
                return False

        watcher = EnabledWatcher(slider)
        slider.installEventFilter(watcher)
        slider.valueChanged.connect(lambda _value: sync())
        line_edit.editingFinished.connect(from_input)
        button.clicked.connect(lambda: slider.setValue(default))
        slider.sync_editor = sync
        sync()

        row = QHBoxLayout()
        row.addWidget(slider)
        row.addWidget(line_edit)
        row.addWidget(button)
        return row

    def _add_scale_reset_button(self, row, attr, slider, line_edit):
        """Add a ↺ glyph that resets an analysis scale; it is greyed out while the value is the default."""
        default = getattr(analysis_options(), attr)
        button = self._make_reset_button(f"{default:g}")

        def on_click():
            line_edit.setText(f"{default:g}")
            line_edit.editingFinished.emit()

        button.clicked.connect(on_click)
        slider.valueChanged.connect(lambda _value: self._update_scale_reset_button(attr))
        line_edit.editingFinished.connect(lambda: self._update_scale_reset_button(attr))
        row.addWidget(button)
        if not hasattr(self, "scale_reset_buttons"):
            self.scale_reset_buttons = {}
        self.scale_reset_buttons[attr] = (button, default)
        self._update_scale_reset_button(attr)

    def _update_scale_reset_button(self, attr):
        button, default = self.scale_reset_buttons[attr]
        button.setEnabled(getattr(self.analysis_options, attr) != default)

    def _refresh_analysis_options_state(self):
        """Keep the analysis option controls and their enabled states in sync with the stored settings."""
        if not hasattr(self, "analysis_options"):
            return

        options = self.analysis_options
        widgets = [
            ("rgb_comparison_checkbox", options.rgb_comparison),
            ("lab_comparison_checkbox", options.lab_comparison),
            ("use_masks_checkbox", options.use_masks),
            ("intersection_mask_checkbox", options.intersection_mask),
            ("mean_comparison_checkbox", options.mean_comparison),
            ("per_pixel_comparison_checkbox", options.per_pixel_comparison),
        ]
        for attr, value in widgets:
            if hasattr(self, attr):
                widget = getattr(self, attr)
                widget.blockSignals(True)
                widget.setChecked(value)
                widget.blockSignals(False)

        if hasattr(self, "use_masks_checkbox"):
            self.use_masks_checkbox.setEnabled(self.use_mask_checkbox.isChecked())

        if hasattr(self, "srgb_scale_slider"):
            slider = self.srgb_scale_slider
            slider.blockSignals(True)
            slider.setValue(int(options.srgb_scale))
            slider.blockSignals(False)
            self.srgb_scale_input.blockSignals(True)
            self.srgb_scale_input.setText(f"{options.srgb_scale:.1f}")
            self.srgb_scale_input.blockSignals(False)
            self.srgb_scale_label.setText(f"sRGB scale: {options.srgb_scale:.1f}")

        for attr, (label, title, slider, line_edit) in getattr(self, "ceilab_scale_controls", {}).items():
            value = getattr(options, attr)
            slider.blockSignals(True)
            slider.setValue(int(value))
            slider.blockSignals(False)
            line_edit.blockSignals(True)
            line_edit.setText(f"{value:.1f}")
            line_edit.blockSignals(False)
            label.setText(f"{title}: {value:.1f}")

        for attr in getattr(self, "scale_reset_buttons", {}):
            self._update_scale_reset_button(attr)

        if hasattr(self, "heatmap_grid_slider_x"):
            slider = self.heatmap_grid_slider_x
            slider.blockSignals(True)
            slider.setValue(options.heatmap_grid_size_x)
            slider.blockSignals(False)
            slider.sync_editor()

        if hasattr(self, "heatmap_grid_slider_y"):
            slider = self.heatmap_grid_slider_y
            slider.blockSignals(True)
            slider.setValue(options.heatmap_grid_size_y)
            slider.blockSignals(False)
            slider.sync_editor()

        intersection_enabled = (
            options.use_masks and self.image_mode != "single" and self._images_have_identical_resolution()
        )
        per_pixel_enabled = self.image_mode != "single" and not (options.use_masks and not options.intersection_mask)
        heatmap_enabled = options.per_pixel_comparison and per_pixel_enabled

        if hasattr(self, "intersection_mask_checkbox"):
            if not intersection_enabled:
                self.analysis_options.intersection_mask = False
                self.intersection_mask_checkbox.blockSignals(True)
                self.intersection_mask_checkbox.setChecked(False)
                self.intersection_mask_checkbox.blockSignals(False)
            self.intersection_mask_checkbox.setEnabled(intersection_enabled)

        if hasattr(self, "per_pixel_comparison_checkbox"):
            if not per_pixel_enabled:
                self.analysis_options.per_pixel_comparison = False
                self.per_pixel_comparison_checkbox.blockSignals(True)
                self.per_pixel_comparison_checkbox.setChecked(False)
                self.per_pixel_comparison_checkbox.blockSignals(False)
            self.per_pixel_comparison_checkbox.setEnabled(per_pixel_enabled)

        if hasattr(self, "heatmap_grid_slider_x"):
            if not heatmap_enabled:
                self.heatmap_grid_slider_x.setEnabled(False)
            else:
                self.heatmap_grid_slider_x.setEnabled(True)
        
        if hasattr(self, "heatmap_grid_slider_y"):
            if not heatmap_enabled:
                self.heatmap_grid_slider_y.setEnabled(False)
            else:
                self.heatmap_grid_slider_y.setEnabled(True)
        self._refresh_disabled_tooltips()

    def _create_analysis_options_tab(self):
        """Tab 4: configure and run the analysis, with the report beside the settings."""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        columns = QHBoxLayout()
        page_layout.addLayout(columns, 1)

        # The status text is still updated by the workflow but is not shown above the report.
        self.status_label = QLabel("Load the selected images for analysis.")

        self.report = QTextEdit()
        self.report.setReadOnly(True)
        report_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        report_font.setPointSize(10)
        self.report.setFont(report_font)
        # Ignore the document's preferred height so the report follows the window height.
        self.report.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.report.setMinimumHeight(0)
        self.report.setPlaceholderText("Analysis summary will appear here.")

        settings_page = QWidget()
        settings_page.setFixedWidth(250)
        columns.addWidget(settings_page)
        columns.addWidget(self.report, 1)
        layout = QVBoxLayout(settings_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop)

        self.analysis_settings_group = QGroupBox("Comparison settings")
        self.analysis_settings_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        options_layout = QVBoxLayout(self.analysis_settings_group)

        self.show_anaysis_options_checkbox = QCheckBox("Show analysis options")
        self.show_anaysis_options_checkbox.setToolTip("Include the selected analysis settings in the results report.")
        self.show_anaysis_options_checkbox.setChecked(self.analysis_options.show_analysis_options)
        self.show_anaysis_options_checkbox.toggled.connect(
            lambda checked: self.analysis_options.sync_from_ui(show_analysis_options=checked)
        )
        self._add_options_heading(options_layout, "Debug data")
        options_layout.addWidget(self.show_anaysis_options_checkbox)

        self.show_image_data_checkbox = QCheckBox("Show image data")
        self.show_image_data_checkbox.setToolTip("Include image dimensions and mask pixel counts in the results report.")
        self.show_image_data_checkbox.setChecked(self.analysis_options.show_image_data)
        self.show_image_data_checkbox.toggled.connect(
            lambda checked: self.analysis_options.sync_from_ui(show_image_data=checked)
        )
        options_layout.addWidget(self.show_image_data_checkbox)

        self.rgb_comparison_checkbox = QCheckBox("sRGB comparison")
        self.rgb_comparison_checkbox.setToolTip("Include sRGB colour comparisons in the results.")
        self.rgb_comparison_checkbox.setChecked(self.analysis_options.rgb_comparison)
        self.rgb_comparison_checkbox.toggled.connect(
            lambda checked: self.analysis_options.sync_from_ui(rgb_comparison=checked)
        )
        self._add_options_heading(options_layout, "Colour space")
        options_layout.addWidget(self.rgb_comparison_checkbox)

        self.lab_comparison_checkbox = QCheckBox("CEILAB comparison")
        self.lab_comparison_checkbox.setToolTip("Include CEILAB colour comparisons in the results.")
        self.lab_comparison_checkbox.setChecked(self.analysis_options.lab_comparison)
        self.lab_comparison_checkbox.toggled.connect(
            lambda checked: self.analysis_options.sync_from_ui(lab_comparison=checked)
        )
        options_layout.addWidget(self.lab_comparison_checkbox)

        self.srgb_scale_label = QLabel(f"sRGB scale: {self.analysis_options.srgb_scale:.1f}")
        self.srgb_scale_label.setToolTip("Set the scale for sRGB values (at 1.0, values range 0-1).")
        self._add_options_heading(options_layout, "Scales")
        options_layout.addWidget(self.srgb_scale_label)
        
        scale_row_srgb = QHBoxLayout()
        self.srgb_scale_slider = QSlider(Qt.Horizontal)
        self.srgb_scale_slider.setToolTip("Adjust the sRGB scale factor.")
        self.srgb_scale_slider.setRange(1, 1000)
        self.srgb_scale_slider.setValue(int(self.analysis_options.srgb_scale))
        self.srgb_scale_slider.setFixedHeight(18)
        self.srgb_scale_input = QLineEdit()
        self.srgb_scale_input.setPlaceholderText("255.0")
        self.srgb_scale_input.setMaximumWidth(56)
        self.srgb_scale_input.setText(str(self.analysis_options.srgb_scale))
        self.srgb_scale_input.setToolTip("Or type the sRGB scale value directly.")
        
        def update_srgb_scale_from_slider(value):
            self.analysis_options.srgb_scale = float(value)
            self.srgb_scale_input.blockSignals(True)
            self.srgb_scale_input.setText(f"{value:.1f}")
            self.srgb_scale_input.blockSignals(False)
            self.srgb_scale_label.setText(f"sRGB scale: {value:.1f}")
        
        def update_srgb_scale_from_input():
            try:
                value = float(self.srgb_scale_input.text())
                value = max(1.0, min(1000.0, value))
                self.srgb_scale_slider.blockSignals(True)
                self.srgb_scale_slider.setValue(int(value))
                self.srgb_scale_slider.blockSignals(False)
                self.analysis_options.srgb_scale = value
                self.srgb_scale_label.setText(f"sRGB scale: {value:.1f}")
            except ValueError:
                self.srgb_scale_input.setText(str(int(self.srgb_scale_slider.value())))
        
        self.srgb_scale_slider.valueChanged.connect(update_srgb_scale_from_slider)
        self.srgb_scale_input.editingFinished.connect(update_srgb_scale_from_input)
        scale_row_srgb.addWidget(self.srgb_scale_slider)
        scale_row_srgb.addWidget(self.srgb_scale_input)
        self._add_scale_reset_button(scale_row_srgb, "srgb_scale", self.srgb_scale_slider, self.srgb_scale_input)
        options_layout.addLayout(scale_row_srgb)

        self.ceilab_scale_controls = {}
        for attr, title, tip, default in (
            ("ceilab_l_scale", "CEILAB L* scale",
             "L* is shown from 0 to this value (CIELAB L* nominally spans 0-100).", "100.0"),
            ("ceilab_ab_scale", "CEILAB a*/b* scale",
             "a* and b* are shown from minus to plus this value (commonly clamped to -128..127).", "128.0"),
        ):
            label = QLabel(f"{title}: {getattr(self.analysis_options, attr):.1f}")
            label.setToolTip(tip)
            options_layout.addWidget(label)

            row = QHBoxLayout()
            slider = QSlider(Qt.Horizontal)
            slider.setToolTip(f"Adjust the {title.lower()} factor.")
            slider.setRange(1, 1000)
            slider.setValue(int(getattr(self.analysis_options, attr)))
            slider.setFixedHeight(18)
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(default)
            line_edit.setMaximumWidth(56)
            line_edit.setText(str(getattr(self.analysis_options, attr)))
            line_edit.setToolTip(f"Or type the {title.lower()} value directly.")

            def from_slider(value, attr=attr, title=title, label=label, line_edit=line_edit):
                setattr(self.analysis_options, attr, float(value))
                line_edit.blockSignals(True)
                line_edit.setText(f"{value:.1f}")
                line_edit.blockSignals(False)
                label.setText(f"{title}: {value:.1f}")

            def from_input(attr=attr, title=title, label=label, slider=slider, line_edit=line_edit):
                try:
                    value = max(1.0, min(1000.0, float(line_edit.text())))
                except ValueError:
                    line_edit.setText(str(int(slider.value())))
                    return
                slider.blockSignals(True)
                slider.setValue(int(value))
                slider.blockSignals(False)
                setattr(self.analysis_options, attr, value)
                label.setText(f"{title}: {value:.1f}")

            slider.valueChanged.connect(from_slider)
            line_edit.editingFinished.connect(from_input)
            row.addWidget(slider)
            row.addWidget(line_edit)
            self._add_scale_reset_button(row, attr, slider, line_edit)
            options_layout.addLayout(row)
            self.ceilab_scale_controls[attr] = (label, title, slider, line_edit)

        self.use_masks_checkbox = QCheckBox("Use masks")
        self.use_masks_checkbox.setToolTip(
            "Compare paper and ink separately using their masks. Enable Use mask on the Mask tab first."
        )
        self.use_masks_checkbox.setChecked(self.analysis_options.use_masks)
        self.use_masks_checkbox.toggled.connect(
            lambda checked: (
                self.analysis_options.sync_from_ui(use_masks=checked),
                self._refresh_analysis_options_state(),
            )
        )
        self._add_options_heading(options_layout, "Masks")
        options_layout.addWidget(self.use_masks_checkbox)

        self.intersection_mask_checkbox = QCheckBox("Combined mask")
        self.intersection_mask_checkbox.setToolTip(
            "Compare only pixels included in the corresponding mask in all selected images. "
            "Requires all images to have the same resolution."
        )
        self.intersection_mask_checkbox.setChecked(self.analysis_options.intersection_mask)
        self.intersection_mask_checkbox.toggled.connect(
            lambda checked: (
                self.analysis_options.sync_from_ui(intersection_mask=checked),
                self._refresh_analysis_options_state(),
            )
        )
        options_layout.addWidget(self.intersection_mask_checkbox)

        self.mean_comparison_checkbox = QCheckBox("Mean comparison")
        self.mean_comparison_checkbox.setToolTip("Compare average colours and report their differences.")
        self.mean_comparison_checkbox.setChecked(self.analysis_options.mean_comparison)
        self.mean_comparison_checkbox.toggled.connect(
            lambda checked: self.analysis_options.sync_from_ui(mean_comparison=checked)
        )
        self._add_options_heading(options_layout, "Micro / macro analysis")
        options_layout.addWidget(self.mean_comparison_checkbox)

        self.per_pixel_comparison_checkbox = QCheckBox("Per-pixel comparison")
        self.per_pixel_comparison_checkbox.setToolTip("Per-pixel comparison and heatmaps are not yet available.")
        self.per_pixel_comparison_checkbox.setChecked(self.analysis_options.per_pixel_comparison)
        self.per_pixel_comparison_checkbox.toggled.connect(
            lambda checked: (
                self.analysis_options.sync_from_ui(per_pixel_comparison=checked),
                self._refresh_analysis_options_state(),
            )
        )
        options_layout.addWidget(self.per_pixel_comparison_checkbox)

        self.heatmap_grid_slider_x = QSlider(Qt.Horizontal)
        self.heatmap_grid_slider_x.setToolTip(
            "Set the planned number of heatmap divisions across the image. Heatmaps are not yet available."
        )
        self.heatmap_grid_slider_x.setRange(2, 1000)
        self.heatmap_grid_slider_x.setValue(self.analysis_options.heatmap_grid_size_x)
        self.heatmap_grid_slider_x.valueChanged.connect(
            lambda value: self.analysis_options.sync_from_ui(heatmap_grid_size_x=value)
        )

        heatmap_label_x = QLabel(f"Heatmap grid divisions: {self.analysis_options.heatmap_grid_size_x}")
        self.heatmap_grid_slider_x.valueChanged.connect(
            lambda value: heatmap_label_x.setText(f"Heatmap x divisions: {value}")
        )
        options_layout.addWidget(heatmap_label_x)
        options_layout.addLayout(self._slider_with_editor(
            self.heatmap_grid_slider_x, analysis_options().heatmap_grid_size_x))

        self.heatmap_grid_slider_y = QSlider(Qt.Horizontal)
        self.heatmap_grid_slider_y.setToolTip(
            "Set the planned number of heatmap divisions down the image. Heatmaps are not yet available."
        )
        self.heatmap_grid_slider_y.setRange(2, 1000)
        self.heatmap_grid_slider_y.setValue(self.analysis_options.heatmap_grid_size_y)
        self.heatmap_grid_slider_y.valueChanged.connect(
            lambda value: self.analysis_options.sync_from_ui(heatmap_grid_size_y=value)
        )

        heatmap_label_y = QLabel(f"Heatmap grid divisions: {self.analysis_options.heatmap_grid_size_y}")
        self.heatmap_grid_slider_y.valueChanged.connect(
            lambda value: heatmap_label_y.setText(f"Heatmap y divisions: {value}")
        )
        options_layout.addWidget(heatmap_label_y)
        options_layout.addLayout(self._slider_with_editor(
            self.heatmap_grid_slider_y, analysis_options().heatmap_grid_size_y))

        layout.addWidget(self.analysis_settings_group)
        self.analyze_button = QPushButton("Analyse")
        self.analyze_button.setToolTip("Run the selected analysis on the loaded images.")
        self.analyze_button.clicked.connect(self.analyze_images)
        self.cancel_analysis_button = QPushButton("Cancel analysis")
        self.cancel_analysis_button.setToolTip(
            "Cancel analysis between calculation steps and keep the previous report."
        )
        self.cancel_analysis_button.setStyleSheet(CANCEL_BUTTON_STYLE)
        self.cancel_analysis_button.clicked.connect(self.cancel_analysis)
        self.cancel_analysis_button.hide()
        layout.addLayout(self._button_row(self.analyze_button, self.cancel_analysis_button))
        self.analysis_progress_bar = QProgressBar()
        self.analysis_progress_bar.setRange(0, 100)
        self.analysis_progress_bar.setValue(0)
        layout.addWidget(self.analysis_progress_bar)
        self.analysis_progress_label = QLabel("Ready to analyse.")
        self.analysis_progress_label.setWordWrap(True)
        layout.addWidget(self.analysis_progress_label)
        for checkbox in self.analysis_settings_group.findChildren(QCheckBox):
            checkbox.toggled.connect(self._reset_analysis_progress)
        for line_edit in self.analysis_settings_group.findChildren(QLineEdit):
            line_edit.editingFinished.connect(self._reset_analysis_progress)
        for slider in self.analysis_settings_group.findChildren(QSlider):
            slider.valueChanged.connect(self._reset_analysis_progress)
        for slider in (self.mask_threshold_slider, self.reference_weight_slider, self.mask_border_slider):
            slider.valueChanged.connect(self._reset_analysis_progress)
        self._refresh_analysis_options_state()
        return page

    def _create_image_results_tab(self):
        """Tab 5: reserve a place for future image results."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)
        layout.addWidget(QLabel("Image results are not available yet."))
        return page

    def dict_to_string(self, d, indent=" "*4, section_break="", _level=0):
        """Convert a nested dict to an indented string.

        Keys with dict values become headers ("key:") with their children
        indented beneath them. Any other value is written inline ("key: value").
        """
        pad = indent * _level
        blocks = []
        for key, value in d.items():
            if isinstance(value, dict):
                block = f"{pad}{key}:"
                if value:
                    block += "\n" + self.dict_to_string(value, indent, section_break, _level + 1)
            else:
                block = f"{pad}{key}: {value}"
            blocks.append(block)

        separator = f"\n{section_break}\n" if _level == 0 else "\n"
        return separator.join(blocks)

    def analyze_images(self):
        """Start the comparison for the loaded images."""
        if self._analysis_thread is not None or not self.analyze_button.isEnabled():
            return
        if any(self.images[key].pixels.get("rgb") is None for key in self._active_image_keys()):
            QMessageBox.warning(self, "Missing input", "Load the selected images before running the analysis.")
            return

        self._analysis_tab_states = {
            self.tab_widget.indexOf(tab): self.tab_widget.isTabEnabled(self.tab_widget.indexOf(tab))
            for tab in (self.load_tab, self.mask_tab, self.check_alignment_tab)
        }
        for index in self._analysis_tab_states:
            self.tab_widget.setTabEnabled(index, False)
        self.analysis_settings_group.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.cancel_analysis_button.setEnabled(True)
        self.cancel_analysis_button.setText("Cancel analysis")
        self.cancel_analysis_button.show()
        self.analysis_progress_label.setText("Starting analysis...")
        self.analysis_progress_bar.setValue(0)
        self.status_label.setText("Analysis in progress. The previous report remains visible until it completes.")

        images = {key: self.images[key] for key in self._active_image_keys()}
        self._analysis_thread = AnalysisThread(images, self.analysis_options, self, mode=self.image_mode)
        self._analysis_thread.progress.connect(self._analysis_progress)
        self._analysis_thread.finished.connect(self._analysis_finished)
        self._analysis_thread.start()
        self._refresh_disabled_tooltips()

    def cancel_analysis(self):
        """Request cooperative cancellation of the current analysis."""
        if self._analysis_thread is None:
            return
        self._analysis_thread.cancel_requested.set()
        self.cancel_analysis_button.setEnabled(False)
        self.cancel_analysis_button.setText("Cancelling...")
        self.analysis_progress_label.setText("Cancelling analysis...")
        self._refresh_disabled_tooltips()

    @Slot(int, int, str)
    def _analysis_progress(self, task, total, description):
        """Display the current analysis stage and estimated midpoint progress."""
        if self._analysis_thread is not None and self._analysis_thread.cancel_requested.is_set():
            return
        self.analysis_progress_bar.setValue(((2 * task - 1) * 100 + total) // (2 * total))
        suffix = " (closing when finished)" if self._close_after_analysis else ""
        self.analysis_progress_label.setText(f"Task {task}/{total}: {description}{suffix}")

    def _analysis_finished(self):
        """Publish successful results and restore controls after the worker finishes."""
        thread = self._analysis_thread
        self._analysis_thread = None
        self.analysis_settings_group.setEnabled(True)
        self.analyze_button.setEnabled(True)
        self.cancel_analysis_button.hide()
        for index, enabled in self._analysis_tab_states.items():
            self.tab_widget.setTabEnabled(index, enabled)
        try:
            if thread.error is not None:
                raise RuntimeError(thread.error)
            if thread.cancel_requested.is_set():
                self.status_label.setText("Analysis cancelled. The previous report is unchanged.")
                self.analysis_progress_label.setText("Analysis cancelled.")
                return
            summary_lines = self.dict_to_string(thread.result, indent=" "*4, section_break="\n"+("-" * 64)+"\n")
            self.report.setPlainText(summary_lines + "\n")
            if self.image_mode == "single":
                self.status_label.setText("Analysis complete. The report describes the image.")
            else:
                self.status_label.setText("Analysis complete. The report compares the before and after images.")
            self.analysis_progress_label.setText(f"Analysis complete ({thread.total_tasks}/{thread.total_tasks} tasks).")
            self.analysis_progress_bar.setValue(100)
            for tab in (self.image_results_tab,):
                self.tab_widget.setTabEnabled(self.tab_widget.indexOf(tab), True)
            if not self._close_after_analysis:
                self.tab_widget.setCurrentWidget(self.analysis_options_tab)
        except Exception as exc:
            self.status_label.setText("Analysis failed. The previous report is unchanged.")
            self.analysis_progress_label.setText(f"Analysis failed. {self.analysis_progress_label.text()}")
            QMessageBox.critical(self, "Analysis failed", f"Unable to analyze the images:\n{exc}")
        finally:
            self._refresh_disabled_tooltips()
            thread.deleteLater()
            if self._close_after_analysis:
                self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
