"""The main window of the detection reviewer app."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import override

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QDoubleSpinBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
)

from vision.common.constants import DEFAULT_CONFIDENCE_THRESHOLD
from vision.reviewer.image import STATUS_COLORS, ImageView
from vision.reviewer.models import ReviewDetection, ReviewSession, ReviewStatus

logger = logging.getLogger(__name__)


class ReviewWindow(QMainWindow):
    """
    Loads in object detections from the ODLC pipeline and allows the user to
    accept/reject the detections before passing them back to the ODLC pipeline
    for final review.

    Attributes
    ----------
    session : ReviewSession
        The detections being reviewed.
    """

    def __init__(self, session: ReviewSession) -> None:
        super().__init__()
        self.session: ReviewSession = session
        self._current_image: str | None = None
        self._updating: bool = False
        self._saved: bool = False
        self._show_all_images: bool = False

        self.setWindowTitle(f"Detection Reviewer - {session.source}")
        self.resize(1600, 950)

        self.view: ImageView = ImageView(self)
        self.view.selection_changed.connect(self._on_view_selection)
        self.setCentralWidget(self.view)

        self.toolbar: QToolBar = self.addToolBar("Review")
        self.toolbar.setIconSize(QSize(16, 16))

        self._build_image_list()
        self._build_detection_list()
        self._build_actions()
        self._build_status_bar()

        self._populate_image_list()
        # Wait for the viewport to be laid out before selecting the first detection
        QTimer.singleShot(0, self._select_first_detection)

    def _build_image_list(self) -> None:
        """Create the left side list of the images being reviewed"""
        self.image_list: QListWidget = QListWidget(self)
        self.image_list.currentRowChanged.connect(self._on_image_changed)

        dock = QDockWidget("Images", self)
        dock.setWidget(self.image_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _listed_images(self) -> list[str]:
        """The images the list is currently showing."""
        if self._show_all_images:
            return self.session.all_images
        return self.session.images

    def _populate_image_list(self) -> None:
        """Fills the image list, keeping the current image selected if it is still listed."""
        images = self._listed_images()
        previous = self._current_image
        row = images.index(previous) if previous in images else 0

        # Selected row can move around, so reselect the correct one when the image
        # list is repopulated
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for image in images:
            item = QListWidgetItem(Path(image).name)
            item.setData(Qt.ItemDataRole.UserRole, image)
            item.setToolTip(image)
            self.image_list.addItem(item)
        self.image_list.setCurrentRow(row)
        self.image_list.blockSignals(False)

        self._refresh_image_labels()
        self._on_image_changed(row)

    def _set_show_all_images(self, show_all: bool) -> None:
        """Switches between listing every captured image and only the ones with detections."""
        self._show_all_images = show_all
        self._populate_image_list()

    def _build_detection_list(self) -> None:
        """Create the right side table of all detections for the current image"""
        self.table: QTableWidget = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(
            ["Category", "Confidence", "Size", "Status"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.table.itemDoubleClicked.connect(self._zoom_to_selected)

        dock = QDockWidget("Detections", self)
        dock.setWidget(self.table)
        dock.setMinimumWidth(340)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _add_to_toolbar(
        self, text: str, shortcut: str, slot: Callable[[], object]
    ) -> QAction:
        """Add an action to the toolbar"""
        action = QAction(text, self)
        action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        self.toolbar.addAction(action)
        return action

    def _build_actions(self) -> None:
        """Create the actions for the review window"""

        self._add_to_toolbar(
            "&Accept",
            "A",
            lambda: self._set_status(ReviewStatus.ACCEPTED),
        )
        self._add_to_toolbar(
            "&Reject",
            "R",
            lambda: self._set_status(ReviewStatus.REJECTED),
        )
        self._add_to_toolbar("Change &Category...", "C", self.change_category)
        self.toolbar.addSeparator()
        self._add_to_toolbar(
            "&Next Detection",
            "Tab",
            lambda: self._step_detection(1),
        )
        self._add_to_toolbar(
            "&Previous Detection",
            "Shift+Tab",
            lambda: self._step_detection(-1),
        )
        self.toolbar.addSeparator()
        self._add_to_toolbar("Next &Image", "]", lambda: self._step_image(1))
        self._add_to_toolbar("Previous I&mage", "[", lambda: self._step_image(-1))
        self.toolbar.addSeparator()
        self._add_to_toolbar("&Fit to Window", "F", self.view.fit_to_window)
        self._add_to_toolbar("Zoom to &Detection", "Z", self._zoom_to_selected)
        self._add_to_toolbar("Zoom &In", "+", lambda: self.view.zoom_by(1.25))
        self._add_to_toolbar("Zoom &Out", "-", lambda: self.view.zoom_by(0.8))
        all_images = QAction("Show All &Images", self)
        all_images.setCheckable(True)
        all_images.setChecked(False)
        all_images.setShortcut(QKeySequence("I"))
        all_images.setToolTip(
            "List every captured image, not just the ones with detections"
        )
        all_images.toggled.connect(self._set_show_all_images)
        self.toolbar.addAction(all_images)

        labels = QAction("Show &Labels", self)
        labels.setCheckable(True)
        labels.setChecked(True)
        labels.setShortcut(QKeySequence("L"))
        labels.toggled.connect(self.view.set_labels_visible)
        self.toolbar.addAction(labels)

        self.toolbar.addSeparator()
        self.toolbar.addWidget(QLabel(" Min confidence: "))
        self.confidence_filter: QDoubleSpinBox = QDoubleSpinBox(self)
        self.confidence_filter.setRange(0.0, 1.0)
        self.confidence_filter.setSingleStep(0.05)
        self.confidence_filter.setDecimals(2)
        self.confidence_filter.setValue(DEFAULT_CONFIDENCE_THRESHOLD)
        self.confidence_filter.valueChanged.connect(self._apply_filter)
        self.toolbar.addWidget(self.confidence_filter)

    def _build_status_bar(self) -> None:
        self.progress_label: QLabel = QLabel("", self)
        self.statusBar().addWidget(self.progress_label)
        self._refresh_progress()

    def _on_image_changed(self, row: int) -> None:
        """Callback when a new image is selected in the image list."""
        if not 0 <= row < self.image_list.count():
            return
        image: str = self.image_list.item(row).data(Qt.ItemDataRole.UserRole)
        self._current_image = image

        path = self.session.resolve_image(image)
        pixmap = QPixmap(str(path)) if path is not None else QPixmap()
        if pixmap.isNull():
            self.statusBar().showMessage(f"Could not load {image}", 8000)

        # Update detection table and view
        detections = self.session.for_image(image)
        self.view.set_image(pixmap, detections)
        self._populate_table(detections)

    def _step_image(self, delta: int) -> None:
        """Go to the next or previous image in the list."""
        row = self.image_list.currentRow() + delta
        if 0 <= row < self.image_list.count():
            self.image_list.setCurrentRow(row)

    def _populate_table(self, detections: list[ReviewDetection]) -> None:
        """Update detections table with new detections"""
        self._updating = True
        self.table.setRowCount(len(detections))
        for row, detection in enumerate(detections):
            values = (
                detection.category,
                f"{detection.confidence:.3f}",
                f"{detection.width:.0f} x {detection.height:.0f}",
                detection.status.value,
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setForeground(STATUS_COLORS[detection.status])
                if column == 0:
                    cell.setData(Qt.ItemDataRole.UserRole, detection)
                self.table.setItem(row, column, cell)
        self.table.resizeColumnsToContents()
        self._updating = False
        self._apply_filter()

    def _row_detections(self) -> list[ReviewDetection]:
        """Every detection in the table, in display order."""
        detections: list[ReviewDetection] = []
        for row in range(self.table.rowCount()):
            cell = self.table.item(row, 0)
            if cell is not None:
                detections.append(cell.data(Qt.ItemDataRole.UserRole))
        return detections

    def _selected_detection(self) -> ReviewDetection | None:
        """The detection selected in the view (which mirrors the table)."""
        return self.view.selected_detection

    def _on_view_selection(self, selected: ReviewDetection | None) -> None:
        """Callback when a detection is selected in the view."""
        if self._updating:
            return
        self._updating = True
        self.table.clearSelection()
        for row, detection in enumerate(self._row_detections()):
            if detection is selected:
                self.table.selectRow(row)
                self.table.setCurrentCell(row, 0)
        self._updating = False

    def _on_table_selection(self) -> None:
        """Callback when a detection is selected in the table."""
        if self._updating:
            return
        self._updating = True
        rows = next(iter(self.table.selectedIndexes()))
        selected_detection = self._row_detections()[rows.row()]
        self.view.select(selected_detection)
        self._updating = False

    def _select_first_detection(self) -> None:
        """Selects and zooms to the first detection of the first image."""
        self._step_detection(1)

    def _filtered_detections(self, image: str) -> list[ReviewDetection]:
        """The detections of an image passing the confidence filter, in table order."""
        minimum = self.confidence_filter.value()
        return [
            detection
            for detection in self.session.for_image(image)
            if detection.confidence >= minimum
        ]

    def _show_detection(self, detection: ReviewDetection) -> None:
        """Selects a detection and zooms the view to it."""
        self.view.select(detection)
        self.view.zoom_to_detection(detection)

    def _step_detection(self, delta: int) -> None:
        """
        Steps to the next/previous detection in the table.
        Jumps to the next/previous image if none are left.
        """
        detections = self._filtered_detections(self._current_image or "")
        selected = self._selected_detection()
        if selected is not None and selected in detections:
            # Go to index + delta
            index = detections.index(selected) + delta
        else:
            # No selection or out of bounds, start from the beginning or end
            index = 0 if delta > 0 else len(detections) - 1

        if 0 <= index < len(detections):
            self._show_detection(detections[index])
            return
        self._step_image_detection(delta)

    def _step_image_detection(self, delta: int) -> None:
        """
        Moves to the nearest image in that direction that still has a detection
        to show, and selects its first (or last, when stepping backwards) one.
        Offers to save the review once the last detection has been stepped past.
        """
        images = self._listed_images()
        row = self.image_list.currentRow()
        rows = range(row + 1, len(images)) if delta > 0 else range(row - 1, -1, -1)
        for next_row in rows:
            detections = self._filtered_detections(images[next_row])
            if not detections:
                continue
            self.image_list.setCurrentRow(next_row)
            self._show_detection(detections[0] if delta > 0 else detections[-1])
            return

        # Nothing left in this direction. Stepping off the end of the last image
        # means every detection has been seen, so offer to save.
        if delta > 0 and self._selected_detection() is not None:
            self._prompt_save()

    def _prompt_save(self) -> None:
        """Asks whether to save now that the last detection has been reviewed."""
        pending = self.session.counts()[ReviewStatus.PENDING]
        summary = f"{pending} detection(s) are still pending."
        choice = QMessageBox.question(
            self,
            "Review complete",
            f"That was the last detection. {summary}\n\nSave the review?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
        )
        if choice == QMessageBox.StandardButton.Save:
            self.save()

    def _set_status(self, status: ReviewStatus) -> None:
        """Applies a review decision to selected detection."""
        selected = self._selected_detection()
        if not selected:
            self.statusBar().showMessage("Select a detection first", 3000)
            return
        selected.status = status
        self._after_edit()
        self._step_detection(1)

    def change_category(self) -> None:
        """Switches the label for the selected detection."""
        selected = self._selected_detection()
        if not selected:
            return
        if selected.category == "person":
            selected.category = "tent"
        elif selected.category == "tent":
            selected.category = "person"
        self._after_edit()

    def _zoom_to_selected(self) -> None:
        """Zooms to the selected detection."""
        selected = self._selected_detection()
        if selected:
            self.view.zoom_to_detection(selected)

    def _apply_filter(self) -> None:
        """Applies the confidence filter to the detections."""
        minimum = self.confidence_filter.value()
        self.view.set_min_confidence(minimum)
        for row, detection in enumerate(self._row_detections()):
            self.table.setRowHidden(row, detection.confidence < minimum)

    def _after_edit(self) -> None:
        """Refreshes everything that shows detection state."""
        selected = self._selected_detection()
        self.view.refresh_items()
        self._populate_table(self.session.for_image(self._current_image or ""))
        self._on_view_selection(selected)
        self._refresh_image_labels()
        self._refresh_progress()

    def _refresh_image_labels(self) -> None:
        """Updates the image list labels with the number of reviewed detections."""
        for row in range(self.image_list.count()):
            item = self.image_list.item(row)
            image: str = item.data(Qt.ItemDataRole.UserRole)
            detections = self.session.for_image(image)
            reviewed = sum(
                1 for d in detections if d.status is not ReviewStatus.PENDING
            )
            item.setText(f"{Path(image).name}  [{reviewed}/{len(detections)}]")

    def _refresh_progress(self) -> None:
        counts = self.session.counts()
        summary = ", ".join(
            f"{counts[status]} {status.value}" for status in ReviewStatus
        )
        self.progress_label.setText(summary)

    def save(self) -> None:
        """Writes the accepted detections to the output JSON file and closes."""
        self.session.save()
        self._saved = True
        self.close()

    @override
    def closeEvent(self, event: QCloseEvent) -> None:
        """Saves the review and closes the window if confirmed"""
        # save() closes the window itself, so there is nothing left to ask about.
        if self._saved:
            super().closeEvent(event)
            return

        choice = QMessageBox.question(
            self,
            "Unsaved review",
            "Save the review before closing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        if choice == QMessageBox.StandardButton.Save:
            self.save()
        super().closeEvent(event)
