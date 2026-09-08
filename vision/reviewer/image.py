"""The zoomable image view used by the detection reviewer."""

from __future__ import annotations

from typing import ClassVar, override

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneHoverEvent,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QStyle,
    QStyleOptionGraphicsItem,
    QWidget,
)

from vision.reviewer.models import ReviewDetection, ReviewStatus

STATUS_COLORS: dict[ReviewStatus, QColor] = {
    ReviewStatus.PENDING: QColor("#ffd23f"),
    ReviewStatus.ACCEPTED: QColor("#3ddc84"),
    ReviewStatus.REJECTED: QColor("#ff5555"),
}

MAX_ZOOM: float = 40.0
MIN_ZOOM: float = 0.01
ZOOM_STEP: float = 1.25
# Above this zoom, pixels are drawn unsmoothed in the image
NEAREST_NEIGHBOR_ZOOM: float = 3.0
# Horizontal padding, in screen pixels, around the text of a box's label
LABEL_PADDING: float = 3.0


class DetectionItem(QGraphicsRectItem):
    """
    One detection's bounding box, drawn in image pixel coordinates.
    Includes label with category/confidence text.

    Attributes
    ----------
    detection : ReviewDetection
        The detection this box represents.
    """

    def __init__(self, detection: ReviewDetection) -> None:
        x1, y1, x2, y2 = detection.bbox
        super().__init__(QRectF(x1, y1, x2 - x1, y2 - y1))
        self.detection: ReviewDetection = detection
        self._hovered: bool = False

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        # Draw smaller boxes on top of large ones so they stay clickable/visible
        self.setZValue(1.0 + 1.0 / (1.0 + detection.area))

        self._label_item: QGraphicsRectItem = QGraphicsRectItem(self)
        self._label_item.setFlag(  # Keep the label the same size
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
        )
        self._label_item.setPos(x1, y1)
        # No outline, already have bbox drawn, and make the label opaque
        self._label_item.setPen(QPen(Qt.PenStyle.NoPen))
        self._label_item.setBrush(QBrush(QColor(0, 0, 0, 170)))

        # Text is set later on refresh, but init the label now
        self._label: QGraphicsSimpleTextItem = QGraphicsSimpleTextItem(self._label_item)
        self._label.setPos(LABEL_PADDING, 0.0)

        self.refresh()

    def refresh(self) -> None:
        """Restyles the box to match the detection's current status and state."""
        color = STATUS_COLORS[self.detection.status]
        pen = QPen(color)
        pen.setCosmetic(True)
        pen.setWidthF(3.0 if self.isSelected() else 2.0 if self._hovered else 1.5)
        if self.detection.status is ReviewStatus.REJECTED:
            pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)

        # Fill bbox bg with slight color if it is the selected detection
        fill = QColor(color)
        fill.setAlpha(45 if self.isSelected() else 0)
        self.setBrush(QBrush(fill))

        # Add text to label and resize label accordingly
        self._label.setText(
            f"{self.detection.category} {self.detection.confidence:.0%}"
        )
        self._label.setBrush(QBrush(color))
        text_size = self._label.boundingRect().size()
        self._label_item.setRect(
            0.0,
            0.0,
            text_size.width() + 2.0 * LABEL_PADDING,
            text_size.height(),
        )

    def set_label_visible(self, visible: bool) -> None:
        """Shows or hides the category/confidence label."""
        self._label_item.setVisible(visible)

    @override
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """
        Qt will by default add a dashed outline if an object is selected, surpress
        that by making the default paint func think that the object is not selected
        """
        option.state &= ~QStyle.StateFlag.State_Selected
        super().paint(painter, option, widget)

    @override
    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        """Override hoverEnterEvent to enable _hovered for thicker outline"""
        self._hovered = True
        self.refresh()
        super().hoverEnterEvent(event)

    @override
    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        """Override hoverLeaveEvent to disable _hovered for thinner outline"""
        self._hovered = False
        self.refresh()
        super().hoverLeaveEvent(event)

    @override
    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: object):
        """
        Override itemChange to refresh on selection change
        This will cause the outline to reset back to normal after selection changes
        """
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.refresh()
        return super().itemChange(change, value)


class ImageView(QGraphicsView):
    """
    Displays one image with its detections drawn on top.

    Signals
    -------
    selection_changed : ReviewDetection | None
        Emitted with a newly selected box, or None when nothing is selected.
    detection_activated : ReviewDetection
        Emitted when a box is double clicked.
    """

    selection_changed: ClassVar[Signal] = Signal(ReviewDetection)
    detection_activated: ClassVar[Signal] = Signal(ReviewDetection)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene: QGraphicsScene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: QGraphicsPixmapItem = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._items: list[DetectionItem] = []
        self._panning: bool = False
        self._pan_origin: QPointF = QPointF()
        self._labels_visible: bool = True
        self._user_zoomed: bool = False

        # Enable AA
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        # Make it so zooming in/out zooms onto the mouse
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # Resize the viewport from the center
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # Disable dragging on the widget
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        # Add background color
        self.setBackgroundBrush(QBrush(QColor("#1b1b1b")))
        # Disable focus by tabbing, enable mouse move events
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)
        # Set callback for selection changes
        self._scene.selectionChanged.connect(self._emit_selection)

    def set_image(self, pixmap: QPixmap, detections: list[ReviewDetection]) -> None:
        """
        Shows an image and the boxes belonging to it

        Parameters
        ----------
        pixmap : QPixmap
            The image to display, if null clears the view
        detections : list[ReviewDetection]
            The detections to draw on top of the image
        """
        # Clear existing items
        for item in self._items:
            self._scene.removeItem(item)
        self._items.clear()

        self._user_zoomed = False
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

        # Add new detections
        for detection in detections:
            item = DetectionItem(detection)
            item.set_label_visible(self._labels_visible)
            self._scene.addItem(item)
            self._items.append(item)

        # Fit image to view size
        self.fit_to_window()

    def item_for(self, detection: ReviewDetection) -> DetectionItem | None:
        """Get the item for a detection"""
        return next((item for item in self._items if item.detection is detection), None)

    def refresh_items(self) -> None:
        """Refreshes every item"""
        for item in self._items:
            item.refresh()

    def set_labels_visible(self, visible: bool) -> None:
        """Shows or hides the label on every box"""
        self._labels_visible = visible
        for item in self._items:
            item.set_label_visible(visible)

    def set_min_confidence(self, minimum: float) -> None:
        """Hides boxes below the given confidence threshold"""
        for item in self._items:
            visible = item.detection.confidence >= minimum
            if not visible:
                item.setSelected(False)
            item.setVisible(visible)

    @property
    def selected_detection(self) -> ReviewDetection | None:
        """The detections whose boxes are currently selected"""
        return next((item.detection for item in self._items if item.isSelected()), None)

    def select(self, detection: ReviewDetection) -> None:
        """Selects the item for the given detection"""
        for item in self._items:
            item.setSelected(item.detection == detection)

    def _emit_selection(self) -> None:
        """Emits the selection_changed signal with the current selected detection"""
        self.selection_changed.emit(self.selected_detection)

    @property
    def zoom(self) -> float:
        """The current scale factor, in screen pixels per image pixel."""
        return self.transform().m11()

    def zoom_by(self, factor: float) -> None:
        """Multiplies the zoom, clamped to the limits."""
        self._user_zoomed = True
        target = min(max(self.zoom * factor, MIN_ZOOM), MAX_ZOOM)
        if self.zoom > 0.0:
            self.scale(target / self.zoom, target / self.zoom)
        self._after_zoom()

    def fit_to_window(self) -> None:
        """Scales the whole image to fit inside the viewport."""
        self._user_zoomed = False
        if self._pixmap_item.pixmap().isNull():
            self.resetTransform()
            self._after_zoom()
            return
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._after_zoom()

    def zoom_to_detection(self, detection: ReviewDetection) -> None:
        """Zooms into the given detection, with context around the detection."""
        item = self.item_for(detection)
        if item is None:
            return
        self._user_zoomed = True
        rect = item.rect()
        margin = max(rect.width(), rect.height()) * 0.75 + 24.0
        self.fitInView(
            rect.adjusted(-margin, -margin, margin, margin),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        # Clamp to MAX_ZOOM
        if self.zoom > MAX_ZOOM:
            self.zoom_by(MAX_ZOOM / self.zoom)
        self._after_zoom()

    def _after_zoom(self) -> None:
        """Updates pixel smoothing based on zoom level"""
        self._pixmap_item.setTransformationMode(
            Qt.TransformationMode.FastTransformation
            if self.zoom >= NEAREST_NEIGHBOR_ZOOM
            else Qt.TransformationMode.SmoothTransformation
        )

    @override
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Changes zoom level based on scroll direction/magnitude"""
        notches = event.angleDelta().y() / 120.0
        if notches == 0.0:
            return
        self.zoom_by(ZOOM_STEP**notches)
        event.accept()

    def _detection_at(self, event: QMouseEvent) -> DetectionItem | None:
        """Returns the item under the current cursor position."""
        item = self.itemAt(event.position().toPoint())
        while item is not None:
            if isinstance(item, DetectionItem):
                return item
            item = item.parentItem()
        return None

    @override
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle enabling dragging/panning the image around"""
        on_empty_space = self._detection_at(event) is None
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and on_empty_space
        ):
            self._scene.clearSelection()
            self._panning = True
            self._pan_origin = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    @override
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Translate mouse movement while holding down mouse to
        panning the image around
        """
        if self._panning:
            delta = event.position() - self._pan_origin
            self._pan_origin = event.position()
            horizontal = self.horizontalScrollBar()
            vertical = self.verticalScrollBar()
            horizontal.setValue(horizontal.value() - int(delta.x()))
            vertical.setValue(vertical.value() - int(delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    @override
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle disabling dragging/panning the image around"""
        if self._panning:
            self._panning = False
            self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    @override
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Zoom to the detection when double-clicking on it"""
        item = self._detection_at(event)
        if item is not None:
            item.setSelected(True)
            self.zoom_to_detection(item.detection)
            self.detection_activated.emit(item.detection)
            event.accept()
            return
        self.fit_to_window()
        event.accept()
