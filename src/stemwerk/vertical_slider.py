from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets


class VerticalStemSlider(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(int)

    def __init__(self, color: str, stem_name: str = "", parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._color = QtGui.QColor(color)
        self._track_color = QtGui.QColor("#1a1a1a")
        self._value = 100
        self._vu_level = 0.0
        self._stem_name = stem_name
        self.setMinimumSize(36, 120)
        self.setMouseTracking(True)

    def value(self) -> int:
        return self._value

    def setValue(self, value: int) -> None:
        clamped = max(0, min(100, int(value)))
        if clamped == self._value:
            return
        self._value = clamped
        self.valueChanged.emit(self._value)
        self.update()

    def set_color(self, color: str) -> None:
        self._color = QtGui.QColor(color)
        self.update()

    def set_vu_level(self, level: float) -> None:
        self._vu_level = max(0.0, min(1.0, level))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        rect = self.rect()
        padding = 6
        vu_width = 6
        track_width = 10
        gap = 4

        available_width = rect.width() - padding * 2
        track_x = rect.left() + padding + (available_width - track_width - vu_width - gap) / 2.0
        track_rect = QtCore.QRectF(
            track_x,
            rect.top() + padding,
            track_width,
            rect.height() - padding * 2,
        )
        vu_rect = QtCore.QRectF(
            track_rect.right() + gap,
            track_rect.top(),
            vu_width,
            track_rect.height(),
        )

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self._track_color)
        painter.drawRoundedRect(track_rect, 4, 4)

        fill_height = track_rect.height() * (self._value / 100.0)
        fill_rect = QtCore.QRectF(
            track_rect.left(),
            track_rect.bottom() - fill_height,
            track_rect.width(),
            fill_height,
        )
        painter.setBrush(self._color)
        painter.drawRoundedRect(fill_rect, 4, 4)

        handle_y = track_rect.bottom() - fill_height
        handle_rect = QtCore.QRectF(
            track_rect.left() - 3,
            handle_y - 2,
            track_rect.width() + 6,
            4,
        )
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.drawRoundedRect(handle_rect, 2, 2)

        painter.setBrush(self._track_color)
        painter.drawRect(vu_rect)
        vu_height = vu_rect.height() * self._vu_level
        vu_fill = QtCore.QRectF(
            vu_rect.left(),
            vu_rect.bottom() - vu_height,
            vu_rect.width(),
            vu_height,
        )
        painter.setBrush(self._color)
        painter.drawRect(vu_fill)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._set_value_from_pos(event.position().y())

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._set_value_from_pos(event.position().y())

    def _set_value_from_pos(self, y: float) -> None:
        rect = self.rect()
        padding = 6
        height = rect.height() - padding * 2
        if height <= 0:
            return
        relative = max(0.0, min((rect.bottom() - padding) - y, height))
        value = int((relative / height) * 100)
        self.setValue(value)
