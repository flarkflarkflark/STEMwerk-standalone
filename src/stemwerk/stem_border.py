from __future__ import annotations

from typing import List, Optional, Union

from PySide6 import QtCore, QtGui, QtWidgets


class StemBorderWidget(QtWidgets.QWidget):
    def __init__(self, colors: Optional[List[Union[str, QtGui.QColor]]] = None, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._colors: List[QtGui.QColor] = []
        self.setFixedHeight(4)
        if colors:
            self.set_colors(colors)

    def set_colors(self, colors: List[Union[str, QtGui.QColor]]) -> None:
        self._colors = [QtGui.QColor(color) for color in colors]
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if not self._colors:
            return
        painter = QtGui.QPainter(self)
        rect = self.rect()
        segment_width = rect.width() / len(self._colors)
        for index, color in enumerate(self._colors):
            segment_rect = QtCore.QRectF(
                rect.left() + segment_width * index,
                rect.top(),
                segment_width,
                rect.height(),
            )
            painter.fillRect(segment_rect, color)
