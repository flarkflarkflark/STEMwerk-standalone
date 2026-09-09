from __future__ import annotations

import math
import time
from typing import List, Optional, Union

from PySide6 import QtCore, QtGui, QtWidgets


class LogoWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._letters = ["S", "T", "E", "M", "w", "e", "r", "k"]
        self._stem_colors: List[QtGui.QColor] = []
        self._text_color = QtGui.QColor("#ffffff")
        self._start_time = time.monotonic()
        self.setFixedHeight(40)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.update)
        self._timer.start()

    def set_colors(self, stem_colors: List[Union[str, QtGui.QColor]], text_color: Union[str, QtGui.QColor]) -> None:
        self._stem_colors = [QtGui.QColor(color) for color in stem_colors]
        self._text_color = QtGui.QColor(text_color)
        self.update()

    def sizeHint(self) -> QtCore.QSize:
        metrics = QtGui.QFontMetrics(QtGui.QFont("Arial", 24, QtGui.QFont.Weight.Bold))
        width = sum(metrics.horizontalAdvance(letter) for letter in self._letters)
        return QtCore.QSize(width, 40)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        font = QtGui.QFont("Arial", 24, QtGui.QFont.Weight.Bold)
        painter.setFont(font)
        metrics = QtGui.QFontMetrics(font)

        widths = [metrics.horizontalAdvance(letter) for letter in self._letters]
        total_width = sum(widths)
        start_x = (self.width() - total_width) / 2.0

        amplitude = max(1, int(math.floor(font.pointSizeF() * 0.08)))
        t = time.monotonic() - self._start_time
        base_y = (self.height() - metrics.height()) / 2.0 + metrics.ascent()

        x = start_x
        for index, letter in enumerate(self._letters):
            y_offset = math.sin(t * 3 + index * 0.5) * amplitude
            if index < 4 and index < len(self._stem_colors):
                painter.setPen(self._stem_colors[index])
            else:
                painter.setPen(self._text_color)
            painter.drawText(QtCore.QPointF(x, base_y + y_offset), letter)
            x += widths[index]
