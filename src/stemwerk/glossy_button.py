from __future__ import annotations

from typing import Optional, Union

from PySide6 import QtCore, QtGui, QtWidgets


class GlossyButton(QtWidgets.QPushButton):
    def __init__(
        self,
        text: str = "",
        base_color: Optional[Union[QtGui.QColor, str]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(text, parent)
        self._base_color = QtGui.QColor("#2a2a2a")
        self._text_color = QtGui.QColor("#ffffff")
        if base_color is not None:
            self.set_base_color(base_color)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self.setMouseTracking(True)
        self.setFlat(True)

    def set_base_color(self, color: Union[QtGui.QColor, str]) -> None:
        self._base_color = QtGui.QColor(color)
        self.update()

    def set_text_color(self, color: Union[QtGui.QColor, str]) -> None:
        self._text_color = QtGui.QColor(color)
        self.update()

    def _multiply_color(self, color: QtGui.QColor, factor: float) -> QtGui.QColor:
        return QtGui.QColor.fromRgbF(
            min(1.0, color.redF() * factor),
            min(1.0, color.greenF() * factor),
            min(1.0, color.blueF() * factor),
            color.alphaF(),
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        rect = self.rect()
        radius = rect.height() / 2.0
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(rect), radius, radius)

        base_color = QtGui.QColor(self._base_color)
        if self.isDown():
            base_color = self._multiply_color(base_color, 0.8)
        elif self.underMouse():
            base_color = self._multiply_color(base_color, 1.2)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(base_color)
        painter.drawPath(path)

        painter.save()
        painter.setClipPath(path)

        hi_color = QtGui.QColor(base_color)
        hi_color.setRedF(min(1.0, hi_color.redF() + 0.3))
        hi_color.setGreenF(min(1.0, hi_color.greenF() + 0.3))
        hi_color.setBlueF(min(1.0, hi_color.blueF() + 0.3))
        highlight_h = max(1, int(rect.height() * 0.42))
        highlight_rect = QtCore.QRectF(rect.left(), rect.top(), rect.width(), highlight_h)
        highlight_grad = QtGui.QLinearGradient(highlight_rect.topLeft(), highlight_rect.bottomLeft())
        highlight_grad.setColorAt(0.0, QtGui.QColor.fromRgbF(hi_color.redF(), hi_color.greenF(), hi_color.blueF(), 0.25))
        highlight_grad.setColorAt(1.0, QtGui.QColor.fromRgbF(hi_color.redF(), hi_color.greenF(), hi_color.blueF(), 0.0))
        painter.fillRect(highlight_rect, highlight_grad)

        band_y = rect.top() + rect.height() * 0.18
        band_h = max(1, int(rect.height() * 0.22))
        band_rect = QtCore.QRectF(rect.left(), band_y, rect.width(), band_h)
        band_grad = QtGui.QLinearGradient(band_rect.topLeft(), band_rect.bottomLeft())
        band_grad.setColorAt(0.0, QtGui.QColor.fromRgbF(1.0, 1.0, 1.0, 0.12))
        band_grad.setColorAt(1.0, QtGui.QColor.fromRgbF(1.0, 1.0, 1.0, 0.0))
        painter.fillRect(band_rect, band_grad)

        shadow_h = max(1, int(rect.height() * 0.35))
        shadow_rect = QtCore.QRectF(rect.left(), rect.bottom() - shadow_h + 1, rect.width(), shadow_h)
        shadow_color = self._multiply_color(base_color, 0.6)
        shadow_grad = QtGui.QLinearGradient(shadow_rect.bottomLeft(), shadow_rect.topLeft())
        shadow_grad.setColorAt(0.0, QtGui.QColor.fromRgbF(shadow_color.redF(), shadow_color.greenF(), shadow_color.blueF(), 0.18))
        shadow_grad.setColorAt(1.0, QtGui.QColor.fromRgbF(shadow_color.redF(), shadow_color.greenF(), shadow_color.blueF(), 0.0))
        painter.fillRect(shadow_rect, shadow_grad)

        inner_color = self._multiply_color(base_color, 0.7)
        inner_top = QtCore.QRectF(rect.left(), rect.top(), rect.width(), 2)
        inner_bottom = QtCore.QRectF(rect.left(), rect.bottom() - 1, rect.width(), 2)
        painter.fillRect(inner_top, QtGui.QColor.fromRgbF(inner_color.redF(), inner_color.greenF(), inner_color.blueF(), 0.2))
        painter.fillRect(inner_bottom, QtGui.QColor.fromRgbF(inner_color.redF(), inner_color.greenF(), inner_color.blueF(), 0.2))

        painter.restore()

        painter.setFont(self.font())
        text_rect = QtCore.QRectF(rect)
        text = self.text()
        if text:
            shadow_color = QtGui.QColor(0, 0, 0, int(255 * 0.5))
            painter.setPen(shadow_color)
            for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                painter.drawText(text_rect.translated(dx, dy), QtCore.Qt.AlignmentFlag.AlignCenter, text)
            shadow_color = QtGui.QColor(0, 0, 0, int(255 * 0.3))
            painter.setPen(shadow_color)
            painter.drawText(text_rect.translated(2, 2), QtCore.Qt.AlignmentFlag.AlignCenter, text)

            painter.setPen(self._text_color)
            painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)
