from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class WaveformWidget(QtWidgets.QWidget):
    position_clicked = QtCore.Signal(float)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._audio_mono: Optional[np.ndarray] = None
        self._sample_rate: Optional[int] = None
        self._peaks: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._stem_overlays: Dict[str, np.ndarray] = {}
        self._stem_colors: Dict[str, str] = {}
        self._stem_states: Dict[str, Dict[str, bool]] = {}
        self._stem_peaks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._playhead_seconds: float = 0.0
        self._last_width: int = 0
        self.setMinimumHeight(150)

    def set_audio_data(self, audio: np.ndarray, sample_rate: int) -> None:
        if audio.ndim == 2:
            mono = audio.mean(axis=1)
        else:
            mono = audio
        self._audio_mono = mono.astype(np.float32, copy=False)
        self._sample_rate = sample_rate
        self._peaks = None
        self._stem_overlays = {}
        self._stem_colors = {}
        self._stem_states = {}
        self._stem_peaks = {}
        self._playhead_seconds = 0.0
        self.update()

    def set_stem_overlays(self, stems: Dict[str, np.ndarray], colors: Dict[str, str]) -> None:
        self._stem_overlays = stems
        self._stem_colors = colors
        self._stem_peaks = {}
        self.update()

    def set_stem_states(self, states: Dict[str, Dict[str, bool]]) -> None:
        self._stem_states = states
        self.update()

    def set_playhead(self, seconds: float) -> None:
        self._playhead_seconds = max(0.0, seconds)
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self._peaks = None
        self._stem_peaks = {}
        super().resizeEvent(event)

    def _build_peaks(self, data: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
        length = data.shape[0]
        if width <= 0 or length == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        indices = np.linspace(0, length, num=width + 1, dtype=np.int64)
        mins = np.zeros(width, dtype=np.float32)
        maxs = np.zeros(width, dtype=np.float32)
        for i in range(width):
            start = indices[i]
            end = indices[i + 1]
            if end <= start:
                mins[i] = 0.0
                maxs[i] = 0.0
                continue
            segment = data[start:end]
            mins[i] = float(segment.min())
            maxs[i] = float(segment.max())
        return mins, maxs

    def _draw_waveform(
        self,
        painter: QtGui.QPainter,
        mins: np.ndarray,
        maxs: np.ndarray,
        color: QtGui.QColor,
        alpha: int,
    ) -> None:
        if mins.size == 0 or maxs.size == 0 or alpha <= 0:
            return
        width = self.width()
        height = self.height()
        mid = height / 2.0
        scale = mid
        path = QtGui.QPainterPath()
        path.moveTo(0, mid)
        for x in range(width):
            y = mid - maxs[x] * scale
            path.lineTo(x, y)
        for x in reversed(range(width)):
            y = mid - mins[x] * scale
            path.lineTo(x, y)
        path.closeSubpath()
        brush_color = QtGui.QColor(color)
        brush_color.setAlpha(alpha)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(brush_color)
        painter.drawPath(path)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, self.palette().window())

        if self._audio_mono is None or self._sample_rate is None:
            painter.setPen(self.palette().text().color())
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, "Drop an audio file to begin")
            return

        width = max(1, self.width())
        if self._peaks is None or self._last_width != width:
            self._peaks = self._build_peaks(self._audio_mono, width)
            self._last_width = width

        mins, maxs = self._peaks
        self._draw_waveform(painter, mins, maxs, self.palette().highlight().color(), 120)

        any_solo = any(state.get("solo") for state in self._stem_states.values())
        for stem_name, stem_data in self._stem_overlays.items():
            state = self._stem_states.get(stem_name, {"enabled": True, "solo": False, "mute": False})
            if not state.get("enabled", True) or state.get("mute"):
                continue
            peaks = self._stem_peaks.get(stem_name)
            if peaks is None or self._last_width != width:
                peaks = self._build_peaks(stem_data, width)
                self._stem_peaks[stem_name] = peaks
            color_hex = self._stem_colors.get(stem_name, "#ffffff")
            stem_color = QtGui.QColor(color_hex)
            if any_solo:
                opacity = 0.8 if state.get("solo") else 0.1
            else:
                opacity = 0.8
            self._draw_waveform(painter, peaks[0], peaks[1], stem_color, int(opacity * 255))

        if self._sample_rate and self._audio_mono.size > 0:
            total_seconds = self._audio_mono.size / float(self._sample_rate)
            if total_seconds > 0:
                ratio = min(self._playhead_seconds / total_seconds, 1.0)
                x = int(ratio * width)
                painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
                painter.drawLine(x, 0, x, self.height())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._sample_rate is None or self._audio_mono is None:
            return
        if self.width() <= 0:
            return
        total_seconds = self._audio_mono.size / float(self._sample_rate)
        ratio = max(0.0, min(event.position().x() / float(self.width()), 1.0))
        self.position_clicked.emit(ratio * total_seconds)
