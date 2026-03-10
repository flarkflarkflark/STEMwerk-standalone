from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from PySide6 import QtCore, QtGui, QtWidgets

from stemwerk_core import get_available_devices

from .player import Player
from .themes import THEMES
from .waveform_widget import WaveformWidget
from .workers import SeparationWorker


STEMS = ["vocals", "drums", "bass", "other"]


class ColorSlider(QtWidgets.QSlider):
    def __init__(self, color: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self._color = color
        self._track_color = "#1a1a1a"
        self.setRange(0, 100)
        self.setValue(100)
        self.setMinimumHeight(16)

    def set_color(self, color: str) -> None:
        self._color = color
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        track_height = 6
        margin = 8
        track = QtCore.QRectF(
            margin,
            rect.center().y() - track_height / 2,
            rect.width() - margin * 2,
            track_height,
        )

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(self._track_color))
        painter.drawRoundedRect(track, 3, 3)

        ratio = (self.value() - self.minimum()) / float(self.maximum() - self.minimum())
        fill = QtCore.QRectF(track.left(), track.top(), track.width() * ratio, track.height())
        painter.setBrush(QtGui.QColor(self._color))
        painter.drawRoundedRect(fill, 3, 3)

        handle_x = track.left() + track.width() * ratio
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
        painter.setBrush(QtGui.QColor(self._color))
        painter.drawEllipse(QtCore.QPointF(handle_x, rect.center().y()), 6, 6)


class VUMeterWidget(QtWidgets.QWidget):
    def __init__(self, color: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._level = 0.0
        self._color = color
        self._background = "#1a1a1a"
        self.setFixedHeight(4)

    def set_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor(self._background))
        fill_width = int(rect.width() * self._level)
        if fill_width > 0:
            painter.fillRect(
                QtCore.QRect(0, 0, fill_width, rect.height()),
                QtGui.QColor(self._color),
            )


class VolumeControl(QtWidgets.QWidget):
    def __init__(self, color: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.slider = ColorSlider(color)
        self.meter = VUMeterWidget(color)
        layout.addWidget(self.slider)
        layout.addWidget(self.meter)
        self.setMinimumWidth(200)

    def set_level(self, level: float) -> None:
        self.meter.set_level(level)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("STEMwerk")
        self.setMinimumSize(900, 650)
        self.setAcceptDrops(True)

        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate: Optional[int] = None
        self._duration: float = 0.0
        self._input_path: Optional[Path] = None
        self._stem_audio: Dict[str, np.ndarray] = {}
        self._stem_files: Dict[str, str] = {}
        self._selected_stem: str = STEMS[0]
        self._worker: Optional[SeparationWorker] = None
        self._slider_updating = False

        self._player = Player()
        self._position_timer = QtCore.QTimer(self)
        self._position_timer.setInterval(100)
        self._position_timer.timeout.connect(self._update_transport_position)

        self._vu_timer = QtCore.QTimer(self)
        self._vu_timer.setInterval(33)
        self._vu_timer.timeout.connect(self._update_vu_meters)

        self._theme = THEMES["classic"]
        self._stem_colors = {
            "vocals": self._theme["stem_colors"][0],
            "drums": self._theme["stem_colors"][1],
            "bass": self._theme["stem_colors"][2],
            "other": self._theme["stem_colors"][3],
            "guitar": self._theme["stem_colors"][4],
            "piano": self._theme["stem_colors"][5],
        }

        self._build_ui()
        self._bind_shortcuts()
        self._refresh_devices()
        self._select_stem(self._selected_stem)
        self._update_transport_state(False)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        toolbar = QtWidgets.QFrame()
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 8, 8, 8)
        toolbar_layout.setSpacing(8)

        self.open_button = QtWidgets.QPushButton("Open File")
        self.open_button.clicked.connect(self._open_file_dialog)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["htdemucs", "htdemucs_ft", "htdemucs_6s"])

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.currentIndexChanged.connect(self._update_status)

        self.separate_button = QtWidgets.QPushButton("Separate")
        self.separate_button.setEnabled(False)
        self.separate_button.clicked.connect(self._start_separation)

        self.export_button = QtWidgets.QPushButton("Export")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_stems)

        toolbar_layout.addWidget(self.open_button)
        toolbar_layout.addWidget(QtWidgets.QLabel("Model:"))
        toolbar_layout.addWidget(self.model_combo)
        toolbar_layout.addWidget(QtWidgets.QLabel("Device:"))
        toolbar_layout.addWidget(self.device_combo)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.separate_button)
        toolbar_layout.addWidget(self.export_button)

        self.waveform = WaveformWidget()
        self.waveform.position_clicked.connect(self._on_waveform_clicked)

        stems_panel = QtWidgets.QFrame()
        stems_panel.setMinimumHeight(120)
        stems_layout = QtWidgets.QGridLayout(stems_panel)
        stems_layout.setContentsMargins(8, 8, 8, 8)
        stems_layout.setHorizontalSpacing(8)
        stems_layout.setVerticalSpacing(6)
        stems_layout.setColumnMinimumWidth(0, 24)
        stems_layout.setColumnMinimumWidth(1, 20)
        stems_layout.setColumnMinimumWidth(2, 60)
        stems_layout.setColumnMinimumWidth(3, 32)
        stems_layout.setColumnMinimumWidth(4, 32)
        stems_layout.setColumnStretch(5, 1)

        self.stem_controls: Dict[str, Dict[str, QtWidgets.QWidget]] = {}
        for row, stem in enumerate(STEMS):
            stems_layout.setRowMinimumHeight(row, 40)

            checkbox = QtWidgets.QCheckBox()
            checkbox.setFixedWidth(24)
            checkbox.setChecked(True)
            checkbox.setToolTip("Include this stem in separation")
            checkbox.stateChanged.connect(self._update_stem_states)

            color = QtWidgets.QLabel()
            color.setFixedSize(20, 20)
            color.setStyleSheet(f"background: {self._stem_colors[stem]}; border: 1px solid #000000;")

            name_button = QtWidgets.QToolButton()
            name_button.setText(stem.capitalize())
            name_button.setFixedWidth(60)
            name_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            name_button.setStyleSheet("border: none; text-align: left;")
            name_button.clicked.connect(lambda _, s=stem: self._select_stem(s))

            solo_button = QtWidgets.QToolButton()
            solo_button.setText("S")
            solo_button.setFixedWidth(32)
            solo_button.setCheckable(True)
            solo_button.setToolTip("Solo this stem (only play this stem)")
            solo_button.clicked.connect(self._update_stem_states)

            mute_button = QtWidgets.QToolButton()
            mute_button.setText("M")
            mute_button.setFixedWidth(32)
            mute_button.setCheckable(True)
            mute_button.setToolTip("Mute this stem")
            mute_button.clicked.connect(self._update_stem_states)

            volume = VolumeControl(self._stem_colors[stem])
            volume.slider.valueChanged.connect(self._update_stem_states)
            volume.slider.setToolTip("Volume: 100%")

            stems_layout.addWidget(checkbox, row, 0)
            stems_layout.addWidget(color, row, 1)
            stems_layout.addWidget(name_button, row, 2)
            stems_layout.addWidget(solo_button, row, 3)
            stems_layout.addWidget(mute_button, row, 4)
            stems_layout.addWidget(volume, row, 5)

            self.stem_controls[stem] = {
                "checkbox": checkbox,
                "solo": solo_button,
                "mute": mute_button,
                "volume": volume,
                "name": name_button,
            }

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.waveform)
        splitter.addWidget(stems_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        transport = QtWidgets.QFrame()
        transport_layout = QtWidgets.QHBoxLayout(transport)
        transport_layout.setContentsMargins(8, 8, 8, 8)
        transport_layout.setSpacing(8)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_playback)

        self.position_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.valueChanged.connect(self._on_slider_changed)

        self.time_label = QtWidgets.QLabel("00:00 / 00:00")
        self.time_label.setObjectName("dimText")

        transport_layout.addWidget(self.play_button)
        transport_layout.addWidget(self.stop_button)
        transport_layout.addWidget(self.position_slider, 1)
        transport_layout.addWidget(self.time_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

        layout.addWidget(toolbar)
        layout.addWidget(splitter, 1)
        layout.addWidget(transport)
        layout.addWidget(self.progress_bar)

        self.setCentralWidget(central)

        status = QtWidgets.QStatusBar()
        self.status_file = QtWidgets.QLabel("No file loaded")
        self.status_info = QtWidgets.QLabel("")
        self.status_device = QtWidgets.QLabel("")
        status.addWidget(self.status_file, 1)
        status.addWidget(self.status_info)
        status.addWidget(self.status_device)
        self.setStatusBar(status)

    def _bind_shortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, self._open_file_dialog)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, self._start_separation)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, self._toggle_play)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+E"), self, self._export_stems)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self, self.close)
        QtGui.QShortcut(QtGui.QKeySequence("1"), self, lambda: self._toggle_stem_checkbox("vocals"))
        QtGui.QShortcut(QtGui.QKeySequence("2"), self, lambda: self._toggle_stem_checkbox("drums"))
        QtGui.QShortcut(QtGui.QKeySequence("3"), self, lambda: self._toggle_stem_checkbox("bass"))
        QtGui.QShortcut(QtGui.QKeySequence("4"), self, lambda: self._toggle_stem_checkbox("other"))
        QtGui.QShortcut(QtGui.QKeySequence("S"), self, self._toggle_selected_solo)
        QtGui.QShortcut(QtGui.QKeySequence("M"), self, self._toggle_selected_mute)

    def _toggle_stem_checkbox(self, stem: str) -> None:
        checkbox = self.stem_controls[stem]["checkbox"]
        checkbox.setChecked(not checkbox.isChecked())

    def _toggle_selected_solo(self) -> None:
        control = self.stem_controls[self._selected_stem]["solo"]
        control.setChecked(not control.isChecked())
        self._update_stem_states()

    def _toggle_selected_mute(self) -> None:
        control = self.stem_controls[self._selected_stem]["mute"]
        control.setChecked(not control.isChecked())
        self._update_stem_states()

    def _select_stem(self, stem: str) -> None:
        self._selected_stem = stem
        for name, controls in self.stem_controls.items():
            button = controls["name"]
            font = button.font()
            font.setBold(name == stem)
            button.setFont(font)

    def _refresh_devices(self) -> None:
        self.device_combo.clear()
        devices = get_available_devices()
        for dev in devices:
            label = f"{dev['name']} ({dev['id']})"
            self.device_combo.addItem(label, dev["id"])

    def _open_file_dialog(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg)",
        )
        if file_path:
            self._load_audio(Path(file_path))

    def _load_audio(self, path: Path) -> None:
        data, sample_rate = sf.read(str(path), dtype="float32")
        self._audio_data = data
        self._sample_rate = int(sample_rate)
        self._input_path = path
        self._duration = data.shape[0] / float(sample_rate)

        self._player.stop()
        self._player.load_audio(data, sample_rate)
        self._player.clear_stems()
        self._stem_audio = {}
        self._stem_files = {}
        self.waveform.set_audio_data(data, sample_rate)
        self._update_stem_states()
        self._update_transport_state(True)
        self._update_status()

    def _update_status(self) -> None:
        if not self._input_path or self._sample_rate is None:
            self.status_file.setText("No file loaded")
            self.status_info.setText("")
            self.status_device.setText("")
            return
        self.status_file.setText(self._input_path.name)
        self.status_info.setText(f"{self._sample_rate} Hz | {self._format_time(self._duration)}")
        device_id = self.device_combo.currentData()
        self.status_device.setText(f"Device: {device_id}")

    def _start_separation(self) -> None:
        if not self._input_path:
            QtWidgets.QMessageBox.warning(self, "No file", "Please open an audio file first.")
            return
        if self._worker and self._worker.isRunning():
            return

        output_dir = self._input_path.parent / f"{self._input_path.stem}_stems"
        output_dir.mkdir(parents=True, exist_ok=True)
        stems = [stem for stem in STEMS if self.stem_controls[stem]["checkbox"].isChecked()]
        device_id = self.device_combo.currentData() or "auto"

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0% - Starting")

        self.separate_button.setEnabled(False)
        self.export_button.setEnabled(False)

        self._worker = SeparationWorker(
            input_file=str(self._input_path),
            output_dir=str(output_dir),
            model=self.model_combo.currentText(),
            device=str(device_id),
            stems=stems,
        )
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.finished.connect(self._on_separation_finished)
        self._worker.error.connect(self._on_separation_error)
        self._worker.start()

    def _on_progress(self, percent: float, message: str) -> None:
        self.progress_bar.setValue(int(percent))
        self.progress_bar.setFormat(f"{percent:.0f}% - {message}")

    def _on_separation_finished(self, stems: Dict[str, str]) -> None:
        self.progress_bar.setVisible(False)
        self.separate_button.setEnabled(True)

        stem_audio: Dict[str, np.ndarray] = {}
        for name, path in stems.items():
            if not os.path.exists(path):
                continue
            data, _ = sf.read(path, dtype="float32")
            stem_audio[name] = data

        self._stem_audio = stem_audio
        self._stem_files = stems
        self._player.set_stems(stem_audio)
        self._update_stem_states()
        self.waveform.set_stem_overlays(stem_audio, self._stem_colors)
        self.waveform.set_stem_states(self._build_waveform_states())
        self._update_transport_state(True)
        self._update_status()

        stem_count = len(stem_audio)
        self.statusBar().showMessage(
            f"Separation complete ({stem_count} stems, {self._duration:.1f}s)",
            5000,
        )

    def _on_separation_error(self, message: str) -> None:
        self.progress_bar.setVisible(False)
        self.separate_button.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Separation failed", message)

    def _apply_button_styles(self) -> None:
        for stem, controls in self.stem_controls.items():
            solo_button: QtWidgets.QToolButton = controls["solo"]  # type: ignore[assignment]
            mute_button: QtWidgets.QToolButton = controls["mute"]  # type: ignore[assignment]
            color = self._stem_colors.get(stem, "#ffffff")

            if solo_button.isChecked():
                solo_button.setStyleSheet(
                    f"background: {color}; color: #ffffff; border: 1px solid #dddddd;"
                )
            else:
                solo_button.setStyleSheet("background: #2a2a2a; color: #888; border: 1px solid #444;")

            if mute_button.isChecked():
                mute_button.setStyleSheet("background: #cc3333; color: #ffffff; border: 1px solid #aa2222;")
            else:
                mute_button.setStyleSheet("background: #2a2a2a; color: #888; border: 1px solid #444;")

    def _build_waveform_states(self) -> Dict[str, Dict[str, bool]]:
        states: Dict[str, Dict[str, bool]] = {}
        for stem, controls in self.stem_controls.items():
            states[stem] = {
                "enabled": controls["checkbox"].isChecked(),
                "solo": controls["solo"].isChecked(),
                "mute": controls["mute"].isChecked(),
            }
        return states

    def _update_stem_states(self) -> None:
        for stem, controls in self.stem_controls.items():
            enabled = controls["checkbox"].isChecked()
            solo = controls["solo"].isChecked()
            mute = controls["mute"].isChecked()
            volume_widget: VolumeControl = controls["volume"]  # type: ignore[assignment]
            volume = volume_widget.slider.value() / 100.0
            volume_widget.slider.setToolTip(f"Volume: {int(volume * 100)}%")
            self._player.update_stem_state(stem, enabled, solo, mute, volume)
        self._apply_button_styles()
        self.waveform.set_stem_states(self._build_waveform_states())

    def _toggle_play(self) -> None:
        if self._audio_data is None:
            return
        if self._player.is_playing:
            self._stop_playback()
        else:
            self._player.play()
            self._position_timer.start()
            self._vu_timer.start()

    def _stop_playback(self) -> None:
        self._player.stop()
        self._position_timer.stop()
        self._vu_timer.stop()
        self._clear_vu_meters()

    def _update_transport_position(self) -> None:
        if self._sample_rate is None:
            return
        if not self._player.is_playing:
            self._position_timer.stop()
            self._vu_timer.stop()
            self._clear_vu_meters()
        seconds = self._player.position
        if self._duration > 0:
            value = int((seconds / self._duration) * 1000)
            self._slider_updating = True
            self.position_slider.setValue(value)
            self._slider_updating = False
        self.time_label.setText(f"{self._format_time(seconds)} / {self._format_time(self._duration)}")
        self.waveform.set_playhead(seconds)

    def _on_slider_changed(self, value: int) -> None:
        if self._duration <= 0 or self._slider_updating:
            return
        seconds = (value / 1000.0) * self._duration
        self._player.seek(seconds)
        self.time_label.setText(f"{self._format_time(seconds)} / {self._format_time(self._duration)}")
        self.waveform.set_playhead(seconds)

    def _on_waveform_clicked(self, seconds: float) -> None:
        self._player.seek(seconds)
        if self._duration > 0:
            value = int((seconds / self._duration) * 1000)
            self._slider_updating = True
            self.position_slider.setValue(value)
            self._slider_updating = False
        self.time_label.setText(f"{self._format_time(seconds)} / {self._format_time(self._duration)}")
        self.waveform.set_playhead(seconds)

    def _update_transport_state(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self.position_slider.setEnabled(enabled)
        self.export_button.setEnabled(enabled and bool(self._stem_files))
        self.separate_button.setEnabled(enabled and not (self._worker and self._worker.isRunning()))

    def _export_stems(self) -> None:
        if not self._stem_files:
            QtWidgets.QMessageBox.information(self, "No stems", "Run separation first.")
            return
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not output_dir:
            return
        exported = 0
        for stem, path in self._stem_files.items():
            controls = self.stem_controls.get(stem)
            if controls and not controls["checkbox"].isChecked():
                continue
            if not os.path.exists(path):
                continue
            target = Path(output_dir) / Path(path).name
            shutil.copy2(path, target)
            exported += 1
        QtWidgets.QMessageBox.information(self, "Export complete", f"Exported {exported} stems.")

    def _update_vu_meters(self) -> None:
        if not self._player.is_playing:
            return
        levels = self._player.get_levels()
        for stem, controls in self.stem_controls.items():
            volume_widget: VolumeControl = controls["volume"]  # type: ignore[assignment]
            volume_widget.set_level(levels.get(stem, 0.0))

    def _clear_vu_meters(self) -> None:
        for controls in self.stem_controls.values():
            volume_widget: VolumeControl = controls["volume"]  # type: ignore[assignment]
            volume_widget.set_level(0.0)

    def _format_time(self, seconds: float) -> str:
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.exists():
                self._load_audio(path)
