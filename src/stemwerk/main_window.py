from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6 import QtCore, QtGui, QtWidgets

from stemwerk_core import get_available_devices
from stemwerk_core.models import AVAILABLE_MODELS

from .export_dialog import ExportDialog
from .glossy_button import GlossyButton
from .logo_widget import LogoWidget
from .player import Player
from .stem_border import StemBorderWidget
from .themes import STEM_COLORS, apply_theme, resolve_theme, to_qcolor
from .vertical_slider import VerticalStemSlider
from .waveform_widget import WaveformWidget
from .workers import SeparationWorker


STEMS_4 = ["vocals", "drums", "bass", "other"]
STEMS_6 = ["vocals", "drums", "bass", "other", "guitar", "piano"]
STEM_LABELS = {
    "vocals": "VOC",
    "drums": "DRM",
    "bass": "BASS",
    "other": "OTH",
    "guitar": "GTR",
    "piano": "PNO",
}
THEME_ORDER = ["classic", "ember", "ice", "mono"]


class GradientWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._top = QtGui.QColor("#1a1a1e")
        self._bottom = QtGui.QColor("#2e2e34")

    def set_colors(self, top: QtGui.QColor, bottom: QtGui.QColor) -> None:
        self._top = QtGui.QColor(top)
        self._bottom = QtGui.QColor(bottom)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        gradient = QtGui.QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, self._top)
        gradient.setColorAt(1.0, self._bottom)
        painter.fillRect(self.rect(), gradient)


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
        self._output_dir: Optional[Path] = None
        self._stem_audio: Dict[str, np.ndarray] = {}
        self._stem_files: Dict[str, str] = {}
        self._selected_stem: str = STEMS_4[0]
        self._worker: Optional[SeparationWorker] = None
        self._slider_updating = False

        self._player = Player()
        self._position_timer = QtCore.QTimer(self)
        self._position_timer.setInterval(100)
        self._position_timer.timeout.connect(self._update_transport_position)

        self._vu_timer = QtCore.QTimer(self)
        self._vu_timer.setInterval(33)
        self._vu_timer.timeout.connect(self._update_vu_meters)

        self._theme_name = "classic"
        self._theme_mode = "dark"
        self._theme = resolve_theme(self._theme_name, self._theme_mode)
        self._theme_colors = {key: to_qcolor(value) for key, value in self._theme.items()}
        self._stem_colors = {
            "vocals": STEM_COLORS[0],
            "drums": STEM_COLORS[1],
            "bass": STEM_COLORS[2],
            "other": STEM_COLORS[3],
            "guitar": STEM_COLORS[4],
            "piano": STEM_COLORS[5],
        }

        self._stems_panel: Optional[QtWidgets.QFrame] = None
        self._stems_layout: Optional[QtWidgets.QHBoxLayout] = None
        self.stem_controls: Dict[str, Dict[str, QtWidgets.QWidget]] = {}

        self._background_widget = GradientWidget()
        self._build_ui()
        self._bind_shortcuts()
        self._refresh_devices()
        self._refresh_output_devices()
        self._apply_theme()
        self._apply_button_styles()
        self._select_stem(self._selected_stem)
        self._update_transport_state(False)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self._background_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.stem_border = StemBorderWidget(STEM_COLORS[:4])

        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header.setFixedHeight(40)

        self.logo_widget = LogoWidget()
        self.logo_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

        self.theme_button = GlossyButton("??")
        self.theme_button.setFixedSize(30, 30)
        self.theme_button.clicked.connect(self._cycle_theme)
        self.theme_button.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.theme_button.customContextMenuRequested.connect(lambda _: self._toggle_theme_mode())

        header_layout.addStretch(1)
        header_layout.addWidget(self.logo_widget, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        header_layout.addStretch(1)
        header_layout.addWidget(self.theme_button, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        toolbar = QtWidgets.QFrame()
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 8, 8, 8)
        toolbar_layout.setSpacing(8)

        self.open_button = GlossyButton("Open File")
        self.open_button.clicked.connect(self._open_file_dialog)

        self.model_combo = QtWidgets.QComboBox()
        for model_id in AVAILABLE_MODELS:
            self.model_combo.addItem(model_id, model_id)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)

        self.quality_combo = QtWidgets.QComboBox()
        self.quality_combo.addItems(["Fast", "Normal", "Best"])
        self.quality_combo.setCurrentText("Normal")

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.currentIndexChanged.connect(self._update_status)

        self.output_combo = QtWidgets.QComboBox()
        self.output_combo.currentIndexChanged.connect(self._on_output_device_changed)

        self.output_folder_button = GlossyButton("Stem folder")
        self.output_folder_button.clicked.connect(self._choose_output_dir)

        self.separate_button = GlossyButton("Separate")
        self.separate_button.setEnabled(False)
        self.separate_button.clicked.connect(self._start_separation)

        self.cancel_button = GlossyButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_separation)

        self.export_button = GlossyButton("Export")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_stems)

        toolbar_layout.addWidget(self.open_button)
        toolbar_layout.addWidget(QtWidgets.QLabel("Model:"))
        toolbar_layout.addWidget(self.model_combo)
        toolbar_layout.addWidget(QtWidgets.QLabel("Quality:"))
        toolbar_layout.addWidget(self.quality_combo)
        toolbar_layout.addWidget(QtWidgets.QLabel("Device:"))
        toolbar_layout.addWidget(self.device_combo)
        toolbar_layout.addWidget(QtWidgets.QLabel("Audio:"))
        toolbar_layout.addWidget(self.output_combo)
        toolbar_layout.addWidget(self.output_folder_button)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.separate_button)
        toolbar_layout.addWidget(self.cancel_button)
        toolbar_layout.addWidget(self.export_button)

        self.waveform = WaveformWidget()
        self.waveform.position_clicked.connect(self._on_waveform_clicked)

        self._stems_panel = QtWidgets.QFrame()
        self._stems_panel.setMinimumHeight(200)
        self._stems_layout = QtWidgets.QHBoxLayout(self._stems_panel)
        self._stems_layout.setContentsMargins(8, 8, 8, 8)
        self._stems_layout.setSpacing(12)

        self._rebuild_stem_controls(self._current_stems_for_model(), preserve_states=False)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.waveform)
        splitter.addWidget(self._stems_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        transport = QtWidgets.QFrame()
        transport_layout = QtWidgets.QHBoxLayout(transport)
        transport_layout.setContentsMargins(8, 8, 8, 8)
        transport_layout.setSpacing(8)

        self.play_button = GlossyButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        self.stop_button = GlossyButton("Stop")
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

        layout.addWidget(self.stem_border)
        layout.addWidget(header)
        layout.addWidget(toolbar)
        layout.addWidget(splitter, 1)
        layout.addWidget(transport)
        layout.addWidget(self.progress_bar)

        self.setCentralWidget(self._background_widget)

        status = QtWidgets.QStatusBar()
        self.status_file = QtWidgets.QLabel("No file loaded")
        self.status_info = QtWidgets.QLabel("")
        self.status_device = QtWidgets.QLabel("")
        self.status_output = QtWidgets.QLabel("")
        self.status_folder = QtWidgets.QLabel("")
        status.addWidget(self.status_file, 1)
        status.addWidget(self.status_info)
        status.addWidget(self.status_device)
        status.addWidget(self.status_output)
        status.addWidget(self.status_folder)
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

    def _current_stems_for_model(self) -> List[str]:
        model = str(self.model_combo.currentData() or self.model_combo.currentText())
        if model == "htdemucs_6s":
            return list(STEMS_6)
        return list(STEMS_4)

    def _on_model_changed(self, model: str) -> None:
        self._rebuild_stem_controls(self._current_stems_for_model(), preserve_states=True)
        self._update_stem_border()

    def _rebuild_stem_controls(self, stems: List[str], preserve_states: bool = True) -> None:
        if self._stems_layout is None:
            return

        previous: Dict[str, Dict[str, object]] = {}
        if preserve_states:
            for stem, controls in self.stem_controls.items():
                volume_slider: VerticalStemSlider = controls["volume"]  # type: ignore[assignment]
                previous[stem] = {
                    "checked": controls["checkbox"].isChecked(),
                    "solo": controls["solo"].isChecked(),
                    "mute": controls["mute"].isChecked(),
                    "volume": float(volume_slider.value()),
                }

        while self._stems_layout.count():
            item = self._stems_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.stem_controls = {}
        for stem in stems:
            column = QtWidgets.QWidget()
            column_layout = QtWidgets.QVBoxLayout(column)
            column_layout.setContentsMargins(4, 4, 4, 4)
            column_layout.setSpacing(6)
            column_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

            label = QtWidgets.QLabel(STEM_LABELS.get(stem, stem[:3].upper()))
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setFixedHeight(22)
            label.setStyleSheet(f"color: {self._stem_colors[stem]}; font-weight: 600;")
            label.mousePressEvent = lambda event, s=stem: self._select_stem(s)  # type: ignore[assignment]

            buttons_row = QtWidgets.QHBoxLayout()
            buttons_row.setSpacing(4)
            buttons_row.setContentsMargins(0, 0, 0, 0)

            solo_button = GlossyButton("S")
            solo_button.setCheckable(True)
            solo_button.setFixedSize(28, 28)
            solo_button.setChecked(bool(previous.get(stem, {}).get("solo", False)))
            solo_button.setToolTip("Solo this stem (only play this stem)")
            solo_button.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Weight.Bold))
            solo_button.toggled.connect(lambda checked, s=stem: self._on_solo_toggled(s, checked))

            mute_button = GlossyButton("M")
            mute_button.setCheckable(True)
            mute_button.setFixedSize(28, 28)
            mute_button.setChecked(bool(previous.get(stem, {}).get("mute", False)))
            mute_button.setToolTip("Mute this stem")
            mute_button.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Weight.Bold))
            mute_button.toggled.connect(lambda checked, s=stem: self._on_mute_toggled(s, checked))

            buttons_row.addWidget(solo_button)
            buttons_row.addWidget(mute_button)

            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(bool(previous.get(stem, {}).get("checked", True)))
            checkbox.setToolTip("Include this stem in separation")
            checkbox.setFixedSize(20, 20)

            slider = VerticalStemSlider(self._stem_colors[stem], stem)
            slider.setValue(int(previous.get(stem, {}).get("volume", 100.0)))
            slider.valueChanged.connect(self._update_waveform_state)
            slider.setToolTip(f"Volume: {slider.value()}%")

            column_layout.addWidget(label)
            column_layout.addLayout(buttons_row)
            column_layout.addWidget(checkbox, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
            column_layout.addWidget(slider, 1)

            self._stems_layout.addWidget(column, 1)

            self.stem_controls[stem] = {
                "checkbox": checkbox,
                "solo": solo_button,
                "mute": mute_button,
                "volume": slider,
                "name": label,
            }

        if stems:
            if self._selected_stem not in stems:
                self._selected_stem = stems[0]
            self._select_stem(self._selected_stem)
        else:
            self._selected_stem = ""

        self._player.prune_stem_states(stems)
        self._apply_button_styles()
        self._update_waveform_state()
        self._update_stem_border()

    def _update_stem_border(self) -> None:
        count = 6 if self.model_combo.currentText() == "htdemucs_6s" else 4
        self.stem_border.set_colors(STEM_COLORS[:count])

    def _toggle_stem_checkbox(self, stem: str) -> None:
        controls = self.stem_controls.get(stem)
        if not controls:
            return
        checkbox = controls["checkbox"]
        checkbox.setChecked(not checkbox.isChecked())

    def _toggle_selected_solo(self) -> None:
        if not self._selected_stem or self._selected_stem not in self.stem_controls:
            return
        control = self.stem_controls[self._selected_stem]["solo"]
        control.setChecked(not control.isChecked())

    def _toggle_selected_mute(self) -> None:
        if not self._selected_stem or self._selected_stem not in self.stem_controls:
            return
        control = self.stem_controls[self._selected_stem]["mute"]
        control.setChecked(not control.isChecked())

    def _select_stem(self, stem: str) -> None:
        self._selected_stem = stem
        for name, controls in self.stem_controls.items():
            label: QtWidgets.QLabel = controls["name"]  # type: ignore[assignment]
            font = label.font()
            font.setBold(name == stem)
            label.setFont(font)

    def _on_solo_toggled(self, stem: str, checked: bool) -> None:
        controls = self.stem_controls.get(stem)
        if not controls:
            return
        mute_button: GlossyButton = controls["mute"]  # type: ignore[assignment]
        if checked and mute_button.isChecked():
            mute_button.blockSignals(True)
            mute_button.setChecked(False)
            mute_button.blockSignals(False)
        self._update_waveform_state()

    def _on_mute_toggled(self, stem: str, checked: bool) -> None:
        controls = self.stem_controls.get(stem)
        if not controls:
            return
        solo_button: GlossyButton = controls["solo"]  # type: ignore[assignment]
        if checked and solo_button.isChecked():
            solo_button.blockSignals(True)
            solo_button.setChecked(False)
            solo_button.blockSignals(False)
        self._update_waveform_state()

    def _refresh_devices(self) -> None:
        self.device_combo.clear()
        self.device_combo.addItem("Auto", "auto")
        self.device_combo.addItem("CPU", "cpu")
        devices = get_available_devices()
        for dev in devices:
            label = f"{dev['name']} ({dev['id']})"
            self.device_combo.addItem(label, dev["id"])

    def _refresh_output_devices(self) -> None:
        self.output_combo.clear()
        self.output_combo.addItem("Default", None)
        for index, device in enumerate(sd.query_devices()):
            if device.get("max_output_channels", 0) > 0:
                name = device.get("name", f"Output {index}")
                label = f"{name} ({index})"
                self.output_combo.addItem(label, index)
        self._on_output_device_changed()

    def _on_output_device_changed(self) -> None:
        device_id = self.output_combo.currentData()
        self._player.set_output_device(device_id)
        self._update_status()

    def _choose_output_dir(self) -> None:
        start_dir = str(self._output_dir or (self._input_path.parent if self._input_path else Path.cwd()))
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose stem output folder",
            start_dir,
        )
        if selected:
            self._output_dir = Path(selected)
            self._update_status()

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
        self._output_dir = path.parent / f"{path.stem}_stems"
        self._duration = data.shape[0] / float(sample_rate)

        self._player.stop()
        self._player.load_audio(data, sample_rate)
        self._player.clear_stems()
        self._stem_audio = {}
        self._stem_files = {}
        self.waveform.set_audio_data(data, sample_rate)
        self._update_waveform_state()
        self._update_transport_state(True)
        self._update_status()

    def _update_status(self) -> None:
        if not self._input_path or self._sample_rate is None:
            self.status_file.setText("No file loaded")
            self.status_info.setText("")
            self.status_device.setText("")
            self.status_output.setText("")
            self.status_folder.setText("")
            return
        self.status_file.setText(self._input_path.name)
        self.status_info.setText(f"{self._sample_rate} Hz | {self._format_time(self._duration)}")
        device_id = self.device_combo.currentData()
        self.status_device.setText(f"Device: {device_id}")
        output_id = self.output_combo.currentData()
        self.status_output.setText(f"Audio: {output_id if output_id is not None else 'Default'}")
        folder = self._output_dir or (self._input_path.parent / f"{self._input_path.stem}_stems")
        self.status_folder.setText(f"Stems: {folder.name}")

    def _start_separation(self) -> None:
        if not self._input_path:
            QtWidgets.QMessageBox.warning(self, "No file", "Please open an audio file first.")
            return
        if self._worker and self._worker.isRunning():
            return

        stems = [
            stem for stem, controls in self.stem_controls.items() if controls["checkbox"].isChecked()
        ]
        if not stems:
            QtWidgets.QMessageBox.warning(
                self,
                "No stems selected",
                "Select at least one stem for separation.",
            )
            return

        output_dir = self._output_dir or (
            self._input_path.parent / f"{self._input_path.stem}_stems"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir
        device_id = self.device_combo.currentData() or "auto"
        model_id = str(self.model_combo.currentData() or self.model_combo.currentText())
        quality = self.quality_combo.currentText().lower()

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0% - Starting")

        self.separate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)

        self._worker = SeparationWorker(
            input_file=str(self._input_path),
            output_dir=str(output_dir),
            model=model_id,
            device=str(device_id),
            quality=quality,
            stems=stems,
        )
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.finished.connect(self._on_separation_finished)
        self._worker.error.connect(self._on_separation_error)
        self._worker.cancelled.connect(self._on_separation_cancelled)
        self._worker.start()

    def _cancel_separation(self) -> None:
        if self._worker and self._worker.isRunning():
            self.cancel_button.setEnabled(False)
            self.progress_bar.setFormat("Cancelling…")
            self._worker.cancel()

    def _on_progress(self, percent: float, message: str) -> None:
        self.progress_bar.setValue(int(percent))
        self.progress_bar.setFormat(f"{percent:.0f}% - {message}")

    def _on_separation_finished(self, stems: Dict[str, str]) -> None:
        self.progress_bar.setVisible(False)
        self.separate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        stem_audio: Dict[str, np.ndarray] = {}
        for name, path in stems.items():
            if not os.path.exists(path):
                continue
            data, _ = sf.read(path, dtype="float32")
            stem_audio[name] = data

        self._stem_audio = stem_audio
        self._stem_files = stems
        self._player.set_stems(stem_audio)
        self.waveform.set_stem_overlays(stem_audio, self._stem_colors)

        available_stems = [stem for stem in STEMS_6 if stem in stem_audio]
        if available_stems:
            self._rebuild_stem_controls(available_stems, preserve_states=True)
        else:
            self._update_waveform_state()

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
        self.cancel_button.setEnabled(False)
        QtWidgets.QMessageBox.critical(self, "Separation failed", message)

    def _on_separation_cancelled(self) -> None:
        self.progress_bar.setVisible(False)
        self.separate_button.setEnabled(bool(self._input_path))
        self.cancel_button.setEnabled(False)
        self.statusBar().showMessage("Separation cancelled.", 5000)

    def _apply_button_styles(self) -> None:
        accent = self._theme_colors["accent"].name()
        primary = self._theme_colors["buttonPrimary"].name()
        dim_text = "#cccccc"

        if hasattr(self, "open_button"):
            self.open_button.set_base_color(accent)
            self.open_button.set_text_color("#ffffff")
        if hasattr(self, "separate_button"):
            self.separate_button.set_base_color(accent)
            self.separate_button.set_text_color("#ffffff")
        if hasattr(self, "cancel_button"):
            self.cancel_button.set_base_color("#cc3333")
            self.cancel_button.set_text_color("#ffffff")
        if hasattr(self, "export_button"):
            self.export_button.set_base_color(accent)
            self.export_button.set_text_color("#ffffff")
        if hasattr(self, "output_folder_button"):
            self.output_folder_button.set_base_color(primary)
            self.output_folder_button.set_text_color("#ffffff")

        if hasattr(self, "play_button"):
            self.play_button.set_base_color(primary)
            self.play_button.set_text_color("#ffffff")
        if hasattr(self, "stop_button"):
            self.stop_button.set_base_color(primary)
            self.stop_button.set_text_color("#ffffff")

        if hasattr(self, "theme_button"):
            self.theme_button.set_base_color(accent)
            self.theme_button.set_text_color("#ffffff")

        for stem, controls in self.stem_controls.items():
            solo_button: GlossyButton = controls["solo"]  # type: ignore[assignment]
            mute_button: GlossyButton = controls["mute"]  # type: ignore[assignment]
            color = self._stem_colors.get(stem, "#ffffff")

            if solo_button.isChecked():
                solo_button.set_base_color(color)
                solo_button.set_text_color("#ffffff")
            else:
                solo_button.set_base_color("#2a2a2a")
                solo_button.set_text_color(dim_text)

            if mute_button.isChecked():
                mute_button.set_base_color("#cc3333")
                mute_button.set_text_color("#ffffff")
            else:
                mute_button.set_base_color("#2a2a2a")
                mute_button.set_text_color(dim_text)

    def _update_stem_states(self) -> None:
        for stem, controls in self.stem_controls.items():
            solo = controls["solo"].isChecked()
            mute = controls["mute"].isChecked()
            volume_slider: VerticalStemSlider = controls["volume"]  # type: ignore[assignment]
            volume = volume_slider.value() / 100.0
            volume_slider.setToolTip(f"Volume: {int(volume * 100)}%")
            self._player.update_stem_state(stem, solo, mute, volume)
        self._apply_button_styles()

    def _update_waveform_state(self) -> None:
        self._update_stem_states()
        any_solo = any(controls["solo"].isChecked() for controls in self.stem_controls.values())
        states: Dict[str, Dict[str, object]] = {}
        for stem, controls in self.stem_controls.items():
            solo = controls["solo"].isChecked()
            mute = controls["mute"].isChecked()
            volume_slider: VerticalStemSlider = controls["volume"]  # type: ignore[assignment]
            volume = volume_slider.value() / 100.0
            visible = (not mute) and (not any_solo or solo)
            states[stem] = {"visible": visible, "opacity": volume}
        self.waveform.set_stem_visibility(states)

    def _toggle_play(self) -> None:
        if self._audio_data is None:
            return
        if self._player.is_playing:
            self._stop_playback()
        else:
            if not self._player.play():
                self._show_playback_error(self._player.last_error or "Playback failed.")
                return
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

    def _show_playback_error(self, message: str) -> None:
        self.statusBar().showMessage(f"Playback error: {message}", 8000)

    def _update_transport_state(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self.position_slider.setEnabled(enabled)
        self.export_button.setEnabled(enabled and bool(self._stem_files))
        running = bool(self._worker and self._worker.isRunning())
        self.separate_button.setEnabled(enabled and not running)
        self.cancel_button.setEnabled(enabled and running)

    def _export_stems(self) -> None:
        if not self._stem_files:
            QtWidgets.QMessageBox.information(self, "No stems", "Run separation first.")
            return
        default_dir = self._input_path.parent if self._input_path else Path.cwd()
        dialog = ExportDialog(self._stem_files, self._stem_colors, default_dir, self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return

        output_dir = dialog.output_dir()
        output_format = dialog.output_format()
        output_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        selected = set(dialog.selected_stems())
        for stem, path in self._stem_files.items():
            if stem not in selected:
                continue
            if not os.path.exists(path):
                continue
            stem_path = Path(path)
            extension = ".wav" if output_format == "wav" else ".flac"
            target = output_dir / f"{stem_path.stem}{extension}"
            if output_format == "wav" and stem_path.suffix.lower() == ".wav":
                shutil.copy2(stem_path, target)
            else:
                data, sample_rate = sf.read(path, dtype="float32")
                sf.write(target, data, sample_rate, format=output_format.upper())
            exported += 1

        self.statusBar().showMessage(f"Exported {exported} stems to {output_dir}", 5000)

    def _update_vu_meters(self) -> None:
        if not self._player.is_playing:
            return
        levels = self._player.get_stem_levels()
        for stem, controls in self.stem_controls.items():
            volume_slider: VerticalStemSlider = controls["volume"]  # type: ignore[assignment]
            volume_slider.set_vu_level(levels.get(stem, 0.0))

    def _clear_vu_meters(self) -> None:
        for controls in self.stem_controls.values():
            volume_slider: VerticalStemSlider = controls["volume"]  # type: ignore[assignment]
            volume_slider.set_vu_level(0.0)

    def _cycle_theme(self) -> None:
        current_index = THEME_ORDER.index(self._theme_name)
        self._theme_name = THEME_ORDER[(current_index + 1) % len(THEME_ORDER)]
        self._apply_theme()

    def _toggle_theme_mode(self) -> None:
        self._theme_mode = "light" if self._theme_mode == "dark" else "dark"
        self._apply_theme()

    def _apply_theme(self) -> None:
        self._theme = resolve_theme(self._theme_name, self._theme_mode)
        self._theme_colors = {key: to_qcolor(value) for key, value in self._theme.items()}
        apply_theme(QtWidgets.QApplication.instance(), self._theme)

        self._update_theme_tooltip()
        self.logo_widget.set_colors(STEM_COLORS[:4], self._theme_colors["text"])
        self._background_widget.set_colors(
            self._theme_colors["bgGradientTop"],
            self._theme_colors["bgGradientBottom"],
        )
        self.waveform.set_theme(
            background=self._theme_colors["bgGradientTop"],
            border=self._theme_colors["border"],
            grid=self._theme_colors["textDim"],
            accent=self._theme_colors["accent"],
            text=self._theme_colors["text"],
        )
        for stem, controls in self.stem_controls.items():
            slider: VerticalStemSlider = controls["volume"]  # type: ignore[assignment]
            slider.set_color(self._stem_colors.get(stem, "#ffffff"))
        self._apply_button_styles()
        self.update()

    def _update_theme_tooltip(self) -> None:
        label = self._theme_name.capitalize()
        self.theme_button.setToolTip(f"Theme: {label} (click to cycle, right-click for light/dark)")

    def _format_time(self, seconds: float) -> str:
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.waitForFinished(3000)
        event.accept()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.exists():
                self._load_audio(path)
