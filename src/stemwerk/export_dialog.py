from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6 import QtWidgets


class ExportDialog(QtWidgets.QDialog):
    def __init__(
        self,
        stems: Dict[str, str],
        colors: Dict[str, str],
        default_dir: Path,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Stems")
        self._stem_checks: Dict[str, QtWidgets.QCheckBox] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        stems_box = QtWidgets.QGroupBox("Stems")
        stems_layout = QtWidgets.QGridLayout(stems_box)
        stems_layout.setContentsMargins(8, 8, 8, 8)
        stems_layout.setHorizontalSpacing(8)
        stems_layout.setVerticalSpacing(6)

        for row, stem in enumerate(stems.keys()):
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(True)
            color = QtWidgets.QLabel()
            color.setFixedSize(14, 14)
            color.setStyleSheet(
                f"background: {colors.get(stem, '#ffffff')}; border: 1px solid #000000;"
            )
            name = QtWidgets.QLabel(stem.capitalize())
            stems_layout.addWidget(checkbox, row, 0)
            stems_layout.addWidget(color, row, 1)
            stems_layout.addWidget(name, row, 2)
            self._stem_checks[stem] = checkbox

        format_row = QtWidgets.QHBoxLayout()
        format_row.addWidget(QtWidgets.QLabel("Format:"))
        self._format_combo = QtWidgets.QComboBox()
        self._format_combo.addItems(["WAV", "FLAC"])
        format_row.addWidget(self._format_combo)
        format_row.addStretch(1)

        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(QtWidgets.QLabel("Output folder:"))
        self._path_edit = QtWidgets.QLineEdit(str(default_dir))
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self._browse_output)
        path_row.addWidget(self._path_edit, 1)
        path_row.addWidget(browse_button)

        buttons = QtWidgets.QDialogButtonBox()
        self._export_button = buttons.addButton("Export", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Cancel", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        self._export_button.clicked.connect(self._on_export)
        buttons.rejected.connect(self.reject)

        layout.addWidget(stems_box)
        layout.addLayout(format_row)
        layout.addLayout(path_row)
        layout.addWidget(buttons)

    def selected_stems(self) -> List[str]:
        return [stem for stem, checkbox in self._stem_checks.items() if checkbox.isChecked()]

    def output_dir(self) -> Path:
        return Path(self._path_edit.text()).expanduser()

    def output_format(self) -> str:
        return self._format_combo.currentText().lower()

    def _browse_output(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if folder:
            self._path_edit.setText(folder)

    def _on_export(self) -> None:
        if not self.selected_stems():
            QtWidgets.QMessageBox.warning(self, "No stems selected", "Select at least one stem to export.")
            return
        if not self._path_edit.text().strip():
            QtWidgets.QMessageBox.warning(self, "No folder", "Select an output folder.")
            return
        self.accept()
