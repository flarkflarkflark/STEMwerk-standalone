from __future__ import annotations

from typing import Dict, List, Optional

from PySide6 import QtCore

from stemwerk_core import StemSeparator


class SeparationWorker(QtCore.QThread):
    progress_updated = QtCore.Signal(float, str)
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(
        self,
        input_file: str,
        output_dir: str,
        model: str,
        device: str,
        stems: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self._input_file = input_file
        self._output_dir = output_dir
        self._model = model
        self._device = device
        self._stems = stems

    def run(self) -> None:
        try:
            separator = StemSeparator(model=self._model, device=self._device)

            def on_progress(percent: float, message: str) -> None:
                self.progress_updated.emit(percent, message)

            separator.on_progress = on_progress
            result = separator.separate(
                self._input_file,
                self._output_dir,
                stems=self._stems,
            )
            stem_paths: Dict[str, str] = {name: str(path) for name, path in result.stems.items()}
            self.finished.emit(stem_paths)
        except Exception as exc:
            self.error.emit(str(exc))
