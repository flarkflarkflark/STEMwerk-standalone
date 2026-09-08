from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PySide6 import QtCore


class SeparationWorker(QtCore.QObject):
    """Run separation outside the Qt process and consume JSONL events."""

    progress_updated = QtCore.Signal(float, str)
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)
    cancelled = QtCore.Signal()

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
        self._stems = stems or []
        self._cancel_requested = False
        self._completed_payload: Optional[Dict[str, object]] = None
        self._protocol_buffer = ""
        self._stderr_buffer = ""

        self._process = QtCore.QProcess(self)
        self._process.setProcessChannelMode(
            QtCore.QProcess.ProcessChannelMode.SeparateChannels
        )
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.errorOccurred.connect(self._on_process_error)
        self._process.finished.connect(self._on_process_finished)

    def start(self) -> None:
        arguments = [
            "-m",
            "stemwerk.runner",
            "--input",
            self._input_file,
            "--output-dir",
            self._output_dir,
            "--model",
            self._model,
            "--device",
            self._device,
        ]
        for stem in self._stems:
            arguments.extend(["--stem", stem])
        self._process.start(sys.executable, arguments)

    def isRunning(self) -> bool:
        return self._process.state() != QtCore.QProcess.ProcessState.NotRunning

    def cancel(self) -> None:
        if not self.isRunning():
            return
        self._cancel_requested = True
        self._process.terminate()
        QtCore.QTimer.singleShot(2500, self._kill_if_running)

    def waitForFinished(self, timeout_ms: int) -> bool:
        return self._process.waitForFinished(timeout_ms)

    def _kill_if_running(self) -> None:
        if self.isRunning():
            self._process.kill()

    def _read_stdout(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self._protocol_buffer += data
        while "\n" in self._protocol_buffer:
            line, self._protocol_buffer = self._protocol_buffer.split("\n", 1)
            self._consume_event(line)

    def _read_stderr(self) -> None:
        data = bytes(self._process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )
        if data:
            self._stderr_buffer += data

    def _consume_event(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            self._stderr_buffer += f"\n[runner protocol] {line}"
            return

        kind = event.get("event")
        if kind == "progress":
            self.progress_updated.emit(
                float(event.get("percent", 0.0)),
                str(event.get("message", "")),
            )
        elif kind == "completed":
            stems = event.get("stems", {})
            self._completed_payload = dict(stems) if isinstance(stems, dict) else {}
        elif kind == "error":
            message = str(event.get("message", "Separation failed."))
            self._stderr_buffer += f"\n{message}"

    def _on_process_error(self, error: QtCore.QProcess.ProcessError) -> None:
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.error.emit(
                "The STEMwerk processing runner could not be started. "
                "Check the Python/runtime installation."
            )

    def _on_process_finished(
        self,
        exit_code: int,
        exit_status: QtCore.QProcess.ExitStatus,
    ) -> None:
        self._read_stdout()
        self._read_stderr()

        if self._cancel_requested:
            self.cancelled.emit()
            return

        if exit_code == 0 and self._completed_payload is not None:
            self.finished.emit(self._completed_payload)
            return

        detail = self._stderr_buffer.strip()
        self.error.emit(detail or f"Processing runner exited with code {exit_code}.")
