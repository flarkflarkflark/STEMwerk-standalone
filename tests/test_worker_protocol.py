from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pytest

QtCore = pytest.importorskip("PySide6.QtCore")

from stemwerk.workers import SeparationWorker


@pytest.fixture(scope="module")
def qapp():
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication(sys.argv[:1])
    yield app


def _make_worker(tmp_path, **overrides):
    kwargs = dict(
        input_file=str(tmp_path / "in.wav"),
        output_dir=str(tmp_path / "out"),
        model="htdemucs",
        device="cpu",
        stems=["vocals"],
        # Unit tests never actually launch a process, but SeparationWorker's
        # constructor would otherwise call resolve_runtime_python(), which
        # depends on what runtime happens to be installed on the machine
        # running the tests. Inject fixed, machine-independent paths instead.
        runtime_python=Path(sys.executable),
        runner_script=Path("/fake/stemwerk/runner.py"),
    )
    kwargs.update(overrides)
    return SeparationWorker(**kwargs)


def test_start_builds_expected_process_arguments(qapp, tmp_path, monkeypatch) -> None:
    captured: List[Tuple[str, List[str]]] = []

    worker = _make_worker(
        tmp_path,
        model="htdemucs_6s",
        quality="best",
        stems=["vocals", "drums"],
    )
    monkeypatch.setattr(
        worker._process,
        "start",
        lambda program, arguments: captured.append((program, list(arguments))),
    )

    worker.start()

    assert len(captured) == 1
    program, arguments = captured[0]
    assert program == str(Path(sys.executable))
    assert arguments[0] == "/fake/stemwerk/runner.py"
    assert "--model" in arguments and arguments[arguments.index("--model") + 1] == "htdemucs_6s"
    assert "--quality" in arguments and arguments[arguments.index("--quality") + 1] == "best"
    assert arguments.count("--stem") == 2
    assert "vocals" in arguments
    assert "drums" in arguments


def test_start_uses_given_runtime_python_not_sys_executable(qapp, tmp_path, monkeypatch) -> None:
    captured: List[Tuple[str, List[str]]] = []

    other_python = tmp_path / "other-python"
    worker = _make_worker(tmp_path, runtime_python=other_python)
    monkeypatch.setattr(
        worker._process,
        "start",
        lambda program, arguments: captured.append((program, list(arguments))),
    )

    worker.start()

    program, _ = captured[0]
    assert program == str(other_python)
    assert program != sys.executable


def test_start_defaults_quality_to_normal(qapp, tmp_path, monkeypatch) -> None:
    captured: List[List[str]] = []

    worker = _make_worker(tmp_path)
    monkeypatch.setattr(
        worker._process,
        "start",
        lambda program, arguments: captured.append(list(arguments)),
    )

    worker.start()

    arguments = captured[0]
    assert arguments[arguments.index("--quality") + 1] == "normal"


def test_cancel_terminates_process_and_flags_cancel_requested(qapp, tmp_path, monkeypatch) -> None:
    worker = _make_worker(tmp_path)

    terminated = []
    monkeypatch.setattr(worker._process, "terminate", lambda: terminated.append(True))
    monkeypatch.setattr(worker, "isRunning", lambda: True)

    worker.cancel()

    assert terminated == [True]
    assert worker._cancel_requested is True


def test_cancel_noop_when_not_running(qapp, tmp_path, monkeypatch) -> None:
    worker = _make_worker(tmp_path)

    terminated = []
    monkeypatch.setattr(worker._process, "terminate", lambda: terminated.append(True))
    monkeypatch.setattr(worker, "isRunning", lambda: False)

    worker.cancel()

    assert terminated == []
    assert worker._cancel_requested is False


def test_cancelled_process_emits_cancelled_not_finished(qapp, tmp_path) -> None:
    worker = _make_worker(tmp_path)

    cancelled_calls: List[bool] = []
    finished_calls: List[dict] = []
    worker.cancelled.connect(lambda: cancelled_calls.append(True))
    worker.finished.connect(lambda payload: finished_calls.append(payload))

    worker._cancel_requested = True
    worker._on_process_finished(0, QtCore.QProcess.ExitStatus.NormalExit)

    assert cancelled_calls == [True]
    assert finished_calls == []


def test_completed_event_emits_finished_signal_with_stem_paths(qapp, tmp_path) -> None:
    worker = _make_worker(tmp_path)

    finished_calls: List[dict] = []
    worker.finished.connect(lambda payload: finished_calls.append(payload))

    worker._consume_event('{"protocol":1,"event":"completed","stems":{"vocals":"vocals.wav"}}')
    worker._on_process_finished(0, QtCore.QProcess.ExitStatus.NormalExit)

    assert finished_calls == [{"vocals": "vocals.wav"}]


def test_error_event_surfaces_message_when_process_fails(qapp, tmp_path) -> None:
    worker = _make_worker(tmp_path)

    error_calls: List[str] = []
    worker.error.connect(lambda message: error_calls.append(message))

    worker._consume_event('{"protocol":1,"event":"error","message":"boom"}')
    worker._on_process_finished(1, QtCore.QProcess.ExitStatus.NormalExit)

    assert error_calls == ["boom"]


def test_failed_to_start_reports_friendly_message(qapp, tmp_path) -> None:
    worker = _make_worker(tmp_path)

    error_calls: List[str] = []
    worker.error.connect(lambda message: error_calls.append(message))

    worker._on_process_error(QtCore.QProcess.ProcessError.FailedToStart)

    assert len(error_calls) == 1
    assert "runner could not be started" in error_calls[0]
