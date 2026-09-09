from __future__ import annotations

import sys
from pathlib import Path

import pytest

from stemwerk.runtime import (
    RUNTIME_ENV_VAR,
    RuntimeResolutionError,
    canonical_runtime_python,
    probe_runtime,
    resolve_runtime_python,
    runner_script_path,
)


def test_runner_script_path_is_next_to_this_module() -> None:
    path = runner_script_path()
    assert path.name == "runner.py"
    assert path.parent.name == "stemwerk"
    assert path.is_file()


def test_canonical_runtime_python_linux(monkeypatch) -> None:
    monkeypatch.setattr("stemwerk.runtime.platform.system", lambda: "Linux")
    path = canonical_runtime_python()
    assert path == Path.home() / ".local" / "share" / "STEMwerk" / ".venv" / "bin" / "python3"


def test_canonical_runtime_python_macos(monkeypatch) -> None:
    monkeypatch.setattr("stemwerk.runtime.platform.system", lambda: "Darwin")
    path = canonical_runtime_python()
    assert path == Path("/Users/Shared/STEMwerk/.venv/bin/python3")


def test_canonical_runtime_python_windows(monkeypatch) -> None:
    monkeypatch.setattr("stemwerk.runtime.platform.system", lambda: "Windows")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Ben\AppData\Local")
    path = canonical_runtime_python()
    assert path == Path(r"C:\Users\Ben\AppData\Local") / "STEMwerk" / ".venv" / "Scripts" / "python.exe"


def test_canonical_runtime_python_windows_without_localappdata(monkeypatch) -> None:
    monkeypatch.setattr("stemwerk.runtime.platform.system", lambda: "Windows")
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    with pytest.raises(RuntimeResolutionError):
        canonical_runtime_python()


def test_resolve_prefers_valid_override(monkeypatch, tmp_path) -> None:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("")
    monkeypatch.setenv(RUNTIME_ENV_VAR, str(fake_python))

    assert resolve_runtime_python() == fake_python


def test_resolve_rejects_invalid_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(RUNTIME_ENV_VAR, str(tmp_path / "does-not-exist"))

    with pytest.raises(RuntimeResolutionError):
        resolve_runtime_python()


def test_resolve_uses_canonical_path_when_no_override(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(RUNTIME_ENV_VAR, raising=False)
    canonical = tmp_path / "canonical-python"
    canonical.write_text("")
    monkeypatch.setattr("stemwerk.runtime.canonical_runtime_python", lambda: canonical)

    assert resolve_runtime_python() == canonical


def test_resolve_raises_with_helpful_message_when_canonical_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(RUNTIME_ENV_VAR, raising=False)
    missing = tmp_path / "missing-python"
    monkeypatch.setattr("stemwerk.runtime.canonical_runtime_python", lambda: missing)

    with pytest.raises(RuntimeResolutionError) as excinfo:
        resolve_runtime_python()

    assert str(missing) in str(excinfo.value)
    assert RUNTIME_ENV_VAR in str(excinfo.value)


_FAKE_RUNNER_TEMPLATE = """\
import json
print(json.dumps({{"protocol": 1, "event": "capabilities", "models": {models!r}, "qualities": {qualities!r}, "devices": {devices!r}, "core_version": {core_version!r}}}))
"""


def _write_fake_runner(tmp_path: Path, *, models=None, qualities=None, devices=None, core_version="0.0.0") -> Path:
    script = tmp_path / "fake_runner.py"
    script.write_text(
        _FAKE_RUNNER_TEMPLATE.format(
            models=models if models is not None else ["htdemucs"],
            qualities=qualities if qualities is not None else ["fast", "normal", "best"],
            devices=devices if devices is not None else [{"id": "auto", "name": "Auto"}],
            core_version=core_version,
        )
    )
    return script


def test_probe_runtime_success(tmp_path) -> None:
    script = _write_fake_runner(tmp_path, models=["htdemucs", "htdemucs_6s"], qualities=["fast", "normal", "best"])

    result = probe_runtime(Path(sys.executable), runner_script=script)

    assert result.ok is True
    assert result.error is None
    assert result.capabilities is not None
    assert result.capabilities.models == ["htdemucs", "htdemucs_6s"]
    assert result.capabilities.qualities == ["fast", "normal", "best"]
    assert result.capabilities.core_version == "0.0.0"


def test_probe_runtime_missing_python(tmp_path) -> None:
    script = _write_fake_runner(tmp_path)

    result = probe_runtime(tmp_path / "no-such-python", runner_script=script)

    assert result.ok is False
    assert "not found" in result.error


def test_probe_runtime_missing_script(tmp_path) -> None:
    result = probe_runtime(Path(sys.executable), runner_script=tmp_path / "no-such-runner.py")

    assert result.ok is False
    assert "not found" in result.error


def test_probe_runtime_no_capabilities_event(tmp_path) -> None:
    script = tmp_path / "silent_runner.py"
    script.write_text("print('not json')\n")

    result = probe_runtime(Path(sys.executable), runner_script=script)

    assert result.ok is False
    assert "capabilities" in result.error


def test_probe_runtime_empty_qualities_is_not_ok(tmp_path) -> None:
    script = _write_fake_runner(tmp_path, qualities=[])

    result = probe_runtime(Path(sys.executable), runner_script=script)

    assert result.ok is False


def test_probe_runtime_times_out(tmp_path) -> None:
    script = tmp_path / "slow_runner.py"
    script.write_text("import time\ntime.sleep(2)\n")

    result = probe_runtime(Path(sys.executable), runner_script=script, timeout=0.2)

    assert result.ok is False
    assert "timed out" in result.error
