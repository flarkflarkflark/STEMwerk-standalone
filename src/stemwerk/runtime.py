from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

RUNTIME_ENV_VAR = "STEMWERK_RUNTIME_PYTHON"
PROBE_TIMEOUT_SECONDS = 20.0


class RuntimeResolutionError(RuntimeError):
    """Raised when no usable STEMwerk processing runtime can be located."""


def runner_script_path() -> Path:
    """Path to runner.py, resolved next to this module.

    This works unchanged whether stemwerk is running from an editable
    development checkout (src/stemwerk/runner.py) or from an installed
    package (.../site-packages/stemwerk/runner.py) -- the processing
    runtime is invoked as `<runtime_python> <this path> ...`, so it never
    needs the `stemwerk` package itself installed, only stemwerk_core.
    """
    return Path(__file__).resolve().with_name("runner.py")


def canonical_runtime_python() -> Path:
    """The per-platform canonical path to the product-wide STEMwerk processing
    runtime. This runtime belongs to the STEMwerk product as a whole (model
    separation, stemwerk-core, PyTorch) and is independent of any single
    STEMwerk application -- standalone, the REAPER integration, or others.
    """
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            raise RuntimeResolutionError(
                "LOCALAPPDATA is not set; cannot locate the STEMwerk runtime."
            )
        return Path(base) / "STEMwerk" / ".venv" / "Scripts" / "python.exe"
    if system == "Darwin":
        return Path("/Users/Shared/STEMwerk/.venv/bin/python3")
    return Path.home() / ".local" / "share" / "STEMwerk" / ".venv" / "bin" / "python3"


def resolve_runtime_python() -> Path:
    """Resolve which Python executable runs stemwerk_core-based separation.

    Resolution order:
    1. The STEMWERK_RUNTIME_PYTHON environment variable (explicit dev/test
       override) -- must point at an existing file.
    2. The canonical per-platform STEMwerk product runtime path.

    Deliberately never falls back to sys.executable or any interpreter found
    on PATH: a wrong interpreter would silently lack stemwerk_core/PyTorch
    and fail deep inside a background subprocess instead of with a clear,
    actionable message here.
    """
    override = os.environ.get(RUNTIME_ENV_VAR)
    if override:
        path = Path(override)
        if not path.is_file():
            raise RuntimeResolutionError(
                f"{RUNTIME_ENV_VAR} is set to '{override}', but no file exists there."
            )
        return path

    path = canonical_runtime_python()
    if not path.is_file():
        raise RuntimeResolutionError(
            "No STEMwerk processing runtime found.\n"
            f"Checked: {path}\n"
            f"Set {RUNTIME_ENV_VAR} to a Python executable with stemwerk-core "
            "installed, or install/repair the STEMwerk runtime."
        )
    return path


@dataclass
class RuntimeCapabilities:
    models: List[str] = field(default_factory=list)
    qualities: List[str] = field(default_factory=list)
    devices: List[Dict[str, Any]] = field(default_factory=list)
    core_version: str = "unknown"


@dataclass
class RuntimeProbeResult:
    ok: bool
    python_path: Path
    capabilities: Optional[RuntimeCapabilities] = None
    error: Optional[str] = None


def probe_runtime(
    python_path: Path,
    runner_script: Optional[Path] = None,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> RuntimeProbeResult:
    """Validate a candidate runtime by running `<python_path> <runner_script> --probe`.

    This is the only way the GUI process learns about available models,
    devices and quality presets: it never imports stemwerk_core itself. A
    successful "capabilities" JSONL event is proof the executable exists,
    starts, can import stemwerk_core, and exposes the model/device/quality
    interface the runner and GUI depend on.
    """
    script = Path(runner_script) if runner_script is not None else runner_script_path()
    python_path = Path(python_path)

    if not python_path.is_file():
        return RuntimeProbeResult(ok=False, python_path=python_path, error=f"Runtime Python not found: {python_path}")
    if not script.is_file():
        return RuntimeProbeResult(ok=False, python_path=python_path, error=f"Runner script not found: {script}")

    try:
        completed = subprocess.run(
            [str(python_path), str(script), "--probe"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except OSError as exc:
        return RuntimeProbeResult(ok=False, python_path=python_path, error=f"Could not start runtime Python: {exc}")
    except subprocess.TimeoutExpired:
        return RuntimeProbeResult(ok=False, python_path=python_path, error="Runtime probe timed out.")

    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "capabilities":
            continue
        qualities = list(event.get("qualities", []))
        if not qualities:
            return RuntimeProbeResult(ok=False, python_path=python_path, error="Runtime capabilities reported no quality presets.")
        capabilities = RuntimeCapabilities(
            models=list(event.get("models", [])),
            qualities=qualities,
            devices=list(event.get("devices", [])),
            core_version=str(event.get("core_version", "unknown")),
        )
        return RuntimeProbeResult(ok=True, python_path=python_path, capabilities=capabilities)

    detail = completed.stderr.strip() or f"exit code {completed.returncode}"
    return RuntimeProbeResult(ok=False, python_path=python_path, error=f"Runtime probe produced no capabilities event ({detail}).")
