from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional


PROTOCOL_VERSION = 1


def emit_event(event: str, **payload: Any) -> None:
    record = {"protocol": PROTOCOL_VERSION, "event": event, **payload}
    sys.stdout.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STEMwerk headless separation runner")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quality", default="normal")
    parser.add_argument("--stem", action="append", dest="stems")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Report available models/devices/quality presets as a capabilities event and exit.",
    )
    return parser


def probe() -> int:
    """Report stemwerk_core's capabilities without running any separation.

    This is the only channel the GUI process uses to learn about models,
    devices and quality presets -- it lets the GUI stay free of a direct
    stemwerk_core/PyTorch dependency while still reflecting the real
    processing runtime's capabilities.
    """
    from stemwerk_core import get_available_devices
    from stemwerk_core.models import AVAILABLE_MODELS

    try:
        from stemwerk_core.separator import QUALITY_PRESETS

        qualities = list(QUALITY_PRESETS)
    except ImportError:
        qualities = ["fast", "normal", "best"]

    try:
        from importlib.metadata import version

        core_version = version("stemwerk-core")
    except Exception:
        core_version = "unknown"

    emit_event(
        "capabilities",
        models=list(AVAILABLE_MODELS),
        qualities=qualities,
        devices=get_available_devices(),
        core_version=core_version,
    )
    return 0


def run(
    input_file: Path,
    output_dir: Path,
    model: str,
    device: str,
    quality: str,
    stems: Optional[Iterable[str]],
) -> int:
    # Keep the protocol stream machine-readable. Backend/library chatter is
    # retained on stderr for diagnostics instead of corrupting JSONL stdout.
    from stemwerk_core import StemSeparator

    protocol_stdout = sys.stdout

    def progress(percent: float, message: str) -> None:
        record = {
            "protocol": PROTOCOL_VERSION,
            "event": "progress",
            "percent": float(percent),
            "message": str(message),
        }
        protocol_stdout.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        protocol_stdout.flush()

    emit_event("started", input=str(input_file), model=model, device=device, quality=quality)
    separator = StemSeparator(model=model, device=device, quality=quality)
    separator.on_progress = progress

    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = separator.separate(
                input_file=input_file,
                output_dir=output_dir,
                stems=list(stems) if stems is not None else None,
            )
    except Exception as exc:
        emit_event(
            "error",
            message=str(exc),
            error_type=type(exc).__name__,
            traceback="".join(traceback.format_exception(exc)),
            quality=quality,
        )
        return 1

    emit_event(
        "completed",
        stems={name: str(path) for name, path in result.stems.items()},
        device_used=result.device_used,
        elapsed=float(result.elapsed),
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.probe:
        return probe()

    if args.input is None or args.output_dir is None:
        parser.error("--input and --output-dir are required unless --probe is set")

    return run(
        input_file=args.input,
        output_dir=args.output_dir,
        model=args.model,
        device=args.device,
        quality=args.quality,
        stems=args.stems,
    )


if __name__ == "__main__":
    raise SystemExit(main())
