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
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quality", default="normal")
    parser.add_argument("--stem", action="append", dest="stems")
    return parser


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
    args = build_parser().parse_args(argv)
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
