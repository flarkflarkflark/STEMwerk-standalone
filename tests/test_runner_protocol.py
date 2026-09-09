from __future__ import annotations

import json
import sys
import types

import pytest

from stemwerk import runner
from stemwerk.runner import PROTOCOL_VERSION, build_parser, emit_event


def test_runner_parser_accepts_multiple_stems() -> None:
    args = build_parser().parse_args(
        [
            "--input",
            "input.wav",
            "--output-dir",
            "stems",
            "--model",
            "htdemucs",
            "--device",
            "cpu",
            "--stem",
            "vocals",
            "--stem",
            "drums",
        ]
    )

    assert args.model == "htdemucs"
    assert args.device == "cpu"
    assert args.stems == ["vocals", "drums"]
    assert args.quality == "normal"


def test_runner_parser_accepts_quality() -> None:
    args = build_parser().parse_args(
        [
            "--input",
            "input.wav",
            "--output-dir",
            "stems",
            "--quality",
            "best",
        ]
    )

    assert args.quality == "best"


def test_runner_event_is_jsonl(monkeypatch, capsys) -> None:
    emit_event("started", model="htdemucs", device="cpu")

    record = json.loads(capsys.readouterr().out)
    assert record == {
        "protocol": PROTOCOL_VERSION,
        "event": "started",
        "model": "htdemucs",
        "device": "cpu",
    }


class _FakeSeparationResult:
    def __init__(self, stems: dict, device_used: str, elapsed: float) -> None:
        self.stems = stems
        self.device_used = device_used
        self.elapsed = elapsed


class _FakeStemSeparator:
    """Stand-in for stemwerk_core.StemSeparator.

    stemwerk-core is not installed in this analysis/dev environment, so runner.run()
    is exercised against a fake module injected into sys.modules instead.
    """

    last_instance: "_FakeStemSeparator | None" = None

    def __init__(self, model: str, device: str, quality: str) -> None:
        self.model = model
        self.device = device
        self.quality = quality
        self.on_progress = None
        _FakeStemSeparator.last_instance = self

    def separate(self, input_file, output_dir, stems=None):
        return _FakeSeparationResult(
            stems={"vocals": str(output_dir) + "/vocals.wav"},
            device_used=self.device,
            elapsed=0.01,
        )


class _FakeFailingStemSeparator(_FakeStemSeparator):
    def separate(self, input_file, output_dir, stems=None):
        raise RuntimeError("boom")


@pytest.fixture
def fake_stemwerk_core(monkeypatch):
    def _install(separator_cls):
        fake_module = types.ModuleType("stemwerk_core")
        fake_module.StemSeparator = separator_cls
        monkeypatch.setitem(sys.modules, "stemwerk_core", fake_module)
        return separator_cls

    return _install


def test_run_passes_quality_to_stem_separator(tmp_path, capsys, fake_stemwerk_core) -> None:
    fake_stemwerk_core(_FakeStemSeparator)

    exit_code = runner.run(
        input_file=tmp_path / "in.wav",
        output_dir=tmp_path / "out",
        model="htdemucs",
        device="cpu",
        quality="best",
        stems=["vocals"],
    )

    assert exit_code == 0
    assert _FakeStemSeparator.last_instance is not None
    assert _FakeStemSeparator.last_instance.quality == "best"

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    started = next(e for e in events if e["event"] == "started")
    assert started["quality"] == "best"
    completed = next(e for e in events if e["event"] == "completed")
    assert completed["stems"]["vocals"].endswith("vocals.wav")


def test_run_reports_quality_in_error_event(tmp_path, capsys, fake_stemwerk_core) -> None:
    fake_stemwerk_core(_FakeFailingStemSeparator)

    exit_code = runner.run(
        input_file=tmp_path / "in.wav",
        output_dir=tmp_path / "out",
        model="htdemucs",
        device="cpu",
        quality="fast",
        stems=None,
    )

    assert exit_code == 1
    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    error_event = next(e for e in events if e["event"] == "error")
    assert error_event["quality"] == "fast"
    assert error_event["message"] == "boom"


def test_main_forwards_quality_argument(tmp_path, capsys, fake_stemwerk_core) -> None:
    fake_stemwerk_core(_FakeStemSeparator)

    exit_code = runner.main(
        [
            "--input",
            str(tmp_path / "in.wav"),
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "htdemucs",
            "--device",
            "cpu",
            "--quality",
            "fast",
            "--stem",
            "vocals",
        ]
    )

    assert exit_code == 0
    assert _FakeStemSeparator.last_instance is not None
    assert _FakeStemSeparator.last_instance.quality == "fast"
