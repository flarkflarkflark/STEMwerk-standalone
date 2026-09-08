from __future__ import annotations

import json

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


def test_runner_event_is_jsonl(monkeypatch, capsys) -> None:
    emit_event("started", model="htdemucs", device="cpu")

    record = json.loads(capsys.readouterr().out)
    assert record == {
        "protocol": PROTOCOL_VERSION,
        "event": "started",
        "model": "htdemucs",
        "device": "cpu",
    }
