# STEMwerk Standalone

Standalone stem separation application for the flarkAUDIO ecosystem.
Powered by PySide6 and stemwerk-core.

## Install

```bash
pip install -e .
pip install -e P:\GIT\STEMwerk-core
```

## Run

```bash
python -m stemwerk.main
```

Or after install:

```bash
stemwerk
```

## Features

- Open audio files (wav, mp3, flac, ogg)
- Visual waveform with stem overlays
- Stem separation with progress updates
- Stem mixing with solo, mute, and volume
- Export stems as WAV


## Processing architecture

The standalone UI launches separation through an isolated headless runner:

    PySide6 UI -> SeparationWorker -> python -m stemwerk.runner -> stemwerk-core

The runner writes versioned JSONL events to stdout and keeps backend/library
diagnostics on stderr. This keeps the UI responsive and makes cancellation,
failure reporting and later queue support possible without creating a second
separation backend.

Current protocol events include:

- `started`
- `progress`
- `completed`
- `error`

The runner is an integration boundary around `stemwerk-core`; model, device
and separation behavior remain owned by the shared core. The REAPER integration
is not changed by this repository.

## Development checks

    python -m py_compile src/stemwerk/*.py
    python -m pytest -q

A real GPU separation is platform validation and must be recorded separately
for Linux ROCm, Windows CUDA/DirectML, macOS CPU/MPS and CPU fallback routes.
