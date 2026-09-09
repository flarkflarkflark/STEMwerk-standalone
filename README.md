# STEMwerk Standalone

Standalone stem separation application for the flarkAUDIO ecosystem.
Powered by PySide6 and stemwerk-core.

## Install

Install this package and your local `stemwerk-core` checkout in editable mode:

```bash
pip install -e .
pip install -e /path/to/STEMwerk-core        # Linux/macOS
pip install -e C:\path\to\STEMwerk-core      # Windows
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
- Visual waveform with stem overlays, glossy-button UI, animated logo and
  stem-color border, with 4 theme presets in dark/light mode
- Stem separation with progress updates, powered by an isolated headless
  runner process (see "Processing architecture" below)
- Model selection from the shared `stemwerk_core.models.AVAILABLE_MODELS`
  registry
- Quality selection (Fast / Normal / Best), passed through to
  `StemSeparator(quality=...)`
- Explicit Auto/CPU device selection plus any hardware devices reported by
  `stemwerk_core.get_available_devices()`
- Per-stem selection (checkboxes plus `1`-`4` shortcuts) to choose which
  stems are returned from separation
- Configurable stem output folder ("Stem folder" button), defaulting to
  `<input>_stems` next to the source file
- Cancel button to stop an in-progress separation; closing the window during
  separation cancels the runner process instead of leaving it hanging
- Stem mixing with solo, mute, and volume; loading of completed separation
  outputs back into the mixer/waveform
- Export stems as WAV or FLAC

## Processing architecture

The standalone UI launches separation through an isolated headless runner —
the GUI never calls `StemSeparator` directly:

    PySide6 UI -> SeparationWorker (QProcess) -> python -m stemwerk.runner -> stemwerk-core

The runner writes versioned JSONL events to stdout and keeps backend/library
diagnostics on stderr. This keeps the UI responsive and makes cancellation,
failure reporting and later queue support possible without creating a second
separation backend.

Current protocol events include:

- `started` (includes the selected model, device and quality)
- `progress`
- `completed`
- `error` (includes quality for diagnostics)

Cancelling from the UI calls `terminate()` on the runner process and
force-kills it after a 2.5s timeout if it has not exited. Closing the main
window while a separation is running cancels the runner and waits briefly
for it to exit before the window closes.

The runner is an integration boundary around `stemwerk-core`; model, device
and separation behavior remain owned by the shared core. The REAPER
integration is not changed by this repository.

## Development checks

    python -m py_compile src/stemwerk/*.py
    python -m pytest -q

## Validation status

- **Linux**: `python -m py_compile` and the full `pytest` suite (including
  targeted runner/worker contract tests for `--quality`, multiple `--stem`
  arguments, JSONL event shape, quality passthrough, and completion/error/
  cancel signal handling) pass in a minimal environment without
  `stemwerk-core`/`soundfile`/`sounddevice` installed. A GUI smoke test
  (open audio, separate, cancel, export, theming) additionally requires
  those three dependencies; see the repository's own record of that run for
  the current result.
- **Windows**: NOT TESTED. Planned for D2.
- **macOS**: NOT TESTED. Planned for D2.

A real GPU separation is platform validation and must be recorded separately
for Linux ROCm, Windows CUDA/DirectML, macOS CPU/MPS and CPU fallback routes.
