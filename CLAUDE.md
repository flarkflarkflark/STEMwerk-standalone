# STEMwerk Standalone

## Project overview
Standalone stem separation app built with PySide6. Uses stemwerk-core as the
separation engine.

## Build instructions
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

## Architecture overview
- `main_window.py` handles the UI and user actions.
- `workers.py` contains QThread workers for separation.
- `player.py` handles audio playback and stem mixing.
- `stemwerk-core` performs the actual separation.

## Dependencies
- PySide6
- stemwerk-core
- sounddevice
- soundfile
- numpy
