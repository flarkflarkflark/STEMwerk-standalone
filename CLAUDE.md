# STEMwerk Standalone

## Project overview
Standalone stem separation app built with PySide6. Uses stemwerk-core as the
separation engine.

## Build instructions

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

## Architecture overview
- `main_window.py` handles the UI, visual identity (theming, glossy buttons,
  waveform/stem mixer) and user actions.
- `workers.py` launches the isolated `stemwerk.runner` process and consumes
  JSONL events (progress, completion, cancellation, errors).
- `runner.py` is the headless JSONL process boundary around `stemwerk-core`.
- `glossy_button.py`, `logo_widget.py`, `stem_border.py`, `themes.py`
  implement the visual identity (glossy buttons, animated logo, stem-color
  border, theme presets).
- `player.py` handles audio playback and stem mixing.
- `stemwerk-core` performs the actual separation.

## Dependencies
- PySide6
- stemwerk-core
- sounddevice
- soundfile
- numpy


## Processing boundary

`runner.py` is the headless JSONL process boundary around `stemwerk-core`.
Do not add model or device implementations here. Keep this repository's
changes standalone-only; the REAPER repository and shared core remain
read-only unless a separate task explicitly authorizes changes.


# Claude Code Instructions

## Git authorship and commit policy

Never add Claude, Anthropic, Claude Sonnet, or any AI assistant as a commit author, committer, co-author, signer, or trailer.

Do not add lines like:

`Co-Authored-By: Claude Sonnet <noreply@anthropic.com>`
`Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`
`Co-Authored-By: Claude <noreply@anthropic.com>`
`Generated-by: Claude`
`AI-generated-by: Claude`

All commits must use the user's configured Git identity only.

Before creating a commit, always inspect the proposed commit message and ensure it contains no AI co-author, AI attribution, or Anthropic metadata.

When asked to commit, use normal commit messages only, for example:

`fix: improve setup repair flow`

Never append co-author trailers unless the user explicitly provides a human co-author line.

## History rewrite safety

Never rewrite public history, force-push, move tags, delete tags, or modify release tags without explicit confirmation from the user.

If a commit is reachable from a release tag, stop and report the risk before proceeding.

## Release safety

For public release assets, always include the version number in filenames.

Use consistent hyphenated names with no spaces.

Preferred format:

`ProjectName-vX.Y.Z-platform-architecture.ext`

Examples:

`STEMwerk-v2.2.2.1-Windows-x64.exe`
`THEMEwerk-v0.1.8-ReaPack.zip`
`DJwerk-v0.2.3-Linux-x64.AppImage`
