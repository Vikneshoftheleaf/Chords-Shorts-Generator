# Music Shorts

A Python CLI worker that deterministically selects songs, synthesizes simple chord/melody audio, and renders an original falling-note visualization with the full 88-key piano visible. There is no YouTube integration or automatic posting.

## Dataset format and rights

`dataset/Hooktheory.json.gz` is the 20 MB Hooktheory archive from [Sheet Sage Data](https://github.com/chrisdonahue/sheetsage-data). It contains native records with beat-aligned `melody` and `harmony` annotations. The CLI uses the refined alignment when available, falls back to user alignment, and skips records without both annotation tracks and usable timing anchors. It converts beat positions to seconds and synthesizes notes; it does not use or download the referenced YouTube audio.

The Sheet Sage repository's [licensing notes](https://github.com/chrisdonahue/sheetsage#licensing-considerations) state that its dataset is CC BY-NC-SA 3.0. The user reports having separate written permission covering the intended commercial/monetized use. Keep that permission available and confirm it covers both redistribution of the archive and publication of derived videos.

Attribution: Chris Donahue, John Thickstun, and Percy Liang, “Melody transcription via generative pre-training,” ISMIR 2022; [Sheet Sage Data](https://github.com/chrisdonahue/sheetsage-data).

For development and tests, the parser also accepts this deliberately small normalized JSON schema compressed with gzip:

```json
{
  "songs": [{
    "id": "unique-id",
    "title": "Song title",
    "tempo": 96,
    "chords": [{"start": 0.0, "duration": 0.8, "notes": [48, 55, 60]}],
    "melody": [{"start": 0.1, "duration": 0.35, "pitch": 72}]
  }]
}
```

Times are seconds; pitches are MIDI integers 0–127. Chord and melody event lists must be non-empty. This normalized format is used only by the synthetic test fixture.

## Setup and use

Requires Python 3.10+. FFmpeg is supplied through the `imageio-ffmpeg` Python dependency.

```sh
python -m pip install -e ".[test]"
pytest
music-shorts --dataset dataset/Hooktheory.json.gz --count 1
```

The CLI defaults to one output per invocation:

```text
music-shorts [--dataset PATH] [--output-dir PATH] [--ledger PATH]
             [--count N] [--max-seconds 1..60]
```

Generated MP4s go to `outputs/`; the persistent SQLite duplicate ledger defaults to `state/generated.sqlite3`. The CLI only writes an ID to the ledger after its video renders successfully. Keep repository size in mind because videos are binary files.

## GitHub Actions

`.github/workflows/generate.yml` is manually dispatched, defaults to one video, and commits outputs plus the SQLite ledger only when they changed. Before use, add this project to a GitHub repository and enable **Settings → Actions → General → Workflow permissions → Read and write permissions**. The workflow only requests `contents: write`, serializes runs per branch, and pushes to the checked-out dispatch branch. YouTube secrets and cloud services are not used.

The real dataset is included, so the default CLI can generate a video locally. The first run may take longer because the archive contains more than 26,000 source records. Generated previews are stored in `outputs/` and their song IDs are recorded in `state/generated.sqlite3`; use the synthetic fixture in `tests/` for fast, repeatable tests.
