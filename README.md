# Music Shorts

A Python CLI worker that deterministically selects songs, synthesizes stereo sampled-piano chord/melody audio, and renders vertical 9:16 piano-roll videos. Each video opens on a dimmed piano-roll background with glowing cartoon cover art, comic-style track details, and phrase-by-phrase Piper voice captions. The voice is slowed slightly; after it ends, the video uses a one-second pause that fades smoothly into the chord progression. The progression ends with the final chord; a reference-inspired like/comment/share row and subscribe button hold over the piano roll for three seconds before the video ends. Comic lettering is used throughout. There is no YouTube integration or automatic posting.

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

## SoundFont and audio

The default renderer uses the [FreePats YDP Grand Piano SoundFont](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html) through FluidSynth. Built from Yamaha Disklavier Pro multisamples, it is published under [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/). Attribute FreePats/Roberto G. and the Zenph Studios multisamples (performed by Dr. Mikhail Krishtal); the accompanying license/attribution text is `assets/soundfonts/YDP-GrandPiano-license.txt`. Its compressed archive is in `assets/soundfonts/`; the generator extracts the SF2 to a local temporary cache on first use.

Artwork uses the public iTunes Search API and only accepts exact normalized artist and track-name matches. Network errors, missing matches, and tracks without artist metadata use a generated local fallback cover, so rendering can continue offline.

FluidSynth renders the sampled piano with velocity variation, stereo samples, chorus disabled, and restrained room reverb. GitHub Actions installs it automatically. On Windows, if FluidSynth is not already on `PATH`, the generator downloads the official FluidSynth 2.6.2 portable build, verifies its SHA-256 checksum, and caches it under the current user's local application data. On Ubuntu, install it with `sudo apt-get install fluidsynth`. If FluidSynth is unavailable, the built-in procedural piano is available with `--procedural-synth`.

Intro narration uses the Piper `en_US-norman-medium` male voice. The model card identifies the LibriVox training recordings as public domain; the model is downloaded on first use, checksum-verified, and cached locally (the GitHub Actions workflow also caches it). Model details and attribution are in `assets/voices/en_US-norman-medium-MODEL_CARD.md`. The intro says, “Here is the chord progression for {track} by {artist},” with a slightly faster speaking rate and slightly increased voice volume; captions are timed to each phrase. A one-second visual fade separates the end of narration from the piano progression. Video lettering uses the Bangers font, distributed under SIL Open Font License 1.1 in `assets/fonts/`.

## Setup and use

Requires Python 3.10+, Piper TTS, and FFmpeg (supplied through the `imageio-ffmpeg` Python dependency). FluidSynth is installed by the GitHub Actions workflow and downloaded automatically on Windows when missing. The first narrated generation downloads the 63.5 MB medium voice model.

```sh
python -m pip install -e ".[test]"
pytest
music-shorts --dataset dataset/Hooktheory.json.gz --count 1
```

To test the full generation pipeline against the real dataset and keep a preview in `outputs/` without modifying the normal ledger:

```sh
python scripts/test_generation.py
```

The test generates an MP4 with up to 60 seconds of piano progression, plus the narrated intro and three-second outro, in `outputs/`. It verifies that the MP4 can be decoded, is portrait at 30 fps, and contains stereo AAC audio; it also checks duplicate selection using a temporary SQLite ledger. The MP4 is kept; the temporary ledger is removed. It skips songs with an existing video so it won't overwrite previous output. Use `--seconds 10`, `--output-dir PATH`, or `--dataset PATH` to customize the run.

The CLI defaults to one output per invocation:

```text
music-shorts [--dataset PATH] [--output-dir PATH] [--ledger PATH]
             [--count N] [--max-seconds 1..60] [--soundfont PATH]
             [--procedural-synth] [--refresh]
```

Generated MP4s are 720×1280 (9:16), 30 fps, and include AAC stereo narration followed by piano audio. The intro uses the piano-roll frame as its background, a glowing cover image, and timed phrase captions without text panels. A one-second pause and smooth fade lead into the progression. The piano section is cut at the final chord (or the configured 60-second piano maximum); the final three-second outro blurs and dims the piano-roll backdrop behind crisp, supersampled CTA icons, white labels, and a red subscribe button. Total video duration includes the narration, transition, and outro in addition to the piano maximum. Horizontal melody and chord bars move across a pitch grid toward the keyboard, with a single comic-style chord-name caption while a chord is active. Video goes to `outputs/`; the persistent SQLite duplicate ledger defaults to `state/generated.sqlite3`. The CLI only writes an ID to the ledger after its video renders successfully. Keep repository size in mind because videos are binary files.

To replace existing videos in `outputs/` with the current vertical visuals and audio, use:

```sh
music-shorts --refresh --count 100
```

`--refresh` only selects songs whose MP4 already exists and updates their existing ledger records; it does not select additional songs.

## GitHub Actions

`.github/workflows/generate.yml` runs hourly at minute 0 (24 scheduled runs per day) and can also be started manually; both modes generate one video by default. Scheduled runs use the default branch. GitHub Actions scheduled events use UTC and can be delayed during high load, so hourly execution is best-effort rather than an exact-time guarantee. Before use, add this project to a GitHub repository and enable **Settings → Actions → General → Workflow permissions → Read and write permissions**. The workflow only requests `contents: write`, serializes runs per branch, and pushes to the checked-out branch. YouTube secrets and cloud services are not used.

The real dataset is included, so the default CLI can generate a video locally. The first run may take longer because the archive contains more than 26,000 source records. Generated previews are stored in `outputs/` and their song IDs are recorded in `state/generated.sqlite3`; use the synthetic fixture in `tests/` for fast, repeatable tests.
