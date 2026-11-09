# Music Shorts

A Python CLI worker that deterministically selects songs, synthesizes stereo sampled-piano chord/melody audio, and renders vertical 9:16 piano-roll videos. Each video opens on a dimmed piano-roll background with glowing cartoon cover art, comic-style track details, and phrase-by-phrase Piper voice captions. The voice is slowed slightly; after it ends, the video uses a one-second pause that fades smoothly into the chord progression. The progression ends with the final chord; a reference-inspired like/comment/share row and subscribe button hold over the piano roll for three seconds before the video ends. Comic lettering is used throughout. A separate, manually invoked command creates a deterministic YouTube title and tags, drafts only the description with Cloudflare Workers AI, and uploads a selected video after explicit confirmation; uploads are never automatic.

## Dataset format and rights

Generation defaults to `dataset/hooktheory_billboard_filtered.json`, a curated 1,623-record subset focused on popular songs. It contains native Hooktheory records with beat-aligned `melody` and `harmony` annotations. The CLI uses the refined alignment when available, falls back to user alignment, and skips records without both annotation tracks and usable timing anchors. It converts beat positions to seconds and synthesizes notes; it does not use or download the referenced YouTube audio. The larger `dataset/Hooktheory.json.gz` archive remains available as an alternative dataset.

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
music-shorts --count 1
```

The loader accepts plain JSON or gzip-compressed JSON. To use the full archive instead, pass `--dataset dataset/Hooktheory.json.gz`.

## Manual YouTube publishing

`music-shorts-publish` hard-codes the title as `Track Name (Artist Name) chord progression` and builds tags from track metadata; Cloudflare Workers AI writes only the description, using song metadata and up to 64 chord events to draft three song-specific paragraphs separated by blank lines. It does not send the video or full dataset to Cloudflare. If Cloudflare is unavailable or returns unusable text, the command previews a deterministic description based on the track metadata and chord progression instead, so AI failures do not block publishing. It previews the title, description, and tags, then uploads only if you type `UPLOAD`. Uploads are explicitly marked **not made for kids**. The video starts with a clean intro title card (artwork, track, and artist, without narration captions). Before setting the thumbnail, the command polls YouTube until video processing succeeds, then extracts and uploads the video's first frame; this ordering is required for YouTube Shorts custom thumbnails. It waits up to 15 minutes and reports the video ID and `--thumbnail-for VIDEO_ID` retry command if processing takes longer. YouTube may take additional time to display a thumbnail change. The default visibility is **public**; choose `--privacy-status private` or `--privacy-status unlisted` to change it. After the video and thumbnail are uploaded successfully, the local MP4 is deleted. If copy generation, confirmation, video upload, or thumbnail upload fails, the MP4 remains available to retry. If video upload succeeded but setting its thumbnail failed, use `--thumbnail-for VIDEO_ID` to retry only the thumbnail step without uploading the video again.

Before running:

1. Enable the YouTube Data API v3 in a Google Cloud project and create an OAuth client ID for a **Desktop app**. Copy its **Client ID** and **Client secret** strings into `.env`; there is no Google OAuth JSON file path. The publisher authorizes with the `youtube.upload` and `youtube.readonly` scopes so it can upload the video and check when processing is complete before setting the Short's thumbnail. On the first local upload, a browser opens for channel authorization; the refresh token is cached in the ignored local file `state/youtube-token.json`. Existing cached tokens are reauthorized if they lack either scope. A configured `YOUTUBE_REFRESH_TOKEN` for non-interactive uploads must also have both scopes.
2. Create a Cloudflare API token with Workers AI access and add the credentials to `.env`:

```dotenv
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_API_TOKEN=your-workers-ai-token
YOUTUBE_CLIENT_ID=your-google-oauth-client-id
YOUTUBE_CLIENT_SECRET=your-google-oauth-client-secret
# Optional for non-interactive use after OAuth has been completed once:
YOUTUBE_REFRESH_TOKEN=
# Optional:
CLOUDFLARE_AI_MODEL=@cf/meta/llama-3.1-8b-instruct
```

`.env` is ignored by Git. Existing shell environment variables take precedence; use `--env-file PATH` to load a different file.

For GitHub Actions, add `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, and `YOUTUBE_REFRESH_TOKEN` as repository Actions secrets. The refresh token must have both `youtube.upload` and `youtube.readonly` scopes; the workflow uses it for non-interactive uploads without opening a browser. GitHub Actions secrets are only available inside workflows that explicitly map them to environment variables; this local CLI cannot read repository secrets directly. Never commit `.env`, the OAuth token cache, or secret values.
3. Generate a video, then associate its dataset song ID and video path:

```powershell
music-shorts-publish --video outputs\VIDEO.mp4 --song-id SONG_ID
```

The command requires `--video` and `--song-id`; it checks that the video's generated filename matches that dataset song ID. You can override the dataset and visibility with `--dataset PATH` and `--privacy-status private|unlisted|public`. Review the preview and confirmation carefully before authorizing a public upload. YouTube API quota and Google OAuth consent/testing settings apply.

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

Generated MP4s are 720×1280 (9:16), 30 fps, and include AAC stereo narration followed by piano audio. The intro uses the piano-roll frame as its background, a glowing cover image, and timed phrase captions without text panels. A one-second pause and smooth fade lead into the progression. The piano section is cut at the final chord (or the configured 60-second piano maximum); the final three-second outro blurs and dims the piano-roll backdrop behind crisp, supersampled CTA icons, white labels, and a red subscribe button. Total video duration includes the narration, transition, and outro in addition to the piano maximum. Horizontal melody and chord bars move across a pitch grid toward the keyboard, with a single comic-style chord-name caption while a chord is active. Video goes to `outputs/`; only one pending MP4 is kept there. A failed/cancelled upload leaves it available for retry; the next generator run removes older MP4s and refuses to replace the remaining pending video. After successful publishing, the publisher removes it. To generate another song, publish the pending video or manually remove it first. Generation accepts `--count 0` or `--count 1` so it does not discard unposted videos in a batch. The persistent SQLite duplicate ledger defaults to `state/generated.sqlite3`; it records song IDs and normalized artist/track identities after videos render successfully, so the same track is skipped even if it appears under a different dataset ID.

To replace existing videos in `outputs/` with the current vertical visuals and audio, use:

```sh
music-shorts --refresh --count 100
```

`--refresh` only selects songs whose MP4 already exists and updates their existing ledger records; it does not select additional songs. It intentionally replaces the existing MP4, so use it only when you want to replace the pending video.

## GitHub Actions

`.github/workflows/generate.yml` runs hourly at minute 0 (up to 24 scheduled public uploads per day) and can also be started manually; each run tests the worker, generates one full-length video with up to 60 seconds of piano progression, and publishes it as public. Scheduled runs use the default branch. GitHub Actions scheduled events use UTC and can be delayed during high load, so hourly execution is best-effort rather than an exact-time guarantee. Before use, add the five publishing credentials above as repository Actions secrets and enable **Settings → Actions → General → Workflow permissions → Read and write permissions** so the workflow can persist the duplicate-prevention ledger. The workflow serializes runs per branch and pushes only the updated ledger to the checked-out branch. It does not store generated MP4s in the repository.

The popular-song dataset is included, so the default CLI can generate a video locally. Generated previews are stored in `outputs/` and their song IDs are recorded in `state/generated.sqlite3`; use the synthetic fixture in `tests/` for fast, repeatable tests.
