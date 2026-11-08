# Music Shorts MVP Project Plan

## Approval and scope

The user approved this plan after reviewing the captured requirements. Build a Python command-line worker that generates a short music-visualization video from a song's chords and melody. YouTube upload is explicitly out of scope for this MVP.

The user confirmed that they have permission to create and publicly publish work derived from the dataset. Avoid using FL Studio branding, logos, or proprietary assets; use an original DAW-inspired interface.

## Runtime and services

- One Python worker, runnable locally and from GitHub Actions.
- Use a lightweight Python CLI and FFmpeg for media encoding; favor dependencies and defaults that work on GitHub-hosted Ubuntu runners.
- Do not add a cloud service or application API.

## MVP behavior

- Load the compressed JSON dataset from `dataset/Hooktheory.json.gz`; validate gzip, JSON, and the expected song/chord/melody data, and give actionable errors for invalid input.
- Select eligible songs deterministically and skip IDs already recorded in a persistent SQLite ledger.
- Synthesize stereo sampled-piano audio from chord and melody events with beat-aligned timing, natural velocity variation, and restrained reverb. Use the FreePats YDP Grand Piano SF2, retaining its CC BY 3.0 attribution and license.
- Render portrait 9:16 videos in a reference-inspired horizontal piano-roll style: a vertical keyboard at the left, melody and chord notes moving horizontally across a pitch grid, distinct note colors, and one chord-name caption while a chord is playing. Do not label every note or key.
- Show track artwork and metadata in a compact header. Use the iTunes Search API only for exact artist/title matches and generate fallback artwork when no match or network access is available. Keep the track title slightly larger than the artist line.
- Add a cartoonized, glowing cover-art intro over a dimmed piano-roll frame, with track metadata, comic-book lettering, and timed phrase captions for a slightly slowed Piper medium male voice.
- Pause for one second after narration and smoothly fade into the piano progression; do not put intro text in boxes or panels.
- End the progression at the final chord (or configured duration cap), then hold the piano-roll view under a reference-inspired like/comment/share row and subscribe button for exactly three seconds.
- Provide CLI controls for output count and output location, with conservative CI defaults and a one-video default per run.
- Store generated MP4 files and the SQLite duplicate-prevention ledger in the repository. The workflow should use a concurrency guard, grant only the write permission it needs, and commit/push only when it generated new output.
- Provide a manually dispatchable GitHub Actions workflow. Do not configure automatic YouTube posting.
- Test the pipeline with a small synthetic fixture so tests do not depend on the full dataset.

## Persistence and repository constraints

- SQLite is the source of truth for generated song IDs and generation metadata; persist the DB file so subsequent workflow runs can avoid duplicates.
- Commit outputs and ledger updates to the checked-out repository as requested. Keep generation configurable and warn about repository growth from video files; do not silently substitute ephemeral Actions artifacts for the requested persistent repository output.
- Configure the workflow for repository write access and a non-destructive push flow. Document that repository Actions settings must allow the workflow to write.

## Data source, rights, and repository constraints

- The original local `dataset/Hooktheory.json.gz` was an HTML `504 Gateway Time-out` response. It has been replaced with the archive from `chrisdonahue/sheetsage-data`; the native Hooktheory record structure and beat-time alignments have now been inspected and integrated.
- The Sheet Sage repository states that its dataset is CC BY-NC-SA 3.0. The user confirms separate written permission covering intended commercial/monetized use. Keep that permission available and ensure it covers redistribution of the archive and publication of derived videos.
- The FreePats YDP Grand Piano SoundFont is published under CC BY 3.0. Include the provided attribution/license text when redistributing it.
- The current workspace is not a Git repository and has no remote. Committing generated files from Actions requires adding this project to a GitHub repository and allowing workflow write permissions.

## Acceptance criteria

1. Unit tests cover parsing/validation, deterministic selection, duplicate skipping, stereo audible piano-style synthesis, portrait MP4 dimensions, AAC audio, and invalid input.
2. An integration test using a small fixture creates a playable portrait MP4 with sound and updates the SQLite ledger; a second run skips that song.
3. The documented CLI and GitHub Actions workflow work without YouTube credentials or external cloud services.
4. Generated previews and ledger entries are stored in the repository after validating the real dataset.
