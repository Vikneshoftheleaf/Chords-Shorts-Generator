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
- Synthesize audio from chord and melody events, then render a short original piano-roll/falling-notes video with the full piano keyboard visible and keys highlighted as notes play.
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
- The current workspace is not a Git repository and has no remote. Committing generated files from Actions requires adding this project to a GitHub repository and allowing workflow write permissions.

## Acceptance criteria

1. Unit tests cover parsing/validation, deterministic selection, duplicate skipping, synthesis/render orchestration, and invalid input.
2. An integration test using a small fixture creates a playable MP4 and updates the SQLite ledger; a second run skips that song.
3. The documented CLI and GitHub Actions workflow work without YouTube credentials or external cloud services.
4. Generated previews and ledger entries are stored in the repository after validating the real dataset.
