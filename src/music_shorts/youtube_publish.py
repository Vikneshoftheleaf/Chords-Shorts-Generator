"""Generate YouTube copy with Cloudflare Workers AI and manually upload a Short."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .pipeline import (
    DatasetError,
    Song,
    _chord_name,
    _format_display_name,
    load_dataset,
    output_filename,
)

DEFAULT_DATASET = Path("dataset/hooktheory_billboard_filtered.json")
DEFAULT_MODEL = "@cf/meta/llama-3.1-8b-instruct"
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"


@dataclass(frozen=True)
class YouTubeCopy:
    title: str
    description: str
    tags: tuple[str, ...]


def _youtube_title(song: Song) -> str:
    track = _format_display_name(song.track_name or song.title, artist=True)
    artist = _format_display_name(song.artist, artist=True)
    if not artist and " - " in track:
        artist, track = track.split(" - ", 1)
    title = f"{track} ({artist}) chord progression" if artist else f"{track} chord progression"
    if len(title) > 100:
        raise RuntimeError("Track and artist names exceed YouTube's 100-character title limit.")
    return title


def build_youtube_copy(song: Song, description: str) -> YouTubeCopy:
    artist = _format_display_name(song.artist, artist=True)
    track = _format_display_name(song.track_name or song.title, artist=True)
    tags = tuple(
        dict.fromkeys(
            tag
            for tag in (
                track,
                artist,
                f"{track} piano",
                "Piano Cover",
                "Piano Chords",
                "Chord Progression",
                "Piano Shorts",
            )
            if tag
        )
    )
    return YouTubeCopy(_youtube_title(song), description, tags)


def build_song_context(song: Song) -> dict[str, Any]:
    """Create bounded musical context without sending audio or source annotations."""
    melody_pitches = [event.notes[0] for event in song.melody]
    return {
        "track": song.track_name or song.title,
        "artist": song.artist,
        "dataset_song_id": song.song_id,
        "tempo_bpm": round(song.tempo, 1),
        "duration_seconds": round(song.duration, 1),
        "chord_progression": [
            _chord_name(event.notes) for event in song.chords[:24]
        ],
        "chords_omitted": max(0, len(song.chords) - 24),
        "melody_note_count": len(melody_pitches),
        "melody_pitch_range_midi": (
            [min(melody_pitches), max(melody_pitches)] if melody_pitches else []
        ),
        "melody_sample_midi": melody_pitches[:24],
    }


def _parse_description_response(response_text: str) -> str:
    start = response_text.find("{")
    if start < 0:
        raise RuntimeError("Cloudflare AI response did not contain a JSON object.")
    try:
        payload, _ = json.JSONDecoder().raw_decode(response_text[start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Cloudflare AI returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Cloudflare AI response must be a JSON object.")

    description = payload.get("description")
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description.strip()) > 5000
    ):
        raise RuntimeError("Generated YouTube description must contain 1 to 5000 characters.")
    return description.strip()


def generate_youtube_description(
    song: Song,
    account_id: str,
    api_token: str,
    model: str = DEFAULT_MODEL,
) -> str:
    context = json.dumps(build_song_context(song), ensure_ascii=False)
    prompt = (
        "Write an original YouTube Shorts description for a piano visualization video. "
        "Use the supplied song data as factual context. Do not invent facts, claim "
        "official status or artist endorsement, quote lyrics, or include URLs. "
        "Return only a JSON object with one string field named description. Keep "
        "the description concise, natural, "
        "and relevant to the chord progression and piano arrangement. Include a "
        "brief, non-pushy call to action and relevant hashtags, including #Shorts.\n\n"
        f"Song data JSON: {context}"
    )
    account = urllib.parse.quote(account_id.strip(), safe="")
    model_path = urllib.parse.quote(model.strip(), safe="@/")
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model_path}"
    request = urllib.request.Request(
        url,
        data=json.dumps({"prompt": prompt, "max_tokens": 500}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Cloudflare Workers AI request failed (HTTP {exc.code}): {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach Cloudflare Workers AI: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cloudflare Workers AI returned invalid JSON.") from exc

    if not isinstance(result, dict) or result.get("success") is False:
        errors = result.get("errors", []) if isinstance(result, dict) else []
        raise RuntimeError(f"Cloudflare Workers AI reported an error: {errors}")
    model_result = result.get("result")
    response_text = model_result.get("response") if isinstance(model_result, dict) else None
    if not isinstance(response_text, str):
        raise RuntimeError("Cloudflare Workers AI response did not contain generated text.")
    return _parse_description_response(response_text)


def upload_video(
    video_path: Path,
    metadata: YouTubeCopy,
    privacy_status: str,
    client_id: str,
    client_secret: str,
    token_path: Path,
    refresh_token: str = "",
) -> str:
    if not client_id.strip() or not client_secret.strip():
        raise RuntimeError(
            "Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET to the OAuth client "
            "values from Google Cloud."
        )
    if not video_path.is_file():
        raise RuntimeError(f"Video file not found: {video_path}")

    try:
        from google.auth.exceptions import GoogleAuthError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise RuntimeError(
            "YouTube upload dependencies are missing. Reinstall with "
            "`python -m pip install -e .`."
        ) from exc

    media_upload = None
    try:
        credentials = None
        if token_path.is_file():
            credentials = Credentials.from_authorized_user_file(
                str(token_path), [YOUTUBE_UPLOAD_SCOPE]
            )
            if credentials.client_id != client_id.strip():
                print(
                    "warning: cached YouTube token belongs to a different OAuth client; "
                    "reauthorizing with YOUTUBE_CLIENT_ID",
                    file=sys.stderr,
                )
                credentials = None
        if credentials is None and refresh_token.strip():
            credentials = Credentials(
                token=None,
                refresh_token=refresh_token.strip(),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client_id.strip(),
                client_secret=client_secret.strip(),
                scopes=[YOUTUBE_UPLOAD_SCOPE],
            )
        if credentials and credentials.refresh_token and not credentials.valid:
            credentials.refresh(Request())
        if not credentials or not credentials.valid:
            flow = InstalledAppFlow.from_client_config(
                {
                    "installed": {
                        "client_id": client_id.strip(),
                        "client_secret": client_secret.strip(),
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": ["http://localhost"],
                    }
                },
                [YOUTUBE_UPLOAD_SCOPE],
            )
            credentials = flow.run_local_server(port=0)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(credentials.to_json(), encoding="utf-8")
        youtube = build("youtube", "v3", credentials=credentials, cache_discovery=False)
        media_upload = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            chunksize=8 * 1024 * 1024,
            resumable=True,
        )
        request = youtube.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": metadata.title,
                    "description": metadata.description,
                    "tags": list(metadata.tags),
                    "categoryId": "10",
                },
                "status": {"privacyStatus": privacy_status},
            },
            media_body=media_upload,
        )
        uploaded = request.execute(num_retries=3)
    except (GoogleAuthError, HttpError, OSError, ValueError) as exc:
        raise RuntimeError(f"YouTube upload failed: {exc}") from exc
    finally:
        if media_upload is not None:
            media_upload.stream().close()

    video_id = uploaded.get("id") if isinstance(uploaded, dict) else None
    if not isinstance(video_id, str) or not video_id:
        raise RuntimeError("YouTube upload completed without returning a video ID.")
    return video_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-shorts-publish",
        description="Generate YouTube copy with Cloudflare AI and manually upload a video.",
    )
    parser.add_argument("--video", type=Path, required=True, help="generated MP4 to upload")
    parser.add_argument("--song-id", required=True, help="song ID associated with the video")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--privacy-status",
        choices=("private", "unlisted", "public"),
        default="public",
        help="YouTube visibility for this upload (default: public)",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=Path("state/youtube-token.json"),
        help="local OAuth token cache (keep private)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Cloudflare Workers AI model (or set CLOUDFLARE_AI_MODEL)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="environment file to load (default: .env; existing environment takes precedence)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file, override=False)
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    client_id = os.environ.get("YOUTUBE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET", "").strip()
    refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN", "").strip()
    if not account_id or not api_token:
        print(
            "error: set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN first",
            file=sys.stderr,
        )
        return 2
    if not client_id or not client_secret:
        print(
            "error: set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET first",
            file=sys.stderr,
        )
        return 2
    if not args.video.is_file():
        print(f"error: video file not found: {args.video}", file=sys.stderr)
        return 2

    try:
        songs = load_dataset(args.dataset)
        song = next((item for item in songs if item.song_id == args.song_id), None)
        if song is None:
            raise DatasetError(f"Song ID {args.song_id!r} was not found in {args.dataset}.")
        expected_video_name = output_filename(song)
        if args.video.name.casefold() != expected_video_name.casefold():
            raise DatasetError(
                f"Video filename does not match song ID {args.song_id!r}; "
                f"expected {expected_video_name}."
            )
        model = args.model or os.environ.get("CLOUDFLARE_AI_MODEL", DEFAULT_MODEL)
        description = generate_youtube_description(song, account_id, api_token, model)
        metadata = build_youtube_copy(song, description)
        print(f"\nTitle: {metadata.title}")
        print(f"\nDescription:\n{metadata.description}")
        print(f"\nTags: {', '.join(metadata.tags)}")
        print(f"\nThe video will be uploaded as {args.privacy_status.upper()}.")
        try:
            confirmation = input("Type UPLOAD to authorize the YouTube upload: ")
        except EOFError:
            print("Upload cancelled; no YouTube upload was made.")
            return 0
        if confirmation.strip() != "UPLOAD":
            print("Upload cancelled; no YouTube upload was made.")
            return 0
        video_id = upload_video(
            args.video,
            metadata,
            args.privacy_status,
            client_id,
            client_secret,
            args.token_file,
            refresh_token,
        )
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"Uploaded {args.privacy_status} video: {video_url}")
        try:
            args.video.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"Upload succeeded ({video_url}), but could not remove local video "
                f"{args.video}: {exc}"
            ) from exc
        print(f"Removed local video {args.video}")
    except (DatasetError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
