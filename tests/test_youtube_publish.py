from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from music_shorts.pipeline import NoteEvent, Song, load_dataset, output_filename
from music_shorts.youtube_publish import (
    YouTubeCopy,
    build_song_context,
    build_youtube_copy,
    generate_youtube_description,
    main,
    _parse_description_response,
)


def sample_song() -> Song:
    return Song(
        song_id="popular-song-001",
        title="Sample Artist - Sample Track",
        tempo=120,
        chords=tuple(
            NoteEvent(float(index), 0.8, (48 + index % 12, 52 + index % 12, 55 + index % 12))
            for index in range(30)
        ),
        melody=tuple(
            NoteEvent(float(index) / 2, 0.2, (72 + index % 8,))
            for index in range(30)
        ),
        artist="Sample Artist",
        track_name="Sample Track",
    )


def test_song_context_is_bounded_and_contains_musical_facts():
    context = build_song_context(sample_song())

    assert context["artist"] == "Sample Artist"
    assert context["track"] == "Sample Track"
    assert context["tempo_bpm"] == 120
    assert len(context["chord_progression"]) == 24
    assert context["chords_omitted"] == 6
    assert context["melody_note_count"] == 30
    assert context["melody_pitch_range_midi"] == [72, 79]
    assert len(context["melody_sample_midi"]) == 24


def test_parse_description_response_accepts_json_fence_and_strips_description():
    result = _parse_description_response(
        '```json\n{"description":"  Piano progression  "}\n```'
    )

    assert result == "Piano progression"


@pytest.mark.parametrize(
    "response",
    [
        "no json",
        '{"description":""}',
        '{"description":2}',
    ],
)
def test_parse_description_response_rejects_invalid_description(response):
    with pytest.raises(RuntimeError):
        _parse_description_response(response)


def test_generate_youtube_description_posts_only_bounded_context(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {
                    "success": True,
                    "result": {
                        "response": json.dumps(
                            {"description": "A piano arrangement of Sample Track."}
                        )
                    },
                }
            ).encode()

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = generate_youtube_description(sample_song(), "account-123", "test-token")

    assert result == "A piano arrangement of Sample Track."
    assert captured["url"].endswith("/accounts/account-123/ai/run/@cf/meta/llama-3.1-8b-instruct")
    assert captured["authorization"] == "Bearer test-token"
    assert captured["timeout"] == 60
    prompt = captured["body"]["prompt"]
    assert "Sample Artist" in prompt
    assert "chord_progression" in prompt
    assert "melody_pitch_range_midi" in prompt
    assert "30" in prompt
    assert "one string field named description" in prompt
    assert "title" not in prompt.lower()


def test_upload_closes_video_stream_after_success(tmp_path, monkeypatch):
    import google.oauth2.credentials
    import googleapiclient.discovery

    from music_shorts.youtube_publish import upload_video

    class FakeCredentials:
        valid = True
        refresh_token = None

        @classmethod
        def from_authorized_user_file(cls, _path, _scopes):
            return cls()

        def to_json(self):
            return "{}"

    class FakeRequest:
        def execute(self, num_retries):
            assert num_retries == 3
            return {"id": "uploaded-id"}

    class FakeYouTube:
        media_body = None

        def videos(self):
            return self

        def insert(self, *, part, body, media_body):
            assert part == "snippet,status"
            assert body["status"]["privacyStatus"] == "private"
            self.media_body = media_body
            return FakeRequest()

    youtube = FakeYouTube()
    monkeypatch.setattr(google.oauth2.credentials, "Credentials", FakeCredentials)
    monkeypatch.setattr(
        googleapiclient.discovery,
        "build",
        lambda *_args, **_kwargs: youtube,
    )
    video_path = tmp_path / "short.mp4"
    video_path.write_bytes(b"video")

    video_id = upload_video(
        video_path,
        YouTubeCopy("Track (Artist) C - G", "Description", ("piano",)),
        "private",
        "client-id",
        "client-secret",
        tmp_path / "token.json",
    )

    assert video_id == "uploaded-id"
    assert youtube.media_body.stream().closed


def test_upload_ignores_cached_token_for_different_client(tmp_path, monkeypatch, capsys):
    import google.oauth2.credentials
    import google_auth_oauthlib.flow
    import googleapiclient.discovery

    from music_shorts.youtube_publish import upload_video

    class FakeCredentials:
        valid = True
        refresh_token = "old-refresh-token"

        def __init__(self, client_id):
            self.client_id = client_id

        @classmethod
        def from_authorized_user_file(cls, _path, _scopes):
            return cls("old-client-id")

        def to_json(self):
            return "{}"

    class FakeRequest:
        def execute(self, num_retries):
            return {"id": "uploaded-id"}

    class FakeYouTube:
        def videos(self):
            return self

        def insert(self, **_kwargs):
            return FakeRequest()

    class FakeFlow:
        @classmethod
        def from_client_config(cls, client_config, _scopes):
            assert client_config["installed"]["client_id"] == "new-client-id"
            return cls()

        def run_local_server(self, port):
            assert port == 0
            return FakeCredentials("new-client-id")

    monkeypatch.setattr(google.oauth2.credentials, "Credentials", FakeCredentials)
    monkeypatch.setattr(google_auth_oauthlib.flow, "InstalledAppFlow", FakeFlow)
    monkeypatch.setattr(
        googleapiclient.discovery,
        "build",
        lambda *_args, **_kwargs: FakeYouTube(),
    )
    video_path = tmp_path / "short.mp4"
    video_path.write_bytes(b"video")
    token_path = tmp_path / "token.json"
    token_path.write_text("stale", encoding="utf-8")

    video_id = upload_video(
        video_path,
        YouTubeCopy("Track (Artist) C - G", "Description", ("piano",)),
        "private",
        "new-client-id",
        "new-client-secret",
        token_path,
    )

    assert video_id == "uploaded-id"
    assert "cached YouTube token belongs to a different OAuth client" in capsys.readouterr().err


def test_youtube_title_uses_exact_track_artist_and_literal_chord_progression():
    song = Song(
        song_id="title-song",
        title="Sample Artist - Sample Track",
        tempo=100,
        chords=(
            NoteEvent(0, 1, (48, 52, 55)),
            NoteEvent(1, 1, (53, 57, 60)),
            NoteEvent(2, 1, (55, 59, 62)),
            NoteEvent(3, 1, (57, 60, 64)),
            NoteEvent(4, 1, (48, 52, 55)),
            NoteEvent(5, 1, (53, 57, 60)),
            NoteEvent(6, 1, (55, 59, 62)),
            NoteEvent(7, 1, (57, 60, 64)),
        ),
        melody=(NoteEvent(0, 1, (72,)),),
        artist="sample-artist",
        track_name="sample-track",
    )

    copy = build_youtube_copy(song, "AI description")

    assert copy.title == "Sample Track (Sample Artist) chord progression"
    assert copy.description == "AI description"
    assert copy.tags == (
        "Sample Track",
        "Sample Artist",
        "Sample Track piano",
        "Piano Cover",
        "Piano Chords",
        "Chord Progression",
        "Piano Shorts",
    )
    assert len(copy.title) <= 100


def test_main_cancellation_never_uploads(tmp_path, monkeypatch, capsys):
    import music_shorts.youtube_publish as publishing

    dataset_path = tmp_path / "songs.json"
    dataset_path.write_bytes(
        gzip.compress(
            json.dumps(
                {
                    "popular-song-001": {
                        "tags": ["MELODY", "HARMONY"],
                        "hooktheory": {
                            "id": "popular-song-001",
                            "artist": "Sample Artist",
                            "song": "Sample Track",
                        },
                        "alignment": {
                            "refined": {"beats": [0, 1, 2], "times": [0, 0.5, 1]},
                        },
                        "annotations": {
                            "melody": [
                                {"onset": 0, "offset": 1, "octave": 1, "pitch_class": 0}
                            ],
                            "harmony": [
                                {
                                    "onset": 0,
                                    "offset": 2,
                                    "root_pitch_class": 0,
                                    "root_position_intervals": [4, 3],
                                }
                            ],
                        },
                    }
                }
            ).encode()
        )
    )
    song = load_dataset(dataset_path)[0]
    video_path = tmp_path / output_filename(song)
    video_path.write_bytes(b"mp4")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")
    monkeypatch.setenv("YOUTUBE_CLIENT_ID", "client-id")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "client-secret")
    monkeypatch.setattr(
        publishing,
        "generate_youtube_description",
        lambda *_args: "Piano arrangement",
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    monkeypatch.setattr(
        publishing,
        "upload_video",
        lambda *_args: pytest.fail("upload must not happen after cancellation"),
    )

    result = main(
        [
            "--video",
            str(video_path),
            "--song-id",
            "popular-song-001",
            "--dataset",
            str(dataset_path),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "PUBLIC" in output
    assert "Upload cancelled" in output


def test_main_loads_dotenv_and_preserves_existing_environment(tmp_path, monkeypatch):
    import music_shorts.youtube_publish as publishing

    dataset_path = tmp_path / "songs.json"
    dataset_path.write_bytes(
        json.dumps(
            {
                "songs": [
                    {
                        "id": "song-1",
                        "title": "Sample Track",
                        "tempo": 120,
                        "chords": [{"start": 0, "duration": 1, "notes": [48, 52, 55]}],
                        "melody": [{"start": 0, "duration": 1, "pitch": 72}],
                    }
                ]
            }
        ).encode()
    )
    song = load_dataset(dataset_path)[0]
    video_path = tmp_path / output_filename(song)
    video_path.write_bytes(b"mp4")
    (tmp_path / ".env").write_text(
        "CLOUDFLARE_ACCOUNT_ID=dotenv-account\n"
        "CLOUDFLARE_API_TOKEN=dotenv-token\n"
        "CLOUDFLARE_AI_MODEL=dotenv-model\n"
        "YOUTUBE_CLIENT_ID=dotenv-client-id\n"
        "YOUTUBE_CLIENT_SECRET=dotenv-client-secret\n"
        "YOUTUBE_REFRESH_TOKEN=dotenv-refresh-token\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.setenv("CLOUDFLARE_AI_MODEL", "shell-model")
    captured = {}

    def fake_generate(_song, account_id, api_token, model):
        captured.update(
            account=account_id,
            token=api_token,
            model=model,
        )
        return "Piano arrangement"

    monkeypatch.setattr(publishing, "generate_youtube_description", fake_generate)
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    result = main(
        [
            "--video",
            str(video_path),
            "--song-id",
            "song-1",
            "--dataset",
            str(dataset_path),
        ]
    )

    assert result == 0
    assert captured == {
        "account": "dotenv-account",
        "token": "dotenv-token",
        "model": "shell-model",
    }


def test_main_upload_requires_exact_confirmation(tmp_path, monkeypatch):
    import music_shorts.youtube_publish as publishing

    dataset_path = tmp_path / "songs.json"
    dataset_path.write_bytes(
        json.dumps(
            {
                "songs": [
                    {
                        "id": "song-1",
                        "title": "Sample Track",
                        "tempo": 120,
                        "artist": "Sample Artist",
                        "track_name": "Sample Track",
                        "chords": [{"start": 0, "duration": 1, "notes": [48, 52, 55]}],
                        "melody": [{"start": 0, "duration": 1, "pitch": 72}],
                    }
                ]
            }
        ).encode()
    )
    song = load_dataset(dataset_path)[0]
    video_path = tmp_path / output_filename(song)
    video_path.write_bytes(b"mp4")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")
    monkeypatch.setenv("YOUTUBE_CLIENT_ID", "configured-client-id")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "configured-client-secret")
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "configured-refresh-token")
    monkeypatch.setattr(
        publishing,
        "generate_youtube_description",
        lambda *_args: "Piano arrangement",
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "UPLOAD")
    captured = {}

    def fake_upload(*args):
        captured["args"] = args
        return "youtube-id"

    monkeypatch.setattr(publishing, "upload_video", fake_upload)
    result = main(
        [
            "--video",
            str(video_path),
            "--song-id",
            "song-1",
            "--dataset",
            str(dataset_path),
        ]
    )

    assert result == 0
    expected_metadata = build_youtube_copy(song, "Piano arrangement")
    assert captured["args"][1] == expected_metadata
    assert captured["args"][2] == "public"
    assert captured["args"][3] == "configured-client-id"
    assert captured["args"][4] == "configured-client-secret"
    assert captured["args"][5] == Path("state/youtube-token.json")
    assert captured["args"][6] == "configured-refresh-token"
    assert not video_path.exists()


def test_main_keeps_video_when_upload_fails(tmp_path, monkeypatch, capsys):
    import music_shorts.youtube_publish as publishing

    dataset_path = tmp_path / "songs.json"
    dataset_path.write_bytes(
        json.dumps(
            {
                "songs": [
                    {
                        "id": "song-1",
                        "title": "Sample Track",
                        "tempo": 120,
                        "chords": [{"start": 0, "duration": 1, "notes": [48, 52, 55]}],
                        "melody": [{"start": 0, "duration": 1, "pitch": 72}],
                    }
                ]
            }
        ).encode()
    )
    song = load_dataset(dataset_path)[0]
    video_path = tmp_path / output_filename(song)
    video_path.write_bytes(b"mp4")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")
    monkeypatch.setenv("YOUTUBE_CLIENT_ID", "client-id")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "client-secret")
    monkeypatch.setattr(
        publishing,
        "generate_youtube_description",
        lambda *_args: "Piano arrangement",
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "UPLOAD")
    monkeypatch.setattr(
        publishing,
        "upload_video",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("simulated upload failure")),
    )

    result = main(
        [
            "--video",
            str(video_path),
            "--song-id",
            "song-1",
            "--dataset",
            str(dataset_path),
        ]
    )

    assert result == 1
    assert video_path.is_file()
    assert "simulated upload failure" in capsys.readouterr().err


def test_main_rejects_video_from_different_song(tmp_path, monkeypatch, capsys):
    import music_shorts.youtube_publish as publishing

    dataset_path = tmp_path / "songs.json"
    dataset_path.write_bytes(
        json.dumps(
            {
                "songs": [
                    {
                        "id": "song-1",
                        "title": "Sample Track",
                        "tempo": 120,
                        "chords": [{"start": 0, "duration": 1, "notes": [48, 52, 55]}],
                        "melody": [{"start": 0, "duration": 1, "pitch": 72}],
                    }
                ]
            }
        ).encode()
    )
    video_path = tmp_path / "different-song.mp4"
    video_path.write_bytes(b"mp4")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")
    monkeypatch.setenv("YOUTUBE_CLIENT_ID", "client-id")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "client-secret")
    monkeypatch.setattr(
        publishing,
        "generate_youtube_description",
        lambda *_args: pytest.fail("copy must not be generated for a mismatched video"),
    )

    result = main(
        [
            "--video",
            str(video_path),
            "--song-id",
            "song-1",
            "--dataset",
            str(dataset_path),
        ]
    )

    assert result == 1
    assert "Video filename does not match" in capsys.readouterr().err
