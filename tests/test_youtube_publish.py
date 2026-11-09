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
    assert len(context["chord_progression"]) == 30
    assert context["chords_omitted"] == 0
    assert context["melody_note_count"] == 30
    assert context["melody_pitch_range_midi"] == [72, 79]
    assert len(context["melody_sample_midi"]) == 24


def test_parse_description_response_accepts_json_fence_and_strips_description():
    result = _parse_description_response(
        '```json\n{"sections":["  Track hook  ","Chord detail","#Shorts"]}\n```'
    )

    assert result == "Track hook\n\nChord detail\n\n#Shorts"


def test_parse_description_response_accepts_plain_text_without_json():
    response = (
        "Sample Track opens with a piano arrangement.\n\n"
        "Hear the chord movement at 120 BPM.\n\n"
        "Follow for more. #Shorts"
    )

    assert _parse_description_response(response) == response


@pytest.mark.parametrize(
    "response",
    [
        (
            "Track hook.\n\nMusical detail.\n\n"
            "Follow along. #Shorts #Piano #Chords #Music"
        ),
        (
            "Track hook #Piano.\n\nMusical detail.\n\n"
            "Follow along. #Shorts"
        ),
        (
            "Track hook.\n\nMusical detail.\n\n"
            "Follow along. #Shorts https://example.com"
        ),
        "Track hook.\n\nMusical detail. #Shorts",
    ],
)
def test_parse_description_response_rejects_spam_or_invalid_format(response):
    with pytest.raises(RuntimeError):
        _parse_description_response(response)


def test_parse_description_response_accepts_three_relevant_hashtags():
    response = (
        "Track hook.\n\nMusical detail.\n\n"
        "Follow along. #Shorts #Piano #ChordProgression"
    )

    assert _parse_description_response(response) == response


@pytest.mark.parametrize(
    "response",
    [
        "",
        '{"malformed":',
        '{"sections":["one","two"]}',
        '{"sections":["one","","three"]}',
        '{"sections":["one",2,"three"]}',
    ],
)
def test_parse_description_response_rejects_invalid_sections(response):
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
                        "response": (
                            "Sample Track opens this piano short.\n\n"
                            "Hear the chord movement at 120 BPM.\n\n"
                            "Follow for more piano progressions. #Shorts"
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

    assert result == (
        "Sample Track opens this piano short.\n\n"
        "Hear the chord movement at 120 BPM.\n\n"
        "Follow for more piano progressions. #Shorts"
    )
    assert captured["url"].endswith("/accounts/account-123/ai/run/@cf/meta/llama-3.1-8b-instruct")
    assert captured["authorization"] == "Bearer test-token"
    assert captured["timeout"] == 60
    prompt = captured["body"]["prompt"]
    assert "Sample Artist" in prompt
    assert "chord_progression" in prompt
    assert "melody_pitch_range_midi" in prompt
    assert "30" in prompt
    assert "exactly three short paragraphs" in prompt
    assert "no more than three hashtags total" in prompt
    assert "#Shorts exactly once" in prompt
    assert "keyword stuffing" in prompt
    assert "chord progression" in prompt
    assert "Do not invent" in prompt
    assert "title" not in prompt.lower()


def test_upload_closes_video_stream_after_success(tmp_path, monkeypatch):
    import google.oauth2.credentials
    import googleapiclient.discovery

    import music_shorts.youtube_publish as publishing
    from music_shorts.youtube_publish import upload_video

    class FakeCredentials:
        valid = True
        refresh_token = None
        client_id = "client-id"

        @classmethod
        def from_authorized_user_file(cls, _path, _scopes=None):
            return cls()

        def has_scopes(self, scopes):
            return set(scopes) == set(publishing.YOUTUBE_SCOPES)

        def to_json(self):
            return "{}"

    class FakeRequest:
        def execute(self, num_retries):
            assert num_retries == 3
            return {"id": "uploaded-id"}

    class FakeYouTube:
        media_body = None
        processing_checks = 0

        def videos(self):
            return self

        def list(self, *, part, id):
            assert part == "processingDetails"
            assert id == "uploaded-id"
            return self

        def insert(self, *, part, body, media_body):
            assert part == "snippet,status"
            assert body["status"]["privacyStatus"] == "private"
            assert body["status"]["selfDeclaredMadeForKids"] is False
            self.media_body = media_body
            return FakeRequest()

        def execute(self, num_retries=0):
            assert num_retries == 0
            self.processing_checks += 1
            return {"items": [{"processingDetails": {"processingStatus": "succeeded"}}]}

    youtube = FakeYouTube()
    monkeypatch.setattr(google.oauth2.credentials, "Credentials", FakeCredentials)
    monkeypatch.setattr(
        googleapiclient.discovery,
        "build",
        lambda *_args, **_kwargs: youtube,
    )
    thumbnail_calls = []
    events = []
    monkeypatch.setattr(
        publishing,
        "_upload_thumbnail",
        lambda service, video_id, path: (
            events.append("thumbnail"),
            thumbnail_calls.append((service, video_id, path)),
        ),
    )
    original_list = youtube.list

    def record_processing_check(*, part, id):
        events.append("processing")
        return original_list(part=part, id=id)

    monkeypatch.setattr(youtube, "list", record_processing_check)
    video_path = tmp_path / "short.mp4"
    video_path.write_bytes(b"video")
    token_path = tmp_path / "token.json"
    token_path.write_text("cached", encoding="utf-8")

    video_id = upload_video(
        video_path,
        YouTubeCopy("Track (Artist) C - G", "Description", ("piano",)),
        "private",
        "client-id",
        "client-secret",
        token_path,
    )

    assert video_id == "uploaded-id"
    assert youtube.media_body.stream().closed
    assert thumbnail_calls == [(youtube, "uploaded-id", video_path)]
    assert events == ["processing", "thumbnail"]


def test_wait_for_video_processing_retries_until_succeeded(monkeypatch):
    import music_shorts.youtube_publish as publishing

    class FakeVideos:
        statuses = iter(("processing", "processing", "succeeded"))

        def list(self, *, part, id):
            assert part == "processingDetails"
            assert id == "youtube-id"
            return self

        def execute(self):
            return {
                "items": [
                    {"processingDetails": {"processingStatus": next(self.statuses)}}
                ]
            }

    class FakeYouTube:
        def __init__(self):
            self.video_api = FakeVideos()

        def videos(self):
            return self.video_api

    sleeps = []
    monkeypatch.setattr(publishing.time, "sleep", sleeps.append)

    publishing._wait_for_video_processing(FakeYouTube(), "youtube-id")

    assert sleeps == [2, 4]


@pytest.mark.parametrize(
    ("status", "reason"),
    [("failed", "uploadFailed"), ("terminated", "terminated")],
)
def test_wait_for_video_processing_reports_terminal_status(status, reason):
    import music_shorts.youtube_publish as publishing

    class FakeVideos:
        def list(self, **_kwargs):
            return self

        def execute(self):
            return {
                "items": [
                    {
                        "processingDetails": {
                            "processingStatus": status,
                            "failureReason": reason,
                        }
                    }
                ]
            }

    class FakeYouTube:
        def videos(self):
            return FakeVideos()

    with pytest.raises(RuntimeError, match=reason):
        publishing._wait_for_video_processing(FakeYouTube(), "youtube-id")


def test_upload_thumbnail_extracts_and_sets_first_frame(tmp_path, monkeypatch):
    import music_shorts.youtube_publish as publishing

    class FakeRequest:
        def execute(self, num_retries):
            assert num_retries == 3

    class FakeThumbnails:
        upload = None

        def set(self, *, videoId, media_body):
            assert videoId == "youtube-id"
            self.upload = media_body
            return FakeRequest()

    class FakeYouTube:
        def __init__(self):
            self.thumbnail_api = FakeThumbnails()

        def thumbnails(self):
            return self.thumbnail_api

    captured = {}

    def fake_extract_first_frame(video, output):
        captured["video"] = video
        captured["output"] = output
        output.write_bytes(b"jpeg")
        return output

    monkeypatch.setattr(publishing, "extract_first_frame", fake_extract_first_frame)
    youtube = FakeYouTube()
    video_path = tmp_path / "short.mp4"
    video_path.write_bytes(b"video")

    publishing._upload_thumbnail(youtube, "youtube-id", video_path)

    assert captured["video"] == video_path
    assert captured["output"].name == "first-frame.jpg"
    assert youtube.thumbnail_api.upload.mimetype() == "image/jpeg"
    assert youtube.thumbnail_api.upload.stream().closed


@pytest.mark.parametrize(
    ("cached_client_id", "cached_has_scopes", "warning"),
    [
        ("old-client-id", True, "different OAuth client"),
        ("new-client-id", False, "needs upload and read-only access"),
    ],
)
def test_upload_reauthorizes_cached_token_when_required(
    tmp_path,
    monkeypatch,
    capsys,
    cached_client_id,
    cached_has_scopes,
    warning,
):
    import google.oauth2.credentials
    import google_auth_oauthlib.flow
    import googleapiclient.discovery

    from music_shorts.youtube_publish import upload_video

    class FakeCredentials:
        valid = True
        refresh_token = "old-refresh-token"

        def __init__(self, client_id, has_required_scopes=True):
            self.client_id = client_id
            self.has_required_scopes = has_required_scopes

        @classmethod
        def from_authorized_user_file(cls, _path):
            return cls(cached_client_id, cached_has_scopes)

        def has_scopes(self, _scopes):
            return self.has_required_scopes

        def to_json(self):
            return "{}"

    class FakeRequest:
        def execute(self, num_retries):
            return {"id": "uploaded-id"}

    class FakeYouTube:
        def videos(self):
            return self

        def list(self, **_kwargs):
            return self

        def insert(self, **_kwargs):
            return FakeRequest()

        def execute(self):
            return {"items": [{"processingDetails": {"processingStatus": "succeeded"}}]}

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
    monkeypatch.setattr(
        "music_shorts.youtube_publish._upload_thumbnail",
        lambda *_args: None,
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
    assert warning in capsys.readouterr().err


def test_refresh_token_failure_does_not_start_interactive_oauth(monkeypatch, tmp_path):
    import google.auth.exceptions
    import google.oauth2.credentials
    import google_auth_oauthlib.flow

    import music_shorts.youtube_publish as publishing

    class FakeCredentials:
        def __init__(self, **_kwargs):
            self.valid = False
            self.refresh_token = "configured-refresh-token"

        def refresh(self, _request):
            raise google.auth.exceptions.RefreshError("invalid_scope")

    class FakeFlow:
        @classmethod
        def from_client_config(cls, *_args, **_kwargs):
            pytest.fail("configured refresh-token failures must not open browser OAuth")

    monkeypatch.setattr(google.oauth2.credentials, "Credentials", FakeCredentials)
    monkeypatch.setattr(google_auth_oauthlib.flow, "InstalledAppFlow", FakeFlow)

    with pytest.raises(RuntimeError, match="authorized for both youtube.upload and youtube.readonly"):
        publishing._youtube_service(
            "client-id",
            "client-secret",
            tmp_path / "missing-token.json",
            "configured-refresh-token",
        )


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


def test_main_yes_skips_confirmation_and_uses_metadata_fallback(
    tmp_path, monkeypatch, capsys
):
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
    monkeypatch.setenv("YOUTUBE_CLIENT_ID", "client-id")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "client-secret")
    monkeypatch.setattr(
        publishing,
        "generate_youtube_description",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("Cloudflare AI response did not contain a JSON object.")
        ),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: pytest.fail("--yes must skip interactive confirmation"),
    )
    captured = {}

    def fake_upload(_path, metadata, *_args):
        captured["metadata"] = metadata
        return "uploaded-id"

    monkeypatch.setattr(publishing, "upload_video", fake_upload)

    result = main(
        [
            "--video",
            str(video_path),
            "--song-id",
            "song-1",
            "--dataset",
            str(dataset_path),
            "--privacy-status",
            "private",
            "--yes",
        ]
    )

    assert result == 0
    assert captured["metadata"].description == publishing._fallback_youtube_description(song)
    assert "#Shorts" in captured["metadata"].description
    assert "metadata-based description instead" in capsys.readouterr().err
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


def test_main_can_retry_thumbnail_for_existing_youtube_video(tmp_path, monkeypatch):
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
    monkeypatch.setattr("builtins.input", lambda _prompt: "THUMBNAIL")
    calls = []
    monkeypatch.setattr(
        publishing,
        "set_existing_video_thumbnail",
        lambda *args: calls.append(args),
    )
    monkeypatch.setattr(
        publishing,
        "generate_youtube_description",
        lambda *_args: pytest.fail("thumbnail retry must not call Cloudflare AI"),
    )

    result = main(
        [
            "--video",
            str(video_path),
            "--song-id",
            "song-1",
            "--dataset",
            str(dataset_path),
            "--thumbnail-for",
            "already-uploaded-id",
        ]
    )

    assert result == 0
    assert calls[0][0] == "already-uploaded-id"
    assert calls[0][1] == video_path
    assert not video_path.exists()


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
