from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import urllib.request
import wave
from pathlib import Path

import pytest
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from music_shorts.pipeline import (
    DatasetError,
    Ledger,
    _midi_track,
    _format_display_name,
    _caption_phrases,
    _font,
    _intro_script,
    INTRO_VOICE_LENGTH_SCALE,
    INTRO_VOICE_VOLUME,
    _lookup_itunes_artwork,
    _progression_excerpt,
    _render_intro_frame,
    _render_outro_frame,
    _render_frame,
    render_video,
    _synthesize_intro,
    CaptionCue,
    generate_song,
    keep_latest_video,
    NoteEvent,
    parse_dataset_bytes,
    _chord_name,
    select_songs,
    synthesize,
    Song,
    TrackArtwork,
)

FIXTURE = Path(__file__).parent / "fixtures" / "song.json"


def fixture_songs():
    return parse_dataset_bytes(gzip.compress(FIXTURE.read_bytes()))


def test_parses_normalized_gzip_fixture():
    songs = fixture_songs()
    assert [song.song_id for song in songs] == ["fixture-song-001", "fixture-song-002"]
    assert songs[0].duration == pytest.approx(1.6)


def test_parses_native_hooktheory_annotations_and_alignment():
    document = {
        "source-song": {
            "tags": ["MELODY", "HARMONY"],
            "hooktheory": {"id": "source-song", "artist": "sample-artist", "song": "sample-title"},
            "alignment": {"refined": {"beats": [0, 1, 2], "times": [10.0, 10.5, 11.0]}},
            "annotations": {
                "melody": [{"onset": 1, "offset": 2, "octave": 0, "pitch_class": 9}],
                "harmony": [{
                    "onset": 0, "offset": 2, "root_pitch_class": 0,
                    "root_position_intervals": [4, 3], "inversion": 1,
                }],
            },
        }
    }
    raw = json.dumps(document).encode()
    songs = parse_dataset_bytes(raw)

    assert len(songs) == 1
    song = songs[0]
    assert song.song_id == "source-song"
    assert song.title == "sample-artist - sample-title"
    assert song.artist == "sample-artist"
    assert song.track_name == "sample-title"
    assert song.tempo == pytest.approx(120)
    assert song.melody[0].notes == (69,)
    assert song.melody[0].start == pytest.approx(0.5)
    assert song.melody[0].duration == pytest.approx(0.5)
    assert song.chords[0].notes == (48, 52, 55)
    assert song.chords[0].duration == pytest.approx(1.0)
    assert parse_dataset_bytes(gzip.compress(raw)) == songs


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"<html>504 Gateway Time-out</html>", "HTML gateway"),
        (gzip.compress(b"{not-json"), "not valid UTF-8 JSON"),
        (gzip.compress(json.dumps({"data": []}).encode()), "Unsupported dataset schema"),
    ],
)
def test_rejects_bad_or_unsupported_dataset(raw, message):
    with pytest.raises(DatasetError, match=message):
        parse_dataset_bytes(raw)


def test_rejects_invalid_note_pitch():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document["songs"][0]["melody"][0]["pitch"] = 200
    with pytest.raises(DatasetError, match="MIDI pitches"):
        parse_dataset_bytes(gzip.compress(json.dumps(document).encode()))


def test_selection_is_deterministic_and_skips_recorded_ids():
    songs = list(reversed(fixture_songs()))
    assert select_songs(songs, set(), 0) == []
    assert [s.song_id for s in select_songs(songs, set(), 2)] == [
        "fixture-song-001", "fixture-song-002"
    ]
    assert [s.song_id for s in select_songs(songs, {"fixture-song-001"}, 2)] == [
        "fixture-song-002"
    ]


def test_selection_skips_repeated_artist_and_track_across_song_ids():
    chords, melody = fixture_songs()[0].chords, fixture_songs()[0].melody
    songs = [
        Song("song-001", "Artist - Track Name", 96, chords, melody, "Artist", "Track Name"),
        Song("song-002", "Artist - Track-Name!", 96, chords, melody, "artist", "Track-Name!"),
        Song("song-003", "Other Artist - Different Track", 96, chords, melody,
             "Other Artist", "Different Track"),
    ]

    assert [song.song_id for song in select_songs(songs, set(), 3)] == [
        "song-001",
        "song-003",
    ]
    assert [song.song_id for song in select_songs(
        songs, set(), 1, already_generated_keys={"artist:artist:track:trackname"}
    )] == ["song-003"]


def test_keep_latest_video_removes_older_mp4s_only(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    older = output_dir / "older.mp4"
    latest = output_dir / "latest.mp4"
    other_file = output_dir / "notes.txt"
    older.write_bytes(b"older")
    latest.write_bytes(b"latest")
    other_file.write_text("keep", encoding="utf-8")
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(latest, ns=(2_000_000_000, 2_000_000_000))

    kept = keep_latest_video(output_dir)

    assert kept == latest
    assert not older.exists()
    assert latest.read_bytes() == b"latest"
    assert other_file.read_text(encoding="utf-8") == "keep"


def test_ledger_persists_duplicate_ids(tmp_path):
    db = tmp_path / "ledger.sqlite3"
    with Ledger(db) as ledger:
        ledger.record("fixture-song-001", Path("outputs/one.mp4"))
    with Ledger(db) as ledger:
        assert ledger.generated_ids() == {"fixture-song-001"}
        with pytest.raises(sqlite3.IntegrityError):
            ledger.record("fixture-song-001", Path("outputs/again.mp4"))


def test_ledger_can_refresh_existing_video_metadata(tmp_path):
    db = tmp_path / "ledger.sqlite3"
    with Ledger(db) as ledger:
        ledger.record("fixture-song-001", Path("outputs/old.mp4"))
        ledger.record("fixture-song-001", Path("outputs/refreshed.mp4"), replace_existing=True)
        assert ledger.generated_ids() == {"fixture-song-001"}
        row = ledger.connection.execute(
            "SELECT output_path FROM generated WHERE song_id = ?",
            ("fixture-song-001",),
        ).fetchone()
        assert row == ("outputs/refreshed.mp4",)


def test_ledger_migrates_existing_database_and_persists_identity_key(tmp_path):
    db = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(db)
    connection.execute(
        """
        CREATE TABLE generated (
            song_id TEXT PRIMARY KEY,
            output_path TEXT NOT NULL,
            generated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )
        """
    )
    connection.execute(
        "INSERT INTO generated(song_id, output_path) VALUES (?, ?)",
        ("previous-song", "outputs/previous.mp4"),
    )
    connection.commit()
    connection.close()

    with Ledger(db) as ledger:
        assert ledger.generated_ids() == {"previous-song"}
        ledger.record(
            "new-song",
            Path("outputs/new.mp4"),
            identity_key="artist:artist:track:newtrack",
        )
        assert ledger.generated_identity_keys() == {"artist:artist:track:newtrack"}


def test_piano_synthesis_is_stereo_audible_and_has_smooth_attack(tmp_path):
    song = fixture_songs()[0]
    wav_path = tmp_path / "piano.wav"
    synthesize(song, wav_path, duration_limit=3)

    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getnchannels() == 2
        assert wav.getframerate() == 44_100
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    samples = samples.reshape(-1, 2).astype(np.float64) / 32768
    assert np.sqrt(np.mean(samples**2)) > 0.08
    assert np.max(np.abs(samples)) < 0.98
    assert np.sqrt(np.mean(samples[:4410] ** 2)) > 0.02
    assert np.any(samples[:, 0] != samples[:, 1])


def test_midi_track_preserves_simultaneous_chords_and_melody():
    track = _midi_track(fixture_songs()[0], duration=3)
    assert track.startswith(b"MTrk")
    assert b"\x90" in track
    assert b"\x91" in track
    assert b"\xff\x51\x03\x07\xa1\x20" in track
    assert track.endswith(b"\xff\x2f\x00")


@pytest.mark.skipif(
    shutil.which("fluidsynth") is None and sys.platform != "win32",
    reason="FluidSynth is not installed",
)
def test_freepats_ydp_soundfont_generates_audible_stereo_audio(tmp_path):
    soundfont = (
        Path(__file__).parents[1]
        / "assets"
        / "soundfonts"
        / "YDP-GrandPiano-SF2.tar.bz2"
    )
    output = generate_song(
        fixture_songs()[0],
        tmp_path / "outputs",
        3,
        soundfont=soundfont,
        include_intro=False,
    )
    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(output), "-map", "0:a:0", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Audio: aac" in probe.stderr and "stereo" in probe.stderr
    decoded_audio = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0:a:0",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    samples = np.frombuffer(decoded_audio.stdout, dtype="<i2").astype(np.float64) / 32768
    assert len(samples) > 44_100 * 3 * 2
    assert np.sqrt(np.mean(samples**2)) > 0.02
    assert np.max(np.abs(samples)) < 0.98


def test_horizontal_piano_roll_shows_melody_and_chord_without_note_labels():
    song = fixture_songs()[0]
    artwork = Image.new("RGB", (32, 32), (12, 34, 56))
    image = _render_frame(song, now=0.2, duration=3, artwork=artwork)
    chord_y = round(152 + (108 - 48) * (1128 / 88) + (1128 / 176))
    melody_y = round(152 + (108 - 72) * (1128 / 88) + (1128 / 176))
    chord_pixel = image.getpixel((120, chord_y))
    melody_pixel = image.getpixel((120, melody_y))
    assert image.size == (720, 1280)
    assert chord_pixel == (240, 46, 73)
    assert melody_pixel == (66, 213, 187)
    assert image.getpixel((40, 40)) == (12, 34, 56)
    assert image.getpixel((400, 5)) == (13, 17, 24)
    assert image.getpixel((400, 1275)) == (23, 31, 40)
    assert sum(
        1
        for x in range(720)
        for y in range(1150, 1220)
        if min(image.getpixel((x, y))) > 235
    ) > 5


def test_itunes_artwork_requires_exact_artist_and_track_match(monkeypatch):
    song = parse_dataset_bytes(
        gzip.compress(
            json.dumps(
                {
                    "source-song": {
                        "tags": ["MELODY", "HARMONY"],
                        "hooktheory": {
                            "id": "source-song",
                            "artist": "sample-artist",
                            "song": "sample-title",
                        },
                        "alignment": {
                            "refined": {"beats": [0, 1], "times": [0.0, 0.5]}
                        },
                        "annotations": {
                            "melody": [
                                {"onset": 0, "offset": 1, "octave": 0, "pitch_class": 9}
                            ],
                            "harmony": [
                                {
                                    "onset": 0,
                                    "offset": 1,
                                    "root_pitch_class": 0,
                                    "root_position_intervals": [4, 3],
                                }
                            ],
                        },
                    }
                }
            ).encode()
        )
    )[0]
    cover = io.BytesIO()
    Image.new("RGB", (8, 8), (90, 120, 150)).save(cover, format="PNG")
    search_response = {
        "results": [
            {
                "artistName": "sample-artist",
                "trackName": "sample-title (Live)",
                "artworkUrl100": "https://example.test/wrong/100x100bb.jpg",
            },
            {
                "artistName": "SAMPLE ARTIST",
                "trackName": "Sample Title",
                "artworkUrl100": "https://example.test/right/100x100bb.jpg",
            },
        ]
    }
    responses = iter(
        [
            io.BytesIO(json.dumps(search_response).encode()),
            io.BytesIO(cover.getvalue()),
        ]
    )
    requested_urls: list[str] = []

    def fake_urlopen(request, timeout):
        assert timeout == 8
        requested_urls.append(request.full_url)
        return next(responses)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    artwork = _lookup_itunes_artwork(song)

    assert artwork is not None
    assert artwork.image.getpixel((0, 0)) == (90, 120, 150)
    assert artwork.artist == "SAMPLE ARTIST"
    assert artwork.track_name == "Sample Title"
    assert "600x600bb.jpg" in requested_urls[1]


def test_fallback_artwork_is_generated_and_deterministic():
    from music_shorts.pipeline import _fallback_artwork

    song = fixture_songs()[0]
    first = _fallback_artwork(song)
    second = _fallback_artwork(song)
    assert first.size == (256, 256)
    assert np.asarray(first).tobytes() == np.asarray(second).tobytes()


def test_intro_script_and_progression_stop_at_final_chord():
    original = fixture_songs()[0]
    song = Song(
        original.song_id,
        "sample-artist - sample-title",
        original.tempo,
        original.chords,
        (*original.melody, NoteEvent(2.2, 0.7, (80,))),
        "sample-artist",
        "sample title",
    )
    progression, duration = _progression_excerpt(song, 60)

    assert _intro_script(song) == (
        "Here is the chord progression for Sample Title by Sample Artist."
    )
    assert duration == pytest.approx(1.6)
    assert progression.duration == pytest.approx(duration)
    assert progression.melody[-1].start + progression.melody[-1].duration <= duration


def test_intro_and_outro_frames_include_requested_content():
    song = fixture_songs()[0]
    artwork = TrackArtwork(Image.new("RGB", (200, 200), "#5790c0"), "Sample Artist", "Sample Title")
    progression = _render_frame(song, 0.2, 1.6, artwork)
    intro = _render_intro_frame(progression, artwork, "This is how you can")
    outro = _render_outro_frame(progression)

    assert intro.size == (720, 1280)
    assert outro.size == (720, 1280)
    assert np.asarray(intro).tobytes() != np.asarray(progression).tobytes()
    assert np.asarray(outro).tobytes() != np.asarray(progression).tobytes()


def test_video_starts_with_clean_intro_card_for_thumbnail(tmp_path, monkeypatch):
    import music_shorts.pipeline as pipeline

    song = fixture_songs()[0]
    artwork = TrackArtwork(
        Image.new("RGB", (200, 200), "#5790c0"),
        "Sample Artist",
        "Sample Title",
    )
    progression_frame = Image.new("RGB", (720, 1280), "#345678")
    monkeypatch.setattr(pipeline, "_render_frame", lambda *_args: progression_frame)
    monkeypatch.setattr(pipeline, "_render_outro_frame", lambda image: image)

    audio_source = tmp_path / "source.wav"
    audio_destination = tmp_path / "mixed.wav"
    audio_duration = 0.1 + pipeline.INTRO_PAUSE_SECONDS + 0.1 + pipeline.OUTRO_SECONDS
    with wave.open(str(audio_source), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(44_100)
        audio.writeframes(bytes(round(audio_duration * 44_100) * 4))

    def fake_mix(_piano, destination, duration, _intro, intro_duration, pause):
        shutil.copyfile(audio_source, destination)
        return intro_duration + pause + duration + pipeline.OUTRO_SECONDS

    monkeypatch.setattr(pipeline, "_mix_intro_and_piano_audio", fake_mix)
    video_path = tmp_path / "intro.mp4"
    render_video(
        song,
        audio_source,
        video_path,
        0.1,
        artwork,
        intro_audio_path=audio_source,
        intro_duration=0.1,
        intro_cues=(CaptionCue("Intro caption", 0, 0.1),),
    )

    decoded = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    actual = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape((1280, 720, 3))
    clean_intro = np.asarray(_render_intro_frame(progression_frame, artwork, None))
    captioned_intro = np.asarray(
        _render_intro_frame(progression_frame, artwork, "Intro caption")
    )

    assert np.abs(actual.astype(np.int16) - clean_intro.astype(np.int16)).mean() < np.abs(
        actual.astype(np.int16) - captioned_intro.astype(np.int16)
    ).mean()


def test_intro_uses_faster_phrase_audio_and_matched_caption_cues(tmp_path, monkeypatch):
    import music_shorts.pipeline as pipeline

    class FakeVoice:
        calls = []

        def synthesize_wav(self, text, wav_file, *, syn_config):
            assert syn_config.length_scale == pytest.approx(INTRO_VOICE_LENGTH_SCALE)
            assert syn_config.volume == pytest.approx(INTRO_VOICE_VOLUME)
            self.calls.append(text)
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22_050)
            wav_file.writeframes(bytes(22_050 // 5 * 2))

    original = fixture_songs()[0]
    song = Song(
        original.song_id,
        original.title,
        original.tempo,
        original.chords,
        original.melody,
        "sample artist",
        "sample title",
    )
    voice = FakeVoice()
    monkeypatch.setattr(pipeline, "_piper_voice", lambda: voice)
    duration, cues = _synthesize_intro(song, tmp_path / "narration.wav")
    expected_phrases = _caption_phrases(_intro_script(song))

    assert tuple(cue.text for cue in cues) == expected_phrases
    assert len(cues) == 3
    assert voice.calls == [_intro_script(song)]
    assert cues[0].start == 0
    assert cues[0].end == pytest.approx(
        duration * len(cues[0].text) / sum(len(cue.text) for cue in cues)
    )
    assert cues[1].start == pytest.approx(cues[0].end)
    assert cues[1].end == pytest.approx(cues[2].start)
    assert cues[-1].end == pytest.approx(duration)
    assert duration == pytest.approx(0.2)
    with wave.open(str(tmp_path / "narration.wav"), "rb") as wav_file:
        assert wav_file.getnframes() / wav_file.getframerate() == pytest.approx(duration)


def test_caption_phrases_split_voice_into_readable_cues_and_use_comic_font():
    sentence = "Here is the chord progression for Orbion by Armin Van Buuren."
    captions = _caption_phrases(sentence)

    assert captions == (
        "Here is the chord",
        "progression for Orbion by",
        "Armin Van Buuren.",
    )
    assert " ".join(captions) == sentence
    assert INTRO_VOICE_LENGTH_SCALE < 1.0
    assert _font(32).getname()[0] == "Bangers"


def test_outro_blurs_backdrop_and_keeps_white_cta_text_sharp():
    source = Image.new("RGB", (720, 1280), "black")
    draw = ImageDraw.Draw(source)
    for x in range(0, 100, 4):
        draw.rectangle((x, 400, x + 1, 450), fill="white")

    outro = np.asarray(_render_outro_frame(source))
    source_patch = np.asarray(source)[400:451, 0:100]
    blurred_patch = outro[400:451, 0:100]
    like_caption = outro[560:600, 80:190]

    assert blurred_patch.std() < source_patch.std()
    assert np.count_nonzero(np.all(like_caption > 245, axis=2)) > 0


@pytest.mark.parametrize(
    ("value", "artist", "expected"),
    [
        ("armin-van-buuren", True, "Armin Van Buuren"),
        ("armin–van-buuren", True, "Armin Van Buuren"),
        ("orbion", False, "Orbion"),
        ("a state of trance", False, "A State Of Trance"),
    ],
)
def test_display_metadata_fixes_hyphens_and_capitalization(value, artist, expected):
    assert _format_display_name(value, artist=artist) == expected


@pytest.mark.parametrize(
    ("notes", "label"),
    [
        ((60, 64, 67), "C Major"),
        ((57, 60, 64), "A Minor"),
        ((60, 64, 67, 70), "C7"),
        ((60, 63, 66, 69), "C Diminished 7"),
    ],
)
def test_chord_caption_uses_chord_name_instead_of_note_list(notes, label):
    assert _chord_name(notes) == label


def test_fixture_generates_playable_mp4_and_second_run_skips(tmp_path):
    assert shutil.which("ffmpeg") or Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file()
    songs = fixture_songs()
    output_dir = tmp_path / "outputs"
    db = tmp_path / "state" / "generated.sqlite3"

    first = select_songs(songs, set(), 1)
    assert len(first) == 1
    output = generate_song(first[0], output_dir, duration_limit=3, include_intro=False)
    assert output.is_file() and output.stat().st_size > 1_000
    with Ledger(db) as ledger:
        ledger.record(first[0].song_id, output)
        assert select_songs(songs, ledger.generated_ids(), 1)[0].song_id == "fixture-song-002"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(output), "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    assert "Video: h264" in probe.stderr and "720x1280" in probe.stderr
    assert "30 fps" in probe.stderr
    assert "Audio: aac" in probe.stderr and "stereo" in probe.stderr
    assert "Duration: 00:00:04.60" in probe.stderr
    decoded_audio = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(output), "-map", "0:a:0",
         "-ac", "2", "-f", "s16le", "-"],
        capture_output=True, check=True,
    )
    audio_samples = np.frombuffer(decoded_audio.stdout, dtype="<i2").astype(np.float64) / 32768
    assert len(audio_samples) > 44_100 * 2
    assert np.sqrt(np.mean(audio_samples**2)) > 0.08
