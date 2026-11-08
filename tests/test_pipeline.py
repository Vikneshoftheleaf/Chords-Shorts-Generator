from __future__ import annotations

import gzip
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest
import imageio_ffmpeg

from music_shorts.pipeline import (
    DatasetError,
    Ledger,
    generate_song,
    parse_dataset_bytes,
    select_songs,
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
    songs = parse_dataset_bytes(gzip.compress(json.dumps(document).encode()))

    assert len(songs) == 1
    song = songs[0]
    assert song.song_id == "source-song"
    assert song.title == "sample-artist - sample-title"
    assert song.tempo == pytest.approx(120)
    assert song.melody[0].notes == (69,)
    assert song.melody[0].start == pytest.approx(0.5)
    assert song.melody[0].duration == pytest.approx(0.5)
    assert song.chords[0].notes == (48, 52, 55)
    assert song.chords[0].duration == pytest.approx(1.0)


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
    assert [s.song_id for s in select_songs(songs, set(), 2)] == [
        "fixture-song-001", "fixture-song-002"
    ]
    assert [s.song_id for s in select_songs(songs, {"fixture-song-001"}, 2)] == [
        "fixture-song-002"
    ]


def test_ledger_persists_duplicate_ids(tmp_path):
    db = tmp_path / "ledger.sqlite3"
    with Ledger(db) as ledger:
        ledger.record("fixture-song-001", Path("outputs/one.mp4"))
    with Ledger(db) as ledger:
        assert ledger.generated_ids() == {"fixture-song-001"}
        with pytest.raises(sqlite3.IntegrityError):
            ledger.record("fixture-song-001", Path("outputs/again.mp4"))


def test_fixture_generates_playable_mp4_and_second_run_skips(tmp_path):
    assert shutil.which("ffmpeg") or Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file()
    songs = fixture_songs()
    output_dir = tmp_path / "outputs"
    db = tmp_path / "state" / "generated.sqlite3"

    first = select_songs(songs, set(), 1)
    assert len(first) == 1
    output = generate_song(first[0], output_dir, duration_limit=3)
    assert output.is_file() and output.stat().st_size > 1_000
    with Ledger(db) as ledger:
        ledger.record(first[0].song_id, output)
        assert select_songs(songs, ledger.generated_ids(), 1)[0].song_id == "fixture-song-002"
    # Decode both streams to ensure this is playable, not merely an MP4-named file.
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(output), "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
