"""Dataset validation, deterministic selection, synthesis, rendering, and ledger."""

from __future__ import annotations

import gzip
import hashlib
import http.client
import io
import json
import math
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import wave
import warnings
import zipfile
import zlib
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


class DatasetError(ValueError):
    """The input is not a supported, validated fixture/song dataset."""


class RenderError(RuntimeError):
    """Audio/video generation failed."""


@dataclass(frozen=True)
class NoteEvent:
    start: float
    duration: float
    notes: tuple[int, ...]


@dataclass(frozen=True)
class Song:
    song_id: str
    title: str
    tempo: float
    chords: tuple[NoteEvent, ...]
    melody: tuple[NoteEvent, ...]
    artist: str = ""
    track_name: str = ""

    @property
    def duration(self) -> float:
        return max(
            (event.start + event.duration for event in (*self.chords, *self.melody)),
            default=0.0,
        )


@dataclass(frozen=True)
class TrackArtwork:
    image: Image.Image
    artist: str
    track_name: str


@dataclass(frozen=True)
class CaptionCue:
    text: str
    start: float
    end: float


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError(f"{label} must be a number.")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise DatasetError(f"{label} must be finite and at least {minimum}.")
    return result


def _parse_events(value: Any, label: str, *, melody: bool) -> tuple[NoteEvent, ...]:
    if not isinstance(value, list) or not value:
        raise DatasetError(f"{label} must be a non-empty list.")
    if len(value) > 20_000:
        raise DatasetError(f"{label} has too many events (maximum 20,000).")

    events: list[NoteEvent] = []
    for index, event in enumerate(value):
        prefix = f"{label}[{index}]"
        if not isinstance(event, dict):
            raise DatasetError(f"{prefix} must be an object.")
        start = _number(event.get("start"), f"{prefix}.start")
        duration = _number(event.get("duration"), f"{prefix}.duration", minimum=0.001)
        raw_notes = [event.get("pitch")] if melody else event.get("notes")
        if not isinstance(raw_notes, list) or not raw_notes:
            key = "pitch" if melody else "notes"
            raise DatasetError(f"{prefix}.{key} must contain at least one MIDI note.")
        notes: list[int] = []
        for note in raw_notes:
            if isinstance(note, bool) or not isinstance(note, int) or not 0 <= note <= 127:
                raise DatasetError(f"{prefix} notes must be integer MIDI pitches from 0 to 127.")
            notes.append(note)
        events.append(NoteEvent(start, duration, tuple(notes)))
    return tuple(events)


def _parse_normalized_dataset(document: dict[str, Any]) -> list[Song]:
    if not isinstance(document.get("songs"), list):
        raise DatasetError(
            "Unsupported dataset schema: expected an object with a 'songs' list "
            "(each song: id, title, tempo, chords, melody)."
        )

    songs: list[Song] = []
    seen: set[str] = set()
    for index, item in enumerate(document["songs"]):
        label = f"songs[{index}]"
        if not isinstance(item, dict):
            raise DatasetError(f"{label} must be an object.")
        song_id, title = item.get("id"), item.get("title")
        if not isinstance(song_id, (str, int)) or not str(song_id).strip():
            raise DatasetError(f"{label}.id must be a non-empty string or integer.")
        song_id = str(song_id)
        if song_id in seen:
            raise DatasetError(f"Duplicate song id {song_id!r} in dataset.")
        seen.add(song_id)
        if not isinstance(title, str) or not title.strip():
            raise DatasetError(f"{label}.title must be a non-empty string.")
        tempo = _number(item.get("tempo"), f"{label}.tempo", minimum=1)
        if tempo > 400:
            raise DatasetError(f"{label}.tempo must not exceed 400 BPM.")
        chords = _parse_events(item.get("chords"), f"{label}.chords", melody=False)
        melody = _parse_events(item.get("melody"), f"{label}.melody", melody=True)
        artist = item.get("artist", "")
        track_name = item.get("track_name", title.strip())
        if not isinstance(artist, str) or not isinstance(track_name, str):
            raise DatasetError(f"{label}.artist and {label}.track_name must be strings when provided.")
        songs.append(
            Song(song_id, title.strip(), tempo, chords, melody, artist.strip(), track_name.strip())
        )
    if not songs:
        raise DatasetError("Dataset contains no songs.")
    return songs


def _beat_seconds(beat: float, beats: list[float], times: list[float]) -> float:
    right = bisect_right(beats, beat)
    if right == 0:
        left, right = 0, 1
    elif right >= len(beats):
        left, right = len(beats) - 2, len(beats) - 1
    else:
        left = right - 1
    beat_span = beats[right] - beats[left]
    ratio = (beat - beats[left]) / beat_span
    return times[left] + ratio * (times[right] - times[left])


def _parse_hooktheory_dataset(document: dict[str, Any]) -> list[Song]:
    songs: list[Song] = []
    for source_id, item in document.items():
        if not isinstance(item, dict):
            continue
        tags = item.get("tags", [])
        if not isinstance(tags, list) or not {"MELODY", "HARMONY"}.issubset(tags):
            continue
        annotations = item.get("annotations")
        alignment = item.get("alignment")
        if not isinstance(annotations, dict) or not isinstance(alignment, dict):
            continue
        melody_data, harmony_data = annotations.get("melody"), annotations.get("harmony")
        if not isinstance(melody_data, list) or not melody_data:
            continue
        if not isinstance(harmony_data, list) or not harmony_data:
            continue

        anchors = None
        for name in ("refined", "user"):
            candidate = alignment.get(name)
            if not isinstance(candidate, dict):
                continue
            beats, times = candidate.get("beats"), candidate.get("times")
            if not isinstance(beats, list) or not isinstance(times, list) or len(beats) < 2:
                continue
            if len(beats) != len(times):
                continue
            parsed_beats = [_number(value, f"{source_id}.{name}.beats") for value in beats]
            parsed_times = [_number(value, f"{source_id}.{name}.times") for value in times]
            if any(b <= a for a, b in zip(parsed_beats, parsed_beats[1:])):
                continue
            if any(b <= a for a, b in zip(parsed_times, parsed_times[1:])):
                continue
            anchors = parsed_beats, parsed_times
            break
        if anchors is None:
            continue
        beats, times = anchors
        origin = times[0]

        def seconds(beat_value: Any, label: str) -> float:
            beat = _number(beat_value, label)
            return max(0.0, _beat_seconds(beat, beats, times) - origin)

        events: dict[str, list[NoteEvent]] = {"melody": [], "harmony": []}
        for event_index, event in enumerate(melody_data):
            label = f"{source_id}.annotations.melody[{event_index}]"
            if not isinstance(event, dict):
                raise DatasetError(f"{label} must be an object.")
            octave, pitch_class = event.get("octave"), event.get("pitch_class")
            if isinstance(octave, bool) or not isinstance(octave, int):
                raise DatasetError(f"{label}.octave must be an integer.")
            if isinstance(pitch_class, bool) or not isinstance(pitch_class, int) or not 0 <= pitch_class <= 11:
                raise DatasetError(f"{label}.pitch_class must be an integer from 0 to 11.")
            pitch = 60 + octave * 12 + pitch_class
            if not 0 <= pitch <= 127:
                raise DatasetError(f"{label} maps outside MIDI pitches 0 to 127.")
            start = seconds(event.get("onset"), f"{label}.onset")
            end = seconds(event.get("offset"), f"{label}.offset")
            if end <= start:
                raise DatasetError(f"{label}.offset must be after onset.")
            events["melody"].append(NoteEvent(start, end - start, (pitch,)))

        for event_index, event in enumerate(harmony_data):
            label = f"{source_id}.annotations.harmony[{event_index}]"
            if not isinstance(event, dict):
                raise DatasetError(f"{label} must be an object.")
            root = event.get("root_pitch_class")
            if isinstance(root, bool) or not isinstance(root, int) or not 0 <= root <= 11:
                raise DatasetError(f"{label}.root_pitch_class must be an integer from 0 to 11.")
            intervals = event.get("root_position_intervals", [])
            if not isinstance(intervals, list):
                raise DatasetError(f"{label}.root_position_intervals must be a list.")
            chord = [48 + root]
            for interval in intervals:
                if isinstance(interval, bool) or not isinstance(interval, int) or not 1 <= interval <= 24:
                    raise DatasetError(f"{label} has an invalid root-position interval.")
                chord.append(chord[-1] + interval)
            if any(not 0 <= pitch <= 127 for pitch in chord):
                raise DatasetError(f"{label} maps outside MIDI pitches 0 to 127.")
            start = seconds(event.get("onset"), f"{label}.onset")
            end = seconds(event.get("offset"), f"{label}.offset")
            if end <= start:
                raise DatasetError(f"{label}.offset must be after onset.")
            events["harmony"].append(NoteEvent(start, end - start, tuple(chord)))

        beat_rates = [
            (time_b - time_a) / (beat_b - beat_a)
            for beat_a, beat_b, time_a, time_b in zip(beats, beats[1:], times, times[1:])
        ]
        tempo = 60.0 / sorted(beat_rates)[len(beat_rates) // 2]
        metadata = item.get("hooktheory", {})
        artist = metadata.get("artist", "") if isinstance(metadata, dict) else ""
        title = metadata.get("song", "") if isinstance(metadata, dict) else ""
        display_title = " - ".join(part for part in (artist, title) if isinstance(part, str) and part)
        song_id = metadata.get("id", source_id) if isinstance(metadata, dict) else source_id
        songs.append(
            Song(
                str(song_id),
                display_title or str(song_id),
                tempo,
                tuple(events["harmony"]),
                tuple(events["melody"]),
                artist.strip() if isinstance(artist, str) else "",
                title.strip() if isinstance(title, str) else "",
            )
        )

    if not songs:
        raise DatasetError(
            "Unsupported dataset schema or no usable Hooktheory records: expected both melody and harmony "
            "annotations with beat-time alignment."
        )
    return songs


def parse_dataset_bytes(raw: bytes) -> list[Song]:
    """Parse the normalized fixture format or the native Sheet Sage Hooktheory export."""
    if not raw.startswith(b"\x1f\x8b"):
        preview = raw[:100].decode("utf-8", errors="replace").strip()
        if preview.lower().startswith(("<!doctype html", "<html")) or "504 Gateway" in preview:
            raise DatasetError(
                "Dataset is an HTML gateway/error response, not gzip data. "
                "Replace dataset/Hooktheory.json.gz with a verified gzip JSON export."
            )
        raise DatasetError("Dataset is not gzip data (missing gzip file signature).")
    try:
        decoded = gzip.decompress(raw)
    except (OSError, EOFError, zlib.error) as exc:
        raise DatasetError(f"Could not decompress gzip dataset: {exc}") from exc
    try:
        document = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Decompressed dataset is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise DatasetError("Unsupported dataset schema: expected a JSON object.")
    if isinstance(document.get("songs"), list):
        return _parse_normalized_dataset(document)
    if not document:
        raise DatasetError(
            "Unsupported dataset schema: expected a 'songs' list or a native Hooktheory record mapping."
        )
    return _parse_hooktheory_dataset(document)


def load_dataset(path: Path) -> list[Song]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"Cannot read dataset at {path}: {exc}") from exc
    return parse_dataset_bytes(raw)


def select_songs(songs: Iterable[Song], already_generated: set[str], count: int) -> list[Song]:
    """Choose the lexicographically first ungenerated IDs, regardless of source order."""
    if count < 0:
        raise ValueError("count must not be negative.")
    return [
        song
        for song in sorted(songs, key=lambda entry: entry.song_id)
        if song.song_id not in already_generated
    ][:count]


class Ledger:
    """Persistent SQLite source of truth for song IDs and generated file metadata."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generated (
                song_id TEXT PRIMARY KEY,
                output_path TEXT NOT NULL,
                generated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
            """
        )
        self.connection.commit()

    def generated_ids(self) -> set[str]:
        return {row[0] for row in self.connection.execute("SELECT song_id FROM generated")}

    def record(self, song_id: str, output_path: Path, *, replace_existing: bool = False) -> None:
        if replace_existing:
            self.connection.execute(
                """
                INSERT INTO generated(song_id, output_path) VALUES (?, ?)
                ON CONFLICT(song_id) DO UPDATE SET
                    output_path = excluded.output_path,
                    generated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                """,
                (song_id, output_path.as_posix()),
            )
        else:
            self.connection.execute(
                "INSERT INTO generated(song_id, output_path) VALUES (?, ?)",
                (song_id, output_path.as_posix()),
            )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


SAMPLE_RATE = 44_100
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 1280
FPS = 30
REVERB_TAIL_SECONDS = 2.5
PIANO_LOW = 21  # A0
PIANO_HIGH = 108  # C8, inclusive
BLACK_PITCH_CLASSES = {1, 3, 6, 8, 10}
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLUIDSYNTH_WINDOWS_URL = (
    "https://github.com/FluidSynth/fluidsynth/releases/download/v2.6.2/"
    "fluidsynth-v2.6.2-win10-x64-cpp11.zip"
)
FLUIDSYNTH_WINDOWS_SHA256 = "808295345d43c9c428f3fa87d29fbf97a9926f356df95559ff2689f3bd4072d7"
PIPER_VOICE_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
    "en/en_US/norman/medium/"
)
PIPER_VOICE_FILENAME = "en_US-norman-medium.onnx"
PIPER_VOICE_SHA256 = "b9739443232a80a59c7d18810dd856899bf16a7964725f5ab81ea49b1351cb71"
PIPER_VOICE_SIZE = 63_531_379
PIPER_CONFIG_FILENAME = f"{PIPER_VOICE_FILENAME}.json"
PIPER_CONFIG_SHA256 = "6c2db7f558a4a8deb9fe822583c1c5105f6c4e834dd0f9de8ad17a888ee9fe1d"
PIPER_CONFIG_SIZE = 4_968
INTRO_PAUSE_SECONDS = 1.0
INTRO_FADE_SECONDS = INTRO_PAUSE_SECONDS
INTRO_VOICE_LENGTH_SCALE = 0.85
INTRO_VOICE_VOLUME = 1.15
OUTRO_SECONDS = 3.0
INTRO_TEXT_TEMPLATE = "Here is the chord progression for {track_info}."


def _cache_directory() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "music-shorts" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "music-shorts"


def _download_verified_file(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    if destination.is_file() and destination.stat().st_size == expected_size:
        digest = hashlib.sha256()
        with destination.open("rb") as cached:
            for block in iter(lambda: cached.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() == expected_sha256:
            return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f"{destination.name}.{os.getpid()}.part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "music-shorts/1.0"})
        digest = hashlib.sha256()
        received = 0
        with urllib.request.urlopen(request, timeout=60) as response, partial_path.open("wb") as target:
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > expected_size:
                    raise RuntimeError(f"Download exceeded the expected size for {destination.name}.")
                digest.update(block)
                target.write(block)
        if received != expected_size:
            raise RuntimeError(
                f"Downloaded {destination.name} has {received} bytes; expected {expected_size}."
            )
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"SHA-256 verification failed for {destination.name}.")
        os.replace(partial_path, destination)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not download Piper voice file {destination.name}: {exc}."
        ) from exc
    finally:
        partial_path.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _piper_voice() -> Any:
    voice_directory = _cache_directory() / "piper" / "en_US-norman-medium-v1.0.0"
    model_path = voice_directory / PIPER_VOICE_FILENAME
    config_path = voice_directory / PIPER_CONFIG_FILENAME
    try:
        _download_verified_file(
            PIPER_VOICE_BASE_URL + PIPER_VOICE_FILENAME + "?download=true",
            model_path,
            expected_size=PIPER_VOICE_SIZE,
            expected_sha256=PIPER_VOICE_SHA256,
        )
        _download_verified_file(
            PIPER_VOICE_BASE_URL + PIPER_CONFIG_FILENAME + "?download=true",
            config_path,
            expected_size=PIPER_CONFIG_SIZE,
            expected_sha256=PIPER_CONFIG_SHA256,
        )
        from piper import PiperVoice

        return PiperVoice.load(str(model_path))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Could not initialize Piper en_US-norman-medium: {exc}") from exc


def _intro_script(song: Song) -> str:
    artist, track = _music_metadata(song)
    track = _format_display_name(track)
    artist = _format_display_name(artist, artist=True)
    if artist:
        return INTRO_TEXT_TEMPLATE.format(track_info=f"{track} by {artist}")
    return INTRO_TEXT_TEMPLATE.format(track_info=track)


def _caption_phrases(text: str, words_per_caption: int = 4) -> tuple[str, ...]:
    sentence_break = text.rfind(". ")
    if sentence_break >= 0:
        return (text[:sentence_break + 1], text[sentence_break + 2:])
    if "," in text:
        lead, continuation = text.split(",", 1)
        if lead.strip() and continuation.strip():
            return (f"{lead.strip()},", continuation.strip())
    words = text.split()
    return tuple(
        " ".join(words[index:index + words_per_caption])
        for index in range(0, len(words), words_per_caption)
    )


def _synthesize_intro(song: Song, destination: Path) -> tuple[float, tuple[CaptionCue, ...]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import wave as wave_module

        from piper.config import SynthesisConfig

        sample_rate = 22_050
        with wave_module.open(str(destination), "wb") as wav_file:
            _piper_voice().synthesize_wav(
                _intro_script(song),
                wav_file,
                syn_config=SynthesisConfig(
                    length_scale=INTRO_VOICE_LENGTH_SCALE,
                    volume=INTRO_VOICE_VOLUME,
                ),
            )
        with wave_module.open(str(destination), "rb") as wav_file:
            if (
                wav_file.getnchannels() != 1
                or wav_file.getsampwidth() != 2
                or wav_file.getframerate() != sample_rate
                or wav_file.getnframes() <= 0
            ):
                raise RuntimeError(
                    "Piper produced unexpected audio; expected non-empty 22.05 kHz mono 16-bit PCM."
                )
            duration = wav_file.getnframes() / sample_rate

        phrases = _caption_phrases(_intro_script(song))
        total_weight = sum(max(len(phrase), 1) for phrase in phrases)
        cues: list[CaptionCue] = []
        elapsed_weight = 0
        for phrase in phrases:
            start = duration * elapsed_weight / total_weight
            elapsed_weight += max(len(phrase), 1)
            end = duration * elapsed_weight / total_weight
            cues.append(CaptionCue(phrase, start, end))
        return duration, tuple(cues)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Could not synthesize intro narration: {exc}") from exc


def _fluidsynth_executable() -> str:
    installed = shutil.which("fluidsynth")
    if installed:
        return installed
    if sys.platform != "win32":
        raise RuntimeError(
            "FluidSynth was not found on PATH. Install FluidSynth, or use --procedural-synth."
        )

    local_app_data = Path(
        os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
    )
    install_dir = local_app_data / "music-shorts" / "tools" / "fluidsynth-2.6.2"
    executable = install_dir / "fluidsynth.exe"
    if executable.is_file():
        return str(executable)

    install_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_path = install_dir.parent / f"fluidsynth-{os.getpid()}.zip"
    staging_dir = install_dir.parent / f"fluidsynth-{os.getpid()}.tmp"
    try:
        request = urllib.request.Request(
            FLUIDSYNTH_WINDOWS_URL,
            headers={"User-Agent": "music-shorts/1.0"},
        )
        digest = hashlib.sha256()
        downloaded_size = 0
        with urllib.request.urlopen(request, timeout=30) as response, archive_path.open("wb") as target:
            while block := response.read(1024 * 1024):
                downloaded_size += len(block)
                if downloaded_size > 32 * 1024 * 1024:
                    raise RuntimeError("FluidSynth Windows download exceeded 32 MiB.")
                digest.update(block)
                target.write(block)
        if digest.hexdigest() != FLUIDSYNTH_WINDOWS_SHA256:
            raise RuntimeError("FluidSynth Windows download failed its SHA-256 integrity check.")

        staging_dir.mkdir()
        expanded_size = 0
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                parts = Path(entry.filename.replace("\\", "/")).parts
                if (
                    len(parts) != 3
                    or parts[-2].lower() != "bin"
                    or entry.is_dir()
                ):
                    continue
                if Path(parts[-1]).name != parts[-1]:
                    raise RuntimeError("FluidSynth archive contains an unsafe filename.")
                expanded_size += entry.file_size
                if entry.file_size > 16 * 1024 * 1024 or expanded_size > 32 * 1024 * 1024:
                    raise RuntimeError("FluidSynth Windows archive exceeded the extraction size limit.")
                with archive.open(entry) as source, (staging_dir / parts[-1]).open("wb") as target:
                    shutil.copyfileobj(source, target)
        if not (staging_dir / "fluidsynth.exe").is_file():
            raise RuntimeError("FluidSynth Windows archive did not contain fluidsynth.exe.")
        if install_dir.exists():
            shutil.rmtree(install_dir)
        os.replace(staging_dir, install_dir)
    except (OSError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            f"Could not install portable FluidSynth automatically: {exc}. "
            "Install FluidSynth on PATH, or use --procedural-synth."
        ) from exc
    finally:
        archive_path.unlink(missing_ok=True)
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return str(executable)


def _add_piano_note(
    audio: np.ndarray,
    start_sample: int,
    pitch: int,
    hold_seconds: float,
    velocity: float,
    pan: float,
) -> None:
    release_seconds = 0.09
    note_samples = max(1, int(hold_seconds * SAMPLE_RATE))
    start = max(0, start_sample)
    end = min(audio.shape[0], start_sample + note_samples + int(release_seconds * SAMPLE_RATE))
    if end <= start:
        return

    times = (np.arange(start, end, dtype=np.float64) - start_sample) / SAMPLE_RATE
    held = times < hold_seconds
    attack = np.sin(np.minimum(times / 0.008, 1.0) * (math.pi / 2)) ** 2
    release = np.ones_like(times)
    release[~held] = np.cos(
        np.minimum((times[~held] - hold_seconds) / release_seconds, 1.0) * (math.pi / 2)
    ) ** 2

    frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
    inharmonicity = min(0.0012, 0.00012 * (frequency / 440.0) ** 0.65)
    decay_seconds = min(3.2, max(0.65, 1.25 + 650.0 / max(frequency, 35.0)))
    envelope = (
        attack
        * release
        * (0.7 * np.exp(-times / decay_seconds) + 0.3 * np.exp(-times / (decay_seconds * 0.28)))
    )
    partials = (1.0, 0.56, 0.34, 0.22, 0.14, 0.09, 0.055, 0.032)
    mono = np.zeros_like(times)
    for partial, amplitude in enumerate(partials, start=1):
        partial_frequency = frequency * partial * math.sqrt(1 + inharmonicity * partial**2)
        detune = 0.0007 if partial > 1 else 0.00045
        mono += amplitude * (
            0.56 * np.sin(2 * math.pi * partial_frequency * (1 - detune) * times)
            + 0.44 * np.sin(2 * math.pi * partial_frequency * (1 + detune) * times)
        )
    mono *= envelope * velocity
    left_gain = math.sqrt(1.0 - pan)
    right_gain = math.sqrt(pan)
    audio[start:end, 0] += mono * left_gain
    audio[start:end, 1] += mono * right_gain


def _add_reverb(audio: np.ndarray) -> np.ndarray:
    dry = audio.copy()
    reverberant = dry.copy()
    delays = (1423, 1877, 2137, 2683)
    feedback = 0.67
    for index, delay in enumerate(delays):
        for repeat in range(1, 19):
            shift = delay * repeat
            if shift >= len(dry):
                break
            gain = 0.23 * feedback ** (repeat - 1)
            source = dry[:-shift]
            reverberant[shift:, 0] += gain * (
                source[:, 0] * (0.72 if index % 2 == 0 else 0.48)
                + source[:, 1] * (0.28 if index % 2 == 0 else 0.52)
            )
            reverberant[shift:, 1] += gain * (
                source[:, 1] * (0.72 if index % 2 == 0 else 0.48)
                + source[:, 0] * (0.28 if index % 2 == 0 else 0.52)
            )
    return dry * 0.76 + reverberant * 0.24


def synthesize(song: Song, destination: Path, duration_limit: float = 15.0) -> float:
    duration = min(max(song.duration, 0.5), duration_limit)
    if not math.isfinite(duration_limit) or duration_limit <= 0 or duration_limit > 60:
        raise ValueError("duration limit must be greater than zero and at most 60 seconds.")
    sample_count = int(math.ceil(duration * SAMPLE_RATE))
    audio = np.zeros((sample_count + int(REVERB_TAIL_SECONDS * SAMPLE_RATE), 2), dtype=np.float64)
    for event in song.chords:
        velocity = 0.2 / math.sqrt(len(event.notes))
        for note_index, pitch in enumerate(event.notes):
            _add_piano_note(
                audio,
                int(event.start * SAMPLE_RATE),
                pitch,
                event.duration,
                velocity,
                0.35 + ((pitch * 7 + note_index * 3) % 30) / 100,
            )
    for event in song.melody:
        for pitch in event.notes:
            _add_piano_note(
                audio,
                int(event.start * SAMPLE_RATE),
                pitch,
                event.duration,
                0.24,
                0.35 + (pitch % 7) * 0.05,
            )

    audio = _add_reverb(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    gain = min(0.19 / max(rms, 1e-8), 0.92 / max(peak, 1e-8))
    audio *= gain
    pcm = np.asarray(np.clip(audio, -1.0, 1.0) * 32767, dtype="<i2")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return duration


def _variable_length_quantity(value: int) -> bytes:
    encoded = [value & 0x7F]
    value >>= 7
    while value:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(encoded))


def _midi_track(song: Song, duration: float) -> bytes:
    events: list[tuple[int, int, bytes]] = [
        (0, -1, b"\xff\x51\x03\x07\xa1\x20"),
    ]
    ticks_per_second = 1920
    for channel, track_events, base_velocity in (
        (0, song.chords, 68),
        (1, song.melody, 82),
    ):
        for event in track_events:
            start = max(0.0, event.start)
            end = min(duration, event.start + event.duration)
            if start >= duration or end <= start:
                continue
            start_tick = round(start * ticks_per_second)
            end_tick = max(start_tick + 1, round(end * ticks_per_second))
            beat = start * song.tempo / 60
            nearest_beat = round(beat)
            on_beat = abs(beat - nearest_beat) < 0.08
            accent = 6 if on_beat and nearest_beat % 4 == 0 else 3 if on_beat else 0
            for pitch in event.notes:
                voicing = (pitch % 3 - 1) * 2 if channel == 0 else 0
                velocity = max(48, min(108, base_velocity + accent + voicing))
                events.append((start_tick, 1, bytes((0x90 | channel, pitch, velocity))))
                events.append((end_tick, 0, bytes((0x80 | channel, pitch, 0))))

    events.sort(key=lambda event: (event[0], event[1]))
    track = bytearray()
    previous_tick = 0
    for tick, _, message in events:
        track.extend(_variable_length_quantity(tick - previous_tick))
        track.extend(message)
        previous_tick = tick

    track_end_tick = round((duration + REVERB_TAIL_SECONDS) * ticks_per_second)
    track.extend(_variable_length_quantity(max(0, track_end_tick - previous_tick)))
    track.extend(b"\xff\x2f\x00")
    return b"MTrk" + struct.pack(">I", len(track)) + track


def synthesize_soundfont(
    song: Song,
    destination: Path,
    soundfont: Path,
    duration_limit: float = 15.0,
) -> float:
    if not math.isfinite(duration_limit) or not 0 < duration_limit <= 60:
        raise ValueError("duration limit must be greater than zero and at most 60 seconds.")
    if not soundfont.is_file():
        raise RuntimeError(
            f"SoundFont not found at {soundfont}. Download the FreePats YDP Grand Piano archive "
            "or pass --soundfont PATH."
        )
    resolved_soundfont = soundfont
    if soundfont.name.lower().endswith((".tar.bz2", ".tbz2")):
        archive_hash = hashlib.sha256()
        with soundfont.open("rb") as archive_file:
            for block in iter(lambda: archive_file.read(1024 * 1024), b""):
                archive_hash.update(block)
        resolved_soundfont = (
            Path(tempfile.gettempdir())
            / "music-shorts-soundfonts"
            / f"YDP-GrandPiano-{archive_hash.hexdigest()}.sf2"
        )
        if not resolved_soundfont.is_file():
            resolved_soundfont.parent.mkdir(parents=True, exist_ok=True)
            partial_path = resolved_soundfont.with_name(
                f"{resolved_soundfont.name}.{os.getpid()}.part"
            )
            try:
                with tarfile.open(soundfont, "r:bz2") as archive:
                    members = [
                        member for member in archive.getmembers()
                        if member.name.endswith("/YDP-GrandPiano-20160804.sf2")
                        and member.isfile()
                    ]
                    if len(members) != 1 or members[0].size > 256 * 1024 * 1024:
                        raise RuntimeError(
                            f"{soundfont} does not contain one valid-sized FreePats YDP Grand Piano SF2."
                        )
                    source = archive.extractfile(members[0])
                    if source is None:
                        raise RuntimeError(f"Could not read the YDP SoundFont from {soundfont}.")
                    with source, partial_path.open("wb") as extracted:
                        shutil.copyfileobj(source, extracted)
                os.replace(partial_path, resolved_soundfont)
            except tarfile.TarError as exc:
                raise RuntimeError(f"Could not read the YDP SoundFont archive {soundfont}: {exc}") from exc
            finally:
                partial_path.unlink(missing_ok=True)
        soundfont = resolved_soundfont

    with soundfont.open("rb") as soundfont_file:
        header = soundfont_file.read(12)
    if header[:4] != b"RIFF" or header[8:12] != b"sfbk":
        raise RuntimeError(f"{soundfont} is not a valid SF2 SoundFont.")

    fluidsynth = _fluidsynth_executable()

    duration = min(max(song.duration, 0.5), duration_limit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    midi_path = destination.with_suffix(".mid")
    track = _midi_track(song, duration)
    midi_path.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, 960) + track)
    command = [
        fluidsynth,
        "-ni",
        "-q",
        "-r",
        str(SAMPLE_RATE),
        "-C",
        "0",
        "-g",
        "0.35",
        "-o",
        "synth.reverb.active=1",
        "-o",
        "synth.reverb.room-size=0.72",
        "-o",
        "synth.reverb.damp=0.55",
        "-o",
        "synth.reverb.width=100",
        "-o",
        "synth.reverb.level=0.42",
        "-F",
        str(destination),
        "-T",
        "wav",
        str(soundfont),
        str(midi_path),
    ]
    try:
        environment = os.environ.copy()
        fluidsynth_directory = str(Path(fluidsynth).resolve().parent)
        environment["PATH"] = os.pathsep.join(
            (fluidsynth_directory, environment.get("PATH", ""))
        )
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        if result.returncode:
            raise RuntimeError(
                f"FluidSynth failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
            )
        with wave.open(str(destination), "rb") as wav:
            if wav.getnchannels() != 2 or wav.getframerate() != SAMPLE_RATE or wav.getsampwidth() != 2:
                raise RuntimeError(
                    "FluidSynth produced an unexpected WAV format; expected 44.1 kHz, stereo, 16-bit PCM."
                )
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()
        if samples.size == 0:
            raise RuntimeError("FluidSynth produced an empty WAV file.")
        audio = samples.reshape(-1, 2).astype(np.float64) / 32768
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))
        gain = min(0.25 / max(rms, 1e-8), 0.88 / max(peak, 1e-8))
        audio = np.asarray(np.clip(audio * gain, -1.0, 1.0) * 32767, dtype="<i2")
        with wave.open(str(destination), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(audio.tobytes())
    except OSError as exc:
        raise RuntimeError(f"Could not synthesize with FluidSynth: {exc}") from exc
    finally:
        midi_path.unlink(missing_ok=True)
    return duration


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.ImageFont:
    font_path = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "Bangers-Regular.ttf"
    try:
        return ImageFont.truetype(str(font_path), size)
    except OSError as exc:
        raise RuntimeError(f"Could not load the bundled comic font at {font_path}: {exc}") from exc


def _music_metadata(song: Song) -> tuple[str, str]:
    track_name = song.track_name or song.title
    artist = song.artist
    if not artist and " - " in song.title:
        artist, parsed_track = song.title.split(" - ", 1)
        track_name = song.track_name or parsed_track
    return artist.strip(), track_name.strip()


def _format_display_name(value: str, *, artist: bool = False) -> str:
    if artist:
        for separator in ("-", "‐", "‑", "‒", "–", "—", "−"):
            value = value.replace(separator, " ")
    return " ".join(value.split()).title()


def _normalize_music_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _lookup_itunes_artwork(song: Song) -> TrackArtwork | None:
    artist, track_name = _music_metadata(song)
    if not artist or not track_name:
        return None

    query = urllib.parse.urlencode(
        {"term": f"{artist} {track_name}", "entity": "song", "limit": 25}
    )
    request = urllib.request.Request(
        f"https://itunes.apple.com/search?{query}",
        headers={"User-Agent": "music-shorts/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            search_bytes = response.read(2 * 1024 * 1024 + 1)
        if len(search_bytes) > 2 * 1024 * 1024:
            raise ValueError("iTunes search response exceeded 2 MiB.")
        search_result = json.loads(search_bytes)
        results = search_result.get("results", []) if isinstance(search_result, dict) else []
        if not isinstance(results, list):
            raise ValueError("iTunes search response did not contain a results list.")
        matched_result = next(
            (
                result
                for result in results
                if isinstance(result, dict)
                and _normalize_music_name(str(result.get("artistName", "")))
                == _normalize_music_name(artist)
                and _normalize_music_name(str(result.get("trackName", "")))
                == _normalize_music_name(track_name)
                and isinstance(result.get("artworkUrl100"), str)
                and result["artworkUrl100"].startswith("https://")
                and isinstance(result.get("artistName"), str)
                and isinstance(result.get("trackName"), str)
            ),
            None,
        )
        if matched_result is None:
            warnings.warn(
                f"No exact iTunes artwork match for {artist} - {track_name}; using generated fallback art.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        artwork_url = matched_result["artworkUrl100"].replace("100x100bb", "600x600bb")
        image_request = urllib.request.Request(
            artwork_url,
            headers={"User-Agent": "music-shorts/1.0"},
        )
        with urllib.request.urlopen(image_request, timeout=8) as response:
            image_bytes = response.read(12 * 1024 * 1024 + 1)
        if len(image_bytes) > 12 * 1024 * 1024:
            raise ValueError("iTunes artwork exceeded 12 MiB.")
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.width > 4096 or source.height > 4096:
                raise ValueError("iTunes artwork dimensions exceeded 4096 pixels.")
            source.load()
            return TrackArtwork(
                source.convert("RGB"),
                matched_result["artistName"],
                matched_result["trackName"],
            )
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        warnings.warn(
            f"Could not load iTunes artwork for {artist} - {track_name} ({exc}); using generated fallback art.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _fallback_artwork(song: Song, size: int = 256) -> Image.Image:
    artist, track_name = _music_metadata(song)
    digest = hashlib.sha256(f"{artist}\0{track_name}".encode("utf-8")).digest()
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    start = tuple(28 + component // 5 for component in digest[:3])
    end = tuple(20 + component // 2 for component in digest[3:6])
    for y in range(size):
        blend = y / max(size - 1, 1)
        color = tuple(round(a * (1 - blend) + b * blend) for a, b in zip(start, end))
        draw.line((0, y, size, y), fill=color)
    center = (size // 2, size // 2)
    radius = size * 0.32
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        outline=(230, 235, 242),
        width=max(2, size // 48),
    )
    note_x, note_y = int(size * 0.45), int(size * 0.32)
    draw.line(
        (note_x + size // 10, note_y, note_x + size // 10, note_y + size // 4),
        fill=(245, 247, 250),
        width=max(3, size // 24),
    )
    draw.ellipse(
        (note_x, note_y + size // 5, note_x + size // 8, note_y + size * 0.28),
        fill=(245, 247, 250),
    )
    return image


def _track_artwork(song: Song) -> TrackArtwork:
    artwork = _lookup_itunes_artwork(song)
    if artwork is not None:
        return artwork
    artist, track_name = _music_metadata(song)
    return TrackArtwork(_fallback_artwork(song), artist, track_name)


def _fit_text(text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if not text:
        return text
    if font.getlength(text) <= max_width:
        return text
    shortened = text
    while shortened:
        candidate = shortened + "..."
        if font.getlength(candidate) <= max_width:
            return candidate
        shortened = shortened[:-1]
    return ""


def _chord_name(notes: tuple[int, ...]) -> str:
    pitch_classes = {pitch % 12 for pitch in notes}
    if not pitch_classes:
        return "No chord"
    root = min(notes) % 12
    intervals = {(pitch_class - root) % 12 for pitch_class in pitch_classes}
    qualities = (
        ({0, 4, 7, 11}, "Major 7"),
        ({0, 3, 7, 10}, "Minor 7"),
        ({0, 4, 7, 10}, "7"),
        ({0, 3, 6, 10}, "Half-diminished"),
        ({0, 3, 6, 9}, "Diminished 7"),
        ({0, 4, 8}, "Augmented"),
        ({0, 3, 6}, "Diminished"),
        ({0, 2, 7}, "Sus 2"),
        ({0, 5, 7}, "Sus 4"),
        ({0, 4, 7, 9}, "Major 6"),
        ({0, 3, 7, 9}, "Minor 6"),
        ({0, 4, 7}, "Major"),
        ({0, 3, 7}, "Minor"),
        ({0, 7}, "5"),
    )
    quality = next(
        (name for template, name in qualities if template.issubset(intervals)),
        "Chord",
    )
    separator = "" if quality == "7" else " "
    return f"{PITCH_NAMES[root]}{separator}{quality}"


def _cartoonize_artwork(image: Image.Image, size: int = 460) -> Image.Image:
    artwork = ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)
    artwork = artwork.filter(ImageFilter.MedianFilter(3))
    artwork = ImageOps.posterize(artwork, 4)
    return ImageEnhance.Color(artwork).enhance(1.35)


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _render_intro_frame(
    runner_background: Image.Image,
    artwork: TrackArtwork,
    caption_text: str | None,
) -> Image.Image:
    background = ImageOps.fit(
        runner_background.convert("RGB"),
        (VIDEO_WIDTH, VIDEO_HEIGHT),
        method=Image.Resampling.LANCZOS,
    ).filter(ImageFilter.GaussianBlur(8))
    background = ImageEnhance.Brightness(background).enhance(0.42)
    image = background.convert("RGBA")
    image.alpha_composite(Image.new("RGBA", image.size, (8, 12, 19, 105)))

    cover = _cartoonize_artwork(artwork.image)
    cover_left, cover_top = (VIDEO_WIDTH - cover.width) // 2, 118
    glow_size = 70
    glow_mask = Image.new("L", (cover.width + glow_size * 2, cover.height + glow_size * 2), 0)
    ImageDraw.Draw(glow_mask).rectangle(
        (glow_size, glow_size, glow_size + cover.width, glow_size + cover.height),
        fill=205,
    )
    glow_mask = glow_mask.filter(ImageFilter.GaussianBlur(32))
    glow = Image.new("RGBA", glow_mask.size, (255, 55, 134, 0))
    glow.putalpha(glow_mask)
    image.alpha_composite(glow, (cover_left - glow_size, cover_top - glow_size))
    image.paste(cover, (cover_left, cover_top))

    draw = ImageDraw.Draw(image)
    track_font, artist_font = _font(58), _font(38)
    track = _fit_text(_format_display_name(artwork.track_name), track_font, 660)
    artist = _fit_text(
        _format_display_name(artwork.artist, artist=True),
        artist_font,
        660,
    )
    draw.text(
        (VIDEO_WIDTH // 2, 680),
        track,
        font=track_font,
        fill="white",
        stroke_width=3,
        stroke_fill="#11131b",
        anchor="mm",
    )
    draw.text(
        (VIDEO_WIDTH // 2, 735),
        artist,
        font=artist_font,
        fill="#f3d9ee",
        stroke_width=2,
        stroke_fill="#11131b",
        anchor="mm",
    )

    if caption_text:
        caption_font = _font(45)
        lines = _wrap_text(caption_text, caption_font, 650)
        line_height = 56
        text_top = 1030 - line_height * (len(lines) - 1) // 2
        for index, line in enumerate(lines):
            draw.text(
                (VIDEO_WIDTH // 2, text_top + index * line_height),
                line,
                font=caption_font,
                fill="#fff5a8",
                stroke_width=4,
                stroke_fill="#15131a",
                anchor="mm",
            )
    return image.convert("RGB")


def _draw_outro_icon(
    draw: ImageDraw.ImageDraw,
    label: str,
    center_x: int,
    top: int,
    scale: int,
) -> None:
    red = "#f20d20"

    def point(x: int, y: int) -> tuple[int, int]:
        return x * scale, y * scale

    if label == "Like":
        wrist = (
            center_x - 34, top + 23, center_x - 18, top + 51
        )
        draw.rounded_rectangle(
            tuple(value * scale for value in wrist),
            radius=4 * scale,
            fill=red,
        )
        draw.polygon(
            [point(x, y) for x, y in (
                (center_x - 18, top + 24), (center_x - 6, top + 7),
                (center_x + 1, top - 3), (center_x + 8, top + 1),
                (center_x + 7, top + 15), (center_x + 25, top + 15),
                (center_x + 31, top + 20), (center_x + 25, top + 46),
                (center_x + 19, top + 51), (center_x - 14, top + 51),
            )],
            fill=red,
        )
    elif label == "Comment":
        draw.rounded_rectangle(
            tuple(value * scale for value in (
                center_x - 34, top, center_x + 34, top + 43
            )),
            radius=14 * scale,
            fill=red,
        )
        draw.polygon(
            [point(x, y) for x, y in (
                (center_x + 11, top + 38), (center_x + 28, top + 54),
                (center_x + 27, top + 38),
            )],
            fill=red,
        )
    else:
        path = [
            point(center_x - 31, top + 38),
            point(center_x - 22, top + 25),
            point(center_x - 11, top + 16),
            point(center_x + 4, top + 10),
            point(center_x + 20, top + 8),
        ]
        draw.line(path, fill=red, width=9 * scale, joint="curve")
        draw.polygon(
            [point(x, y) for x, y in (
                (center_x + 4, top - 4), (center_x + 36, top + 1),
                (center_x + 21, top + 29), (center_x + 20, top + 13),
            )],
            fill=red,
        )


def _render_outro_frame(frame: Image.Image) -> Image.Image:
    blurred_background = frame.convert("RGB").filter(ImageFilter.GaussianBlur(14))
    image = blurred_background.convert("RGBA")
    image.alpha_composite(Image.new("RGBA", image.size, (0, 0, 0, 62)))

    scale = 3
    cta_layer = Image.new("RGBA", (VIDEO_WIDTH * scale, VIDEO_HEIGHT * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(cta_layer)
    labels = ("Like", "Comment", "Share")
    centers = (135, 360, 585)
    for label, center_x in zip(labels, centers):
        _draw_outro_icon(draw, label, center_x, 502, scale)
        draw.text(
            (center_x * scale, 580 * scale),
            label,
            font=_font(34 * scale),
            fill="white",
            anchor="mm",
        )
    subscribe_box = (74, 664, VIDEO_WIDTH - 74, 802)
    draw.rounded_rectangle(
        tuple(value * scale for value in subscribe_box),
        radius=69 * scale,
        fill="#f20d20",
    )
    bell_x, bell_y = 166, 733
    draw.polygon(
        [(x * scale, y * scale) for x, y in (
            (bell_x - 22, bell_y + 12), (bell_x - 15, bell_y + 2),
            (bell_x - 13, bell_y - 19), (bell_x - 5, bell_y - 29),
            (bell_x + 6, bell_y - 29), (bell_x + 15, bell_y - 18),
            (bell_x + 17, bell_y + 2), (bell_x + 24, bell_y + 12),
        )],
        fill="white",
    )
    draw.ellipse(
        tuple(value * scale for value in (bell_x - 4, bell_y - 36, bell_x + 5, bell_y - 27)),
        fill="white",
    )
    draw.ellipse(
        tuple(value * scale for value in (bell_x - 4, bell_y + 12, bell_x + 5, bell_y + 21)),
        fill="white",
    )
    draw.text(
        (385 * scale, 733 * scale),
        "Subscribe",
        font=_font(54 * scale),
        fill="white",
        anchor="mm",
    )
    cta_layer = cta_layer.resize(image.size, Image.Resampling.LANCZOS)
    image.alpha_composite(cta_layer)
    return image.convert("RGB")


def _render_frame(
    song: Song,
    now: float,
    duration: float,
    artwork: TrackArtwork | Image.Image | None = None,
) -> Image.Image:
    image = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), "#0d1118")
    draw = ImageDraw.Draw(image)
    artist, track_name = _music_metadata(song)
    if artwork is None:
        artwork = TrackArtwork(_fallback_artwork(song), artist, track_name)
    if isinstance(artwork, Image.Image):
        artwork = TrackArtwork(artwork, artist, track_name)
    header_art = artwork.image.resize((112, 112), Image.Resampling.LANCZOS)
    image.paste(header_art, (20, 20))
    header_font = _font(28)
    artist_font = _font(21)
    track_label = _fit_text(
        _format_display_name(artwork.track_name),
        header_font,
        VIDEO_WIDTH - 170,
    )
    artist_label = _fit_text(
        _format_display_name(artwork.artist, artist=True),
        artist_font,
        VIDEO_WIDTH - 170,
    )
    draw.text((150, 34), track_label, font=header_font, fill="#ffffff")
    draw.text((152, 78), artist_label, font=artist_font, fill="#d0d5dc")
    current_chord = next(
        (event for event in song.chords if event.start <= now < event.start + event.duration),
        None,
    )
    chord_label = _chord_name(current_chord.notes) if current_chord else ""

    keyboard_right = 80
    roll_left, roll_right = keyboard_right, VIDEO_WIDTH
    roll_top, roll_bottom = 152, VIDEO_HEIGHT
    row_height = (roll_bottom - roll_top) / (PIANO_HIGH - PIANO_LOW + 1)
    lookahead = 3.2
    playhead_x = roll_left + 16
    pixels_per_second = (roll_right - playhead_x) / lookahead

    def row_bounds(pitch: int) -> tuple[int, int]:
        row = PIANO_HIGH - pitch
        top = round(roll_top + row * row_height)
        bottom = round(roll_top + (row + 1) * row_height)
        return top, max(top + 1, bottom)

    def event_x(event_time: float) -> float:
        return playhead_x + (event_time - now) * pixels_per_second

    beat_seconds = 60 / song.tempo
    first_beat = math.floor(now / beat_seconds)
    beat = first_beat
    while True:
        beat_time = beat * beat_seconds
        x = event_x(beat_time)
        if x > roll_right:
            break
        if x >= roll_left:
            color = "#303b47" if beat % 4 == 0 else "#202a34"
            draw.line((int(x), roll_top, int(x), roll_bottom), fill=color,
                      width=2 if beat % 4 == 0 else 1)
        beat += 1

    active_pitches = {
        pitch
        for event in (*song.chords, *song.melody)
        if event.start <= now < event.start + event.duration
        for pitch in event.notes
    }
    active_chord_pitches = {
        pitch
        for event in song.chords
        if event.start <= now < event.start + event.duration
        for pitch in event.notes
    }
    for pitch in range(PIANO_HIGH, PIANO_LOW - 1, -1):
        y0, y1 = row_bounds(pitch)
        is_black = pitch % 12 in BLACK_PITCH_CLASSES
        row_fill = "#121920" if is_black else "#171f28"
        draw.rectangle((roll_left, y0, roll_right, y1), fill=row_fill)
        draw.line((roll_left, y1, roll_right, y1), fill="#232e39", width=1)

        key_fill = "#1d242d" if is_black else "#dce1e6"
        if pitch in active_pitches:
            key_fill = "#d94858" if pitch in active_chord_pitches else "#34a995"
        key_width = 50 if is_black else keyboard_right - 4
        draw.rectangle((0, y0, key_width, y1), fill=key_fill,
                       outline="#4a535d" if is_black else "#a5adb5", width=1)

    def draw_notes(events: tuple[NoteEvent, ...], color: str, active_color: str) -> None:
        for event in events:
            event_end = event.start + event.duration
            if event_end <= now or event.start > now + lookahead:
                continue
            x0 = max(float(roll_left + 1), event_x(event.start))
            x1 = min(float(roll_right - 1), event_x(event_end))
            if x1 <= x0:
                continue
            fill = active_color if event.start <= now < event_end else color
            for pitch in event.notes:
                if not PIANO_LOW <= pitch <= PIANO_HIGH:
                    continue
                y0, y1 = row_bounds(pitch)
                draw.rounded_rectangle((int(x0), y0 + 1, int(x1), y1 - 1),
                                       radius=3, fill=fill)

    draw_notes(song.chords, "#842e3a", "#f02e49")
    draw_notes(song.melody, "#236c68", "#42d5bb")
    draw.line((playhead_x, roll_top, playhead_x, roll_bottom), fill="#f3f5f8", width=2)
    draw.rounded_rectangle((playhead_x - 5, roll_top, playhead_x + 5, roll_top + 9),
                           radius=3, fill="#f3f5f8")
    if chord_label:
        chord_font = _font(38)
        label_box = draw.textbbox((0, 0), chord_label, font=chord_font)
        label_width = label_box[2] - label_box[0]
        label_x = (VIDEO_WIDTH - label_width) // 2
        label_y = VIDEO_HEIGHT - 248
        draw.text(
            (label_x, label_y),
            chord_label,
            font=chord_font,
            fill="#ffffff",
            stroke_width=3,
            stroke_fill="#10141a",
        )
    return image


def _progression_excerpt(song: Song, duration_limit: float) -> tuple[Song, float]:
    if not math.isfinite(duration_limit) or not 0 < duration_limit <= 60:
        raise ValueError("duration limit must be greater than zero and at most 60 seconds.")
    last_chord_end = max(
        event.start + event.duration for event in song.chords
    )
    duration = min(last_chord_end, duration_limit)

    def clip_events(events: tuple[NoteEvent, ...]) -> tuple[NoteEvent, ...]:
        clipped: list[NoteEvent] = []
        for event in events:
            start = max(event.start, 0.0)
            end = min(event.start + event.duration, duration)
            if end > start:
                clipped.append(NoteEvent(start, end - start, event.notes))
        return tuple(clipped)

    return (
        Song(
            song.song_id,
            song.title,
            song.tempo,
            clip_events(song.chords),
            clip_events(song.melody),
            song.artist,
            song.track_name,
        ),
        duration,
    )


def _mix_intro_and_piano_audio(
    piano_path: Path,
    destination: Path,
    progression_duration: float,
    intro_path: Path | None,
    intro_duration: float,
    intro_pause_duration: float,
) -> float:
    total_duration = intro_duration + intro_pause_duration + progression_duration + OUTRO_SECONDS
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if intro_path is None:
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(piano_path),
            "-filter_complex",
            (
                f"[0:a]aresample={SAMPLE_RATE},"
                "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"apad=pad_dur={OUTRO_SECONDS},atrim=duration={total_duration:.6f},"
                "asetpts=PTS-STARTPTS[out]"
            ),
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            str(destination),
        ]
    else:
        delay_ms = round((intro_duration + intro_pause_duration) * 1000)
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(intro_path),
            "-i",
            str(piano_path),
            "-filter_complex",
            (
                f"[0:a]aresample={SAMPLE_RATE},"
                "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                "asetpts=PTS-STARTPTS[voice];"
                f"[1:a]aresample={SAMPLE_RATE},"
                "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"adelay=delays={delay_ms}:all=1[piano];"
                "[voice][piano]amix=inputs=2:duration=longest:"
                "dropout_transition=0:normalize=0,"
                f"apad=pad_dur={OUTRO_SECONDS},"
                f"atrim=duration={total_duration:.6f},"
                "asetpts=PTS-STARTPTS[out]"
            ),
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            str(destination),
        ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RenderError(
            f"FFmpeg audio mix failed ({result.returncode}): {result.stderr.strip()}"
        )
    if not destination.is_file() or destination.stat().st_size < 1_000:
        raise RenderError("Audio mix did not produce a valid WAV file.")
    return total_duration


def render_video(
    song: Song,
    audio_path: Path,
    output_path: Path,
    duration: float,
    artwork: TrackArtwork | None = None,
    *,
    intro_audio_path: Path | None = None,
    intro_duration: float = 0.0,
    intro_cues: tuple[CaptionCue, ...] = (),
) -> None:
    if not math.isfinite(duration) or duration <= 0 or duration > 60:
        raise ValueError("progression duration must be greater than zero and at most 60 seconds.")
    if not math.isfinite(intro_duration) or intro_duration < 0:
        raise ValueError("intro duration must be finite and at least zero.")
    if (intro_audio_path is None) != (intro_duration == 0):
        raise ValueError("intro audio path and duration must both be provided or omitted.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if artwork is None:
        artwork = _track_artwork(song)
    if intro_audio_path is None and intro_cues:
        raise ValueError("caption cues require intro narration.")
    intro_pause_duration = INTRO_PAUSE_SECONDS if intro_audio_path is not None else 0.0
    total_duration = intro_duration + intro_pause_duration + duration + OUTRO_SECONDS
    frame_count = max(1, math.ceil(total_duration * FPS))
    temp_dir = output_path.parent / f".{output_path.stem}.frames-{hashlib.sha1(song.song_id.encode()).hexdigest()[:8]}"
    mixed_audio_path = output_path.with_suffix(".mixed.wav")
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        opening_progression_frame = (
            _render_frame(song, 0.0, duration, artwork)
            if intro_audio_path is not None
            else None
        )
        intro_base_frame = (
            _render_intro_frame(opening_progression_frame, artwork, None)
            if opening_progression_frame is not None
            else None
        )
        caption_frames = {
            cue.text: _render_intro_frame(opening_progression_frame, artwork, cue.text)
            for cue in intro_cues
        } if opening_progression_frame is not None else {}
        last_progression_frame = _render_frame(
            song,
            max(0.0, duration - 1 / FPS),
            duration,
            artwork,
        )
        outro_frame = _render_outro_frame(last_progression_frame)
        for frame in range(frame_count):
            now = frame / FPS
            if intro_base_frame is not None and now < intro_duration:
                active_cue = next(
                    (cue for cue in intro_cues if cue.start <= now < cue.end),
                    None,
                )
                image = caption_frames[active_cue.text] if active_cue else intro_base_frame
            elif now >= intro_duration + intro_pause_duration + duration:
                image = outro_frame
            elif intro_base_frame is not None and now < intro_duration + intro_pause_duration:
                fade_progress = min(
                    (now - intro_duration) / INTRO_FADE_SECONDS,
                    1.0,
                )
                alpha = fade_progress * fade_progress * (3 - 2 * fade_progress)
                image = Image.blend(intro_base_frame, opening_progression_frame, alpha)
            else:
                progression_time = now - intro_duration - intro_pause_duration
                progression_frame = _render_frame(
                    song,
                    progression_time,
                    duration,
                    artwork,
                )
                image = progression_frame
            image.save(temp_dir / f"frame_{frame:05d}.png", optimize=False)
        total_duration = _mix_intro_and_piano_audio(
            audio_path,
            mixed_audio_path,
            duration,
            intro_audio_path,
            intro_duration,
            intro_pause_duration,
        )
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(temp_dir / "frame_%05d.png"),
            "-i",
            str(mixed_audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            f"{total_duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RenderError(f"FFmpeg failed ({result.returncode}): {result.stderr.strip()}")
        if not output_path.is_file() or output_path.stat().st_size < 1_000:
            raise RenderError("FFmpeg completed but did not create a valid video file.")
        probe = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(output_path), "-map", "0:a:0", "-f", "null", "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        if "Audio:" not in probe.stderr or "Video:" not in probe.stderr:
            raise RenderError("Generated MP4 is missing a decodable audio or video stream.")
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, RenderError):
            raise
        raise RenderError(f"Could not render MP4: {exc}") from exc
    finally:
        mixed_audio_path.unlink(missing_ok=True)
        for frame_path in temp_dir.glob("*.png"):
            frame_path.unlink(missing_ok=True)
        temp_dir.rmdir()


def output_filename(song: Song) -> str:
    readable = "".join(c if c.isalnum() or c in "-_" else "-" for c in song.song_id)
    readable = readable.strip("-_")[:48] or "song"
    digest = hashlib.sha256(song.song_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}.mp4"


def generate_song(
    song: Song,
    output_dir: Path,
    duration_limit: float = 15.0,
    soundfont: Path | None = None,
    *,
    include_intro: bool = True,
) -> Path:
    output_path = output_dir / output_filename(song)
    audio_path = output_path.with_suffix(".wav")
    intro_path = output_path.with_suffix(".intro.wav")
    try:
        progression, progression_duration = _progression_excerpt(song, duration_limit)
        if soundfont is None:
            synthesize(progression, audio_path, progression_duration)
        else:
            synthesize_soundfont(progression, audio_path, soundfont, progression_duration)
        artwork = _track_artwork(song)
        if include_intro:
            intro_duration, intro_cues = _synthesize_intro(song, intro_path)
        else:
            intro_duration, intro_cues = 0.0, ()
        render_video(
            progression,
            audio_path,
            output_path,
            progression_duration,
            artwork,
            intro_audio_path=intro_path if include_intro else None,
            intro_duration=intro_duration,
            intro_cues=intro_cues,
        )
    finally:
        audio_path.unlink(missing_ok=True)
        audio_path.with_suffix(".mid").unlink(missing_ok=True)
        intro_path.unlink(missing_ok=True)
    return output_path
