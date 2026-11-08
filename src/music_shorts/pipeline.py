"""Dataset validation, deterministic selection, synthesis, rendering, and ledger."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sqlite3
import wave
import zlib
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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

    @property
    def duration(self) -> float:
        return max(
            (event.start + event.duration for event in (*self.chords, *self.melody)),
            default=0.0,
        )


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
        songs.append(Song(song_id, title.strip(), tempo, chords, melody))
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

    def record(self, song_id: str, output_path: Path) -> None:
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
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 540
FPS = 12
PIANO_LOW = 21  # A0
PIANO_HIGH = 108  # C8, inclusive
BLACK_PITCH_CLASSES = {1, 3, 6, 8, 10}
NOTE_COLORS = ("#31d7c2", "#a889ff", "#ffb454", "#ff7189")


def synthesize(song: Song, destination: Path, duration_limit: float = 15.0) -> float:
    duration = min(max(song.duration, 0.5), duration_limit)
    if duration_limit <= 0 or duration > 60:
        raise ValueError("duration limit must be greater than zero and at most 60 seconds.")
    sample_count = int(math.ceil(duration * SAMPLE_RATE))
    audio = np.zeros(sample_count, dtype=np.float64)
    for event_index, event in enumerate((*song.chords, *song.melody)):
        start = max(0, int(event.start * SAMPLE_RATE))
        end = min(sample_count, int((event.start + event.duration) * SAMPLE_RATE))
        if end <= start:
            continue
        indices = np.arange(end - start, dtype=np.float64)
        event_seconds = indices / SAMPLE_RATE
        event_length = max((end - start) / SAMPLE_RATE, 0.001)
        attack = np.minimum(event_seconds / 0.015, 1.0)
        release = np.minimum((event_length - event_seconds) / 0.04, 1.0)
        envelope = np.maximum(0.0, np.minimum(attack, release))
        level = 0.10 / max(1, len(event.notes))
        for note_index, pitch in enumerate(event.notes):
            frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
            phase = (event_index * 0.17 + note_index * 0.31) % (2 * math.pi)
            audio[start:end] += (
                np.sin(2 * math.pi * frequency * event_seconds + phase) * envelope * level
            )
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.92:
        audio *= 0.92 / peak
    pcm = np.asarray(np.clip(audio, -1.0, 1.0) * 32767, dtype="<i2")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return duration


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",),
        ("arialbd.ttf" if bold else "arial.ttf",),
    )
    for (name,) in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _render_frame(song: Song, now: float, duration: float) -> Image.Image:
    image = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), "#11151c")
    draw = ImageDraw.Draw(image)
    title_font, small_font = _font(23, True), _font(13)
    draw.rectangle((0, 0, VIDEO_WIDTH, 62), fill="#1b222d")
    draw.text((24, 17), song.title[:52], font=title_font, fill="#f3f5f8")
    draw.text((VIDEO_WIDTH - 165, 24), f"{song.tempo:g} BPM  |  {now:04.1f}s",
              font=small_font, fill="#9eaab9")

    left, right, top, bottom = 36, VIDEO_WIDTH - 24, 86, 372
    draw.rectangle((left, top, right, bottom), fill="#161c25")
    for i in range(13):
        y = top + i * (bottom - top) // 12
        draw.line((left, y, right, y), fill="#27313e", width=1)
    lookahead = 3.5
    for pitch in range(PIANO_LOW, PIANO_HIGH + 1):
        x = left + (pitch - PIANO_LOW) * (right - left) / (PIANO_HIGH - PIANO_LOW + 1)
        if pitch % 12 == 0:
            draw.line((x, top, x, bottom), fill="#34404e", width=1)
    for event_index, event in enumerate((*song.chords, *song.melody)):
        event_end = event.start + event.duration
        if event_end < now or event.start > now + lookahead:
            continue
        y_bottom = bottom - (event.start - now) / lookahead * (bottom - top)
        y_top = y_bottom - max(7, event.duration / lookahead * (bottom - top))
        for note_index, pitch in enumerate(event.notes):
            if pitch < PIANO_LOW or pitch > PIANO_HIGH:
                continue
            x0 = left + (pitch - PIANO_LOW) * (right - left) / 88
            x1 = x0 + max(4, (right - left) / 88 - 2)
            draw.rounded_rectangle(
                (int(x0), int(max(top, y_top)), int(x1), int(min(bottom, y_bottom))),
                radius=3,
                fill=NOTE_COLORS[(event_index + note_index) % len(NOTE_COLORS)],
            )

    key_top, key_bottom = 394, 496
    draw.rectangle((24, key_top - 8, VIDEO_WIDTH - 24, key_bottom + 1), fill="#202936")
    key_width = (VIDEO_WIDTH - 48) / 88
    active = {
        pitch
        for event in (*song.chords, *song.melody)
        if event.start <= now < event.start + event.duration
        for pitch in event.notes
    }
    for pitch in range(PIANO_LOW, PIANO_HIGH + 1):
        x0 = 24 + (pitch - PIANO_LOW) * key_width
        x1 = x0 + key_width
        is_black = pitch % 12 in BLACK_PITCH_CLASSES
        if not is_black:
            color = "#36d9c0" if pitch in active else "#e8edf2"
            draw.rectangle((int(x0), key_top, int(x1), key_bottom), fill=color,
                          outline="#647080", width=1)
    for pitch in range(PIANO_LOW, PIANO_HIGH + 1):
        if pitch % 12 not in BLACK_PITCH_CLASSES:
            continue
        x0 = 24 + (pitch - PIANO_LOW) * key_width
        x1 = x0 + key_width * 0.65
        color = "#2ab7a4" if pitch in active else "#10141a"
        draw.rectangle((int(x0), key_top, int(x1), key_top + 62), fill=color,
                      outline="#596473", width=1)
    draw.text((24, 512), "88-KEY RANGE  A0 — C8", font=small_font, fill="#8f9bab")
    progress_width = int((VIDEO_WIDTH - 48) * min(1.0, now / max(duration, 0.001)))
    draw.rectangle((24, 532, VIDEO_WIDTH - 24, 537), fill="#303a47")
    draw.rectangle((24, 532, 24 + progress_width, 537), fill="#31d7c2")
    return image


def render_video(song: Song, audio_path: Path, output_path: Path, duration: float) -> None:
    if duration <= 0 or duration > 60:
        raise ValueError("render duration must be greater than zero and at most 60 seconds.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, math.ceil(duration * FPS))
    temp_dir = output_path.parent / f".{output_path.stem}.frames-{hashlib.sha1(song.song_id.encode()).hexdigest()[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for frame in range(frame_count):
            image = _render_frame(song, frame / FPS, duration)
            image.save(temp_dir / f"frame_{frame:05d}.png", optimize=False)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", str(temp_dir / "frame_%05d.png"), "-i", str(audio_path),
            "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", str(output_path),
        ]
        import subprocess

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RenderError(f"FFmpeg failed ({result.returncode}): {result.stderr.strip()}")
        if not output_path.is_file() or output_path.stat().st_size < 1_000:
            raise RenderError("FFmpeg completed but did not create a valid video file.")
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, RenderError):
            raise
        raise RenderError(f"Could not render MP4: {exc}") from exc
    finally:
        for frame_path in temp_dir.glob("*.png"):
            frame_path.unlink(missing_ok=True)
        temp_dir.rmdir()


def output_filename(song: Song) -> str:
    readable = "".join(c if c.isalnum() or c in "-_" else "-" for c in song.song_id)
    readable = readable.strip("-_")[:48] or "song"
    digest = hashlib.sha256(song.song_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}.mp4"


def generate_song(song: Song, output_dir: Path, duration_limit: float = 15.0) -> Path:
    output_path = output_dir / output_filename(song)
    audio_path = output_path.with_suffix(".wav")
    try:
        duration = synthesize(song, audio_path, duration_limit)
        render_video(song, audio_path, output_path, duration)
    finally:
        audio_path.unlink(missing_ok=True)
    return output_path
