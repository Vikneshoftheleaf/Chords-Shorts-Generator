"""Generate and validate a video in outputs/ using the real dataset."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

from music_shorts.pipeline import (
    Ledger,
    generate_song,
    load_dataset,
    output_filename,
    select_songs,
)

ROOT = Path(__file__).resolve().parents[1]
SOUNDFONT = ROOT / "assets" / "soundfonts" / "YDP-GrandPiano-SF2.tar.bz2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "dataset" / "Hooktheory.json.gz",
        help="path to the Hooktheory gzip JSON dataset",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=60.0,
        help="maximum video duration in seconds (default: 60, the Shorts limit)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="directory to keep the generated MP4 (default: outputs)",
    )
    parser.add_argument(
        "--soundfont",
        type=Path,
        default=SOUNDFONT,
        help="SF2 piano soundfont or FreePats YDP .tar.bz2 archive (default: YDP Grand Piano)",
    )
    parser.add_argument(
        "--procedural-synth",
        action="store_true",
        help="use the built-in piano-style synth instead of FluidSynth/SF2",
    )
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        parser.error("--seconds must be greater than 0 and at most 60")

    try:
        songs = load_dataset(args.dataset)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        available_songs = [
            song
            for song in songs
            if not (args.output_dir / output_filename(song)).exists()
        ]
        selected = select_songs(available_songs, set(), 1)
        if not selected:
            raise RuntimeError(f"No song without an existing video was found in {args.output_dir}.")
        song = selected[0]

        with tempfile.TemporaryDirectory(prefix="music-shorts-test-") as temp:
            temp_dir = Path(temp)
            soundfont = None if args.procedural_synth else args.soundfont
            output = generate_song(song, args.output_dir, args.seconds, soundfont=soundfont)
            if not output.is_file() or output.stat().st_size <= 1_000:
                raise RuntimeError("Generation did not produce a non-empty MP4.")

            with Ledger(temp_dir / "state" / "generated.sqlite3") as ledger:
                ledger.record(song.song_id, output)
                if song.song_id not in ledger.generated_ids():
                    raise RuntimeError("The generated song ID was not saved to the SQLite ledger.")
                next_song = select_songs(songs, ledger.generated_ids(), 1)[0]
                if next_song.song_id == song.song_id:
                    raise RuntimeError("The ledger did not prevent duplicate selection.")

            completed = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-hide_banner",
                    "-i",
                    str(output),
                    "-f",
                    "null",
                    "-",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            details = completed.stderr
            if (
                "720x1280" not in details
                or "30 fps" not in details
                or "Audio: aac" not in details
                or "stereo" not in details
            ):
                raise RuntimeError("The generated MP4 is missing portrait video or stereo AAC audio.")
            if not re.search(r"Duration: \d{2}:\d{2}:\d{2}\.\d{2}", details):
                raise RuntimeError("Could not verify the generated MP4 duration.")
            print(f"PASS: generated and decoded {output.resolve()} ({output.stat().st_size:,} bytes)")
            print(f"Song: {song.title} (ID: {song.song_id})")
            print("PASS: song ID persisted and excluded from duplicate selection")
            print("The video is kept; the temporary SQLite ledger is removed automatically.")
    except (OSError, RuntimeError, ValueError, IndexError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
