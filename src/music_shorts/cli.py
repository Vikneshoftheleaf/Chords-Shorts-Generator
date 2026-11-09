"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import (
    DatasetError,
    Ledger,
    _song_identity_key,
    generate_song,
    keep_latest_video,
    load_dataset,
    output_filename,
    select_songs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-shorts",
        description="Synthesize and render vertical piano-study videos from Hooktheory JSON.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/hooktheory_billboard_filtered.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--ledger", type=Path, default=Path("state/generated.sqlite3"))
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="number of videos to generate or refresh (0 or 1; upload before generating the next)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=60.0,
        help="maximum piano progression duration (default: 60 seconds)",
    )
    parser.add_argument(
        "--soundfont",
        type=Path,
        default=Path("assets/soundfonts/YDP-GrandPiano-SF2.tar.bz2"),
        help="SF2 file or FreePats YDP Grand Piano .tar.bz2 archive",
    )
    parser.add_argument(
        "--procedural-synth",
        action="store_true",
        help="use the built-in piano-style synth instead of FluidSynth/SF2",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="regenerate songs whose video already exists in the output directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count not in (0, 1):
        print("error: --count must be zero or one; publish the pending video before generating another", file=sys.stderr)
        return 2
    if not 0 < args.max_seconds <= 60:
        print("error: --max-seconds must be greater than 0 and at most 60", file=sys.stderr)
        return 2
    try:
        if args.count == 0:
            print("No songs requested (--count 0).")
            return 0
        songs = load_dataset(args.dataset)
        pending_video = keep_latest_video(args.output_dir)
        if pending_video is not None and not args.refresh:
            raise RuntimeError(
                f"A video is waiting in {pending_video}. Publish it successfully to remove it, "
                "or remove it manually before generating another."
            )
        with Ledger(args.ledger) as ledger:
            generated_ids = ledger.generated_ids()
            generated_identity_keys = ledger.generated_identity_keys()
            if args.refresh:
                songs_to_refresh = [
                    song for song in songs if (args.output_dir / output_filename(song)).is_file()
                ]
                selected = select_songs(songs_to_refresh, set(), args.count)
            else:
                selected = select_songs(
                    songs,
                    generated_ids,
                    args.count,
                    already_generated_keys=generated_identity_keys,
                )
            if not selected:
                if args.count == 0:
                    print("No songs requested (--count 0).")
                elif args.refresh:
                    print("No existing videos found to refresh.")
                else:
                    print("No eligible songs; all valid songs are already in the ledger.")
                return 0
            for song in selected:
                soundfont = None if args.procedural_synth else args.soundfont
                output = generate_song(song, args.output_dir, args.max_seconds, soundfont=soundfont)
                ledger.record(
                    song.song_id,
                    output,
                    identity_key=_song_identity_key(song),
                    replace_existing=args.refresh,
                )
                print(f"Generated {output} (song ID {song.song_id})")
    except (DatasetError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
