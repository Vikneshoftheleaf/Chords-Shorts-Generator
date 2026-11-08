"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import DatasetError, Ledger, generate_song, load_dataset, select_songs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-shorts",
        description="Synthesize and render falling-note videos from Hooktheory gzip JSON.",
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset/Hooktheory.json.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--ledger", type=Path, default=Path("state/generated.sqlite3"))
    parser.add_argument("--count", type=int, default=1, help="maximum new videos to generate (default: 1)")
    parser.add_argument("--max-seconds", type=float, default=15.0, help="maximum excerpt duration (default: 15)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 0:
        print("error: --count must be zero or greater", file=sys.stderr)
        return 2
    if not 0 < args.max_seconds <= 60:
        print("error: --max-seconds must be greater than 0 and at most 60", file=sys.stderr)
        return 2
    try:
        songs = load_dataset(args.dataset)
        with Ledger(args.ledger) as ledger:
            selected = select_songs(songs, ledger.generated_ids(), args.count)
            if not selected:
                if args.count == 0:
                    print("No songs requested (--count 0).")
                else:
                    print("No eligible songs; all valid songs are already in the ledger.")
                return 0
            for song in selected:
                output = generate_song(song, args.output_dir, args.max_seconds)
                ledger.record(song.song_id, output)
                print(f"Generated {output} (song ID {song.song_id})")
    except (DatasetError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
