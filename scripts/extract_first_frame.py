"""Extract the first video frame as an image."""

from __future__ import annotations

import argparse
from pathlib import Path

from music_shorts.video_utils import extract_first_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="path to the input video")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="image path (default: <video>_first_frame.jpg)",
    )
    args = parser.parse_args()

    video = args.video.expanduser()
    if not video.is_file():
        parser.error(f"video does not exist: {video}")
    output = args.output or video.with_name(f"{video.stem}_first_frame.jpg")
    output = output.expanduser()
    if output.exists():
        parser.error(f"output already exists: {output}")

    try:
        extract_first_frame(video, output)
    except (OSError, RuntimeError) as exc:
        print(f"Could not extract first frame: {exc}")
        return 1

    print(f"Saved first frame to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
