"""Video frame extraction helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg


def extract_first_frame(video: Path, output: Path) -> Path:
    """Extract frame zero from a video into a new image file."""
    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    try:
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-n",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not run FFmpeg: {exc}") from exc

    if result.returncode or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(
            "Could not extract the video's first frame: "
            f"{result.stderr.strip() or 'FFmpeg produced no image.'}"
        )
    return output
