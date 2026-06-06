"""Extract first-frame left/right pairs from MV-HEVC stereo videos.

Usage:
    python scripts/extract_frames.py --input-dir data/raw --output-dir data/frames
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from experiments.lib.io_video import extract_stereo_frame


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="data/raw")
    p.add_argument("--output-dir", default="data/frames")
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--pattern", default="*.mov")
    args = p.parse_args(argv)

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(in_dir.glob(args.pattern))
    if not videos:
        print(f"no videos matching {args.pattern} in {in_dir}")
        return 1

    for video in videos:
        left, right = extract_stereo_frame(video, args.frame_index)
        for side, img in (("left", left), ("right", right)):
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{video.stem}_{side}.png"), bgr)
        print(f"{video.name}: {left.shape[1]}x{left.shape[0]} -> {video.stem}_left/right.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
