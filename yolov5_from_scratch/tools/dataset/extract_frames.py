import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from yolov5_from_scratch.paths import DATASETS_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Uniformly sample frames from videos for annotation"
    )
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATASETS_DIR / "traffic_pilot"),
    )
    parser.add_argument("--frames-per-video", type=int, default=120)
    parser.add_argument(
        "--every-seconds",
        type=float,
        default=0.0,
        help="Use a fixed time interval instead of --frames-per-video",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Downscale extracted frames; 0 keeps original size",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resize_frame(frame, max_width):
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    height = max(1, int(round(frame.shape[0] * max_width / frame.shape[1])))
    return cv2.resize(
        frame,
        (max_width, height),
        interpolation=cv2.INTER_AREA,
    )


def sample_indices(frame_count, fps, frames_per_video, every_seconds):
    if frame_count <= 0:
        return []
    if every_seconds > 0:
        step = max(1, int(round(fps * every_seconds)))
        return list(range(0, frame_count, step))
    if frames_per_video <= 0:
        raise ValueError("--frames-per-video must be greater than 0")
    count = min(frame_count, frames_per_video)
    if count == 1:
        return [frame_count // 2]
    return sorted(
        set(
            int(round(value))
            for value in np.linspace(0, frame_count - 1, count)
        )
    )


def extract_video(
    source,
    images_dir,
    writer,
    args,
    source_index,
):
    source = Path(source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    indices = sample_indices(
        frame_count,
        fps,
        args.frames_per_video,
        args.every_seconds,
    )

    prefix = f"{source_index:02d}_{source.stem}"
    saved = 0
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                print(f"warning: could not read {source} frame {frame_index}")
                continue

            original_height, original_width = frame.shape[:2]
            frame = resize_frame(frame, args.max_width)
            output_name = f"{prefix}_{frame_index:06d}.jpg"
            output_path = images_dir / output_name
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output already exists: {output_path}. Use --overwrite."
                )

            ok = cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            if not ok:
                raise RuntimeError(f"Could not write image: {output_path}")

            writer.writerow(
                [
                    output_name,
                    str(source.resolve()),
                    frame_index,
                    f"{frame_index / fps:.3f}",
                    original_width,
                    original_height,
                    frame.shape[1],
                    frame.shape[0],
                    "unlabeled",
                ]
            )
            saved += 1
    finally:
        capture.release()

    print(
        f"{source}: sampled={len(indices)} saved={saved} "
        f"fps={fps:.3f} frames={frame_count}",
        flush=True,
    )
    return saved


def main():
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.max_width < 0:
        raise ValueError("--max-width must be 0 or greater")

    output_dir = Path(args.output)
    images_dir = output_dir / "images" / "unlabeled"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "frames.csv"

    total = 0
    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "image",
                "source",
                "frame_index",
                "timestamp_seconds",
                "original_width",
                "original_height",
                "saved_width",
                "saved_height",
                "status",
            ]
        )
        for source_index, source in enumerate(args.source, start=1):
            total += extract_video(
                source,
                images_dir,
                writer,
                args,
                source_index,
            )

    print(f"saved images: {total}", flush=True)
    print(f"images_dir: {images_dir.resolve()}", flush=True)
    print(f"manifest: {manifest_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
