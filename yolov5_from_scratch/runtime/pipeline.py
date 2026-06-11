import csv
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from .geometry import (
    output_dimensions,
    predict_frame,
    resize_for_output,
    scale_detections,
)
from .media import IMAGE_EXTENSIONS, iter_sources, open_capture, output_name
from .rendering import (
    detections_for_display,
    draw_detections,
    draw_line_and_counts,
    draw_performance,
    draw_roi,
)
from .tracking import update_tracks


DETECTION_COLUMNS = [
    "source",
    "frame",
    "timestamp_seconds",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "crossing_direction",
]


class NullWriter:
    def writerow(self, _row):
        return None


def _write_detection(csv_writer, source, frame_index, timestamp, detection, track_id, direction=""):
    csv_writer.writerow(
        [
            str(source),
            frame_index,
            f"{timestamp:.3f}",
            track_id if track_id is not None else "",
            detection.class_id,
            detection.class_name,
            f"{detection.confidence:.6f}",
            *(f"{value:.2f}" for value in detection.xyxy),
            direction or "",
        ]
    )


def _render_frame(frame, detections, track_ids, line_counter, inference_ms, args):
    rendered, scale_x, scale_y = resize_for_output(frame, args.output_width)
    visible, visible_track_ids = detections_for_display(
        detections,
        track_ids,
        args.display_conf,
    )
    draw_detections(
        rendered,
        scale_detections(visible, scale_x, scale_y),
        visible_track_ids,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf,
    )
    draw_line_and_counts(rendered, line_counter)
    if args.draw_roi:
        draw_roi(rendered, args.roi)
    draw_performance(rendered, inference_ms)
    return rendered


def process_image(source, predictor, tracker, line_counter, save_dir, csv_writer, args):
    image = cv2.imread(str(source))
    if image is None:
        raise RuntimeError(f"Could not read image: {source}")

    started = time.perf_counter()
    detections = predict_frame(predictor, image, args.roi)
    inference_ms = (time.perf_counter() - started) * 1000.0
    track_ids = (
        update_tracks(tracker, detections, args.track_conf)
        if args.track
        else [None] * len(detections)
    )
    for detection, track_id in zip(detections, track_ids):
        _write_detection(csv_writer, source, 0, 0.0, detection, track_id)

    rendered = _render_frame(
        image,
        detections,
        track_ids,
        line_counter,
        inference_ms,
        args,
    )
    output_path = None
    if args.save_media:
        output_path = save_dir / output_name(source, Path(source).suffix.lower())
        cv2.imwrite(str(output_path), rendered)
    print(
        f"image: {source} | detections={len(detections)} "
        f"| inference={inference_ms:.1f} ms "
        f"| saved={output_path or 'disabled'}",
        flush=True,
    )


def process_video(
    source,
    predictor,
    tracker,
    line_counter,
    save_dir,
    csv_writer,
    summary_writer,
    args,
):
    capture = open_capture(source)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError(f"Could not read the first frame from: {source}")
    height, width = first_frame.shape[:2]
    output_width, output_height = output_dimensions(width, height, args.output_width)

    output_path = None
    writer = None
    if args.save_media:
        output_path = save_dir / output_name(source, ".mp4")
        if len(args.codec) != 4:
            capture.release()
            raise ValueError("--codec must contain exactly four characters")
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*args.codec),
            fps,
            (output_width, output_height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {output_path}")

    tracker.reset()
    line_counter.reset()
    frame_index = 0
    inference_times = []
    pending_frame = first_frame
    try:
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                ok, frame = capture.read()
                if not ok:
                    break

            started = time.perf_counter()
            detections = predict_frame(predictor, frame, args.roi)
            inference_ms = (time.perf_counter() - started) * 1000.0
            inference_times.append(inference_ms)
            track_ids = (
                update_tracks(tracker, detections, args.track_conf)
                if args.track
                else [None] * len(detections)
            )
            timestamp = frame_index / fps

            for detection, track_id in zip(detections, track_ids):
                direction = ""
                if track_id is not None:
                    direction = line_counter.update(
                        tracker.tracks[track_id],
                        frame.shape,
                        detection.class_name,
                    )
                _write_detection(
                    csv_writer,
                    source,
                    frame_index,
                    timestamp,
                    detection,
                    track_id,
                    direction,
                )

            rendered = _render_frame(
                frame,
                detections,
                track_ids,
                line_counter,
                inference_ms,
                args,
            )
            if writer is not None:
                writer.write(rendered)

            if args.view:
                cv2.imshow("YOLO camera application", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
            if frame_index % 100 == 0:
                print(f"video: {source} | processed frames={frame_index}", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.view:
            cv2.destroyAllWindows()

    mean_ms = float(np.mean(inference_times)) if inference_times else 0.0
    for (class_name, direction), count in sorted(line_counter.counts.items()):
        summary_writer.writerow([str(source), class_name, direction, count])
    print(
        f"video: {source} | frames={frame_index} | mean inference={mean_ms:.1f} ms "
        f"| tracks={tracker.next_id - 1} | counts={dict(line_counter.counts)} "
        f"| saved={output_path or 'disabled'}",
        flush=True,
    )


def run_sources(predictor, tracker, line_counter, save_dir, args):
    csv_path = save_dir / "detections.csv"
    counts_path = save_dir / "counts.csv"
    with ExitStack() as stack:
        if args.save_csv:
            csv_file = stack.enter_context(
                csv_path.open("w", newline="", encoding="utf-8")
            )
            counts_file = stack.enter_context(
                counts_path.open("w", newline="", encoding="utf-8")
            )
            csv_writer = csv.writer(csv_file)
            summary_writer = csv.writer(counts_file)
            csv_writer.writerow(DETECTION_COLUMNS)
            summary_writer.writerow(["source", "class_name", "direction", "count"])
        else:
            csv_writer = NullWriter()
            summary_writer = NullWriter()

        for source in iter_sources(args.source):
            tracker.reset()
            line_counter.reset()
            is_image = isinstance(source, Path) and source.suffix.lower() in IMAGE_EXTENSIONS
            if is_image:
                process_image(
                    source,
                    predictor,
                    tracker,
                    line_counter,
                    save_dir,
                    csv_writer,
                    args,
                )
            else:
                process_video(
                    source,
                    predictor,
                    tracker,
                    line_counter,
                    save_dir,
                    csv_writer,
                    summary_writer,
                    args,
                )

    if args.save_csv:
        print(f"detections CSV = {csv_path.resolve()}", flush=True)
        print(f"counts CSV = {counts_path.resolve()}", flush=True)
    else:
        print("CSV output = disabled", flush=True)
