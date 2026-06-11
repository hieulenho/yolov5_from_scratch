import argparse
import csv
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference import Detection, YOLOv5Predictor, load_class_names


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v"}
TRAFFIC_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect, track, and count traffic objects in images or video"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(ROOT / "runs" / "train" / "exp" / "weights" / "best.pt"),
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "datasets" / "coco2017" / "dataset.yaml"),
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="Low default because the current checkpoint has low confidence scores",
    )
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Detect all dataset classes instead of traffic classes only",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class names or IDs. Overrides the traffic class filter.",
    )
    parser.add_argument(
        "--track",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--track-iou", type=float, default=0.30)
    parser.add_argument(
        "--track-center-distance",
        type=float,
        default=1.0,
        help="Fallback center distance normalized by box diagonal; 0 disables it",
    )
    parser.add_argument(
        "--track-conf",
        type=float,
        default=0.10,
        help="Only detections at or above this confidence receive track IDs",
    )
    parser.add_argument(
        "--display-conf",
        type=float,
        default=0.10,
        help="Only detections at or above this confidence are drawn",
    )
    parser.add_argument("--track-max-age", type=int, default=30)
    parser.add_argument("--min-track-hits", type=int, default=5)
    parser.add_argument(
        "--count-conf",
        type=float,
        default=0.15,
        help="Minimum average confidence of a track before it can be counted",
    )
    parser.add_argument(
        "--count-line",
        choices=["none", "horizontal", "vertical"],
        default="horizontal",
    )
    parser.add_argument(
        "--line-position",
        type=float,
        default=0.5,
        help="Normalized line position in the range [0, 1]",
    )
    parser.add_argument("--line-margin", type=float, default=2.0)
    parser.add_argument(
        "--count-directions",
        nargs="*",
        choices=[
            "top_to_bottom",
            "bottom_to_top",
            "left_to_right",
            "right_to_left",
        ],
        default=None,
        help="Optional crossing directions to count",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Normalized inference crop, e.g. --roi 0.1 0.2 0.9 1.0",
    )
    parser.add_argument(
        "--draw-roi",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=1280,
        help="Downscale saved media to this width; 0 keeps original size",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec for saved video",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(ROOT / "runs" / "detect"),
    )
    parser.add_argument("--name", type=str, default="traffic")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--hide-conf", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def increment_path(path, exist_ok=False):
    path = Path(path)
    if exist_ok or not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.name}{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate output directory near {path}")


def resolve_class_filter(tokens, names, all_classes=False):
    if tokens is None:
        if all_classes:
            return None
        return {i for i, name in enumerate(names) if name in TRAFFIC_CLASSES}

    name_to_id = {name.lower(): i for i, name in enumerate(names)}
    class_ids = set()
    for token in tokens:
        token_text = str(token).strip()
        if token_text.lstrip("-").isdigit():
            class_id = int(token_text)
            if not 0 <= class_id < len(names):
                raise ValueError(f"Class ID is out of range: {class_id}")
            class_ids.add(class_id)
            continue

        class_id = name_to_id.get(token_text.lower())
        if class_id is None:
            raise ValueError(f"Unknown class name: {token_text}")
        class_ids.add(class_id)
    return class_ids


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    return intersection / max(area1 + area2 - intersection, 1e-9)


def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def normalized_center_distance(box1, box2):
    center1 = box_center(box1)
    center2 = box_center(box2)
    distance = float(np.hypot(center1[0] - center2[0], center1[1] - center2[1]))
    diagonal1 = float(np.hypot(box1[2] - box1[0], box1[3] - box1[1]))
    diagonal2 = float(np.hypot(box2[2] - box2[0], box2[3] - box2[1]))
    return distance / max(diagonal1, diagonal2, 1e-9)


def validate_roi(roi):
    if roi is None:
        return None
    x1, y1, x2, y2 = (float(value) for value in roi)
    if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        raise ValueError("--roi values must be between 0 and 1")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("--roi requires X2 > X1 and Y2 > Y1")
    return x1, y1, x2, y2


def roi_pixel_bounds(frame_shape, roi):
    height, width = frame_shape[:2]
    if roi is None:
        return 0, 0, width, height
    x1, y1, x2, y2 = roi
    left = max(0, min(width - 1, int(round(x1 * width))))
    top = max(0, min(height - 1, int(round(y1 * height))))
    right = max(left + 1, min(width, int(round(x2 * width))))
    bottom = max(top + 1, min(height, int(round(y2 * height))))
    return left, top, right, bottom


def offset_detections(detections, offset_x, offset_y):
    return [
        Detection(
            xyxy=(
                detection.xyxy[0] + offset_x,
                detection.xyxy[1] + offset_y,
                detection.xyxy[2] + offset_x,
                detection.xyxy[3] + offset_y,
            ),
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
        )
        for detection in detections
    ]


def predict_frame(predictor, frame, roi):
    left, top, right, bottom = roi_pixel_bounds(frame.shape, roi)
    crop = frame[top:bottom, left:right]
    detections = predictor.predict(crop)
    return offset_detections(detections, left, top)


def output_dimensions(width, height, output_width):
    if output_width <= 0 or width <= output_width:
        return width, height
    output_height = max(2, int(round(height * output_width / width)))
    if output_height % 2:
        output_height += 1
    return int(output_width), output_height


def resize_for_output(image, output_width):
    target_width, target_height = output_dimensions(
        image.shape[1],
        image.shape[0],
        output_width,
    )
    if (target_width, target_height) == (image.shape[1], image.shape[0]):
        return image.copy(), 1.0, 1.0
    resized = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized, target_width / image.shape[1], target_height / image.shape[0]


def scale_detections(detections, scale_x, scale_y):
    return [
        Detection(
            xyxy=(
                detection.xyxy[0] * scale_x,
                detection.xyxy[1] * scale_y,
                detection.xyxy[2] * scale_x,
                detection.xyxy[3] * scale_y,
            ),
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
        )
        for detection in detections
    ]


@dataclass
class Track:
    track_id: int
    class_id: int
    bbox: tuple
    center: tuple
    previous_center: tuple = None
    hits: int = 1
    missed: int = 0
    confidence_sum: float = 0.0
    max_confidence: float = 0.0

    @property
    def average_confidence(self):
        return self.confidence_sum / max(self.hits, 1)


class IoUTracker:
    def __init__(
        self,
        iou_threshold=0.3,
        max_age=30,
        center_distance_threshold=1.0,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.center_distance_threshold = float(center_distance_threshold)
        self.next_id = 1
        self.tracks = {}

    def reset(self):
        self.next_id = 1
        self.tracks.clear()

    def update(self, detections):
        for track in self.tracks.values():
            track.missed += 1

        candidates = []
        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                if track.class_id != detection.class_id:
                    continue
                iou = box_iou(track.bbox, detection.xyxy)
                if iou >= self.iou_threshold:
                    score = 2.0 + iou
                    candidates.append((score, track_id, detection_index))
                    continue

                distance = normalized_center_distance(
                    track.bbox,
                    detection.xyxy,
                )
                if (
                    self.center_distance_threshold > 0
                    and distance <= self.center_distance_threshold
                ):
                    score = 1.0 - distance / self.center_distance_threshold
                    candidates.append((score, track_id, detection_index))

        matched_tracks = set()
        matched_detections = set()
        assignments = [None] * len(detections)
        for _, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            track = self.tracks[track_id]
            track.previous_center = track.center
            track.bbox = detections[detection_index].xyxy
            track.center = box_center(track.bbox)
            track.hits += 1
            track.missed = 0
            track.confidence_sum += detections[detection_index].confidence
            track.max_confidence = max(
                track.max_confidence,
                detections[detection_index].confidence,
            )
            assignments[detection_index] = track_id

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = Track(
                track_id=track_id,
                class_id=detection.class_id,
                bbox=detection.xyxy,
                center=box_center(detection.xyxy),
                confidence_sum=detection.confidence,
                max_confidence=detection.confidence,
            )
            assignments[detection_index] = track_id

        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed > self.max_age
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

        return assignments


class LineCounter:
    def __init__(
        self,
        orientation="horizontal",
        position=0.5,
        margin=2.0,
        min_track_hits=2,
        min_confidence=0.15,
        allowed_directions=None,
    ):
        if not 0.0 <= position <= 1.0:
            raise ValueError("--line-position must be between 0 and 1")
        self.orientation = orientation
        self.position = float(position)
        self.margin = float(margin)
        self.min_track_hits = int(min_track_hits)
        self.min_confidence = float(min_confidence)
        self.allowed_directions = (
            None
            if allowed_directions is None
            else set(allowed_directions)
        )
        self.counted_tracks = set()
        self.track_sides = {}
        self.counts = Counter()

    def reset(self):
        self.counted_tracks.clear()
        self.track_sides.clear()
        self.counts.clear()

    def update(self, track, frame_shape, class_name):
        if self.orientation == "none":
            return None

        frame_h, frame_w = frame_shape[:2]
        if self.orientation == "horizontal":
            line = frame_h * self.position
            previous = (
                track.previous_center[1]
                if track.previous_center is not None
                else None
            )
            current = track.center[1]
            positive_direction = "top_to_bottom"
            negative_direction = "bottom_to_top"
        else:
            line = frame_w * self.position
            previous = (
                track.previous_center[0]
                if track.previous_center is not None
                else None
            )
            current = track.center[0]
            positive_direction = "left_to_right"
            negative_direction = "right_to_left"

        def side(value):
            if value < line - self.margin:
                return -1
            if value > line + self.margin:
                return 1
            return 0

        current_side = side(current)
        previous_side = self.track_sides.get(track.track_id)
        if previous_side is None and track.previous_center is not None:
            previous_side = side(previous)

        if current_side != 0:
            self.track_sides[track.track_id] = current_side

        if (
            track.previous_center is None
            or track.hits < self.min_track_hits
            or track.average_confidence < self.min_confidence
            or track.track_id in self.counted_tracks
            or previous_side not in (-1, 1)
            or current_side not in (-1, 1)
            or previous_side == current_side
        ):
            return None

        direction = (
            positive_direction
            if previous_side < current_side
            else negative_direction
        )
        if (
            self.allowed_directions is not None
            and direction not in self.allowed_directions
        ):
            return None

        self.counted_tracks.add(track.track_id)
        self.counts[(class_name, direction)] += 1
        return direction


def color_for_class(class_id):
    palette = (
        (255, 56, 56),
        (255, 157, 151),
        (255, 112, 31),
        (255, 178, 29),
        (207, 210, 49),
        (72, 249, 10),
        (146, 204, 23),
        (61, 219, 134),
        (26, 147, 52),
        (0, 212, 187),
        (44, 153, 168),
        (0, 194, 255),
    )
    rgb = palette[class_id % len(palette)]
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def draw_line_and_counts(image, line_counter):
    if line_counter.orientation == "none":
        return

    height, width = image.shape[:2]
    if line_counter.orientation == "horizontal":
        y = int(height * line_counter.position)
        cv2.line(image, (0, y), (width - 1, y), (0, 255, 255), 2)
    else:
        x = int(width * line_counter.position)
        cv2.line(image, (x, 0), (x, height - 1), (0, 255, 255), 2)

    summary = Counter()
    for (class_name, _), value in line_counter.counts.items():
        summary[class_name] += value

    y_text = 28
    for class_name, value in sorted(summary.items()):
        text = f"{class_name}: {value}"
        cv2.putText(
            image,
            text,
            (10, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_text += 26


def draw_roi(image, roi):
    if roi is None:
        return
    left, top, right, bottom = roi_pixel_bounds(image.shape, roi)
    cv2.rectangle(
        image,
        (left, top),
        (right - 1, bottom - 1),
        (255, 255, 0),
        2,
    )


def draw_performance(image, inference_ms):
    fps = 1000.0 / max(inference_ms, 1e-6)
    text = f"{inference_ms:.1f} ms | {fps:.1f} FPS"
    text_size, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2,
    )
    x = max(10, image.shape[1] - text_size[0] - 12)
    cv2.putText(
        image,
        text,
        (x, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_detections(
    image,
    detections,
    track_ids,
    hide_labels=False,
    hide_conf=False,
):
    for detection, track_id in zip(detections, track_ids):
        x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy)
        color = color_for_class(detection.class_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        if hide_labels:
            continue
        label = detection.class_name
        if track_id is not None:
            label += f" #{track_id}"
        if not hide_conf:
            label += f" {detection.confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def update_tracks(tracker, detections, min_confidence):
    selected_indices = [
        index
        for index, detection in enumerate(detections)
        if detection.confidence >= min_confidence
    ]
    selected_detections = [detections[index] for index in selected_indices]
    selected_assignments = tracker.update(selected_detections)
    assignments = [None] * len(detections)
    for detection_index, track_id in zip(selected_indices, selected_assignments):
        assignments[detection_index] = track_id
    return assignments


def detections_for_display(detections, track_ids, min_confidence):
    selected = [
        (detection, track_id)
        for detection, track_id in zip(detections, track_ids)
        if detection.confidence >= min_confidence
    ]
    return (
        [item[0] for item in selected],
        [item[1] for item in selected],
    )


def iter_sources(source):
    path = Path(source)
    if path.is_dir():
        files = [
            item
            for item in sorted(path.rglob("*"))
            if item.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        ]
        if not files:
            raise FileNotFoundError(f"No supported media found in: {path}")
        return files
    if path.is_file():
        return [path]
    if source.isdigit():
        return [int(source)]
    raise FileNotFoundError(f"Source not found: {source}")


def output_name(source, suffix):
    if isinstance(source, int):
        return f"webcam_{source}{suffix}"
    return f"{Path(source).stem}{suffix}"


def open_capture(source):
    capture_source = source if isinstance(source, int) else str(source)
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return capture


def process_image(
    source,
    predictor,
    tracker,
    line_counter,
    save_dir,
    csv_writer,
    args,
):
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
        csv_writer.writerow(
            [
                str(source),
                0,
                0.0,
                track_id if track_id is not None else "",
                detection.class_id,
                detection.class_name,
                f"{detection.confidence:.6f}",
                *(f"{value:.2f}" for value in detection.xyxy),
                "",
            ]
        )

    rendered, scale_x, scale_y = resize_for_output(image, args.output_width)
    display_detections, display_track_ids = detections_for_display(
        detections,
        track_ids,
        args.display_conf,
    )
    render_detections = scale_detections(
        display_detections,
        scale_x,
        scale_y,
    )
    draw_detections(
        rendered,
        render_detections,
        display_track_ids,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf,
    )
    if args.draw_roi:
        draw_roi(rendered, args.roi)
    draw_performance(rendered, inference_ms)
    output_path = save_dir / output_name(source, Path(source).suffix.lower())
    cv2.imwrite(str(output_path), rendered)
    print(
        f"image: {source} | detections={len(detections)} "
        f"| inference={inference_ms:.1f} ms | saved={output_path}",
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
    output_width, output_height = output_dimensions(
        width,
        height,
        args.output_width,
    )

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
                direction = None
                if track_id is not None:
                    direction = line_counter.update(
                        tracker.tracks[track_id],
                        frame.shape,
                        detection.class_name,
                    )
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

            rendered, scale_x, scale_y = resize_for_output(
                frame,
                args.output_width,
            )
            display_detections, display_track_ids = detections_for_display(
                detections,
                track_ids,
                args.display_conf,
            )
            render_detections = scale_detections(
                display_detections,
                scale_x,
                scale_y,
            )
            draw_detections(
                rendered,
                render_detections,
                display_track_ids,
                hide_labels=args.hide_labels,
                hide_conf=args.hide_conf,
            )
            draw_line_and_counts(rendered, line_counter)
            if args.draw_roi:
                draw_roi(rendered, args.roi)
            draw_performance(rendered, inference_ms)
            writer.write(rendered)

            if args.view:
                cv2.imshow("YOLOv5 traffic detection", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
            if frame_index % 100 == 0:
                print(f"video: {source} | processed frames={frame_index}", flush=True)
    finally:
        capture.release()
        writer.release()
        if args.view:
            cv2.destroyAllWindows()

    mean_ms = float(np.mean(inference_times)) if inference_times else 0.0
    for (class_name, direction), count in sorted(line_counter.counts.items()):
        summary_writer.writerow([str(source), class_name, direction, count])
    print(
        f"video: {source} | frames={frame_index} | mean inference={mean_ms:.1f} ms "
        f"| tracks={tracker.next_id - 1} | counts={dict(line_counter.counts)} "
        f"| saved={output_path}",
        flush=True,
    )


def main():
    args = parse_args()
    args.roi = validate_roi(args.roi)
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if not 0.0 <= args.track_conf <= 1.0:
        raise ValueError("--track-conf must be between 0 and 1")
    if not 0.0 <= args.display_conf <= 1.0:
        raise ValueError("--display-conf must be between 0 and 1")
    if not 0.0 <= args.count_conf <= 1.0:
        raise ValueError("--count-conf must be between 0 and 1")
    if args.output_width < 0:
        raise ValueError("--output-width must be 0 or greater")
    names = load_class_names(args.data)
    class_filter = resolve_class_filter(
        args.classes,
        names,
        all_classes=args.all_classes,
    )
    if args.count_line != "none" and not args.track:
        raise ValueError("Line counting requires tracking; use --track")

    save_dir = increment_path(
        Path(args.project) / args.name,
        exist_ok=args.exist_ok,
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    predictor = YOLOv5Predictor(
        weights=args.weights,
        data_yaml=args.data,
        img_size=args.img_size,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_det=args.max_det,
        device=args.device,
        amp=args.amp,
        class_filter=class_filter,
    )
    tracker = IoUTracker(
        iou_threshold=args.track_iou,
        max_age=args.track_max_age,
        center_distance_threshold=args.track_center_distance,
    )
    line_counter = LineCounter(
        orientation=args.count_line,
        position=args.line_position,
        margin=args.line_margin,
        min_track_hits=args.min_track_hits,
        min_confidence=args.count_conf,
        allowed_directions=args.count_directions,
    )

    print(f"device = {predictor.device}", flush=True)
    print(f"weights = {Path(args.weights).resolve()}", flush=True)
    print(f"checkpoint epoch = {predictor.checkpoint_epoch}", flush=True)
    print(f"roi = {args.roi or 'full frame'}", flush=True)
    print(
        f"thresholds = detect:{args.conf:.3f} track:{args.track_conf:.3f} "
        f"display:{args.display_conf:.3f} count_avg:{args.count_conf:.3f}",
        flush=True,
    )
    print(
        "classes = "
        + (
            "all"
            if class_filter is None
            else ", ".join(names[class_id] for class_id in sorted(class_filter))
        ),
        flush=True,
    )
    print(f"save_dir = {save_dir.resolve()}", flush=True)

    csv_path = save_dir / "detections.csv"
    counts_path = save_dir / "counts.csv"
    with (
        open(csv_path, "w", newline="", encoding="utf-8") as csv_file,
        open(counts_path, "w", newline="", encoding="utf-8") as counts_file,
    ):
        csv_writer = csv.writer(csv_file)
        summary_writer = csv.writer(counts_file)
        csv_writer.writerow(
            [
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
        )
        summary_writer.writerow(["source", "class_name", "direction", "count"])

        for source in iter_sources(args.source):
            tracker.reset()
            line_counter.reset()
            suffix = "" if isinstance(source, int) else source.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
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

    print(f"detections CSV = {csv_path.resolve()}", flush=True)
    print(f"counts CSV = {counts_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
