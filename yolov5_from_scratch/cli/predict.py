import argparse
from pathlib import Path

import yaml

from yolov5_from_scratch.paths import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_WEIGHTS,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
)
from yolov5_from_scratch.runtime.counting import LineCounter
from yolov5_from_scratch.runtime.detector import YOLOv5Predictor, load_class_names
from yolov5_from_scratch.runtime.geometry import validate_roi
from yolov5_from_scratch.runtime.media import increment_path
from yolov5_from_scratch.runtime.pipeline import run_sources
from yolov5_from_scratch.runtime.tracking import IoUTracker


def build_parser(default_source=None):
    parser = argparse.ArgumentParser(
        description="Detect, track, and count any configured object classes"
    )
    parser.add_argument(
        "--config",
        help="optional YAML application profile",
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument(
        "--source",
        default=default_source,
        help="image, video, directory, camera index, or RTSP/HTTP stream",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="class names or IDs; omit to detect every class in the data config",
    )
    parser.add_argument(
        "--track",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--track-iou", type=float, default=0.30)
    parser.add_argument("--track-center-distance", type=float, default=1.0)
    parser.add_argument("--track-conf", type=float, default=0.25)
    parser.add_argument("--display-conf", type=float, default=0.25)
    parser.add_argument("--track-max-age", type=int, default=30)
    parser.add_argument("--min-track-hits", type=int, default=5)
    parser.add_argument("--count-conf", type=float, default=0.30)
    parser.add_argument(
        "--count-line",
        choices=["none", "horizontal", "vertical"],
        default="none",
    )
    parser.add_argument("--line-position", type=float, default=0.5)
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
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
    )
    parser.add_argument(
        "--draw-roi",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--project", default=str(PREDICTIONS_DIR))
    parser.add_argument("--name", default="predict")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--save-media",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save rendered images/video",
    )
    parser.add_argument(
        "--save-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save detections.csv and counts.csv",
    )
    parser.add_argument(
        "--view",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--hide-conf", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser


def resolve_class_filter(tokens, names):
    if tokens is None:
        return None

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


def validate_args(args):
    if args.source is None:
        raise ValueError("--source is required unless it is set by --config")
    args.roi = validate_roi(args.roi)
    for name in ("conf", "track_conf", "display_conf", "count_conf"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.output_width < 0:
        raise ValueError("--output-width must be 0 or greater")
    if args.count_line != "none" and not args.track:
        raise ValueError("Line counting requires tracking; use --track")
    return args


def load_profile(path):
    profile_path = Path(path).resolve()
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = yaml.safe_load(handle) or {}
    if not isinstance(profile, dict):
        raise ValueError(f"Application profile must be a YAML mapping: {profile_path}")

    valid_keys = {
        action.dest
        for action in build_parser()._actions
        if action.dest not in {"help", "config"}
    }
    unknown = sorted(set(profile) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown profile option(s): {', '.join(unknown)}")

    for key in ("weights", "data", "project"):
        value = profile.get(key)
        if value and not Path(str(value)).is_absolute():
            profile[key] = str((PROJECT_ROOT / str(value)).resolve())
    return profile


def run(args):
    args = validate_args(args)
    if not args.save_media and not args.save_csv and not args.view:
        raise ValueError(
            "Nothing is enabled: use --view, --save-media, or --save-csv"
        )
    names = load_class_names(args.data)
    class_filter = resolve_class_filter(args.classes, names)
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
    selected_names = (
        names if class_filter is None else [names[index] for index in sorted(class_filter)]
    )
    print(f"classes = {', '.join(selected_names)}", flush=True)
    print(f"save_dir = {save_dir.resolve()}", flush=True)
    run_sources(predictor, tracker, line_counter, save_dir, args)
    return save_dir


def main(argv=None, default_source=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args(argv)
    parser = build_parser(default_source=default_source)
    if known.config:
        parser.set_defaults(**load_profile(known.config))
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
