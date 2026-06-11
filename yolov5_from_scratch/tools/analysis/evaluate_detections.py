import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

from yolov5_from_scratch.data.dataset import img2label_path, resolve_data_root


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate predict.py detections.csv against YOLO labels"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.25, 0.30, 0.40, 0.50],
    )
    return parser.parse_args()


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    return intersection / max(area1 + area2 - intersection, 1e-9)


def load_names(config):
    names = config["names"]
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=lambda value: int(value))]
    return list(names)


def read_yolo_labels(path, width, height):
    labels = defaultdict(list)
    if not path.is_file():
        return labels
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        class_id, center_x, center_y, box_width, box_height = map(
            float,
            line.split(),
        )
        center_x *= width
        center_y *= height
        box_width *= width
        box_height *= height
        labels[int(class_id)].append(
            (
                center_x - box_width / 2.0,
                center_y - box_height / 2.0,
                center_x + box_width / 2.0,
                center_y + box_height / 2.0,
            )
        )
    return labels


def load_ground_truth(data_yaml, split):
    with data_yaml.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_root = resolve_data_root(data_yaml, config["path"])
    image_dir = data_root / config[split]
    ground_truth = {}
    for image_path in sorted(image_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        ground_truth[image_path.name] = read_yolo_labels(
            img2label_path(image_path, data_root),
            width,
            height,
        )
    return config, ground_truth


def load_predictions(csv_path):
    predictions = defaultdict(lambda: defaultdict(list))
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_name = Path(row["source"]).name
            predictions[image_name][int(row["class_id"])].append(
                {
                    "confidence": float(row["confidence"]),
                    "box": tuple(
                        float(row[key]) for key in ("x1", "y1", "x2", "y2")
                    ),
                }
            )
    return predictions


def evaluate(ground_truth, predictions, confidence, iou_threshold, num_classes):
    stats = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in range(num_classes)}
    for image_name, image_ground_truth in ground_truth.items():
        image_predictions = predictions.get(image_name, {})
        for class_id in range(num_classes):
            truth_boxes = image_ground_truth.get(class_id, [])
            predicted = sorted(
                (
                    item
                    for item in image_predictions.get(class_id, [])
                    if item["confidence"] >= confidence
                ),
                key=lambda item: item["confidence"],
                reverse=True,
            )
            matched = set()
            for prediction in predicted:
                best_index = -1
                best_iou = 0.0
                for index, truth_box in enumerate(truth_boxes):
                    if index in matched:
                        continue
                    iou = box_iou(prediction["box"], truth_box)
                    if iou > best_iou:
                        best_index = index
                        best_iou = iou
                if best_index >= 0 and best_iou >= iou_threshold:
                    matched.add(best_index)
                    stats[class_id]["tp"] += 1
                else:
                    stats[class_id]["fp"] += 1
            stats[class_id]["fn"] += len(truth_boxes) - len(matched)
    return stats


def metrics(stats):
    true_positive = sum(item["tp"] for item in stats.values())
    false_positive = sum(item["fp"] for item in stats.values())
    false_negative = sum(item["fn"] for item in stats.values())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return true_positive, false_positive, false_negative, precision, recall, f1


def main():
    args = parse_args()
    csv_path = args.run_dir.resolve() / "detections.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Detection CSV not found: {csv_path}")
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("--iou must be greater than 0 and at most 1")

    config, ground_truth = load_ground_truth(args.data.resolve(), args.split)
    names = load_names(config)
    predictions = load_predictions(csv_path)

    results = []
    print("conf    precision  recall     f1       tp     fp     fn")
    for threshold in args.thresholds:
        stats = evaluate(
            ground_truth,
            predictions,
            threshold,
            args.iou,
            len(names),
        )
        row = (threshold, stats, metrics(stats))
        results.append(row)
        tp, fp, fn, precision, recall, f1 = row[2]
        print(
            f"{threshold:0.2f}    {precision:0.4f}     {recall:0.4f}   "
            f"{f1:0.4f}   {tp:5d}  {fp:5d}  {fn:5d}"
        )

    best_threshold, best_stats, best_metrics = max(
        results,
        key=lambda item: item[2][-1],
    )
    print(f"\nbest confidence={best_threshold:.2f} at IoU={args.iou:.2f}")
    print("class         precision  recall     f1       tp     fp     fn")
    for class_id, class_name in enumerate(names):
        class_metrics = metrics({class_id: best_stats[class_id]})
        tp, fp, fn, precision, recall, f1 = class_metrics
        print(
            f"{class_name:12s}  {precision:0.4f}     {recall:0.4f}   "
            f"{f1:0.4f}   {tp:5d}  {fp:5d}  {fn:5d}"
        )


if __name__ == "__main__":
    main()
