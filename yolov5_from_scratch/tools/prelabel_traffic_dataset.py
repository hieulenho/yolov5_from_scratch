import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import to_tensor


TRAFFIC_CLASS_MAP = {
    1: (0, "person"),
    3: (1, "car"),
    4: (2, "motorcycle"),
    6: (3, "bus"),
    8: (4, "truck"),
}
COMPETING_VEHICLE_CLASSES = {1, 3, 4}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a five-class YOLO traffic dataset with TorchVision pre-labels"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/traffic_pilot"),
        help="Pilot dataset containing images/unlabeled and frames.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/traffic5"),
    )
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument(
        "--vehicle-nms-iou",
        type=float,
        default=0.70,
        help="Suppress overlapping car/bus/truck boxes with conflicting classes",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.20,
        help="Contiguous middle fraction held out from every source video",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--link-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use hard links where possible, then fall back to copying",
    )
    return parser.parse_args()


def read_manifest(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def assign_splits(rows, val_fraction):
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("--val-fraction must be greater than 0 and less than 0.5")

    grouped = {}
    for row in rows:
        grouped.setdefault(row["source"], []).append(row)

    result = {}
    for source, source_rows in grouped.items():
        source_rows.sort(key=lambda item: int(item["frame_index"]))
        val_count = max(1, round(len(source_rows) * val_fraction))
        val_start = (len(source_rows) - val_count) // 2
        val_end = val_start + val_count
        for index, row in enumerate(source_rows):
            result[row["image"]] = "val" if val_start <= index < val_end else "train"
    return result


def prepare_output(output_dir):
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new directory or remove the generated dataset first."
        )
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    return output_dir


def place_image(source, destination, link_images):
    if link_images:
        try:
            destination.hardlink_to(source)
            return "linked"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copied"


def xyxy_to_yolo(box, width, height):
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0.0 or box_height <= 0.0:
        return None
    return (
        (x1 + x2) / (2.0 * width),
        (y1 + y2) / (2.0 * height),
        box_width / width,
        box_height / height,
    )


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    return intersection / max(area1 + area2 - intersection, 1e-9)


def suppress_competing_vehicle_labels(candidates, iou_threshold):
    if iou_threshold <= 0.0:
        return candidates

    kept = []
    for candidate in sorted(
        candidates,
        key=lambda item: item["confidence"],
        reverse=True,
    ):
        class_id = candidate["target_class_id"]
        is_competing_vehicle = class_id in COMPETING_VEHICLE_CLASSES
        conflicts = any(
            is_competing_vehicle
            and previous["target_class_id"] in COMPETING_VEHICLE_CLASSES
            and previous["target_class_id"] != class_id
            and box_iou(candidate["box"], previous["box"]) >= iou_threshold
            for previous in kept
        )
        if not conflicts:
            kept.append(candidate)
    return kept


def load_model(device):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.to(device).eval()
    return model


def main():
    args = parse_args()
    source_dir = args.source.resolve()
    images_dir = source_dir / "images" / "unlabeled"
    manifest_path = source_dir / "frames.csv"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if not 0.0 <= args.vehicle_nms_iou <= 1.0:
        raise ValueError("--vehicle-nms-iou must be between 0 and 1")

    rows = read_manifest(manifest_path)
    splits = assign_splits(rows, args.val_fraction)
    output_dir = prepare_output(args.output)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"device={device}", flush=True)
    print("loading TorchVision Faster R-CNN weights...", flush=True)
    model = load_model(device)

    audit_path = output_dir / "prelabels.csv"
    summary = Counter()
    with audit_path.open("w", encoding="utf-8", newline="") as audit_handle:
        writer = csv.writer(audit_handle)
        writer.writerow(
            [
                "image",
                "split",
                "source",
                "frame_index",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )

        for index, row in enumerate(rows, start=1):
            image_name = row["image"]
            image_path = images_dir / image_name
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            height, width = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = to_tensor(rgb).to(device)
            with torch.inference_mode():
                prediction = model([tensor])[0]

            split = splits[image_name]
            output_image = output_dir / "images" / split / image_name
            place_image(image_path, output_image, args.link_images)
            label_path = output_dir / "labels" / split / Path(image_name).with_suffix(
                ".txt"
            )

            candidates = []
            for box, score, coco_class_id in zip(
                prediction["boxes"].detach().cpu(),
                prediction["scores"].detach().cpu(),
                prediction["labels"].detach().cpu(),
            ):
                confidence = float(score)
                if confidence < args.conf:
                    continue
                class_mapping = TRAFFIC_CLASS_MAP.get(int(coco_class_id))
                if class_mapping is None:
                    continue
                target_class_id, class_name = class_mapping
                yolo_box = xyxy_to_yolo(box.tolist(), width, height)
                if yolo_box is None:
                    continue

                candidates.append(
                    {
                        "target_class_id": target_class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "box": tuple(float(value) for value in box.tolist()),
                        "yolo_box": yolo_box,
                    }
                )

            candidates = suppress_competing_vehicle_labels(
                candidates,
                args.vehicle_nms_iou,
            )
            label_rows = []
            for candidate in candidates:
                target_class_id = candidate["target_class_id"]
                class_name = candidate["class_name"]
                confidence = candidate["confidence"]
                yolo_box = candidate["yolo_box"]
                label_rows.append(
                    f"{target_class_id} "
                    + " ".join(f"{value:.6f}" for value in yolo_box)
                )
                x1, y1, x2, y2 = candidate["box"]
                writer.writerow(
                    [
                        image_name,
                        split,
                        row["source"],
                        row["frame_index"],
                        target_class_id,
                        class_name,
                        f"{confidence:.6f}",
                        f"{x1:.2f}",
                        f"{y1:.2f}",
                        f"{x2:.2f}",
                        f"{y2:.2f}",
                    ]
                )
                summary[f"class:{class_name}"] += 1

            label_path.write_text(
                "\n".join(label_rows) + ("\n" if label_rows else ""),
                encoding="ascii",
            )
            summary[f"images:{split}"] += 1
            summary[f"labels:{split}"] += len(label_rows)
            if not label_rows:
                summary[f"empty:{split}"] += 1

            print(
                f"[{index:03d}/{len(rows):03d}] {split:5s} "
                f"{image_name}: {len(label_rows)} labels",
                flush=True,
            )

    classes_path = output_dir / "classes.txt"
    classes_path.write_text(
        "person\ncar\nmotorcycle\nbus\ntruck\n",
        encoding="ascii",
    )

    print("\nPre-label summary", flush=True)
    for key in sorted(summary):
        print(f"{key}={summary[key]}", flush=True)
    print(f"dataset={output_dir}", flush=True)
    print(f"audit={audit_path}", flush=True)


if __name__ == "__main__":
    main()
