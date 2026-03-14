import argparse
from pathlib import Path
from collections import defaultdict
import yaml
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_label_file(label_path):
    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label format: {line}")

        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        rows.append((cls_id, x, y, w, h))

    return rows


def get_label_dir_from_image_dir(image_dir: Path, data_root: Path) -> Path:
    relative = image_dir.relative_to(data_root)
    relative_parts = list(relative.parts)

    if relative_parts[0] != "images":
        raise ValueError(f"Expected image path under 'images/', got: {image_dir}")

    relative_parts[0] = "labels"
    return data_root.joinpath(*relative_parts)


def check_split(data_root, split_dir, num_classes):
    image_dir = data_root / split_dir
    label_dir = get_label_dir_from_image_dir(image_dir, data_root)

    if not image_dir.exists():
        print(f"[ERROR] Missing image dir: {image_dir}")
        return

    print(f"\n=== Checking split: {split_dir} ===")
    images = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])

    total_images = 0
    total_objects = 0
    missing_labels = []
    bad_images = []
    bad_labels = []
    empty_labels = []
    class_counts = defaultdict(int)

    for img_path in images:
        total_images += 1

        img = cv2.imread(str(img_path))
        if img is None:
            bad_images.append(str(img_path))
            continue

        label_path = label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            missing_labels.append(str(label_path))
            continue

        try:
            rows = read_label_file(label_path)

            if len(rows) == 0:
                empty_labels.append(str(label_path))
                continue

            for cls_id, x, y, w, h in rows:
                if not (0 <= cls_id < num_classes):
                    raise ValueError(f"class_id out of range: {cls_id}")

                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    raise ValueError(f"box value out of range: {(x, y, w, h)}")

                if w <= 0 or h <= 0:
                    raise ValueError(f"invalid width/height: {(w, h)}")

                class_counts[cls_id] += 1
                total_objects += 1

        except Exception as e:
            bad_labels.append((str(label_path), str(e)))

    print(f"Images: {total_images}")
    print(f"Objects: {total_objects}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Bad images: {len(bad_images)}")
    print(f"Bad labels: {len(bad_labels)}")
    print(f"Empty labels: {len(empty_labels)}")

    if class_counts:
        print("Class distribution:")
        for cls_id in sorted(class_counts.keys()):
            print(f"  class {cls_id}: {class_counts[cls_id]}")

    if missing_labels[:10]:
        print("\nSome missing labels:")
        for x in missing_labels[:10]:
            print(" ", x)

    if bad_images[:10]:
        print("\nSome bad images:")
        for x in bad_images[:10]:
            print(" ", x)

    if bad_labels[:10]:
        print("\nSome bad labels:")
        for x, err in bad_labels[:10]:
            print(" ", x, "->", err)

    if empty_labels[:10]:
        print("\nSome empty labels:")
        for x in empty_labels[:10]:
            print(" ", x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to dataset yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.data)
    data_root = Path(cfg["path"])
    names = cfg["names"]

    if isinstance(names, dict):
        num_classes = len(names)
    elif isinstance(names, list):
        num_classes = len(names)
    else:
        raise ValueError("Invalid names field in yaml")

    for split_key in ["train", "val", "test"]:
        if split_key not in cfg:
            continue
        split_dir = cfg[split_key]
        check_split(data_root, split_dir, num_classes)


if __name__ == "__main__":
    main()