import argparse
from pathlib import Path
import random
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
            continue

        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        rows.append((cls_id, x, y, w, h))

    return rows


def yolo_to_xyxy(x, y, w, h, img_w, img_h):
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return x1, y1, x2, y2


def get_label_dir_from_image_dir(image_dir: Path, data_root: Path) -> Path:
    relative = image_dir.relative_to(data_root)
    relative_parts = list(relative.parts)

    if relative_parts[0] != "images":
        raise ValueError(f"Expected image path under 'images/', got: {image_dir}")

    relative_parts[0] = "labels"
    return data_root.joinpath(*relative_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--save-dir", type=str, default="runs/vis")
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()

    cfg = load_yaml(args.data)
    data_root = Path(cfg["path"])
    names = cfg["names"]

    if isinstance(names, dict):
        class_names = {int(k): v for k, v in names.items()}
    else:
        class_names = {i: n for i, n in enumerate(names)}

    image_dir = data_root / cfg[args.split]
    label_dir = get_label_dir_from_image_dir(image_dir, data_root)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    random.shuffle(images)
    images = images[:args.num_samples]

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        label_path = label_dir / f"{img_path.stem}.txt"
        rows = read_label_file(label_path) if label_path.exists() else []

        for cls_id, xc, yc, bw, bh in rows:
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = class_names.get(cls_id, str(cls_id))
            cv2.putText(
                img,
                text,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        save_path = save_dir / img_path.name
        cv2.imwrite(str(save_path), img)

    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    main()