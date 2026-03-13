import json
import argparse
from pathlib import Path
from collections import defaultdict
import yaml


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    w = w / img_w
    h = h / img_h
    return x_center, y_center, w, h


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_category_mapping(categories):
    categories = sorted(categories, key=lambda c: c["id"])
    cat_id_to_yolo = {}
    names = {}
    for yolo_id, cat in enumerate(categories):
        cat_id_to_yolo[cat["id"]] = yolo_id
        names[yolo_id] = cat["name"]
    return cat_id_to_yolo, names


def convert_one_split(json_path, image_dir, label_dir, cat_id_to_yolo):
    data = load_json(json_path)

    images = {img["id"]: img for img in data["images"]}
    anns_by_image = defaultdict(list)

    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue

        image_id = ann["image_id"]
        category_id = ann["category_id"]
        bbox = ann["bbox"]

        if image_id not in images:
            continue
        if category_id not in cat_id_to_yolo:
            continue

        img_info = images[image_id]
        img_w = img_info["width"]
        img_h = img_info["height"]

        x_center, y_center, w, h = coco_bbox_to_yolo(bbox, img_w, img_h)

        # clip basic safety
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)

        if w <= 0.0 or h <= 0.0:
            continue

        yolo_cls = cat_id_to_yolo[category_id]
        anns_by_image[image_id].append((yolo_cls, x_center, y_center, w, h))

    label_dir.mkdir(parents=True, exist_ok=True)

    num_images = 0
    num_labels = 0

    for image_id, img_info in images.items():
        file_name = img_info["file_name"]
        stem = Path(file_name).stem
        label_path = label_dir / f"{stem}.txt"

        rows = anns_by_image.get(image_id, [])

        # create empty txt only if you want; here we skip empty files
        if not rows:
            continue

        with open(label_path, "w", encoding="utf-8") as f:
            for cls_id, xc, yc, bw, bh in rows:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        num_images += 1
        num_labels += len(rows)

    return num_images, num_labels


def save_dataset_yaml(out_yaml, root_dir, names):
    cfg = {
        "path": str(root_dir).replace("\\", "/"),
        "train": "images/train2017",
        "val": "images/val2017",
        "names": names
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="dataset root, e.g. datasets/coco2017")
    args = parser.parse_args()

    root = Path(args.root)
    ann_train = root / "annotations" / "instances_train2017.json"
    ann_val = root / "annotations" / "instances_val2017.json"

    img_train = root / "images" / "train2017"
    img_val = root / "images" / "val2017"

    lbl_train = root / "labels" / "train2017"
    lbl_val = root / "labels" / "val2017"

    train_data = load_json(ann_train)
    val_data = load_json(ann_val)

    cat_id_to_yolo, names = build_category_mapping(train_data["categories"])

    # sanity check categories between train/val
    val_cat_id_to_yolo, val_names = build_category_mapping(val_data["categories"])
    if names != val_names:
        raise ValueError("Train and val category definitions do not match.")

    n_img_train, n_lab_train = convert_one_split(ann_train, img_train, lbl_train, cat_id_to_yolo)
    n_img_val, n_lab_val = convert_one_split(ann_val, img_val, lbl_val, cat_id_to_yolo)

    save_dataset_yaml(root / "dataset.yaml", root, names)

    print("Done converting COCO -> YOLO")
    print(f"Train labeled images: {n_img_train}, objects: {n_lab_train}")
    print(f"Val labeled images:   {n_img_val}, objects: {n_lab_val}")
    print(f"Saved yaml: {root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()