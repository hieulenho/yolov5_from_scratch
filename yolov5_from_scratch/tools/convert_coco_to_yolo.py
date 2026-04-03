import sys
from pathlib import Path
import json
import argparse
from collections import defaultdict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import load_yaml, resolve_data_root


def coco_box_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x1 = max(0.0, min(float(x), float(img_w)))
    y1 = max(0.0, min(float(y), float(img_h)))
    x2 = max(0.0, min(float(x + w), float(img_w)))
    y2 = max(0.0, min(float(y + h), float(img_h)))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0.0 or bh <= 0.0:
        return None
    xc = (x1 + x2) * 0.5 / float(img_w)
    yc = (y1 + y2) * 0.5 / float(img_h)
    bw /= float(img_w)
    bh /= float(img_h)
    return xc, yc, bw, bh


def build_name_to_idx(cfg_names):
    if isinstance(cfg_names, dict):
        pairs = sorted((int(k), v) for k, v in cfg_names.items())
        names = [v for _, v in pairs]
    else:
        names = list(cfg_names)
    return {name: i for i, name in enumerate(names)}


def convert_split(data_root: Path, name_to_idx, split: str, create_empty: bool = True, overwrite: bool = True):
    ann_path = data_root / 'annotations' / f'instances_{split}.json'
    if not ann_path.exists():
        raise FileNotFoundError(f'Annotation file not found: {ann_path}')

    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
    grouped = defaultdict(list)

    skipped_crowd = 0
    skipped_bad = 0
    skipped_class = 0

    for ann in coco['annotations']:
        if ann.get('iscrowd', 0):
            skipped_crowd += 1
            continue
        img = images.get(ann['image_id'])
        if img is None:
            skipped_bad += 1
            continue
        cat_name = cat_id_to_name.get(ann['category_id'])
        if cat_name not in name_to_idx:
            skipped_class += 1
            continue
        box = coco_box_to_yolo(ann['bbox'], img['width'], img['height'])
        if box is None:
            skipped_bad += 1
            continue
        grouped[ann['image_id']].append((name_to_idx[cat_name], *box))

    image_dir = data_root / 'images' / split
    label_dir = data_root / 'labels' / split
    label_dir.mkdir(parents=True, exist_ok=True)

    written_files = 0
    written_rows = 0
    empty_files = 0
    missing_images = 0

    for img_id, img in images.items():
        img_file = image_dir / img['file_name']
        if not img_file.exists():
            missing_images += 1
            continue
        label_file = label_dir / Path(img['file_name']).with_suffix('.txt')
        rows = grouped.get(img_id, [])

        if rows or create_empty:
            if label_file.exists() and not overwrite:
                continue
            with open(label_file, 'w', encoding='utf-8') as f:
                for cls_id, xc, yc, bw, bh in rows:
                    f.write(f'{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')
            written_files += 1
            written_rows += len(rows)
            if not rows:
                empty_files += 1

    print(f'[{split}] images={len(images)} labels_written={written_files} rows={written_rows}')
    print(f'[{split}] empty_label_files={empty_files} missing_images={missing_images}')
    print(f'[{split}] skipped_crowd={skipped_crowd} skipped_bad={skipped_bad} skipped_class={skipped_class}')
    print(f'[{split}] label_dir={label_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-yaml', type=str, default=str(ROOT / 'datasets' / 'coco2017' / 'dataset.yaml'))
    parser.add_argument('--splits', nargs='+', default=['train2017', 'val2017'])
    parser.add_argument('--no-empty', action='store_true')
    parser.add_argument('--no-overwrite', action='store_true')
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml).resolve()
    cfg = load_yaml(data_yaml)
    data_root = resolve_data_root(data_yaml, cfg.get('path', '.'))
    name_to_idx = build_name_to_idx(cfg['names'])

    print(f'data_yaml = {data_yaml}')
    print(f'data_root = {data_root}')

    for split in args.splits:
        convert_split(
            data_root=data_root,
            name_to_idx=name_to_idx,
            split=split,
            create_empty=not args.no_empty,
            overwrite=not args.no_overwrite,
        )


if __name__ == '__main__':
    main()
