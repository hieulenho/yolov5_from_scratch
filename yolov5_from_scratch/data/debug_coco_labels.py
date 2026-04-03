import sys
from pathlib import Path
import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import load_yaml, resolve_data_root, img2label_path, parse_yolo_label_file, IMG_EXTS


def scan_images(image_dir: Path):
    if not image_dir.exists():
        return []
    return sorted([p for p in image_dir.rglob('*') if p.suffix.lower() in IMG_EXTS])


def main():
    data_yaml = ROOT / 'datasets' / 'coco2017' / 'dataset.yaml'
    cfg = load_yaml(data_yaml)
    data_root = resolve_data_root(data_yaml, cfg.get('path', '.'))

    print(f'data_yaml = {data_yaml}')
    print(f'data_root = {data_root} | exists = {data_root.exists()}')

    for split in ['train', 'val']:
        if split not in cfg:
            continue
        image_dir = data_root / cfg[split]
        images = scan_images(image_dir)
        print(f'{split:5} images = {image_dir} | exists = {image_dir.exists()} | count = {len(images)}')

        label_dir = data_root / cfg[split].replace('images/', 'labels/')
        label_files = sorted(label_dir.rglob('*.txt')) if label_dir.exists() else []
        print(f'{split:5} labels = {label_dir} | exists = {label_dir.exists()} | count = {len(label_files)}')

        if images:
            print(f'  first image = {images[0].name}')
            sample = random.sample(images, k=min(5, len(images)))
            print('  sample label check:')
            for im_file in sample:
                lb = img2label_path(im_file, data_root)
                rows = parse_yolo_label_file(lb)
                size = lb.stat().st_size if lb.exists() else -1
                print(f'    {im_file.name}: label_exists={lb.exists()} size={size} n={len(rows)}')

        non_empty_found = False
        if label_dir.exists():
            for lb in label_files[:2000]:
                rows = parse_yolo_label_file(lb)
                if len(rows) > 0:
                    print(f'  first non-empty label = {lb}')
                    print(f'  rows[:5] = {rows[:5]}')
                    non_empty_found = True
                    break
        if not non_empty_found:
            print('  no non-empty label found in first scan window')


if __name__ == '__main__':
    main()
