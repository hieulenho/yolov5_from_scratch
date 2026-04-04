import sys
from pathlib import Path
import argparse
import numpy as np
import torch

FILE = Path(__file__).resolve()
# assumes this script may be copied into yolov5_from_scratch/data or project root
# user should adjust ROOT if needed after copying
ROOT = Path.cwd() / 'yolov5_from_scratch'
if (ROOT / 'data').exists() and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloader, YOLODataset


def summarize_dataset_samples(ds: YOLODataset, limit: int = 20):
    print('\n=== cache/sample summary ===', flush=True)
    n = len(ds.samples)
    label_counts = [int(s['labels'].shape[0]) for s in ds.samples]
    non_empty = sum(c > 0 for c in label_counts)
    print(f'total samples = {n}', flush=True)
    print(f'non-empty samples in cache = {non_empty}', flush=True)
    print(f'total cached labels = {sum(label_counts)}', flush=True)

    shown = 0
    for i, s in enumerate(ds.samples):
        c = int(s['labels'].shape[0])
        if c > 0:
            print(f'cache sample idx={i} file={Path(s["im_file"]).name} n={c}', flush=True)
            print(s['labels'][:5], flush=True)
            shown += 1
            if shown >= limit:
                break
    if shown == 0:
        print('No non-empty cached samples found.', flush=True)


def summarize_getitem(ds: YOLODataset, max_scan: int = 200):
    print('\n=== __getitem__ summary ===', flush=True)
    shown = 0
    for i in range(min(max_scan, len(ds))):
        img, targets, meta = ds[i]
        n = int(targets.shape[0])
        if n > 0:
            print(f'getitem idx={i} file={Path(meta["im_file"]).name} n={n}', flush=True)
            print(targets[:5], flush=True)
            shown += 1
            if shown >= 10:
                break
    if shown == 0:
        print('No non-empty __getitem__ samples found in scan window.', flush=True)


def summarize_loader(loader, max_batches: int = 20):
    print('\n=== dataloader/collate summary ===', flush=True)
    shown = 0
    for bi, (imgs, targets, metas) in enumerate(loader):
        print(f'batch={bi} imgs={tuple(imgs.shape)} n_targets={targets.shape[0]}', flush=True)
        if targets.numel() > 0:
            print(targets[:10], flush=True)
            shown += 1
            if shown >= 5:
                break
        if bi + 1 >= max_batches:
            break
    if shown == 0:
        print('No non-empty batches found in scan window.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-yaml', type=str, default=str(Path.cwd() / 'yolov5_from_scratch' / 'datasets' / 'coco2017' / 'dataset.yaml'))
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--rebuild-cache', action='store_true')
    parser.add_argument('--min-box-size', type=float, default=2.0)
    args = parser.parse_args()

    dataset, loader = build_dataloader(
        data_yaml=args.data_yaml,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=0,
        cache_labels=True,
        cache_images=False,
        augment=False,
        shuffle=False,
        persistent_workers=False,
        verbose=True,
        rebuild_cache=args.rebuild_cache,
        min_box_size=args.min_box_size,
    )

    summarize_dataset_samples(dataset)
    summarize_getitem(dataset)
    summarize_loader(loader)


if __name__ == '__main__':
    main()
