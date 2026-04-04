import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloader
from models.yolo import YOLOv5FromScratch
from loss.loss import YoloLoss


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(self.count, 1)

    def update(self, value, n=1):
        self.sum += float(value) * int(n)
        self.count += int(n)


class LossMeters:
    def __init__(self):
        self.loss = AverageMeter()
        self.lbox = AverageMeter()
        self.lobj = AverageMeter()
        self.lcls = AverageMeter()

    def update(self, loss_items, n):
        self.loss.update(loss_items["loss"], n)
        self.lbox.update(loss_items["lbox"], n)
        self.lobj.update(loss_items["lobj"], n)
        self.lcls.update(loss_items["lcls"], n)

    def as_dict(self):
        return {
            "loss": self.loss.avg,
            "lbox": self.lbox.avg,
            "lobj": self.lobj.avg,
            "lcls": self.lcls.avg,
        }


class CounterMeter:
    def __init__(self):
        self.images = 0
        self.targets = 0
        self.empty_images = 0
        self.empty_batches = 0
        self.positive_matches = 0

    def as_dict(self):
        return {
            "images": int(self.images),
            "targets": int(self.targets),
            "empty_images": int(self.empty_images),
            "empty_batches": int(self.empty_batches),
            "positive_matches": int(self.positive_matches),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLOv5FromScratch checkpoint")
    parser.add_argument("--data", type=str, default=str(ROOT / "datasets" / "coco2017" / "dataset.yaml"))
    parser.add_argument("--weights", type=str, default="", help="checkpoint path (.pt). If empty, evaluate random-init model")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--single-cls", action="store_true")
    parser.add_argument("--cache-labels", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--min-box-size", type=float, default=2.0)
    parser.add_argument("--anchor-t", type=float, default=6.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--project", type=str, default=str(ROOT / "runs" / "val"))
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--save-json", action="store_true")
    return parser.parse_args()


def load_data_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_num_classes(data_cfg, single_cls=False):
    if single_cls:
        return 1
    names = data_cfg.get("names")
    if isinstance(names, (list, tuple)):
        return len(names)
    if isinstance(names, dict):
        return len(names)
    if "nc" in data_cfg:
        return int(data_cfg["nc"])
    raise ValueError("Cannot infer number of classes from dataset.yaml")


def get_device(device_arg: str):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_model(model, weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    return ckpt


@torch.no_grad()
def validate(model, criterion, loader, device, args):
    model.eval()
    meters = LossMeters()
    counters = CounterMeter()
    start = time.time()
    nb = len(loader)
    autocast_enabled = bool(args.amp and device.type == "cuda")

    for batch_idx, (imgs, targets, metas) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        counters.images += int(imgs.size(0))
        counters.targets += int(targets.shape[0])

        if targets.numel() == 0:
            counters.empty_batches += 1
            counters.empty_images += int(imgs.size(0))
        else:
            for i in range(imgs.size(0)):
                if int((targets[:, 0] == i).sum().item()) == 0:
                    counters.empty_images += 1

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            outputs = model(imgs)
            _, _, indices, _ = criterion.build_targets(outputs, targets)
            batch_pos = sum(x[0].numel() for x in indices)
            counters.positive_matches += int(batch_pos)
            _, loss_items = criterion(outputs, targets)

        meters.update(loss_items, imgs.size(0))

        if batch_idx % args.print_freq == 0 or batch_idx == nb - 1:
            stats = meters.as_dict()
            print(
                f"val | batch {batch_idx + 1}/{nb} | "
                f"loss={stats['loss']:.4f} | lbox={stats['lbox']:.4f} | "
                f"lobj={stats['lobj']:.4f} | lcls={stats['lcls']:.4f} | "
                f"targets={int(counters.targets)} | pos={int(counters.positive_matches)}",
                flush=True,
            )

        if args.max_batches > 0 and (batch_idx + 1) >= args.max_batches:
            break

    stats = meters.as_dict()
    stats.update(counters.as_dict())
    stats["time_sec"] = time.time() - start
    stats["avg_targets_per_image"] = stats["targets"] / max(stats["images"], 1)
    stats["avg_positive_matches_per_image"] = stats["positive_matches"] / max(stats["images"], 1)
    return stats


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    data_cfg = load_data_yaml(args.data)
    nc = get_num_classes(data_cfg, args.single_cls)
    device = get_device(args.device)

    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"device = {device}", flush=True)
    print(f"data = {Path(args.data).resolve()}", flush=True)
    print(f"split = {args.split}", flush=True)
    print(f"nc = {nc}", flush=True)
    print(f"save_dir = {save_dir}", flush=True)

    _, loader = build_dataloader(
        data_yaml=args.data,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        augment=False,
        cache_labels=args.cache_labels,
        cache_images=args.cache_images,
        single_cls=args.single_cls,
        shuffle=False,
        persistent_workers=args.workers > 0,
        verbose=True,
        rebuild_cache=args.rebuild_cache,
        min_box_size=args.min_box_size,
    )

    model = YOLOv5FromScratch(nc=nc).to(device)
    criterion = YoloLoss(model.head, nc=nc, anchor_t=args.anchor_t).to(device)

    ckpt_meta = None
    if args.weights:
        ckpt_meta = load_checkpoint_model(model, args.weights, device)
        print(f"loaded weights from: {args.weights}", flush=True)
        if isinstance(ckpt_meta, dict) and "epoch" in ckpt_meta:
            print(f"checkpoint epoch = {ckpt_meta['epoch']}", flush=True)

    results = validate(model, criterion, loader, device, args)

    print("\n===== validation summary =====", flush=True)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}", flush=True)
        else:
            print(f"{k}: {v}", flush=True)

    if args.save_json:
        payload = {
            "args": vars(args),
            "results": results,
        }
        if isinstance(ckpt_meta, dict):
            payload["checkpoint_meta_keys"] = sorted(list(ckpt_meta.keys()))
        with open(save_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"saved json -> {save_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
