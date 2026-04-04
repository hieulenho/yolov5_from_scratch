import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv5FromScratch end-to-end")
    parser.add_argument("--data", type=str, default=str(ROOT / "datasets" / "coco2017" / "dataset.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--single-cls", action="store_true")
    parser.add_argument("--cache-labels", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--min-box-size", type=float, default=2.0)

    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "AdamW"])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lrf", type=float, default=1e-2, help="final lr factor for cosine schedule")
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--clip-grad", type=float, default=10.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--anchor-t", type=float, default=6.0)

    parser.add_argument("--val", action="store_true", help="run validation each epoch")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=20)

    parser.add_argument("--project", type=str, default=str(ROOT / "runs" / "train"))
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-period", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def make_optimizer(args, model):
    if args.optimizer == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    if args.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def make_scheduler(args, optimizer):
    epochs = max(args.epochs, 1)

    def lf(epoch):
        # cosine from 1.0 -> lrf
        return ((1 + math.cos(math.pi * epoch / epochs)) / 2) * (1 - args.lrf) + args.lrf

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


def get_device(device_arg: str):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, best_val_loss, args):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def load_checkpoint(resume_path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    return start_epoch, best_val_loss


def train_one_epoch(model, criterion, optimizer, loader, device, epoch, args, scaler=None):
    model.train()
    meters = LossMeters()
    start = time.time()
    nb = len(loader)
    nw = max(round(args.warmup_epochs * nb), 1) if args.warmup_epochs > 0 else 0
    autocast_enabled = bool(args.amp and device.type == "cuda")

    for batch_idx, (imgs, targets, _) in enumerate(loader):
        ni = epoch * nb + batch_idx
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if nw > 0 and ni < nw:
            warm = (ni + 1) / nw
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * warm
                if "momentum" in pg:
                    pg["momentum"] = 0.8 + (args.momentum - 0.8) * warm

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            outputs = model(imgs)
            loss, loss_items = criterion(outputs, targets)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is NaN/Inf at epoch={epoch} batch={batch_idx}: {loss_items}")

        if scaler is not None and autocast_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            optimizer.step()

        meters.update(loss_items, imgs.size(0))

        if batch_idx % args.print_freq == 0 or batch_idx == nb - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            stats = meters.as_dict()
            print(
                f"train | epoch {epoch + 1} | batch {batch_idx + 1}/{nb} | "
                f"lr={cur_lr:.6g} | loss={stats['loss']:.4f} | lbox={stats['lbox']:.4f} | "
                f"lobj={stats['lobj']:.4f} | lcls={stats['lcls']:.4f}",
                flush=True,
            )

        if args.max_train_batches > 0 and (batch_idx + 1) >= args.max_train_batches:
            break

    stats = meters.as_dict()
    stats["time_sec"] = time.time() - start
    return stats


@torch.no_grad()
def validate(model, criterion, loader, device, epoch, args):
    model.eval()
    meters = LossMeters()
    start = time.time()
    autocast_enabled = bool(args.amp and device.type == "cuda")
    nb = len(loader)

    for batch_idx, (imgs, targets, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            outputs = model(imgs)
            _, loss_items = criterion(outputs, targets)

        meters.update(loss_items, imgs.size(0))

        if batch_idx % args.print_freq == 0 or batch_idx == nb - 1:
            stats = meters.as_dict()
            print(
                f"val   | epoch {epoch + 1} | batch {batch_idx + 1}/{nb} | "
                f"loss={stats['loss']:.4f} | lbox={stats['lbox']:.4f} | "
                f"lobj={stats['lobj']:.4f} | lcls={stats['lcls']:.4f}",
                flush=True,
            )

        if args.max_val_batches > 0 and (batch_idx + 1) >= args.max_val_batches:
            break

    stats = meters.as_dict()
    stats["time_sec"] = time.time() - start
    return stats


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    data_cfg = load_data_yaml(args.data)
    nc = get_num_classes(data_cfg, args.single_cls)
    device = get_device(args.device)

    save_dir = Path(args.project) / args.name
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"device = {device}", flush=True)
    print(f"data = {Path(args.data).resolve()}", flush=True)
    print(f"nc = {nc}", flush=True)
    print(f"save_dir = {save_dir}", flush=True)

    train_dataset, train_loader = build_dataloader(
        data_yaml=args.data,
        split="train",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        augment=True,
        cache_labels=args.cache_labels,
        cache_images=args.cache_images,
        single_cls=args.single_cls,
        shuffle=True,
        persistent_workers=args.workers > 0,
        verbose=True,
        rebuild_cache=args.rebuild_cache,
        min_box_size=args.min_box_size,
    )

    val_loader = None
    if args.val:
        _, val_loader = build_dataloader(
            data_yaml=args.data,
            split="val",
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.workers,
            augment=False,
            cache_labels=args.cache_labels,
            cache_images=False,
            single_cls=args.single_cls,
            shuffle=False,
            persistent_workers=args.workers > 0,
            verbose=True,
            rebuild_cache=False,
            min_box_size=args.min_box_size,
        )

    model = YOLOv5FromScratch(nc=nc).to(device)
    criterion = YoloLoss(model.head, nc=nc, anchor_t=args.anchor_t).to(device)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)

    history = []
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        print(f"\n========== epoch {epoch + 1}/{args.epochs} ==========" , flush=True)
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            epoch=epoch,
            args=args,
            scaler=scaler,
        )

        val_stats = None
        if val_loader is not None and ((epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs):
            val_stats = validate(
                model=model,
                criterion=criterion,
                loader=val_loader,
                device=device,
                epoch=epoch,
                args=args,
            )

        scheduler.step()

        row = {
            "epoch": epoch + 1,
            "train": train_stats,
            "val": val_stats,
        }
        history.append(row)
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        last_path = weights_dir / "last.pt"
        save_checkpoint(last_path, epoch, model, optimizer, scheduler, scaler, best_val_loss, args)

        should_save_epoch = args.save_period > 0 and ((epoch + 1) % args.save_period == 0)
        if should_save_epoch:
            save_checkpoint(weights_dir / f"epoch_{epoch + 1:03d}.pt", epoch, model, optimizer, scheduler, scaler, best_val_loss, args)

        if val_stats is not None and val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            save_checkpoint(weights_dir / "best.pt", epoch, model, optimizer, scheduler, scaler, best_val_loss, args)
            print(f"saved new best.pt | val_loss={best_val_loss:.4f}", flush=True)

        train_msg = (
            f"epoch {epoch + 1}: train loss={train_stats['loss']:.4f} "
            f"(lbox={train_stats['lbox']:.4f}, lobj={train_stats['lobj']:.4f}, lcls={train_stats['lcls']:.4f})"
        )
        if val_stats is not None:
            train_msg += (
                f" | val loss={val_stats['loss']:.4f} "
                f"(lbox={val_stats['lbox']:.4f}, lobj={val_stats['lobj']:.4f}, lcls={val_stats['lcls']:.4f})"
            )
        print(train_msg, flush=True)

    print(f"training done in {(time.time() - train_start) / 3600:.2f} hours", flush=True)
    print(f"artifacts saved to: {save_dir}", flush=True)


if __name__ == "__main__":
    main()
