import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from yolov5_from_scratch.data.dataset import build_dataloader
from yolov5_from_scratch.loss.loss import YoloLoss
from yolov5_from_scratch.models.yolo import YOLOv5FromScratch
from yolov5_from_scratch.paths import DATASETS_DIR, TRAINING_DIR
from yolov5_from_scratch.training.meters import LossMeters


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv5FromScratch end-to-end")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATASETS_DIR / "coco2017" / "dataset.yaml"),
    )
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

    parser.add_argument("--project", type=str, default=str(TRAINING_DIR))
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="initialize model weights without restoring optimizer or epoch",
    )
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


def get_class_names(data_cfg):
    names = data_cfg.get("names")
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=lambda value: int(value))]
    if isinstance(names, (list, tuple)):
        return list(names)
    return []


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
    ckpt = torch.load(
        resume_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        checkpoint_args = ckpt.get("args", {})
        checkpoint_optimizer = (
            checkpoint_args.get("optimizer")
            if isinstance(checkpoint_args, dict)
            else None
        )
        current_optimizer = optimizer.__class__.__name__
        if (
            checkpoint_optimizer
            and checkpoint_optimizer.lower() != current_optimizer.lower()
        ):
            raise ValueError(
                "Optimizer mismatch while resuming: checkpoint uses "
                f"{checkpoint_optimizer}, but --optimizer is {current_optimizer}"
            )
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    return start_epoch, best_val_loss


def load_resume_history(resume_path, save_dir, completed_epochs):
    resume_path = Path(resume_path)
    candidates = [
        Path(save_dir) / "history.json",
        resume_path.parent / "history.json",
        resume_path.parent.parent / "history.json",
    ]
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.is_file():
            continue
        seen.add(resolved)
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                history = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(history, list):
            continue
        return [
            row
            for row in history
            if isinstance(row, dict)
            and int(row.get("epoch", 0)) <= completed_epochs
        ]
    return []


def get_checkpoint_class_names(checkpoint):
    checkpoint_args = checkpoint.get("args", {})
    source_data = checkpoint_args.get("data") if isinstance(checkpoint_args, dict) else None
    if not source_data:
        return []
    source_data_path = Path(source_data)
    if not source_data_path.is_file():
        source_text = str(source_data_path).lower()
        if "coco2017" in source_text:
            source_data_path = DATASETS_DIR / "coco2017" / "dataset.yaml"
        elif "traffic5" in source_text:
            source_data_path = DATASETS_DIR / "traffic5" / "dataset.yaml"
    if not source_data_path.is_file():
        return []
    return get_class_names(load_data_yaml(source_data_path))


def remap_detect_parameter(source, target, source_names, target_names):
    if source.ndim not in (1, 4) or target.ndim != source.ndim:
        return None
    num_anchors = 3
    if source.shape[0] % num_anchors or target.shape[0] % num_anchors:
        return None

    source_no = source.shape[0] // num_anchors
    target_no = target.shape[0] // num_anchors
    if source_no != len(source_names) + 5 or target_no != len(target_names) + 5:
        return None
    if source.shape[1:] != target.shape[1:]:
        return None

    source_view = source.view(num_anchors, source_no, *source.shape[1:])
    target_view = target.clone().view(num_anchors, target_no, *target.shape[1:])
    target_view[:, :5] = source_view[:, :5]

    source_name_to_id = {
        str(name).strip().lower(): index for index, name in enumerate(source_names)
    }
    for target_id, target_name in enumerate(target_names):
        source_id = source_name_to_id.get(str(target_name).strip().lower())
        if source_id is not None:
            target_view[:, 5 + target_id] = source_view[:, 5 + source_id]
    return target_view.view_as(target)


def load_pretrained_weights(weights_path, model, target_names, device="cpu"):
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    source_state = checkpoint.get("model", checkpoint)
    target_state = model.state_dict()
    source_names = get_checkpoint_class_names(checkpoint)
    transfer_state = {}
    remapped_head = []
    skipped = []

    for key, source_value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            skipped.append(key)
            continue
        if source_value.shape == target_value.shape:
            transfer_state[key] = source_value
            continue
        if key.startswith("head.m.") and source_names and target_names:
            remapped = remap_detect_parameter(
                source_value,
                target_value,
                source_names,
                target_names,
            )
            if remapped is not None:
                transfer_state[key] = remapped
                remapped_head.append(key)
                continue
        skipped.append(key)

    missing, unexpected = model.load_state_dict(transfer_state, strict=False)
    return {
        "loaded": len(transfer_state),
        "source_total": len(source_state),
        "source_names": source_names,
        "remapped_head": remapped_head,
        "skipped": skipped,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


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
        with torch.amp.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
        ):
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

        with torch.amp.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
        ):
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
    if args.resume and args.weights:
        raise ValueError("--resume and --weights cannot be used together")
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
    if args.weights:
        transfer = load_pretrained_weights(
            args.weights,
            model=model,
            target_names=get_class_names(data_cfg),
            device=device,
        )
        print(
            f"initialized from {args.weights} | "
            f"loaded={transfer['loaded']}/{transfer['source_total']} | "
            f"remapped_head={len(transfer['remapped_head'])} | "
            f"skipped={len(transfer['skipped'])}",
            flush=True,
        )
        if not transfer["source_names"]:
            print(
                "warning: source class names were unavailable; "
                "detection head class channels were not transferred",
                flush=True,
            )
    criterion = YoloLoss(model.head, nc=nc, anchor_t=args.anchor_t).to(device)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=bool(args.amp and device.type == "cuda"),
    )

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
        print(
            f"resumed from {args.resume} | next epoch = {start_epoch + 1}",
            flush=True,
        )

    history = (
        load_resume_history(args.resume, save_dir, start_epoch)
        if args.resume
        else []
    )
    if history:
        print(f"restored {len(history)} history rows", flush=True)
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

        is_new_best = False
        if val_stats is not None and val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            is_new_best = True

        last_path = weights_dir / "last.pt"
        save_checkpoint(last_path, epoch, model, optimizer, scheduler, scaler, best_val_loss, args)

        should_save_epoch = args.save_period > 0 and ((epoch + 1) % args.save_period == 0)
        if should_save_epoch:
            save_checkpoint(weights_dir / f"epoch_{epoch + 1:03d}.pt", epoch, model, optimizer, scheduler, scaler, best_val_loss, args)

        if is_new_best:
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
