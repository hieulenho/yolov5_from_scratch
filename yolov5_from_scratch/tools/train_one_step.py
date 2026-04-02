import sys
from pathlib import Path
import time
import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloader
from models.yolo import YOLOv5FromScratch
from loss.loss import YoloLoss


def print_target_stats(targets: torch.Tensor):
    print(f"n_targets = {targets.shape[0]}", flush=True)
    if targets.numel() == 0:
        return

    print("targets[:10] =", flush=True)
    print(targets[:10], flush=True)
    print(
        "batch_idx min/max:",
        int(targets[:, 0].min().item()),
        int(targets[:, 0].max().item()),
        flush=True,
    )
    print(
        "cls min/max:",
        int(targets[:, 1].min().item()),
        int(targets[:, 1].max().item()),
        flush=True,
    )
    print(
        "xywh min:",
        targets[:, 2:6].min(dim=0).values,
        flush=True,
    )
    print(
        "xywh max:",
        targets[:, 2:6].max(dim=0).values,
        flush=True,
    )


def print_batch_breakdown(targets: torch.Tensor, metas):
    bs = len(metas)
    counts = [0] * bs

    if targets.numel() > 0:
        for i in range(bs):
            counts[i] = int((targets[:, 0] == i).sum().item())

    print("targets per image:", counts, flush=True)
    for i, meta in enumerate(metas):
        print(
            f"  [{i}] {Path(meta['im_file']).name} | "
            f"orig={meta['orig_shape']} resized={meta['resized_shape']} "
            f"ratio={meta['ratio']} pad={meta['pad']} n={counts[i]}",
            flush=True,
        )


def print_match_stats(indices):
    total_pos = 0
    for i, (b, a, gj, gi) in enumerate(indices):
        n_pos = b.numel()
        total_pos += n_pos
        print(f"scale {i}: positives = {n_pos}", flush=True)
        if n_pos > 0:
            uniq_anchor, anchor_count = a.unique(return_counts=True)
            anchor_info = [
                (int(k.item()), int(v.item()))
                for k, v in zip(uniq_anchor, anchor_count)
            ]
            print(f"  anchor usage = {anchor_info}", flush=True)
            print(
                f"  gi range = [{int(gi.min().item())}, {int(gi.max().item())}] | "
                f"gj range = [{int(gj.min().item())}, {int(gj.max().item())}]",
                flush=True,
            )
    print(f"total positives = {total_pos}", flush=True)


def main():
    torch.manual_seed(0)

    data_yaml = ROOT / "datasets" / "coco2017" / "dataset.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = "train"
    img_size = 640
    batch_size = 4
    max_train_steps = 20
    max_scan_batches = 200

    print(f"device = {device}", flush=True)
    print(f"data_yaml = {data_yaml}", flush=True)

    t0 = time.time()
    print("[1] before build_dataloader", flush=True)
    dataset, loader = build_dataloader(
        data_yaml=str(data_yaml),
        split=split,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=0,
        cache_labels=True,
        cache_images=False,
        augment=False,
        shuffle=False,
        persistent_workers=False,
        verbose=True,
    )
    print(f"[2] after build_dataloader | dt={time.time() - t0:.2f}s", flush=True)
    print(f"dataset len = {len(dataset)}", flush=True)

    model = YOLOv5FromScratch(nc=80).to(device)
    criterion = YoloLoss(model.head, nc=80, anchor_t=6.0).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
    )
    model.train()

    trained_steps = 0
    scanned_batches = 0

    for batch_idx, (imgs, targets, metas) in enumerate(loader):
        scanned_batches += 1
        print(f"\n========== batch {batch_idx} ==========", flush=True)
        print(f"imgs.shape = {tuple(imgs.shape)}", flush=True)
        print_target_stats(targets)
        print_batch_breakdown(targets, metas)

        if targets.shape[0] == 0:
            print("skip empty batch", flush=True)
            if scanned_batches >= max_scan_batches:
                print("Reached max_scan_batches while only seeing empty batches.", flush=True)
                break
            continue

        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        t1 = time.time()
        outputs = model(imgs)
        print(f"forward dt = {time.time() - t1:.4f}s", flush=True)
        for i, out in enumerate(outputs):
            print(f"outputs[{i}].shape = {tuple(out.shape)}", flush=True)

        tcls, tbox, indices, anch = criterion.build_targets(outputs, targets)
        print_match_stats(indices)

        if sum(x[0].numel() for x in indices) == 0:
            print("All scales have 0 positives -> skip backward for this batch", flush=True)
            if scanned_batches >= max_scan_batches:
                print("Reached max_scan_batches without finding a positive batch.", flush=True)
                break
            continue

        t2 = time.time()
        loss, loss_items = criterion(outputs, targets)
        print(f"loss dt = {time.time() - t2:.4f}s", flush=True)
        print("loss items:", loss_items, flush=True)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is NaN/Inf: {loss_items}")

        optimizer.zero_grad(set_to_none=True)
        t3 = time.time()
        loss.backward()
        print(f"backward dt = {time.time() - t3:.4f}s", flush=True)

        grad_found = False
        for name, p in model.named_parameters():
            if p.grad is not None:
                print(f"first grad mean | {name} = {p.grad.abs().mean().item():.6f}", flush=True)
                grad_found = True
                break
        if not grad_found:
            raise RuntimeError("No gradients found")

        optimizer.step()
        trained_steps += 1
        print(f"trained_steps = {trained_steps}", flush=True)

        if trained_steps >= max_train_steps:
            break
        if scanned_batches >= max_scan_batches:
            break

    print("train_one_step: OK", flush=True)


if __name__ == "__main__":
    main()