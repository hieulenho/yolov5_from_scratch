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



def find_positive_batch(loader, model, criterion, device, max_scan_batches=100):
    for batch_idx, (imgs, targets, metas) in enumerate(loader):
        print(f"\n[scan] batch {batch_idx} | targets={targets.shape}", flush=True)
        if targets.numel() == 0:
            continue

        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(imgs)
        tcls, tbox, indices, anch = criterion.build_targets(outputs, targets)
        total_pos = sum(x[0].numel() for x in indices)
        print(f"[scan] positives = {total_pos}", flush=True)
        if total_pos > 0:
            return imgs, targets, metas, outputs, indices

        if batch_idx + 1 >= max_scan_batches:
            break

    raise RuntimeError("Could not find a positive batch. Try --split train or --rebuild-cache.")



def main():
    torch.manual_seed(0)

    data_yaml = ROOT / "datasets" / "coco2017" / "dataset.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    t0 = time.time()
    print("[1] before build_dataloader", flush=True)
    dataset, loader = build_dataloader(
        data_yaml=str(data_yaml),
        split="train",
        img_size=640,
        batch_size=4,
        num_workers=0,
        cache_labels=True,
        cache_images=False,
        augment=False,
        shuffle=False,
        verbose=True,
        rebuild_cache=True,
    )
    print(f"[2] after build_dataloader | dt={time.time()-t0:.2f}s", flush=True)

    print("[3] before model init", flush=True)
    model = YOLOv5FromScratch(nc=80).to(device)
    criterion = YoloLoss(model.head, nc=80).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
    )
    model.train()
    print("[4] after model init", flush=True)

    print("[5] scanning for a positive batch", flush=True)
    imgs, targets, metas, outputs, indices = find_positive_batch(
        loader=loader,
        model=model,
        criterion=criterion,
        device=device,
        max_scan_batches=100,
    )

    print("imgs:", imgs.shape, flush=True)
    print("targets:", targets.shape, flush=True)
    print("targets sample:", flush=True)
    print(targets[:10], flush=True)
    for i, out in enumerate(outputs):
        print(f"outputs[{i}]: {out.shape}", flush=True)

    for i in range(len(outputs)):
        b, a, gj, gi = indices[i]
        print(f"scale {i}: positives = {b.numel()}", flush=True)

    t1 = time.time()
    print("[6] before criterion(outputs, targets)", flush=True)
    loss, items = criterion(outputs, targets)
    print(f"[7] after criterion(outputs, targets) | dt={time.time()-t1:.2f}s", flush=True)
    print("loss items:", items, flush=True)

    t2 = time.time()
    print("[8] before backward", flush=True)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    print(f"[9] after backward | dt={time.time()-t2:.2f}s", flush=True)

    grad_found = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"grad mean {name}: {p.grad.abs().mean().item():.6f}", flush=True)
            grad_found = True
            break

    assert grad_found, "No gradients found"
    assert torch.isfinite(loss).all(), "Loss is NaN or Inf"

    print("[10] before optimizer.step()", flush=True)
    optimizer.step()
    print("[11] after optimizer.step()", flush=True)

    print("test_loss: OK", flush=True)


if __name__ == "__main__":
    main()
