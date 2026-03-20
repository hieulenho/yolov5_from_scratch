import sys
from pathlib import Path
import time

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("[1] importing build_dataloader...", flush=True)
from data.dataset import build_dataloader

print("[2] importing model...", flush=True)
from models.yolo import YOLOv5FromScratch

print("[3] importing loss...", flush=True)
from loss.loss import YoloLoss


def main():
    torch.manual_seed(0)

    data_yaml = ROOT / "datasets" / "coco2017" / "dataset.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[4] device = {device}", flush=True)

    t0 = time.time()
    print("[5] before build_dataloader", flush=True)
    dataset, loader = build_dataloader(
        data_yaml=str(data_yaml),
        split="val",          # debug bằng val trước
        img_size=320,         # giảm tải mạnh
        batch_size=1,         # giảm tải mạnh
        num_workers=0,
        cache_labels=True,
        cache_images=False,
        augment=False,
        shuffle=False,
        verbose=True,
    )
    print(f"[6] after build_dataloader | dt={time.time()-t0:.2f}s", flush=True)

    t1 = time.time()
    print("[7] before next(iter(loader))", flush=True)
    imgs, targets, metas = next(iter(loader))
    print(f"[8] after next(iter(loader)) | dt={time.time()-t1:.2f}s", flush=True)

    print("imgs:", imgs.shape, flush=True)
    print("targets:", targets.shape, flush=True)
    if targets.numel() > 0:
        print("targets sample:")
        print(targets[:5], flush=True)

    t2 = time.time()
    print("[9] before model init", flush=True)
    model = YOLOv5FromScratch(nc=80).to(device)
    criterion = YoloLoss(model.head, nc=80).to(device)
    print(f"[10] after model init | dt={time.time()-t2:.2f}s", flush=True)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
    )

    model.train()

    imgs = imgs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    t3 = time.time()
    print("[11] before forward", flush=True)
    outputs = model(imgs)
    print(f"[12] after forward | dt={time.time()-t3:.2f}s", flush=True)

    for i, out in enumerate(outputs):
        print(f"outputs[{i}]: {out.shape}", flush=True)

    t4 = time.time()
    print("[13] before criterion(outputs, targets)", flush=True)
    loss, items = criterion(outputs, targets)
    print(f"[14] after criterion(outputs, targets) | dt={time.time()-t4:.2f}s", flush=True)
    print("loss items:", items, flush=True)

    t5 = time.time()
    print("[15] before build_targets(debug)", flush=True)
    tcls, tbox, indices, anch = criterion.build_targets(outputs, targets)
    for i in range(len(outputs)):
        b, a, gj, gi = indices[i]
        print(f"scale {i}: positives = {b.numel()}", flush=True)
    print(f"[16] after build_targets(debug) | dt={time.time()-t5:.2f}s", flush=True)

    t6 = time.time()
    print("[17] before backward", flush=True)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    print(f"[18] after backward | dt={time.time()-t6:.2f}s", flush=True)

    grad_found = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"grad mean {name}: {p.grad.abs().mean().item():.6f}", flush=True)
            grad_found = True
            break

    assert grad_found, "No gradients found"
    assert torch.isfinite(loss).all(), "Loss is NaN or Inf"

    t7 = time.time()
    print("[19] before optimizer.step()", flush=True)
    optimizer.step()
    print(f"[20] after optimizer.step() | dt={time.time()-t7:.2f}s", flush=True)

    print("test_loss: OK", flush=True)


if __name__ == "__main__":
    main()