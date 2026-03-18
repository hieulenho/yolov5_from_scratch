import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloader
from models.yolo import YOLOv5FromScratch
from loss.loss import YoloLoss


def main(): # Kiểm tra loss và gradient
    torch.manual_seed(0)

    data_yaml = ROOT / "datasets" / "coco2017" / "dataset.yaml"

    dataset, loader = build_dataloader(
        data_yaml=str(data_yaml),
        split="train",
        img_size=640,
        batch_size=2,
        num_workers=0,
        cache_labels=True,
        cache_images=False,
        verbose=False,
    )

    model = YOLOv5FromScratch(nc=80)
    criterion = YoloLoss(model.head, nc=80)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
    )

    model.train()

    imgs, targets, metas = next(iter(loader))

    print("imgs:", imgs.shape)
    print("targets:", targets.shape)
    if targets.numel() > 0:
        print("targets sample:")
        print(targets[:5])

    outputs = model(imgs)

    for i, out in enumerate(outputs):
        print(f"outputs[{i}]:", out.shape)

    loss, items = criterion(outputs, targets)
    print("loss items:", items)

    # debug positives
    tcls, tbox, indices, anch = criterion.build_targets(outputs, targets)
    for i in range(len(outputs)):
        b, a, gj, gi = indices[i]
        print(f"scale {i}: positives = {b.numel()}")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    grad_found = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"grad mean {name}: {p.grad.abs().mean().item():.6f}")
            grad_found = True
            break

    assert grad_found, "No gradients found"
    assert torch.isfinite(loss).all(), "Loss is NaN or Inf"

    optimizer.step()
    print("Backward + optimizer step: OK")


if __name__ == "__main__":
    main()