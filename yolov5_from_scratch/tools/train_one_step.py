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


def main():
    torch.manual_seed(0)

    data_yaml = ROOT / "datasets" / "coco2017" / "dataset.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset, loader = build_dataloader(
        data_yaml=str(data_yaml),
        split="train",
        img_size=640,
        batch_size=2,
        num_workers=0,
        cache_labels=True,
        cache_images=False,
        augment=True,
        verbose=False,
    )

    model = YOLOv5FromScratch(nc=80).to(device)
    criterion = YoloLoss(model.head, nc=80)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
    )

    model.train()

    max_steps = 5
    step = 0

    for imgs, targets, metas in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(imgs)
        loss, loss_items = criterion(outputs, targets)

        tcls, tbox, indices, anch = criterion.build_targets(outputs, targets)
        pos_counts = []
        for i in range(len(outputs)):
            b, a, gj, gi = indices[i]
            pos_counts.append(int(b.numel()))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print(
            f"step={step} "
            f"loss={loss_items['loss']:.4f} "
            f"lbox={loss_items['lbox']:.4f} "
            f"lobj={loss_items['lobj']:.4f} "
            f"lcls={loss_items['lcls']:.4f} "
            f"n_targets={targets.shape[0]} "
            f"positives={pos_counts}"
        )

        assert torch.isfinite(loss).all(), "Loss is NaN or Inf"

        step += 1
        if step >= max_steps:
            break

    print("train_one_step: OK")


if __name__ == "__main__":
    main()