import sys
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(ROOT / "datasets" / "coco2017" / "dataset.yaml"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device = {device}", flush=True)
    print(f"data = {args.data}", flush=True)

    _, loader = build_dataloader(
        data_yaml=args.data,
        split="train",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        cache_labels=True,
        cache_images=False,
        augment=True,
        shuffle=True,
        persistent_workers=args.workers > 0,
        verbose=True,
        rebuild_cache=args.rebuild_cache,
    )

    model = YOLOv5FromScratch(nc=80).to(device)
    criterion = YoloLoss(model.head, nc=80).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        model.train()
        print(f"\n===== epoch {epoch + 1}/{args.epochs} =====", flush=True)

        seen = 0
        for batch_idx, (imgs, targets, metas) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(imgs)
            loss, items = criterion(outputs, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            seen += imgs.shape[0]
            print(
                f"batch {batch_idx} | seen={seen} | "
                f"loss={items['loss']:.4f} lbox={items['lbox']:.4f} "
                f"lobj={items['lobj']:.4f} lcls={items['lcls']:.4f}",
                flush=True,
            )

            if args.max_batches_per_epoch > 0 and batch_idx + 1 >= args.max_batches_per_epoch:
                break

    print("train.py: OK", flush=True)


if __name__ == "__main__":
    main()
