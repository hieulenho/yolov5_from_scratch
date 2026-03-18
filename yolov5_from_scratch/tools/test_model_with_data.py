import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("[1] importing build_dataloader...", flush=True)
from data.dataset import build_dataloader

print("[2] importing model...", flush=True)
from models.yolo import YOLOv5FromScratch


def main():
    data_yaml = ROOT / "datasets" / "coco2017" / "dataset.yaml"

    print("[3] before build_dataloader", flush=True)
    dataset, loader = build_dataloader(
        data_yaml=str(data_yaml),
        split="val",          # dùng val trước cho nhẹ hơn train
        img_size=640,
        batch_size=2,
        num_workers=0,
        cache_labels=True,
        cache_images=False,
        augment=False,
        shuffle=False,
        verbose=True,         # bật log lên
    )
    print("[4] after build_dataloader", flush=True)

    print(f"[5] dataset len = {len(dataset)}", flush=True)

    print("[6] before next(iter(loader))", flush=True)
    imgs, targets, metas = next(iter(loader))
    print("[7] after next(iter(loader))", flush=True)

    print("imgs:", imgs.shape, flush=True)
    print("targets:", targets.shape, flush=True)
    if targets.numel() > 0:
        print(targets[:5], flush=True)

    print("[8] before model init", flush=True)
    model = YOLOv5FromScratch(nc=80)
    model.eval()
    print("[9] after model init", flush=True)

    print("[10] before forward", flush=True)
    with torch.no_grad():
        outputs = model(imgs)
    print("[11] after forward", flush=True)

    for i, out in enumerate(outputs):
        print(f"outputs[{i}]:", out.shape, flush=True)


if __name__ == "__main__":
    main()