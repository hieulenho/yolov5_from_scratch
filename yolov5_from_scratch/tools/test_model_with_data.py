import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloader
from models.yolo import YOLOv5FromScratch


def main():
    data_yaml = ROOT / "datasets" / "coco2017" / "dataset.yaml"

    dataset, loader = build_dataloader(
        data_yaml=str(data_yaml),
        split="train",
        img_size=640,
        batch_size=2,
        num_workers=0,
        cache_labels=True,
        cache_images=False,
        verbose=True,
    )

    model = YOLOv5FromScratch(nc=80)
    model.eval()

    imgs, targets, metas = next(iter(loader))
    print("imgs:", imgs.shape)
    print("targets:", targets.shape)

    with torch.no_grad():
        outputs = model(imgs)

    for i, out in enumerate(outputs):
        print(f"outputs[{i}]:", out.shape)


if __name__ == "__main__":
    main()