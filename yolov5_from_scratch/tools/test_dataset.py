import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloader


def main():
    data_yaml = ROOT / "datasets" / "coco2017" / "dataset.yaml"

    print("ROOT:", ROOT)
    print("DATA YAML:", data_yaml)
    print("DATA YAML exists:", data_yaml.exists())

    dataset, loader = build_dataloader(
        data_yaml=str(data_yaml),
        split="train",
        img_size=640,
        batch_size=4,
        num_workers=0,   # Windows: debug trước với 0
        cache_labels=True,
        cache_images=False,
        verbose=True,
    )

    print("\n=== Single sample check ===")
    img, targets, meta = dataset[0]
    print("single image shape:", img.shape)
    print("single image dtype:", img.dtype)
    print("single image min/max:", float(img.min()), float(img.max()))
    print("single targets shape:", targets.shape)
    if targets.numel() > 0:
        print("single targets first rows:\n", targets[:5])
    print("single meta:", meta)

    print("\n=== One batch check ===")
    imgs, batch_targets, metas = next(iter(loader))
    print("batch imgs shape:", imgs.shape)
    print("batch imgs dtype:", imgs.dtype)
    print("batch targets shape:", batch_targets.shape)

    if batch_targets.numel() > 0:
        print("batch targets first rows:\n", batch_targets[:10])
        print("batch idx min/max:", int(batch_targets[:, 0].min()), int(batch_targets[:, 0].max()))
        print("cls min/max:", int(batch_targets[:, 1].min()), int(batch_targets[:, 1].max()))
        print("xywh min:", float(batch_targets[:, 2:6].min()))
        print("xywh max:", float(batch_targets[:, 2:6].max()))

    print("first meta:", metas[0])


if __name__ == "__main__":
    main()

