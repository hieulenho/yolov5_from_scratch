import torch

from yolov5_from_scratch.data.dataset import build_dataloader
from yolov5_from_scratch.models.yolo import YOLOv5FromScratch
from yolov5_from_scratch.paths import DATASETS_DIR



def main():
    data_yaml = DATASETS_DIR / "coco2017" / "dataset.yaml"

    print("[1] before build_dataloader", flush=True)
    dataset, loader = build_dataloader(
        data_yaml=str(data_yaml),
        split="train",
        img_size=640,
        batch_size=2,
        num_workers=0,
        cache_labels=True,
        cache_images=False,
        augment=False,
        shuffle=False,
        verbose=True,
        rebuild_cache=True,
    )
    print("[2] after build_dataloader", flush=True)
    print(f"dataset len = {len(dataset)}", flush=True)

    imgs = None
    targets = None
    metas = None
    for batch_idx, batch in enumerate(loader):
        imgs, targets, metas = batch
        print(f"[3] scanned batch {batch_idx} | targets={targets.shape}", flush=True)
        if targets.numel() > 0:
            break
    else:
        raise RuntimeError("No non-empty batch found")

    print("imgs:", imgs.shape, flush=True)
    print("targets:", targets.shape, flush=True)
    if targets.numel() > 0:
        print(targets[:5], flush=True)

    print("[4] before model init", flush=True)
    model = YOLOv5FromScratch(nc=80)
    model.eval()
    print("[5] after model init", flush=True)

    print("[6] before forward", flush=True)
    with torch.no_grad():
        outputs = model(imgs)
    print("[7] after forward", flush=True)

    for i, out in enumerate(outputs):
        print(f"outputs[{i}]:", out.shape, flush=True)


if __name__ == "__main__":
    main()
