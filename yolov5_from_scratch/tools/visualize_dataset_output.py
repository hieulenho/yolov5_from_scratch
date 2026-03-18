import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from data.dataset import build_dataloader




def draw_yolo_boxes(img, targets, class_names=None):
    """
    img: uint8 RGB image, shape [H, W, 3]
    targets: tensor/ndarray [N, 5] = [cls, x, y, w, h] normalized
    """
    out = img.copy()
    h, w = out.shape[:2]

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    for row in targets:
        cls_id, xc, yc, bw, bh = row.tolist()

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # label = str(int(cls_id))
        # if class_names is not None and int(cls_id) in class_names:
        #     label = class_names[int(cls_id)]
            
        label = str(int(cls_id))
        if class_names is not None:
            if isinstance(class_names, dict):
                label = class_names.get(int(cls_id), label)
            elif isinstance(class_names, (list, tuple)) and 0 <= int(cls_id) < len(class_names):
                label = class_names[int(cls_id)]

        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return out


def main():
    save_dir = ROOT / "runs" / "vis_dataset_output"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset, _ = build_dataloader(
        data_yaml=str(ROOT / "datasets" / "coco2017" / "dataset.yaml"),
        split="train",
        img_size=640,
        batch_size=4,
        num_workers=0,
        cache_labels=True,
        cache_images=False,
        verbose=False,
    )

    for i in range(12):
        img_tensor, targets, meta = dataset[i]

        img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        img = draw_yolo_boxes(img, targets, class_names=dataset.names)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        stem = Path(meta["im_file"]).stem
        save_path = save_dir / f"{i:02d}_{stem}.jpg"
        cv2.imwrite(str(save_path), img_bgr)

    print(f"Saved to: {save_dir}")


    # count = 0
    # i = 0
    # while count < 12 and i < len(dataset):
    #     img_tensor, targets, meta = dataset[i]
    #     if len(targets) == 0:
    #         i += 1
    #         continue

    #     img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    #     img = draw_yolo_boxes(img, targets, class_names=dataset.names)

    #     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     stem = Path(meta["im_file"]).stem
    #     save_path = save_dir / f"{count:02d}_{stem}.jpg"
    #     cv2.imwrite(str(save_path), img_bgr)

    #     print(f"saved: {save_path.name} | n_targets={len(targets)}")
    #     count += 1
    #     i += 1

if __name__ == "__main__":
    main()
