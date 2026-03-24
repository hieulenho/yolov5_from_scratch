import sys
from pathlib import Path
import time
import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# print("[1] importing build_dataloader...", flush=True)
from data.dataset import build_dataloader

# print("[2] importing model...", flush=True)
from models.yolo import YOLOv5FromScratch

# print("[3] importing loss...", flush=True)
from loss.loss import YoloLoss


def main():
    torch.manual_seed(0)

    data_yaml = ROOT / "datasets" / "coco2017" / "dataset.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"[4] device: {device}", flush=True)

    t0 = time.time()
    print("[5] before build_dataloader", flush=True)
    dataset, loader = build_dataloader(
        data_yaml=str(data_yaml),
        split="val",          # debug bằng val trước
        img_size=320,         # giảm tải
        batch_size=1,         # giảm tải
        num_workers=0,
        cache_labels=True,
        cache_images=False,
        augment=False,        # tắt augment để debug
        shuffle=False,
        persistent_workers=False,
        verbose=True,
    )
    # print(f"[6] after build_dataloader | dt={time.time()-t0:.2f}s", flush=True)

    # print(f"[7] dataset len = {len(dataset)}", flush=True)

    model = YOLOv5FromScratch(nc=80).to(device)
    criterion = YoloLoss(model.head, nc=80, anchor_t=6.0).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
    )
    model.train()
    # print("[8] model/criterion/optimizer ready", flush=True)

    step = 0
    max_steps = 3 
    

    # for imgs, targets, metas in loader:
    #     print("[9] got batch from loader", flush=True)

    #     imgs = imgs.to(device, non_blocking=True)
    #     targets = targets.to(device, non_blocking=True)
    #     print("[10] batch moved to device", flush=True)

    #     outputs = model(imgs)
    #     print("[11] after forward", flush=True)

    #     loss, loss_items = criterion(outputs, targets)
        

    #     tcls, tbox, indices, anch = criterion.build_targets(outputs, targets)
    #     for i in range(len(outputs)):
    #         b, a, gj, gi = indices[i]
    #         print(f"scale {i}: positives = {b.numel()}", flush=True)

    #     print(f"[12] after loss | {loss_items}", flush=True)

    #     optimizer.zero_grad(set_to_none=True)
    #     print("[13] before backward", flush=True)
    #     loss.backward()
    #     print("[14] after backward", flush=True)

    #     optimizer.step()
    #     print("[15] after optimizer.step()", flush=True)

    #     step += 1
    #     if step >= max_steps:
    #         break

    # print("train_one_step: OK", flush=True)
    for imgs, targets, metas in loader:
        # if targets.shape[0] == 0:
        #     print("skip empty batch", flush=True)
        #     continue
        print(f"\nn_targets = {targets.shape[0]}", flush=True)
        if targets.numel() > 0:
            print(targets[:10], flush=True)
        
        if targets.shape[0] == 0:
            print("skip empty batch", flush=True)
            continue

        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(imgs)
        loss, loss_items = criterion(outputs, targets)

        print(f"n_targets = {targets.shape[0]}", flush=True)
        if targets.numel() > 0:
            print(targets[:10], flush=True)

        tcls, tbox, indices, anch = criterion.build_targets(outputs, targets)
        for i in range(len(outputs)):
            b, a, gj, gi = indices[i]
            print(f"scale {i}: positives = {b.numel()}", flush=True)

        print("loss items:", loss_items, flush=True)

        tcls, tbox, indices, anch = criterion.build_targets(outputs, targets)
        for i in range(len(outputs)):
            b, a, gj, gi = indices[i]
            print(f"scale {i}: positives = {b.numel()}", flush=True)
        print("loss items:", loss_items, flush=True)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


        step += 1
        if step >= max_steps:
            break
    print("train_one_step: OK", flush=True)

if __name__ == "__main__":
    main()