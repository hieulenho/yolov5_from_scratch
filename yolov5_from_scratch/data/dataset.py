import os
import cv2
import yaml
import math
import time
import random
import pickle
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

cv2.setNumThreads(0)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_root(data_yaml, cfg_path_value):
    """
    Resolve dataset root robustly for:
    - path: datasets/coco2017
    - path: .
    - absolute paths
    """
    yaml_path = Path(data_yaml).resolve()
    yaml_dir = yaml_path.parent
    root = Path(str(cfg_path_value))

    if root.is_absolute():
        return root

    candidates = [
        (Path.cwd() / root).resolve(),
        (yaml_dir / root).resolve(),
        yaml_dir.resolve(),
    ]

    for c in candidates:
        if (c / "images").exists():
            return c

    return candidates[0]


def img2label_path(im_file: Path, data_root: Path) -> Path:
    rel = im_file.relative_to(data_root)
    parts = list(rel.parts)
    if parts[0] != "images":
        raise ValueError(f"Expected image path under 'images/', got: {im_file}")
    parts[0] = "labels"
    return data_root.joinpath(*parts).with_suffix(".txt")


def parse_yolo_label_file(label_path: Path, num_classes=None, single_cls=False):
    """
    Return np.ndarray [N, 5] with [cls, x, y, w, h], normalized
    """
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])

        if num_classes is not None and not (0 <= cls_id < num_classes):
            continue

        if single_cls:
            cls_id = 0

        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))
        w = float(np.clip(w, 0.0, 1.0))
        h = float(np.clip(h, 0.0, 1.0))

        if w <= 0.0 or h <= 0.0:
            continue

        rows.append([cls_id, x, y, w, h])

    if len(rows) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)


def yolo_xywhn_to_xyxy(labels, img_w, img_h):
    """
    labels: [N, 5] -> [cls, x1, y1, x2, y2] in pixels
    """
    if labels.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    out = labels.copy()
    x = labels[:, 1] * img_w
    y = labels[:, 2] * img_h
    w = labels[:, 3] * img_w
    h = labels[:, 4] * img_h

    out[:, 1] = x - w / 2
    out[:, 2] = y - h / 2
    out[:, 3] = x + w / 2
    out[:, 4] = y + h / 2
    return out


def xyxy_to_yolo_xywhn(labels, img_w, img_h, eps=1e-6):
    """
    labels: [N, 5] -> [cls, x, y, w, h] normalized
    """
    if labels.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    out = labels.copy()
    x1 = labels[:, 1]
    y1 = labels[:, 2]
    x2 = labels[:, 3]
    y2 = labels[:, 4]

    xc = ((x1 + x2) / 2) / max(img_w, eps)
    yc = ((y1 + y2) / 2) / max(img_h, eps)
    bw = (x2 - x1) / max(img_w, eps)
    bh = (y2 - y1) / max(img_h, eps)

    out[:, 1] = xc
    out[:, 2] = yc
    out[:, 3] = bw
    out[:, 4] = bh
    return out


def clip_boxes_xyxy(labels, img_w, img_h):
    if labels.size == 0:
        return labels

    labels[:, 1] = labels[:, 1].clip(0, img_w)
    labels[:, 2] = labels[:, 2].clip(0, img_h)
    labels[:, 3] = labels[:, 3].clip(0, img_w)
    labels[:, 4] = labels[:, 4].clip(0, img_h)
    return labels


def filter_invalid_boxes_xyxy(labels, min_size=2.0):
    if labels.size == 0:
        return labels

    w = labels[:, 3] - labels[:, 1]
    h = labels[:, 4] - labels[:, 2]
    keep = (w >= min_size) & (h >= min_size)
    return labels[keep]


def letterbox(
    image,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scale_fill=False,
    scaleup=True,
    stride=32,
):
    """
    Resize and pad image while keeping aspect ratio.

    Returns:
        image_out
        ratio: (rw, rh)
        pad: (dw, dh)  # half-padding
    """
    shape = image.shape[:2]  # (h, w)

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw = dw % stride
        dh = dh % stride
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    return image, ratio, (dw, dh)


def augment_hsv(image, hgain=0.015, sgain=0.7, vgain=0.4):
    if hgain == 0 and sgain == 0 and vgain == 0:
        return image

    r = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1.0

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] * r[0]) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)

    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return image


class YOLODataset(Dataset):
    """
    Production-friendly baseline dataset for YOLO-style training.

    Output per sample:
        img_tensor:  [3, H, W], float32 in [0,1]
        targets:     [N, 5], float32, [cls, x, y, w, h] normalized
        meta:        dict
    """

    def __init__(
        self,
        data_yaml,
        split="train",
        img_size=640,
        stride=32,
        augment=False,
        hyp=None,
        rect=False,
        cache_labels=True,
        cache_images=False,
        single_cls=False,
        verbose=True,
    ):
        super().__init__()

        self.data_yaml = str(data_yaml)
        self.cfg = load_yaml(data_yaml)
        self.data_root = resolve_data_root(data_yaml, self.cfg["path"])
        self.split = split
        self.img_size = img_size
        self.stride = stride
        self.augment = augment
        self.rect = rect
        self.cache_labels = cache_labels
        self.cache_images = cache_images
        self.single_cls = single_cls
        self.verbose = verbose

        self.hyp = hyp or {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "fliplr": 0.5,
        }

        if split not in self.cfg:
            raise ValueError(f"Split '{split}' not found in {data_yaml}")

        names = self.cfg["names"]
        if isinstance(names, dict):
            self.names = {int(k): v for k, v in names.items()}
        else:
            self.names = {i: n for i, n in enumerate(names)}

        self.num_classes = 1 if single_cls else len(self.names)

        self.image_dir = self.data_root / self.cfg[split]
        self.im_files = self._scan_images(self.image_dir)

        if len(self.im_files) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

        self.cache_path = self.data_root / f".{split}_labels.cache.pkl"
        self.samples = self._load_or_build_cache() if cache_labels else self._build_cache(save=False)

        # optional image cache
        self.imgs = [None] * len(self.samples) if cache_images else None

        if self.verbose:
            print(f"[YOLODataset] split={split}")
            print(f"[YOLODataset] data_root={self.data_root}")
            print(f"[YOLODataset] images={len(self.samples)}")
            print(f"[YOLODataset] classes={self.num_classes}")
            print(f"[YOLODataset] cache_labels={self.cache_labels}")
            print(f"[YOLODataset] cache_images={self.cache_images}")
            print(f"[YOLODataset] augment={self.augment}")

    def _scan_images(self, image_dir: Path):
        files = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        return sorted(files)

    def _build_cache(self, save=True):
        t0 = time.time()
        samples = []
        bad_images = 0
        total_labels = 0

        for im_file in self.im_files:
            img = cv2.imread(str(im_file))
            if img is None:
                bad_images += 1
                continue

            h, w = img.shape[:2]
            label_file = img2label_path(im_file, self.data_root)
            labels = parse_yolo_label_file(
                label_file,
                num_classes=None if self.single_cls else len(self.names),
                single_cls=self.single_cls,
            )

            total_labels += len(labels)

            samples.append(
                {
                    "im_file": str(im_file),
                    "label_file": str(label_file),
                    "shape": (h, w),
                    "labels": labels,
                }
            )

        if save:
            payload = {
                "version": 1,
                "data_root": str(self.data_root),
                "split": self.split,
                "img_size": self.img_size,
                "samples": samples,
            }
            with open(self.cache_path, "wb") as f:
                pickle.dump(payload, f)

        if self.verbose:
            dt = time.time() - t0
            print(f"[cache] built in {dt:.2f}s")
            print(f"[cache] usable images={len(samples)}")
            print(f"[cache] bad images={bad_images}")
            print(f"[cache] total labels={total_labels}")

        return samples

    def _load_or_build_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    payload = pickle.load(f)

                if (
                    payload.get("version") == 1
                    and payload.get("data_root") == str(self.data_root)
                    and payload.get("split") == self.split
                ):
                    if self.verbose:
                        print(f"[cache] loaded {self.cache_path}")
                    return payload["samples"]
            except Exception:
                pass

        return self._build_cache(save=True)

    def __len__(self):
        return len(self.samples)

    def _load_image(self, index):
        if self.cache_images and self.imgs[index] is not None:
            return self.imgs[index].copy()

        im_file = self.samples[index]["im_file"]
        img = cv2.imread(im_file)
        if img is None:
            raise RuntimeError(f"Failed to read image: {im_file}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.cache_images:
            self.imgs[index] = img.copy()

        return img

    def _load_labels(self, index):
        return self.samples[index]["labels"].copy()

    def __getitem__(self, index):
        sample = self.samples[index]

        # 1) image
        img = self._load_image(index)
        h0, w0 = img.shape[:2]

        # 2) labels: [N,5] normalized on original image
        labels = self._load_labels(index)

        # 3) convert to xyxy pixels on original image
        labels_xyxy = yolo_xywhn_to_xyxy(labels, w0, h0)

        # 4) letterbox
        img, ratio, (dw, dh) = letterbox(
            img,
            new_shape=(self.img_size, self.img_size),
            auto=False,
            scale_fill=False,
            scaleup=True,
            stride=self.stride,
        )
        h, w = img.shape[:2]

        # 5) move boxes with same resize + pad
        if labels_xyxy.size > 0:
            labels_xyxy[:, [1, 3]] = labels_xyxy[:, [1, 3]] * ratio[0] + dw
            labels_xyxy[:, [2, 4]] = labels_xyxy[:, [2, 4]] * ratio[1] + dh
            labels_xyxy = clip_boxes_xyxy(labels_xyxy, w, h)
            labels_xyxy = filter_invalid_boxes_xyxy(labels_xyxy, min_size=2.0)

        # 6) lightweight augment
        if self.augment:
            img = augment_hsv(
                img,
                hgain=self.hyp.get("hsv_h", 0.015),
                sgain=self.hyp.get("hsv_s", 0.7),
                vgain=self.hyp.get("hsv_v", 0.4),
            )

            if labels_xyxy.size > 0 and random.random() < self.hyp.get("fliplr", 0.5):
                img = np.fliplr(img)
                x1 = labels_xyxy[:, 1].copy()
                x2 = labels_xyxy[:, 3].copy()
                labels_xyxy[:, 1] = w - x2
                labels_xyxy[:, 3] = w - x1

        # 7) back to normalized xywh
        targets = xyxy_to_yolo_xywhn(labels_xyxy, w, h)

        # 8) final clamp
        if targets.size > 0:
            targets[:, 1:] = np.clip(targets[:, 1:], 0.0, 1.0)

        # 9) image -> tensor
        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # HWC -> CHW
        img_tensor = torch.from_numpy(img).float().div_(255.0)

        # 10) targets -> tensor
        targets_tensor = torch.from_numpy(targets).float()

        meta = {
            "im_file": sample["im_file"],
            "orig_shape": (h0, w0),
            "resized_shape": (h, w),
            "ratio": ratio,
            "pad": (dw, dh),
        }

        return img_tensor, targets_tensor, meta


def yolo_collate_fn(batch):
    """
    Batch output:
        imgs:    [B, 3, H, W]
        targets: [M, 6] = [batch_idx, cls, x, y, w, h]
        metas:   tuple(dict)
    """
    imgs, targets, metas = zip(*batch)
    imgs = torch.stack(imgs, dim=0)

    new_targets = []
    for i, t in enumerate(targets):
        if t.numel() == 0:
            continue
        batch_idx = torch.full((t.shape[0], 1), i, dtype=t.dtype)
        t = torch.cat([batch_idx, t], dim=1)
        new_targets.append(t)

    if len(new_targets) > 0:
        new_targets = torch.cat(new_targets, dim=0)
    else:
        new_targets = torch.zeros((0, 6), dtype=torch.float32)

    return imgs, new_targets, metas


def build_dataloader(
    data_yaml,
    split="train",
    img_size=640,
    batch_size=16,
    num_workers=4,
    stride=32,
    augment=None,
    hyp=None,
    cache_labels=True,
    cache_images=False,
    single_cls=False,
    shuffle=None,
    pin_memory=True,
    persistent_workers=True,
    drop_last=False,
    verbose=True,
):
    if augment is None:
        augment = split == "train"

    if shuffle is None:
        shuffle = split == "train"

    dataset = YOLODataset(
        data_yaml=data_yaml,
        split=split,
        img_size=img_size,
        stride=stride,
        augment=augment,
        hyp=hyp,
        cache_labels=cache_labels,
        cache_images=cache_images,
        single_cls=single_cls,
        verbose=verbose,
    )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=yolo_collate_fn,
        drop_last=drop_last,
        persistent_workers=(persistent_workers and num_workers > 0),
    )

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(**loader_kwargs)
    return dataset, loader