import cv2
import yaml
import time
import random
import pickle
import hashlib
import re
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
    yaml_path = Path(data_yaml).resolve()
    yaml_dir = yaml_path.parent
    root = Path(str(cfg_path_value or "."))

    if root.is_absolute():
        return root.resolve()

    candidates = [
        (yaml_dir / root).resolve(),
        yaml_dir.resolve(),
        (Path.cwd() / root).resolve(),
        Path.cwd().resolve(),
    ]

    for c in candidates:
        if (c / "images").exists() or (c / "annotations").exists() or (c / "labels").exists():
            return c

    return candidates[0]


def _split_label_line(line: str):
    line = line.strip().replace("\ufeff", "")
    if not line:
        return []
    return [p for p in re.split(r"[\s,]+", line) if p]


def parse_yolo_label_file(label_path: Path, num_classes=None, single_cls=False, return_debug=False):
    """
    Return np.ndarray [N, 5] with [cls, x, y, w, h], normalized.
    Accept both space-separated and comma-separated rows.
    """
    debug = {
        "exists": label_path.exists(),
        "rows_total": 0,
        "rows_kept": 0,
        "rows_bad_cols": 0,
        "rows_bad_values": 0,
        "rows_bad_class": 0,
        "rows_bad_box": 0,
        "examples": [],
    }

    if not label_path.exists():
        out = np.zeros((0, 5), dtype=np.float32)
        return (out, debug) if return_debug else out

    rows = []
    with open(label_path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = [x.rstrip("\n\r") for x in f.readlines() if x.strip()]

    for line in lines:
        debug["rows_total"] += 1
        parts = _split_label_line(line)
        if len(parts) != 5:
            debug["rows_bad_cols"] += 1
            if len(debug["examples"]) < 5:
                debug["examples"].append(f"bad_cols: {line}")
            continue

        try:
            cls_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except Exception:
            debug["rows_bad_values"] += 1
            if len(debug["examples"]) < 5:
                debug["examples"].append(f"bad_values: {line}")
            continue

        if num_classes is not None and not (0 <= cls_id < num_classes):
            debug["rows_bad_class"] += 1
            if len(debug["examples"]) < 5:
                debug["examples"].append(f"bad_class: {line}")
            continue

        if single_cls:
            cls_id = 0

        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))
        w = float(np.clip(w, 0.0, 1.0))
        h = float(np.clip(h, 0.0, 1.0))

        if w <= 0.0 or h <= 0.0:
            debug["rows_bad_box"] += 1
            if len(debug["examples"]) < 5:
                debug["examples"].append(f"bad_box: {line}")
            continue

        rows.append([cls_id, x, y, w, h])
        debug["rows_kept"] += 1

    if len(rows) == 0:
        out = np.zeros((0, 5), dtype=np.float32)
        return (out, debug) if return_debug else out

    out = np.asarray(rows, dtype=np.float32)
    return (out, debug) if return_debug else out


def yolo_xywhn_to_xyxy(labels, img_w, img_h):
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
    shape = image.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

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
    CACHE_VERSION = 4

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
        rebuild_cache=False,
        min_box_size=2.0,
    ):
        super().__init__()
        self.data_yaml = str(data_yaml)
        self.cfg = load_yaml(data_yaml)
        self.data_root = resolve_data_root(data_yaml, self.cfg.get("path", "."))
        self.split = split
        self.img_size = img_size
        self.stride = stride
        self.augment = augment
        self.rect = rect
        self.cache_labels = cache_labels
        self.cache_images = cache_images
        self.single_cls = single_cls
        self.verbose = verbose
        self.rebuild_cache = rebuild_cache
        self.min_box_size = float(min_box_size)
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
        self.image_dir = (self.data_root / self.cfg[split]).resolve()
        self.im_files = self._scan_images(self.image_dir)
        if len(self.im_files) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

        self.label_dir = self._guess_label_dir()
        self.label_index = self._build_label_index(self.label_dir)

        self.cache_path = self.data_root / f".{split}_labels.cache.pkl"
        self.cache_fingerprint = self._compute_cache_fingerprint(self.im_files)

        if cache_labels:
            self.samples = self._load_or_build_cache()
        else:
            self.samples = self._build_cache(save=False)

        self.imgs = [None] * len(self.samples) if cache_images else None

        if self.verbose:
            print(f"[YOLODataset] split={split}")
            print(f"[YOLODataset] data_root={self.data_root}")
            print(f"[YOLODataset] image_dir={self.image_dir}")
            print(f"[YOLODataset] label_dir={self.label_dir} | exists={self.label_dir.exists()}")
            print(f"[YOLODataset] indexed_label_files={len(self.label_index)}")
            print(f"[YOLODataset] images={len(self.samples)}")
            print(f"[YOLODataset] classes={self.num_classes}")
            print(f"[YOLODataset] cache_labels={self.cache_labels}")
            print(f"[YOLODataset] cache_images={self.cache_images}")
            print(f"[YOLODataset] rebuild_cache={self.rebuild_cache}")
            print(f"[YOLODataset] min_box_size={self.min_box_size}")
            print(f"[YOLODataset] augment={self.augment}")

    def _scan_images(self, image_dir: Path):
        files = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        return sorted(files)

    def _guess_label_dir(self):
        split_rel = str(self.cfg[self.split]).replace("\\", "/")
        candidate_str = split_rel.replace("images/", "labels/")
        candidates = [
            (self.data_root / candidate_str).resolve(),
            (self.data_root / "labels" / Path(split_rel).name).resolve(),
            (self.image_dir.parent.parent / "labels" / self.image_dir.name).resolve(),
            (self.data_root / "labels").resolve(),
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    def _build_label_index(self, label_dir: Path):
        index = {}
        if not label_dir.exists():
            return index
        for p in label_dir.rglob("*.txt"):
            index.setdefault(p.stem, []).append(p)
        return index

    def _resolve_label_file(self, im_file: Path) -> Path:
        # Standard case: data_root/images/... -> data_root/labels/.../same_name.txt
        try:
            rel = im_file.relative_to(self.data_root)
            parts = list(rel.parts)
            if parts and parts[0] == "images":
                parts[0] = "labels"
                candidate = self.data_root.joinpath(*parts).with_suffix(".txt")
                if candidate.exists():
                    return candidate
        except Exception:
            pass

        # Sibling replacement: .../images/<split>/<name>.jpg -> .../labels/<split>/<name>.txt
        candidate = (self.image_dir.parent.parent / "labels" / self.image_dir.name / im_file.with_suffix(".txt").name).resolve()
        if candidate.exists():
            return candidate

        # Indexed fallback by stem
        hits = self.label_index.get(im_file.stem, [])
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            for h in hits:
                if h.parent.name == self.image_dir.name:
                    return h
            return hits[0]

        # Default expected path even if missing
        return (self.label_dir / im_file.with_suffix(".txt").name).resolve()

    def _stat_token(self, path: Path) -> str:
        if not path.exists():
            return "missing"
        st = path.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"

    def _compute_cache_fingerprint(self, im_files):
        h = hashlib.sha256()
        h.update(str(self.data_root.resolve()).encode("utf-8"))
        h.update(str(Path(self.data_yaml).resolve()).encode("utf-8"))
        h.update(str(self.num_classes).encode("utf-8"))
        h.update(str(self.single_cls).encode("utf-8"))
        h.update(str(self.min_box_size).encode("utf-8"))

        for im_file in im_files:
            label_file = self._resolve_label_file(im_file)
            h.update(str(im_file.resolve()).encode("utf-8"))
            h.update(self._stat_token(im_file).encode("utf-8"))
            h.update(str(label_file.resolve()).encode("utf-8"))
            h.update(self._stat_token(label_file).encode("utf-8"))

        return h.hexdigest()

    def _build_cache(self, save=True):
        t0 = time.time()
        samples = []
        bad_images = 0
        total_labels = 0
        total_empty = 0
        existing_label_files = 0
        parser_debug_examples = []
        missing_label_examples = []

        for im_file in self.im_files:
            img = cv2.imread(str(im_file))
            if img is None:
                bad_images += 1
                continue

            h, w = img.shape[:2]
            label_file = self._resolve_label_file(im_file)
            if label_file.exists():
                existing_label_files += 1
            else:
                if len(missing_label_examples) < 5:
                    missing_label_examples.append(str(label_file))

            labels, debug = parse_yolo_label_file(
                label_file,
                num_classes=None if self.single_cls else len(self.names),
                single_cls=self.single_cls,
                return_debug=True,
            )
            if len(labels) == 0:
                total_empty += 1
                if debug["exists"] and debug["rows_total"] > 0 and len(parser_debug_examples) < 5:
                    parser_debug_examples.append({
                        "label_file": str(label_file),
                        "debug": debug,
                    })
            total_labels += len(labels)
            samples.append(
                {
                    "im_file": str(im_file),
                    "label_file": str(label_file),
                    "shape": (h, w),
                    "labels": labels.astype(np.float32),
                }
            )

        if save:
            payload = {
                "version": self.CACHE_VERSION,
                "data_root": str(self.data_root),
                "split": self.split,
                "img_size": self.img_size,
                "num_classes": self.num_classes,
                "single_cls": self.single_cls,
                "min_box_size": self.min_box_size,
                "image_dir": str(self.image_dir),
                "label_dir": str(self.label_dir),
                "fingerprint": self.cache_fingerprint,
                "samples": samples,
            }
            with open(self.cache_path, "wb") as f:
                pickle.dump(payload, f)

        if self.verbose:
            dt = time.time() - t0
            print(f"[cache] built in {dt:.2f}s")
            print(f"[cache] usable images={len(samples)}")
            print(f"[cache] bad images={bad_images}")
            print(f"[cache] existing label files={existing_label_files}")
            print(f"[cache] images with 0 labels={total_empty}")
            print(f"[cache] total labels={total_labels}")
            if missing_label_examples:
                print("[cache] missing label examples:")
                for p in missing_label_examples:
                    print(f"  - {p}")
            if parser_debug_examples:
                print("[cache] parser rejected label examples:")
                for item in parser_debug_examples:
                    dbg = item["debug"]
                    print(f"  - {item['label_file']}")
                    print(
                        f"    rows_total={dbg['rows_total']} kept={dbg['rows_kept']} bad_cols={dbg['rows_bad_cols']} "
                        f"bad_values={dbg['rows_bad_values']} bad_class={dbg['rows_bad_class']} bad_box={dbg['rows_bad_box']}"
                    )
                    for ex in dbg["examples"]:
                        print(f"    ex: {ex}")

        return samples

    def _load_or_build_cache(self):
        if self.rebuild_cache:
            if self.verbose:
                print("[cache] rebuild requested -> ignore existing cache")
            return self._build_cache(save=True)

        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    payload = pickle.load(f)

                cache_ok = (
                    payload.get("version") == self.CACHE_VERSION
                    and payload.get("data_root") == str(self.data_root)
                    and payload.get("split") == self.split
                    and payload.get("num_classes") == self.num_classes
                    and payload.get("single_cls") == self.single_cls
                    and float(payload.get("min_box_size", -1.0)) == self.min_box_size
                    and payload.get("image_dir") == str(self.image_dir)
                    and payload.get("label_dir") == str(self.label_dir)
                    and payload.get("fingerprint") == self.cache_fingerprint
                )

                if cache_ok:
                    if self.verbose:
                        print(f"[cache] loaded {self.cache_path}")
                    return payload["samples"]

                if self.verbose:
                    print("[cache] stale cache detected -> rebuilding")
            except Exception as e:
                if self.verbose:
                    print(f"[cache] failed to read cache ({e}) -> rebuilding")

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
        img = self._load_image(index)
        h0, w0 = img.shape[:2]
        labels = self._load_labels(index)

        labels_xyxy = yolo_xywhn_to_xyxy(labels, w0, h0)

        img, ratio, (dw, dh) = letterbox(
            img,
            new_shape=(self.img_size, self.img_size),
            auto=False,
            scale_fill=False,
            scaleup=True,
            stride=self.stride,
        )
        h, w = img.shape[:2]

        if labels_xyxy.size > 0:
            labels_xyxy[:, [1, 3]] = labels_xyxy[:, [1, 3]] * ratio[0] + dw
            labels_xyxy[:, [2, 4]] = labels_xyxy[:, [2, 4]] * ratio[1] + dh
            labels_xyxy = clip_boxes_xyxy(labels_xyxy, w, h)
            labels_xyxy = filter_invalid_boxes_xyxy(labels_xyxy, min_size=self.min_box_size)

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

        targets = xyxy_to_yolo_xywhn(labels_xyxy, w, h)
        if targets.size > 0:
            targets[:, 1:] = np.clip(targets[:, 1:], 0.0, 1.0)

        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        img_tensor = torch.from_numpy(img).float().div_(255.0)

        targets = np.ascontiguousarray(targets, dtype=np.float32)
        targets_tensor = torch.from_numpy(targets)

        meta = {
            "im_file": sample["im_file"],
            "label_file": sample["label_file"],
            "orig_shape": (h0, w0),
            "resized_shape": (h, w),
            "ratio": ratio,
            "pad": (dw, dh),
            "num_classes": self.num_classes,
        }
        return img_tensor, targets_tensor, meta


def yolo_collate_fn(batch):
    imgs, targets, metas = zip(*batch)
    imgs = torch.stack(imgs, dim=0)

    new_targets = []
    for i, t in enumerate(targets):
        if t.numel() == 0:
            continue
        batch_idx = torch.full((t.shape[0], 1), float(i), dtype=torch.float32)
        t = t.to(torch.float32)
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
    cache_labels=False,
    cache_images=False,
    single_cls=False,
    shuffle=None,
    pin_memory=True,
    persistent_workers=True,
    drop_last=False,
    verbose=True,
    rebuild_cache=False,
    min_box_size=2.0,
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
        rebuild_cache=rebuild_cache,
        min_box_size=min_box_size,
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
