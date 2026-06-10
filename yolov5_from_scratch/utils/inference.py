from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torchvision.ops import batched_nms

from data.dataset import letterbox
from models.yolo import YOLOv5FromScratch


@dataclass
class Detection:
    xyxy: tuple
    confidence: float
    class_id: int
    class_name: str


def load_class_names(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = cfg["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def resolve_device(device_arg=""):
    if device_arg:
        device = torch.device(device_arg)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def scale_boxes(boxes, ratio, pad, original_shape):
    boxes = boxes.clone()
    ratio_x, ratio_y = ratio
    pad_x, pad_y = pad
    orig_h, orig_w = original_shape

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - float(pad_x)) / float(ratio_x)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - float(pad_y)) / float(ratio_y)
    boxes[:, [0, 2]].clamp_(0, orig_w)
    boxes[:, [1, 3]].clamp_(0, orig_h)
    return boxes


class YOLOv5Predictor:
    def __init__(
        self,
        weights,
        data_yaml,
        img_size=640,
        conf_threshold=0.05,
        iou_threshold=0.45,
        max_det=300,
        device="",
        amp=True,
        class_filter=None,
    ):
        self.weights = Path(weights)
        self.data_yaml = Path(data_yaml)
        self.img_size = int(img_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_det = int(max_det)
        self.device = resolve_device(device)
        self.amp = bool(amp and self.device.type == "cuda")
        self.names = load_class_names(self.data_yaml)
        self.class_filter = None if class_filter is None else set(class_filter)

        if not self.weights.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.weights}")

        self.model = YOLOv5FromScratch(nc=len(self.names)).to(self.device)
        checkpoint = torch.load(
            self.weights,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.checkpoint_epoch = (
            int(checkpoint["epoch"]) + 1
            if isinstance(checkpoint, dict) and "epoch" in checkpoint
            else None
        )

    def preprocess(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image is empty")

        original_shape = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized, ratio, pad = letterbox(
            image_rgb,
            new_shape=(self.img_size, self.img_size),
            auto=False,
            scale_fill=False,
            scaleup=True,
            stride=32,
        )
        tensor = torch.from_numpy(
            np.ascontiguousarray(resized.transpose(2, 0, 1))
        ).float()
        tensor = tensor.div_(255.0).unsqueeze(0).to(self.device)
        return tensor, ratio, pad, original_shape

    def _decode(self, raw_outputs):
        return torch.cat(
            [
                self.model.head.decode_one(output, i).view(
                    output.shape[0], -1, self.model.head.no
                )
                for i, output in enumerate(raw_outputs)
            ],
            dim=1,
        )

    def _postprocess_one(self, prediction, ratio, pad, original_shape):
        class_confidence, class_ids = prediction[:, 5:].max(dim=1)
        scores = prediction[:, 4] * class_confidence
        keep = scores >= self.conf_threshold

        if self.class_filter is not None:
            class_mask = torch.zeros_like(keep)
            for class_id in self.class_filter:
                class_mask |= class_ids == int(class_id)
            keep &= class_mask

        if not keep.any():
            return []

        prediction = prediction[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        xywh = prediction[:, :4]
        boxes = torch.stack(
            (
                xywh[:, 0] - xywh[:, 2] / 2,
                xywh[:, 1] - xywh[:, 3] / 2,
                xywh[:, 0] + xywh[:, 2] / 2,
                xywh[:, 1] + xywh[:, 3] / 2,
            ),
            dim=1,
        )

        selected = batched_nms(
            boxes.float(),
            scores.float(),
            class_ids,
            self.iou_threshold,
        )[: self.max_det]
        boxes = scale_boxes(
            boxes[selected].float(),
            ratio=ratio,
            pad=pad,
            original_shape=original_shape,
        ).cpu()
        scores = scores[selected].float().cpu()
        class_ids = class_ids[selected].cpu()

        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            class_id = int(class_id)
            detections.append(
                Detection(
                    xyxy=tuple(float(v) for v in box.tolist()),
                    confidence=float(score),
                    class_id=class_id,
                    class_name=self.names[class_id],
                )
            )
        return detections

    @torch.inference_mode()
    def predict(self, image_bgr):
        tensor, ratio, pad, original_shape = self.preprocess(image_bgr)
        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.amp,
        ):
            raw_outputs = self.model(tensor)
            decoded = self._decode(raw_outputs)

        return self._postprocess_one(
            decoded[0].float(),
            ratio=ratio,
            pad=pad,
            original_shape=original_shape,
        )
