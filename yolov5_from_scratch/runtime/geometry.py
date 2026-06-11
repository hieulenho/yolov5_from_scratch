import cv2
import numpy as np

from .detector import Detection


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    return intersection / max(area1 + area2 - intersection, 1e-9)


def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def normalized_center_distance(box1, box2):
    center1 = box_center(box1)
    center2 = box_center(box2)
    distance = float(np.hypot(center1[0] - center2[0], center1[1] - center2[1]))
    diagonal1 = float(np.hypot(box1[2] - box1[0], box1[3] - box1[1]))
    diagonal2 = float(np.hypot(box2[2] - box2[0], box2[3] - box2[1]))
    return distance / max(diagonal1, diagonal2, 1e-9)


def validate_roi(roi):
    if roi is None:
        return None
    x1, y1, x2, y2 = (float(value) for value in roi)
    if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        raise ValueError("--roi values must be between 0 and 1")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("--roi requires X2 > X1 and Y2 > Y1")
    return x1, y1, x2, y2


def roi_pixel_bounds(frame_shape, roi):
    height, width = frame_shape[:2]
    if roi is None:
        return 0, 0, width, height
    x1, y1, x2, y2 = roi
    left = max(0, min(width - 1, int(round(x1 * width))))
    top = max(0, min(height - 1, int(round(y1 * height))))
    right = max(left + 1, min(width, int(round(x2 * width))))
    bottom = max(top + 1, min(height, int(round(y2 * height))))
    return left, top, right, bottom


def offset_detections(detections, offset_x, offset_y):
    return [
        Detection(
            xyxy=(
                detection.xyxy[0] + offset_x,
                detection.xyxy[1] + offset_y,
                detection.xyxy[2] + offset_x,
                detection.xyxy[3] + offset_y,
            ),
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
        )
        for detection in detections
    ]


def predict_frame(predictor, frame, roi):
    left, top, right, bottom = roi_pixel_bounds(frame.shape, roi)
    crop = frame[top:bottom, left:right]
    detections = predictor.predict(crop)
    return offset_detections(detections, left, top)


def output_dimensions(width, height, output_width):
    if output_width <= 0 or width <= output_width:
        return width, height
    output_height = max(2, int(round(height * output_width / width)))
    if output_height % 2:
        output_height += 1
    return int(output_width), output_height


def resize_for_output(image, output_width):
    target_width, target_height = output_dimensions(
        image.shape[1],
        image.shape[0],
        output_width,
    )
    if (target_width, target_height) == (image.shape[1], image.shape[0]):
        return image.copy(), 1.0, 1.0
    resized = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized, target_width / image.shape[1], target_height / image.shape[0]


def scale_detections(detections, scale_x, scale_y):
    return [
        Detection(
            xyxy=(
                detection.xyxy[0] * scale_x,
                detection.xyxy[1] * scale_y,
                detection.xyxy[2] * scale_x,
                detection.xyxy[3] * scale_y,
            ),
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
        )
        for detection in detections
    ]
