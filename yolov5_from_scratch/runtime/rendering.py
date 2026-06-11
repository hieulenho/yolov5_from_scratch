from collections import Counter

import cv2

from .geometry import roi_pixel_bounds


def color_for_class(class_id):
    palette = (
        (255, 56, 56),
        (255, 157, 151),
        (255, 112, 31),
        (255, 178, 29),
        (207, 210, 49),
        (72, 249, 10),
        (146, 204, 23),
        (61, 219, 134),
        (26, 147, 52),
        (0, 212, 187),
        (44, 153, 168),
        (0, 194, 255),
    )
    rgb = palette[class_id % len(palette)]
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def draw_line_and_counts(image, line_counter):
    if line_counter.orientation == "none":
        return

    height, width = image.shape[:2]
    if line_counter.orientation == "horizontal":
        y = int(height * line_counter.position)
        cv2.line(image, (0, y), (width - 1, y), (0, 255, 255), 2)
    else:
        x = int(width * line_counter.position)
        cv2.line(image, (x, 0), (x, height - 1), (0, 255, 255), 2)

    summary = Counter()
    for (class_name, _), value in line_counter.counts.items():
        summary[class_name] += value

    y_text = 28
    for class_name, value in sorted(summary.items()):
        cv2.putText(
            image,
            f"{class_name}: {value}",
            (10, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_text += 26


def draw_roi(image, roi):
    if roi is None:
        return
    left, top, right, bottom = roi_pixel_bounds(image.shape, roi)
    cv2.rectangle(
        image,
        (left, top),
        (right - 1, bottom - 1),
        (255, 255, 0),
        2,
    )


def draw_performance(image, inference_ms):
    fps = 1000.0 / max(inference_ms, 1e-6)
    text = f"{inference_ms:.1f} ms | {fps:.1f} FPS"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    x = max(10, image.shape[1] - text_size[0] - 12)
    cv2.putText(
        image,
        text,
        (x, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_detections(
    image,
    detections,
    track_ids,
    hide_labels=False,
    hide_conf=False,
):
    for detection, track_id in zip(detections, track_ids):
        x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy)
        color = color_for_class(detection.class_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        if hide_labels:
            continue
        label = detection.class_name
        if track_id is not None:
            label += f" #{track_id}"
        if not hide_conf:
            label += f" {detection.confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def detections_for_display(detections, track_ids, min_confidence):
    selected = [
        (detection, track_id)
        for detection, track_id in zip(detections, track_ids)
        if detection.confidence >= min_confidence
    ]
    return (
        [item[0] for item in selected],
        [item[1] for item in selected],
    )
