import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference import scale_boxes
from utils.predict import Detection, IoUTracker, LineCounter, resolve_class_filter


def test_scale_boxes():
    boxes = torch.tensor([[100.0, 160.0, 300.0, 360.0]])
    scaled = scale_boxes(
        boxes,
        ratio=(2.0, 2.0),
        pad=(20.0, 40.0),
        original_shape=(200, 200),
    )
    expected = torch.tensor([[40.0, 60.0, 140.0, 160.0]])
    assert torch.allclose(scaled, expected)


def test_class_filter():
    names = ["person", "car", "dog"]
    assert resolve_class_filter(["car", "0"], names) == {0, 1}
    assert resolve_class_filter(None, names, all_classes=True) is None


def test_tracker_and_counter():
    tracker = IoUTracker(iou_threshold=0.1, max_age=2)
    counter = LineCounter(
        orientation="horizontal",
        position=0.5,
        margin=3.0,
        min_track_hits=2,
    )

    first = [
        Detection(
            xyxy=(40.0, 35.0, 60.0, 55.0),
            confidence=0.9,
            class_id=2,
            class_name="car",
        )
    ]
    second = [
        Detection(
            xyxy=(40.0, 39.0, 60.0, 59.0),
            confidence=0.9,
            class_id=2,
            class_name="car",
        )
    ]
    third = [
        Detection(
            xyxy=(40.0, 45.0, 60.0, 65.0),
            confidence=0.9,
            class_id=2,
            class_name="car",
        )
    ]

    first_ids = tracker.update(first)
    first_track = tracker.tracks[first_ids[0]]
    assert counter.update(first_track, (100, 100, 3), "car") is None

    second_ids = tracker.update(second)
    assert first_ids == second_ids
    track = tracker.tracks[second_ids[0]]
    assert counter.update(track, (100, 100, 3), "car") is None

    third_ids = tracker.update(third)
    assert second_ids == third_ids
    track = tracker.tracks[third_ids[0]]
    direction = counter.update(track, (100, 100, 3), "car")
    assert direction == "top_to_bottom"
    assert counter.counts[("car", "top_to_bottom")] == 1


def main():
    test_scale_boxes()
    test_class_filter()
    test_tracker_and_counter()
    print("test_inference: OK")


if __name__ == "__main__":
    main()
