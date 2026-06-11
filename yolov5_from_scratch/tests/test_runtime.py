import torch

from yolov5_from_scratch.cli.predict import resolve_class_filter
from yolov5_from_scratch.runtime.counting import LineCounter
from yolov5_from_scratch.runtime.detector import Detection, scale_boxes
from yolov5_from_scratch.runtime.geometry import (
    offset_detections,
    output_dimensions,
    roi_pixel_bounds,
    validate_roi,
)
from yolov5_from_scratch.runtime.media import iter_sources, output_name
from yolov5_from_scratch.runtime.rendering import detections_for_display
from yolov5_from_scratch.runtime.tracking import IoUTracker, update_tracks


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
    assert resolve_class_filter(None, names) is None


def test_roi_and_output_geometry():
    roi = validate_roi((0.1, 0.2, 0.9, 1.0))
    assert roi_pixel_bounds((100, 200, 3), roi) == (20, 20, 180, 100)
    detections = [
        Detection(
            xyxy=(1.0, 2.0, 11.0, 12.0),
            confidence=0.8,
            class_id=2,
            class_name="car",
        )
    ]
    shifted = offset_detections(detections, 20, 30)
    assert shifted[0].xyxy == (21.0, 32.0, 31.0, 42.0)
    assert output_dimensions(3840, 2160, 1280) == (1280, 720)
    assert output_dimensions(1080, 1920, 1280) == (1080, 1920)


def test_tracking_threshold_and_center_fallback():
    tracker = IoUTracker(
        iou_threshold=0.3,
        max_age=2,
        center_distance_threshold=1.0,
    )
    low = Detection(
        xyxy=(0.0, 0.0, 20.0, 20.0),
        confidence=0.08,
        class_id=2,
        class_name="car",
    )
    strong = Detection(
        xyxy=(30.0, 0.0, 50.0, 20.0),
        confidence=0.8,
        class_id=2,
        class_name="car",
    )
    assignments = update_tracks(tracker, [low, strong], min_confidence=0.1)
    assert assignments[0] is None
    assert assignments[1] == 1

    moved = Detection(
        xyxy=(45.0, 0.0, 65.0, 20.0),
        confidence=0.7,
        class_id=2,
        class_name="car",
    )
    moved_assignments = update_tracks(tracker, [moved], min_confidence=0.1)
    assert moved_assignments == [1]
    assert tracker.tracks[1].hits == 2
    assert abs(tracker.tracks[1].average_confidence - 0.75) < 1e-6


def test_display_filter():
    detections = [
        Detection((0, 0, 10, 10), 0.08, 0, "person"),
        Detection((10, 10, 20, 20), 0.4, 2, "car"),
    ]
    visible, track_ids = detections_for_display(
        detections,
        [None, 3],
        min_confidence=0.1,
    )
    assert len(visible) == 1
    assert visible[0].class_name == "car"
    assert track_ids == [3]


def test_stream_source():
    source = "rtsp://camera.local/stream"
    assert iter_sources(source) == [source]
    assert output_name(source, ".mp4") == "stream.mp4"


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
    test_roi_and_output_geometry()
    test_tracking_threshold_and_center_fallback()
    test_display_filter()
    test_stream_source()
    test_tracker_and_counter()
    print("test_runtime: OK")


if __name__ == "__main__":
    main()
