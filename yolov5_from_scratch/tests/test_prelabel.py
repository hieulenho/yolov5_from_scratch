from yolov5_from_scratch.tools.dataset.prelabel_traffic import (
    assign_splits,
    suppress_competing_vehicle_labels,
)


def candidate(class_id, confidence, box):
    return {
        "target_class_id": class_id,
        "class_name": str(class_id),
        "confidence": confidence,
        "box": box,
        "yolo_box": (0.5, 0.5, 0.2, 0.2),
    }


def test_split_is_contiguous_per_source():
    rows = [
        {"source": "a.mp4", "image": f"a_{index}.jpg", "frame_index": str(index)}
        for index in range(10)
    ]
    splits = assign_splits(rows, 0.2)
    val_images = [name for name, split in splits.items() if split == "val"]
    assert val_images == ["a_4.jpg", "a_5.jpg"]


def test_competing_vehicle_labels_are_suppressed():
    box = (10.0, 10.0, 100.0, 100.0)
    candidates = [
        candidate(3, 0.80, box),
        candidate(4, 0.70, box),
    ]
    kept = suppress_competing_vehicle_labels(candidates, 0.70)
    assert len(kept) == 1
    assert kept[0]["target_class_id"] == 3


def test_person_and_motorcycle_overlap_is_preserved():
    box = (10.0, 10.0, 100.0, 100.0)
    candidates = [
        candidate(0, 0.90, box),
        candidate(2, 0.80, box),
    ]
    kept = suppress_competing_vehicle_labels(candidates, 0.70)
    assert len(kept) == 2


if __name__ == "__main__":
    test_split_is_contiguous_per_source()
    test_competing_vehicle_labels_are_suppressed()
    test_person_and_motorcycle_overlap_is_preserved()
    print("test_prelabel: OK")
