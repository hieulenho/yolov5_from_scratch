import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from yolov5_from_scratch.training.train import (
    load_resume_history,
    remap_detect_parameter,
)


def test_detect_head_class_remap():
    source_names = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
    ]
    target_names = ["person", "car", "motorcycle", "bus", "truck"]
    source_no = len(source_names) + 5
    target_no = len(target_names) + 5
    source = torch.arange(3 * source_no, dtype=torch.float32)
    target = torch.full((3 * target_no,), -1.0)

    remapped = remap_detect_parameter(
        source,
        target,
        source_names,
        target_names,
    ).view(3, target_no)
    source = source.view(3, source_no)

    assert torch.equal(remapped[:, :5], source[:, :5])
    source_class_ids = [0, 2, 3, 5, 7]
    for target_id, source_id in enumerate(source_class_ids):
        assert torch.equal(
            remapped[:, 5 + target_id],
            source[:, 5 + source_id],
        )


def test_load_flattened_resume_history():
    with TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoint"
        checkpoint_dir.mkdir()
        history = [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}]
        (checkpoint_dir / "history.json").write_text(
            json.dumps(history),
            encoding="utf-8",
        )
        restored = load_resume_history(
            checkpoint_dir / "last.pt",
            Path(temp_dir) / "new_run",
            completed_epochs=2,
        )
        assert restored == history[:2]


if __name__ == "__main__":
    test_detect_head_class_remap()
    test_load_flattened_resume_history()
    print("test_training_transfer: OK")
