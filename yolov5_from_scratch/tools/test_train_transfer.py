import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.train import remap_detect_parameter


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


if __name__ == "__main__":
    test_detect_head_class_remap()
    print("test_train_transfer: OK")
