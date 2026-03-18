import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbone import YOLOBackbone
from models.neck import YOLOPAN
from models.head import DetectHead
from models.yolo import YOLOv5FromScratch


def test_backbone():
    print("\n=== test_backbone ===")
    model = YOLOBackbone()
    x = torch.randn(2, 3, 640, 640)
    p3, p4, p5 = model(x)

    print("p3:", p3.shape)
    print("p4:", p4.shape)
    print("p5:", p5.shape)

    assert p3.shape == (2, 128, 80, 80)
    assert p4.shape == (2, 256, 40, 40)
    assert p5.shape == (2, 512, 20, 20)


def test_neck():
    print("\n=== test_neck ===")
    neck = YOLOPAN()

    p3 = torch.randn(2, 128, 80, 80)
    p4 = torch.randn(2, 256, 40, 40)
    p5 = torch.randn(2, 512, 20, 20)

    n3, n4, n5 = neck(p3, p4, p5)

    print("n3:", n3.shape)
    print("n4:", n4.shape)
    print("n5:", n5.shape)

    assert n3.shape == (2, 128, 80, 80)
    assert n4.shape == (2, 256, 40, 40)
    assert n5.shape == (2, 512, 20, 20)


def test_head():
    print("\n=== test_head ===")
    nc = 80
    head = DetectHead(nc=nc, ch=(128, 256, 512), na=3)

    n3 = torch.randn(2, 128, 80, 80)
    n4 = torch.randn(2, 256, 40, 40)
    n5 = torch.randn(2, 512, 20, 20)

    outputs = head([n3, n4, n5])

    for i, out in enumerate(outputs):
        print(f"head out[{i}]:", out.shape)

    assert outputs[0].shape == (2, 3, 80, 80, 85)
    assert outputs[1].shape == (2, 3, 40, 40, 85)
    assert outputs[2].shape == (2, 3, 20, 20, 85)


def test_full_model():
    print("\n=== test_full_model ===")
    model = YOLOv5FromScratch(nc=80)
    x = torch.randn(2, 3, 640, 640)
    outputs = model(x)

    for i, out in enumerate(outputs):
        print(f"model out[{i}]:", out.shape)

    assert outputs[0].shape == (2, 3, 80, 80, 85)
    assert outputs[1].shape == (2, 3, 40, 40, 85)
    assert outputs[2].shape == (2, 3, 20, 20, 85)


def main():
    test_backbone()
    test_neck()
    test_head()
    test_full_model()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()