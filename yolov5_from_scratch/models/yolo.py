import torch.nn as nn
from .backbone import YOLOBackbone
from .neck import YOLOPAN
from .head import DetectHead

anchors = (
    ((10, 13), (16, 30), (33, 23)),      # P3/8
    ((30, 61), (62, 45), (59, 119)),     # P4/16
    ((116, 90), (156, 198), (373, 326)), # P5/32
)

class YOLOv5FromScratch(nn.Module):
    def __init__(self, nc=80):
        super().__init__()
        self.backbone = YOLOBackbone()
        self.neck = YOLOPAN()
        self.head = DetectHead(nc=nc, ch=(128, 256, 512), na=3)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)
        outputs = self.head([n3, n4, n5])
        return outputs