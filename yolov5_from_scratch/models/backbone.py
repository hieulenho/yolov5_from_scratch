import torch.nn as nn
from .common import ConvBNAct, C3, SPPF


class YOLOBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNAct(3, 32, k=6, s=2, p=2)      # 640 -> 320
        self.stage1_conv = ConvBNAct(32, 64, k=3, s=2)   # 320 -> 160
        self.stage1_c3 = C3(64, 64, n=1)

        self.stage2_conv = ConvBNAct(64, 128, k=3, s=2)  # 160 -> 80
        self.stage2_c3 = C3(128, 128, n=2)               # P3

        self.stage3_conv = ConvBNAct(128, 256, k=3, s=2) # 80 -> 40
        self.stage3_c3 = C3(256, 256, n=3)               # P4

        self.stage4_conv = ConvBNAct(256, 512, k=3, s=2) # 40 -> 20
        self.stage4_c3 = C3(512, 512, n=1)
        self.sppf = SPPF(512, 512)                       # P5

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1_c3(self.stage1_conv(x))

        p3 = self.stage2_c3(self.stage2_conv(x))
        p4 = self.stage3_c3(self.stage3_conv(p3))
        p5 = self.sppf(self.stage4_c3(self.stage4_conv(p4)))

        return p3, p4, p5