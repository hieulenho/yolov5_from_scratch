import torch
import torch.nn as nn
from .common import ConvBNAct, C3


class YOLOPAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # top-down
        self.reduce_p5 = ConvBNAct(512, 256, 1, 1)
        self.c3_p4 = C3(256 + 256, 256, n=1, shortcut=False)

        self.reduce_p4 = ConvBNAct(256, 128, 1, 1)
        self.c3_p3 = C3(128 + 128, 128, n=1, shortcut=False)

        # bottom-up
        self.down_p3 = ConvBNAct(128, 128, 3, 2)
        self.c3_n4 = C3(128 + 256, 256, n=1, shortcut=False)

        self.down_n4 = ConvBNAct(256, 256, 3, 2)
        self.c3_n5 = C3(256 + 256, 512, n=1, shortcut=False)

    def forward(self, p3, p4, p5):
        p5_reduced = self.reduce_p5(p5)
        x = self.upsample(p5_reduced)
        x = torch.cat([x, p4], dim=1)
        p4_fused = self.c3_p4(x)

        p4_reduced = self.reduce_p4(p4_fused)
        x = self.upsample(p4_reduced)
        x = torch.cat([x, p3], dim=1)
        n3 = self.c3_p3(x)

        x = self.down_p3(n3)
        x = torch.cat([x, p4_fused], dim=1)
        n4 = self.c3_n4(x)

        x = self.down_n4(n4)
        x = torch.cat([x, p5_reduced], dim=1)
        n5 = self.c3_n5(x)

        return n3, n4, n5