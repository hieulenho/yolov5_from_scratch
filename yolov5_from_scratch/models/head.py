import torch
import torch.nn as nn

anchors = (
    ((10, 13), (16, 30), (33, 23)),      # P3/8
    ((30, 61), (62, 45), (59, 119)),     # P4/16
    ((116, 90), (156, 198), (373, 326)), # P5/32
)
class DetectHead(nn.Module):
    def __init__(self, nc=80, ch=(128, 256, 512), na=3):
        super().__init__()
        self.nc = nc
        self.na = na
        self.no = nc + 5

        self.m = nn.ModuleList([
            nn.Conv2d(ch[0], na * self.no, kernel_size=1),
            nn.Conv2d(ch[1], na * self.no, kernel_size=1),
            nn.Conv2d(ch[2], na * self.no, kernel_size=1),
        ])



    def forward_one(self, x, conv):
        x = conv(x)  # [B, na*no, H, W]
        b, _, h, w = x.shape
        x = x.view(b, self.na, self.no, h, w)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [B, A, H, W, no]
        return x

    def forward(self, features):
        assert len(features) == 3
        return [self.forward_one(f, conv) for f, conv in zip(features, self.m)]
        