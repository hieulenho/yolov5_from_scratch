import torch
import torch.nn as nn


class DetectHead(nn.Module):
    def __init__(
        self,
        nc=80,
        ch=(128, 256, 512),
        na=3,
        anchors=(
            ((10, 13), (16, 30), (33, 23)),
            ((30, 61), (62, 45), (59, 119)),
            ((116, 90), (156, 198), (373, 326)),
        ),
        strides=(8, 16, 32),
    ):
        super().__init__()
        self.nc = nc
        self.na = na
        self.no = nc + 5
        self.nl = len(ch)

        self.m = nn.ModuleList([
            nn.Conv2d(ch[0], na * self.no, kernel_size=1),
            nn.Conv2d(ch[1], na * self.no, kernel_size=1),
            nn.Conv2d(ch[2], na * self.no, kernel_size=1),
        ])

        self.register_buffer(
            "anchors",
            torch.tensor(anchors, dtype=torch.float32).view(self.nl, self.na, 2)
        )  # [nl, na, 2]

        self.register_buffer(
            "stride",
            torch.tensor(strides, dtype=torch.float32)
        )  # [nl]

    def forward_one(self, x, conv):
        x = conv(x)  # [B, na*no, H, W]
        b, _, h, w = x.shape
        x = x.view(b, self.na, self.no, h, w)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [B, A, H, W, no]
        return x

    def forward(self, features):
        return [self.forward_one(f, conv) for f, conv in zip(features, self.m)]

    def make_grid(self, nx, ny, device):
        y, x = torch.meshgrid(
            torch.arange(ny, device=device),
            torch.arange(nx, device=device),
            indexing="ij",
        )
        grid = torch.stack((x, y), dim=-1).float()
        return grid.view(1, 1, ny, nx, 2)

    def decode_one(self, p, i):
        """
        p: raw prediction [B, A, H, W, no]
        return decoded prediction in image scale:
            xywh -> pixels on input image
        """
        b, a, ny, nx, no = p.shape
        grid = self.make_grid(nx, ny, p.device)
        anchor = self.anchors[i].view(1, a, 1, 1, 2)

        xy = (p[..., 0:2].sigmoid() * 2.0 - 0.5 + grid) * self.stride[i]
        wh = (p[..., 2:4].sigmoid() * 2.0).pow(2) * anchor
        obj = p[..., 4:5].sigmoid()
        cls = p[..., 5:].sigmoid()

        return torch.cat([xy, wh, obj, cls], dim=-1)