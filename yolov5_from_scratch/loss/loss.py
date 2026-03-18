import torch
import torch.nn as nn


def bbox_iou_xywh(box1, box2, eps=1e-7):
    """
    box1, box2: [N, 4] in xywh format
    return: IoU [N]
    """
    b1_x1 = box1[:, 0] - box1[:, 2] / 2
    b1_y1 = box1[:, 1] - box1[:, 3] / 2
    b1_x2 = box1[:, 0] + box1[:, 2] / 2
    b1_y2 = box1[:, 1] + box1[:, 3] / 2

    b2_x1 = box2[:, 0] - box2[:, 2] / 2
    b2_y1 = box2[:, 1] - box2[:, 3] / 2
    b2_x2 = box2[:, 0] + box2[:, 2] / 2
    b2_y2 = box2[:, 1] + box2[:, 3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
    union = area1 + area2 - inter

    return inter / (union + eps)


class YoloLoss(nn.Module):
    def __init__(
        self,
        head,
        nc=80,
        anchor_t=4.0,
        box_gain=0.05,
        obj_gain=1.0,
        cls_gain=0.5,
    ):
        super().__init__()
        self.nc = nc
        self.nl = head.nl
        self.na = head.na

        self.register_buffer("anchors", head.anchors.clone())  # [nl, na, 2] pixel-space
        self.register_buffer("stride", head.stride.clone())    # [nl]

        self.anchor_t = anchor_t
        self.box_gain = box_gain
        self.obj_gain = obj_gain
        self.cls_gain = cls_gain

        self.bce_obj = nn.BCEWithLogitsLoss()
        self.bce_cls = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def build_targets(self, preds, targets):
        """
        preds: list of 3 tensors
            p[i] shape = [B, A, H, W, no]
        targets: [M, 6] = [batch_idx, cls, x, y, w, h] normalized to image
        """
        device = targets.device
        tcls, tbox, indices, anch = [], [], [], []

        if targets.shape[0] == 0:
            for _ in preds:
                empty_long = torch.zeros(0, dtype=torch.long, device=device)
                tcls.append(empty_long)
                tbox.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
                indices.append((empty_long, empty_long, empty_long, empty_long))
                anch.append(torch.zeros((0, 2), dtype=torch.float32, device=device))
            return tcls, tbox, indices, anch

        nt = targets.shape[0]

        for i, p in enumerate(preds):
            bsz, na, ny, nx, no = p.shape

            # convert normalized-image xywh -> grid xywh
            gain = torch.tensor([1, 1, nx, ny, nx, ny], device=device, dtype=targets.dtype)
            t = targets * gain  # [nt, 6]

            # anchors in grid units
            anchors_grid = self.anchors[i] / self.stride[i]  # [na, 2]

            # repeat targets for each anchor
            t_repeat = t[None].repeat(self.na, 1, 1)  # [na, nt, 6]
            a_idx = torch.arange(self.na, device=device).view(self.na, 1).repeat(1, nt)

            # ratio matching
            r = t_repeat[..., 4:6] / anchors_grid[:, None, :]  # [na, nt, 2]
            max_ratio = torch.max(r, 1.0 / (r + 1e-9)).max(dim=2).values
            mask = max_ratio < self.anchor_t

            t_match = t_repeat[mask]
            a_match = a_idx[mask]

            if t_match.shape[0] == 0:
                empty_long = torch.zeros(0, dtype=torch.long, device=device)
                tcls.append(empty_long)
                tbox.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
                indices.append((empty_long, empty_long, empty_long, empty_long))
                anch.append(torch.zeros((0, 2), dtype=torch.float32, device=device))
                continue

            b_idx = t_match[:, 0].long()
            c_idx = t_match[:, 1].long()
            gxy = t_match[:, 2:4]
            gwh = t_match[:, 4:6]

            gij = gxy.long()
            gi = gij[:, 0].clamp_(0, nx - 1)
            gj = gij[:, 1].clamp_(0, ny - 1)

            tcls.append(c_idx)
            tbox.append(torch.cat([gxy, gwh], dim=1))  # [n_pos, 4] in grid xywh
            indices.append((b_idx, a_match.long(), gj, gi))
            anch.append(anchors_grid[a_match.long()])

        return tcls, tbox, indices, anch

    def forward(self, preds, targets):
        device = preds[0].device
        targets = targets.to(device)

        tcls, tbox, indices, anch = self.build_targets(preds, targets)

        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lcls = torch.zeros(1, device=device)

        for i, p in enumerate(preds):
            # p: [B, A, H, W, no]
            obj_target = torch.zeros_like(p[..., 4])  # [B, A, H, W]

            b, a, gj, gi = indices[i]

            if b.numel() > 0:
                ps = p[b, a, gj, gi]  # [n_pos, no]

                # decode positives in grid units
                pxy = ps[:, 0:2].sigmoid() * 2.0 - 0.5
                pxy = pxy + torch.stack((gi, gj), dim=1).float()

                pwh = (ps[:, 2:4].sigmoid() * 2.0).pow(2) * anch[i]

                pbox = torch.cat([pxy, pwh], dim=1)  # [n_pos, 4]
                iou = bbox_iou_xywh(pbox, tbox[i])

                lbox += (1.0 - iou).mean()

                obj_target[b, a, gj, gi] = iou.detach().clamp(0).to(obj_target.dtype)

                if self.nc > 1:
                    t = torch.zeros_like(ps[:, 5:])
                    t[torch.arange(ps.shape[0], device=device), tcls[i]] = 1.0
                    lcls += self.bce_cls(ps[:, 5:], t)

            lobj += self.bce_obj(p[..., 4], obj_target)

        loss = self.box_gain * lbox + self.obj_gain * lobj + self.cls_gain * lcls

        loss_items = {
            "loss": float(loss.detach()),
            "lbox": float(lbox.detach()),
            "lobj": float(lobj.detach()),
            "lcls": float(lcls.detach()),
        }
        return loss, loss_items