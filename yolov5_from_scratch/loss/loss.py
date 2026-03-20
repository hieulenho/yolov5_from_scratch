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
        anchor_t=6.0,
        box_gain=0.05,
        obj_gain=1.0,
        cls_gain=0.5,
    ):
        super().__init__()
        self.nc = nc
        self.nl = head.nl
        self.na = head.na

        # head.anchors: [nl, na, 2] in pixel space
        # head.stride:  [nl]
        self.register_buffer("anchors", head.anchors.clone())
        self.register_buffer("stride", head.stride.clone())

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
        targets: [M, 6] = [batch_idx, cls, x, y, w, h]
                 xywh are normalized to input image size
        """
        device = preds[0].device
        tcls, tbox, indices, anch = [], [], [], []

        if targets.shape[0] == 0:
            for _ in preds:
                empty_long = torch.zeros(0, dtype=torch.long, device=device)
                tcls.append(empty_long)
                tbox.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
                indices.append((empty_long, empty_long, empty_long, empty_long))
                anch.append(torch.zeros((0, 2), dtype=torch.float32, device=device))
            return tcls, tbox, indices, anch

        for i, p in enumerate(preds):
            _, _, ny, nx, _ = p.shape

            # normalized image xywh -> grid xywh
            gain = targets.new_tensor([1, 1, nx, ny, nx, ny])
            t = targets * gain  # [M, 6]

            b = t[:, 0].long()
            c = t[:, 1].long()
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]

            # anchors in grid units
            anchors_grid = self.anchors[i] / self.stride[i]  # [na, 2]

            # choose best anchor per target on this scale
            # r: [na, M, 2]
            r = gwh[None] / (anchors_grid[:, None] + 1e-9)
            max_ratio = torch.max(r, 1.0 / (r + 1e-9)).amax(dim=2)  # [na, M]
            best_ratio, best_a = max_ratio.min(dim=0)  # [M]

            # normal filtering
            keep = (gwh[:, 0] > 0) & (gwh[:, 1] > 0) & (best_ratio < self.anchor_t)

            # fallback: if scale matched nothing but batch has targets,
            # keep best anchor for all valid boxes so training can start
            if keep.sum() == 0 and targets.shape[0] > 0:
                keep = (gwh[:, 0] > 0) & (gwh[:, 1] > 0)

            b = b[keep]
            c = c[keep]
            gxy = gxy[keep]
            gwh = gwh[keep]
            a = best_a[keep].long()

            if b.numel() == 0:
                empty_long = torch.zeros(0, dtype=torch.long, device=device)
                tcls.append(empty_long)
                tbox.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
                indices.append((empty_long, empty_long, empty_long, empty_long))
                anch.append(torch.zeros((0, 2), dtype=torch.float32, device=device))
                continue

            gij = gxy.long()
            gi = gij[:, 0].clamp_(0, nx - 1)
            gj = gij[:, 1].clamp_(0, ny - 1)

            tcls.append(c)
            tbox.append(torch.cat([gxy, gwh], dim=1))  # absolute grid xywh
            indices.append((b, a, gj, gi))
            anch.append(anchors_grid[a])

        return tcls, tbox, indices, anch

    def forward(self, preds, targets):
        """
        preds: list of raw predictions
            each p has shape [B, A, H, W, no]
        targets: [M, 6] = [batch_idx, cls, x, y, w, h]
        """
        device = preds[0].device
        targets = targets.to(device)

        tcls, tbox, indices, anch = self.build_targets(preds, targets)

        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lcls = torch.zeros(1, device=device)

        for i, p in enumerate(preds):
            # objectness target map
            obj_target = torch.zeros_like(p[..., 4])  # [B, A, H, W]

            b, a, gj, gi = indices[i]

            if b.numel() > 0:
                # positive predictions only
                ps = p[b, a, gj, gi]  # [n_pos, no]

                # decode positives in grid units
                pxy = ps[:, 0:2].sigmoid() * 2.0 - 0.5
                pxy = pxy + torch.stack((gi, gj), dim=1).float()

                pwh = (ps[:, 2:4].sigmoid() * 2.0).pow(2) * anch[i]

                pbox = torch.cat([pxy, pwh], dim=1)  # [n_pos, 4]
                iou = bbox_iou_xywh(pbox, tbox[i])

                # box loss
                lbox += (1.0 - iou).mean()

                # objectness target at positive locations
                obj_target[b, a, gj, gi] = iou.detach().clamp(0).to(obj_target.dtype)

                # class loss
                if self.nc > 1:
                    t = torch.zeros_like(ps[:, 5:])
                    t[torch.arange(ps.shape[0], device=device), tcls[i]] = 1.0
                    lcls += self.bce_cls(ps[:, 5:], t)

            # objectness loss on full feature map
            lobj += self.bce_obj(p[..., 4], obj_target)

        loss = self.box_gain * lbox + self.obj_gain * lobj + self.cls_gain * lcls

        loss_items = {
            "loss": float(loss.detach()),
            "lbox": float(lbox.detach()),
            "lobj": float(lobj.detach()),
            "lcls": float(lcls.detach()),
        }
        return loss, loss_items