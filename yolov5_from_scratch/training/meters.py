class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(self.count, 1)

    def update(self, value, n=1):
        self.sum += float(value) * int(n)
        self.count += int(n)


class LossMeters:
    def __init__(self):
        self.loss = AverageMeter()
        self.lbox = AverageMeter()
        self.lobj = AverageMeter()
        self.lcls = AverageMeter()

    def update(self, loss_items, n):
        self.loss.update(loss_items["loss"], n)
        self.lbox.update(loss_items["lbox"], n)
        self.lobj.update(loss_items["lobj"], n)
        self.lcls.update(loss_items["lcls"], n)

    def as_dict(self):
        return {
            "loss": self.loss.avg,
            "lbox": self.lbox.avg,
            "lobj": self.lobj.avg,
            "lcls": self.lcls.avg,
        }


class CounterMeter:
    def __init__(self):
        self.images = 0
        self.targets = 0
        self.empty_images = 0
        self.empty_batches = 0
        self.positive_matches = 0

    def as_dict(self):
        return {
            "images": int(self.images),
            "targets": int(self.targets),
            "empty_images": int(self.empty_images),
            "empty_batches": int(self.empty_batches),
            "positive_matches": int(self.positive_matches),
        }
