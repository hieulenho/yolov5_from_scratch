import sys
import os
from numpy import insert
import torch
from torch.utils.data import dataset


FILE = Path.(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_dataloader

from models.yolo import YOLOv5FromScratch

from loss.loss import YoloLoss

def main():
    torch.manual_seed(0)

    data_yaml = ROOT / "dataset" / "coco2017" / "dataset.yaml"