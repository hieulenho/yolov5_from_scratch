# YOLOv5 From Scratch - Traffic Video Detection

Project nay cai dat mot mo hinh YOLOv5 nho tu dau bang PyTorch. Checkpoint COCO
hien tai co the duoc dung de phat hien, theo doi va dem cac doi tuong giao thong
trong anh hoac video.

## Checkpoint hien tai

- `best.pt`: epoch 91, duoc chon theo validation loss.
- `last.pt`: epoch 100.
- Kiem tra COCO val 5.000 anh:
  - `best.pt`: mAP@0.5:0.95 = 0.0090, mAP@0.5 = 0.0208.
  - `last.pt`: mAP@0.5:0.95 = 0.0088, mAP@0.5 = 0.0205.

Chat luong hien tai con thap. Pipeline detection/video dung de quan sat model,
xay ung dung va thu thap du lieu fine-tune; chua nen dung cho he thong thuc te.

## Cai dat

Moi truong da duoc kiem tra:

- Python 3.12
- PyTorch 2.5.1 + CUDA 12.1
- torchvision 0.20.1
- OpenCV 4.13

```powershell
pip install -r requirements.txt
```

Neu can CUDA, nen cai `torch` va `torchvision` theo wheel phu hop voi driver/GPU
truoc khi cai cac dependency con lai.

## Detect mot anh

Chay tu thu muc goc repository:

```powershell
python yolov5_from_scratch/utils/predict.py `
  --source yolov5_from_scratch/datasets/coco2017/images/val2017/val2017/000000000139.jpg `
  --conf 0.05 `
  --count-line none
```

Output duoc luu trong `yolov5_from_scratch/runs/detect/traffic`.

## Detect, track va dem video giao thong

```powershell
python yolov5_from_scratch/utils/predict.py `
  --source D:\videos\traffic.mp4 `
  --weights yolov5_from_scratch/runs/train/exp/weights/best.pt `
  --conf 0.05 `
  --count-line horizontal `
  --line-position 0.55
```

Mac dinh chi giu cac lop:

- `person`
- `bicycle`
- `car`
- `motorcycle`
- `bus`
- `truck`

Ket qua gom:

- Video/anh da ve bounding box, confidence va track ID.
- `detections.csv`: detection theo tung frame.
- `counts.csv`: tong so doi tuong cat qua vach theo lop va huong.

Vach ngang dem `top_to_bottom` va `bottom_to_top`. Vach doc dem
`left_to_right` va `right_to_left`.

## Webcam

```powershell
python yolov5_from_scratch/utils/predict.py --source 0 --view
```

Nhan `q` de dung cua so xem truc tiep.

## Tuy chon huu ich

Detect tat ca 80 lop COCO:

```powershell
python yolov5_from_scratch/utils/predict.py --source input.mp4 --all-classes
```

Chi detect mot so lop:

```powershell
python yolov5_from_scratch/utils/predict.py `
  --source input.mp4 `
  --classes car motorcycle bus truck
```

Tat tracking va dem:

```powershell
python yolov5_from_scratch/utils/predict.py `
  --source input.mp4 `
  --no-track `
  --count-line none
```

Chi xu ly 100 frame dau de smoke test:

```powershell
python yolov5_from_scratch/utils/predict.py `
  --source input.mp4 `
  --max-frames 100
```

## Han che cua tracker

Tracker hien tai ghep box bang IoU va class. No nhe, khong can dependency ngoai,
nhung co the doi ID khi doi tuong bi che khuat, di nhanh hoac roi khoi frame.
Khi detection da tot hon, nen thay bang ByteTrack de co ID on dinh hon.

## Buoc fine-tune tiep theo

1. Dung `predict.py` tren video giao thong thuc te de tim failure case.
2. Trich va gan nhan 1.000-3.000 frame cho 5-6 lop giao thong.
3. Bo sung che do load `best.pt` theo weights-only, khong resume optimizer cu.
4. Sua loss/target assignment va tich hop COCO mAP vao validation.
5. Fine-tune voi learning rate nho va chon checkpoint theo mAP.
6. Sau khi detection tot, thay tracker IoU bang ByteTrack va export ONNX.

## Kiem thu nhanh

```powershell
python yolov5_from_scratch/tools/test_model.py
python yolov5_from_scratch/tools/test_inference.py
```
