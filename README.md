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
va co them center-distance fallback khi box khong con overlap. No van co the doi
ID khi doi tuong bi che khuat lau hoac khi nhieu nguoi dung sat nhau. Khi
detection da tot hon, nen thay bang ByteTrack de co ID on dinh hon.

## Cau hinh da kiem tra voi hai video thuc te

Video ngang 4K, chi theo doi xe:

```powershell
python yolov5_from_scratch/utils/predict.py `
  --source D:\videos\traffic.mp4 `
  --classes car bus truck `
  --img-size 960 `
  --conf 0.05 `
  --track-conf 0.12 `
  --display-conf 0.12 `
  --count-conf 0.18 `
  --min-track-hits 8 `
  --track-center-distance 1.0 `
  --roi 0.08 0.12 0.92 1.0 `
  --line-position 0.65 `
  --count-directions top_to_bottom `
  --output-width 960 `
  --no-draw-roi `
  --name traffic_improved
```

Video doc, canh dong nguoi va xe may:

```powershell
python yolov5_from_scratch/utils/predict.py `
  --source D:\videos\traffic2.mp4 `
  --classes person car motorcycle bus truck `
  --img-size 960 `
  --conf 0.05 `
  --track-conf 0.12 `
  --display-conf 0.12 `
  --count-conf 0.18 `
  --min-track-hits 8 `
  --track-center-distance 0.8 `
  --roi 0.0 0.05 1.0 0.98 `
  --line-position 0.58 `
  --count-directions top_to_bottom `
  --output-width 720 `
  --no-draw-roi `
  --name traffic2_improved
```

`--roi X1 Y1 X2 Y2` dung toa do chuan hoa theo frame goc. Model chi suy luan
ben trong ROI, sau do box duoc dua ve toa do frame goc. `detections.csv` luon
giu toa do frame goc, ke ca khi `--output-width` lam video output nho hon.

Ba nguong confidence co vai tro rieng:

- `--conf`: detection toi thieu duoc luu vao CSV.
- `--track-conf`: detection toi thieu duoc cap track ID.
- `--display-conf`: detection toi thieu duoc ve len video.
- `--count-conf`: confidence trung binh toi thieu cua ca track de duoc dem.

## Phan tich mot run

```powershell
python yolov5_from_scratch/tools/analyze_detection_run.py `
  yolov5_from_scratch/runs/detect/traffic_improved
```

Cong cu in tong detection, so track, do dai track, confidence va crossing.
Them `--json` neu can output co cau truc.

Ket qua sau khi cai tien pipeline:

- `traffic.mp4`: 1.073 track xuong 124; median track tu 9 len 40 frame.
- `traffic2.mp4`: 632 track xuong 190; median track tu 12 len 41 frame.
- Video output 4K giam tu 508 MB xuong 88 MB.
- Video output doc giam tu 299 MB xuong 138 MB.

So dem cua video doc van la `person`, khong phai `motorcycle`, vi checkpoint
hien tai thuong nhan nguoi lai xe may thanh nguoi. Can fine-tune de sua loi nay.

## Tao bo frame de gan nhan

```powershell
python yolov5_from_scratch/tools/extract_video_frames.py `
  --source D:\videos\traffic.mp4 D:\videos\traffic2.mp4 `
  --output yolov5_from_scratch/datasets/traffic_pilot `
  --frames-per-video 120 `
  --max-width 1280 `
  --jpeg-quality 90
```

Bo pilot hien tai da duoc tao tai:

```text
yolov5_from_scratch/datasets/traffic_pilot/
├── classes.txt
├── frames.csv
└── images/
    └── unlabeled/
```

Co 240 frame, tong dung luong khoang 92 MB. `frames.csv` luu video nguon,
frame index, timestamp va kich thuoc goc.

Quy tac gan nhan nam trong
[TRAFFIC_ANNOTATION_GUIDE.md](TRAFFIC_ANNOTATION_GUIDE.md). Cau hinh dataset
5 lop sau khi chia train/val nam trong
`yolov5_from_scratch/configs/traffic5.yaml`.

## Tao pre-label 5 lop

Dung Faster R-CNN pretrained cua TorchVision de tao nhan ban dau:

```powershell
cd yolov5_from_scratch
python tools/prelabel_traffic_dataset.py `
  --source datasets/traffic_pilot `
  --output datasets/traffic5 `
  --conf 0.60 `
  --vehicle-nms-iou 0.70 `
  --val-fraction 0.20 `
  --device cuda
```

Dataset hien tai co:

- 192 anh train, 48 anh validation.
- 6.853 box: 4.677 person, 1.210 car, 351 motorcycle, 245 bus, 370 truck.
- Validation la mot doan thoi gian lien tuc o giua moi video.
- `prelabels.csv` luu confidence va toa do goc de audit.

Day la pre-label, chua phai ground truth. Can uu tien sua cac box `bus`/`truck`,
box bi bo sot va rider/motorcycle truoc khi dung ket qua cho bao cao chinh thuc.

Kiem tra cau truc:

```powershell
python tools/check_dataset.py `
  --data configs/traffic5.yaml `
  --strict-missing-labels
```

## Fine-tune tu best.pt

`--weights` chi nap trong so va tao optimizer moi. Detection head 80 lop duoc
anh xa theo ten class sang head 5 lop. Khong dung `--resume` khi doi tu COCO
80 lop sang traffic 5 lop.

```powershell
python utils/train.py `
  --data configs/traffic5.yaml `
  --weights runs/train/exp/weights/best.pt `
  --epochs 50 `
  --img-size 640 `
  --batch-size 8 `
  --workers 4 `
  --device cuda `
  --amp `
  --optimizer AdamW `
  --lr 0.001 `
  --val `
  --project runs/train `
  --name traffic5_finetune
```

Sau khi fine-tune:

1. Chay checkpoint moi tren hai video goc.
2. So sanh `motorcycle`, false positive va do on dinh tracking.
3. Sua thu cong pre-label sai, them failure case va train vong tiep theo.
4. Khi detection du tot, thay tracker IoU bang ByteTrack va export ONNX.

## Kiem thu nhanh

```powershell
python yolov5_from_scratch/tools/test_model.py
python yolov5_from_scratch/tools/test_inference.py
python yolov5_from_scratch/tools/test_prelabel_traffic.py
python yolov5_from_scratch/tools/test_train_transfer.py
```
