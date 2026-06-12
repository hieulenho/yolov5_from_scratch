# YOLOv5 From Scratch - Camera Object Detection

Dự án gồm ba phần độc lập:

- Train và fine-tune YOLOv5 tự xây dựng.
- Detect, track và đếm object từ ảnh, video, webcam hoặc RTSP.
- Chuẩn bị dataset, đánh giá kết quả và quản lý artifact.

Runtime không còn bị khóa vào traffic. Model và file dataset YAML quyết định
các class có thể nhận diện; `--classes` chỉ chọn một nhóm class để hiển thị,
tracking hoặc đếm.

## Cấu trúc

```text
predict.py                 Detect ảnh/video/thư mục/stream
camera.py                  Chạy camera, mặc định source 0
train.py                   Train hoặc fine-tune
validate.py                Validate checkpoint
yolov5_from_scratch/
  cli/                     CLI và application profile
  runtime/                 Detector, tracker, counter, renderer, pipeline
  training/                Train/validate engine
  models/ data/ loss/      Thành phần YOLO cốt lõi
  configs/                 Dataset và application YAML
  tools/                   Dataset, analysis, dev, maintenance
  tests/                   Test
datasets/                  Dataset cục bộ, không commit Git
artifacts/
  checkpoints/             Checkpoint cần giữ
  predictions/             Video/ảnh/CSV đầu ra
  reports/                 Báo cáo đánh giá
docs/                      Kiến trúc và quy tắc gán nhãn
```

Chi tiết: [docs/architecture.md](docs/architecture.md).

## Cài đặt

```powershell
cd D:\yolov5_from_scratch
pip install -r requirements.txt
```

Có thể cài project dạng editable để dùng các lệnh `yolo-predict`,
`yolo-train`, `yolo-validate`:

```powershell
pip install -e .
```

## Checkpoint hiện có

```text
artifacts/checkpoints/coco80/best.pt
artifacts/checkpoints/coco80/last.pt
artifacts/checkpoints/traffic5/best.pt
artifacts/checkpoints/traffic5/last.pt
```

- `coco80`: model 80 class, dùng để thử nhiều loại object.
- `traffic5`: `person`, `car`, `motorcycle`, `bus`, `truck`.
- `traffic5/best.pt` là epoch 21/50, validation loss `0.1815`.

## Chạy video mới

Ứng dụng đếm xe máy đã được lưu thành profile:

```powershell
python predict.py `
  --config yolov5_from_scratch/configs/apps/motorcycle_counter.yaml `
  --source "D:\video\traffic3.mp4" `
  --name traffic3_motorcycle
```

Kết quả:

```text
artifacts/predictions/traffic3_motorcycle/
  traffic3.mp4
  detections.csv
  counts.csv
```

Giá trị truyền trực tiếp trên command line luôn ghi đè giá trị trong profile.

## Detect nhiều object

Dùng checkpoint COCO 80 class và không truyền `--classes`:

```powershell
python predict.py `
  --weights artifacts/checkpoints/coco80/best.pt `
  --data datasets/coco2017/dataset.yaml `
  --source "D:\video\input.mp4" `
  --conf 0.05 `
  --count-line none `
  --name all_objects
```

Chỉ chọn một số class:

```powershell
python predict.py `
  --weights artifacts/checkpoints/coco80/best.pt `
  --data datasets/coco2017/dataset.yaml `
  --source "D:\video\input.mp4" `
  --classes person car dog cat `
  --conf 0.05 `
  --count-line none `
  --name selected_objects
```

## Camera

Webcam USB mặc định:

```powershell
python camera.py `
  --config yolov5_from_scratch/configs/apps/generic_camera.yaml
```

Profile này mặc định dùng `--no-save-media --no-save-csv` để camera chạy lâu
không làm đầy ổ đĩa. Thêm `--save-media` hoặc `--save-csv` khi thật sự cần lưu.

Camera index khác:

```powershell
python camera.py --source 1 --view
```

RTSP:

```powershell
python camera.py `
  --source "rtsp://user:password@192.168.1.10:554/stream" `
  --weights artifacts/checkpoints/coco80/best.pt `
  --data datasets/coco2017/dataset.yaml `
  --view
```

Nhấn `q` để đóng cửa sổ khi dùng `--view`.

## Train traffic5

Fine-tune từ checkpoint COCO:

```powershell
python train.py `
  --data yolov5_from_scratch/configs/traffic5.yaml `
  --weights artifacts/checkpoints/coco80/best.pt `
  --epochs 50 `
  --img-size 640 `
  --batch-size 8 `
  --workers 4 `
  --device cuda `
  --amp `
  --optimizer AdamW `
  --lr 0.001 `
  --val `
  --name traffic5_next
```

`--weights` khởi tạo từ trọng số đã train nhưng tạo optimizer và lịch learning
rate mới. `--resume` chỉ dùng để khôi phục đúng một run bị gián đoạn, với cùng
kiến trúc, số class, optimizer và tổng số epoch đã định trước.

Checkpoint traffic5 đã hoàn tất 50 epoch. Cách khuyến nghị để train thêm là mở
một giai đoạn mới 30 epoch từ `last.pt`:

```powershell
python train.py `
  --data yolov5_from_scratch/configs/traffic5.yaml `
  --weights artifacts/checkpoints/traffic5/last.pt `
  --epochs 30 `
  --img-size 640 `
  --batch-size 8 `
  --workers 4 `
  --device cuda `
  --amp `
  --optimizer AdamW `
  --lr 0.001 `
  --val `
  --project artifacts/training `
  --name traffic5_resume
```

Không dùng `--resume ... --epochs 80` cho trường hợp này vì learning-rate
scheduler của run cũ đã đi hết chu kỳ 50 epoch. Khi một run đang dở thực sự bị
ngắt, `--resume` sẽ khôi phục model, optimizer, scheduler, AMP scaler và lịch sử.

## Công cụ dataset

Kiểm tra dataset:

```powershell
python -m yolov5_from_scratch.tools.dataset.check `
  --data yolov5_from_scratch/configs/traffic5.yaml `
  --strict-missing-labels
```

Trích frame:

```powershell
python -m yolov5_from_scratch.tools.dataset.extract_frames `
  --source D:\video\traffic.mp4 `
  --output datasets/traffic_pilot `
  --frames-per-video 120
```

Pre-label traffic:

```powershell
python -m yolov5_from_scratch.tools.dataset.prelabel_traffic `
  --source datasets/traffic_pilot `
  --output datasets/traffic5 `
  --conf 0.60 `
  --device cuda
```

Quy tắc nhãn: [docs/traffic_annotation.md](docs/traffic_annotation.md).

## Đánh giá và phân tích

```powershell
python -m yolov5_from_scratch.tools.analysis.analyze_run `
  artifacts/predictions/motorcycle_counter_final
```

```powershell
python -m yolov5_from_scratch.tools.analysis.evaluate_detections `
  artifacts/reports/traffic5_validation `
  --data yolov5_from_scratch/configs/traffic5.yaml
```

## Kiểm thử

```powershell
python -m yolov5_from_scratch.tests.test_model
python -m yolov5_from_scratch.tests.test_runtime
python -m yolov5_from_scratch.tests.test_prelabel
python -m yolov5_from_scratch.tests.test_training_transfer
```

## Dọn artifact

Mặc định chỉ liệt kê prediction cũ hơn 7 ngày:

```powershell
python -m yolov5_from_scratch.tools.maintenance.clean_artifacts
```

Thực sự xóa, nhưng giữ các demo chỉ định:

```powershell
python -m yolov5_from_scratch.tools.maintenance.clean_artifacts `
  --older-than-days 7 `
  --keep motorcycle_counter_final traffic3_motorcycle `
  --apply
```

## Giới hạn hiện tại

- Traffic5 được fine-tune phần lớn từ pseudo-label, chưa thay thế ground truth
  được kiểm duyệt thủ công.
- Tracker hiện tại dựa trên IoU và khoảng cách tâm; camera triển khai lâu dài
  nên nâng cấp sang ByteTrack.
- Luồng RTSP hiện chưa tự kết nối lại khi camera hoặc mạng bị ngắt.
- Checkpoint `coco80` hỗ trợ đủ 80 class về mặt kiến trúc nhưng độ chính xác
  hiện còn thấp; cần train/đánh giá lại trước khi xem là model đa vật thể dùng
  trong sản phẩm.
- Chọn checkpoint hiện vẫn dựa trên validation loss; bước tiếp theo nên tích
  hợp mAP50-95 vào quá trình train.
- Trước khi dùng số đếm trong thực tế, cần kiểm tra trên nhiều góc camera,
  ban đêm, mưa, che khuất và mật độ đông.
