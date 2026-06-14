# Project Architecture

```text
.
|-- predict.py                    # image/video/directory/stream inference
|-- camera.py                     # camera entry point, defaults to source 0
|-- train.py                      # training entry point
|-- validate.py                   # validation entry point
|-- yolov5_from_scratch/
|   |-- cli/                      # argument parsing and application profiles
|   |-- runtime/                  # detector, geometry, tracking, counting, rendering
|   |-- training/                 # train/validate engines and shared meters
|   |-- models/                   # YOLO backbone, neck, and head
|   |-- data/                     # dataset and augmentation
|   |-- loss/                     # YOLO loss
|   |-- configs/                  # dataset and application YAML files
|   |-- tools/
|   |   |-- dataset/              # convert, inspect, sample, pre-label
|   |   |-- analysis/             # evaluate and summarize runs
|   |   `-- dev/                  # debugging helpers
|   `-- tests/                    # executable smoke tests and run_all
|-- datasets/                     # ignored, local datasets only
|-- artifacts/                    # ignored checkpoints, predictions, reports
`-- docs/
```

## Runtime Flow

`cli.predict` builds the application, then delegates to small reusable modules:

1. `runtime.detector` loads the checkpoint and runs YOLO inference.
2. `runtime.geometry` handles ROI, coordinate transforms, and output scaling.
3. `runtime.tracking` assigns persistent object IDs.
4. `runtime.counting` detects configured line crossings.
5. `runtime.rendering` draws boxes, IDs, counts, and performance.
6. `runtime.pipeline` reads media/camera streams and writes video plus CSV files.

The runtime does not assume traffic classes. The selected dataset config and
optional `--classes` argument define which objects are used.

## Artifact Policy

- `artifacts/checkpoints/`: only useful `best.pt`, `last.pt`, args, and history.
- `artifacts/predictions/`: user-facing outputs worth keeping.
- `artifacts/reports/`: CSV evaluation and validation reports.
- Temporary experiments should use a descriptive `--name` and be removed after
  comparison.
