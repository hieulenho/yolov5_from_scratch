import os
import subprocess
import sys


TEST_MODULES = [
    "yolov5_from_scratch.tests.test_model",
    "yolov5_from_scratch.tests.test_runtime",
    "yolov5_from_scratch.tests.test_dataset",
    "yolov5_from_scratch.tests.test_model_with_data",
    "yolov5_from_scratch.tests.test_loss",
    "yolov5_from_scratch.tests.test_prelabel",
    "yolov5_from_scratch.tests.test_training_transfer",
]


def main():
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for module in TEST_MODULES:
        print(f"\n===== {module} =====", flush=True)
        subprocess.run(
            [sys.executable, "-m", module],
            check=True,
            env=env,
        )
    print(f"\nAll {len(TEST_MODULES)} test modules passed.", flush=True)


if __name__ == "__main__":
    main()
