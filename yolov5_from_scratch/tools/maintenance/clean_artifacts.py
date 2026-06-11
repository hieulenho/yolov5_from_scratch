import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from yolov5_from_scratch.paths import ARTIFACTS_DIR


ALLOWED_CATEGORIES = ("predictions", "reports", "training")


def parse_args():
    parser = argparse.ArgumentParser(
        description="List or remove old generated artifact directories"
    )
    parser.add_argument(
        "--category",
        choices=ALLOWED_CATEGORIES,
        default="predictions",
    )
    parser.add_argument(
        "--older-than-days",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--keep",
        nargs="*",
        default=[],
        help="artifact directory names that must never be removed",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform deletion; default mode only prints candidates",
    )
    return parser.parse_args()


def cleanup_candidates(root, older_than_days, keep):
    cutoff = datetime.now() - timedelta(days=older_than_days)
    keep = set(keep)
    if not root.is_dir():
        return []
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_dir()
        and path.name not in keep
        and datetime.fromtimestamp(path.stat().st_mtime) < cutoff
    ]


def safe_remove(path, category_root):
    resolved = path.resolve()
    root = category_root.resolve()
    if resolved.parent != root:
        raise RuntimeError(f"Refusing to remove path outside category root: {resolved}")
    shutil.rmtree(resolved)


def main():
    args = parse_args()
    if args.older_than_days < 0:
        raise ValueError("--older-than-days must be 0 or greater")

    category_root = ARTIFACTS_DIR / args.category
    candidates = cleanup_candidates(
        category_root,
        args.older_than_days,
        args.keep,
    )
    mode = "DELETE" if args.apply else "DRY-RUN"
    print(f"mode={mode} category={category_root.resolve()}")
    for path in candidates:
        print(path.resolve())
        if args.apply:
            safe_remove(path, category_root)
    print(f"candidates={len(candidates)}")


if __name__ == "__main__":
    main()
