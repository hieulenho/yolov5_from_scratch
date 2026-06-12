from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v"}


def increment_path(path, exist_ok=False):
    path = Path(path)
    if exist_ok or not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.name}{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate output directory near {path}")


def iter_sources(source):
    if isinstance(source, int):
        return [source]

    source_text = str(source)
    if source_text.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return [source_text]
    if source_text.isdigit():
        return [int(source_text)]

    path = Path(source_text)
    if path.is_dir():
        files = [
            item
            for item in sorted(path.rglob("*"))
            if item.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        ]
        if not files:
            raise FileNotFoundError(f"No supported media found in: {path}")
        return files
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Source not found: {source}")


def output_name(source, suffix):
    if isinstance(source, int):
        return f"camera_{source}{suffix}"
    source_text = str(source)
    if "://" in source_text:
        return f"stream{suffix}"
    return f"{Path(source_text).stem}{suffix}"


def source_label(source):
    if isinstance(source, int):
        return f"camera:{source}"
    source_text = str(source)
    if "://" not in source_text:
        return source_text

    parsed = urlsplit(source_text)
    hostname = parsed.hostname or ""
    if parsed.port is not None:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit(
        (
            parsed.scheme,
            hostname,
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


def open_capture(source):
    capture_source = source if isinstance(source, int) else str(source)
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video or camera source: {source_label(source)}"
        )
    return capture
