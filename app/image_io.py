"""OpenCV image I/O supporting Unicode paths on Windows."""

from pathlib import Path

import numpy as np


def read_image(path: str | Path) -> np.ndarray | None:
    """Decode a BGR image, returning None for missing or invalid files."""
    import cv2

    try:
        payload = Path(path).read_bytes()
        if not payload:
            return None
        return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Encode using the target extension, then write with Python's path handling."""
    import cv2

    destination = Path(path)
    ok, encoded = cv2.imencode(destination.suffix, image)
    if not ok:
        raise OSError(f"图像编码失败: {destination}")
    destination.write_bytes(encoded.tobytes())
