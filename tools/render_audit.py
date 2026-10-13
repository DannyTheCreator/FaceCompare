"""Capture real Qt widgets in an isolated empty gallery; never opens a camera."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from app.config import StorageConfig, load_config
from app.service import FaceComparisonService
from app.ui.enroll_dialog import EnrollDialog
from app.ui.gallery_dialog import GalleryDialog
from app.ui.main_window import MainWindow
from app.ui.style import configure_application


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Model/config paths; gallery stays temporary")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    application = QApplication([])
    configure_application(application)

    def capture(widget, name):
        widget.show()
        application.processEvents()
        if not widget.grab().save(str(args.output / name)):
            raise OSError(name)

    with tempfile.TemporaryDirectory(prefix="facecompare-audit-") as temporary:
        root = Path(temporary)
        cfg = replace(
            load_config(args.config), storage=StorageConfig(root / "gallery.db", root / "samples")
        )
        service = FaceComparisonService(cfg)
        window = MainWindow(service)
        try:
            capture(window, "01-start.png")
            window.camera_combo.setFocus()
            window.camera_combo.setCurrentIndex(window.camera_combo.count() - 1)
            capture(window, "02-device.png")
            enroll = EnrollDialog(service, lambda: None, window)
            enroll.capture_button.click()
            capture(enroll, "03-enroll.png")
            enroll.close()
            gallery = GalleryDialog(service, window)
            capture(gallery, "05-gallery.png")
            gallery.close()
            # Step 4: real recognition of an empty image, not a fabricated face result.
            import time

            import numpy as np

            from app.types import FramePacket

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            packet = FramePacket(1, time.perf_counter(), frame, window.stream_id)
            window._on_frame(packet)
            result = service.recognize(
                frame,
                frame_id=1,
                captured_monotonic=packet.captured_monotonic,
                write_log=False,
                stream_id=window.stream_id,
            )
            window._on_result(result, frame)
            capture(window, "04-recognition.png")
        finally:
            window.close()
            application.processEvents()
    print(f"Saved 5 isolated Qt screenshots to {args.output}")


if __name__ == "__main__":
    main()
