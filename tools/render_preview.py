"""Render the real startup UI with a temporary empty gallery and no camera."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication

from app.config import StorageConfig, load_config
from app.service import FaceComparisonService
from app.ui.main_window import MainWindow
from app.ui.style import configure_application


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs/assets/ui-preview.png")
    args = parser.parse_args()
    application = QApplication([])
    configure_application(application)
    with tempfile.TemporaryDirectory(prefix="facecompare-preview-") as temporary:
        root = Path(temporary)
        config = replace(
            load_config(), storage=StorageConfig(root / "gallery.db", root / "samples")
        )
        window = MainWindow(FaceComparisonService(config))
        window.show()
        application.processEvents()
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(args.output)):
                raise OSError(f"Cannot save preview: {args.output}")
        finally:
            window.close()
            application.processEvents()
    print(f"Rendered empty-gallery preview: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
