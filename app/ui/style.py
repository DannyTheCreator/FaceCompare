"""Shared Qt appearance with a Chinese-capable font fallback."""

import os
import sys
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase


def configure_application(application) -> None:
    application.setStyle("Fusion")
    families = QFontDatabase.families()
    if sys.platform == "win32" and "Microsoft YaHei" not in families:
        # Offscreen/minimal Windows runtimes may not discover system fonts automatically.
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
            families = QFontDatabase.families()
    for family in ("Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC"):
        if family in families:
            application.setFont(QFont(family, 10))
            break
