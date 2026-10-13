"""Shared Qt appearance with a Chinese-capable font fallback."""

import os
import sys
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QLabel

STYLESHEET = """
    QMainWindow, QDialog { background: #f3f6f7; }
    QWidget { color: #20343b; font-size: 13px; }
    QLabel { background: transparent; border: none; }
    QLabel#brandMark { background: #0f766e; color: white; border-radius: 12px;
        font-size: 25px; font-weight: 700; }
    QLabel#brandTitle { font-size: 25px; font-weight: 700; color: #16363c; }
    QLabel[role="subtitle"] { color: #526c73; font-size: 12px; }
    QLabel[role="sectionTitle"] { font-size: 15px; font-weight: 600; color: #23434a; }
    QLabel[role="pageTitle"] { font-size: 23px; font-weight: 700; }
    QLabel[role="badge"] { background: #e4f2ee; color: #20756a;
        border: 1px solid #d4e9e2; border-radius: 14px; padding: 5px 12px; }
    QLabel[role="deviceBadge"] { background: #eef3f4; color: #576f76;
        border-radius: 10px; padding: 4px 10px; font-size: 11px; }
    QLabel[role="metricValue"] { color: #183b41; font-size: 25px; font-weight: 600; }
    QLabel#resultValue { font-size: 27px; font-weight: 700; color: #6b8289; }
    QLabel#resultValue[tone="success"] { color: #0f766e; }
    QLabel#resultValue[tone="warning"] { color: #b45309; }
    QLabel#resultValue[tone="error"] { color: #b91c1c; }
    QLabel#resultValue[tone="info"] { color: #1e6475; font-size: 22px; }
    QLabel[role="deviceBadge"][tone="success"] { background: #dcf2e8; color: #16705b; }
    QLabel[role="deviceBadge"][tone="warning"] { background: #fff2d6; color: #9a650f; }
    QLabel[role="deviceBadge"][tone="error"] { background: #fee9e5; color: #b33b2f; }
    QFrame[role="card"] { background: white; border: 1px solid #dfe8e9;
        border-radius: 14px; }
    QFrame[role="softCard"] { background: #eaf2f2; border: none; border-radius: 10px; }
    QFrame#header { background: transparent; border: none; }
    QScrollArea#rightScroll, QWidget#rightContent { background: transparent; border: none; }
    QPushButton { background: white; color: #36565f; border: 1px solid #d8e3e5;
        border-radius: 8px; padding: 9px 14px; font-weight: 500; }
    QPushButton:hover { background: #edf6f4; border-color: #b2d3cb; }
    QPushButton:pressed { background: #dfeee9; }
    QPushButton[role="primary"] { background: #0f766e; color: white; border-color: #0f766e; }
    QPushButton[role="primary"]:hover { background: #115f59; border-color: #115f59; }
    QPushButton[role="danger"] { color: #b5483b; background: #fff8f6; border-color: #ecd8d3; }
    QPushButton[role="danger"]:hover { background: #ffebe6; }
    QPushButton:disabled { color: #a6b5b9; background: #f1f5f5; border-color: #e6eded; }
    QPushButton[role="primary"]:disabled { background: #dbeae6; color: #99b5ae;
        border-color: #dbeae6; }
    QPushButton:focus, QComboBox:focus, QLineEdit:focus { border: 1px solid #0f766e; }
    QComboBox, QLineEdit { background: #fcfdfd; color: #29464d; border: 1px solid #d7e3e5;
        border-radius: 8px; padding: 9px 10px; selection-background-color: #c7e7de; }
    QComboBox::drop-down { border: none; width: 24px; }
    QComboBox QAbstractItemView { background: white; border: 1px solid #d7e3e5;
        selection-background-color: #e2f0ec; selection-color: #1b594e; padding: 4px; }
    QLineEdit:disabled { background: #f1f5f5; color: #a6b5b9; }
    QCheckBox { spacing: 8px; color: #526d75; padding: 4px 0; }
    QCheckBox::indicator { width: 17px; height: 17px; }
    QTabWidget::pane { border: none; background: transparent; }
    QTabBar::tab { background: #edf3f3; color: #789096; border: none;
        border-radius: 7px; margin: 0 6px 10px 0; padding: 8px 17px; }
    QTabBar::tab:selected { background: #e0f0eb; color: #166a5d; font-weight: 600; }
    QTabBar::tab:hover { color: #166a5d; }
    QTreeWidget { background: white; alternate-background-color: #f7fafa;
        border: 1px solid #e6eded; border-radius: 8px; outline: none;
        selection-background-color: #e0f0eb; selection-color: #1b594e; }
    QTreeWidget::item { padding: 7px 4px; border: none; }
    QHeaderView::section { background: #f3f7f7; color: #566f76; border: none;
        border-bottom: 1px solid #e6eded; padding: 8px 5px; font-size: 11px; }
    QPlainTextEdit { background: #f5f8f8; color: #536d75; border: none;
        border-radius: 8px; padding: 8px; font-size: 11px; }
    QToolButton { background: transparent; border: none; color: #73878e;
        padding: 4px; text-align: left; }
    QProgressBar { background: #e8f0ee; border: none; border-radius: 5px;
        min-height: 8px; max-height: 8px; }
    QProgressBar::chunk { background: #31917b; border-radius: 5px; }
    QSplitter::handle { background: transparent; width: 14px; }
    QStatusBar { color: #809298; background: #f3f6f7; padding: 2px 16px; font-size: 11px; }
    QStatusBar::item { border: none; }
    QScrollBar:vertical { background: transparent; width: 7px; margin: 3px 0; }
    QScrollBar::handle:vertical { background: #d7e2e3; border-radius: 3px; min-height: 25px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""


def apply_theme(widget) -> None:
    widget.setStyleSheet(STYLESHEET)


def set_tone(widget, tone: str) -> None:
    widget.setProperty("tone", tone)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class ImagePreview(QLabel):
    """Paint a camera empty state, while retaining QLabel's image and text APIs."""

    def __init__(self, text: str, subtitle: str = "", parent=None):
        super().__init__(text, parent)
        self.subtitle = subtitle
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setStyleSheet("background: #142d33; border-radius: 12px; color: #a4c2c3;")

    def paintEvent(self, event) -> None:
        pixmap = self.pixmap()
        if pixmap is not None and not pixmap.isNull():
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        gradient = QLinearGradient(bounds.topLeft(), bounds.bottomRight())
        gradient.setColorAt(0, QColor("#19383e"))
        gradient.setColorAt(1, QColor("#10272d"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(bounds, 12, 12)
        large = self.width() > 400 and self.height() > 220
        center_x, center_y = self.width() / 2, self.height() / 2
        if large:
            painter.setPen(QPen(QColor("#203d43"), 1))
            for x in range(36, self.width(), 48):
                for y in range(30, self.height(), 48):
                    painter.drawPoint(x, y)
            painter.setPen(QPen(QColor("#47746f"), 2))
            painter.setBrush(QColor("#21464a"))
            painter.drawRoundedRect(QRectF(center_x - 35, center_y - 77, 70, 52), 13, 13)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#89bcb2"), 2))
            painter.drawEllipse(QRectF(center_x - 12, center_y - 63, 24, 24))
            painter.drawLine(int(center_x), int(center_y - 25), int(center_x), int(center_y - 15))
            painter.drawLine(
                int(center_x - 15), int(center_y - 15), int(center_x + 15), int(center_y - 15)
            )
        font = QFont(self.font())
        font.setPixelSize(17 if large else 12)
        font.setWeight(QFont.Weight.Medium)
        painter.setFont(font)
        painter.setPen(QColor("#b8d1d0"))
        painter.drawText(
            QRectF(24, center_y + (8 if large else -20), self.width() - 48, 45),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            self.text(),
        )
        if large and self.subtitle:
            font.setPixelSize(12)
            font.setWeight(QFont.Weight.Normal)
            painter.setFont(font)
            painter.setPen(QColor("#89abaa"))
            painter.drawText(
                QRectF(24, center_y + 48, self.width() - 48, 40),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self.subtitle,
            )
        painter.end()


def configure_application(application) -> None:
    application.setStyle("Fusion")
    application.setStyleSheet(STYLESHEET)
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
