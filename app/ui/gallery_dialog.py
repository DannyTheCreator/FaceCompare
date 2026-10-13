from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ..service import FaceComparisonService
from ..workers import TaskWorker


class GalleryDialog(QDialog):
    def __init__(self, service: FaceComparisonService, parent=None):
        super().__init__(parent)
        self.service = service
        self.worker = None
        self.setWindowTitle("标准库管理")
        self.resize(800, 520)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["姓名 / 样本", "人员ID", "样本数 / 会话", "创建时间"])
        self.tree.setAlternatingRowColors(True)
        self.preview = QLabel("选择样本可预览")
        self.preview.setMinimumSize(280, 280)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background: #202020; color: #ddd;")

        self.delete_button = QPushButton("删除所选")
        self.delete_button.setEnabled(False)
        self.status_label = QLabel("选择人员或样本后可删除；删除人员会同时删除全部样本。")
        self.status_label.setWordWrap(True)
        self.check_button = QPushButton("检查库完整性")
        self.cleanup_button = QPushButton("重试待清理图片")
        self.refresh_button = QPushButton("刷新")
        button_row = QHBoxLayout()
        button_row.addWidget(self.refresh_button)
        button_row.addWidget(self.check_button)
        button_row.addWidget(self.cleanup_button)
        button_row.addStretch()
        button_row.addWidget(self.delete_button)

        content = QHBoxLayout()
        content.addWidget(self.tree, 3)
        content.addWidget(self.preview, 2)
        layout = QVBoxLayout(self)
        layout.addLayout(content)
        layout.addWidget(self.status_label)
        layout.addLayout(button_row)

        self.tree.currentItemChanged.connect(self._selection_changed)
        self.delete_button.clicked.connect(self._delete_selected)
        self.refresh_button.clicked.connect(self.reload)
        self.check_button.clicked.connect(self._check_integrity)
        self.cleanup_button.clicked.connect(self._cleanup)
        self.reload()

    def reload(self) -> None:
        self.tree.clear()
        self.preview.clear()
        self.preview.setText("选择样本可预览")
        self.delete_button.setEnabled(False)
        samples_by_person = {
            person.person_id: self.service.repository.list_samples(person.person_id)
            for person in self.service.repository.list_persons()
        }
        for person in self.service.repository.list_persons():
            item = QTreeWidgetItem(
                [person.name, str(person.person_id), str(person.sample_count), person.created_at]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, ("person", person.person_id))
            self.tree.addTopLevelItem(item)
            for sample in samples_by_person[person.person_id]:
                child = QTreeWidgetItem(
                    [
                        sample.sample_id[:12],
                        "",
                        sample.session_id[:12],
                        sample.created_at,
                    ]
                )
                child.setData(
                    0,
                    Qt.ItemDataRole.UserRole,
                    ("sample", sample.sample_id, str(sample.image_path)),
                )
                item.addChild(child)
            item.setExpanded(True)
        for column in range(4):
            self.tree.resizeColumnToContents(column)
        if not self.tree.topLevelItemCount():
            self.status_label.setText("标准库为空。返回主界面，开启摄像头后使用“人员录入”。")

    def _selection_changed(self, current, _previous) -> None:
        self.delete_button.setEnabled(current is not None and self.worker is None)
        self.preview.clear()
        self.preview.setText("选择样本可预览")
        if current is None:
            return
        payload = current.data(0, Qt.ItemDataRole.UserRole)
        if not payload or payload[0] != "sample":
            self.preview.setText("选择样本可预览")
            self.preview.setPixmap(QPixmap())
            return
        pixmap = QPixmap(payload[2])
        if pixmap.isNull():
            self.preview.setText("样本文件不存在")
            return
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _delete_selected(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            return
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if not payload:
            return
        kind = payload[0]
        detail = "该人员及其全部样本" if kind == "person" else "该样本"
        answer = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除{detail}吗？此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        def operation():
            if kind == "person":
                self.service.delete_person(int(payload[1]))
            else:
                self.service.delete_sample(str(payload[1]))
            remaining = self.service.repository.cleanup_pending()
            return f"标准库已更新；待清理图片 {remaining} 张。" + (
                "文件可能被占用，可稍后重试。" if remaining else ""
            )

        self._run(operation)

    def _run(self, operation):
        if self.worker is not None:
            return
        worker = TaskWorker(operation, self)
        self.worker = worker
        self.status_label.setText("正在处理，请稍候…")
        for widget in (
            self.tree,
            self.delete_button,
            self.refresh_button,
            self.check_button,
            self.cleanup_button,
        ):
            widget.setEnabled(False)
        worker.succeeded.connect(self._done)
        worker.failed.connect(lambda message: self.status_label.setText(f"操作失败：{message}"))
        worker.finished.connect(self._finished)
        worker.start()

    def _done(self, message):
        self.reload()
        self.status_label.setText(str(message))

    def _finished(self):
        worker = self.worker
        self.worker = None
        worker.deleteLater()
        for widget in (self.tree, self.refresh_button, self.check_button, self.cleanup_button):
            widget.setEnabled(True)
        self.delete_button.setEnabled(self.tree.currentItem() is not None)

    def _check_integrity(self):
        def operation():
            result = self.service.repository.integrity_report(
                self.service.model_hash, self.service.config.preprocess_version
            )
            labels = {
                "missing_images": "缺失图片",
                "unsafe_paths": "越界路径",
                "invalid_features": "损坏特征",
                "incompatible_samples": "不兼容样本",
                "orphan_images": "未引用图片",
                "pending_deletions": "待清理图片",
            }
            return "；".join(f"{labels[key]} {value}" for key, value in result.items())

        self._run(operation)

    def _cleanup(self):
        self._run(
            lambda: f"清理后仍有 {self.service.repository.cleanup_pending()} 张待处理；未引用图片不会自动删除。"
        )

    def reject(self):
        if self.worker is None:
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)
