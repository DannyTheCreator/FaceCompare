from __future__ import annotations

import uuid
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ..service import FaceComparisonService
from ..types import PreparedSample
from ..workers import TaskWorker


class EnrollDialog(QDialog):
    def __init__(
        self,
        service: FaceComparisonService,
        frame_provider: Callable[[], np.ndarray | None],
        parent=None,
    ):
        super().__init__(parent)
        self.service = service
        self.frame_provider = frame_provider
        self.samples: list[PreparedSample] = []
        self.worker: TaskWorker | None = None
        self._saved = False
        self.setWindowTitle("人员录入 / 补充样本")
        self.setMinimumWidth(500)

        self.person_combo = QComboBox()
        self.person_combo.addItem("新增人员", None)
        for person in service.repository.list_persons():
            self.person_combo.addItem(
                f"{person.name}（ID {person.person_id}，{person.sample_count} 张）",
                person.person_id,
            )
        self.name_edit = QLineEdit()
        self.external_id_edit = QLineEdit()
        self.external_id_edit.setPlaceholderText("可选，例如学号；姓名不作为主键")
        self.session_edit = QLineEdit(uuid.uuid4().hex[:12])
        self.session_edit.setToolTip("同一次拍摄共用会话编号；注册、验证、测试必须使用不同会话。")

        form = QFormLayout()
        form.addRow("录入方式", self.person_combo)
        form.addRow("姓名", self.name_edit)
        form.addRow("内部编号", self.external_id_edit)
        form.addRow("拍摄会话", self.session_edit)

        self.hint = QLabel(
            "请依次采集正面、轻微左右转头和自然表情。每次只保留一张脸，避免连续重复帧。"
        )
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #555;")
        self.count_label = QLabel()
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label = QLabel("等待采集")
        self.status_label.setWordWrap(True)

        self.capture_button = QPushButton("采集当前画面")
        self.undo_button = QPushButton("撤销上一张")
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.capture_button)
        buttons_row.addWidget(self.undo_button)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Save).setText("保存录入")
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.hint)
        layout.addWidget(self.count_label)
        layout.addWidget(self.status_label)
        layout.addLayout(buttons_row)
        layout.addWidget(self.button_box)

        self.person_combo.currentIndexChanged.connect(self._person_changed)
        self.name_edit.textChanged.connect(self._update_state)
        self.session_edit.textChanged.connect(self._update_state)
        self.capture_button.clicked.connect(self._capture)
        self.undo_button.clicked.connect(self._undo)
        self.button_box.accepted.connect(self._save)
        self.button_box.rejected.connect(self.reject)
        self._person_changed()
        self._update_state()

    def _person_changed(self) -> None:
        # Samples collected for one identity must not move to another identity.
        self.samples.clear()
        self.status_label.setText("等待采集")
        is_new = self.person_combo.currentData() is None
        self.name_edit.setEnabled(is_new)
        self.external_id_edit.setEnabled(is_new)
        self._update_state()

    def _required_count(self) -> int:
        if self.person_combo.currentData() is None:
            return self.service.config.enrollment.minimum_samples
        return 1

    def _update_state(self) -> None:
        minimum = self._required_count()
        target = self.service.config.enrollment.target_samples
        self.count_label.setText(
            f"已采集 {len(self.samples)} 张；最低 {minimum} 张，建议目标 {target} 张"
        )
        busy = self.worker is not None
        self.session_edit.setEnabled(not busy)
        self.person_combo.setEnabled(not busy)
        self.capture_button.setEnabled(not busy)
        self.undo_button.setEnabled(bool(self.samples) and not busy)
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setEnabled(not busy)
        can_save = len(self.samples) >= minimum and bool(self.session_edit.text().strip())
        if self.person_combo.currentData() is None:
            can_save = can_save and bool(self.name_edit.text().strip())
        self.button_box.button(QDialogButtonBox.StandardButton.Save).setEnabled(
            can_save and not busy
        )

    def _capture(self) -> None:
        frame = self.frame_provider()
        if frame is None:
            self.status_label.setText("摄像头尚未提供画面，请先在主界面开启摄像头")
            return
        if self.worker is not None:
            return
        self.status_label.setText("正在检查画面…")
        self._run_task(lambda: self.service.prepare_enrollment_frame(frame), self._sample_ready)

    def _run_task(self, operation, callback) -> None:
        worker = TaskWorker(operation, self)
        self.worker = worker
        worker.succeeded.connect(callback)
        worker.failed.connect(self.status_label.setText)
        worker.finished.connect(self._task_finished)
        self._update_state()
        worker.start()

    def _task_finished(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self._update_state()
        if self._saved:
            self.accept()

    def _sample_ready(self, payload) -> None:
        sample, quality = payload
        if sample is None:
            self.status_label.setText(quality.message)
            return
        if self.samples:
            similarity = max(float(np.dot(sample.feature, item.feature)) for item in self.samples)
            if similarity > 0.9995:
                self.status_label.setText("该画面与已采样本过于相似，请改变姿态或表情后重试")
                return
        self.samples.append(sample)
        self.status_label.setText(
            f"采集成功：清晰度 {quality.blur_variance:.1f}，亮度 {quality.brightness:.1f}"
        )

    def _undo(self) -> None:
        if self.samples:
            self.samples.pop()
            self.status_label.setText("已撤销上一张样本")
        self._update_state()

    def _save(self) -> None:
        if self.worker is not None:
            return
        samples = list(self.samples)
        name = self.name_edit.text().strip() or None
        external_id = self.external_id_edit.text().strip() or None
        existing = self.person_combo.currentData()
        session_id = self.session_edit.text().strip()
        self._saved = False

        def save():
            return self.service.commit_enrollment(
                samples,
                name=name,
                external_id=external_id,
                existing_person_id=existing,
                session_id=session_id,
            )

        self.status_label.setText("正在保存…")
        self._run_task(save, self._save_ready)

    def _save_ready(self, person_id: int) -> None:
        self._saved = True
        self.status_label.setText(f"录入完成，人员 ID：{person_id}")
        # _task_finished closes the dialog after the worker exits.

    def reject(self) -> None:
        if self.worker is None:
            super().reject()

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)
