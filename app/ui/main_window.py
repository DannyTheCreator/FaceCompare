from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..service import FaceComparisonService
from ..types import FramePacket, RecognitionResult, SceneResult
from ..workers import CameraWorker, InferenceWorker
from .enroll_dialog import EnrollDialog
from .gallery_dialog import GalleryDialog


def _pixmap_from_bgr(image: np.ndarray, target_size) -> QPixmap:
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    height, width, channels = rgb.shape
    qimage = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage).scaled(
        target_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class MainWindow(QMainWindow):
    def __init__(self, service: FaceComparisonService):
        super().__init__()
        self.service = service
        self.camera_worker: CameraWorker | None = None
        self.inference_worker = InferenceWorker(service, self)
        self.inference_worker.result_ready.connect(self._on_result)
        self.inference_worker.inference_error.connect(self._on_inference_error)
        self.inference_worker.start()
        self.current_packet: FramePacket | None = None
        self.stream_id = 0
        self._accept_results = True
        self._mode_generation = 0
        self.last_submit_monotonic = 0.0
        self._preview_times = []
        self._result_times = []
        self._last_queue_ms = None

        self.setWindowTitle("FaceCompare · 人脸比对")
        self.resize(1320, 880)
        self._build_ui()
        self._refresh_gallery_status()
        self.result_timer = QTimer(self)
        self.result_timer.setSingleShot(True)
        self.result_timer.timeout.connect(lambda: self._clear_result("结果已过期，等待重新识别"))
        self.rate_timer = QTimer(self)
        self.rate_timer.timeout.connect(self._update_rates)
        self.rate_timer.start(1000)

    def _build_ui(self) -> None:
        self.preview = QLabel("请选择设备并开启摄像头")
        self.preview.setMinimumSize(560, 420)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background: #151515; color: #ddd; font-size: 18px;")

        self.camera_combo = QComboBox()
        self.camera_combo.setAccessibleName("摄像头编号")
        self.camera_combo.setToolTip(
            "编号来自配置，并不表示设备已检测到。打开后请用实时画面确认设备。"
        )
        for index in self.service.config.camera.probe_indices:
            self.camera_combo.addItem(f"摄像头 {index}", index)
        default_index = self.camera_combo.findData(self.service.config.camera.default_index)
        if default_index >= 0:
            self.camera_combo.setCurrentIndex(default_index)
        self.open_button = QPushButton("开启摄像头")
        self.close_button = QPushButton("关闭摄像头")
        self.close_button.setEnabled(False)
        self.capture_button = QPushButton("抓拍识别")
        self.capture_button.setEnabled(False)
        self.continuous_check = QCheckBox("连续识别")
        self.scene_check = QCheckBox("多人识别 / 跟踪")
        self.scene_check.setToolTip("轻量IoU轨迹关联；连续多次匹配后确认姓名，快速交叉可能换号")

        camera_row = QHBoxLayout()
        camera_row.addWidget(self.camera_combo)
        camera_row.addWidget(self.open_button)
        camera_row.addWidget(self.close_button)
        camera_row.addWidget(self.capture_button)
        camera_row.addWidget(self.continuous_check)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.scene_check)
        mode_row.addStretch()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.analysis_preview = QLabel("识别分析快照，等待识别")
        self.analysis_preview.setMinimumSize(560, 420)
        self.analysis_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analysis_preview.setStyleSheet("background: #151515; color: #ddd; font-size: 18px;")
        self.views = QTabWidget()
        self.views.addTab(self.preview, "实时画面")
        self.views.addTab(self.analysis_preview, "识别分析快照")
        left_layout.addWidget(self.views, 1)
        left_layout.addLayout(camera_row)
        left_layout.addLayout(mode_row)
        self.guide_label = QLabel("① 选择并开启设备  →  ② 录入人员  →  ③ 抓拍或连续识别")
        self.guide_label.setWordWrap(True)
        left_layout.addWidget(self.guide_label)
        self.rate_label = QLabel("预览 FPS —  |  有效结果 — 次/秒  |  排队 — ms")
        self.rate_label.setToolTip(
            "最近5秒的预览与有效结果到达速率；后台识别吞吐请使用视频基准测量。"
        )
        self.rate_label.setWordWrap(True)
        left_layout.addWidget(self.rate_label)

        result_group = QGroupBox("识别结果")
        result_form = QFormLayout(result_group)
        self.result_label = QLabel("等待识别")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.score_label = QLabel("—")
        self.threshold_label = QLabel(f"{self.service.matcher.threshold:.3f}")
        self.latency_label = QLabel("—")
        self.updated_label = QLabel("—")
        result_form.addRow("状态", self.result_label)
        result_form.addRow("余弦相似度", self.score_label)
        result_form.addRow("当前阈值", self.threshold_label)
        result_form.addRow("算法 / 端到端", self.latency_label)
        result_form.addRow("更新时间", self.updated_label)
        self.result_preview = QLabel("识别快照")
        self.result_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_preview.setMinimumSize(220, 130)
        self.result_preview.setMaximumHeight(190)
        result_form.addRow(self.result_preview)

        self.enroll_button = QPushButton("人员录入 / 补充样本")
        self.gallery_button = QPushButton("标准库管理")
        self.gallery_status = QLabel()
        self.gallery_status.setWordWrap(True)
        actions = QGroupBox("标准库")
        actions_layout = QVBoxLayout(actions)
        actions_layout.addWidget(self.gallery_status)
        actions_layout.addWidget(self.enroll_button)
        actions_layout.addWidget(self.gallery_button)

        self.candidate_tree = QTreeWidget()
        self.candidate_tree.setHeaderLabels(["轨迹 / 排名", "结果 / 候选", "相似度", "连续确认"])
        for column in (0, 2, 3):
            self.candidate_tree.header().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.candidate_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.candidate_tree.setAlternatingRowColors(True)
        self.candidate_tree.setMinimumHeight(150)
        self.candidate_tree.setToolTip("候选排序只用于解释匹配；未知或待确认状态不会因此接受身份")
        candidate_group = QGroupBox("人脸与候选解释")
        candidate_layout = QVBoxLayout(candidate_group)
        candidate_layout.addWidget(self.candidate_tree)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.log)

        right = QWidget()
        right.setMinimumWidth(460)
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(result_group)
        right_layout.addWidget(actions)
        right_layout.addWidget(candidate_group)
        right_layout.addWidget(log_group, 1)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([800, 500])
        self.setCentralWidget(splitter)
        self.setStyleSheet("""
            QMainWindow, QDialog { background: #f5f7fb; }
            QGroupBox { background: white; border: 1px solid #dde3ee;
                border-radius: 10px; margin-top: 14px; padding: 18px 12px 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; color: #334155; }
            QPushButton { background: #e8eef9; color: #1d365e; border: 1px solid #ccd8ec;
                border-radius: 6px; padding: 9px 12px; }
            QPushButton:hover { background: #d8e5fb; }
            QPushButton:disabled { color: #9aa5b5; background: #f1f4f9; }
            QComboBox, QLineEdit { padding: 7px; border: 1px solid #ccd8ec; border-radius: 5px; }
            QPlainTextEdit { border: none; background: #f8fafc; color: #475569; }
        """)
        self.statusBar().showMessage("模型已加载，等待开启摄像头")

        self.open_button.clicked.connect(self._open_camera)
        self.close_button.clicked.connect(self._stop_camera)
        self.capture_button.clicked.connect(self._capture)
        self.enroll_button.clicked.connect(self._open_enrollment)
        self.gallery_button.clicked.connect(self._open_gallery)
        self.scene_check.toggled.connect(self._mode_changed)

    def _mode_changed(self) -> None:
        self._mode_generation += 1
        self.inference_worker.clear_pending()
        self._clear_result()
        self.last_submit_monotonic = 0.0
        self.views.setCurrentIndex(1 if self.scene_check.isChecked() else 0)

    def _submit(self, packet: FramePacket) -> None:
        self.inference_worker.submit_latest(
            replace(
                packet,
                mode="scene" if self.scene_check.isChecked() else "single",
                mode_generation=self._mode_generation,
            )
        )

    def _append_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{stamp}] {text}")

    def _open_camera(self) -> None:
        if not self._stop_camera():
            return
        self.stream_id += 1
        cfg = self.service.config.camera
        worker = CameraWorker(
            self.camera_combo.currentData(),
            cfg.width,
            cfg.height,
            cfg.preview_fps,
            self,
            stream_id=self.stream_id,
        )
        worker.frame_ready.connect(self._on_frame)
        worker.camera_opened.connect(self._on_camera_opened)
        worker.camera_error.connect(self._on_camera_error)
        worker.finished.connect(self._on_camera_finished)
        self.camera_worker = worker
        self.open_button.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self.statusBar().showMessage("正在打开摄像头…")
        worker.start()

    def _stop_camera(self) -> bool:
        self.stream_id += 1
        self.inference_worker.clear_pending()
        self.current_packet = None
        self._preview_times.clear()
        self._result_times.clear()
        self._last_queue_ms = None
        self._clear_result()
        worker = self.camera_worker
        if worker is not None:
            worker.request_stop()
            if not worker.wait(0):
                self.capture_button.setEnabled(False)
                self.close_button.setEnabled(False)
                self.open_button.setEnabled(False)
                self.statusBar().showMessage("正在等待摄像头释放…")
                return False
            self.camera_worker = None
            worker.deleteLater()
        self.capture_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self.open_button.setEnabled(True)
        self.camera_combo.setEnabled(True)
        self.preview.clear()
        self.preview.setText("请选择设备并开启摄像头")
        self._preview_times.clear()
        self._result_times.clear()
        self.rate_label.setText("预览 FPS —  |  有效结果 — 次/秒  |  排队 — ms")
        return True

    def _clear_result(self, message: str = "等待识别") -> None:
        if hasattr(self, "result_timer"):
            self.result_timer.stop()
        self.result_label.setText(message)
        self.result_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #64748b;")
        for label in (self.score_label, self.latency_label, self.updated_label):
            label.setText("—")
        self.result_preview.clear()
        self.result_preview.setText("识别快照")
        self.candidate_tree.clear()
        self.analysis_preview.clear()
        self.analysis_preview.setText("识别分析快照，等待识别")

    def _on_camera_opened(self, width: int, height: int) -> None:
        if (
            self.sender() is not self.camera_worker
            or self.camera_worker.stream_id != self.stream_id
        ):
            return
        self.capture_button.setEnabled(True)
        self.close_button.setEnabled(True)
        self.statusBar().showMessage(f"摄像头已连接，实际分辨率 {width}×{height}")
        self._append_log(f"摄像头已连接，实际分辨率 {width}×{height}")

    def _on_camera_error(self, message: str) -> None:
        if (
            self.sender() is not self.camera_worker
            or self.camera_worker.stream_id != self.stream_id
        ):
            return
        self.current_packet = None
        self.stream_id += 1
        self.inference_worker.clear_pending()
        self._clear_result("摄像头异常")
        self._append_log(message)
        self.statusBar().showMessage(message)
        QMessageBox.warning(self, "摄像头异常", message)

    def _on_camera_finished(self) -> None:
        if self.sender() is not self.camera_worker:
            return
        self.stream_id += 1
        self.current_packet = None
        self.inference_worker.clear_pending()
        worker = self.camera_worker
        self.camera_worker = None
        worker.deleteLater()
        self._clear_result()
        self.capture_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self.open_button.setEnabled(True)
        self.camera_combo.setEnabled(True)
        self.preview.clear()
        self.preview.setText("设备已断开，请选择设备并重新开启")

    def _on_frame(self, packet: FramePacket) -> None:
        if packet.stream_id != self.stream_id:
            return
        self.current_packet = packet
        self._preview_times.append(time.perf_counter())
        self._update_rates()
        self.preview.setPixmap(_pixmap_from_bgr(packet.image, self.preview.size()))
        if self.continuous_check.isChecked():
            now = time.perf_counter()
            interval = self.service.config.camera.recognition_interval_ms / 1000
            if now - self.last_submit_monotonic >= interval:
                self.last_submit_monotonic = now
                self._submit(packet)

    def _capture(self) -> None:
        if self.current_packet is None:
            return
        self._submit(self.current_packet)
        self.statusBar().showMessage("已提交抓拍识别")

    def _on_result(
        self, result: RecognitionResult | SceneResult, image: np.ndarray | None = None
    ) -> None:
        ttl_seconds = self.service.config.camera.result_ttl_ms / 1000
        age_seconds = time.perf_counter() - result.captured_monotonic
        if (
            not self._accept_results
            or self.current_packet is None
            or result.stream_id != self.stream_id
            or result.mode_generation != self._mode_generation
            or result.gallery_version != self.service.repository.gallery_version()
            or age_seconds >= ttl_seconds
        ):
            return
        self._result_times.append(time.perf_counter())
        self._update_rates(result.timings_ms.get("queue_wait"))
        if isinstance(result, SceneResult):
            self._on_scene(result, image, ttl_seconds - age_seconds)
            return
        if result.status == "matched":
            label = result.person_name or "匹配成功"
            color = "#137333"
        elif result.status == "unknown":
            label = "未知人员"
            color = "#b06000"
        else:
            label = result.message
            color = "#b3261e"
        self.result_label.setText(label)
        self.result_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
        self.score_label.setText("—" if result.score is None else f"{result.score:.4f}")
        algorithm_ms = result.timings_ms.get("algorithm_total", 0.0)
        self.latency_label.setText(f"{algorithm_ms:.1f} ms / {result.end_to_end_ms:.1f} ms")
        self.updated_label.setText(time.strftime("%H:%M:%S"))
        self.candidate_tree.clear()
        self._add_candidates(result)
        for warning in result.warnings:
            self._append_log(warning)
        self._append_log(
            f"frame={result.frame_id} status={result.status} "
            f"score={result.score if result.score is not None else '-'} "
            f"algorithm={algorithm_ms:.1f}ms e2e={result.end_to_end_ms:.1f}ms"
        )
        self.result_timer.start(max(1, int((ttl_seconds - age_seconds) * 1000)))
        if image is not None:
            import cv2

            snapshot = image.copy()
            if result.face is not None:
                x, y, w, h = (int(round(v)) for v in result.face[:4])
                cv2.rectangle(snapshot, (x, y), (x + w, y + h), (40, 210, 90), 2)
            self.result_preview.setPixmap(_pixmap_from_bgr(snapshot, self.result_preview.size()))
            self.analysis_preview.setPixmap(
                _pixmap_from_bgr(snapshot, self.analysis_preview.size())
            )

    def _add_candidates(self, result: RecognitionResult, parent=None) -> None:
        for rank, candidate in enumerate(result.candidates, start=1):
            item = QTreeWidgetItem(
                [
                    f"Top {rank}",
                    candidate.person_name,
                    f"{candidate.score:.4f}",
                    "候选",
                ]
            )
            if parent is None:
                self.candidate_tree.addTopLevelItem(item)
            else:
                parent.addChild(item)

    def _on_scene(self, scene: SceneResult, image: np.ndarray | None, remaining: float) -> None:
        confirmed = sum(face.confirmed for face in scene.faces)
        self.result_label.setText(f"{scene.detected_count} 张人脸 · 已确认 {confirmed}")
        self.result_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #1d365e;")
        self.score_label.setText("见下方逐脸结果")
        algorithm_ms = scene.timings_ms.get("algorithm_total", 0)
        self.latency_label.setText(f"{algorithm_ms:.1f} ms / {scene.end_to_end_ms:.1f} ms")
        self.updated_label.setText(time.strftime("%H:%M:%S"))
        self.candidate_tree.clear()
        for face in scene.faces:
            label = face.person_name or face.message
            row = QTreeWidgetItem(
                [
                    f"Track {face.track_id}",
                    label,
                    "—" if face.score is None else f"{face.score:.4f}",
                    f"{face.confirmation_hits}/{self.service.config.tracking.confirmation_frames}",
                ]
            )
            self.candidate_tree.addTopLevelItem(row)
            self._add_candidates(face, row)
            row.setExpanded(False)
        if scene.detected_count > len(scene.faces):
            self._append_log(
                f"检测到 {scene.detected_count} 张脸，本次仅处理 {len(scene.faces)} 张"
            )
        self.result_timer.start(max(1, int(remaining * 1000)))
        self._append_log(
            f"scene frame={scene.frame_id} faces={scene.detected_count} confirmed={confirmed}"
        )
        if image is not None:
            import cv2

            snapshot = image.copy()
            for face in scene.faces:
                if face.face is None:
                    continue
                x, y, w, h = (int(round(v)) for v in face.face[:4])
                color = (40, 210, 90) if face.confirmed else (40, 170, 240)
                cv2.rectangle(snapshot, (x, y), (x + w, y + h), color, 2)
                # ASCII track labels avoid cv2.putText's lack of Chinese glyph support.
                cv2.putText(
                    snapshot,
                    f"#{face.track_id} {face.status}",
                    (max(0, x), max(18, y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
            self.result_preview.setPixmap(_pixmap_from_bgr(snapshot, self.result_preview.size()))
            self.analysis_preview.setPixmap(
                _pixmap_from_bgr(snapshot, self.analysis_preview.size())
            )

    def _on_inference_error(self, message: str, stream_id: int, mode_generation: int) -> None:
        if (
            not self._accept_results
            or self.current_packet is None
            or stream_id != self.stream_id
            or mode_generation != self._mode_generation
        ):
            return
        self._clear_result("系统异常")
        self.result_label.setText("系统异常")
        self._append_log(message)
        self.statusBar().showMessage(message)
        self.continuous_check.setChecked(False)

    def _frame_copy(self) -> np.ndarray | None:
        return None if self.current_packet is None else self.current_packet.image.copy()

    def _open_enrollment(self) -> None:
        self._mode_generation += 1
        was_continuous = self.continuous_check.isChecked()
        self.continuous_check.setChecked(False)
        self._accept_results = False
        self.inference_worker.clear_pending()
        try:
            dialog = EnrollDialog(self.service, self._frame_copy, self)
            dialog.exec()
        finally:
            self._accept_results = True
            self._clear_result()
            self.continuous_check.setChecked(was_continuous)
            self._refresh_gallery_status()

    def _open_gallery(self) -> None:
        self._mode_generation += 1
        was_continuous = self.continuous_check.isChecked()
        self.continuous_check.setChecked(False)
        self._accept_results = False
        self.inference_worker.clear_pending()
        try:
            GalleryDialog(self.service, self).exec()
        finally:
            self._accept_results = True
            self._clear_result()
            self.continuous_check.setChecked(was_continuous)
            self._refresh_gallery_status()

    def _refresh_gallery_status(self) -> None:
        persons = self.service.repository.list_persons()
        samples = sum(item.sample_count for item in persons)
        compatible = sum(self.service.matcher.snapshot.sample_counts)
        self.gallery_status.setText(
            f"人员 {len(persons)} 名，样本 {samples} 张；当前模型兼容样本 {compatible} 张；"
            f"库版本 {self.service.repository.gallery_version()}"
        )
        if compatible == 0:
            self.gallery_status.setText(
                self.gallery_status.text() + "\n请先录入当前模型的样本，才能识别姓名。"
            )

    def _update_rates(self, queue_ms=None) -> None:
        if queue_ms is not None:
            self._last_queue_ms = queue_ms
        now = time.perf_counter()
        rates = []
        for stamps in (self._preview_times, self._result_times):
            stamps[:] = [stamp for stamp in stamps if now - stamp <= 5]
            rates.append(
                (len(stamps) - 1) / (stamps[-1] - stamps[0])
                if len(stamps) > 1 and stamps[-1] > stamps[0]
                else None
            )
        preview, inference = ["—" if rate is None else f"{rate:.1f}" for rate in rates]
        queued = "—" if self._last_queue_ms is None else f"{self._last_queue_ms:.1f}"
        self.rate_label.setText(
            f"预览 FPS {preview}  |  有效结果 {inference} 次/秒  |  排队 {queued} ms"
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self._accept_results = False
        if not self._stop_camera():
            event.ignore()
            QTimer.singleShot(500, self.close)
            return
        self.inference_worker.request_stop()
        if not self.inference_worker.wait(250):
            event.ignore()
            self.statusBar().showMessage("正在等待识别任务退出…")
            QTimer.singleShot(500, self.close)
            return
        event.accept()
