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
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QToolButton,
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
from .style import ImagePreview, apply_theme, set_tone


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
        self.resize(1400, 920)
        self.setMinimumSize(1080, 760)
        self._build_ui()
        self._refresh_gallery_status()
        self.result_timer = QTimer(self)
        self.result_timer.setSingleShot(True)
        self.result_timer.timeout.connect(lambda: self._clear_result("结果已过期，等待重新识别"))
        self.rate_timer = QTimer(self)
        self.rate_timer.timeout.connect(self._update_rates)
        self.rate_timer.start(1000)

    def _build_ui(self) -> None:
        def section(title):
            label = QLabel(title)
            label.setProperty("role", "sectionTitle")
            return label

        def card():
            frame = QFrame()
            frame.setProperty("role", "card")
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(18, 16, 18, 16)
            layout.setSpacing(12)
            layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
            return frame, layout

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(24, 20, 24, 8)
        root_layout.setSpacing(18)
        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        mark = QLabel("F")
        mark.setObjectName("brandMark")
        mark.setFixedSize(46, 46)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(mark)
        brand = QVBoxLayout()
        brand.setSpacing(2)
        title = QLabel("FaceCompare")
        title.setObjectName("brandTitle")
        subtitle = QLabel("人脸识别工作台 · 采集、识别与人员管理")
        subtitle.setProperty("role", "subtitle")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        header_layout.addLayout(brand)
        header_layout.addStretch()
        local_badge = QLabel("本地运行")
        local_badge.setProperty("role", "badge")
        header_layout.addWidget(local_badge)
        root_layout.addWidget(header)

        self.preview = ImagePreview(
            "请选择设备并开启摄像头", "实时画面将在这里显示 · 图像仅在本机处理"
        )
        self.preview.setMinimumSize(560, 350)
        self.analysis_preview = ImagePreview(
            "识别分析快照，等待识别", "识别完成后，查看人脸位置与匹配结果"
        )
        self.analysis_preview.setMinimumSize(560, 350)
        self.views = QTabWidget()
        self.views.addTab(self.preview, "实时画面")
        self.views.addTab(self.analysis_preview, "识别分析快照")
        preview_card, preview_layout = card()
        preview_heading = QHBoxLayout()
        preview_heading.addWidget(section("摄像头工作区"))
        preview_heading.addStretch()
        self.device_badge = QLabel("设备未连接")
        self.device_badge.setProperty("role", "deviceBadge")
        preview_heading.addWidget(self.device_badge)
        preview_layout.addLayout(preview_heading)
        preview_layout.addWidget(self.views, 1)

        self.camera_combo = QComboBox()
        self.camera_combo.setAccessibleName("摄像头编号")
        self.camera_combo.setMinimumWidth(130)
        self.camera_combo.setToolTip(
            "编号来自配置，并不表示设备已检测到。打开后请用实时画面确认设备。"
        )
        for index in self.service.config.camera.probe_indices:
            self.camera_combo.addItem(f"摄像头 {index}", index)
        default_index = self.camera_combo.findData(self.service.config.camera.default_index)
        if default_index >= 0:
            self.camera_combo.setCurrentIndex(default_index)
        self.open_button = QPushButton("开启摄像头")
        self.open_button.setProperty("role", "primary")
        self.close_button = QPushButton("关闭摄像头")
        self.close_button.setEnabled(False)
        self.capture_button = QPushButton("抓拍识别")
        self.capture_button.setProperty("role", "primary")
        self.capture_button.setEnabled(False)
        self.continuous_check = QCheckBox("连续识别")
        self.scene_check = QCheckBox("多人识别 / 跟踪")
        self.scene_check.setToolTip("连续多次匹配后确认姓名；快速移动或交叉可能改变轨迹号")
        controls, controls_layout = card()
        camera_row = QHBoxLayout()
        camera_row.setSpacing(10)
        camera_row.addWidget(self.camera_combo, 1)
        camera_row.addWidget(self.open_button)
        camera_row.addWidget(self.close_button)
        camera_row.addWidget(self.capture_button)
        controls_layout.addLayout(camera_row)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(24)
        mode_row.addWidget(self.continuous_check)
        mode_row.addWidget(self.scene_check)
        mode_row.addStretch()
        controls_layout.addLayout(mode_row)
        self.guide_label = QLabel("01  开启设备     →     02  录入人员     →     03  开始识别")
        self.guide_label.setProperty("role", "subtitle")
        self.guide_label.setWordWrap(True)
        controls_layout.addWidget(self.guide_label)
        self.rate_label = QLabel("预览 FPS —  |  有效结果 — 次/秒  |  排队 — ms")
        self.rate_label.setProperty("role", "subtitle")
        self.rate_label.setToolTip(
            "最近5秒的预览与有效结果到达速率；后台识别吞吐请使用视频基准测量。"
        )
        self.rate_label.setWordWrap(True)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)
        left_layout.addWidget(preview_card, 1)
        left_layout.addWidget(controls)
        left_layout.addWidget(self.rate_label)

        result_card, result_layout = card()
        result_layout.addWidget(section("识别结果"))
        self.result_label = QLabel("等待识别")
        self.result_label.setObjectName("resultValue")
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        result_form = QFormLayout()
        result_form.setVerticalSpacing(7)
        self.score_label = QLabel("—")
        self.threshold_label = QLabel(f"{self.service.matcher.threshold:.3f}")
        self.latency_label = QLabel("—")
        self.updated_label = QLabel("—")
        for title, value in (
            ("余弦相似度", self.score_label),
            ("接受阈值", self.threshold_label),
            ("算法 / 端到端", self.latency_label),
            ("最近更新", self.updated_label),
        ):
            name = QLabel(title)
            name.setProperty("role", "subtitle")
            result_form.addRow(name, value)
        result_layout.addLayout(result_form)
        self.result_preview = ImagePreview("识别快照将在这里显示")
        self.result_preview.setMinimumSize(220, 90)
        self.result_preview.setMaximumHeight(110)
        result_layout.addWidget(self.result_preview)

        actions, actions_layout = card()
        actions_layout.addWidget(section("人员标准库"))
        metrics = QHBoxLayout()
        self.person_count = QLabel("0")
        self.sample_count = QLabel("0")
        for caption, value in (("已录入人员", self.person_count), ("人脸样本", self.sample_count)):
            column = QVBoxLayout()
            column.setSpacing(0)
            value.setProperty("role", "metricValue")
            column.addWidget(value)
            label = QLabel(caption)
            label.setProperty("role", "subtitle")
            column.addWidget(label)
            metrics.addLayout(column, 1)
        actions_layout.addLayout(metrics)
        self.gallery_status = QLabel()
        self.gallery_status.setProperty("role", "subtitle")
        self.gallery_status.setWordWrap(True)
        actions_layout.addWidget(self.gallery_status)
        self.enroll_button = QPushButton("录入 / 补充样本")
        self.enroll_button.setProperty("role", "primary")
        self.enroll_button.setAccessibleName("人员录入 / 补充样本")
        self.gallery_button = QPushButton("标准库管理")
        action_row = QHBoxLayout()
        action_row.addWidget(self.enroll_button, 1)
        action_row.addWidget(self.gallery_button, 1)
        actions_layout.addLayout(action_row)

        self.candidate_tree = QTreeWidget()
        self.candidate_tree.setHeaderLabels(["轨迹 / 排名", "结果 / 候选", "相似度", "确认"])
        for column in (0, 2, 3):
            self.candidate_tree.header().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.candidate_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.candidate_tree.setAlternatingRowColors(True)
        self.candidate_tree.setMinimumHeight(115)
        self.candidate_tree.setToolTip("候选排序用于解释匹配；未知或待确认状态不会接受身份")
        candidates, candidates_layout = card()
        candidates_layout.addWidget(section("人脸与候选"))
        candidates_layout.addWidget(self.candidate_tree, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setMaximumHeight(85)
        self.log.setPlaceholderText("设备连接与识别事件会显示在这里")
        self.log.setVisible(False)
        log_toggle = QToolButton()
        log_toggle.setText("运行日志")
        log_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        log_toggle.setArrowType(Qt.ArrowType.RightArrow)
        log_toggle.setCheckable(True)
        log_toggle.toggled.connect(self.log.setVisible)
        log_toggle.toggled.connect(
            lambda checked: log_toggle.setArrowType(
                Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
            )
        )
        right = QWidget()
        right.setObjectName("rightContent")
        right.setMinimumWidth(390)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(14)
        right_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        right_layout.addWidget(result_card)
        right_layout.addWidget(actions)
        right_layout.addWidget(candidates, 1)
        right_layout.addWidget(log_toggle)
        right_layout.addWidget(self.log)
        right_scroll = QScrollArea()
        right_scroll.setObjectName("rightScroll")
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(410)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setWidget(right)
        splitter = QSplitter()
        splitter.setHandleWidth(16)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([850, 450])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)
        apply_theme(self)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().showMessage("模型已就绪 · 开启摄像头开始使用")

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
        self.device_badge.setText("连接中")
        set_tone(self.device_badge, "warning")
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
                self.device_badge.setText("释放中")
                set_tone(self.device_badge, "warning")
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
        self.device_badge.setText("设备未连接")
        set_tone(self.device_badge, "neutral")
        self._preview_times.clear()
        self._result_times.clear()
        self.rate_label.setText("预览 FPS —  |  有效结果 — 次/秒  |  排队 — ms")
        return True

    def _clear_result(self, message: str = "等待识别") -> None:
        if hasattr(self, "result_timer"):
            self.result_timer.stop()
        self.result_label.setText(message)
        set_tone(self.result_label, "neutral")
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
        self.device_badge.setText("设备已连接")
        set_tone(self.device_badge, "success")
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
        self.device_badge.setText("连接异常")
        set_tone(self.device_badge, "error")
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
        self.device_badge.setText("设备未连接")
        set_tone(self.device_badge, "neutral")

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
            tone = "success"
        elif result.status == "unknown":
            label = "未知人员"
            tone = "warning"
        else:
            label = result.message
            tone = "error"
        self.result_label.setText(label)
        set_tone(self.result_label, tone)
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
        set_tone(self.result_label, "info")
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
        set_tone(self.result_label, "error")
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
        self.person_count.setText(str(len(persons)))
        self.sample_count.setText(str(samples))
        self.gallery_status.setText(
            f"当前模型兼容样本 {compatible} 张 · 库版本 {self.service.repository.gallery_version()}"
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
