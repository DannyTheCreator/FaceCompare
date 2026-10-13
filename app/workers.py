from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import replace

from PySide6.QtCore import QThread, Signal

from .service import FaceComparisonService
from .tracking import SceneTracker
from .types import FramePacket


class CameraWorker(QThread):
    frame_ready = Signal(object)
    camera_opened = Signal(int, int)
    camera_error = Signal(str)

    def __init__(
        self, index: int, width: int, height: int, fps: int, parent=None, stream_id: int = 0
    ):
        super().__init__(parent)
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.stream_id = stream_id
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        capture = None
        try:
            import cv2

            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            capture = cv2.VideoCapture(self.index, backend)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(self.index)
            if not capture.isOpened():
                self.camera_error.emit(f"无法打开摄像头 {self.index}，请检查设备占用或编号")
                return
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.camera_opened.emit(actual_w, actual_h)
            frame_id = 0
            failed_reads = 0
            target_period = 1.0 / self.fps
            while not self._stop_event.is_set():
                started = time.perf_counter()
                ok, frame = capture.read()
                captured = time.perf_counter()
                if not ok or frame is None:
                    failed_reads += 1
                    if failed_reads >= 5:
                        self.camera_error.emit("摄像头连续读取失败，设备可能已拔出")
                        break
                    self.msleep(30)
                    continue
                failed_reads = 0
                frame_id += 1
                self.frame_ready.emit(FramePacket(frame_id, captured, frame, self.stream_id))
                remaining = target_period - (time.perf_counter() - started)
                if remaining > 0:
                    self.msleep(max(1, int(remaining * 1000)))
        except Exception as exc:
            self.camera_error.emit(f"摄像头线程异常: {exc}")
        finally:
            if capture is not None:
                capture.release()


class InferenceWorker(QThread):
    result_ready = Signal(object, object)
    inference_error = Signal(str, int, int)

    def __init__(self, service: FaceComparisonService, parent=None):
        super().__init__(parent)
        self.service = service
        self._queue: queue.Queue[FramePacket | None] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._tracker: SceneTracker | None = None

    def submit_latest(self, packet: FramePacket) -> None:
        """Keep only the newest pending frame to prevent latency accumulation."""

        if self._stop_event.is_set():
            return
        packet = replace(packet, submitted_monotonic=time.perf_counter())
        try:
            self._queue.put_nowait(packet)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(packet)
        except queue.Full:
            pass

    def request_stop(self) -> None:
        self._stop_event.set()
        self.clear_pending()

    def clear_pending(self) -> None:
        """Drop pending work when the device or gallery changes."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if packet is None:
                break
            started = time.perf_counter()
            queue_started = time.perf_counter()
            if self._stop_event.is_set():
                break
            if (
                started - packet.captured_monotonic
                >= self.service.config.camera.result_ttl_ms / 1000
            ):
                continue
            try:
                if packet.mode == "scene":
                    if self._tracker is None:
                        self._tracker = SceneTracker(self.service.config.tracking)
                    scene = self.service.recognize_scene(
                        packet.image,
                        packet.frame_id,
                        packet.captured_monotonic,
                        packet.stream_id,
                    )
                    result = self._tracker.update(
                        replace(scene, mode_generation=packet.mode_generation)
                    )
                else:
                    if self._tracker is not None:
                        self._tracker.reset()
                    result = self.service.recognize(
                        packet.image,
                        frame_id=packet.frame_id,
                        captured_monotonic=packet.captured_monotonic,
                        stream_id=packet.stream_id,
                    )
                    result = replace(result, mode_generation=packet.mode_generation)
                result.timings_ms["queue_wait"] = max(
                    0, (queue_started - (packet.submitted_monotonic or queue_started)) * 1000
                )
                if not self._stop_event.is_set():
                    self.result_ready.emit(result, packet.image)
            except Exception as exc:
                self.inference_error.emit(
                    f"识别线程异常: {exc}", packet.stream_id, packet.mode_generation
                )


class TaskWorker(QThread):
    """Run one enrollment operation without blocking the Qt event loop."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

    def run(self) -> None:
        try:
            self.succeeded.emit(self.operation())
        except Exception as exc:
            self.failed.emit(str(exc))
