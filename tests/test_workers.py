import time
from types import SimpleNamespace

import numpy as np
import pytest

from app.types import FramePacket
from app.workers import CameraWorker, InferenceWorker
from experiments.video_benchmark import run_variant


def test_latest_queue_replaces_pending_frames_and_rejects_after_stop(qapp):
    worker = InferenceWorker(SimpleNamespace())
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    worker.submit_latest(FramePacket(1, 0, image))
    worker.submit_latest(FramePacket(2, 0, image))
    assert worker._queue.qsize() == 1
    assert worker._queue.get_nowait().frame_id == 2
    worker.request_stop()
    worker.submit_latest(FramePacket(3, 0, image))
    assert worker._queue.empty()


def test_video_consumer_failure_propagates_instead_of_returning_partial_metrics(
    tmp_path, monkeypatch
):
    import cv2

    class Capture:
        def isOpened(self):
            return True

        def get(self, key):
            return 100.0

        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def release(self):
            pass

    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return None  # Warmup succeeds; the actual consumer must propagate failure.
        raise ValueError("broken inference")

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: Capture())
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="推理失败"):
        run_variant(
            SimpleNamespace(recognize=fail),
            tmp_path / "video.mp4",
            latest_only=True,
            detect_scale=1.0,
            match_mode="matrix",
            max_seconds=0.1,
        )
    assert time.perf_counter() - started < 3


def test_camera_read_failure_releases_capture(qapp, monkeypatch):
    import cv2

    class Capture:
        released = False

        def isOpened(self):
            return True

        def set(self, *args):
            pass

        def get(self, *args):
            return 640

        def read(self):
            return False, None

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    worker = CameraWorker(0, 640, 480, 30)
    failures = []
    worker.camera_error.connect(failures.append)
    worker.run()
    assert capture.released
    assert len(failures) == 1 and "连续读取失败" in failures[0]


def test_expired_pending_frame_never_reaches_service(qapp):
    from PySide6.QtTest import QTest

    from app.config import load_config

    calls = []
    worker = InferenceWorker(
        SimpleNamespace(config=load_config(), recognize=lambda *a, **k: calls.append(k))
    )
    worker.submit_latest(
        FramePacket(1, time.perf_counter() - 20, np.zeros((4, 4, 3), dtype=np.uint8))
    )
    worker.start()
    QTest.qWait(50)
    worker.request_stop()
    assert worker.wait(1000)
    assert calls == []
