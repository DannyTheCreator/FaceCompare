import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialogButtonBox

from app.config import load_config
from app.matcher import Candidate
from app.repository import GalleryRepository
from app.types import FramePacket, RecognitionResult, SceneResult
from app.ui.enroll_dialog import EnrollDialog
from app.ui.main_window import MainWindow


def fake_service(tmp_path):
    repository = GalleryRepository(tmp_path / "gallery.db", tmp_path / "samples")
    config = load_config()
    return SimpleNamespace(
        config=config,
        repository=repository,
        matcher=SimpleNamespace(threshold=0.42, snapshot=SimpleNamespace(sample_counts=())),
    )


def test_entering_name_updates_save_button_and_switch_resets_samples(
    qapp, tmp_path, prepared_sample
):
    service = fake_service(tmp_path)
    service.repository.create_person("已有人员")
    dialog = EnrollDialog(service, lambda: None)
    dialog.samples = [prepared_sample] * service.config.enrollment.minimum_samples
    dialog._update_state()
    save_button = dialog.button_box.button(QDialogButtonBox.StandardButton.Save)
    assert not save_button.isEnabled()
    dialog.name_edit.setText("新人员")
    assert save_button.isEnabled()
    dialog.person_combo.setCurrentIndex(1)
    assert dialog.samples == []
    dialog.close()


def test_async_save_closes_only_after_worker_finishes(qapp, tmp_path, prepared_sample):
    service = fake_service(tmp_path)
    service.commit_enrollment = lambda *args, **kwargs: 1
    dialog = EnrollDialog(service, lambda: None)
    dialog.samples = [prepared_sample] * service.config.enrollment.minimum_samples
    dialog.name_edit.setText("测试")
    dialog._save()
    deadline = time.perf_counter() + 3
    while dialog.worker is not None and time.perf_counter() < deadline:
        QTest.qWait(10)
    assert dialog.worker is None
    assert dialog.result() == dialog.DialogCode.Accepted


def test_stale_camera_and_gallery_results_are_ignored(qapp, tmp_path):
    window = MainWindow(fake_service(tmp_path))
    now = time.perf_counter()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    window.stream_id = 5
    window.current_packet = FramePacket(1, now, image, 5)
    result = RecognitionResult(1, now, now, "matched", "ok", 0.42, person_name="Test", stream_id=4)
    try:
        window._on_result(result, image)
        assert window.result_label.text() == "等待识别"
        window._on_result(replace(result, stream_id=5, gallery_version=99), image)
        assert window.result_label.text() == "等待识别"
        window._on_result(replace(result, stream_id=5), image)
        assert window.result_label.text() == "Test"
        window._on_frame(FramePacket(2, now, image, 4))
        assert window.current_packet.frame_id == 1
        window._stop_camera()
        assert window.result_label.text() == "等待识别"
    finally:
        window.close()


def test_expired_result_is_not_displayed(qapp, tmp_path):
    window = MainWindow(fake_service(tmp_path))
    now = time.perf_counter()
    window.current_packet = FramePacket(1, now, np.zeros((100, 100, 3), dtype=np.uint8))
    result = RecognitionResult(1, now - 10, now, "matched", "ok", 0.42, person_name="old")
    try:
        window._on_result(result)
        assert window.result_label.text() == "等待识别"
    finally:
        window.close()


def test_scene_ui_lists_tracks_and_ignores_previous_mode(qapp, tmp_path):
    window = MainWindow(fake_service(tmp_path))
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    now = time.perf_counter()
    window.current_packet = FramePacket(1, now, image)
    window.scene_check.setChecked(True)
    face = np.zeros(15, dtype=np.float32)
    face[:4] = [10, 10, 80, 80]
    result = RecognitionResult(
        1,
        now,
        now,
        "pending",
        "等待连续确认",
        0.42,
        candidates=(Candidate(1, "Candidate", 0.8),),
        face=face,
        track_id=1,
        confirmation_hits=1,
    )
    scene = SceneResult(
        1,
        now,
        now,
        (result,),
        1,
        0.42,
        0,
        0,
        {"algorithm_total": 1},
        mode_generation=window._mode_generation,
    )
    try:
        window._on_result(scene, image)
        assert window.candidate_tree.topLevelItemCount() == 1
        assert window.candidate_tree.topLevelItem(0).childCount() == 1
        assert "已确认 0" in window.result_label.text()
        assert not window.analysis_preview.pixmap().isNull()
        window.scene_check.setChecked(False)
        window._on_result(scene, image)
        assert window.result_label.text() == "等待识别"
    finally:
        window.close()


def test_old_mode_error_does_not_interrupt_current_result(qapp, tmp_path):
    window = MainWindow(fake_service(tmp_path))
    window.current_packet = FramePacket(
        1, time.perf_counter(), np.zeros((10, 10, 3), dtype=np.uint8)
    )
    try:
        window._mode_generation = 5
        window._on_inference_error("old failure", window.stream_id, 4)
        assert window.result_label.text() == "等待识别"
    finally:
        window.close()


def test_gallery_empty_state_and_async_integrity(qapp, tmp_path):
    from app.ui.gallery_dialog import GalleryDialog

    service = fake_service(tmp_path)
    service.model_hash = "synthetic"
    dialog = GalleryDialog(service)
    try:
        assert not dialog.delete_button.isEnabled()
        assert "标准库为空" in dialog.status_label.text()
        dialog.check_button.click()
        assert not dialog.refresh_button.isEnabled()
        deadline = time.perf_counter() + 3
        while dialog.worker is not None and time.perf_counter() < deadline:
            QTest.qWait(10)
        assert dialog.worker is None
        assert "缺失图片 0" in dialog.status_label.text()
    finally:
        dialog.close()


def test_camera_stop_does_not_block_on_slow_driver(qapp, tmp_path):
    class SlowCamera:
        def request_stop(self):
            pass

        def wait(self, milliseconds):
            assert milliseconds == 0
            return False

    window = MainWindow(fake_service(tmp_path))
    window.camera_worker = SlowCamera()
    try:
        assert not window._stop_camera()
        assert not window.open_button.isEnabled()
        assert "等待摄像头释放" in window.statusBar().currentMessage()
    finally:
        window.camera_worker = None
        window.close()


def test_opened_signal_after_stop_does_not_reenable_capture(qapp, tmp_path):
    from app.workers import CameraWorker

    window = MainWindow(fake_service(tmp_path))
    worker = CameraWorker(0, 640, 480, 30, stream_id=1)
    window.camera_worker = worker
    window.stream_id = 2
    worker.camera_opened.connect(window._on_camera_opened)
    worker.camera_error.connect(window._on_camera_error)
    try:
        worker.camera_opened.emit(640, 480)
        worker.camera_error.emit("old camera failure")
        assert not window.capture_button.isEnabled()
        assert window.result_label.text() == "等待识别"
    finally:
        window.camera_worker = None
        worker.deleteLater()
        window.close()
