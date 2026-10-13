from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from app import __version__
from app.config import load_config


def environment_check(config_path: str | None) -> int:
    errors: list[str] = []
    try:
        config = load_config(config_path)
        print(f"[OK] 配置: {config.root / 'config.json' if config_path is None else config_path}")
    except Exception as exc:
        print(f"[FAIL] 配置: {exc}")
        return 1
    for module_name in ("numpy", "cv2", "PySide6"):
        try:
            module = importlib.import_module(module_name)
            print(f"[OK] {module_name}: {getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
            print(f"[FAIL] {module_name}: {exc}")
    from app.model_manager import model_downloads, verify_model

    for item in model_downloads(config):
        try:
            digest = verify_model(item)
            print(f"[OK] {item.name}: SHA-256 {digest}")
        except Exception as exc:
            errors.append(str(exc))
            print(f"[FAIL] {exc}")
    if not errors:
        try:
            from app.detector import FaceDetector
            from app.recognizer import FaceRecognizer

            FaceDetector(config.models.detector, config.detection)
            FaceRecognizer(config.models.recognizer)
            print("[OK] 两个模型均可加载")
        except Exception as exc:
            errors.append(str(exc))
            print(f"[FAIL] 模型加载: {exc}")
    print(f"[OK] 数据库路径: {config.storage.database}")
    if errors:
        print("\n环境未就绪。先执行: pip install -r requirements.txt")
        print("模型缺失时执行: python main.py --download-models")
        return 1
    print("\n环境检查通过。")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="外置摄像头人脸比对系统")
    parser.add_argument("--version", action="version", version=f"FaceCompare {__version__}")
    parser.add_argument("--config", help="自定义 config.json 路径")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true", help="检查依赖、配置、模型校验和加载")
    actions.add_argument(
        "--download-models", action="store_true", help="下载官方 YuNet 与 SFace 模型"
    )
    actions.add_argument("--image", type=Path, help="识别图片，不启动界面或摄像头")
    parser.add_argument("--json", action="store_true", help="图片识别输出 JSON")
    parser.add_argument("--multi-face", action="store_true", help="图片中逐脸识别，不做跨帧确认")
    parser.add_argument("--overwrite", action="store_true", help="重新下载已有模型")
    args = parser.parse_args()
    if args.overwrite and not args.download_models:
        parser.error("--overwrite 需要与 --download-models 一起使用")
    if args.json and args.image is None:
        parser.error("--json 需要与 --image 一起使用")
    if args.multi_face and args.image is None:
        parser.error("--multi-face 需要与 --image 一起使用")
    return args


def result_payload(result) -> dict:
    return {
        "status": result.status,
        "message": result.message,
        "person_id": result.person_id,
        "person_name": result.person_name,
        "score": result.score,
        "threshold": result.threshold,
        "candidates": [asdict(candidate) for candidate in result.candidates],
        "box": result.face[:4].tolist() if result.face is not None else None,
        "timings_ms": result.timings_ms,
        "quality": result.quality.as_dict() if result.quality else None,
    }


def main() -> int:
    args = parse_args()
    if args.check:
        return environment_check(args.config)
    try:
        config = load_config(args.config)
        if args.download_models:
            from app.model_manager import download_models

            manifest = download_models(config, overwrite=args.overwrite)
            print(f"模型清单已写入: {manifest}")
            return 0
        if args.image is not None:
            from app.image_io import read_image
            from app.service import FaceComparisonService

            image = read_image(args.image)
            if image is None:
                raise ValueError(f"图片不存在或无法解码: {args.image}")
            service = FaceComparisonService(config)
            if getattr(args, "multi_face", False):
                scene = service.recognize_scene(image)
                payload = {
                    "detected_count": scene.detected_count,
                    "processed_count": len(scene.faces),
                    "faces": [result_payload(face) for face in scene.faces],
                    "timings_ms": scene.timings_ms,
                }
                message = f"检测到 {scene.detected_count} 张脸，已处理 {len(scene.faces)} 张"
            else:
                result = service.recognize(image, write_log=False)
                payload = result_payload(result)
                message = result.message
            print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else message)
            return 0
    except Exception as exc:
        print(f"操作失败: {exc}", file=sys.stderr)
        return 1
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        from app.service import FaceComparisonService
        from app.ui.main_window import MainWindow
        from app.ui.style import configure_application
    except ImportError as exc:
        print(f"依赖缺失: {exc}\n请执行: pip install -r requirements.txt", file=sys.stderr)
        return 1
    application = QApplication(sys.argv)
    configure_application(application)
    application.setApplicationName("人脸比对系统")
    try:
        service = FaceComparisonService(config)
    except Exception as exc:
        QMessageBox.critical(None, "启动失败", str(exc))
        return 1
    window = MainWindow(service)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
