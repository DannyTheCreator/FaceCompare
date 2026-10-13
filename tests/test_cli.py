import json

import pytest

import main


def test_image_command_returns_json_without_loading_ui(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    import numpy as np

    import app.image_io
    import app.service

    monkeypatch.setattr(
        main,
        "parse_args",
        lambda: SimpleNamespace(
            config=None,
            check=False,
            download_models=False,
            image=tmp_path / "query.jpg",
            json=True,
        ),
    )
    monkeypatch.setattr(
        app.image_io, "read_image", lambda path: np.zeros((4, 4, 3), dtype=np.uint8)
    )
    from app.types import RecognitionResult

    result = RecognitionResult(0, 0, 0, "unknown", "未知人员", 0.42, score=0.1)
    monkeypatch.setattr(
        app.service,
        "FaceComparisonService",
        lambda config: SimpleNamespace(recognize=lambda image, **kwargs: result),
    )
    # No QApplication is constructed; the command uses only service and image I/O.
    assert main.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "unknown"


def test_image_read_error_has_nonzero_exit(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    monkeypatch.setattr(
        main,
        "parse_args",
        lambda: SimpleNamespace(
            config=None,
            check=False,
            download_models=False,
            image=tmp_path / "missing.jpg",
            json=True,
        ),
    )
    assert main.main() == 1
    assert "无法解码" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments", [["--json"], ["--overwrite"], ["--check", "--image", "a.jpg"]]
)
def test_incompatible_cli_flags_are_rejected(monkeypatch, arguments):
    import sys

    monkeypatch.setattr(sys, "argv", ["main.py", *arguments])
    with pytest.raises(SystemExit) as exc:
        main.parse_args()
    assert exc.value.code == 2


def test_multiface_json_outputs_each_face(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    import numpy as np

    import app.image_io
    import app.service
    from app.types import SceneResult

    monkeypatch.setattr(
        main,
        "parse_args",
        lambda: SimpleNamespace(
            config=None,
            check=False,
            download_models=False,
            image=tmp_path / "query.jpg",
            json=True,
            multi_face=True,
        ),
    )
    monkeypatch.setattr(
        app.image_io, "read_image", lambda path: np.zeros((4, 4, 3), dtype=np.uint8)
    )
    scene = SceneResult(0, 0, 0, (), 0, 0.42, 0, 0, {})
    monkeypatch.setattr(
        app.service,
        "FaceComparisonService",
        lambda config: SimpleNamespace(recognize_scene=lambda image: scene),
    )
    assert main.main() == 0
    assert json.loads(capsys.readouterr().out)["faces"] == []
