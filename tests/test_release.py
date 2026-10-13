import zipfile

from tools.build_source_archive import build_archive


def test_source_archive_excludes_personal_data_weights_and_local_notes(tmp_path):
    root = tmp_path / "source"
    for name in [
        "main.py",
        "README.md",
        "AGENTS.md",
        "skills/facecompare-check/SKILL.md",
        "refine-logs/EXPERIMENT_PLAN.md",
        "docs/PUBLISHING.md",
        "docs/HARDENING.md",
        "docs/UI_AUDIT.md",
        "docs/UPGRADES.md",
        "docs/assets/audit/01-start.png",
        "docs/ARCHITECTURE.md",
        "data/gallery.db",
        "data/samples/1/face.jpg",
        "models/private.onnx",
        "experiments/manifests/private.csv",
        "experiments/manifests/example.csv",
        "app/module.py",
        "app/__pycache__/cached.py",
        ".github/workflows/ci.yml",
    ]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder")
    output = tmp_path / "release.zip"
    assert build_archive(root, output) == 6
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "main.py",
            "README.md",
            "docs/ARCHITECTURE.md",
            "app/module.py",
            "experiments/manifests/example.csv",
            ".github/workflows/ci.yml",
        }
    assert output.with_suffix(".zip.sha256").is_file()
    first = output.read_bytes()
    build_archive(root, output)
    assert output.read_bytes() == first
