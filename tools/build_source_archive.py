"""Build an allowlisted source archive without local data, weights or environments."""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import __version__

ROOT_FILES = {
    "AGENTS.md",
    "main.py",
    "config.json",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "README.md",
    "README_EN.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
}


def source_files(root: Path) -> list[Path]:
    """Select known project files; personal manifests and generated outputs are excluded."""
    selected = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for directory in ("app", "experiments", "tests", "tools"):
        selected.extend((root / directory).rglob("*.py"))
    for directory, patterns in (
        ("docs", ("*.md",)),
        (".github", ("*.yml", "*.md")),
        ("skills/facecompare-check", ("SKILL.md",)),
    ):
        for pattern in patterns:
            selected.extend((root / directory).rglob(pattern))
    for relative in (
        "data/.gitkeep",
        "experiments/results/.gitkeep",
        "experiments/manifests/example.csv",
        "docs/assets/ui-preview.png",
        "docs/assets/audit/01-start.png",
        "docs/assets/audit/02-device.png",
        "docs/assets/audit/03-enroll.png",
        "docs/assets/audit/04-recognition.png",
        "docs/assets/audit/05-gallery.png",
        "refine-logs/EXPERIMENT_PLAN.md",
        "refine-logs/EXPERIMENT_PLAN_20261013.md",
        "refine-logs/EXPERIMENT_TRACKER.md",
        "refine-logs/EXPERIMENT_TRACKER_20261013.md",
        "refine-logs/MANIFEST.md",
    ):
        if (root / relative).is_file():
            selected.append(root / relative)
    safe = set()
    for path in selected:
        if any(part.is_symlink() for part in (path, *path.parents)):
            continue
        relative = path.relative_to(root)
        if relative.as_posix() != "experiments/results/.gitkeep" and any(
            part in {"__pycache__", ".venv", "results"} for part in relative.parts[:-1]
        ):
            continue
        safe.add(path)
    return sorted(safe)


def build_archive(root: Path, output: Path) -> int:
    root, output = root.resolve(), output.resolve()
    files = source_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(
                path.relative_to(root).as_posix(), date_time=(2026, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{checksum}  {output.name}\n", encoding="utf-8"
    )
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dist" / f"FaceCompare-{__version__}-source.zip",
    )
    args = parser.parse_args()
    count = build_archive(PROJECT_ROOT, args.output)
    print(f"Created {args.output} ({count} files, no local data or weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
