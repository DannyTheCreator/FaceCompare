"""Run the local project checks without camera access or gallery writes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", action="store_true", help="Also verify locally downloaded models"
    )
    args = parser.parse_args()
    env = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONUTF8": "1",
    }
    checks = [
        ["-m", "ruff", "check", ".", "--no-cache"],
        ["-m", "ruff", "format", "--check", ".", "--no-cache"],
        ["-m", "pytest", "-q", "-p", "no:cacheprovider"],
    ]
    if args.models:
        checks.append(["main.py", "--check"])
    failed = []
    for command in checks:
        print("CHECK:", " ".join(command), flush=True)
        if subprocess.run([sys.executable, *command], cwd=ROOT, env=env).returncode:
            failed.append(command)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
