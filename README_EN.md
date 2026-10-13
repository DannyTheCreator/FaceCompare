# FaceCompare

A local desktop face comparison application using **YuNet**, **SFace**, **OpenCV**, and **PySide6**. Supports USB camera capture, incremental enrollment, open-set 1:N identification, and reproducible threshold and latency experiments.

[中文说明](README.md) · [User guide (Chinese)](docs/操作手册.md) · [Architecture](docs/ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md)

[![Checks](https://github.com/DannyTheCreator/FaceCompare/actions/workflows/ci.yml/badge.svg)](https://github.com/DannyTheCreator/FaceCompare/actions/workflows/ci.yml)

![FaceCompare recognition workspace](docs/assets/ui-preview.png)

The refreshed workspace rendered from the real application with an empty gallery and no personal data. The desktop interface is in Chinese.

## Features

- Card-based workspace with separate camera controls, recognition results, gallery counts, and expandable run logs.
- Live preview, snapshot recognition, continuous recognition and image-only JSON CLI.
- Identity templates with cosine similarity and unknown-person rejection.
- Enrollment, additional samples, preview, deletion, persistence and immediate template updates.
- Single-face, size, blur and brightness quality feedback.
- Background inference/enrollment, latest-frame scheduling, stale-result guards and expiration.
- Expected SHA-256 verification for downloaded and cached models.
- Validation-only calibration, data-isolation checks and image/video benchmarks.
- Multi-face processing, lightweight IoU association and consecutive identity confirmation.
- Top-K candidates, an analysis-snapshot view and prediction-based chart reports.

## Quick start

Use Python 3.11 or 3.12. Windows 10/11 and a USB camera are recommended for camera use. No GPU is required. CI covers Windows/Ubuntu; camera support needs verification on the target device.

```powershell
git clone https://github.com/DannyTheCreator/FaceCompare.git
cd FaceCompare
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --download-models
python main.py --check
python main.py
```

On Linux, activate with `source .venv/bin/activate`. Ubuntu also needs the Qt EGL runtime: `sudo apt-get install -y libegl1`. The first model download requires network access and verifies fixed SHA-256 hashes. Enrollment and inference then run locally on the CPU. The initial gallery is empty; enroll each person with at least 5 clear images.

```sh
python main.py --image /path/to/query.jpg --json
python main.py --image /path/to/group.jpg --multi-face --json
python main.py --config config.local.json
python main.py --version
```

Image recognition uses the configured gallery without opening a camera or writing a recognition log. It creates an empty gallery if none exists. Outcomes such as `unknown` and `no_face` return exit code 0; input/environment errors return 1.

Switch between the live view and analysis snapshot on the left. The sidebar shows results, person/sample counts, and per-face candidates; it scrolls in smaller windows. Expand the run log at the bottom for device and recognition events. Enrollment includes sample-count progress, and gallery management pairs the person list with a sample preview.

## Identification and evaluation

YuNet detects the face and five landmarks. SFace aligns it and extracts an embedding. Samples are normalized, averaged per identity and normalized again. The highest cosine similarity must meet the threshold to accept an identity.

The default threshold `0.42` requires calibration and is not a probability. Use independent enrollment, validation and test sessions; calibrate on validation only and freeze the configuration before testing.

Enable multi-face mode and continuous recognition for track labels. Three consecutive matches confirm an identity by default; changes, failures, missing detections and context changes reset evidence. Enrollment remains single-face. The lightweight IoU tracker is intended for small scenes with slow motion and may switch IDs during crossings or fast motion. Static multi-face queries do not perform temporal confirmation.

## Experiments and performance

Keep enrollment, validation and final test data separate. Calibrate on validation only and use a freeze record for final testing. The [experiment guide](docs/EXPERIMENTS.md) contains commands, manifest fields and metric definitions. Processing failures are separate from successful unknown rejection.

A separate [AT&T/ORL public-image benchmark](docs/PUBLIC_BENCHMARK.md) with a 20-person gallery produced 60/60 correct known queries, 94/100 correct unknown rejections, 6 detection failures, and 0/94 false accepts among valid unknown queries. B0 → B4 algorithm p50 was 35.51 → 29.08 ms.

This benchmark uses a separate low-resolution quality configuration and an explicit image split without per-image session metadata. It does not establish default camera or cross-session performance. Detection failures are not counted as successful unknown rejections. [Performance analysis](docs/PERFORMANCE.md) separately documents synthetic matching and queue measurements.

## Development

```sh
python -m pip install -r requirements-dev.txt
python tools/check_project.py
```

Checks run Ruff and pytest with temporary data and offscreen Qt windows, without opening a camera. Add `--models` after downloading the models to verify them. See [Contributing](CONTRIBUTING.md) for development guidance.

## Scope and licenses

Images, features and gallery databases remain in local `data/` and are not included in the repository. Enroll consenting participants and protect gallery backups. Dedicated liveness/presentation-attack detection and cross-population recognition validation are not implemented. Application code: [MIT](LICENSE). Models and dependencies retain their own licenses: [third-party notices](THIRD_PARTY_NOTICES.md).
