# FaceCompare

A local desktop face comparison application using **YuNet**, **SFace**, **OpenCV**, and **PySide6**. Supports USB camera capture, incremental enrollment, open-set 1:N identification, and reproducible threshold and latency experiments.

[中文说明](README.md) · [Architecture](docs/ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md)

![Empty-gallery startup screen](docs/assets/ui-preview.png)

The preview is rendered from the real application with a temporary empty gallery, no camera and no personal data.

## Features

- Live preview, snapshot recognition, continuous recognition and image-only JSON CLI.
- Identity templates with cosine similarity and unknown-person rejection.
- Enrollment, additional samples, preview, deletion, persistence and immediate template updates.
- Single-face, size, blur and brightness quality feedback.
- Background inference/enrollment, latest-frame scheduling, stale-result guards and expiration.
- Expected SHA-256 verification for downloaded and cached models.
- Validation-only calibration, data-isolation checks and image/video benchmarks.
- Regression tests, Ruff, GitHub Actions and a source-only archive builder.
- Multi-face processing, lightweight IoU association and consecutive identity confirmation.
- Top-K candidates, an analysis-snapshot view and prediction-based chart reports.

## Quick start

Use Python 3.11 or 3.12. Windows is the primary target for camera use. CI is also configured for Ubuntu; target-device camera support still needs verification.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --download-models
python main.py --check
python main.py
```

On Linux, activate with `source .venv/bin/activate`. Models are downloaded once; enrollment and inference then run locally on the CPU.

```sh
python main.py --image /path/to/query.jpg --json
python main.py --image /path/to/group.jpg --multi-face --json
python main.py --config config.local.json
python main.py --version
```

Image recognition uses the configured gallery without opening a camera or writing a recognition log. It creates an empty gallery if none exists. Outcomes such as `unknown` and `no_face` return exit code 0; input/environment errors return 1.

## Identification and evaluation

YuNet detects the face and five landmarks. SFace aligns it and extracts an embedding. Samples are normalized, averaged per identity and normalized again. The highest cosine similarity must meet the threshold to accept an identity.

The default threshold `0.42` requires calibration and is not a probability. Use independent enrollment, validation and test sessions; calibrate on validation only and freeze the configuration before testing.

Enable multi-face mode and continuous recognition for track labels. Three consecutive matches confirm an identity by default; changes, failures, missing detections and context changes reset evidence. Enrollment remains single-face. The original IoU tracker is a small-scene baseline, not ByteTrack, and may switch IDs during crossings or fast motion. Static multi-face queries do not perform temporal confirmation.

```sh
python experiments/evaluate.py experiments/manifests/local.csv --split validation --calibrate-target-far 0.05 --freeze-output experiments/results/run.freeze.json --output experiments/results/validation
# Copy the full-precision calibrated threshold into the config before testing.
python experiments/evaluate.py experiments/manifests/local.csv --split test --freeze experiments/results/run.freeze.json --output experiments/results/test
python experiments/benchmark.py experiments/manifests/local.csv --split test --repeats 3
python experiments/video_benchmark.py data/evaluation/fixed_video.mp4 --repeats 3 --max-seconds 30
python experiments/report.py experiments/results/test/predictions.csv --threshold 0.42 --output experiments/results/report
```

Copy the example manifest first. Paths are relative to the CSV. Write the calibrated threshold back before final testing. Processing failures are separate from successful unknown rejection. Benchmarks report algorithm latency, application-side latency and dropped frames separately. They do not establish universal accuracy or speed guarantees.

Version 0.4 opens experiment galleries read-only and rejects missing databases. Final tests require a validation-generated freeze record covering settings, both model hashes, logical gallery contents, image contents, source and environment. Image SHA-256 checks catch renamed copies; re-encoded duplicates still require session discipline. Gallery deletion records retryable file cleanup in the database transaction. The UI provides integrity counts, explicit enrollment sessions, and background gallery operations.

Run `python tools/check_project.py --models` for local checks. See [engineering evidence](docs/HARDENING.md), [measured performance](docs/PERFORMANCE.md), [UI evidence and limits](docs/UI_AUDIT.md), and the portable [project skill](skills/facecompare-check/SKILL.md). Synthetic measurements are not biometric accuracy evidence.

Reports provide score histograms, a 1:N open-set operating curve and scene summaries. This is not a pair-verification ROC and cannot justify tuning on test data. Further plans: [upgrade notes](docs/UPGRADES.md).

An actual [AT&T/ORL public-image benchmark](docs/PUBLIC_BENCHMARK.md) achieved 60/60 correct known queries, 94/100 correct unknown rejections, 6 detection failures and 0/94 false accepts among valid unknown queries. B0/B4 algorithm p50 was 35.51/29.08 ms. This uses a separate low-resolution quality configuration and an explicit image split with unavailable session metadata; it is not evidence for default camera or session-separated performance. `tools/prepare_orl.py` creates an isolated gallery and records provenance. Evaluation requires `--allow-image-split` plus the usual validation-only calibration and final freeze.

## Development and publishing

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -m "not integration" -q
python -m pytest -m integration -q
python tools/build_source_archive.py
```

Integration tests require the downloaded models and use a blank frame and temporary database. Unit/UI tests require no camera or downloads. Publish the contents of `face_compare` as the repository root, including `.github`. Source archives exclude local biometric data, weights, real manifests and environments.

## Scope and licenses

Dedicated liveness/presentation-attack detection is not implemented; no cross-population recognition results are published. Application code: [MIT](LICENSE). Models and dependencies retain their own licenses: [third-party notices](THIRD_PARTY_NOTICES.md).
