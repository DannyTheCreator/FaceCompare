# Third-party notices

The root MIT license applies to FaceCompare application code, not third-party models or dependencies. This source distribution bundles neither model weights nor dependency binaries.

The optional local public-photo benchmark downloads the [AT&T Database of Faces (ORL), hosted by Cambridge](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html). Credit: **AT&T Laboratories Cambridge**, as requested by the source. Images are not bundled or relicensed under MIT. See the source page before using or redistributing the data, and [our benchmark protocol](docs/PUBLIC_BENCHMARK.md) for the fixed archive checksum and experimental limitations.

| Model | Official source | Upstream directory license |
| --- | --- | --- |
| YuNet 2023mar | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | [MIT, Shiqi Yu](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE) |
| SFace 2021dec | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | [Apache-2.0](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE) |

The directory README files identify these licenses. If redistributing model binaries separately, retain the corresponding upstream license and attribution files.

| File | Expected SHA-256 |
| --- | --- |
| `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

Interface references: [OpenCV face tutorial](https://docs.opencv.org/4.12.0/d0/dd4/tutorial_dnn_face.html), [Qt QThread](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html). Model implementations: [YuNet](https://github.com/ShiqiYu/libfacedetection.train), [SFace](https://github.com/zhongyy/SFace).

Design comparisons: [DeepFace](https://github.com/serengil/deepface), [InsightFace FaceAnalysis](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/face_analysis.py), [Supervision trackers](https://supervision.roboflow.com/latest/trackers/). These libraries are not vendored or installed. The lightweight IoU and confirmation code is an original baseline, not their tracker implementation.

This project implements its own configuration, storage, templates, quality feedback, UI state, evaluation and archive layers around the pretrained models. Dependencies are installed from PyPI rather than vendored; refer to the installed package metadata and upstream licenses for exact versions:

- [NumPy](https://numpy.org/), [OpenCV Python](https://github.com/opencv/opencv-python)
- [PySide6 / Qt for Python](https://doc.qt.io/qtforpython-6/licenses.html), [Matplotlib](https://matplotlib.org/)
- [pytest](https://docs.pytest.org/), [pytest-cov](https://github.com/pytest-dev/pytest-cov), [Ruff](https://github.com/astral-sh/ruff)
