import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def prepared_sample():
    import numpy as np

    from app.types import PreparedSample, QualityReport

    return PreparedSample(
        np.full((64, 64, 3), 128, dtype=np.uint8),
        np.array([0.8, 0.2], dtype=np.float32),
        np.zeros(15, dtype=np.float32),
        QualityReport(True, "accepted", "ok", 1, 0.5, 100, 128),
    )
