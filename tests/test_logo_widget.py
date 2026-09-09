from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QtCore = pytest.importorskip("PySide6.QtCore")

from stemwerk.logo_widget import LogoWidget


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_logo_widget_has_nonzero_size_hint(qapp) -> None:
    logo = LogoWidget()
    hint = logo.sizeHint()
    assert hint.width() > 0
    assert hint.height() == 40


def test_logo_widget_gets_real_width_in_header_stretch_layout(qapp) -> None:
    """Regression test for a real bug: main_window.py's header row places the
    logo between two addStretch(1) spacers with an explicit stretch factor of
    0 (see _build_ui). Without a real sizeHint, Qt collapsed the widget to
    zero width and the animated logo silently never rendered. This reproduces
    that exact layout pattern.
    """
    header = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(header)
    logo = LogoWidget()
    logo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
    button = QtWidgets.QPushButton("??")
    button.setFixedSize(30, 30)

    layout.addStretch(1)
    layout.addWidget(logo, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
    layout.addStretch(1)
    layout.addWidget(button, 0, QtCore.Qt.AlignmentFlag.AlignRight)

    header.resize(900, 40)
    header.show()
    qapp.processEvents()

    assert logo.width() > 0
