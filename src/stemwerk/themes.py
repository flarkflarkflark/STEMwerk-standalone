from __future__ import annotations

from typing import Dict, List

from PySide6 import QtWidgets
from PySide6.QtGui import QColor, QPalette

Theme = Dict[str, object]

THEMES: Dict[str, Theme] = {
    "classic": {
        "bg": "#1a1a2e",
        "panel": "#22223b",
        "accent": "#ff7a18",
        "text": "#f5f5f5",
        "text_dim": "#b0b0b0",
        "border": "#2f2f4a",
        "stem_colors": ["#ff6b6b", "#f4d35e", "#6bcB77", "#4d96ff", "#ff9f1c", "#9c89b8"],
    },
    "ember": {
        "bg": "#2d1b00",
        "panel": "#3a2400",
        "accent": "#ff9f1c",
        "text": "#fff1d6",
        "text_dim": "#c7b08b",
        "border": "#4a2f00",
        "stem_colors": ["#ff6b35", "#ffb703", "#f25f5c", "#ffa552", "#f77f00", "#d62828"],
    },
    "ice": {
        "bg": "#0a1628",
        "panel": "#12213a",
        "accent": "#35baf6",
        "text": "#e8f6ff",
        "text_dim": "#9fb8c9",
        "border": "#1c2d4a",
        "stem_colors": ["#5bc0eb", "#9bc53d", "#e55934", "#7fdbff", "#3dccc7", "#c7f9cc"],
    },
    "mono": {
        "bg": "#1a1a1a",
        "panel": "#222222",
        "accent": "#ffffff",
        "text": "#f0f0f0",
        "text_dim": "#9a9a9a",
        "border": "#2c2c2c",
        "stem_colors": ["#d9d9d9", "#bfbfbf", "#a6a6a6", "#8c8c8c", "#737373", "#595959"],
    },
}


def apply_theme(app: QtWidgets.QApplication, theme_name: str) -> None:
    theme = THEMES.get(theme_name, THEMES["classic"])
    bg = theme["bg"]
    panel = theme["panel"]
    accent = theme["accent"]
    text = theme["text"]
    text_dim = theme["text_dim"]
    border = theme["border"]

    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(bg))
    palette.setColor(QPalette.Base, QColor(panel))
    palette.setColor(QPalette.WindowText, QColor(text))
    palette.setColor(QPalette.Text, QColor(text))
    palette.setColor(QPalette.Button, QColor(panel))
    palette.setColor(QPalette.ButtonText, QColor(text))
    palette.setColor(QPalette.Highlight, QColor(accent))
    palette.setColor(QPalette.HighlightedText, QColor(bg))
    app.setPalette(palette)

    app.setStyleSheet(
        f"""
        QMainWindow {{
            background: {bg};
        }}
        QWidget {{
            color: {text};
            background: {bg};
        }}
        QToolBar, QFrame {{
            background: {panel};
            border: 1px solid {border};
        }}
        QPushButton, QToolButton {{
            background: {panel};
            border: 1px solid {border};
            padding: 6px 10px;
        }}
        QPushButton:hover, QToolButton:hover {{
            border-color: {accent};
        }}
        QComboBox, QLineEdit {{
            background: {panel};
            border: 1px solid {border};
            padding: 4px;
        }}
        QSlider::groove:horizontal {{
            height: 6px;
            background: {border};
        }}
        QSlider::handle:horizontal {{
            width: 12px;
            background: {accent};
            margin: -4px 0;
        }}
        QProgressBar {{
            background: {panel};
            border: 1px solid {border};
            text-align: center;
        }}
        QProgressBar::chunk {{
            background: {accent};
        }}
        QLabel#dimText {{
            color: {text_dim};
        }}
        """
    )
