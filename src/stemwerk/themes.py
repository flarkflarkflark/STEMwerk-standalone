from __future__ import annotations

from typing import Dict, Tuple

from PySide6 import QtWidgets
from PySide6.QtGui import QColor, QPalette

ColorTuple = Tuple[float, float, float]
ThemeVariant = Dict[str, ColorTuple]

STEM_COLORS = ["#FF6464", "#64C8FF", "#9664FF", "#64FF96", "#FFB464", "#FF78C8"]

THEMES: Dict[str, Dict[str, ThemeVariant]] = {
    "classic": {
        "dark": {
            "bg": (0.18, 0.18, 0.20),
            "bgGradientTop": (0.10, 0.10, 0.12),
            "bgGradientBottom": (0.18, 0.18, 0.20),
            "inputBg": (0.12, 0.12, 0.14),
            "text": (1.0, 1.0, 1.0),
            "textDim": (0.7, 0.7, 0.7),
            "accent": (0.3, 0.5, 0.8),
            "accentHover": (0.4, 0.6, 0.9),
            "button": (0.2, 0.4, 0.7),
            "buttonHover": (0.3, 0.5, 0.8),
            "buttonPrimary": (0.2, 0.5, 0.3),
            "buttonPrimaryHover": (0.3, 0.6, 0.4),
            "border": (0.6, 0.6, 0.6),
        },
        "light": {
            "bg": (0.92, 0.92, 0.94),
            "bgGradientTop": (0.96, 0.96, 0.98),
            "bgGradientBottom": (0.88, 0.88, 0.90),
            "inputBg": (0.85, 0.85, 0.87),
            "text": (0.1, 0.1, 0.1),
            "textDim": (0.3, 0.3, 0.3),
            "accent": (0.2, 0.4, 0.7),
            "accentHover": (0.3, 0.5, 0.8),
            "button": (0.3, 0.5, 0.75),
            "buttonHover": (0.4, 0.6, 0.85),
            "buttonPrimary": (0.25, 0.55, 0.35),
            "buttonPrimaryHover": (0.35, 0.65, 0.45),
            "border": (0.4, 0.4, 0.4),
        },
    },
    "ember": {
        "dark": {
            "accent": (0.75, 0.35, 0.25),
            "accentHover": (0.85, 0.45, 0.35),
            "button": (0.55, 0.25, 0.2),
            "buttonHover": (0.65, 0.35, 0.3),
            "buttonPrimary": (0.5, 0.35, 0.2),
            "buttonPrimaryHover": (0.6, 0.45, 0.3),
            "bgGradientTop": (0.11, 0.09, 0.08),
            "bgGradientBottom": (0.18, 0.14, 0.12),
        },
        "light": {
            "accent": (0.75, 0.35, 0.25),
            "accentHover": (0.85, 0.45, 0.35),
            "button": (0.55, 0.25, 0.2),
            "buttonHover": (0.65, 0.35, 0.3),
            "buttonPrimary": (0.5, 0.35, 0.2),
            "buttonPrimaryHover": (0.6, 0.45, 0.3),
            "bgGradientTop": (0.11, 0.09, 0.08),
            "bgGradientBottom": (0.18, 0.14, 0.12),
        },
    },
    "ice": {
        "dark": {
            "accent": (0.2, 0.65, 0.75),
            "accentHover": (0.3, 0.75, 0.85),
            "button": (0.2, 0.5, 0.6),
            "buttonHover": (0.3, 0.6, 0.7),
            "buttonPrimary": (0.2, 0.55, 0.55),
            "buttonPrimaryHover": (0.3, 0.65, 0.65),
            "bgGradientTop": (0.08, 0.1, 0.12),
            "bgGradientBottom": (0.14, 0.18, 0.2),
        },
        "light": {
            "accent": (0.2, 0.65, 0.75),
            "accentHover": (0.3, 0.75, 0.85),
            "button": (0.2, 0.5, 0.6),
            "buttonHover": (0.3, 0.6, 0.7),
            "buttonPrimary": (0.2, 0.55, 0.55),
            "buttonPrimaryHover": (0.3, 0.65, 0.65),
            "bgGradientTop": (0.08, 0.1, 0.12),
            "bgGradientBottom": (0.14, 0.18, 0.2),
        },
    },
    "mono": {
        "dark": {
            "accent": (0.55, 0.55, 0.6),
            "accentHover": (0.65, 0.65, 0.7),
            "button": (0.35, 0.35, 0.4),
            "buttonHover": (0.45, 0.45, 0.5),
            "buttonPrimary": (0.4, 0.4, 0.45),
            "buttonPrimaryHover": (0.5, 0.5, 0.55),
            "bgGradientTop": (0.11, 0.11, 0.12),
            "bgGradientBottom": (0.16, 0.16, 0.17),
        },
        "light": {
            "accent": (0.55, 0.55, 0.6),
            "accentHover": (0.65, 0.65, 0.7),
            "button": (0.35, 0.35, 0.4),
            "buttonHover": (0.45, 0.45, 0.5),
            "buttonPrimary": (0.4, 0.4, 0.45),
            "buttonPrimaryHover": (0.5, 0.5, 0.55),
            "bgGradientTop": (0.11, 0.11, 0.12),
            "bgGradientBottom": (0.16, 0.16, 0.17),
        },
    },
}


def resolve_theme(name: str, mode: str) -> ThemeVariant:
    base = THEMES["classic"][mode]
    overrides = THEMES.get(name, {}).get(mode, {})
    resolved = dict(base)
    resolved.update(overrides)
    return resolved


def to_qcolor(color: ColorTuple, alpha: float = 1.0) -> QColor:
    r = max(0, min(1, color[0]))
    g = max(0, min(1, color[1]))
    b = max(0, min(1, color[2]))
    qcolor = QColor.fromRgbF(r, g, b, alpha)
    return qcolor


def apply_theme(app: QtWidgets.QApplication | None, theme: ThemeVariant) -> None:
    if app is None:
        return
    bg = to_qcolor(theme["bg"])
    input_bg = to_qcolor(theme["inputBg"])
    text = to_qcolor(theme["text"])
    accent = to_qcolor(theme["accent"])
    border = to_qcolor(theme["border"])

    palette = app.palette()
    palette.setColor(QPalette.Window, bg)
    palette.setColor(QPalette.Base, input_bg)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, input_bg)
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.Highlight, accent)
    palette.setColor(QPalette.HighlightedText, bg)
    app.setPalette(palette)

    app.setStyleSheet(
        f"""
        QWidget {{
            color: {text.name()};
        }}
        QFrame {{
            background: {input_bg.name()};
            border: 1px solid {border.name()};
        }}
        QComboBox, QLineEdit {{
            background: {input_bg.name()};
            border: 1px solid {border.name()};
            padding: 4px;
        }}
        QSlider::groove:horizontal {{
            height: 6px;
            background: {border.name()};
        }}
        QSlider::handle:horizontal {{
            width: 12px;
            background: {accent.name()};
            margin: -4px 0;
        }}
        QProgressBar {{
            background: {input_bg.name()};
            border: 1px solid {border.name()};
            text-align: center;
        }}
        QProgressBar::chunk {{
            background: {accent.name()};
        }}
        QLabel#dimText {{
            color: {to_qcolor(theme["textDim"]).name()};
        }}
        """
    )
