"""Design system, color palette, and typography for Smart Study Monitor.

Centralizes all visual properties to ensure a consistent, modern AI
productivity monitor aesthetic rather than a debugging feed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2

# Color format: BGR (Blue, Green, Red) for OpenCV compatibility
ColorBGR = Tuple[int, int, int]


@dataclass(frozen=True)
class UIColors:
    """Centralized color palette with semantic meanings."""

    # Panels and surfaces (Dark obsidian / charcoal)
    PANEL_BG: ColorBGR = (20, 21, 25)
    PANEL_BORDER: ColorBGR = (50, 55, 68)
    CARD_BG: ColorBGR = (25, 26, 32)
    CARD_BORDER: ColorBGR = (60, 66, 80)
    DIVIDER: ColorBGR = (42, 45, 55)

    # Typography
    TEXT_PRIMARY: ColorBGR = (250, 250, 252)    # Crisp high-contrast white
    TEXT_SECONDARY: ColorBGR = (165, 170, 182)  # Muted silver for labels
    TEXT_DIM: ColorBGR = (110, 115, 128)        # Dark gray for shortcuts/units

    # Status Semantics
    SUCCESS: ColorBGR = (75, 215, 120)          # Emerald green (Detected / Open / Center)
    WARNING: ColorBGR = (50, 180, 255)          # Warm amber (Turned away / Hysteresis)
    DANGER: ColorBGR = (75, 75, 235)            # Soft crimson (Closed eyes / Disconnected)
    ACCENT: ColorBGR = (245, 185, 45)           # Clean cyan/blue accent for indicators
    NEUTRAL: ColorBGR = (125, 130, 140)         # Slate gray (Unknown / Idle)


@dataclass(frozen=True)
class UITypography:
    """Typography scales and line weights for OpenCV Hershey rendering."""

    FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    LINE_TYPE: int = cv2.LINE_AA

    # Scales
    SCALE_TITLE: float = 0.54
    SCALE_HEADER: float = 0.44
    SCALE_BADGE: float = 0.50
    SCALE_METRIC: float = 0.44
    SCALE_CAPTION: float = 0.38

    # Thicknesses
    THICK_TITLE: int = 2
    THICK_HEADER: int = 1
    THICK_BADGE: int = 2
    THICK_METRIC: int = 1
    THICK_CAPTION: int = 1


@dataclass(frozen=True)
class UILayout:
    """Spacing, margins, and layout dimensions."""

    HEADER_HEIGHT: int = 44
    FOOTER_HEIGHT: int = 32
    CARD_MARGIN_X: int = 18
    CARD_MARGIN_BOTTOM: int = 14
    CARD_PADDING_X: int = 14
    CARD_PADDING_Y: int = 12
    DEFAULT_CARD_WIDTH: int = 270
    DEFAULT_CARD_HEIGHT: int = 135
    DEFAULT_OPACITY: float = 0.80
