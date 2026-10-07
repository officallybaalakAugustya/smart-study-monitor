"""Object detection modules for Smart Study Monitor.

Provides PhoneDetector for real-time mobile phone visibility detection.
"""

from detection.phone_detector import (
    PhoneDetector,
    PhoneDetectionResult,
    PhoneBoundingBox,
)

__all__ = [
    "PhoneDetector",
    "PhoneDetectionResult",
    "PhoneBoundingBox",
]
