"""Behavior Engine package for Smart Study Monitor.

Combines individual CV signals (face detection, eye tracking, head pose,
phone detection) into stable, temporally-confirmed behavioral states.
"""

from behavior.behavior_engine import (
    BehaviorEngine,
    BehaviorResult,
    BehaviorState,
    BehaviorReason,
)

__all__ = [
    "BehaviorEngine",
    "BehaviorResult",
    "BehaviorState",
    "BehaviorReason",
]
