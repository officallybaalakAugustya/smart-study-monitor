"""Formatting and data transformation utilities for the analytics dashboard.

Separates data preparation and formatting from Streamlit UI rendering to
enable deterministic unit testing without a browser or UI context.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from analytics.models import SessionSummary
from behavior.behavior_engine import BehaviorState


# Semantic color palette matching the Smart Study Monitor UI
STATE_COLORS: Dict[str, str] = {
    BehaviorState.FOCUSED.value: "#22C55E",         # Emerald Green (Positive)
    BehaviorState.DISTRACTED.value: "#F59E0B",      # Amber (Warning)
    BehaviorState.DROWSY.value: "#EF4444",          # Red (Alert)
    BehaviorState.PHONE_DETECTED.value: "#EC4899",  # Pink/Purple (Alert)
    BehaviorState.AWAY_FROM_DESK.value: "#64748B",  # Slate (Away)
    BehaviorState.NO_FACE.value: "#94A3B8",         # Muted Slate (Unobservable)
    BehaviorState.UNKNOWN.value: "#CBD5E1",         # Light Slate (Invalid)
}

STATE_LABELS: Dict[str, str] = {
    BehaviorState.FOCUSED.value: "Focused",
    BehaviorState.DISTRACTED.value: "Distracted",
    BehaviorState.DROWSY.value: "Drowsy",
    BehaviorState.PHONE_DETECTED.value: "Phone Detected",
    BehaviorState.AWAY_FROM_DESK.value: "Away From Desk",
    BehaviorState.NO_FACE.value: "No Face Detected",
    BehaviorState.UNKNOWN.value: "Unknown / Incomplete",
}


def format_duration(seconds: float) -> str:
    """Formats duration in seconds to a human-readable HH:MM:SS string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string (e.g. '01:26:14' or '00:05:32').
    """
    total_sec = max(0, int(round(seconds)))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_iso_datetime(iso_str: Optional[str], include_time: bool = True) -> str:
    """Formats an ISO 8601 string to a readable date/time string.

    Args:
        iso_str: ISO 8601 timestamp string.
        include_time: If True, includes HH:MM:SS.

    Returns:
        Formatted string (e.g. 'Oct 7, 2026, 01:42:18') or '--' if None/empty.
    """
    if not iso_str or not isinstance(iso_str, str):
        return "--"
    try:
        clean_ts = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_ts)
        if include_time:
            return dt.strftime("%b %d, %Y, %H:%M:%S")
        return dt.strftime("%b %d, %Y")
    except Exception:
        return str(iso_str)


def format_focus_score(score: Optional[float]) -> str:
    """Formats the Focus Score for presentation.

    Args:
        score: Focus score float in [0.0, 100.0] or None.

    Returns:
        Formatted score string (e.g. '84.7' or 'N/A').
    """
    if score is None:
        return "N/A"
    return f"{score:.1f}"


def get_state_color(state: str) -> str:
    """Returns the hex color code for a behavioral state."""
    return STATE_COLORS.get(state, "#94A3B8")


def get_state_label(state: str) -> str:
    """Returns a friendly display label for a behavioral state."""
    return STATE_LABELS.get(state, state)


def prepare_state_distribution_data(
    summary: SessionSummary,
    include_zero_duration: bool = False,
) -> List[Dict[str, Any]]:
    """Prepares structured distribution rows for charts and distribution cards.

    Args:
        summary: The SessionSummary instance.
        include_zero_duration: Whether to include states with 0 seconds duration.

    Returns:
        List of dicts with state metadata, duration, percentage, and color.
    """
    if summary is None or getattr(summary, "state_durations", None) is None:
        return []

    dur_dict = summary.state_durations.to_dict()
    total_seconds = summary.duration_seconds
    if total_seconds <= 0.0:
        total_seconds = summary.state_durations.total_seconds

    rows: List[Dict[str, Any]] = []
    for state_key in [
        BehaviorState.FOCUSED.value,
        BehaviorState.DISTRACTED.value,
        BehaviorState.DROWSY.value,
        BehaviorState.PHONE_DETECTED.value,
        BehaviorState.AWAY_FROM_DESK.value,
        BehaviorState.NO_FACE.value,
        BehaviorState.UNKNOWN.value,
    ]:
        dur = max(0.0, float(dur_dict.get(state_key, 0.0)))
        if dur == 0.0 and not include_zero_duration:
            continue

        pct = (dur / total_seconds * 100.0) if total_seconds > 0.0 else 0.0
        rows.append({
            "state": state_key,
            "label": get_state_label(state_key),
            "duration_seconds": dur,
            "duration_formatted": format_duration(dur),
            "percentage": pct,
            "color": get_state_color(state_key),
        })

    return rows


def prepare_event_table_data(timeline_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepares structured rows for the chronological event table.

    Args:
        timeline_segments: List of timeline segment dictionaries from SessionAnalytics.

    Returns:
        List of formatted event table rows.
    """
    rows: List[Dict[str, Any]] = []
    for seg in timeline_segments:
        state_key = str(seg.get("state", "UNKNOWN"))
        dur_sec = float(seg.get("duration_seconds", 0.0))
        conf = float(seg.get("confidence", 0.0))

        rows.append({
            "Time": format_iso_datetime(seg.get("start_time"), include_time=True),
            "End Time": format_iso_datetime(seg.get("end_time"), include_time=True),
            "State": get_state_label(state_key),
            "Raw State": state_key,
            "Duration": format_duration(dur_sec),
            "Duration (s)": round(dur_sec, 1),
            "Confidence": f"{conf:.2f}",
            "Reason": str(seg.get("reason", "")),
            "Color": get_state_color(state_key),
        })

    return rows


def prepare_session_history_data(summaries: List[SessionSummary]) -> List[Dict[str, Any]]:
    """Prepares summary rows for the historical session list and trend charts.

    Args:
        summaries: List of completed SessionSummary objects.

    Returns:
        List of formatted history rows.
    """
    rows: List[Dict[str, Any]] = []
    for s in summaries:
        score_display = format_focus_score(s.focus_score)
        rows.append({
            "Session ID": s.session_id,
            "Date": format_iso_datetime(s.start_time, include_time=False),
            "Start Time": format_iso_datetime(s.start_time, include_time=True),
            "Duration": format_duration(s.duration_seconds),
            "Duration (s)": s.duration_seconds,
            "Focus Score": score_display,
            "Focus Score Value": s.focus_score if s.focus_score is not None else float("nan"),
            "Focused Time": format_duration(s.focused_seconds),
            "Transitions": s.transition_count,
            "Status": s.status,
        })
    return rows
