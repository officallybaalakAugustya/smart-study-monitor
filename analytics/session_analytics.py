"""Session Analytics module for Smart Study Monitor.

Reconstructs behavioral timelines from stored state transition events,
computes per-state durations, and produces comprehensive session summaries
including Focus Scores without direct SQL access.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from analytics.focus_score import FocusScoreCalculator
from analytics.models import FocusScoreResult, SessionSummary, StateDurations
from behavior.behavior_engine import BehaviorState
from session.event_logger import EventLogger


class SessionAnalytics:
    """Provides high-level session analytics and behavioral timeline reconstruction.

    Consumes data via EventLogger. Does NOT access SQLite with raw SQL directly.
    """

    def __init__(
        self,
        event_logger: Optional[EventLogger] = None,
        focus_calculator: Optional[FocusScoreCalculator] = None,
    ) -> None:
        """Initializes SessionAnalytics.

        Args:
            event_logger: EventLogger data access instance. If None, creates a default.
            focus_calculator: FocusScoreCalculator instance. If None, creates a default.
        """
        self._logger = event_logger or EventLogger()
        self._focus_calculator = focus_calculator or FocusScoreCalculator()

    @property
    def event_logger(self) -> EventLogger:
        """Returns the underlying EventLogger."""
        return self._logger

    @property
    def focus_calculator(self) -> FocusScoreCalculator:
        """Returns the underlying FocusScoreCalculator."""
        return self._focus_calculator

    def get_state_durations(
        self,
        session_id: int,
        as_of: Optional[datetime] = None,
    ) -> StateDurations:
        """Reconstructs state durations for a session from its transition events.

        Timeline reconstruction rules:
            1. Events are sorted chronologically.
            2. The first event establishes the starting state from the session start time.
            3. Each state lasts until the timestamp of the next transition event.
            4. The final recorded state extends until the session end_time (or as_of timestamp).
            5. Durations are non-overlapping and sum to the total timeline duration.

        Args:
            session_id: Target session ID.
            as_of: Reference end datetime if the session has no end_time (active session).

        Returns:
            StateDurations object containing reconstructed seconds for all states.
        """
        session_data = self._logger.get_session(session_id)
        if not session_data:
            return StateDurations()

        events = self._logger.get_session_events(session_id)
        if not events:
            return StateDurations()

        # Sort events chronologically to guarantee correct timeline reconstruction
        sorted_events = sorted(events, key=lambda ev: ev["timestamp"])

        # Parse session start time
        session_start = self._parse_iso_timestamp(session_data["start_time"])

        # Parse session end time if available, or use as_of, or fall back to last event timestamp
        session_end = None
        if session_data.get("end_time"):
            session_end = self._parse_iso_timestamp(session_data["end_time"])
        elif as_of is not None:
            session_end = as_of

        # Map to accumulate duration per canonical state
        accumulated: Dict[str, float] = {state.value: 0.0 for state in BehaviorState}

        num_events = len(sorted_events)

        # Parse event timestamps
        parsed_events = []
        for ev in sorted_events:
            ts = self._parse_iso_timestamp(ev["timestamp"])
            state_str = str(ev["state"]).strip()
            parsed_events.append((ts, state_str))

        # First state begins at min(session_start, first_event_time)
        first_ts, first_state = parsed_events[0]
        effective_start = min(session_start, first_ts) if session_start else first_ts

        # Determine intervals between consecutive events
        for i in range(num_events - 1):
            curr_ts = effective_start if i == 0 else parsed_events[i][0]
            next_ts = parsed_events[i + 1][0]
            state = parsed_events[i][1]

            delta_sec = max(0.0, (next_ts - curr_ts).total_seconds())
            if state in accumulated:
                accumulated[state] += delta_sec
            else:
                accumulated[state] = accumulated.get(state, 0.0) + delta_sec

        # Final event state extends until session_end (or lasts at least 0.0)
        last_ts, last_state = parsed_events[-1]
        last_start = effective_start if num_events == 1 else last_ts

        if session_end is not None:
            final_delta_sec = max(0.0, (session_end - last_start).total_seconds())
        else:
            # If session is active and no as_of provided, use the duration recorded on the last event
            # or 0.0
            last_event_duration = max(0.0, float(sorted_events[-1].get("state_duration", 0.0)))
            final_delta_sec = last_event_duration

        if last_state in accumulated:
            accumulated[last_state] += final_delta_sec
        else:
            accumulated[last_state] = accumulated.get(last_state, 0.0) + final_delta_sec

        return StateDurations.from_dict(accumulated)

    def get_transition_count(self, session_id: int) -> int:
        """Returns the number of behavioral state transitions in the session.

        The first event establishes the baseline state and is not counted as a transition.
        """
        events = self._logger.get_session_events(session_id)
        if len(events) <= 1:
            return 0
        return len(events) - 1

    def get_focus_score(
        self,
        session_id: int,
        as_of: Optional[datetime] = None,
    ) -> FocusScoreResult:
        """Calculates the FocusScoreResult for a session."""
        durations = self.get_state_durations(session_id, as_of=as_of)
        return self._focus_calculator.calculate(durations)

    def analyze_session(
        self,
        session_id: int,
        as_of: Optional[datetime] = None,
    ) -> Optional[SessionSummary]:
        """Performs comprehensive analysis and returns a complete SessionSummary."""
        session_data = self._logger.get_session(session_id)
        if not session_data:
            return None

        durations = self.get_state_durations(session_id, as_of=as_of)
        transition_count = self.get_transition_count(session_id)
        focus_result = self._focus_calculator.calculate(durations)

        # Total duration prefers session duration_seconds, or reconstructed total
        duration_sec = float(session_data.get("duration_seconds") or 0.0)
        if duration_sec <= 0.0:
            duration_sec = durations.total_seconds

        return SessionSummary(
            session_id=session_id,
            start_time=session_data["start_time"],
            end_time=session_data.get("end_time"),
            duration_seconds=duration_sec,
            status=session_data.get("status", "UNKNOWN"),
            state_durations=durations,
            transition_count=transition_count,
            focus_score_result=focus_result,
        )

    def get_session_summary(
        self,
        session_id: int,
        as_of: Optional[datetime] = None,
    ) -> Optional[SessionSummary]:
        """Alias for analyze_session for clear interface naming."""
        return self.analyze_session(session_id, as_of=as_of)

    def get_recent_session_summaries(self, limit: int = 10) -> List[SessionSummary]:
        """Retrieves summaries for the most recent sessions."""
        recent_sessions = self._logger.get_recent_sessions(limit=limit)
        summaries = []
        for s in recent_sessions:
            summary = self.analyze_session(s["id"])
            if summary:
                summaries.append(summary)
        return summaries

    def get_completed_session_summaries(self, limit: Optional[int] = None) -> List[SessionSummary]:
        """Retrieves summaries for completed study sessions, newest first."""
        completed_sessions = self._logger.get_completed_sessions(limit=limit)
        summaries = []
        for s in completed_sessions:
            summary = self.analyze_session(s["id"])
            if summary:
                summaries.append(summary)
        return summaries

    def get_timeline_segments(
        self,
        session_id: int,
        as_of: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Reconstructs contiguous timeline intervals from state transition events.

        Returns:
            List of dicts, each with keys:
            - state: str
            - start_time: str (ISO)
            - end_time: str (ISO)
            - duration_seconds: float
            - confidence: float
            - reason: str
        """
        session_data = self._logger.get_session(session_id)
        if not session_data:
            return []

        events = self._logger.get_session_events(session_id)
        if not events:
            return []

        sorted_events = sorted(events, key=lambda ev: ev["timestamp"])
        session_start = self._parse_iso_timestamp(session_data["start_time"])

        session_end = None
        if session_data.get("end_time"):
            session_end = self._parse_iso_timestamp(session_data["end_time"])
        elif as_of is not None:
            session_end = as_of

        num_events = len(sorted_events)
        parsed = []
        for ev in sorted_events:
            ts = self._parse_iso_timestamp(ev["timestamp"])
            parsed.append((ts, ev))

        first_ts, first_ev = parsed[0]
        effective_start = min(session_start, first_ts) if session_start else first_ts

        segments: List[Dict[str, Any]] = []

        # Intermediate segments
        for i in range(num_events - 1):
            curr_start = effective_start if i == 0 else parsed[i][0]
            next_start = parsed[i + 1][0]
            ev = parsed[i][1]

            dur = max(0.0, (next_start - curr_start).total_seconds())
            segments.append({
                "state": str(ev["state"]).strip(),
                "start_time": curr_start.isoformat(),
                "end_time": next_start.isoformat(),
                "duration_seconds": dur,
                "confidence": float(ev.get("confidence", 0.0)),
                "reason": str(ev.get("reason", "")),
            })

        # Final segment
        last_start = effective_start if num_events == 1 else parsed[-1][0]
        last_ev = parsed[-1][1]

        if session_end is not None:
            final_dur = max(0.0, (session_end - last_start).total_seconds())
            final_end_str = session_end.isoformat()
        else:
            final_dur = max(0.0, float(last_ev.get("state_duration", 0.0)))
            from datetime import timedelta
            final_end_str = (last_start + timedelta(seconds=final_dur)).isoformat()

        segments.append({
            "state": str(last_ev["state"]).strip(),
            "start_time": last_start.isoformat(),
            "end_time": final_end_str,
            "duration_seconds": final_dur,
            "confidence": float(last_ev.get("confidence", 0.0)),
            "reason": str(last_ev.get("reason", "")),
        })

        return segments

    @staticmethod
    def _parse_iso_timestamp(ts_str: str) -> datetime:
        """Safely parses an ISO 8601 string into a timezone-aware datetime."""
        clean_ts = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_ts)
        return dt
