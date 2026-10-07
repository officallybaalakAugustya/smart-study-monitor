"""Event Logger module for Smart Study Monitor.

Provides high-level data access and persistence for study sessions and
behavioral state transition events using SQLite.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from behavior.behavior_engine import BehaviorState
from database.database import Database

logger = logging.getLogger(__name__)


class EventLogger:
    """Manages persistence and retrieval of sessions and behavior events in SQLite."""

    VALID_STATUSES = {"ACTIVE", "COMPLETED", "CANCELLED"}
    VALID_STATES = {state.value for state in BehaviorState}

    def __init__(self, db: Optional[Database] = None) -> None:
        """Initializes the event logger.

        Args:
            db: Database instance. If None, creates a default Database instance.
        """
        self._db = db or Database()

    @property
    def database(self) -> Database:
        """Returns the underlying Database instance."""
        return self._db

    def initialize_database(self) -> None:
        """Ensures the database and required tables are initialized."""
        self._db.initialize()

    def create_session(
        self,
        start_time: str,
        created_at: Optional[str] = None,
        status: str = "ACTIVE",
    ) -> int:
        """Creates a new study session row.

        Args:
            start_time: ISO 8601 wall-clock timestamp when session began.
            created_at: ISO 8601 wall-clock timestamp of creation. If None, uses start_time.
            status: Initial session status ("ACTIVE", "COMPLETED", "CANCELLED").

        Returns:
            The generated integer session ID.

        Raises:
            ValueError: If inputs fail validation.
        """
        self._validate_timestamp(start_time, "start_time")
        created_ts = created_at or start_time
        self._validate_timestamp(created_ts, "created_at")

        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {self.VALID_STATUSES}")

        self.initialize_database()

        cursor = self._db.execute(
            """
            INSERT INTO sessions (start_time, end_time, duration_seconds, status, created_at)
            VALUES (?, NULL, 0.0, ?, ?);
            """,
            (start_time, status, created_ts),
        )
        session_id = cursor.lastrowid
        if session_id is None:
            raise RuntimeError("Failed to insert session row")
        return int(session_id)

    def end_session(
        self,
        session_id: int,
        end_time: str,
        duration_seconds: float,
        status: str = "COMPLETED",
    ) -> bool:
        """Updates a session row to record completion or cancellation.

        Args:
            session_id: Target session ID.
            end_time: ISO 8601 wall-clock timestamp when session ended.
            duration_seconds: Total runtime duration in seconds (>= 0).
            status: Final status ("COMPLETED" or "CANCELLED").

        Returns:
            True if a row was updated, False if session_id was not found.

        Raises:
            ValueError: If inputs fail validation.
        """
        if session_id <= 0:
            raise ValueError(f"session_id must be > 0, got {session_id}")
        self._validate_timestamp(end_time, "end_time")

        if duration_seconds < 0.0:
            raise ValueError(f"duration_seconds must be >= 0.0, got {duration_seconds}")

        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {self.VALID_STATUSES}")

        cursor = self._db.execute(
            """
            UPDATE sessions
            SET end_time = ?, duration_seconds = ?, status = ?
            WHERE id = ?;
            """,
            (end_time, float(duration_seconds), status, session_id),
        )
        return cursor.rowcount > 0

    def log_behavior_event(
        self,
        session_id: int,
        timestamp: str,
        state: Union[BehaviorState, str],
        state_duration: float,
        confidence: float,
        reason: str,
        signals: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Logs a behavioral event into behavior_events table.

        Args:
            session_id: Valid active session ID.
            timestamp: ISO 8601 wall-clock timestamp when transition occurred.
            state: BehaviorState enum or state string name.
            state_duration: Duration of this state in seconds (>= 0).
            confidence: Confidence score in [0.0, 1.0].
            reason: Stable structured reason code.
            signals: Optional dictionary of sensor and derived inputs.

        Returns:
            The generated integer event ID.

        Raises:
            ValueError: If inputs fail validation.
        """
        if session_id <= 0:
            raise ValueError(f"session_id must be > 0, got {session_id}")

        self._validate_timestamp(timestamp, "timestamp")

        state_str = state.value if isinstance(state, BehaviorState) else str(state)
        if state_str not in self.VALID_STATES:
            raise ValueError(f"Invalid behavior state '{state_str}'. Must be one of {self.VALID_STATES}")

        if state_duration < 0.0:
            raise ValueError(f"state_duration must be >= 0.0, got {state_duration}")

        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")

        if not reason or not str(reason).strip():
            raise ValueError("reason must be a non-empty string code")

        # Verify session exists
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session with id {session_id} does not exist")

        signals_json = json.dumps(signals if signals is not None else {}, default=str)

        cursor = self._db.execute(
            """
            INSERT INTO behavior_events (session_id, timestamp, state, state_duration, confidence, reason, signals_json)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                session_id,
                timestamp,
                state_str,
                float(state_duration),
                float(confidence),
                str(reason).strip(),
                signals_json,
            ),
        )
        event_id = cursor.lastrowid
        if event_id is None:
            raise RuntimeError("Failed to insert behavior event row")
        return int(event_id)

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves session details by session_id."""
        if session_id <= 0:
            return None
        row = self._db.fetchone("SELECT * FROM sessions WHERE id = ?;", (session_id,))
        if row is None:
            return None
        return dict(row)

    def get_session_events(self, session_id: int) -> List[Dict[str, Any]]:
        """Retrieves all behavior events for a specific session ordered chronologically.

        Deserializes signals_json into a signals dictionary.
        """
        if session_id <= 0:
            return []
        rows = self._db.fetchall(
            "SELECT * FROM behavior_events WHERE session_id = ? ORDER BY id ASC;",
            (session_id,),
        )
        events = []
        for row in rows:
            event_dict = dict(row)
            raw_json = event_dict.get("signals_json")
            if raw_json:
                try:
                    event_dict["signals"] = json.loads(raw_json)
                except Exception:
                    event_dict["signals"] = {}
            else:
                event_dict["signals"] = {}
            events.append(event_dict)
        return events

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves the most recent sessions ordered from newest to oldest."""
        if limit <= 0:
            return []
        rows = self._db.fetchall(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?;",
            (limit,),
        )
        return [dict(r) for r in rows]

    def get_completed_sessions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves completed study sessions ordered newest first."""
        query = "SELECT * FROM sessions WHERE status = 'COMPLETED' ORDER BY id DESC"
        params: list[Any] = []
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._db.fetchall(query, tuple(params))
        return [dict(r) for r in rows]

    def recover_unfinished_sessions(self, fallback_status: str = "CANCELLED") -> List[int]:
        """Recovers any unfinished ACTIVE sessions lingering in the database (e.g. from an abrupt exit).

        For each ACTIVE session:
          1. Queries the latest behavior_event recorded for that session.
          2. Sets end_time to the latest event's timestamp (or start_time if no events were recorded).
          3. Computes duration_seconds between start_time and end_time.
          4. Updates session status to fallback_status ('CANCELLED').
          5. Preserves all recorded events without modification.

        Returns:
            List of recovered session IDs.
        """
        self.initialize_database()
        active_rows = self._db.fetchall(
            "SELECT id, start_time FROM sessions WHERE status = 'ACTIVE' ORDER BY id ASC;"
        )
        if not active_rows:
            return []

        recovered_ids: List[int] = []
        for row in active_rows:
            sid = int(row["id"])
            start_iso = str(row["start_time"])

            last_ev = self._db.fetchone(
                "SELECT timestamp FROM behavior_events WHERE session_id = ? ORDER BY id DESC LIMIT 1;",
                (sid,),
            )
            end_iso = str(last_ev["timestamp"]) if last_ev else start_iso

            try:
                start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
                duration = max(0.0, (end_dt - start_dt).total_seconds())
            except Exception:
                duration = 0.0

            self._db.execute(
                """
                UPDATE sessions
                SET end_time = ?, duration_seconds = ?, status = ?
                WHERE id = ?;
                """,
                (end_iso, duration, fallback_status, sid),
            )
            logger.info(
                "Recovered unfinished session #%d (marked %s, duration=%.1fs)",
                sid,
                fallback_status,
                duration,
            )
            recovered_ids.append(sid)

        return recovered_ids

    def close(self) -> None:
        """Safely closes the underlying database connection."""
        self._db.close()

    @staticmethod
    def _validate_timestamp(timestamp_str: str, field_name: str) -> None:
        """Validates that a string is a non-empty valid ISO 8601 timestamp."""
        if not timestamp_str or not isinstance(timestamp_str, str):
            raise ValueError(f"{field_name} must be a non-empty ISO 8601 timestamp string")
        try:
            # Handles Python standard ISO formats including +00:00, Z, offsets
            clean_ts = timestamp_str.replace("Z", "+00:00")
            datetime.fromisoformat(clean_ts)
        except Exception as exc:
            raise ValueError(f"{field_name} is not a valid ISO 8601 timestamp: '{timestamp_str}'") from exc
