"""Tests for database/database.py.

Verifies database initialization, table creation, foreign keys, constraints,
parameterized queries, and connection lifecycle.
"""

import sqlite3
import pytest
from pathlib import Path

from database.database import Database


class TestDatabaseInit:
    def test_in_memory_initialization(self):
        db = Database(db_path=":memory:")
        db.initialize()
        conn = db.get_connection()
        assert conn is not None

        # Check tables exist
        tables = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('sessions', 'behavior_events');"
        )
        table_names = {row["name"] for row in tables}
        assert "sessions" in table_names
        assert "behavior_events" in table_names
        db.close()

    def test_file_based_creates_parent_directory(self, tmp_path: Path):
        db_file = tmp_path / "nested" / "subfolder" / "test.db"
        assert not db_file.parent.exists()

        db = Database(db_path=db_file)
        db.initialize()
        assert db_file.parent.exists()
        assert db_file.exists()
        db.close()

    def test_idempotent_initialization(self, tmp_path: Path):
        db_file = tmp_path / "test.db"
        db = Database(db_path=db_file)
        db.initialize()

        # Insert test session
        db.execute(
            "INSERT INTO sessions (start_time, status, created_at) VALUES (?, ?, ?);",
            ("2026-10-06T10:00:00Z", "ACTIVE", "2026-10-06T10:00:00Z"),
        )
        row = db.fetchone("SELECT COUNT(*) as cnt FROM sessions;")
        assert row["cnt"] == 1

        # Re-initialize - data must NOT be destroyed
        db.initialize()
        row2 = db.fetchone("SELECT COUNT(*) as cnt FROM sessions;")
        assert row2["cnt"] == 1
        db.close()


class TestDatabaseConstraints:
    def test_foreign_key_constraint_enforced(self):
        db = Database(db_path=":memory:")
        db.initialize()

        # Inserting behavior_event with non-existent session_id must fail
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                """
                INSERT INTO behavior_events (session_id, timestamp, state, state_duration, confidence, reason, signals_json)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (999, "2026-10-06T10:00:00Z", "FOCUSED", 0.0, 0.9, "focused", "{}"),
            )
        db.close()

    def test_status_check_constraint(self):
        db = Database(db_path=":memory:")
        db.initialize()

        # Valid statuses
        for valid_status in ("ACTIVE", "COMPLETED", "CANCELLED"):
            db.execute(
                "INSERT INTO sessions (start_time, status, created_at) VALUES (?, ?, ?);",
                ("2026-10-06T10:00:00Z", valid_status, "2026-10-06T10:00:00Z"),
            )

        # Invalid status must fail
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO sessions (start_time, status, created_at) VALUES (?, ?, ?);",
                ("2026-10-06T10:00:00Z", "INVALID_STATUS", "2026-10-06T10:00:00Z"),
            )
        db.close()

    def test_cascade_delete(self):
        db = Database(db_path=":memory:")
        db.initialize()

        cursor = db.execute(
            "INSERT INTO sessions (start_time, status, created_at) VALUES (?, ?, ?);",
            ("2026-10-06T10:00:00Z", "ACTIVE", "2026-10-06T10:00:00Z"),
        )
        session_id = cursor.lastrowid

        db.execute(
            """
            INSERT INTO behavior_events (session_id, timestamp, state, state_duration, confidence, reason, signals_json)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (session_id, "2026-10-06T10:00:00Z", "FOCUSED", 0.0, 0.9, "focused", "{}"),
        )
        assert db.fetchone("SELECT COUNT(*) as cnt FROM behavior_events;")["cnt"] == 1

        # Delete session
        db.execute("DELETE FROM sessions WHERE id = ?;", (session_id,))
        assert db.fetchone("SELECT COUNT(*) as cnt FROM behavior_events;")["cnt"] == 0
        db.close()


class TestDatabaseOperations:
    def test_context_manager(self, tmp_path: Path):
        db_file = tmp_path / "test_ctx.db"
        with Database(db_path=db_file) as db:
            db.execute(
                "INSERT INTO sessions (start_time, status, created_at) VALUES (?, ?, ?);",
                ("2026-10-06T10:00:00Z", "ACTIVE", "2026-10-06T10:00:00Z"),
            )
            rows = db.fetchall("SELECT * FROM sessions;")
            assert len(rows) == 1
        # DB connection should be closed after exit
        assert db._conn is None

    def test_safe_repeated_close(self):
        db = Database(db_path=":memory:")
        db.initialize()
        db.close()
        db.close()  # No error
