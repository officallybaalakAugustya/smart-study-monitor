"""Database management module for Smart Study Monitor.

Provides SQLite initialization, schema definition, and safe query execution.
Designed for local study session persistence and behavioral event tracking.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

from config.config import BASE_DIR, config


class Database:
    """Manages SQLite database connections, schema setup, and query execution."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        timeout_sec: Optional[float] = None,
    ) -> None:
        """Initializes the database manager.

        Args:
            db_path: Path to SQLite database file or ':memory:'. If None, defaults
                     to config.database.db_path.
            timeout_sec: Timeout in seconds for database locks. If None, defaults
                         to config.database.timeout_sec.
        """
        if db_path is not None:
            raw_path = str(db_path)
            if raw_path != ":memory:":
                p = Path(raw_path)
                self._db_path = str(p if p.is_absolute() else (BASE_DIR / p).resolve())
            else:
                self._db_path = ":memory:"
        else:
            self._db_path = str(config.database.db_path)

        self._timeout_sec = (
            timeout_sec if timeout_sec is not None else config.database.timeout_sec
        )
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def db_path(self) -> str:
        """Returns the database file path or identifier."""
        return self._db_path

    def get_connection(self) -> sqlite3.Connection:
        """Gets or establishes the SQLite connection."""
        if self._conn is None:
            # Ensure parent directory exists for file-based databases
            if self._db_path != ":memory:":
                path_obj = Path(self._db_path)
                path_obj.parent.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(
                self._db_path,
                timeout=self._timeout_sec,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row

            # Enforce foreign key constraints
            self._conn.execute("PRAGMA foreign_keys = ON;")

            # Enable WAL mode for file-based DBs for better concurrency and write performance
            if self._db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL;")

        return self._conn

    def initialize(self) -> None:
        """Initializes the database schema idempotently.

        Safe to call multiple times; existing data is preserved.
        """
        conn = self.get_connection()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_seconds REAL,
                    status TEXT NOT NULL CHECK(status IN ('ACTIVE', 'COMPLETED', 'CANCELLED')),
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS behavior_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    state TEXT NOT NULL,
                    state_duration REAL NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    signals_json TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_behavior_events_session_id
                ON behavior_events(session_id);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_start_time
                ON sessions(start_time);
                """
            )

    def execute(self, query: str, params: Union[tuple, list, dict] = ()) -> sqlite3.Cursor:
        """Executes a single parameterized query and returns the cursor."""
        conn = self.get_connection()
        with conn:
            return conn.execute(query, params)

    def executemany(self, query: str, seq_of_params: Iterable[Union[tuple, list, dict]]) -> sqlite3.Cursor:
        """Executes a query against all parameter sequences."""
        conn = self.get_connection()
        with conn:
            return conn.executemany(query, seq_of_params)

    def fetchone(self, query: str, params: Union[tuple, list, dict] = ()) -> Optional[sqlite3.Row]:
        """Executes a parameterized query and fetches the first row."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchone()

    def fetchall(self, query: str, params: Union[tuple, list, dict] = ()) -> List[sqlite3.Row]:
        """Executes a parameterized query and fetches all rows."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def close(self) -> None:
        """Safely closes the database connection if open."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            finally:
                self._conn = None

    def __enter__(self) -> Database:
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
