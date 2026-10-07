"""Smart Study Monitor - Analytics Dashboard.

Main Streamlit application entry point. Visualizes completed study sessions,
behavioral timeline reconstructions, state distributions, and Focus Scores.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from analytics.session_analytics import SessionAnalytics
from config.config import config
from database.database import Database
from dashboard.components import (
    render_event_table,
    render_focus_score_breakdown,
    render_header,
    render_session_history,
    render_sidebar,
    render_state_distribution,
    render_summary_metrics,
    render_timeline,
)
from session.event_logger import EventLogger


def init_analytics_layer() -> tuple[EventLogger, SessionAnalytics, str]:
    """Initializes the read-only analytics layer with the configured SQLite path.

    Returns:
        Tuple of (EventLogger, SessionAnalytics, db_path_str).
    """
    database = Database(db_path=config.database.db_path)
    logger = EventLogger(db=database)
    analytics = SessionAnalytics(event_logger=logger)
    return logger, analytics, str(database.db_path)


def main() -> None:
    dash_cfg = config.dashboard
    st.set_page_config(
        page_title=dash_cfg.page_title,
        page_icon=dash_cfg.page_icon,
        layout=dash_cfg.layout,
        initial_sidebar_state=dash_cfg.initial_sidebar_state,
    )

    # Initialize data layer
    logger, analytics, db_path_str = init_analytics_layer()
    db_file = Path(db_path_str)

    # Sidebar refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Empty/Missing database check
    if not db_file.exists():
        render_header()
        st.warning(
            f"### ⚠️ Database Not Found\n\n"
            f"No SQLite database exists at `{db_path_str}`.\n\n"
            f"Please run the study monitor first to record a session:\n"
            f"```bash\npython app.py\n```"
        )
        return

    # Check for active session
    recent = logger.get_recent_sessions(limit=1)
    active_session = recent[0] if recent and recent[0].get("status") == "ACTIVE" else None

    # Load completed session summaries
    try:
        completed_summaries = analytics.get_completed_session_summaries()
    except Exception as exc:
        render_header()
        st.error(f"Error accessing session database: {exc}")
        return

    # Render sidebar session selector
    selected_session_id = render_sidebar(
        completed_summaries=completed_summaries,
        active_session=active_session,
        db_path=db_path_str,
    )

    render_header()

    if not completed_summaries or selected_session_id is None:
        st.info(
            "### 📖 No Completed Sessions Yet\n\n"
            "Completed study sessions will appear here once you finish a session in the monitor.\n\n"
            "To start a study session, run:\n"
            "```bash\npython app.py\n```"
        )
        return

    # Load analytics for selected session
    selected_summary = analytics.get_session_summary(selected_session_id)
    if not selected_summary:
        st.error(f"Could not load analytics for Session #{selected_session_id}.")
        return

    timeline_segments = analytics.get_timeline_segments(selected_session_id)

    # 1. Top Summary Cards
    render_summary_metrics(selected_summary)

    # 2. Focus Score Transparency Breakdown
    render_focus_score_breakdown(selected_summary)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. State Distribution Breakdown
    render_state_distribution(selected_summary)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Behavioral Timeline Chart
    render_timeline(timeline_segments)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Chronological Event Table
    render_event_table(timeline_segments)

    st.markdown("<br>", unsafe_allow_html=True)

    # 6. Historical Sessions Log & Trend Chart
    render_session_history(completed_summaries)


if __name__ == "__main__":
    main()
