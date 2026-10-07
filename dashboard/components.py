"""Streamlit UI components for the Smart Study Monitor analytics dashboard.

Provides modular, testable presentation components for metrics, distribution
charts, behavioral timelines, and historical session logs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from analytics.models import SessionSummary
from dashboard.formatting import (
    format_duration,
    format_focus_score,
    format_iso_datetime,
    get_state_color,
    get_state_label,
    prepare_event_table_data,
    prepare_session_history_data,
    prepare_state_distribution_data,
)


def render_header() -> None:
    """Renders the dashboard top header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            """
            <div style="margin-bottom: 0.5rem;">
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em;">
                    SMART STUDY MONITOR
                </h1>
                <p style="margin: 0.2rem 0 0 0; color: #64748B; font-size: 1.05rem;">
                    Study Session Analytics & Behavioral Insights
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 0.8rem;">
                <span style="background-color: #E2E8F0; color: #334155; padding: 0.3rem 0.7rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600;">
                    Offline Analytics
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()


def render_sidebar(
    completed_summaries: List[SessionSummary],
    active_session: Optional[Dict[str, Any]] = None,
    db_path: str = "",
) -> Optional[int]:
    """Renders the sidebar navigation and session selector.

    Args:
        completed_summaries: List of completed SessionSummary objects (newest first).
        active_session: Active session row dict if one is currently in progress.
        db_path: Path to the SQLite database.

    Returns:
        The selected session_id, or None if no completed sessions exist.
    """
    st.sidebar.markdown("### 📚 Sessions")

    if active_session:
        st.sidebar.info(
            f"🟢 **Session #{active_session['id']} in progress**\n"
            f"Started: {format_iso_datetime(active_session['start_time'])}\n\n"
            f"*Active sessions appear in analytics once completed.*"
        )

    if not completed_summaries:
        st.sidebar.warning("No completed study sessions found.")
        st.sidebar.markdown("---")
        st.sidebar.caption(f"📁 Database: `{db_path}`")
        return None

    # Construct session options
    session_map: Dict[str, int] = {}
    options: List[str] = []
    for s in completed_summaries:
        score_txt = f"Score: {format_focus_score(s.focus_score)}"
        dur_txt = format_duration(s.duration_seconds)
        date_txt = format_iso_datetime(s.start_time, include_time=False)
        label = f"Session #{s.session_id} • {date_txt} ({dur_txt}, {score_txt})"
        options.append(label)
        session_map[label] = s.session_id

    selected_label = st.sidebar.selectbox(
        "Select Completed Session:",
        options=options,
        index=0,
    )
    selected_id = session_map.get(selected_label)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"📁 Database: `{db_path}`")
    st.sidebar.caption(f"Total Completed Sessions: {len(completed_summaries)}")

    return selected_id


def render_summary_metrics(summary: SessionSummary) -> None:
    """Renders top KPI metric cards for the selected session."""
    c1, c2, c3, c4 = st.columns(4)

    score_val = summary.focus_score
    score_display = f"{score_val:.1f}" if score_val is not None else "N/A"
    score_help = (
        "Calculated from scorable duration (FOCUSED / Scored Duration * 100). "
        "NO_FACE and UNKNOWN are excluded by default."
    )

    with c1:
        st.metric(
            label="Focus Score",
            value=f"{score_display} / 100",
            help=score_help,
        )
    with c2:
        st.metric(
            label="Session Duration",
            value=format_duration(summary.duration_seconds),
            help="Total recorded time from session start to session end.",
        )
    with c3:
        st.metric(
            label="Focused Time",
            value=format_duration(summary.focused_seconds),
            help="Total confirmed time spent in attentive posture.",
        )
    with c4:
        st.metric(
            label="Behavior Transitions",
            value=str(summary.transition_count),
            help="Number of times behavioral state transitioned between states.",
        )


def render_focus_score_breakdown(summary: SessionSummary) -> None:
    """Renders transparent explanation and mathematical breakdown of the Focus Score."""
    res = summary.focus_score_result

    with st.expander("ℹ️ Focus Score Breakdown & Transparency", expanded=(res.score is None)):
        if res.score is None:
            st.warning(
                f"**Focus Score Unavailable**: {res.reason or 'Insufficient scorable session data.'}"
            )
        else:
            st.markdown(
                f"""
                **Calculation Summary:**
                - **Focused Duration**: `{format_duration(res.focused_seconds)}` ({res.focused_seconds:.1f}s)
                - **Scored Duration**: `{format_duration(res.scored_seconds)}` ({res.scored_seconds:.1f}s)
                - **Excluded Duration**: `{format_duration(res.excluded_seconds)}` ({res.excluded_seconds:.1f}s) — *NO_FACE and UNKNOWN states are excluded from the denominator.*
                - **Formula**: `({res.focused_seconds:.1f}s / {res.scored_seconds:.1f}s) × 100 = {res.score:.2f}`
                """
            )

        st.caption(
            "📌 **Disclaimer**: The Focus Score is an application-defined behavioral metric "
            "based on computer-vision observations. It is not a clinical, psychological, "
            "or scientifically validated assessment of human intelligence, cognitive attention, "
            "or academic productivity."
        )


def render_state_distribution(summary: SessionSummary) -> None:
    """Renders state duration distribution cards and a horizontal breakdown chart."""
    st.subheader("📊 Behavioral State Distribution")

    dist_rows = prepare_state_distribution_data(summary, include_zero_duration=False)
    if not dist_rows:
        st.info("No state durations recorded for this session.")
        return

    # Visual horizontal progress / distribution bar
    col_chart, col_legend = st.columns([3, 2])

    with col_chart:
        # Create a clean horizontal bar chart using Plotly
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            for row in dist_rows:
                fig.add_trace(go.Bar(
                    y=["Session Time"],
                    x=[row["duration_seconds"]],
                    name=row["label"],
                    orientation="h",
                    marker=dict(color=row["color"]),
                    hovertemplate=f"<b>{row['label']}</b><br>Duration: {row['duration_formatted']} ({row['percentage']:.1f}%)<extra></extra>",
                ))
            fig.update_layout(
                barmode="stack",
                height=130,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                xaxis=dict(showgrid=True, title="Duration (seconds)"),
                yaxis=dict(showticklabels=False),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            # Fallback to Streamlit native progress if plotly has issues
            for row in dist_rows:
                st.write(f"**{row['label']}**: {row['duration_formatted']} ({row['percentage']:.1f}%)")
                st.progress(min(1.0, max(0.0, row["percentage"] / 100.0)))

    with col_legend:
        # Render clean summary table
        df_dist = pd.DataFrame([
            {
                "State": r["label"],
                "Duration": r["duration_formatted"],
                "Share": f"{r['percentage']:.1f}%",
            }
            for r in dist_rows
        ])
        st.dataframe(df_dist, use_container_width=True, hide_index=True)


def render_timeline(timeline_segments: List[Dict[str, Any]]) -> None:
    """Renders chronological timeline chart and transitions view."""
    st.subheader("⏱️ Session Timeline")

    if not timeline_segments:
        st.info("No timeline events available for this session.")
        return

    # Chronological Plotly Gantt / timeline representation
    try:
        import plotly.express as px
        df_timeline = pd.DataFrame([
            {
                "State": get_state_label(s["state"]),
                "Raw State": s["state"],
                "Start": s["start_time"],
                "End": s["end_time"],
                "Duration": format_duration(s["duration_seconds"]),
                "Duration_Sec": s["duration_seconds"],
                "Color": get_state_color(s["state"]),
                "Reason": s.get("reason", ""),
            }
            for s in timeline_segments
        ])

        color_discrete_map = {
            get_state_label(st_key): get_state_color(st_key)
            for st_key in [
                "FOCUSED", "DISTRACTED", "DROWSY", "PHONE_DETECTED",
                "AWAY_FROM_DESK", "NO_FACE", "UNKNOWN"
            ]
        }

        fig = px.timeline(
            df_timeline,
            x_start="Start",
            x_end="End",
            y="State",
            color="State",
            color_discrete_map=color_discrete_map,
            hover_data={"Duration": True, "Reason": True, "Start": True, "End": True},
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis=dict(title="Time", showgrid=True),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def render_event_table(timeline_segments: List[Dict[str, Any]]) -> None:
    """Renders detailed table of state transitions and reasons."""
    st.subheader("📋 State Transition Log")

    if not timeline_segments:
        st.info("No state transitions recorded.")
        return

    table_data = prepare_event_table_data(timeline_segments)
    df_events = pd.DataFrame(table_data)

    st.dataframe(
        df_events[["Time", "State", "Duration", "Confidence", "Reason"]],
        use_container_width=True,
        hide_index=True,
    )


def render_session_history(summaries: List[SessionSummary]) -> None:
    """Renders completed session history table and trend chart if multiple sessions exist."""
    st.subheader("📜 Session History")

    if not summaries:
        st.info("No historical sessions recorded.")
        return

    history_data = prepare_session_history_data(summaries)
    df_history = pd.DataFrame(history_data)

    # Trend Chart if >= 2 completed sessions
    valid_scores = [s for s in summaries if s.focus_score is not None]
    if len(valid_scores) >= 2:
        st.markdown("#### Focus Score Trend")
        try:
            import plotly.express as px
            df_trend = pd.DataFrame([
                {
                    "Session": f"Session #{s.session_id}",
                    "Date": format_iso_datetime(s.start_time, include_time=True),
                    "Focus Score": s.focus_score,
                }
                for s in reversed(valid_scores)  # chronological order for trend
            ])
            fig = px.line(
                df_trend,
                x="Date",
                y="Focus Score",
                markers=True,
                title="Focus Score over Sessions",
            )
            fig.update_yaxes(range=[0, 105])
            fig.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    st.dataframe(
        df_history[["Session ID", "Date", "Duration", "Focus Score", "Focused Time", "Transitions", "Status"]],
        use_container_width=True,
        hide_index=True,
    )
