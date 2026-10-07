"""Overlay renderer for Smart Study Monitor.

Draws high-contrast, translucent HUD panels, status badges, metrics,
and controls on the video frame with zero algorithmic logic.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ui.theme import UIColors, UITypography, UILayout
from vision.face_detector import FaceDetectionResult
from vision.eye_tracker import EyeTrackingResult, EyeState
from vision.head_pose import HeadPoseResult, HeadDirection
from detection.phone_detector import PhoneDetectionResult
from behavior.behavior_engine import BehaviorResult, BehaviorState


class OverlayRenderer:
    """Manages all graphical HUD elements and status overlays."""

    def __init__(self, opacity: float = UILayout.DEFAULT_OPACITY) -> None:
        self.opacity = opacity

    # ------------------------------------------------------------------
    # Drawing primitives (fast & memory-safe)
    # ------------------------------------------------------------------

    def draw_panel(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        bg_color: Tuple[int, int, int] = UIColors.CARD_BG,
        border_color: Optional[Tuple[int, int, int]] = UIColors.CARD_BORDER,
        opacity: Optional[float] = None,
    ) -> None:
        """Renders an alpha-blended translucent rectangular card panel."""
        alpha = opacity if opacity is not None else self.opacity
        fh, fw = frame.shape[:2]

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)

        if x1 >= x2 or y1 >= y2:
            return

        # Perform sub-ROI alpha blending (avoiding full-frame copies)
        roi = frame[y1:y2, x1:x2]
        overlay = np.full_like(roi, bg_color, dtype=np.uint8)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

        if border_color is not None:
            cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), border_color, 1, cv2.LINE_AA)

    def draw_status_dot(
        self,
        frame: np.ndarray,
        center_x: int,
        center_y: int,
        color: Tuple[int, int, int],
        filled: bool = True,
        radius: int = 4,
    ) -> None:
        """Draws an anti-aliased geometric status circle."""
        thickness = -1 if filled else 1
        cv2.circle(frame, (center_x, center_y), radius, color, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Header and Footer
    # ------------------------------------------------------------------

    def draw_header(
        self,
        frame: np.ndarray,
        fps: float,
        is_active: bool = True,
    ) -> None:
        """Draws the top application header with title and live status badge."""
        w = frame.shape[1]
        h_bar = UILayout.HEADER_HEIGHT

        # Background panel
        self.draw_panel(frame, 0, 0, w, h_bar, bg_color=UIColors.PANEL_BG, border_color=None, opacity=0.88)
        cv2.line(frame, (0, h_bar - 1), (w, h_bar - 1), UIColors.PANEL_BORDER, 1, cv2.LINE_AA)

        # Title and category
        cv2.putText(
            frame,
            "SMART STUDY MONITOR",
            (18, 28),
            UITypography.FONT,
            UITypography.SCALE_TITLE,
            UIColors.TEXT_PRIMARY,
            UITypography.THICK_TITLE,
            UITypography.LINE_TYPE,
        )

        cv2.putText(
            frame,
            "Real-Time Monitor",
            (240, 28),
            UITypography.FONT,
            UITypography.SCALE_HEADER,
            UIColors.TEXT_SECONDARY,
            UITypography.THICK_HEADER,
            UITypography.LINE_TYPE,
        )

        # Live status badge (Right-aligned)
        badge_text = "MONITORING ACTIVE" if is_active else "IDLE"
        dot_color = UIColors.SUCCESS if is_active else UIColors.NEUTRAL
        badge_x = w - 180

        self.draw_status_dot(frame, badge_x, 26, dot_color, filled=is_active, radius=4)
        cv2.putText(
            frame,
            badge_text,
            (badge_x + 12, 30),
            UITypography.FONT,
            UITypography.SCALE_CAPTION,
            dot_color,
            1,
            UITypography.LINE_TYPE,
        )

    def draw_footer(
        self,
        frame: np.ndarray,
        fps: float,
        show_landmarks: bool,
        debug_mode: bool,
        show_hud: bool,
    ) -> None:
        """Draws the bottom status bar with FPS and interactive controls."""
        h, w = frame.shape[:2]
        f_h = UILayout.FOOTER_HEIGHT
        y_top = h - f_h

        # Background panel
        self.draw_panel(frame, 0, y_top, w, f_h, bg_color=UIColors.PANEL_BG, border_color=None, opacity=0.88)
        cv2.line(frame, (0, y_top), (w, y_top), UIColors.PANEL_BORDER, 1, cv2.LINE_AA)

        # FPS indicator
        fps_color = UIColors.SUCCESS if fps >= 20.0 else UIColors.WARNING
        cv2.putText(
            frame,
            f"FPS {fps:4.1f}",
            (18, y_top + 21),
            UITypography.FONT,
            UITypography.SCALE_METRIC,
            fps_color,
            1,
            UITypography.LINE_TYPE,
        )

        # Interactive controls (Right-aligned)
        mesh_state = "ON" if show_landmarks else "OFF"
        debug_state = "ON" if debug_mode else "OFF"
        controls_text = f"[Q] Exit   [L] Mesh: {mesh_state}   [D] Debug: {debug_state}   [H] HUD"
        
        cv2.putText(
            frame,
            controls_text,
            (w - 410, y_top + 21),
            UITypography.FONT,
            UITypography.SCALE_CAPTION,
            UIColors.TEXT_SECONDARY,
            1,
            UITypography.LINE_TYPE,
        )

    # ------------------------------------------------------------------
    # Data Cards (Lower Left and Lower Right)
    # ------------------------------------------------------------------

    def draw_face_eye_card(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        face_result: FaceDetectionResult,
        eye_result: EyeTrackingResult,
        debug_mode: bool = False,
    ) -> None:
        """Draws the Face Detection and Eye Tracking status card."""
        self.draw_panel(frame, x, y, w, h, bg_color=UIColors.CARD_BG, border_color=UIColors.CARD_BORDER)

        # Card header
        cv2.putText(
            frame,
            "FACE & EYE TRACKING",
            (x + 14, y + 22),
            UITypography.FONT,
            UITypography.SCALE_HEADER,
            UIColors.TEXT_SECONDARY,
            1,
            UITypography.LINE_TYPE,
        )
        cv2.line(frame, (x + 14, y + 28), (x + w - 14, y + 28), UIColors.DIVIDER, 1, cv2.LINE_AA)

        # Face status row
        if face_result.face_detected:
            face_label = "FACE DETECTED"
            face_color = UIColors.SUCCESS
            dot_filled = True
        else:
            face_label = "NO FACE IN FRAME"
            face_color = UIColors.NEUTRAL
            dot_filled = False

        self.draw_status_dot(frame, x + 20, y + 46, face_color, filled=dot_filled, radius=4)
        cv2.putText(
            frame,
            face_label,
            (x + 32, y + 50),
            UITypography.FONT,
            UITypography.SCALE_METRIC,
            face_color,
            1,
            UITypography.LINE_TYPE,
        )

        # Eye status row
        state = eye_result.eye_state
        if state is EyeState.OPEN:
            eye_label = "EYES OPEN"
            eye_color = UIColors.SUCCESS
        elif state is EyeState.CLOSED:
            eye_label = "EYES CLOSED"
            eye_color = UIColors.DANGER
        else:
            eye_label = "EYES UNKNOWN"
            eye_color = UIColors.NEUTRAL

        self.draw_status_dot(frame, x + 20, y + 74, eye_color, filled=(state is not EyeState.UNKNOWN), radius=4)
        cv2.putText(
            frame,
            eye_label,
            (x + 32, y + 78),
            UITypography.FONT,
            UITypography.SCALE_BADGE,
            eye_color,
            UITypography.THICK_BADGE,
            UITypography.LINE_TYPE,
        )

        # EAR Metric row
        if eye_result.average_ear is not None:
            ear_str = f"Avg EAR: {eye_result.average_ear:.2f}"
            ear_color = UIColors.TEXT_PRIMARY
        else:
            ear_str = "Avg EAR: --"
            ear_color = UIColors.TEXT_DIM

        cv2.putText(
            frame,
            ear_str,
            (x + 32, y + 104),
            UITypography.FONT,
            UITypography.SCALE_METRIC,
            ear_color,
            1,
            UITypography.LINE_TYPE,
        )

        # Debug mode: extra precision
        if debug_mode and eye_result.left_ear is not None and eye_result.right_ear is not None:
            debug_text = f"L: {eye_result.left_ear:.2f}  R: {eye_result.right_ear:.2f}"
            cv2.putText(
                frame,
                debug_text,
                (x + 32, y + 122),
                UITypography.FONT,
                UITypography.SCALE_CAPTION,
                UIColors.TEXT_DIM,
                1,
                UITypography.LINE_TYPE,
            )

    def draw_head_pose_card(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        pose_result: HeadPoseResult,
        debug_mode: bool = False,
    ) -> None:
        """Draws the Head Pose Estimation and Orientation status card."""
        self.draw_panel(frame, x, y, w, h, bg_color=UIColors.CARD_BG, border_color=UIColors.CARD_BORDER)

        # Card header
        cv2.putText(
            frame,
            "HEAD ORIENTATION",
            (x + 14, y + 22),
            UITypography.FONT,
            UITypography.SCALE_HEADER,
            UIColors.TEXT_SECONDARY,
            1,
            UITypography.LINE_TYPE,
        )
        cv2.line(frame, (x + 14, y + 28), (x + w - 14, y + 28), UIColors.DIVIDER, 1, cv2.LINE_AA)

        # Direction status badge
        direction = pose_result.direction
        if direction is HeadDirection.CENTER:
            dir_text = "HEAD CENTERED"
            dir_color = UIColors.SUCCESS
        elif direction in (HeadDirection.LEFT, HeadDirection.RIGHT, HeadDirection.UP, HeadDirection.DOWN):
            dir_text = f"LOOKING {direction.value}"
            dir_color = UIColors.WARNING
        else:
            dir_text = "HEAD UNKNOWN"
            dir_color = UIColors.NEUTRAL

        self.draw_status_dot(frame, x + 20, y + 50, dir_color, filled=(direction is not HeadDirection.UNKNOWN), radius=4)
        cv2.putText(
            frame,
            dir_text,
            (x + 32, y + 54),
            UITypography.FONT,
            UITypography.SCALE_BADGE,
            dir_color,
            UITypography.THICK_BADGE,
            UITypography.LINE_TYPE,
        )

        # Angles row (Yaw & Pitch)
        if pose_result.yaw is not None and pose_result.pitch is not None:
            angles_str = f"Yaw: {pose_result.yaw:+4.1f}   Pitch: {pose_result.pitch:+4.1f}"
            angles_color = UIColors.TEXT_PRIMARY
        else:
            angles_str = "Yaw: --      Pitch: --"
            angles_color = UIColors.TEXT_DIM

        cv2.putText(
            frame,
            angles_str,
            (x + 32, y + 80),
            UITypography.FONT,
            UITypography.SCALE_METRIC,
            angles_color,
            1,
            UITypography.LINE_TYPE,
        )

        # Roll / Guideline row
        if pose_result.roll is not None:
            roll_str = f"Roll: {pose_result.roll:+4.1f} deg"
            roll_color = UIColors.TEXT_SECONDARY
        else:
            roll_str = "Roll: --"
            roll_color = UIColors.TEXT_DIM

        cv2.putText(
            frame,
            roll_str,
            (x + 32, y + 104),
            UITypography.FONT,
            UITypography.SCALE_METRIC,
            roll_color,
            1,
            UITypography.LINE_TYPE,
        )

        if debug_mode and pose_result.yaw is not None:
            cv2.putText(
                frame,
                f"State: {direction.name}",
                (x + 32, y + 122),
                UITypography.FONT,
                UITypography.SCALE_CAPTION,
                UIColors.TEXT_DIM,
                1,
                UITypography.LINE_TYPE,
            )

    def draw_phone_card(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        phone_result: PhoneDetectionResult,
        debug_mode: bool = False,
    ) -> None:
        """Draws the Phone Detection status card."""
        self.draw_panel(frame, x, y, w, h, bg_color=UIColors.CARD_BG, border_color=UIColors.CARD_BORDER)

        # Card header
        cv2.putText(
            frame,
            "PHONE DETECTION",
            (x + 14, y + 20),
            UITypography.FONT,
            UITypography.SCALE_HEADER,
            UIColors.TEXT_SECONDARY,
            1,
            UITypography.LINE_TYPE,
        )
        cv2.line(frame, (x + 14, y + 26), (x + w - 14, y + 26), UIColors.DIVIDER, 1, cv2.LINE_AA)

        # Status badge row
        if phone_result.phone_detected:
            dot_color = UIColors.WARNING  # Warm amber (non-alarming detection)
            badge_label = "PHONE DETECTED"
            dot_filled = True
        else:
            dot_color = UIColors.NEUTRAL  # Slate gray
            badge_label = "NOT DETECTED"
            dot_filled = False

        self.draw_status_dot(frame, x + 20, y + 46, dot_color, filled=dot_filled, radius=4)
        cv2.putText(
            frame,
            badge_label,
            (x + 32, y + 50),
            UITypography.FONT,
            UITypography.SCALE_BADGE,
            dot_color,
            UITypography.THICK_BADGE,
            UITypography.LINE_TYPE,
        )

        # Detail row: confidence or waiting text
        if phone_result.phone_detected and phone_result.confidence is not None:
            pct = int(round(phone_result.confidence * 100))
            count_suffix = f" ({len(phone_result.detections)} in view)" if len(phone_result.detections) > 1 else ""
            conf_str = f"Confidence: {pct}%{count_suffix}"
            cv2.putText(
                frame,
                conf_str,
                (x + 32, y + 70),
                UITypography.FONT,
                UITypography.SCALE_METRIC,
                UIColors.TEXT_PRIMARY,
                1,
                UITypography.LINE_TYPE,
            )
        else:
            cv2.putText(
                frame,
                "No phone in view",
                (x + 32, y + 70),
                UITypography.FONT,
                UITypography.SCALE_CAPTION,
                UIColors.TEXT_DIM,
                1,
                UITypography.LINE_TYPE,
            )

        # Debug row
        if debug_mode:
            cache_flag = "Yes" if phone_result.is_cached else "No"
            debug_str = f"Infer: {phone_result.inference_time_ms:.1f}ms | Cache: {cache_flag}"
            cv2.putText(
                frame,
                debug_str,
                (x + 14, y + 90),
                UITypography.FONT,
                UITypography.SCALE_CAPTION,
                UIColors.TEXT_DIM,
                1,
                UITypography.LINE_TYPE,
            )

    def draw_behavior_card(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        behavior_result: BehaviorResult,
        debug_mode: bool = False,
    ) -> None:
        """Draws the Behavior Engine status card showing current behavioral state."""
        self.draw_panel(frame, x, y, w, h, bg_color=UIColors.CARD_BG, border_color=UIColors.CARD_BORDER)

        # Card header
        cv2.putText(
            frame,
            "BEHAVIOR STATUS",
            (x + 14, y + 20),
            UITypography.FONT,
            UITypography.SCALE_HEADER,
            UIColors.TEXT_SECONDARY,
            1,
            UITypography.LINE_TYPE,
        )
        cv2.line(frame, (x + 14, y + 26), (x + w - 14, y + 26), UIColors.DIVIDER, 1, cv2.LINE_AA)

        # Map state to display label and color
        state = behavior_result.state
        state_display = {
            BehaviorState.FOCUSED: ("FOCUSED", UIColors.SUCCESS),
            BehaviorState.DISTRACTED: ("DISTRACTED", UIColors.WARNING),
            BehaviorState.DROWSY: ("DROWSY", UIColors.DANGER),
            BehaviorState.PHONE_DETECTED: ("PHONE DETECTED", UIColors.WARNING),
            BehaviorState.AWAY_FROM_DESK: ("AWAY FROM DESK", UIColors.NEUTRAL),
            BehaviorState.NO_FACE: ("NO FACE", UIColors.NEUTRAL),
            BehaviorState.UNKNOWN: ("INITIALIZING", UIColors.NEUTRAL),
        }
        label, color = state_display.get(state, ("UNKNOWN", UIColors.NEUTRAL))

        # Status dot and label
        dot_filled = state not in (BehaviorState.UNKNOWN, BehaviorState.NO_FACE)
        self.draw_status_dot(frame, x + 20, y + 46, color, filled=dot_filled, radius=4)
        cv2.putText(
            frame,
            label,
            (x + 32, y + 50),
            UITypography.FONT,
            UITypography.SCALE_BADGE,
            color,
            UITypography.THICK_BADGE,
            UITypography.LINE_TYPE,
        )

        # Duration row
        duration = behavior_result.state_duration
        if duration < 60:
            dur_str = f"Duration: {duration:.0f}s"
        else:
            dur_str = f"Duration: {int(duration // 60)}m {int(duration % 60)}s"

        cv2.putText(
            frame,
            dur_str,
            (x + 32, y + 70),
            UITypography.FONT,
            UITypography.SCALE_METRIC,
            UIColors.TEXT_PRIMARY,
            1,
            UITypography.LINE_TYPE,
        )

        # Debug row: reason code
        if debug_mode:
            reason_val = behavior_result.reason.value if hasattr(behavior_result.reason, "value") else str(behavior_result.reason)
            reason_str = f"Reason: {reason_val}"
            cv2.putText(
                frame,
                reason_str,
                (x + 14, y + 90),
                UITypography.FONT,
                UITypography.SCALE_CAPTION,
                UIColors.TEXT_DIM,
                1,
                UITypography.LINE_TYPE,
            )

    # ------------------------------------------------------------------
    # Master Render Entry Point
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        fps: float,
        face_result: FaceDetectionResult,
        eye_result: EyeTrackingResult,
        pose_result: HeadPoseResult,
        phone_result: Optional[PhoneDetectionResult] = None,
        behavior_result: Optional[BehaviorResult] = None,
        visual_alert: Optional[object] = None,
        show_hud: bool = True,
        show_landmarks: bool = True,
        show_phone_card: bool = True,
        debug_mode: bool = False,
    ) -> None:
        """Renders the complete modern UI overlay onto the frame."""
        if not show_hud:
            return

        h, w = frame.shape[:2]

        # 1. Top Header
        self.draw_header(frame, fps, is_active=True)

        # 2. Responsive Card Dimensions (Docked at bottom corners)
        card_w = min(UILayout.DEFAULT_CARD_WIDTH, int(w * 0.36))
        card_h = UILayout.DEFAULT_CARD_HEIGHT if not debug_mode else UILayout.DEFAULT_CARD_HEIGHT + 14
        margin_x = UILayout.CARD_MARGIN_X
        card_y = h - UILayout.FOOTER_HEIGHT - card_h - UILayout.CARD_MARGIN_BOTTOM

        # 3. Left Card: Face & Eyes
        self.draw_face_eye_card(
            frame,
            x=margin_x,
            y=card_y,
            w=card_w,
            h=card_h,
            face_result=face_result,
            eye_result=eye_result,
            debug_mode=debug_mode,
        )

        # 4. Right Card: Head Orientation
        self.draw_head_pose_card(
            frame,
            x=w - card_w - margin_x,
            y=card_y,
            w=card_w,
            h=card_h,
            pose_result=pose_result,
            debug_mode=debug_mode,
        )

        # 5. Top-Right Card: Phone Detection (if available and enabled)
        if show_phone_card and phone_result is not None:
            phone_w = min(240, int(w * 0.30))
            phone_h = 82 if not debug_mode else 100
            self.draw_phone_card(
                frame,
                x=w - phone_w - margin_x,
                y=UILayout.HEADER_HEIGHT + 10,
                w=phone_w,
                h=phone_h,
                phone_result=phone_result,
                debug_mode=debug_mode,
            )

        # 6. Top-Left Card: Behavior Status (if available)
        if behavior_result is not None:
            behavior_w = min(240, int(w * 0.30))
            behavior_h = 82 if not debug_mode else 100
            self.draw_behavior_card(
                frame,
                x=margin_x,
                y=UILayout.HEADER_HEIGHT + 10,
                w=behavior_w,
                h=behavior_h,
                behavior_result=behavior_result,
                debug_mode=debug_mode,
            )

        # 7. Visual Alert Banner (Phase 7 - non-blocking overlay)
        if visual_alert is not None and hasattr(visual_alert, "draw"):
            visual_alert.draw(frame)

        # 8. Bottom Footer
        self.draw_footer(
            frame,
            fps,
            show_landmarks=show_landmarks,
            debug_mode=debug_mode,
            show_hud=show_hud,
        )
