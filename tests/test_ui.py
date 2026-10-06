"""Unit tests for ui/theme.py and ui/overlay.py.

Verifies HUD rendering, layout calculations, color themes, card generation,
and error handling across synthetic frames without requiring a physical display.
"""

import numpy as np
import pytest

from ui.theme import UIColors, UITypography, UILayout
from ui.overlay import OverlayRenderer
from vision.face_detector import FaceDetectionResult
from vision.eye_tracker import EyeTrackingResult, EyeState
from vision.head_pose import HeadPoseResult, HeadDirection


class TestUITheme:
    def test_colors_format(self):
        """All theme colors must be valid 3-tuples of ints in [0, 255]."""
        colors = [
            UIColors.PANEL_BG,
            UIColors.PANEL_BORDER,
            UIColors.CARD_BG,
            UIColors.CARD_BORDER,
            UIColors.TEXT_PRIMARY,
            UIColors.TEXT_SECONDARY,
            UIColors.TEXT_DIM,
            UIColors.SUCCESS,
            UIColors.WARNING,
            UIColors.DANGER,
            UIColors.ACCENT,
            UIColors.NEUTRAL,
        ]
        for c in colors:
            assert isinstance(c, tuple)
            assert len(c) == 3
            assert all(0 <= val <= 255 for val in c)

    def test_typography_constants(self):
        assert UITypography.SCALE_TITLE > 0
        assert UITypography.SCALE_HEADER > 0
        assert UITypography.THICK_TITLE >= 1

    def test_layout_constants(self):
        assert UILayout.HEADER_HEIGHT > 0
        assert UILayout.FOOTER_HEIGHT > 0
        assert UILayout.DEFAULT_CARD_WIDTH > 0
        assert UILayout.DEFAULT_CARD_HEIGHT > 0
        assert 0.0 <= UILayout.DEFAULT_OPACITY <= 1.0


class TestOverlayRenderer:
    def test_init(self):
        renderer = OverlayRenderer(opacity=0.75)
        assert renderer.opacity == 0.75

    def test_draw_panel_mutates_roi(self):
        renderer = OverlayRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        renderer.draw_panel(frame, 10, 10, 100, 100)
        # The region inside (10..110, 10..110) should now have non-zero pixels
        roi = frame[10:110, 10:110]
        assert np.count_nonzero(roi) > 0
        # Region outside should still be zero
        outside = frame[200:300, 200:300]
        assert np.count_nonzero(outside) == 0

    def test_draw_panel_out_of_bounds_handled_safely(self):
        renderer = OverlayRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Negative x/y and larger than frame dimensions
        renderer.draw_panel(frame, -50, -50, 100, 100)
        renderer.draw_panel(frame, 600, 450, 200, 200)
        renderer.draw_panel(frame, 1000, 1000, 50, 50)
        # Should not raise any error

    def test_draw_status_dot(self):
        renderer = OverlayRenderer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        renderer.draw_status_dot(frame, 50, 50, UIColors.SUCCESS, filled=True)
        assert np.count_nonzero(frame) > 0

    def test_draw_header(self):
        renderer = OverlayRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        renderer.draw_header(frame, fps=30.0, is_active=True)
        header_area = frame[0:UILayout.HEADER_HEIGHT, :]
        assert np.count_nonzero(header_area) > 0

    def test_draw_footer(self):
        renderer = OverlayRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        renderer.draw_footer(frame, fps=28.5, show_landmarks=True, debug_mode=False, show_hud=True)
        footer_area = frame[480 - UILayout.FOOTER_HEIGHT : 480, :]
        assert np.count_nonzero(footer_area) > 0

    def test_draw_face_eye_card_open(self):
        renderer = OverlayRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face_res = FaceDetectionResult(face_detected=True)
        eye_res = EyeTrackingResult(left_ear=0.31, right_ear=0.32, average_ear=0.315, eye_state=EyeState.OPEN)
        renderer.draw_face_eye_card(frame, 20, 200, 250, 130, face_res, eye_res, debug_mode=False)
        assert np.count_nonzero(frame[200:330, 20:270]) > 0

    def test_draw_face_eye_card_no_face(self):
        renderer = OverlayRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face_res = FaceDetectionResult(face_detected=False)
        eye_res = EyeTrackingResult(eye_state=EyeState.UNKNOWN)
        renderer.draw_face_eye_card(frame, 20, 200, 250, 130, face_res, eye_res, debug_mode=True)
        assert np.count_nonzero(frame[200:330, 20:270]) > 0

    def test_draw_head_pose_card_directions(self):
        renderer = OverlayRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for direction in [HeadDirection.CENTER, HeadDirection.LEFT, HeadDirection.RIGHT, HeadDirection.UP, HeadDirection.DOWN, HeadDirection.UNKNOWN]:
            pose_res = HeadPoseResult(yaw=12.0, pitch=-5.0, roll=1.0, direction=direction)
            renderer.draw_head_pose_card(frame, 350, 200, 250, 130, pose_res, debug_mode=True)
            assert np.count_nonzero(frame[200:330, 350:600]) > 0

    def test_master_render_hud_toggle(self):
        renderer = OverlayRenderer()
        frame = np.zeros((720, 960, 3), dtype=np.uint8)
        original = frame.copy()

        face_res = FaceDetectionResult(face_detected=True)
        eye_res = EyeTrackingResult(average_ear=0.32, eye_state=EyeState.OPEN)
        pose_res = HeadPoseResult(yaw=2.0, pitch=-3.0, roll=0.0, direction=HeadDirection.CENTER)

        # When show_hud=False, frame must remain completely unmutated
        renderer.render(
            frame,
            fps=30.0,
            face_result=face_res,
            eye_result=eye_res,
            pose_result=pose_res,
            show_hud=False,
        )
        np.testing.assert_array_equal(frame, original)

        # When show_hud=True, frame must be updated
        renderer.render(
            frame,
            fps=30.0,
            face_result=face_res,
            eye_result=eye_res,
            pose_result=pose_res,
            show_hud=True,
        )
        assert not np.array_equal(frame, original)

    def test_render_multiple_resolutions(self):
        renderer = OverlayRenderer()
        face_res = FaceDetectionResult(face_detected=True)
        eye_res = EyeTrackingResult(average_ear=0.30, eye_state=EyeState.OPEN)
        pose_res = HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, direction=HeadDirection.CENTER)

        for w, h in [(640, 480), (960, 720), (1280, 720), (800, 600)]:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            renderer.render(
                frame,
                fps=29.0,
                face_result=face_res,
                eye_result=eye_res,
                pose_result=pose_res,
                show_hud=True,
                show_landmarks=True,
                debug_mode=True,
            )
            assert np.count_nonzero(frame) > 0
