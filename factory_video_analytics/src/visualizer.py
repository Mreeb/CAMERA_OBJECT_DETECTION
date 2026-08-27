"""Frame visualizer with zone polygons, motion badges, bottom-center trails, and HUD dashboard."""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .tracker import ConfirmedTrack
from .zone_engine import ZoneEngine

# Color Palette (BGR)
COLOR_PERSON = (50, 205, 50)       # Lime Green
COLOR_ARTICLE = (255, 140, 0)      # Dark Orange
COLOR_LINE = (0, 255, 255)         # Yellow
COLOR_ZONE_WORK = (50, 205, 50)    # Green
COLOR_ZONE_STORAGE = (255, 140, 0) # Orange
COLOR_ZONE_LOADING = (0, 215, 255) # Gold
COLOR_ZONE_EXCLUSION = (100, 100, 100) # Gray

COLOR_MOVING = (0, 255, 0)         # Bright Green
COLOR_LOW_MOTION = (0, 165, 255)   # Amber
COLOR_EXT_LOW_MOTION = (0, 69, 255)# Orange-Red
COLOR_STATIONARY = (180, 180, 180) # Gray
COLOR_UNKNOWN = (150, 150, 150)    # Dim Gray


class FrameVisualizer:
    """Renders clean bounding boxes, motion badges, and HUD stats."""

    def __init__(
        self,
        draw_zones: bool = False,
        draw_boxes: bool = True,
        draw_trails: bool = False,
        draw_hud: bool = False,
    ):
        self.draw_zones = draw_zones
        self.draw_boxes = draw_boxes
        self.draw_trails = draw_trails
        self.draw_hud = draw_hud

    def draw(
        self,
        frame: np.ndarray,
        tracks: List[ConfirmedTrack],
        zone_engine: Optional[ZoneEngine],
        camera_id: str,
        capture_fps: float,
        inference_fps: float,
        latency_ms: float,
        confirmed_transfers: int = 0,
    ) -> Tuple[np.ndarray, int]:
        """Draws clean bounding boxes and returns (annotated_frame, rendered_person_boxes_count)."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Draw Zones only if explicitly enabled
        if self.draw_zones and zone_engine:
            self._draw_zones(annotated, zone_engine, w, h)

        # 2. Draw Bounding Boxes, Anchors, and Motion Badges
        rendered_person_count = 0
        if self.draw_boxes:
            rendered_person_count = self._draw_boxes(annotated, tracks)

        # 5. Draw HUD Dashboard
        if self.draw_hud:
            self._draw_hud(
                annotated,
                tracks,
                camera_id,
                capture_fps,
                inference_fps,
                latency_ms,
                confirmed_transfers,
                rendered_person_count,
            )

        return annotated, rendered_person_count

    def _draw_zones(self, img: np.ndarray, zone_engine: ZoneEngine, w: int, h: int) -> None:
        overlay = img.copy()

        # Draw functional zones
        for zone in zone_engine.zones:
            pts = np.array(zone_engine.get_pixel_polygon(zone, w, h), dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], zone.color)
                cv2.polylines(img, [pts], True, zone.color, 2, cv2.LINE_AA)
                # Label
                centroid_x = int(np.mean(pts[:, 0]))
                centroid_y = int(np.mean(pts[:, 1]))
                cv2.putText(
                    img,
                    zone.name.upper(),
                    (centroid_x - 40, centroid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Draw exclusion zones
        for ex_zone in zone_engine.exclusion_zones:
            pts = np.array(zone_engine.get_pixel_polygon(ex_zone, w, h), dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], (80, 80, 80))
                cv2.polylines(img, [pts], True, (120, 120, 120), 1, cv2.LINE_AA)
                centroid_x = int(np.mean(pts[:, 0]))
                centroid_y = int(np.mean(pts[:, 1]))
                cv2.putText(
                    img,
                    "[EXCLUDED]",
                    (centroid_x - 35, centroid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    def _draw_counting_line(self, img: np.ndarray, zone_engine: ZoneEngine, w: int, h: int) -> None:
        cl = zone_engine.counting_line
        p1 = (int(cl.point1[0] * w), int(cl.point1[1] * h))
        p2 = (int(cl.point2[0] * w), int(cl.point2[1] * h))

        cv2.line(img, p1, p2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(img, p1, p2, COLOR_LINE, 2, cv2.LINE_AA)
        cv2.circle(img, p1, 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p2, 5, (0, 0, 255), -1, cv2.LINE_AA)

        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        counts_str = f"A->B: {zone_engine.line_crossings['A_to_B']} | B->A: {zone_engine.line_crossings['B_to_A']}"
        cv2.putText(img, f"{cl.name} ({counts_str})", (mid_x - 120, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    def _draw_trails(self, img: np.ndarray, tracks: List[ConfirmedTrack]) -> None:
        for track in tracks:
            pts = list(track.smoother.trail)
            if len(pts) < 2:
                continue

            color = COLOR_PERSON if track.is_person else COLOR_ARTICLE
            for i in range(1, len(pts)):
                p_start = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p_end = (int(pts[i][0]), int(pts[i][1]))
                alpha = i / len(pts)
                thickness = max(1, int(alpha * 2))
                cv2.line(img, p_start, p_end, color, thickness, cv2.LINE_AA)

    def _draw_boxes(self, img: np.ndarray, tracks: List[ConfirmedTrack]) -> int:
        person_count = 0

        for track in tracks:
            if not track.is_visible:
                continue

            x1, y1, x2, y2 = [int(v) for v in track.box]
            color = COLOR_PERSON if track.is_person else COLOR_ARTICLE
            if track.is_person:
                person_count += 1

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Bottom-center anchor point for people, center for articles
            anchor = (int(track.bottom_center[0]), int(track.bottom_center[1]))
            cv2.circle(img, anchor, 4, color, -1, cv2.LINE_AA)

            # Motion State Badge
            state = track.motion_state
            if state == "Extended Low Motion":
                state_badge = "[EXT LOW MOTION]"
                badge_col = COLOR_EXT_LOW_MOTION
            elif state == "Low Motion":
                state_badge = "[LOW MOTION]"
                badge_col = COLOR_LOW_MOTION
            elif state == "Moving":
                state_badge = "[MOVING]"
                badge_col = COLOR_MOVING
            elif state == "Stopped":
                state_badge = "[STOPPED]"
                badge_col = (0, 215, 255)
            elif state == "Stationary":
                state_badge = "[STATIONARY]"
                badge_col = COLOR_STATIONARY
            else:
                state_badge = f"[{state.upper()}]"
                badge_col = COLOR_UNKNOWN

            label = f"WORKER #{track.session_id} ({track.confidence:.0%}) {state_badge}"
            if track.current_zone != "Unknown":
                label += f" • {track.current_zone}"

            (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            bg_y1 = max(0, y1 - t_h - 6)
            bg_y2 = y1
            cv2.rectangle(img, (x1, bg_y1), (x1 + t_w + 8, bg_y2), (10, 10, 10), -1)
            cv2.rectangle(img, (x1, bg_y1), (x1 + t_w + 8, bg_y2), badge_col, 1)
            cv2.putText(img, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        return person_count

    def _draw_hud(
        self,
        img: np.ndarray,
        tracks: List[ConfirmedTrack],
        camera_id: str,
        capture_fps: float,
        inference_fps: float,
        latency_ms: float,
        confirmed_transfers: int,
        rendered_person_count: int,
    ) -> None:
        hud_w = 460
        hud_h = 100

        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)
        cv2.rectangle(img, (10, 10), (10 + hud_w, 10 + hud_h), (70, 70, 70), 1)

        # Title
        cv2.putText(
            img,
            f"CAMERA: {camera_id.upper()} (LIVE MONITORING)",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )

        visible_people = [t for t in tracks if t.is_person and t.is_visible]
        moving_people = len([t for t in visible_people if t.motion_state == "Moving"])
        low_motion_people = len([t for t in visible_people if t.motion_state in ("Low Motion", "Extended Low Motion")])
        visible_articles = len([t for t in tracks if not t.is_person and t.is_visible])

        line1 = f"Cap FPS: {capture_fps:.1f} | Model FPS: {inference_fps:.1f} | Latency: {latency_ms:.0f}ms"
        line2 = f"Visible People: {len(visible_people)} (Moving: {moving_people}, Low Motion: {low_motion_people})"
        line3 = f"Visible Articles: {visible_articles} | Confirmed Transfers: {confirmed_transfers}"

        cv2.putText(img, line1, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, line2, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 205, 50), 1, cv2.LINE_AA)
        cv2.putText(img, line3, (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 140, 0), 1, cv2.LINE_AA)
