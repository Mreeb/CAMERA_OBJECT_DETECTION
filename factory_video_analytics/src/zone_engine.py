"""Polygon zone engine, point-in-polygon attribution, exclusion filtering, and line crossings."""

from dataclasses import dataclass, field
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ZoneDefinition:
    """Represents a named functional zone polygon."""
    id: str
    name: str
    zone_type: str  # "work", "storage", "loading", "exclusion", "pickup", "destination"
    polygon: List[List[float]]  # Normalized [[x, y], ...] in 0.0 - 1.0 range
    color: Tuple[int, int, int] = (200, 200, 200)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "zone_type": self.zone_type,
            "polygon": self.polygon,
            "color": list(self.color),
        }


@dataclass
class CountingLineDefinition:
    """Represents a directional counting line segment."""
    enabled: bool = False
    name: str = "Counting Line"
    point1: Tuple[float, float] = (0.2, 0.5)  # Normalized (x, y)
    point2: Tuple[float, float] = (0.8, 0.5)
    target_classes: List[str] = field(default_factory=lambda: ["cardboard box"])
    count_people: bool = False
    debounce_seconds: float = 2.0


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def point_side_of_line(
    p: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    return (line_end[0] - line_start[0]) * (p[1] - line_start[1]) - (line_end[1] - line_start[1]) * (p[0] - line_start[0])


class ZoneEngine:
    """Evaluates spatial zones, exclusion masks, and directional counting line crossings."""

    def __init__(
        self,
        zones_config: Optional[List[Dict[str, Any]]] = None,
        counting_line_config: Optional[Dict[str, Any]] = None,
    ):
        self.zones: List[ZoneDefinition] = []
        self.exclusion_zones: List[ZoneDefinition] = []
        self.load_zones(zones_config or [])

        # Counting line
        cl_cfg = counting_line_config or {}
        self.counting_line = CountingLineDefinition(
            enabled=cl_cfg.get("enabled", False),
            name=cl_cfg.get("name", "Counting Line"),
            point1=tuple(cl_cfg.get("point1", [0.2, 0.5])),
            point2=tuple(cl_cfg.get("point2", [0.8, 0.5])),
            target_classes=cl_cfg.get("target_classes", ["cardboard box"]),
            count_people=cl_cfg.get("count_people", False),
            debounce_seconds=float(cl_cfg.get("debounce_seconds", 2.0)),
        )

        # Counting state
        self.line_crossings: Dict[str, int] = {"A_to_B": 0, "B_to_A": 0}
        self.last_track_crossing_time: Dict[int, float] = {}

    def load_zones(self, zones_data: List[Dict[str, Any]]) -> None:
        """Parses and loads zone definitions."""
        self.zones.clear()
        self.exclusion_zones.clear()

        for z in zones_data:
            color = tuple(z.get("color", [200, 200, 200]))
            zone = ZoneDefinition(
                id=z.get("id", "zone"),
                name=z.get("name", "Zone"),
                zone_type=z.get("type", "work"),
                polygon=z.get("polygon", []),
                color=color,
            )
            if zone.zone_type == "exclusion":
                self.exclusion_zones.append(zone)
            else:
                self.zones.append(zone)

        logger.info(f"Loaded {len(self.zones)} functional zones and {len(self.exclusion_zones)} exclusion zones.")

    def get_pixel_polygon(self, zone: ZoneDefinition, width: int, height: int) -> List[Tuple[float, float]]:
        """Converts normalized polygon to pixel coordinates."""
        return [(p[0] * width, p[1] * height) for p in zone.polygon]

    def is_in_exclusion_zone(self, point: Tuple[float, float], width: int, height: int) -> bool:
        """Checks if a point falls inside any exclusion polygon."""
        for ex_zone in self.exclusion_zones:
            poly_px = self.get_pixel_polygon(ex_zone, width, height)
            if len(poly_px) >= 3 and point_in_polygon(point, poly_px):
                return True
        return False

    def get_zone_for_point(self, point: Tuple[float, float], width: int, height: int) -> Optional[ZoneDefinition]:
        """Finds the matching functional zone for a point."""
        for zone in self.zones:
            poly_px = self.get_pixel_polygon(zone, width, height)
            if len(poly_px) >= 3 and point_in_polygon(point, poly_px):
                return zone
        return None

    def check_line_crossing(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float],
        class_name: str,
        track_id: int,
        timestamp: float,
        width: int,
        height: int,
    ) -> Optional[str]:
        """Evaluates if a track center crossed the counting line in direction A_to_B or B_to_A."""
        if not self.counting_line.enabled:
            return None

        is_person = class_name.lower() in ("person", "worker")
        if is_person and not self.counting_line.count_people:
            return None

        if not is_person and self.counting_line.target_classes:
            if class_name.lower() not in [c.lower() for c in self.counting_line.target_classes]:
                return None

        # Check debounce
        last_time = self.last_track_crossing_time.get(track_id, -999.0)
        if (timestamp - last_time) < self.counting_line.debounce_seconds:
            return None

        p1_px = (self.counting_line.point1[0] * width, self.counting_line.point1[1] * height)
        p2_px = (self.counting_line.point2[0] * width, self.counting_line.point2[1] * height)

        if segments_intersect(prev_pos, curr_pos, p1_px, p2_px):
            side_prev = point_side_of_line(prev_pos, p1_px, p2_px)
            side_curr = point_side_of_line(curr_pos, p1_px, p2_px)

            direction = "A_to_B" if side_prev > 0 and side_curr <= 0 else "B_to_A"
            self.line_crossings[direction] += 1
            self.last_track_crossing_time[track_id] = timestamp
            return direction

        return None
