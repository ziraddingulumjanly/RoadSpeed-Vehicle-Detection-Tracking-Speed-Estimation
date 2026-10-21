"""Homography and camera-coordinate validation."""

from collections import defaultdict, deque
from math import dist, isfinite

import cv2
import numpy as np


def make_homography(image_points, world_points_meters, frame_size: tuple[int, int]):
    """Transform image pixels to road-plane meters using four matched road points."""
    if image_points is None or world_points_meters is None:
        raise ValueError(
            "Calibration required: set speed.image_points and speed.world_points_meters "
            "in config.yaml. Run 'python main.py inspect' to view a pixel grid."
        )
    try:
        src = np.asarray(image_points, dtype=np.float32)
        dst = np.asarray(world_points_meters, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise ValueError("Calibration coordinates must be numeric") from exc
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("Exactly four (x, y) pairs are required in each calibration list")
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise ValueError("All calibration coordinates must be finite")
    width, height = frame_size
    if np.any(src[:, 0] < 0) or np.any(src[:, 0] >= width) or np.any(src[:, 1] < 0) or np.any(src[:, 1] >= height):
        raise ValueError(f"Image points must be inside the original {width}x{height} video frame")
    if abs(cv2.contourArea(src)) < 1 or abs(cv2.contourArea(dst)) < 1e-6:
        raise ValueError("Calibration points must span nonzero road and image areas")
    if not (cv2.isContourConvex(src) and cv2.isContourConvex(dst)):
        raise ValueError("Both lists must follow matching corners of convex road quadrilaterals")
    # Opposite winding directions indicate a mirrored (usually mistaken) correspondence.
    if cv2.contourArea(src, oriented=True) * cv2.contourArea(dst, oriented=True) < 0:
        raise ValueError("Image and world points must use the same corner order/direction")
    H = cv2.getPerspectiveTransform(src, dst)
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("Degenerate calibration; choose four distinct road-plane points")
    return H, src


def _project(pixel: tuple[float, float], homography: np.ndarray) -> tuple[float, float] | None:
    mapped = cv2.perspectiveTransform(np.asarray([[pixel]], dtype=np.float32), homography)[0, 0]
    if not np.isfinite(mapped).all():
        return None
    return float(mapped[0]), float(mapped[1])


class SpeedEstimator:
    """Speed from bottom-center track displacement in calibrated road coordinates."""

    def __init__(self, homography: np.ndarray, image_polygon: np.ndarray,
                 window_seconds: float, min_elapsed_seconds: float, max_gap_seconds: float):
        self.H = homography
        self.polygon = image_polygon
        self.window = window_seconds
        self.min_elapsed = min_elapsed_seconds
        self.max_gap = max_gap_seconds
        self.history = defaultdict(deque)

    def update(self, track_id: int, contact_px: tuple[float, float], time_s: float) -> float | None:
        # Extrapolating far outside the four-point region can produce meaningless km/h.
        if cv2.pointPolygonTest(self.polygon, contact_px, False) < 0:
            self.history.pop(track_id, None)
            return None
        point_m = _project(contact_px, self.H)
        if point_m is None:
            self.history.pop(track_id, None)
            return None
        records = self.history[track_id]
        if records and (time_s <= records[-1][0] or time_s - records[-1][0] > self.max_gap):
            records.clear()
        records.append((time_s, point_m))
        while len(records) > 1 and time_s - records[0][0] > self.window:
            records.popleft()
        elapsed = records[-1][0] - records[0][0]
        if elapsed < self.min_elapsed:
            return None
        kmh = 3.6 * dist(records[0][1], records[-1][1]) / elapsed
        return kmh if isfinite(kmh) else None

    def discard(self, track_id: int) -> None:
        self.history.pop(track_id, None)
