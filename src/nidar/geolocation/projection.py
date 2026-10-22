"""Ground-projection of detections -> lat/lon estimates with uncertainty.

IMPORTANT: drone GPS != survivor position. Without camera calibration +
mounting angle + attitude, outputs are labelled `drone_location` (the drone's
own position). Only with full inputs do we emit `projected_target`.
Flat-ground nadir-ish assumption; uncertainty grows with off-nadir angle and
missing calibration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ProjectionResult:
    lat: float
    lon: float
    kind: str  # 'drone_location' | 'projected_target'
    uncertainty_m: float


def project_to_ground(px: float, py: float, img_w: int, img_h: int, drone_lat: float,
                      drone_lon: float, alt_m: float, yaw_deg: float = 0.0,
                      hfov_deg: float | None = None,
                      mount_pitch_deg: float = 90.0) -> ProjectionResult:
    """Project pixel (px,py) to ground. Falls back to drone_location if uncalibrated."""
    if hfov_deg is None or alt_m <= 0:
        return ProjectionResult(drone_lat, drone_lon, "drone_location", alt_m * 0.5)
    # ground footprint width from HFOV (nadir approx)
    footprint_w = 2 * alt_m * math.tan(math.radians(hfov_deg / 2))
    footprint_h = footprint_w * img_h / max(img_w, 1)
    dx_m = (px - img_w / 2) / img_w * footprint_w
    dy_m = (py - img_h / 2) / img_h * footprint_h
    yaw = math.radians(yaw_deg)
    rx = dx_m * math.cos(yaw) - dy_m * math.sin(yaw)
    ry = dx_m * math.sin(yaw) + dy_m * math.cos(yaw)
    lat = drone_lat + ry / 111320.0
    lon = drone_lon + rx / (111320.0 * max(math.cos(math.radians(drone_lat)), 0.2))
    off_nadir = abs(mount_pitch_deg - 90.0) / 90.0
    unc = 2.0 + 0.1 * alt_m + off_nadir * alt_m * 0.5 + math.hypot(dx_m, dy_m) * 0.1
    return ProjectionResult(lat, lon, "projected_target", unc)
