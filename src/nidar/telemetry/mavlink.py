"""Optional MAVLink telemetry (requires pymavlink + flight controller).

Vision inference NEVER depends on this: callers catch all exceptions and
continue with telemetry=None.
"""
from __future__ import annotations

import time

from nidar.telemetry.mock import TelemetrySample


class MAVLinkTelemetry:
    def __init__(self, endpoint: str = "udp:127.0.0.1:14550") -> None:
        try:
            from pymavlink import mavutil
        except ImportError as e:
            raise RuntimeError("pymavlink not installed (pip install pymavlink)") from e
        self._conn = mavutil.mavlink_connection(endpoint)
        self._conn.wait_heartbeat(timeout=10)

    def read(self) -> TelemetrySample | None:
        try:
            msg = self._conn.recv_match(type="GLOBAL_POSITION_INT", blocking=False, timeout=0.1)
            if msg is None:
                return None
            return TelemetrySample(
                timestamp=time.time(), lat=msg.lat / 1e7, lon=msg.lon / 1e7,
                alt_m=msg.relative_alt / 1000.0, yaw=getattr(msg, "hdg", 0) / 100.0,
                source="mavlink",
            )
        except Exception:
            return None

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


def open_telemetry(cfg: dict):
    source = str(cfg.get("source", "mock")).lower()
    if source == "mavlink":
        return MAVLinkTelemetry(cfg.get("mavlink_endpoint", "udp:127.0.0.1:14550"))
    from nidar.telemetry.mock import MockTelemetry

    return MockTelemetry()
