"""Mock telemetry for development without a drone (sinusoidal drift around a home point)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class TelemetrySample:
    timestamp: float
    lat: float
    lon: float
    alt_m: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    source: str = "mock"


class MockTelemetry:
    def __init__(self, home_lat: float = 51.5, home_lon: float = -0.12, alt_m: float = 20.0) -> None:
        self.home_lat, self.home_lon, self.alt_m = home_lat, home_lon, alt_m
        self._t0 = time.time()

    def read(self) -> TelemetrySample:
        t = time.time() - self._t0
        return TelemetrySample(
            timestamp=time.time(),
            lat=self.home_lat + 1e-5 * math.sin(t / 10),
            lon=self.home_lon + 1e-5 * math.cos(t / 10),
            alt_m=self.alt_m, yaw=(t * 2) % 360,
        )

    def close(self) -> None:
        pass
