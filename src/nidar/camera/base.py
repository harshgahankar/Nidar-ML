"""Camera abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CameraBase(ABC):
    @abstractmethod
    def open(self) -> None: ...
    @abstractmethod
    def read(self) -> np.ndarray | None: ...
    @abstractmethod
    def close(self) -> None: ...
    def __enter__(self): self.open(); return self
    def __exit__(self, *a): self.close()
