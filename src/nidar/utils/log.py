"""Centralised logging setup."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure root logger once and return the nidar logger."""
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("nidar")
