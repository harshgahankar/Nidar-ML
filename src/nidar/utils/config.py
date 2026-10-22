"""YAML config loading + env checks."""
from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def check_environment() -> dict[str, Any]:
    """Return versions of key deps; never raises, marks missing modules."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for mod in ("ultralytics", "cv2", "torch", "numpy", "yaml", "PIL", "fastapi", "pymavlink"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception as e:  # noqa: BLE001
            info[mod] = f"missing ({e})"
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
    except Exception:
        info["cuda_available"] = False
    return info


def main() -> None:
    import json

    print(json.dumps(check_environment(), indent=2))


if __name__ == "__main__":
    main()
