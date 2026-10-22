"""Minimal ground-station API (FastAPI) — detection events + status.

Serve: uvicorn nidar.groundstation.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from pathlib import Path


def create_app(evidence_dir: str | Path = "outputs/evidence"):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="NIDAR ground station")
    app.state = {"last": {}}  # type: ignore[attr-defined]

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/detections")
    def detections(limit: int = 50):
        p = Path(evidence_dir) / "detections.jsonl"
        rows = []
        if p.exists():
            import json

            lines = p.read_text().splitlines()[-limit:]
            rows = [json.loads(ln) for ln in lines if ln.strip()]
        return JSONResponse({"count": len(rows), "events": rows})

    @app.get("/status")
    def status():
        return JSONResponse({"live": app.state.get("last", {}),
                             "note": "Core CV pipeline runs independently of this UI."})

    return app


app = create_app()
