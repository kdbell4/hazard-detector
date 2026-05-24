"""
FastAPI application — serves the dashboard and all API endpoints.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp.engine import DetectionEngine
from webapp.state import SharedState

# ─────────────────────────────────────────────
# Single shared state instance
# ─────────────────────────────────────────────
state = SharedState()

# ─────────────────────────────────────────────
# Lifespan: start / stop the engine thread
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = DetectionEngine(state)
    engine.daemon = True
    engine.start()
    try:
        yield
    finally:
        engine.stop()
        engine.join(timeout=3.0)
        print("Engine joined.")

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(title="Hazard Detector", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.get("/video_feed")
async def video_feed():
    """MJPEG streaming endpoint — embed as <img src='/video_feed'>."""
    async def generate():
        try:
            while True:
                frame = state.get_frame()
                if frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                await asyncio.sleep(0.033)   # ~30 FPS cap
        except (GeneratorExit, asyncio.CancelledError):
            return

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
async def api_status():
    return JSONResponse(state.get_status_snapshot())


@app.get("/api/log")
async def api_log():
    return JSONResponse({"entries": state.get_log_snapshot()})


@app.get("/api/stats")
async def api_stats():
    snap = state.get_status_snapshot()
    return JSONResponse({
        "fps":              snap["fps"],
        "uptime_seconds":   snap["uptime_seconds"],
        "total_detections": snap["total_detections"],
    })


# ─────────────────────────────────────────────
# Toggle endpoints
# ─────────────────────────────────────────────

class ToggleModelRequest(BaseModel):
    model: str
    enabled: bool


class ToggleAudioRequest(BaseModel):
    enabled: bool


@app.post("/api/toggle_model")
async def toggle_model(req: ToggleModelRequest):
    ok = state.toggle_model(req.model, req.enabled)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Unknown model: '{req.model}'")
    return JSONResponse({"model": req.model, "enabled": req.enabled})


@app.post("/api/toggle_audio")
async def toggle_audio(req: ToggleAudioRequest):
    state.toggle_audio(req.enabled)
    return JSONResponse({"audio_enabled": req.enabled})
