"""FastAPI server that exposes the sign-language recognizer for React."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from text_to_speech import synthesize_speech
from vsl_recognition import SignLanguageRecognizer

recognizer = SignLanguageRecognizer()
tts_output_dir = Path("Outputs/app_predictions")

app = FastAPI(title="Vietnamese Sign Language API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory=tts_output_dir), name="audio")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/predict/video")
async def predict_from_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    start = time.perf_counter()
    try:
        result = recognizer.predict_from_video(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    latency = time.perf_counter() - start

    audio_file = tts_output_dir / f"prediction_{int(time.time())}.mp3"
    try:
        synthesize_speech(result.label, audio_file)
        audio_url = f"/audio/{audio_file.name}"
    except Exception:
        audio_url = None

    return {
        "label": result.label,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "latency_ms": latency * 1000,
        "audio_url": audio_url,
    }


# Serve frontend
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve the frontend for all routes that are not API routes."""
    # pylint: disable=unused-argument
    return FileResponse("frontend/dist/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
