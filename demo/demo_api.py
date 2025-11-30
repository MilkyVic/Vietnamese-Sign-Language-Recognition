"""Demo FastAPI server that always returns a scripted prediction.

Frontend stays the same; point it to this API (e.g., http://127.0.0.1:8002)
and every /predict/video call will produce a fixed scripted response plus TTS.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from load_env import load_dotenv
from text_to_speech import synthesize_speech
from openai import OpenAI

# Load .env flexibly (project dir and its parent where the real .env lives)
_here = Path(__file__).resolve().parent
load_dotenv(_here.parent / ".env")
load_dotenv(_here.parent.parent / ".env")

SCRIPT_TEXT = (
    "500 ngàn 1 đêm không anh"
)

tts_output_dir = Path("Outputs/demo_predictions")
tts_output_dir.mkdir(parents=True, exist_ok=True)
# Point to the new main UI build folder (Kinesis 3)
_kinesis_dist = _here.parent / "Kinesis 3" / "dist"

app = FastAPI(title="Vietnamese Sign Language API - Demo (scripted)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "mode": "scripted-demo"}


def build_response_text() -> str:
    """Return the full scripted text."""
    return SCRIPT_TEXT


@app.post("/predict/video")
async def predict_from_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    # We ignore the actual video content; just consume the upload to satisfy client.
    await file.read()

    label = build_response_text()
    probabilities = [1.0]
    confidence = 1.0

    audio_file = tts_output_dir / f"demo_prediction_{int(time.time())}.mp3"
    audio_url = None
    try:
        # Use a Vietnamese-friendly voice to keep demo audio consistent.
        synthesize_speech(label, audio_file, voice="coral")
        audio_url = f"/audio/{audio_file.name}"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "latency_ms": 0,
        "audio_url": audio_url,
    }


# Serve frontend build (unchanged)
app.mount("/audio", StaticFiles(directory=tts_output_dir), name="audio")
app.mount("/assets", StaticFiles(directory=_kinesis_dist / "assets"), name="assets")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Speech-to-text demo using gpt-4o-transcribe."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    # Persist temp file for OpenAI API
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix or ".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    client = OpenAI()
    try:
        with open(tmp_path, "rb") as audio_f:
            result = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_f,
                language="vi",
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"text": result.text, "language": "vi"}


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve the frontend for all routes that are not API routes."""
    # pylint: disable=unused-argument
    index_file = _kinesis_dist / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=500, detail="Frontend build (kinesis/dist/index.html) not found")
    return FileResponse(index_file)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("demo.demo_api:app", host="0.0.0.0", port=8002, reload=True)
