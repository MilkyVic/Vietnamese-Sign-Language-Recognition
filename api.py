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
from openai import OpenAI

from load_env import load_dotenv
from text_to_speech import synthesize_speech
from vsl_recognition import SignLanguageRecognizer

# Load environment variables whether the server is started from repo root or the
# inner project directory. The second call is a no-op if the first succeeds.
_here = Path(__file__).resolve().parent
load_dotenv(_here / ".env")
load_dotenv(_here.parent / ".env")

recognizer = SignLanguageRecognizer()
tts_output_dir = Path("Outputs/app_predictions")
tts_output_dir.mkdir(parents=True, exist_ok=True)
_kinesis_dist = Path("Kinesis 3/dist")

app = FastAPI(title="Vietnamese Sign Language API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory=tts_output_dir), name="audio")
client = OpenAI()


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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

    return {
        "label": result.label,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "latency_ms": latency * 1000,
        "audio_url": audio_url,
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Speech-to-text using OpenAI gpt-4o-transcribe."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix or ".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

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


# Serve frontend from Kinesis 3 build (includes static like /logo.png)
if not _kinesis_dist.exists():
    raise RuntimeError("Kinesis 3 build not found. Please run npm run build in Kinesis 3.")

app.mount("/", StaticFiles(directory=_kinesis_dist, html=True), name="kinesis")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=True)
