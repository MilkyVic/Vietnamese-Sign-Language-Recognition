"""FastAPI server that exposes the sign-language recognizer for React."""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
_kinesis_dist = Path("Kinesis3/dist")
_video_dir = _here / "Dataset" / "Videos"
_poster_dir = _here / "Dataset" / "Posters"
_poster_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Vietnamese Sign Language API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory=tts_output_dir), name="audio")
# Expose learning videos so the frontend can stream them directly.
app.mount("/videos", StaticFiles(directory=_video_dir), name="videos")
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


def _safe_video_filename(video_name: str) -> str:
    """Prevent path traversal by normalizing to basename."""
    return Path(video_name).name


def _poster_path_for(video_name: str) -> Path:
    safe_name = _safe_video_filename(video_name)
    stem = Path(safe_name).stem
    return _poster_dir / f"{stem}.jpg"


def _ensure_poster(video_name: str) -> Path:
    safe_name = _safe_video_filename(video_name)
    video_path = _video_dir / safe_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    poster_path = _poster_path_for(video_name)
    if poster_path.exists():
        return poster_path

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        "1",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(poster_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="ffmpeg is required to generate posters.") from exc
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode(errors="ignore")
        raise HTTPException(status_code=500, detail=f"Poster generation failed: {err[:200]}") from exc

    return poster_path


@app.get("/video-poster/{video_name}")
async def video_poster(video_name: str):
    """
    Return a poster image for a given video. Generates and caches if missing.
    """
    poster_path = _ensure_poster(video_name)
    if not poster_path.exists():
        raise HTTPException(status_code=404, detail="Poster not found")
    return FileResponse(poster_path, media_type="image/jpeg")


@app.get("/learning-library")
async def learning_library(page: int = 1, limit: int = 12, q: str = ""):
    """
    List learning videos with labels from Dataset/Text/label.csv.
    Returns video URLs already exposed via /videos.
    """
    csv_path = _here / "Dataset" / "Text" / "label.csv"

    if not csv_path.exists():
        raise HTTPException(status_code=500, detail="label.csv not found in Dataset/Text")
    if not _video_dir.exists():
        raise HTTPException(status_code=500, detail="Videos directory not found in Dataset/Videos")

    items: list[dict[str, str | int]] = []
    try:
        query = (q or "").strip().lower()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                video_name = (row.get("VIDEO") or row.get("video") or "").strip()
                label = (row.get("LABEL") or row.get("label") or "Video hướng dẫn").strip()
                if not video_name:
                    continue
                if not (_video_dir / video_name).exists():
                    # Skip entries that don't have a matching video file.
                    continue
                if query and query not in label.lower():
                    continue
                items.append(
                    {
                        "id": row.get("ID") or idx,
                        "label": label,
                        "videoUrl": f"/videos/{video_name}",
                        "posterUrl": f"/video-poster/{video_name}",
                        "filename": video_name,
                    }
                )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read learning library: {exc}") from exc

    # Pagination
    limit = max(1, min(limit, 100))
    page = max(1, page)
    total = len(items)
    total_pages = max(1, (total + limit - 1) // limit)
    if page > total_pages:
        page = total_pages
    start = (page - 1) * limit
    end = start + limit

    return {
        "items": items[start:end],
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": total_pages,
    }


# Serve frontend from Kinesis3 build (includes static like /logo.png)
if not _kinesis_dist.exists():
    raise RuntimeError("Kinesis3 build not found. Please run npm run build in Kinesis3.")

app.mount("/", StaticFiles(directory=_kinesis_dist, html=True), name="kinesis")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=True)
