# Vietnamese Sign Language Recognition

A lightweight FastAPI + React (Vite) app for Vietnamese Sign Language (VSL): browse/sign videos, run speech-to-text via OpenAI, and serve local dataset videos/posters.

## Requirements
- Python 3.10+ (recommended)
- Node.js 18+ and npm
- FFmpeg (for video posters) — verify with `ffmpeg -version`
- OpenAI API key (`OPENAI_API_KEY` in `.env`)
- (Optional) GPU for training/augmentation

## Project Layout
- `api.py` — FastAPI backend (serves Kinesis3 build, `/learning-library`, `/sign-animation`, `/transcribe`, posters, videos)
- `Kinesis3/` — frontend (Vite/React)
- `Dataset/Videos/` — video files (served at `/videos/...`)
- `Dataset/Text/label.csv` — labels for videos (`VIDEO`, `LABEL`)
- `Outputs/app_predictions/` — TTS output folder
- Augmentation/training scripts: `create_data_augment.py`, `create_data_augment_gpu.py`, `download_data.py`, `trainning.ipynb`

## Setup

### 1) Clone and enter
```bash
git clone <repo-url>
cd Vietnamese-Sign-Language-Recognition
```

### 2) Python env + deps
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install --upgrade pip
pip install -r requirements.txt
```
(If mediapipe/protobuf conflicts, pin: `pip install "protobuf<4" "mediapipe==0.10.9"`.)

### 3) Node deps + build frontend
```bash
cd Kinesis3
npm install
npm run build
cd ..
```
Build outputs to `Kinesis3/dist` (served by FastAPI).

### 4) FFmpeg
Install if missing (pick one):
- Windows: `winget install --id=Gyan.FFmpeg.Full -e` or `choco install ffmpeg -y`
- Ubuntu/Debian: `sudo apt update && sudo apt install -y ffmpeg`
- macOS (Homebrew): `brew install ffmpeg`

### 5) Environment variables
Create `.env` in repo root:
```
OPENAI_API_KEY=sk-...
```
(Backend auto-loads `.env`.)

## Running

### Backend
From repo root:
```bash
.\.venv\Scripts\activate
python -m uvicorn api:app --reload --port 8001
```
- Serves frontend at `/` from `Kinesis3/dist`
- Videos at `/videos/<file>`
- Posters at `/video-poster/<file>`
- Learning library at `/learning-library`
- STT at `/transcribe` (uses `gpt-4o-transcribe`)
- Sign animation mapping at `/sign-animation` (maps text to library video or 404 if not found)

### Frontend
Already served by FastAPI after `npm run build`. Open `http://127.0.0.1:8001`.

## Data Notes
- `Dataset/Text/label.csv` has ~12,418 rows but only ~4,000 actual video files; many labels are duplicates or missing video files. Clean or dedupe if needed before training/serving.
- `download_data.py` appends all fetched entries without deduplication; run cleanup to avoid duplicate labels.

## Training / Augmentation (optional)
- `download_data.py` — fetch videos/labels from QIPEDC.
- `create_data_augment.py` / `_gpu.py` — extract landmarks, augment sequences, save `.npz`.
- `trainning.ipynb` — train the model (GPU recommended).
