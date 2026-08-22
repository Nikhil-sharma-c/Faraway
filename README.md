# ProctorAI — Organized Project

This is an organized copy of the working Face-Detection/ProctorAI application. The original folder is unchanged. The migration keeps the existing Flask API, computer-vision pipeline, model assets, legacy pages, datasets, and benchmark scripts while adding a React supervisor console.

## Structure

```text
project/
├── frontend/                 # React + Vite supervisor console
│   ├── src/
│   ├── package.json
│   └── legacy/                # Existing HTML/CSS/JS pages retained for compatibility
├── backend/
│   ├── server.py              # Existing Flask API, adjusted to project paths
│   ├── routes/                # Route modules for future extraction
│   ├── services/              # Service modules for future extraction
│   ├── database/              # Database adapters/migrations
│   └── requirements.txt
├── ml/
│   ├── proctor_ai.py          # Face mesh, gaze, pose, temporal risk engine
│   ├── face_recog.py          # Face detection/recognition/gallery
│   ├── phone_detect.py        # Phone/person ROI detector
│   ├── models/                # ONNX, MediaPipe and YOLO assets
│   └── requirements.txt
├── scripts/                   # Scraping, training, evaluation, benchmarks
├── data/                      # Raw, processed and exam-cheating dataset
├── tests/
├── docs/                      # Existing project documentation
├── .env                      # Local-only runtime settings (not for commit)
└── .gitignore
```

## Run locally

### Backend

```bash
cd backend
python -m venv .venv
# Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
pip install -r ../ml/requirements.txt
python server.py
```

The API listens on `http://localhost:5001`. It serves the legacy pages and API routes. PostgreSQL is still required for the existing authentication/admin/report features; set `DATABASE_URL` in `.env` or the environment before using those features.

### React frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api`, `/video_feed`, and `/models` to the Flask backend. The console polls `/api/status` and links to the retained monitoring, enrollment, replay, reports, and admin pages.

### ML-only monitor

The previous standalone monitor is available as `ml/legacy_monitor.py`. It writes `snapshot.json` in its working directory and is retained as a compatibility utility; the Flask worker is the main integrated runtime.

## Migration notes

- The original project at `faraway/Face-Detection-Project-using-opencv` was not modified.
- Model files were copied into `ml/models`; generated/downloaded weights remain ignored by Git.
- The large Flask file is intentionally preserved so behavior and API compatibility are not lost. `routes/`, `services/`, and `database/` are ready for incremental extraction.
- Secrets and runtime config must be supplied locally; do not commit `.env` or `backend/config.json`.
