# Development quickstart

## Prerequisites
- Python 3.10+ with `pip`
- Node.js 18+ and npm

Install Python dependencies once:
```bash
pip install -r requirements.txt
```

## Run the API (FastAPI)
From the repo root:
```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```
The API reads `data/metadata.db` for library stats and allows CORS from `http://localhost:5173`.

You can also use the module entrypoint:
```bash
python -m api
```

## Run the UI (Vite + React + TypeScript)
```bash
cd ui
npm install
npm run dev
```
This starts the UI dev server on `http://localhost:5173`.

## Smoke test the API
With the API running, check the library status endpoint:
```bash
curl http://127.0.0.1:8000/api/library/status
```
You should see JSON containing track counts and coverage percentages.
