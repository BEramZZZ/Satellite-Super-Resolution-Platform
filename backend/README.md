# Satellite Super-Resolution Backend

Minimal FastAPI backend for the satellite super-resolution website.

## Current scope

This first step only exposes a health endpoint. No ML models are loaded yet.

### Endpoints

- `GET /` — basic service information
- `GET /api/health` — backend health check

## Setup

From this `backend/` directory:

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API should then be available at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Next step

Once this endpoint is confirmed working, we will add the first real inference endpoint for **Real-ESRGAN zero-shot**.
