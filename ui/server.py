import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


APP_NAME = "file_organizer_ui"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = Path.home() / ".file_organizer.log"
CATEGORIES_FILE = Path.home() / ".file_organizer_discovered_categories.json"


app = FastAPI(title="File Organizer UI", version="0.1.0")


# Serve static UI
static_dir = PROJECT_ROOT / "ui" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="static")


def _tail_file(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return "(log file not found)"
    try:
        # Try using tail for efficiency on large files
        result = subprocess.run(["tail", f"-n{lines}", str(path)], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass

    # Fallback: read last N lines manually
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            content = f.readlines()
        return "".join(content[-lines:])
    except Exception:
        return "(unable to read log)"


def _get_status() -> Dict[str, Any]:
    # Basic process status check (same heuristic as CLI helper)
    try:
        current_pid = os.getpid()
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        running_pids = []
        for line in lines:
            if "file_organizer.py" in line and "python" in line and "manage_organizer.sh" not in line:
                parts = line.split()
                if len(parts) >= 2:
                    pid = int(parts[1])
                    if pid != current_pid:
                        running_pids.append(pid)
        is_running = len(running_pids) > 0
    except Exception:
        is_running = False
        running_pids = []

    return {
        "running": is_running,
        "pids": running_pids,
        "log_file": str(LOG_FILE),
        "categories_file": str(CATEGORIES_FILE),
    }


@app.get("/api/status")
def api_status():
    return JSONResponse(_get_status())


@app.get("/api/logs")
def api_logs(lines: int = 200):
    content = _tail_file(LOG_FILE, max(1, min(lines, 2000)))
    return JSONResponse({"lines": lines, "content": content})


@app.get("/api/categories")
def api_categories():
    if not CATEGORIES_FILE.exists():
        return JSONResponse({"categories": {}, "message": "No categories file yet."})
    try:
        with CATEGORIES_FILE.open("r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        return JSONResponse({"categories": data})
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Categories file is not valid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health endpoint
@app.get("/api/health")
def api_health():
    return {"status": "ok"}


# Root serves the UI index
@app.get("/")
def root_index():
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))


