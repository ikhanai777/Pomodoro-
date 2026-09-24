"""DepthForge local server: serves the web app and runs full 3D reconstruction jobs.

    python app.py            (or start.bat / start.sh)

Then open http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from config import APP_DIR, settings  # noqa: E402
from engines import fal_engine, photogrammetry, triposr_engine  # noqa: E402
from jobs import JobManager  # noqa: E402

app = FastAPI(title="DepthForge", docs_url="/api/docs", openapi_url="/api/openapi.json")
jobs = JobManager(settings.data_dir)

IMAGE_ENGINES = {"triposr-local": triposr_engine.run_job, **{k: fal_engine.run_job for k in fal_engine.MODELS}}
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _engines() -> dict:
    image = {"triposr-local": triposr_engine.availability(), **fal_engine.availability()}
    return {"image": image, "photos": {"photogrammetry": photogrammetry.availability()}}


@app.get("/api/health")
def health():
    return {"ok": True, "engines": _engines(), "threads": settings.threads,
            "presets": list(photogrammetry.PRESETS)}


async def _save_uploads(files: list[UploadFile], dest: Path) -> int:
    limit = settings.max_upload_mb * 1024 * 1024
    total = 0
    for i, f in enumerate(files):
        name = SAFE_NAME.sub("_", Path(f.filename or f"file{i}").name)[-120:] or f"file{i}"
        with open(dest / f"{i:04d}_{name}", "wb") as out:
            while chunk := await f.read(1 << 20):
                total += len(chunk)
                if total > limit:
                    raise HTTPException(413, f"Upload larger than {settings.max_upload_mb} MB")
                out.write(chunk)
    return total


@app.post("/api/jobs/image")
async def create_image_job(image: UploadFile = File(...), engine: str = Form("triposr-local"),
                           options: str = Form("{}")):
    if engine not in IMAGE_ENGINES:
        raise HTTPException(400, f"Unknown engine {engine}")
    av = _engines()["image"][engine]
    if not av["available"]:
        raise HTTPException(409, av.get("reason") or "Engine not available")
    try:
        params = json.loads(options or "{}")
    except ValueError:
        raise HTTPException(400, "options must be JSON")
    job = jobs.create("image", engine, params)
    await _save_uploads([image], job.dir / "input")
    jobs.submit(job, IMAGE_ENGINES[engine])
    return job.public()


@app.post("/api/jobs/photos")
async def create_photos_job(files: list[UploadFile] = File(...), quality: str = Form("standard"),
                            crop: bool = Form(True)):
    if quality not in photogrammetry.PRESETS:
        raise HTTPException(400, f"quality must be one of {list(photogrammetry.PRESETS)}")
    av = photogrammetry.availability()
    if not av["available"]:
        raise HTTPException(409, "Photogrammetry tools missing: " + ", ".join(av["missing"]))
    job = jobs.create("photos", "photogrammetry", {"quality": quality, "crop": crop, "files": len(files)})
    await _save_uploads(files, job.dir / "input")
    jobs.submit(job, photogrammetry.run_job)
    return job.public()


@app.get("/api/jobs")
def list_jobs():
    items = sorted(jobs.jobs.values(), key=lambda j: j.created, reverse=True)[:50]
    return [j.public(log_lines=0) for j in items]


def _job_or_404(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "No such job")
    return job


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return _job_or_404(jid).public()


@app.delete("/api/jobs/{jid}")
def cancel_job(jid: str):
    job = _job_or_404(jid)
    jobs.cancel(job)
    return job.public()


@app.get("/api/jobs/{jid}/model.glb")
def job_model(jid: str):
    job = _job_or_404(jid)
    if not job.result:
        raise HTTPException(404, "Model not ready")
    path = (job.dir / job.result).resolve()
    if job.dir.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "Model missing")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"depthforge-{jid}.glb")


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    return JSONResponse({"detail": str(exc)}, status_code=500)


# The web app itself (index.html and friends). Mounted last so /api wins.
app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="app")


if __name__ == "__main__":
    import uvicorn
    print(f"DepthForge running at http://{settings.host}:{settings.port}")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
