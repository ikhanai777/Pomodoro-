"""Single image → full 360° textured model via fal.ai hosted models (needs FAL_KEY, billed per run).

Uses fal's queue REST API directly: submit, poll status, fetch result, download the GLB.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path

import httpx

from config import settings
from jobs import Job
from tools import ToolError

QUEUE = os.environ.get("FAL_QUEUE_URL", "https://queue.fal.run")

# engine id -> (fal app id, name of the image field, extra inputs, label)
MODELS = {
    "fal-hunyuan3d": ("fal-ai/hunyuan3d/v2", "input_image_url", {"textured_mesh": True},
                      "Hunyuan3D-2 (cloud) — best detail and texture"),
    "fal-trellis":   ("fal-ai/trellis", "image_url", {},
                      "TRELLIS (cloud) — clean geometry, good texture"),
    "fal-triposr":   ("fal-ai/triposr", "image_url", {"output_format": "glb"},
                      "TripoSR (cloud) — fastest, lower detail"),
}


def availability() -> dict:
    return {eid: {"available": bool(settings.fal_key), "label": label,
                  "reason": "" if settings.fal_key else "Set FAL_KEY to enable cloud engines"}
            for eid, (_, _, _, label) in MODELS.items()}


def _find_glb_url(obj) -> str | None:
    """Result shapes differ between models; take the first URL that looks like a mesh."""
    if isinstance(obj, dict):
        for key in ("model_mesh", "model_glb", "mesh", "glb", "model"):
            v = obj.get(key)
            if isinstance(v, dict) and isinstance(v.get("url"), str):
                return v["url"]
            if isinstance(v, str) and v.startswith("http"):
                return v
        for v in obj.values():
            u = _find_glb_url(v)
            if u:
                return u
    elif isinstance(obj, list):
        for v in obj:
            u = _find_glb_url(v)
            if u:
                return u
    elif isinstance(obj, str) and obj.startswith("http") and obj.split("?")[0].lower().endswith((".glb", ".gltf")):
        return obj
    return None


def run_job(job: Job, base_url: str | None = None) -> Path:
    base_url = base_url or QUEUE
    if not settings.fal_key:
        raise ToolError("FAL_KEY is not set. Add your fal.ai API key to config.json or the FAL_KEY environment variable.")
    app_id, field, extra, _ = MODELS[job.engine]
    app_id = settings.fal_models.get(job.engine, app_id)
    src = next((job.dir / "input").iterdir())
    mime = mimetypes.guess_type(src.name)[0] or "image/png"
    data_uri = f"data:{mime};base64," + base64.b64encode(src.read_bytes()).decode()
    headers = {"Authorization": f"Key {settings.fal_key}"}
    with httpx.Client(timeout=httpx.Timeout(60, read=300), headers=headers, follow_redirects=True) as http:
        job.set("Uploading to the cloud", 0.03)
        r = http.post(f"{base_url}/{app_id}", json={field: data_uri, **extra, **job.params.get("fal_inputs", {})})
        if r.status_code in (401, 403):
            raise ToolError("fal.ai rejected the API key (check FAL_KEY).")
        r.raise_for_status()
        sub = r.json()
        rid = sub["request_id"]
        status_url = sub.get("status_url") or f"{base_url}/{app_id}/requests/{rid}/status"
        response_url = sub.get("response_url") or f"{base_url}/{app_id}/requests/{rid}"
        cancel_url = sub.get("cancel_url") or f"{base_url}/{app_id}/requests/{rid}/cancel"
        job.log(f"fal request {rid}")
        job.set("Generating 3D model in the cloud", 0.08)
        t0, seen = time.time(), 0
        while True:
            if job.cancel.is_set():
                try:
                    http.put(cancel_url)
                finally:
                    job.check_cancel()
            st = http.get(status_url, params={"logs": 1}).json()
            for entry in (st.get("logs") or [])[seen:]:
                job.log(str(entry.get("message", entry)))
            seen = len(st.get("logs") or [])
            status = st.get("status")
            if status == "COMPLETED":
                if st.get("error"):
                    raise ToolError(f"fal.ai: {st['error']}")
                break
            if status == "IN_QUEUE":
                job.set(message=f"Queued at fal.ai (position {st.get('queue_position', '?')})")
            else:
                job.set(message="Generating…")
            # Typical runs take 20–90 s; creep towards 90% so the bar keeps moving.
            job.set(progress=0.08 + 0.82 * (1 - 1 / (1 + (time.time() - t0) / 45)))
            time.sleep(1.5)
        res = http.get(response_url)
        if res.status_code >= 400:
            raise ToolError(f"fal.ai returned {res.status_code}: {res.text[:300]}")
        url = _find_glb_url(res.json())
        if not url:
            raise ToolError("fal.ai finished but returned no model file: " + res.text[:300])
        job.set("Downloading model", 0.92)
    out = job.dir / "model.glb"
    # Separate client: the file host must not receive the API key.
    with httpx.Client(timeout=httpx.Timeout(60, read=300), follow_redirects=True) as plain, plain.stream("GET", url) as dl:
        dl.raise_for_status()
        with open(out, "wb") as f:
            for chunk in dl.iter_bytes():
                job.check_cancel()
                f.write(chunk)
    return out
