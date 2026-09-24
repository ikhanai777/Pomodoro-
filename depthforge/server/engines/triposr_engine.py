"""Single image → full 360° model with TripoSR, running locally on the CPU (free, offline after first run).

Needs the TripoSR repo (tools/TripoSR) and requirements-local-ai.txt. The first run downloads
~1.7 GB of weights from Hugging Face. Expect a few minutes per model on a laptop CPU.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

from config import settings
from jobs import Job
from tools import ToolError

_model = None
_rembg = None
_lock = threading.Lock()


def _repo() -> Path:
    return Path(settings.triposr_dir) if settings.triposr_dir else settings.tools_dir / "TripoSR"


def availability() -> dict:
    repo = _repo()
    missing = []
    if not (repo / "tsr" / "system.py").is_file():
        missing.append(f"TripoSR code (clone it to {repo})")
    for mod in ("torch", "omegaconf", "einops", "transformers", "rembg", "skimage", "huggingface_hub"):
        if importlib.util.find_spec(mod) is None:
            missing.append(mod)
    return {"available": not missing, "label": "TripoSR (local CPU) — free, a few minutes, vertex colours",
            "reason": ("Missing: " + ", ".join(missing)) if missing else ""}


def _load(job: Job):
    global _model, _rembg
    if _model is not None:
        return _model, _rembg
    repo = _repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if importlib.util.find_spec("torchmcubes") is None:
        sys.path.insert(0, str(Path(__file__).parent / "mcubes_shim"))
    import rembg
    import torch
    from tsr.system import TSR

    torch.set_num_threads(settings.threads)
    job.set("Loading TripoSR (first run downloads ~1.7 GB)", 0.05)
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to("cpu")
    _model, _rembg = model, rembg.new_session()
    return _model, _rembg


def run_job(job: Job) -> Path:
    av = availability()
    if not av["available"]:
        raise ToolError("Local TripoSR is not installed. " + av["reason"])
    with _lock:
        import numpy as np
        import torch
        from PIL import Image

        model, session = _load(job)
        from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

        job.set("Removing background", 0.15)
        src = next((job.dir / "input").iterdir())
        image = Image.open(src)
        if job.params.get("keep_background"):
            image = image.convert("RGBA")
        else:
            image = remove_background(image, session)  # skipped if the PNG already has transparency
            if image.mode != "RGBA":
                image = image.convert("RGBA")
        image = resize_foreground(image, float(job.params.get("foreground_ratio", 0.85)))
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        image = Image.fromarray((arr * 255.0).astype(np.uint8))
        image.save(job.dir / "preprocessed.png")
        job.check_cancel()

        job.set("Inferring 3D shape", 0.25)
        with torch.no_grad():
            scene_codes = model([image], device="cpu")
        job.check_cancel()

        res = int(job.params.get("resolution", 256))
        job.set(f"Extracting mesh ({res}³ grid)", 0.55)
        meshes = model.extract_mesh(scene_codes, True, resolution=res)
        mesh = to_gradio_3d_orientation(meshes[0])
        job.check_cancel()

        job.set("Saving model", 0.95)
        mesh.fix_normals()
        out = job.dir / "model.glb"
        mesh.export(str(out))
        return out
