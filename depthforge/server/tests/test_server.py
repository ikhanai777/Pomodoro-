"""Tests for the DepthForge server. Run from depthforge/server:  python -m pytest tests -q

The photogrammetry end-to-end test needs COLMAP + OpenMVS and a folder of photos:
    DEPTHFORGE_TEST_PHOTOS=path/to/photos python -m pytest tests -q -k photogrammetry
"""
from __future__ import annotations

import http.server
import importlib.util
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["DEPTHFORGE_DATA_DIR"] = str(tmp_path_factory.mktemp("data"))
    os.environ.setdefault("DEPTHFORGE_TOOLS_DIR", str(tmp_path_factory.mktemp("tools")))
    from fastapi.testclient import TestClient
    import app as app_module
    return TestClient(app_module.app), app_module


def _png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 80, 40)).save(buf, "PNG")
    return buf.getvalue()


def test_health_lists_engines(client):
    c, _ = client
    r = c.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert {"triposr-local", "fal-hunyuan3d", "fal-trellis", "fal-triposr"} <= set(body["engines"]["image"])
    assert "photogrammetry" in body["engines"]["photos"]
    assert body["presets"] == ["draft", "standard", "high"]


def test_serves_web_app(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200 and "DepthForge" in r.text


def test_rejects_unavailable_engine_and_bad_input(client):
    c, app_module = client
    app_module.settings.fal_key = ""
    r = c.post("/api/jobs/image", files={"image": ("a.png", _png(), "image/png")}, data={"engine": "fal-trellis"})
    assert r.status_code == 409
    r = c.post("/api/jobs/image", files={"image": ("a.png", _png(), "image/png")}, data={"engine": "nope"})
    assert r.status_code == 400
    assert c.get("/api/jobs/doesnotexist").status_code == 404


# ------------------------------------------------------------------ fal.ai engine against a fake queue API
class FakeFal(http.server.BaseHTTPRequestHandler):
    polls = 0
    submitted: dict = {}

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeFal.submitted = {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        base = f"http://127.0.0.1:{self.server.server_port}"
        self._json({"request_id": "r1", "status_url": base + "/status", "response_url": base + "/result"})

    def do_GET(self):
        base = f"http://127.0.0.1:{self.server.server_port}"
        if self.path.startswith("/status"):
            FakeFal.polls += 1
            done = FakeFal.polls >= 2
            self._json({"status": "COMPLETED" if done else "IN_PROGRESS", "logs": [{"message": "step"}]})
        elif self.path == "/result":
            self._json({"model_mesh": {"url": base + "/files/model.glb", "content_type": "model/gltf-binary"}})
        elif self.path == "/files/model.glb":
            FakeFal.download_auth = self.headers.get("Authorization")
            data = b"glTF" + b"\0" * 16
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json({"detail": "nope"}, 404)


def test_fal_engine_round_trip(client, monkeypatch):
    c, app_module = client
    from engines import fal_engine
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeFal)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(fal_engine, "QUEUE", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setattr(fal_engine.settings, "fal_key", "test-key")
    monkeypatch.setattr(fal_engine.time, "sleep", lambda s: None)

    r = c.post("/api/jobs/image", files={"image": ("cup.png", _png(), "image/png")}, data={"engine": "fal-trellis"})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    for _ in range(100):
        job = c.get(f"/api/jobs/{jid}").json()
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    srv.shutdown()
    assert job["status"] == "done", job
    assert FakeFal.submitted["path"] == "/fal-ai/trellis"
    assert FakeFal.submitted["auth"] == "Key test-key"
    assert FakeFal.submitted["body"]["image_url"].startswith("data:image/png;base64,")
    assert FakeFal.download_auth is None  # the key is never sent to the file host
    glb = c.get(job["result_url"].replace("api/", "/api/", 1))
    assert glb.status_code == 200 and glb.content.startswith(b"glTF")
    assert any(j["id"] == jid for j in c.get("/api/jobs").json())


def test_find_glb_url_handles_result_shapes():
    from engines.fal_engine import _find_glb_url
    assert _find_glb_url({"model_mesh": {"url": "https://x/a.glb"}}) == "https://x/a.glb"
    assert _find_glb_url({"model_glb": {"url": "https://x/b.glb"}}) == "https://x/b.glb"
    assert _find_glb_url({"output": [{"file": "https://x/c.glb?sig=1"}]}) == "https://x/c.glb?sig=1"
    assert _find_glb_url({"nothing": 1}) is None


# ------------------------------------------------------------------ photogrammetry helpers
def test_camera_up_from_colmap_images(tmp_path):
    from engines.photogrammetry import camera_up
    # Identity rotation: camera y points down in world, so "up" is -y.
    (tmp_path / "images.txt").write_text("# header\n1 1 0 0 0 0 0 0 1 a.jpg\n\n2 1 0 0 0 1 0 0 1 b.jpg\n\n")
    up = camera_up(tmp_path / "images.txt")
    assert np.allclose(up, [0, -1, 0])


def test_ingest_extracts_sharpest_video_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from engines import photogrammetry
    from jobs import Job
    job = Job(id="t", kind="photos", engine="photogrammetry", dir=tmp_path)
    (tmp_path / "input").mkdir()
    vw = cv2.VideoWriter(str(tmp_path / "input" / "walk.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    rng = np.random.default_rng(0)
    for i in range(60):
        frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        if i % 3:  # blur two of every three frames
            frame = cv2.GaussianBlur(frame, (15, 15), 5)
        vw.write(frame)
    vw.release()
    preset = dict(photogrammetry.PRESETS["draft"], max_images=10)
    from_video = photogrammetry.ingest(job, tmp_path / "images", preset)
    frames = sorted((tmp_path / "images").glob("*.jpg"))
    assert from_video and len(frames) == 10


def test_ingest_refuses_too_few_photos(tmp_path):
    from PIL import Image
    from engines import photogrammetry
    from jobs import Job
    from tools import ToolError
    job = Job(id="t", kind="photos", engine="photogrammetry", dir=tmp_path)
    (tmp_path / "input").mkdir()
    for i in range(3):
        Image.new("RGB", (50, 50)).save(tmp_path / "input" / f"{i}.jpg")
    with pytest.raises(ToolError, match="at least 8"):
        photogrammetry.ingest(job, tmp_path / "images", photogrammetry.PRESETS["draft"])


@pytest.mark.skipif(not os.environ.get("DEPTHFORGE_TEST_PHOTOS"), reason="set DEPTHFORGE_TEST_PHOTOS to a folder of photos")
def test_photogrammetry_end_to_end(tmp_path):
    import shutil
    import trimesh
    from engines import photogrammetry
    from jobs import JobManager
    if not photogrammetry.availability()["available"]:
        pytest.skip("COLMAP/OpenMVS not installed")
    jm = JobManager(tmp_path)
    job = jm.create("photos", "photogrammetry", {"quality": "draft"})
    for p in sorted(Path(os.environ["DEPTHFORGE_TEST_PHOTOS"]).glob("*.jpg")):
        shutil.copy(p, job.dir / "input" / p.name)
    out = photogrammetry.run_job(job)
    scene = trimesh.load(out, force="scene")
    assert sum(len(g.faces) for g in scene.geometry.values()) > 1000


# ------------------------------------------------------------------ torchmcubes replacement
@pytest.mark.skipif(importlib.util.find_spec("torch") is None or importlib.util.find_spec("skimage") is None,
                    reason="needs torch and scikit-image")
def test_mcubes_shim_matches_torchmcubes_conventions():
    import torch
    import trimesh
    sys.path.insert(0, str(SERVER / "engines" / "mcubes_shim"))
    import torchmcubes as shim  # our module (the real one is not installed in test envs)
    n = 40
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    f = ((i - 15) / 10.0) ** 2 + ((j - 20) / 7.0) ** 2 + ((k - 22) / 4.0) ** 2
    v, faces = shim.marching_cubes(torch.from_numpy((1 - f).astype(np.float32)), 0.0)
    m = trimesh.Trimesh(v.numpy(), faces.numpy())
    ext = m.bounds[1] - m.bounds[0]
    # torchmcubes returns (x, y, z) = (dim2, dim1, dim0)
    assert np.allclose(ext, [8, 14, 20], atol=0.6)
    assert m.is_watertight
    assert m.volume < 0  # same winding as torchmcubes (TripoSR fixes normals afterwards)


def test_finalize_keeps_textured_mesh_whole_and_crops(tmp_path):
    """UV seams duplicate vertices; clean-up must not mistake texture patches for floating fragments."""
    import trimesh
    from PIL import Image
    from engines.photogrammetry import finalize
    from jobs import Job
    big = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    # Split every face into its own vertices, like a heavily patched texture atlas.
    unmerged = trimesh.Trimesh(big.vertices[big.faces].reshape(-1, 3), np.arange(len(big.faces) * 3).reshape(-1, 3), process=False)
    uv = np.random.default_rng(1).random((len(unmerged.vertices), 2))
    unmerged.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.new("RGB", (8, 8), (90, 120, 200)))
    speck = trimesh.creation.icosphere(subdivisions=0, radius=0.02)  # 20 faces, well under 1%
    speck.apply_translation([0, 1.5, 0])
    far = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    far.apply_translation([30, 0, 0])
    scene = trimesh.Scene()
    scene.add_geometry(unmerged, geom_name="obj")
    src = tmp_path / "in.glb"
    scene.export(src)
    # Put the speck and the far box into the same mesh as extra components.
    loaded = trimesh.load(src, force="scene")
    g = list(loaded.geometry.values())[0]
    merged = trimesh.util.concatenate([g, trimesh.Trimesh(speck.vertices, speck.faces), trimesh.Trimesh(far.vertices, far.faces)])
    out_scene = trimesh.Scene(); out_scene.add_geometry(merged, geom_name="obj"); out_scene.export(src)
    job = Job(id="t", kind="photos", engine="photogrammetry", dir=tmp_path)
    out = finalize(src, tmp_path / "model.glb", np.array([0, 1.0, 0]), job, focus=(np.zeros(3), 5.0))
    result = list(trimesh.load(out, force="scene").geometry.values())[0]
    assert len(result.faces) == len(big.faces)  # sphere kept whole; speck (<1%) and far box (cropped) removed
    assert any("Cropped" in l for l in job.logs) and any("fragment" in l for l in job.logs)
