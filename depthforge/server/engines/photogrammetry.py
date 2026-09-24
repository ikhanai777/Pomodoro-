"""Multi-photo photogrammetry: COLMAP (structure from motion) + OpenMVS (dense mesh + texture), CPU only.

Pipeline: photos/video → frames → SIFT features → matching → sparse reconstruction → undistort
→ OpenMVS densify → mesh → (refine) → texture → GLB, re-oriented so the cameras' "up" is +Y.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from config import settings
from jobs import Job
from tools import ToolError, find_tool, help_text, run

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}

PRESETS = {
    # max_images: frames/photos kept; max_side: pixels; level: OpenMVS resolution level (higher = coarser)
    "draft":    dict(max_images=40,  max_side=1200, level=2, refine=False, texture_level=1),
    "standard": dict(max_images=80,  max_side=1600, level=1, refine=False, texture_level=0),
    "high":     dict(max_images=150, max_side=2400, level=1, refine=True,  texture_level=0),
}

OPENMVS_TOOLS = ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh"]


def locate_tools() -> dict:
    colmap = find_tool("colmap", settings.colmap, settings.tools_dir / "colmap")
    mvs = {t: find_tool(t, settings.openmvs, settings.tools_dir / "openmvs") for t in OPENMVS_TOOLS}
    return {"colmap": colmap, **mvs}


def availability() -> dict:
    t = locate_tools()
    missing = [k for k, v in t.items() if v is None and k != "RefineMesh"]
    return {"available": not missing, "missing": missing,
            "tools": {k: str(v) if v else None for k, v in t.items()}}


# ---------------------------------------------------------------- ingest
def _sharpness(gray: np.ndarray) -> float:
    import cv2
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _save(img: Image.Image, path: Path, max_side: int) -> None:
    img = img.convert("RGB")
    s = max_side / max(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    img.save(path, "JPEG", quality=94)


def extract_video(path: Path, out: Path, want: int, max_side: int, prefix: str, job: Job) -> int:
    """Pick `want` frames spread evenly through the video, choosing the sharpest of a few candidates each time."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        raise ToolError(f"Could not read video {path.name}")
    want = max(1, min(want, total))
    seg = total / want
    candidates: dict[int, int] = {}  # frame index -> segment
    for k in range(want):
        a, b = int(k * seg), max(int(k * seg) + 1, int((k + 1) * seg))
        for f in np.linspace(a, b - 1, num=min(3, b - a), dtype=int):
            candidates[int(f)] = k
    best: dict[int, tuple[float, np.ndarray]] = {}
    idx = 0
    while idx < total:
        job.check_cancel()
        if not cap.grab():
            break
        if idx in candidates:
            ok, frame = cap.retrieve()
            if ok:
                small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
                score = _sharpness(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
                k = candidates[idx]
                if k not in best or score > best[k][0]:
                    best[k] = (score, frame)
        idx += 1
        if idx % 200 == 0:
            job.set(progress=0.05 * idx / total)
    cap.release()
    for k, (_, frame) in sorted(best.items()):
        _save(Image.fromarray(frame[:, :, ::-1]), out / f"{prefix}_{k:04d}.jpg", max_side)
    return len(best)


def ingest(job: Job, images: Path, preset: dict) -> bool:
    """Copy/resize photos and extract video frames into `images`. Returns True if frames come from video."""
    files = sorted(p for p in (job.dir / "input").iterdir() if p.is_file())
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXT]
    photos = [p for p in files if p.suffix.lower() in IMAGE_EXT]
    images.mkdir(parents=True, exist_ok=True)
    max_n, side = preset["max_images"], preset["max_side"]
    if photos and len(photos) > max_n:
        keep = np.linspace(0, len(photos) - 1, max_n).round().astype(int)
        job.log(f"Using {max_n} of {len(photos)} photos (preset limit).")
        photos = [photos[i] for i in sorted(set(keep))]
    for i, p in enumerate(photos):
        job.check_cancel()
        try:
            with Image.open(p) as im:
                _save(ImageOps.exif_transpose(im), images / f"img_{i:04d}.jpg", side)
        except OSError as e:
            job.log(f"Skipped {p.name}: {e}")
    for vi, v in enumerate(videos):
        n = extract_video(v, images, max(8, (max_n - len(photos)) // max(1, len(videos))), side, f"vid{vi}", job)
        job.log(f"Extracted {n} frames from {v.name}")
    count = len(list(images.glob("*.jpg")))
    if count < 8:
        raise ToolError(f"Only {count} usable images. Photogrammetry needs at least 8 (30–80 recommended), "
                        "taken all the way around the object with plenty of overlap.")
    job.log(f"{count} images ready.")
    return bool(videos) and not photos


# ---------------------------------------------------------------- helpers
def _flag(help_txt: str, *candidates: str) -> str | None:
    for c in candidates:
        if "--" + c in help_txt:
            return c
    return None


def _progress_parser(job: Job, lo: float, hi: float, total: int | None = None):
    pct = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
    reg = re.compile(r"Registering image #\d+ \((\d+)\)")

    def on_line(line: str) -> None:
        f = None
        m = reg.search(line)
        if m and total:
            f = int(m.group(1)) / total
        else:
            m = pct.search(line)
            if m:
                f = float(m.group(1)) / 100
        if f is not None and 0 <= f <= 1:
            job.set(progress=lo + (hi - lo) * f)
    return on_line


def _best_sparse(colmap: Path, sparse: Path, job: Job) -> tuple[Path, int]:
    best, best_n = None, -1
    for d in sorted(p for p in sparse.iterdir() if p.is_dir()):
        n = -1
        txt = help_text(str(colmap), "model_analyzer", "--path", str(d))
        m = re.search(r"Registered images:\s*(\d+)", txt)
        if m:
            n = int(m.group(1))
        elif (d / "images.bin").is_file():
            n = (d / "images.bin").stat().st_size // 1000
        job.log(f"Sparse model {d.name}: {n} registered images")
        if n > best_n:
            best, best_n = d, n
    if best is None:
        raise ToolError("COLMAP could not reconstruct the scene. Use more photos with more overlap "
                        "(each spot seen from 3+ photos), and avoid shiny, transparent or plain untextured objects.")
    return best, best_n


def camera_poses(images_txt: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(centre, viewing direction, up) of every camera in a COLMAP images.txt, in world coordinates."""
    poses = []
    lines = [l for l in images_txt.read_text().splitlines() if l and not l.startswith("#")]
    for line in lines[0::2]:
        parts = line.split()
        if len(parts) < 8:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        R = np.array([  # world -> camera
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        centre = -R.T @ np.array([tx, ty, tz])
        poses.append((centre, R[2], -R[1]))  # camera looks along +z; its y axis points down
    return poses


def camera_up(images_txt: Path) -> np.ndarray | None:
    """Average world-space 'up' of all cameras."""
    poses = camera_poses(images_txt)
    if not poses:
        return None
    up = np.mean([p[2] for p in poses], axis=0)
    n = np.linalg.norm(up)
    return up / n if n > 1e-6 else None


def focus_sphere(images_txt: Path, scale: float = 0.6) -> tuple[np.ndarray, float] | None:
    """The point the cameras look at (least squares over their viewing rays), and a radius around it."""
    poses = camera_poses(images_txt)
    if len(poses) < 3:
        return None
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, d, _ in poses:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    try:
        centre = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    dist = np.median([np.linalg.norm(c - centre) for c, _, _ in poses])
    # Cameras must actually face the point (not a degenerate, all-parallel capture).
    facing = np.mean([np.dot(d, centre - c) > 0 for c, d, _ in poses])
    return (centre, float(dist * scale)) if facing > 0.8 else None


def finalize(src: Path, out: Path, up: np.ndarray | None, job: Job,
             focus: tuple[np.ndarray, float] | None = None) -> Path:
    """Load the textured mesh, crop to the object, drop floating fragments, stand it upright, centre it, write GLB."""
    import trimesh
    scene = trimesh.load(str(src), force="scene")
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        mesh = scene.geometry[gname]
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) < 1000:
            continue
        if focus is not None:
            centre, radius = focus
            world = trimesh.transform_points(mesh.vertices, T)
            inside = np.linalg.norm(world - centre, axis=1) <= radius
            keep = inside[mesh.faces].any(axis=1)
            if 100 < keep.sum() < len(keep):
                job.log(f"Cropped to the object: kept {int(keep.sum())} of {len(keep)} faces.")
                mesh.update_faces(keep)
                mesh.remove_unreferenced_vertices()
        # Mask out small disconnected fragments in place (splitting would copy the texture per piece).
        # Textured meshes repeat vertices along UV seams, so connect faces by vertex position, not index.
        _, inverse = trimesh.grouping.unique_rows(mesh.vertices, digits=6)
        faces = inverse[mesh.faces]
        comps = trimesh.graph.connected_components(trimesh.graph.face_adjacency(faces=faces),
                                                   nodes=np.arange(len(faces)), min_len=1)
        total = len(mesh.faces)
        small = [c for c in comps if len(c) < 0.01 * total]
        if small and len(small) < len(comps):
            mask = np.ones(total, dtype=bool)
            for c in small:
                mask[c] = False
            mesh.update_faces(mask)
            mesh.remove_unreferenced_vertices()
            job.log(f"Removed {len(small)} small floating fragments ({total - int(mask.sum())} faces).")
    if up is not None:
        scene.apply_transform(trimesh.geometry.align_vectors(up, [0, 1, 0]))
    lo, hi = scene.bounds
    size = float(np.max(hi - lo)) or 1.0
    T = np.eye(4)
    T[:3, :3] *= 2.0 / size
    T[:3, 3] = -(lo + hi) / 2 * (2.0 / size)
    scene.apply_transform(T)
    scene.export(str(out))
    return out


# ---------------------------------------------------------------- main
def run_job(job: Job) -> Path:
    tools = locate_tools()
    missing = [k for k, v in tools.items() if v is None and k != "RefineMesh"]
    if missing:
        raise ToolError("Photogrammetry tools not found: " + ", ".join(missing) + ". Run setup.ps1 or set COLMAP_PATH / OPENMVS_PATH.")
    preset = PRESETS.get(job.params.get("quality", "standard"), PRESETS["standard"])
    T = settings.threads
    work = (job.dir / "work").resolve()
    images, db, sparse, dense = work / "images", work / "database.db", work / "sparse", work / "dense"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    colmap = tools["colmap"]
    ctx = dict(cwd=work, log=job.log, cancel=job.cancel)

    job.set("Preparing photos", 0.0)
    from_video = ingest(job, images, preset)
    n_images = len(list(images.glob("*.jpg")))
    same_size = len({Image.open(p).size for p in images.glob("*.jpg")}) == 1

    job.set("Finding features", 0.05)
    h = help_text(str(colmap), "feature_extractor", "-h")
    grp = _flag(h, "FeatureExtraction.use_gpu", "SiftExtraction.use_gpu") or "SiftExtraction.use_gpu"
    cmd = [colmap, "feature_extractor", "--database_path", db, "--image_path", images,
           "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "1" if (from_video or same_size) else "0",
           "--" + grp, "0"]
    for flag, val in (("FeatureExtraction.max_image_size", preset["max_side"]), ("SiftExtraction.max_image_size", preset["max_side"]),
                      ("FeatureExtraction.num_threads", T), ("SiftExtraction.num_threads", T)):
        if "--" + flag in h:
            cmd += ["--" + flag, val]
    run(cmd, **ctx, on_line=_progress_parser(job, 0.05, 0.12))

    job.set("Matching photos", 0.12)
    matcher = "sequential_matcher" if from_video else "exhaustive_matcher"
    h = help_text(str(colmap), matcher, "-h")
    grp = _flag(h, "FeatureMatching.use_gpu", "SiftMatching.use_gpu") or "SiftMatching.use_gpu"
    cmd = [colmap, matcher, "--database_path", db, "--" + grp, "0"]
    for flag, val in (("FeatureMatching.num_threads", T), ("SiftMatching.num_threads", T),
                      ("SequentialMatching.overlap", 15), ("SequentialMatching.quadratic_overlap", 1)):
        if "--" + flag in h:
            cmd += ["--" + flag, val]
    run(cmd, **ctx, on_line=_progress_parser(job, 0.12, 0.25))

    job.set("Reconstructing camera positions", 0.25)
    sparse.mkdir()
    h = help_text(str(colmap), "mapper", "-h")
    cmd = [colmap, "mapper", "--database_path", db, "--image_path", images, "--output_path", sparse]
    if "--Mapper.num_threads" in h:
        cmd += ["--Mapper.num_threads", T]
    if "--Mapper.ba_use_gpu" in h:
        cmd += ["--Mapper.ba_use_gpu", 0]
    run(cmd, **ctx, on_line=_progress_parser(job, 0.25, 0.42, total=n_images))
    best, n_reg = _best_sparse(colmap, sparse, job)
    if 0 <= n_reg < 5:
        raise ToolError(f"Only {n_reg} of {n_images} photos could be placed. Take more photos with more overlap.")
    job.set(message=f"{n_reg} of {n_images} photos placed")

    job.set("Undistorting photos", 0.42)
    run([colmap, "image_undistorter", "--image_path", images, "--input_path", best, "--output_path", dense,
         "--output_type", "COLMAP", "--max_image_size", preset["max_side"]], **ctx)
    up = focus = None
    try:
        txt = work / "sparse_txt"
        txt.mkdir()
        run([colmap, "model_converter", "--input_path", dense / "sparse", "--output_path", txt, "--output_type", "TXT"], **ctx)
        up = camera_up(txt / "images.txt")
        if job.params.get("crop", True):
            focus = focus_sphere(txt / "images.txt")
    except (ToolError, OSError) as e:
        job.log(f"Could not read camera orientation ({e}); model may need manual rotation.")

    def mvs(tool: str, args: list, lo: float, hi: float, cuda: bool = True):
        exe = tools[tool]
        h = help_text(str(exe), "--help")
        extra = ["-w", work, "--max-threads", T] if "max-threads" in h else ["-w", work]
        if cuda and "cuda-device" in h:
            extra += ["--cuda-device", "-2"]  # force CPU: the GPU path needs a modern CUDA card
        run([exe, *args, *extra], **ctx, on_line=_progress_parser(job, lo, hi))

    job.set("Importing into OpenMVS", 0.44)
    h = help_text(str(tools["InterfaceCOLMAP"]), "--help")
    img_flag = "--image-folder" if "image-folder" in h else None
    mvs("InterfaceCOLMAP", ["-i", dense, "-o", work / "scene.mvs"] + ([img_flag, dense / "images"] if img_flag else []), 0.44, 0.46, cuda=False)

    job.set("Building dense point cloud", 0.46)
    mvs("DensifyPointCloud", ["-i", work / "scene.mvs", "-o", work / "scene_dense.mvs",
                              "--resolution-level", preset["level"]], 0.46, 0.72)

    job.set("Building mesh", 0.72)
    mvs("ReconstructMesh", ["-i", work / "scene_dense.mvs", "-o", work / "scene_mesh.mvs"], 0.72, 0.80)
    state = {"stem": "scene_mesh"}

    def mesh_input(tool: str) -> list:
        """OpenMVS 2.x writes the mesh as <stem>.ply and takes it back with -m; 1.x kept it inside <stem>.mvs."""
        ply = work / (state["stem"] + ".ply")
        if ply.is_file() and "mesh-file" in help_text(str(tools[tool]), "--help"):
            return ["-i", work / "scene_dense.mvs", "-m", ply]
        return ["-i", work / (state["stem"] + ".mvs")]

    if preset["refine"] and tools["RefineMesh"]:
        job.set("Refining mesh detail", 0.80)
        mvs("RefineMesh", mesh_input("RefineMesh") + ["-o", work / "scene_refine.mvs",
                                                     "--resolution-level", preset["level"]], 0.80, 0.88)
        state["stem"] = "scene_refine"

    job.set("Texturing", 0.88)
    h = help_text(str(tools["TextureMesh"]), "--help")
    args = mesh_input("TextureMesh") + ["-o", work / "scene_texture.mvs"]
    if "resolution-level" in h:
        args += ["--resolution-level", preset["texture_level"]]
    # GLB first; PLY (with a PNG texture) as a fallback. OpenMVS's OBJ writer is not used: it crashes in some builds.
    types = ["glb", "ply"] if "export-type" in h else [None]
    for i, kind in enumerate(types):
        try:
            mvs("TextureMesh", args + (["--export-type", kind] if kind else []), 0.88, 0.96)
            break
        except ToolError:
            if i == len(types) - 1:
                raise
            job.log(f"TextureMesh could not write {kind}; retrying with {types[i + 1]}.")

    job.set("Packaging model", 0.96)
    outs = [p for ext in (".glb", ".ply", ".obj") for p in sorted(work.glob(f"scene_texture*{ext}"))]
    if not outs:
        raise ToolError("TextureMesh finished but produced no mesh file.")
    return finalize(outs[0], job.dir / "model.glb", up, job, focus)
