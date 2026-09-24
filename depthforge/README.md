# DepthForge

Turns images into textured 3D models. It has three modes:

| Mode | Input | Output | Runs on |
|---|---|---|---|
| **Relief** | 1 image | 2.5D relief, solid or mirrored volume | Your browser, no install |
| **360° from image** | 1 image | Full 3D model; the AI invents the hidden sides | Local server: TripoSR on CPU, or cloud AI via fal.ai |
| **Photo scan** | 20–150 photos or a video | Full 3D model, accurate on every side (photogrammetry) | Local server: COLMAP + OpenMVS on CPU |

One image can't show the back of an object. **360° from image** gives a plausible full model quickly. **Photo scan** measures every side, like a real 3D scanner.

## Quick start on Windows

1. Install [Python 3.11](https://www.python.org/downloads/) and tick **Add python.exe to PATH**.
2. Open PowerShell in `depthforge\server` and run:
   ```powershell
   powershell -ExecutionPolicy Bypass -File setup.ps1            # server + photogrammetry tools
   powershell -ExecutionPolicy Bypass -File setup.ps1 -LocalAI   # optional: add the local TripoSR engine (~3 GB)
   ```
   Everything installs inside `depthforge\server`: `.venv\` holds the Python packages, and `tools\` holds COLMAP, OpenMVS and TripoSR.
3. Double-click `start.bat`. Your browser opens at http://127.0.0.1:8765.

For Linux or macOS: install COLMAP and OpenMVS with your package manager or from their releases. Then run `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` and `./start.sh`.

## The engines

### 360° from image
- **TripoSR (local CPU)**: free and offline after the first run, which downloads ~1.7 GB of weights. Takes about 2–6 minutes per model on a laptop CPU. Colour is stored per vertex, and detail is moderate.
- **Cloud engines via [fal.ai](https://fal.ai)**: put your key in `server\config.json` as `"fal_key": "..."`, or set the `FAL_KEY` environment variable. Each run costs a few cents and takes 20–90 s. The image is sent to fal.ai.
  - **Hunyuan3D-2**: the most detailed geometry and textures.
  - **TRELLIS**: clean shapes and good textures.
  - **TripoSR**: the fastest.

For the best results, use one whole object on a plain background (see the image prompt tips below).

### Photo scan
COLMAP works out where each photo was taken. OpenMVS then builds a dense point cloud, a mesh and a texture. Everything runs on the CPU:

| Preset | Photos used | Time for ~50 photos on a 4-core laptop |
|---|---|---|
| Draft | up to 40, 1200 px | 10–20 min |
| Standard | up to 80, 1600 px | 30–60 min |
| High | up to 150, 2400 px, plus mesh refinement | 1–3 h |

Shooting tips:
- Take a photo every 10–15°, in 2–3 rings at different heights.
- Every part of the object should appear in 3 or more photos.
- Use soft, even light.
- Matte, textured objects work best.
- A walk-around video also works; the sharpest frames are picked automatically.

Jobs run one at a time in the background. The page shows the current stage, progress and the tool log. You can close the tab and come back, and finished models stay under **Recent models**.

## A note on older NVIDIA GPUs

Kepler cards such as the Quadro K5100M are no longer supported by current CUDA or PyTorch releases. DepthForge therefore uses the CPU builds of COLMAP and OpenMVS, and runs TripoSR on the CPU. 16 GB of RAM is enough for the Draft and Standard presets.

## Configuration

`server\config.json` (created from `config.example.json`) or environment variables:

| Setting | Env var | Default |
|---|---|---|
| `host` / `port` | `DEPTHFORGE_HOST` / `DEPTHFORGE_PORT` | `127.0.0.1` / `8765` |
| `fal_key` | `FAL_KEY` | none (cloud engines off) |
| `colmap` | `COLMAP_PATH` | searches `tools\colmap`, then PATH |
| `openmvs` | `OPENMVS_PATH` | searches `tools\openmvs`, then PATH |
| `threads` | `DEPTHFORGE_THREADS` | CPU cores − 1 |
| `data_dir` | `DEPTHFORGE_DATA_DIR` | `server\data` (job files and models) |

Keep `host` at `127.0.0.1` unless you want other devices on your network to use the server. It has no login.

## Relief mode only (no install)

Relief mode also works from any static web server, e.g. `python -m http.server -d depthforge`. The other two modes need `start.bat`.

## Troubleshooting

- **"Only N of M photos could be placed"**: the photos don't overlap enough. Add more photos from in-between angles, and avoid plain backgrounds.
- **Photo scan fails partway**: open **Log** in the job card. The last lines show which tool failed and why.
- **A model is lying on its side**: use the ⟲ X / Y / Z buttons above the viewer before exporting.
- **Relief mode's AI depth doesn't load**: the first run downloads the model from Hugging Face, so check your internet connection. Quick mode works offline.

## Tests

```sh
cd depthforge/server
python -m pytest tests -q
DEPTHFORGE_TEST_PHOTOS=path/to/photos python -m pytest tests -q -k photogrammetry   # full pipeline, needs COLMAP + OpenMVS
```
