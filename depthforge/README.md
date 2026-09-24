# DepthForge — any image → 3D model

A single-file web app that turns a photo into a textured, exportable 3D mesh. Everything runs in the browser; the image never leaves the device.

## How it works

1. **Depth estimation**: [Depth Anything V2 Small](https://huggingface.co/onnx-community/depth-anything-v2-small) runs on-device through Transformers.js. It uses WebGPU when the browser has it and falls back to WASM. The model is about 27 MB, downloaded once and then cached by the browser. If the model can't be loaded (for example, when offline), a **Quick** heuristic is used instead. It combines colour contrast against the frame border, shading and a centre prior.
2. **Meshing**: the depth map is resampled onto a grid of up to 512² vertices, smoothed, and passed through a contrast curve. It is then displaced into a surface:
   - **Background cut** drops vertices below a depth threshold, which isolates the subject.
   - **Edge tearing** removes the stretched "curtain" triangles at depth discontinuities.
3. **Volume**:
   - **Relief** is a single surface.
   - **Solid** adds a flat back and side walls, which makes a watertight, manifold mesh ready for 3D printing.
   - **Full volume** mirrors the front onto the back for rounded objects.
4. **Rendering**: Three.js with PBR material, image-based lighting, ACES tone mapping and soft shadows. The view modes are Photo, Clay, Depth and Wire.
5. **Export**:
   - **GLB**: textured, for Blender, Unity, web or AR viewers.
   - **STL**: 100 mm on the longest side, for slicers.
   - **OBJ**: geometry and UVs.

## Run

Serve the folder over HTTP. ES modules and the model download need a real origin, not `file://`:

```sh
npx serve depthforge   # or: python3 -m http.server -d depthforge
```

Then drop, paste or pick an image, or try one of the built-in samples.

## Tips

- Single objects on plain backgrounds give the cleanest models. Raise **Background cut** until the backdrop disappears.
- For portraits, use a depth of about 0.4–0.6 and smoothing of 2–3.
- If near and far look swapped (e.g. with some drawings), tick **Invert depth**.
