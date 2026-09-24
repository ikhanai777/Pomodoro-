"""Drop-in replacement for `torchmcubes.marching_cubes`, built on scikit-image.

TripoSR imports torchmcubes, which has to be compiled from source (painful on Windows).
This module is only put on sys.path when the real package is missing.
Output matches torchmcubes: vertices as (x, y, z) = (dim2, dim1, dim0), faces as int64.
"""
import numpy as np
import torch
from skimage.measure import marching_cubes as _sk_marching_cubes


def marching_cubes(vol: torch.Tensor, thresh: float):
    v = vol.detach().float().cpu().numpy()
    if not (v.min() <= thresh <= v.max()):
        return torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0, 3), dtype=torch.int64)
    verts, faces, _, _ = _sk_marching_cubes(v, level=thresh)
    verts = np.ascontiguousarray(verts[:, ::-1])          # (d0, d1, d2) -> (x=d2, y=d1, z=d0)
    faces = np.ascontiguousarray(faces[:, ::-1])          # axis swap mirrors, so flip winding back
    return torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.int64))
