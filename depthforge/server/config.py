"""Runtime settings, read from environment variables and an optional config.json.

Environment variables win over config.json. Paths may point at a tool's folder
(it is searched recursively) or directly at the executable.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
APP_DIR = SERVER_DIR.parent  # depthforge/ (the static front end)


def _load_file() -> dict:
    path = Path(os.environ.get("DEPTHFORGE_CONFIG", SERVER_DIR / "config.json"))
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    data_dir: Path = SERVER_DIR / "data"
    tools_dir: Path = SERVER_DIR / "tools"
    colmap: str = ""        # path to colmap(.exe) or its folder; "" = search tools_dir and PATH
    openmvs: str = ""       # folder containing DensifyPointCloud(.exe) etc.; "" = search tools_dir and PATH
    triposr_dir: str = ""   # clone of github.com/VAST-AI-Research/TripoSR; "" = tools_dir/TripoSR
    fal_key: str = ""
    threads: int = max(1, (os.cpu_count() or 2) - 1)
    max_upload_mb: int = 1024
    fal_models: dict = field(default_factory=dict)  # overrides for engine id -> fal app id

    @classmethod
    def load(cls) -> "Settings":
        raw = _load_file()
        env = {
            "host": os.environ.get("DEPTHFORGE_HOST"),
            "port": os.environ.get("DEPTHFORGE_PORT"),
            "data_dir": os.environ.get("DEPTHFORGE_DATA_DIR"),
            "tools_dir": os.environ.get("DEPTHFORGE_TOOLS_DIR"),
            "colmap": os.environ.get("COLMAP_PATH"),
            "openmvs": os.environ.get("OPENMVS_PATH"),
            "triposr_dir": os.environ.get("TRIPOSR_DIR"),
            "fal_key": os.environ.get("FAL_KEY"),
            "threads": os.environ.get("DEPTHFORGE_THREADS"),
        }
        merged = {**raw, **{k: v for k, v in env.items() if v}}
        s = cls()
        for k, v in merged.items():
            if not hasattr(s, k):
                continue
            cur = getattr(s, k)
            if isinstance(cur, Path):
                v = Path(v).expanduser()
                if not v.is_absolute():
                    v = SERVER_DIR / v
            elif isinstance(cur, bool):
                v = bool(v)
            elif isinstance(cur, int):
                v = int(v)
            setattr(s, k, v)
        if s.threads <= 0:
            s.threads = max(1, (os.cpu_count() or 2) - 1)
        s.data_dir.mkdir(parents=True, exist_ok=True)
        return s


settings = Settings.load()
