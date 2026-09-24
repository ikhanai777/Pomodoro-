"""A small background job queue. Heavy jobs run one at a time so they don't fight over CPU and RAM."""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from tools import Cancelled


@dataclass
class Job:
    id: str
    kind: str                 # "image" | "photos"
    engine: str
    dir: Path
    params: dict = field(default_factory=dict)
    status: str = "queued"    # queued | running | done | error | cancelled
    stage: str = "Waiting in queue"
    progress: float = 0.0     # 0..1 overall
    message: str = ""
    error: str = ""
    result: Optional[str] = None  # file name inside dir
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    logs: deque = field(default_factory=lambda: deque(maxlen=400))
    cancel: threading.Event = field(default_factory=threading.Event)

    def log(self, line: str) -> None:
        self.logs.append(line)

    def set(self, stage: Optional[str] = None, progress: Optional[float] = None, message: Optional[str] = None) -> None:
        if stage is not None:
            self.stage = stage
            self.log(f"== {stage}")
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        if message is not None:
            self.message = message

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    def public(self, log_lines: int = 40) -> dict:
        now = time.time()
        return {
            "id": self.id, "kind": self.kind, "engine": self.engine, "status": self.status,
            "stage": self.stage, "progress": round(self.progress, 4), "message": self.message,
            "error": self.error, "created": self.created,
            "elapsed": round(((self.finished or now) - (self.started or now)), 1) if self.started else 0,
            "result_url": f"api/jobs/{self.id}/model.glb" if self.result else None,
            "log": list(self.logs)[-log_lines:],
        }

    def save(self) -> None:
        data = self.public(log_lines=400)
        data.update(params=self.params, result=self.result, finished=self.finished, started=self.started)
        (self.dir / "job.json").write_text(json.dumps(data, indent=1), encoding="utf-8")


class JobManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir / "jobs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.q: "queue.Queue[tuple[Job, Callable[[Job], Path]]]" = queue.Queue()
        self._load_history()
        threading.Thread(target=self._worker, daemon=True, name="depthforge-worker").start()

    def _load_history(self) -> None:
        for f in sorted(self.data_dir.glob("*/job.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                status = d["status"] if d["status"] in ("done", "error", "cancelled") else "error"
                job = Job(id=d["id"], kind=d["kind"], engine=d["engine"], dir=f.parent, params=d.get("params", {}),
                          status=status, stage=d.get("stage", ""), progress=d.get("progress", 0),
                          message=d.get("message", ""), error=d.get("error") or ("Interrupted by a server restart" if status != d["status"] else ""),
                          result=d.get("result"), created=d.get("created", 0), started=d.get("started"), finished=d.get("finished"))
                job.logs.extend(d.get("log", []))
                self.jobs[job.id] = job
            except (OSError, ValueError, KeyError):
                continue

    def create(self, kind: str, engine: str, params: dict) -> Job:
        jid = uuid.uuid4().hex[:16]
        d = self.data_dir / jid
        (d / "input").mkdir(parents=True)
        job = Job(id=jid, kind=kind, engine=engine, dir=d, params=params)
        self.jobs[jid] = job
        return job

    def submit(self, job: Job, fn: Callable[[Job], Path]) -> None:
        position = sum(1 for j in self.jobs.values() if j.status in ("queued", "running"))
        job.message = "Starting…" if position <= 1 else f"{position - 1} job(s) ahead of this one"
        job.save()
        self.q.put((job, fn))

    def get(self, jid: str) -> Optional[Job]:
        return self.jobs.get(jid)

    def cancel(self, job: Job) -> None:
        job.cancel.set()
        if job.status == "queued":
            job.status, job.stage, job.finished = "cancelled", "Cancelled", time.time()
            job.save()

    def _worker(self) -> None:
        while True:
            job, fn = self.q.get()
            if job.cancel.is_set():
                continue
            job.status, job.started = "running", time.time()
            try:
                out = fn(job)
                job.result = Path(out).name
                job.status, job.stage, job.progress = "done", "Done", 1.0
                job.message = f"Finished in {time.time() - job.started:.0f} s"
            except Cancelled:
                job.status, job.stage, job.message = "cancelled", "Cancelled", ""
            except Exception as e:  # report every failure to the UI rather than dying
                job.status, job.error = "error", str(e) or e.__class__.__name__
                job.log(traceback.format_exc())
            finally:
                job.finished = time.time()
                try:
                    job.save()
                except OSError:
                    pass
