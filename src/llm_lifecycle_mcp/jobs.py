"""One-at-a-time async job registry for long, GPU-heavy operations.

Guardrails are built in: only one job runs at a time, and a cooldown spaces runs
apart so nothing hammers the hardware back-to-back. Jobs run as async subprocesses,
so the server stays responsive while a benchmark or quantize is in flight.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"  # running | done | failed
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    returncode: Optional[int] = None
    result: Optional[dict] = None
    output_tail: str = ""


class JobManager:
    def __init__(self, cooldown_s: float = 30.0) -> None:
        self.jobs: dict[str, Job] = {}
        self.current: Optional[str] = None
        self.last_finished: float = 0.0
        self.cooldown_s = cooldown_s
        self._lock = asyncio.Lock()

    async def submit(
        self,
        kind: str,
        cmd: list[str],
        cwd: Optional[str] = None,
        parse: Optional[Callable[[str], dict]] = None,
    ) -> str:
        async with self._lock:
            if self.current is not None:
                raise RuntimeError(f"A job is already running ({self.current}). One at a time.")
            remaining = self.cooldown_s - (time.time() - self.last_finished)
            if remaining > 0:
                raise RuntimeError(f"Cooldown active — wait {remaining:.0f}s before the next job.")
            job = Job(id=uuid.uuid4().hex[:8], kind=kind)
            self.jobs[job.id] = job
            self.current = job.id
        asyncio.create_task(self._run(job, cmd, cwd, parse))
        return job.id

    async def _run(self, job: Job, cmd, cwd, parse) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
            text = out.decode("utf-8", "ignore")
            job.output_tail = text[-4000:]
            job.returncode = proc.returncode
            job.status = "done" if proc.returncode == 0 else "failed"
            if parse and job.status == "done":
                try:
                    job.result = parse(text)
                except Exception as e:  # noqa: BLE001
                    job.result = {"parse_error": str(e)}
        except Exception as e:  # noqa: BLE001
            job.status = "failed"
            job.output_tail += f"\n{type(e).__name__}: {e}"
        finally:
            job.finished = time.time()
            self.last_finished = time.time()
            self.current = None

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)


jobs = JobManager()
