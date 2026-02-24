"""
FastAPI backend for the Concept Attention Explorer web UI.

Architecture:
  - POST /api/analyze     → start a job (returns job_id)
  - GET  /api/job/{id}    → poll job status + results
  - WS   /ws/{id}         → stream per-step heatmaps in real time
  - Static files at /     → serve index.html + assets

Usage (Colab):
    from web.server import create_app, run_server
    run_server(pipeline, port=8000)
"""
import asyncio
import base64
import io
import json
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    image_b64: str          # base64-encoded PNG/JPEG
    prompt: str
    concepts: list[str]
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 20
    image2image_strength: float = 0.8
    guidance: float = 4.0
    seed: int = 0
    softmax_temperature: float = 1000.0
    cmap: str = "plasma"


class JobStatus(BaseModel):
    job_id: str
    status: str             # "pending" | "running" | "done" | "error"
    progress: int = 0       # 0-100
    error: str | None = None
    # Populated when done:
    original_image_b64: str | None = None
    generated_image_b64: str | None = None
    original_heatmaps_b64: list[str] | None = None
    concepts: list[str] | None = None
    num_steps: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# In-memory job store
# ─────────────────────────────────────────────────────────────────────────────

class Job:
    def __init__(self, job_id: str, request: AnalyzeRequest):
        self.job_id = job_id
        self.request = request
        self.status = "pending"
        self.progress = 0
        self.error: str | None = None
        self.total_steps = 0
        self.current_step = 0
        # Final results
        self.original_image_b64: str | None = None
        self.generated_image_b64: str | None = None
        self.original_heatmaps_b64: list[str] | None = None
        self.concepts: list[str] | None = None
        # Per-step heatmaps for replay (list of list[str], [step][concept])
        self.temporal_heatmaps_b64: list[list[str]] = []
        # Active WebSocket connections waiting on this job
        self._ws_queue: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._ws_queue = asyncio.Queue()

    def _send_ws(self, msg: dict):
        """Thread-safe: put a message into the WebSocket queue."""
        if self._loop and self._ws_queue:
            self._loop.call_soon_threadsafe(self._ws_queue.put_nowait, msg)

    def on_step(self, step_idx: int, total_steps: int, b64_heatmaps: list[str]):
        self.current_step = step_idx + 1
        self.total_steps = total_steps
        self.progress = int(100 * self.current_step / max(total_steps, 1))
        self.temporal_heatmaps_b64.append(b64_heatmaps)
        self._send_ws({
            "type": "step",
            "step": step_idx,
            "total_steps": total_steps,
            "progress": self.progress,
            "heatmaps": b64_heatmaps,
        })

    def finish(self, orig_b64, gen_b64, orig_hm_b64, concepts):
        self.status = "done"
        self.progress = 100
        self.original_image_b64 = orig_b64
        self.generated_image_b64 = gen_b64
        self.original_heatmaps_b64 = orig_hm_b64
        self.concepts = concepts
        self._send_ws({
            "type": "done",
            "original_image": orig_b64,
            "generated_image": gen_b64,
            "original_heatmaps": orig_hm_b64,
            "concepts": concepts,
            "num_steps": len(self.temporal_heatmaps_b64),
        })

    def fail(self, error: str):
        self.status = "error"
        self.error = error
        self._send_ws({"type": "error", "error": error})


# Global job store
_jobs: dict[str, Job] = {}
_pipeline = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pil_to_b64(img: PIL.Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _b64_to_pil(b64: str) -> PIL.Image.Image:
    return PIL.Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner (executes in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(job: Job):
    global _pipeline
    try:
        req = job.request
        job.status = "running"

        original_image = _b64_to_pil(req.image_b64)

        temporal_output = _pipeline.compare_images(
            original_image=original_image,
            prompt=req.prompt,
            concepts=req.concepts,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            image2image_strength=req.image2image_strength,
            guidance=req.guidance,
            seed=req.seed,
            softmax_temperature=req.softmax_temperature,
            cmap=req.cmap,
            on_step_callback=job.on_step,
        )

        orig_b64 = _pil_to_b64(temporal_output.original_image)
        gen_b64 = _pil_to_b64(temporal_output.generated_image)
        orig_hm_b64 = [_pil_to_b64(hm) for hm in temporal_output.original_heatmaps]

        job.finish(orig_b64, gen_b64, orig_hm_b64, req.concepts)

    except Exception as e:
        job.fail(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app(pipeline) -> FastAPI:
    global _pipeline
    _pipeline = pipeline

    app = FastAPI(title="Concept Attention Explorer")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── REST endpoints ─────────────────────────────────────────────────────

    @app.post("/api/analyze")
    async def start_analyze(req: AnalyzeRequest):
        job_id = str(uuid.uuid4())
        job = Job(job_id, req)
        loop = asyncio.get_event_loop()
        job.set_loop(loop)
        _jobs[job_id] = job

        thread = threading.Thread(target=_run_pipeline, args=(job,), daemon=True)
        thread.start()

        return {"job_id": job_id}

    @app.get("/api/job/{job_id}")
    async def get_job(job_id: str):
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobStatus(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            error=job.error,
            original_image_b64=job.original_image_b64,
            generated_image_b64=job.generated_image_b64,
            original_heatmaps_b64=job.original_heatmaps_b64,
            concepts=job.concepts,
            num_steps=len(job.temporal_heatmaps_b64),
        )

    @app.get("/api/job/{job_id}/temporal")
    async def get_temporal(job_id: str):
        """Return all per-step heatmaps for replay (only available when done)."""
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "done":
            raise HTTPException(status_code=202, detail="Job not yet complete")
        return {
            "temporal_heatmaps": job.temporal_heatmaps_b64,
            "concepts": job.concepts,
            "num_steps": len(job.temporal_heatmaps_b64),
        }

    # ── WebSocket for real-time streaming ──────────────────────────────────

    @app.websocket("/ws/{job_id}")
    async def websocket_stream(websocket: WebSocket, job_id: str):
        await websocket.accept()
        job = _jobs.get(job_id)
        if job is None:
            await websocket.send_json({"type": "error", "error": "Job not found"})
            await websocket.close()
            return

        if job._ws_queue is None:
            await websocket.send_json({"type": "error", "error": "Job queue not initialized"})
            await websocket.close()
            return

        try:
            # Replay any already-completed steps (handles late WebSocket connection)
            for step_idx, b64_heatmaps in enumerate(job.temporal_heatmaps_b64):
                await websocket.send_json({
                    "type": "step",
                    "step": step_idx,
                    "total_steps": job.total_steps,
                    "progress": int(100 * (step_idx + 1) / max(job.total_steps, 1)),
                    "heatmaps": b64_heatmaps,
                })

            if job.status == "done":
                await websocket.send_json({
                    "type": "done",
                    "original_image": job.original_image_b64,
                    "generated_image": job.generated_image_b64,
                    "original_heatmaps": job.original_heatmaps_b64,
                    "concepts": job.concepts,
                    "num_steps": len(job.temporal_heatmaps_b64),
                })
                return
            elif job.status == "error":
                await websocket.send_json({"type": "error", "error": job.error})
                return

            # Stream new messages from the queue
            while True:
                try:
                    msg = await asyncio.wait_for(job._ws_queue.get(), timeout=120.0)
                    await websocket.send_json(msg)
                    if msg.get("type") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    # Keep connection alive with a ping
                    await websocket.send_json({"type": "ping"})

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({"type": "error", "error": str(e)})
            except Exception:
                pass

    # ── Static files (frontend) ────────────────────────────────────────────

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


def run_server(pipeline, host: str = "0.0.0.0", port: int = 8000, **kwargs):
    """Start the uvicorn server. Call from Colab or local."""
    app = create_app(pipeline)
    uvicorn.run(app, host=host, port=port, **kwargs)
