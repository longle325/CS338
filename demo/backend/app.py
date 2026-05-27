from __future__ import annotations

import os
import json
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .core import (
    OUTPUT_ROOT,
    ROOT_DIR,
    CompareSettings,
    example_items,
    normalize_object_class,
    object_classes,
    read_json,
    run_compare,
    utc_now,
    write_json,
)

API_PREFIX = os.environ.get("OMNITRY_BACKEND_API_PREFIX", "/api/v1").rstrip("/")
MAX_UPLOAD_MB = int(os.environ.get("OMNITRY_BACKEND_MAX_UPLOAD_MB", "30"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("OMNITRY_BACKEND_WORKERS", "1")))

app = FastAPI(
    title="OmniTry Live Inference Backend",
    version="0.1.0",
    description="Backend API for comparing pretrained OmniTry with pretrained + geometry affordance inference.",
)

cors_raw = os.environ.get("OMNITRY_BACKEND_CORS_ORIGINS", "*")
cors_origins = [item.strip() for item in cors_raw.split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials=False if cors_origins == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=str(OUTPUT_ROOT)), name="artifacts")
app.mount("/demo-examples", StaticFiles(directory=str(ROOT_DIR / "demo_example")), name="demo_examples")


def status_path(run_id: str) -> Path:
    return OUTPUT_ROOT / run_id / "status.json"


def run_dir(run_id: str) -> Path:
    return OUTPUT_ROOT / run_id


def load_status(run_id: str) -> dict:
    path = status_path(run_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return read_json(path)


def update_status(run_id: str, **updates) -> dict:
    path = status_path(run_id)
    current = read_json(path) if path.is_file() else {"run_id": run_id}
    current.update(updates)
    current["updated_at"] = utc_now()
    write_json(path, current)
    return current


async def save_upload(upload: UploadFile, output_path: Path) -> dict:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{upload.filename or 'upload'} is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload is larger than {MAX_UPLOAD_MB}MB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_suffix(".upload")
    raw_path.write_bytes(data)
    try:
        with Image.open(raw_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.save(output_path, quality=95)
            width, height = image.size
    except UnidentifiedImageError as exc:
        raw_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {upload.filename}") from exc
    raw_path.unlink(missing_ok=True)
    return {"filename": upload.filename, "path": str(output_path), "width": width, "height": height}


def submit_job(run_id: str, settings: CompareSettings) -> None:
    def callback(stage: str, message: str) -> None:
        update_status(run_id, status="running", stage=stage, message=message)

    def worker() -> None:
        try:
            update_status(run_id, status="running", stage="queued", message="Job picked up by backend worker")
            result = run_compare(run_id=run_id, run_dir=run_dir(run_id), settings=settings, update=callback)
            update_status(
                run_id,
                status="complete",
                stage="complete",
                message="Compare run complete",
                result=result,
                result_url=f"/artifacts/{run_id}/result.json",
            )
        except Exception as exc:  # noqa: BLE001 - API should return the captured failure.
            update_status(
                run_id,
                status="failed",
                stage="failed",
                message=str(exc),
                traceback=traceback.format_exc(),
            )

    EXECUTOR.submit(worker)


@app.get(f"{API_PREFIX}/health")
def health() -> dict:
    return {
        "status": "ok",
        "time": utc_now(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "output_root": str(OUTPUT_ROOT),
        "object_classes": object_classes(),
    }


@app.get(f"{API_PREFIX}/classes")
def classes() -> dict:
    return {
        "classes": object_classes(),
        "aliases": {
            "eyeglasses": "glasses",
            "earring": "earrings",
            "shoes": "shoe",
            "bowtie": "bow tie",
        },
    }


@app.get(f"{API_PREFIX}/examples")
def examples() -> dict:
    return {"items": example_items()}


@app.get(f"{API_PREFIX}/runs")
def list_runs(limit: int = 30) -> dict:
    items = []
    for path in sorted(OUTPUT_ROOT.glob("*/status.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            items.append(read_json(path))
        except json.JSONDecodeError:
            continue
        if len(items) >= limit:
            break
    return {"items": items}


@app.post(f"{API_PREFIX}/runs/compare")
async def create_compare_run(
    person_image: UploadFile = File(...),
    object_image: UploadFile = File(...),
    object_class: str = Form(...),
    optional_prompt: str = Form(""),
    steps: int = Form(20),
    guidance_scale: float = Form(30.0),
    seed: int = Form(-1),
    geometry_candidate_count: int = Form(2),
    run_pretrained: bool = Form(True),
    run_geometry: bool = Form(True),
) -> dict:
    try:
        normalized_class = normalize_object_class(object_class)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not run_pretrained and not run_geometry:
        raise HTTPException(status_code=400, detail="At least one branch must be enabled.")

    run_id = uuid.uuid4().hex[:12]
    directory = run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=False)

    person_meta = await save_upload(person_image, directory / "person.jpg")
    object_meta = await save_upload(object_image, directory / "object.jpg")

    settings = CompareSettings(
        object_class=normalized_class,
        optional_prompt=optional_prompt,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        geometry_candidate_count=geometry_candidate_count,
        run_pretrained=run_pretrained,
        run_geometry=run_geometry,
    )
    status = {
        "run_id": run_id,
        "status": "queued",
        "stage": "queued",
        "message": "Compare job queued",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "settings": settings.__dict__,
        "inputs": {
            "person": person_meta,
            "object": object_meta,
            "person_url": f"/artifacts/{run_id}/person.jpg",
            "object_url": f"/artifacts/{run_id}/object.jpg",
        },
        "status_url": f"{API_PREFIX}/runs/{run_id}",
        "artifact_base_url": f"/artifacts/{run_id}",
    }
    write_json(status_path(run_id), status)
    submit_job(run_id, settings)
    return status


@app.get(f"{API_PREFIX}/runs/{{run_id}}")
def get_run(run_id: str) -> dict:
    return load_status(run_id)


@app.get(f"{API_PREFIX}/runs/{{run_id}}/result")
def get_result(run_id: str) -> dict:
    path = run_dir(run_id) / "result.json"
    if not path.is_file():
        status = load_status(run_id)
        raise HTTPException(status_code=409, detail={"message": "Result is not ready", "status": status})
    return read_json(path)


@app.get(f"{API_PREFIX}/runs/{{run_id}}/diagnostics/{{branch}}")
def get_diagnostics(run_id: str, branch: str) -> FileResponse:
    if branch not in {"pretrained", "geometry"}:
        raise HTTPException(status_code=404, detail="branch must be pretrained or geometry")
    path = run_dir(run_id) / f"{branch}_diagnostics.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="diagnostics not found")
    return FileResponse(path, media_type="text/markdown")
