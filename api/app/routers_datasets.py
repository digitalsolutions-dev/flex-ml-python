import re, uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from .models import RegisterDatasetRequest, ProfileResponse, SuggestResponse, RunRequest, TaskStatus
from .db import call_proc
from .tasks import profile_dataset_task, run_pipeline_task
from .s3 import upload_stream
from flexml.io_s3 import head_object
from flexml.io_df import sample_for_suggest
from flexml.suggest import suggest_from_sample

router = APIRouter(prefix="/datasets", tags=["datasets"])


# ---------- New one-shot ingest endpoint ----------
@router.post("/ingest")
def ingest_file(
        file: UploadFile = File(...),
        name: Optional[str] = Form(None),
        key_prefix: Optional[str] = Form("uploads/"),
        auto_profile: Optional[bool] = Form(False),
):
    """
    Single call to upload a file to S3 and register a dataset.
    Returns dataset_id (and optional profile_task_id if auto_profile=true).
    """
    # sanitize filename for S3 (keep extension)
    orig = file.filename or "upload.dat"
    base, ext = re.match(r"^(.*?)(\.[^.]*)?$", orig).groups()
    safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", (base or "file")).strip("._-") or "file"
    safe_ext = ext or ""
    key = f"{key_prefix}{uuid.uuid4().hex}_{safe_base}{safe_ext}"

    # stream to S3
    ctype = file.content_type or "application/octet-stream"
    upload_stream(file.file, key, ctype)

    # register in DB
    rows = call_proc("sp_dataset_create", (key, name or orig, ctype))
    if not rows:
        raise HTTPException(500, "Failed to create dataset record")
    dataset_id = rows[0]["id"]

    # optional background profiling
    if auto_profile:
        t = profile_dataset_task.delay(dataset_id, key, name or orig)
        return {"dataset_id": dataset_id, "s3_key": key, "profile_task_id": t.id}

    return {"dataset_id": dataset_id, "s3_key": key}


# ---------- Legacy endpoints (kept for compatibility) ----------
@router.post("/register")
def register(req: RegisterDatasetRequest):
    try:
        head_object(req.s3_key)
    except Exception as e:
        raise HTTPException(400, f"S3 object not found: {e}")
    rows = call_proc("sp_dataset_create", (req.s3_key, req.name, req.mime or "application/octet-stream"))
    dataset_id = rows[0]["id"]
    return {"dataset_id": dataset_id, "s3_key": req.s3_key}


@router.post("/{dataset_id}/profile", response_model=ProfileResponse)
def profile(dataset_id: int):
    ds = call_proc("sp_dataset_get", (dataset_id,))
    if not ds: raise HTTPException(404, "dataset not found")
    s3_key, name = ds[0]["s3_key"], ds[0]["name"]
    task = profile_dataset_task.delay(dataset_id, s3_key, name)
    return ProfileResponse(task_id=task.id)


@router.get("/{dataset_id}/suggest", response_model=SuggestResponse)
def suggest(dataset_id: int):
    ds = call_proc("sp_dataset_get", (dataset_id,))
    if not ds: raise HTTPException(404, "dataset not found")
    s3_key, name = ds[0]["s3_key"], ds[0]["name"]
    sample = sample_for_suggest(s3_key, name)
    options = suggest_from_sample(sample)
    return SuggestResponse(options=options)


@router.post("/runs", response_model=TaskStatus)
def run(req: RunRequest):
    ds = call_proc("sp_dataset_get", (req.dataset_id,))
    if not ds: raise HTTPException(404, "dataset not found")
    s3_key, name = ds[0]["s3_key"], ds[0]["name"]
    t = run_pipeline_task.delay(req.dataset_id, s3_key, name, req.config.model_dump())
    return TaskStatus(task_id=t.id, state="PENDING")
