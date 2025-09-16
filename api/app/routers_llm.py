from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from .db import call_proc
from .llm.orchestrator import run_with_tools

router = APIRouter(prefix="/llm", tags=["llm"])


class AskRequest(BaseModel):
    dataset_id: int
    question: str


class RunRequest(BaseModel):
    dataset_id: int
    goal: str
    allow_ml: bool = True


@router.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    ds = call_proc("sp_dataset_get", (req.dataset_id,))
    if not ds:
        raise HTTPException(404, "dataset not found")
    key = ds[0]["s3_key"]
    result = run_with_tools(user_content=req.question, dataset_key=key, allow_ml=False)
    return result


@router.post("/run")
def run(req: RunRequest) -> Dict[str, Any]:
    ds = call_proc("sp_dataset_get", (req.dataset_id,))
    if not ds:
        raise HTTPException(404, "dataset not found")
    key = ds[0]["s3_key"]
    result = run_with_tools(user_content=req.goal, dataset_key=key, allow_ml=req.allow_ml)
    return result
