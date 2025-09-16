from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class PresignRequest(BaseModel):
    filename: str
    content_type: Optional[str] = "application/octet-stream"
    key_prefix: Optional[str] = "uploads/"


class PresignResponse(BaseModel):
    url: str
    fields: Dict[str, Any]
    key: str


class RegisterDatasetRequest(BaseModel):
    name: str
    s3_key: str
    mime: Optional[str] = None


class ProfileResponse(BaseModel):
    task_id: str


class SuggestResponse(BaseModel):
    options: List[Dict[str, Any]]


class RunConfig(BaseModel):
    type: str
    # forecasting params
    ts_col: Optional[str] = None
    target: Optional[str] = None
    horizon: Optional[int] = None
    exog_cols: Optional[List[str]] = None
    step: Optional[int] = None
    initial_train: Optional[int] = None
    model_name: Optional[str] = None
    # clustering params
    k: Optional[int] = None
    # classification params
    mode: Optional[str] = None
    clf_model_key: Optional[str] = None
    return_file: Optional[bool] = False


class RunRequest(BaseModel):
    dataset_id: int
    config: RunConfig


class TaskStatus(BaseModel):
    task_id: str
    state: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
