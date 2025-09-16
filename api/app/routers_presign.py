from fastapi import APIRouter
from .s3 import presign_post
from .models import PresignRequest, PresignResponse

router = APIRouter(prefix="/presign-upload", tags=["upload"])


@router.post("", response_model=PresignResponse)
def presign(req: PresignRequest):
    p = presign_post(req.filename, req.content_type or "application/octet-stream", req.key_prefix or "uploads/")
    return PresignResponse(url=p["url"], fields=p["fields"], key=p["key"])
