import os, uuid, boto3
from typing import Dict, Any
from boto3.s3.transfer import TransferConfig

_s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "eu-central-1"))
BUCKET = os.environ["S3_BUCKET"]


# Tuning for multi-part uploads; adjust if needed
def _transfer_config() -> TransferConfig:
    chunk_mb = int(os.environ.get("S3_MULTIPART_CHUNK_MB", "8"))
    concurr = int(os.environ.get("S3_MAX_UPLOAD_CONCURRENCY", "4"))
    return TransferConfig(
        multipart_threshold=chunk_mb * 1024 * 1024,
        multipart_chunksize=chunk_mb * 1024 * 1024,
        max_concurrency=concurr,
        use_threads=True,
    )


def presign_post(filename: str, content_type: str, key_prefix: str = "uploads/") -> Dict[str, Any]:
    key = f"{key_prefix}{uuid.uuid4().hex}_{filename}"
    resp = _s3.generate_presigned_post(
        Bucket=BUCKET,
        Key=key,
        Fields={"Content-Type": content_type},
        Conditions=[{"Content-Type": content_type}],
        ExpiresIn=3600
    )
    resp["key"] = key
    return resp


def upload_stream(fileobj, key: str, content_type: str = "application/octet-stream") -> Dict[str, Any]:
    """
    Server-side streaming upload to S3 from a file-like object.
    Uses multi-part upload for large files without loading into memory.
    """
    fileobj.seek(0)
    _s3.upload_fileobj(
        Fileobj=fileobj,
        Bucket=BUCKET,
        Key=key,
        ExtraArgs={"ContentType": content_type},
        Config=_transfer_config(),
    )
    return {"bucket": BUCKET, "key": key}
