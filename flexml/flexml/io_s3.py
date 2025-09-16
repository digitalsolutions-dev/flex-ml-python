import os, boto3

BUCKET = os.environ.get("S3_BUCKET")
REGION = os.environ.get("AWS_REGION", "eu-central-1")
_s3 = boto3.client("s3", region_name=REGION)


def get_bytes(key: str) -> bytes:
    return _s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def put_bytes(key: str, data: bytes, content_type="application/octet-stream"):
    _s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=content_type)
    return {"bucket": BUCKET, "key": key}


def head_object(key: str):
    return _s3.head_object(Bucket=BUCKET, Key=key)
