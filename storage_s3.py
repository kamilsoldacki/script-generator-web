import os
import json
from typing import Any

import boto3


def _s3_client():
    # Works with AWS S3, Cloudflare R2, Backblaze B2 (S3-compatible).
    endpoint_url = (os.environ.get("S3_ENDPOINT_URL") or "").strip() or None
    region_name = (os.environ.get("S3_REGION") or "").strip() or None
    access_key_id = (os.environ.get("S3_ACCESS_KEY_ID") or "").strip()
    secret_access_key = (os.environ.get("S3_SECRET_ACCESS_KEY") or "").strip()

    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=access_key_id or None,
        aws_secret_access_key=secret_access_key or None,
    )


def _bucket() -> str:
    b = (os.environ.get("S3_BUCKET") or "").strip()
    if not b:
        raise RuntimeError("Missing S3_BUCKET env var.")
    return b


def put_text(*, key: str, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
    s3 = _s3_client()
    s3.put_object(
        Bucket=_bucket(),
        Key=key,
        Body=text.encode("utf-8"),
        ContentType=content_type,
    )


def put_json(*, key: str, data: dict[str, Any]) -> None:
    put_text(key=key, text=json.dumps(data, ensure_ascii=False, indent=2), content_type="application/json; charset=utf-8")


def presign_get(*, key: str, expires_seconds: int = 60 * 60 * 24) -> str:
    s3 = _s3_client()
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": _bucket(), "Key": key},
        ExpiresIn=int(expires_seconds),
    )

