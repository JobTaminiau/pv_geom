"""Remote storage: telling remote from local, listing a prefix, and caching a
remote object to local disk for readers that need a file path (LAZ, OGR).

The cache is idempotent — a second call for the same URI is a local lookup.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

DEFAULT_CACHE = Path(tempfile.gettempdir()) / "pv_geom_cache"


def is_remote(uri: str | Path) -> bool:
    s = str(uri)
    return s.startswith(("s3://", "gs://", "http://", "https://"))


class RemoteFileMissing(FileNotFoundError):
    """Raised when an s3:// URI returns 404. Lets the runner skip gracefully
    rather than crashing the whole pipeline on a single missing LAZ tile."""


def list_s3_uris(prefix_uri: str) -> set[str]:
    """List all object URIs under an ``s3://bucket/prefix`` with one paginated
    LIST. Lets the runner drop tiles absent from the bucket before dispatch
    instead of paying a 404 round-trip per missing tile on every worker."""
    s = str(prefix_uri)
    if not s.startswith("s3://"):
        raise ValueError(f"expected an s3:// prefix; got {s}")
    bucket, _, key_prefix = s[len("s3://"):].partition("/")

    import boto3

    uris: set[str] = set()
    paginator = boto3.client("s3").get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            uris.add(f"s3://{bucket}/{obj['Key']}")
    return uris


def localize(uri: str | Path, cache_dir: Path | None = None) -> Path:
    """Return a local Path for ``uri``. Downloads from S3 once if needed.

    Raises ``RemoteFileMissing`` for HTTP 404 — the caller decides whether
    to skip or fail. Other transport errors propagate as the underlying
    ``botocore`` exception.
    """
    s = str(uri)
    if not is_remote(s):
        return Path(s)
    if not s.startswith("s3://"):
        raise NotImplementedError(f"only s3:// remote URIs are supported; got {s}")

    cache_dir = cache_dir or DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    bucket, _, key = s[len("s3://"):].partition("/")
    fname = key.replace("/", "__")
    local = cache_dir / fname
    if not local.exists() or local.stat().st_size == 0:
        import boto3
        from botocore.exceptions import ClientError
        try:
            boto3.client("s3").download_file(bucket, key, str(local))
        except ClientError as exc:
            err = exc.response.get("Error", {})
            if err.get("Code") in ("404", "NoSuchKey") or "404" in str(exc):
                local.unlink(missing_ok=True)
                raise RemoteFileMissing(f"s3://{bucket}/{key} not found") from exc
            raise
    return local
