"""Diagnostic: run boto3 calls on a Coiled worker to confirm what creds and
permissions it actually has against the asu-nsf-phoenix bucket. Single
worker, ~2 minutes total.
"""

from __future__ import annotations

from typing import Any


def _probe() -> dict[str, Any]:
    """Runs ON the worker. Returns a dict of probe results."""
    import socket

    import boto3
    from botocore.exceptions import ClientError

    out: dict[str, Any] = {"hostname": socket.gethostname()}

    sts = boto3.client("sts")
    try:
        ident = sts.get_caller_identity()
        out["caller_identity"] = {
            "Account": ident.get("Account"),
            "Arn": ident.get("Arn"),
            "UserId": ident.get("UserId"),
        }
    except ClientError as e:
        out["caller_identity_error"] = str(e)

    s3 = boto3.client("s3")

    try:
        s3.head_bucket(Bucket="asu-nsf-phoenix")
        out["head_bucket"] = "OK"
    except ClientError as e:
        out["head_bucket"] = f"{e.response['ResponseMetadata']['HTTPStatusCode']} {e.response['Error'].get('Code')}"

    try:
        loc = s3.get_bucket_location(Bucket="asu-nsf-phoenix")
        out["bucket_location"] = loc.get("LocationConstraint") or "us-east-1"
    except ClientError as e:
        out["bucket_location_error"] = str(e)

    key = "data/lidar_data/USGS_LPC_AZ_MaricopaPinal_2020_B20_w0432n3719.laz"
    try:
        h = s3.head_object(Bucket="asu-nsf-phoenix", Key=key)
        out["head_object"] = {"size_mb": round(h.get("ContentLength", 0) / 1e6, 1)}
    except ClientError as e:
        code = e.response["ResponseMetadata"]["HTTPStatusCode"]
        out["head_object"] = f"{code} {e.response['Error'].get('Code')}"

    try:
        lst = s3.list_objects_v2(Bucket="asu-nsf-phoenix", Prefix="data/lidar_data/", MaxKeys=2)
        out["list_objects_v2"] = {
            "count": len(lst.get("Contents", [])),
            "first_keys": [c["Key"] for c in lst.get("Contents", [])][:2],
        }
    except ClientError as e:
        code = e.response["ResponseMetadata"]["HTTPStatusCode"]
        out["list_objects_v2"] = f"{code} {e.response['Error'].get('Code')}"

    return out


def main() -> None:
    import json

    import coiled
    from dask.distributed import Client

    from pv_geom.coiled_env import REGION, SOFTWARE_ENV, install_pv_geom_on_workers

    print(f"[probe] spinning up 1-worker probe cluster in {REGION}")
    cluster = coiled.Cluster(
        name="pv-geom-aws-probe",
        n_workers=1,
        worker_cpu=2,
        worker_memory="4GiB",
        software=SOFTWARE_ENV,
        region=REGION,
    )
    client = Client(cluster)
    try:
        install_pv_geom_on_workers(client)
        result = client.submit(_probe).result()
        print("[probe] result:")
        print(json.dumps(result, indent=2, default=str))
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
