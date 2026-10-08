# Running in the cloud

A large area runs on a [Coiled](https://coiled.io) cluster: workers sit beside
the LiDAR bucket, read tiles in-region, and write partitions straight to S3.

## Set up once

```bash
pip install "pv-geom[coiled]"
coiled login
```

Workers need read access to the LiDAR bucket and write access to the output
prefix. For a bucket in another account, grant the Coiled role
`s3:GetObject`, `s3:ListBucket` and `s3:GetBucketLocation` in the bucket
policy. `scripts/coiled_aws_probe.py my_area.yaml` checks access from a real
worker.

## Configure

```yaml
study:
  output: s3://my-bucket/pv-geom/my_area

inputs:
  polygons: s3://my-bucket/inputs/pv_polygons.parquet
  lidar_prefix: s3://lidar-bucket/tiles

compute:
  backend: coiled
  memory_budget_gb: 6          # about half of worker_memory
  coiled:
    n_workers: 40
    worker_memory: 16GiB
    worker_cpu: 2
    usd_per_worker_hour: 0.10  # optional: lets --dry-run estimate the bill
```

The rest is worked out unless you set it:

| Setting | Left on `auto` it is |
| --- | --- |
| `package_source` | The exact commit your machine is running |
| `region` | Where the LiDAR bucket is |
| `software` | An environment named after pv-geom's dependency list, built on first use |
| `name` | `pv-geom-<study.name>` |

## Run

```bash
pv-geom run --config my_area.yaml --dry-run     # size, time and cost estimate
pv-geom run --config my_area.yaml
```

Two checks protect you from running the wrong code:

- Before a cluster starts, the run stops if your checkout has **uncommitted or
  unpushed changes**. Workers install from the remote, so they would run
  something other than what is on your machine.
- After the workers install, each is asked what it has. The run stops if any
  reports a different version or commit.

## If it stops

```bash
pv-geom run --config my_area.yaml --resume
```

Only the tile groups without a partition are run again. A tile group whose
task failed leaves its polygons in `part-unmeasured.parquet` with a status
that says why, so every input polygon is always accounted for.

## Memory

Tiles are decoded a chunk at a time and only returns near a polygon are kept.
If a tile group would still exceed `compute.memory_budget_gb`, its polygons are
measured in spatial batches. Results are identical either way. Keep one task
per worker (`worker_threads: 1`) unless the budget times the thread count fits
comfortably in a worker's memory.
