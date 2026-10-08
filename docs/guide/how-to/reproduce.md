# Reproducing a run

Every run's `manifest.json` records three things under `reproducibility`:

- a **content hash** of the rows, which ignores the run id, how the work was
  partitioned, the order of rows and the package version;
- the **inputs** read: size and SHA-256 of the polygon and footprint layers,
  and the names and sizes of the LiDAR tiles;
- the **environment**: Python, platform, and library versions.

## Is this output what its manifest says?

```bash
pv-geom verify out/my_area
```

prints `intact` when the rows still hash to the recorded value, and exits with
an error if a partition has been changed or replaced.

## Do two outputs agree?

```bash
pv-geom verify out/my_area --against out/my_area_rerun
```

If they differ it says which columns, in how many rows, and by how much.

## Rerun and check

```bash
pv-geom reproduce out/my_area --out out/my_area_rerun
```

reruns from the manifest's configuration and inputs, compares, and notes
anything that explains a difference: a library version that changed, or an
input file that is no longer the one the original run read.

## How exact is "the same"?

- **On one machine and environment: bit for bit.** The content hashes are equal.
- **Across platforms: within 0.0001** (degrees or metres), with every row,
  label and count identical. Measured between Linux and Windows, the largest
  differences are 0.000015 degrees and a hundred-millionth of a metre.
  Linear-algebra libraries differ in their last bits by platform and
  processor, so equal hashes cannot be promised across machines. Pass
  `--tolerance 0` to demand them anyway.

To pin the environment, install from the repository's `uv.lock`
(`uv sync --frozen`) at the commit named in the manifest.
