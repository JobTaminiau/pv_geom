"""Build the stratified ground-truth validation sample for Label Studio.

Draws a seeded, stratified sample from the full-atlas output (out_full_v0.1.0),
renders two aerial chips per polygon (detail + context, Maricopa 2024 ortho via
_aerial.py), and writes a ready-to-import Label Studio bundle:

    <out>/chips/<polygon_id>_{detail,context}.jpg
    <out>/tasks.json            blind tasks (no model prediction leaks)
    <out>/labeling_config.xml   Label Studio labeling interface
    <out>/sample.csv            sampled rows + model predictions (analysis key)
    <out>/sample_meta.json      seed + per-stratum population/sample counts
    <out>/README.md             exact launch + import steps

The eight strata partition the full 349k-row output, so design-weighted
(population/sample) estimates from the labels are unbiased for the whole run.
Analysis counterpart: scripts/_label_analysis.py.

Usage:
    uv run python scripts/_label_sample.py                 # sample + render
    uv run python scripts/_label_sample.py --no-render     # sample only
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyproj
from PIL import Image, ImageDraw
from shapely import wkb

sys.path.insert(0, str(Path(__file__).parent))
from _aerial import aerial_basemap  # noqa: E402

SEED = 20260730
DETAIL_LOD = 13   # ~6 cm/px
CONTEXT_LOD = 10  # ~0.5 m/px
CONTEXT_HALF_M = 100.0

_TO_WGS84 = pyproj.Transformer.from_crs("EPSG:6341", "EPSG:4326", always_xy=True)

COLS = [
    "polygon_id", "parent_polygon_id", "geometry", "mounting_type",
    "mounting_confidence", "mounting_rule", "on_building",
    "height_above_ground_m", "area_m2", "aspect_ratio", "panel_rmse_m", "flags",
]

ROOFTOP_LABELS = {
    "flush_mount_pitched_roof", "flush_mount_flat_roof",
    "tilted_rack_rooftop", "east_west_rack_rooftop",
}


def _strata(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Disjoint boolean masks that partition every row of the output."""
    amb = df.mounting_type == "ambiguous"
    amb_recoverable = (
        amb & df.panel_rmse_m.notna() & df.on_building
        & df.height_above_ground_m.notna()
    )
    return {
        # the on-building pole_mount contradiction population (edge-overshoot hypothesis)
        "pole_on_building": (df.mounting_type == "pole_mount") & df.on_building,
        "pole_off_building": (df.mounting_type == "pole_mount") & ~df.on_building,
        # carport reroute (on-building) vs plain off-building carport
        "carport_on_building": (df.mounting_type == "carport") & df.on_building,
        "carport_off_building": (df.mounting_type == "carport") & ~df.on_building,
        # fit-ok on-building ambiguous = the superclass-recovery candidates
        "ambiguous_recoverable": amb_recoverable,
        "ambiguous_other": amb & ~amb_recoverable,
        "rooftop": df.mounting_type.isin(ROOFTOP_LABELS),
        "ground_r4r5": df.mounting_type.isin(
            {"ground_mount_fixed", "ground_mount_tracker_suspected"}
        ),
    }


ALLOC = {
    "pole_on_building": 40,
    "pole_off_building": 30,
    "carport_on_building": 40,
    "carport_off_building": 40,
    "ambiguous_recoverable": 45,
    "ambiguous_other": 30,
    "rooftop": 45,
    "ground_r4r5": 30,
}  # total 300


def draw_sample(out_dir: Path, full_dir: Path) -> pd.DataFrame:
    files = glob.glob(str(full_dir / "*.parquet"))
    if not files:
        raise SystemExit(f"no parquet partitions under {full_dir}")
    df = ds.dataset(files, format="parquet").to_table(columns=COLS).to_pandas()
    print(f"loaded {len(df)} rows from {len(files)} partitions")

    masks = _strata(df)
    covered = np.zeros(len(df), dtype=int)
    for m in masks.values():
        covered += m.to_numpy().astype(int)
    assert (covered == 1).all(), "strata must partition the output exactly once"

    rng = np.random.default_rng(SEED)
    picks, meta = [], {}
    for name, mask in masks.items():
        pool = df.index[mask]
        n = min(ALLOC[name], len(pool))
        chosen = rng.choice(pool, size=n, replace=False)
        sub = df.loc[chosen].copy()
        sub["stratum"] = name
        picks.append(sub)
        meta[name] = {"population": int(mask.sum()), "sampled": n}
        print(f"  {name:24s} pop={mask.sum():7d} sampled={n}")

    sample = pd.concat(picks).reset_index(drop=True)
    geoms = [wkb.loads(b) for b in sample["geometry"]]
    cx = [g.centroid.x for g in geoms]
    cy = [g.centroid.y for g in geoms]
    lon, lat = _TO_WGS84.transform(cx, cy)
    sample["lon"], sample["lat"] = lon, lat
    sample["flags"] = sample["flags"].apply(lambda f: "|".join(f))

    out_dir.mkdir(parents=True, exist_ok=True)
    sample.drop(columns=["geometry"]).to_csv(out_dir / "sample.csv", index=False)
    (out_dir / "sample_meta.json").write_text(
        json.dumps({"seed": SEED, "total_rows": len(df), "strata": meta}, indent=2),
        encoding="utf-8",
    )
    sample["_geom"] = geoms
    return sample


def _render_chip(
    geom_utm, path: Path, lod: int, half_m: float | None
) -> None:
    """Render one chip: aerial + red polygon outline, cropped to the box."""
    minx, miny, maxx, maxy = geom_utm.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    if half_m is None:  # detail: polygon + margin, floor 15 m half-width
        half_m = max(15.0, max(maxx - minx, maxy - miny) / 2 + 8.0)
    lon0, lat0 = _TO_WGS84.transform(cx - half_m, cy - half_m)
    lon1, lat1 = _TO_WGS84.transform(cx + half_m, cy + half_m)

    img_arr, ext = aerial_basemap((lon0, lat0, lon1, lat1), target_lod=lod)
    e_lon0, e_lon1, e_lat0, e_lat1 = ext  # (lon_min, lon_max, lat_bottom, lat_top)
    h, w = img_arr.shape[:2]

    def to_px(lon, lat):
        x = (lon - e_lon0) / (e_lon1 - e_lon0) * w
        y = (e_lat1 - lat) / (e_lat1 - e_lat0) * h
        return x, y

    img = Image.fromarray(img_arr)
    drw = ImageDraw.Draw(img)
    polys = geom_utm.geoms if geom_utm.geom_type == "MultiPolygon" else [geom_utm]
    for p in polys:
        xs, ys = zip(*p.exterior.coords)
        lons, lats = _TO_WGS84.transform(xs, ys)
        drw.line(
            [to_px(lo, la) for lo, la in zip(lons, lats)],
            fill=(255, 40, 40), width=3 if lod >= 12 else 2,
        )

    x0, y0 = to_px(lon0, lat1)  # crop to the requested box
    x1, y1 = to_px(lon1, lat0)
    img = img.crop((int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))))
    img.save(path, quality=88)


def render_all(sample: pd.DataFrame, out_dir: Path) -> None:
    chips = out_dir / "chips"
    chips.mkdir(exist_ok=True)
    for i, row in sample.iterrows():
        d_path = chips / f"{row.polygon_id}_detail.jpg"
        c_path = chips / f"{row.polygon_id}_context.jpg"
        if not (d_path.exists() and c_path.exists()):
            _render_chip(row._geom, d_path, DETAIL_LOD, None)
            _render_chip(row._geom, c_path, CONTEXT_LOD, CONTEXT_HALF_M)
        if (i + 1) % 25 == 0:
            print(f"  chips {i + 1}/{len(sample)}")


def write_bundle(sample: pd.DataFrame, out_dir: Path) -> None:
    # Blind tasks: no mounting_type / confidence / stratum — join back via
    # polygon_id -> sample.csv at analysis time (_label_analysis.py).
    tasks = [
        {
            "data": {
                "polygon_id": row.polygon_id,
                "detail": f"/data/local-files/?d=chips/{row.polygon_id}_detail.jpg",
                "context": f"/data/local-files/?d=chips/{row.polygon_id}_context.jpg",
                "area_m2": round(float(row.area_m2), 1),
                "maps_url": (
                    f"<a href='https://www.google.com/maps/@{row.lat},{row.lon},60m"
                    f"/data=!3m1!1e3' target='_blank'>open in Google Maps (satellite)</a>"
                ),
            }
        }
        for row in sample.itertuples()
    ]
    (out_dir / "tasks.json").write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    (out_dir / "labeling_config.xml").write_text(LABEL_CONFIG, encoding="utf-8")
    (out_dir / "README.md").write_text(
        LAUNCH_README.format(root=out_dir.resolve()), encoding="utf-8"
    )
    print(f"wrote {len(tasks)} tasks + config + README to {out_dir}")


LABEL_CONFIG = """\
<View>
  <View style="display:flex; align-items:flex-start; gap:16px">
    <View style="flex:55%">
      <Header value="Detail (~6 cm/px)" size="4"/>
      <Image name="detail" value="$detail" zoom="true" zoomControl="true" rotateControl="false"/>
    </View>
    <View style="flex:45%">
      <Header value="Context (200 m across)" size="4"/>
      <Image name="context" value="$context" zoom="true" zoomControl="true" rotateControl="false"/>
      <Text name="area" value="Polygon area: $area_m2 m²"/>
      <HyperText name="maps" value="$maps_url" inline="true"/>
    </View>
  </View>
  <Header value="What is the red-outlined PV array mounted on?"/>
  <Choices name="superclass" toName="detail" choice="single" required="true">
    <Choice value="rooftop" hint="On a building roof (house, commercial, warehouse)"/>
    <Choice value="canopy_carport" hint="Open-sided canopy: parking carport, shade structure, patio cover"/>
    <Choice value="ground_mount" hint="Racks/trackers on the ground (incl. utility-scale rows)"/>
    <Choice value="pole_mount" hint="Small array on a single pole/mast"/>
    <Choice value="not_pv" hint="False detection: not a solar panel"/>
    <Choice value="cant_tell" hint="Genuinely undecidable from the imagery"/>
  </Choices>
  <Header value="Polygon / imagery issues (optional)" size="5"/>
  <Choices name="issues" toName="detail" choice="multiple" required="false">
    <Choice value="polygon_overshoots_edge" hint="Outline extends past the panel onto ground/roof"/>
    <Choice value="fragment_of_larger_array" hint="Polygon covers only part of a visibly larger array"/>
    <Choice value="imagery_changed" hint="Scene differs from detection-time imagery (construction, removal)"/>
  </Choices>
  <TextArea name="notes" toName="detail" placeholder="Notes (optional)" maxSubmissions="1" rows="2"/>
</View>
"""

LAUNCH_README = """\
# pv-geom ground-truth labeling (v0.1.0 sample)

300 polygons, stratified over the full-atlas output (strata + weights in
`sample_meta.json`). Tasks are blind — model predictions live only in
`sample.csv`, joined back at analysis time by `polygon_id`.

## Launch (PowerShell)

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "{root}"
label-studio start
```

## One-time project setup

1. Create project "pv-geom v0.1.0 ground truth".
2. Settings -> Labeling Interface -> Code: paste `labeling_config.xml`.
3. Settings -> Cloud Storage -> Add Source Storage -> Local files:
   - Absolute local path: `{root}\\chips`
   - Leave "Treat every bucket object as a source file" OFF.
   - Add Storage (no sync needed — tasks carry the file references).
4. Import -> upload `tasks.json`.

## Labeling guidance

- Label what the structure IS, not what the model said (you can't see that).
- `rooftop` vs `canopy_carport`: canopies are open-sided — look for cars,
  shadows passing under, no walls. Phoenix parking canopies are common.
- `pole_mount`: one mast, small array, typically over yard/desert ground.
- Use the Google Maps link for a second opinion / oblique context; prefer
  `cant_tell` over guessing.
- Tick `polygon_overshoots_edge` whenever the red outline clearly extends
  past the panel — this directly tests the fake-canopy-evidence hypothesis.

## Afterwards

Export -> JSON, then:

```powershell
uv run python scripts/_label_analysis.py <export.json>
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-dir", default="out_full_v0.1.0")
    ap.add_argument("--out", default="data/validation/v0.1.0")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    sample = draw_sample(out_dir, Path(args.full_dir))
    write_bundle(sample, out_dir)
    if not args.no_render:
        render_all(sample, out_dir)
        print("done — see README.md in the output dir for launch steps")


if __name__ == "__main__":
    main()
