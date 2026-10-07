"""Build a reference table from the U.S. Large-Scale Solar Photovoltaic Database.

USPVDB (USGS / LBNL) maps about 6,600 facilities of roughly 1 MW and larger:
ground arrays, large rooftops and parking canopies. Each has a digitised
boundary and, from EIA Form 860, a tilt, an azimuth and an operating year.

Unlike a point location, the boundary says which polygons belong to a facility
without anyone looking at their angles, so with ``--polygons`` this script
fills ``polygon_ids`` itself: every polygon whose centroid lies inside the
boundary (buffered by ``--buffer`` metres, since boundaries are drawn loosely).

What the reference is, and is not:

- One tilt and azimuth per facility (``scope = system``). Where a plant reports
  several, USPVDB stores their average, which may describe no array at all.
- Reported, not surveyed (``grade = documented``).
- ``present_by`` is set to the end of the operating year. Check it against
  imagery where it matters.
- Trackers are left out: their attitude at the moment of the flight is unknown.

Usage:
    python scripts/uspvdb_references.py --bbox -113 32.5 -111 34.5 \\
        --polygons detections.parquet --id-col cluster_id --out refs.csv
"""

from __future__ import annotations

import argparse
import io
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

ARCHIVE = "https://energy.usgs.gov/uspvdb/assets/data/uspvdbGeoJSON.zip"


def read_facilities(source: str | None) -> gpd.GeoDataFrame:
    """The USPVDB facility layer, from a local GeoJSON/zip or the USGS site."""
    if source is None:
        tmp = Path(tempfile.mkdtemp()) / "uspvdbGeoJSON.zip"
        urllib.request.urlretrieve(ARCHIVE, tmp)
        source = str(tmp)
    if not str(source).endswith(".zip"):
        return gpd.read_file(source)
    with zipfile.ZipFile(source) as archive:
        member = next(n for n in archive.namelist() if n.lower().endswith((".geojson", ".json")))
        return gpd.read_file(io.BytesIO(archive.read(member)))


def build(facilities: gpd.GeoDataFrame, bbox: tuple[float, float, float, float],
          polygons: gpd.GeoDataFrame | None = None, id_col: str = "polygon_id",
          buffer_m: float = 10.0) -> pd.DataFrame:
    fac = facilities.to_crs("EPSG:4326").cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].copy()
    fac = fac[fac["p_axis"].astype(str).str.lower() == "fixed-tilt"]
    fac = fac[pd.to_numeric(fac["p_tilt"], errors="coerce").notna()]

    members: dict[int, list[str]] = {}
    if polygons is not None and len(fac):
        metric = polygons.estimate_utm_crs()
        cent = gpd.GeoDataFrame({"pid": polygons[id_col].astype(str).to_numpy()},
                                geometry=polygons.geometry.to_crs(metric).centroid.values,
                                crs=metric)
        zones = fac[["case_id", "geometry"]].to_crs(metric)
        zones["geometry"] = zones.geometry.buffer(buffer_m)
        hit = gpd.sjoin(cent, zones, predicate="within")
        members = {int(k): sorted(v) for k, v in hit.groupby("case_id")["pid"]}

    rows = []
    for f in fac.itertuples(index=False):
        azimuth = pd.to_numeric(f.p_azimuth, errors="coerce")
        year = pd.to_numeric(f.p_year, errors="coerce")
        rows.append({
            "reference_id": f"uspvdb-{int(f.case_id)}",
            "polygon_ids": ";".join(members.get(int(f.case_id), [])),
            "tilt_deg": float(f.p_tilt),
            "azimuth_deg": None if pd.isna(azimuth) else float(azimuth),
            "system_id": int(f.case_id),
            "source": f"USPVDB {int(f.case_id)} / EIA plant {f.eia_id}: {f.p_name}",
            "scope": "system", "grade": "documented",
            "tilt_resolution_deg": 1.0, "azimuth_resolution_deg": 1.0, "suspect": False,
            "present_by": "" if pd.isna(year) else f"{int(year)}-12-31", "absent_at": "",
            "notes": f"{f.p_sys_type}; {f.p_cap_dc} MW DC; operating year {f.p_year}; "
                     f"digitisation confidence {f.p_dig_conf}; facility-level EIA-860 angles "
                     "(averaged where a plant reports several)",
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--uspvdb", help="Local USPVDB GeoJSON or zip (else downloaded).")
    ap.add_argument("--polygons", help="Polygon layer to match to facility boundaries.")
    ap.add_argument("--id-col", default="polygon_id")
    ap.add_argument("--buffer", type=float, default=10.0,
                    help="Metres to grow each facility boundary by when matching.")
    args = ap.parse_args()
    polygons = gpd.read_parquet(args.polygons) if args.polygons else None
    refs = build(read_facilities(args.uspvdb), tuple(args.bbox), polygons, args.id_col,
                 args.buffer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    refs.to_csv(args.out, index=False)
    matched = int((refs["polygon_ids"] != "").sum()) if len(refs) else 0
    print(f"{len(refs)} fixed-tilt facilities written to {args.out}; "
          f"{matched} with matched polygons.")


if __name__ == "__main__":
    main()
