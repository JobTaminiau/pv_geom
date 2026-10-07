"""Prepare an accuracy set from PVDAQ's documented systems and public 3DEP LiDAR.

PVDAQ's documented (non-PVOutput) systems report tilt and azimuth to a degree
and can be located. This script does the mechanical preparation around them;
the polygons themselves are drawn by a person (see docs/validation.md).

    sites    list the candidate systems, one row per reported mount
    lidar    for each site, find a 3DEP LiDAR collection flown while the system
             was operating, download the tile(s) and keep a clip around the site
    render   write the images polygons are drawn on: NAIP for identifying the
             array, LiDAR height and intensity for outlining it

Everything it reads is public: the PVDAQ inventory (OEDI), 3DEP point clouds
(USGS National Map) and NAIP imagery (USDA, via the USGS image service).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

INVENTORY = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/systems_20250729.csv"
EPT_INDEX = ("https://raw.githubusercontent.com/hobu/usgs-lidar/master/boundaries/"
             "resources.geojson")
NAIP = ("https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/"
        "exportImage")
CLIP_RADIUS_M = 150.0
MIN_KW = 5.0          # smaller arrays cannot be outlined on 60 cm imagery

# Coordinates shared by many unrelated systems are district centroids, not sites.
MAX_SYSTEMS_PER_COORDINATE = 10
# Known slips in the inventory.
FIXES = {1416: {"longitude": -78.6341}}       # Raleigh, NC: longitude sign


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pv-geom accuracy set"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def candidate_sites(inventory: pd.DataFrame) -> pd.DataFrame:
    s = inventory.loc[:, ~inventory.columns.str.startswith("Unnamed")].copy()
    for sid, fix in FIXES.items():
        for col, val in fix.items():
            s.loc[s["system_id"] == sid, col] = val
    s = s[~s["system_public_name"].astype(str).str.contains("pvoutput", case=False)]
    for col in ("tilt", "azimuth", "latitude", "longitude", "dc_capacity_kW"):
        s[col] = pd.to_numeric(s[col], errors="coerce")
    s = s[(s["tracking"] != "tracking") & s["tilt"].notna() & (s["azimuth"] >= 0)
          & s["latitude"].notna()]
    coord = s["latitude"].round(4).astype(str) + "," + s["longitude"].round(4).astype(str)
    names = s["system_public_name"].str.replace(r"[\s\-]*\d+[a-zA-Z]?$", "", regex=True)
    crowded = coord.map(coord.value_counts()) >= MAX_SYSTEMS_PER_COORDINATE
    district = names.str.contains("District|Parish", case=False)
    s = s[~crowded & ~district & (s["dc_capacity_kW"].fillna(0) >= MIN_KW)]
    out = pd.DataFrame({
        "system_id": s["system_id"].astype(int), "name": s["system_public_name"],
        "location": s["site_location"], "latitude": s["latitude"],
        "longitude": s["longitude"], "mount_type": s["type"], "tilt_deg": s["tilt"],
        "azimuth_deg": s["azimuth"], "dc_kw": s["dc_capacity_kW"],
        "first_data": pd.to_datetime(s["first_timestamp"], errors="coerce").dt.date,
        "last_data": pd.to_datetime(s["last_timestamp"], errors="coerce").dt.date,
    })
    return out.sort_values(["latitude", "longitude", "system_id"]).reset_index(drop=True)


def _project_year(name: str) -> int | None:
    years = [int(y) for y in re.findall(r"(?<!\d)(19\d\d|20\d\d)(?!\d)", name)]
    return years[0] if years else None       # the flight year comes first; a later one is publication


def ept_resources(lon: float, lat: float, index: dict) -> list[dict]:
    """USGS 3DEP collections (Entwine point tiles on AWS) covering a point."""
    from shapely.geometry import Point, shape

    here = Point(lon, lat)
    out = []
    for f in index["features"]:
        if shape(f["geometry"]).contains(here):
            name = f["properties"]["name"]
            out.append({"name": name, "url": f["properties"]["url"],
                        "year": _project_year(name)})
    return out


def choose_collection(resources: list[dict], first_year: int, last_year: int) -> dict | None:
    """The latest collection flown while the system was reporting data, else
    the earliest flown after it started."""
    dated = [r for r in resources if r["year"]]
    during = [r for r in dated if first_year < r["year"] <= last_year]
    after = [r for r in dated if r["year"] > last_year]
    if during:
        return max(during, key=lambda r: r["year"])
    return min(after, key=lambda r: r["year"]) if after else None


def read_ept(url: str, lon: float, lat: float, radius_m: float) -> dict | None:
    """Every return within ``radius_m`` of a point, from an Entwine point-tile
    set, fetching only the octree nodes that overlap it.

    Returns arrays in the set's own CRS (USGS serves Web Mercator) plus the
    header facts needed to write them back out.
    """
    import laspy
    from pyproj import CRS, Transformer

    root = url.rsplit("/", 1)[0]
    meta = json.loads(_get(url))
    srs = meta["srs"]
    crs = CRS.from_user_input(srs.get("wkt") or f"{srs['authority']}:{srs['horizontal']}")
    x0, y0 = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    # Web Mercator stretches distance by 1/cos(latitude).
    r = radius_m / np.cos(np.radians(lat)) if crs.to_epsg() == 3857 else radius_m
    box = (x0 - r, y0 - r, x0 + r, y0 + r)
    bx0, by0, _, bx1, by1, _ = meta["bounds"]

    def node_box(dep: int, i: int, j: int) -> tuple[float, float, float, float]:
        step_x, step_y = (bx1 - bx0) / 2 ** dep, (by1 - by0) / 2 ** dep
        return bx0 + i * step_x, by0 + j * step_y, bx0 + (i + 1) * step_x, by0 + (j + 1) * step_y

    def overlaps(nb: tuple[float, float, float, float]) -> bool:
        return nb[0] <= box[2] and nb[2] >= box[0] and nb[1] <= box[3] and nb[3] >= box[1]

    chunks: dict[str, list[np.ndarray]] = {k: [] for k in
                                           ("x", "y", "z", "cls", "intensity", "gps")}
    encoding = None
    stack = ["0-0-0-0"]
    hierarchy: dict[str, int] = {}
    loaded: set[str] = set()
    while stack:
        key = stack.pop()
        dep, i, j, k = (int(v) for v in key.split("-"))
        if not overlaps(node_box(dep, i, j)):
            continue
        if key not in hierarchy or hierarchy[key] == -1:
            if key in loaded:
                continue
            loaded.add(key)
            hierarchy.update(json.loads(_get(f"{root}/ept-hierarchy/{key}.json")))
        if hierarchy.get(key, 0) <= 0:
            continue
        las = laspy.read(io.BytesIO(_get(f"{root}/ept-data/{key}.laz", timeout=300)))
        if encoding is None:
            encoding = las.header.global_encoding
        x, y = np.asarray(las.x), np.asarray(las.y)
        near = (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])
        if near.any():
            chunks["x"].append(x[near])
            chunks["y"].append(y[near])
            chunks["z"].append(np.asarray(las.z)[near])
            chunks["cls"].append(np.asarray(las.classification)[near])
            chunks["intensity"].append(np.asarray(las.intensity)[near])
            chunks["gps"].append(np.asarray(las.gps_time)[near])
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    child = f"{dep + 1}-{2 * i + di}-{2 * j + dj}-{2 * k + dk}"
                    if child in hierarchy:
                        stack.append(child)
    if not chunks["x"]:
        return None
    out = {k: np.concatenate(v) for k, v in chunks.items()}
    out["crs"], out["encoding"] = crs, encoding
    return out


def write_clip(pts: dict, lon: float, lat: float, out: Path) -> str:
    """Write the returns as a LAZ tile in the site's UTM zone (metres)."""
    import laspy
    from pyproj import CRS, Transformer
    from pyproj.aoi import AreaOfInterest
    from pyproj.database import query_utm_crs_info

    utm = query_utm_crs_info(datum_name="NAD83(2011)",
                             area_of_interest=AreaOfInterest(lon, lat, lon, lat))
    target = CRS.from_epsg(int(utm[0].code))
    x, y = Transformer.from_crs(pts["crs"], target, always_xy=True).transform(pts["x"], pts["y"])
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([np.floor(x.min()), np.floor(y.min()), 0.0])
    header.add_crs(target)
    if pts["encoding"] is not None:
        header.global_encoding.gps_time_type = pts["encoding"].gps_time_type
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, pts["z"]
    las.classification = pts["cls"].astype(np.uint8)
    las.intensity = pts["intensity"].astype(np.uint16)
    las.gps_time = pts["gps"]
    las.write(str(out))
    return f"EPSG:{utm[0].code}"


def cmd_sites(args: argparse.Namespace) -> None:
    raw = Path(args.inventory).read_bytes() if args.inventory else _get(INVENTORY)
    sites = candidate_sites(pd.read_csv(io.BytesIO(raw), low_memory=False))
    args.out.mkdir(parents=True, exist_ok=True)
    sites.to_csv(args.out / "sites.csv", index=False)
    print(f"{len(sites)} reported mounts at "
          f"{sites[['latitude', 'longitude']].round(4).drop_duplicates().shape[0]} locations "
          f"written to {args.out / 'sites.csv'}")


def cmd_lidar(args: argparse.Namespace) -> None:
    sites = pd.read_csv(args.out / "sites.csv", parse_dates=["first_data", "last_data"])
    index_path = args.out / "ept_resources.geojson"
    if not index_path.exists():
        index_path.write_bytes(_get(EPT_INDEX))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    log = []
    for (lat, lon), group in sites.groupby([sites["latitude"].round(4),
                                            sites["longitude"].round(4)]):
        key = f"{int(group['system_id'].iloc[0])}"
        if args.only and key not in args.only:
            continue
        site_dir = args.out / key
        first, last = group["first_data"].min().year, group["last_data"].max().year
        entry = {"site": key, "systems": group["system_id"].tolist(),
                 "name": group["name"].iloc[0], "operating": f"{first}-{last}"}
        resources = ept_resources(lon, lat, index)
        entry["collections"] = sorted(r["name"] for r in resources)
        chosen = choose_collection(resources, first, last)
        if chosen is None:
            entry["status"] = "no collection during or after operation"
        else:
            entry["chosen"] = chosen["name"]
            entry["in_operating_period"] = bool(first < chosen["year"] <= last)
            clip = site_dir / "lidar_clip.laz"
            if clip.exists():
                entry["status"] = "ok"
            else:
                try:
                    pts = read_ept(chosen["url"], lon, lat, CLIP_RADIUS_M)
                except Exception as exc:
                    pts, entry["status"] = None, f"read failed: {exc}"
                if pts is not None:
                    site_dir.mkdir(exist_ok=True)
                    entry["crs"] = write_clip(pts, lon, lat, clip)
                    entry["returns"] = len(pts["x"])
                    entry["status"] = "ok"
                elif "status" not in entry:
                    entry["status"] = "no returns at the site"
        log.append(entry)
        print(key, entry["name"][:38], "|", entry["operating"], "|", entry.get("chosen", "-"),
              "|", entry.get("returns", ""), "|", entry["status"], flush=True)
    (args.out / "lidar_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


def cmd_render(args: argparse.Namespace) -> None:
    import laspy
    import matplotlib
    from matplotlib.figure import Figure
    from pyproj import Transformer

    from pv_geom.utils.crs import horizontal

    matplotlib.use("Agg")
    sites = pd.read_csv(args.out / "sites.csv")
    half = args.half_width
    for clip in sorted(args.out.glob("*/lidar_clip.laz")):
        site = sites[sites["system_id"] == int(clip.parent.name)].iloc[0]
        if args.only and clip.parent.name not in args.only:
            continue
        las = laspy.read(str(clip))
        crs = las.header.parse_crs()
        h = horizontal(crs)
        unit = h.axis_info[0].unit_conversion_factor
        to_xy = Transformer.from_crs("EPSG:4326", h, always_xy=True)
        cx, cy = to_xy.transform(site["longitude"] + args.shift_e / 85_000.0,
                                 site["latitude"] + args.shift_n / 111_000.0)
        x, y, z = (np.asarray(las.x) - cx) * unit, (np.asarray(las.y) - cy) * unit, \
            np.asarray(las.z) * unit
        keep = (np.abs(x) < half) & (np.abs(y) < half) & (np.asarray(las.classification) != 7)
        x, y, z, inten = x[keep], y[keep], z[keep], np.asarray(las.intensity)[keep]
        cell = 2 * half / args.pixels
        ix = np.clip(((x + half) / cell).astype(int), 0, args.pixels - 1)
        iy = np.clip(((half - y) / cell).astype(int), 0, args.pixels - 1)
        height = np.full((args.pixels, args.pixels), np.nan)
        np.fmax.at(height, (iy, ix), z)
        tone = np.full((args.pixels, args.pixels), np.nan)
        np.fmax.at(tone, (iy, ix), inten.astype(float))
        to_ll = Transformer.from_crs(h, "EPSG:4326", always_xy=True)
        w, s_ = to_ll.transform(cx - half / unit, cy - half / unit)
        e, n = to_ll.transform(cx + half / unit, cy + half / unit)
        query = urllib.parse.urlencode({"bbox": f"{w},{s_},{e},{n}", "bboxSR": 4326,
                                        "imageSR": 4326, "size": f"{args.pixels},{args.pixels}",
                                        "format": "png", "f": "image"})
        naip_path = clip.parent / "naip.png"
        naip_path.write_bytes(_get(f"{NAIP}?{query}"))
        import matplotlib.image as mpimg

        fig = Figure(figsize=(18, 6.4), layout="constrained")
        axes = fig.subplots(1, 3)
        ext = (-half, half, -half, half)
        axes[0].imshow(mpimg.imread(str(naip_path)), extent=ext)
        lo, hi = np.nanpercentile(height, [2, 98])
        axes[1].imshow(height, extent=ext, cmap="viridis", vmin=lo, vmax=hi)
        lo, hi = np.nanpercentile(tone, [2, 98])
        axes[2].imshow(tone, extent=ext, cmap="gray", vmin=lo, vmax=hi)
        for ax, title in zip(axes, ("NAIP", "LiDAR highest return (m)", "LiDAR intensity"),
                             strict=True):
            ax.set_title(f"{title} - {site['name'][:38]}", fontsize=9)
            ax.set_xticks(np.arange(-half, half + 1, args.grid))
            ax.set_yticks(np.arange(-half, half + 1, args.grid))
            ax.grid(color="white", linewidth=0.4, alpha=0.6)
            ax.tick_params(labelsize=6)
        fig.savefig(clip.parent / "draw.png", dpi=100)
        (clip.parent / "frame.json").write_text(json.dumps(
            {"crs": h.to_wkt(), "centre_xy": [cx, cy], "unit_to_m": unit, "half_width_m": half,
             "density_per_m2": float(keep.sum() / (2 * half) ** 2)}, indent=2), encoding="utf-8")
        print(clip.parent.name, site["name"][:40], f"{keep.sum() / (2 * half) ** 2:.1f} pts/m2")


def cmd_polygons(args: argparse.Namespace) -> None:
    """Turn the drawn outlines (``polygons.json``: label, system_id and corner
    coordinates in metres east/north of the drawing frame's centre) into a
    polygon layer in the LiDAR's CRS, and draw them back over the images."""
    import geopandas as gpd
    import matplotlib.image as mpimg
    from matplotlib.figure import Figure
    from shapely.geometry import Polygon

    for spec in sorted(args.out.glob("*/polygons.json")):
        if args.only and spec.parent.name not in args.only:
            continue
        frame = json.loads((spec.parent / "frame.json").read_text(encoding="utf-8"))
        cx, cy = frame["centre_xy"]
        unit = frame["unit_to_m"]
        drawn = json.loads(spec.read_text(encoding="utf-8"))
        geoms = [Polygon([(cx + x / unit, cy + y / unit) for x, y in d["xy"]]) for d in drawn]
        gdf = gpd.GeoDataFrame(
            {"polygon_id": [f"{spec.parent.name}-{d['label']}" for d in drawn],
             "system_id": [str(d["system_id"]) for d in drawn]},
            geometry=geoms, crs=frame["crs"])
        gdf.to_parquet(spec.parent / "polygons.parquet")
        half = frame["half_width_m"]
        fig = Figure(figsize=(6.4, 6.4), layout="constrained")
        ax = fig.subplots()
        ax.imshow(mpimg.imread(str(spec.parent / "naip.png")), extent=(-half, half, -half, half))
        for d in drawn:
            xy = np.array([*d["xy"], d["xy"][0]])
            ax.plot(xy[:, 0], xy[:, 1], color="yellow", linewidth=1.0)
            ax.annotate(d["label"], xy[:-1].mean(axis=0), color="yellow", fontsize=7, ha="center")
        fig.savefig(spec.parent / "drawn.png", dpi=100)
        print(spec.parent.name, len(gdf), "polygons,",
              f"{gdf.geometry.area.sum() * unit ** 2:.0f} m2")


def cmd_measure(args: argparse.Namespace) -> None:
    """Measure every drawn site and compare with the reported mounts."""
    import geopandas as gpd
    import laspy

    import pv_geom
    from pv_geom.config import PVGeomConfig
    from pv_geom.validation import compare_to_reference, normalise_references

    sites = pd.read_csv(args.out / "sites.csv")
    measured, refs = [], []
    for layer in sorted(args.out.glob("*/polygons.parquet")):
        site_dir = layer.parent
        tiles = site_dir / "tiles"
        tiles.mkdir(exist_ok=True)
        if not (tiles / "clip.laz").exists():
            (tiles / "clip.laz").write_bytes((site_dir / "lidar_clip.laz").read_bytes())
        # The outlines were drawn on the LiDAR itself, so they are of its date.
        with laspy.open(str(tiles / "clip.laz")) as reader:
            from pv_geom.io.lidar import _gps_range_to_dates

            gps = np.asarray(next(reader.chunk_iterator(200_000)).gps_time)
            flown = _gps_range_to_dates(gps, reader.header)[1]
        out = site_dir / "output"
        if args.rerun and out.exists():
            shutil.rmtree(out)
        if not (out / "manifest.json").exists():
            pv_geom.run(PVGeomConfig(), polygons=layer, polygon_id_col="polygon_id",
                        lidar_prefix=tiles, output=out, polygon_vintage=str(flown),
                        backend="local", use_dask=False)
        polys = gpd.read_parquet(layer)
        rows = pv_geom.load(out)
        rows["site"] = site_dir.name
        measured.append(pd.DataFrame(rows.drop(columns="geometry")))
        # A polygon lists every system it may belong to ("1283;1433") when the
        # records cannot be told apart on the ground.
        ids = sorted({int(v) for joined in polys["system_id"] for v in str(joined).split(";")})
        for sid in ids:
            group = polys[[str(sid) in str(j).split(";") for j in polys["system_id"]]]
            shared = any(";" in str(j) for j in group["system_id"])
            for i, m in enumerate(sites[sites["system_id"] == sid].itertuples()):
                refs.append({
                    "reference_id": f"pvdaq-{sid}-m{i}", "system_id": sid,
                    "polygon_ids": ";".join(group["polygon_id"]),
                    "tilt_deg": m.tilt_deg, "azimuth_deg": m.azimuth_deg,
                    "source": f"PVDAQ {sid}: {m.name}",
                    "scope": ("mount" if (sites["system_id"] == sid).sum() == 1 and not shared
                              else "system"),
                    "grade": "documented", "present_by": m.first_data,
                    "notes": f"{m.mount_type}; {m.dc_kw:.1f} kW DC; {m.location}"})
    table = pd.concat(measured, ignore_index=True)
    table.to_csv(args.out / "measurements.csv", index=False)
    references = pd.DataFrame(refs)
    references.to_csv(args.out / "references.csv", index=False)
    result = compare_to_reference(table, normalise_references(references))
    result.write(args.out)
    pd.set_option("display.width", 220)
    print(result.references[["reference_id", "scope", "eligibility", "outcome", "n_facets",
                             "measured_area_m2", "reference_tilt_deg", "tilt_error_deg",
                             "reference_azimuth_deg", "azimuth_error_deg",
                             "area_share_agreeing"]].round(2).to_string(index=False))
    print(json.dumps(result.summary["mount_scope_accuracy"], indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["sites", "lidar", "render", "polygons", "measure"])
    ap.add_argument("--out", type=Path, required=True, help="Working directory.")
    ap.add_argument("--inventory", help="Local PVDAQ systems CSV (else downloaded).")
    ap.add_argument("--only", nargs="*", help="render: site ids to draw.")
    ap.add_argument("--half-width", type=float, default=60.0, help="render: metres.")
    ap.add_argument("--pixels", type=int, default=480)
    ap.add_argument("--grid", type=float, default=10.0, help="render: grid spacing, metres.")
    ap.add_argument("--shift-e", type=float, default=0.0, help="render: recentre east, m.")
    ap.add_argument("--shift-n", type=float, default=0.0, help="render: recentre north, m.")
    ap.add_argument("--rerun", action="store_true", help="measure: redo existing outputs.")
    args = ap.parse_args()
    {"sites": cmd_sites, "lidar": cmd_lidar, "render": cmd_render, "polygons": cmd_polygons,
     "measure": cmd_measure}[args.command](args)


if __name__ == "__main__":
    main()
