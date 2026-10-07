"""Score a Label Studio export against the model's mounting labels.

Joins the (blind) annotations back to model predictions via sample.csv,
then prints:
  - fine confusion: 9 model labels x human labels
  - 3-way superclass confusion (rooftop / canopy / ground)
  - per-stratum accuracy and the design-weighted overall accuracy
    (strata partition the full output; weights from sample_meta.json)
  - polygon-overshoot issue rates by stratum (fake-canopy-evidence check)

Usage:
    uv run python scripts/_label_analysis.py <ls_export.json> \
        [--sample-dir data/validation/v0.1.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PRED_SUPER = {
    "flush_mount_pitched_roof": "rooftop",
    "flush_mount_flat_roof": "rooftop",
    "tilted_rack_rooftop": "rooftop",
    "east_west_rack_rooftop": "rooftop",
    "carport": "canopy",
    "ground_mount_fixed": "ground",
    "ground_mount_tracker_suspected": "ground",
    "pole_mount": "ground",
    "ambiguous": "ambiguous",
}
TRUTH_SUPER = {
    "rooftop": "rooftop",
    "canopy_carport": "canopy",
    "ground_mount": "ground",
    "pole_mount": "ground",
}


def parse_export(path: Path) -> pd.DataFrame:
    rows = []
    for task in json.loads(path.read_text(encoding="utf-8")):
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        rec = {"polygon_id": task["data"]["polygon_id"], "truth": None,
               "issues": "", "notes": ""}
        for res in anns[0].get("result", []):
            if res["from_name"] == "superclass":
                rec["truth"] = res["value"]["choices"][0]
            elif res["from_name"] == "issues":
                rec["issues"] = "|".join(res["value"]["choices"])
            elif res["from_name"] == "notes":
                rec["notes"] = " ".join(res["value"].get("text", []))
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=Path)
    ap.add_argument("--sample-dir", type=Path, default=Path("data/validation/v0.1.0"))
    args = ap.parse_args()

    labels = parse_export(args.export)
    sample = pd.read_csv(args.sample_dir / "sample.csv")
    meta = json.loads((args.sample_dir / "sample_meta.json").read_text())
    df = sample.merge(labels, on="polygon_id", how="inner")
    print(f"labeled {len(df)}/{len(sample)} sampled polygons "
          f"({len(labels)} annotations in export)\n")
    if df.empty:
        return

    print("=== fine confusion: model label x human label ===")
    print(pd.crosstab(df.mounting_type, df.truth, margins=True).to_string(), "\n")

    scored = df[df.truth.isin(TRUTH_SUPER)].copy()
    excluded = df[~df.truth.isin(TRUTH_SUPER)]
    print(f"excluded from scoring: {len(excluded)} "
          f"({excluded.truth.value_counts().to_dict()})\n")

    scored["pred_super"] = scored.mounting_type.map(PRED_SUPER)
    scored["truth_super"] = scored.truth.map(TRUTH_SUPER)
    print("=== superclass confusion (model x human) ===")
    print(pd.crosstab(scored.pred_super, scored.truth_super, margins=True).to_string(), "\n")

    decided = scored[scored.pred_super != "ambiguous"].copy()
    decided["correct"] = decided.pred_super == decided.truth_super
    print("=== per-stratum superclass accuracy (decided rows only) ===")
    per = decided.groupby("stratum").agg(n=("correct", "size"),
                                         acc=("correct", "mean")).round(3)
    print(per.to_string(), "\n")

    # Design-weighted accuracy over the DECIDED part of the output: weight each
    # stratum by its population share among strata that produce decided rows.
    strata = meta["strata"]
    w_acc, w_tot = 0.0, 0.0
    for name, grp in decided.groupby("stratum"):
        pop = strata[name]["population"]
        w_acc += pop * grp.correct.mean()
        w_tot += pop
    if w_tot:
        print(f"design-weighted superclass accuracy (decided strata, "
              f"pop={w_tot:,.0f}): {w_acc / w_tot:.3f}\n")

    # What do the ambiguous rows actually look like to a human?
    amb = scored[scored.pred_super == "ambiguous"]
    if len(amb):
        print("=== human labels for model-ambiguous rows ===")
        print(amb.groupby(["stratum", "truth_super"]).size().to_string(), "\n")

    print("=== polygon_overshoots_edge rate by stratum ===")
    df["overshoot"] = df.issues.fillna("").str.contains("polygon_overshoots_edge")
    print(df.groupby("stratum").agg(n=("overshoot", "size"),
                                    overshoot=("overshoot", "mean")).round(3).to_string())

    out = args.sample_dir / "scored.csv"
    df.to_csv(out, index=False)
    print(f"\njoined per-row table written to {out}")


if __name__ == "__main__":
    main()
