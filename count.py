"""Score every window with the trained probe, keep each lot's best, and write the courts list.

    python count.py --threshold 0.9

Writes data/results.csv (best window per lot), data/courts.csv and data/courts.geojson (one
row per distinct court: lat, lon, score, Google Maps link).
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import BLOCKS, DATA, PROBE, TO_4326, load_probe, probe_score

CAND = DATA / "candidates.npz"                 # all windows of lots that ever scored above CAND_MIN
CAND_MIN = 0.15
CACHE = DATA / "best_per_lot.parquet"        # cached scores; rebuilt when the probe or windows change
CACHE_STAMP = DATA / "best_per_lot.stamp"

DEDUPE_M = 40.0   # Web Mercator metres (~29 m on the ground): flagged windows this close are one court


def best_per_lot(probe=None):
    """DataFrame parcel_id, score, x, y (EPSG:3857), lat, lon: the best-scoring window of every lot."""
    probe = probe or load_probe()
    if probe is None:
        raise SystemExit("no trained probe - run train_probe.py first")
    files = sorted(BLOCKS.glob("*.npz"))
    stamp = f"{PROBE.stat().st_mtime_ns}-{len(files)}-{CAND.stat().st_mtime_ns if CAND.exists() else 0}"
    if CACHE.exists() and CACHE_STAMP.exists() and CACHE_STAMP.read_text() == stamp:
        return pd.read_parquet(CACHE)
    if CAND.exists():
        z = np.load(CAND)
        d = pd.DataFrame({"parcel_id": z["pid"], "score": probe_score(z["emb"], probe), "x": z["x"], "y": z["y"]})
        df = d.loc[d.groupby("parcel_id")["score"].idxmax()].reset_index(drop=True)
        df["lon"], df["lat"] = TO_4326.transform(df.x.values, df.y.values)
        df.to_parquet(CACHE)
        CACHE_STAMP.write_text(stamp)
        return df
    parts = []
    for f in files:
        z = np.load(f)
        if not len(z["pid"]):
            continue
        d = pd.DataFrame({"parcel_id": z["pid"], "score": probe_score(z["emb"], probe), "x": z["x"], "y": z["y"]})
        parts.append(d.loc[d.groupby("parcel_id")["score"].idxmax()])
    if not parts:
        raise SystemExit("no windows found - run detect.py first")
    df = pd.concat(parts, ignore_index=True).drop_duplicates("parcel_id")
    df["lon"], df["lat"] = TO_4326.transform(df.x.values, df.y.values)
    df = df.reset_index(drop=True)
    df.to_parquet(CACHE)
    CACHE_STAMP.write_text(stamp)
    return df


def build_candidates(probe=None):
    """Keep every window of each lot whose best score is >= CAND_MIN, so later retraining rescoring is fast."""
    probe = probe or load_probe()
    pid, x, y, emb = [], [], [], []
    for f in sorted(BLOCKS.glob("*.npz")):
        z = np.load(f)
        if not len(z["pid"]):
            continue
        s = probe_score(z["emb"], probe)
        best = pd.Series(s).groupby(z["pid"]).transform("max").values
        k = best >= CAND_MIN
        pid.append(z["pid"][k]); x.append(z["x"][k]); y.append(z["y"][k]); emb.append(z["emb"][k])
    np.savez(CAND, pid=np.concatenate(pid), x=np.concatenate(x), y=np.concatenate(y), emb=np.concatenate(emb))
    print(f"{sum(len(p) for p in pid):,} candidate windows -> {CAND}")


def merge_duplicates(flagged):
    """One court often straddles neighbouring lots; keep the highest-scoring lot in each cluster."""
    keep = []
    for r in flagged.sort_values("score", ascending=False).itertuples():
        if all(np.hypot(r.x - k.x, r.y - k.y) > DEDUPE_M for k in keep):
            keep.append(r)
    return pd.DataFrame(keep).drop(columns="Index") if keep else flagged.iloc[:0]


def maps_link(lat, lon):
    return f"https://www.google.com/maps/@?api=1&map_action=map&center={lat:.6f},{lon:.6f}&zoom=20&basemap=satellite"


def write_courts(courts):
    """Write data/courts.csv and data/courts.geojson (lat, lon, Google Maps link) for a courts DataFrame."""
    courts = courts.copy()
    courts["google_maps"] = [maps_link(a, b) for a, b in zip(courts.lat, courts.lon)]
    out = courts[["lat", "lon", "google_maps"]].round({"lat": 6, "lon": 6}).reset_index(drop=True)
    out.insert(0, "court_id", range(1, len(out) + 1))
    out.to_csv(DATA / "courts.csv", index=False)
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
              "properties": {"court_id": r.court_id, "google_maps": r.google_maps}} for r in out.itertuples()]
    (DATA / "courts.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print(f"{len(out)} courts -> data/courts.csv, data/courts.geojson")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, help="score at or above which a lot counts as a court")
    ap.add_argument("--confirmed", action="store_true",
                    help="write the courts list from the courts you confirmed in labels.csv (not from scores)")
    ap.add_argument("--rebuild-candidates", action="store_true", help="rebuild data/candidates.npz from all windows")
    args = ap.parse_args()
    if args.rebuild_candidates:
        if CAND.exists():
            CAND.unlink()
        build_candidates()
        return
    if args.confirmed:
        from common import LABELS
        lab = pd.read_csv(LABELS)
        lab = lab[lab.label == 1].reset_index(drop=True)
        lab["score"] = 1.0
        courts = merge_duplicates(lab[["parcel_id", "x", "y", "score"]]).copy()
        courts["lon"], courts["lat"] = TO_4326.transform(courts.x.values, courts.y.values)
        write_courts(courts.sort_values(["lat", "lon"]))
        return
    if args.threshold is None:
        ap.error("--threshold is required")

    df = best_per_lot()
    df["flag"] = df["score"] >= args.threshold
    df.drop(columns=["x", "y"]).round({"score": 4, "lat": 6, "lon": 6}).to_csv(DATA / "results.csv", index=False)
    courts = merge_duplicates(df[df.flag]).copy()
    courts["google_maps"] = [maps_link(a, b) for a, b in zip(courts.lat, courts.lon)]
    out = courts[["parcel_id", "lat", "lon", "score", "google_maps"]].round({"score": 4, "lat": 6, "lon": 6})
    out.to_csv(DATA / "courts.csv", index=False)
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
              "properties": {"parcel_id": int(r.parcel_id), "score": r.score, "google_maps": r.google_maps}}
             for r in out.itertuples()]
    (DATA / "courts.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print(f"{len(df):,} lots scored; {int(df.flag.sum())} flagged at >= {args.threshold}, "
          f"{len(out)} distinct courts after merging neighbours -> data/courts.csv, data/courts.geojson")


if __name__ == "__main__":
    main()
