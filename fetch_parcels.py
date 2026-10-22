"""Download Toronto property boundaries and keep house-sized residential lots.

The dataset has no land-use field (only PARCELID, FEATURE_TYPE, STATEDAREA), so
"residential" is approximated as ordinary parcels (FEATURE_TYPE == COMMON) whose
stated area is typical of a house lot. Schools, parks and big commercial sites
fall outside this range.
"""
import argparse

import geopandas as gpd
import requests
from tqdm import tqdm

from common import DATA, PARCELS, PARCELS_RAW

GPKG_URL = ("https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/1acaa8b0-f235-4df6-8305-02025ccdeb07/"
            "resource/7e9fff54-71c8-4958-b74a-2419f727c8a7/download/property-boundaries-4326.gpkg")


def download():
    DATA.mkdir(exist_ok=True)
    with requests.get(GPKG_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(PARCELS_RAW, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-area", type=float, default=150, help="min lot area, m^2")
    ap.add_argument("--max-area", type=float, default=1500, help="max lot area, m^2")
    ap.add_argument("--redownload", action="store_true")
    args = ap.parse_args()

    if args.redownload or not PARCELS_RAW.exists():
        download()
    g = gpd.read_file(PARCELS_RAW)
    g["area_m2"] = g["STATEDAREA"].str.extract(r"([\d.]+)")[0].astype(float)
    keep = (g["FEATURE_TYPE"] == "COMMON") & g["area_m2"].between(args.min_area, args.max_area)
    g = g[keep & g.geometry.notna()][["PARCELID", "area_m2", "geometry"]].to_crs(3857)
    g = g.rename(columns={"PARCELID": "parcel_id"}).reset_index(drop=True)
    g.to_parquet(PARCELS)
    print(f"{len(g):,} residential-sized parcels -> {PARCELS}")


if __name__ == "__main__":
    main()
