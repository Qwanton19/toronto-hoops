"""Grids of aerial tiles for every court in data/courts.csv, labelled with court_id, for review.

    python montage_courts.py [--per-page 72] [--cols 9]

Writes data/court_grids/courts_01.jpg, courts_02.jpg, ... (same tile size and layout as review.py).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image, ImageDraw

from common import DATA, TO_3857, fetch_image

HALF = 20.0   # Web Mercator metres around each court shown in a tile
TILE = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-page", type=int, default=72)
    ap.add_argument("--cols", type=int, default=9)
    args = ap.parse_args()

    courts = pd.read_csv(DATA / "courts.csv")
    xy = TO_3857.transform(courts.lon.values, courts.lat.values)
    courts["x"], courts["y"] = xy

    def tile(r):
        img = fetch_image((r.x - HALF, r.y - HALF, r.x + HALF, r.y + HALF)).resize((TILE, TILE))
        ImageDraw.Draw(img).text((4, 4), f"#{r.court_id}", fill=(255, 255, 0))
        return img

    out = DATA / "court_grids"
    out.mkdir(exist_ok=True)
    for old in out.glob("courts_*.jpg"):
        old.unlink()
    size = TILE + 4
    for n, start in enumerate(range(0, len(courts), args.per_page), 1):
        page = courts.iloc[start:start + args.per_page]
        with ThreadPoolExecutor(4) as ex:
            tiles = list(ex.map(tile, page.itertuples()))
        rows = -(-len(tiles) // args.cols)
        m = Image.new("RGB", (args.cols * size, rows * size), (25, 25, 25))
        for i, im in enumerate(tiles):
            m.paste(im, ((i % args.cols) * size, (i // args.cols) * size))
        path = out / f"courts_{n:02d}.jpg"
        m.save(path, quality=88)
        print(f"{path.name}: courts #{page.court_id.iloc[0]}-#{page.court_id.iloc[-1]}")


if __name__ == "__main__":
    main()
