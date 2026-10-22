"""Montage of candidate windows for eyeballing; also records the order for label.py.

    python review.py --top 48                 # highest-scoring lots
    python review.py --band 0.3 0.9 --top 48  # random lots scoring in a band (to look for misses)
    python review.py --skip 48 --top 48       # next page

Writes data/review.jpg and data/review_order.csv (rank, parcel_id, x, y, score).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from common import DATA, LABELS, RES, fetch_image
from count import best_per_lot, merge_duplicates

HALF = 20.0   # Web Mercator metres around the window centre shown in each tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=48)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--band", nargs=2, type=float, metavar=("LO", "HI"))
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--sorted", action="store_true", help="with --band: highest scores first instead of random order")
    ap.add_argument("--unseen", action="store_true", help="skip lots within 40 m of anything already labeled")
    ap.add_argument("--confirmed", action="store_true", help="show the courts you confirmed in labels.csv (merged)")
    ap.add_argument("--zoom", help="comma-separated ranks from the last review to show larger -> data/zoom.jpg")
    args = ap.parse_args()

    if args.zoom:
        order = pd.read_csv(DATA / "review_order.csv").set_index("rank")
        ranks = [int(v) for v in args.zoom.split(",")]
        with ThreadPoolExecutor(4) as ex:
            tiles = list(ex.map(lambda k: fetch_image((order.x[k] - 14, order.y[k] - 14, order.x[k] + 14, order.y[k] + 14))
                                .resize((380, 380)), ranks))
        cols = min(4, len(tiles))
        m = Image.new("RGB", (cols * 384, -(-len(tiles) // cols) * 384), (25, 25, 25))
        for n, (im, k) in enumerate(zip(tiles, ranks)):
            ImageDraw.Draw(im).text((4, 4), f"#{k}", fill=(255, 255, 0))
            m.paste(im, ((n % cols) * 384, (n // cols) * 384))
        m.save(DATA / "zoom.jpg", quality=90)
        return

    if args.confirmed:
        lab = pd.read_csv(LABELS)
        lab = lab[lab.label == 1].rename(columns={"parcel_id": "parcel_id"}).copy()
        lab["score"] = 1.0
        df = merge_duplicates(lab[["parcel_id", "x", "y", "score"]].reset_index(drop=True))
        print(f"{len(lab)} confirmed windows -> {len(df)} distinct courts")
        args.top = args.top or len(df)
        sel = df.iloc[args.skip:args.skip + args.top].reset_index(drop=True)
    else:
        df = best_per_lot()
    if not args.confirmed and args.unseen and LABELS.exists():
        lab = pd.read_csv(LABELS)
        from scipy.spatial import cKDTree
        near, _ = cKDTree(lab[["x", "y"]].values).query(df[["x", "y"]].values)
        df = df[near > 40].reset_index(drop=True)
    if args.confirmed:
        pass
    elif args.band:
        lo, hi = args.band
        df = df[(df.score >= lo) & (df.score < hi)]
        df = df.sort_values("score", ascending=False) if args.sorted else df.sample(frac=1, random_state=0)
        print(f"{len(df)} lots in band [{lo}, {hi})")
        sel = df.iloc[args.skip:args.skip + args.top]
    else:
        # collapse neighbouring lots that show the same court so each court appears once
        sel = merge_duplicates(df[df.score >= 0.02]).iloc[args.skip:args.skip + args.top]
    sel = sel.reset_index(drop=True)

    def tile(r):
        img = fetch_image((r.x - HALF, r.y - HALF, r.x + HALF, r.y + HALF))
        return img.resize((200, 200))

    with ThreadPoolExecutor(4) as ex:
        tiles = list(ex.map(tile, sel.itertuples()))
    size = 204
    rows = -(-len(tiles) // args.cols)
    m = Image.new("RGB", (args.cols * size, rows * size), (25, 25, 25))
    for n, (im, r) in enumerate(zip(tiles, sel.itertuples())):
        ImageDraw.Draw(im).text((4, 4), f"#{args.skip + n + 1} {r.score:.2f}", fill=(255, 255, 0))
        m.paste(im, ((n % args.cols) * size, (n // args.cols) * size))
    m.save(DATA / "review.jpg", quality=88)
    order = sel[["parcel_id", "x", "y", "score"]].copy()
    order.insert(0, "rank", np.arange(args.skip + 1, args.skip + 1 + len(order)))
    order.to_csv(DATA / "review_order.csv", index=False)
    print(f"showing ranks {args.skip + 1}-{args.skip + len(sel)} -> data/review.jpg")


if __name__ == "__main__":
    main()
