"""Embed sliding windows over every residential lot, one imagery block at a time.

Each block writes data/windows/{ix}_{iy}.npz with one row per window (parcel id, centre in
EPSG:3857, CLIP embedding). Runs resume where they stopped, and because every embedding is
kept, the classifier can be retrained and everything rescored without re-downloading imagery.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shapely
from tqdm import tqdm

from common import (BLOCK, BLOCKS, MARGIN, MIN_COMPACT, MIN_SIDE_M, RES, STRIDE, TO_3857, Clip, fetch_image,
                    load_parcels)


def block_bounds(ix, iy):
    return (ix * BLOCK - MARGIN, iy * BLOCK - MARGIN, (ix + 1) * BLOCK + MARGIN, (iy + 1) * BLOCK + MARGIN)


def lot_mask(geoms):
    """Keep compact lots wide enough to be a house lot; drops slivers and odd shapes."""
    rect = shapely.minimum_rotated_rectangle(geoms)
    compact = shapely.area(geoms) / np.maximum(shapely.area(rect), 1e-9) >= MIN_COMPACT
    short = np.empty(len(geoms))
    for i, r in enumerate(rect):
        xs, ys = r.exterior.coords.xy
        short[i] = min(np.hypot(xs[1] - xs[0], ys[1] - ys[0]), np.hypot(xs[2] - xs[1], ys[2] - ys[1]))
    return compact & (short >= MIN_SIDE_M)


def lot_points(geom):
    """Grid of window centres (EPSG:3857) inside a lot; falls back to one interior point."""
    step = STRIDE * RES
    x0, y0, x1, y1 = geom.bounds
    xs = np.arange(x0 + step / 2, x1, step)
    ys = np.arange(y0 + step / 2, y1, step)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    inside = shapely.contains_xy(geom, gx, gy) if len(gx) else np.zeros(0, bool)
    if inside.any():
        return np.column_stack([gx[inside], gy[inside]])
    p = geom.representative_point()
    return np.array([[p.x, p.y]])


def process_block(clip, img, bounds, pids, geoms):
    pts, owner = [], []
    for pid, geom in zip(pids, geoms):
        p = lot_points(geom)
        pts.append(p)
        owner.append(np.full(len(p), pid, dtype=np.int64))
    pts, owner = np.concatenate(pts), np.concatenate(owner)
    emb = clip.embed_windows(img, bounds, pts)
    ok = ~np.isnan(emb[:, 0])
    return dict(pid=owner[ok], x=pts[ok, 0], y=pts[ok, 1], emb=emb[ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                    help="only process parcels inside this lon/lat box")
    ap.add_argument("--max-blocks", type=int, help="stop after N new blocks (trial runs)")
    ap.add_argument("--workers", type=int, default=3, help="parallel imagery downloads")
    args = ap.parse_args()

    parcels = load_parcels()
    if args.bbox:
        x0, y0 = TO_3857.transform(args.bbox[0], args.bbox[1])
        x1, y1 = TO_3857.transform(args.bbox[2], args.bbox[3])
        parcels = parcels.cx[x0:x1, y0:y1]
    geoms = parcels.geometry.values
    keep = lot_mask(geoms)
    pid = parcels["parcel_id"].values[keep]
    geoms = geoms[keep]
    cen = shapely.centroid(geoms)
    ix = np.floor(shapely.get_x(cen) / BLOCK).astype(int)
    iy = np.floor(shapely.get_y(cen) / BLOCK).astype(int)
    del parcels, cen
    order = np.lexsort((iy, ix))
    pid, geoms, ix, iy = pid[order], geoms[order], ix[order], iy[order]
    keys = ix.astype(np.int64) * 1_000_000 + (iy + 500_000)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    blocks = [((int(ix[s]), int(iy[s])), s, e) for s, e in zip(starts, ends)]
    BLOCKS.mkdir(parents=True, exist_ok=True)
    todo = [b for b in blocks if not (BLOCKS / f"{b[0][0]}_{b[0][1]}.npz").exists()]
    if args.max_blocks:
        todo = todo[:args.max_blocks]
    print(f"{len(pid):,} usable lots in {len(blocks)} blocks, {len(todo)} to do")

    clip = Clip()

    def download(b):
        try:
            return b, fetch_image(block_bounds(*b[0]))
        except RuntimeError as e:
            return b, e

    with ThreadPoolExecutor(args.workers) as pool:
        pending = collections.deque()
        it = iter(todo)
        for b in it:
            pending.append(pool.submit(download, b))
            if len(pending) >= args.workers * 2:
                break
        for _ in tqdm(range(len(todo)), unit="block"):
            b, img = pending.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(pool.submit(download, nxt))
            (kx, ky), s, e = b
            if isinstance(img, Exception):
                tqdm.write(f"skip {(kx, ky)}: {img}")
                continue
            out = process_block(clip, img, block_bounds(kx, ky), pid[s:e], geoms[s:e])
            np.savez(BLOCKS / f"{kx}_{ky}.npz", **out)


if __name__ == "__main__":
    main()
