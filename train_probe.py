"""Train the court classifier (small MLP on CLIP window embeddings).

Training data:
  * positives: OpenStreetMap basketball courts, OSM courts that sit inside house lots (weighted up),
    and every court you confirmed with label.py (weighted up)
  * negatives: OSM tennis courts and swimming pools (look-alikes), random windows on residential
    lots, and every window you rejected with label.py (hard negatives)

    python train_probe.py [--holdout]

--holdout also reports how many known backyard courts the probe would have found had it not
seen them. The base embeddings are cached in data/train_cache.npz; label embeddings in
data/labels_emb.pkl.
"""
import argparse
import json
import pickle
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import shapely
from tqdm import tqdm

from common import (BLOCK, DATA, DIM, LABELS, PROBE, RES, TO_3857, Clip, fetch_image, load_parcels,
                    probe_score)
from detect import block_bounds, lot_mask

CACHE = DATA / "train_cache.npz"
LABEL_CACHE = DATA / "labels_emb.pkl"
OSM_NEG_CACHE = DATA / "osm_neg_cache.npz"
POOL_NEG_CACHE = DATA / "osm_neg_pool.npz"
SHIFTS = [(0, 0), (24, 0), (-24, 0), (0, 24), (0, -24)]   # px


def osm_points(name, pure_basketball=False):
    pts = []
    for e in json.load(open(DATA / f"osm_{name}.json")):
        if pure_basketball and e.get("tags", {}).get("sport") != "basketball":
            continue
        if e["type"] == "node":
            lon, lat = e["lon"], e["lat"]
        elif "center" in e:
            lon, lat = e["center"]["lon"], e["center"]["lat"]
        else:
            continue
        pts.append(TO_3857.transform(lon, lat))
    return pts


def embed_many(clip, pts, augment, desc):
    with ThreadPoolExecutor(3) as ex:
        imgs = list(tqdm(ex.map(lambda p: _fetch(p), pts), total=len(pts), desc=desc))
    return [_embed_fetched(clip, p, im, augment) for p, im in zip(pts, imgs)]


def _fetch(pt):
    mx, my = pt
    b = (mx - 60, my - 60, mx + 60, my + 60)
    try:
        return b, fetch_image(b)
    except RuntimeError:
        return None


def _embed_fetched(clip, pt, fetched, augment):
    if fetched is None:
        return np.zeros((0, DIM), np.float16)
    b, img = fetched
    mx, my = pt
    shifts = SHIFTS if augment else [(0, 0)]
    pts = [(mx + dx * RES, my + dy * RES) for dx, dy in shifts]
    parts = [clip.embed_windows(img, b, pts)]
    if augment:
        parts.append(clip.embed_windows(img, b, pts, flip=True))
    e = np.concatenate(parts)
    return e[~np.isnan(e[:, 0])]


def random_lot_embeddings(clip, n_blocks=150, per_block=40, seed=0):
    """Windows centred on random points of random residential lots across the city."""
    rng = random.Random(seed)
    parcels = load_parcels()
    geoms = parcels.geometry.values[lot_mask(parcels.geometry.values)]
    del parcels
    cen = shapely.centroid(geoms)
    ix = np.floor(shapely.get_x(cen) / BLOCK).astype(int)
    iy = np.floor(shapely.get_y(cen) / BLOCK).astype(int)
    blocks = {}
    for i, k in enumerate(zip(ix, iy)):
        blocks.setdefault(k, []).append(i)
    keys = list(blocks)
    rng.shuffle(keys)

    def one(k):
        b = block_bounds(*k)
        try:
            return k, b, fetch_image(b)
        except RuntimeError:
            return k, b, None

    out = []
    with ThreadPoolExecutor(3) as ex:
        for k, b, img in tqdm(ex.map(one, keys[:n_blocks]), total=n_blocks, desc="lot windows"):
            if img is None:
                continue
            pts = []
            for gi in rng.sample(blocks[k], min(per_block, len(blocks[k]))):
                geom = geoms[gi]
                x0, y0, x1, y1 = geom.bounds
                for _ in range(20):
                    px, py = rng.uniform(x0, x1), rng.uniform(y0, y1)
                    if shapely.contains_xy(geom, px, py):
                        pts.append((px, py))
                        break
            e = clip.embed_windows(img, b, pts)
            out.append(e[~np.isnan(e[:, 0])])
    return np.concatenate(out)


def build_base(clip):
    """kinds: 0 OSM court, 1 tennis/pool, 2 random lot window, 3 OSM court inside a house lot."""
    X, y, kind, grp = [], [], [], []

    def add(sets, label, k, off):
        for i, e in enumerate(sets):
            if len(e):
                X.append(e); y.extend([label] * len(e)); kind.extend([k] * len(e)); grp.extend([off + i] * len(e))

    pos = osm_points("basketball", pure_basketball=True)
    neg = osm_points("tennis")[:400] + osm_points("pool")[:400]
    print(f"{len(pos)} basketball courts, {len(neg)} tennis/pool look-alikes")
    add(embed_many(clip, pos, True, "OSM courts"), 1, 0, 0)
    add(embed_many(clip, neg, False, "look-alikes"), 0, 1, 10_000)
    bd = pd.read_csv(DATA / "osm_backyard_courts.csv")
    add(embed_many(clip, list(zip(bd.x, bd.y)), True, "backyard courts"), 1, 3, 100_000)
    lot = random_lot_embeddings(clip)
    X.append(lot); y.extend([0] * len(lot)); kind.extend([2] * len(lot)); grp.extend(range(200_000, 200_000 + len(lot)))
    np.savez(CACHE, X=np.concatenate(X), y=np.array(y), kind=np.array(kind), grp=np.array(grp))


def osm_negatives(clip, per_kind=None):
    """Windows on OSM parking lots, playgrounds and large buildings: look-alikes the model must reject."""
    if OSM_NEG_CACHE.exists():
        return np.load(OSM_NEG_CACHE)["X"].astype(np.float32)
    rng = random.Random(1)
    per_kind = per_kind or {"parking": 900, "playground": 500, "bigbuildings": 900}
    X = []
    for name, n in per_kind.items():
        pts = osm_points(name)
        rng.shuffle(pts)
        for e in embed_many(clip, pts[:n], False, f"osm {name}"):
            if len(e):
                X.append(e)
    X = np.concatenate(X)
    np.savez(OSM_NEG_CACHE, X=X)
    return X.astype(np.float32)


def label_embeddings(clip):
    """Embeddings for every window in labels.csv (positives augmented), cached by location."""
    if not LABELS.exists():
        return np.zeros((0, DIM), np.float32), np.zeros(0, int)
    cache = pickle.load(open(LABEL_CACHE, "rb")) if LABEL_CACHE.exists() else {}
    df = pd.read_csv(LABELS)
    todo = [(r.x, r.y, r.label) for r in df.itertuples() if (round(r.x, 1), round(r.y, 1), r.label) not in cache]
    if todo:
        for lab in (0, 1):
            pts = [(x, y) for x, y, l in todo if l == lab]
            for (x, y), e in zip(pts, embed_many(clip, pts, bool(lab), f"labels={lab}")):
                cache[(round(x, 1), round(y, 1), lab)] = e
        pickle.dump(cache, open(LABEL_CACHE, "wb"))
    X, y = [], []
    for r in df.itertuples():
        e = cache.get((round(r.x, 1), round(r.y, 1), r.label), np.zeros((0, DIM)))
        X.append(e); y += [r.label] * len(e)
    return np.concatenate(X).astype(np.float32), np.array(y)


def fit_one(X, y, w, epochs, hidden, seed):
    """One small MLP (d -> hidden -> 1), class-balanced, sample-weighted BCE. Returns (W1, b1, W2, b2)."""
    import torch
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    wt = torch.tensor(w, dtype=torch.float32)
    wt = wt * torch.where(yt == 1, (yt == 0).sum() / (yt == 1).sum(), torch.ones(()))
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(0.5),
                              torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-2)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(Xt)).split(512):
            opt.zero_grad()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[idx])[:, 0], yt[idx], reduction="none")
                    * wt[idx]).sum() / wt[idx].sum()
            loss.backward()
            opt.step()
    return (net[0].weight.detach().numpy().T.copy(), net[0].bias.detach().numpy().copy(),
            net[3].weight.detach().numpy().T.copy(), net[3].bias.detach().numpy().copy())


def fit(X, y, w, members=5, epochs=60, hidden=256):
    """Ensemble of `members` MLPs with different seeds; scores average their logits."""
    parts = [fit_one(X, y, w, epochs, hidden, seed) for seed in range(members)]
    return dict(W1=np.stack([p[0] for p in parts]), b1=np.stack([p[1] for p in parts]),
                W2=np.stack([p[2] for p in parts]), b2=np.stack([p[3] for p in parts]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true",
                    help="hold out half of the known backyard courts and report recall on them")
    args = ap.parse_args()

    clip = Clip()
    if not CACHE.exists():
        build_base(clip)
    z = np.load(CACHE)
    X, y, kind, grp = z["X"].astype(np.float32), z["y"], z["kind"], z["grp"]
    w = np.ones(len(y), np.float32)
    w[kind == 3] = 4.0
    on = osm_negatives(clip)
    if POOL_NEG_CACHE.exists():
        pools = np.load(POOL_NEG_CACHE)["X"].astype(np.float32)
    else:
        rng = random.Random(2)
        pts = osm_points("pool")
        rng.shuffle(pts)
        pools = np.concatenate([e for e in embed_many(clip, pts[:1200], False, "osm pools") if len(e)]).astype(np.float32)
        np.savez(POOL_NEG_CACHE, X=pools)
    on = np.concatenate([on, pools])
    X = np.concatenate([X, on]); y = np.concatenate([y, np.zeros(len(on), int)])
    w = np.concatenate([w, np.ones(len(on), np.float32)])
    kind = np.concatenate([kind, np.full(len(on), 5)]); grp = np.concatenate([grp, np.arange(len(on)) + 400_000])
    lx, ly = label_embeddings(clip)
    if len(lx):
        X = np.concatenate([X, lx]); y = np.concatenate([y, ly])
        w = np.concatenate([w, np.where(ly == 1, 4.0, 2.0).astype(np.float32)])
        kind = np.concatenate([kind, np.full(len(ly), 4)]); grp = np.concatenate([grp, np.arange(len(ly)) + 300_000])
    print(f"{int(y.sum())} positive / {int((1 - y).sum())} negative windows "
          f"({int((kind == 4).sum())} from your labels)")

    if args.holdout:
        held = (kind == 3) & (grp % 2 == 1)
        clf = fit(X[~held], y[~held], w[~held])
        p_held = probe_score(X[held], clf)
        for name, mask in [("random lot windows", kind == 2), ("your rejected windows", (kind == 4) & (y == 0))]:
            if mask.any():
                thr = np.quantile(probe_score(X[mask], clf), 0.99)
                print(f"vs {name}: threshold at 1% FPR = {thr:.2f}; held-out backyard court windows above it: "
                      f"{(p_held >= thr).mean():.0%} ({int(held.sum())} windows, {len(set(grp[held]))} courts)")
    # progress metric: average precision on a held-out fifth of the windows you labelled
    lab = np.flatnonzero(kind == 4)
    rng = np.random.default_rng(0)
    test = rng.choice(lab, size=len(lab) // 5, replace=False) if len(lab) > 50 else np.zeros(0, int)
    if len(test) and len(set(y[test])) == 2:
        from sklearn.metrics import average_precision_score
        keep = np.ones(len(y), bool)
        keep[test] = False
        m = fit(X[keep], y[keep], w[keep])
        print(f"held-out labelled windows: average precision {average_precision_score(y[test], probe_score(X[test], m)):.3f} "
              f"({int(y[test].sum())} courts / {int((1 - y[test]).sum())} not)")
    clf = fit(X, y, w)
    np.savez(PROBE, **clf)
    print(f"saved {PROBE}")


if __name__ == "__main__":
    main()
