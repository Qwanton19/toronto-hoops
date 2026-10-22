"""Shared helpers: paths, imagery export, GPU window extraction and the CLIP wrapper."""
import io
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from pyproj import Transformer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
BLOCKS = DATA / "windows"          # per-block window embeddings written by detect.py
PARCELS_RAW = DATA / "property-boundaries-4326.gpkg"
PARCELS = DATA / "residential.parquet"
PROBE = DATA / "probe.npz"
LABELS = DATA / "labels.csv"       # reviewed windows: x, y (EPSG:3857), label (1 court / 0 not)

EXPORT_URL = "https://gis.toronto.ca/arcgis/rest/services/basemap/cot_ortho/MapServer/export"

# Web Mercator metres per pixel. 0.15 is ~0.108 m on the ground at Toronto's latitude.
RES = 0.15
BLOCK = 300.0     # block edge in Web Mercator metres (2000 px at RES)
MARGIN = 45.0     # extra imagery around each block so edge windows are complete
WIN = 128         # sliding-window edge in px (~14 m on the ground)
STRIDE = 80       # window spacing in px
CLIP_PX = 224
MIN_COMPACT = 0.6   # lot area / min rotated rectangle area; drops slivers
MIN_SIDE_M = 7.0    # shortest lot side, Web Mercator metres

MODEL, PRETRAINED, DIM = "ViT-B-16-quickgelu", "openai", 512

TO_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
TO_4326 = Transformer.from_crs(3857, 4326, always_xy=True)


def fetch_image(bounds, res=RES, retries=4):
    """Export an orthophoto for bounds=(minx, miny, maxx, maxy) in EPSG:3857."""
    minx, miny, maxx, maxy = bounds
    w, h = round((maxx - minx) / res), round((maxy - miny) / res)
    params = dict(bbox=f"{minx},{miny},{maxx},{maxy}", bboxSR=3857, imageSR=3857,
                  size=f"{w},{h}", format="jpg", f="image")
    for attempt in range(retries):
        try:
            r = requests.get(EXPORT_URL, params=params, timeout=120)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                return Image.open(io.BytesIO(r.content)).convert("RGB")
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"export failed for {bounds}")


class Clip:
    """CLIP image encoder that crops sliding windows on the GPU (no per-window PIL work)."""

    def __init__(self, name=MODEL, pretrained=PRETRAINED):
        import open_clip
        import torch
        import torchvision
        self.torch, self.tv = torch, torchvision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        self.model = model.visual.to(self.device, self.dtype).eval()
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

    def embed_windows(self, img, img_bounds, points, flip=False, batch=256):
        """Embed WIN-px windows centred on EPSG:3857 `points` (n,2) inside `img`.

        Returns (n, DIM) float16 unit vectors. Points whose window leaves the image get NaN rows.
        """
        torch = self.torch
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        out = np.full((len(points), DIM), np.nan, np.float16)
        if not len(points):
            return out
        minx, _, _, maxy = img_bounds
        px = (points[:, 0] - minx) / RES
        py = (maxy - points[:, 1]) / RES
        ok = (px - WIN / 2 >= 0) & (py - WIN / 2 >= 0) & (px + WIN / 2 <= img.width) & (py + WIN / 2 <= img.height)
        idx = np.flatnonzero(ok)
        if not len(idx):
            return out
        t = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).to(self.device).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            for i in range(0, len(idx), batch):
                sel = idx[i:i + batch]
                boxes = torch.tensor(np.stack([px[sel] - WIN / 2, py[sel] - WIN / 2,
                                               px[sel] + WIN / 2, py[sel] + WIN / 2], 1),
                                     dtype=torch.float32, device=self.device)
                boxes = torch.cat([torch.zeros(len(sel), 1, device=self.device), boxes], 1)
                x = self.tv.ops.roi_align(t, boxes, CLIP_PX, spatial_scale=1.0, sampling_ratio=2, aligned=True)
                if flip:
                    x = x.flip(-1)
                x = ((x - self.mean) / self.std).to(self.dtype)
                e = self.model(x)
                e = e / e.norm(dim=-1, keepdim=True)
                out[sel] = e.float().cpu().numpy().astype(np.float16)
        return out


def load_parcels():
    import geopandas as gpd
    return gpd.read_parquet(PARCELS)


def load_probe():
    """Trained probe parameters (a dict of arrays), or None if not trained yet."""
    if not PROBE.exists():
        return None
    return dict(np.load(PROBE))


def probe_score(emb, probe):
    """Court probability: mean logit of an ensemble of small MLPs (W1 (K,d,h), b1 (K,h), W2 (K,h,1), b2 (K,1))."""
    x = emb.astype(np.float32)
    logit = 0.0
    for k in range(probe["W1"].shape[0]):
        h = np.maximum(x @ probe["W1"][k] + probe["b1"][k], 0)
        logit = logit + (h @ probe["W2"][k] + probe["b2"][k])[:, 0]
    return 1 / (1 + np.exp(-logit / probe["W1"].shape[0]))
