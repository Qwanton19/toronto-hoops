"""Regenerate the chart images used in README.md.

    python generate_report_assets.py

Writes docs/images/court_distribution_map.png (where the confirmed courts sit across the
city) and docs/images/labeling_progress.png (growth of the hand-labeled training set across
the review rounds below). Needs `pip install matplotlib`.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import DATA

OUT = DATA.parent / "docs" / "images"

# Cumulative (confirmed court, rejected non-court) window counts at four checkpoints across the
# review rounds: the first calibration pass, the point the full-city scan began, a mid-review
# checkpoint, and the final state of data/labels.csv.
CHECKPOINTS = [
    ("Initial calibration", 14, 39),
    ("Full-city scan begins", 126, 469),
    ("Mid-review", 215, 588),
    ("Final", 287, 1237),
]


def plot_map():
    courts = pd.read_csv(DATA / "courts.csv")
    lon_span = courts.lon.max() - courts.lon.min()
    lat_span = courts.lat.max() - courts.lat.min()
    aspect = 1 / np.cos(np.radians(courts.lat.mean()))  # longitude degrees are shorter than latitude here
    fig, ax = plt.subplots(figsize=(7, 7 * lat_span / (lon_span * aspect) + 1.2))
    ax.scatter(courts.lon, courts.lat, s=14, c="#2a7f62", edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_aspect(aspect)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{len(courts)} confirmed backyard basketball courts\nCity of Toronto")
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "court_distribution_map.png", dpi=150)
    plt.close(fig)


def plot_labeling_progress():
    names = [c[0] for c in CHECKPOINTS]
    courts = [c[1] for c in CHECKPOINTS]
    rejected = [c[2] for c in CHECKPOINTS]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, courts, "o-", color="#2a7f62", label="Confirmed courts", linewidth=2)
    ax.plot(x, rejected, "o-", color="#9a9a9a", label="Reviewed, rejected", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Cumulative labeled examples")
    ax.set_title("Growth of the hand-labeled training set\nacross review-and-retrain rounds")
    ax.legend()
    ax.grid(alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "labeling_progress.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plot_map()
    plot_labeling_progress()
    print(f"wrote {OUT / 'court_distribution_map.png'}")
    print(f"wrote {OUT / 'labeling_progress.png'}")
