"""Record what you saw in the last review montage.

    python label.py --courts 1,2,3,16     # those ranks are courts; every other tile is not
    python label.py --courts 1,2 --only-courts   # add only the courts (e.g. when reviewing a low band)

Reads data/review_order.csv (written by review.py) and appends to data/labels.csv.
"""
import argparse

import pandas as pd

from common import LABELS, DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--courts", default="", help="comma-separated ranks in the last review that are real courts")
    ap.add_argument("--skip", default="", help="comma-separated ranks you are unsure about; left unlabeled")
    ap.add_argument("--only-courts", action="store_true", help="do not label the other tiles as negatives")
    args = ap.parse_args()
    order = pd.read_csv(DATA / "review_order.csv")
    courts = {int(v) for v in args.courts.split(",") if v}
    order["label"] = order["rank"].isin(courts).astype(int)
    order = order[~order["rank"].isin({int(v) for v in args.skip.split(",") if v})]
    if args.only_courts:
        order = order[order.label == 1]
    new = order[["parcel_id", "x", "y", "label"]]
    old = pd.read_csv(LABELS) if LABELS.exists() else new.iloc[:0]
    allrows = pd.concat([old, new]).drop_duplicates(["x", "y"], keep="last")
    allrows.to_csv(LABELS, index=False)
    print(f"labels: {int(allrows.label.sum())} courts, {int((1 - allrows.label).sum())} not courts")


if __name__ == "__main__":
    main()
