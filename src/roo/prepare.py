"""Milestone 1a - turn 11 GB of CSV into a few hundred MB of Parquet.

The raw Open Bandit Dataset ships 89 columns per row, 80 of which are a
user-item affinity vector. Every downstream step re-reads the logs, and a
5.9 GB CSV scan costs minutes each time, so this converts once.

Kept: timestamp, item_id, position, click, propensity_score, and the four
user features. Dropped: the 80 affinity columns, which are all zero for a
large share of rows - that claim is checked here rather than assumed, and the
measured fill rate is written into the report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "open_bandit_dataset"
OUT = ROOT / "data" / "parquet"

CORE = ["timestamp", "item_id", "position", "click", "propensity_score"]
USER = [f"user_feature_{i}" for i in range(4)]
POLICIES = ("random", "bts")
CAMPAIGNS = ("all", "men", "women")


def affinity_fill_rate(csv: Path, n: int = 200_000) -> float:
    """What fraction of the 80 affinity columns are non-zero, on a sample.

    Worth measuring before dropping them: if they carried signal we would be
    throwing away the only user-item interaction features in the dataset.
    """
    cols = [f"user-item_affinity_{i}" for i in range(80)]
    head = pd.read_csv(csv, usecols=cols, nrows=n)
    return float((head.to_numpy() != 0).mean())


def convert(policy: str, campaign: str) -> dict:
    csv = RAW / policy / campaign / f"{campaign}.csv"
    dest = OUT / f"{policy}_{campaign}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = pd.read_csv(
        csv,
        usecols=CORE + USER,
        dtype={"item_id": "int16", "position": "int8", "click": "int8",
               "propensity_score": "float32"},
    )
    # Explicit parse, not read_csv(parse_dates=...): these are tz-aware ISO8601
    # strings and parse_dates left them as object, which Parquet then stored as
    # str. That surfaces much later as "Can only use .dt accessor" rather than
    # at read time, so convert here and assert it took.
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), "timestamp not parsed"
    for c in USER:                       # hashed categoricals, not numbers
        df[c] = df[c].astype("category")
    df.to_parquet(dest, index=False, compression="zstd")

    items = pd.read_csv(RAW / policy / campaign / "item_context.csv", index_col=0)
    items.to_parquet(OUT / f"items_{policy}_{campaign}.parquet", index=False)

    return {
        "policy": policy, "campaign": campaign,
        "rows": len(df), "n_items": int(df.item_id.nunique()),
        "csv_mb": round(csv.stat().st_size / 1e6, 1),
        "parquet_mb": round(dest.stat().st_size / 1e6, 1),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--campaigns", nargs="*", default=list(CAMPAIGNS))
    a = p.parse_args()

    rows = []
    for policy in POLICIES:
        for campaign in a.campaigns:
            r = convert(policy, campaign)
            rows.append(r)
            print(f"{policy:7} {campaign:6} {r['rows']:>10,} rows  "
                  f"{r['csv_mb']:>7.1f} MB csv -> {r['parquet_mb']:>6.1f} MB parquet  "
                  f"{r['seconds']:>5.1f}s", flush=True)

    fill = affinity_fill_rate(RAW / "random" / "all" / "all.csv")
    meta = {"tables": rows, "affinity_nonzero_fraction": fill,
            "total_parquet_mb": round(sum(r["parquet_mb"] for r in rows), 1)}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "prepare.json").write_text(json.dumps(meta, indent=1))
    print(f"\naffinity columns non-zero: {fill:.4%} of entries "
          f"-> {'carries signal, do not drop' if fill > 0.01 else 'effectively empty, dropped'}")
    print(f"total parquet: {meta['total_parquet_mb']} MB")


if __name__ == "__main__":
    main()
