"""Milestone 4 - the regimes where off-policy evaluation stops working.

Milestone 3 came out clean: every estimator within 2% of truth, weights capped
at 9.64, clipping pointless. That is not a general result about off-policy
evaluation, it is a fact about evaluating FROM uniform-random logs, which is
the easy direction. Three stress tests, each with a known answer:

  1. REVERSE DIRECTION
     Log with the concentrated policy (BTS) and evaluate the uniform one.
     Now the weight is 1/80 divided by BTS's probability, which is tiny for
     items BTS learned to avoid. This is the realistic setting: a company has
     logs from the ranker it already deployed, not from a random one.

  2. SHRINKING SAMPLE
     Subsample the logs and watch the interval. The question is not whether
     the estimate degrades - it must - but whether the reported confidence
     interval degrades honestly with it, i.e. still covers the truth.

  3. BROKEN SUPPORT
     Delete some items from the logging data entirely, so the target policy
     puts mass on actions that were never logged. This is the assumption that
     fails silently in production, when an item is new or was suppressed.

The third one carries the real lesson. Effective sample size is the diagnostic
everyone reaches for, and it CANNOT see a support violation: the weights that
survive look perfectly healthy, because the problematic actions contribute no
rows at all. The diagnostic that does see it is the target policy's probability
mass on unlogged actions, which is computable without any labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import target_policy
from ope import ess, ips, snips

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "parquet"
REPORTS = ROOT / "reports"


def uniform_policy(pi_like: pd.DataFrame, n_items: int) -> pd.DataFrame:
    """pi_random(a|pos) = 1/n_items, on the same (position, item) grid."""
    out = pi_like[["position", "item_id"]].copy()
    out["prob"] = 1.0 / n_items
    return out


def evaluate(logs: pd.DataFrame, pi: pd.DataFrame, truth: float,
             label: str) -> dict:
    """Run IPS/SNIPS on `logs` for target policy `pi`, and report diagnostics."""
    key = pi.set_index(["position", "item_id"]).prob
    idx = pd.MultiIndex.from_arrays([logs.position, logs.item_id])
    p_t = np.nan_to_num(key.reindex(idx).to_numpy(), nan=0.0)
    w = p_t / logs.propensity_score.to_numpy()
    r = logs.click.to_numpy()

    # the support diagnostic: how much target mass sits on actions never logged
    logged = set(map(tuple, logs[["position", "item_id"]].drop_duplicates().to_numpy()))
    unlogged = pi[~pi.apply(lambda x: (x.position, x.item_id) in logged, axis=1)]
    missing_mass = float(unlogged.prob.sum() / pi.prob.sum())

    a, b = ips(w, r), snips(w, r)
    for e in (a, b):
        e["rel_error"] = e["value"] / truth - 1
        e["covers_truth"] = bool(e["ci95"][0] <= truth <= e["ci95"][1])
    return {
        "label": label, "n": int(len(logs)), "truth": truth,
        "ips": a, "snips": b,
        "ess": ess(w), "ess_frac": ess(w) / len(logs),
        "max_weight": float(w.max()), "p99_weight": float(np.percentile(w, 99)),
        "target_mass_on_unlogged_actions": missing_mass,
    }


def reverse_direction(rnd: pd.DataFrame, bts: pd.DataFrame) -> dict:
    """Log with BTS (concentrated), evaluate uniform random (truth known)."""
    n_items = int(max(rnd.item_id.nunique(), bts.item_id.nunique()))
    pi_unif = uniform_policy(target_policy(bts), n_items)
    return evaluate(bts, pi_unif, float(rnd.click.mean()), "bts_logs -> evaluate random")


def sample_curve(logs: pd.DataFrame, pi: pd.DataFrame, truth: float,
                 fracs=(0.001, 0.005, 0.02, 0.1, 0.5, 1.0), seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for f in fracs:
        idx = rng.choice(len(logs), size=max(2, int(len(logs) * f)), replace=False)
        r = evaluate(logs.iloc[idx], pi, truth, f"frac={f}")
        rows.append({"frac": f, "n": r["n"], "value": r["ips"]["value"],
                     "rel_error": r["ips"]["rel_error"],
                     "ci_width": r["ips"]["ci95"][1] - r["ips"]["ci95"][0],
                     "covers_truth": r["ips"]["covers_truth"],
                     "ess_frac": r["ess_frac"]})
    return rows


def broken_support(rnd: pd.DataFrame, bts: pd.DataFrame,
                   drops=(0, 5, 10, 20, 40, 60)) -> list[dict]:
    """Remove the best items from the LOGS, keep them in the target policy.

    Dropping the highest-CTR items is the adversarial choice on purpose: a
    random drop mostly removes actions the target policy barely uses, which
    understates how badly this fails.
    """
    truth = float(bts.click.mean())
    pi = target_policy(bts)
    ctr = rnd.groupby("item_id").click.mean().sort_values(ascending=False)
    rows = []
    for k in drops:
        drop = set(ctr.index[:k])
        sub = rnd[~rnd.item_id.isin(drop)]
        r = evaluate(sub, pi, truth, f"dropped_top_{k}")
        rows.append({
            "items_dropped": k, "n": r["n"],
            "value": r["ips"]["value"], "rel_error": r["ips"]["rel_error"],
            "covers_truth": r["ips"]["covers_truth"],
            "ess_frac": r["ess_frac"],
            "target_mass_on_unlogged_actions": r["target_mass_on_unlogged_actions"],
        })
    return rows


def run(campaign: str = "all") -> dict:
    rnd = pd.read_parquet(PARQUET / f"random_{campaign}.parquet").reset_index(drop=True)
    bts = pd.read_parquet(PARQUET / f"bts_{campaign}.parquet").reset_index(drop=True)
    pi_bts = target_policy(bts)

    forward = evaluate(rnd, pi_bts, float(bts.click.mean()),
                       "random_logs -> evaluate bts")
    reverse = reverse_direction(rnd, bts)

    n_items = int(max(rnd.item_id.nunique(), bts.item_id.nunique()))
    pi_unif = uniform_policy(pi_bts, n_items)

    out = {
        "campaign": campaign,
        "forward": forward,
        "reverse": reverse,
        "sample_curve_reverse": sample_curve(bts, pi_unif, float(rnd.click.mean())),
        "broken_support": broken_support(rnd, bts),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"stress_{campaign}.json").write_text(json.dumps(out, indent=1))
    return out


def demo() -> None:
    """Self-check: a deliberately broken support must be caught by the support
    diagnostic, and must NOT be caught by effective sample size."""
    n = 4000
    rng = np.random.default_rng(0)
    item = rng.integers(0, 4, n)
    logs = pd.DataFrame({
        "position": np.ones(n, dtype=int), "item_id": item,
        "click": (item == 3).astype(int),          # only item 3 ever clicks
        "propensity_score": np.full(n, 0.25),
    })
    # target policy spreads over all four items
    pi = pd.DataFrame({"position": [1] * 4, "item_id": [0, 1, 2, 3],
                       "prob": [0.25] * 4})

    full = evaluate(logs, pi, truth=0.25, label="full")
    assert full["target_mass_on_unlogged_actions"] == 0.0
    assert abs(full["ips"]["rel_error"]) < 0.15, full["ips"]

    # now delete the only item that generates clicks from the logs
    broken = evaluate(logs[logs.item_id != 3], pi, truth=0.25, label="broken")
    assert broken["target_mass_on_unlogged_actions"] == 0.25, broken
    assert broken["ips"]["value"] == 0.0                       # badly wrong
    # ESS is blind to it: the surviving weights are perfectly well behaved
    assert broken["ess_frac"] > 0.9, broken["ess_frac"]
    print("self-check ok")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("campaign", nargs="?", default="all")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        demo()
        return

    o = run(a.campaign)
    print("=== direction matters ===")
    print(f"{'setting':32} {'truth':>8} {'IPS':>9} {'rel err':>9} {'max w':>9} {'ESS':>8}")
    for k in ("forward", "reverse"):
        d = o[k]
        print(f"{d['label']:32} {d['truth']:8.5f} {d['ips']['value']:9.5f} "
              f"{d['ips']['rel_error']:+8.1%} {d['max_weight']:9.1f} {d['ess_frac']:7.2%}")

    print("\n=== shrinking sample (reverse direction) ===")
    print(f"{'frac':>7} {'n':>12} {'rel err':>9} {'CI width':>10} {'ESS':>8}  covers")
    for c in o["sample_curve_reverse"]:
        print(f"{c['frac']:7g} {c['n']:12,} {c['rel_error']:+8.1%} "
              f"{c['ci_width']:10.5f} {c['ess_frac']:7.2%}  "
              f"{'yes' if c['covers_truth'] else 'NO'}")

    print("\n=== broken support: ESS cannot see it ===")
    print(f"{'dropped':>8} {'rel err':>9} {'ESS':>8} {'unlogged mass':>14}  covers")
    for c in o["broken_support"]:
        print(f"{c['items_dropped']:8} {c['rel_error']:+8.1%} {c['ess_frac']:7.2%} "
              f"{c['target_mass_on_unlogged_actions']:13.1%}  "
              f"{'yes' if c['covers_truth'] else 'NO'}")


if __name__ == "__main__":
    main()
