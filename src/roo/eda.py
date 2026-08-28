"""Milestone 1 - establish the ground truth, then check it is usable.

This project only means anything if there is a true answer to score against.
Open Bandit Dataset provides one: ZOZOTOWN ran two policies on live traffic,
a uniform-random one and a Bernoulli Thompson Sampling one, and logged the
propensity of every action. So the online value of each policy is measured
directly from its own logs, and off-policy estimators computed on the random
logs can be checked against the BTS logs' measured CTR.

That only works if some assumptions actually hold, so this checks them rather
than citing them:

  ground truth   CTR of each policy, with a confidence interval. The interval
                 matters: CTR here is well under 1%, so "BTS beats random" is
                 a claim about a difference of a fraction of a percent, and
                 without the interval there is no way to know if that survives
                 sampling noise.
  overlap        off-policy evaluation needs the logging policy to give every
                 action a non-zero chance. Uniform random does by construction
                 - but it is worth confirming every item actually appears.
  same period    if the two policies ran at different times, any CTR gap
                 confounds policy with seasonality, and the "true answer" is
                 not a true answer at all.
  position       the logs have slots, and slot 1 is not slot 3. Ignoring that
                 silently averages three different questions together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "parquet"
REPORTS = ROOT / "reports"


def ci(clicks: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson interval for a proportion.

    Normal-approximation intervals are unreliable at CTRs this low; Wilson
    stays sane when p is near zero, which is the entire regime here.
    """
    if n == 0:
        return (float("nan"),) * 3
    p = clicks / n
    z = stats.norm.ppf(1 - alpha / 2)
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, centre - half, centre + half


def load(policy: str, campaign: str) -> pd.DataFrame:
    return pd.read_parquet(PARQUET / f"{policy}_{campaign}.parquet")


def describe(policy: str, campaign: str, df: pd.DataFrame) -> dict:
    p, lo, hi = ci(int(df.click.sum()), len(df))
    return {
        "policy": policy, "campaign": campaign,
        "rows": len(df), "clicks": int(df.click.sum()),
        "ctr": p, "ctr_lo": lo, "ctr_hi": hi,
        "n_items": int(df.item_id.nunique()),
        "positions": sorted(int(x) for x in df.position.unique()),
        "propensity_min": float(df.propensity_score.min()),
        "propensity_max": float(df.propensity_score.max()),
        "t_start": str(df.timestamp.min()), "t_end": str(df.timestamp.max()),
    }


def run(campaign: str = "all") -> dict:
    rnd, bts = load("random", campaign), load("bts", campaign)
    out = {"campaign": campaign,
           "random": describe("random", campaign, rnd),
           "bts": describe("bts", campaign, bts)}

    # --- the number every later estimator has to reproduce -----------------
    a, b = out["random"], out["bts"]
    lift = b["ctr"] / a["ctr"] - 1
    # difference of two proportions, normal approx on the difference
    se = np.sqrt(a["ctr"] * (1 - a["ctr"]) / a["rows"]
                 + b["ctr"] * (1 - b["ctr"]) / b["rows"])
    diff = b["ctr"] - a["ctr"]
    z = diff / se
    out["ground_truth"] = {
        "ctr_random": a["ctr"], "ctr_bts": b["ctr"],
        "absolute_diff": diff, "relative_lift": lift,
        "diff_se": float(se), "z": float(z),
        "p_value": float(2 * stats.norm.sf(abs(z))),
        "diff_ci95": [float(diff - 1.96 * se), float(diff + 1.96 * se)],
    }

    # --- assumption checks -------------------------------------------------
    n_items = max(a["n_items"], b["n_items"])
    expected = 1 / n_items
    uniform_ok = bool(np.allclose(rnd.propensity_score.unique(), expected, atol=1e-4))
    overlap_days = len(
        set(rnd.timestamp.dt.date.unique()) & set(bts.timestamp.dt.date.unique()))
    out["checks"] = {
        "n_items": n_items,
        "random_propensity_is_uniform": uniform_ok,
        "random_propensity_expected": expected,
        "random_propensity_observed": sorted(
            float(x) for x in rnd.propensity_score.unique()[:5]),
        "all_items_seen_by_random": int(rnd.item_id.nunique()) == n_items,
        "all_items_seen_by_bts": int(bts.item_id.nunique()) == n_items,
        "shared_days": overlap_days,
        "random_days": int(rnd.timestamp.dt.date.nunique()),
        "bts_days": int(bts.timestamp.dt.date.nunique()),
    }

    # --- position ----------------------------------------------------------
    pos = {}
    for name, df in (("random", rnd), ("bts", bts)):
        g = df.groupby("position").agg(n=("click", "size"), c=("click", "sum"))
        pos[name] = {int(k): {"n": int(r.n), "clicks": int(r.c),
                              "ctr": float(r.c / r.n)} for k, r in g.iterrows()}
    out["by_position"] = pos

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"eda_{campaign}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("campaign", nargs="?", default="all")
    o = run(p.parse_args().campaign)

    a, b, g, c = o["random"], o["bts"], o["ground_truth"], o["checks"]
    print(f"=== campaign '{o['campaign']}' ===")
    for d in (a, b):
        print(f"{d['policy']:7} {d['rows']:>10,} rows  {d['clicks']:>7,} clicks  "
              f"CTR {d['ctr']:.5f}  [{d['ctr_lo']:.5f}, {d['ctr_hi']:.5f}]")
    print(f"\nGROUND TRUTH  bts vs random: {g['relative_lift']:+.2%} relative, "
          f"{g['absolute_diff']:+.5f} absolute")
    print(f"              95% CI on the difference "
          f"[{g['diff_ci95'][0]:+.5f}, {g['diff_ci95'][1]:+.5f}]  p={g['p_value']:.2e}")
    print("\nassumption checks")
    for k, v in c.items():
        print(f"  {k:32} {v}")


if __name__ == "__main__":
    main()
