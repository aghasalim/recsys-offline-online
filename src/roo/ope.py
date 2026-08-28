"""Milestone 3 - off-policy estimators that use the logged propensities.

Milestone 2's naive estimators were wrong by -30% to +54% because they ignored
the one thing the logs actually record: the probability with which each action
was taken. Reweighting by that probability turns "what happened under the
logging policy" into "what would happen under the target policy".

  IPS    weight every logged reward by pi_target(a|x) / pi_logging(a|x).
         Unbiased, and textbook-notorious for its variance. I predicted in
         milestone 1 that the weights would be heavy-tailed here, and that was
         wrong: they top out at 9.64. Uniform logging gives every action
         probability 1/80, so a weight cannot exceed 80 * max pi_target, and
         BTS's most concentrated action only reaches 0.1205. The variance
         problem needs a logging policy that is itself concentrated, which
         this dataset does not provide from the random side.
  SNIPS  divide by the sum of weights rather than n. Slightly biased, usually
         much lower variance, and it cannot return a value outside the range
         of observed rewards - which plain IPS can.
  DR     combine a click model with an IPS correction on its residual. If
         either the model or the propensities are right, it is consistent.
         The model is fit OUT OF FOLD, because fitting q on the same rows it
         corrects is how DR quietly turns back into the direct method.

Reported next to every estimate:

  ESS    effective sample size, (sum w)^2 / sum w^2. Says how many of the
         1.37 million logged rows are really contributing. A small ESS means
         the estimate rests on a handful of impressions no matter how large
         the dataset looks.
  max w  the single largest importance weight. One row with weight 500 is a
         500-row opinion.

Standard errors are analytic rather than bootstrapped: IPS and DR are means of
per-row scores, so std/sqrt(n) is exact and costs nothing on 1.4M rows. SNIPS
is a ratio, so it gets the delta method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import q_empirical, target_policy, time_split

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "parquet"
REPORTS = ROOT / "reports"


def importance_weights(df: pd.DataFrame, pi: pd.DataFrame) -> np.ndarray:
    """pi_target(a|pos) / pi_logging(a|pos) for each logged row.

    pi_logging comes from the dataset's own propensity_score column rather than
    being assumed uniform - if the logging policy were not what we think, that
    is where it would show up.
    """
    key = pi.set_index(["position", "item_id"]).prob
    idx = pd.MultiIndex.from_arrays([df.position, df.item_id])
    p_target = key.reindex(idx).to_numpy()
    p_target = np.nan_to_num(p_target, nan=0.0)
    return p_target / df.propensity_score.to_numpy()


def ess(w: np.ndarray) -> float:
    """Effective sample size of a set of importance weights."""
    s = w.sum()
    return float(s * s / np.square(w).sum()) if s > 0 else 0.0


def ips(w: np.ndarray, r: np.ndarray) -> dict:
    s = w * r
    n = len(s)
    v = float(s.mean())
    se = float(s.std(ddof=1) / np.sqrt(n))
    return {"value": v, "se": se, "ci95": [v - 1.96 * se, v + 1.96 * se]}


def snips(w: np.ndarray, r: np.ndarray) -> dict:
    """Self-normalised IPS, with a delta-method standard error."""
    n = len(w)
    num, den = (w * r).mean(), w.mean()
    v = float(num / den)
    # var of ratio ~ (1/den^2) * var(w*r - v*w) / n
    resid = w * r - v * w
    se = float(resid.std(ddof=1) / (den * np.sqrt(n)))
    return {"value": v, "se": se, "ci95": [v - 1.96 * se, v + 1.96 * se]}


def dr(w: np.ndarray, r: np.ndarray, q_logged: np.ndarray, baseline: np.ndarray) -> dict:
    """Doubly robust: model prediction plus IPS-corrected residual."""
    s = baseline + w * (r - q_logged)
    n = len(s)
    v = float(s.mean())
    se = float(s.std(ddof=1) / np.sqrt(n))
    return {"value": v, "se": se, "ci95": [v - 1.96 * se, v + 1.96 * se]}


def q_lookup(q: pd.DataFrame, position: pd.Series, item: pd.Series) -> np.ndarray:
    key = q.set_index(["position", "item_id"]).q
    idx = pd.MultiIndex.from_arrays([position, item])
    return np.nan_to_num(key.reindex(idx).to_numpy(), nan=0.0)


def dm_baseline(q: pd.DataFrame, pi: pd.DataFrame, position: pd.Series) -> np.ndarray:
    """sum_a pi(a|pos) q(a,pos), evaluated per row via its position."""
    m = pi.merge(q[["position", "item_id", "q"]], on=["position", "item_id"], how="left")
    m["q"] = m.q.fillna(0.0)
    per_pos = m.assign(v=m.prob * m.q).groupby("position").v.sum()
    return per_pos.reindex(position).to_numpy()


def cross_fitted_dr(rnd: pd.DataFrame, pi: pd.DataFrame, w: np.ndarray) -> dict:
    """DR with 2-fold cross-fitting over time, so q is always out of fold.

    Fitting the click model on the same rows whose residual it then corrects
    biases DR toward the direct method - the correction term shrinks because
    the model has already seen those clicks.
    """
    a, b = time_split(rnd, train_days=4)
    parts = []
    for fit, evl in ((a, b), (b, a)):
        q = q_empirical(fit)
        m = rnd.index.isin(evl.index)
        parts.append((
            m,
            q_lookup(q, evl.position, evl.item_id),
            dm_baseline(q, pi, evl.position),
        ))
    q_logged = np.empty(len(rnd))
    base = np.empty(len(rnd))
    for m, ql, bl in parts:
        q_logged[m] = ql
        base[m] = bl
    return dr(w, rnd.click.to_numpy(), q_logged, base)


def clip_sweep(w: np.ndarray, r: np.ndarray, truth: float,
               caps: tuple[float, ...] = (1, 2, 5, 10, 25, 50, 100, np.inf)) -> list[dict]:
    """Clipping trades variance for bias. Show the whole curve, not one point."""
    rows = []
    for c in caps:
        wc = np.minimum(w, c)
        e = ips(wc, r)
        rows.append({"cap": None if np.isinf(c) else float(c),
                     "value": e["value"], "se": e["se"],
                     "rel_error": e["value"] / truth - 1,
                     "ess": ess(wc), "clipped_frac": float((w > c).mean())})
    return rows


def run(campaign: str = "all") -> dict:
    rnd = pd.read_parquet(PARQUET / f"random_{campaign}.parquet").reset_index(drop=True)
    bts = pd.read_parquet(PARQUET / f"bts_{campaign}.parquet")
    truth = float(bts.click.mean())

    pi = target_policy(bts)
    w = importance_weights(rnd, pi)
    r = rnd.click.to_numpy()

    est = {
        "ips": ips(w, r),
        "snips": snips(w, r),
        "dr_crossfit": cross_fitted_dr(rnd, pi, w),
    }
    # direct method on the same rows, for a like-for-like comparison
    q_all = q_empirical(rnd)
    v_dm = float(dm_baseline(q_all, pi, rnd.position).mean())
    est["direct_method"] = {"value": v_dm, "se": float("nan"),
                            "ci95": [float("nan"), float("nan")]}

    for v in est.values():
        v["rel_error"] = v["value"] / truth - 1
        lo, hi = v["ci95"]
        v["truth_inside_ci"] = bool(lo <= truth <= hi) if np.isfinite(lo) else None

    out = {
        "campaign": campaign, "truth_bts_ctr": truth, "n_rows": int(len(rnd)),
        "weights": {
            "mean": float(w.mean()), "max": float(w.max()),
            "p99": float(np.percentile(w, 99)), "frac_zero": float((w == 0).mean()),
            "ess": ess(w), "ess_frac": ess(w) / len(w),
        },
        "estimators": est,
        "clipping": clip_sweep(w, r, truth),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"ope_{campaign}.json").write_text(json.dumps(out, indent=1))
    return out


def demo() -> None:
    """Self-check on a two-arm problem whose true value is arithmetic."""
    # logging: uniform over 2 items. target: always item 1.
    # item 1 clicks with prob 1, item 0 with prob 0 -> true value of target = 1.0
    n = 10_000
    rng = np.random.default_rng(0)
    item = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "position": np.ones(n, dtype=int), "item_id": item,
        "click": item.astype(int), "propensity_score": np.full(n, 0.5),
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC"),
    })
    pi = pd.DataFrame({"position": [1, 1], "item_id": [0, 1], "prob": [0.0, 1.0]})
    w = importance_weights(df, pi)
    r = df.click.to_numpy()

    assert abs(ips(w, r)["value"] - 1.0) < 0.05, ips(w, r)
    assert abs(snips(w, r)["value"] - 1.0) < 0.05, snips(w, r)
    # weights are 0 or 2, so ESS is about half the rows
    assert 0.4 < ess(w) / n < 0.6, ess(w) / n

    # a target identical to the logging policy must recover the logged mean
    pi_same = pd.DataFrame({"position": [1, 1], "item_id": [0, 1], "prob": [0.5, 0.5]})
    w_same = importance_weights(df, pi_same)
    assert np.allclose(w_same, 1.0)
    assert abs(ips(w_same, r)["value"] - r.mean()) < 1e-9
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
    t, wt = o["truth_bts_ctr"], o["weights"]
    print(f"=== campaign '{o['campaign']}'   truth {t:.5f}   n={o['n_rows']:,} ===\n")
    print(f"{'estimator':16} {'estimate':>10} {'rel err':>9} {'95% CI':>22}  covers truth")
    for k, v in o["estimators"].items():
        lo, hi = v["ci95"]
        ci = f"[{lo:.5f}, {hi:.5f}]" if np.isfinite(lo) else "n/a"
        cov = "-" if v["truth_inside_ci"] is None else ("yes" if v["truth_inside_ci"] else "NO")
        print(f"{k:16} {v['value']:10.5f} {v['rel_error']:+8.1%} {ci:>22}  {cov}")

    print(f"\nweights: mean {wt['mean']:.3f}  p99 {wt['p99']:.1f}  max {wt['max']:.1f}  "
          f"zero {wt['frac_zero']:.1%}")
    print(f"ESS {wt['ess']:,.0f} of {o['n_rows']:,} rows ({wt['ess_frac']:.2%})")

    print(f"\n{'clip cap':>9} {'estimate':>10} {'rel err':>9} {'ESS':>12} {'% clipped':>10}")
    for c in o["clipping"]:
        cap = "none" if c["cap"] is None else f"{c['cap']:g}"
        print(f"{cap:>9} {c['value']:10.5f} {c['rel_error']:+8.1%} "
              f"{c['ess']:12,.0f} {c['clipped_frac']:9.2%}")


if __name__ == "__main__":
    main()
