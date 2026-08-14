"""Milestone 2 - the naive offline evaluation, and how wrong it is.

The task is deliberately narrow so that it can be graded: using ONLY the
uniform-random logs, estimate the online CTR of the BTS policy. Milestone 1
measured that answer directly from BTS's own logs (0.00495), so every estimate
here has a right answer to be scored against.

Three naive estimators, each one a thing real teams actually ship:

  naive mean    ignore the policy entirely and quote the logged CTR. This is
                what "our historical CTR is 0.35%" means when someone says it
                about a system whose ranker has since changed.
  replay        keep only the logged rows where the logging policy happened to
                pick what the target policy would pick, and average their
                clicks. Intuitive, widely used, and it silently changes the
                population being averaged over.
  direct method fit a click model q(x, a) on the logs, then average its
                predictions under the target policy's action distribution.
                Its error is whatever the model gets wrong, and nothing in the
                offline metrics tells you how large that is.

A supervised ranking metric (AUC) is reported next to them on purpose. AUC
describes the click model; it says nothing about the value of the policy built
from it, and the point of this milestone is that the two can disagree.

Splitting is by TIME, not at random: the logs are 7 consecutive days, and a
random split lets the model see the future of the same campaign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from eda import ci
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "parquet"
REPORTS = ROOT / "reports"


def time_split(df: pd.DataFrame, train_days: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = np.sort(df.timestamp.dt.date.unique())
    cut = days[train_days]
    return df[df.timestamp.dt.date < cut], df[df.timestamp.dt.date >= cut]


def target_policy(bts: pd.DataFrame) -> pd.DataFrame:
    """pi_bts(a | position), estimated from BTS's own logs.

    BTS updates in batches rather than per user, so its action distribution is
    modelled as context-free but position-dependent - the same approximation
    the Open Bandit benchmark uses. Returns a (position, item_id) -> prob table
    that sums to 1 within each position.
    """
    counts = bts.groupby(["position", "item_id"]).size().rename("n").reset_index()
    counts["prob"] = counts.n / counts.groupby("position").n.transform("sum")
    return counts


def position_weights(df: pd.DataFrame) -> pd.Series:
    return df.position.value_counts(normalize=True).sort_index()


def q_empirical(train: pd.DataFrame) -> pd.DataFrame:
    """Non-parametric click model: CTR of each (item, position) on random logs.

    Under uniform logging this is an unbiased estimate of each cell's true
    reward, so it is the strongest simple direct method - which makes it the
    fair version of the naive approach rather than a straw man.
    """
    g = train.groupby(["position", "item_id"]).agg(n=("click", "size"),
                                                   c=("click", "sum")).reset_index()
    g["q"] = g.c / g.n
    return g


def dm_value(q: pd.DataFrame, pi: pd.DataFrame, w: pd.Series) -> float:
    """Direct method: sum_pos P(pos) sum_a pi(a|pos) q(a,pos)."""
    m = pi.merge(q[["position", "item_id", "q"]], on=["position", "item_id"], how="left")
    m["q"] = m.q.fillna(0.0)                       # unseen cell contributes nothing
    per_pos = m.assign(v=m.prob * m.q).groupby("position").v.sum()
    return float((per_pos * w.reindex(per_pos.index)).sum())


def replay_value(test: pd.DataFrame, pi: pd.DataFrame) -> tuple[float, int, tuple]:
    """Keep rows where the logged action is the target policy's argmax action.

    Returns a Wilson interval too. Replay throws away ~99% of the logs, so the
    obvious objection to any error it shows is that it is just noise. The
    interval is what settles that: if the true value sits outside it, the error
    is bias, not variance.
    """
    best = pi.loc[pi.groupby("position").prob.idxmax(), ["position", "item_id"]]
    best = set(map(tuple, best.to_numpy()))
    mask = [(p, i) in best for p, i in zip(test.position, test.item_id, strict=True)]
    sel = test[np.array(mask)]
    if not len(sel):
        return float("nan"), 0, (float("nan"), float("nan"))
    p_, lo, hi = ci(int(sel.click.sum()), len(sel))
    return p_, len(sel), (lo, hi)


def logistic_dm(train: pd.DataFrame, test: pd.DataFrame, pi: pd.DataFrame,
                w: pd.Series) -> dict:
    """A parametric click model, so AUC can be reported next to the value error."""
    cols = ["item_id", "position"]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    xtr = enc.fit_transform(train[cols].astype(str))
    xte = enc.transform(test[cols].astype(str))

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(xtr, train.click)
    p_te = clf.predict_proba(xte)[:, 1]

    # value under the target policy: predict q for every (position, item) cell
    cells = pi[["position", "item_id"]].copy()
    q = clf.predict_proba(enc.transform(cells[cols].astype(str)))[:, 1]
    cells["q"] = q
    return {
        "auc": float(roc_auc_score(test.click, p_te)),
        "log_loss": float(log_loss(test.click, p_te)),
        "value": dm_value(cells, pi, w),
    }


def run(campaign: str = "all", train_days: int = 5) -> dict:
    rnd = pd.read_parquet(PARQUET / f"random_{campaign}.parquet")
    bts = pd.read_parquet(PARQUET / f"bts_{campaign}.parquet")
    truth = float(bts.click.mean())

    tr, te = time_split(rnd, train_days)
    pi = target_policy(bts)
    w = position_weights(rnd)

    q = q_empirical(tr)
    v_dm = dm_value(q, pi, w)
    v_replay, n_replay, replay_ci = replay_value(te, pi)
    v_naive = float(rnd.click.mean())
    lg = logistic_dm(tr, te, pi, w)

    def err(v: float) -> dict:
        return {"value": v, "abs_error": v - truth, "rel_error": v / truth - 1}

    out = {
        "campaign": campaign, "truth_bts_ctr": truth,
        "truth_random_ctr": float(rnd.click.mean()),
        "train_rows": len(tr), "test_rows": len(te),
        "train_days": train_days,
        "estimators": {
            "naive_mean": err(v_naive),
            "replay": {**err(v_replay), "n_matched": n_replay,
                       "pct_of_test_kept": n_replay / len(te),
                       "ci95": list(replay_ci),
                       "truth_inside_ci": bool(replay_ci[0] <= truth <= replay_ci[1])},
            "direct_method_empirical": err(v_dm),
            "direct_method_logistic": {**err(lg["value"]), "auc": lg["auc"],
                                       "log_loss": lg["log_loss"]},
        },
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"baseline_{campaign}.json").write_text(json.dumps(out, indent=1))
    return out


def demo() -> None:
    """Self-check the estimator plumbing on data where the answer is arithmetic."""
    # two items, one position; item 1 always clicks, item 0 never does
    df = pd.DataFrame({
        "position": [1] * 8,
        "item_id": [0, 1] * 4,
        "click": [0, 1] * 4,
        "timestamp": pd.to_datetime(["2020-01-01"] * 8, utc=True),
    })
    q = q_empirical(df)
    assert q.set_index("item_id").q.to_dict() == {0: 0.0, 1: 1.0}

    w = pd.Series({1: 1.0})
    # a policy that always picks item 1 must be valued at exactly 1.0
    pi_all1 = pd.DataFrame({"position": [1, 1], "item_id": [0, 1], "prob": [0.0, 1.0]})
    assert dm_value(q, pi_all1, w) == 1.0
    # a 50/50 policy must be valued at exactly 0.5
    pi_half = pd.DataFrame({"position": [1, 1], "item_id": [0, 1], "prob": [0.5, 0.5]})
    assert dm_value(q, pi_half, w) == 0.5
    # replay against the always-item-1 policy keeps only the item-1 rows
    v, n, (lo, hi) = replay_value(df, pi_all1)
    assert (v, n) == (1.0, 4), (v, n)
    assert lo <= 1.0 <= hi, (lo, hi)
    print("self-check ok")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("campaign", nargs="?", default="all")
    p.add_argument("--train-days", type=int, default=5)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        demo()
        return

    o = run(a.campaign, a.train_days)
    t = o["truth_bts_ctr"]
    print(f"=== campaign '{o['campaign']}' ===")
    print(f"TRUTH   BTS online CTR {t:.5f}   (random {o['truth_random_ctr']:.5f})")
    print(f"train {o['train_rows']:,} rows / test {o['test_rows']:,} rows "
          f"(first {o['train_days']} days vs rest)\n")
    print(f"{'estimator':26} {'estimate':>10} {'rel. error':>12}  notes")
    for k, v in o["estimators"].items():
        note = ""
        if "auc" in v:
            note = f"AUC {v['auc']:.4f}, log loss {v['log_loss']:.5f}"
        if "n_matched" in v:
            inside = "truth INSIDE CI" if v["truth_inside_ci"] else "truth OUTSIDE CI -> real bias"
            note = (f"kept {v['n_matched']:,} rows ({v['pct_of_test_kept']:.2%}), "
                    f"CI [{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}] {inside}")
        print(f"{k:26} {v['value']:10.5f} {v['rel_error']:+11.1%}  {note}")


if __name__ == "__main__":
    main()
