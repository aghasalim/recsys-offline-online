"""Milestone 5 - an evaluation harness that refuses to answer when it cannot.

Milestone 4 showed that off-policy evaluation fails in a specific, nasty way:
the estimate comes back precise, the confidence interval comes back narrow, and
both are wrong by 90%. Nothing about the output looks suspicious. The failure is
only visible in a diagnostic nobody computes.

So the deliverable of this project is not a model, it is a gate. `audit()`
returns a value only when the checks pass. When they do not, it returns
`value=None` and the reasons - because a withheld number cannot be pasted into
a slide, and a wrong one can.

THRESHOLDS, AND WHY THEY ARE NOT TUNED
Milestone 4 measured that the relative error tracks the target policy's
probability mass on unlogged actions almost exactly (17.7% mass -> -19.0%
error, 88.6% -> -89.6%). So the threshold is not a knob to be fitted, it is
whatever error you are willing to tolerate: allow 1% unlogged mass if you can
live with roughly 1% bias. The default is 1%.

The ESS floor is an absolute count, not a fraction. 0.16% ESS is fine on 12M
rows (20k effective) and fatal on 100k rows (160 effective), and only the
absolute number distinguishes those. 1,000 is the usual rule of thumb for a
normal approximation to be worth quoting.

Both thresholds are arguments. Neither was chosen by looking at which value
made the answers come out right.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ope import ess, ips, snips

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "parquet"
REPORTS = ROOT / "reports"


@dataclass
class Audit:
    status: str                       # "ok" | "warn" | "refuse"
    value: float | None               # withheld entirely when status == "refuse"
    ci95: list[float] | None
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def audit(logs: pd.DataFrame, pi: pd.DataFrame, *,
          max_unlogged_mass: float = 0.01,
          min_ess: float = 1000.0,
          warn_ci_ratio: float = 0.5,
          estimator: str = "snips") -> Audit:
    """Estimate the target policy's value, or refuse and say why.

    logs must have columns: position, item_id, click, propensity_score.
    pi must have columns: position, item_id, prob.
    """
    key = pi.set_index(["position", "item_id"]).prob
    idx = pd.MultiIndex.from_arrays([logs.position, logs.item_id])
    p_t = np.nan_to_num(key.reindex(idx).to_numpy(), nan=0.0)
    w = p_t / logs.propensity_score.to_numpy()
    r = logs.click.to_numpy()

    logged = set(map(tuple, logs[["position", "item_id"]].drop_duplicates().to_numpy()))
    mass_total = float(pi.prob.sum())
    unlogged = float(
        pi.loc[[(p, i) not in logged for p, i in zip(pi.position, pi.item_id, strict=True)],
               "prob"].sum() / mass_total)

    e = (snips if estimator == "snips" else ips)(w, r)
    n_eff = ess(w)
    ci_ratio = (e["ci95"][1] - e["ci95"][0]) / e["value"] if e["value"] > 0 else np.inf

    # For a 0.5% click rate, ESS on the weights is not the binding constraint -
    # the effective number of POSITIVE events is. 1,890 effective rows sounds
    # comfortable and is 9 expected clicks.
    eff_clicks = float(n_eff * r.mean())
    diagnostics = {
        "n_rows": int(len(logs)), "ess": n_eff, "ess_frac": n_eff / len(logs),
        "effective_clicks": eff_clicks,
        "max_weight": float(w.max()), "p99_weight": float(np.percentile(w, 99)),
        "unlogged_target_mass": unlogged,
        "ci_width_over_estimate": float(ci_ratio),
        "estimator": estimator,
    }

    reasons: list[str] = []
    if unlogged > max_unlogged_mass:
        reasons.append(
            f"{unlogged:.1%} of the target policy's probability mass is on actions "
            f"that never appear in these logs (limit {max_unlogged_mass:.1%}). "
            f"Expect a bias of roughly that size, and note the confidence interval "
            f"will NOT reflect it.")
    if n_eff < min_ess:
        reasons.append(
            f"effective sample size is {n_eff:,.0f} (limit {min_ess:,.0f}). "
            f"{len(logs):,} logged rows carry the weight of {n_eff:,.0f}.")

    if reasons:
        return Audit("refuse", None, None, reasons, diagnostics)

    warn: list[str] = []
    if ci_ratio > warn_ci_ratio:
        warn.append(f"interval is {ci_ratio:.0%} as wide as the estimate itself")
    return Audit("warn" if warn else "ok", e["value"], e["ci95"], warn, diagnostics)


# --------------------------------------------------------------------------
# Does the gate actually work? Scored against the known answers, not asserted.
# --------------------------------------------------------------------------

def scenarios(campaign: str = "all") -> list[dict]:
    from baseline import target_policy
    from stress import uniform_policy

    rnd = pd.read_parquet(PARQUET / f"random_{campaign}.parquet")
    bts = pd.read_parquet(PARQUET / f"bts_{campaign}.parquet")
    pi_bts = target_policy(bts)
    n_items = int(max(rnd.item_id.nunique(), bts.item_id.nunique()))
    pi_unif = uniform_policy(pi_bts, n_items)
    ctr = rnd.groupby("item_id").click.mean().sort_values(ascending=False)

    out = [
        {"name": "forward (random logs -> bts)", "logs": rnd, "pi": pi_bts,
         "truth": float(bts.click.mean())},
        {"name": "reverse (bts logs -> random)", "logs": bts, "pi": pi_unif,
         "truth": float(rnd.click.mean())},
    ]
    for k in (5, 20, 60):
        out.append({
            "name": f"broken support (top {k} items unlogged)",
            "logs": rnd[~rnd.item_id.isin(set(ctr.index[:k]))],
            "pi": pi_bts, "truth": float(bts.click.mean())})
    for frac, seed in ((0.0005, 0), (0.00005, 0)):
        sub = rnd.sample(frac=frac, random_state=seed)
        out.append({"name": f"tiny sample (n={len(sub):,})", "logs": sub,
                    "pi": pi_bts, "truth": float(bts.click.mean())})
    return out


def validate(campaign: str = "all", tolerance: float = 0.10) -> dict:
    """Run the gate over every scenario and check its decisions against truth.

    Scored on INTERVAL COVERAGE, not point-estimate error. An earlier version
    graded |point estimate - truth| against a tolerance and called a wide-but-
    honest interval a failure: at n=6,872 the estimate was 43% high, but its
    interval was [0.00251, 0.01164], which contains the truth, and the gate had
    already flagged it as 129% as wide as the estimate. Reporting a correct
    interval is the gate doing its job, so the criterion is:

      correct  = reported an interval that covers the truth,
                 OR refused one that would not have covered it.

    `tolerance` is kept only to report point-estimate quality alongside.
    """
    rows = []
    for sc in scenarios(campaign):
        a = audit(sc["logs"], sc["pi"])
        would_be = snips(
            np.nan_to_num(sc["pi"].set_index(["position", "item_id"]).prob.reindex(
                pd.MultiIndex.from_arrays([sc["logs"].position, sc["logs"].item_id])
            ).to_numpy(), nan=0.0) / sc["logs"].propensity_score.to_numpy(),
            sc["logs"].click.to_numpy())["value"]
        actual_err = would_be / sc["truth"] - 1
        full = audit(sc["logs"], sc["pi"], max_unlogged_mass=1.0, min_ess=0.0)
        lo, hi = full.ci95
        would_cover = bool(lo <= sc["truth"] <= hi)
        acceptable = would_cover
        reported = a.status != "refuse"
        rows.append({
            "scenario": sc["name"], "status": a.status,
            "reported_value": a.value, "would_be_value": would_be,
            "truth": sc["truth"], "actual_rel_error": actual_err,
            "acceptable": acceptable, "reported": reported,
            "would_be_ci": [lo, hi], "would_cover_truth": would_cover,
            "point_within_tolerance": bool(abs(actual_err) <= tolerance),
            "decision_correct": reported == acceptable,
            "unlogged_target_mass": a.diagnostics["unlogged_target_mass"],
            "ess": a.diagnostics["ess"],
            "reasons": a.reasons,
        })
    ok = sum(r["decision_correct"] for r in rows)
    out = {"campaign": campaign, "tolerance": tolerance,
           "decisions_correct": ok, "n_scenarios": len(rows), "rows": rows}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"harness_{campaign}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def demo() -> None:
    """Self-check: the gate must refuse exactly the cases it should."""
    n = 4000
    rng = np.random.default_rng(0)
    item = rng.integers(0, 4, n)
    logs = pd.DataFrame({
        "position": np.ones(n, dtype=int), "item_id": item,
        "click": (item == 3).astype(int), "propensity_score": np.full(n, 0.25)})
    pi = pd.DataFrame({"position": [1] * 4, "item_id": [0, 1, 2, 3], "prob": [0.25] * 4})

    good = audit(logs, pi)
    assert good.status in ("ok", "warn"), good
    assert good.value is not None and abs(good.value - 0.25) < 0.05, good

    # delete the only clicking item -> must refuse, and must withhold the number
    bad = audit(logs[logs.item_id != 3], pi)
    assert bad.status == "refuse", bad
    assert bad.value is None and bad.ci95 is None, bad
    assert any("never appear" in r for r in bad.reasons), bad.reasons

    # a tiny log -> must refuse on effective sample size
    tiny = audit(logs.head(40), pi)
    assert tiny.status == "refuse" and tiny.value is None, tiny
    assert any("effective sample size" in r for r in tiny.reasons), tiny.reasons
    print("self-check ok")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("campaign", nargs="?", default="all")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--export-grid", action="store_true")
    a = p.parse_args()
    if a.self_check:
        demo()
        return
    if a.export_grid:
        export_grid(a.campaign)
        return

    o = validate(a.campaign)
    print(f"=== evaluation gate, {o['n_scenarios']} scenarios "
          f"(tolerance +-{o['tolerance']:.0%}) ===\n")
    print(f"{'scenario':38} {'gate':>8} {'true err':>9} {'unlogged':>9} "
          f"{'ESS':>10}  correct")
    for r in o["rows"]:
        print(f"{r['scenario']:38} {r['status']:>8} {r['actual_rel_error']:+8.1%} "
              f"{r['unlogged_target_mass']:8.1%} {r['ess']:10,.0f}  "
              f"{'yes' if r['decision_correct'] else 'NO'}"
              f"{'' if r['would_cover_truth'] else '   (CI would miss truth)'}")
    print(f"\ncorrect decisions: {o['decisions_correct']}/{o['n_scenarios']}")
    for r in o["rows"]:
        if r["reasons"]:
            print(f"\n{r['scenario']} refused because:")
            for x in r["reasons"]:
                print(f"  - {x}")




def export_grid(campaign: str = "all") -> Path:
    """Precompute diagnostics for a grid of scenarios, for the demo app.

    The gate's decision is a function of three scalars - unlogged mass, ESS and
    interval width - so the app can re-run the gate at any threshold from this
    file alone. That keeps the demo honest (the numbers are the full-data ones
    from the README, not a subsample) without shipping 11 GB.
    """
    from baseline import target_policy
    from stress import uniform_policy

    rnd = pd.read_parquet(PARQUET / f"random_{campaign}.parquet")
    bts = pd.read_parquet(PARQUET / f"bts_{campaign}.parquet")
    pi_bts = target_policy(bts)
    n_items = int(max(rnd.item_id.nunique(), bts.item_id.nunique()))
    pi_unif = uniform_policy(pi_bts, n_items)
    ctr = rnd.groupby("item_id").click.mean().sort_values(ascending=False)
    truth_bts, truth_rnd = float(bts.click.mean()), float(rnd.click.mean())

    grid = []

    def add(name, group, logs, pi, truth, knob=None):
        a = audit(logs, pi, max_unlogged_mass=1.0, min_ess=0.0)   # diagnostics only
        grid.append({
            "name": name, "group": group, "knob": knob, "truth": truth,
            "value": a.value, "ci95": a.ci95, "rel_error": a.value / truth - 1,
            **a.diagnostics,
        })

    add("random logs -> evaluate BTS", "direction", rnd, pi_bts, truth_bts)
    add("BTS logs -> evaluate random", "direction", bts, pi_unif, truth_rnd)
    for k in (0, 2, 5, 10, 20, 40, 60):
        add(f"top {k} items missing from logs", "support",
            rnd[~rnd.item_id.isin(set(ctr.index[:k]))], pi_bts, truth_bts, knob=k)
    for f in (0.00005, 0.0005, 0.005, 0.05, 0.5, 1.0):
        sub = rnd.sample(frac=f, random_state=0) if f < 1.0 else rnd
        add(f"n = {len(sub):,}", "sample_size", sub, pi_bts, truth_bts, knob=len(sub))

    out = ROOT / "app_data" / f"grid_{campaign}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"campaign": campaign, "scenarios": grid}, indent=1))
    print(f"wrote {out}  ({len(grid)} scenarios, {out.stat().st_size/1e3:.1f} KB)")
    return out

if __name__ == "__main__":
    main()
