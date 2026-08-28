"""Demo for the off-policy evaluation gate.

The point of this app is not to show a number. It is to let you break the
estimate and watch the gate catch it, drag the sliders until the diagnostics
go red, and see the estimator return a confident, precise, wrong answer while
the gate refuses to report it.

Everything shown is precomputed on the full dataset (app_data/grid_all.json).
The gate's decision depends only on three scalars, unlogged target mass,
effective sample size, and interval width, so the thresholds are live: moving
them re-runs the real gate logic, not a cached answer. The true values are
revealed on purpose, because the whole project is about the offline estimate
being checkable against them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
GRID = ROOT / "app_data" / "grid_all.json"

st.set_page_config(page_title="Off-policy evaluation gate", page_icon="🚦",
                   layout="wide")


@st.cache_data
def load() -> list[dict]:
    return json.loads(GRID.read_text())["scenarios"]


def gate(s: dict, max_mass: float, min_ess: float, warn_ratio: float) -> tuple[str, list[str]]:
    """The same rules as harness.audit(), applied to precomputed diagnostics."""
    reasons = []
    if s["unlogged_target_mass"] > max_mass:
        reasons.append(
            f"**{s['unlogged_target_mass']:.1%}** of the target policy's probability "
            f"mass is on actions that never appear in these logs (limit "
            f"{max_mass:.1%}). Expect a bias of roughly that size, and the "
            f"confidence interval will *not* reflect it.")
    if s["ess"] < min_ess:
        reasons.append(
            f"effective sample size is **{s['ess']:,.0f}** (limit {min_ess:,.0f}). "
            f"{s['n_rows']:,} logged rows carry the weight of {s['ess']:,.0f}.")
    if reasons:
        return "refuse", reasons
    if s["ci_width_over_estimate"] > warn_ratio:
        return "warn", [f"interval is {s['ci_width_over_estimate']:.0%} as wide as "
                        f"the estimate itself"]
    return "ok", []


scen = load()
st.title("Off-policy evaluation, with a gate")
st.caption(
    "Estimating a recommender policy's online CTR from logged data. Every "
    "scenario below has a **known** true value, measured from a real A/B test, "
    "so each estimate can be graded rather than admired."
)

with st.sidebar:
    st.subheader("Gate thresholds")
    max_mass = st.slider("Max unlogged target mass", 0.0, 1.0, 0.01, 0.01,
                         help="Measured relationship: relative error tracks this "
                              "almost 1:1. Set it to the bias you can tolerate.")
    min_ess = st.select_slider("Min effective sample size",
                               options=[0, 100, 1_000, 10_000, 100_000], value=1_000)
    warn_ratio = st.slider("Warn if interval wider than", 0.05, 2.0, 0.5, 0.05)
    st.divider()
    st.caption(
        "Thresholds are not fitted. The mass limit is whatever bias you accept; "
        "the ESS floor is the usual ~1,000 rule of thumb for quoting a normal "
        "approximation."
    )

groups = {"direction": "Which policy did the logging?",
          "support": "Items missing from the logs",
          "sample_size": "How much data"}
tabs = st.tabs(list(groups.values()))

for tab, key in zip(tabs, groups, strict=True):
    with tab:
        rows = [s for s in scen if s["group"] == key]
        if key == "direction":
            pick = st.radio("Scenario", [s["name"] for s in rows], horizontal=True)
        else:
            labels = [s["name"] for s in rows]
            pick = st.select_slider("Scenario", options=labels, value=labels[0])
        s = next(x for x in rows if x["name"] == pick)

        status, reasons = gate(s, max_mass, min_ess, warn_ratio)
        c1, c2 = st.columns([2, 3])

        with c1:
            if status == "refuse":
                st.error("### REFUSED\nNo estimate reported.")
                for r in reasons:
                    st.markdown(f"- {r}")
            else:
                if status == "warn":
                    st.warning("### REPORTED, with a caveat")
                    for r in reasons:
                        st.markdown(f"- {r}")
                else:
                    st.success("### REPORTED")
                st.metric("Estimated CTR", f"{s['value']:.5f}",
                          f"{s['rel_error']:+.1%} vs truth")
                st.caption(f"95% CI [{s['ci95'][0]:.5f}, {s['ci95'][1]:.5f}]")

        with c2:
            st.markdown("**What the estimator would have returned**")
            st.metric("True online CTR (measured A/B)", f"{s['truth']:.5f}")
            st.metric("Estimate, ungated", f"{s['value']:.5f}",
                      f"{s['rel_error']:+.1%} error", delta_color="inverse")
            if status == "refuse" and abs(s["rel_error"]) > 0.1:
                st.error(f"Had the gate not fired, this run would have reported a "
                         f"number that is **{s['rel_error']:+.1%}** wrong, with a "
                         f"narrow confidence interval and nothing else to warn you.")

        st.markdown("**Diagnostics**")
        st.dataframe(pd.DataFrame([{
            "rows": f"{s['n_rows']:,}",
            "effective sample size": f"{s['ess']:,.0f}",
            "ESS %": f"{s['ess_frac']:.2%}",
            "max importance weight": f"{s['max_weight']:,.1f}",
            "unlogged target mass": f"{s['unlogged_target_mass']:.1%}",
            "CI width / estimate": f"{s['ci_width_over_estimate']:.0%}",
        }]), hide_index=True, use_container_width=True)

st.divider()
with st.expander("Why a gate, and what it does not do"):
    st.markdown("""
**The failure this is built for.** Off-policy evaluation breaks in a way that
looks fine from the outside: the estimate is precise, the confidence interval is
narrow, and both are wrong by up to 93%. Removing the top-CTR items from the
logs drives the error to −89.6% while *effective sample size rises* to 47.6%
the diagnostic everyone reaches for moves in the wrong direction, because the
weights that survive are beautifully well conditioned.

**What actually detects it** is free and needs no labels: the target policy's
probability mass on actions that never appear in the logs. Measured on this
dataset it tracks the error almost 1:1 to 17.7% mass → −19.0% error, 88.6% →
−89.6%.

**Limits, stated plainly.**
- Validated on 7 scenarios from one dataset: 6/7 correct decisions, **0 false
  accepts**, 1 conservative refusal. It has never yet reported a wrong number,
  which is the property that matters, but 7 scenarios is not a guarantee.
- The 1:1 mass→error relationship was measured with IPS. The gate defaults to
  SNIPS, whose self-normalisation damps the bias, so the gate is *conservative*
  when using SNIPS, that is the single incorrect decision above.
- Everything here assumes the logged propensities are correct. If they are
  wrong, every number on this page is wrong and no diagnostic here would know.
""")
