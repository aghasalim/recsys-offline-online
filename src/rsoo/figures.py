"""Draw the additional README figures from reports/*.json.

The pipeline already writes eda_all.png, ope_all.png and stress_all.png. These
three cover the parts of the argument those do not: how the naive baselines fail,
how support erosion breaks the estimate, and what the decision gate actually did.

Reads the saved reports only -- no data download, no re-evaluation.

    python -m rsoo.figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def load(name: str) -> dict:
    return json.loads((REPORTS / f"{name}.json").read_text())


def baselines(out: Path) -> Path:
    """The two things people actually do, against the corrected estimators.

    The naive mean is 30% low because it averages a policy that was not the one
    being evaluated. Replay is 54% high and throws away 98.8% of the test rows to
    get there. Both are wrong in the direction that matters: one kills a good
    policy, the other ships a bad one.
    """
    base = load("baseline_all")
    ope = load("ope_all")
    truth = base["truth_bts_ctr"]

    entries = [
        ("naive mean", base["estimators"]["naive_mean"]["value"], "#b2182b", None),
        ("replay", base["estimators"]["replay"]["value"], "#b2182b",
         base["estimators"]["replay"]["ci95"]),
    ]
    for name in ("ips", "snips", "dr_crossfit"):
        est = ope["estimators"][name]
        entries.append((name.replace("_", " "), est["value"], "#1a9850", est["ci95"]))

    figure, ax = plt.subplots(figsize=(10, 4.8))
    positions = np.arange(len(entries))
    for index, (label, value, colour, ci) in enumerate(entries):
        ax.barh(index, value * 100, 0.55, color=colour, edgecolor="0.3", lw=0.5)
        if ci:
            ax.plot([ci[0] * 100, ci[1] * 100], [index, index], color="0.2", lw=2)
        error = (value - truth) / truth * 100
        ax.text(max(value, ci[1] if ci else value) * 100 + 0.03, index,
                f"{error:+.0f}%", va="center", fontsize=9,
                color="#b2182b" if abs(error) > 10 else "#1a9850")
    ax.axvline(truth * 100, color="0.2", ls="--", lw=1.6)
    ax.text(truth * 100, -0.55, "  true online CTR", fontsize=9, color="0.3",
            va="bottom")
    ax.set_yticks(positions)
    ax.set_yticklabels([e[0] for e in entries])
    ax.invert_yaxis()
    ax.set_xlabel("estimated CTR (%)")
    kept = base["estimators"]["replay"]["pct_of_test_kept"] * 100
    ax.set_title(
        "Green estimators land within 2% of the truth. Replay is 54% high after "
        f"discarding\n{100 - kept:.1f}% of the test rows to find matches.",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def support(out: Path) -> Path:
    """What happens as the logging policy stops covering the target policy.

    Dropping items from the logs moves probability mass onto actions that were
    never logged. The estimate degrades smoothly and the interval stops covering
    the truth, which is the failure mode worth being able to detect.
    """
    rows = load("stress_all")["broken_support"]
    dropped = [r["items_dropped"] for r in rows]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.4), sharex=True)
    covers = [r["covers_truth"] for r in rows]
    colours = ["#1a9850" if c else "#b2182b" for c in covers]
    left.bar(range(len(rows)), [abs(r["rel_error"]) * 100 for r in rows],
             0.6, color=colours, edgecolor="0.3", lw=0.5)
    left.set_xticks(range(len(rows)))
    left.set_xticklabels(dropped)
    left.set_ylabel("|relative error| (%)")
    left.set_title("red = the 95% interval no longer covers the truth", fontsize=10)
    left.spines[["top", "right"]].set_visible(False)

    right.plot(range(len(rows)), [r["ess_frac"] * 100 for r in rows], "o-",
               color="#2166ac", lw=2, label="effective sample size")
    right.plot(range(len(rows)),
               [r["target_mass_on_unlogged_actions"] * 100 for r in rows], "s--",
               color="#b2182b", lw=2, label="target mass on unlogged actions")
    right.set_xticks(range(len(rows)))
    right.set_xticklabels(dropped)
    right.set_ylabel("% ")
    right.set_title("and the diagnostic that sees it coming", fontsize=10)
    right.legend(frameon=False, fontsize=8)
    right.spines[["top", "right"]].set_visible(False)

    figure.suptitle("items dropped from the logs", fontsize=10, y=0.02,
                    color="0.35")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def gate(out: Path) -> Path:
    """What the decision gate did on each scenario.

    The deliverable is not an estimate, it is a ship / do-not-ship call with a
    stated tolerance. This is every scenario the harness was run on, and whether
    the gate got it right.
    """
    harness = load("harness_all")
    rows = harness["rows"]
    tolerance = harness["tolerance"]

    figure, ax = plt.subplots(figsize=(11, 4.8))
    positions = np.arange(len(rows))
    errors = [r["actual_rel_error"] * 100 for r in rows]
    colours = ["#1a9850" if r.get("acceptable") == r.get("reported") else "#b2182b"
               for r in rows]
    ax.barh(positions, errors, color=colours, edgecolor="0.3", lw=0.5)
    ax.axvline(tolerance * 100, color="0.35", ls="--", lw=1.3)
    ax.axvline(-tolerance * 100, color="0.35", ls="--", lw=1.3)
    ax.axvline(0, color="0.2", lw=1.0)
    ax.set_yticks(positions)
    ax.set_yticklabels([r["scenario"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("actual relative error (%)")
    correct = harness["decisions_correct"]
    ax.set_title(
        f"Dashed lines are the +/-{tolerance:.0%} tolerance. The gate made the "
        f"right call on {correct} of {harness['n_scenarios']} scenarios.",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    for path in (
        baselines(REPORTS / "baselines.png"),
        support(REPORTS / "support.png"),
        gate(REPORTS / "gate.png"),
    ):
        print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
