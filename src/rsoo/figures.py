"""Draw every README figure, from the saved reports rather than the raw logs.

Each figure here reads reports/*.json and app_data/grid_all.json, so the whole
set redraws on a machine that never downloaded the 11 GB dataset, and no figure
can disagree with the numbers quoted in the README. The pipeline steps write the
JSON; this writes the pictures.

    python src/rsoo/figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

from style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

# Red for the estimates you should not trust and green for the ones you can,
# because that is how the README and the pipeline figures already talk about
# them. Anything with no meaning attached takes a colour from the palette.
BAD, GOOD = "#b2182b", "#1a9850"
DIAG = PALETTE[0]


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
        ("naive mean", base["estimators"]["naive_mean"]["value"], BAD, None),
        ("replay", base["estimators"]["replay"]["value"], BAD,
         base["estimators"]["replay"]["ci95"]),
    ]
    labels = {"ips": "IPS", "snips": "SNIPS", "dr_crossfit": "DR cross-fit"}
    for name, label in labels.items():
        est = ope["estimators"][name]
        entries.append((label, est["value"], GOOD, est["ci95"]))

    figure, ax = plt.subplots(figsize=(9.8, 4.6))
    right = max((e[3][1] if e[3] else e[1]) for e in entries) * 100
    for index, (_label, value, colour, ci) in enumerate(entries):
        ax.barh(index, value * 100, 0.55, color=colour, edgecolor="0.3", lw=0.5)
        if ci:
            ax.plot([ci[0] * 100, ci[1] * 100], [index, index], color="0.15", lw=2)
        error = (value - truth) / truth * 100
        ax.text(max(value, ci[1] if ci else value) * 100 + right * 0.02, index,
                f"{error:+.0f}%", va="center", fontsize=9.5,
                color=BAD if abs(error) > 10 else GOOD)

    ax.axvline(truth * 100, color="0.2", ls="--", lw=1.4)
    ax.set_yticks(np.arange(len(entries)))
    ax.set_yticklabels([e[0] for e in entries])
    ax.set_xlim(0, right * 1.13)
    ax.set_ylim(len(entries) - 0.35, -0.6)
    ax.text(truth * 100 + right * 0.01, len(entries) - 0.45,
            f"measured online CTR, {truth * 100:.3f}%", fontsize=9, color="0.3",
            va="center", ha="left")
    ax.set_xlabel("estimated click-through rate (%)")
    kept = base["estimators"]["replay"]["pct_of_test_kept"] * 100
    ax.text(right * 0.02, 1, f"kept {kept:.2f}% of the test rows", color="white",
            fontsize=9, va="center", ha="left")
    titled(ax, "The two shortcuts miss the truth in opposite directions",
           "BTS click-through rate from 1.37M uniform-random rows, black lines are the "
           "95% intervals")
    figure.tight_layout()
    figure.savefig(out)
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
    x = np.arange(len(rows))

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.6, 4.8), sharex=True)

    covers = [r["covers_truth"] for r in rows]
    left.bar(x, [abs(r["rel_error"]) * 100 for r in rows], 0.6,
             color=[GOOD if c else BAD for c in covers], edgecolor="0.3", lw=0.5)
    left.set_ylabel("absolute relative error (%)")
    left.set_ylim(0, 105)
    left.legend(handles=[Patch(facecolor=GOOD, label="95% interval covers the truth"),
                         Patch(facecolor=BAD, label="it does not")],
                loc="upper left", bbox_to_anchor=(0.0, 0.97))
    titled(left, "Five missing items are enough to lose the truth",
           "IPS on the random logs, highest-CTR items deleted from the logs "
           "and left in the target policy")

    right.plot(x, [r["ess_frac"] * 100 for r in rows], "o-", color=DIAG,
               label="effective sample size, share of rows")
    right.plot(x, [r["target_mass_on_unlogged_actions"] * 100 for r in rows], "s--",
               color=BAD, label="target mass on actions the logs never show")
    right.set_ylabel("share (%)")
    right.set_ylim(-4, 105)
    right.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86))
    titled(right, "The usual diagnostic moves the wrong way",
           "at 40 items removed the error is -90% and ESS has nearly doubled")

    for ax in (left, right):
        ax.set_xticks(x)
        ax.set_xticklabels(dropped)
        ax.set_xlabel("items withheld from the logs (count)")
    figure.tight_layout()
    figure.savefig(out)
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
    tolerance = harness["tolerance"] * 100

    figure, ax = plt.subplots(figsize=(11.2, 5.0))
    y = np.arange(len(rows))
    errors = [r["actual_rel_error"] * 100 for r in rows]
    ax.barh(y, errors, 0.6, edgecolor="0.3", lw=0.5,
            color=[GOOD if r["decision_correct"] else BAD for r in rows])
    ax.axvline(tolerance, color="0.35", ls="--", lw=1.3)
    ax.axvline(-tolerance, color="0.35", ls="--", lw=1.3)
    ax.axvline(0, color="0.2", lw=1.0)

    ax.set_xlim(-112, 46)
    ax.set_ylim(len(rows) - 0.4, -0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([r["scenario"] for r in rows], fontsize=9)
    for index, row in enumerate(rows):
        note = "" if row["decision_correct"] else ", wrongly"
        ax.text(12, index, f"gate said {row['status']}{note}", va="center", fontsize=9,
                color="0.25")
    ax.text(-tolerance - 2, len(rows) - 0.5, f"+/-{tolerance:.0f}% tolerance",
            fontsize=9, color="0.35", ha="right", va="center")
    ax.set_xlabel("true relative error of the estimate (%)")
    titled(ax, f"The gate made the right call on {harness['decisions_correct']} of "
               f"{harness['n_scenarios']} scenarios",
           "green where the gate's report-or-refuse call matched whether its interval "
           "would have covered the truth")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def _upto(x, y, cut):
    """The polyline (x, y) clipped at cut, with the cut segment interpolated."""
    k = int(np.searchsorted(x, cut, side="right"))
    if k == len(x):
        return x, y
    return np.append(x[:k], cut), np.append(y[:k], np.interp(cut, x, y))


def anim_support(out: Path, per_step: int = 11, hold: int = 22, fps: int = 15) -> Path:
    """The support sweep replayed, estimate on the left, diagnostics on the right.

    Reads the committed grid the demo app ships, so nothing is estimated here and
    the GIF is the same on every run. The point is the two panels together: the
    interval on the left stays narrow and stops containing the truth, while the
    effective sample size on the right refuses to react.
    """
    grid = json.loads((ROOT / "app_data" / "grid_all.json").read_text())
    rows = sorted((s for s in grid["scenarios"] if s["group"] == "support"),
                  key=lambda s: s["knob"])
    truth = rows[0]["truth"] * 100
    x = np.arange(len(rows), dtype=float)
    value = np.array([r["value"] for r in rows]) * 100
    lo = np.array([r["ci95"][0] for r in rows]) * 100
    hi = np.array([r["ci95"][1] for r in rows]) * 100
    ess = np.array([r["ess_frac"] for r in rows]) * 100
    mass = np.array([r["unlogged_target_mass"] for r in rows]) * 100
    error = np.array([r["rel_error"] for r in rows]) * 100

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    left.axhline(truth, color="0.2", ls="--", lw=1.4)
    left.text(len(rows) - 1, truth + 0.02, "measured online CTR of BTS", fontsize=9,
              color="0.3", ha="right", va="bottom")
    left.set_ylim(0, truth * 1.35)
    left.set_ylabel("estimated click-through rate (%)")
    titled(left, "The estimate walks away and never says so",
           "SNIPS with its 95% interval, one point per sweep step")

    right.set_ylim(-4, 105)
    right.set_ylabel("share (%)")
    titled(right, "One diagnostic follows it, the other does not",
           "unlogged target mass tracks the error, effective sample size does not")

    for ax in (left, right):
        ax.set_xlim(-0.35, len(rows) - 0.65)
        ax.set_xticks(x)
        ax.set_xticklabels([r["knob"] for r in rows])
        ax.set_xlabel("items withheld from the logs (count)")

    art = {
        "line": left.plot([], [], color=PALETTE[4], lw=2.2, zorder=3)[0],
        "head": left.plot([], [], "o", markersize=7, color=PALETTE[4], zorder=5)[0],
        "readout": left.text(0.02, 0.03, "", transform=left.transAxes, fontsize=9.5,
                             color="0.3", ha="left", va="bottom"),
        "ess": right.plot([], [], "o-", color=DIAG, zorder=3,
                          label="effective sample size, share of rows")[0],
        "mass": right.plot([], [], "s--", color=BAD, zorder=3,
                           label="target mass on actions the logs never show")[0],
    }
    right.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86))
    for i in range(len(rows)):
        colour = GOOD if lo[i] <= truth <= hi[i] else BAD
        art[f"ci{i}"] = left.plot([], [], color=colour, lw=8, alpha=0.6,
                                  solid_capstyle="butt", zorder=2)[0]
        art[f"pt{i}"] = left.plot([], [], "o", markersize=4.5, color=colour, zorder=4)[0]
    left.legend(handles=[Patch(facecolor=GOOD, alpha=0.6, label="interval covers the truth"),
                         Patch(facecolor=BAD, alpha=0.6, label="it does not")],
                loc="lower left", bbox_to_anchor=(0.0, 0.09))
    figure.tight_layout()

    frames = per_step * (len(rows) - 1) + 1
    cuts = np.linspace(0, len(rows) - 1, frames)

    def draw(i):
        cut = cuts[min(i, frames - 1)]
        cx, cy = _upto(x, value, cut)
        art["line"].set_data(cx, cy)
        art["head"].set_data(cx[-1:], cy[-1:])
        for j in range(len(rows)):
            shown = [j, j] if x[j] <= cut else []
            art[f"ci{j}"].set_data(shown, [lo[j], hi[j]] if shown else [])
            art[f"pt{j}"].set_data(shown[:1], [value[j]] if shown else [])
        last = int(np.searchsorted(x, cut, side="right")) - 1
        art["readout"].set_text(f"{rows[last]['knob']} items withheld, "
                                f"error {error[last]:+.0f}%, "
                                f"unlogged mass {mass[last]:.0f}%")
        for key, series in (("ess", ess), ("mass", mass)):
            k = last + 1
            art[key].set_data(x[:k], series[:k])
        return list(art.values())

    anim = FuncAnimation(figure, draw, frames=frames + hold, interval=1000 // fps,
                         blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    _shrink_gif(out)
    return out


def _shrink_gif(path: Path, colours: int = 64) -> None:
    """Rewrite the GIF on one shared palette, which roughly halves the file.

    PillowWriter gives every frame its own full palette. Consecutive frames here
    differ in a few hundred pixels, so one palette taken from a middle frame
    covers all of them and lets the encoder store only what changed. Disposal is
    left unset on purpose: disposal=2 forces a full redraw per frame and made the
    file bigger.
    """
    from PIL import Image

    source = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(source.convert("RGB"))
            durations.append(source.info.get("duration", 62))
            source.seek(source.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(colours, method=Image.Quantize.MEDIANCUT)
    quantised = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    quantised[0].save(path, save_all=True, append_images=quantised[1:], loop=0,
                      duration=durations, optimize=True)


def ground_truth(out: Path) -> Path:
    """The measured answer every later estimator is graded against.

    Both policies ran on live traffic over the same seven days, so the gap
    between them is a measurement rather than a model output. The right panel is
    the check that keeps it honest: the gap holds in all three slots, so it is
    not one position doing all the work.
    """
    o = load("eda_all")
    truth = o["ground_truth"]
    keys = ("random", "bts")
    names = ["uniform random", "Bernoulli TS"]
    colours = [PALETTE[5], PALETTE[0]]
    ctr = [o[k]["ctr"] * 100 for k in keys]
    err = [[(o[k]["ctr"] - o[k]["ctr_lo"]) * 100 for k in keys],
           [(o[k]["ctr_hi"] - o[k]["ctr"]) * 100 for k in keys]]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.2, 4.7), sharey=True)
    left.bar(names, ctr, 0.5, color=colours, edgecolor="0.3", lw=0.5, yerr=err,
             capsize=4, error_kw={"ecolor": "0.15", "elinewidth": 1.6})
    for index, (value, key) in enumerate(zip(ctr, keys, strict=True)):
        left.text(index, value + max(ctr) * 0.07, f"{value:.3f}%", ha="center",
                  fontsize=9.5, color="0.25")
        left.text(index, max(ctr) * 0.04, f"{o[key]['rows']:,} impressions",
                  ha="center", fontsize=9, color="white")
    left.set_ylim(0, max(ctr) * 1.35)
    left.set_ylabel("click-through rate (%)")
    titled(left, f"Bernoulli TS is {truth['relative_lift']:.1%} better",
           f"live traffic, same 7 days, p = {truth['p_value']:.1e}, "
           "bars are 95% Wilson intervals")

    for name, key, colour in zip(names, keys, colours, strict=True):
        by_slot = o["by_position"][key]
        slots = sorted(int(s) for s in by_slot)
        right.plot(slots, [by_slot[str(s)]["ctr"] * 100 for s in slots], "o-",
                   color=colour, label=name)
    right.set_xticks(slots)
    right.set_xlabel("slot on the page")
    right.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02))
    titled(right, "The gap holds in every slot",
           "slot 1 is not slot 3, so the check is done per slot")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def corrected(out: Path) -> Path:
    """The propensity correction working, and clipping making it worse.

    All four corrected estimators sit within 0.1 points of each other, which is
    the uncomfortable part: this benchmark cannot rank them. The right panel is
    the standard advice failing, because there is no variance problem here to
    trade bias against.
    """
    o = load("ope_all")
    truth = o["truth_bts_ctr"] * 100
    weights = o["weights"]
    labels = {"ips": "IPS", "snips": "SNIPS", "dr_crossfit": "DR cross-fit",
              "direct_method": "direct method"}

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.6, 4.7))

    for index, key in enumerate(labels):
        est = o["estimators"][key]
        value = est["value"] * 100
        lo, hi = est["ci95"][0] * 100, est["ci95"][1] * 100
        if np.isfinite(lo):
            left.plot([lo, hi], [index, index], color=GOOD, lw=9, alpha=0.55,
                      solid_capstyle="butt", zorder=2)
            left.plot([value], [index], "o", color=GOOD, markersize=6, zorder=4)
            note = f"{est['rel_error']:+.1%}"
        else:
            left.plot([value], [index], "o", color=DIAG, markersize=6, zorder=4)
            note = f"{est['rel_error']:+.1%}, no interval"
        left.text(max(hi, value) + 0.002 if np.isfinite(hi) else value + 0.002,
                  index, note, va="center", fontsize=9.5, color="0.25")

    left.axvline(truth, color="0.2", ls="--", lw=1.4)
    left.set_xlim(0.474, 0.552)
    left.set_ylim(len(labels) - 0.4, -0.75)
    left.set_yticks(np.arange(len(labels)))
    left.set_yticklabels(list(labels.values()))
    left.text(truth + 0.002, -0.55, f"measured online CTR, {truth:.3f}%",
              fontsize=9, color="0.3", va="center", ha="left")
    left.text(0.02, 0.05, f"largest importance weight {weights['max']:.2f}, "
                          f"effective sample size {weights['ess_frac']:.1%} of "
                          f"{o['n_rows']:,} rows",
              transform=left.transAxes, fontsize=9, color="0.3")
    left.set_xlabel("estimated click-through rate (%)")
    titled(left, "The correction lands every estimator inside 2%",
           "green bars are the 95% intervals, all four agree to 0.1 points")

    caps = [c for c in o["clipping"] if c["cap"] is not None]
    right.plot([c["cap"] for c in caps], [c["value"] * 100 for c in caps],
               color="0.65", lw=1.6, zorder=1)
    for c in caps:
        binds = c["clipped_frac"] > 0
        right.plot([c["cap"]], [c["value"] * 100], "o", markersize=7,
                   color=BAD if binds else GOOD, zorder=3)
    right.axhline(truth, color="0.2", ls="--", lw=1.4)
    right.set_xscale("log")
    right.set_ylim(0, truth * 1.45)
    right.set_xlabel("cap applied to the importance weight")
    right.set_ylabel("IPS estimate of CTR (%)")
    right.text(caps[0]["cap"] * 1.15, caps[0]["value"] * 100,
               f"{caps[0]['clipped_frac']:.1%} of the weights cut", fontsize=9,
               color="0.3", va="center")
    right.text(2e2, truth - truth * 0.1, f"no weight exceeds {weights['max']:.2f},\n"
                                         "so these caps change nothing",
               fontsize=9, color="0.3", va="top", ha="right")
    right.legend(handles=[Patch(facecolor=BAD, label="the cap binds"),
                          Patch(facecolor=GOOD, label="the cap does nothing")],
                 loc="lower right", bbox_to_anchor=(1.0, 0.02))
    titled(right, "Clipping the weights only adds bias here",
           "every cap that binds costs accuracy, every cap that does not is a no-op")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def direction(out: Path) -> Path:
    """Two of the three stress tests: which policy logged, and how little data.

    Swapping the direction leaves the estimate accurate and the weights
    unrecognisable, which is the trap: nothing in the number warns you. The
    right panel is the well-behaved case for contrast, where the interval grows
    as the sample shrinks and keeps covering the truth.
    """
    o = load("stress_all")
    rows = [("random logs,\nevaluate BTS", o["forward"], DIAG),
            ("BTS logs,\nevaluate random", o["reverse"], BAD)]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.6, 4.7))

    # Dots, not bars. The axis is logarithmic, so a bar drawn from x=1 would have
    # a length proportional to log10(value), and the 1297x gap between these two
    # weights would render as about 4x. A dot puts the value at its position and
    # lets nothing encode it by length.
    for index, (_label, d, colour) in enumerate(rows):
        top = d["max_weight"]
        left.plot([top], [index], marker="o", markersize=11, color=colour,
                  linestyle="none", zorder=3)
        shown = f"{top:,.0f}" if top >= 100 else f"{top:.2f}"
        left.text(top * 1.7, index,
                  f"max weight {shown}\n"
                  f"ESS {d['ess_frac']:.2%} of {d['n']:,} rows",
                  va="center", fontsize=9.5, color="0.25")
    left.set_xscale("log")
    left.set_xlim(1, 2e6)
    left.set_ylim(1.45, -0.5)
    left.set_yticks([0, 1])
    left.set_yticklabels([r[0] for r in rows])
    left.set_xlabel("largest importance weight (log scale)")
    # Both numbers come from the file, so the claim cannot drift away from it.
    blowup = o["reverse"]["max_weight"] / o["forward"]["max_weight"]
    worst = max(abs(d["value"] / d["truth"] - 1.0) * 100
                for d in (o["forward"], o["reverse"])) if "value" in o["forward"] else None
    tail = (f"both within {worst:.1f}% of the truth"
            if worst is not None else "both still close to the truth")
    titled(left, f"The weights move {blowup:,.0f}x, the answer barely moves",
           f"same dataset, same estimator, {tail}")

    curve = o["sample_curve_reverse"]
    truth = o["reverse"]["truth"] * 100
    right.axhline(truth, color="0.2", ls="--", lw=1.4)
    right.plot([d["n"] for d in curve], [d["value"] * 100 for d in curve],
               color="0.65", lw=1.6, zorder=1)
    for d in curve:
        colour = GOOD if d["covers_truth"] else BAD
        half = d["ci_width"] / 2 * 100
        right.plot([d["n"], d["n"]], [d["value"] * 100 - half, d["value"] * 100 + half],
                   color=colour, lw=9, alpha=0.55, solid_capstyle="butt", zorder=2)
        right.plot([d["n"]], [d["value"] * 100], "o", color=colour, markersize=6,
                   zorder=4)
    right.set_xscale("log")
    right.set_ylim(0, truth * 2.0)
    right.set_xlabel("rows kept from the BTS logs (log scale)")
    right.set_ylabel("IPS estimate of CTR (%)")
    right.text(0.02, 0.05, "dashed line is the measured online CTR of random, "
                           f"{truth:.3f}%",
               transform=right.transAxes, fontsize=9, color="0.3")
    right.legend(handles=[Patch(facecolor=GOOD, alpha=0.55,
                                label="95% interval covers the truth")],
                 loc="upper right", bbox_to_anchor=(1.0, 0.98))
    titled(right, "The interval widens honestly as data shrinks",
           "the same reverse-direction estimate, 12,357 rows up to 12.36M")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def main() -> None:
    for path in (
        ground_truth(REPORTS / "eda_all.png"),
        baselines(REPORTS / "baselines.png"),
        corrected(REPORTS / "ope_all.png"),
        direction(REPORTS / "stress_all.png"),
        support(REPORTS / "support.png"),
        gate(REPORTS / "gate.png"),
        anim_support(REPORTS / "support-erosion.gif"),
    ):
        print(f"-> {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
