// Check that the README says what the data files say.
//
// Everything else here verifies that one JSON file follows from another. None
// of it looks at the README, and the README is what a reader actually believes.
// Every number in it was copied across by hand at some point, so a stale table
// after a rerun, or a digit typed wrong, would survive every other check in
// this directory.
//
// This reads README.md, parses its four tables structurally rather than by
// hunting for numbers, formats the corresponding value out of the JSON the same
// way the README formats it, and requires the two strings to be equal. Then it
// does the same for the sentences that carry a number.
//
// Node's standard library only.

"use strict";
const fs = require("fs");
const path = require("path");

const root = process.argv[2] || ".";
const readme = fs.readFileSync(path.join(root, "README.md"), "utf8");
const load = (p) =>
  JSON.parse(
    fs.readFileSync(path.join(root, p), "utf8").replace(/\bNaN\b/g, "null")
      .replace(/-?\bInfinity\b/g, "null")
  );

const eda = load("reports/eda_all.json");
const base = load("reports/baseline_all.json");
const ope = load("reports/ope_all.json");
const stress = load("reports/stress_all.json");
const harness = load("reports/harness_all.json");

let failures = 0;

// The README writes minus as U+2212, uses ** for emphasis, and is hard
// wrapped. None of those is a difference in a number.
const norm = (s) =>
  s.replace(/\u2212/g, "-").replace(/\*\*/g, "").replace(/\u00a0/g, " ")
    .replace(/\s+/g, " ").trim();

const commas = (x) =>
  String(Math.round(x)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
const fixed = (x, d) => x.toFixed(d);
const pct = (x, d) => (x >= 0 ? "+" : "") + (100 * x).toFixed(d) + "%";
const interval = (a, b, d) => `[${a.toFixed(d)}, ${b.toFixed(d)}]`;

function cell(label, got, want) {
  const ok = norm(String(got)) === norm(String(want));
  if (!ok) failures++;
  console.log(
    `  ${label.padEnd(42)} README ${String(want).padEnd(24)} data ${String(got).padEnd(24)} ${ok ? "ok" : "FAIL"}`
  );
}

// --------------------------------------------------------------- tables ---

// Return the data rows of the markdown table whose header row contains every
// one of `headers`, as arrays of trimmed cells.
function table(headers) {
  const lines = readme.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.startsWith("|")) continue;
    const cells = line.split("|").slice(1, -1).map((c) => c.trim());
    if (!headers.every((h) => cells.includes(h))) continue;
    const rows = [];
    for (let j = i + 2; j < lines.length && lines[j].startsWith("|"); j++) {
      rows.push(lines[j].split("|").slice(1, -1).map((c) => c.trim()));
    }
    return { header: cells, rows };
  }
  throw new Error(`no table in README.md with columns ${headers.join(", ")}`);
}

function checkTable(name, headers, spec) {
  const t = table(headers);
  console.log(`\n${name}: ${t.rows.length} rows`);
  if (t.rows.length !== spec.length) {
    console.log(`  FAIL the README has ${t.rows.length} rows, the data has ${spec.length}`);
    failures++;
    return;
  }
  t.rows.forEach((row, i) => {
    const want = spec[i];
    if (want.label !== undefined) cell(`row ${i + 1} label`, want.label, row[0]);
    Object.entries(want.cells).forEach(([col, value]) => {
      const at = t.header.indexOf(col);
      if (at < 0) {
        console.log(`  FAIL no column ${col}`);
        failures++;
        return;
      }
      cell(`${row[0].slice(0, 28)} / ${col}`, value, row[at]);
    });
  });
}

// 1. the known answer
checkTable("the known answer", ["policy", "impressions", "clicks", "CTR", "95% CI"], [
  {
    label: "uniform random",
    cells: {
      impressions: commas(eda.random.rows),
      clicks: commas(eda.random.clicks),
      CTR: fixed(eda.random.ctr, 5),
      "95% CI": interval(eda.random.ctr_lo, eda.random.ctr_hi, 5),
    },
  },
  {
    label: "Bernoulli TS",
    cells: {
      impressions: commas(eda.bts.rows),
      clicks: commas(eda.bts.clicks),
      CTR: fixed(eda.bts.ctr, 5),
      "95% CI": interval(eda.bts.ctr_lo, eda.bts.ctr_hi, 5),
    },
  },
]);

// 2. what the shortcuts get wrong
const shortcut = (key) => ({
  cells: {
    estimate: fixed(base.estimators[key].value, 5),
    "error vs truth": pct(base.estimators[key].rel_error, 1),
  },
});
checkTable("what the shortcuts get wrong", ["estimator", "estimate", "error vs truth"],
  ["naive_mean", "replay", "direct_method_empirical", "direct_method_logistic"].map(shortcut));

// 3. what the propensity correction fixes
const corrected = (key) => {
  const e = ope.estimators[key];
  return {
    cells: {
      estimate: fixed(e.value, 5),
      "error vs truth": pct(e.rel_error, 1),
      "95% CI": interval(e.ci95[0], e.ci95[1], 5),
      "covers truth": e.truth_inside_ci ? "yes" : "no",
    },
  };
};
checkTable("what the propensity correction fixes",
  ["estimator", "estimate", "error vs truth", "95% CI", "covers truth"],
  ["ips", "snips", "dr_crossfit"].map(corrected));

// 4. the gate
checkTable("the gate", ["scenario", "gate", "true error", "unlogged mass", "ESS", "correct"],
  harness.rows.map((r) => ({
    cells: {
      gate: r.status,
      "true error": pct(r.actual_rel_error, 1),
      "unlogged mass": pct(r.unlogged_target_mass, 1).replace("+", ""),
      ESS: commas(r.ess),
      correct: r.decision_correct ? "yes" : "no",
    },
  })));

// ---------------------------------------------------------------- prose ---
// Each of these is a sentence in the README that carries a number. The number
// is rebuilt from the JSON and the sentence has to contain it.

function prose(label, needle) {
  const ok = norm(readme).includes(norm(needle));
  if (!ok) failures++;
  console.log(`  ${label.padEnd(42)} ${ok ? "ok  " : "FAIL"} ${needle}`);
}

console.log("\nsentences that carry a number");
const gt = eda.ground_truth;
prose("headline lift", `BTS is ${pct(gt.relative_lift, 2)} better than random`);
prose("headline p value", `p = ${gt.p_value.toExponential(1)}`);
prose("days and items",
  `${eda.checks.shared_days} days of live traffic, both policies running concurrently over ${eda.checks.n_items} items`);
prose("the two shortcuts disagree by",
  `disagree by ${Math.round(100 * (base.estimators.replay.rel_error - base.estimators.naive_mean.rel_error))} points`);
prose("replay interval",
  `95% interval of ${interval(base.estimators.replay.ci95[0], base.estimators.replay.ci95[1], 5)}`);
prose("logistic AUC", `AUC, ${fixed(base.estimators.direct_method_logistic.auc, 4)}`);
prose("largest weight",
  `caps the largest weight in ${fixed(ope.n_rows / 1e6, 2)} million rows at ${fixed(ope.weights.max, 2)}`);
prose("reverse weight blow up",
  `from ${fixed(stress.forward.max_weight, 1)} to ${commas(stress.reverse.max_weight)}`);
prose("reverse ESS collapse",
  `from ${(100 * stress.forward.ess_frac).toFixed(1)}% to ${(100 * stress.reverse.ess_frac).toFixed(2)}%`);

const drop40 = stress.broken_support.find((s) => s.items_dropped === 40);
prose("support sweep at 40 dropped",
  `at 40 items removed the estimate is off by ${pct(drop40.rel_error, 1)} and ESS has gone up to ${(100 * drop40.ess_frac).toFixed(1)}%`);
prose("gate score",
  `${harness.decisions_correct} of ${harness.n_scenarios} right at a ${(100 * harness.tolerance).toFixed(0)}% tolerance`);
prose("the documented false accept",
  `its interval ${interval(stress.reverse.snips.ci95[0], stress.reverse.snips.ci95[1], 6)} misses the truth`);
prose("reverse diagnostics in Limitations",
  `at max weight ${commas(stress.reverse.max_weight)} and ESS ${(100 * stress.reverse.ess_frac).toFixed(2)}%`);

// The one claim in the corrected section that is a bound, not a quotation.
const worst = Math.max(
  ...["ips", "snips", "dr_crossfit", "direct_method"].map((k) =>
    Math.abs(ope.estimators[k].rel_error))
);
const bound = worst < 0.02;
if (!bound) failures++;
console.log(`  ${"collapses to under 2%".padEnd(42)} ${bound ? "ok  " : "FAIL"} worst corrected error ${pct(worst, 2)}`);

if (failures > 0) {
  console.log(`\n${failures} README numbers do not match the data files`);
  process.exit(1);
}
console.log("\nevery number in the README's four tables and its numbered sentences matches");
