# Recompute the two tables nothing else here touches: the propensity corrected
# estimators, and the stress tests.
#
# reports/ope_all.json and reports/stress_all.json carry the README's third and
# fourth sections. Both are built the same way: a point estimate, a standard
# error, and then an interval, a relative error and a coverage flag derived
# from those two. The derived columns are the ones a reader actually looks at,
# and they were written by the same function that produced the estimate, so
# nothing has ever checked that they follow from it.
#
# This rederives every one of them, in Ruby, and then checks the sentences the
# README makes out of them: that clipping only adds bias on this dataset, that
# reversing the direction takes the largest weight from 9.6 to 12,500 and the
# effective sample size from 26.8% to 0.16%, and that the relative error tracks
# the target policy's mass on unlogged actions across the support sweep.
#
# It also checks the two measured CTRs are the same value in every file that
# quotes them, which is the one thing that would break every table at once.

require "json"

ROOT = ARGV[0] || "."
Z = 1.96                 # the multiplier src/roo/ope.py uses for its intervals
EXACT = 0.0
NEAR = 1e-12

$failures = 0

def check(label, got, want, rtol = EXACT)
  d = (got - want).abs
  rel = want.zero? ? d : d / want.abs
  ok = rel <= rtol
  $failures += 1 unless ok
  printf("  %-34s ruby %-22.15g published %-22.15g rel %.1e  %s\n",
         label, got, want, rel, ok ? "ok" : "FAIL")
end

def claim(label, ok, detail)
  $failures += 1 unless ok
  printf("  %-34s %s  %s\n", label, ok ? "ok  " : "FAIL", detail)
end

def load(rel)
  JSON.parse(File.read(File.join(ROOT, rel)), allow_nan: true)
end

ope     = load("reports/ope_all.json")
stress  = load("reports/stress_all.json")
eda     = load("reports/eda_all.json")
base    = load("reports/baseline_all.json")
harness = load("reports/harness_all.json")

# ------------------------------------------------------------------------
puts "the corrected estimators, derived columns rebuilt from value and se"
truth = ope["truth_bts_ctr"]
n_rows = ope["n_rows"]

ope["estimators"].each do |name, e|
  next if e["se"].nil? || e["se"].to_f.nan?   # the direct method has no se
  check("#{name} ci95 lo", e["value"] - Z * e["se"], e["ci95"][0])
  check("#{name} ci95 hi", e["value"] + Z * e["se"], e["ci95"][1])
  check("#{name} rel_error", e["value"] / truth - 1.0, e["rel_error"])
  inside = e["ci95"][0] <= truth && truth <= e["ci95"][1]
  claim("#{name} truth_inside_ci", inside == e["truth_inside_ci"],
        "interval [#{'%.6f' % e['ci95'][0]}, #{'%.6f' % e['ci95'][1]}], truth #{'%.6f' % truth}")
end
check("weights ess_frac", ope["weights"]["ess"] / n_rows, ope["weights"]["ess_frac"])

# ------------------------------------------------------------------------
# The README says clipping "only adds bias here". That is a statement about the
# whole sweep, not one row: every cap at or above the largest weight has to be
# a no-op, and every cap below it has to move the estimate further from the
# truth, not closer.
puts "\nthe clipping sweep, against the claim that clipping only adds bias"
max_w = ope["weights"]["max"]
uncapped = ope["clipping"].find { |c| c["cap"].nil? }
raise "no uncapped row in the clipping sweep" unless uncapped

no_ops = 0
ope["clipping"].each do |c|
  cap = c["cap"]
  if cap.nil? || cap >= max_w
    no_ops += 1
    label = cap.nil? ? "no cap" : "cap #{'%g' % cap}"
    claim("#{label} changes nothing",
          c["value"] == uncapped["value"] && c["clipped_frac"] == 0.0,
          "value #{'%.9f' % c['value']}, clipped #{'%.4f' % c['clipped_frac']}%")
  else
    claim("cap #{'%g' % cap} is further from truth",
          c["rel_error"].abs > uncapped["rel_error"].abs,
          "rel error #{'%+.1f' % (100 * c['rel_error'])}% against " \
          "#{'%+.1f' % (100 * uncapped['rel_error'])}% uncapped")
  end
  check("cap #{cap.nil? ? 'none' : '%g' % cap} rel_error",
        c["value"] / truth - 1.0, c["rel_error"], NEAR)
end
claim("every cap at or above max weight", no_ops == 5,
      "#{no_ops} of the 8 caps are at or above the largest weight #{'%.3f' % max_w}")

# Clipping trades variance for bias, so the effective sample size it buys has
# to fall as the cap is loosened, and the share of rows it touches with it.
caps = ope["clipping"].reject { |c| c["cap"].nil? }
monotone = caps.each_cons(2).all? { |a, b| b["ess"] <= a["ess"] + 1e-9 }
claim("ESS falls as the cap loosens", monotone,
      "#{'%.0f' % caps.first['ess']} at cap #{'%g' % caps.first['cap']} down to " \
      "#{'%.0f' % caps.last['ess']} at cap #{'%g' % caps.last['cap']}")
claim("clipped share falls with it",
      caps.each_cons(2).all? { |a, b| b["clipped_frac"] <= a["clipped_frac"] + 1e-15 },
      "#{'%.4f' % caps.first['clipped_frac']} at cap #{'%g' % caps.first['cap']} " \
      "down to #{'%.4f' % caps.last['clipped_frac']} at cap #{'%g' % caps.last['cap']}")

# A row that says it clipped nothing has to report the unclipped estimate, and a
# row that says it clipped something must not. That ties the two columns
# together, which is the only thing stopping either drifting on its own.
ope["clipping"].each do |c|
  label = c["cap"].nil? ? "no cap" : "cap #{'%g' % c['cap']}"
  claim("#{label} share matches the value",
        (c["clipped_frac"] == 0.0) == (c["value"] == uncapped["value"]),
        "clipped #{'%.4f' % c['clipped_frac']} of rows, " \
        "estimate #{c['value'] == uncapped['value'] ? 'unchanged' : 'moved'}")
end

# ------------------------------------------------------------------------
puts "\nthe stress tests, and the same forward numbers reached from two files"
%w[ips snips].each do |k|
  check("forward #{k} matches ope_all", stress["forward"][k]["value"],
        ope["estimators"][k]["value"])
  check("forward #{k} se matches ope_all", stress["forward"][k]["se"],
        ope["estimators"][k]["se"])
end

%w[forward reverse].each do |dir|
  d = stress[dir]
  check("#{dir} ess_frac", d["ess"] / d["n"], d["ess_frac"])
  %w[ips snips].each do |k|
    e = d[k]
    check("#{dir} #{k} ci95 lo", e["value"] - Z * e["se"], e["ci95"][0])
    check("#{dir} #{k} ci95 hi", e["value"] + Z * e["se"], e["ci95"][1])
    check("#{dir} #{k} rel_error", e["value"] / d["truth"] - 1.0, e["rel_error"], NEAR)
    inside = e["ci95"][0] <= d["truth"] && d["truth"] <= e["ci95"][1]
    claim("#{dir} #{k} covers_truth", inside == e["covers_truth"],
          "truth #{'%.6f' % d['truth']} against [#{'%.6f' % e['ci95'][0]}, " \
          "#{'%.6f' % e['ci95'][1]}]")
  end
end

# The README's sentence: reversing the direction takes the largest weight from
# 9.6 to 12,500 and the effective sample size from 26.8% to 0.16%.
f, r = stress["forward"], stress["reverse"]
claim("reverse blows up the weights", r["max_weight"] > 1000 * f["max_weight"],
      "max weight #{'%.1f' % f['max_weight']} forward, " \
      "#{'%.0f' % r['max_weight']} reverse")
claim("reverse collapses the ESS", r["ess_frac"] < f["ess_frac"] / 100,
      "#{'%.1f' % (100 * f['ess_frac'])}% forward, " \
      "#{'%.2f' % (100 * r['ess_frac'])}% reverse")

# ------------------------------------------------------------------------
# "What does track the error is the target policy's probability mass on actions
# the logs never saw." Across the support sweep that is a monotone relationship,
# and the correlation between the two is the number nobody computed.
puts "\nthe support sweep: does unlogged mass track the error"
sweep = stress["broken_support"]
mass = sweep.map { |s| s["target_mass_on_unlogged_actions"] }
err = sweep.map { |s| s["rel_error"].abs }
ess_frac = sweep.map { |s| s["ess_frac"] }

claim("unlogged mass rises with drops", mass.each_cons(2).all? { |a, b| b > a },
      "#{'%.1f' % (100 * mass.first)}% to #{'%.1f' % (100 * mass.last)}% over " \
      "#{sweep.length} steps")
claim("the error rises with it", err.each_cons(2).all? { |a, b| b > a },
      "#{'%.1f' % (100 * err.first)}% to #{'%.1f' % (100 * err.last)}%")

def pearson(a, b)
  n = a.length
  ma = a.sum / n.to_f
  mb = b.sum / n.to_f
  cov = a.zip(b).map { |x, y| (x - ma) * (y - mb) }.sum
  va = a.map { |x| (x - ma)**2 }.sum
  vb = b.map { |y| (y - mb)**2 }.sum
  cov / Math.sqrt(va * vb)
end

r_mass = pearson(mass, err)
r_ess = pearson(ess_frac, err)
claim("mass correlates with the error", r_mass > 0.9,
      "pearson r = #{'%.4f' % r_mass} over #{sweep.length} steps")
claim("ESS does not", r_ess.abs < r_mass,
      "pearson r = #{'%+.4f' % r_ess}, which is the diagnostic the README says fails")

# ------------------------------------------------------------------------
puts "\nthe two measured CTRs, in every file that quotes them"
bts = eda["ground_truth"]["ctr_bts"]
rnd = eda["ground_truth"]["ctr_random"]
{
  "eda_all bts block"        => eda["bts"]["ctr"],
  "baseline_all truth"       => base["truth_bts_ctr"],
  "ope_all truth"            => ope["truth_bts_ctr"],
  "stress_all forward truth" => stress["forward"]["truth"],
  "harness_all row 0 truth"  => harness["rows"][0]["truth"],
}.each { |k, v| check(k, v, bts) }
{
  "eda_all random block"     => eda["random"]["ctr"],
  "baseline_all truth"       => base["truth_random_ctr"],
  "stress_all reverse truth" => stress["reverse"]["truth"],
  "harness_all row 1 truth"  => harness["rows"][1]["truth"],
}.each { |k, v| check(k, v, rnd) }

if $failures > 0
  puts "\n#{$failures} Ruby checks failed"
  exit 1
end
puts "\nRuby rebuilds every derived column in ope_all.json and stress_all.json"
