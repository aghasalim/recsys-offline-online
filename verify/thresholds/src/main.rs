//! Were the gate's thresholds fitted to the answer?
//!
//! src/roo/harness.py says its two thresholds "were not chosen by looking at
//! which value made the answers come out right", and the README repeats it. The
//! repository has no way to check that, because it only ever ran the gate at
//! one setting: 1% unlogged mass and an effective sample size floor of 1,000.
//! Nobody has seen the rest of the plane.
//!
//! This walks all of it. The gate's decision is a step function of the two
//! thresholds, so the plane splits into finitely many regions inside which
//! every scenario decides the same way, and enumerating one point per region
//! is an exact answer rather than a sample. That is done first. Then the same
//! question is answered a second way, by brute force over a fine grid of the
//! whole plane, which has to find the same maxima; the two disagreeing would
//! mean the region enumeration missed something.
//!
//! Three things come out of it:
//!
//!   1. at the published thresholds the decisions must match
//!      reports/harness_all.json exactly, which is the differential check
//!   2. the best score any threshold pair can reach on the seven published
//!      scenarios, and on all the distinct scenarios in the grid
//!   3. how much of the plane reaches each score, which is what says whether a
//!      good setting is a broad region or a knife edge
//!
//! No crates. The JSON here is machine written and shallow, so it is read with
//! a scanner that resolves every field by name.

use std::env;
use std::fs;
use std::process::exit;

/// The published defaults, from harness.audit()'s signature.
const PUBLISHED_MASS: f64 = 0.01;
const PUBLISHED_ESS: f64 = 1000.0;

/// Brute force resolution. 4000 x 4000 is 16 million threshold pairs, each
/// scored against every scenario.
const GRID: usize = 4000;
const LOG_ESS_MAX: f64 = 6.0;

#[derive(Clone, Debug, PartialEq)]
struct Scenario {
    name: String,
    ess: f64,
    unlogged: f64,
    truth: f64,
    lo: f64,
    hi: f64,
}

impl Scenario {
    /// True when the published interval would have covered the truth, which is
    /// validate()'s grading rule.
    fn covers(&self) -> bool {
        self.truth >= self.lo && self.truth <= self.hi
    }
    fn refuses(&self, max_mass: f64, min_ess: f64) -> bool {
        self.unlogged > max_mass || self.ess < min_ess
    }
    fn correct_at(&self, max_mass: f64, min_ess: f64) -> bool {
        !self.refuses(max_mass, min_ess) == self.covers()
    }
}

// ------------------------------------------------------------------ JSON ---

/// Byte offset just past `"key":` inside `s`, searching only outside strings is
/// unnecessary here because no value in these files contains a quoted key.
fn after_key(s: &str, key: &str) -> Option<usize> {
    let pat = format!("\"{}\"", key);
    let mut from = 0;
    while let Some(rel) = s[from..].find(&pat) {
        let at = from + rel + pat.len();
        let rest = &s[at..];
        let trimmed = rest.trim_start();
        if trimmed.starts_with(':') {
            let colon = at + (rest.len() - trimmed.len()) + 1;
            return Some(colon + (s[colon..].len() - s[colon..].trim_start().len()));
        }
        from = at;
    }
    None
}

fn number(s: &str, key: &str) -> f64 {
    let at = after_key(s, key)
        .unwrap_or_else(|| die(&format!("no key \"{}\" in a block that needs one", key)));
    let rest = &s[at..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E'))
        .unwrap_or(rest.len());
    rest[..end]
        .parse()
        .unwrap_or_else(|_| die(&format!("\"{}\" is not a number: {:?}", key, &rest[..end])))
}

fn string(s: &str, key: &str) -> String {
    let at = after_key(s, key).unwrap_or_else(|| die("missing string key"));
    let rest = &s[at + 1..];
    rest[..rest.find('"').unwrap_or(0)].to_string()
}

/// Two element numeric array under `key`.
fn pair(s: &str, key: &str) -> (f64, f64) {
    let at = after_key(s, key).unwrap_or_else(|| die("missing array key"));
    let rest = &s[at + 1..];
    let close = rest.find(']').unwrap_or_else(|| die("unterminated array"));
    let parts: Vec<f64> = rest[..close]
        .split(',')
        .map(|p| p.trim().parse().unwrap_or_else(|_| die("bad array element")))
        .collect();
    if parts.len() != 2 {
        die("interval does not have two ends");
    }
    (parts[0], parts[1])
}

/// Split the array that follows `key` into its top level object slices.
fn objects<'a>(s: &'a str, key: &str) -> Vec<&'a str> {
    let at = after_key(s, key).unwrap_or_else(|| die("missing array of objects"));
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let (mut depth, mut start, mut in_string, mut escaped) = (0usize, 0usize, false, false);
    for i in at..bytes.len() {
        let c = bytes[i] as char;
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    out.push(&s[start..=i]);
                }
            }
            ']' if depth == 0 => break,
            _ => {}
        }
    }
    out
}

fn die(msg: &str) -> ! {
    eprintln!("thresholds: {}", msg);
    exit(2)
}

// ----------------------------------------------------------------- sweep ---

/// One representative threshold from every region the decisions can occupy.
///
/// The mass predicate is `unlogged > m`, so the decisions change only as m
/// crosses an observed mass from below, and the regions are [v_i, v_i+1). The
/// ESS predicate is `ess < e`, so its regions are (v_i, v_i+1], represented by
/// the next representable double above v_i. Every region has positive width,
/// which is what lets the brute force pass below find them all too.
fn mass_candidates(values: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = values.to_vec();
    v.push(0.0);
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

fn ess_candidates(values: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = values
        .iter()
        .map(|x| f64::from_bits(x.to_bits() + 1))
        .collect();
    v.push(0.0);
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

/// The maximal runs of ESS floor that reach `target` at a fixed mass, printed
/// as the half open intervals they really are.
fn ess_regions(set: &[Scenario], mass: f64, target: usize, observed: &[f64]) -> Vec<(f64, f64)> {
    let mut v: Vec<f64> = observed.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    let mut bounds = vec![0.0];
    bounds.extend(v.iter().cloned());
    bounds.push(f64::INFINITY);

    let mut out: Vec<(f64, f64)> = Vec::new();
    for i in 0..bounds.len() - 1 {
        let probe = if bounds[i] == 0.0 {
            f64::MIN_POSITIVE
        } else {
            f64::from_bits(bounds[i].to_bits() + 1)
        };
        if score(set, mass, probe) == target {
            match out.last_mut() {
                Some(last) if last.1 == bounds[i] => last.1 = bounds[i + 1],
                _ => out.push((bounds[i], bounds[i + 1])),
            }
        }
    }
    out
}

fn score(set: &[Scenario], m: f64, e: f64) -> usize {
    set.iter().filter(|s| s.correct_at(m, e)).count()
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| ".".into());
    let grid_txt = fs::read_to_string(format!("{}/app_data/grid_all.json", root))
        .unwrap_or_else(|e| die(&format!("cannot read grid_all.json: {}", e)));
    let harness_txt = fs::read_to_string(format!("{}/reports/harness_all.json", root))
        .unwrap_or_else(|e| die(&format!("cannot read harness_all.json: {}", e)));

    let all: Vec<Scenario> = objects(&grid_txt, "scenarios")
        .iter()
        .map(|o| Scenario {
            name: string(o, "name"),
            ess: number(o, "ess"),
            unlogged: number(o, "unlogged_target_mass"),
            truth: number(o, "truth"),
            lo: pair(o, "ci95").0,
            hi: pair(o, "ci95").1,
        })
        .collect();

    // Three grid entries are the same scenario reached three ways. Deduplicate
    // so the wider count is over distinct evidence.
    let mut distinct: Vec<Scenario> = Vec::new();
    for s in &all {
        if !distinct.iter().any(|d| d.ess == s.ess && d.unlogged == s.unlogged
            && d.lo == s.lo && d.hi == s.hi) {
            distinct.push(s.clone());
        }
    }

    // The seven the README publishes, matched to the grid by effective sample
    // size, which identifies a scenario's log set.
    let rows = objects(&harness_txt, "rows");
    let mut published: Vec<Scenario> = Vec::new();
    let mut failures = 0;
    println!("the seven published scenarios, at the published thresholds");
    for row in &rows {
        let ess = number(row, "ess");
        let name = string(row, "scenario");
        let s = match distinct.iter().find(|d| d.ess == ess) {
            Some(s) => s.clone(),
            None => {
                println!("  {:<40} FAIL no grid entry at ESS {:.6}", name, ess);
                failures += 1;
                continue;
            }
        };
        let refused = s.refuses(PUBLISHED_MASS, PUBLISHED_ESS);
        let want_status = string(row, "status");
        let want_correct = row.contains("\"decision_correct\": true");
        let got_status = if refused { "refuse" } else { "ok" };
        let got_correct = s.correct_at(PUBLISHED_MASS, PUBLISHED_ESS);
        let bad = got_status != want_status || got_correct != want_correct;
        failures += bad as usize;
        println!(
            "  {:<40} {:<7} correct {:<5} {}",
            name, got_status, got_correct,
            if bad { "FAIL" } else { "agrees with harness_all.json" }
        );
        published.push(s);
    }

    let want_total = number(&harness_txt, "decisions_correct") as usize;
    let got_total = score(&published, PUBLISHED_MASS, PUBLISHED_ESS);
    if got_total != want_total || published.len() != rows.len() {
        failures += 1;
    }
    println!(
        "  {} of {} correct at mass {} and ESS floor {}, harness_all.json publishes {}",
        got_total, published.len(), PUBLISHED_MASS, PUBLISHED_ESS, want_total
    );

    // --- exact region enumeration -----------------------------------------
    let masses: Vec<f64> = distinct.iter().map(|s| s.unlogged).collect();
    let esss: Vec<f64> = distinct.iter().map(|s| s.ess).collect();
    let cm = mass_candidates(&masses);
    let ce = ess_candidates(&esss);
    println!(
        "\nexact enumeration: {} x {} = {} regions cover the whole threshold plane",
        cm.len(), ce.len(), cm.len() * ce.len()
    );

    let mut best_pub = (0usize, 0.0, 0.0);
    let mut best_all = (0usize, 0.0, 0.0);
    let mut best_joint: Option<(usize, usize, f64, f64)> = None;
    for &m in &cm {
        for &e in &ce {
            let sp = score(&published, m, e);
            let sa = score(&distinct, m, e);
            if sp > best_pub.0 {
                best_pub = (sp, m, e);
            }
            if sa > best_all.0 {
                best_all = (sa, m, e);
            }
            let better = match best_joint {
                None => true,
                Some((bp, ba, _, _)) => (sp + sa) > (bp + ba),
            };
            if better {
                best_joint = Some((sp, sa, m, e));
            }
        }
    }
    println!(
        "  best on the seven published scenarios: {} of {}, first reached at mass {:.6} ESS {:.1}",
        best_pub.0, published.len(), best_pub.1, best_pub.2
    );
    println!(
        "  best on all {} distinct scenarios:      {} of {}, first reached at mass {:.6} ESS {:.1}",
        distinct.len(), best_all.0, distinct.len(), best_all.1, best_all.2
    );
    let published_all = score(&distinct, PUBLISHED_MASS, PUBLISHED_ESS);
    println!(
        "  the published thresholds score {} of {} on the same {} scenarios",
        published_all, distinct.len(), distinct.len()
    );
    if let Some((sp, sa, m, e)) = best_joint {
        println!(
            "  best pair on both at once: {} of {} and {} of {}, at mass {:.6} ESS {:.1}",
            sp, published.len(), sa, distinct.len(), m, e
        );
    }

    // What it costs to fix the one false accept the README documents.
    let regions = ess_regions(&published, PUBLISHED_MASS, best_pub.0, &esss);
    for (lo, hi) in &regions {
        let inside: Vec<f64> = ce
            .iter()
            .cloned()
            .filter(|e| e > lo && (!hi.is_finite() || e <= hi))
            .collect();
        let wider: Vec<usize> = inside
            .iter()
            .map(|&e| score(&distinct, PUBLISHED_MASS, e))
            .collect();
        println!(
            "  at the published mass, every ESS floor above {:.1} and up to {} reaches {} of {}",
            lo,
            if hi.is_finite() { format!("{:.1}", hi) } else { "no bound".to_string() },
            best_pub.0, published.len()
        );
        println!(
            "  across that whole region the wider set scores {} to {} of {}, \n\
             \x20 against {} of {} at the published floor of {:.0}",
            wider.iter().min().copied().unwrap_or(0),
            wider.iter().max().copied().unwrap_or(0),
            distinct.len(), published_all, distinct.len(), PUBLISHED_ESS
        );
    }
    let both = cm.iter().any(|&m| {
        ce.iter().any(|&e| {
            score(&published, m, e) == best_pub.0 && score(&distinct, m, e) == best_all.0
        })
    });
    println!(
        "  a single pair reaching {} of {} and {} of {} at the same time: {}",
        best_pub.0, published.len(), best_all.0, distinct.len(),
        if both { "yes" } else { "no" }
    );

    // --- brute force, as a check on the enumeration ------------------------
    let mut area = [0u64; 32];
    let mut area_all = [0u64; 32];
    let (mut bf_pub, mut bf_all) = (0usize, 0usize);
    for i in 0..GRID {
        let m = (i as f64 + 0.5) / GRID as f64;
        for j in 0..GRID {
            let e = 10f64.powf((j as f64 + 0.5) / GRID as f64 * LOG_ESS_MAX);
            let sp = score(&published, m, e);
            let sa = score(&distinct, m, e);
            area[sp] += 1;
            area_all[sa] += 1;
            bf_pub = bf_pub.max(sp);
            bf_all = bf_all.max(sa);
        }
    }
    let cells = (GRID * GRID) as f64;
    println!(
        "\nbrute force over {} x {} = {} threshold pairs, mass in [0, 1] and \n\
         ESS floor in [1, 1e{:.0}]",
        GRID, GRID, GRID * GRID, LOG_ESS_MAX
    );
    println!("  published set, share of the plane by score");
    for s in (0..=published.len()).rev() {
        if area[s] > 0 {
            println!("    {} of {}: {:6.2}%", s, published.len(), 100.0 * area[s] as f64 / cells);
        }
    }
    println!("  all {} distinct scenarios, share of the plane by score", distinct.len());
    for s in (0..=distinct.len()).rev() {
        if area_all[s] > 0 {
            println!("    {} of {}: {:6.2}%", s, distinct.len(), 100.0 * area_all[s] as f64 / cells);
        }
    }

    if bf_pub != best_pub.0 || bf_all != best_all.0 {
        println!(
            "\nFAIL brute force found {} and {}, the region enumeration found {} and {}",
            bf_pub, bf_all, best_pub.0, best_all.0
        );
        failures += 1;
    } else {
        println!(
            "\nbrute force and the exact enumeration agree on both maxima, {} and {}",
            bf_pub, bf_all
        );
    }

    if failures > 0 {
        println!("\n{} checks failed", failures);
        exit(1);
    }
    println!("the gate rebuilt at the published thresholds matches reports/harness_all.json");
}
