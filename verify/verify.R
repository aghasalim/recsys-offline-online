# Independent statistical check of the claim the whole project rests on.
#
# The README's opening table says BTS beats random by +42.77% with p = 2.5e-166,
# and every later section is scored against that. It came from scipy, through
# src/roo/eda.py. This redoes the inference in base R, from the per position
# counts in reports/eda_all.json, and then adds two things the Python never
# computed:
#
#   an exact interval   the published CTR intervals are Wilson. Clopper-Pearson
#                       is exact and conservative, so it must be the wider of
#                       the two. If the published interval were the wider one,
#                       something is wrong with it.
#   an interval on the  the README states +42.77% as a point. There is no
#   relative lift       interval on it anywhere in the repository. Two routes
#                       are computed here, a delta method on the log ratio and
#                       200,000 parametric bootstrap draws, and they have to
#                       agree with each other and contain the published point.
#
# Base R only, so CI needs nothing beyond r-base-core.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
set.seed(20260901)

BOOT <- 200000
failures <- 0

# R's own decimal reader is accurate to about one unit in the last place, so a
# double written by Python and read back here can differ from the same double
# recomputed here in the 17th digit. That is a property of the reader, not a
# disagreement, so "exact" means one part in 1e-15 rather than bit equality.
EXACT <- 1e-15

check <- function(label, got, want, rtol) {
    d <- abs(got - want)
    rel <- if (want != 0) d / abs(want) else d
    ok <- rel <= rtol
    failures <<- failures + !ok
    cat(sprintf("  %-26s R %-22.15g published %-22.15g rel %.1e  %s\n",
                label, got, want, rel, if (ok) "ok" else "FAIL"))
}

claim <- function(label, ok, detail) {
    failures <<- failures + !ok
    cat(sprintf("  %-26s %s  %s\n", label, if (ok) "ok  " else "FAIL", detail))
}

# ---------------------------------------------------------------- JSON ----
# Enough of a reader for machine written, indented JSON with no escaped quotes
# in its keys. Not a parser: it finds a key and returns either the number after
# it or the span of the object it opens.

jblock <- function(txt, key) {
    m <- regexpr(paste0('"', key, '"[[:space:]]*:[[:space:]]*\\{'), txt)
    if (m == -1) stop(sprintf("no object %s", key))
    start <- m + attr(m, "match.length") - 1
    chars <- strsplit(substring(txt, start), "")[[1]]
    depth <- 0
    for (i in seq_along(chars)) {
        if (chars[i] == "{") depth <- depth + 1
        else if (chars[i] == "}") {
            depth <- depth - 1
            if (depth == 0) return(substring(txt, start, start + i - 1))
        }
    }
    stop(sprintf("unterminated object %s", key))
}

jnum <- function(txt, key) {
    m <- regexpr(paste0('"', key, '"[[:space:]]*:[[:space:]]*-?[0-9][0-9.eE+-]*'), txt)
    if (m == -1) stop(sprintf("no number %s", key))
    s <- substring(txt, m, m + attr(m, "match.length") - 1)
    as.numeric(sub('^.*:[[:space:]]*', "", s))
}

jarray2 <- function(txt, key) {
    m <- regexpr(paste0('"', key, '"[[:space:]]*:[[:space:]]*\\[[^]]*\\]'), txt)
    if (m == -1) stop(sprintf("no array %s", key))
    s <- substring(txt, m, m + attr(m, "match.length") - 1)
    as.numeric(strsplit(sub('^.*\\[', "", sub("\\]$", "", s)), ",")[[1]])
}

eda <- paste(readLines(file.path(root, "reports", "eda_all.json"), warn = FALSE),
             collapse = "\n")
stress <- paste(readLines(file.path(root, "reports", "stress_all.json"), warn = FALSE),
                collapse = "\n")

# --------------------------------------------------- counts and CTR ------

by_pos <- jblock(eda, "by_position")
counts <- function(policy) {
    blk <- jblock(by_pos, policy)
    n <- sum(sapply(c("1", "2", "3"), function(s) jnum(jblock(blk, s), "n")))
    k <- sum(sapply(c("1", "2", "3"), function(s) jnum(jblock(blk, s), "clicks")))
    c(n = n, k = k)
}
r <- counts("random")
b <- counts("bts")
gt <- jblock(eda, "ground_truth")

cat("counts summed from by_position, against the per policy blocks\n")
check("random impressions", r["n"], jnum(jblock(eda, "random"), "rows"), EXACT)
check("random clicks", r["k"], jnum(jblock(eda, "random"), "clicks"), EXACT)
check("bts impressions", b["n"], jnum(jblock(eda, "bts"), "rows"), EXACT)
check("bts clicks", b["k"], jnum(jblock(eda, "bts"), "clicks"), EXACT)

p_r <- r["k"] / r["n"]
p_b <- b["k"] / b["n"]
z975 <- qnorm(0.975)

wilson <- function(k, n, z = z975) {
    p <- k / n
    d <- 1 + z^2 / n
    centre <- (p + z^2 / (2 * n)) / d
    half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / d
    c(centre - half, centre + half)
}

cat("\nCTR and the published Wilson interval, rebuilt in base R\n")
check("random ctr", p_r, jnum(jblock(eda, "random"), "ctr"), EXACT)
check("bts ctr", p_b, jnum(jblock(eda, "bts"), "ctr"), EXACT)
for (pol in c("random", "bts")) {
    k <- if (pol == "random") r["k"] else b["k"]
    n <- if (pol == "random") r["n"] else b["n"]
    w <- wilson(k, n)
    blk <- jblock(eda, pol)
    check(paste(pol, "ctr_lo"), w[1], jnum(blk, "ctr_lo"), 1e-13)
    check(paste(pol, "ctr_hi"), w[2], jnum(blk, "ctr_hi"), 1e-13)
}

# --- the exact interval the repository does not publish -------------------
# Clopper-Pearson inverts the binomial directly and is conservative by
# construction, so it cannot be narrower than Wilson. That is a property of the
# two methods, not a tuned tolerance.
cat("\nClopper-Pearson exact intervals, which the repository never computed\n")
clopper <- function(k, n, alpha = 0.05) {
    c(qbeta(alpha / 2, k, n - k + 1), qbeta(1 - alpha / 2, k + 1, n - k))
}
for (pol in c("random", "bts")) {
    k <- if (pol == "random") r["k"] else b["k"]
    n <- if (pol == "random") r["n"] else b["n"]
    cp <- clopper(k, n)
    w <- wilson(k, n)
    claim(paste(pol, "exact is wider"), diff(cp) > diff(w),
          sprintf("exact [%.7f, %.7f] width %.3e, published Wilson width %.3e",
                  cp[1], cp[2], diff(cp), diff(w)))
    claim(paste(pol, "exact covers ctr"), cp[1] <= k / n && k / n <= cp[2],
          sprintf("ctr %.7f", k / n))
}

# --- the headline test ----------------------------------------------------
cat("\nthe two proportion test, two routes in R against scipy's\n")
diff_hat <- p_b - p_r
se <- sqrt(p_r * (1 - p_r) / r["n"] + p_b * (1 - p_b) / b["n"])
z <- diff_hat / se
p_from_norm <- 2 * pnorm(-abs(z))
p_from_chisq <- pchisq(z^2, df = 1, lower.tail = FALSE)

check("absolute_diff", diff_hat, jnum(gt, "absolute_diff"), EXACT)
check("diff_se", se, jnum(gt, "diff_se"), EXACT)
check("z", z, jnum(gt, "z"), EXACT)
check("relative_lift", p_b / p_r - 1, jnum(gt, "relative_lift"), EXACT)
check("p_value via pnorm", p_from_norm, jnum(gt, "p_value"), 1e-12)
check("p_value via pchisq", p_from_chisq, jnum(gt, "p_value"), 1e-12)
ci <- jarray2(gt, "diff_ci95")
check("diff_ci95 lo", diff_hat - 1.96 * se, ci[1], EXACT)
check("diff_ci95 hi", diff_hat + 1.96 * se, ci[2], EXACT)

# --- an interval on the relative lift -------------------------------------
cat("\nan interval on the +42.77%, which the README states as a point\n")
lift <- p_b / p_r - 1
log_se <- sqrt((1 - p_b) / (b["n"] * p_b) + (1 - p_r) / (r["n"] * p_r))
delta_ci <- exp(log(p_b / p_r) + c(-1, 1) * z975 * log_se) - 1

draws_r <- rbinom(BOOT, r["n"], p_r) / r["n"]
draws_b <- rbinom(BOOT, b["n"], p_b) / b["n"]
boot_ci <- unname(quantile(draws_b / draws_r - 1, c(0.025, 0.975)))

cat(sprintf("  delta method on the log ratio  [%+.2f%%, %+.2f%%]\n",
            100 * delta_ci[1], 100 * delta_ci[2]))
cat(sprintf("  %s parametric bootstrap draws  [%+.2f%%, %+.2f%%]\n",
            format(BOOT, big.mark = ",", scientific = FALSE), 100 * boot_ci[1], 100 * boot_ci[2]))
claim("published lift is inside", lift >= boot_ci[1] && lift <= boot_ci[2],
      sprintf("point %+.4f%%", 100 * lift))
rel_gap <- max(abs(delta_ci - boot_ci)) / diff(delta_ci)
claim("two routes agree", rel_gap < 0.02,
      sprintf("worst end differs by %.2f%% of the interval width", 100 * rel_gap))
claim("lift interval excludes zero", boot_ci[1] > 0,
      sprintf("lower end %+.2f%%", 100 * boot_ci[1]))

# --- the one thing the README admits it gets wrong ------------------------
# Limitations says the reverse direction's interval "misses the truth by 1.0
# half-widths". Nothing computed that; it is a claim about two numbers sitting
# in reports/stress_all.json.
cat("\nthe Limitations claim, from reports/stress_all.json\n")
rev <- jblock(stress, "reverse")
rev_snips <- jblock(rev, "snips")
rev_ci <- jarray2(rev_snips, "ci95")
truth_rev <- jnum(rev, "truth")
half <- diff(rev_ci) / 2
miss <- (truth_rev - mean(rev_ci)) / half
claim("reverse interval misses", !(rev_ci[1] <= truth_rev && truth_rev <= rev_ci[2]),
      sprintf("truth %.6f, interval [%.6f, %.6f]", truth_rev, rev_ci[1], rev_ci[2]))
claim("by 1.0 half-widths", sprintf("%.1f", miss) == "1.0",
      sprintf("measured %.4f half-widths", miss))

if (failures > 0) {
    cat(sprintf("\n%d R checks failed\n", failures))
    quit(status = 1)
}
cat("\nR reproduces the headline test exactly and puts an interval on the lift\n")
