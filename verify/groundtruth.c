/* Recompute the ground truth table in the README, in C.
 *
 * The README opens with impressions, clicks, CTR and a 95% interval for each
 * policy, then "BTS is +42.77% better than random, p = 2.5e-166". Every one of
 * those came out of src/roo/eda.py, through pandas and scipy. Nothing checked
 * them, because everything downstream reads the same file.
 *
 * reports/eda_all.json carries a rawer layer than the headline: by_position
 * holds the per-slot impression and click counts, and those sum to the totals.
 * So this reads only the per-position counts and derives the whole table from
 * them: totals, CTR, the Wilson interval, the difference of proportions, its
 * standard error, z, and the p value. Then it compares against the published
 * ground_truth block and exits non-zero on any disagreement.
 *
 * Two constants are deliberately not hard coded. The Wilson interval needs
 * qnorm(0.975), which scipy supplies to the Python; here it is solved from
 * erfc by Newton iteration, so a wrong constant cannot be shared between the
 * two implementations. The p value uses erfc directly rather than a normal
 * survival function, which is a different code path to scipy's.
 *
 * Every field is located by name. A key that moves, or a block that gains a
 * field, cannot silently shift what this reads.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_JSON (1 << 20)

static char doc[MAX_JSON];

/* ---------------------------------------------------------------- JSON ---
 * The file is machine written, indented, and never contains an escaped quote
 * in a key. That is enough structure for a scanner that finds a key inside a
 * given span and returns the span of its value. It is not a general parser and
 * does not pretend to be.
 */

typedef struct { const char *b, *e; } Span;

static Span whole(void) { Span s = { doc, doc + strlen(doc) }; return s; }

/* Span of the brace or bracket delimited value starting at p. */
static Span bracketed(const char *p, const char *end)
{
    char open = *p, close = (open == '{') ? '}' : ']';
    int depth = 0;
    int in_string = 0;
    const char *q = p;
    for (; q < end; q++) {
        if (in_string) {
            if (*q == '\\') q++;
            else if (*q == '"') in_string = 0;
            continue;
        }
        if (*q == '"') in_string = 1;
        else if (*q == open) depth++;
        else if (*q == close && --depth == 0) { q++; break; }
    }
    Span s = { p, q };
    return s;
}

/* Find "key" at any depth inside span and return the span of its value. */
static int find_key(Span in, const char *key, Span *out)
{
    char pat[128];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    size_t n = strlen(pat);

    for (const char *p = in.b; p + n <= in.e; p++) {
        if (memcmp(p, pat, n) != 0) continue;
        const char *q = p + n;
        while (q < in.e && (*q == ' ' || *q == '\n' || *q == '\t')) q++;
        if (q >= in.e || *q != ':') continue;
        q++;
        while (q < in.e && (*q == ' ' || *q == '\n' || *q == '\t')) q++;
        if (q >= in.e) return 0;
        *out = (*q == '{' || *q == '[') ? bracketed(q, in.e) : (Span){ q, in.e };
        return 1;
    }
    return 0;
}

/* Value of "key" inside span, as a double. Aborts rather than defaulting: a
 * missing field is a broken file, not a zero. */
static double num(Span in, const char *key)
{
    Span v;
    if (!find_key(in, key, &v)) {
        fprintf(stderr, "eda_all.json has no key \"%s\" where one is required\n", key);
        exit(2);
    }
    return strtod(v.b, NULL);
}

/* Value of "key" inside span, as an object span. */
static Span obj(Span in, const char *key)
{
    Span v;
    if (!find_key(in, key, &v) || *v.b != '{') {
        fprintf(stderr, "eda_all.json has no object \"%s\"\n", key);
        exit(2);
    }
    return v;
}

/* ------------------------------------------------------------ normal ---- */

/* Two sided normal tail. erfc is in libm and is not the code path scipy takes,
 * which is the point of computing it here. */
static double two_sided_p(double z) { return erfc(fabs(z) / sqrt(2.0)); }

/* qnorm(1 - alpha/2), by Newton on erfc. d/dz erfc(z/sqrt2) = -sqrt(2/pi)
 * exp(-z^2/2). Converges from 2.0 in a handful of steps for alpha = 0.05.
 * pi comes from acos rather than M_PI, which -std=c99 hides on glibc. */
static double z_for(double alpha)
{
    const double pi = acos(-1.0);
    double z = 2.0;
    for (int i = 0; i < 60; i++) {
        double f = two_sided_p(z) - alpha;
        double d = -sqrt(2.0 / pi) * exp(-0.5 * z * z);
        double step = f / d;
        z -= step;
        if (fabs(step) < 1e-16 * fabs(z)) break;
    }
    return z;
}

/* --------------------------------------------------------------- check --- */

static int failures = 0;

static void cmp(const char *label, double got, double want, double rtol)
{
    double d = fabs(got - want);
    double rel = want != 0.0 ? d / fabs(want) : d;
    int bad = !(rel <= rtol);
    failures += bad;
    printf("  %-24s C %-22.15g  published %-22.15g  rel %.1e  %s\n",
           label, got, want, rel, bad ? "FAIL" : "ok");
}

static void cmp_long(const char *label, long got, long want)
{
    int bad = got != want;
    failures += bad;
    printf("  %-24s C %-22ld  published %-22ld  %-9s %s\n",
           label, got, want, "", bad ? "FAIL" : "ok");
}

/* Sum the per position counts of one policy and check each slot's own CTR. */
static void totals(Span by_position, const char *policy, long *n, long *clicks)
{
    Span p = obj(by_position, policy);
    *n = 0;
    *clicks = 0;
    for (int slot = 1; slot <= 3; slot++) {
        char key[8];
        snprintf(key, sizeof key, "%d", slot);
        Span s = obj(p, key);
        long ni = (long)num(s, "n"), ci = (long)num(s, "clicks");
        double ctr = num(s, "ctr");
        char label[64];
        snprintf(label, sizeof label, "%s slot %d ctr", policy, slot);
        cmp(label, (double)ci / (double)ni, ctr, 0.0);
        *n += ni;
        *clicks += ci;
    }
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    char path[1024];
    snprintf(path, sizeof path, "%s/reports/eda_all.json", root);

    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    size_t got = fread(doc, 1, sizeof doc - 1, f);
    doc[got] = '\0';
    fclose(f);
    if (got == 0) { fprintf(stderr, "%s is empty\n", path); return 2; }

    Span all = whole();
    Span by_position = obj(all, "by_position");
    Span gt = obj(all, "ground_truth");

    const double alpha = 0.05;
    const double z975 = z_for(alpha);
    printf("qnorm(0.975) solved from erfc: %.15f\n", z975);
    printf("\nper position counts summed to totals, from by_position\n");

    long n_r, c_r, n_b, c_b;
    totals(by_position, "random", &n_r, &c_r);
    totals(by_position, "bts", &n_b, &c_b);

    printf("\ntotals against the published per policy block\n");
    Span rnd = obj(all, "random"), bts = obj(all, "bts");
    cmp_long("random impressions", n_r, (long)num(rnd, "rows"));
    cmp_long("random clicks", c_r, (long)num(rnd, "clicks"));
    cmp_long("bts impressions", n_b, (long)num(bts, "rows"));
    cmp_long("bts clicks", c_b, (long)num(bts, "clicks"));

    /* CTR and the Wilson interval, which is what eda.py's ci() computes. */
    printf("\nCTR and Wilson 95%% interval, derived from the counts\n");
    double p_r = (double)c_r / (double)n_r, p_b = (double)c_b / (double)n_b;
    cmp("random ctr", p_r, num(rnd, "ctr"), 0.0);
    cmp("bts ctr", p_b, num(bts, "ctr"), 0.0);

    struct { const char *name; double p; long n; Span blk; } pol[2] = {
        { "random", p_r, n_r, rnd }, { "bts", p_b, n_b, bts }
    };
    for (int i = 0; i < 2; i++) {
        double p = pol[i].p, z = z975;
        double n = (double)pol[i].n;
        double d = 1.0 + z * z / n;
        double centre = (p + z * z / (2.0 * n)) / d;
        double half = z * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d;
        char lo[64], hi[64];
        snprintf(lo, sizeof lo, "%s ctr_lo", pol[i].name);
        snprintf(hi, sizeof hi, "%s ctr_hi", pol[i].name);
        cmp(lo, centre - half, num(pol[i].blk, "ctr_lo"), 1e-12);
        cmp(hi, centre + half, num(pol[i].blk, "ctr_hi"), 1e-12);
    }

    /* The headline comparison. */
    printf("\nthe headline, against the published ground_truth block\n");
    double diff = p_b - p_r;
    double se = sqrt(p_r * (1.0 - p_r) / (double)n_r + p_b * (1.0 - p_b) / (double)n_b);
    double zstat = diff / se;
    double pval = two_sided_p(zstat);

    cmp("absolute_diff", diff, num(gt, "absolute_diff"), 0.0);
    cmp("relative_lift", p_b / p_r - 1.0, num(gt, "relative_lift"), 0.0);
    cmp("diff_se", se, num(gt, "diff_se"), 0.0);
    cmp("z", zstat, num(gt, "z"), 0.0);
    cmp("p_value", pval, num(gt, "p_value"), 1e-12);

    Span ci;
    if (!find_key(gt, "diff_ci95", &ci) || *ci.b != '[') {
        fprintf(stderr, "no diff_ci95 array\n");
        return 2;
    }
    const char *q = ci.b + 1;
    double want_lo = strtod(q, (char **)&q);
    while (*q == ',' || *q == ' ' || *q == '\n') q++;
    double want_hi = strtod(q, NULL);
    cmp("diff_ci95 lo", diff - 1.96 * se, want_lo, 0.0);
    cmp("diff_ci95 hi", diff + 1.96 * se, want_hi, 0.0);

    if (failures) {
        printf("\n%d C checks disagree with reports/eda_all.json\n", failures);
        return 1;
    }
    printf("\nC rebuilds the whole ground truth table from the per position "
           "counts and agrees\n");
    return 0;
}
