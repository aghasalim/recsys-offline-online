# Offline metrics vs online lift, checking recommender evaluation against a known answer

Offline recommender metrics decide whether a policy ships, and they are usually
validated against nothing. The **Open Bandit Dataset** (ZOZOTOWN) logged real
production traffic under a uniform-random policy and a Bernoulli Thompson Sampling
policy at the same time, and recorded the probability of every action taken, so
both policies' true online CTR is measured. The whole project is one question:
using only the random logs, can you work out how well BTS performs?

Every number here was produced by the code in this repo. Full write-up in
[notes/METHODS.md](notes/METHODS.md), decision trail with the wrong turns kept in
at [NOTES.md](NOTES.md).

## The known answer

7 days of live traffic, both policies running concurrently over 80 items.

| policy | impressions | clicks | CTR | 95% CI |
|---|---|---|---|---|
| uniform random | 1,374,327 | 4,768 | 0.00347 | [0.00337, 0.00357] |
| Bernoulli TS | 12,357,200 | 61,208 | 0.00495 | [0.00491, 0.00499] |

**BTS is +42.77% better than random**, p = 2.5e-166. That is the target every
estimator has to hit. Four fairness assumptions get checked rather than cited,
including that both policies ran the same 7 days, so BTS's higher CTR cannot just
be a better week: [notes/METHODS.md](notes/METHODS.md#1-the-known-answer).

![ground truth](reports/eda_all.png)

## What the shortcuts get wrong

| estimator | estimate | error vs truth |
|---|---|---|
| quote the logged CTR | 0.00347 | **−30.0%** |
| replay / matched rows | 0.00763 | **+54.0%** |
| direct method, empirical | 0.00475 | −4.1% |
| direct method, logistic | 0.00355 | −28.4% |

Two reasonable-sounding methods disagree by 84 points and land on opposite sides
of the truth. Replay's error is bias, not small-sample noise: the true value sits
outside its 95% interval of [0.00551, 0.01054]. The logistic click model has the
respectable AUC, 0.5386, and the worse value estimate, so the supervised metric
ranks the two in the opposite order from the decision being made.
[Detail](notes/METHODS.md#2-what-the-naive-offline-evaluation-says).

![the two shortcuts against the corrected estimators](reports/baselines.png)

## What the propensity correction fixes

| estimator | estimate | error vs truth | 95% CI | covers truth |
|---|---|---|---|---|
| IPS | 0.00504 | **+1.7%** | [0.00480, 0.00527] | yes |
| SNIPS | 0.00503 | **+1.6%** | [0.00479, 0.00527] | yes |
| doubly robust (cross-fitted) | 0.00503 | **+1.6%** | [0.00480, 0.00527] | yes |

A range of −30% to +54% collapses to under 2% and every interval covers the truth.
I predicted heavy-tailed weights and there are none: uniform logging caps the
largest weight in 1.37 million rows at 9.64. Clipping, the standard advice, only
adds bias here, and all four corrected estimators agree to within 0.1 points, so
this benchmark cannot separate them.
[Weights and the clipping sweep](notes/METHODS.md#3-what-the-propensity-correction-fixes).

![off-policy evaluation](reports/ope_all.png)

## Where it breaks

Evaluating from uniform-random logs is the easy direction and nobody has those
logs in production, so: three stress tests, each still graded against a known
answer. Reversing the direction takes the largest weight from 9.6 to 12,500 and
effective sample size from 26.8% to 0.16%. Shrinking the data keeps the intervals
honest. Deleting the highest-CTR items from the logs is the one that breaks it,
and ESS does not merely miss that failure, it moves the wrong way: at 40 items
removed the estimate is off by −89.6% and ESS has gone up to 47.6%. What does
track the error, at no cost, is the target policy's probability mass on actions
the logs never saw. [Full stress tests](notes/METHODS.md#4-where-it-breaks).

![where it breaks](reports/stress_all.png)

![support erosion and the diagnostic that anticipates it](reports/support.png)

![the support sweep, step by step](reports/support-erosion.gif)

*One step per frame, deleting the highest-CTR items from the logs. The
estimator and the target policy never change, so the drift away from the
measured true CTR line is support erosion alone.*

## The deliverable is a gate, not a model

`harness.audit()` returns a value only when the diagnostics pass, and otherwise
returns `value=None` plus the reasons, because a withheld number cannot be pasted
into a slide and a wrong one can. The thresholds are not fitted. Graded on
interval coverage against the known answers, 6 of 7 right at a 10% tolerance:

| scenario | gate | true error | unlogged mass | ESS | correct |
|---|---|---|---|---|---|
| forward (random → BTS) | ok | +1.6% | 0.0% | 367,933 | yes |
| reverse (BTS → random) | ok | −5.0% | 0.0% | 19,910 | **no** |
| top 5 items unlogged | refuse | −7.9% | 17.7% | 310,817 | yes |
| top 20 items unlogged | refuse | −28.3% | 66.8% | 259,170 | yes |
| top 60 items unlogged | refuse | −63.9% | 95.2% | 135,418 | yes |
| n = 687 | refuse | −94.3% | 4.6% | 196 | yes |
| n = 69 | refuse | −100.0% | 74.1% | 24 | yes |

The thresholds, the API and the two things I got wrong while building it are in
[notes/METHODS.md](notes/METHODS.md#5-the-deliverable-is-a-gate-not-a-model).

![what the gate decided on each scenario](reports/gate.png)

## Limitations

The gate's one false accept is the open problem. In the reverse direction its
interval [0.003131, 0.003464] misses the truth by 1.0 half-widths, because at max
weight 12,500 and ESS 0.16% the normal approximation behind the standard error is
marginally anti-conservative. Tightening a threshold to catch that would mean
picking the threshold by looking at the answer, so it stays documented; a
bootstrap or empirical-Bernstein interval is the honest fix. What tightening it
would cost is measured rather than guessed in [the threshold
sweep](#everything-here-is-recomputed-in-another-language) below: an ESS floor
above 19,910 does get all seven scenarios right, and costs one or two correct
decisions on the wider grid of scenarios the gate was never scored on.
Everything here also rests on one 7-day window from one retailer, and the BTS
target policy is modelled as context-free but position-dependent, matching the
Open Bandit benchmark.

## Everything here is recomputed in another language

Every number above came out of one pipeline: pandas and numpy under `src/roo/`,
writing JSON into `reports/`. Everything downstream reads that JSON, this README
included, so a mistake in the pipeline had nothing to catch it. The self-checks
CI already runs test that each estimator returns the right answer on hand-built
data; they say nothing about the numbers actually published.

So each published table is now rebuilt from a rawer layer of the same data by an
implementation in another language, and CI fails if any two disagree. Two of
those rawer layers were already in the repository without being used as one.
`reports/eda_all.json` records per-position impression and click counts that sum
to the headline table. `app_data/grid_all.json` records, for fifteen scenarios,
the three scalars the gate decides on, which is one level below the gate table
itself.

| implementation | what it recomputes, and from what | agreement |
|---|---|---|
| [`verify/groundtruth.c`](verify/groundtruth.c) | the known-answer table from the per-position counts: totals, CTR, both Wilson intervals, the difference of proportions, z and p | 22 of 23 fields bit-identical, p value 2.8e-14 relative |
| [`verify/gate.sql`](verify/gate.sql) | the gate table in SQLite, from the per-scenario diagnostics in `app_data/grid_all.json` | exact, 0.0e+00, on all 7 scenarios |
| [`verify/gocheck`](verify/gocheck) | structure of all 7 published data files, then the gate table a second time | 7 of 7 files sound, gate exact |
| [`verify/verify.R`](verify/verify.R) | the headline test in base R by two routes, plus an exact interval and a bootstrap the repository never had | exact to 1e-15, p value within 1.0e-14 |
| [`verify/estimators.rb`](verify/estimators.rb) | every derived column of `ope_all.json` and `stress_all.json`, from its own estimate and standard error | exact, 0.0e+00 |
| [`verify/readme.js`](verify/readme.js) | this README against the files: 65 table cells and 14 sentences that carry a number | every one matches |
| [`verify/thresholds`](verify/thresholds) | whether the gate's two thresholds were fitted, over the entire threshold plane | see below |

Run them all with [`./verify/verify.sh`](verify/verify.sh), which prints
`7 passed, 0 failed, 0 skipped`. Each is skipped with a message if its toolchain
is missing, so a partial install still runs the rest.

**R puts an interval on a number this repo only ever stated as a point.** The
+42.77% lift has no interval anywhere above. Base R produces one two ways, a
delta method on the log ratio and 200,000 parametric bootstrap draws, and gets
[+38.63%, +47.03%] and [+38.66%, +47.08%], whose worst end differs by 0.57% of
the interval width. It also computes the Clopper-Pearson exact intervals, which
come out wider than the published Wilson ones as they have to, and it puts a
number on the failure Limitations admits: the reverse direction's interval misses
the truth by 1.0339 half-widths.

**The Rust answers a question that was never asked.** `harness.py` says its two
thresholds were not chosen by looking at which value made the answers come out
right, and there was no way to check that, because the gate had only ever been
run at one setting. The decision is a step function of the two thresholds, so
the plane splits into 126 regions and enumerating one point per region is an
exact answer; a brute force pass over 16,000,000 threshold pairs finds the same
maxima. At the published 1% and 1,000 the gate scores 6 of 7 on the scenarios
above, and 11 of 13 on the distinct scenarios in `app_data/grid_all.json`. Some
setting does get all 7: any ESS floor above 19,910 and up to 367,933, which is
4.76% of the plane. Across that whole region the wider set drops to 9 or 10 of
13. The best any pair reaches on the wider set is 12 of 13, at a 5.98% mass
limit and an ESS floor just above 196, and no single pair reaches both maxima.
So the false accept in Limitations is fixable and the fix costs more than it
buys, which is what "picking the threshold by looking at the answer" means in
numbers.

**Go found something nobody was looking for.** Two of the seven files are not
valid JSON. Python writes bare `NaN` and `Infinity` for the direct method's
missing standard error and for one interval width ratio that divides by zero,
and no parser is obliged to accept either. Both are legitimate where they sit,
so the check is that they appear nowhere else, along with duplicate keys,
inverted intervals and negative counts.

**The harness is itself checked.** CI corrupts `reports/harness_all.json`,
requires the harness to reject it, restores it and requires a pass. A check that
cannot fail is not evidence. Each implementation catches what it is responsible
for and nothing more: moving one click between position slots is caught by C and
R; nudging the published p value by C, R and JavaScript; widening a gate
interval until it covers the truth by SQL, Go and Rust; raising the published
gate score to 7 of 7 by SQL, Go, JavaScript and Rust; moving an IPS interval end
by Ruby and JavaScript; telling one clipping row it clipped nothing by Ruby
alone; writing a key twice by Go alone; and a stale ESS left in the README table
by JavaScript alone.

## Demo

```bash
uv run streamlit run app.py
# or
docker build -t roo . && docker run -p 7860:7860 roo
```

The gate's decision depends on three scalars, so the thresholds are live: move
them and the real gate logic re-runs on the full-data numbers above. True values
are shown on purpose, so you can watch the estimator be confidently wrong while
the gate withholds it.

## Reproducing

```bash
uv sync
curl -L -o data/obd.zip https://research.zozo.com/data_release/open_bandit_dataset.zip
unzip -q data/obd.zip -d data/          # 400 MB download, 11 GB unpacked

uv run python src/roo/prepare.py --campaigns all  # 7 GB csv -> 78 MB parquet
uv run python src/roo/eda.py all                  # the known answer
uv run python src/roo/baseline.py all             # the naive estimators
uv run python src/roo/ope.py all                  # IPS / SNIPS / DR
uv run python src/roo/stress.py all               # where the correction fails
uv run python src/roo/harness.py all              # the gate, scored against truth
uv run python src/roo/harness.py all --export-grid # data for the demo app

# self-checks: no dataset needed, so CI does not need the 11 GB download
for m in baseline ope stress harness; do uv run python src/roo/$m.py --self-check; done
```

Raw logs and the derived Parquet are gitignored. Stack and roadmap in
[notes/METHODS.md](notes/METHODS.md#9-stack).

## Layout

```
src/roo/    prepare, eda, baseline, ope, stress, harness, one per milestone
src/rsoo/   figure generation
reports/    json results and the figures above
app_data/   precomputed diagnostics for app.py, the Streamlit demo
notes/      METHODS.md, the full write-up
verify/     the published numbers, recomputed in seven other languages
```

## Data, author, licence

[Open Bandit Dataset](https://research.zozo.com/data.html) (ZOZO Research),
released for research use.

Built by Aghasalim Mustafazada, a third-year Applied Computer Science (AI)
student. My code is MIT, see [LICENSE](LICENSE).

## References

The papers and sources this implementation follows. Each one is here because
the code uses the method, the dataset or the metric it describes.

- **Dudík, Langford, Li. Doubly Robust Policy Evaluation and Learning. ICML 2011.** [arXiv:1103.4601](https://arxiv.org/abs/1103.4601) the doubly robust estimator.
- **Swaminathan, Joachims. The Self-Normalized Estimator for Counterfactual Learning. NeurIPS 2015.** SNIPS, the self normalised variant.
- **Gilotte, Calauzènes, Nedelec, Abraham, Dollé. Offline A/B testing for Recommender Systems. WSDM 2018.** [arXiv:1801.07030](https://arxiv.org/abs/1801.07030) the offline versus online comparison this repo is built around.
