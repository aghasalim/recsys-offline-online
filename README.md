# Offline metrics vs online lift — checking recommender evaluation against a known answer

Built by a third-year Applied Computer Science (AI) student.

> **Status: in progress.** Milestones 1–3 are done — ground truth established,
> naive evaluation measured against it, and the propensity-corrected estimators
> built and scored. Milestone 4 is the interesting one: finding the regime where
> the correction stops working. Numbers below are real and reproducible.

The problem with a recommender portfolio project is that the metric you report
is not the thing anyone cares about. You train a ranker, you report NDCG@10 or
AUC, and none of it tells you whether deploying it would earn more clicks than
what you already have. The offline number and the online number are different
quantities, and a project that only shows the offline one cannot tell you if it
was any good.

So this uses the one public dataset where the online answer is already known.
**Open Bandit Dataset** (ZOZOTOWN) logged real production traffic under two
different policies at the same time — a uniform-random one and a Bernoulli
Thompson Sampling one — and recorded the probability of every action taken.
That means the true online CTR of each policy is measured, and any offline
estimate computed from one policy's logs can be graded against it.

The whole project is one question: **using only the random logs, can you work
out how well BTS performs?** The answer is written down before any estimator
runs.

---

## The known answer

7 days of live traffic, both policies running concurrently over 80 items.

| policy | impressions | clicks | CTR | 95% CI |
|---|---|---|---|---|
| uniform random | 1,374,327 | 4,768 | 0.00347 | [0.00337, 0.00357] |
| Bernoulli TS | 12,357,200 | 61,208 | 0.00495 | [0.00491, 0.00499] |

**BTS is +42.77% better than random**, p = 2.5e-166. That is the target every
estimator has to hit.

Four things have to hold for this to be a fair test, so I checked them instead
of citing them (`make eda`):

| assumption | check | result |
|---|---|---|
| logging policy is really uniform | distinct propensity values | exactly one, 0.0125 = 1/80 |
| every action has support | items seen under random | 80 / 80 |
| no seasonality confound | days both policies ran | 7 shared out of 7 |
| slots are not interchangeable | CTR by position | reported separately |

The third one is the one that would have killed the project quietly. If BTS had
run a different week, its higher CTR could just be a better week, and there
would be no true answer to grade against.

![ground truth](reports/eda_all.png)

## What the naive offline evaluation says

Estimating BTS's CTR from the random logs, the way it usually gets done:

| estimator | estimate | error vs truth | |
|---|---|---|---|
| quote the logged CTR | 0.00347 | **−30.0%** | ignores the policy entirely |
| replay / matched rows | 0.00763 | **+54.0%** | kept 1.23% of the logs |
| direct method, empirical | 0.00475 | −4.1% | per-(item, slot) CTR table |
| direct method, logistic | 0.00355 | −28.4% | AUC 0.5386 |

**Two reasonable-sounding methods disagree by 84 points and land on opposite
sides of the truth.** Ship replay and you claim BTS is half again better than it
is. Quote historical CTR and you claim it is a third worse.

Replay throws away 99% of the data, so the obvious objection is that its error
is just noise from a small sample. It is not: the true value 0.00495 sits
**outside** replay's 95% interval of [0.00551, 0.01054]. That is bias from
quietly changing which impressions get averaged, not variance.

### The AUC column is the point

The logistic click model produces a respectable-looking **AUC of 0.5386** and a
log loss you could put on a slide, and its value estimate is off by 28%. The
empirical direct method is a lookup table — no model, no AUC to report — and it
is off by 4%.

So the supervised metric ranks the two approaches in the **opposite order** from
the thing being decided. That is the whole argument for this project: an offline
ranking metric is a statement about a click model, and the decision you are
making is about a policy.

## What the propensity correction fixes

The logs record the probability of every action taken. Reweighting each logged
reward by `pi_target(a|x) / pi_logging(a|x)` converts "what happened under
random" into "what would happen under BTS":

| estimator | estimate | error vs truth | 95% CI | covers truth |
|---|---|---|---|---|
| IPS | 0.00504 | **+1.7%** | [0.00480, 0.00527] | yes |
| SNIPS | 0.00503 | **+1.6%** | [0.00479, 0.00527] | yes |
| doubly robust (cross-fitted) | 0.00503 | **+1.6%** | [0.00480, 0.00527] | yes |
| direct method, all data | 0.00503 | +1.6% | — | — |

A range of −30% to +54% collapses to under 2%, and every interval contains the
true value. That is the correction doing exactly what the theory says.

![off-policy evaluation](reports/ope_all.png)

### I predicted the weights would be heavy-tailed. They are not.

Milestone 1 ended by noting that BTS puts 12% of its impressions on one item
where random puts 1.25% on each, and I said importance weights over that gap
would be heavy-tailed and wreck IPS. Measured, the largest weight in 1.37
million rows is **9.64**.

The reason is arithmetic I should have done first. Uniform logging gives every
action probability 1/80, so no weight can exceed `80 × max pi_target`, and BTS's
most concentrated action is 0.1205 → 9.64. Heavy tails need a *logging* policy
that is concentrated, not a target policy. Evaluating from uniform-random logs
is the easy direction, and this dataset only supports the easy direction from
the random side.

So the honest summary is that IPS's famous variance problem does not appear
here at all. Effective sample size is 367,933 of 1,374,327 rows (26.8%) — a real
cost, but nowhere near collapse.

### Clipping the weights, the standard advice, only makes it worse

| clip cap | estimate | error | ESS | % clipped |
|---|---|---|---|---|
| 1 | 0.00204 | −58.9% | 838,761 | 24.5% |
| 2 | 0.00309 | −37.7% | 642,757 | 15.1% |
| 5 | 0.00457 | −7.7% | 439,608 | 5.4% |
| 10 | 0.00504 | +1.7% | 367,933 | 0.0% |
| none | 0.00504 | +1.7% | 367,933 | 0.0% |

Clipping trades variance for bias, and here there is no variance problem to
trade against — so every cap that binds is pure damage, and every cap that
doesn't bind is a no-op. "Clip your importance weights" is good advice in the
regime it was written for, and actively harmful in this one.

### The uncomfortable part

All four corrected estimators agree to within 0.1 points of each other. That
means this benchmark **cannot tell them apart** — it is a well-behaved problem
with 80 actions, full support, and uniform logging. Any claim that DR beats IPS
would not be supported by this evidence.

Note also that the direct method scores +1.6% here versus −4.1% in milestone 2.
That is not a method difference: milestone 2 fit it on 5 days and this fits it
on all 7. Comparing those two numbers directly would be comparing training set
sizes and calling it an estimator comparison.

## Reproduce

```bash
uv sync
# 400 MB download, 11 GB unpacked
curl -L -o data/obd.zip https://research.zozo.com/data_release/open_bandit_dataset.zip
unzip -q data/obd.zip -d data/

uv run python src/roo/prepare.py --campaigns all   # 7 GB csv -> 78 MB parquet
uv run python src/roo/eda.py all                   # the known answer + assumption checks
uv run python src/roo/baseline.py all              # the naive estimators
uv run python src/roo/ope.py all                   # IPS / SNIPS / DR + diagnostics
```

Self-checks, which assert each estimator returns the arithmetically correct
answer on hand-built data:

```bash
uv run python src/roo/baseline.py --self-check
uv run python src/roo/ope.py --self-check
```

The raw logs and the derived Parquet are gitignored. 80 of the 89 raw columns
are a user-item affinity vector that is **0.0637% non-zero**, measured before
dropping it, which is most of why 7 GB compresses to 78 MB.

## Roadmap

- [x] **1 — Ground truth.** Prepare the logs, measure both policies' true online
      CTR, verify the four assumptions that make the comparison fair.
- [x] **2 — Naive baseline.** Replay, direct method, and a supervised ranker,
      each scored against the known answer.
- [x] **3 — Corrected estimators.** IPS, self-normalised IPS, doubly robust
      with cross-fitting, plus ESS, weight tails and a clipping sweep.
- [ ] **4 — When the correction fails.** Weight clipping, heavy tails, and the
      regime where every estimator is untrustworthy.
- [ ] **5 — Deployment.** An evaluation harness plus a demo that shows the
      offline/online gap interactively.
- [ ] **6 — Docs.** Full write-up with the failures kept in.

Milestone 4 exists because milestone 3 came out clean. Evaluating from
uniform-random logs is the easy direction: full support, bounded weights, every
estimator agreeing. The hard direction is evaluating a *concentrated* policy's
logs — swap logging and target, or evaluate a deterministic greedy ranker, and
the weights are no longer bounded by 9.64. That is where IPS should break, and
where the diagnostics built here have to earn their place.

## Stack

Python 3.12, pandas, scikit-learn, SciPy, NumPy, matplotlib, PyArrow. Managed
with `uv`, linted with `ruff`.

## Data

[Open Bandit Dataset](https://research.zozo.com/data.html) (ZOZO Research),
released for research use. My code is MIT.
