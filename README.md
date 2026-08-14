# Offline metrics vs online lift — checking recommender evaluation against a known answer

Built by a third-year Applied Computer Science (AI) student.

> **Status: in progress.** Milestones 1 and 2 are done — the ground truth is
> established and the naive offline evaluation is measured against it. The
> corrected estimators (IPS, SNIPS, doubly robust) are next. Numbers below are
> real and reproducible; nothing here is a placeholder.

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

## Reproduce

```bash
uv sync
# 400 MB download, 11 GB unpacked
curl -L -o data/obd.zip https://research.zozo.com/data_release/open_bandit_dataset.zip
unzip -q data/obd.zip -d data/

uv run python src/roo/prepare.py --campaigns all   # 7 GB csv -> 78 MB parquet
uv run python src/roo/eda.py all                   # the known answer + assumption checks
uv run python src/roo/baseline.py all              # the naive estimators
```

Self-checks, which assert each estimator returns the arithmetically correct
answer on hand-built data:

```bash
uv run python src/roo/baseline.py --self-check
```

The raw logs and the derived Parquet are gitignored. 80 of the 89 raw columns
are a user-item affinity vector that is **0.0637% non-zero**, measured before
dropping it, which is most of why 7 GB compresses to 78 MB.

## Roadmap

- [x] **1 — Ground truth.** Prepare the logs, measure both policies' true online
      CTR, verify the four assumptions that make the comparison fair.
- [x] **2 — Naive baseline.** Replay, direct method, and a supervised ranker,
      each scored against the known answer.
- [ ] **3 — Corrected estimators.** IPS, self-normalised IPS, doubly robust,
      with effective sample size and weight diagnostics.
- [ ] **4 — When the correction fails.** Weight clipping, heavy tails, and the
      regime where every estimator is untrustworthy.
- [ ] **5 — Deployment.** An evaluation harness plus a demo that shows the
      offline/online gap interactively.
- [ ] **6 — Docs.** Full write-up with the failures kept in.

Milestone 1 already shows why milestone 3 will be hard: BTS puts 11% of its
impressions on a single item where random puts 1.25% on each. Importance
weights over that gap are heavy-tailed, which is precisely the regime where IPS
has enormous variance.

## Stack

Python 3.12, pandas, scikit-learn, SciPy, NumPy, matplotlib, PyArrow. Managed
with `uv`, linted with `ruff`.

## Data

[Open Bandit Dataset](https://research.zozo.com/data.html) (ZOZO Research),
released for research use. My code is MIT.
