# Decision trail

A running log of what I tried, what broke, and what I caught. Newest entries at
the bottom. This file is the point of the project, the estimators are downstream
of it.

Ground rule I set for myself: **no number in this file that I did not personally
run.** Several entries below are me being wrong in writing, because deleting
those would make the log useless.

---

## 1. Picking a dataset where the answer is already known

A recommender project has a structural problem: the metric you can compute
offline (NDCG, AUC, recall@k) is not the quantity you care about (clicks if you
deploy it). You can report the first one forever without learning whether it
predicts the second. Most portfolio projects stop there, and there is no way for
the reader, or the author, to know if the work was any good.

So the dataset choice came before anything else, and the requirement was: *the
online answer must already have been measured.* That rules out MovieLens and
every offline ranking benchmark.

**Open Bandit Dataset** (ZOZOTOWN) qualifies. Two policies, uniform random and
Bernoulli Thompson Sampling, ran on live traffic *at the same time*, with the
probability of every action logged. So each policy's true online CTR is a
measured quantity, and any offline estimate can be graded against it.

The whole project reduces to one gradeable question: **from the random logs
alone, how well does BTS perform?**

## 2. Checking the assumptions instead of citing them

The known answer is only fair if four things hold, so I measured all four
(`eda.py`) rather than repeating them from the paper:

| assumption | result |
|---|---|
| logging policy is uniform | exactly one distinct propensity, 0.0125 = 1/80 |
| every action has support | 80/80 items appear under random |
| no seasonality confound | both policies ran the same 7 days |
| slots are not interchangeable | CTR reported per position |

The third is the one that would have killed the project silently. If BTS had run
a different week, its higher CTR could just be a better week, and there would be
no true answer at all. It shares all 7 days, so the comparison is clean.

Ground truth: random 0.00347, BTS 0.00495, **+42.77%**, p = 2.5e-166.

## 3. A parsing bug that Parquet made invisible

`pd.read_csv(..., parse_dates=["timestamp"])` silently did nothing on these
tz-aware ISO8601 strings, the column stayed`object`, and`to_parquet` happily
stored it as`str`. Nothing failed at write time. It surfaced two steps later as
`Can only use .dt accessor with datetimelike values`, in a function that had
nothing to do with parsing.

Fixed by converting explicitly and asserting the dtype took, in`prepare.py`.
The assert is the point: a silent type downgrade that only explodes downstream
is worse than a crash.

Also measured before discarding: the 80`user-item_affinity_*` columns are
**0.0637% non-zero**. That is why 7 GB of CSV becomes 78 MB of Parquet. I
checked rather than assumed, because if they had carried signal they were the
only user-item interaction features in the dataset.

## 4. The naive baseline, and the number I would have shipped

Estimating BTS's CTR from the random logs the usual ways:

| method | estimate | error |
|---|---|---|
| quote the logged CTR | 0.00347 | −30.0% |
| replay / matched rows | 0.00763 | **+54.0%** |
| direct method, empirical | 0.00475 | −4.1% |
| direct method, logistic | 0.00355 | −28.4% |

Two respectable-sounding methods disagree by 84 points and land on opposite
sides of the truth.

Replay keeps only 1.23% of rows, so the obvious objection is that +54% is
small-sample noise. It is not: truth 0.00495 sits outside replay's 95% interval
[0.00551, 0.01054]. I added that interval specifically because I expected to be
asked, and I would rather answer it than assert it.

**The part I would put in front of an interviewer:** the logistic click model
has a real, reportable AUC of 0.5386 and its value estimate is off by 28%. The
empirical direct method is a lookup table with no AUC to report and is off by
4%. The supervised metric ranks the two approaches in the *opposite* order from
the decision being made.

## 5. I predicted heavy-tailed weights. I was wrong.

At the end of milestone 1 I wrote that BTS concentrates 12% of impressions on
one item where random spreads 1.25% each, so importance weights would be
heavy-tailed and IPS would blow up.

Measured: the largest weight across 1,374,327 rows is **9.64**.

The reason is arithmetic I should have done before writing the prediction.
Uniform logging gives every action probability 1/80, so a weight cannot exceed
`80 × max π_target`, and BTS's most concentrated action is 0.1205 → 9.64. Heavy
tails require a concentrated **logging** policy. Evaluating *from* uniform-random
logs is the easy direction, and I had the direction backwards.

Consequence: IPS, SNIPS, cross-fitted DR and the direct method all land within
1.6 to 1.7% of truth, and all three intervals cover it.

## 6. Clipping made it worse, and all four estimators agreed

Weight clipping is standard advice. Here every cap that binds is pure bias
(cap 1 → −58.9%, cap 5 → −7.7%) and every cap ≥ 10 is a no-op, because nothing
exceeds 9.64. Clipping trades variance for bias and there was no variance
problem to trade against.

Two things I made myself write down rather than gloss:

- All four corrected estimators agree to within 0.1 points, which means **this
  benchmark cannot discriminate between them.** "DR beats IPS" would not be
  supported by this evidence.
- The direct method scores +1.6% here and −4.1% in milestone 2. That is a
  training-set-size difference (7 days vs 5), not a method difference.
  Comparing those two numbers directly would be a mistake.

## 7. Making it hard on purpose

Milestone 3 came out clean, which was a warning rather than a result. So I built
the regimes it had avoided:

**Direction.** Swap which policy did the logging: max weight goes 9.6 → **12,500**
and ESS 26.77% → **0.16%**. Twelve million rows behave like twenty thousand. The
estimate still lands at −1.0%, and that is the dangerous part, it survived
because the log is enormous, not because the setup was sound.

**Sample size.** Subsampled to 12k rows the point estimate bounces ±15%, but
every interval still covers the truth. The intervals degrade honestly.

**Support.** Delete the top-CTR items from the logs while leaving them in the
target policy:

| items removed | error | ESS | unlogged mass |
|---|---|---|---|
| 0 | +1.7% | 26.77% | 0.0% |
| 20 | −68.4% | 25.11% | 66.8% |
| 40 | **−89.6%** | **47.62%** | 88.6% |
| 60 | −93.1% | 39.41% | 95.2% |

**Effective sample size does not merely fail to detect this, it moves in the
wrong direction.** At −89.6% error, ESS has *risen* to 47.6%, because deleting
actions leaves a beautifully well-conditioned set of surviving weights. The
confidence intervals stop covering the truth from 5 items onward. The estimator
is confidently, precisely wrong, and every diagnostic anyone routinely computes
looks healthy.

What does detect it costs nothing and needs no labels: the target policy's
probability mass on unlogged actions, which tracks the error almost 1:1 (17.7%
vs −19.0%, 88.6% vs −89.6%). The missing mass *is* the uncounted value.

## 8. Building a gate, and grading it on the wrong thing

Given a failure that looks healthy from the outside, the useful artefact is not
a better estimator, it is something that refuses to answer.`harness.audit()`
returns`value=None` plus reasons when the checks fail, a withheld number
cannot be pasted into a slide, a wrong one can.

The thresholds are deliberately not fitted. The unlogged-mass limit is whatever
bias you accept, because error ≈ mass was *measured* in §7. The ESS floor is an
absolute count, not a fraction, because 0.16% ESS is fine on 12M rows and fatal
on 100k.

**My first scoring criterion was wrong.** I graded the gate on
|point estimate − truth| ≤ 10%, which failed the`n = 6,872` scenario. Looking
at it properly: the estimate was 43% high, but its interval was
[0.00251, 0.01164], which *contains* the truth, and the gate had already warned
that the interval was 129% as wide as the estimate. Grading a point estimate
when the tool reports an interval is simply the wrong test. Re-scored on
interval coverage.

That re-scoring also moved which scenario fails, which is the useful part.

## 9. ESS is the wrong sufficiency check for a rare outcome

Chasing the above,`n = 6,872` has an ESS of 1,890, comfortably over my 1,000
floor, and amounts to **9 expected clicks** at a 0.5% click rate. The binding
constraint for a rare event is the effective number of *positive* events, not of
rows. Added`effective_clicks` to the diagnostics.

## 10. The gate's one false accept, left in place on purpose

Scored on interval coverage: **6/7**. The failure is the reverse direction.

- truth 0.003469
- estimate 0.003297 (−5.0%)
- 95% CI **[0.003131, 0.003464]**: misses the truth by 1.0 half-widths

Support is perfect and ESS is 19,910, so every check passes. The cause is that
the interval itself is untrustworthy here: with max weight 12,500 and ESS at
0.16%, the sum is dominated by a thin tail and the normal approximation behind
`std/√n` is marginally anti-conservative. The point estimate is fine; the
*uncertainty* is understated.

I could close this by tightening a threshold until this case refuses. I did not,
because any threshold that catches it would have been chosen by looking at the
answer, the exact failure this project is about. The honest fix is an interval
that does not assume a light tail (bootstrap or empirical Bernstein), and that
is the open problem the project ends on.

## 11. Smaller things worth remembering

- Appending a function *after* the`if __name__ == "__main__"` block means
`main()` runs before it exists. Cost me one`NameError`; entrypoints now sit
  at the bottom of every module.
- The self-checks earned their place twice: once catching my own arithmetic in
  a kNN assertion, once catching a signature change when`replay_value` grew a
  confidence interval and`demo()` still unpacked two values.
- Every self-check asserts a metric **fails** on a deliberately wrong input, not
  just that it runs.`aupro`-style "returns a number" tests would have passed
  through all of the bugs above.
