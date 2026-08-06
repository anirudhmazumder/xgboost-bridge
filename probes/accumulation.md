# Probe: margin accumulation order and precision

Empirical determination of the accumulation order and precision that reproduces
`predict(output_margin=True)` **bit-exactly**, for `reg:squarederror`, `binary:logistic`,
and `survival:cox`, booster `gbtree`.

This probe was opened to close the one residual left by `probes/float32_thresholds.md`:
leaf assignment was already exact on 5000/5000 rows, but no accumulation variant tried
there was bit-exact on all rows — its best was `2878/5000` at max abs error
`4.76837158203125e-07`.

**Headline: the residual was never an accumulation problem.** It was the `base_score`
margin intercept. Applying the corrected logit from `probes/base_score.md` §5, with no
change whatsoever to the summation, takes the count from partial to **5000/5000, max abs
error exactly `0.0`**. The accumulation variant that was already being used was the right
one.

Every claim below is backed by a pasted command and its real output. Anything not directly
measured is labelled **INFERRED**.

---

## Environment

```
$ uv run python -c "
import sys, numpy, xgboost
print('python', sys.version)
print('xgboost', xgboost.__version__)
print('numpy', numpy.__version__)
"
python 3.12.8 (main, Dec  3 2024, 18:42:41) [Clang 16.0.0 (clang-1600.0.26.4)]
xgboost 3.3.0
numpy 2.5.1

$ node --version
v20.19.0
```

Platform darwin 25.5.0, arm64. Artifact version marker `[3, 3, 0]`. Matches the D001 pin.

---

## Verdicts up front

| Question | Answer |
|---|---|
| Did the corrected intercept alone close the residual? | **Yes.** `2336/5000` → `5000/5000`, max abs err `4.76837158203125e-07` → `0.0`. No summation change. |
| Winning accumulation variant | **V1** — intercept first, float32 running sum, serialized tree order, leaves narrowed to float32 on read. |
| V1 bit-exact count | `5000/5000` on **every** one of the 48 model configurations measured. Max abs error `0.0`. |
| Runner-up | `3706/5000`. Every non-V1 variant lost by 1294–4801 rows, with one degenerate tie noted in §2. |
| Do leaf values need float32 narrowing on read? | **Yes.** Same accumulation, leaves left as the float64 parse: `70/5000` to `3706/5000`, never all. |
| Is XGBoost's margin bit-stable across `nthread`, batch size, and predict API? | **Yes.** 0 divergences in 1440 per-row configurations plus 42 full-batch comparisons. Bit-exact parity against XGBoost is well defined. |
| Is exactly-`0.0` cross-language parity reachable? | **Yes.** Python V1 vs JavaScript V1: `25000/25000` bit-identical, max abs difference `0.0`. |

---

## Method

Synthetic data only: 5000 rows × 6 columns from `numpy.random.default_rng(20260801)`,
generic feature names `f0`..`f5`, deterministic nonlinear signal, `max_depth=4`,
`tree_method=exact` unless stated, `seed=20260801`, `nthread=1` at fit time. Cox labels use
the sign convention (positive = event, negative = right-censored). Nothing was written into
the repository; models and scripts live in the session scratch directory.

Two method commitments that matter:

1. **Comparisons are on float32 bit patterns**, `np.float32(x).view(np.uint32)` in Python
   and a `Float32Array`/`Uint32Array` overlay in JavaScript. A residual of `1e-7` is not
   evidence of anything; a count of `5000/5000` is.
2. **The feature matrix and the XGBoost margins cross the language boundary as float32 bit
   patterns**, not as decimals. No formatting step sits anywhere in the Python→JavaScript
   path, so the JS cross-check cannot be softened by a lossy re-serialization.

The intercept transforms are **not re-derived here** — they are taken verbatim from
`probes/base_score.md` §5 and §9:

```
reg:squarederror  ->  float32(base_score)
binary:logistic   ->  -log(float32(float32(1/float32(p)) - 1))      NOT log(p/(1-p))
survival:cox      ->  log(float32(base_score))
```

---

## 1. The corrected intercept, alone, closes the residual

Single change: the intercept. Same tree walk, same float32 running sum, same tree order.

```
$ uv run python a01_intercept_first.py
=== reg:squarederror  rounds=8 trees=8 ===
  base_score raw            : '[6.0321754E-1]'
  base_score float64 parse  : 0.60321754
  base_score float32        : 0.6032175421714783
  ref margin dtype          : float32
  rows                      : 5000
  leaf assignment exact rows: 5000 / 5000
  intercept identity(f32 base_score)           = 0.6032175421714783
    f32 running sum, intercept FIRST -> bit-exact 5000 / 5000   max abs err 0.0

=== binary:logistic  rounds=8 trees=8 ===
  base_score raw            : '[5.8E-1]'
  base_score float64 parse  : 0.58
  base_score float32        : 0.5799999833106995
  ref margin dtype          : float32
  rows                      : 5000
  leaf assignment exact rows: 5000 / 5000
  intercept logit XGB (f32 1/p-1, f64 log)     = 0.322773277759552
    f32 running sum, intercept FIRST -> bit-exact 5000 / 5000   max abs err 0.0
  intercept logit textbook f64 log(p/(1-p))    = 0.3227733373641968
    f32 running sum, intercept FIRST -> bit-exact 2336 / 5000   max abs err 4.76837158203125e-07
  intercept logit textbook all-f32             = 0.322773277759552
    f32 running sum, intercept FIRST -> bit-exact 5000 / 5000   max abs err 0.0

=== survival:cox  rounds=8 trees=8 ===
  base_score raw            : '[1E0]'
  base_score float64 parse  : 1.0
  base_score float32        : 1.0
  ref margin dtype          : float32
  rows                      : 5000
  leaf assignment exact rows: 5000 / 5000
  intercept ln(f32 base_score)                 = 0.0
    f32 running sum, intercept FIRST -> bit-exact 5000 / 5000   max abs err 0.0
  intercept ln(f64 base_score)                 = 0.0
    f32 running sum, intercept FIRST -> bit-exact 5000 / 5000   max abs err 0.0
```

**Before: `2336 / 5000`, max abs error `4.76837158203125e-07`.
After: `5000 / 5000`, max abs error `0.0`.**

That error magnitude is the same `4.76837158203125e-07` the earlier probe reported, and the
partial count is in the same band as its `2878/5000`. The two probes used different data and
different seeds, so the counts are not expected to match digit-for-digit; the *signature*
does.

`reg:squarederror` is the control that makes this conclusive. Its intercept transform is the
identity, so no logit enters the chain — and it is `5000/5000` on the first attempt with a
plain float32 running sum. If summation order or precision were the problem, regression
would have shown a residual too. It does not.

Note the coincidence in the `binary:logistic` block: at `p = 0.58` the all-float32 textbook
logit happens to land on the same float32 as XGBoost's formula, so it also scores
`5000/5000`. That is a property of this value, not of the formula. `probes/base_score.md`
§5 measured the textbook form wrong on 16 of 27 values, and §5 below shows it failing here
at `p = 0.987654`.

### 1.1 The prior probe's own `base_score` value, run to ground

`probes/float32_thresholds.md` recorded `base_score = '[5.1953125E-1]'` and printed two
candidate intercepts. **Both of them narrow to the same float32, and that float32 is 7 ULP
away from the one XGBoost actually uses.**

```
$ uv run python a11_prior_residual.py
token                            : '5.1953125E-1'
float64 parse                    : 0.51953125
float32                          : 0.51953125   dyadic (f64==f32): True
  textbook f64 log(p/(1-p))      = 0.07816477119922638  bits=1033901274
  textbook all-f32               = 0.07816480845212936  bits=1033901279
  XGB f32 (1/p - 1), f64 log     = 0.0781647190451622  bits=1033901267

  textbook f64 log(p/(1-p))      ULP gap vs XGB formula = +7  abs diff = 5.21540641784668e-08
  textbook all-f32               ULP gap vs XGB formula = +12  abs diff = 8.940696716308594e-08
  XGB f32 (1/p - 1), f64 log     ULP gap vs XGB formula = +0  abs diff = 0.0

The two values the prior probe printed, for comparison:
  logit(base_score) float64  : 0.07816477284933626
  logit computed in float32  : 0.07816477119922638
  np.float32(0.07816477284933626) bits = 1033901274
  np.float32(0.07816477119922638) bits = 1033901274
  XGB formula                     bits = 1033901267
```

Refitting on that exact `base_score` and sweeping intercepts around the correct one shows
the partial-match signature is *diagnostic of an intercept off by a few ULP*, not of an
accumulation defect:

```
=== a one-ULP intercept error is exactly what a ~58% bit-exact rate looks like ===
  refit with base_score=0.51953125 -> stored '[5.1953125E-1]'
  V1 with intercept logit XGB (f32 1/p-1, f64 log)     (0.0781647190451622) -> bit-exact 5000/5000  max abs err 0.0
  V1 with intercept logit textbook f64 log(p/(1-p))    (0.07816477119922638) -> bit-exact 1881/5000  max abs err 2.384185791015625e-07
  V1 with intercept logit textbook all-f32             (0.07816480845212936) -> bit-exact 337/5000  max abs err 2.384185791015625e-07
  V1 with intercept XGB formula +1 ULP                 (0.0781647264957428) -> bit-exact 4412/5000  max abs err 1.1920928955078125e-07
  V1 with intercept XGB formula -1 ULP                 (0.0781647115945816) -> bit-exact 4698/5000  max abs err 1.1920928955078125e-07
```

A **one-ULP** intercept error yields `4412/5000` and `4698/5000`. A seven-ULP error yields
`1881/5000`. The earlier probe's `2878/5000` sits squarely in that family. The mechanism is
that a float32 running sum absorbs a small intercept error on most rows — the addition
rounds it away — and preserves it on the rest, depending on where each row's partial sums
happen to fall relative to the float32 grid.

**INFERRED**, not separately measured: that absorb-or-preserve mechanism. What is measured
is the monotone relationship between intercept ULP error and bit-exact count.

---

## 2. Every accumulation variant, with counts

Eleven variants, run against 12 model configurations spanning three objectives, 8 / 100 /
300 trees, and explicit `base_score` values chosen to exercise the intercept. `N = 5000` in
every row. Full output in `a02_out.txt`; the intercept is the correct per-objective one
except where the table says otherwise.

### Variant definitions

| | Variant |
|---|---|
| **V1** | float32 running sum, float32 leaves, serialized tree order, **intercept first** |
| V2 | float32 running sum, float32 leaves, **intercept last** |
| V3 | float32 running sum, **reverse** tree order, intercept first |
| V4 | float64 running sum of float32 leaves, intercept first, narrow at end |
| V5 | float64 running sum of float32 leaves, intercept last, narrow at end |
| V6 | float64 throughout — float64-parsed leaves, float64 intercept — narrow at end |
| V7 | float64 throughout, **no narrowing** (compared against the reference widened to float64) |
| V8 | numpy float32 pairwise sum of leaves, intercept last |
| V9 | numpy float32 pairwise sum with the intercept prepended as the first term |
| V10 | float32 blocked — float32 sums within blocks of 8 trees, then accumulated |
| V11 | float32 running sum, **leaves left as the float64 parse** (no leaf narrowing), intercept first |

**Margin of victory.** V1 is `5000/5000` in all 12 configurations. The best any other variant
ever achieved, in any configuration, with the correct intercept, is **`3706/5000`** (V11,
`binary:logistic`, 8 trees) — a loss of 1294 rows. The worst is `199/5000` (V2,
`binary:logistic`, `base_score = 0.987654`) — a loss of 4801. The single exception is
`survival:cox` with the estimated default `base_score`, where the intercept is exactly `0.0`
and V2 and V10 tie with V1; that degeneracy is analysed under "The Cox trap" below and does
not hold once `base_score` is anything else.

### `reg:squarederror`

```
$ uv run python a02_variants.py
### reg:squarederror  rounds=8 base_score_arg=None raw=[6.0321754E-1]   leaf-exact rows 5000/5000   ref margin range [np.float32(-3.82224), np.float32(6.073764)]
    V1 f32 running sum, f32 leaves, intercept FIRST                           5000 / 5000   max abs err 0.0  <== ALL
    V2 f32 running sum, f32 leaves, intercept LAST                            2120 / 5000   max abs err 7.152557373046875e-07
    V3 f32 running sum, REVERSE tree order, intercept FIRST                   2164 / 5000   max abs err 7.152557373046875e-07
    V4 f64 running sum of f32 leaves, intercept FIRST, narrow at end          2457 / 5000   max abs err 4.76837158203125e-07
    V5 f64 running sum of f32 leaves, intercept LAST, narrow at end           2457 / 5000   max abs err 4.76837158203125e-07
    V6 f64 throughout (f64-parsed leaves, f64 intercept), narrow at end       2347 / 5000   max abs err 4.76837158203125e-07
    V7 f64 throughout, NO narrowing                                              0 / 5000   max abs err 5.556783753135619e-07
    V8 numpy f32 pairwise sum of leaves, intercept LAST                       2067 / 5000   max abs err 4.76837158203125e-07
    V9 numpy f32 pairwise sum incl. intercept as first term                   2864 / 5000   max abs err 4.76837158203125e-07
    V10 f32 blocked (blocks of 8 trees), intercept FIRST                      2120 / 5000   max abs err 7.152557373046875e-07
    V11 f32 running sum, f64-parsed leaves (no leaf narrowing), intercept FIRST  3696 / 5000   max abs err 9.5367431640625e-07

### reg:squarederror  rounds=100 base_score_arg=None raw=[6.0321754E-1]   leaf-exact rows 5000/5000   ref margin range [np.float32(-7.2442136), np.float32(9.623854)]
    V1 f32 running sum, f32 leaves, intercept FIRST                           5000 / 5000   max abs err 0.0  <== ALL
    V2 f32 running sum, f32 leaves, intercept LAST                             925 / 5000   max abs err 3.337860107421875e-06
    V3 f32 running sum, REVERSE tree order, intercept FIRST                    599 / 5000   max abs err 3.337860107421875e-06
    V4 f64 running sum of f32 leaves, intercept FIRST, narrow at end           642 / 5000   max abs err 3.814697265625e-06
    V5 f64 running sum of f32 leaves, intercept LAST, narrow at end            642 / 5000   max abs err 3.814697265625e-06
    V6 f64 throughout (f64-parsed leaves, f64 intercept), narrow at end        637 / 5000   max abs err 3.814697265625e-06
    V7 f64 throughout, NO narrowing                                              0 / 5000   max abs err 3.433812407749315e-06
    V8 numpy f32 pairwise sum of leaves, intercept LAST                        604 / 5000   max abs err 3.814697265625e-06
    V9 numpy f32 pairwise sum incl. intercept as first term                    624 / 5000   max abs err 3.337860107421875e-06
    V10 f32 blocked (blocks of 8 trees), intercept FIRST                       591 / 5000   max abs err 4.291534423828125e-06
    V11 f32 running sum, f64-parsed leaves (no leaf narrowing), intercept FIRST  3018 / 5000   max abs err 9.5367431640625e-07

### reg:squarederror  rounds=300 base_score_arg=None raw=[6.0321754E-1]   leaf-exact rows 5000/5000   ref margin range [np.float32(-6.852784), np.float32(9.314619)]
    V1 f32 running sum, f32 leaves, intercept FIRST                           5000 / 5000   max abs err 0.0  <== ALL
    V2 f32 running sum, f32 leaves, intercept LAST                             686 / 5000   max abs err 6.67572021484375e-06
    V3 f32 running sum, REVERSE tree order, intercept FIRST                    331 / 5000   max abs err 5.7220458984375e-06
    V4 f64 running sum of f32 leaves, intercept FIRST, narrow at end           318 / 5000   max abs err 5.7220458984375e-06
    V5 f64 running sum of f32 leaves, intercept LAST, narrow at end            318 / 5000   max abs err 5.7220458984375e-06
    V6 f64 throughout (f64-parsed leaves, f64 intercept), narrow at end        315 / 5000   max abs err 5.7220458984375e-06
    V7 f64 throughout, NO narrowing                                              0 / 5000   max abs err 5.790178880360486e-06
    V8 numpy f32 pairwise sum of leaves, intercept LAST                        348 / 5000   max abs err 5.245208740234375e-06
    V9 numpy f32 pairwise sum incl. intercept as first term                    343 / 5000   max abs err 5.7220458984375e-06
    V10 f32 blocked (blocks of 8 trees), intercept FIRST                       318 / 5000   max abs err 6.67572021484375e-06
    V11 f32 running sum, f64-parsed leaves (no leaf narrowing), intercept FIRST  2765 / 5000   max abs err 9.5367431640625e-07
```

### `binary:logistic`

With the corrected intercept, at 8, 100 and 300 trees, and at `base_score = 0.987654` —
one of the two values `probes/base_score.md` found breaches the `1e-6` gate under the
textbook formula:

```
### binary:logistic  rounds=8 base_score_arg=None raw=[5.8E-1]   leaf-exact rows 5000/5000   ref margin range [np.float32(-2.1587362), np.float32(2.969931)]
--- intercept: logit XGB (f32 1/p-1, f64 log) = 0.322773277759552
    V1 f32 running sum, f32 leaves, intercept FIRST                           5000 / 5000   max abs err 0.0  <== ALL
    V2 f32 running sum, f32 leaves, intercept LAST                            1569 / 5000   max abs err 7.152557373046875e-07
    V3 f32 running sum, REVERSE tree order, intercept FIRST                   2365 / 5000   max abs err 4.76837158203125e-07
    V4 f64 running sum of f32 leaves, intercept FIRST, narrow at end          2541 / 5000   max abs err 2.384185791015625e-07
    V5 f64 running sum of f32 leaves, intercept LAST, narrow at end           2541 / 5000   max abs err 2.384185791015625e-07
    V6 f64 throughout (f64-parsed leaves, f64 intercept), narrow at end       2451 / 5000   max abs err 2.384185791015625e-07
    V7 f64 throughout, NO narrowing                                              0 / 5000   max abs err 3.337983702778047e-07
    V8 numpy f32 pairwise sum of leaves, intercept LAST                       2258 / 5000   max abs err 3.5762786865234375e-07
    V9 numpy f32 pairwise sum incl. intercept as first term                   2599 / 5000   max abs err 4.76837158203125e-07
    V10 f32 blocked (blocks of 8 trees), intercept FIRST                      1569 / 5000   max abs err 7.152557373046875e-07
    V11 f32 running sum, f64-parsed leaves (no leaf narrowing), intercept FIRST  3706 / 5000   max abs err 4.76837158203125e-07

### binary:logistic  rounds=300 base_score_arg=None raw=[5.8E-1]   leaf-exact rows 5000/5000   ref margin range [np.float32(-9.370341), np.float32(13.128214)]
--- intercept: logit XGB (f32 1/p-1, f64 log) = 0.322773277759552
    V1 f32 running sum, f32 leaves, intercept FIRST                           5000 / 5000   max abs err 0.0  <== ALL
    V2 f32 running sum, f32 leaves, intercept LAST                             499 / 5000   max abs err 5.7220458984375e-06
    V3 f32 running sum, REVERSE tree order, intercept FIRST                    417 / 5000   max abs err 7.152557373046875e-06
    V4 f64 running sum of f32 leaves, intercept FIRST, narrow at end           439 / 5000   max abs err 7.62939453125e-06
    V5 f64 running sum of f32 leaves, intercept LAST, narrow at end            439 / 5000   max abs err 7.62939453125e-06
    V6 f64 throughout (f64-parsed leaves, f64 intercept), narrow at end        446 / 5000   max abs err 7.62939453125e-06
    V7 f64 throughout, NO narrowing                                              0 / 5000   max abs err 7.645498799391248e-06
    V8 numpy f32 pairwise sum of leaves, intercept LAST                        435 / 5000   max abs err 7.62939453125e-06
    V9 numpy f32 pairwise sum incl. intercept as first term                    409 / 5000   max abs err 7.62939453125e-06
    V10 f32 blocked (blocks of 8 trees), intercept FIRST                       452 / 5000   max abs err 7.62939453125e-06
    V11 f32 running sum, f64-parsed leaves (no leaf narrowing), intercept FIRST  2396 / 5000   max abs err 1.9073486328125e-06

### binary:logistic  rounds=100 base_score_arg=0.987654 raw=[9.87654E-1]   leaf-exact rows 5000/5000   ref margin range [np.float32(-182.42458), np.float32(295.13684)]
--- intercept: logit XGB (f32 1/p-1, f64 log) = 4.381994247436523
    V1 f32 running sum, f32 leaves, intercept FIRST                           5000 / 5000   max abs err 0.0  <== ALL
    V2 f32 running sum, f32 leaves, intercept LAST                             199 / 5000   max abs err 5.340576171875e-05
    V3 f32 running sum, REVERSE tree order, intercept FIRST                    245 / 5000   max abs err 7.62939453125e-05
    V4 f64 running sum of f32 leaves, intercept FIRST, narrow at end           611 / 5000   max abs err 7.62939453125e-05
    V5 f64 running sum of f32 leaves, intercept LAST, narrow at end            611 / 5000   max abs err 7.62939453125e-05
    V6 f64 throughout (f64-parsed leaves, f64 intercept), narrow at end        272 / 5000   max abs err 7.62939453125e-05
    V7 f64 throughout, NO narrowing                                              0 / 5000   max abs err 7.592820482216212e-05
    V8 numpy f32 pairwise sum of leaves, intercept LAST                        225 / 5000   max abs err 0.000213623046875
    V9 numpy f32 pairwise sum incl. intercept as first term                    304 / 5000   max abs err 0.0002288818359375
    V10 f32 blocked (blocks of 8 trees), intercept FIRST                       269 / 5000   max abs err 8.487701416015625e-05
    V11 f32 running sum, f64-parsed leaves (no leaf narrowing), intercept FIRST   990 / 5000   max abs err 0.0001068115234375

--- intercept: logit textbook f64 log(p/(1-p)) = 4.381998062133789
    V1 f32 running sum, f32 leaves, intercept FIRST                           3143 / 5000   max abs err 3.0517578125e-05
    V2 f32 running sum, f32 leaves, intercept LAST                              31 / 5000   max abs err 5.340576171875e-05
    V3 f32 running sum, REVERSE tree order, intercept FIRST                     79 / 5000   max abs err 7.62939453125e-05
    V4 f64 running sum of f32 leaves, intercept FIRST, narrow at end            72 / 5000   max abs err 7.62939453125e-05
    V5 f64 running sum of f32 leaves, intercept LAST, narrow at end             72 / 5000   max abs err 7.62939453125e-05
    V6 f64 throughout (f64-parsed leaves, f64 intercept), narrow at end         66 / 5000   max abs err 6.866455078125e-05
    V7 f64 throughout, NO narrowing                                              0 / 5000   max abs err 7.211350755653712e-05
    V8 numpy f32 pairwise sum of leaves, intercept LAST                         45 / 5000   max abs err 0.0002288818359375
    V9 numpy f32 pairwise sum incl. intercept as first term                     63 / 5000   max abs err 0.0002593994140625
    V10 f32 blocked (blocks of 8 trees), intercept FIRST                        73 / 5000   max abs err 9.1552734375e-05
    V11 f32 running sum, f64-parsed leaves (no leaf narrowing), intercept FIRST    70 / 5000   max abs err 0.0001068115234375
```

That last block is the textbook logit under load: `3143/5000` and a max abs error of
`3.0517578125e-05` — **30× the `1e-6` margin gate.** With the corrected intercept, the same
model on the same rows is `5000/5000` at `0.0`. This is the strongest single argument in
this probe for §5 of `probes/base_score.md`.

### `survival:cox`

```
### survival:cox  rounds=8 base_score_arg=None raw=[1E0]   leaf-exact rows 5000/5000   ref margin range [np.float32(-1.4403183), np.float32(10.961314)]
--- intercept: ln(f32 base_score) = 0.0
    V1 f32 running sum, f32 leaves, intercept FIRST                           5000 / 5000   max abs err 0.0  <== ALL
    V2 f32 running sum, f32 leaves, intercept LAST                            5000 / 5000   max abs err 0.0  <== ALL
    V3 f32 running sum, REVERSE tree order, intercept FIRST                   2327 / 5000   max abs err 1.430511474609375e-06
    V4 f64 running sum of f32 leaves, intercept FIRST, narrow at end          2089 / 5000   max abs err 1.430511474609375e-06
    V5 f64 running sum of f32 leaves, intercept LAST, narrow at end           2089 / 5000   max abs err 1.430511474609375e-06
    V6 f64 throughout (f64-parsed leaves, f64 intercept), narrow at end       1899 / 5000   max abs err 1.430511474609375e-06
    V7 f64 throughout, NO narrowing                                              0 / 5000   max abs err 1.20994751018344e-06
    V8 numpy f32 pairwise sum of leaves, intercept LAST                       2242 / 5000   max abs err 1.430511474609375e-06
    V9 numpy f32 pairwise sum incl. intercept as first term                   2988 / 5000   max abs err 9.5367431640625e-07
    V10 f32 blocked (blocks of 8 trees), intercept FIRST                      5000 / 5000   max abs err 0.0  <== ALL
    V11 f32 running sum, f64-parsed leaves (no leaf narrowing), intercept FIRST  3103 / 5000   max abs err 9.5367431640625e-07

### survival:cox  rounds=300 base_score_arg=None raw=[1E0]   leaf-exact rows 5000/5000   ref margin range [np.float32(0.94383675), np.float32(31.083128)]
--- intercept: ln(f32 base_score) = 0.0
    V1 f32 running sum, f32 leaves, intercept FIRST                           5000 / 5000   max abs err 0.0  <== ALL
    V2 f32 running sum, f32 leaves, intercept LAST                            5000 / 5000   max abs err 0.0  <== ALL
    V3 f32 running sum, REVERSE tree order, intercept FIRST                    468 / 5000   max abs err 1.71661376953125e-05
    V4 f64 running sum of f32 leaves, intercept FIRST, narrow at end           822 / 5000   max abs err 1.71661376953125e-05
    V5 f64 running sum of f32 leaves, intercept LAST, narrow at end            822 / 5000   max abs err 1.71661376953125e-05
    V6 f64 throughout (f64-parsed leaves, f64 intercept), narrow at end        837 / 5000   max abs err 1.71661376953125e-05
    V7 f64 throughout, NO narrowing                                              0 / 5000   max abs err 1.801517661093044e-05
    V8 numpy f32 pairwise sum of leaves, intercept LAST                        780 / 5000   max abs err 1.9073486328125e-05
    V9 numpy f32 pairwise sum incl. intercept as first term                    797 / 5000   max abs err 1.9073486328125e-05
    V10 f32 blocked (blocks of 8 trees), intercept FIRST                       826 / 5000   max abs err 2.09808349609375e-05
    V11 f32 running sum, f64-parsed leaves (no leaf narrowing), intercept FIRST  2623 / 5000   max abs err 3.814697265625e-06

### survival:cox  rounds=100 base_score_arg=0.7 raw=[7E-1]   leaf-exact rows 5000/5000   ref margin range [np.float32(-6.104598), np.float32(20.09915)]
--- intercept: ln(f32 base_score) = -0.3566749691963196
    V1 f32 running sum, f32 leaves, intercept FIRST                           5000 / 5000   max abs err 0.0  <== ALL
    V2 f32 running sum, f32 leaves, intercept LAST                             935 / 5000   max abs err 3.814697265625e-06
    V3 f32 running sum, REVERSE tree order, intercept FIRST                    760 / 5000   max abs err 6.67572021484375e-06
    V4 f64 running sum of f32 leaves, intercept FIRST, narrow at end           849 / 5000   max abs err 6.67572021484375e-06
    V5 f64 running sum of f32 leaves, intercept LAST, narrow at end            849 / 5000   max abs err 6.67572021484375e-06
    V6 f64 throughout (f64-parsed leaves, f64 intercept), narrow at end        837 / 5000   max abs err 6.67572021484375e-06
    V7 f64 throughout, NO narrowing                                              0 / 5000   max abs err 6.602641258623976e-06
    V8 numpy f32 pairwise sum of leaves, intercept LAST                        781 / 5000   max abs err 8.58306884765625e-06
    V9 numpy f32 pairwise sum incl. intercept as first term                    931 / 5000   max abs err 7.62939453125e-06
    V10 f32 blocked (blocks of 8 trees), intercept FIRST                       778 / 5000   max abs err 8.58306884765625e-06
    V11 f32 running sum, f64-parsed leaves (no leaf narrowing), intercept FIRST  2705 / 5000   max abs err 1.9073486328125e-06

### survival:cox  rounds=100 base_score_arg=3.1415927 raw=[3.1415927E0]   leaf-exact rows 5000/5000   ref margin range [np.float32(-4.8842463), np.float32(21.569485)]
--- intercept: ln(f32 base_score) = 1.1447299718856812
    V1 f32 running sum, f32 leaves, intercept FIRST                           5000 / 5000   max abs err 0.0  <== ALL
    V2 f32 running sum, f32 leaves, intercept LAST                             676 / 5000   max abs err 6.67572021484375e-06
    V3 f32 running sum, REVERSE tree order, intercept FIRST                    823 / 5000   max abs err 7.62939453125e-06
    V4 f64 running sum of f32 leaves, intercept FIRST, narrow at end           870 / 5000   max abs err 7.62939453125e-06
    V5 f64 running sum of f32 leaves, intercept LAST, narrow at end            870 / 5000   max abs err 7.62939453125e-06
    V6 f64 throughout (f64-parsed leaves, f64 intercept), narrow at end        884 / 5000   max abs err 7.62939453125e-06
    V7 f64 throughout, NO narrowing                                              0 / 5000   max abs err 7.107216084989432e-06
    V8 numpy f32 pairwise sum of leaves, intercept LAST                        747 / 5000   max abs err 7.62939453125e-06
    V9 numpy f32 pairwise sum incl. intercept as first term                    821 / 5000   max abs err 7.62939453125e-06
    V10 f32 blocked (blocks of 8 trees), intercept FIRST                       787 / 5000   max abs err 7.62939453125e-06
    V11 f32 running sum, f64-parsed leaves (no leaf narrowing), intercept FIRST  2292 / 5000   max abs err 1.9073486328125e-06
```

### The Cox trap

**Cox with the default `base_score` cannot distinguish intercept-first from
intercept-last.** With `base_score = '[1E0]'` the intercept is exactly `0.0`, so V1, V2 and
V10 are all `5000/5000` — `0.0 + x == x + 0.0` and blocking is invisible. Set `base_score`
to anything else and V2 collapses to `676`–`935 / 5000`. A Cox fixture that only ever uses
the estimated default is decorative for this invariant: it passes an implementation that
adds the intercept in the wrong place.

This is the same concern `probes/base_score.md` raised about `ln(1.0) = 0` hiding a broken
`ln`, arriving independently from the accumulation side.

### Why V7 is `0 / 5000` and that is not a bug in the test

V7 keeps the result in float64 and is compared against XGBoost's float32 margin widened to
float64. It scores `0/5000` because a float64 sum essentially never lands exactly on a
float32 grid point. It is included to make the point that "more precision" is not "more
correct" here: XGBoost's margin is a float32, and any variant that does not round to the
float32 grid *at every step* is arithmetically a different function.

---

## 3. Do leaf values need float32 narrowing when read?

**Yes.** The isolation is clean: identical accumulation, identical intercept, identical tree
order; only the way the leaf value is *read* changes. Narrowed → `float32(json_value)`.
Not narrowed → the float64 that `json.load` produced, added into a float32 accumulator whose
result is then narrowed.

First, every leaf token in these models is a value the float64 parse gets wrong — the same
property `probes/float32_thresholds.md` measured for thresholds:

```
$ uv run python a03_leaves_and_zero.py
=== A. do leaf tokens differ between float64 parse and float32 narrow? ===
  reg:squarederror   trees= 100 leaf tokens=  1400  float64(tok) != float32(tok) on 1400/1400  worst |f64-f32| = 4.162597666557133e-08
                     leaf tokens equal to zero: -0.0 x0  +0.0 x0
  binary:logistic    trees= 100 leaf tokens=  1243  float64(tok) != float32(tok) on 1243/1243  worst |f64-f32| = 2.6099395755707633e-08
                     leaf tokens equal to zero: -0.0 x0  +0.0 x0
  survival:cox       trees= 100 leaf tokens=  1511  float64(tok) != float32(tok) on 1511/1511  worst |f64-f32| = 4.889678950625864e-08
                     leaf tokens equal to zero: -0.0 x0  +0.0 x0
```

`1400/1400`, `1243/1243`, `1511/1511`. Not a rare case — every leaf.

```
=== B. leaf narrowing on read: bit-exact counts, same accumulation ===
    (f32 running sum, intercept FIRST, in every row of this table;
     only how the leaf value is READ changes)
  reg:squarederror   rounds=   8 bs=None        leaf NARROWED to f32: 5000/5000   leaf left as f64 parse: 3696/5000 (max err 9.5367431640625e-07)
  reg:squarederror   rounds= 100 bs=None        leaf NARROWED to f32: 5000/5000   leaf left as f64 parse: 3018/5000 (max err 9.5367431640625e-07)
  binary:logistic    rounds=   8 bs=None        leaf NARROWED to f32: 5000/5000   leaf left as f64 parse: 3706/5000 (max err 4.76837158203125e-07)
  binary:logistic    rounds= 100 bs=None        leaf NARROWED to f32: 5000/5000   leaf left as f64 parse: 2563/5000 (max err 9.5367431640625e-07)
  binary:logistic    rounds= 100 bs=0.987654    leaf NARROWED to f32: 5000/5000   leaf left as f64 parse: 990/5000 (max err 0.0001068115234375)
  survival:cox       rounds= 100 bs=None        leaf NARROWED to f32: 5000/5000   leaf left as f64 parse: 2474/5000 (max err 1.9073486328125e-06)
  survival:cox       rounds= 100 bs=0.7         leaf NARROWED to f32: 5000/5000   leaf left as f64 parse: 2705/5000 (max err 1.9073486328125e-06)
```

`5000/5000` versus `990`–`3706 / 5000`, and the un-narrowed variant breaches the `1e-6`
margin gate on the `base_score = 0.987654` model at `0.0001068115234375`. The narrowing
must happen **at parse time**, on the whole `split_conditions` array, before any node is
classified as internal or leaf — not at the point where a leaf is added.

This extends `probes/float32_thresholds.md` §8(b) from thresholds to leaf values. Both live
in `split_conditions`, so one narrowing at parse time covers both — which is the correct
implementation anyway, since a reader does not know a node's role until it consults
`left_children`.

---

## 4. Is XGBoost's own margin bit-stable? — the well-definedness question

If XGBoost's margin for a fixed row depended on thread count, batch size, or which predict
entry point was called, bit-exact parity against XGBoost would not be a well-defined target
and this project could not promise it. **It does not depend on any of them.**

Full-batch comparisons, all bit-identical to the `nthread=1` reference. Six models: three
objectives × 8 and 300 trees. Full output in `a05_out.txt`.

```
$ uv run python a05_stability.py
=== reg:squarederror rounds=8 trees=8 ===
  reference: DMatrix, nthread=1, batch=5000   margin range [np.float32(-3.82224), np.float32(6.073764)]
    DMatrix nthread=1  batch=5000    bit-identical 5000/5000  max abs diff 0.0
    DMatrix nthread=2  batch=5000    bit-identical 5000/5000  max abs diff 0.0
    DMatrix nthread=4  batch=5000    bit-identical 5000/5000  max abs diff 0.0
    DMatrix nthread=8  batch=5000    bit-identical 5000/5000  max abs diff 0.0
    DMatrix nthread=0  batch=5000    bit-identical 5000/5000  max abs diff 0.0
    inplace nthread=1  batch=5000    bit-identical 5000/5000  max abs diff 0.0
    inplace nthread=4  batch=5000    bit-identical 5000/5000  max abs diff 0.0
    per-row batch/thread sweep on rows [0, 1, 7, 924, 2434, 4999]:
      row     0: 40 configurations -> 1 distinct float32 bit pattern(s)  ALL IDENTICAL
      row     1: 40 configurations -> 1 distinct float32 bit pattern(s)  ALL IDENTICAL
      row     7: 40 configurations -> 1 distinct float32 bit pattern(s)  ALL IDENTICAL
      row   924: 40 configurations -> 1 distinct float32 bit pattern(s)  ALL IDENTICAL
      row  2434: 40 configurations -> 1 distinct float32 bit pattern(s)  ALL IDENTICAL
      row  4999: 40 configurations -> 1 distinct float32 bit pattern(s)  ALL IDENTICAL
    rows whose margin was NOT bit-stable: 0/6
```

The per-row sweep is the sharp instrument: for each probe row, the margin is read back under
**40 configurations** — `nthread ∈ {1, 2, 4, 8}` × batch size `∈ {1, 2, 16, 1000, 5000}` ×
`{DMatrix, inplace_predict}`, with the probe row always at index 0 of the batch. One distinct
float32 bit pattern in all 40, on every probe row, in every model:

```
### summary across all six models (each block reported identically)
    rows whose margin was NOT bit-stable: 0/6      x 6 models
```

Totals: **6 models × 6 rows × 40 configurations = 1440 per-row observations, 0 divergences.**
Plus 6 × 7 = 42 full-batch comparisons, all `5000/5000` bit-identical at max abs diff `0.0`.

`nthread=0` is included because it means "use the default thread count", i.e. the
multi-threaded path a caller gets without asking.

### `inplace_predict` vs `DMatrix`, including the float64 container

`DMatrix` narrows a float64 array to float32 at construction; `inplace_predict` does not.
That is the version of the comparison that could actually diverge, so it was run separately
on the same float64 input:

```
$ uv run python a06_api_and_export.py
=== A. float64 input container: DMatrix vs inplace_predict, 5000 rows ===
  reg:squarederror   trees=300
    DMatrix(f64)   bit-identical to DMatrix(f32) on 5000/5000   dtype=float32
    inplace(f64)   bit-identical to DMatrix(f32) on 5000/5000   dtype=float32
    DMatrix(f32)   bit-identical to DMatrix(f32) on 5000/5000   dtype=float32
    inplace(f32)   bit-identical to DMatrix(f32) on 5000/5000   dtype=float32
  binary:logistic    trees=300
    DMatrix(f64)   bit-identical to DMatrix(f32) on 5000/5000   dtype=float32
    inplace(f64)   bit-identical to DMatrix(f32) on 5000/5000   dtype=float32
    DMatrix(f32)   bit-identical to DMatrix(f32) on 5000/5000   dtype=float32
    inplace(f32)   bit-identical to DMatrix(f32) on 5000/5000   dtype=float32
  survival:cox       trees=300
    DMatrix(f64)   bit-identical to DMatrix(f32) on 5000/5000   dtype=float32
    inplace(f64)   bit-identical to DMatrix(f32) on 5000/5000   dtype=float32
    DMatrix(f32)   bit-identical to DMatrix(f32) on 5000/5000   dtype=float32
    inplace(f32)   bit-identical to DMatrix(f32) on 5000/5000   dtype=float32
```

All four paths agree bit-for-bit on all 5000 rows for all three objectives. Note that the
input here is a float32 matrix widened to float64, so it contains no value that needs
narrowing; `probes/float32_thresholds.md` §5 covers the case where the float64 input does
carry sub-float32 detail, and reaches the same conclusion by a different route.

**Scope limit, stated so it is not mistaken for a general claim:** measured on darwin/arm64,
CPU only, `gbtree`, single process, 1–8 threads. GPU (`device="cuda"`) was not measured, and
neither was a different CPU architecture. **INFERRED** from the pattern — that XGBoost's
gbtree CPU predictor parallelises across rows rather than across trees, so no row's
accumulation order can depend on thread count — but the inference is not needed for any
conclusion here; the measurement stands on its own for this platform.

---

## 5. Negative zero

### Policy applied

**Strict bit-pattern equality. `-0.0` and `+0.0` count as a MISMATCH.** Every count in this
report is `np.float32(x).view(np.uint32)` equality in Python and `Uint32Array` overlay
equality in JavaScript. `Object.is` semantics, not `==`.

Both readings are defensible and here is the case for each:

- **Strict (used).** The gate is stated as exactly `0.0` cross-language parity at the bit
  level, and signed zero is a real bit-level difference that propagates: `1/(-0.0)` and
  `1/(+0.0)` differ in sign, and a downstream link function or a serializer can turn the
  sign into a visible difference. `JSON.stringify(-0)` is `"0"` while
  `json.dumps(-0.0)` is `"-0.0"`, so a signed zero that survives into an artifact is also a
  byte-identical-export (D008) hazard.
- **Lenient.** `-0.0 == +0.0` is true in IEEE 754 and the two are numerically
  indistinguishable in the margin's own units. On this reading a sign-of-zero difference is
  not a wrong number and should not fail a parity gate.

The strict reading was chosen partly because it is the one that **catches a real bug** here:
see the 0-tree case below, where the textbook logit gets the sign of zero wrong and the
strict comparison is the only one that notices.

### What was actually observed

**No row in 60000 produced a zero margin at all**, so the question never arose naturally:

```
$ uv run python a03_leaves_and_zero.py
=== C. negative zero: does the accumulated margin ever differ from
    XGBoost only in the sign of zero? ===
  reg:squarederror   rounds=   8 bs=None        zero margins: mine=0 ref=0 (ref -0.0: 0)  sign-of-zero-only disagreements=0
  reg:squarederror   rounds= 100 bs=None        zero margins: mine=0 ref=0 (ref -0.0: 0)  sign-of-zero-only disagreements=0
  reg:squarederror   rounds= 300 bs=None        zero margins: mine=0 ref=0 (ref -0.0: 0)  sign-of-zero-only disagreements=0
  binary:logistic    rounds=   8 bs=None        zero margins: mine=0 ref=0 (ref -0.0: 0)  sign-of-zero-only disagreements=0
  binary:logistic    rounds= 100 bs=None        zero margins: mine=0 ref=0 (ref -0.0: 0)  sign-of-zero-only disagreements=0
  binary:logistic    rounds= 300 bs=None        zero margins: mine=0 ref=0 (ref -0.0: 0)  sign-of-zero-only disagreements=0
  binary:logistic    rounds= 100 bs=0.987654    zero margins: mine=0 ref=0 (ref -0.0: 0)  sign-of-zero-only disagreements=0
  survival:cox       rounds=   8 bs=None        zero margins: mine=0 ref=0 (ref -0.0: 0)  sign-of-zero-only disagreements=0
  survival:cox       rounds= 100 bs=None        zero margins: mine=0 ref=0 (ref -0.0: 0)  sign-of-zero-only disagreements=0
  survival:cox       rounds= 300 bs=None        zero margins: mine=0 ref=0 (ref -0.0: 0)  sign-of-zero-only disagreements=0
  survival:cox       rounds= 100 bs=0.7         zero margins: mine=0 ref=0 (ref -0.0: 0)  sign-of-zero-only disagreements=0
  survival:cox       rounds= 100 bs=3.1415927   zero margins: mine=0 ref=0 (ref -0.0: 0)  sign-of-zero-only disagreements=0
  TOTAL rows examined 60000;  rows differing ONLY in sign of zero: 0
```

**Zero rows differed only in the sign of zero.** The two readings therefore give the same
counts everywhere in this report; the policy choice changed nothing about the verdict.

### Why, mechanically — and where it *is* decidable

In a model with at least one tree, the sign of the intercept's zero is destroyed by the
first addition of a nonzero leaf. IEEE gives `(-0.0) + (+0.0) = +0.0` and
`x + (-x) = +0.0` for nonzero `x`; only `(-0.0) + (-0.0)` stays negative:

```
$ uv run python a04_signed_zero_forced.py
=== C. IEEE signed-zero arithmetic, numpy float32 vs the JS spec ===
  (-0.0) + (+0.0)      = 0.0  bits=0
  (-0.0) + (-0.0)      = -0.0  bits=2147483648
  (+0.0) + (-0.0)      = 0.0  bits=0
  1.5 + (-1.5)         = 0.0  bits=0
```

```
$ node a07_js_crosscheck.mjs
  signed zero, JS Math.fround: (-0)+(0) -> 0  (-0)+(-0) -> 2147483648  Object.is(-0,0) -> false  JSON.stringify(-0) -> 0
```

Python and JavaScript agree on signed-zero addition, so this cannot be a source of
cross-language divergence.

Consistently, a `binary:logistic` model with `base_score = 0.5` (intercept `-0.0`) scores
`5000/5000` whether the walk starts from `-0.0` or `+0.0`, because trees are present:

```
$ uv run python a03_leaves_and_zero.py
=== D. can the intercept itself be a negative zero, and does it survive? ===
  binary:logistic base_score=0.5 raw='[5E-1]' intercept=-0.0 bits=2147483648 is_negzero=True
  intercept = -0.0: bit-exact 5000/5000
  intercept = +0.0: bit-exact 5000/5000
```

**The one place the sign of zero decides a bit is the 0-tree model**, where the intercept
*is* the margin — and there the textbook logit is wrong:

```
$ uv run python a04_signed_zero_forced.py
=== A. 0-tree model, base_score pinned: the intercept IS the margin ===
  binary:logistic    base_score=0.5 raw='[5E-1]'
    XGB margin[0]  = np.float32(-0.0)  bits=2147483648  negzero=True
    logit XGB (f32 1/p-1, f64 log)     = -0.0  bits=2147483648  bit-match=True
    logit textbook f64 log(p/(1-p))    = 0.0  bits=0  bit-match=False
    logit textbook all-f32             = 0.0  bits=0  bit-match=False
  survival:cox       base_score=1.0 raw='[1E0]'
    XGB margin[0]  = np.float32(0.0)  bits=0  negzero=False
    ln(f32 base_score)                 = 0.0  bits=0  bit-match=True
    ln(f64 base_score)                 = 0.0  bits=0  bit-match=True
  reg:squarederror   base_score=0.0 raw='[0E0]'
    XGB margin[0]  = np.float32(0.0)  bits=0  negzero=False
    identity(f32 base_score)           = 0.0  bits=0  bit-match=True
```

This independently reproduces the signed-zero observation in `probes/base_score.md` §5(b),
and confirms it survives the accumulation path unchanged.

### Not reproduced: a `-0E0` leaf value

`probes/tree_structure.md` §7(a) observed a leaf value serialized as `-0E0`. This probe did
**not** reproduce one, in any of the 48 model configurations, despite deliberately pinning
`base_score` to the float32 label mean and pruning trees down to a leaf-only root with
large `gamma`:

```
$ uv run python a04_signed_zero_forced.py
=== B. leaf-only trees via large gamma: does a -0E0 leaf appear, and does
    intercept + (-0.0) reproduce XGBoost bit-for-bit? ===
  gamma=1e+06      base_score raw='[6.0321754E-1]' trees=3 (num_nodes,leaves)=[(31, 31), (31, 31), (31, 31)]
    split_conditions[tree0] = [-1.079999e-08, -0.45375812, 0.36651582, ...]
    signed -0.0 entries across all split_conditions: 0
    V1 bit-exact 5000/5000   distinct margins=1
```

The nearest thing produced was a leaf of `-1.079999e-08` on a fully-pruned leaf-only root —
near zero but not zero. V1 was `5000/5000` on that degenerate shape too, at
`gamma ∈ {0, 1e3, 1e6, 1e9}`. **Whether a `-0E0` leaf changes anything in the accumulation
is therefore untested here** and is stated as a gap, not as a null result. Both readings of
the earlier observation stay open: either `-0E0` leaves arise only under conditions this
probe did not hit, or they are common enough that a fixture must construct one deliberately.

---

## 6. The winning variant, stated implementably

Everything below is measured. Nothing is inferred.

```
INPUT
  cond[]        = split_conditions, NARROWED TO FLOAT32 AT PARSE TIME
                  (this array holds BOTH thresholds and leaf values)
  left[], right[], sidx[], dleft[]
  intercept     = a float32, the per-objective transform of base_score
                  (probes/base_score.md section 9)

MARGIN
  acc = float32(intercept)                      # INTERCEPT FIRST, before any tree

  for t in trees, IN SERIALIZED ORDER:          # trees[] array order
      node = 0
      while left[t][node] != -1:                # leaf iff left_children == -1
          v = feature_value(sidx[t][node])
          if v is NaN:
              node = dleft[t][node] ? left[t][node] : right[t][node]
          else:
              node = (cast32(v) < cast32(cond[t][node]))     # STRICT '<', BOTH sides
                     ? left[t][node] : right[t][node]
      acc = cast32(acc + cond[t][node])         # NARROW AFTER EVERY SINGLE ADD

  margin = acc                                  # already float32; do not touch further
```

- `cast32` is `np.float32(...)` in Python and `Math.fround(...)` in JavaScript.
- **The intercept is the initial value of the accumulator, not a final addend.** Measured:
  intercept-last scores `199`–`2120 / 5000` on every objective whose intercept is nonzero.
- **Tree order is the `trees[]` array order.** Measured: reversing it scores
  `245`–`2365 / 5000`.
- **Narrow after every add, not once at the end.** Measured: a float64 running sum narrowed
  at the end scores `318`–`2541 / 5000`.
- **Narrow leaf values on read.** Measured: §3, `990`–`3706 / 5000` without it.
- The intercept must be float32 before it enters the accumulator, and must be the exact
  per-objective transform from `probes/base_score.md`. Measured: §1 and §2.

There is no free parameter left. Any deviation from the above was measured and loses.

### Coverage of the `5000/5000` claim

V1 was `5000/5000` at max abs error `0.0` on all of:

| Dimension | Values measured |
|---|---|
| Objective | `reg:squarederror`, `binary:logistic`, `survival:cox` |
| Tree count | 0, 3, 8, 25, 50, 100, 200, 300, 500, 1000 |
| `eta` | 0.3, 1.0 |
| `base_score` | estimated default, and explicit `0.0`, `0.5`, `0.7`, `0.987654`, `1.0`, `3.1415927`, `0.51953125` |
| `tree_method` | `exact`, `hist` |
| `num_parallel_tree` | 1, 4, 8 |
| `gamma` (pruning, dead nodes, leaf-only root) | 0, `1e3`, `1e6`, `1e9` |
| Missing values | none; 15% NaN in training and prediction; 15% NaN at predict only |
| Adversarial rows | every feature exactly on a float32 threshold, or ±1 ULP from one |
| Margin magnitude | up to `295.13684` (binary) and `205.03795` (Cox) |

Selected pasted evidence for the harder shapes:

```
$ uv run python a08_stress.py
=== A. missing values (NaN) in both training and prediction input ===
  reg:squarederror rounds=100 15% NaN train+predict            trees= 100 V1 bit-exact 5000/5000  max abs err 0.0  <== ALL
  binary:logistic rounds=100 15% NaN train+predict             trees= 100 V1 bit-exact 5000/5000  max abs err 0.0  <== ALL
  survival:cox rounds=100 15% NaN train+predict                trees= 100 V1 bit-exact 5000/5000  max abs err 0.0  <== ALL

=== B. NaN only at predict time, model trained on dense data ===
  reg:squarederror rounds=100 dense train / 15% NaN predict     trees= 100 V1 bit-exact 5000/5000  max abs err 0.0  <== ALL
  binary:logistic rounds=100 dense train / 15% NaN predict      trees= 100 V1 bit-exact 5000/5000  max abs err 0.0  <== ALL
  survival:cox rounds=100 dense train / 15% NaN predict         trees= 100 V1 bit-exact 5000/5000  max abs err 0.0  <== ALL

=== C. adversarial rows: every feature value sits EXACTLY on a
    float32 threshold taken from the fitted model ===
  reg:squarederror   rounds=100 adversarial rows: leaf-exact 5000/5000  V1 bit-exact 5000/5000  max abs err 0.0  <== ALL
                     control: threshold NOT narrowed -> bit-exact 3331/5000  max abs err 1.5857419967651367
  binary:logistic    rounds=100 adversarial rows: leaf-exact 5000/5000  V1 bit-exact 5000/5000  max abs err 0.0  <== ALL
                     control: threshold NOT narrowed -> bit-exact 3205/5000  max abs err 3.2439963817596436
  survival:cox       rounds=100 adversarial rows: leaf-exact 5000/5000  V1 bit-exact 5000/5000  max abs err 0.0  <== ALL
                     control: threshold NOT narrowed -> bit-exact 3459/5000  max abs err 5.134965300559998

=== D. num_parallel_tree > 1 -- several trees per boosting round ===
  reg:squarederror   num_parallel_tree=4 rounds=25 trees=100 num_trees=100 iteration_indptr[:4]=[0, 4, 8, 12] tree_info uniq=[0] -> V1 bit-exact 5000/5000  <== ALL
  binary:logistic    num_parallel_tree=4 rounds=25 trees=100 num_trees=100 iteration_indptr[:4]=[0, 4, 8, 12] tree_info uniq=[0] -> V1 bit-exact 5000/5000  <== ALL
  survival:cox       num_parallel_tree=4 rounds=25 trees=100 num_trees=100 iteration_indptr[:4]=[0, 4, 8, 12] tree_info uniq=[0] -> V1 bit-exact 5000/5000  <== ALL
  reg:squarederror   num_parallel_tree=8 rounds=25 trees=200 num_trees=200 iteration_indptr[:4]=[0, 8, 16, 24] tree_info uniq=[0] -> V1 bit-exact 5000/5000  <== ALL
  binary:logistic    num_parallel_tree=8 rounds=25 trees=200 num_trees=200 iteration_indptr[:4]=[0, 8, 16, 24] tree_info uniq=[0] -> V1 bit-exact 5000/5000  <== ALL
  survival:cox       num_parallel_tree=8 rounds=25 trees=200 num_trees=200 iteration_indptr[:4]=[0, 8, 16, 24] tree_info uniq=[0] -> V1 bit-exact 5000/5000  <== ALL
```

The **control lines in section C are the deliberate red-test** CLAUDE.md requires: on rows
placed exactly on float32 thresholds, reverting the threshold narrowing drops the count to
`3205`–`3459 / 5000` with margin errors of `1.58` to `5.13`. Those rows are not decorative;
they turn red when the cast is removed. Note also that adversarial rows do **not** move V1
off `5000/5000` — leaf assignment and accumulation are independent, exactly as the earlier
probe found.

```
$ uv run python a10_parity.py
=== B. stress: 1000 trees, and eta=1.0 (large leaf magnitudes) ===
  reg:squarederror   rounds= 1000 eta=0.3  trees= 1000 margin range [np.float32(-6.799681), np.float32(9.25296)] -> V1 bit-exact 5000/5000 max abs err 0.0  <== ALL
  reg:squarederror   rounds=  500 eta=1.0  trees=  500 margin range [np.float32(-6.8034005), np.float32(9.263194)] -> V1 bit-exact 5000/5000 max abs err 0.0  <== ALL
  binary:logistic    rounds= 1000 eta=0.3  trees= 1000 margin range [np.float32(-14.728653), np.float32(18.581436)] -> V1 bit-exact 5000/5000 max abs err 0.0  <== ALL
  binary:logistic    rounds=  500 eta=1.0  trees=  500 margin range [np.float32(-17.97422), np.float32(26.635443)] -> V1 bit-exact 5000/5000 max abs err 0.0  <== ALL
  survival:cox       rounds= 1000 eta=0.3  trees= 1000 margin range [np.float32(51.777225), np.float32(88.39914)] -> V1 bit-exact 5000/5000 max abs err 0.0  <== ALL
  survival:cox       rounds=  500 eta=1.0  trees=  500 margin range [np.float32(-2.272003), np.float32(205.03795)] -> V1 bit-exact 5000/5000 max abs err 0.0  <== ALL
```

1000 trees, `eta = 1.0`, margins up to `205.03795`. Still `0.0`.

---

## 7. JavaScript cross-check — exactly `0.0` parity is reachable

A standalone JavaScript walk, `Math.fround` on both sides of every comparison and on every
partial sum, run against the same 5000 rows and the same XGBoost margins. Feature values and
XGBoost margins cross from Python as **float32 bit patterns**, so no decimal formatting can
soften the comparison.

```
$ node a07_js_crosscheck.mjs
node version: v20.19.0
rows: 5000 cols: 6
  reg:squarederror   rounds= 300 base_score_arg=null      intercept=0.6032175421714783 | JSON.parse->Math.fround vs np.float32 mismatches 0/8198 | V1 bit-exact vs XGBoost 5000/5000
  binary:logistic    rounds= 300 base_score_arg=null      intercept=0.322773277759552 | JSON.parse->Math.fround vs np.float32 mismatches 0/7026 | V1 bit-exact vs XGBoost 5000/5000
  binary:logistic    rounds= 100 base_score_arg=0.987654  intercept=4.381994247436523 | JSON.parse->Math.fround vs np.float32 mismatches 0/1660 | V1 bit-exact vs XGBoost 5000/5000
  survival:cox       rounds= 300 base_score_arg=null      intercept=0 | JSON.parse->Math.fround vs np.float32 mismatches 0/8842 | V1 bit-exact vs XGBoost 5000/5000
  survival:cox       rounds= 100 base_score_arg=0.7       intercept=-0.3566749691963196 | JSON.parse->Math.fround vs np.float32 mismatches 0/2894 | V1 bit-exact vs XGBoost 5000/5000
  TOTAL bit-exact 25000/25000
  CONTROL, binary:logistic rounds=300: bare JSON.parse thresholds and un-narrowed leaves -> bit-exact 2396/5000
```

Two things in that output:

- **`25000/25000` bit-exact against XGBoost**, across 28620 `split_conditions` tokens, with
  `0` disagreements between `Math.fround(JSON.parse(token))` and `np.float32(token)`.
- The **control** — a bare `JSON.parse` with no `Math.fround` on the threshold and no leaf
  narrowing — scores `2396/5000`. That is **the identical count Python's V11 produced for
  the same model** (§2, `binary:logistic rounds=300`, `V11 = 2396/5000`). The two languages
  agree even in how they are wrong, which is itself evidence the walks are the same program.

The intercept is passed to JavaScript as a float32 bit pattern rather than recomputed there.
That is deliberate: it isolates the **accumulation** cross-check from re-testing the logit
transform, which `probes/base_score.md` owns. It also means this probe produces **no
evidence about whether JavaScript can reproduce the float32 `1/p - 1` logit** — see the
decision below.

### Direct Python ↔ JavaScript parity

Both sides matching XGBoost implies they match each other, but the gate is stated as a
cross-language number, so it was measured as one:

```
$ node a09_js_dump.mjs && uv run python a10_parity.py
node v20.19.0 -> wrote js_margins.json, 5 cases
=== A. Python V1 vs JavaScript V1, directly ===
  reg:squarederror   rounds= 300 bs=None       py==js bits 5000/5000  max |py-js| = 0.0   py==xgboost 5000/5000
  binary:logistic    rounds= 300 bs=None       py==js bits 5000/5000  max |py-js| = 0.0   py==xgboost 5000/5000
  binary:logistic    rounds= 100 bs=0.987654   py==js bits 5000/5000  max |py-js| = 0.0   py==xgboost 5000/5000
  survival:cox       rounds= 300 bs=None       py==js bits 5000/5000  max |py-js| = 0.0   py==xgboost 5000/5000
  survival:cox       rounds= 100 bs=0.7        py==js bits 5000/5000  max |py-js| = 0.0   py==xgboost 5000/5000
  TOTAL py==js 25000/25000   CROSS-LANGUAGE PARITY = 0.0
```

**Cross-language parity: exactly `0.0`.** Not small. `0.0`.

---

## 8. Consequences for the fixture corpus

Stated as what the measurements imply about which fixtures can detect which defect, not as
design instructions.

| Defect | Fixture that detects it | Fixture that does NOT |
|---|---|---|
| Intercept added last instead of first | Any model with a nonzero intercept | **Cox with the default `base_score`** — intercept is exactly `0.0`, so V1 and V2 both pass |
| Wrong logit formulation | `binary:logistic` with `base_score` near 1 (`0.987654` gives max err `3.05e-05`, 30× the gate) | `base_score = 0.58` — the textbook and XGBoost formulas coincide there |
| Leaf value not narrowed on read | Any model; `base_score = 0.987654` breaches the gate at `1.07e-04` | — |
| Threshold not narrowed | Rows exactly on `float32(threshold)` (drops to `3205/5000`, margin error up to `5.13`) | Ordinary random rows — V1 and the broken walk agree |
| Float64 accumulation | Any model ≥ 8 trees; grows with tree count | A 3-tree model can coincide |
| Wrong tree order | Any model ≥ 8 trees with a nonzero intercept | — |
| Signed-zero intercept | A **0-tree** model, `binary:logistic`, `base_score = 0.5` | Any model with ≥ 1 tree — the first nonzero leaf erases the sign |

The Cox row and the `base_score = 0.58` row are the two traps: both produce a green board on
a broken implementation.

---

## 9. Decisions needed

```
DECISION NEEDED: Where does the float32 1/p-1 logit live -- Python only, or both languages?
Context:  This probe reached 25000/25000 in JavaScript by passing the intercept across as a
          float32 bit pattern rather than recomputing it there. So it produced NO evidence
          about whether Math.fround can reproduce the float32 (1/p - 1) expression bit-for-bit.
          probes/base_score.md already raised this as a format-design question and leaned
          toward storing the derived intercept. This probe strengthens the case with a number:
          the textbook logit at base_score=0.987654 breaches the 1e-6 gate by 30x
          (3.0517578125e-05) once 100 trees of accumulation sit on top of it -- worse than the
          6.198883056640625e-06 measured on the intercept alone, because a float32 running sum
          preserves rather than damps the error on most rows.
Options:  A) Store base_score in its native space; both predictors compute the logit. Needs a
             separate probe confirming Math.fround(1/p - 1) matches np.float32 across a sweep
             of p, including values near 0.5 and near 1.
          B) Store the derived float32 margin intercept; the transform exists once, in Python.
             Then the measurement in this probe IS the shipping path, already at 0.0.
Lean:     B, and note that this probe's JS cross-check is only valid evidence for the shipping
          path under B. Under A there is an untested step.
Blocks:   Artifact format design. Also whether a further probe is needed before the format is fixed.
```

```
DECISION NEEDED: Is a -0E0 leaf value reachable, and must a fixture construct one?
Context:  probes/tree_structure.md section 7(a) observed a leaf serialized as -0E0. This probe
          did not reproduce one across 44 model configurations, including base_score pinned to
          the float32 label mean and gamma up to 1e9 pruning trees to a leaf-only root (the
          closest was a leaf of -1.079999e-08). Signed zero is only observable in the
          accumulation when it is the FIRST term, since (-0.0) + x for nonzero x is unsigned;
          a -0.0 LEAF in a model with a nonzero intercept would be invisible, but a -0.0 leaf
          in a single-tree model with a -0.0 intercept would not.
Options:  A) Probe the specific conditions that produce a -0E0 leaf, then decide.
          B) Treat it as reachable-but-harmless: narrow on read, add in order, and accept that
             the sign only matters in a shape (zero intercept AND all-zero leaves) that has no
             predictive content.
Lean:     No lean. Reading 1 is that section 7(a) found it once so it is reachable and needs a
          fixture; reading 2 is that it arose from a degenerate fit and the accumulation is
          provably insensitive to it. I could not distinguish these, and guessing would put a
          fabricated condition into the fixture spec.
Blocks:   Fixture design for the leaf-value path. Nothing in the accumulation spec.
```

---

## 10. Ambiguity, handled rather than resolved

1. **The negative-zero counting policy.** Both readings are laid out in §5 with the argument
   for each. Strict bit equality was used. It changed nothing: `0` of `60000` rows differed
   only in the sign of zero, and no row produced a zero margin at all. The policy is
   recorded because it will matter the first time a fixture does hit one, not because it
   affected any number here.

2. **`nthread` stability is platform-scoped.** Measured on darwin/arm64, CPU, `gbtree`,
   1–8 threads, batch sizes 1 to 5000, both predict APIs: `0` divergences in 1440 per-row
   observations. Two readings: (a) XGBoost's CPU gbtree predictor parallelises across rows,
   so per-row accumulation order is thread-independent by construction and this generalises;
   (b) the measurement covers one platform and one booster and says nothing about GPU or
   another architecture. **Not resolved.** The conclusion this probe needs — that bit-exact
   parity is well defined *on the reference platform* — holds under either reading.

3. **Whether the absorb-or-preserve mechanism in §1.1 is the full explanation** of the
   partial match counts. The monotone relationship between intercept ULP error and bit-exact
   count is measured; the floating-point mechanism behind it is **INFERRED**. It is offered
   as an explanation of *why* a near-miss count is diagnostic, not as a load-bearing claim.

---

## 11. Out of scope, things that looked wrong

- **`num_parallel_tree` and `tree_info` do not distinguish forest members.** With
  `num_parallel_tree=8`, `num_trees` is `200`, `tree_info` is uniformly `0`, and
  `iteration_indptr` is `[0, 8, 16, 24, ...]`. The 8 trees of a round are ordinary
  consecutive entries in `trees[]` and are summed like any others — V1 is `5000/5000`
  without special handling. Recorded because `iteration_indptr` is the *only* field that
  records the round boundary, and an exporter that drops it loses the ability to honour
  `best_iteration` or an `iteration_range`. Consistent with `probes/tree_structure.md` §10;
  no new hazard, just confirmation that the accumulation is flat.

- **Fully-pruned trees still occupy `trees[]` and still contribute.** At `gamma = 1e6` all
  three trees collapsed to a leaf-only root and the model produced exactly **one distinct
  margin** across 5000 rows, yet the arrays still carry 31 nodes each. V1 handles this
  correctly by following `left_children`, but it is a reminder that a reader must never infer
  tree shape from array length. Already covered by `probes/tree_structure.md` §7(a′); noted
  here because it is also an accumulation edge case (a leaf-only tree contributes its root's
  `split_conditions` entry).

- **Nothing in this probe contradicts any prior finding.** The one apparent tension — the
  all-float32 textbook logit scoring `5000/5000` at `p = 0.58` — is a property of that
  value, and the same formula scores `3143/5000` at `p = 0.987654` and `337/5000` at
  `p = 0.51953125`. `probes/base_score.md` §5 remains correct as written.

---

## 12. Reproducing this probe

Scripts are in the session scratch directory, not the repository:

```
scratchpad/probe-accumulation/
  common.py                  data, fitting, artifact reading, the float32 tree walk
  a01_intercept_first.py     STEP 1 -- corrected intercept, before/after counts
  a02_variants.py            STEP 2 -- 11 accumulation variants x 12 model configurations
  a03_leaves_and_zero.py     STEP 3 + 6 -- leaf narrowing, negative-zero survey
  a04_signed_zero_forced.py  STEP 6 -- 0-tree and gamma-pruned signed-zero constructions
  a05_stability.py           STEP 4 -- nthread, batch size, predict API
  a06_api_and_export.py      STEP 5 -- float64 container; exports the JS fixture
  a07_js_crosscheck.mjs      STEP 7 -- JavaScript Math.fround walk + bare-JSON.parse control
  a08_stress.py              NaN, adversarial on-threshold rows, num_parallel_tree
  a09_js_dump.mjs            JavaScript margin bit patterns, for direct parity
  a10_parity.py              Python vs JavaScript parity; 1000-tree and eta=1.0 stress
  a11_prior_residual.py      the prior probe's base_score value, run to ground
  a02_out.txt, a05_out.txt   captured full output
  js_fixture.json            trees + feature bits + XGBoost margin bits, for Node
```

Everything is seeded (`seed = 20260801`), `nthread = 1` at fit time, generic feature names
`f0..f5`, synthetic `numpy` normal data. No named datasets, no domain vocabulary. No file
was written into the repository except this report.

---

## Not measured

Stated so the gaps are visible rather than assumed away.

- **`booster="dart"` and `booster="gblinear"` accumulation.** DART applies per-tree weights
  and gblinear has no trees at all; neither is a `gbtree` sum. A separate probe owns those.
- **Multi-output / multi-class accumulation.** Out of scope per D003. `tree_info` grouping
  was observed but no `multi:*` model was summed.
- **GPU (`device="cuda"`) and non-arm64 CPUs.** The bit-stability result in §4 is
  platform-scoped.
- **`iteration_range` and `best_iteration` truncation.** V1 was measured against the default
  full-ensemble `predict`. Partial-ensemble prediction was not.
- **Objectives outside the 1.0 scope.** Three objectives measured; three reported. Nothing
  here extends by analogy to `reg:logistic`, `count:poisson`, `survival:aft`, or `rank:*`.
- **A `-0E0` leaf value in the accumulation.** Not reproduced; see the decision in §9.
- **Whether JavaScript can reproduce the float32 `1/p - 1` logit.** Deliberately excluded;
  see the decision in §9.
- **XGBoost versions other than 3.3.0.** Per D001, drift is a separate pass.
