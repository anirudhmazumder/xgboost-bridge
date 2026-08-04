# Probe — the output transform: objective pairing, and XGBoost's internal precision

Closes **FORMAT.md §14 gap G1**. Establishes, by measurement, which transform maps
`predict(output_margin=True)` to `predict()` for each of the three in-scope objectives,
and what numeric precision XGBoost evaluates that transform in.

Every claim is backed by a pasted command and its real output. Anything not directly
measured is labelled **INFERRED**. Ambiguity is presented, not resolved.

Fitted models and scripts lived entirely outside the repository. Nothing was written
into the tree except this file.

---

## Environment

```
$ uv run python -c "
import sys, numpy, xgboost, mpmath, platform
print('python', sys.version)
print('xgboost', xgboost.__version__)
print('numpy', numpy.__version__)
print('mpmath', mpmath.__version__)
print('platform', platform.platform())
print('machine', platform.machine())
"
python 3.12.8 (main, Dec  3 2024, 18:42:41) [Clang 16.0.0 (clang-1600.0.26.4)]
xgboost 3.3.0
numpy 2.5.1
mpmath 1.4.1
platform macOS-26.5.2-arm64-arm-64bit
machine arm64
```

The `mpmath` reference precision, printed from the running process:

```
mp.mp.dps=50  mp.mp.prec=169 bits
```

All findings are for `xgboost 3.3.0` exactly. No version-drift claim is made here.

---

## Verdicts up front

| Objective | `output_transform` | Precision XGBoost uses internally | Bit-exact vs `predict()` |
|---|---|---|---|
| `reg:squarederror` | **`identity`** | n/a — `predict()` is bit-identical to the margin | **27500/27500** |
| `binary:logistic` | **`sigmoid`** | **float32**, and the margin is **clamped below at `float32(-88.7)`** | **35000/35000** |
| `survival:cox` | **`exp`** | **float32**, **no clamp** — saturates to `+inf` | **45000/45000** |

Total across 43 fitted models: **107500/107500** float32 bit-exact.

| Question | Answer, measured |
|---|---|
| float32 or float64 internally? | **float32.** On the rows where the two hypotheses disagree, XGBoost sides with float32 **2772/2772** and with float64 **0/2772**. |
| Margin-level Python vs XGBoost | **`0.0`**, bit-exact `2500/2500`, at every `base_score` in the sweep, for all three objectives. Inside the `1e-6` gate. **Caveat:** for `binary:logistic` this holds for `base_score` inside `[1e-6, 1-1e-6]`; outside that window the intercept recipe itself fails and the margin error reaches `73.68` — §10. |
| Output-level Python vs XGBoost, `binary:logistic` | max abs `8.77e-08`. **Inside `1e-6`.** |
| Output-level Python vs XGBoost, `survival:cox` | max abs **`6.96e+23`**, and **`+inf`** on 734/2500 rows in one config. **Breaches `1e-6` by construction** — see §6, this is an absolute-vs-relative framing problem, not a defect. Max *relative* error is `5.9e-08`. |
| Output-level Python vs XGBoost, `reg:squarederror` | **`0.0`**. |
| `numpy` float64 `exp` vs correctly rounded | **max 1 ULP**, 158/95011 not correctly rounded. |
| `numpy` float64 `1/(1+exp(-m))` vs correctly rounded | **max 2 ULP**, 25521/85012 not correctly rounded — plus a hard **range failure** below `m = -709.78`, §7. |
| `predict()` return | plain 1-D `ndarray`, `dtype=float32`, shape `(N,)`, for all three objectives. |
| Cox output ever ≤ 0? | **Never negative, never exactly `0.0`, never `NaN`.** Can be `+inf`. Subnormals observed. |

**New findings this probe produced that no prior probe recorded, and that FORMAT.md does
not currently account for:**

1. `binary:logistic`'s transform **clamps the margin below at `float32(-88.7)`**. Any
   margin below that returns the single value `3.006635794144578e-39`, never `0.0` (§3).
   A float64 sigmoid returns `1.5e-89` on such a row — a **relative error of `1.0`**.
2. `survival:cox` **has no such clamp** and returns `+inf` for margins above ≈`88.72`
   (§4). The absolute divergence against any float64 transform is then literally
   infinite.
3. **Out of scope but wrong, reported loudly (§10):** `binary:logistic` clamps
   `base_score` to `[float32(1e-6), float32(1)−float32(1e-6)]` when deriving the margin
   intercept, while **storing the unclamped value**. The `probes/base_score.md` §9 /
   D015 intercept recipe therefore scores **0/2500** at `base_score = 1e-7` and below,
   and at `0.9999991` and above. It is not a rounding error: at `base_score=1e-38` the
   intercept is wrong by **73.68**.

---

## Method

Synthetic data only. 2500 rows × 6 columns from `numpy.random.default_rng(20260802)`,
generic feature names `f0`…`f5`, `nthread=1`, `seed=20260802`, `max_depth=4`,
`tree_method=exact`. Cox labels use the sign convention (positive = event, negative =
right-censored). Tree counts 0–600, `eta` 0.3 and 1.0, so margins span
`[-748.25, +386.64]` for `binary:logistic` and `[-94.63, +212.83]` for `survival:cox`.

Two reference computations are used and they are kept distinct:

- **The reference margin walk** — the normative recipe from `probes/accumulation.md` §6,
  vectorized over rows. `split_conditions` is loaded as `np.float32` at parse time; both
  sides of every comparison are float32 arrays; `acc` is float32 and `float32 + float32
  → float32` in numpy, so narrow-after-every-add holds by construction of the dtypes.
  The intercept is the `probes/base_score.md` §9 per-objective transform of the stored
  `base_score`.
- **The candidate output transforms** — applied to `float64(margin_f32)` per FORMAT.md
  §5.1, or to the float32 margin with every intermediate kept float32, depending on
  which hypothesis is under test. Which one is in use is stated at every table.

**Verdicts rest on float32 bit patterns** (`np.float32(x).view(np.uint32)`), not on
residuals. Residuals are reported because the brief asks for them and because they show
the margin of victory; a hit count of `2500/2500` is the evidence.

---

## 1. The objective → transform mapping

Every candidate is evaluated in **float64 from the float32 margin**, and compared against
XGBoost's own `predict()` on the same rows. All candidates are reported, not only the
winner.

```
$ uv run python s02_candidates.py
==============================================================================
objective = reg:squarederror   rounds=300 eta=0.3 max_depth=4 rows=2500
  stored base_score        = '[1.0088582E1]'  -> float32 10.088582038879395
  margin  dtype=float32 shape=(2500,) ndim=1 C_contig=True
  predict dtype=float32 shape=(2500,) ndim=1 C_contig=True
  margin  min=np.float32(-13.473269) max=np.float32(34.58281)
  predict min=np.float32(-13.473269) max=np.float32(34.58281)
  predict n_negative=184 n_zero=0 n_nonfinite=0
  n_distinct margins=2499  n_distinct predict=2499
  candidate residuals, max abs |candidate(margin_f64) - predict()| :
    identity              m          max_abs_resid = 0.0                        nonfinite=0   <== SMALLEST
    negate               -m          max_abs_resid = 69.16561889648438          nonfinite=0
    sigmoid   1/(1+exp(-m))          max_abs_resid = 33.58280944824219          nonfinite=0
    sigmoid_flip 1/(1+e^m)           max_abs_resid = 34.58280944824219          nonfinite=0
    sigmoid_tanh .5(1+tanh(m/2))     max_abs_resid = 33.58280944824219          nonfinite=0
    exp              exp(m)          max_abs_resid = 1045016895607439.0         nonfinite=0
    exp_neg         exp(-m)          max_abs_resid = 710190.4369526674          nonfinite=0
    expm1          expm1(m)          max_abs_resid = 1045016895607438.0         nonfinite=0
    softplus  log1p(exp(m))          max_abs_resid = 13.473270870684201         nonfinite=0
    log              log(m)          max_abs_resid = 31.039452726398792         nonfinite=184
    relu         max(m, 0)           max_abs_resid = 13.47326946258545          nonfinite=0

==============================================================================
objective = binary:logistic   rounds=300 eta=0.3 max_depth=4 rows=2500
  stored base_score        = '[4.98E-1]'  -> float32 0.49799999594688416
  margin  dtype=float32 shape=(2500,) ndim=1 C_contig=True
  predict dtype=float32 shape=(2500,) ndim=1 C_contig=True
  margin  min=np.float32(-11.983445) max=np.float32(10.597893)
  predict min=np.float32(6.246736e-06) max=np.float32(0.9999751)
  predict n_negative=0 n_zero=0 n_nonfinite=0
  n_distinct margins=2500  n_distinct predict=2500
  candidate residuals, max abs |candidate(margin_f64) - predict()| :
    identity              m          max_abs_resid = 11.983451414277624         nonfinite=0
    negate               -m          max_abs_resid = 11.983438920805384         nonfinite=0
    sigmoid   1/(1+exp(-m))          max_abs_resid = 8.618250790792814e-08      nonfinite=0   <== SMALLEST
    sigmoid_flip 1/(1+e^m)           max_abs_resid = 0.9999875065275278         nonfinite=0
    sigmoid_tanh .5(1+tanh(m/2))     max_abs_resid = 8.618250801895044e-08      nonfinite=0
    exp              exp(m)          max_abs_resid = 40049.352816263796         nonfinite=0
    exp_neg         exp(-m)          max_abs_resid = 160082.59303603956         nonfinite=0
    expm1          expm1(m)          max_abs_resid = 40048.352816263796         nonfinite=0
    softplus  log1p(exp(m))          max_abs_resid = 9.597942644229347          nonfinite=0
    log              log(m)          max_abs_resid = 1.8575912207741334         nonfinite=1255
    relu         max(m, 0)           max_abs_resid = 9.597917675971985          nonfinite=0

==============================================================================
objective = survival:cox   rounds=300 eta=0.3 max_depth=4 rows=2500
  stored base_score        = '[1E0]'  -> float32 1.0
  margin  dtype=float32 shape=(2500,) ndim=1 C_contig=True
  predict dtype=float32 shape=(2500,) ndim=1 C_contig=True
  margin  min=np.float32(-8.161462) max=np.float32(20.444147)
  predict min=np.float32(0.0002854448) max=np.float32(7.564501e+08)
  predict n_negative=0 n_zero=0 n_nonfinite=0
  n_distinct margins=2500  n_distinct predict=2500
  candidate residuals, max abs |candidate(margin_f64) - predict()| :
    identity              m          max_abs_resid = 756450091.5558529          nonfinite=0
    negate               -m          max_abs_resid = 756450132.4441471          nonfinite=0
    sigmoid   1/(1+exp(-m))          max_abs_resid = 756450111.0                nonfinite=0
    sigmoid_flip 1/(1+e^m)           max_abs_resid = 756450112.0                nonfinite=0
    sigmoid_tanh .5(1+tanh(m/2))     max_abs_resid = 756450111.0                nonfinite=0
    exp              exp(m)          max_abs_resid = 25.136664867401123         nonfinite=0   <== SMALLEST
    exp_neg         exp(-m)          max_abs_resid = 756450112.0                nonfinite=0
    expm1          expm1(m)          max_abs_resid = 26.136664867401123         nonfinite=0
    softplus  log1p(exp(m))          max_abs_resid = 756450091.5558529          nonfinite=0
    log              log(m)          max_abs_resid = 756450108.9823034          nonfinite=497
    relu         max(m, 0)           max_abs_resid = 756450091.5558529          nonfinite=0
```

Absolute residual is a misleading yardstick on `survival:cox`, where `predict()` spans
`[2.85e-04, 7.56e+08]` — `exp`'s absolute residual of `25.14` is a *relative* residual of
`3.3e-08`. The same candidate set scored on max **relative** residual:

```
$ uv run python s16_rel_candidates.py
reg:squarederror   |predict| in [0.00464955298230052, 34.58280944824219]
    identity              m          max_rel_resid = 0.0                          <== SMALLEST
    negate               -m          max_rel_resid = 2.0
    sigmoid   1/(1+exp(-m))          max_rel_resid = 108.2872200289843
    sigmoid_flip 1/(1+e^m)           max_rel_resid = 108.78721912822196
    sigmoid_tanh .5(1+tanh(m/2))     max_rel_resid = 108.2872200289843
    exp              exp(m)          max_rel_resid = 30217813771649.957
    exp_neg         exp(-m)          max_rel_resid = 52711.06904859495
    expm1          expm1(m)          max_rel_resid = 30217813771649.926
    softplus  log1p(exp(m))          max_rel_resid = 149.57882230592827
    log              log(m)          max_rel_resid = 124.5678305914144
    relu         max(m, 0)           max_rel_resid = 1.0

binary:logistic   |predict| in [6.246736120374408e-06, 0.9999750852584839]
    identity              m          max_rel_resid = 1918354.0305460154
    negate               -m          max_rel_resid = 1918352.0305460154
    sigmoid   1/(1+exp(-m))          max_rel_resid = 1.0421397373421966e-07       <== SMALLEST
    sigmoid_flip 1/(1+e^m)           max_rel_resid = 160081.59897549698
    sigmoid_tanh .5(1+tanh(m/2))     max_rel_resid = 1.0421397373421966e-07       <== SMALLEST
    exp              exp(m)          max_rel_resid = 40050.35066039813
    exp_neg         exp(-m)          max_rel_resid = 25626597626.544975
    expm1          expm1(m)          max_rel_resid = 160083.5989692502
    softplus  log1p(exp(m))          max_rel_resid = 9.598181780447431
    log              log(m)          max_rel_resid = 3.267852552486752
    relu         max(m, 0)           max_rel_resid = 9.598156811567977

survival:cox   |predict| in [0.00028544481028802693, 756450112.0]
    identity              m          max_rel_resid = 28593.08342903089
    negate               -m          max_rel_resid = 28591.08342903089
    sigmoid   1/(1+exp(-m))          max_rel_resid = 0.9999999986780358
    sigmoid_flip 1/(1+e^m)           max_rel_resid = 3501.3044757020257
    sigmoid_tanh .5(1+tanh(m/2))     max_rel_resid = 0.9999999986780358
    exp              exp(m)          max_rel_resid = 5.896253432614965e-08        <== SMALLEST
    exp_neg         exp(-m)          max_rel_resid = 12273138.926350038
    expm1          expm1(m)          max_rel_resid = 3503.3041903386556
    softplus  log1p(exp(m))          max_rel_resid = 0.9999999729735686
    log              log(m)          max_rel_resid = 8.686373487547094
    relu         max(m, 0)           max_rel_resid = 1.0
```

**The mapping, measured:**

| Objective | Winner | Margin of victory over the runner-up |
|---|---|---|
| `reg:squarederror` | `identity`, residual **exactly `0.0`** | next best `1.0` relative (`relu`) — infinite ratio |
| `binary:logistic` | `sigmoid`, `1.04e-07` relative | next distinct function `3.27` relative (`log`) — **7.5 orders** |
| `survival:cox` | `exp`, `5.90e-08` relative | next best `0.99999997` relative (`softplus`) — **7.2 orders** |

Two things the residual table cannot do, stated so they are not over-read:

- `sigmoid` and `sigmoid_tanh` are **the same mathematical function** in two
  formulations, and they tie to `1.0421397373421966e-07`. A residual test cannot
  separate formulations of one function; only the bit-exact test in §2 can, and it
  selects a specific one.
- `identity` scoring exactly `0.0` on `reg:squarederror` means `predict()` is
  **bit-identical** to `predict(output_margin=True)`, confirmed separately at
  `27500/27500` in §11.

---

## 2. Precision — float32, decided on the disagreement subset

`predict()` returns `float32` (§8), so every hypothesis must be narrowed to float32
before comparison. That narrowing hides most of the difference: the two hypotheses agree
on 64% of rows. **The question is only decidable on the rows where they disagree**, so
that subset is isolated and counted.

Hypotheses, all starting from the identical float32 margin:

- **float64** — the transform entirely in float64, narrowed once at the end.
- **float32** — every intermediate narrowed to float32, using numpy's float32 `exp`.
- plus a correctly-rounded-float32-`exp` variant, and the exact value from `mpmath`
  rounded once to float32, and rounded float64-then-float32, to check whether the
  answer is an artifact of numpy's particular `exp`.

```
$ uv run python s03_precision.py
objective=binary:logistic  base_score_arg=None  rounds=300  rows=2500
  margin range [np.float32(-11.983445), np.float32(10.597893)]
  predict() dtype=float32   n=2500
  bit-exact match against XGBoost predict(), per hypothesis:
    f64 np.exp, narrow at end                   1600/2500  max_abs_diff=5.960464477539063e-08
    f64 math.exp, narrow at end                 1600/2500  max_abs_diff=5.960464477539063e-08
    f32 np.exp, every step f32                  2500/2500  max_abs_diff=0.0
    f32 correctly-rounded exp, f32 steps        2495/2500  max_abs_diff=1.1920928955078125e-07
    exact -> float32 (one rounding)             1600/2500  max_abs_diff=5.960464477539063e-08
    exact -> float64 -> float32                 1600/2500  max_abs_diff=5.960464477539063e-08
  DISAGREEMENT SUBSET  ('f64 np.exp, narrow at end' vs 'f32 np.exp, every step f32'): 900/2500 rows
    on those 900 rows: XGBoost == f64-hypothesis  0/900
    on those 900 rows: XGBoost == f32-hypothesis  900/900
    on those 900 rows: XGBoost == neither          0/900
      row     1 margin=np.float32(2.8851357)
        xgboost  = 0.9471067786216736  bits=1064465815
        f64 hyp  = 0.9471067190170288  bits=1064465814
        f32 hyp  = 0.9471067786216736  bits=1064465815
      row     5 margin=np.float32(-7.37625)
        xgboost  = 0.0006255523185245693  bits=975436827
        f64 hyp  = 0.0006255523767322302  bits=975436828
        f32 hyp  = 0.0006255523185245693  bits=975436827
```

Same script, all six configurations, the deciding lines:

```
objective=binary:logistic  base_score_arg=None  rounds=300
    f64 np.exp, narrow at end                   1600/2500
    f32 np.exp, every step f32                  2500/2500
  DISAGREEMENT SUBSET: 900/2500 rows
    on those 900 rows: XGBoost == f64-hypothesis  0/900
    on those 900 rows: XGBoost == f32-hypothesis  900/900

objective=binary:logistic  base_score_arg=0.05  rounds=200
    f64 np.exp, narrow at end                   1586/2500
    f32 np.exp, every step f32                  2500/2500
  DISAGREEMENT SUBSET: 914/2500 rows
    on those 914 rows: XGBoost == f64-hypothesis  0/914
    on those 914 rows: XGBoost == f32-hypothesis  914/914

objective=binary:logistic  base_score_arg=0.95  rounds=200
    f64 np.exp, narrow at end                   1586/2500
    f32 np.exp, every step f32                  2500/2500
  DISAGREEMENT SUBSET: 914/2500 rows
    on those 914 rows: XGBoost == f64-hypothesis  0/914
    on those 914 rows: XGBoost == f32-hypothesis  914/914

objective=survival:cox  base_score_arg=None  rounds=300
    f64 np.exp, narrow at end                   2483/2500
    f32 np.exp (float32 exp)                    2500/2500
  DISAGREEMENT SUBSET: 17/2500 rows
    on those 17 rows: XGBoost == f64-hypothesis  0/17
    on those 17 rows: XGBoost == f32-hypothesis  17/17

objective=survival:cox  base_score_arg=0.25  rounds=200
    f64 np.exp, narrow at end                   2486/2500
    f32 np.exp (float32 exp)                    2500/2500
  DISAGREEMENT SUBSET: 14/2500 rows
    on those 14 rows: XGBoost == f64-hypothesis  0/14
    on those 14 rows: XGBoost == f32-hypothesis  14/14

objective=survival:cox  base_score_arg=7.5  rounds=200
    f64 np.exp, narrow at end                   2487/2500
    f32 np.exp (float32 exp)                    2500/2500
  DISAGREEMENT SUBSET: 13/2500 rows
    on those 13 rows: XGBoost == f64-hypothesis  0/13
    on those 13 rows: XGBoost == f32-hypothesis  13/13
```

### The counts that decide it

| Hypothesis | `binary:logistic`, 3 configs | `survival:cox`, 3 configs | Total |
|---|---|---|---|
| **float32 throughout** | 7500/7500 | 7500/7500 | **15000/15000** |
| **float64, narrowed at end** | 4772/7500 | 7456/7500 | 12228/15000 |
| On the **disagreement subset**, XGBoost == float32 | 2728/2728 | 44/44 | **2772/2772** |
| On the **disagreement subset**, XGBoost == float64 | 0/2728 | 0/44 | **0/2772** |

2728 = 900 + 914 + 914 (logistic). 44 = 17 + 14 + 13 (Cox). 2772 = 2728 + 44.

**Verdict: XGBoost 3.3.0 computes the output transform in float32.** Not "probably" —
`0/2772` for float64 on the only rows that can distinguish them.

Two supporting details, both measured:

- `f64 np.exp` and `f64 math.exp` give **identical** counts (1600/1600, 1586/1586,
  2483/2483, …). The result is not an artifact of which float64 `exp` is used.
- `exact -> float32 (one rounding)` from `mpmath` also scores 1600/2500, i.e. it matches
  the float64 hypothesis rather than XGBoost. So XGBoost is not computing a
  *correctly rounded* float32 sigmoid either; it is computing a float32 sigmoid from a
  float32 `expf` that carries its own rounding error. `f32 correctly-rounded exp`
  scores 2495–2499/2500 where numpy's own float32 `exp` scores 2500/2500. So on 1–5 rows
  per configuration, **XGBoost agrees with numpy's `expf` and disagrees with a correctly
  rounded one.** *INFERRED* that those rows are the ones where numpy's `expf` is 1 ULP
  off correctly rounded; the alternative is that the divergence enters at the divide
  rather than the `exp`. Distinguishing them was not measured.

That last point is worth stating precisely, because it is stronger than it looks. On
this platform numpy's float32 `exp` and the platform `libm`'s `expf` are bit-identical:

```
$ uv run python s09_ulp.py
  n=50008
  numpy float32 exp vs platform libm expf (ctypes): bit-exact 50008/50008
  numpy float32 exp   ULP histogram vs mpmath: {0: 49686, 1: 322}
    MAX = 1 at m=75.54653930664062 numpy=6.448298520364634e+32 mpmath=6.448298907220896e+32
  platform expf       ULP histogram vs mpmath: {0: 49686, 1: 322}
    MAX = 1 at m=75.54653930664062 expf=6.448298520364634e+32 mpmath=6.448298907220896e+32
```

**INFERRED, not measured:** that XGBoost's C++ transform calls the same platform `expf`
these two do. What is *measured* is only that all three agree bit-for-bit on every
sampled point including the points where `expf` is 1 ULP wrong. Confirming the call
would need a symbol trace or a source read, neither of which this probe did.

---

## 3. `binary:logistic` clamps the margin below at `float32(-88.7)`

This was not being looked for. It surfaced as **one row in 2500** that no hypothesis
reproduced — the exact frequency at which a defect survives review.

At `base_score = 0.987654` with 200 rounds the margin range is
`[-110.90, +114.16]`. In float32, `expf(m)` overflows to `+inf` for `m > 88.7228`, so a
plain float32 `1/(1+expf(-m))` returns **exactly `0.0`** for `m < -88.7228`. XGBoost
does not:

```
$ uv run python s05_saturation.py
rows=2500  margin range [-110.90191650390625, 114.16361236572266]
predict range [3.006635794144578e-39, 1.0]
predict n_exactly_zero=0  n_exactly_one=54  n_subnormal=1

bit-exact vs predict(), ALL rows:
  f32 naive 1/(1+expf(-m))            2499/2500
  f64 naive, narrowed                 1650/2500
  f32 sign-stable branch              1941/2500
  f32 divide of narrowed f64 exp      2492/2500

the 8 most negative and 8 most positive tail rows, verbatim:
  row  1094 margin=-110.90191650390625
        xgboost predict()          = 3.006635794144578e-39  bits=2145607
        f32 naive 1/(1+expf(-m))   = 0.0
        f64 naive narrowed         = 0.0
        f32 sign-stable branch     = 0.0
```

The floor value identifies the clamp uniquely. An exhaustive scan of **every** float32 in
`[-90, -88]`:

```
$ uv run python s06_clamp.py
observed predict() floor value = 3.006635794144578e-39  bits=2145607
  is subnormal float32: True
  1/target = 3.3259765015353694e+38

candidate clamp constants c: float32 value of 1/(1+expf(+c))
  c=88.0         expf(c)=1.6516362661361307e+38     1/(1+expf(c))=6.054601485195952e-39      bits=4320708 match=False
  c=88.5         expf(c)=2.723087918012828e+38      1/(1+expf(c))=3.672301610145117e-39      bits=2620642 match=False
  c=88.7         expf(c)=3.325976864406685e+38      1/(1+expf(c))=3.006635794144578e-39      bits=2145607 match=True
  c=88.72        expf(c)=3.3931806003874245e+38     1/(1+expf(c))=2.947087615903095e-39      bits=2103112 match=False
  c=88.7228      expf(c)=3.4026946730843054e+38     1/(1+expf(c))=2.938847980932865e-39      bits=2097232 match=False
  c=89.0         expf(c)=inf                        1/(1+expf(c))=0.0                        bits=0 match=False

exhaustive: every float32 m in [-90, -88] whose f32 sigmoid has the target bits
  scanned 262145 float32 values; 1 produce the target bits
  range of matching m: [-88.69999694824219, -88.69999694824219]
```

**Exactly one** of 262145 candidate float32 values reproduces the observed bits, and it is
`float32(-88.7) = -88.69999694824219`. Confirmed on 2056 more below-clamp rows across
three configurations, including margins as low as `-748.25`:

```
$ uv run python s07_clamp_confirm.py
clamp constant under test: float32(-88.7) = -88.69999694824219  bits=3266405990

binary:logistic, configurations chosen to drive margins past the clamp
  eta=1.0 rounds=300 base_score=None
    margin range [-20.916406631469727, 17.788005828857422]  rows below clamp = 0   rows above +88.7 = 0
      unclamped f32 sigmoid            all 2500/2500   below-clamp subset None/0
      lower-clamped f32 sigmoid        all 2500/2500   below-clamp subset None/0
      two-sided-clamped f32 sigmoid    all 2500/2500   below-clamp subset None/0

  eta=1.0 rounds=300 base_score=0.987654
    margin range [-748.246337890625, 386.6369323730469]  rows below clamp = 1959   rows above +88.7 = 286
    predict range [3.006635794144578e-39, 1.0]  n_exact_zero=0  n_exact_one=288
      unclamped f32 sigmoid            all 541/2500   below-clamp subset 0/1959
      lower-clamped f32 sigmoid        all 2500/2500   below-clamp subset 1959/1959
      two-sided-clamped f32 sigmoid    all 2500/2500   below-clamp subset 1959/1959
      distinct predict() values on the 1959 below-clamp rows: [3.006635794144578e-39]
      distinct bit patterns: [2145607]

  eta=1.0 rounds=600 base_score=0.05
    margin range [-327.9169921875, 355.8014221191406]  rows below clamp = 97   rows above +88.7 = 87
    predict range [3.006635794144578e-39, 1.0]  n_exact_zero=0  n_exact_one=666
      unclamped f32 sigmoid            all 2403/2500   below-clamp subset 0/97
      lower-clamped f32 sigmoid        all 2500/2500   below-clamp subset 97/97
      two-sided-clamped f32 sigmoid    all 2500/2500   below-clamp subset 97/97
      distinct predict() values on the 97 below-clamp rows: [3.006635794144578e-39]
      distinct bit patterns: [2145607]
```

`0/1959` for the unclamped form, `1959/1959` for the clamped form, and **one** distinct
bit pattern across all 1959 rows regardless of whether the margin is `-88.71` or `-748.25`.

`predict()` for `binary:logistic` was **never** exactly `0.0` in any configuration
measured — `n_exact_zero=0` in all of them. It **was** exactly `1.0` on 288/2500 rows.

### Ambiguity A1 — clamp on the input, or floor on the output

Both readings fit every measurement and this probe does not choose between them:

- **(a)** the margin is clamped to `≥ float32(-88.7)` before the sigmoid;
- **(b)** the sigmoid output is floored at `1/(1+expf(88.7))`.

Because the float32 sigmoid is monotone, (a) and (b) produce identical output on every
input. What *is* measured, and is the same under both readings:
`predict()` on any margin below `float32(-88.7)` is exactly `3.006635794144578e-39`,
bits `2145607`. Distinguishing them would need a source read or an input on which
monotonicity fails, and neither exists.

### Ambiguity A2 — the upper clamp is unobservable

`two-sided-clamped f32 sigmoid` and `lower-clamped f32 sigmoid` both score `2500/2500`,
including on the 286 rows above `+88.7`. They cannot be separated, because
`1/(1+expf(-88.7))` and `1/(1+expf(-386.64))` are **both exactly `1.0`** in float32. So:
whether XGBoost also clamps above at `+88.7` is **not decidable from `predict()`**. It
does not matter numerically — the observable output is `1.0` either way.

`predict(output_margin=True)` is **not** clamped: it returned `-748.246337890625`. The
clamp lives inside the transform path only.

---

## 4. `survival:cox` has no clamp and saturates to `+inf`

The asymmetry with §3 is measured, not assumed.

```
$ uv run python s06_clamp.py
survival:cox at large margins -- does exp saturate to inf or clamp?
  eta=1.0 rounds=400: margin range [1.0444279909133911, 120.70205688476562]
    predict range [2.8417725563049316, inf]  n_inf=734 n_zero=0 n_nan=0
    f32 exp: n_inf=734  bit-exact vs predict 2500/2500
    f64 exp narrowed: n_inf=734  bit-exact vs predict 2489/2500
    argmax row 193: margin=120.70205688476562 predict=inf f32exp=inf f64exp=inf
    rows with margin > 88.0 : 734
      row 4 margin=112.06353759765625 predict=inf f32exp=inf
      row 7 margin=112.06353759765625 predict=inf f32exp=inf
```

`734/2500` rows return `+inf`. If Cox clamped at `88.7` the way logistic does, those rows
would return `expf(88.7) = 3.325976864406685e+38`, a finite float32. They do not.

The lower tail is likewise unclamped — a margin of `-94.63` produces
`expf(-94.63)`, not `expf(-88.7)`:

```
$ uv run python s07_clamp_confirm.py
survival:cox, driven to both tails
  eta=0.3 rounds=300 base_score=1e-12 stored='[1E-12]' intercept=-27.63102149963379
    margin range [-35.1800537109375, -6.171255111694336]
    predict range [5.266193356701933e-16, 0.002088612876832485]
    predict n_negative=0 n_exact_zero=0 n_inf=0 n_nan=0 n_subnormal=0
    f32 exp        bit-exact 2500/2500
    f64 exp narrow bit-exact 2491/2500

$ uv run python s08_cox_tail.py
eta=0.3 rounds=300 base_score=1e-38 stored='[1E-38]' float32=9.999999350456404e-39 intercept=-87.49822998046875
  margin range [-94.62601470947266, -66.37680053710938]
  predict range [8.025236305188227e-42, 1.4890929218137535e-29]
  n_negative=0 n_exact_zero=0 n_inf=0 n_nan=0 n_subnormal=467
  rows with margin < -103.28 : 0
  f32 exp        bit-exact 2500/2500
  f64 exp narrow bit-exact 2493/2500
    row  1205 margin=-94.62601470947266 predict=8.025236305188227e-42 f32exp=8.025236305188227e-42 f64exp_narrowed=8.025236305188227e-42
```

`predict() = 8.025236305188227e-42` at margin `-94.63`, a **float32 subnormal**, not the
`3.0e-39` a `-88.7` clamp would give. `467/2500` outputs were subnormal.

The float32 `exp` underflow boundary, for reference:

```
float32 exp underflow reference points:
  np.exp(np.float32(-87.0)) = 1.6458114537543937e-38
  np.exp(np.float32(-95.0)) = 5.521115949439779e-42
  np.exp(np.float32(-103.0)) = 1.401298464324817e-45
  np.exp(np.float32(-103.28)) = 1.401298464324817e-45
  np.exp(np.float32(-104.0)) = 0.0
  np.exp(np.float32(-200.0)) = 0.0
```

**Not measured:** no fitted Cox model in this probe produced a margin below `-103.28`
(`rows with margin < -103.28 : 0` in all three configurations), so a Cox `predict()` of
exactly `0.0` was **never observed**. **INFERRED:** it is reachable, since float32 `exp`
returns `0.0` below `-104.0` and no clamp was found. Confirming it needs a model with a
more negative intercept than `ln(float32(1e-38)) = -87.498`, which is the most negative
this probe could reach — `base_score` smaller than that stores a value whose float32
`1/p` overflows.

---

## 5. `base_score` sweep — reported per value, never averaged

Reference walk vs `predict(output_margin=True)` at the margin, and float64 transform of
the float32 margin vs `predict()` at the output. `1e-6` gate checked explicitly at both.

`base_score = 0.5` is included because it is the known degenerate case
(`probes/accumulation.md` §8): the logistic intercept is exactly `-0.0` there, and Cox's
estimated default gives exactly `0.0`. Values far from it are included for the same
reason.

```
$ uv run python s04_sweep.py
==============================================================================
objective=reg:squarederror   output_transform=identity   rounds=200 eta=0.3 max_depth=4 rows=2500
  gate: margin <= 1e-06, output <= 1e-06
  --- base_score arg = None
      stored='[1.0088582E1]'  float32=10.088582038879395  intercept=10.088582038879395  intercept_bits=1092709077
      margin range [-13.595444679260254, 34.59975051879883]
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_rel_err=0.0   float32(py_out) bit-exact vs predict() 2500/2500
  --- base_score arg = 0.5
      stored='[5E-1]'  float32=0.5  intercept=0.5  intercept_bits=1056964608
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=0.0  PASS vs 1e-06
  --- base_score arg = -7.5
      stored='[-7.5E0]'  float32=-7.5  intercept=-7.5  intercept_bits=3236954112
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=0.0  PASS vs 1e-06
  --- base_score arg = 100.0
      stored='[1E2]'  float32=100.0  intercept=100.0  intercept_bits=1120403456
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=0.0  PASS vs 1e-06
  --- base_score arg = 0.987654
      stored='[9.87654E-1]'  float32=0.9876539707183838  intercept=0.9876539707183838  intercept_bits=1065146084
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=0.0  PASS vs 1e-06
```

| `reg:squarederror` `base_score` | intercept | margin max abs err | output max abs err | gate |
|---|---|---|---|---|
| `None` → `10.088582` | `10.088582038879395` | **`0.0`** | **`0.0`** | PASS / PASS |
| `0.5` | `0.5` | **`0.0`** | **`0.0`** | PASS / PASS |
| `-7.5` | `-7.5` | **`0.0`** | **`0.0`** | PASS / PASS |
| `100.0` | `100.0` | **`0.0`** | **`0.0`** | PASS / PASS |
| `0.987654` | `0.9876539707183838` | **`0.0`** | **`0.0`** | PASS / PASS |

```
==============================================================================
objective=binary:logistic   output_transform=sigmoid   rounds=200 eta=0.3 max_depth=4 rows=2500
  --- base_score arg = None
      stored='[4.98E-1]'  float32=0.49799999594688416  intercept=-0.007999997586011887  intercept_bits=3154317932
      margin range [-10.650321006774902, 9.437246322631836]
      predict range [2.3692673494224437e-05, 0.9999202489852905]
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=8.474226997901013e-08  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_rel_err=1.1048204156348363e-07   float32(py_out) bit-exact vs predict() 1647/2500
  --- base_score arg = 0.5
      stored='[5E-1]'  float32=0.5  intercept=-0.0  intercept_bits=2147483648
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=8.35906645013651e-08  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_rel_err=1.0186656758122632e-07   float32(py_out) bit-exact vs predict() 1590/2500
  --- base_score arg = 0.05
      stored='[5E-2]'  float32=0.05000000074505806  intercept=-2.944438934326172  intercept_bits=3225186736
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=8.485134805891192e-08  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_rel_err=1.0260785176352246e-07   float32(py_out) bit-exact vs predict() 1586/2500
  --- base_score arg = 0.95
      stored='[9.5E-1]'  float32=0.949999988079071  intercept=2.9444382190704346  intercept_bits=1077703085
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=8.589636857347926e-08  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_rel_err=1.0758997221848816e-07   float32(py_out) bit-exact vs predict() 1586/2500
  --- base_score arg = 0.987654
      stored='[9.87654E-1]'  float32=0.9876539707183838  intercept=4.381994247436523  intercept_bits=1082931532
      margin range [-110.90191650390625, 114.16361236572266]
      predict range [3.006635794144578e-39, 1.0]
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=8.753329094890461e-08  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_rel_err=0.9999999997720557   float32(py_out) bit-exact vs predict() 1650/2500
  --- base_score arg = 0.48
      stored='[4.8E-1]'  float32=0.47999998927116394  intercept=-0.08004285395145416  intercept_bits=3181636994
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=8.765505354890735e-08  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_rel_err=1.084044033047996e-07   float32(py_out) bit-exact vs predict() 1657/2500
```

| `binary:logistic` `base_score` | intercept | margin max abs err | output max abs err | output max rel err | gate |
|---|---|---|---|---|---|
| `None` → `0.498` | `-0.007999997586011887` | **`0.0`** | `8.474226997901013e-08` | `1.10e-07` | PASS / PASS |
| **`0.5`** (degenerate) | **`-0.0`**, bits `2147483648` | **`0.0`** | `8.35906645013651e-08` | `1.02e-07` | PASS / PASS |
| `0.05` | `-2.944438934326172` | **`0.0`** | `8.485134805891192e-08` | `1.03e-07` | PASS / PASS |
| `0.95` | `2.9444382190704346` | **`0.0`** | `8.589636857347926e-08` | `1.08e-07` | PASS / PASS |
| `0.987654` | `4.381994247436523` | **`0.0`** | `8.753329094890461e-08` | **`0.9999999998`** | PASS / PASS |
| `0.48` | `-0.08004285395145416` | **`0.0`** | `8.765505354890735e-08` | `1.08e-07` | PASS / PASS |

The `0.99999999977` relative error at `base_score = 0.987654` is the §3 clamp, on one
row. Absolute error there is `3.0e-39`, so the `1e-6` absolute gate does not see it at
all. §6 has the numbers.

```
==============================================================================
objective=survival:cox   output_transform=exp   rounds=200 eta=0.3 max_depth=4 rows=2500
  --- base_score arg = None
      stored='[1E0]'  float32=1.0  intercept=0.0  intercept_bits=0
      margin range [-8.77393627166748, 18.38960075378418]
      predict range [0.00015471338701900095, 96939800.0]
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=2.7663360238075256  BREACH vs 1e-06
      OUTPUT py-vs-xgb : max_rel_err=5.837661785201937e-08   float32(py_out) bit-exact vs predict() 2489/2500
  --- base_score arg = 0.5
      stored='[5E-1]'  float32=0.5  intercept=-0.6931471824645996  intercept_bits=3207688728
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=2.7515079975128174  BREACH vs 1e-06
      OUTPUT py-vs-xgb : max_rel_err=5.83093827776303e-08   float32(py_out) bit-exact vs predict() 2491/2500
  --- base_score arg = 0.25
      stored='[2.5E-1]'  float32=0.25  intercept=-1.3862943649291992  intercept_bits=3216077336
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=1.9386094510555267  BREACH vs 1e-06
      OUTPUT py-vs-xgb : max_rel_err=5.9264570381630625e-08   float32(py_out) bit-exact vs predict() 2486/2500
  --- base_score arg = 7.5
      stored='[7.5E0]'  float32=7.5  intercept=2.0149030685424805  intercept_bits=1073804332
      predict range [0.0010209030006080866, 810706240.0]
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=29.750519394874573  BREACH vs 1e-06
      OUTPUT py-vs-xgb : max_rel_err=5.886625963341525e-08   float32(py_out) bit-exact vs predict() 2487/2500
  --- base_score arg = 3.1415927
      stored='[3.1415927E0]'  float32=3.1415927410125732  intercept=1.1447299718856812  intercept_bits=1066567299
      MARGIN py-vs-xgb : bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
      OUTPUT py-vs-xgb : max_abs_err=4.546594977378845  BREACH vs 1e-06
      OUTPUT py-vs-xgb : max_rel_err=5.727536606160499e-08   float32(py_out) bit-exact vs predict() 2484/2500
```

| `survival:cox` `base_score` | intercept | margin max abs err | output max abs err | output max rel err | gate |
|---|---|---|---|---|---|
| **`None` → `1.0`** (degenerate) | **`0.0`**, bits `0` | **`0.0`** | **`2.7663360238075256`** | `5.84e-08` | PASS / **BREACH** |
| `0.5` | `-0.6931471824645996` | **`0.0`** | **`2.7515079975128174`** | `5.83e-08` | PASS / **BREACH** |
| `0.25` | `-1.3862943649291992` | **`0.0`** | **`1.9386094510555267`** | `5.93e-08` | PASS / **BREACH** |
| `7.5` | `2.0149030685424805` | **`0.0`** | **`29.750519394874573`** | `5.89e-08` | PASS / **BREACH** |
| `3.1415927` | `1.1447299718856812` | **`0.0`** | **`4.546594977378845`** | `5.73e-08` | PASS / **BREACH** |

**Margin level: `0.0` at every single `base_score` value, for all three objectives,
all 16 configurations of this sweep, `2500/2500` bit-exact each.** That is inside the `1e-6` gate with
room to spare, and it is better than the "low single-digit `1e-7`" the brief expected —
it is exactly zero.

**Output level for `survival:cox` breaches the `1e-6` absolute gate at every single
`base_score` value.** §6.

---

## 6. Output-level divergence — the framing, then the numbers

**State this before reading any output-level number.** XGBoost evaluates its transform in
C++ `libm` (§2: measured to be a float32 computation on this platform). This library will
call a bundled transform instead (D030). A divergence of roughly 1–2 ULP at the output is
therefore **expected by construction and is not a regression**. Nobody reviewing a later
report should read the output-level figures below as a defect.

That framing covers `binary:logistic` and `reg:squarederror`. It does **not** cover
`survival:cox`, where the divergence is not 1–2 ULP of anything.

```
$ uv run python s12_divergence.py
------------------------------------------------------------------------------
reg:squarederror  base_score=None rounds=200 eta=0.3
  MARGIN  bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
  OUTPUT  xgb_inf=0  py_inf=0  xgb_inf_while_py_finite=0
  OUTPUT  max_abs_err (finite pairs only) = 0.0  PASS vs 1e-06
  OUTPUT  max_rel_err (finite pairs only) = 0.0
------------------------------------------------------------------------------
binary:logistic  base_score=0.987654 rounds=200 eta=0.3
  margin[-110.90191650390625, 114.16361236572266]
  MARGIN  bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
  OUTPUT  max_abs_err (finite pairs only) = 8.753329094890461e-08  PASS vs 1e-06
  OUTPUT  max_rel_err (finite pairs only) = 0.9999999997720557
    rows below the -88.7 clamp: 1
    example row 1094: margin=-110.90191650390625 xgb=3.006635794144578e-39 py_f64=6.853456015150463e-49
      abs diff = 3.006635793459232e-39   rel diff = 0.9999999997720557
------------------------------------------------------------------------------
binary:logistic  base_score=0.987654 rounds=300 eta=1.0
  margin[-748.246337890625, 386.6369323730469]
  MARGIN  bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
  OUTPUT  max_abs_err (finite pairs only) = 2.2100462790126257e-11  PASS vs 1e-06
  OUTPUT  max_rel_err (finite pairs only) = 1.0
    rows below the -88.7 clamp: 1959
    example row 0: margin=-204.50521850585938 xgb=3.006635794144578e-39 py_f64=1.5293682943107931e-89
      abs diff = 3.006635794144578e-39   rel diff = 1.0
------------------------------------------------------------------------------
survival:cox  base_score=7.5 rounds=400 eta=1.0
  margin[1.0444279909133911, 120.70205688476562]
  MARGIN  bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
  OUTPUT  xgb_inf=734  py_inf=0  xgb_inf_while_py_finite=734
  OUTPUT  max_abs_err (finite pairs only) = 6.955377819341459e+23  BREACH vs 1e-06
  OUTPUT  max_rel_err (finite pairs only) = 5.480421454834668e-08
    example row 4: margin=112.06353759765625 xgb=inf py_f64=4.6620400622695623e+48   -> absolute divergence is +inf
------------------------------------------------------------------------------
survival:cox  base_score=1e-38 rounds=300 eta=0.3
  margin[-94.62601470947266, -66.37680053710938]
  MARGIN  bit-exact 2500/2500  max_abs_err=0.0  PASS vs 1e-06
  OUTPUT  max_abs_err (finite pairs only) = 1.1848462734550676e-37  PASS vs 1e-06
  OUTPUT  max_rel_err (finite pairs only) = 5.6613137378082795e-05
```

### Three regimes where the `1e-6` **absolute** output gate is unattainable, per objective

| Regime | Objective | What happens | Abs err | Rel err |
|---|---|---|---|---|
| Large hazard ratio | `survival:cox` | output `~1e8` in float32; 1 ULP there is `~8` | up to **`29.75`** at ordinary settings, **`6.96e+23`** at `eta=1.0` | `5.5e-08` – `5.9e-08` |
| Overflow | `survival:cox` | XGBoost `+inf` (734/2500 rows), float64 transform finite `4.66e+48` | **`+inf`** | n/a |
| Subnormal output | `survival:cox` | XGBoost float32 subnormal loses mantissa bits | `1.18e-37` (PASS) | **`5.66e-05`** |
| Clamp floor | `binary:logistic` | XGBoost `3.0066e-39`, float64 transform `1.5e-89` | `3.0e-39` (PASS) | **`1.0`** |

**Top-line finding, stated loudly as the brief requires:** the output-level absolute
error against XGBoost **exceeds `1e-6` for `survival:cox` at every `base_score` value
tested, by up to 23 orders of magnitude, and is `+inf` in one configuration.**

This is not a bug in the reference walk — the margin is `0.0` bit-exact in the very same
runs. It is what "absolute error `≤ 1e-6`" means when applied to an output whose natural
scale is `1e-16` to `1e+38`. The mirror-image problem appears for `binary:logistic` in
the clamp tail, where absolute error passes trivially (`3e-39`) while relative error is
`1.0` — a value the wrong side of every decision boundary anyone would draw on it.

**This probe does not recommend a tolerance** (out of scope, D030). It reports that a
single absolute number cannot serve as an output-level gate across these three
objectives, and leaves that to a decision. See the DECISION NEEDED block at the end.

---

## 7. `numpy` float64 `sigmoid` and `exp` against a 50-digit reference

Reference: `mpmath` at `mp.mp.dps = 50` (`mp.mp.prec = 169` bits), rounded once to the
target format. ULP error is the distance in representable values. **MAX** reported, never
mean, per FORMAT.md §5.5.

```
$ uv run python s10_ulp_clean.py
mp.mp.dps=50  mp.mp.prec=169 bits

sigmoid 1.0/(1.0+np.exp(-m))   region m >= -709.7   n=85012
  ULP histogram (float64, vs mpmath 50 dps): {0: 59491, 1: 25323, 2: 198}
  MAX ULP = 2    fraction not correctly rounded = 25521/85012
    ulp=     2  m=-7.67799197385255e-07    numpy=0.49999980805020056        mpmath=0.49999980805020067
    ulp=     2  m=-1.2003467034237935e-06  numpy=0.49999969991332405        mpmath=0.49999969991332416
    ulp=     2  m=-1.7647439884140192e-06  numpy=0.499999558814003          mpmath=0.4999995588140029
    ulp=     2  m=-2.389825419470992e-06   numpy=0.4999994025436452         mpmath=0.4999994025436451

exp np.exp(m)   full float64 range   n=95011
  ULP histogram (float64, vs mpmath 50 dps): {0: 94853, 1: 158}
  MAX ULP = 1    fraction not correctly rounded = 158/95011
    ulp=     1  m=685.3243451377324        numpy=4.2912261952623654e+297    mpmath=4.291226195262365e+297
    ulp=     1  m=645.4491185338671        numpy=2.0653351129240882e+280    mpmath=2.0653351129240886e+280
    ulp=     1  m=606.4015683777766        numpy=2.2743377709654457e+263    mpmath=2.274337770965446e+263
```

`math.exp` is bit-identical to `np.exp` on the same grid:

```
$ uv run python s09_ulp.py
math.exp vs mpmath, same exp grid
  ULP histogram: {0: 94854, 1: 158}
  MAX ULP = 1 at m=-12.19021075264093 math=5.07994180249677e-06 mpmath=5.0799418024967705e-06
```

| Function, float64 | MAX ULP vs `mpmath` 50 dps | Not correctly rounded |
|---|---|---|
| `np.exp(m)` | **1** | 158/95011 = 0.17% |
| `math.exp(m)` | **1** | 158/95011 = 0.17% |
| `1.0/(1.0+np.exp(-m))`, `m ≥ -709.7` | **2** | 25521/85012 = **30.0%** |

The sigmoid's 2 ULP is the composition, not the `exp`: one rounding in `exp`, one in the
add, one in the divide. **This is the number that calibrates "1–2 ULP" for D030's bundled
sigmoid**: `libm`-based float64 `1/(1+exp(-m))` is itself up to 2 ULP off correctly
rounded and wrong on 30% of inputs. A bundled implementation at ≤2 ULP is not worse than
the platform composition it replaces.

### The naive float64 sigmoid formula has a hard range failure, not just a rounding error

This is separate from the ULP figures and is why the sigmoid grid above stops at
`-709.7`. `1.0/(1.0+exp(-m))` in float64 **returns exactly `0.0`** once `exp(-m)`
overflows, while the true value is still a representable float64 subnormal:

```
the naive float64 sigmoid formula's overflow cliff, exact boundary:
  m=-709.0                 exp(-m)=8.218407461554972e+307     naive=1.216780750623423e-308 true(f64)=1.216780750623423e-308  agree=True
  m=-709.7                 exp(-m)=1.6549840276802644e+308    naive=6.04235438695844e-309 true(f64)=6.04235438695844e-309  agree=True
  m=-709.7827128933839     exp(-m)=1.7976931348620688e+308    naive=5.56268464626877e-309 true(f64)=5.56268464626877e-309  agree=True
  m=-709.7827128933841     exp(-m)=inf                        naive=0.0      true(f64)=5.562684646267504e-309  agree=False
  m=-709.79                exp(-m)=inf                        naive=0.0      true(f64)=5.52229610670219e-309  agree=False
  m=-710.0                 exp(-m)=inf                        naive=0.0      true(f64)=4.47628622567513e-309  agree=False
  m=-745.0                 exp(-m)=inf                        naive=0.0      true(f64)=5e-324  agree=False
  m=-745.1332191019411     exp(-m)=inf                        naive=0.0      true(f64)=5e-324  agree=False
  m=-745.2                 exp(-m)=inf                        naive=0.0      true(f64)=0.0  agree=True
  m=-800.0                 exp(-m)=inf                        naive=0.0      true(f64)=0.0  agree=True
  sampled 3000 points in [-745.0, -709.79]: naive returns exactly 0.0 on 3000/3000, true value is nonzero on 3000/3000
    max |naive - true| over that band = 5.3635618960223e-309
```

Boundary: `m = -709.7827128933839` is the last float64 the naive formula gets right;
`m = -709.7827128933841` is the first it gets wrong. The band where it is wrong is
`[-745.1332191019411, -709.7827128933841]` — **3000/3000** sampled points return `0.0`
where the true value is nonzero. Relative error is `1.0` throughout that band.

Margins in that band are reachable: this probe measured a `binary:logistic` margin of
`-748.246337890625`. Directly relevant to D030's mandated validation list, which already
names the subnormal transition and the saturation boundaries.

### float32 side, for reference

XGBoost's transform was measured to be float32 (§2), so this calibrates the thing being
compared against:

```
  numpy float32 exp vs platform libm expf (ctypes): bit-exact 50008/50008
  numpy float32 exp   ULP histogram vs mpmath: {0: 49686, 1: 322}
    MAX = 1 at m=75.54653930664062 numpy=6.448298520364634e+32 mpmath=6.448298907220896e+32
  platform expf       ULP histogram vs mpmath: {0: 49686, 1: 322}
    MAX = 1 at m=75.54653930664062 expf=6.448298520364634e+32 mpmath=6.448298907220896e+32
```

Max 1 ULP, 322/50008 not correctly rounded (0.64%).

---

## 8. What `predict()` returns

```
$ uv run python s11_consolidate.py
what predict() returns: shape / dtype / call-path agreement
  reg:squarederror
    predict(dm)                  type=ndarray dtype=float32 shape=(2500,)
    predict(dm, strict_shape)    type=ndarray dtype=float32 shape=(2500, 1)
    predict(dm, output_margin)   type=ndarray dtype=float32 shape=(2500,)
    inplace_predict(x)           type=ndarray dtype=float32 shape=(2500,)
    predict(dm) vs inplace_predict(x) bit-exact 2500/2500
    predict(dm) vs strict_shape.ravel() bit-exact 2500/2500
  binary:logistic
    predict(dm)                  type=ndarray dtype=float32 shape=(2500,)
    predict(dm, strict_shape)    type=ndarray dtype=float32 shape=(2500, 1)
    predict(dm, output_margin)   type=ndarray dtype=float32 shape=(2500,)
    inplace_predict(x)           type=ndarray dtype=float32 shape=(2500,)
    predict(dm) vs inplace_predict(x) bit-exact 2500/2500
    predict(dm) vs strict_shape.ravel() bit-exact 2500/2500
  survival:cox
    predict(dm)                  type=ndarray dtype=float32 shape=(2500,)
    predict(dm, strict_shape)    type=ndarray dtype=float32 shape=(2500, 1)
    predict(dm, output_margin)   type=ndarray dtype=float32 shape=(2500,)
    inplace_predict(x)           type=ndarray dtype=float32 shape=(2500,)
    predict(dm) vs inplace_predict(x) bit-exact 2500/2500
    predict(dm) vs strict_shape.ravel() bit-exact 2500/2500
```

- **All three objectives: plain 1-D `numpy.ndarray`, `dtype=float32`, shape `(N,)`,
  C-contiguous.** No structured type, no object array, no tuple, no list of classes.
- `strict_shape=True` returns `(N, 1)` — 2-D, bit-identical after `ravel()`.
- `inplace_predict` is bit-identical to `predict(DMatrix)` on all three (finite inputs;
  `probes/tree_structure.md` §6.1 already records that they differ on `±inf`, which this
  probe did not re-test).
- `predict()` returning `float32` means **the float32-vs-float64 question in §2 is about
  the internal computation only.** Even a float64-internal transform would be narrowed to
  float32 on the way out.

---

## 9. `survival:cox` — is `exp` the whole story?

| Property | Measured across 18 Cox configurations |
|---|---|
| `predict() == float32(expf(float32(margin)))` | **45000/45000 bit-exact** |
| Ever negative | **`n_negative=0`** in every configuration |
| Ever exactly `0.0` | **`n_exact_zero=0`** in every configuration |
| Ever `NaN` | **`n_nan=0`** in every configuration |
| Ever `+inf` | **Yes** — 358, 734, 1329 rows in three configurations, for margins > ≈`88.72` |
| Subnormal | **Yes** — 317, 467 rows, min observed `8.025236305188227e-42` |
| Any clamp | **None found.** §4 |

```
$ uv run python s11_consolidate.py
survival:cox   winning hypothesis: f32 exp
  bs=None         rounds=   0 eta=0.3  trees=0  margin[0.5,0.5]  predict[1.64872,1.64872]  bit-exact 2500/2500
  bs=None         rounds=   1 eta=0.3  trees=1  margin[-0.198194,3.95926]  predict[0.82021,52.4184]  bit-exact 2500/2500
  bs=None         rounds= 200 eta=0.3  trees=200  margin[-8.77394,18.3896]  predict[0.000154713,9.69398e+07]  bit-exact 2500/2500
  bs=None         rounds= 300 eta=1.0  trees=300  margin[-0.792413,134.943]  predict[0.452751,inf]  bit-exact 2500/2500
  bs=0.5          rounds= 300 eta=1.0  trees=300  margin[-1.48546,212.83]  predict[0.226398,inf]  bit-exact 2500/2500
  bs=7.5          rounds= 400 eta=1.0  trees=400  margin[1.04443,120.702]  predict[2.84177,inf]  bit-exact 2500/2500
  bs=100.0        rounds= 200 eta=0.3  trees=200  margin[-4.27122,23.057]  predict[0.0139647,1.03168e+10]  bit-exact 2500/2500
  bs=1e-38        rounds= 300 eta=1.0  trees=300  margin[-88.2962,159.824]  predict[4.50235e-39,inf]  bit-exact 2500/2500
  TOTAL 45000/45000
```

**`exp` is the whole story: nothing else is applied.** `predict()` is exactly
`expf(margin)` on 45000/45000 rows, with no clamp, no floor, no ceiling, and no
post-processing. Strict positivity is a *consequence* of `exp`, not an extra step —
`exp` of a finite float32 is never negative and only underflows to `0.0` below `-104`,
which no fitted model here reached (§4).

**Whether that value is called a "hazard ratio" is nomenclature this probe cannot
measure.** What is measured: it is `exp(margin)`, strictly positive, unbounded above, and
its ratio between two rows is `exp(margin_a − margin_b)`. Any interpretation beyond that
identity is not evidence produced here.

---

## 10. Out of scope, and it looked wrong — `binary:logistic` clamps `base_score`

**This is not the question I was assigned, and it is the loudest thing this probe found.**
It surfaced because the brief required sweeping `base_score` far from `0.5`.

`binary:logistic` clamps `base_score` to `[float32(1e-6), float32(1) − float32(1e-6)]`
when deriving the margin intercept, and **stores the unclamped value**. The
`probes/base_score.md` §9 / D015 recipe therefore returns the wrong intercept outside
that window, with no error.

```
$ uv run python s15_bs_clamp.py
clamp bounds under test: float32(1e-6)=9.999999974752427e-07  float32(1)-float32(1e-6)=0.9999989867210388

           arg           stored         recipe(stored)        recipe(clamped)    walk bits  clamped bits
         1e-38        '[1E-38]'     -87.49822998046875    -13.815509796142578       0/2500     2500/2500
         1e-12        '[1E-12]'     -27.63102149963379    -13.815509796142578       0/2500     2500/2500
         1e-08         '[1E-8]'     -18.42068099975586    -13.815509796142578       0/2500     2500/2500
         1e-07         '[1E-7]'     -16.11809539794922    -13.815509796142578       0/2500     2500/2500
       9.9e-07       '[9.9E-7]'    -13.825559616088867    -13.815509796142578       0/2500     2500/2500
         1e-06         '[1E-6]'    -13.815509796142578    -13.815509796142578    2500/2500     2500/2500
       1.1e-06       '[1.1E-6]'    -13.720199584960938    -13.720199584960938    2500/2500     2500/2500
         1e-05         '[1E-5]'     -11.51291561126709     -11.51291561126709    2500/2500     2500/2500
           0.5         '[5E-1]'                   -0.0                   -0.0    2500/2500     2500/2500
      0.999999   '[9.99999E-1]'     13.745160102844238     13.745160102844238    2500/2500     2500/2500
     0.9999991  '[9.999991E-1]'     13.862943649291992     13.745160102844238       0/2500     2500/2500
     0.9999995  '[9.999995E-1]'     14.556090354919434     13.745160102844238       0/2500     2500/2500
     0.9999999  '[9.999999E-1]'     15.942384719848633     13.745160102844238       0/2500     2500/2500
  0.9999999999          '[1E0]'                   None     13.745160102844238 n/a (domain)     2500/2500
```

The clamped recipe is `2500/2500` on **all 14** values, including the seven where the
plain recipe is `0/2500`. Error magnitudes at the margin:

```
$ uv run python s14_extreme_bs.py
  bs=1e-38            rounds=  1 stored='[1E-38]'    recipe_intercept=-87.49822998046875   bit-exact 0/2500 max_abs_err=73.68272018432617   <== RECIPE FAILS
  bs=1e-12            rounds=  1 stored='[1E-12]'    recipe_intercept=-27.63102149963379   bit-exact 0/2500 max_abs_err=13.815511703491211   <== RECIPE FAILS
  bs=1e-07            rounds=  1 stored='[1E-7]'     recipe_intercept=-16.11809539794922   bit-exact 0/2500 max_abs_err=2.3025856018066406   <== RECIPE FAILS
  bs=1e-06            rounds=  1 stored='[1E-6]'     recipe_intercept=-13.815509796142578  bit-exact 2500/2500 max_abs_err=0.0
  bs=0.5              rounds=  1 stored='[5E-1]'     recipe_intercept=-0.0                 bit-exact 2500/2500 max_abs_err=0.0
  bs=0.999999         rounds=  1 stored='[9.99999E-1]' recipe_intercept=13.745160102844238 bit-exact 2500/2500 max_abs_err=0.0
  bs=0.9999995        rounds=  1 stored='[9.999995E-1]' recipe_intercept=14.556090354919434 bit-exact 0/2500 max_abs_err=0.8109302520751953   <== RECIPE FAILS
  bs=0.9999999        rounds=  1 stored='[9.999999E-1]' recipe_intercept=15.942384719848633 bit-exact 0/2500 max_abs_err=2.1972246170043945   <== RECIPE FAILS
```

Margin error **`73.68`** at `base_score=1e-38`, `13.82` at `1e-12`, `2.30` at `1e-7`,
`2.20` at `0.9999999`. Seven orders of magnitude past the `1e-6` gate. And it is not a
crash — `predict()` returns a perfectly plausible probability.

Two distinct failure modes for an exporter that derives the intercept from stored
`base_score` per D015:

- **Silent wrong number:** `base_score` in `(0, 1e-6)` or in `(1−1e-6, 1)` that still
  rounds to a float32 other than `1.0`. The derived intercept is wrong by up to `73.68`
  and export's own agreement assertion cannot see it, because both sides of that
  assertion use the same wrong recipe.
- **Loud failure:** `base_score = 1 − 1e-10` stores as `'[1E0]'`, so
  `float32(1/1.0) − 1 = 0.0` and `−log(0)` is a **domain error** (`ValueError: math
  domain error`, which is exactly how my first sweep script died). A crash is the
  acceptable outcome, but it happens for a *legal* `base_score` value.

The other two objectives are **not** clamped, measured over 12 more orders of magnitude:

```
survival:cox, same question -- is base_score clamped there?
  bs=1e-38        stored='[1E-38]'   recipe_intercept=-87.49822998046875     bit-exact 2500/2500
  bs=1e-12        stored='[1E-12]'   recipe_intercept=-27.63102149963379     bit-exact 2500/2500
  bs=1e-06        stored='[1E-6]'    recipe_intercept=-13.815510749816895    bit-exact 2500/2500
  bs=0.25         stored='[2.5E-1]'  recipe_intercept=-1.3862943649291992    bit-exact 2500/2500
  bs=1.0          stored='[1E0]'     recipe_intercept=0.0                    bit-exact 2500/2500
  bs=7.5          stored='[7.5E0]'   recipe_intercept=2.0149030685424805     bit-exact 2500/2500
  bs=1000000.0    stored='[1E6]'     recipe_intercept=13.815510749816895     bit-exact 2500/2500
  bs=1e+30        stored='[1E30]'    recipe_intercept=69.07755279541016      bit-exact 2500/2500
  bs=1e+38        stored='[1E38]'    recipe_intercept=87.49822998046875      bit-exact 2500/2500

reg:squarederror, same question
  bs=-1e+30       stored='[-1E30]'   recipe_intercept=-1.0000000150474662e+30 bit-exact 2500/2500
  bs=-1000000.0   stored='[-1E6]'    recipe_intercept=-1000000.0             bit-exact 2500/2500
  bs=0.0          stored='[0E0]'     recipe_intercept=0.0                    bit-exact 2500/2500
  bs=1e+30        stored='[1E30]'    recipe_intercept=1.0000000150474662e+30 bit-exact 2500/2500
  bs=1e+38        stored='[1E38]'    recipe_intercept=9.999999680285692e+37  bit-exact 2500/2500
```

**Ambiguity A3.** The exact bounds are bracketed, not pinned to a single float32:

- Lower bound is in `(float32(9.9e-7), float32(1e-6)]` — `9.9e-7` clamps, `1e-6` does not.
- Upper bound is in `[float32(0.999999), float32(0.9999991))` — `0.999999` (float32
  `0.9999989867210388`) does not clamp, `0.9999991` does.

Both brackets are **consistent with** bounds of `float32(1e-6)` and
`float32(1) − float32(1e-6) = 0.9999989867210388`, and the observed clamped intercepts
`-13.815509796142578` and `13.745160102844238` are exactly the recipe applied to those
two values. A second reading — that the constant is some other value inside each bracket
— is not excluded by these measurements. Nailing it would need a bit-level bisection
sweep of `base_score`, which this probe did not run because it is outside its question.

---

## 11. Consolidated confirmation table

```
$ uv run python s11_consolidate.py
==============================================================================
TOTALS
  reg:squarederror     27500/27500
  binary:logistic      35000/35000
  survival:cox         45000/45000
```

Winning hypotheses, exactly as coded:

```
reg:squarederror  out = margin_f32                                          (identity)
binary:logistic   out = f32( 1 / (1 + expf( -max(margin_f32, f32(-88.7)) )) )
survival:cox      out = f32( expf(margin_f32) )
```

Coverage of those 43 models: tree counts `0, 1, 8, 200, 300, 400, 600`; `eta` `0.3, 1.0`;
`base_score` estimated-default plus `0.05, 0.25, 0.5, 0.95, 0.987654, 3.1415927, 7.5,
100.0, 1e-12, 1e-38`; margins from `-748.25` to `+386.64` (logistic) and `-94.63` to
`+212.83` (Cox); outputs including `+inf`, exactly `1.0`, float32 subnormals, and the
clamp floor.

Two degenerate shapes worth pointing at, both from that run:

```
  bs=None         rounds=   0 eta=0.3  trees=0  margin[0.5,0.5]  predict[0.5,0.5]  bit-exact 2500/2500          (reg:squarederror)
  bs=None         rounds=   0 eta=0.3  trees=0  margin[0.5,0.5]  predict[0.622459,0.622459]  bit-exact 2500/2500  (binary:logistic)
  bs=None         rounds=   0 eta=0.3  trees=0  margin[0.5,0.5]  predict[1.64872,1.64872]  bit-exact 2500/2500    (survival:cox)
```

At **0 boosting rounds with `base_score` unset**, `probes/base_score.md` §7 records that
the *intercept* transform is not applied — the margin is the raw stored `0.5`. This probe
adds: the **output** transform *is* still applied to that margin.
`sigmoid(0.5) = 0.622459` and `exp(0.5) = 1.64872`. **The two transforms are independent
and the §7 trap does not extend to the output transform.** That is a direct measurement
in support of FORMAT.md §5.7 keeping them separate.

Determinism of the measurements themselves:

```
$ uv run python s13_misc.py
refit determinism: same params twice -> identical predict()?
  reg:squarederror     predict bit-exact 2500/2500   margin bit-exact 2500/2500
  binary:logistic      predict bit-exact 2500/2500   margin bit-exact 2500/2500
  survival:cox         predict bit-exact 2500/2500   margin bit-exact 2500/2500
```

---

## 12. Ambiguity, presented rather than resolved

| # | Ambiguity | Both readings | What would settle it |
|---|---|---|---|
| A1 | The `binary:logistic` clamp | (a) input clamped to `≥ float32(-88.7)`; (b) output floored at `1/(1+expf(88.7))` | Source read. Observationally identical because float32 sigmoid is monotone. The *observable* is identical under both. |
| A2 | Upper clamp for `binary:logistic` | (a) clamped at `+88.7`; (b) not clamped above | Not decidable from `predict()`: both give exactly `1.0`. Numerically irrelevant. |
| A3 | Exact `base_score` clamp bounds (§10) | (a) exactly `float32(1e-6)` and `float32(1)−float32(1e-6)`; (b) some other constant inside the measured brackets | A bit-level bisection sweep of `base_score`. Not run — outside this probe's question. |
| A4 | Whether XGBoost calls the platform `expf` | (a) it does; (b) it uses an independent float32 `exp` that happens to agree bit-for-bit on all 50008 sampled points, including where `expf` is 1 ULP wrong | Symbol trace or source read. Reading (a) is **INFERRED**, not measured. |

---

## 13. Not measured — stated so nothing here is over-read

- **XGBoost versions other than 3.3.0.** Everything above is 3.3.0. The `-88.7` clamp in
  particular is a magic constant and is exactly the kind of thing
  `probes/version_drift.md` §3 shows can move between versions. Not checked.
- **Platforms other than macOS 26.5.2 / arm64.** The `expf` agreement in §2 and the ULP
  histograms in §7 are single-platform measurements. FORMAT.md §5.3 already records why
  a single platform pair does not bound another.
- **A Cox `predict()` of exactly `0.0`.** Not produced. Believed reachable; see §4.
- **`binary:logistic` `predict()` of exactly `0.0`.** Not produced in any configuration,
  and §3 makes it look unreachable while the clamp is in place. Not proven unreachable.
- **GPU / `device="cuda"` prediction.** Not tested. The transform is a separate kernel
  there; nothing here transfers.
- **Objectives outside the three in scope.** No behaviour above should be extended by
  analogy to `reg:logistic`, `count:poisson`, `survival:aft`, or `rank:*`.
- **`num_class` for binary classification** (gap G3). Not this probe's question; not
  investigated.
- **Cross-language `libm` divergence.** Settled by D030; explicitly out of scope; no
  JavaScript was written and `Math.exp` was not tested.
- **Multi-output / multi-target.** Out of scope per D017.

---

## 14. Decisions needed

```
DECISION NEEDED: The output-level 1e-6 ABSOLUTE gate is unattainable for survival:cox
Context:  The margin gate is met perfectly -- 0.0 bit-exact, 2500/2500, at every
          base_score value for all three objectives. But survival:cox's output is an
          unbounded exp(margin): measured predict() up to 1.03e+10 at ordinary settings
          and +inf on 734/2500 rows at eta=1.0. A float64 transform of the identical
          float32 margin diverges from XGBoost's float32 transform by up to 29.75 in
          absolute terms at eta=0.3 (max_rel_err 5.89e-08), 6.96e+23 at eta=1.0, and
          +inf where XGBoost overflows and float64 does not. The mirror problem exists
          for binary:logistic: in the clamp tail the ABSOLUTE error is 3.0e-39 (passes
          trivially) while the RELATIVE error is 1.0. Neither objective is broken; the
          single absolute number is the wrong instrument for both.
Options:  A) Keep 1e-6 absolute, and scope it to the MARGIN only. Output-vs-XGBoost
             becomes a relative check. Measured relative error is 5.7e-08 to 5.9e-08
             for Cox and ~1.1e-07 for logistic -- both comfortably inside 1e-6 relative.
             Does not cover the clamp tail or the +inf rows.
          B) Keep 1e-6 absolute but per-objective: absolute for identity and sigmoid,
             relative for exp. Two rules to state, and someone will apply the wrong one.
          C) Compare in ULP of the float32 output. Scale-free, covers subnormals, and
             covers +inf if inf==inf counts as agreement. Needs a ULP helper in the
             harness and a decision on what to do at the clamp boundary.
Lean:     No lean on the instrument -- that is a gate design decision, not a
          measurement. But whichever is chosen MUST say what happens on the +inf rows
          and on the clamp-floor rows, because those are the two places where every
          candidate instrument returns something degenerate, and a gate that silently
          skips them is worse than no gate.
Blocks:   Phase 6/7 output-level verification for survival:cox and for the
          binary:logistic saturation fixtures. Does NOT block the margin gate, which is
          met at exactly 0.0.
```

```
DECISION NEEDED: Does the predictor reproduce XGBoost's -88.7 sigmoid clamp?
Context:  Measured: binary:logistic predict() clamps the margin below at
          float32(-88.7) = -88.69999694824219, so every margin below that returns
          exactly 3.006635794144578e-39 (bits 2145607) and NEVER 0.0. Confirmed on
          1959/1959, 97/97, and 1/1 below-clamp rows across three configurations;
          the unclamped form scores 0/1959 on those rows. Margins down to -748.25
          were produced by an ordinary fit (base_score=0.987654, eta=1.0, 300 rounds).
          survival:cox has NO analogous clamp -- confirmed at margin -94.63, where
          predict() is expf(-94.63)=8.03e-42, not the clamp value. FORMAT.md section
          5.1 currently specifies an unclamped float64 sigmoid, which returns 1.5e-89
          on such a row: relative error 1.0 against XGBoost, absolute error 3.0e-39.
Options:  A) Reproduce the clamp. Output matches XGBoost's saturation behaviour on the
             tail. Bakes an XGBoost magic constant into the format's semantics, and it
             is version-fragile in exactly the way version_drift.md warns about.
          B) Do not reproduce it. FORMAT.md section 5.1 stands as written. The library
             returns the mathematically better answer (1.5e-89 is closer to the true
             sigmoid than 3.0e-39 is) and diverges from XGBoost by relative 1.0 on
             saturated rows. Must be documented, or it reads as a bug forever.
          C) Refuse: export raises if any training-set margin falls below the clamp.
             Rejects a legitimately-fitted model, which section 14/G3 already flags as
             the worst failure direction.
Lean:     B, weakly, on the D026/D030 principle that cross-language reproducibility and
             a defensible transform beat matching XGBoost bit-for-bit at the output
             stage -- and because a probability of 3e-39 versus 1e-89 changes no
             decision anyone makes. But this is a spec change either way: FORMAT.md
             section 5.1's "Consequence, accepted deliberately" paragraph says the
             difference is "bounded by the 1e-6 gate", and on these rows the RELATIVE
             difference is 1.0, so the sentence as written is no longer accurate.
Blocks:   FORMAT.md section 5.1 wording; the saturation fixtures D030 mandates
          ("margins large enough to saturate sigmoid at exactly 0 and exactly 1" --
          measured: exactly 1 is reachable, 288/2500 rows; exactly 0 is NOT reachable
          through XGBoost, because the clamp prevents it).
```

```
DECISION NEEDED: binary:logistic clamps base_score but stores it unclamped -- D015 recipe
                 is wrong outside [1e-6, 1-1e-6]
Context:  Not this probe's assigned question; found while sweeping base_score. XGBoost
          clamps base_score to approximately [float32(1e-6), float32(1)-float32(1e-6)]
          when deriving the logistic intercept, and stores the UNCLAMPED value. The
          probes/base_score.md section 9 recipe applied to the stored value scores
          0/2500 on 7 of 14 tested values, with margin error 73.68 at base_score=1e-38,
          13.82 at 1e-12, 2.30 at 1e-7, 2.20 at 0.9999999. Applying the recipe to the
          clamped value scores 2500/2500 on all 14. survival:cox and reg:squarederror
          are NOT clamped (verified 1e-38 to 1e38, and -1e30 to 1e38). Separately,
          base_score=1-1e-10 stores as '[1E0]', so float32(1/p)-1 is 0.0 and -log(0)
          raises ValueError: math domain error.
          This directly affects the D015 export-time intercept derivation, and export's
          own "intercept agrees with transform of provenance.base_score" assertion
          CANNOT catch it, because both sides of that assertion use the same recipe.
Options:  A) Add the clamp to the exporter's logistic intercept derivation, and pin the
             constants with a fixture at 1e-7 and 0.9999999. Needs the exact bounds,
             which this probe only bracketed (ambiguity A3).
          B) Make export raise if base_score falls outside the measured safe window
             [1e-6, 1-1e-6]. Loud, cheap, and the values are pathological in practice.
             Rejects a model XGBoost accepts.
          C) Stop deriving the intercept from base_score at all: read the actual margin
             of the fitted model on a probe row and back out the intercept. Immune to
             every clamp, present and future. Changes what export actually does.
Lean:     A follow-up probe first -- the exact bounds are ambiguous (A3) and option A
          cannot be implemented without them. Between A and B: B is safe today and A is
          correct; B does not preclude A later. C is worth considering on its own
          merits, since it removes a whole class of this failure.
Blocks:   The D015 export-time intercept derivation and its self-check. Not blocking
          for base_score in [1e-6, 1-1e-6], which is every realistic value and every
          value the existing fixture plan uses.
```

---

## 15. Reproducing this probe

Scripts, in the scratch directory, none in the repository:

| Script | What it establishes |
|---|---|
| `common.py` | data generation, fit helper, `base_score` parse, section 9 intercept, vectorized reference walk |
| `s02_candidates.py` | §1 candidate residuals, absolute; `predict()` dtype/shape |
| `s16_rel_candidates.py` | §1 candidate residuals, relative |
| `s03_precision.py` | §2 float32 vs float64, six configs, disagreement subsets |
| `s05_saturation.py` | §3 the one anomalous row |
| `s06_clamp.py` | §3 exhaustive float32 scan identifying `float32(-88.7)`; §4 Cox `+inf` |
| `s07_clamp_confirm.py` | §3 clamp confirmed on 2056 below-clamp rows; §4 both Cox tails |
| `s08_cox_tail.py` | §4 Cox lower tail, subnormals, no exact zero |
| `s04_sweep.py` | §5 `base_score` sweep, margin and output error per value |
| `s12_divergence.py` | §6 output-level divergence including `+inf` and clamp rows |
| `s09_ulp.py`, `s10_ulp_clean.py` | §7 ULP vs `mpmath` 50 dps; float64 sigmoid overflow cliff; float32 side |
| `s11_consolidate.py` | §8, §11 consolidated 107500/107500; return shape |
| `s13_misc.py` | §11 refit determinism; stored `base_score` at extreme values |
| `s14_extreme_bs.py`, `s15_bs_clamp.py` | §10 the `base_score` clamp, characterized |

Fixed inputs throughout: `numpy.random.default_rng(20260802)` for features,
`default_rng(20260803)` for labels, `seed=20260802`, `nthread=1`, `tree_method=exact`,
2500 rows × 6 columns, feature names `f0`…`f5`.
