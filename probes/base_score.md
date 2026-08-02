# Probe — `base_score`: storage space and margin-intercept transform

Empirical findings on what space `base_score` is stored in inside XGBoost's serialized
model, for each of the three in-scope objectives, and exactly what transform recovers
the margin intercept.

Every claim below is backed by a pasted command and its real output. Anything not
directly measured is labelled **INFERRED**. Three items are surfaced as open decisions
rather than resolved.

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
```

Matches the D001 reference pin. All findings below are for `xgboost 3.3.0` exactly.

---

## Verdicts up front

| Objective | Stored space | Transform to margin intercept | Discrimination |
|---|---|---|---|
| `reg:squarederror` | **margin space (= label space)** | **identity** | 13/13 bit-exact; `ln` off by up to 1227 |
| `binary:logistic` | **probability space** | **logit** — but *not* the textbook formula, see §5 | 27/27 bit-exact; `identity` off by up to 4.6 |
| `survival:cox` | **hazard-ratio space** | **`ln`** | 17/17 bit-exact; `identity` off by up to 118.6 |

| Hypothesis | Verdict |
|---|---|
| **H1** — `reg:squarederror`: `boost_from_average` can overwrite a user-supplied `base_score` at fit time | **REPRODUCED, BUT ONLY UNDER A CONDITION THE HYPOTHESIS DOES NOT STATE.** Supplying `base_score` *by itself* is safe — XGBoost silently flips `boost_from_average` to `0` and the explicit value survives verbatim. The overwrite happens only when the caller forces `boost_from_average=1` *alongside* an explicit `base_score`. See §6. **And it is not regression-specific:** the identical overwrite occurs for `binary:logistic` and `survival:cox`. |
| **H2** — `binary:logistic`: stored in probability space, intercept is `logit(base_score)` | **REPRODUCED as to the space. THE TRANSFORM IS ONLY APPROXIMATELY RIGHT.** The textbook `log(p/(1-p))` is bit-wrong on **16 of 27** measured values and exceeds the ≤`1e-6` margin gate on 2 of them (worst `6.198883056640625e-06`). XGBoost computes the logit through a **float32 `1/p - 1` intermediate**. See §5 — this is the most consequential finding in this probe. |
| **H3** — Cox/survival requires `ln(base_score)` | **REPRODUCED.** 17/17 bit-exact. One qualification: the input must be snapped to float32 *before* the `log`, or 2 of 17 values come out 1 ULP wrong. |

---

## Method

Synthetic data only: 200 rows x 4 columns from `numpy.random.default_rng(20260801)`,
generic feature names `f0`..`f3`, `max_depth=3`, `eta=0.3`, `nthread=1`. Cox labels use
the sign convention (positive = event, negative = right-censored). Scripts and fitted
binaries lived entirely outside the repository; nothing was written into the tree.

The measurement of the intercept is deliberately **leaf-free**. Two earlier approaches
were tried and both proved confounded:

- **Summing per-tree leaf contributions and taking the residual** — the leaf values
  returned by `get_dump()` are rounded decimals, so a margin rebuilt from them is not
  bit-exact even with the correct intercept. This produced *contradictory* per-value
  results (float64 logit won at `p=0.3`, float32 logit won at `p=0.7`), which is the
  signature of a confounded measurement rather than a real finding.
- **Slicing a fitted booster to zero trees** — `booster[0:0]` does not do this. See §8.

The method that works, and that every number in §3–§5 comes from:

1. Fit *N*>0 rounds and read the serialized `base_score`.
2. Fit a **second** model with **0 rounds** and `base_score` pinned to exactly that
   value. §6 establishes that an explicitly-supplied `base_score` is preserved verbatim,
   so this model's margin is a constant equal to `transform(base_score)` — exactly, in
   float32, with no leaf value anywhere in the chain.
3. Compare **float32 bit patterns**, not residuals. A hit count of 27/27 is evidence; a
   residual of `1e-7` is not.

**Precision used:** all comparisons are on `np.float32` bit patterns
(`np.float32(x).view(np.uint32)`). Where a candidate transform is computed in float64
this is stated explicitly per row. Residuals are reported in float64 for readability
only; the verdicts rest on the bit comparison.

---

## 1. Where `base_score` lives — and it is a bracketed string

Single JSON path, identical across all three objectives:

```
learner.learner_model_param.base_score
```

**Top-line finding: the value is a JSON *string*, and it is *bracketed*.** It is not a
JSON number. A parser that expects a number, or that expects a bare numeric string,
fails on every artifact.

```
$ uv run python probe.py
reg:squarederror  |  tag=default_0trees  |  base_score arg=None  |  rounds=0
saved model file: reg_squarederror_default_0trees.json
JSON path:        learner.learner_model_param.base_score
raw form:         '[5E-1]'   (python type after json.load: str)
file substring:   '"base_score":"[5E-1]","boost_from_average":"1","num_class":"'
parse note:       bracketed 1-vector string; inner token = '5E-1'
```

Raw forms observed, verbatim from the saved files:

| Objective | Condition | Raw serialized form |
|---|---|---|
| `reg:squarederror` | default, 5 rounds | `'[3.9882263E1]'` |
| `reg:squarederror` | `base_score=0.25` | `'[2.5E-1]'` |
| `binary:logistic` | default, 5 rounds | `'[4.8E-1]'` |
| `binary:logistic` | `base_score=0.25` | `'[2.5E-1]'` |
| `survival:cox` | default, 5 rounds | `'[1E0]'` |
| `survival:cox` | `base_score=0.25` | `'[2.5E-1]'` |
| `reg:squarederror` | `base_score=-100.0` | `'[-1E2]'` |
| `reg:squarederror` | `base_score=0.0` | `'[0E0]'` |

Notes on the format, all measured:

- Bracketed, comma-free, single element in every case observed. It reads as a
  1-element vector serialization. **INFERRED:** the bracket is a vector notation that
  would carry more than one element for a multi-output or multi-class model. Not
  measured — multi-class is out of scope per D003. Confirming it would need a
  `multi:softprob` fit, which this probe did not perform.
- Exponent notation is always present and always uppercase `E`, including for
  integral values (`'[1E0]'`, `'[0E0]'`).
- Negative values serialize with a leading minus inside the bracket (`'[-1E2]'`).

### Full `learner_model_param`, all three objectives

```
$ uv run python probe7.py
  reg:squarederror
    learner.learner_model_param = {"base_score": "[3.9882263E1]", "boost_from_average": "1", "num_class": "0", "num_feature": "4", "num_target": "1"}
    learner top-level keys      = ['attributes', 'feature_names', 'feature_types', 'gradient_booster', 'learner_model_param', 'objective']
    doc top-level keys          = ['learner', 'version']
    doc['version']              = [3, 3, 0]
    'intercept' anywhere in file: False
    objective block             = {"name": "reg:squarederror", "reg_loss_param": {"scale_pos_weight": "1"}}

  binary:logistic
    learner.learner_model_param = {"base_score": "[4.8E-1]", "boost_from_average": "1", "num_class": "0", "num_feature": "4", "num_target": "1"}
    'intercept' anywhere in file: False
    objective block             = {"name": "binary:logistic", "reg_loss_param": {"scale_pos_weight": "1"}}

  survival:cox
    learner.learner_model_param = {"base_score": "[1E0]", "boost_from_average": "1", "num_class": "0", "num_feature": "4", "num_target": "1"}
    'intercept' anywhere in file: False
    objective block             = {"name": "survival:cox"}
```

Answers to the two structural questions asked:

- **Does the field name or location differ across objectives?** No. Same key, same path,
  same string-in-brackets encoding for all three. Only the *meaning* of the number
  changes.
- **Is there a separate intercept field distinct from `base_score`?** No. The substring
  `intercept` does not appear anywhere in any saved model file, for any of the three
  objectives. `base_score` is the only intercept carrier.
- `boost_from_average` sits **in the saved model file**, adjacent to `base_score` in
  `learner_model_param` — not only in the config. It is available to an exporter reading
  the artifact.

### Do the several serialized views ever disagree?

They do not. Checked `save_model(.json)`, `save_raw(raw_format="json")`,
`save_config()`, and all of those again after a `load_model` round trip through both
JSON and UBJ, for all three objectives, default and explicit:

```
$ uv run python probe8.py
binary:logistic  base_score arg=None
  save_model(.json) file             = '[4.8E-1]'       -> 0.48    float32=0.47999998927116394
  save_raw(json)                     = '[4.8E-1]'       -> 0.48    float32=0.47999998927116394
  save_config()                      = '[4.8E-1]'       -> 0.48    float32=0.47999998927116394
  reload(json) -> save_raw(json)     = '[4.8E-1]'       -> 0.48    float32=0.47999998927116394
  reload(json) -> save_config()      = '[4.8E-1]'       -> 0.48    float32=0.47999998927116394
  reload(ubj) -> save_raw(json)      = '[4.8E-1]'       -> 0.48    float32=0.47999998927116394
  reload(ubj) -> save_config()       = '[4.8E-1]'       -> 0.48    float32=0.47999998927116394

  distinct raw strings : ['[4.8E-1]']
  distinct float32 bits: [1056293519]
  ALL VIEWS AGREE: strings=True  float32=True
  margin bitwise identical after reload(json): True
  margin bitwise identical after reload(ubj) : True
```

All six objective/condition combinations reported `ALL VIEWS AGREE: strings=True
float32=True` and bitwise-identical margins after both round trips.

`base_score` does **not** appear in `get_dump()` output in either format:

```
$ uv run python probe7.py
  reg:squarederror   dump_format=text  trees=2 contains 'base_score': False  chars=975
  reg:squarederror   dump_format=json  trees=2 contains 'base_score': False  chars=2675
  reg:squarederror   save_config() contains 'base_score': True
  binary:logistic    dump_format=text  trees=2 contains 'base_score': False  chars=975
  binary:logistic    dump_format=json  trees=2 contains 'base_score': False  chars=2675
  binary:logistic    save_config() contains 'base_score': True
  survival:cox       dump_format=text  trees=2 contains 'base_score': False  chars=829
  survival:cox       dump_format=json  trees=2 contains 'base_score': False  chars=2287
  survival:cox       save_config() contains 'base_score': True
```

### Serialization precision: not lossy

The decimal carries enough digits to identify the internal float32 uniquely. Seven
*adjacent* float32 values around `0.48` each produced a **distinct** serialized string —
no collisions:

```
$ uv run python probe5.py
E1  is the serialized decimal lossy? sweep adjacent float32 near 0.48
  k=-3  input float64=0.4799998998641968
        serialized  = '[4.799999E-1]'
  k=-2  input float64=0.47999992966651917
        serialized  = '[4.7999993E-1]'
  k=-1  input float64=0.47999995946884155
        serialized  = '[4.7999996E-1]'
  k=+0  input float64=0.47999998927116394
        serialized  = '[4.8E-1]'
  k=+1  input float64=0.48000001907348633
        serialized  = '[4.8000002E-1]'
  k=+2  input float64=0.4800000488758087
        serialized  = '[4.8000005E-1]'
  k=+3  input float64=0.4800000786781311
        serialized  = '[4.8000008E-1]'

  collisions (one serialized string, several distinct float32 inputs):
    '[4.799999E-1]' <- 1 input (no collision)
    '[4.7999993E-1]' <- 1 input (no collision)
    '[4.7999996E-1]' <- 1 input (no collision)
    '[4.8E-1]' <- 1 input (no collision)
    '[4.8000002E-1]' <- 1 input (no collision)
    '[4.8000005E-1]' <- 1 input (no collision)
    '[4.8000008E-1]' <- 1 input (no collision)
```

And the float32 bit pattern round-trips through the file exactly:

```
E1b  digits carried by the serializer
  input 0.123456789
     serialized      = '[1.2345679E-1]'
     float32(input)  = 0.12345679104328156
     float32(reparse)= 0.12345679104328156
     float32 round-trips through the file? True
  input 0.3333333333333333
     serialized      = '[3.3333334E-1]'
     float32(input)  = 0.3333333432674408
     float32(reparse)= 0.3333333432674408
     float32 round-trips through the file? True
  input 39.88226318359375
     serialized      = '[3.9882263E1]'
     float32(input)  = 39.88226318359375
     float32(reparse)= 39.88226318359375
     float32 round-trips through the file? True
```

Confirmed across all 57 sweep values in §3–§5: `f32 round-trip = True` on every row.
The serialized decimal is the shortest that round-trips in **float32** — consistent with
the threshold-representation invariant, and *not* a float64 value.

**Precision of the round trip: exact at float32, lossy at float64.** `float(stored)` is
`0.48`, whereas the value XGBoost holds is `0.47999998927116394`. Reading the field into
a float64 and using it unsnapped is a float32-discipline violation of exactly the kind
D004 covers, and §5 shows it changes the answer.

---

## 2. Measured margin intercepts, full precision

From the leaf-free method (0-round model, `base_score` pinned to the serialized value):

| Objective | Serialized | Measured margin intercept | float32 bits |
|---|---|---|---|
| `reg:squarederror` (default) | `'[3.9882263E1]'` | `39.88226318359375` | 1109362544 |
| `reg:squarederror` (`=0.25`) | `'[2.5E-1]'` | `0.25` | 1048576000 |
| `binary:logistic` (default) | `'[4.8E-1]'` | `-0.08004285395145416` | 3181636994 |
| `binary:logistic` (`=0.25`) | `'[2.5E-1]'` | `-1.0986123085021973` | 3213664084 |
| `survival:cox` (default) | `'[1E0]'` | `0.0` | 0 |
| `survival:cox` (`=0.25`) | `'[2.5E-1]'` | `-1.3862943649291992` | 3216077336 |

Every one of these is a constant across all 200 rows (`residual spread = 0.0`), because
the model has no trees.

---

## 3. Residual per candidate transform — the winners win by orders of magnitude

Numbers are `abs(float32(candidate) - measured)`. `BIT-EXACT` means the float32 bit
patterns are identical.

### `reg:squarederror`

```
$ uv run python probe4.py
reg:squarederror   base_score arg = None
  after 5 rounds: base_score = '[3.9882263E1]' (str)
                  boost_from_average = '1'
  MEASURED INTERCEPT = 39.88226318359375   float32 bits = 1109362544
    identity (f64)      = 39.882263          f32=39.88226318359375  bits=1109362544 resid=0.0  <== BIT-EXACT
    identity (f32 in)   = 39.88226318359375  f32=39.88226318359375  bits=1109362544 resid=0.0  <== BIT-EXACT
    ln (f64)            = 3.685931688719395  f32=3.68593168258667   bits=1080813134 resid=36.19633150100708
    ln (f32 in)         = 3.685931693322788  f32=3.68593168258667   bits=1080813134 resid=36.19633150100708
  BIT-EXACT transforms: ['identity (f64)', 'identity (f32 in)']
```

`logit` is **undefined** here — the stored value `39.882263` is outside `(0,1)`. That is
itself a signal: a `reg:squarederror` `base_score` is routinely outside the domain of
`logit`, so a mistakenly applied logit would raise or produce NaN rather than a plausible
wrong number. The dangerous direction is the reverse.

With `base_score=0.25` (where all three transforms *are* defined, so the discrimination
is real rather than an artifact of domain):

```
reg:squarederror   base_score arg = 0.25
  MEASURED INTERCEPT = 0.25   float32 bits = 1048576000
    identity (f64)                = 0.25                 bits=1048576000 resid=0.0  <== BIT-EXACT
    identity (f32 in)             = 0.25                 bits=1048576000 resid=0.0  <== BIT-EXACT
    ln (f64)                      = -1.3862943611198906  bits=3216077336 resid=1.6362943649291992
    ln (f32 in)                   = -1.3862943611198906  bits=3216077336 resid=1.6362943649291992
    logit log(p/(1-p)) (f64)      = -1.0986122886681098  bits=3213664084 resid=1.3486123085021973
    logit log(p/(1-p)) (f32 in)   = -1.0986122886681098  bits=3213664084 resid=1.3486123085021973
    logit -log(1/p-1) (f64)       = -1.0986122886681098  bits=3213664084 resid=1.3486123085021973
    logit -log(1/p-1) (f32 in)    = -1.0986122886681098  bits=3213664084 resid=1.3486123085021973
    logit log(p)-log1p(-p) (f64)  = -1.0986122886681096  bits=3213664084 resid=1.3486123085021973
    logit log(p)-log1p(-p) (f32 in)= -1.0986122886681096 bits=3213664084 resid=1.3486123085021973
  BIT-EXACT transforms: ['identity (f64)', 'identity (f32 in)']
```

Identity `0.0` versus `1.35` and `1.64`. The margin is **10 orders**, not a coin flip.

13-value sweep, including negatives and zero:

```
$ uv run python probe6.py
reg:squarederror: sweep of 13 base_score values
                   input       serialized       measured intercept  f32 round-trip
                  -100.0         '[-1E2]'                   -100.0  True
                    -7.5       '[-7.5E0]'                     -7.5  True
                    -1.0         '[-1E0]'                     -1.0  True
                    -0.3        '[-3E-1]'     -0.30000001192092896  True
                     0.0          '[0E0]'                      0.0  True
                    0.25       '[2.5E-1]'                     0.25  True
                     0.3         '[3E-1]'      0.30000001192092896  True
                     0.5         '[5E-1]'                      0.5  True
                     1.0          '[1E0]'                      1.0  True
                     7.5        '[7.5E0]'                      7.5  True
               39.882263  '[3.9882263E1]'        39.88226318359375  True
                   100.0          '[1E2]'                    100.0  True
               1234.5678  '[1.2345677E3]'       1234.5677490234375  True

  bit-exact hit counts over 13 values:
    identity                            13/13   worst abs diff = 0.0  <== ALL
    ln                                   0/13   worst abs diff = 1227.4492726325989
    logit XGB                            0/13   worst abs diff = 1.3486123085021973
```

**Verdict: `reg:squarederror` stores `base_score` in margin space (identical to label
space, since the link is the identity). Intercept = `float32(base_score)`.**
Negative and zero values are accepted and stored, which confirms the space is not
constrained to `(0,1)` or to positives.

### `survival:cox`

```
$ uv run python probe4.py
survival:cox   base_score arg = None
  after 5 rounds: base_score = '[1E0]' (str)
                  boost_from_average = '1'
  MEASURED INTERCEPT = 0.0   float32 bits = 0
    identity (f64)     = 1.0  f32=1.0  bits=1065353216 resid=1.0
    identity (f32 in)  = 1.0  f32=1.0  bits=1065353216 resid=1.0
    ln (f64)           = 0.0  f32=0.0  bits=0           resid=0.0  <== BIT-EXACT
    ln (f32 in)        = 0.0  f32=0.0  bits=0           resid=0.0  <== BIT-EXACT
  BIT-EXACT transforms: ['ln (f64)', 'ln (f32 in)']

survival:cox   base_score arg = 0.25
  MEASURED INTERCEPT = -1.3862943649291992   float32 bits = 3216077336
    identity (f64)                 = 0.25                 bits=1048576000 resid=1.6362943649291992
    ln (f64)                       = -1.3862943611198906  bits=3216077336 resid=0.0  <== BIT-EXACT
    ln (f32 in)                    = -1.3862943611198906  bits=3216077336 resid=0.0  <== BIT-EXACT
    logit log(p/(1-p)) (f64)       = -1.0986122886681098  bits=3213664084 resid=0.28768205642700195
    logit -log(1/p-1) (f64)        = -1.0986122886681098  bits=3213664084 resid=0.28768205642700195
    logit log(p)-log1p(-p) (f64)   = -1.0986122886681096  bits=3213664084 resid=0.28768205642700195
  BIT-EXACT transforms: ['ln (f64)', 'ln (f32 in)']

survival:cox   base_score arg = 7.5
  MEASURED INTERCEPT = 2.0149030685424805   float32 bits = 1073804332
    identity (f64)  = 7.5                  bits=1089470464 resid=5.4850969314575195
    ln (f64)        = 2.0149030205422647   bits=1073804332 resid=0.0  <== BIT-EXACT
    ln (f32 in)     = 2.0149030205422647   bits=1073804332 resid=0.0  <== BIT-EXACT
  BIT-EXACT transforms: ['ln (f64)', 'ln (f32 in)']
```

17-value sweep:

```
$ uv run python probe6.py
survival:cox: sweep of 17 base_score values
  bit-exact hit counts over 17 values:
    ln (f32 in, f64 log)                17/17   worst abs diff = 0.0  <== ALL
    ln (f64 in, f64 log)                15/17   worst abs diff = 1.1920928955078125e-07
    identity                             0/17   worst abs diff = 118.64011669158936
    logit XGB                            0/17   worst abs diff = 1.2039727568626404
```

**Verdict: `survival:cox` stores `base_score` in hazard-ratio space. Intercept =
`ln(base_score)`.** H3 reproduced.

The 2/17 gap between the two `ln` variants is the float32-discipline point, and the exact
two values are named:

```
$ uv run python probe9.py
survival:cox -- ln with float64 input vs float32 input
  DIVERGES v=0.7          serialized='[7E-1]'         measured=-0.3566749691963196
     ln(f64 in)=-0.3566749393939972 exact=False  err=2.9802322387695312e-08
     ln(f32 in)=-0.3566749691963196 exact=True  err=0.0
  DIVERGES v=3.1415927    serialized='[3.1415927E0]'  measured=1.1447299718856812
     ln(f64 in)=1.1447298526763916 exact=False  err=1.1920928955078125e-07
     ln(f32 in)=1.1447299718856812 exact=True  err=0.0
```

**The stored value must be snapped to float32 before the `log`.** `ln(0.48)` and
`ln(float32(0.48))` are different numbers, and XGBoost uses the latter. Errors are only
~`1e-7`, so this will not trip the ≤`1e-6` margin gate on its own — but it *will* make
cross-language parity nonzero, which the gate requires to be exactly `0.0`.

---

## 4. `binary:logistic` — space confirmed, and it is probability space

```
$ uv run python probe4.py
binary:logistic   base_score arg = 0.25
  MEASURED INTERCEPT = -1.0986123085021973   float32 bits = 3213664084
    identity (f64)                  = 0.25                 bits=1048576000 resid=1.3486123085021973
    ln (f64)                        = -1.3862943611198906  bits=3216077336 resid=0.28768205642700195
    logit log(p/(1-p)) (f64)        = -1.0986122886681098  bits=3213664084 resid=0.0  <== BIT-EXACT
    logit log(p/(1-p)) (f32 in)     = -1.0986122886681098  bits=3213664084 resid=0.0  <== BIT-EXACT
    logit -log(1/p-1) (f64)         = -1.0986122886681098  bits=3213664084 resid=0.0  <== BIT-EXACT
    logit -log(1/p-1) (f32 in)      = -1.0986122886681098  bits=3213664084 resid=0.0  <== BIT-EXACT
    logit log(p)-log1p(-p) (f64)    = -1.0986122886681096  bits=3213664084 resid=0.0  <== BIT-EXACT
    logit log(p)-log1p(-p) (f32 in) = -1.0986122886681096  bits=3213664084 resid=0.0  <== BIT-EXACT
  BIT-EXACT transforms: [all six logit variants]
```

Identity `1.35`, `ln` `0.288`, logit `0.0`. The space is unambiguously **probability
space** and the transform is unambiguously **logit**. 27-value sweep:

```
$ uv run python probe6.py
binary:logistic: sweep of 27 base_score values
  bit-exact hit counts over 27 values:
    logit XGB (f32 1/p-1, f64 log)      27/27   worst abs diff = 0.0  <== ALL
    logit XGB (all float32)             27/27   worst abs diff = 0.0  <== ALL
    logit naive f64 log(p/(1-p))        11/27   worst abs diff = 6.198883056640625e-06
    logit naive f32-in log(p/(1-p))      8/27   worst abs diff = 3.814697265625e-06
    identity                             0/27   worst abs diff = 4.605119952932
    ln                                   0/27   worst abs diff = 4.605175048112869
```

So `logit` versus `identity`/`ln` is settled by orders of magnitude. But note that the
*textbook* logit scores only 11/27. That is §5.

---

## 5. Top-line finding — the logit formulation matters, and the textbook one is wrong

**`binary:logistic` computes the prob→margin conversion with a float32 `1/p - 1`
intermediate.** Getting the space right and the formula "right" in the textbook sense
still produces wrong last bits, and on 2 of 27 measured values it breaches the
≤`1e-6` margin gate.

The formulation that is bit-exact on all 27 values:

```python
p = np.float32(base_score)                     # snap the stored value first
t = np.float32(np.float32(np.float32(1.0) / p) - np.float32(1.0))   # float32 throughout
intercept = np.float32(-math.log(float(t)))
```

Equivalently in JS terms: `-Math.log(Math.fround(Math.fround(1 / Math.fround(p)) - 1))`.

The discriminating evidence — six formulations against the measured intercept, at the
value where they diverge most:

```
$ uv run python probe5.py
E2b  float32-intermediate formulations vs the measured intercept

  arg=0.48  serialized='[4.8E-1]'
    measured intercept = -0.08004285395145416  bits=3181636994
    f64 log(p/(1-p)) of f32 p        = -0.0800427506576553   bits=3181636980 ULP gap= -14  absdiff=1.043081283569336e-07
    f64 log(p/(1-p)) of f64 arg      = -0.0800427076735365   bits=3181636974 ULP gap= -20  absdiff=1.4901161193847656e-07
    all-f32 log(p/(1-p))             = -0.08004270493984222  bits=3181636974 ULP gap= -20  absdiff=1.4901161193847656e-07
    f32 ratio, f64 log               = -0.08004270270648271  bits=3181636974 ULP gap= -20  absdiff=1.4901161193847656e-07
    f32 (1/p - 1), f64 log           = -0.08004285439265127  bits=3181636994 ULP gap=  +0  absdiff=0.0  <== BIT-EXACT
```

Read that carefully. **Computing the ratio `p/(1-p)` in float32 is not the same as
computing `1/p - 1` in float32, and only the latter matches.** Being generically
"float32-disciplined" is not sufficient; the specific expression XGBoost evaluates has to
be reproduced. Four plausible float32-respecting formulations are all 14–20 ULP wrong.

Two independent confirmations of the mechanism:

**(a) Adjacent float32 inputs collapse onto identical intercepts**, which is what a
lossy float32 intermediate does and what an accurate float64 logit would not do:

```
$ uv run python probe5.py
  k=-3  input float64=0.4799998998641968     intercept = -0.08004307746887207  bits=3181637024
  k=-2  input float64=0.47999992966651917    intercept = -0.08004307746887207  bits=3181637024
  k=-1  input float64=0.47999995946884155    intercept = -0.08004285395145416  bits=3181636994
  k=+0  input float64=0.47999998927116394    intercept = -0.08004285395145416  bits=3181636994
  k=+1  input float64=0.48000001907348633    intercept = -0.08004263788461685  bits=3181636965
  k=+2  input float64=0.4800000488758087     intercept = -0.08004241436719894  bits=3181636935
  k=+3  input float64=0.4800000786781311     intercept = -0.08004241436719894  bits=3181636935
```

Distinct float32 inputs, identical intercepts, in pairs. §1 already ruled out lossy
serialization as the cause — all seven serialized to distinct strings.

**(b) Signed zero.** At `base_score = 0.5` the measured intercept is **negative zero**,
which is what `-log(1/0.5 - 1) = -log(1.0) = -0.0` produces and what
`log(0.5/0.5) = +0.0` does not:

```
$ uv run python probe6.py
signed-zero note
  binary:logistic base_score=0.5 -> serialized '[5E-1]'
  measured intercept = -0.0   bits = 2147483648   is negative zero: True
  logit XGB      = np.float32(-0.0)  bits = 2147483648
  naive f64 logit= np.float32(0.0)  bits = 0
  numerically equal: True   bitwise equal: False
```

`survival:cox` at `base_score=1.0` gives **positive** zero (`bits = 0`), matching
`ln(1.0)`. The two objectives differ in the sign of zero.

### Which values the textbook formula gets wrong, and by how much

```
$ uv run python probe7.py
A  naive float64 logit: per-value error vs the 1e-6 gate
           p               measured        naive f64 logit                  abs err  gate
        0.01     -4.595119953155518     -4.595119953155518                      0.0  ok
        0.25    -1.0986123085021973    -1.0986123085021973                      0.0  ok
        0.45    -0.2006707787513733   -0.20067068934440613    8.940696716308594e-08  ok
        0.48   -0.08004285395145416   -0.08004270493984222   1.4901161193847656e-07  ok
        0.49  -0.040005315095186234  -0.040005333721637726    1.862645149230957e-08  ok
         0.5                   -0.0                    0.0                      0.0  ok
        0.75     1.0986121892929077     1.0986123085021973   1.1920928955078125e-07  ok
         0.9     2.1972241401672363     2.1972246170043945     4.76837158203125e-07  ok
        0.95     2.9444382190704346      2.944438934326172    7.152557373046875e-07  ok
        0.99        4.5951247215271      4.595119953155518     4.76837158203125e-06  BREACHES 1e-6
    0.987654      4.381994247436523       4.38200044631958    6.198883056640625e-06  BREACHES 1e-6

  worst naive error: p=0.987654 err=6.198883056640625e-06
  values breaching the 1e-6 gate: 2/22 -> [0.99, 0.987654]
  same values under the XGB float32-intermediate formula:
    p=0.99       err = 0.0   bit-exact = True
    p=0.987654   err = 0.0   bit-exact = True
```

```
$ uv run python probe9.py
binary:logistic -- which values the naive float64 logit gets RIGHT
  naive f64 logit bit-exact on   : [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.3333333, 0.4, 0.51, 0.8, 0.1234567]
  naive f64 logit WRONG on       : [0.45, 0.48, 0.49, 0.499, 0.5, 0.501, 0.52, 0.55, 0.6, 0.7, 0.75, 0.9, 0.95, 0.99, 0.6666667, 0.987654]
  -> 16/27 values silently wrong with the textbook formula
```

This is exactly the project's stated failure signature: **correct on most values, wrong on
some, no error raised.** A boundary fixture built only from `0.25` or `0.5` would pass a
buggy implementation. Any fixture corpus for this transform must include values near
`0.5` (`0.48`, `0.52`) and near `1.0` (`0.99`, `0.987654`) — those are where it breaks.

---

## 6. H1 — the `boost_from_average` overwrite

H1 reproduces, but the hypothesis as written is missing the condition, and it is wrong
about the scope.

**An explicit `base_score` survives by default.** Supplying `base_score` causes XGBoost
to silently flip `boost_from_average` to `0`:

```
$ uv run python probe7.py
D  H1 head-on: explicit base_score AND boost_from_average=1
  reg:squarederror   passed base_score=0.25, boost_from_average=None  -> stored='[2.5E-1]'   (0.25)  serialized bfa='0'  survived=True
  reg:squarederror   passed base_score=0.25, boost_from_average=0     -> stored='[2.5E-1]'   (0.25)  serialized bfa='0'  survived=True
  reg:squarederror   passed base_score=0.25, boost_from_average=1     -> stored='[3.9882263E1]' (39.882263)  serialized bfa='1'  survived=False
  binary:logistic    passed base_score=0.25, boost_from_average=None  -> stored='[2.5E-1]'   (0.25)  serialized bfa='0'  survived=True
  binary:logistic    passed base_score=0.25, boost_from_average=0     -> stored='[2.5E-1]'   (0.25)  serialized bfa='0'  survived=True
  binary:logistic    passed base_score=0.25, boost_from_average=1     -> stored='[4.8E-1]'   (0.48)  serialized bfa='1'  survived=False
  survival:cox       passed base_score=0.25, boost_from_average=None  -> stored='[2.5E-1]'   (0.25)  serialized bfa='0'  survived=True
  survival:cox       passed base_score=0.25, boost_from_average=0     -> stored='[2.5E-1]'   (0.25)  serialized bfa='0'  survived=True
  survival:cox       passed base_score=0.25, boost_from_average=1     -> stored='[1E0]'      (1.0)  survived=False
```

Two corrections to H1:

1. **The overwrite requires `boost_from_average=1` to be forced explicitly alongside
   `base_score`.** `base_score` alone is safe: 0.25 in, `'[2.5E-1]'` out, verbatim,
   across every objective and every round count tested.
2. **It is not specific to `reg:squarederror`.** `binary:logistic` and `survival:cox`
   overwrite identically. H1 attributes this to regression; the measurement does not.

What the estimated value is, per objective, and **when** the estimation happens:

```
$ uv run python probe7.py
  reg:squarederror   base_score UNSET, rounds=0 -> stored='[5E-1]'         bfa='1'   label mean(f64)=39.88226314544678
  reg:squarederror   base_score UNSET, rounds=1 -> stored='[3.9882263E1]'  bfa='1'   label mean(f64)=39.88226314544678
  reg:squarederror   base_score UNSET, rounds=5 -> stored='[3.9882263E1]'  bfa='1'   label mean(f64)=39.88226314544678
  binary:logistic    base_score UNSET, rounds=0 -> stored='[5E-1]'         bfa='1'   label mean(f64)=0.48
  binary:logistic    base_score UNSET, rounds=1 -> stored='[4.8E-1]'       bfa='1'   label mean(f64)=0.48
  binary:logistic    base_score UNSET, rounds=5 -> stored='[4.8E-1]'       bfa='1'   label mean(f64)=0.48
  survival:cox       base_score UNSET, rounds=0 -> stored='[5E-1]'         bfa='1'   label mean(f64)=1.5502546086907387
  survival:cox       base_score UNSET, rounds=1 -> stored='[1E0]'          bfa='1'   label mean(f64)=1.5502546086907387
  survival:cox       base_score UNSET, rounds=5 -> stored='[1E0]'          bfa='1'   label mean(f64)=1.5502546086907387
```

- `reg:squarederror` — estimate is the **label mean**, `39.882263`, in label space.
- `binary:logistic` — estimate is the **label mean in probability space**, `0.48`
  (96 positives / 200). Not the logit of it.
- `survival:cox` — estimate is **`1.0`**, i.e. margin intercept `0.0`. It is *not* the
  label mean (`1.5502546`), and not any function of it that this probe identified.
  **INFERRED:** Cox appears to have no intercept estimation and to fall back to a fixed
  `1.0`. Confirming that would need fits across several datasets with different label
  distributions; this probe used one, so the alternative reading — that `1.0` is a
  coincidence of this data — is not excluded.
- The estimation happens on the **first boosting round**, not at configure time. At
  `rounds=0` the value is still the untransformed default `0.5` for all three.

---

## 7. Trap — a zero-round model with `base_score` unset has no transform applied

This is the sharpest measurement trap found, and it is worth recording because the
obvious way to isolate the intercept walks straight into it.

**A model with 0 boosting rounds and `base_score` left unset emits the stored value as
the margin directly — the per-objective transform is never applied.**

```
$ uv run python probe10.py
zero-round models: base_score UNSET vs SET-to-the-same-value

  reg:squarederror
    UNSET (default)          trees=0 stored='[5E-1]'   bfa='1'
      margin        = array([0.5], dtype=float32)
      transform says= 0.5
      transform applied: True
    SET to 0.5 explicitly    trees=0 stored='[5E-1]'   bfa='0'
      margin        = array([0.5], dtype=float32)
      transform applied: True

  binary:logistic
    UNSET (default)          trees=0 stored='[5E-1]'   bfa='1'
      margin        = array([0.5], dtype=float32)
      transform says= -0.0
      transform applied: False
    SET to 0.5 explicitly    trees=0 stored='[5E-1]'   bfa='0'
      margin        = array([-0.], dtype=float32)
      transform applied: True

  survival:cox
    UNSET (default)          trees=0 stored='[5E-1]'   bfa='1'
      margin        = array([0.5], dtype=float32)
      transform says= -0.6931471824645996
      transform applied: False
    SET to 0.5 explicitly    trees=0 stored='[5E-1]'   bfa='0'
      margin        = array([-0.6931472], dtype=float32)
      transform applied: True
```

For `binary:logistic` the zero-round default model's margin is `0.5` — which is neither
`logit(0.5) = 0` nor a margin at all. `reg:squarederror` is unaffected only because its
transform is the identity, so the trap is invisible there.

Consequence for method: the brief's suggestion to isolate the intercept with a zero-tree
model is correct **only if `base_score` is pinned explicitly**. Had this probe used a
zero-round model with defaults to calibrate, it would have concluded that
`binary:logistic` and `survival:cox` both use the identity transform — a confident,
plausible, wrong finding of exactly the kind this project exists to prevent. Flagged as
a decision below, because an exporter can encounter this state.

---

## 8. `booster[0:0]` silently returns the whole model

Out of scope, but a silent-wrong-number hazard worth recording since it defeated an
earlier measurement approach in this probe.

```
$ uv run python probe3.py
L2  booster slicing behaviour
  full booster trees            = 4
  booster[0:0] -> trees = 4
  booster[0:1] -> trees = 1
  booster[0:2] -> trees = 2
  booster[1:1] -> XGBoostError: ... Check failed: end != begin (1 vs. 1) : Empty slice is not allowed.
  booster[2:2] -> XGBoostError: ... Check failed: end != begin (2 vs. 2) : Empty slice is not allowed.
  booster[0:4] -> trees = 4

  predict with iteration_range:
  iteration_range=(0, 0) -> n unique margins = 66
  iteration_range=(0, 1) -> n unique margins = 8
  iteration_range=(0, 4) -> n unique margins = 66
```

`booster[1:1]` and `booster[2:2]` raise "Empty slice is not allowed" — but `booster[0:0]`
returns all four trees with no error and no warning, because `(0, 0)` is overloaded to
mean "all iterations". Same for `iteration_range=(0, 0)` in `predict`. Any code that
computes a slice bound arithmetically and can land on `(0, 0)` gets the full ensemble
while believing it got none.

---

## 9. Summary — what a correct implementation must do

Stated as measured behaviour, not as a design recommendation.

1. Read `learner.learner_model_param.base_score` as a **string**, strip the brackets,
   parse the single element. Fail loudly on a non-string, on a missing bracket, or on
   more than one element (per D007) — a 2-element vector has not been observed and its
   meaning is not established.
2. **Snap to float32 immediately on parse.** `float(stored)` is not the value XGBoost
   holds; `float32(float(stored))` is. §1 confirms the bit pattern round-trips exactly at
   float32 and §3/§5 confirm the transforms disagree if it is not snapped.
3. Apply the per-objective transform:
   - `reg:squarederror` → `float32(base_score)`
   - `binary:logistic` → `-log(float32(float32(1/float32(p)) - 1))`, **not**
     `log(p/(1-p))`
   - `survival:cox` → `log(float32(base_score))`
4. Do not read `boost_from_average` as a signal about whether the value needs
   transforming for any model with ≥1 tree — the transform applies regardless. It is only
   relevant to the 0-tree case in §7.

---

## Decisions needed

```
DECISION NEEDED: How to handle a 0-tree artifact whose base_score was never estimated
Context:  A model fitted with 0 boosting rounds and base_score unset serializes
          base_score='[5E-1]' with boost_from_average='1', and XGBoost emits the margin
          as 0.5 raw -- the per-objective transform is NOT applied (probes/base_score.md
          section 7). Every model with >=1 tree does apply it. So the artifact is
          internally ambiguous: the same field means different things depending on tree
          count, and only for the estimated-value path.
Options:  A) Refuse to export a 0-tree model -- raise. Consistent with D007; costs the
             ability to round-trip a degenerate model nobody would deploy.
          B) Reproduce XGBoost's behaviour exactly, keying on (tree count == 0 AND
             boost_from_average == 1). Bit-faithful; encodes an XGBoost quirk into the
             format and needs a fixture that looks like a bug.
          C) Export it with the transform applied, diverging from XGBoost on this one
             degenerate case. Simplest predictor; breaks the parity gate on that fixture.
Lean:     A. A 0-tree model carries no learned structure, the state is XGBoost's own
          inconsistency rather than model content, and raising is what D007 prescribes
          for something whose meaning is not established.
Blocks:   Artifact format design (Phase 3) -- specifically whether boost_from_average
          must be carried in the artifact at all. Under A it need not be.
```

```
DECISION NEEDED: Does the artifact store base_score or the derived margin intercept?
Context:  The logit transform for binary:logistic is only bit-reproducible through a
          float32 1/p-1 intermediate (section 5). Storing base_score in probability
          space means BOTH Python and JS must reimplement that exact expression, and any
          divergence between them shows up as nonzero cross-language parity. Storing the
          already-transformed margin intercept means the transform runs once, in Python,
          at export.
Options:  A) Store base_score in its native per-objective space; both predictors
             transform. Mirrors XGBoost; duplicates a delicate float32 expression in two
             languages; two places to get the 1/p-1 form wrong.
          B) Store the derived margin intercept as a float32; predictors add it directly.
             The transform exists in exactly one place. Artifact is one step further from
             the XGBoost original, and inspection tooling no longer shows the user the
             base_score they passed in.
          C) Store both, and have the loader verify they agree.
Lean:     B. The float32 logit formulation is the single most error-prone thing this
          probe found -- 16/27 values silently wrong with the obvious formula. Having it
          in one language rather than two roughly halves the exposure, and the parity
          gate then tests tree-walk arithmetic rather than re-testing a transform.
Blocks:   Artifact format design (Phase 3). Also the JS numerical core scope: under B,
          JS never needs a logit at all for the intercept.
```

```
DECISION NEEDED: Is survival:cox's estimated base_score of 1.0 fixed, or data-dependent?
Context:  With base_score unset, survival:cox stored '[1E0]' -- intercept exactly 0.0 --
          while the label mean was 1.5502546 (section 6). reg:squarederror and
          binary:logistic both stored the label mean in their respective spaces. Cox
          stored neither the mean nor any function of it this probe identified. Measured
          on ONE dataset, so 1.0 being a fixed fallback and 1.0 being a coincidence are
          both consistent with the evidence.
Options:  A) Probe further: fit Cox on several datasets with deliberately different label
             scales and censoring rates, check whether base_score ever leaves 1.0.
          B) Treat it as opaque -- read whatever is stored and apply ln. Correct either
             way for export, since the value is read rather than derived.
Lean:     B for the implementation, A as a cheap follow-up before 1.0. The exporter never
          needs to predict the value, so this does not block anything -- but if Cox's
          base_score is always 1.0, the fixture corpus needs an explicitly-set Cox
          base_score or it will only ever exercise ln(1.0) = 0, which is the one input
          where a broken ln implementation still looks right.
Blocks:   Nothing. Fixture design should not ship without resolving it.
```

---

## Not measured

Stated so nothing here is mistaken for a finding.

- **Multi-element `base_score` vectors.** Every observed value had exactly one element.
  Whether the bracket carries more for multi-output/multi-class is inferred from the
  notation only. Out of scope per D003.
- **The sklearn wrapper surface** (`intercept_`, `get_params()['base_score']`).
  `scikit-learn` is not installed in the workspace and this probe does not install
  anything. `xgb.XGBRegressor(...)` raises `ImportError: sklearn needs to be installed in
  order to use this module`. The `save_model` / `save_raw` paths used throughout are the
  documented serialization interface and are unaffected.
- **Objectives outside the 1.0 scope.** No behaviour here should be extended by analogy
  to `reg:logistic`, `count:poisson`, `survival:aft`, or `rank:*`. Three objectives were
  measured; three objectives are reported.
- **XGBoost versions other than 3.3.0.** Per D001, drift is a separate pass.
