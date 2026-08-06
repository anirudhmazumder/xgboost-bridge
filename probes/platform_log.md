# Probe — XGBoost's intercept logarithm is platform-dependent

Established by this probe: **XGBoost's own margin-space intercept differs between
darwin/arm64 and linux/x86_64**, on the same model file, the same `base_score` and
the same XGBoost 3.3.0. On 29 of 58 measured inputs, by exactly 1 ULP.

That makes "bit-exact against XGBoost" and "platform-independent" incompatible
goals rather than a choice of recipe, which is what D053 acts on.

Superseded by this probe: the conclusion drawn in `probes/base_score.md` §6 and
recorded as **D040** — that `np.log` of a float32 is the route XGBoost takes,
full stop. It is the route XGBoost takes *on darwin/arm64*. The measurement was
correct and the generalisation was not.

Run it yourself:

```bash
uv run python probes/platform_log_probe.py                      # this machine
gh workflow run probe-platform.yml -f script=platform_log_probe.py   # linux/x86_64
```

---

## 1. How CI found it

The first successful CI run reported **18 Python failures on linux/x86_64 that
pass on darwin/arm64**. Every one was the derived intercept differing from
XGBoost's observed zero-tree margin by exactly 1 ULP, in both directions:

```
base_score=0.6   derived 0x3ECF9922  vs XGBoost 0x3ECF9921
base_score=0.3   derived 0xBF58E882  vs XGBoost 0xBF58E883
```

`InterceptMismatchError` fired, so no wrong number was produced — the export
gate refused the model. Loud failure, as intended. But it refused *valid*
models, which is a defect, and it meant the suite could only ever pass on one
platform.

## 2. The first pass got it backwards

The obvious hypothesis is that numpy's float32 `log` differs by platform while
XGBoost's is stable. Thirteen hand-picked "interesting" `base_score` values were
measured on darwin against numpy, XGBoost, and a 50-digit `mpmath` reference:

| route | agreement with XGBoost, 13 values |
|---|---|
| `np.log` of a float32 | 13/13 |
| `mpmath`, correctly rounded | 13/13 |

All three agreed everywhere, and darwin's value equalled linux's *XGBoost*
column on every one — which reads as "XGBoost is correctly rounded and stable,
numpy is the mover." A separate 49,640-value sweep then found `np.log(f32)`
disagreeing with correctly-rounded on 0.1249% of inputs while
`f32(math.log(f64))` matched it on all of them, apparently confirming it.

**Both conclusions were wrong**, because neither sample distinguished the
candidates. Selecting 40 inputs *because* `np.log` and the correctly-rounded
value disagree there, and then asking XGBoost, reverses the result:

| route | agreement with XGBoost, 40 discriminating values |
|---|---|
| `np.log` of a float32 | **40/40** |
| `mpmath`, correctly rounded | **0/40** |

XGBoost is **not** correctly rounded, and on darwin it reproduces numpy's
float32 `log` exactly, including where both miss. This is the lesson D040
already recorded, walked into again one level up: *a sample that does not
deliberately target the inputs where two candidates diverge cannot distinguish
them, and its silence is not evidence of equivalence.*

## 3. The measurement that settles it

`probes/platform_log_probe.py`, run unchanged on both platforms. Inputs are
embedded as float32 **bit patterns** so the comparison cannot be a test of two
decimal parsers, and were selected on darwin as values where `np.log` and a
correctly-rounded logarithm disagree. Five ordinary values per objective are
included as a control.

Agreement with XGBoost's own observed intercept, 58 inputs:

| route | darwin/arm64 | linux/x86_64 |
|---|---|---|
| `np.float32(np.log(f32))` | **58/58** | 36/58 |
| `np.float32(math.log(f64))` | 10/58 | 39/58 |
| `Decimal.ln()` at 40 digits, narrowed | 10/58 | 39/58 |
| `mpmath` at 60 digits, narrowed | 10/58 | 39/58 |

The 10 are the control values. **No route is exact on both platforms**, and the
correctly-rounded route is not exact on either — so implementing a
correctly-rounded logarithm would be exactly as wrong, just differently.

Environments: CPython 3.12.8 / numpy 2.5.1 / XGBoost 3.3.0 / Apple libm, and
CPython 3.12.3 / numpy 2.5.1 / XGBoost 3.3.0 / glibc 2.39.

### XGBoost's own value, diffed across the two platforms

The decisive comparison is XGBoost's column against itself:

| group | inputs | XGBoost differs by platform |
|---|---|---|
| `survival:cox`, discriminating | 24 | **14** |
| `survival:cox`, control | 5 | 0 |
| `binary:logistic`, discriminating | 24 | **15** |
| `binary:logistic`, control | 5 | 0 |
| **total** | **58** | **29** |

```
base_score=0x3F7F9A17   darwin=0xBACBFA9D   linux=0xBACBFA9C
base_score=0x3ED942E3   darwin=0xBF5B7308   linux=0xBF5B7307
```

Every difference is exactly 1 ULP. The direction is **not** systematic — darwin
is higher on 15 and lower on 14 — which rules out a constant offset or a
differing clamp and leaves last-place rounding in the platform's `logf` as the
explanation. IEEE-754 mandates correct rounding only for `+ − × ÷ √` and fma;
`logf` is not covered, and no two `libm` implementations agree. This is the same
finding as `probes/output_transform.md` §4 for `exp`, which is why both packages
bundle `exp` — the one transcendental left on a platform implementation is the
one that broke.

**Consequence for upstream, stated plainly:** XGBoost's own margins for
`binary:logistic` and `survival:cox` are not reproducible across these two
platforms at the last bit. That is an upstream property this library can record
but not fix.

## 4. What is confined to the logarithm, and what is not

`reg:squarederror` takes the identity link, so no logarithm runs. Its intercept
is bit-exact on both platforms by every route, which localises the divergence to
the logarithm rather than to `base_score` parsing, the clamp, the
`boost_from_average` cell, or emission. Pinned by
`test_regression_recipe_is_exact_because_no_logarithm_runs`.

Inference is unaffected. The intercept is a stored artifact field, and neither
predictor computes a logarithm — `grep -rE 'Math\.log|\blog\(' packages/js/src/`
returns nothing. Cross-language parity remains exactly `0.0` at both measurement
points.

## 5. A second, smaller platform difference

For `survival:cox` at a negative `base_score`, XGBoost returns a NaN whose
**sign bit differs by platform**: `0x7FC00000` on darwin, `0xFFC00000` on linux.
Both are quiet NaNs. IEEE-754 does not specify the sign of a NaN produced by
`log` of a negative number, so this is not a behaviour to pin; two of the 18
failures were a test that pinned it. The behaviour that matters — the value is
not finite, and export refuses it — holds on both platforms and is what is
asserted now.

## 6. Determinism of the replacement

D053 reads the intercept out of the engine, which for a model with trees means a
zero-boosting-round refit. Export determinism is a gate, so the refit was
measured rather than assumed: 8 configurations × 15 repeats gave **one distinct
bit pattern each**, and the value was stable across separate interpreter
invocations. There are no boosting rounds to vary.

## 7. Ambiguities this probe does not close

- **Which** `logf` XGBoost reaches is not established — whether it links the
  system `libm` directly or a vendored routine. It does not matter for D053,
  which observes the result rather than reproducing the mechanism, and no probe
  available here can see inside the compiled call.
- Only two platforms are measured. A third may agree with neither. The design is
  indifferent to this by construction, which is the point of observing rather
  than deriving, but the claim "29 of 58" is about these two only.
- Whether the divergence extends to XGBoost's **leaf values** during training is
  not measured. It would not affect export, which reads leaves from the
  serialized model rather than recomputing them, but a model *trained* on one
  platform and one trained on another may not be the same model.
