# Verification

What this library has measured, by what method, at what numbers — and, in the last section, **what it has not measured.** If you are deciding whether to trust these predictions, read that section too.

Every figure here is reproducible from a clone. The commands are at the end.

---

## What is checked, and against what

The distinction that matters most: **agreement between this library's two predictors is not evidence that either is correct.** Two implementations of the same mistake agree perfectly. So correctness is measured against oracles outside the library, and cross-language agreement is a separate check with a separate purpose.

| Property | Oracle | Result |
|---|---|---|
| Margin correctness | **XGBoost's own `predict(output_margin=True)`** | **bit-exact, 289/289 corpus rows** |
| Output correctness | **XGBoost's own `predict()`** | max **relative** error `9.56e-08` (bound: `1e-6`) |
| `sigmoid` / `exp` accuracy | **`mpmath` at 50 digits**, per language independently | max **1 ULP** (`exp`), **2 ULP** (`sigmoid`) |
| Python ↔ JavaScript agreement | each other, on bit patterns | **exactly `0.0`**, at two measurement points |
| Artifact shape | JSON Schema, draft 2020-12 | all 23 fixtures validate |

Test suites: **977** Python, **112** Node. No skipped tests, no `xfail`.

---

## Numerical accuracy, per objective

Against XGBoost's recorded output across the whole fixture corpus:

| Objective | Rows | Output bit-exact | Max relative error |
|---|---|---|---|
| `reg:squarederror` | 174 | **174/174** | `0.0` |
| `binary:logistic` | 89 | 87/89 | `9.56e-08` |
| `survival:cox` | 26 | 22/26 | `7.03e-08` |

**Margins are bit-exact on all 289 rows, for every objective.**

Six *output* rows differ from XGBoost in the last bit. That is expected and unavoidable: XGBoost's own `expf` is not correctly rounded, so no implementation can match it bit-for-bit at the output stage. Note where the differences fall — `reg:squarederror` uses the identity transform and differs on **nothing**, while every difference is on one of the two objectives that call the bundled `exp`. That is the pattern a correct implementation produces.

Those six are pinned as an exact set of `(fixture, row)` pairs plus their count, in both languages. Movement in **either** direction fails the suite, improvements included — so the check stays a tripwire rather than becoming a tolerance band a future defect could hide inside.

### Why `sigmoid` and `exp` are implemented here rather than called

IEEE-754 requires correct rounding only for `+ − × ÷ √` and fused multiply-add. `exp` is not covered, and no two `libm` implementations agree: V8's differs from Apple's on **4.2%** of sigmoid and **9.6%** of `exp` evaluations, by up to 2 ULP. A library calling the platform's `exp` cannot give two languages the same answer.

So both packages build the transform from the four correctly-rounded operations and exact powers of two, evaluate it under float32 semantics, and reproduce XGBoost's clamps — including the `binary:logistic` floor, which returns exactly `3.006635794144578e-39` and never `0.0`.

**What this means for you:** comparing this library's probability against `scipy.special.expit`, `Math.exp`, or XGBoost's own output will show last-bit differences. That is deliberate, bounded, and the price of the two predictors agreeing exactly with each other.

---

## Cross-language agreement

Both measurement points — the raw margin, and the final output after the transform — compared as **bit patterns**, never with `==`. (`-0.0 == 0.0` is true in both languages, and they are different values; the corpus contains a fixture whose margin is exactly `-0.0` to catch a comparison that gets this wrong.)

| Corpus | Rows | Margin | Output | Measured on |
|---|---|---|---|---|
| Fixture corpus | 299 | **0 mismatches** | **0 mismatches** | both platforms, in CI |
| Generated, adversarial | **100,000** | **0 mismatches** | **0 mismatches** | both platforms, in CI |

The generated rows are adversarial rather than uniform, because uniform rows cannot find the defect that matters — **0 of 20,000 random continuous rows** detect an incorrectly-cast float32 comparison. The composition is counted and reported on every run rather than estimated:

| Row class | Count | What it is for |
|---|---|---|
| Exactly on a `float32` split threshold | **68,664** | Equality routes RIGHT under a strict `<`. An implementation that casts only one side diverges here and nowhere else |
| **Narrows onto** a threshold without equalling it | **20,511** | The only class that pins the *sample* side of the cast — see below |
| The two adjacent `float32` values | 774 | Confirms the routing flips where it should |
| Format edges — subnormals, `±3.4e38`, past `1e30` | 252 | Where `Math.fround` and `np.float32` could part company |
| Missing (`NaN`) | 23 | The default-direction path |
| Fully random | 9,776 | A control, so the run is not exclusively pathological |

Both languages are fed identical float64 **bit patterns** from the same generated file, and the harness checks that rather than assuming it, so the comparison cannot become a test of two number parsers. The generator is seeded, so the 100,000 rows are **identical on both platforms** — the two runs compare the same inputs, not merely the same number of inputs.

**Why the narrows-onto class exists.** A row sitting exactly on a threshold carries a value that is already `float32`-exact, so narrowing the *sample* is a no-op and an implementation that skips it still routes correctly. The rows that catch that are float64 values which round to the threshold without equalling it: narrowed they compare equal and route RIGHT, un-narrowed a value just below routes LEFT — a whole subtree of difference from one missing cast. Reverting `Math.fround` on the sample in the JavaScript walk, the 299-row corpus reports **1** mismatch; this reports **1,279 of 20,000**.

**Two protections this harness does not claim to pin,** because they absorb each other: the parse-time `Float32Array` and the threshold-side `Math.fround`. Reverting *either* alone leaves parity at exactly `0.0`, since each narrows the threshold on its own. They are pinned by direct type assertions instead — the JavaScript suite asserts `node_values` loads into a `Float32Array` and the Python suite asserts its `dtype` — which is the only kind of check that can see a storage type.

Sensitivity is re-confirmed on every run rather than once: a one-ULP injection at **each** measurement point must produce exactly one mismatch, because "0 across 100,000 rows" is also what a silently broken comparator reports.

The two CI platforms also differ in JavaScript engine version — Node 20 on Linux, Node 24 locally — so the `0.0` spans two V8 releases as well as two `libm`s.

### Row counts: 289 and 299 are both correct

289 is the number of corpus rows carrying XGBoost ground truth — the denominator for correctness against XGBoost. 299 adds 10 rows carrying `±inf` feature values, which deliberately have **no** ground truth, because XGBoost itself answers them two different ways (it raises through `DMatrix` and treats them as ordinary comparable values through `inplace_predict`). Those 10 still count for cross-language agreement, because agreeing on *refusing* an input is as much a property as agreeing on a value.

---

## What this library refuses

Refusing loudly is a feature. Each refusal exists because the alternative is a plausible wrong number, and each is pinned by a fixture.

| Refused | Why |
|---|---|
| `dart` | Only one signal distinguishes it inside a serialized model, so it cannot be detected reliably. A `dart` model with no dropout is byte-identical to `gbtree` and does export — correctly, because it *is* a tree ensemble |
| `gblinear` | Deprecated upstream, and a separate inference path with no trees |
| Multi-class, multi-target | An objective allow-list alone lets `reg:squarederror` with `num_target=2` through, producing two outputs per row that a scalar predictor would silently accept |
| Categorical splits | They invert the child convention — in-set goes *right* — and their threshold slot holds a subnormal rather than a threshold |
| Models with no feature names | A strict-key policy with no keys to check reads as enforced and is not |
| Early-stopped models with an ambiguous tree count | The effective count is not a property of the model: the same file answers differently loaded as a `Booster` versus through the scikit-learn estimator, diverging by `1.55`, with no field distinguishing them |
| XGBoost versions outside the tested list | Version drift here is silent — 3.4.0-dev relocated a field, and 3.3.0 reads such a model returning **0 of 400 rows correct at max error 1.26, with no warning and exit code 0** |
| `±inf` feature values | Upstream is itself inconsistent, as above |
| Non-finite intercepts | Reachable and silent: Cox at `base_score=0.0` gives `-inf`, and at any negative value `NaN`, both accepted by XGBoost without complaint |

`NaN` is **not** refused. It is the missing value, and it routes by the tree's default direction.

---

## What is NOT verified

The honest limits. Read this before relying on the numbers above in an environment unlike the one they were measured in.

### Most figures above were measured on one machine

**darwin/arm64** — CPython 3.12.8, numpy 2.5.1, XGBoost 3.3.0, Node 20.19.0 / 24.7.0 / 24.18.0. The installed wheel was additionally exercised at the declared floor, Python 3.10.20 with numpy 1.24.4, on that same machine.

CI now also runs on **linux/x86_64** — glibc 2.39, CPython 3.12.3, numpy 2.5.1, XGBoost 3.3.0, Node 20. Confirmed there, not inferred:

| Check | linux/x86_64 | darwin/arm64 |
|---|---|---|
| Python suite | **977 passed** | 977 passed |
| Node suite | **112 passed** | 112 passed |
| Margin parity, Python ↔ JavaScript | **exactly `0.0`** | exactly `0.0` |
| Output parity, Python ↔ JavaScript | **exactly `0.0`** | exactly `0.0` |
| Rows compared, fixture corpus | 299 (289 valued, 10 refused by both) | 299 |
| Rows compared, generated adversarial | **100,000, `0.0` at both points** | 100,000, `0.0` |
| Wheel installs from a clean environment and predicts | **yes** | yes |
| npm tarball installs from a clean project and predicts | **yes** | yes |

The reasoning for why these results were *expected* to carry across platforms was:

- The float32 arithmetic is built only from `+ − × ÷` and exact powers of two, all of which IEEE-754 requires to be correctly rounded. Those cannot vary by platform.
- The transform calls no `libm`, which is the one component that demonstrably *does* vary by platform.
- The margin comparison against XGBoost was measured bit-stable across thread counts, batch sizes, and both prediction APIs — 1440 per-row observations, zero divergences.

That reasoning also named two things it could not rule out, the first being "an x86-64 `libm` differing somewhere a platform function is still reached that is believed to be off the path." **That is exactly what the first Linux run found**, and it is worth being blunt that reasoning identified the hole and only measurement found what was in it.

#### What Linux found: XGBoost's own intercept is not portable

For `binary:logistic` and `survival:cox`, XGBoost computes the margin intercept with the platform's `logf`. IEEE-754 requires correct rounding only for `+ − × ÷ √` and fma — `logf` is not covered. Measured on 58 inputs selected because they discriminate the candidate implementations, **XGBoost's own intercept differs between darwin/arm64 and linux/x86_64 on 29 of them**, always by exactly 1 ULP, in both directions.

Same model file, same `base_score`, same XGBoost 3.3.0. This is an upstream property, not a defect in this library, and it means **XGBoost's own predictions for these two objectives are not bit-reproducible across these platforms.**

The library previously *derived* this intercept and refused any model where its derivation did not match XGBoost bit-for-bit. That refusal is the correct failure mode — it never produced a wrong number — but it fired on valid models: 13 of 50 ordinary `binary:logistic` `base_score` values on Linux. The intercept is now read out of the engine instead, so export is bit-exact against whatever XGBoost is present, on any platform.

**What this means for you:**

- Export is bit-exact against your own XGBoost, wherever you run it.
- An artifact exported on macOS and one exported on Linux from the same model **may differ in the `intercept` field by one float32 step**, for these two objectives. `reg:squarederror` uses the identity link and is unaffected.
- **Inference is unaffected and remains bit-identical everywhere.** The intercept is a stored number and neither predictor computes a logarithm, so cross-language agreement is exactly `0.0` regardless of which platform produced the artifact.
- If you need artifacts to be byte-identical across platforms, export them on one platform. Determinism *within* a platform is byte-identical and tested.

Full evidence, including the first pass at this investigation that reached the opposite conclusion from a sample that could not distinguish the candidates, is in [`probes/platform_log.md`](probes/platform_log.md).

A second, smaller difference: for `survival:cox` at a negative `base_score`, XGBoost returns a NaN whose sign bit differs by platform. Both are quiet NaNs, IEEE-754 does not specify the sign for `log` of a negative, and such a model is refused either way.

**Still unmeasured:** XGBoost's own `expf` differing by architecture, which would move the six accepted *output* differences rather than the margins. **GPU was not measured at all.**

### Specific gaps, named rather than glossed

- The **published JSON Schema enforces roughly a third** of what the format specification requires. Ten of eleven deliberately-malformed artifacts validate against it. Both shipped predictors reject ten and nine of those respectively, so no wrong number reaches you *through them* — but if you validate against the schema and then walk the artifact yourself, it is weaker than it looks.
- The **scikit-learn `XGBClassifier` export path is unmeasured.** The arity checks were verified against `xgboost.train` models only.
- The **`save_best=True` early-stopping callback is unmeasured.** If it trims the model, export proceeds — which is safe, because a trimmed model is unambiguous — but that is inferred, not observed.
- The **exact mechanism of the logistic clamp** is undecidable by measurement: an input clamp and an output floor are observationally identical, and an upper clamp is undetectable because both forms give exactly `1.0`. The constant is pinned; the mechanism is not.
- **A feature value that is finite as float64 but infinite once narrowed is accepted, while an explicitly infinite one is refused.** `1e39` is a legal float64; narrowed to `float32` it is `+inf`, and the walk then compares `inf` against thresholds and routes consistently. Both predictors agree on every such row — the 100,000-row parity run includes them and reports `0` mismatches — so this is not a cross-language defect. But it is an inconsistency in the refusal: `±inf` arriving directly raises, and `1e39` arriving and *becoming* `inf` does not. Python also emits `RuntimeWarning: overflow encountered in cast` on those rows, 1,160 times in the scale run, which the harness counts rather than letting scroll past. Whether such a value should be refused is an open behavioural question, not a resolved one.
- **Two artifact refusals are impossible in JavaScript.** `format_version: 1.0` is indistinguishable from `1` after `JSON.parse`. The Python reader rejects it; the JavaScript reader cannot.

### The evidence itself is committed

Every empirical claim above traces to a report under [`probes/`](probes/) — 11 files recording the commands run and their actual output, including the cases where a measurement contradicted an expectation. Where something could not be established, those reports say so.

---

## Reproducing these numbers

```bash
uv sync                                    # Python workspace
uv run pytest                              # 977 tests, including the parity harness
uv run python parity/run_parity.py         # cross-language agreement, both points
uv run python parity/run_parity_scale.py   # the same, over ~100,000 adversarial rows

npm --prefix packages/js install
npm --prefix packages/js run typecheck     # separate from the build, deliberately
npm --prefix packages/js test              # 112 tests, against the built bundle

./tools/clean_install_python.sh            # wheel contents, then install and predict
./tools/clean_install_js.sh                # tarball contents, then install and predict
```

The two clean-install scripts build the packages, verify their contents, install them into environments containing nothing else, and predict from outside the repository. They exist because everything else in this repository tests the source tree rather than the package a user receives — and two defects reached that gap before these scripts did.
