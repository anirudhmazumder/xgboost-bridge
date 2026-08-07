# xgboost-bridge

Read this before changing anything. The invariants below are not style preferences — each one exists because violating it produces *plausible wrong numbers* rather than an error.

Several of them replaced an earlier belief that looked sound and was measured false. Where that happened it is marked, because the superseded version is the one a reasonable person would guess.

## What this is

`xgboost-bridge` exports trained XGBoost models as portable JSON artifacts and runs zero-dependency JavaScript inference in browser and edge environments.

It exists because standard XGBoost→ONNX conversion **fails silently**: the conversion succeeds, inference runs, and the predictions are wrong. No exception, no warning. Every design choice in this repository follows from that failure mode. A crash is an acceptable outcome; a wrong number is not.

| Package | Registry | Role |
|---|---|---|
| `xgboost-bridge` | PyPI | Python export, inspection, reference predictor |
| `xgboost-predictor` | npm | Zero-dependency browser/Node inference |

**1.0 scope:** binary classification, regression, Cox survival. **`gbtree` only** — dart and gblinear raise (see below).

This is a general-purpose library. **No application-specific vocabulary anywhere — variable names included.** If a name would only make sense to someone working on one particular problem, it is the wrong name. Enforced by an executable, self-testing scrub in the Python suite.

## Layout

```
packages/python/src/xgboost_bridge/   Python package
packages/js/src/                      TypeScript package
fixtures/                             fixture corpus + generators
schema/                               JSON Schema for the artifact format
probes/                               recorded empirical findings
FORMAT.md                             artifact format specification
VERIFICATION.md                       what is measured, and what is not
docs/DECISIONS.md                     durable decisions and rationale
```

Python is a `uv` workspace with members `packages/python` and `fixtures`. JavaScript builds with `tsup`.

## Invariants

### Float32 split precision

XGBoost's engine compares features against thresholds in **float32**. Exported threshold values are the shortest decimal that round-trips in float32 — not a bit-identical float64.

Cast **both sides** of every comparison, and the operator is **strict `<` with equality routing RIGHT** — measured on 104/104 internal nodes of the primary model plus every internal node of seven further models:

```python
np.float32(value) < np.float32(threshold)
```

```javascript
Math.fround(value) < Math.fround(splitCond)
```

Casting only the sample value is insufficient. It produced a **6.6-percentage-point probability error** on a real row. This is the highest-value invariant in the codebase; it belongs on the first line of the tree walk, not patched in afterward.

**Narrowing happens at parse time, not at comparison time.** `JSON.parse` returns float64 unconditionally, and on 104/104 measured thresholds that float64 is a different number from the engine's float32. Thresholds and leaf values load into a `Float32Array` / `dtype=np.float32` array so the invariant is a property of the data structure rather than a discipline every future reader must remember.

**Leaf values need narrowing too.** Un-narrowed scored 990–3706/5000 bit-exact and breached the gate at `1.07e-04`.

### The accumulation recipe is normative

1. Narrow thresholds **and** leaf values to float32 on read.
2. Initialize the accumulator with the float32 intercept — **before any tree**.
3. Walk trees in serialized order. Leaf iff `left_children[i] == -1`.
4. Narrow the accumulator to float32 **after every single addition**.

Every deviation was measured and every one loses: intercept added last scores 199–2120/5000, reversed tree order 245–2365/5000, a float64 sum narrowed once at the end 318–2541/5000.

### `base_score` is per-objective, and logistic is not what it looks like

> **Superseded belief:** the logistic intercept is `logit(base_score)`. It is not. Textbook `log(p/(1-p))` breaches the `1e-6` gate at `7.63e-06`.

- `reg:squarederror` — identity.
- `survival:cox` — `log(f32(base_score))`.
- `binary:logistic` — `-log(f32(f32(1/p) - 1))`, and **`p` is clamped to `[f32(1e-6), f32(1 - 1e-6)]` before the transform while being stored unclamped.** The unclamped recipe is wrong by up to **13.8 in margin space** at extreme `base_score`, and raises `math domain error` at `base_score = 1 - 1e-10`, which stores as `[1E0]`.

**The intercept is read out of the engine, not computed.** `objectives.observe_intercept` is what export ships; `derive_intercept` documents how the engine gets there and is off the export path.

> **Superseded belief:** that a float32 `np.log` *is* XGBoost's logarithm. It is XGBoost's logarithm **on darwin/arm64**. XGBoost calls the platform's `logf`, IEEE-754 does not require `logf` to be correctly rounded, and XGBoost's own intercept differs between darwin/arm64 and linux/x86_64 by exactly 1 ULP on **29 of 58** discriminating inputs. Agreement with the engine: `np.log` 58/58 on darwin and 36/58 on Linux; the float64 route 10/58 and 39/58; a correctly-rounded 60-digit reference also 10/58 and 39/58 — so the engine is not correctly rounded either, and **no fixed recipe is exact on both platforms.**

"Bit-exact against XGBoost" and "platform-independent" are therefore incompatible here rather than a choice of recipe. A 1-ULP intercept error is silent — it shifts every margin the model produces — so the value comes from the engine. Deriving it guarantees a spurious refusal somewhere: it refused 13 of 50 ordinary logistic `base_score` values on Linux, which is 18 test failures that pass on darwin. Artifacts exported on the two platforms may differ in this one field by 1 ULP; inference does not, because the value is stored and no consumer computes a logarithm. See D053 and `probes/platform_log.md`.

**Both logarithms are still float32 logarithms** inside `derive_intercept` — `np.log` of a float32, **not** `np.float32(math.log(float(x)))`. The two routes differ on only 0.055% of float32 inputs, which is why two independent sweeps (79 values and 1432 values) both concluded "no difference."

**The lesson repeated itself one level up.** A first pass at the platform investigation used 13 hand-picked `base_score` values, found `np.log`, XGBoost and a correctly-rounded reference in agreement on all 13, and concluded that XGBoost was correctly rounded and numpy was the mover. Both halves were false, and selecting 40 inputs *because* the routes disagree there reversed the result immediately (`np.log` 40/40, correctly-rounded 0/40). Knowing the rule is not the same as applying it.

> **The general lesson, which applies to every numerical check here:** a sample that does not deliberately target the inputs where two candidate implementations diverge cannot distinguish them, and its silence is not evidence of equivalence. Find the disagreeing inputs first, then ask XGBoost.

**`boost_from_average` selects the intercept space and is load-bearing.** At `learner.learner_model_param.boost_from_average`. With **zero trees and `boost_from_average == "1"`, the margin is the RAW `base_score`**, not the link transform — logistic default gives `0.5` where the link gives `-0.0`; Cox default gives `0.5` where the link gives `-0.693147`.

**Never infer a new objective's space by analogy.** Every objective gets its own verification against a real fitted artifact, recorded under `probes/`.

### The output transform is bundled, runs in float32, and reproduces XGBoost's clamps

The margin→output transform is a **separate concern** from the intercept transform. Do not collapse them.

Both packages implement `sigmoid` and `exp` themselves. Neither calls `Math.exp`, `math.exp`, or `np.exp` on the prediction path.

> **Why bundled:** V8's `exp` and Apple's `libm` differ on 4.2% of sigmoid and 9.6% of `exp` evaluations, by up to 2 ULP. IEEE-754 mandates correct rounding only for `+ − × ÷ √` and fma — `exp` is not required to be correctly rounded and no two `libm` implementations agree. This is not a precision-width question and widening does not fix it.

Evaluated under **float32** semantics, via `Math.fround` per step in JS and `np.float32` per step in Python. Simulating float32 arithmetic as a float64 operation followed by a narrowing is **exact** for `+ − × ÷` — float64 carries more than twice float32's significand, so double-rounding cannot occur for these four operations. That is a property of the format, not an observation. It does **not** extend to `exp`, which is precisely why `exp` must be built from the four operations rather than called.

XGBoost transforms in float32 **with clamps**: `binary:logistic` floors at margin `f32(-88.7)` returning exactly `3.006635794144578e-39` and never `0.0`; `survival:cox` has no clamp and returns `+inf` above margin ≈ `88.72`. Float64 was not off by a ULP in the tail — it was relative error `1.0` below the logistic floor and finite-versus-`inf` for Cox. Clamp constants are XGBoost internals, are **version-sensitive**, and are pinned empirically under `probes/`.

**Bit-exactness with XGBoost at the output is unreachable and is not a goal** — XGBoost's own `expf` is not correctly rounded (an mpmath-exact reference scores 1600/2500 against it).

### Every check needs an oracle independent of what it checks

Before accepting any validation, state what its oracle is and why the oracle cannot share the defect. If the answer is "it compares our value to our other value," the check is decorative.

This is not abstract. An earlier export assertion compared a derived intercept against a re-derivation of the same recipe, so an error in the recipe could not make it fire. It was replaced with a comparison against **XGBoost's own observed zero-tree margin**.

- Numerical implementations validate against a **high-precision reference** (`mpmath`, 50 digits), **per side, independently**. Cross-language agreement is a *separate* check and is never evidence of correctness — two identical implementations agreeing proves only that the code was written twice.
- Export-side assertions validate against **XGBoost's observed output**, never against a re-derivation of the export recipe.
- Redundant safeguards are untested safeguards. If removing either of two overlapping protections breaks no test, neither is pinned. Test each **independently** where that is achievable; where it is not, **pin the minimal detectable combination and name what remains unpinned.**

  > **Superseded belief:** *"Test each independently, or collapse them to one."* The second half was too strong. It assumed the redundancy was always incidental — that if you cannot separate two protections you should delete one. Measured counterexample: the accumulator's float32 discipline rests on **three** narrowings (the seed, the per-leaf cast, the outer cast) and any one of the three is sufficient alone, so every single-site revert stays green. Collapsing would delete working protection from a public, normative function to buy a tidier coverage map. Naming the gap is more honest than a collapse that pretends the question away. See D064 and the `absorbs` field in `tools/revert_harness.py`.

**An independent oracle asked a non-discriminating question still proves nothing.** These are two requirements, not one, and satisfying the first is the more comfortable half.

The fixture corpus is the worked example, and it matters because the reassuring sentence about it is *true*. "Our fixtures come from `booster.predict()`, never from our own walk" — correct, and the generator additionally re-walks every row and refuses to write a fixture it cannot reproduce bit-exactly. A reader who stops there concludes the corpus cannot launder a defect.

It could. The oracle was never the weakness; **the rows were.** Every value the corpus carried was already exact in float32, so narrowing it is a no-op and an implementation missing the sample-side cast routes all of them correctly. Measured: that cast was reverted, the corpus regenerated, and **generation succeeded silently.** The refusal was real, the ground truth was real, and the question being asked could not distinguish the correct implementation from the broken one.

So when a check is defended on the strength of its oracle, ask the second question too: *which input would fail if the thing under test were wrong?* If there is no answer, the oracle's independence is decoration. The corpus generator now asks XGBoost about 1322 rows that **narrow onto** a threshold without equalling it, and the same revert now fails at generation time. D064; and `fixtures/tests/test_fixture_door.py`, which exists because that probe's own failure mode is silence rather than error.

### dart and gblinear are refused

> **Superseded belief:** dart can be detected by two independent signals. It cannot. Exactly **one** in-artifact signal exists (`weight_drop`), confirmed by exhaustive key census. The string "dart" appears nowhere in the artifact, and dart at `rate_drop=0` is byte-identical to `gbtree`.

Raise on `weight_drop` at **either** JSON path — `gradient_booster.weight_drop` or `gradient_booster.model.weight_drop`, because it relocated between versions. Raise on `gblinear`.

The two-signal rule survives as a **rejection** test, which is what it is good at. A dart model with no dropout exports fine, and that is correct: it is indistinguishable from a tree ensemble because it *is* one.

`gblinear`'s `shotgun` updater is non-deterministic — the cause is thread parallelism, not seeding (12/12 distinct weight vectors at `nthread=4`, 1/12 at `nthread=1`). Pin `coord_descent` anywhere it appears.

### Version drift is real, silent, and only a ceiling catches it

XGBoost 3.4.0-dev relocated `weight_drop`. XGBoost 3.3.0 loads such an artifact, returns 0/400 rows correct at max error `1.26`, **emits zero warnings, exits 0**, and drops the field on re-save.

Unrecognized-**field** detection catches *additions* and **structurally cannot catch relocations or removals** — a missing optional field is not an unknown field. Only an explicit version ceiling defends against this class. `COMPAT.md` records the versions **actually probed, as a list**. An untested version is an unrecognized input.

### Export-time gate fields

All are JSON **strings** — an integer comparison silently never fires. Export requires the objective in the allow-list **and** `num_target == "1"` **and** `num_class ∈ {"0", "1"}` **and** the per-tree `size_leaf_vector == "1"` check.

`num_class` can legitimately be `"1"` on a single-output model; requiring `"0"` falsely rejects valid models. `size_leaf_vector` exists **only per-tree**, never in `learner_model_param`, so a zero-tree model has zero occurrences and needs an explicit rule.

An objective-name allow-list alone is not enough: `reg:squarederror` with `num_target=2` is an in-scope objective producing `(N,2)` margins, and it arrives through a permitted door.

## Design commitments

- **Strict feature keys.** Exact match — no missing, no extra. Lenient handling turns a typo into a missing-value path, which is legitimate model structure, so the mistake compounds into a confident wrong number. Rationale in `COMPAT.md`.
- **Feature names are required.** `feature_names == []` when a model is fit from a bare array; export raises. A strict-key policy with no keys to check reads as enforced and is not.
- **No `fromFile` in JavaScript.** Consumers do their own I/O and call `fromJSON`.
- **Fail loudly on anything unrecognized.** Unknown objective, booster, field, or out-of-range version marker — raise. Never default, never guess, never skip.
- **Deterministic export.** Byte-identical output, tested explicitly. `learner.attributes` is the only nondeterministic surface; it is excluded except by explicit whitelist.
- **Signed zero is never normalized.** It is reachable through an ordinary default — `binary:logistic` at `base_score=0.5` gives an intercept of exactly `-0.0`.
- **Zero JavaScript runtime dependencies.** Non-negotiable. Not "few." Zero.
- **The version marker is the migration mechanism.** Do not reserve structural space for unimplemented features.

Durable decisions and their rationale live in `docs/DECISIONS.md`, including which decisions superseded which and on what evidence. Add to it rather than relitigating.

## Verification gates

| Check | Threshold |
|---|---|
| Python suite | All pass; test count never decreases |
| Node suite | All pass; test count never decreases |
| **Margin parity, Python vs JS** | **Exactly `0.0`**, bit-pattern comparison, no tolerance |
| **Output parity, Python vs JS** | **Exactly `0.0`**, bit-pattern comparison, no tolerance |
| Python vs XGBoost, margin | **Absolute** ≤ `1e-6`. Currently `0.0` bit-exact — treat regression from that as a defect |
| Python vs XGBoost, output | **Relative** ≤ `1e-6`. `±inf` must match as bit patterns; NaN is always a failure; where XGBoost's value is `0.0` or `-0.0`, require exact bit equality rather than a ratio |
| Bundled transform | Max ULP vs `mpmath` at 50 digits, **per side independently**, reported as max never mean |
| `tsc --noEmit` | Clean, run as a step separate from the build |
| JS tests import from `dist/` | Confirmed — never from `src/` |
| Export determinism | Byte-identical across runs |
| JS runtime dependencies | **0** |
| Vocabulary scrub | Executable, self-testing, clean |

**Cross-language parity and upstream accuracy are different gates.** Parity is exact equality and no tolerance question touches it. Conflating the two is how a tolerance leaks into the parity gate.

Comparison is on **bit patterns**, not `==`: `-0.0 == 0.0` is true and they are different artifacts.

A **nonzero parity number means a bit-level defect** — a missing narrowing site, a transform in the wrong space, or an operation that is not correctly-rounded. Diagnose it. Never accept a tiny nonzero parity and move on; that is precisely the silent failure this project exists to prevent.

## Rules for changes

- **Never loosen, skip, or `xfail` a test.** If a test is in the way, it is telling you something. Report it; do not disable it.
- **Never add a dependency** without it being an explicit, separate decision. Zero JS runtime dependencies is absolute.
- **Never decide a numerical fact by reasoning about what should be true.** If a probe can settle it, run the probe. Every falsified belief in this project was inference that looked sound.
- **Report ambiguity rather than resolving it silently.** A confident guess about empirical XGBoost behavior propagates into every downstream number.
- Verify each protection by reverting it **in isolation** and confirming the specific tests go red. `tools/revert_harness.py` does this mechanically; run it, and treat a revert that turns **nothing** red as a finding rather than a pass. Three things look identical from outside — a test that was renamed away, another site absorbing the failure, and a revert that no longer removes what it claims to. Rule out the third first; it has happened.

  Where a protection genuinely cannot be isolated, **pin the minimal detectable combination and name what remains unpinned** rather than collapsing the redundancy away. The float32 narrowing sites are the worked example and they split two ways: the threshold-side cast in `walk_margin` *is* independently pinnable and was found unpinned, while the accumulator's three narrowings are mutually absorbing by construction and are pinned as a set. Both outcomes are recorded in D064; neither is a licence to revert a pair and call it one test.

## Commands

```bash
uv sync                                    # Python workspace
uv run pytest                              # Python suite
npm --prefix packages/js install           # JS deps
npm --prefix packages/js run build         # bundle to dist/
npm --prefix packages/js run typecheck     # tsc --noEmit, separate step
npm --prefix packages/js test              # build, then run suite against dist/
```
