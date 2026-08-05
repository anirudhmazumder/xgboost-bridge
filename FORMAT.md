# Artifact format, version 1

Normative specification for the JSON artifact produced by `xgboost-bridge` and consumed by `xgboost-predictor`.

Every requirement here is traceable to measured evidence under `probes/` or to a recorded decision in `DECISIONS.md`. Where a rule exists because a plausible alternative was measured and produced wrong numbers, the measurement is cited inline. Nothing in this document is a preference.

Where this document says **MUST**, a conforming implementation that does otherwise is wrong, not merely different. There is no **SHOULD** in this specification; every rule is exact, because two implementations in two languages are written from it and must agree to exactly `0.0`.

**Status:** approved 2026-08-01; corrected 2026-08-02 after two probes closed the blocking evidence gaps and falsified four rules as originally written. §14 records what changed. No open gaps block implementation.

---

## 1. Scope

Version 1 supports **scalar-output tree ensembles only**:

| Dimension | Supported in v1 |
|---|---|
| Booster | `gbtree` only (D016) |
| Objective | `reg:squarederror`, `binary:logistic`, `survival:cox` |
| Output arity | Exactly one output per row (D017) |
| Split type | Numeric only; categorical splits are refused at export (D016, §11) |

Anything outside this table is refused at export time with a loud error. Nothing outside this table has structural space reserved for it in the format (D003): the format version marker is the migration mechanism, and a shape designed against an unimplemented feature is usually the wrong shape.

---

## 2. Two version markers, and they are different things

The artifact carries two version numbers with unrelated jobs. Confusing them is the most likely misreading of this document, so they are named to make that hard.

| Field | Meaning | Who enforces it |
|---|---|---|
| `format_version` | The version of **this specification**. Integer. `1`. | The reader, on every load |
| `provenance.xgboost_version` | The XGBoost version that produced the source model. String. | The exporter, at export time |

`format_version` is the only version a predictor examines. A reader **MUST** raise on any value other than `1` — including `0`, `2`, a string `"1"`, a float `1.0`, or a missing field. It **MUST NOT** attempt best-effort handling of an unrecognized version (D007).

Migration is expressed by incrementing `format_version` and nothing else. There is no minor version, no feature-flag map, and no capability negotiation.

`provenance.xgboost_version` is recorded but never used for inference. The exporter enforces an **enumerated ceiling**: it raises unless the producing version is one that has actually been probed. `COMPAT.md` carries that list.

> **Why an enumerated list and not a range (D018).** XGBoost 3.4.0-dev relocated `weight_drop` from `gradient_booster.weight_drop` to `gradient_booster.model.weight_drop`. XGBoost 3.3.0 loads such an artifact, returns predictions with max error `1.26`, zero rows correct, **zero warnings, exit code 0**, and silently drops the field on re-save (`probes/version_drift.md` §3). The structural lesson is in that report's §5: unrecognized-**field** detection catches *additions* and cannot catch *relocations or removals*, because a missing optional field is not an unknown field. Only an explicit upper bound defends against this class. A guessed range is not an upper bound; it is an assumption.

---

## 3. The envelope

```
{
  "format_version": 1,
  "objective":      <string>,
  "output_transform": <string>,
  "intercept":      <number>,
  "feature_names":  [<string>, ...],
  "trees":          [<tree>, ...],
  "provenance":     { ... }
}
```

All seven keys are **required**. A reader **MUST** raise if any is absent, and **MUST** raise on any key it does not recognize, at every level of the document (D007).

---

## 4. `objective`

A string, exactly one of:

```
"reg:squarederror"   "binary:logistic"   "survival:cox"
```

Any other value raises at both export and load. The reader does not pattern-match, prefix-match, or normalize case.

**`objective` is non-operative metadata with an export-time validation role.** Its only job is the cross-check against `output_transform` (§5), performed in Python at export. It does **not** select an intercept transform — under D015 no intercept transform exists at runtime at all (§6).

A predictor, in either language, **MUST NOT** branch on this field. No `if objective == ...` on any prediction path. The test suite in both languages asserts this, so that a future contributor cannot quietly make it load-bearing: once any code path branches on `objective`, the field stops being metadata and becomes a second source of truth about behavior that `output_transform` already determines.

---

## 5. `output_transform`

A string, exactly one of:

```
"identity"   "sigmoid"   "exp"
```

This is the transform from margin space to the predictor's returned value. It is **carried explicitly rather than derived from `objective`.**

> **Why explicit.** Deriving it would mean that adding an objective later silently changes transform behavior through a shared code path, which is the shape of a silent-wrongness bug. Carrying it explicitly makes the reader's job a lookup with no inference, and lets export assert the objective/transform pairing — redundancy that costs nothing because it lives only in Python.

The exporter **MUST** assert the pairing and raise on a mismatch. The pairing is measured, not assumed (`probes/output_transform.md`):

| `objective` | `output_transform` | Winner residual | Runner-up |
|---|---|---|---|
| `reg:squarederror` | `identity` | `0.0` exactly | `relu`, `1.0` |
| `binary:logistic` | `sigmoid` | `1.042e-07` | `log`, `3.268` |
| `survival:cox` | `exp` | `5.896e-08` | `softplus`, `0.99999997` |

Eleven candidate transforms were scored per objective across 43 fitted models; the winners are bit-exact on 107500/107500 rows. For Cox, `predict() == expf(margin)` on 45000/45000 — `exp` is the whole story and nothing further is applied.

### 5.1 Precision of the output transform — normative

**Margins are float32 throughout, and the output transform is evaluated under float32 semantics.**

```
Python       every intermediate wrapped in np.float32(...)
JavaScript   every intermediate wrapped in Math.fround(...)
```

A predictor **MUST NOT** evaluate the transform in float64 and **MUST NOT** call a platform transcendental (§5.4).

> **Why float32 arithmetic can be simulated exactly, which is what makes this safe.** Performing a float64 operation and then narrowing to float32 is **exact** for `+ − × ÷`: float64 carries more than twice float32's significand, so double-rounding cannot occur for these four operations. This is a property of the IEEE-754 formats, not an empirical observation, and it holds identically in both languages.
>
> It does **not** extend to `exp`. That is precisely why `exp` must be *built from* the four operations rather than called — see §5.4.
>
> IEEE-754 mandates correct rounding only for `+ − × ÷ √` and fused multiply-add. **`exp` is not required to be correctly rounded, and no two `libm` implementations agree.** Measured: V8's `exp` differs from Apple's `libm` on **4.2%** of sigmoid evaluations and **9.6%** of `exp` evaluations, by up to **2 ULP** (worst sigmoid case at margin `0.9417615532875061`: `0.7194553455999664` versus `0.7194553455999666`). Python's `np.exp` and `math.exp` agree on 6009/6009, locating the split between runtimes rather than within one. This is not a precision-width question, and widening does not fix it.

**Why float32 and not float64.** XGBoost transforms in float32 **with clamps** (§5.2). A float64 transform is not off by a ULP in the tail — it is qualitatively wrong there: **relative error `1.0`** below the logistic clamp floor, and finite-versus-`inf` for Cox. Divergence from upstream that is itself silent is the thing this library exists to surface, so upstream is matched in the tail rather than diverged from quietly.

Cross-language parity remains exactly `0.0` under float32 semantics, because every operation on both sides is one this format controls. There is no `libm` anywhere in the path.

### 5.2 XGBoost's clamps are reproduced

Measured behavior that the transform **MUST** reproduce:

| Objective | Clamp |
|---|---|
| `binary:logistic` | Floors at margin `f32(-88.7)`, returning exactly `3.006635794144578e-39` — **never** `0.0`. The sole float32 input producing those bits is `-88.69999694824219`, found by exhaustive scan of all 262145 float32 values in `[-90, -88]` |
| `survival:cox` | **No clamp.** Returns `+inf` above margin ≈ `88.72` (734/2500 rows measured). Never negative, never exactly `0.0`, never NaN |
| `reg:squarederror` | Identity; no clamp |

Two consequences worth stating because they are easy to get backwards:

- Sigmoid output of exactly `1` **is** reachable (288/2500 rows measured). Sigmoid output of exactly `0` is **not** reachable through XGBoost, because the clamp floor prevents it. A fixture demanding saturation at `0` is testing something XGBoost cannot produce.
- Whether the logistic clamp is an input clamp on the margin or an output floor is **observationally identical** and is not resolved here; either implementation satisfies this specification. An upper clamp at `+88.7` is undetectable because both clamped and unclamped give exactly `1.0`.

**Clamp constants are XGBoost internals and are version-sensitive in exactly the way `weight_drop` proved to be** (§2). They fall under the version ceiling and are re-probed whenever the tested version list widens.

**Bit-exactness with XGBoost at the output is unreachable and is not a goal.** XGBoost's own `expf` is not correctly rounded — an mpmath-exact reference rounded to float32 scores 1600/2500 against it. Do not chase it.

### 5.3 The parity gate has two measurement points

Cross-language parity is checked at **both**, and both are exactly `0.0`, bit-identical:

1. **The margin.**
2. **The final output**, after the output transform.

Phase 8 checks both. A margin-only parity check passes while a transform mismatch ships, which is the failure this specification exists to prevent, relocated one stage downstream.

Neither figure carries a tolerance. §5.4 is what makes the second one attainable.

### 5.4 The transform is bundled, not the platform's

Both packages implement `sigmoid` and `exp` themselves. Neither calls `Math.exp`, `math.exp`, `np.exp`, or any other platform transcendental on the prediction path.

> **Why not just state a tolerance instead (the option that was rejected).** A tolerance would have to be a number, and no honest number exists. The 2-ULP figure in §5.1 was measured on exactly one platform pair — V8 against Apple `libm`. glibc's `exp` is a third implementation, and recent glibc is correctly rounded where V8's fdlibm-derived port is not, so a Linux CI runner produces a *different* divergence than a macOS laptop and **neither measurement bounds the other**. Publishing a bound measured on one pair and applying it to every platform a consumer runs on is precisely the silently-generalized numerical claim this project exists to prevent — appearing in the verification gate itself.
>
> The 2 ULP does not matter on its own terms: `3e-16` at a probability changes no decision anyone makes. What matters is that **exact equality is a tripwire and any tolerance is a band that real bugs hide inside.** Set the gate at 2 ULP and a genuine 1-ULP defect passes it forever. This project's methodological claim is that exactness detects what tolerance conceals; accepting a tolerance here would trade that away to avoid work.

Accuracy cost is not a design input. A bundled implementation at ~1 ULP against a correctly-rounded reference, versus `libm`'s ~0.5 ULP, is a relative difference around `1e-16` — sixteen orders of magnitude below the `1e-6` output gate.

### 5.5 Implementation constraints for the bundled transform

Bit-identity across languages is a property that can be lost by accident. These are requirements, not guidance.

- **Only `+`, `−`, `*`, `/`, and exact power-of-two scaling.** Nothing else in IEEE-754 is mandated correctly-rounded.
- **Each operation is a separate statement with an explicit named intermediate.** Do not write a fused expression and rely on neither runtime contracting it into an FMA. The guarantee must come from how the code is written, not from what the runtimes happen to do today.
- **No vectorization in the reference transform.** Scalar, explicit, boring.
- **Argument-reduction constants are split hi/lo and written as literal float64 bit patterns in both languages**, never as decimal strings that each language's parser rounds independently. The test suite verifies that the two sides' constants parse to identical bits before anything downstream is trusted.

### 5.6 Validating the bundled transform

**Each side is validated independently against a high-precision reference — `mpmath` at 50 digits — and never against the other side.**

Cross-language agreement is a **separate** check and is **not** evidence of correctness. Two identical implementations agreeing proves only that the same code was written twice. Bit-identical wrong is still wrong, and it is invisible to the parity harness precisely because both sides agree perfectly.

Per objective, sampling at least `1e6` points across the full representable input range, and reporting **max** ULP error against the reference, never mean:

- overflow and underflow boundaries
- the subnormal transition
- `+inf`, `-inf`, `NaN`
- exact `0.0` and `-0.0`
- margins at and beyond **both clamp boundaries** (§5.2)
- saturation at exactly `1`, which is reachable. Saturation at exactly `0` is **not** reachable through XGBoost — the logistic clamp floor prevents it, so a fixture demanding it tests nothing

This is now the most dangerous code in the repository: novel numerical code is new silent-wrongness surface, introduced in the one place the project previously had none. The adversarial-fixture treatment applies to it in full, including the revert-and-confirm-red methodology of D019.

### 5.7 The accuracy gate against XGBoost is relative; the parity gate is exact

**These are two different gates and conflating them is how a tolerance leaks into the parity gate.** Cross-language parity (§5.3) is exact equality and no tolerance question touches it. This section is only about the Python-vs-XGBoost *accuracy* comparison.

| Comparison | Instrument |
|---|---|
| Margin, Python vs XGBoost | **Absolute** ≤ `1e-6`. Currently `0.0` bit-exact at all 16 measured sweep configurations — treat any regression from that as a defect to diagnose, not headroom to spend |
| Output, Python vs XGBoost | **Relative** ≤ `1e-6`, computed against XGBoost's value |

> **Why the output gate is relative.** An absolute bound flags the wrong objective and misses the real defect. Cox output is a hazard ratio spanning `2.85e-04` to `7.56e+08`, so absolute error there is meaningless — measured `1.94` up to `6.96e+23`, plus rows at `+inf` — while its max *relative* error is `5.7e-08`. Meanwhile `binary:logistic` passes an absolute gate trivially while being **relatively 100% wrong** below the clamp floor if the transform is implemented in float64.

Rules for the cases that otherwise fall out of the comparison silently:

- **`+inf` and `-inf`** — must match as **bit patterns**. Never divide, and never let an infinity drop out of the comparison.
- **NaN** — always a failure, on either side, with no exception. NaN compares unequal to everything including itself, so a naive harness silently *skips* exactly these rows.
- **XGBoost's value exactly `0.0` or `-0.0`** — require exact bit-pattern equality rather than computing a ratio.
- The harness reports **max relative error and the row that produced it**, never a mean. A mean hides one catastrophic row in a large corpus.

Because the transform is bundled and XGBoost's is `libm`-based, a small divergence at the output is expected **by construction** and must not be read as a regression. Bit-exactness with XGBoost is unreachable (§5.2).

### 5.8 The output transform is a different thing from the intercept transform

These two are separate concerns and this specification keeps them separate:

- The **intercept transform** (per-objective, `logit`/`ln`/identity) happens **at export only**, in Python, and has **no runtime representation** (§6).
- The **output transform** happens **at predict time**, in both languages, and is the field specified here.

Collapsing them would be a serious error. §6 removes the first from the artifact; it does not remove the second.

---

## 6. `intercept` — the single operative numeric intercept

A JSON number. The **margin-space** intercept, already transformed, exactly representable as float32.

A predictor **MUST**:
- narrow it to float32 on read,
- use it as the **initial value** of the margin accumulator,
- and **never** transform it in any way.

A predictor **MUST NOT** apply `logit`, `ln`, `exp`, or any other function to this field. There is no objective-dependent branch on this value in either language.

> **Why the artifact stores a derived intercept rather than `base_score` (D015).** The per-objective link space is the single largest source of silent wrongness in this project's history, and this decision removes it from the artifact entirely. `binary:logistic`'s float32 `1/p − 1` form, `survival:cox`'s `ln`, and `reg:squarederror`'s identity all collapse into this one float32 field.
>
> The transform is not merely delicate, it is *specifically* delicate. The textbook `log(p/(1-p))` is not equivalent: it is bit-wrong on 16 of 27 measured values and **breaches the `1e-6` margin gate** (`probes/base_score.md` §5). Independently reproduced during Phase 2 review at 100 trees: `7.63e-06` at `base_score=0.987654` and `1.91e-06` at `0.48`, against `0.0` for the correct form. Reproducing it requires the exact float32 expression, not generic float32 discipline. Implementing that once, in one language, is categorically safer than mirroring it in two.

### 6.1 Deriving the intercept at export — the exact rule

Two things select the space. Both are read from the source model; neither is guessed.

**Step 1 — `boost_from_average` decides whether a transform applies at all.** Read `learner.learner_model_param.boost_from_average`. When there are **zero trees** and it is `"1"`, XGBoost emits the **raw `base_score`** as the margin, with no link transform. Verified: logistic default gives margin `0.5` where the link would give `-0.0`; Cox default gives `0.5` where the link would give `-0.693147`. Every model with at least one tree applies the transform.

This field is load-bearing and was previously absent from this specification. Flipping that one string moves the margin between `0.5` and `-0.0`.

**Step 2 — clamp, then transform, per objective.**

| Objective | Rule |
|---|---|
| `reg:squarederror` | `f32(base_score)`. No clamp |
| `survival:cox` | **float32** `log(f32(base_score))`. No clamp — verified across 34 values from `1.4e-45` to `3.4e38` |
| `binary:logistic` | **Clamp `p` to `[f32(1e-6), f32(1 - 1e-6)]` first**, then **float32** `-log(f32(f32(1/p) - 1))` |

**The logarithm is a float32 logarithm in both cases, not a float64 logarithm narrowed afterwards.** In Python that is `np.log` applied to a float32; it is *not* `np.float32(math.log(float(x)))`. The two differ, and only the float32 route matches XGBoost.

> **Why this is called out, and how it was nearly missed.** The two routes disagree on only **0.055%** of float32 inputs, so a sweep that samples ordinary values finds nothing. Two separate sweeps — one of 79 values, one of 1432 — both concluded "no difference," and one of them explicitly recorded logistic as unaffected.
>
> Isolating the discriminating values first and *then* asking XGBoost settles it immediately. Of 13,421,774 float32 values in `[0.4, 1.2]`, 7387 make the two routes disagree. On a 120-value sample of those, for Cox: the float32 log matches XGBoost **120/120** and the float64 route **0/120**. For logistic, on 75 disagreeing values: **75/75** versus **0/75**.
>
> The lesson is methodological and applies to every numerical check in this repository: a sample that does not deliberately target the cases where two candidate implementations differ cannot distinguish them, and its silence is not evidence of equivalence. Rare-but-systematic is the signature this project exists to catch.

Getting this wrong does not merely shift a digit — because §6.2 requires the derived intercept to match XGBoost's observed margin **bit-for-bit**, a float64 logarithm makes export raise spuriously on roughly one in two thousand models, with no indication that the rule rather than the model is at fault.

> **Why the clamp, and why it is easy to miss.** XGBoost clamps `base_score` before deriving the logistic intercept **but stores the value unclamped.** Applying the recipe to the stored value is wrong by up to **13.8 in margin space**. Independently verified: at `base_score=1e-12` the unclamped recipe gives `-27.631021` where XGBoost's intercept is `-13.815510`; at `1e-7`, `-16.118095` versus `-13.815510`; at `0.9999999`, `15.942385` versus `13.745160`. Clamping first reproduces XGBoost exactly in every case. Additionally `base_score = 1 - 1e-10` stores as `[1E0]`, where the unclamped recipe raises `math domain error` outright.

### 6.2 The export-time intercept check uses an independent oracle

The exporter **MUST** validate the derived intercept against **XGBoost's own observed margin on a tree-free model**, requiring bit equality. It **MUST NOT** validate by re-deriving the intercept from `base_score` and comparing the two derivations.

Which tree-free model, exactly, depends on the case — and getting this backwards makes the check reject correct intercepts:

| Source model | Oracle |
|---|---|
| Has one or more trees | Fit a fresh model with **zero** boosting rounds, same objective, `base_score` passed **explicitly**, and read `predict(output_margin=True)` |
| Already has zero trees | Use **the source model's own** margin directly |

> **Why the zero-tree case cannot use the refit.** Passing `base_score` explicitly flips `boost_from_average` to `"0"`, which puts the refit in *link* space. But a zero-tree model whose `boost_from_average` is `"1"` is correct precisely *because* its margin is the **raw** `base_score` (§6.1). Refitting would compare a raw-space intercept against a link-space oracle and reject the one configuration §6.3 requires as a fixture. Reading the source model's own margin is still XGBoost's observed output rather than a re-derivation, so the independent-oracle property holds.

> **Why this replaces the original check.** The first version of this rule compared the stored `base_score` against a re-derivation of the export recipe. Both sides ran the same recipe, so **an error in the recipe could not make the check fire** — and the `base_score` clamp above is exactly such an error, which that check would have passed. A validation whose oracle shares the defect it is looking for is decorative.
>
> This generalizes and is a standing rule for this codebase: every check must have an oracle independent of the thing it checks. State what the oracle is and why it cannot share the defect. "It compares our value to our other value" is not an oracle.

### 6.3 Signed zero is reachable and is not normalized

`intercept` can legitimately be **negative zero**, and it arrives through an ordinary default: `binary:logistic` with `base_score = 0.5` gives `-log(f32(1/0.5 − 1)) = -log(1) = -0.0`, bit pattern `0x80000000`. Verified during Phase 2 review.

Consequently:
- The exporter **MUST** emit `-0.0` as `-0.0` and **MUST NOT** normalize it to `0.0`.
- Parity comparison is on **bit patterns**, not `==`. In JavaScript `-0 === 0` is `true` while `Object.is(-0, 0)` is `false`; a parity harness using `===` cannot see this difference (`probes/float32_thresholds.md` §2).

A hazard for any future tooling: `JSON.stringify(-0)` in JavaScript emits `0`, silently destroying the sign. Python's `json.dumps(-0.0)` emits `-0.0` and is correct. This is one more reason export is Python-only.

**Required fixture, and it must be built carefully or it tests nothing.** The corpus **MUST** include a case where the intercept *is* the entire output — a zero-tree model, or one whose leaves are all zero — at `base_score = 0.5` for `binary:logistic`. That is the only configuration in which `-0.0` survives to the output rather than being absorbed by the first addition.

**`base_score = 0.5` MUST be passed explicitly.** On a zero-tree model left at the default, `boost_from_average` is `"1"` and XGBoost emits the raw `0.5` rather than the link-transformed `-0.0` (§6.1). A fixture built from the default is therefore in the *wrong space* and would test nothing at all — while looking exactly like a signed-zero fixture. Passing `base_score` explicitly flips `boost_from_average` and the transform is applied.

This correction exists because the first version of this requirement did not say it, and the fixture it specified would have been built wrong.

---

## 7. `feature_names` and input mapping

An array of strings. **Required, non-empty, and every entry unique.** Its length is the model's feature count; no separate count field exists, because a second source of truth for the same fact is a second thing that can disagree.

Index `i` in this array is the column index referenced by `split_indices` (§8).

A reader **MUST** reject a prediction input whose key set is not **exactly** equal to `feature_names` — no missing key, no extra key (D005).

> **Why names are required, and why export raises without them (D021).** A model fit from a bare array serializes `feature_names` as `[]` while `num_feature` is nonzero (`probes/tree_structure.md` §10.1). A strict-key policy with no keys to check reads as enforced and is not, which is worse than no policy: the caller believes a typo will be caught. So the exporter raises and requires the caller to supply names explicitly.

Lenient key handling is specifically forbidden because a missing key routes down the missing-value branch, which is legitimate model structure — so a typo becomes a confident wrong number rather than an error, and the error compounds across the ensemble.

---

## 8. Tree representation

Each element of `trees` is an object of **parallel arrays**, one entry per node:

```
{
  "left_children":  [<int>, ...],
  "right_children": [<int>, ...],
  "split_indices":  [<int>, ...],
  "node_values":    [<number>, ...],
  "default_left":   [<0|1>, ...]
}
```

All five keys are required. All five arrays **MUST** have identical length; a reader **MUST** raise otherwise. There is no node-count field — the array length is the count, for the same single-source-of-truth reason as §7.

| Field | Semantics |
|---|---|
| `left_children[i]` | Left child index, or `-1` if node `i` is a leaf |
| `right_children[i]` | Right child index. `-1` at a leaf |
| `split_indices[i]` | Feature column index at an internal node. `0` at a leaf, and never read there |
| `node_values[i]` | **Internal node:** the float32 split threshold. **Leaf:** the leaf output value |
| `default_left[i]` | `1` = a missing value routes to `left_children[i]`; `0` = to `right_children[i]` |

Node `0` is the root. **A node is a leaf if and only if `left_children[i] == -1`.**

> **Why that leaf test.** It is the only test measured to hold in every observed tree shape (`probes/tree_structure.md` §2). `right_children[i] == -1` coincides at leaves for scalar trees but does *not* hold for vector-leaf trees, where `right_children` at a leaf carries a block index instead. Vector-leaf trees are refused in v1, but a leaf test that is accidentally correct is a latent bug, so the reliable test is specified.

`default_left` is `0`/`1` integers rather than JSON booleans, for compact packing into a `Uint8Array`. Its value at a leaf is meaningless and never read.

### 8.1 `node_values` deliberately keeps XGBoost's overloading

One array carries both thresholds and leaf outputs, exactly as XGBoost's `split_conditions` does. This is a deliberate choice against separating them.

> **Why keep it.** Leaf values require float32 narrowing just as thresholds do — without it, accumulation scores `990`–`3706 / 5000` bit-exact and breaches the `1e-6` gate at `1.07e-04` (`probes/accumulation.md` §3, §6). A single array is loaded into a single `Float32Array` (§9.2), so **both roles are narrowed by one act of construction.** Two arrays would create two narrowing sites, one of which could be forgotten — precisely the redundancy trap D019 warns about. Keeping the overloading also makes export a near-copy of the source array rather than a transformation, and transformations are where export bugs live.
>
> The cost is a name that carries two meanings. That is mitigated by the leaf test being exact and total: given `left_children[i]`, the role of `node_values[i]` is never ambiguous.

### 8.2 Trees are ordered and the order is normative

`trees` is walked in array order and the order is part of the artifact's meaning. Reversing it scores `245`–`2365 / 5000` bit-exact (`probes/accumulation.md` §6). No tree carries an identifier: XGBoost's `id` field is positional and is renumbered by model slicing, so it is not a stable identity and is omitted (`probes/tree_structure.md` §10.2).

### 8.3 Dead nodes are neutralized in place, not removed

Pruned trees in XGBoost retain unreachable nodes in the arrays, marked by `split_indices == 2147483647`, with **stale** `parents` links, and the dead set is **not** in general a trailing suffix — at `gamma=50.0` the dead indices are `[31,32,33,34,37,38,41,42,45,...]`, interleaved with live ones (`probes/tree_structure.md` §7a′).

A v1 artifact contains **no reachable garbage and no out-of-range values**, while keeping every node index exactly as XGBoost assigned it.

The exporter walks each tree from node `0`, marks every reachable node, and then **overwrites each unreachable node's entries with canonical safe values**:

```
split_indices[i]  = 0
node_values[i]    = 0.0
left_children[i]  = -1
right_children[i] = -1
default_left[i]   = 0
```

Array lengths are unchanged. No index is renumbered. Every child reference in the artifact still points where XGBoost pointed it.

> **Why neutralize rather than compact or carry.** This is the reachability walk without the renumbering. Compaction would remap every child reference, and a remapping bug silently reroutes a live sample — real risk for a size win. Carrying the nodes verbatim would instead force a permanent exception into reader validation: it could never assert that a `split_indices` value is in range, because `2147483647` is legitimate in a legitimately pruned model, and that exception would apply to every artifact rather than only pruned ones. Neutralizing keeps the range check total and the indices untouched: strictly less risk than compaction, strictly better validation than carrying.
>
> The cost is size. A heavily-pruned tree still carries its dead slots, which matters for browser delivery — at `gamma=1e9` a 59-node tree has 58 dead nodes. Compaction is a v1.1 optimization, available once the walk has fixture coverage behind it, and it does **not** change the format: an artifact with no dead slots is already valid under this specification.

**Verification is mandatory, to the same standard compaction would have required.** A neutralization that clears a *live* node is silent wrongness — the walk would read a `0.0` threshold or a spurious leaf and return a plausible wrong number. So:

- The exporter **MUST** walk each neutralized tree and compare against XGBoost's own `predict(output_margin=True)`, raising on any difference.
- The fixture corpus **MUST** include a pruned model in which a neutralized node **would have been visited if the reachability walk were wrong** — that is, a dead node that is a child of a live node's stale link. A pruned-model fixture that a broken walk still passes is decorative.

Two independent markers of deadness were measured to agree on all six `gamma` sweeps: unreachability from the root, and `split_indices == 2147483647` (`probes/tree_structure.md` §7a′, and ambiguity A1 there). The exporter uses **reachability** as the definition, because it is the property the walk actually depends on, and **asserts** that the `split_indices` marker agrees — raising if it does not, since a disagreement would mean the model's shape is not what either probe measured.

---

## 9. Numeric encoding — the crux

This section is why the format exists in this shape. XGBoost's engine compares in float32; a float64 value that merely looks close routes a real sample down the wrong branch, and the resulting prediction is plausible.

### 9.1 Emission rule (exporter)

For every value in `node_values` and for `intercept`, the exporter emits the float32 value **widened to float64 and serialized by Python's shortest-round-trip repr** — that is, `json.dumps(float(np.float32(x)))`. For example the float32 nearest `0.1` is emitted as `0.10000000149011612`.

The exporter **MUST NOT** apply any rounding, truncation, or formatting step to these values. No `round(x, n)`, no `%.6g`, no "tidy up the artifact" pass.

> **Why, with the numbers.** Re-emitting thresholds at reduced precision lands on a *different float32*: 5 digits corrupts 324/341 values, 7 digits corrupts 226/341, and **8 digits still corrupts 2/341** (`probes/float32_thresholds.md` §8c). Nine significant digits is the floor. The 2/341 case at 8 digits is the worst possible failure rate — frequent enough to produce wrong numbers, rare enough to survive review. The chosen rule satisfies the floor by construction, since a shortest round-trip float64 repr of a float32-valued double always recovers that float32; measured drift `0/341` (`probes/float32_thresholds.md` §8d).

Values are emitted as **plain JSON numbers**, never strings.

> **Why not mirror XGBoost's own encoding.** XGBoost stores `base_score` as a JSON *string containing a bracketed array* — `"[4.8E-1]"` — requiring two parses (`probes/base_score.md` §1). Its threshold tokens are always uppercase-`E` exponent notation and drop the decimal point for single-digit mantissas, so `0.5` is written `5E-1`; a regex expecting `D.D+E` rejects valid tokens (`probes/float32_thresholds.md` §2). Both are upstream quirks with no upside for a format we control, and the second is an active trap for a hand-written parser. This format uses ordinary JSON numbers throughout.

### 9.2 Narrowing rule (reader)

**Narrowing happens at parse time, not at comparison time.**

```
Python       arr = np.asarray(json_list, dtype=np.float32)
JavaScript   arr = Float32Array.from(jsonArray)
```

`JSON.parse` returns float64 unconditionally, and on **104/104** measured thresholds that float64 is a different number from the engine's float32 (`probes/float32_thresholds.md` §8b).

Narrowing the float64 parse is **exact** — `float32(float64(text)) == float32(text)` on 341/341 tokens — so no re-quantization is needed; the parse is not the hazard (`probes/float32_thresholds.md` §8a). The hazard is *using* the un-narrowed value.

Storing `node_values` in a `Float32Array` / `dtype=np.float32` array is required rather than incidental. It makes the invariant a **property of the data structure** instead of a discipline every future reader has to remember. If narrowing lived at the comparison site, any other consumer of the threshold — a re-serializer, an inspection utility, an arithmetic transform — reintroduces the bug (`probes/float32_thresholds.md` §9).

### 9.3 Non-finite values

`node_values` and `intercept` **MUST** be finite. A reader **MUST** raise on `Infinity`, `-Infinity`, or `NaN`, in any spelling.

No non-finite threshold was observed, and training on infinite features raises upstream — but that is evidence about training on finite data, not proof that no code path can emit one, so the reader raises rather than assumes (`probes/float32_thresholds.md` §2).

Separately, **non-finite values in prediction input raise** (D022). Upstream is inconsistent here: `±inf` raises through `DMatrix` but is treated as an ordinary comparable value through `inplace_predict`, so the same input yields two different predictions depending on the call path (`probes/tree_structure.md` §6.1). This library picks one behavior — raise — and pins it with a fixture.

`NaN` in prediction input is **not** an error. It is the missing value, and it routes by `default_left`.

---

## 10. Normative prediction algorithm

This is specification, not guidance. Every deviation below it was measured and every one loses.

```
INPUT
  intercept    float32
  trees        in artifact order
  node_values  float32 array per tree   (thresholds AND leaf values)
  left_children, right_children, split_indices, default_left

MARGIN
  acc = float32(intercept)                       # INTERCEPT FIRST, before any tree

  for t in trees:                                # artifact array order
      node = 0
      while left_children[t][node] != -1:        # leaf iff left child == -1
          v = input[ feature_names[ split_indices[t][node] ] ]
          if isNaN(v):
              node = default_left[t][node] ? left_children[t][node]
                                           : right_children[t][node]
          else:
              node = cast32(v) < cast32(node_values[t][node])   # STRICT '<', BOTH sides
                     ? left_children[t][node] : right_children[t][node]
      acc = cast32(acc + node_values[t][node])   # NARROW AFTER EVERY SINGLE ADD

  margin = acc                                   # already float32; touch it no further

OUTPUT
  apply output_transform to margin
```

`cast32` is `np.float32(...)` in Python and `Math.fround(...)` in JavaScript.

Measured constraints, each with the cost of getting it wrong (`probes/accumulation.md` §6, and independently reproduced during Phase 2 review):

| Rule | Measured cost of the alternative |
|---|---|
| Both sides of the comparison cast | One-sided casting: a 6.6-percentage-point probability error on a real row (`probes/float32_thresholds.md` §6) |
| Strict `<`; **equality routes RIGHT** | Measured on 104/104 nodes plus 195 more across 7 models |
| Intercept is the accumulator's initial value | Intercept last: `199`–`2120 / 5000`; breaches the gate at `1.34e-05` |
| Trees in artifact order | Reversed: `245`–`2365 / 5000` |
| Narrow after **every** add | float64 sum narrowed once at the end: `318`–`2541 / 5000` |
| Leaf values narrowed on read | Un-narrowed: `990`–`3706 / 5000` |

Correct implementation: **5000/5000 bit-exact against `predict(output_margin=True)`, max abs error `0.0`**, across 3 objectives × tree counts 0–1000 × two `tree_method`s. Python-vs-JavaScript reproduced at `0.0`.

> **A fixture-design trap that belongs in this specification, not only in the test suite.** At `base_score = 0.5` **every** wrong variant above scores 5000/5000, because the logistic intercept is exactly `-0.0` and intercept placement stops mattering. `survival:cox` has the same trap at its estimated default, where the intercept is exactly `0.0`. A corpus built on those values validates a broken implementation (`probes/accumulation.md` §8, and verified during Phase 2 review). Boundary values near `1.0` are required.

### 10.1 Verification of the two narrowing sites

Per D019 each narrowing site is verified **in isolation** — reverted one at a time, never as a pair — and if a site cannot be made to fail on its own, that is reported rather than covered by a test that proves nothing.

**There are five float32 sites in the walk, not two:** the sample cast and the threshold cast in the comparison, the leaf cast and the accumulator wrap on the add, and the intercept cast. Measured revert results, each site reverted alone:

| Site | Tests red | Rows wrong |
|---|---|---|
| sample cast | 2 | 126/629 |
| threshold cast | 1 | 96/464 |
| leaf cast | 2 | 980/2000 |
| intercept cast | 2 | 35/2000 |
| accumulator wrap | **0** | — |

The accumulator wrap is a **provable** no-op rather than an unmeasured one: given the leaf cast and the intercept cast, both operands of that addition are already float32, and wrapping a float32 result in another narrowing changes no bit. No input can make it fail while the other two stand, so any test claiming to pin it would be decorative. It is retained as defence-in-depth and disclosed here, which is what this section requires. What *is* load-bearing — the accumulator staying float32 across every add — is pinned by the float64-running-sum variant at 473/2000.

> **A correction, because the earlier version of this section drew the wrong conclusion.** It stated that leaf narrowing was absorbed by per-add narrowing, citing a variant that still scored 5000/5000. That absorption is an artefact of **NumPy's NEP 50 weak-scalar promotion**, not a property of the arithmetic: it holds only while the un-narrowed values arrive as *weak* Python floats. Handed a strong `dtype=np.float64` array, leaf narrowing is not absorbed at all and 980/2000 rows come back wrong, which is the regime `probes/accumulation.md` §6 measured at 990–3706/5000.
>
> The practical consequence for tests, and it is easy to get wrong: a harness that feeds the walk Python floats masks **four of the five sites**, because whichever operand still carries a float32 drags the whole operation into float32. Pinning a cast requires handing the walk a **strong float64** input — an `np.float64` row or a `dtype=np.float64` value array — so that the cast is the only thing keeping the operation in float32. `np.float64(0.1) < np.float32(0.1)` is `True` while `np.float32(0.1) < np.float32(0.1)` is `False`; that one row is the whole difference between a test that pins the invariant and one that merely appears to.

---

## 11. Export-time validation

The exporter raises on every condition below. Each is a wrong-number path, not a limitation.

**Booster (D016).** Raise if `weight_drop` is present at **either** `gradient_booster.weight_drop` **or** `gradient_booster.model.weight_drop`. Raise on `gblinear`.

> Both paths must be checked because the field relocated between 3.3.0 and 3.4.0-dev (`probes/version_drift.md` §3). Only one in-artifact dart signal exists, confirmed by exhaustive key census (`probes/boosters.md` §2) — so the two-signal rule cannot be satisfied for *acceptance*, but works perfectly for *refusal*. A model trained as `dart` with `rate_drop=0`/`skip_drop=0` is byte-identical to `gbtree` and exports fine; that is a feature of D016, not a gap.

**Output arity (D017).** In addition to the objective allow-list, require:

| Field | Location | Required value |
|---|---|---|
| `num_target` | `learner.learner_model_param` | `"1"` |
| `num_class` | `learner.learner_model_param` | `"0"` **or** `"1"` |
| `size_leaf_vector` | **each tree's** `tree_param` | `"1"` for every tree; a model with **zero trees vacuously passes** |

**All four gate fields — including `objective.name` — are JSON strings and MUST be compared as strings.** An integer comparison silently never fires: `num_class == 0` is `False` against `"0"`. Verified for all four.

> **`num_class` accepts `"1"`, and requiring `"0"` was a defect.** Independently verified across all three in-scope objectives: `binary:logistic` with `num_class=1` produces the *same model* — trees byte-identical, margins bit-identical 400/400, `predict()` shape `(400,)` — and requiring `"0"` **rejects all three**. A false rejection is worse than over-strictness because it reads as correct strictness and is not. Relaxing to `{"0", "1"}` admitted nothing extra across a 23-shape table with zero false acceptances; `multi:softprob` with `num_class=1` is still caught by the objective allow-list.
>
> **`size_leaf_vector` exists only per-tree**, never in `learner_model_param` (census over 20 artifacts: occurrences equal the tree count). A scalar comparison against a model-level field has no referent, and on a zero-tree model there is nothing to read at all — hence the explicit vacuous-pass rule. Without it the gate rejects every zero-round model, which §16 and §6.3 both require to be exportable.
>
> An objective-name allow-list alone has a hole: `reg:squarederror` with `num_target=2` is an in-scope objective producing `tree_info=[0,1,0,1,...]`, a two-element `base_score`, and `(N,2)` margins (`probes/multiclass_extensibility.md` §7). A scalar predictor accepts it and returns confident wrong numbers.

Zero-boosting-round models serialize `"trees": []` — **present and empty, never absent**, verified on all three objectives. A reader handles the empty array; it does not need to handle a missing key.

**Categorical splits.** Raise if **any** of: `split_type` contains `1`; `categories_nodes` is non-empty; `learner.feature_types` contains `'c'`. All three were measured present together (`probes/tree_structure.md` §9); all three are checked because this is a refusal test, where redundancy is free.

> Silently reading a categorical split as numeric is a wrong-number path for two independent reasons. Categorical splits **invert the child convention** — in-set routes to the *right* child, the opposite of a numeric split — and `split_conditions` at such a node holds the smallest positive subnormal float32 (`1e-45`), which is not a threshold at all.

**Feature names (D021).** Raise if `feature_names` is empty, contains duplicates, or its length disagrees with `num_feature`.

**XGBoost version (D018).** Raise unless the producing version is in the tested list in `COMPAT.md`.

**Early stopping (D038).** Raise when `iteration_indptr[best_iteration + 1] != len(trees)`, reading `best_iteration` from `learner.attributes.best_iteration` (a JSON string) and `iteration_indptr` from `learner.gradient_booster.model.iteration_indptr`. The error directs the caller to slice the model explicitly — `booster[0:best_iteration + 1]` — and re-export.

> **Why refusal rather than a choice.** The effective tree count is not a property of the model. Measured on one file on disk: loaded as a bare `Booster` it matches a walk over **all** trees 2500/2500; loaded through a scikit-learn estimator it matches a walk over `best_iteration + 1` **iterations** 2500/2500. Max divergence between the two readings is `1.55` in margin space, and **no field in the model distinguishes them**. Either choice is silently wrong for half of callers, on every row.
>
> The predicate is deliberately not "`best_iteration` is present." With `early_stopping_rounds` set but never fired, both readings agree, and refusing on presence alone would reject unambiguous models.
>
> Not version-dependent — re-measured byte-identical on XGBoost 3.4.0. It is API-path-dependent, which a version ceiling cannot defend against; only refusal can.

**Neutralization self-check (§8.3).** Raise if any neutralized tree's walk disagrees with `predict(output_margin=True)`, and raise if the reachability marking disagrees with the `split_indices == 2147483647` marker.

**Non-finite intercept.** Raise if the derived intercept is not finite, per §9.3.

> This is reachable and XGBoost does not announce it. `survival:cox` with `base_score = 0.0` yields an intercept of `-inf` (bits `0xFF800000`), and with any negative `base_score` yields `NaN` (`0x7FC00000`) — both accepted by XGBoost with no error and no warning. The derivation reproduces them bit-for-bit, which is necessary for the oracle in §6.2 to agree, so the refusal belongs at export rather than in the derivation. Note that a bit-pattern oracle would happily match `NaN` against `NaN`: the check that catches this is the finiteness requirement, not the comparison.

**Intercept agreement, against an independent oracle (§6.2).** Raise if the derived `intercept` disagrees bit-for-bit with **XGBoost's own zero-tree margin** for the same configuration. Python-only; costs nothing at runtime. Do **not** implement this as a comparison against a re-derivation of the export recipe — that check cannot fire on a recipe error, which is the class of error it exists to catch.

---

## 12. Determinism

The same model exported twice **MUST** produce byte-identical output (D008).

- Object keys sorted lexicographically at every level.
- Compact separators, no insignificant whitespace, `\n` line endings, UTF-8, no trailing newline ambiguity — the exporter writes exactly one.
- Numbers formatted only by the rule in §9.1. No locale-dependent formatting anywhere.
- No timestamps, hostnames, paths, or environment-derived values in any field.
- Signed zero preserved, never normalized (§6.1).

**Excluded because it varies between fits.** `learner.attributes` is the only nondeterministic surface measured in the source model — early stopping writes `best_score` as a full-precision string (`probes/tree_structure.md` §11.1). Per D020 nothing from it reaches the artifact except by explicit whitelist, and **the v1 whitelist is empty.** Determinism by construction beats determinism by hope.

`loss_changes` drifts in its final digits between XGBoost versions (`probes/version_drift.md` §3). It is not a prediction-path field and is not in this format (§15), so the drift cannot reach an artifact.

---

## 13. Reader behavior on anything unrecognized

A conforming reader raises — never defaults, never guesses, never skips (D007) — on:

- `format_version` absent or not exactly `1`
- any unrecognized key, at any level of the document
- any required key absent
- a value of the wrong JSON type, including a numeric field carried as a string
- an `objective` or `output_transform` outside its enumerated set
- an objective/transform pairing that does not match
- the five tree arrays having unequal lengths
- a child index out of range, or a `split_indices` value outside `[0, len(feature_names))`
- a non-finite `node_values` entry or `intercept`
- `feature_names` empty, or containing a duplicate
- a prediction input whose key set is not exactly `feature_names`
- a non-finite value in prediction input

Errors carry structured attributes — which key, which index, what was expected — not only a message string, so a caller can branch on the failure programmatically.

A reader **MUST NOT** raise on a node that is unreachable from the root. Neutralized dead slots are legitimate artifact content (§8.3), they are indistinguishable from a leaf carrying value `0.0`, and the walk never visits them. A reader that rejected unreachable nodes would reject every pruned model.

The reader takes a parsed object. There is no `fromFile` in JavaScript (D006): filesystem access is unavailable in browsers and differs across edge runtimes, so a file loader would either add a dependency or split the bundle by runtime. Neither is acceptable against a zero-dependency universal bundle.

---

## 14. Evidence gaps, and what closing them changed

**G1 and G3 are closed.** Both were probed (`probes/output_transform.md`, `probes/arity_gate.md`) and both falsified rules this specification had asserted. The corrections are folded into the sections above; they are listed here because a reader who saw the earlier version needs to know what moved and why.

| Was | Is | Evidence |
|---|---|---|
| Output transform widens to float64 on both sides | Evaluated under **float32** semantics, reproducing XGBoost's clamps (§5.1, §5.2) | XGBoost transforms in float32: 400/400 bit-exact versus 236/400 for float64, and on all 164 rows where the hypotheses disagree, XGBoost matches float32 164/164 and float64 0/164 |
| Output gate is absolute ≤ `1e-6` | **Relative** ≤ `1e-6`, with explicit `inf`/NaN/signed-zero rules (§5.7) | Cox absolute error reaches `6.96e+23` with `+inf` rows while its relative error is `5.7e-08`; logistic passes absolutely while being relatively 100% wrong below the clamp |
| Intercept is `-log(f32(f32(1/p) − 1))` of the stored `base_score` | **Clamp `p` first** to `[f32(1e-6), f32(1 − 1e-6)]` (§6.1) | Unclamped is wrong by up to `13.8` in margin space, and raises `math domain error` at `base_score = 1 − 1e-10` |
| Export asserts intercept against a re-derivation | Asserts against **XGBoost's observed zero-tree margin** (§6.2) | The original check applies the same recipe to both sides, so the clamp defect above could not make it fire |
| `num_class == "0"` | `num_class ∈ {"0", "1"}`, compared as a **string** (§11) | `"1"` is producible on a genuine single-output model, verified across all three objectives; the old rule rejected all three |
| `size_leaf_vector == "1"` model-level | **Per-tree**, with zero trees a vacuous pass (§11) | The field exists only per-tree; a zero-tree model has zero occurrences and the old rule had no referent |
| Signed-zero fixture at `base_score = 0.5` | Same, but `base_score` **passed explicitly** (§6.3) | At the default, `boost_from_average == "1"` makes a zero-tree margin the raw `0.5`, not the link-transformed `-0.0` — the fixture would have been built in the wrong space |

Two of these were defects in reasoning rather than gaps in coverage — the self-blind assertion and the float64 transform — and both are recorded rather than quietly overwritten, because the reasoning failures generalize.

**G2 — `learner.gradient_booster.model.cats` purpose unknown.** *Still open, not blocking.* Empty in every model probed, including two with genuine categorical splits (`probes/tree_structure.md` §3.1). Categorical models are refused in v1, so this cannot affect a v1 artifact. It needs probing before any categorical support.

**G4 — the logistic clamp's exact form.** *Open, not blocking.* Whether the clamp is applied to the margin on input or as a floor on output is observationally identical, and an upper clamp at `+88.7` is undetectable because clamped and unclamped both give exactly `1.0`. Either implementation satisfies §5.2. The measured floor constant is pinned; its mechanism is not.

---

## 15. Deliberately not in the format

Each omission is a decision, and none of these fields is reserved for later (D003).

| Omitted | Why |
|---|---|
| `base_score` as an operative field | D015. Retained in `provenance`, read by nothing |
| `boost_from_average` | An **export-time input**, not artifact content. It selects which intercept space applies (§6.1) and its effect is already baked into `intercept`. Carrying it would invite a predictor to branch on it |
| `base_weights` | Unshrunk node weight, not the leaf output. A walk using it is off by `5.10` in margin (`probes/tree_structure.md` §4) |
| `loss_changes`, `sum_hessian`, `parents` | Never read by any walk that reproduces `predict()` at `0.0` (`probes/tree_structure.md` §12). `parents` is additionally *stale* after pruning, and `loss_changes` drifts across versions |
| `id` | Positional, renumbered by slicing; not a stable identity |
| `num_nodes`, `num_feature` | Derivable from array lengths. A second source of truth for the same fact is a second thing that can disagree |
| `num_deleted` | After neutralization a dead slot is indistinguishable from a leaf carrying `0.0` (§8.3), so the reader could not verify the count. A number no consumer can check is a claim, not evidence |
| `split_type` | All splits are numeric; categorical is refused at export (§11) |
| `categories`, `categories_nodes`, `categories_segments`, `categories_sizes`, `cats` | Same |
| `tree_info` | v1 has exactly one output group (D017), so it would be uniformly `0`. Grouping lives *outside* tree objects in XGBoost, so adding it later is additive and needs no restructuring (`probes/multiclass_extensibility.md` §4) |
| `weight_drop` | dart is refused (D016). No tree-weight array exists in this format |
| `leaf_weights`, vector leaves | `size_leaf_vector == 1` is required (§11) |
| `best_iteration` | Models with an ambiguous tree count are refused outright (D038, §11), so there is no ambiguous count for an artifact to record |
| `missing=` parameter | Not recorded in the source model at all (`probes/tree_structure.md` §6.2), so it cannot be preserved. `NaN` is the missing value |
| `learner.attributes` | Nondeterministic; whitelist is empty (D020, §12) |
| Reserved space for multi-class, multi-target, categorical, dart | D003. The version marker is the migration mechanism |

---

## 16. Worked example

A `binary:logistic` model, two features, two trees. The second tree is a single leaf — a shape that occurs in practice and that the format handles as a degenerate case of the general layout rather than a special case (`probes/tree_structure.md` §7a).

Source model had `base_score = 0.6`, so `intercept = -log(f32(f32(1/0.6) − 1)) = 0.40546515583992004` (float32 `0.40546516`). Computed, not illustrative.

```json
{
  "feature_names": ["feature_a", "feature_b"],
  "format_version": 1,
  "intercept": 0.40546515583992004,
  "objective": "binary:logistic",
  "output_transform": "sigmoid",
  "provenance": {
    "base_score": "[6E-1]",
    "exporter_version": "0.1.0.dev0",
    "xgboost_version": "3.3.0"
  },
  "trees": [
    {
      "default_left": [1, 0, 0],
      "left_children": [1, -1, -1],
      "node_values": [0.5, -0.25, 0.75],
      "right_children": [2, -1, -1],
      "split_indices": [0, 0, 0]
    },
    {
      "default_left": [0],
      "left_children": [-1],
      "node_values": [0.125],
      "right_children": [-1],
      "split_indices": [0]
    }
  ]
}
```

Predicting `{"feature_a": 0.25, "feature_b": 9.0}`:

1. `acc = f32(0.40546515583992004)` → `0.40546516`
2. Tree 0, node 0: `f32(0.25) < f32(0.5)` is true → left → node 1, a leaf. `acc = f32(0.40546516 + (-0.25))` → `0.15546516`
3. Tree 1, node 0 is a leaf. `acc = f32(0.15546516 + 0.125)` → `0.28046516`
4. `margin = 0.28046516`; `sigmoid(margin) = 0.5696602463722229`

> The sigmoid decimal is the **float32** result, as §5.1 requires. A float64 sigmoid of the same margin gives `0.5696602593994496`, which narrows to the same float32 bit pattern `0x3f11d541` — so the two are not in conflict, but an implementer comparing decimals against the wrong one will conclude they have a bug. Earlier revisions of this example printed the float64 value.

Note that `feature_b` is never read by any split, yet it **must** still be present in the input: the key set must equal `feature_names` exactly (§7). Omitting it raises rather than being treated as missing — which is the entire point of D005, since a missing value is legitimate model structure and would silently produce a different, plausible number.

Note also that keys are sorted at every level (§12).

**`provenance.base_score` is the raw string exactly as XGBoost stored it** — `"[6E-1]"`, a JSON string containing a bracketed array — not a parsed or float32-snapped number. It is copied through with zero interpretation.

> **Why raw rather than tidied.** Provenance exists to record what the source model actually contained. A parsed, snapped number would be a *derived* value sitting in a block whose whole purpose is fidelity, and a derived number in a provenance block is an invitation for someone to use it — at which point the field stops being non-operative. The raw string cannot be mistaken for something a predictor should read, because nothing downstream accepts that shape. The operative intercept is `intercept`, and it is the only numeric field a predictor touches (§6).
